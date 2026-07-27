"""Patent-agnostic MinerU runner.

End-to-end pipeline for one patent:
    1. Download PDF (if not already on disk)
    2. Run MinerU via subprocess (uses venv_mineru)
    3. Post-process MinerU's per-page markdown into the canonical
       {patent_id}/all_pages/page_*.md layout that HARVEST + Markush
       (and now the MinerU-aware IUPAC loader) consume

Idempotent: skips work if `all_pages/` is already populated.

Usage:
    python -m patentdb.scripts.eval.run_mineru_for_patent US8952177
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

import requests

from patentdb.core import config

logger = logging.getLogger(__name__)


# Resolve venv_mineru's binary regardless of where this is invoked from.
REPO_ROOT = config.REPO_ROOT
MINERU_BIN = REPO_ROOT / "venv_mineru" / "bin" / "mineru"
RAW_OUTPUT_ROOT = REPO_ROOT / "mineru_output"


# ── PDF download (patent-agnostic) ────────────────────────────────


def _download_patent_pdf(patent_id: str, target: Path) -> None:
    """Find the PDF link on the Google Patents page for `patent_id` and
    download to `target`. Patent-agnostic.

    Google Patents serves the PDF via a `<meta itemprop="pdf">` tag or
    a download link in the page HTML. We hit the page, scrape the URL,
    follow it.
    """
    page_url = f"https://patents.google.com/patent/{patent_id}/en"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; patent-extraction)"}
    logger.info("fetching Google Patents page for %s", patent_id)
    r = requests.get(page_url, timeout=30, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(
            f"Google Patents returned {r.status_code} for {patent_id}"
        )
    # Two link forms:
    #   <meta itemprop="pdf" content="https://patentimages.storage.googleapis.com/...">
    #   <a href="https://patentimages.storage.googleapis.com/.../{pid}.pdf">
    pdf_url: str | None = None
    m = re.search(
        r'<meta itemprop="pdf"\s+content="([^"]+)"',
        r.text,
    )
    if m:
        pdf_url = m.group(1)
    else:
        m = re.search(
            r'<a[^>]+href="(https://patentimages\.storage\.googleapis\.com/[^"]+\.pdf)"',
            r.text,
        )
        if m:
            pdf_url = m.group(1)
    if not pdf_url:
        raise RuntimeError(
            f"could not find PDF URL on Google Patents page for {patent_id}"
        )
    logger.info("downloading PDF from %s", pdf_url)
    pdf_resp = requests.get(pdf_url, timeout=120, headers=headers, stream=True)
    if pdf_resp.status_code != 200:
        raise RuntimeError(
            f"PDF fetch returned {pdf_resp.status_code} for {patent_id}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        for chunk in pdf_resp.iter_content(chunk_size=64 * 1024):
            f.write(chunk)
    size_kb = target.stat().st_size / 1024
    logger.info("PDF saved to %s (%.1f KB)", target, size_kb)


# ── MinerU invocation ─────────────────────────────────────────────


def _run_mineru(pdf_path: Path, out_dir: Path) -> Path:
    """Run MinerU on `pdf_path`, output to `out_dir`. Returns the
    `mineru_output/{stem}/auto/` directory. Raises on failure.
    """
    if not MINERU_BIN.exists():
        raise RuntimeError(
            f"MinerU binary not found at {MINERU_BIN}. "
            f"Activate venv_mineru and ensure mineru is installed."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(MINERU_BIN),
        "-p", str(pdf_path),
        "-o", str(out_dir),
        "-m", "auto",
        "-b", "pipeline",
    ]
    logger.info("running MinerU: %s", " ".join(cmd))
    # MinerU is slow (5-10 min on small patents, 30-60+ min on long
    # BDB-rich patents). No timeout — let it finish. Callers that need
    # bounded wall time should invoke this in a backgrounded process.
    result = subprocess.run(
        cmd, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MinerU failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
        )
    stem = pdf_path.stem
    auto_dir = out_dir / stem / "auto"
    if not auto_dir.exists():
        # MinerU may use a different layout; search for it.
        candidates = list(out_dir.glob(f"{stem}*/auto"))
        if not candidates:
            raise RuntimeError(
                f"MinerU completed but no auto/ directory found for {stem} "
                f"under {out_dir}"
            )
        auto_dir = candidates[0]
    return auto_dir


# ── Post-process: MinerU output → canonical all_pages/page_*.md ────


_PAGE_BREAK_RE = re.compile(r"<!-- *page\s+(\d+) *-->", re.IGNORECASE)


def _split_markdown_into_pages(md_text: str) -> list[str]:
    """Split MinerU's full-document markdown on page-break markers if
    they exist. Falls back to a single-page chunk if no markers found —
    that's still useful for density-based IUPAC extraction.
    """
    if not _PAGE_BREAK_RE.search(md_text):
        return [md_text]
    # Use page markers as cut points; each chunk includes the marker.
    cuts = [m.start() for m in _PAGE_BREAK_RE.finditer(md_text)]
    cuts.append(len(md_text))
    pages = []
    for i in range(len(cuts) - 1):
        chunk = md_text[cuts[i]:cuts[i + 1]].strip()
        if chunk:
            pages.append(chunk)
    return pages


def _postprocess_to_all_pages(auto_dir: Path, all_pages: Path) -> int:
    """Convert MinerU's auto/ output into `{patent_id}/all_pages/page_*.md`.

    Returns number of pages written. MinerU produces one big {stem}.md
    file by default; if it has page-break HTML comments, we split.
    Otherwise we write the whole thing as page_0001.md — still works
    for the density-based IUPAC extractor (it operates on a blob).
    """
    md_files = list(auto_dir.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"no .md files in {auto_dir}")
    # Concatenate all .md files MinerU produced (usually just one)
    md_text = "\n\n".join(
        p.read_text(encoding="utf-8") for p in sorted(md_files)
    )
    pages = _split_markdown_into_pages(md_text)
    all_pages.mkdir(parents=True, exist_ok=True)
    # Clear any stale pages
    for old in all_pages.glob("page_*.md"):
        old.unlink()
    for i, page in enumerate(pages, start=1):
        target = all_pages / f"page_{i:04d}.md"
        target.write_text(page, encoding="utf-8")
    return len(pages)


# ── Public entry ───────────────────────────────────────────────────


def run_mineru_for_patent(
    patent_id: str,
    *,
    force: bool = False,
) -> Path:
    """End-to-end MinerU pass for one patent. Patent-agnostic.

    Returns the populated `data/{patent_id}/all_pages` directory.

    Idempotent in two directions:
      - SUCCESS cache: if `{patent_id}/all_pages/` already has pages,
        skip immediately.
      - FAILURE cache: if a prior MinerU attempt left a `.mineru_failed`
        marker under `mineru_output/{patent_id}/`, skip — re-running
        the same model on the same PDF will just fail the same way.
        Caller falls through to GP-only extraction. Delete the marker
        (or pass `force=True`) to retry.
    """
    pages_dir = REPO_ROOT / patent_id / "all_pages"
    if not force and pages_dir.exists():
        existing = list(pages_dir.glob("page_*.md"))
        if existing:
            logger.info(
                "%s: all_pages/ already has %d pages — skipping (force=False)",
                patent_id, len(existing),
            )
            return pages_dir

    failure_marker = RAW_OUTPUT_ROOT / patent_id / ".mineru_failed"
    if not force and failure_marker.exists():
        logger.info(
            "%s: prior MinerU attempt failed (marker at %s) — skipping. "
            "Delete the marker or pass force=True to retry.",
            patent_id, failure_marker,
        )
        return pages_dir

    pdf_path = REPO_ROOT / patent_id / f"{patent_id}.pdf"
    if not pdf_path.exists():
        _download_patent_pdf(patent_id, pdf_path)
    else:
        logger.info("%s: PDF already on disk at %s", patent_id, pdf_path)

    raw_out = RAW_OUTPUT_ROOT
    try:
        auto_dir = _run_mineru(pdf_path, raw_out)
    except Exception as e:
        # Persist a marker so future runs don't keep grinding on the
        # same failure. Includes the error for diagnostics.
        failure_marker.parent.mkdir(parents=True, exist_ok=True)
        failure_marker.write_text(f"{e}\n")
        logger.warning(
            "%s: MinerU failed; wrote failure marker to %s — pipeline "
            "will fall through to GP-only extraction on this and future runs",
            patent_id, failure_marker,
        )
        raise
    n_pages = _postprocess_to_all_pages(auto_dir, pages_dir)
    logger.info("%s: %d page(s) written to %s", patent_id, n_pages, pages_dir)
    return pages_dir


# ── CLI ────────────────────────────────────────────────────────────


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Patent-agnostic MinerU runner")
    ap.add_argument("patent_ids", nargs="+",
                    help="One or more patent IDs (e.g. US8952177)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if all_pages/ exists")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    rc = 0
    for pid in args.patent_ids:
        try:
            run_mineru_for_patent(pid, force=args.force)
        except Exception as e:
            logger.error("%s: %s", pid, e)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(_cli())
