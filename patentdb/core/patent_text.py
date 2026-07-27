"""Single source for loading patent text from disk.

Before this module: 4+ places open `output_v2/gpatents_cache/{pid}.json`
directly with their own error handling and cache-path logic. This module
is the canonical loader.

Two functions:
  - `load_gp_description(patent_id)` — the Google Patents description
    field (the main text body we extract from).
  - `load_full_patent_text(patent_id)` — concatenated per-page markdown
    + GP description, used by the FSM pipeline for assay extraction.

Both apply Stage 0 normalization (mojibake repair + HTML unescape +
NFKC) before returning, so callers don't need to remember to do it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .assay_fsm.normalizer import normalize_page

logger = logging.getLogger(__name__)


def gp_cache_path(patent_id: str) -> Path:
    """Canonical filesystem path of the Google-Patents description cache."""
    return config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"


def load_gp_description(patent_id: str, *, normalize: bool = True) -> str:
    """Load the Google-Patents description text for `patent_id`.

    Returns "" if the cache doesn't exist or is malformed (never raises).
    With `normalize=True` (default), applies Stage 0 normalization
    (mojibake + HTML unescape + NFKC).
    """
    p = gp_cache_path(patent_id)
    if not p.exists():
        logger.debug("no GP cache for %s at %s", patent_id, p)
        return ""
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        logger.warning("GP cache hydrate failed for %s: %r", patent_id, e)
        return ""
    text = (data.get("description") or "")
    if normalize and text:
        text = normalize_page(text)
    return text


def load_gp_claims(patent_id: str, *, normalize: bool = True) -> str:
    """Load the Google-Patents claims section."""
    p = gp_cache_path(patent_id)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text())
    except Exception:
        return ""
    text = (data.get("claims") or "")
    if normalize and text:
        text = normalize_page(text)
    return text


def load_full_patent_text(
    patent_id: str,
    data_dir: Path | None = None,
    *,
    normalize: bool = True,
) -> str:
    """Concatenated full patent text — per-page markdown files +
    Google Patents description.

    Used by the FSM pipeline for end-to-end assay extraction. The
    order matches what the cheap pipeline processes: markdown pages
    first (in alpha order), then GP description.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    pieces: list[str] = []
    for subdir in ("all_pages", "iupacs_clean"):
        pages_dir = data_dir / patent_id / subdir
        if not pages_dir.exists():
            continue
        for page_file in sorted(pages_dir.glob("page_*.md")):
            try:
                pieces.append(page_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    desc = load_gp_description(patent_id, normalize=False)   # we normalize at the end once
    if desc:
        pieces.append(desc)
    out = "\n".join(pieces)
    if normalize and out:
        out = normalize_page(out)
    return out


def load_patent_description(
    patent_id: str,
    *,
    prefer_format: str = "auto",
    data_dir: Path | None = None,
    normalize: bool = True,
) -> tuple[str, str]:
    """Return (description_text, source_format) for any patent.

    The patent-agnostic primary entry point for IUPAC/density extraction.
    **HTML-first** — Google Patents clean HTML beats MinerU OCR markdown
    in every controlled comparison we've run (no `<|ref|>` tags, no
    [[bbox]] artifacts, no mid-name line wraps). Markdown is the
    fallback for patents that aren't on Google Patents at all.

    Resolution order with `prefer_format="auto"`:
        1. output_v2/gpatents_cache/{patent_id}.json    (HTML scrape)
        2. {data_dir}/{patent_id}/all_pages/page_*.md   (MinerU markdown)

    With `prefer_format="markdown"` or `prefer_format="html"`, that source
    is required — a missing file raises FileNotFoundError. Useful for
    A/B comparisons (Gate D in the plan).

    Markdown pages are concatenated in lexical `page_*.md` order with
    "\\n\\n" separators so the existing density-scoring strategies
    (1–4) operate on a single blob, identical to the HTML path. No
    per-page granularity is exposed.

    Returns ("", source_format) if no source produces text.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    pages_dir = data_dir / patent_id / "all_pages"

    # 1. HTML scrape (preferred — clean text, no OCR artifacts).
    #    Critical for IUPAC extraction quality: MinerU markdown carries
    #    `<|ref|>...<|/ref|>` and `[[114,99,...]]` detection-tag pollution
    #    mid-IUPAC-name. HTML stays clean.
    if prefer_format in ("auto", "html"):
        text = load_gp_description(patent_id, normalize=normalize)
        if text:
            return text, "google_html"
        if prefer_format == "html":
            raise FileNotFoundError(
                f"no HTML cache for {patent_id} at {gp_cache_path(patent_id)}"
            )

    # 2. Markdown fallback — patents not on Google Patents
    if prefer_format in ("auto", "markdown"):
        if pages_dir.exists():
            page_files = sorted(pages_dir.glob("page_*.md"))
            if page_files:
                pieces = []
                for pf in page_files:
                    try:
                        pieces.append(pf.read_text(encoding="utf-8"))
                    except Exception as e:
                        logger.warning("read failed for %s: %r", pf, e)
                if pieces:
                    text = "\n\n".join(pieces)
                    if normalize:
                        text = normalize_page(text)
                    return text, "mineru_markdown"
        if prefer_format == "markdown":
            raise FileNotFoundError(
                f"no markdown pages for {patent_id} at {pages_dir}"
            )

    return "", "none"
