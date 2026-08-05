"""Single source for loading patent text from disk.

Before this module: 4+ places open `output_v2/gpatents_cache/{pid}.json`
directly with their own error handling and cache-path logic. This module
is the canonical loader.

Source ranking, everywhere in this module: the patent's own USPTO XML first,
Google Patents and MinerU below it. GP and PubChem are for RESOLVING (turning
a name into a structure) and for filling gaps, not for supplying the text —
taking the values from the XML and the names from a scrape of the same
document is what made a 250-line `GP107 -> 107` reconciliation bridge
necessary in the first place.

  - `load_uspto_description(patent_id)` — the patent's own description text.
  - `load_gp_description(patent_id)` — the Google Patents description field.
  - `load_full_patent_text(patent_id)` — all sources concatenated, XML first,
    used by the FSM pipeline for assay extraction.

Both apply Stage 0 normalization (mojibake repair + HTML unescape +
NFKC) before returning, so callers don't need to remember to do it.
"""
from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path

from . import config
from .assay_fsm.normalizer import normalize_page

logger = logging.getLogger(__name__)


def uspto_xml_path(patent_id: str) -> Path:
    """Canonical filesystem path of the patent's own grant/publication XML."""
    return config.OUTPUT_DIR / "uspto_xml" / f"{patent_id}.xml"


# The description body. Everything outside it is front matter — title, abstract,
# claims, citations — and pulling those in gives the density scorers a second
# copy of every compound name that appears in the claims.
_DESC_RE = re.compile(r"<description[^>]*>(.*?)</description>", re.S | re.I)
# Block elements that must become a line break, or an example header runs
# straight into the paragraph beneath it and `terminate_name` has to undo it.
_BLOCK_RE = re.compile(
    r"</?(?:p|heading|li|ul|ol|br|div|tr|entry|row|section|"
    r"description-of-drawings|tables?)\b[^>]*>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def load_uspto_description(patent_id: str, *, normalize: bool = True) -> str:
    """The description text out of the patent's own XML. Never raises.

    Block-level tags become newlines BEFORE the rest are stripped. Flattening
    every tag to a space instead — the obvious one-liner — welds each example
    header onto the synthesis paragraph below it, which is precisely the
    unterminated-name defect `core/name_boundary` exists to clean up. Producing
    it here and repairing it downstream would be silly.

    `<tables>` is treated as a block boundary rather than removed: the CALS
    cells hold compound ids and values that the density scorers legitimately
    read, and a helper that silently dropped `<tables>` has already cost this
    project a false "the data isn't in this patent" once.
    """
    p = uspto_xml_path(patent_id)
    if not p.exists():
        return ""
    try:
        raw = p.read_text(errors="ignore")
    except OSError as e:
        logger.warning("read failed for %s: %r", p, e)
        return ""
    m = _DESC_RE.search(raw)
    body = m.group(1) if m else raw
    body = _BLOCK_RE.sub("\n", body)
    body = _TAG_RE.sub(" ", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n\s*\n+", "\n\n", body)
    text = body.strip()
    return normalize_page(text) if (normalize and text) else text


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

    Used by the FSM pipeline for end-to-end assay extraction.

    USPTO XML LEADS. This concatenated markdown pages FIRST and the GP
    description second, with the patent's own XML absent — so the OCR'd text
    came before both cleaner sources, and any scorer that takes the first
    match of a compound got the OCR'd rendering of it. Same inversion as
    `load_patent_description` had, in the function next to it.

    Order now: XML, then GP, then markdown. Concatenation means nothing is
    lost either way; it is which copy a first-match scan reaches first that
    changes.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    pieces: list[str] = []
    xml_text = load_uspto_description(patent_id, normalize=False)
    if xml_text:
        pieces.append(xml_text)
    desc_first = load_gp_description(patent_id, normalize=False)
    if desc_first:
        pieces.append(desc_first)
    for subdir in ("all_pages", "iupacs_clean"):
        pages_dir = data_dir / patent_id / subdir
        if not pages_dir.exists():
            continue
        for page_file in sorted(pages_dir.glob("page_*.md")):
            try:
                pieces.append(page_file.read_text(encoding="utf-8"))
            except Exception:
                pass
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

    **USPTO XML FIRST** — the patent's own published text, which is what the
    assay path has always used and what the source ranking in the README says
    is tier 1. This loader did not know the XML existed: it went straight to
    Google Patents, so names came from a scrape of the document while values
    came from the document, and MinerU OCR sat one fallback below.

    That mattered. US11292791's stored names carried `imidazo[4,5-flquinolin`
    (`f]` read as `fl`) and `1,6'-biosquinolin]`, neither of which appears in
    the USPTO XML or the GP cache — OCR artifacts from a run made before GP
    was cached, when MinerU was the only source left. A tier-3 OCR fallback
    should never have been reachable for a patent whose tier-1 text was
    sitting on disk.

    The XML is also more complete: 896,523 characters against GP's 835,891 on
    US11292791.

    Resolution order with `prefer_format="auto"`:
        1. output_v2/uspto_xml/{patent_id}.xml         (the patent's own text)
        2. output_v2/gpatents_cache/{patent_id}.json   (HTML scrape)
        3. {data_dir}/{patent_id}/all_pages/page_*.md  (MinerU markdown)

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

    # USPTO XML is NOT preferred for description TEXT, and the attempt to make
    # it so was reverted. It remains tier 1 for ASSAY TABLES (`uspto_assays`
    # reads the CALS markup directly) and for MARKUSH (`<chemistry>` anchors
    # carry document position), neither of which comes through this loader.
    #
    # What preferring it here actually bought, measured: nothing. The two texts
    # are word-for-word identical — 147,996 words against 147,998 on
    # US11292791, and every one of 1,079 diff blocks is GP mojibake against
    # correct XML Unicode, which `core.name_boundary.demojibake` already
    # repairs. What it COST was severe and took three separate investigations
    # to see:
    #
    #   `routes/google_patents.py` gates the Strategy 0 harvest on
    #   `text_source_format in ("google_html", ...)`, so returning "uspto_xml"
    #   silently switched off `gp_embedded_meta` — 11,633 of 17,359 resolved
    #   structures, 65% of the corpus. It also flipped `is_clean` to False,
    #   routing the cleanest source in the system down the OCR-repair cascade.
    #
    #   Every cached LLM prompt is keyed on the text it was built from, so the
    #   change invalidated the lot: only 14 of 182 name-repair prompts on
    #   US10544143 still exist in `output_v2/cache`. 92% became questions we
    #   had never asked, which is most of what a re-extraction would have paid
    #   for.
    #
    # `load_uspto_description` is kept — it is correct, and callers that want
    # the patent's own text can ask for it by name with prefer_format.
    if prefer_format == "uspto_xml":
        text = load_uspto_description(patent_id, normalize=normalize)
        if text:
            return text, "uspto_xml"
        raise FileNotFoundError(
            f"no USPTO XML for {patent_id} at {uspto_xml_path(patent_id)}")

    # 2. HTML scrape (clean text, no OCR artifacts).
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
