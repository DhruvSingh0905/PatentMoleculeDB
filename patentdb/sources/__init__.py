"""Ranked text/table sources for a patent.

One module per way of getting a patent's text and tables, tried in quality
order. The ranking is not a preference — it is a measured statement about how
much reconstruction each source forces us to do:

  1. uspto_xml   — USPTO's own grant XML. CALS table markup, no OCR. Exact
                   cells, qualifiers preserved. Grants from 2002 onward.
  2. google_html — Google Patents HTML. Clean prose, but no table structure,
                   and its machine-read chemistry is unreliable (measured at
                   0.2% exact-match against BindingDB; see docs/notes/).
  3. mineru_ocr  — PDF→Markdown OCR. Recovers table structure by inference and
                   pays for it in `<|ref|>` tags, bbox pollution, merged rows
                   and character errors. Last resort.

`resolve` walks the tiers in order and returns the first that yields text,
recording which tier answered so provenance survives downstream. A tier that
raises is logged and skipped; a tier that is merely absent for this patent is
not an error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from . import uspto_xml

logger = logging.getLogger(__name__)


@dataclass
class SourceResult:
    text: str
    tables: list          # list[uspto_xml.Table] when tier 1 answered, else []
    tier: str             # "uspto_xml" | "google_html" | "mineru_ocr" | "none"
    structured_tables: bool   # True only when cells came from real markup


def resolve(patent_id: str, *, allow_tiers: tuple[str, ...] = (
    "uspto_xml", "google_html", "mineru_ocr",
)) -> SourceResult:
    """Return the best available text+tables for `patent_id`."""

    if "uspto_xml" in allow_tiers:
        try:
            xml = uspto_xml.fetch_grant_xml(patent_id)
            text = uspto_xml.description_text(xml)
            tables = uspto_xml.parse_tables(xml)
            if text or tables:
                logger.info(
                    "source %s: uspto_xml (%d chars, %d tables)",
                    patent_id, len(text), len(tables),
                )
                return SourceResult(text, tables, "uspto_xml", True)
        except uspto_xml.UsptoUnavailable as e:
            # Expected for pre-2002 grants and publications — not a failure.
            logger.info("source %s: uspto_xml unavailable (%s)", patent_id, e)
        except Exception as e:
            # Transport/parse trouble IS a failure worth seeing. Falling through
            # silently is how this codebase historically turned outages into
            # "the patent had no compounds".
            logger.warning("source %s: uspto_xml errored: %r", patent_id, e)

    # Tiers 2 and 3 stay behind the existing loader, which already implements
    # the GP-then-MinerU order. Kept as one call so there is a single place
    # that knows the legacy resolution rules.
    if "google_html" in allow_tiers or "mineru_ocr" in allow_tiers:
        from ..core.patent_text import load_patent_description
        prefer = "auto"
        if "google_html" not in allow_tiers:
            prefer = "markdown"
        elif "mineru_ocr" not in allow_tiers:
            prefer = "html"
        try:
            text, fmt = load_patent_description(patent_id, prefer_format=prefer)
        except FileNotFoundError:
            text, fmt = "", "none"
        if text:
            tier = "google_html" if fmt == "google_html" else "mineru_ocr"
            logger.info("source %s: %s (%d chars)", patent_id, tier, len(text))
            return SourceResult(text, [], tier, False)

    logger.warning("source %s: no tier produced text", patent_id)
    return SourceResult("", [], "none", False)
