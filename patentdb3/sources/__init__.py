"""The patent's own grant XML. There is exactly ONE source, and this is it.

v2 ranked three tiers — `uspto_xml`, then Google Patents HTML, then MinerU OCR
— and v3 carries only the first. That is a real narrowing and it should be
stated plainly rather than implied by an absent file:

  KEPT     uspto_xml   — USPTO's own grant XML. OASIS/CALS table markup, so
                         exact cells, qualifiers intact, no OCR anywhere.
                         Coverage begins with grants from 2002.

  DROPPED  google_html — clean prose but NO table structure, and its
                         machine-read chemistry measured 0.2% exact-match
                         against BindingDB. For a package whose entire output
                         is table-derived assay rows it contributes nothing to
                         the measurement half. The 209-document scrape still
                         sits in `output_v2/gpatents_cache/` (194 MB), unread.

  DROPPED  mineru_ocr  — PDF -> Markdown OCR. Recovered table structure by
                         inference and paid for it in `<|ref|>` tags, bbox
                         pollution, merged rows and character errors.

WHAT THE NARROWING COSTS, precisely: a patent with no grant XML — granted
before 2002, or simply absent from the USPTO index — now has NO fallback. It
raises `UsptoUnavailable`, `verify` prints "no XML" and skips it, and the patent
contributes nothing. In v2 it fell through to Google Patents and contributed
prose. Nothing here silently degrades; the patent is just absent from the run,
and the skip is printed. Watch for it when a corpus is drawn from BindingDB,
which cites plenty of pre-2002 patents.
"""
from __future__ import annotations
