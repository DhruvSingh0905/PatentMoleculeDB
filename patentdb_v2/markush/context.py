"""R-group option text hygiene for the Markush TAGGING route.

What is left here is two string helpers — `_clean_rgroup_text` and
`_is_valid_rgroup_option` — imported by `routes/text_markush.py`.

The LM-driven multi-agent Markush extractor that used to live in this
module is gone: `extract_markush_multiagent`, `_extract_formula`,
`_extract_formula_from_image`, `_extract_substituents`,
`_symbolic_r_group_parse`, `_check_consistency`,
`classify_markush_difficulty`, `get_context_for_prompt` and their four
prompt constants.

Two independent reasons, either sufficient:

  - **Nothing called any of them.** `routes/text_markush.py` imports
    exactly the two helpers below; no other module in the tree referenced
    the rest. The live Markush path is
    `text_markush.extract_markush_dict_hybrid(patent_id, text)`, which
    runs over text the caller already has.

  - **They read MinerU markdown and page images.** Every one of them
    globbed `DATA_DIR/{pid}/{iupacs_clean,all_pages}/page_*.md` or
    `IMAGES_DIR/{pid}`, and the pipeline no longer generates or reads OCR.
    They could not have run even if something called them.

CLAUDE.md already records that Markush ENUMERATION is not in the pipeline
(`markush/enumerate.py` + `step.py` are held out in `_attic/`). This was
the LM half of the same retired machinery, left behind.
"""

from __future__ import annotations

import re


def _clean_rgroup_text(text: str) -> str:
    """Clean R-group definition text before splitting into options.

    Removes pagination artifacts, figure references, LaTeX fragments,
    and normalizes whitespace and formatting.
    """
    t = text
    # Remove page references: [123, 456], page 123, FIG. 1
    t = re.sub(r'\[\[?\d{2,4}[,\s\d]*\]\]?', '', t)
    t = re.sub(r'(?:page|FIG\.?)\s*\d+', '', t, flags=re.IGNORECASE)
    # Remove "text" annotations from OCR
    t = re.sub(r'\btext\b', '', t, flags=re.IGNORECASE)
    # Remove "(continued)" markers
    t = re.sub(r'\(continued\)', '', t, flags=re.IGNORECASE)
    # Remove LaTeX math fragments
    t = re.sub(r'\\[\(\)]', '', t)
    t = re.sub(r'\\mathrm\{[^}]*\}', '', t)
    t = re.sub(r'\\\w+', '', t)
    # Remove bare numbers that are page/figure refs (3+ digits alone)
    t = re.sub(r'\b\d{3,}\b', '', t)
    # Fix "G- 1" → remove if it's a page ref (G-1 followed by G-2 pattern)
    t = re.sub(r'G-\s*\d+\.?\s*', '', t)
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    # Fix broken hyphens from line breaks
    t = re.sub(r'- (\w)', r'-\1', t)
    return t


def _is_valid_rgroup_option(option: str) -> bool:
    """Filter out junk R-group options (page refs, empty, non-chemical)."""
    o = option.strip().lower()
    if len(o) < 2:
        return False
    # Reject pure numbers (page refs)
    if re.match(r'^\d+$', o):
        return False
    # Reject "in another embodiment" etc.
    if any(kw in o for kw in ['embodiment', 'wherein', 'aspect', 'example', 'table', 'figure']):
        return False
    # Must contain at least one chemical term or be a known pattern
    chemical_terms = [
        'alkyl', 'aryl', 'hetero', 'cyclo', 'phenyl', 'methyl', 'ethyl',
        'propyl', 'butyl', 'fluoro', 'chloro', 'bromo', 'amino', 'hydroxy',
        'oxy', 'thio', 'sulfonyl', 'carbonyl', 'hydrogen', 'halogen', 'cyano',
        'morpholin', 'piperidin', 'pyrrolidin', 'pyridin', 'pyrimidin',
        'absent', 'h', 'f', 'cl', 'br', 'oh', 'nh',
    ]
    if any(term in o for term in chemical_terms):
        return True
    # Also accept if it looks like a chemical formula
    if re.search(r'[CNOS]\d|[a-z]-[a-z]', o):
        return True
    return False
