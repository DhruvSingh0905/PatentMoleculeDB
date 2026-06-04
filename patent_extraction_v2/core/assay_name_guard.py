"""Validation guard for assay names entering the pipeline.

Why this exists
---------------
Without this guard the LLM (in `realign_table_via_llm` / IUPAC pattern
discovery) is free to invent placeholder column names — `Activity1`,
`col_0`, `col1_IC50` — and to take non-assay table headers verbatim —
`[M+H]`, `Method`, `HPLC tR`, `Molecular Weight`. Those names then:

  1. Get persisted into `data/assay_patterns.discoveries.json` and
     auto-promote after 3 observations (no name validation in
     `IupacPatternLibrary.add_discovery`).
  2. Get cached by `AdaptiveExtractionCache` under a content-blind
     "structural" fingerprint, where the *cached row data carries
     compound IDs and values from the patent that produced the rule* —
     so any other patent with a structurally-similar table inherits
     them. (See `adaptive_extraction_rules.json`: 229 rule entries from
     14 different patents, 2,129 entries with `first_observed = "?"`.)
  3. Get emitted into `assay_tables.json` as `assay_name`. Downstream
     consumers (the hand-check workbook's `_v2_canonical_assays`) then
     pick the most-frequent placeholder as a "primary assay" column —
     which is how `US11292791` ended up with `activity1_uM = 0` on every
     row and 0/660 value-matches against BDB.

The single guard `is_valid_assay_name()` is meant to be called at every
LLM-output ingestion point so the placeholder/non-assay leak is closed at
the source, not whitewashed downstream.
"""
from __future__ import annotations

import re

# Placeholder names the LLM emits when it can't identify the real assay.
# The pattern library / cache used to accept these wholesale; here we
# reject them — better an empty row than a phantom 0-valued assay.
_PLACEHOLDER_PATTERNS = [
    re.compile(r"^Activity\d+$", re.IGNORECASE),                # Activity1, Activity2
    re.compile(r"^Activity\d+\s*[_-]?\s*\w*$", re.IGNORECASE),   # Activity1_grade
    re.compile(r"^activity[_-]?col\d+$", re.IGNORECASE),         # activity_col1
    re.compile(r"^col[_-]?\d+$", re.IGNORECASE),                 # col_0, col0, col-0
    re.compile(r"^col\d+[_-]\w+$", re.IGNORECASE),               # col1_IC50, col1_grade
    re.compile(r"^column\s*\d+$", re.IGNORECASE),                # Column 1
    re.compile(r"^value\s*\d+$", re.IGNORECASE),                 # Value 1
    re.compile(r"^assay\s*\d+$", re.IGNORECASE),                 # Assay 1
    re.compile(r"^unnamed[_:\s-]?\d*$", re.IGNORECASE),          # Unnamed, Unnamed_0
    # OR-disjunction placeholders the LLM emits when it can't decide
    # between two column candidates: "IC50 or Activity", "EC50 or potency".
    # Real assays never use the word "or" in their name.
    re.compile(r"^\w+\s*[_-]?\s*or\s*[_-]?\s*\w+$", re.IGNORECASE),
    re.compile(r"^(ic50|ec50|cc50|ki|kd)[_\s]+or[_\s]+\w+$", re.IGNORECASE),
]

# Table-header content that is NOT an assay readout — mass-spec masses,
# HPLC method labels / retention times, MW, etc. Patents print these
# alongside assays; the LLM-realigner prompt's "use the column header
# verbatim" rule lets them through. We reject them centrally.
_NON_ASSAY_PATTERNS = [
    re.compile(r"^\[?m\s*[+\-]\s*h\]?\+?$", re.IGNORECASE),       # [M+H], M+H, [M-H], M-H, [M+H]+
    re.compile(r"^m\s*\+\s*1\s*$", re.IGNORECASE),                # M+1
    re.compile(r"^method$", re.IGNORECASE),                       # Method (HPLC method)
    re.compile(r"^hplc(\s+\w+)*$", re.IGNORECASE),                # HPLC method, HPLC tR
    re.compile(r"^lcms?(\s+\w+)*$", re.IGNORECASE),               # LCMS, LCMS method, LCMS m/z
    re.compile(r"^ms(\s+\w+)*$", re.IGNORECASE),                  # MS, MS m/z
    re.compile(r"^m/z$", re.IGNORECASE),                          # m/z
    re.compile(r"^molecular\s+weight$", re.IGNORECASE),           # Molecular Weight
    re.compile(r"^mw$", re.IGNORECASE),                           # MW
    re.compile(r"^mol\.?\s*wt\.?$", re.IGNORECASE),               # Mol Wt, Mol. Wt.
    re.compile(r"^mass$", re.IGNORECASE),                         # Mass
    re.compile(r"^exact\s+mass$", re.IGNORECASE),                 # Exact Mass
    re.compile(r"^(retention\s+time|rt|tr)(\s*\(min\))?$", re.IGNORECASE),  # Retention Time, RT, tR (min)
    re.compile(r"^(hplc\s+)?tr(\s*\(min\))?$", re.IGNORECASE),    # HPLC tR (min), tR
    re.compile(r"^purity(\s*\(%\))?$", re.IGNORECASE),            # Purity, Purity (%)
    re.compile(r"^yield(\s*\(%\))?$", re.IGNORECASE),             # Yield
    re.compile(r"^logp$", re.IGNORECASE),                         # LogP
    re.compile(r"^clogp$", re.IGNORECASE),                        # cLogP
    re.compile(r"^tpsa$", re.IGNORECASE),                         # TPSA
    re.compile(r"^calc\.?\s*mass$", re.IGNORECASE),               # Calc Mass
    re.compile(r"^found$", re.IGNORECASE),                        # Found (MS data column)
]


# A name that explicitly carries a real readout type is NOT a placeholder,
# even in an "X or Y" form. "IC50 or activity" names the IC50 readout — the
# LLM is offering an alias, not hedging between two unknown columns. (Dropping
# "IC50_or_activity" cost US11254686 its 639 real primary-IC50 values.)
_READOUT_RE = re.compile(
    r"\b(ic50|ec50|cc50|gi50|ed50|ld50|ki50|ki|kd|kb|pic50|pec50)\b",
    re.IGNORECASE,
)


def is_placeholder_name(name: str) -> bool:
    """True iff the name matches a known LLM placeholder pattern. These
    are names the LLM invents when it can't identify the real assay —
    `Activity1`, `col_0`, `col1_IC50`, etc. An OR-disjunction that names a
    real readout (`IC50 or activity`) is exempt — it's a real assay column."""
    if not name:
        return False
    s = name.strip()
    if not any(p.match(s) for p in _PLACEHOLDER_PATTERNS):
        return False
    # Exempt readout-bearing OR-names: the readout IS identified. Normalize
    # underscores to spaces first so the readout's `\b` boundary matches
    # ("IC50_or_activity" → "ic50 or activity").
    norm = f" {s.lower().replace('_', ' ')} "
    if " or " in norm and _READOUT_RE.search(norm):
        return False
    return True


def is_non_assay_name(name: str) -> bool:
    """True iff the name is a non-assay table-column type — MS mass,
    HPLC method/retention, molecular weight, purity, yield, etc."""
    if not name:
        return False
    s = name.strip()
    if any(p.match(s) for p in _NON_ASSAY_PATTERNS):
        return True
    # Substring markers that are NEVER part of a real biological assay name —
    # mass-spec / NMR / chromatography characterization columns whose full
    # text varies too much for an anchored regex ("MS m/z 1st eluting isomer",
    # "1H NMR (CDCl3)", "LRMS m/z [M+H]+", "HPLC tR (min)"). If the name
    # contains any of these, it's characterization data, not an assay readout.
    low = s.lower()
    return any(mark in low for mark in _NON_ASSAY_SUBSTRINGS)


# Characterization markers — presence anywhere in the name disqualifies it.
# These are analytical/QC columns patents print next to assays, not bioassay
# readouts. Targeted multi-word forms ("qc method", "analytical method") avoid
# over-rejecting legit technique-named assays (e.g. "FRET", "HTRF").
_NON_ASSAY_SUBSTRINGS = (
    "m/z", "nmr", "eluting", "lcms", "lrms", "hrms", "[m+h]", "[m-h]",
    "retention time", "rt (min)", "tr (min)", "exact mass", "molecular weight",
    "qc method", "analytical method", "hplc method", "lc method", "purity",
)


def is_valid_assay_name(name: str | None) -> bool:
    """The single guard: True iff `name` is plausibly a real biological
    assay readout (kept in the pipeline) — i.e. it's a non-empty string
    that's neither an LLM placeholder nor a known non-assay column.

    Callers (LLM-realigner output, IUPAC pattern discovery, FSM→AssayResult
    conversion, AdaptiveExtractionCache.put, workbook canonical-assay
    selection) MUST filter through this before persisting / surfacing the
    name. Returning False is non-destructive — the row is just dropped
    rather than emitted under a placeholder name.
    """
    if not name:
        return False
    s = str(name).strip()
    if not s:
        return False
    if is_placeholder_name(s):
        return False
    if is_non_assay_name(s):
        return False
    return True


def filter_valid_assay_names(names) -> list[str]:
    """Convenience wrapper: filter an iterable of names down to the valid ones."""
    return [n for n in names if is_valid_assay_name(n)]
