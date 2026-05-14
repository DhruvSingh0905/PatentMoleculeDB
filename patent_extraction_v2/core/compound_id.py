"""Single source of truth for compound-id parsing across the codebase.

Why this exists: every time a new patent uses a different prefix
("Cpd. No 19", "Compound 5A", "Cmp. No. 12", "Example 1", "Ex. 7")
we keep adding ad-hoc regexes in different places. That's exactly
the anti-pattern CLAUDE.md warns against. This module reads the
canonical prefix list from `data/assay_vocabulary.json` (the
`COMPOUND_ID_PREFIX` entry) and exposes ONE parser function +
ONE compiled regex that every extractor imports.

Surface forms covered (loaded from vocabulary at import time):
  Cpd, Cpd., Cpd. No, Cpd. No.
  Compound, Compound No, Compound No.
  Cmp, Cmp., Cmp No, Cmp No., Cmp. No, Cmp. No.
  Example, Example No, Example No.
  Ex, Ex.

Adding a new surface form is a vocabulary JSON edit, not a code
change. Anywhere in the codebase that needs to detect "the patent
mentions compound X here" should call `parse_compound_id` or
`COMPOUND_ID_RE`.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


_VOCAB_PATH = Path(__file__).resolve().parent.parent / "data" / "assay_vocabulary.json"


@lru_cache(maxsize=1)
def _load_prefixes() -> list[str]:
    """Read COMPOUND_ID_PREFIX surface forms from canonical vocab."""
    if not _VOCAB_PATH.exists():
        return ["Cpd", "Cpd.", "Compound", "Example", "Ex", "Ex.",
                "Cmp", "Cmp.", "Cmp No.", "Cmp. No.", "Cpd. No.",
                "Compound No.", "Example No."]
    data = json.loads(_VOCAB_PATH.read_text())
    out = []
    for entry in data.get("tokens", []):
        if entry.get("class") == "COMPOUND_ID_PREFIX":
            out.extend(entry.get("surface_forms", []))
    # Sort longest-first so "Compound No." matches before "Compound"
    return sorted(set(out), key=lambda s: (-len(s), s))


@lru_cache(maxsize=1)
def _build_compound_id_regex() -> re.Pattern[str]:
    """One regex that matches any known compound-id prefix + an id.

    The compound id is captured in named group `cid` and supports:
      - bare digits: 1, 42, 113
      - letter-suffix stereoisomers: 5A, 5B, 38A, 92AA, 167AB, 92aa
    Whitespace is forgiving — accepts no-space, single-space, or
    multi-space between prefix and id.
    """
    prefixes = _load_prefixes()
    # Escape and join. Allow optional trailing period/space variations.
    # E.g., "Cpd. No." should also match "Cpd. No" or "Cpd No." or "Cpd No"
    prefix_alts = []
    for p in prefixes:
        # Normalize input → escape for regex, then make `.` and ` ` flexible
        esc = re.escape(p)
        # Make periods optional
        esc = esc.replace(r"\.", r"\.?")
        # Make spaces match any whitespace run
        esc = esc.replace(r"\ ", r"\s+")
        prefix_alts.append(esc)
    prefix_re = "|".join(prefix_alts)
    pattern = (
        rf"(?<![A-Za-z0-9.])"
        rf"(?:{prefix_re})\s+"
        rf"(?P<cid>\d{{1,4}}[A-Za-z]{{0,4}})"
        rf"(?![A-Za-z0-9])"
    )
    return re.compile(pattern, re.IGNORECASE)


@lru_cache(maxsize=1)
def _build_paren_compound_regex() -> re.Pattern[str]:
    """Match the reverse-parenthetical form: `(Compound 5A)` / `(Cpd. No 19)`.

    Used when an IUPAC name precedes a parenthetical compound-id, e.g.:
        "Synthesis of <IUPAC> (Compound 5A) and <IUPAC> (Compound 5B)"
    """
    prefixes = _load_prefixes()
    prefix_alts = []
    for p in prefixes:
        esc = re.escape(p).replace(r"\.", r"\.?").replace(r"\ ", r"\s+")
        prefix_alts.append(esc)
    prefix_re = "|".join(prefix_alts)
    return re.compile(
        rf"\(\s*(?:{prefix_re})\s+(?P<cid>\d{{1,4}}[A-Za-z]{{0,4}})\s*\)",
        re.IGNORECASE,
    )


COMPOUND_ID_RE = _build_compound_id_regex()
COMPOUND_ID_PAREN_RE = _build_paren_compound_regex()


def parse_compound_id(text: str) -> str | None:
    """Return the FIRST compound-id found in `text` using any known
    prefix surface form. Returns the id string (e.g. "5A") or None.

    Use this anywhere you need to "find the compound mentioned in
    this string" — synthesis prose, BDB ligand-name suffixes,
    table cells, etc.

    Examples (all return "5A"):
        "Compound 5A"
        "Cpd. No. 5A"
        "Cmp 5A"
        "Example No. 5A"
        "(Compound 5A)"   — handled by `parse_compound_id_paren`
    """
    m = COMPOUND_ID_RE.search(text or "")
    return m.group("cid") if m else None


def parse_compound_id_paren(text: str) -> str | None:
    """Return the compound-id from a parenthetical mention `(Compound 5A)`."""
    m = COMPOUND_ID_PAREN_RE.search(text or "")
    return m.group("cid") if m else None


def find_all_compound_ids(text: str) -> list[tuple[int, str]]:
    """Return every (offset, compound_id) match in `text`."""
    return [(m.start(), m.group("cid")) for m in COMPOUND_ID_RE.finditer(text or "")]


# ── Canonical key normalization ────────────────────────────────────
# Used wherever code needs to convert a raw compound-id string (from
# HARVEST output, table cells, regex captures, Compound.example_number)
# into a stable dict key. ONE convention shared across the pipeline:
#
#   - HTML markup stripped
#   - Known prefixes ("Compound ", "Cpd. No. ", "Example ", etc.) removed
#   - Trailing "Step X" qualifiers preserved as-is (sub-steps are distinct)
#   - Whitespace collapsed
#   - Letter suffix UPPERCASED (patent convention: "5A" not "5a")
#   - Numeric stem preserved as-is
#
# Returns None if the result doesn't look like a valid compound-id
# (e.g. pure-letter strings, empty after stripping). Callers that need
# to keep an unparseable raw string (e.g. "Intermediate A") should fall
# back to the raw value when this returns None.

import html as _html

_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Match the canonical key shape: optional letter prefix + 1-5 digits + 0-3 letter suffix.
_VALID_KEY_RE = re.compile(r"^[A-Za-z]*\d{1,5}[A-Za-z]{0,3}$")


@lru_cache(maxsize=1)
def _build_prefix_strip_regex() -> re.Pattern[str]:
    """Anchored regex that strips any known compound-id prefix from a string.
    Built from the same vocabulary as `_build_compound_id_regex` so adding
    a new surface form to the JSON updates BOTH parser and key-normalizer.

    Whitespace WITHIN the prefix is optional (`\\s*`) so "Cpd.No. 5" is
    handled the same as "Cpd. No. 5". Trailing whitespace between prefix
    and the id is stripped by the caller.
    """
    prefixes = _load_prefixes()
    alts = []
    for p in prefixes:
        esc = re.escape(p).replace(r"\.", r"\.?").replace(r"\ ", r"\s*")
        alts.append(esc)
    return re.compile(rf"^(?:{'|'.join(alts)})\s*", re.IGNORECASE)


def normalize_cid_key(raw: str) -> str | None:
    """Canonical compound-id key normalizer.

    Returns the cleaned key (e.g. "5A", "172"), or None if the input
    can't be reduced to a valid id shape. Use this to produce dict keys
    for `example_index` / `assay_tables` so all writers agree.

    Examples:
        "Compound 5A"      → "5A"
        "Cpd. No. 172"     → "172"
        "<td>5</td>"       → "5"
        "Example No. 92AA" → "92AA"
        "5a"               → "5A"
        ""                 → None
        "Intermediate A"   → None  (no digits)
    """
    if not raw:
        return None
    s = _html.unescape(raw)
    s = _HTML_TAG_RE.sub("", s).strip()
    s = _build_prefix_strip_regex().sub("", s).strip()
    s = s.replace(" ", "")
    if not s or not _VALID_KEY_RE.match(s):
        return None
    # Uppercase the letter suffix (and any letter-prefix) so "5a" and
    # "5A" collapse — patent convention is uppercase.
    return s.upper()
