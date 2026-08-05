"""Discovered assay-row regex patterns — extracted by HARVEST/LLM and
applied deterministically across the rest of the patent text.

Lifecycle:
    pending  → freshly extracted from one chunk
    auto_loaded → ≥3 distinct patent fingerprints have used it
    promoted → curator approved (moves to canonical store)

Storage: `data/assay_patterns.discoveries.json` — mirror of the
`assay_vocabulary.discoveries.json` machinery so the same auto-promote
and curator-review tooling works.

Public API:
    add_pattern(regex, column_assays, example_match, patent_id)
        Append/update an entry. Returns True if newly added.
    apply_patterns_to_text(text, patent_id)
        Iterate auto_loaded + freshly-added patterns, run each across
        `text`, return a flat list of ActivityTuple-shaped dicts.
"""
from __future__ import annotations

import bisect
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from ..assay_name_guard import is_valid_assay_name

logger = logging.getLogger(__name__)

_PATTERNS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "assay_patterns.discoveries.json"
_PATTERN_PREVIEW_CHARS = 80     # what we log for traceability


def _read() -> dict:
    if not _PATTERNS_PATH.exists():
        return {"schema_version": "1.0", "tokens": []}
    try:
        return json.loads(_PATTERNS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"schema_version": "1.0", "tokens": []}


def _write(data: dict) -> None:
    _PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = str(date.today())
    _PATTERNS_PATH.write_text(json.dumps(data, indent=2))


def _canonicalize_regex(regex: str) -> str:
    """Normalize a regex to a canonical form so semantically-similar
    patterns produced by the LLM collide on the same fingerprint.

    Without this, LLM variations like ``\\+{1,4}`` vs ``\\+{1,5}`` get
    different SHA hashes, never accumulate 3 patent fingerprints, and
    never auto-promote — the library stays in "pending" forever even
    when every new patent re-discovers the same convention. With this,
    `(?P<cid>\\d+)\\s+(?P<value0>\\+{1,4})` and
    `(?P<cid>\\d{1,3})\\s+(?P<value0>\\+{1,5})` both reduce to the
    same canonical form and auto-promote on the 3rd patent.
    """
    canon = regex
    # Collapse digit-quantifier variants to a single form: \d+, \d{1,4},
    # \d{1,5} → \d+
    canon = re.sub(r"\\d\{[\d,]+\}", r"\\d+", canon)
    # Collapse grade-symbol quantifier variants: \+{1,4} / \+{1,5} →
    # \+{1,8}; same for \*
    canon = re.sub(r"\\(\+|\*)\{[\d,]+\}", r"\\\1{1,8}", canon)
    # Collapse whitespace runs: \s+, \s{1,4}, \s{2,} → \s+
    canon = re.sub(r"\\s\{[\d,]+\}", r"\\s+", canon)
    # Strip anchors (^ / $) — the same pattern with/without anchors is
    # the same table format.
    canon = canon.lstrip("^").rstrip("$")
    return canon


def _pattern_key(regex: str) -> str:
    """Stable identity for dedup, computed from the canonicalized regex
    so semantically-equivalent LLM variations collide."""
    import hashlib
    canon = _canonicalize_regex(regex)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def add_pattern(
    regex: str,
    column_assays: list[str],
    example_match: str,
    patent_id: str,
    header_text: str = "",
) -> bool:
    """Append or update a row-pattern entry. Returns True when newly added.

    `header_text` is the table's header line as the LLM saw it at discovery
    (e.g. "Example No.  CBP IC50 (μM gmean)  BRD4 IC50 (μM gmean)"). It is the
    pattern's ANCHOR: at fire time the pattern is applied to a patent only
    when this header (its ordered assay-column tokens, within a compact span)
    actually appears in that patent's text. This decouples the structural
    row-regex (reusable across patents) from the assay labels (which only
    apply where the matching header is locally present) — so a pattern's
    output depends solely on the text it's run against, deterministically,
    regardless of which patent discovered it. Legacy entries with no
    `header_text` fall back to an anchor derived from `column_assays`.
    """
    if not regex:
        return False
    try:
        re.compile(regex)
    except re.error as e:
        logger.warning("assay_pattern_library: rejected bad regex %r: %s", regex, e)
        return False

    # GUARD: a discovered pattern whose `column_assays` are LLM placeholders
    # (`Activity1`, `col_0`, `col1_IC50`, …) or non-assay column types
    # (`[M+H]`, `Method`, `Molecular Weight`, …) gets rejected at the source.
    # Without this guard the pattern auto-promotes to `auto_loaded` after 3
    # observations and starts injecting phantom assay rows into
    # assay_tables.json on every future patent (US11292791's 704 rows of
    # `Activity1 = 0.0` are exactly this leak).
    if not column_assays or not any(is_valid_assay_name(n) for n in column_assays):
        logger.warning(
            "assay_pattern_library: rejected pattern from %s — no valid assay "
            "names in column_assays=%r", patent_id, column_assays,
        )
        return False
    # We keep the original column_assays list intact (preserving the
    # regex's value-group indexing); partial placeholders are filtered out
    # at apply time below.

    data = _read()
    tokens: list[dict[str, Any]] = data.setdefault("tokens", [])
    key = _pattern_key(regex)
    existing = next((t for t in tokens if t.get("key") == key), None)
    if existing is None:
        tokens.append({
            "key": key,
            "regex": regex,
            "column_assays": list(column_assays),
            "header_text": (header_text or "")[:300],
            "example_match": example_match[:160],
            "status": "pending",
            "fingerprints_observed": [patent_id],
            "first_seen_patent": patent_id,
            "first_seen": str(date.today()),
            "n_observations": 1,
        })
        _write(data)
        logger.info(
            "assay_pattern_library: NEW pattern %s for %s — %r…",
            key, patent_id, regex[:_PATTERN_PREVIEW_CHARS],
        )
        return True
    # Update
    if patent_id not in existing.get("fingerprints_observed", []):
        existing.setdefault("fingerprints_observed", []).append(patent_id)
    existing["n_observations"] = existing.get("n_observations", 0) + 1
    # Backfill a header anchor if this observation supplied one and the
    # stored entry lacks it (older entries were discovered before headers
    # were captured).
    if header_text and not existing.get("header_text"):
        existing["header_text"] = header_text[:300]
    if (
        existing.get("status") == "pending"
        and len(existing["fingerprints_observed"]) >= 3
        and existing["n_observations"] >= 3
    ):
        existing["status"] = "auto_loaded"
        logger.info(
            "assay_pattern_library: pattern %s promoted to auto_loaded",
            key,
        )
    _write(data)
    return False


def _load_active_patterns() -> list[dict[str, Any]]:
    """All patterns with status in {pending, auto_loaded, promoted}.
    Pending entries are still applied — caller decides per-patent
    whether to trust them based on the fingerprints_observed list.
    """
    data = _read()
    return [
        t for t in data.get("tokens", [])
        if t.get("status") in ("pending", "auto_loaded", "promoted")
        and t.get("regex")
    ]


# ── Header-anchored firing: deterministic + patent-independent output ──
#
# A pattern's row-regex (`\d+\s+\d+\.\d+`) is purely structural and matches
# data rows in ANY patent; its `column_assays` are the labels the LLM read
# off the ORIGINATING table's header. Firing the regex broadly and trusting
# those labels is the contamination root cause (a US10273259 RORγ pattern
# matching a Nav-channel patent).
#
# The deterministic rule: a pattern's HEADER ANCHOR — the ordered, distinctive
# tokens of its column headers — must appear, in order and within a compact
# span, somewhere in the target patent's text. If it does, the patent contains
# that exact table and the labels are correct; if it doesn't, the pattern does
# not fire at all. The decision depends only on the text the pattern is run
# against, so output is identical regardless of which patent discovered the
# pattern or in what order the corpus was processed.

# Words/units that don't distinguish one assay from another.
_GENERIC_LABEL_TOKENS = {
    "the", "and", "for", "with", "value", "values", "mean", "gmean", "median",
    "ic", "ic50", "ec", "ec50", "ki", "kd", "cc50", "kd0",
    "um", "μm", "nm", "nm)", "mm", "m+h", "m+", "m-h",
    "min", "hr", "hrs", "sec",
    "binding", "activity", "potency", "assay", "test",
    "data", "result", "percent", "example", "compound", "number", "structure",
}
# A "salient" token is alphanumeric, ≥ 3 chars (or a digit-bearing identifier
# like "P2X3" / "NaV1.5" / "BRD4"), and not in the generic stoplist.
_TOKEN_RE = re.compile(r"[A-Za-zα-ωΑ-Ωγ][A-Za-zα-ωΑ-Ωγ0-9.]+", re.UNICODE)

# Patents commonly write spelled-out Greek ("RORgamma") while LLM headers use
# the letter ("RORγ"); transliterate so the two compare equal.
_GREEK_TO_LATIN = {
    "α":"alpha","β":"beta","γ":"gamma","δ":"delta","ε":"epsilon","ζ":"zeta",
    "η":"eta","θ":"theta","ι":"iota","κ":"kappa","λ":"lambda","μ":"mu",
    "ν":"nu","ξ":"xi","ο":"omicron","π":"pi","ρ":"rho","σ":"sigma","τ":"tau",
    "υ":"upsilon","φ":"phi","χ":"chi","ψ":"psi","ω":"omega",
}

# Max normalized-char span the header anchor's tokens may span — a real header
# line keeps its column names close together; tokens scattered across prose
# don't count as a header.
_HEADER_SPAN = 240


def _lc_greek(s: str) -> str:
    """Lowercase + Greek→Latin, but KEEP separators so word boundaries are
    preserved for anchor matching (the search side)."""
    return "".join(_GREEK_TO_LATIN.get(c, c) for c in s.lower())


def _normalize(s: str) -> str:
    """Lowercase + Greek→Latin + strip non-alphanumerics. Used to build the
    anchor tokens themselves (`NaV1.5` → `nav15`, `RORγ` → `rorgamma`)."""
    return re.sub(r"[^a-z0-9]", "", _lc_greek(s))


def _salient_tokens(label: str) -> list[str]:
    """Distinctive tokens (generic words removed) from a label or header."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(label or ""):
        if len(tok) < 3 or tok.lower() in _GENERIC_LABEL_TOKENS:
            continue
        out.append(tok)
    return out


def _header_anchor(entry: dict[str, Any]) -> list[str]:
    """Ordered, normalized, de-duplicated salient tokens that identify this
    pattern's table header. Prefer the captured `header_text`; fall back to
    the `column_assays` (their concatenation is effectively the header)."""
    src = entry.get("header_text") or " ".join(entry.get("column_assays") or [])
    seen: set[str] = set()
    anchor: list[str] = []
    for tok in _salient_tokens(src):
        n = _normalize(tok)
        if n and n not in seen:
            seen.add(n)
            anchor.append(n)
    return anchor


# ── Unit resolution ─────────────────────────────────────────────
#
# Both emit sites in `apply_patterns_to_text` used to write the literal
# `"unit": ""`. Measured on US9694016 (2026-08-04): 13,582 rows over 2,215
# compound ids, unit distribution `{'': 13582}` — Example 1's B-Raf IC50,
# which the patent prints as 0.000145 μM, replayed as a bare 0.0003.
#
# The discovered regex captures `cid` and `value0…valueN` only, so the unit
# has to come from the column header. Two sources, in order, and NEITHER is
# an inheritance rule: a unit is taken only from a header naming THIS column.
#   1. the column name itself — 53 of the library's 124 names carry it,
#      `P2X3 IC50 (μM)`;
#   2. this patent's own text — a header whose normalised form equals the
#      column name, with a unit printed against it: US9694016's TABLE 2
#      prints `b-Raf IC-50 (μM)` and the column is named `B-Raf IC50`.
# A block-level fallback ("nearest unit upstream") is deliberately NOT one of
# them; that is the rule that stamped nM onto US11254686's dimensionless
# Ratio column and produced 99 records reading 2.24 nM against BindingDB's
# 300 nM. A row whose scale cannot be named is dropped instead — the same
# call `AssayRecord.missing_fields()` makes, which nothing on this path
# applied before.

# Bare `M` and `nm` are deliberately NOT here. `M` matched the `M` of
# `MS (M+H)+` and stamped molar onto 1,619 US10273259 mass-spec rows; `nm` is
# a wavelength. Neither is worth the one real molar column it might cost.
_UNIT_LITERALS = r"nM|μM|uM|µM|mcM|mM|pM|fM|%|mg/mL|μg/mL|ug/mL|mL/min/kg|uL/min/mg"
# Concentration units only — the ones safe to read off a bare name tail.
# `M` and `%` are excluded there: every name ending in a capital M would
# otherwise claim molar.
_CONC_LITERALS = r"nM|μM|uM|µM|mcM|mM|pM|fM"
# The unit need not be the WHOLE bracket: `CBP IC50 (μM gmean)` is 2,874 rows
# on US11292791 and the qualifier is the aggregation, not the scale.
_NAME_UNIT_RE = re.compile(
    r"[\(\[][^)\]]*?(?<![A-Za-z])(" + _UNIT_LITERALS + r")(?![A-Za-z])[^)\]]*[\)\]]"
)
# …and it need not be bracketed at all: `RORγ Binding IC50 μM` is 3,307 rows
# across US10273259 + US20240010684A1.
_NAME_UNIT_TAIL_RE = re.compile(
    r"(?<![A-Za-z])(" + _CONC_LITERALS + r")\s*$"
)
# A header cell and the unit printed against it: up to 60 chars of label
# ending immediately before the bracketed unit, no newline crossing.
_HEADER_UNIT_RE = re.compile(
    r"([A-Za-zͰ-Ͽ][^\n\r]{0,60}?)\s*[\(\[]\s*("
    + _UNIT_LITERALS + r")\s*[\)\]]"
)


def _unit_from_name(assay_name: str) -> str:
    m = _NAME_UNIT_RE.search(assay_name or "")
    if m:
        return m.group(1)
    m = _NAME_UNIT_TAIL_RE.search(assay_name or "")
    return m.group(1) if m else ""


def _text_header_units(text: str) -> dict[str, str]:
    """{normalised header label → unit} for every `Label (unit)` in `text`.

    Keyed on the normalised label so `b-Raf IC-50` and the column name
    `B-Raf IC50` collapse to the same key — the two spellings of one column
    that kept 9,178 of US9694016's rows unitless.
    """
    out: dict[str, str] = {}
    for m in _HEADER_UNIT_RE.finditer(text):
        key = _normalize(m.group(1))
        if key and key not in out:
            out[key] = m.group(2)
    return out


def _resolve_unit(assay_name: str, header_units: dict[str, str]) -> str:
    unit = _unit_from_name(assay_name)
    if unit:
        return unit
    key = _normalize(re.sub(r"[\(\[].*", "", assay_name or ""))
    if not key:
        return ""
    if key in header_units:
        return header_units[key]
    # The header may carry a leading cell the column name doesn't
    # (`Cmpd  b-Raf IC-50 (μM)`), so accept a header that ENDS with it.
    for hkey, unit in header_units.items():
        if hkey.endswith(key):
            return unit
    return ""


def _token_subpattern(tok: str) -> str:
    """Regex for one normalized anchor token, matched on WORD BOUNDARIES with
    internal separators allowed. `raf` matches `Raf`/`RAF` but NOT the `raf`
    inside `draft`/`craft`; `nav15` matches `NaV 1.5`/`NaV1.5`/`Nav-1.5`. The
    boundary guard is what stops a short token from substring-matching inside
    an unrelated word — the B-Raf→`raf` cross-patent leak."""
    inner = r"[^a-z0-9]*".join(re.escape(c) for c in tok)
    return r"(?<![a-z0-9])" + inner + r"(?![a-z0-9])"


def _anchor_regex(anchor: list[str]) -> "re.Pattern[str]":
    """Compiled regex requiring the anchor tokens in order, each on word
    boundaries, within `_HEADER_SPAN` chars of each other."""
    gap = r"[\s\S]{0,%d}?" % _HEADER_SPAN
    return re.compile(gap.join(_token_subpattern(t) for t in anchor))


# `_anchor_present` used to live here, answering "does this anchor occur at
# all". `_anchor_spans` below answers "and WHERE", which the locality gate
# needs and which subsumes the boolean — an empty span list is an absent
# anchor, and still means the pattern does not fire.


# ── The anchor is not enough on its own: provenance-scoped firing ──
#
# The comment above claims a pattern's output "depends solely on the text it's
# run against". That was the design; the anchor as built is too weak to deliver
# it, and the gap is a fabricated MEANING, not a fabricated number.
#
# Measured 2026-08-04 over the 22-patent corpus (117,739 pattern-library rows):
#   * 0 of 116 library entries carry a real `header_text`, so EVERY anchor is
#     the degraded `column_assays` fallback in `_header_anchor`;
#   * 30 of those anchors are a SINGLE token, because `_GENERIC_LABEL_TOKENS`
#     correctly strips `binding`/`ic50` and `_salient_tokens` drops `μM` as
#     too short — `RORγ Binding IC50 μM` reduces to `['rorgamma']`;
#   * US20240010684A1 is a MASP-1/MASP-2 complement patent (226 mentions of
#     "MASP", ONE of "RORgamma", in a list of unrelated therapeutics). That one
#     prose word opened the gate for US10273259's pattern and its structural
#     regex — `(?P<cid>\d+)\s+(?P<value0>\d+\.\d+)`, which matches "integer,
#     then decimal" in any document — then ran document-wide for **593 rows**
#     of `RORγ Binding IC50 μM`, every one of them read out of HPLC method
#     prose: `SB-C18 2.7 μm` → `compound_id 18, RORγ Binding IC50 = 2.7 μM`.
#
# `output_validator` cannot see this. It corroborates `(cid, value)` and never
# looks at `assay_name`, so a foreign label pinned to the host's own adjacent
# numbers corroborates trivially. The number is real; the target is invented.
#
# Two properties separate that from the reuse the library exists for. Compare
# it with US8952177 → US9745328, which is correct: US9745328 prints
# `FLAP Binding wild type HTRF … Human Whole Blood LTB4 IC 50` itself, over its
# own TABLE 5 and TABLE 6.
#
#   anchor size    leak 1 token          legitimate 8 tokens
#   locality       min 30,782 chars      max 17,292 chars from the anchor
#                  from its anchor
#
# So a foreign pattern must clear both: an anchor of ≥2 salient tokens (one
# word of boilerplate is not a table header), and rows that actually sit under
# that header rather than elsewhere in the document.
#
# Why this is scoped to FOREIGN patterns — entries whose `first_seen_patent`
# is some other document. Applying either test to native patterns as well was
# measured and costs the corpus its extraction:
#
#   gate (applied to ALL patterns)      native rows kept (of 113,046)
#   anchor ≥ 2 tokens                    59,769   (−53,277)
#   anchor ≥ 3 tokens                    51,168   (−61,878)
#   locality ≤ 20k                       72,736   (−40,310)
#   both, foreign only                  113,046   (unchanged)
#
# A native pattern's labels came off THIS patent's own header, so a mislocated
# native row is a within-patent misassignment — a real and larger defect (the
# locality figures above are its size), but a different one, and not something
# to fix by silently deleting a third of the corpus's rows. Consulting
# `first_seen_patent` re-introduces the provenance dependence the design tried
# to avoid; that is the honest trade until `header_text` is captured at
# discovery, which is the fix that would let the gate be text-only again.

# A single salient token is boilerplate, not a header. Blocks 593/593 leaked
# rows; costs 0 of the 3,048 legitimate US9745328 rows (8-token anchor).
_FOREIGN_MIN_ANCHOR_TOKENS = 2
# A row may sit this far downstream of the header that names it. 20,000 chars
# covers US9745328's TABLE 5 (15,365 chars) with room to spare — its furthest
# legitimate row is 17,292 away — while the nearest leaked row is 30,782 away.
_FOREIGN_ANCHOR_LOCALITY = 20_000
# …and a little upstream, for the caption/units line printed after the header
# cells but before the first data row.
_FOREIGN_ANCHOR_BACKREACH = 200


def _lc_greek_indexed(text: str) -> tuple[str, tuple[list[int], list[int]] | None]:
    """`_lc_greek(text)` plus the map back to offsets in `text`.

    The anchor is searched in Greek-expanded space (`γ` → `gamma`), so an
    offset there is NOT an offset in the source — it drifts by 4 chars per
    Greek letter seen so far, and patent text is full of `μM`. The locality
    gate compares anchor positions against `re.Match.start()` offsets in the
    raw text, so the two have to be reconciled or the comparison is nonsense.

    Returns `(lc_text, None)` when nothing expanded (the common case, no cost),
    else `(lc_text, (starts, shifts))` for `_to_source_offset` to bisect.
    """
    parts: list[str] = []
    starts: list[int] = []
    shifts: list[int] = []
    j = 0
    shift = 0
    for ch in text:
        low = ch.lower()
        rep = _GREEK_TO_LATIN.get(low, low)
        if len(rep) != 1:
            starts.append(j)
            shift += len(rep) - 1
            shifts.append(shift)
        parts.append(rep)
        j += len(rep)
    lc = "".join(parts)
    return lc, ((starts, shifts) if starts else None)


def _to_source_offset(off: int, index: tuple[list[int], list[int]] | None) -> int:
    """Offset in `_lc_greek` space → offset in the source text."""
    if index is None:
        return off
    starts, shifts = index
    k = bisect.bisect_right(starts, off) - 1
    return off - (shifts[k] if k >= 0 else 0)


def _anchor_spans(
    lc_text: str,
    index: tuple[list[int], list[int]] | None,
    anchor: list[str],
) -> list[tuple[int, int]]:
    """Every place the header anchor occurs, as source-text (start, end)."""
    if not anchor:
        return []
    return [
        (_to_source_offset(m.start(), index), _to_source_offset(m.end(), index))
        for m in _anchor_regex(anchor).finditer(lc_text)
    ]


def _is_foreign(entry: dict[str, Any], patent_id: str) -> bool:
    """True when this pattern's labels were read off a DIFFERENT patent's
    header. An entry with no `first_seen_patent` is a `fresh_patterns` item —
    discovered on this patent, during this run, by the call that just paid for
    it — and is native.
    """
    src = (entry.get("first_seen_patent") or "").strip().upper()
    return bool(src) and src != (patent_id or "").strip().upper()


def _under_anchor(offset: int, spans: list[tuple[int, int]]) -> bool:
    """True when `offset` falls under one of the anchor's occurrences."""
    return any(
        start - _FOREIGN_ANCHOR_BACKREACH <= offset <= end + _FOREIGN_ANCHOR_LOCALITY
        for start, end in spans
    )


def apply_patterns_to_text(
    text: str,
    patent_id: str,
    *,
    fresh_patterns: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run every active pattern across `text`. Returns a list of
    ActivityTuple-shaped dicts (compound_id, assay_name, value, unit,
    qualifier, n_runs, source_quote, source_offset, confidence).

    Args:
        text: full patent text (any source).
        patent_id: for source_quote tagging in returned dicts.
        fresh_patterns: patterns just-discovered this run — included
            alongside library patterns so this run benefits immediately
            (auto_loaded promotion takes effect on the NEXT run, but
            fresh patterns should still apply NOW for cost amortization).
    """
    if not text:
        return []
    active = _load_active_patterns()
    if fresh_patterns:
        # Avoid double-counting patterns that are both fresh AND already
        # in the library: key on regex hash.
        existing_keys = {t["key"] for t in active if "key" in t}
        for fp in fresh_patterns:
            if not isinstance(fp, dict):
                continue
            rx = fp.get("regex") or ""
            k = _pattern_key(rx)
            if k in existing_keys:
                continue
            # Stamp provenance: a fresh pattern was discovered on THIS patent
            # by the run that is now applying it, so `_is_foreign` must not
            # subject it to the cross-patent gate. `add_pattern` records the
            # same value in the library; the caller's dict does not carry it.
            active.append({"first_seen_patent": patent_id, **fp, "key": k})
            existing_keys.add(k)

    # Lowercase + Greek-normalize ONCE (keeping separators so the anchor's
    # word-boundary check works). The header-anchor check runs against this
    # per pattern — the deterministic gate: a pattern fires on this patent
    # only if its header is present here. `lc_index` maps positions in that
    # expanded text back to `text`, for the foreign-pattern locality gate.
    lc_text, lc_index = _lc_greek_indexed(text)
    # Built once per call: the patent's own `Label (unit)` headers.
    header_units = _text_header_units(text)

    out: list[dict[str, Any]] = []
    seen_per_pattern: dict[str, int] = {}
    n_anchor_skipped = 0
    n_foreign_thin_anchor = 0
    n_foreign_far = 0
    n_no_unit = 0
    unit_cache: dict[str, str] = {}
    for entry in active:
        rx = entry.get("regex") or ""
        column_assays = entry.get("column_assays") or []
        # HEADER-ANCHOR GATE (per pattern): the pattern's header must appear
        # in THIS patent's text. If it doesn't, the row-regex may still match
        # (it's generic) but the labels don't belong here — skip the whole
        # pattern. This was intended to make output depend only on the local
        # text; with no entry carrying a real `header_text` it decides only
        # "some word from the header occurs somewhere", which is why the
        # foreign gate below exists.
        anchor = _header_anchor(entry)
        spans = _anchor_spans(lc_text, lc_index, anchor)
        if not spans:
            n_anchor_skipped += 1
            continue
        # FOREIGN-PATTERN GATE: these labels were read off another patent's
        # header, so they have to earn their way in. See the block above
        # `_FOREIGN_MIN_ANCHOR_TOKENS` for the measurements.
        foreign = _is_foreign(entry, patent_id)
        if foreign and len(anchor) < _FOREIGN_MIN_ANCHOR_TOKENS:
            n_foreign_thin_anchor += 1
            continue
        try:
            # MULTILINE: discovered row-regexes are written to match "a single
            # data row" and frequently carry ^…$ anchors meaning line-start /
            # line-end. Without MULTILINE those anchors bind to string
            # start/end and the pattern silently never fires on a multi-row
            # table — dead recall. MULTILINE makes them fire per line.
            compiled = re.compile(rx, re.MULTILINE)
        except re.error:
            continue
        n_for_pattern = 0
        for m in compiled.finditer(text):
            cid = (m.groupdict().get("cid") or "").strip()
            if not cid:
                continue
            # A foreign label belongs to the table its header heads, not to
            # every "integer, then decimal" in the document. This is what
            # separates US9745328's rows (all within 17,292 chars of the
            # header they inherit) from US20240010684A1's (all at least
            # 30,782 chars away, in the HPLC methods section).
            if foreign and not _under_anchor(m.start(), spans):
                n_foreign_far += 1
                continue
            # value0, value1, …
            for i, assay in enumerate(column_assays):
                # GUARD: skip placeholder (Activity1, col_0) / non-assay
                # ([M+H], Method, Molecular Weight) column names — defence
                # in depth with the discovery-time guard in add_pattern.
                if not is_valid_assay_name(assay):
                    continue
                grp = f"value{i}"
                raw_val = (m.groupdict().get(grp) or "").strip()
                if not raw_val or raw_val.lower() in ("nt", "nd", "—", "-"):
                    continue
                if assay not in unit_cache:
                    unit_cache[assay] = _resolve_unit(assay, header_units)
                unit = unit_cache[assay]
                if not unit:
                    # No scale, no measurement. See the note above
                    # `_UNIT_LITERALS` for why no block-level fallback.
                    n_no_unit += 1
                    continue
                # numeric or letter-grade?
                try:
                    v_num = float(raw_val)
                    out.append({
                        "compound_id": cid,
                        "assay_name": assay,
                        "value": v_num,
                        "unit": unit,
                        "qualifier": None,
                        "n_runs": None,
                        "source_quote": m.group(0)[:80],
                        "source_offset": m.start(),
                        "confidence": "medium",
                        "validation_reason": f"pattern_library:{entry.get('key','')}",
                    })
                    n_for_pattern += 1
                except ValueError:
                    # Letter grade — keep as categorical (value=None;
                    # the workbook surfaces these distinctly).
                    out.append({
                        "compound_id": cid,
                        "assay_name": assay,
                        "value": None,
                        "value_categorical": raw_val,
                        "unit": unit,
                        "qualifier": None,
                        "n_runs": None,
                        "source_quote": m.group(0)[:80],
                        "source_offset": m.start(),
                        "confidence": "medium",
                        "validation_reason": f"pattern_library:{entry.get('key','')}:letter_grade",
                    })
                    n_for_pattern += 1
        if n_for_pattern:
            seen_per_pattern[entry.get("key", "?")] = n_for_pattern
    if n_anchor_skipped:
        logger.info(
            "assay_pattern_library: %s — %d pattern(s) skipped "
            "(header anchor absent in this patent's text; their tables "
            "are not present here)", patent_id, n_anchor_skipped,
        )
    if n_foreign_thin_anchor or n_foreign_far:
        logger.info(
            "assay_pattern_library: %s — cross-patent gate blocked %d "
            "pattern(s) on a single-token anchor and %d row-match(es) sitting "
            "outside the anchored region",
            patent_id, n_foreign_thin_anchor, n_foreign_far,
        )
    if seen_per_pattern:
        logger.info(
            "assay_pattern_library: %s — %d rows extracted via patterns: %s",
            patent_id, sum(seen_per_pattern.values()), seen_per_pattern,
        )
    return out
