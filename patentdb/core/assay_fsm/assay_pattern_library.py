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


def _anchor_present(lc_text: str, anchor: list[str]) -> bool:
    """True iff the header anchor appears (tokens in order, word-bounded,
    within `_HEADER_SPAN`) in `lc_text` (lowercased + Greek-normalized, with
    separators kept). Empty anchor → False (never fire a pattern we can't
    anchor)."""
    if not anchor:
        return False
    return _anchor_regex(anchor).search(lc_text) is not None


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
            active.append({**fp, "key": k})
            existing_keys.add(k)

    # Lowercase + Greek-normalize ONCE (keeping separators so the anchor's
    # word-boundary check works). The header-anchor check runs against this
    # per pattern — the deterministic gate: a pattern fires on this patent
    # only if its header is present here.
    lc_text = _lc_greek(text)

    out: list[dict[str, Any]] = []
    seen_per_pattern: dict[str, int] = {}
    n_anchor_skipped = 0
    for entry in active:
        rx = entry.get("regex") or ""
        column_assays = entry.get("column_assays") or []
        # HEADER-ANCHOR GATE (per pattern, deterministic): the pattern's
        # header must appear in THIS patent's text. If it doesn't, the
        # row-regex may still match (it's generic) but the labels don't
        # belong here — skip the whole pattern. This is what makes the
        # library's output depend only on the local text, never on which
        # patent the pattern came from.
        anchor = _header_anchor(entry)
        if not _anchor_present(lc_text, anchor):
            n_anchor_skipped += 1
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
                # numeric or letter-grade?
                try:
                    v_num = float(raw_val)
                    out.append({
                        "compound_id": cid,
                        "assay_name": assay,
                        "value": v_num,
                        "unit": "",
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
                        "unit": "",
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
    if seen_per_pattern:
        logger.info(
            "assay_pattern_library: %s — %d rows extracted via patterns: %s",
            patent_id, sum(seen_per_pattern.values()), seen_per_pattern,
        )
    return out
