"""Adaptive LLM-populated cache of table extraction rules.

Why this exists
---------------
Static regex parsers handle predictable table formats well, but patents
introduce new formats every time a new pharma family is filed. Examples
of formats that break baseline parsers:

  - Header column "CC50" recognized only as a chemistry term, not an
    assay (US11312727 p220 — silently dropped, GT coverage 56%)
  - Side-by-side two-up tables where the parser only reads the left half
  - Bleed-through OCR where atom labels (OH, O) split data rows
  - Header rows wrapped across two lines

Per the rebuild plan: "Adaptive caches over hand-curated rules. When a
new edge case shows up, fire a single low-cost LM call to learn the
mapping, cache it, never ask the LM again."

This module:
  1. Detects when the baseline parser likely UNDER-extracted (column
     mismatch, row count anomaly, missing assay keywords)
  2. Builds a stable fingerprint from the page's header + structural
     shape (NOT patent-specific content)
  3. Looks up the fingerprint in cache; if hit, returns the cached rule
  4. If miss, fires ONE LM call to derive a structured rule
  5. Validates the rule (must produce ≥1 row, valid numerics)
  6. Caches the rule keyed on fingerprint — reusable across patents
     with the same table shape

Generic — same logic runs on every patent. The cache key is structural
(header tokens, column count, row patterns), not content (no patent IDs,
no compound names).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from . import config
from .api_client import call_claude_text

logger = logging.getLogger(__name__)


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass
class AssayColumn:
    """One assay column derived by the LLM from page text."""
    header_text: str               # raw header as it appears on the page
    canonical_name: str            # e.g., "IC50", "EC50", "CC50", "FRET_IC50"
    unit: str                      # "μM", "nM", "M", "%"
    col_position: int = -1         # 0-indexed column position (when known)


@dataclass
class ExtractionRule:
    """Structured rule for parsing a specific table format.

    Produced by the LLM when the baseline regex parser misses columns
    or rows. Stored in `AdaptiveExtractionCache` keyed on the page's
    structural fingerprint so future pages with the same shape reuse
    the rule for free.
    """
    compound_id_col: str           # raw header text of the compound-ID column
    assay_cols: list[AssayColumn] = field(default_factory=list)
    qualifier_chars: list[str] = field(default_factory=lambda: ["<", ">", "≤", "≥"])
    value_format: str = "decimal"  # "decimal" | "scientific" | "letter_grade"
    legend: dict[str, float] = field(default_factory=dict)  # for letter-grade tables: {"A": 0.075, ...}
    notes: str = ""


# ── Fingerprint builder ─────────────────────────────────────────────


_HEADER_LIKE_RE = re.compile(r"\b(IC50|EC50|CC50|Ki|Kd|FRET|EC_50|IC_50)\b", re.IGNORECASE)


def fingerprint_table_shape(
    headers: list[str],
    n_rows: int,
    sample_cells: list[str],
) -> str:
    """Build a stable hash that captures the STRUCTURE of a table format
    independent of specific values or compound IDs.

    Inputs that go into the hash:
      - normalized header tokens (lowercased, whitespace-collapsed)
      - column count
      - number of data rows (bucketed: 1-5, 6-20, 21-100, 100+)
      - cell-shape signature: replace digit runs with "N", letter
        sequences with "X" — captures column types without leaking values

    Two pages with the same overall shape (same header layout + row
    count bucket + cell pattern) produce the same fingerprint, so the
    cached rule applies.
    """
    norm_headers = sorted(re.sub(r"\s+", " ", h.strip().lower()) for h in headers if h)
    row_bucket = (
        "small" if n_rows <= 5 else
        "medium" if n_rows <= 20 else
        "large" if n_rows <= 100 else
        "xlarge"
    )
    cell_sig = []
    for c in sample_cells[:20]:
        s = re.sub(r"\d+", "N", c.strip()[:40])
        s = re.sub(r"[A-Za-z]+", "X", s)
        cell_sig.append(s)
    payload = json.dumps({
        "headers": norm_headers,
        "n_cols": len(headers),
        "row_bucket": row_bucket,
        "cell_sig": cell_sig,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Trigger detection ───────────────────────────────────────────────


def detect_column_miss(
    page_text: str,
    extracted_assay_names: set[str],
) -> list[str]:
    """Return assay-keyword tokens present in the page text but NOT in
    the extracted assay-name set. Drives the firer's trigger condition.

    Conservative — only flags well-known assay tokens (IC50, EC50, Ki,
    CC50, etc.) that aren't covered by the existing extraction. Missing
    novel/exotic assays is acceptable; over-firing the LLM is not.
    """
    found_keywords = {m.group(0).upper().replace("_", "")
                      for m in _HEADER_LIKE_RE.finditer(page_text)}
    extracted_norm = {re.sub(r"[^A-Z0-9]", "", n.upper()) for n in extracted_assay_names}

    missed: list[str] = []
    for kw in sorted(found_keywords):
        kw_norm = re.sub(r"[^A-Z0-9]", "", kw)
        # Already covered if any extracted assay name contains this keyword
        if any(kw_norm in n for n in extracted_norm):
            continue
        missed.append(kw)
    return missed


# ── Cache implementation ────────────────────────────────────────────


class AdaptiveExtractionCache:
    """JSON-backed cache of LLM-derived extraction rules.

    File layout:
        output_v2/text_extraction/_cache/adaptive_extraction_rules.json
        {
          "<fingerprint_hash>": {
            "rule": { ExtractionRule fields ... },
            "first_observed": "<patent_id>",
            "n_uses": 4,
            "last_used": "2026-05-02"
          },
          ...
        }
    """

    def __init__(self, cache_path: Path | None = None):
        if cache_path is None:
            cache_path = config.OUTPUT_DIR / "text_extraction" / "_cache" / "adaptive_extraction_rules.json"
        self.cache_path = cache_path
        self._cache = self._load()

    def _load(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text())
        except Exception as e:
            logger.warning(f"Could not read adaptive cache {self.cache_path}: {e!r}")
            return {}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2))

    def get(self, fingerprint: str) -> ExtractionRule | None:
        entry = self._cache.get(fingerprint)
        if not entry:
            return None
        try:
            rule_dict = entry["rule"]
            assay_cols = [AssayColumn(**c) for c in rule_dict.get("assay_cols", [])]
            rule = ExtractionRule(
                compound_id_col=rule_dict["compound_id_col"],
                assay_cols=assay_cols,
                qualifier_chars=rule_dict.get("qualifier_chars", ["<", ">"]),
                value_format=rule_dict.get("value_format", "decimal"),
                legend=rule_dict.get("legend", {}),
                notes=rule_dict.get("notes", ""),
            )
            entry["n_uses"] = entry.get("n_uses", 0) + 1
            entry["last_used"] = str(date.today())
            self._save()
            return rule
        except Exception as e:
            logger.warning(f"Cached rule {fingerprint} hydrate failed: {e!r}")
            return None

    def put(
        self,
        fingerprint: str,
        rule: ExtractionRule,
        first_observed_patent: str,
    ) -> None:
        self._cache[fingerprint] = {
            "rule": {
                "compound_id_col": rule.compound_id_col,
                "assay_cols": [asdict(c) for c in rule.assay_cols],
                "qualifier_chars": rule.qualifier_chars,
                "value_format": rule.value_format,
                "legend": rule.legend,
                "notes": rule.notes,
            },
            "first_observed": first_observed_patent,
            "n_uses": 1,
            "last_used": str(date.today()),
        }
        self._save()


# ── LLM rule derivation ─────────────────────────────────────────────


_RULE_PROMPT = """\
You are a patent assay-table extraction assistant. The page below
contains an assay table that the baseline parser PARTIALLY extracted.

Baseline parser found these assay columns: {extracted_names}
But the page text mentions these additional assay keywords that may
correspond to columns the parser missed: {missed_keywords}

Output a JSON extraction rule that lists ALL assay columns in the table,
including any the parser missed. Be precise about column position
(0-indexed, where compound_id is typically position 0).

Page text (truncated):
<<<
{page_text}
>>>

Output JSON ONLY, no prose. Schema:
{{
  "compound_id_col": "<exact header text of the compound-ID column>",
  "assay_cols": [
    {{"header_text": "<as it appears>", "canonical_name": "IC50|EC50|CC50|Ki|Kd|FRET_IC50|...",
      "unit": "μM|nM|M|%|...", "col_position": <0-indexed int>}},
    ...
  ],
  "qualifier_chars": ["<", ">", "≤", "≥"],
  "value_format": "decimal|scientific|letter_grade",
  "legend": {{"A": 0.075, ...}},  // empty object if not letter-grade
  "notes": "<one short sentence on what was non-obvious>"
}}
"""


# ── Row-realignment firer ───────────────────────────────────────────


_REALIGN_PROMPT = """\
You are a patent assay-table extraction assistant. The page below
contains one or more assay tables. The baseline parser detected
structural signals that the OCR mangled the row alignment (mis-placed
compound IDs, leading orphan data rows from a previous page,
compound-ID rows with no associated values after coalescing).

Re-extract the data ROWS, mapping each REAL compound to its assay
values. Drop OCR-noise rows (atom-label bleed-through, structure
fragments, page-number headers, prose). Return clean rows in the
order they appear in the patent's actual table — NOT necessarily the
order of the OCR'd HTML.

Page text:
<<<
{page_text}
>>>

Output JSON ONLY, no prose. Schema:
{{
  "rows": [
    {{
      "compound_id": "<patent's exact compound id, e.g. '5A', '43A', '162'>",
      "values": {{"<canonical_assay_name>": <number>, ...}},
      "qualifiers": {{"<canonical_assay_name>": "<", ...}},  // optional, only when value has < or >
      "n_runs": {{"<canonical_assay_name>": <int>, ...}}  // optional, only when patent annotates replicate count
    }},
    ...
  ],
  "units": {{"<canonical_assay_name>": "<unit>", ...}},
  "notes": "<one sentence on what required disambiguation>"
}}

Rules:
  - canonical_assay_name: use the column header verbatim if it already
    looks canonical (IC50, EC50, CC50, Ki, FRET_IC50, etc.). Otherwise
    use the shortest unambiguous form derived from the column header.
    Do NOT invent assay names that aren't in the source.
  - n_runs: many patents annotate replicate count immediately after
    the value, e.g. `0.0038 (8)` means the value 0.0038 was averaged
    over 8 runs. When such an annotation exists, capture it in n_runs.
    Skip n_runs entries when the patent doesn't annotate one.
"""


def _parse_realign_json(raw_stripped: str) -> dict | None:
    """Robust JSON parser for the realigner's output.

    Handles three failure modes empirically observed:
      1. Strict JSON (happy path)
      2. Truncation mid-row — the response hit max_tokens before the
         closing `}` and `]`. Strategy: strip trailing partial-row
         tokens, append `]}` to close `rows` array + outer object,
         then retry parse. This recovers the rows that DID complete.
      3. Trailing prose after the JSON object — strip everything after
         the first balanced `{...}` block.

    Returns parsed dict or None if no recovery succeeded.
    """
    # First try: strict parse
    try:
        return json.loads(raw_stripped)
    except json.JSONDecodeError:
        pass

    # Second try: extract the first {...} block via balanced-brace match
    # (regex-greedy). Handles trailing prose.
    m = re.search(r"\{.*\}", raw_stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Third try: truncation recovery. Strip back to the last COMPLETE
    # row entry (inside the `rows` array), then close the array + object.
    # Pattern: each row is a JSON object `{...}` ending in `}` followed
    # by `,` (between rows) or eventually `]` (end of array).
    #
    # Walk from start, tracking brace depth. When depth=0 again after
    # entering the rows array, that's a row boundary. Truncate at the
    # latest such boundary, then append `]}` to close.
    if '"rows"' not in raw_stripped:
        return None

    # Find the start of the rows array
    rows_start = raw_stripped.find("[", raw_stripped.find('"rows"'))
    if rows_start < 0:
        return None

    # Walk forward, tracking depth & string state, recording the offset
    # AFTER each complete row object inside the rows array.
    depth = 0
    in_string = False
    escape = False
    last_complete_row_end = -1
    for i in range(rows_start + 1, len(raw_stripped)):
        ch = raw_stripped[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # End of a row object inside rows[]
                last_complete_row_end = i + 1
        elif ch == "]" and depth == 0:
            # Real end of rows array — let the strict parser handle it
            break

    if last_complete_row_end < 0:
        return None

    # Build a recovered JSON string: prefix + completed rows + ]}
    candidate = raw_stripped[:last_complete_row_end] + "]}"
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last-ditch — try without optional trailing fields
        logger.debug("realign firer: truncation-recovery parse failed")
        return None


@dataclass
class RealignedRow:
    """One compound's assay values from the LLM realigner."""
    compound_id: str
    values: dict[str, float] = field(default_factory=dict)        # name → value
    qualifiers: dict[str, str] = field(default_factory=dict)       # name → '<' '>' '≤' '≥'
    n_runs: dict[str, int] = field(default_factory=dict)           # name → replicate count (when patent annotates)


@dataclass
class RealignedTable:
    """LLM-realigned table contents."""
    rows: list[RealignedRow]
    units: dict[str, str] = field(default_factory=dict)            # name → unit
    notes: str = ""


def realign_table_via_llm(
    page_text: str,
    max_text_chars: int = 6000,
    max_tokens: int = 8000,
) -> RealignedTable | None:
    """Fire one LM call to re-extract assay-table rows when the
    baseline parser flagged alignment uncertainty. Returns a
    RealignedTable with cleaned rows, or None on failure.

    Caller is responsible for caching by table fingerprint.

    Defaults sized for full-page assay tables: ~6000 source chars and
    ~8000 output tokens covers ≥250 rows of compact JSON. The earlier
    2500-token cap truncated mid-row on tables like US11312727 TABLE 1A
    (252 compounds) and US10214537 TABLE 11 (600+ compounds), causing
    the JSON parser to fail and the firer to return no rows.
    """
    truncated = page_text[:max_text_chars]
    prompt = _REALIGN_PROMPT.format(page_text=truncated)
    try:
        raw = call_claude_text(prompt, max_tokens=max_tokens)
    except Exception as e:
        logger.warning(f"realign firer LM call failed: {e!r}")
        return None
    if not raw:
        return None

    raw_stripped = raw.strip()
    if raw_stripped.startswith("```"):
        raw_stripped = re.sub(r"^```(?:json)?\s*", "", raw_stripped)
        raw_stripped = re.sub(r"\s*```$", "", raw_stripped)

    parsed = _parse_realign_json(raw_stripped)
    if parsed is None:
        return None

    rows: list[RealignedRow] = []
    for r in parsed.get("rows", []) or []:
        cid = str(r.get("compound_id") or "").strip()
        if not cid:
            continue
        vals_raw = r.get("values") or {}
        vals: dict[str, float] = {}
        for name, v in vals_raw.items():
            try:
                vals[str(name)] = float(v)
            except (TypeError, ValueError):
                continue
        quals = {str(k): str(v) for k, v in (r.get("qualifiers") or {}).items() if v}
        n_runs_raw = r.get("n_runs") or {}
        n_runs: dict[str, int] = {}
        for k, v in n_runs_raw.items():
            try:
                n_runs[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        if vals:
            rows.append(RealignedRow(
                compound_id=cid, values=vals, qualifiers=quals, n_runs=n_runs,
            ))

    if not rows:
        logger.warning("realign firer: LM returned no usable rows, dropping")
        return None

    units = {str(k): str(v) for k, v in (parsed.get("units") or {}).items() if v}
    return RealignedTable(rows=rows, units=units, notes=str(parsed.get("notes") or ""))


def derive_rule_via_llm(
    page_text: str,
    extracted_assay_names: list[str],
    missed_keywords: list[str],
    max_text_chars: int = 4000,
) -> ExtractionRule | None:
    """Fire one LM call to derive a structured extraction rule.

    Returns the rule on success, None on parse failure. Caller is
    responsible for caching.
    """
    truncated = page_text[:max_text_chars]
    prompt = _RULE_PROMPT.format(
        extracted_names=extracted_assay_names or "(none)",
        missed_keywords=missed_keywords,
        page_text=truncated,
    )
    try:
        raw = call_claude_text(prompt, max_tokens=1500)
    except Exception as e:
        logger.warning(f"adaptive firer LM call failed: {e!r}")
        return None

    if not raw:
        return None

    # Strip leading/trailing prose if any (model sometimes wraps in markdown)
    raw_stripped = raw.strip()
    if raw_stripped.startswith("```"):
        raw_stripped = re.sub(r"^```(?:json)?\s*", "", raw_stripped)
        raw_stripped = re.sub(r"\s*```$", "", raw_stripped)

    try:
        parsed = json.loads(raw_stripped)
    except json.JSONDecodeError:
        # Retry: extract first {...} block
        m = re.search(r"\{.*\}", raw_stripped, re.DOTALL)
        if not m:
            logger.warning(f"adaptive firer: LM output not JSON-parseable: {raw_stripped[:200]!r}")
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"adaptive firer: JSON parse failed even after extraction: {e!r}")
            return None

    try:
        assay_cols = [
            AssayColumn(**{k: v for k, v in c.items() if k in {"header_text", "canonical_name", "unit", "col_position"}})
            for c in parsed.get("assay_cols", [])
        ]
        rule = ExtractionRule(
            compound_id_col=parsed.get("compound_id_col", ""),
            assay_cols=assay_cols,
            qualifier_chars=parsed.get("qualifier_chars", ["<", ">", "≤", "≥"]),
            value_format=parsed.get("value_format", "decimal"),
            legend=parsed.get("legend", {}) or {},
            notes=parsed.get("notes", ""),
        )
    except Exception as e:
        logger.warning(f"adaptive firer: rule construction failed: {e!r}")
        return None

    if not rule.compound_id_col or not rule.assay_cols:
        logger.warning("adaptive firer: rule missing required fields, dropping")
        return None

    return rule
