"""Structural table fingerprint, its JSON-backed cache, and the row realigner.

The name is historical. The module was written to buy an extraction RULE from
a model when the baseline parser under-read a table, cache it by structural
fingerprint, and replay it free on every patent with the same shape. That half
never had a caller and is gone (see the note below `logger`); what is left is
the fingerprint function, the file-backed cache several tiers share, and
`realign_table_via_llm` — the one paid entry point, behind
`config.ASSAY_REALIGN_ENABLED`, default OFF.

Two consumers reach the cache through this class:
  - `assay_fsm.llm_realigner` — the realigner's per-region row cache
  - `assay_fsm.harvest.orchestrator` — HARVEST's per-chunk cache
and `harvest.iupac_orchestrator` reads the same FILE through its own
`_read_cache()`. All three key their entries with a namespace prefix
(`plaintext:`, `iupac:`, `iupac_targeted:`) to keep those spaces apart.

The fingerprint is deliberately content-blind — header tokens, column count,
row-count bucket, cell shape — so it carries no patent ids and no compound
names. `llm_realigner` nevertheless prefixes the patent id onto its key: a
purely structural key returned patent A's rows for patent B, verified across
14 patents sharing this file.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .api_client import call_claude_text
from .assay_name_guard import is_valid_assay_name

logger = logging.getLogger(__name__)


# The `ExtractionRule` / `AssayColumn` pair that used to head this module is
# gone, with `derive_rule_via_llm`, `_RULE_PROMPT`, `detect_column_miss` and
# `AdaptiveExtractionCache.get`/`.put`. Between them they were the rule half
# the docstring describes, and `derive_rule_via_llm` was the second of this
# file's two ungated `call_claude_text` sites.
#
# They had ZERO callers anywhere outside `_attic`. `iupac_orchestrator` and
# `harvest/orchestrator` share the same JSON FILE but reach it through
# `_read_cache()` or `cache._cache[...]` directly, never through the
# ExtractionRule accessors, so nothing ever produced or consumed a rule. The
# live surface is the fingerprint + the row realigner below.


# ── Fingerprint builder ─────────────────────────────────────────────


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


# ── Cache implementation ────────────────────────────────────────────


class AdaptiveExtractionCache:
    """JSON-backed cache shared by the realigner and the HARVEST tiers.

    File layout:
        output_v2/text_extraction/_cache/adaptive_extraction_rules.json
        {
          "<namespace>:<...>:<fingerprint_hash>": {
            "rule": { whatever the owning tier stores — rows, pairs, ... },
            "first_observed": "<patent_id>",
            "n_uses": 4,
            "last_used": "2026-05-02"
          },
          ...
        }

    Every reader goes through `_cache` directly and writes its own entry
    shape; the typed `get`/`put` accessors that used to hydrate an
    `ExtractionRule` had no callers and are gone with it.
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
    patent_id: str = "",
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
    if not config.ASSAY_REALIGN_ENABLED:
        # Second half of the gate. `llm_realigner.realign_region` checks it
        # too; this one is what a NEW call site cannot get around. `43d037e`
        # deleted two functions that between them held the only
        # `LLM_RECOVERY_ENABLED` check in their module, and the flag went
        # with them.
        return None
    truncated = page_text[:max_text_chars]
    prompt = _REALIGN_PROMPT.format(page_text=truncated)
    try:
        raw = call_claude_text(prompt, max_tokens=max_tokens,
                               patent_id=patent_id, cost_category="realign")
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

    # GUARD: drop any assay-name key the LLM invented as a placeholder
    # (Activity1, col_0, col1_IC50, …) or fabricated from a non-assay
    # column (M+H, Method, HPLC tR, Molecular Weight, …). See
    # `assay_name_guard.is_valid_assay_name` for the full pattern list.
    # Filtering HERE (at LLM output) prevents the bad names from being
    # cached by AdaptiveExtractionCache.put under a structural fingerprint
    # and re-firing on every later patent with a similar table shape —
    # the root cause behind the cross-patent contamination we observed
    # (RORγ in K-Ras / B-Raf patents) and the US11292791 100%-zero
    # Activity1 column.
    rows: list[RealignedRow] = []
    n_dropped_names = 0
    for r in parsed.get("rows", []) or []:
        cid = str(r.get("compound_id") or "").strip()
        if not cid:
            continue
        vals_raw = r.get("values") or {}
        vals: dict[str, float] = {}
        for name, v in vals_raw.items():
            if not is_valid_assay_name(name):
                n_dropped_names += 1
                continue
            try:
                vals[str(name)] = float(v)
            except (TypeError, ValueError):
                continue
        quals = {str(k): str(v) for k, v in (r.get("qualifiers") or {}).items()
                 if v and is_valid_assay_name(k)}
        n_runs_raw = r.get("n_runs") or {}
        n_runs: dict[str, int] = {}
        for k, v in n_runs_raw.items():
            if not is_valid_assay_name(k):
                continue
            try:
                n_runs[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        if vals:
            rows.append(RealignedRow(
                compound_id=cid, values=vals, qualifiers=quals, n_runs=n_runs,
            ))

    if n_dropped_names:
        logger.info("realign firer: guard dropped %d placeholder/non-assay name(s)",
                    n_dropped_names)
    if not rows:
        logger.warning("realign firer: LM returned no usable rows, dropping")
        return None

    units = {str(k): str(v) for k, v in (parsed.get("units") or {}).items()
             if v and is_valid_assay_name(k)}
    return RealignedTable(rows=rows, units=units, notes=str(parsed.get("notes") or ""))
