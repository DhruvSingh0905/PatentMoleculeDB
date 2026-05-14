"""Stage 5b — FSM ↔ LLM cross-validation + merge.

Both the FSM (Stages 1-4) and the LLM (Stage 5) extract rows from
the same region. This stage merges the two by compound_id, preferring
the LLM's value on disagreement and never dropping a row that either
side extracted.

Disagreement log: every (compound_id, FSM_value, LLM_value) tuple
where the values differ goes to
`output_v2/text_extraction/_logs/fsm_llm_disagreements.jsonl`.
This is the data feed for `vocab_learner.py` to discover new tokens
the FSM is mis-tokenizing.

The merge logic:
  1. Build dict[cpd_id_normalized] for both sides.
  2. For each cpd_id present in EITHER side:
     - Both present, values match (within 5% tolerance) → keep LLM
       (LLM is the source of truth; FSM may have lost qualifier/n_runs)
     - Both present, values differ → keep LLM, log disagreement
     - LLM only → keep LLM
     - FSM only → keep FSM (may be a row the LLM truncated past)
  3. Map merged rows to AssayResult records using the LLM's units +
     column-name map.

`AssayResult` is the existing pydantic model from
`core.models` — we don't introduce a new schema.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..adaptive_extraction_cache import RealignedRow
from ..models import AssayResult
from .row_aligner import AlignedRow, ValueCell

logger = logging.getLogger(__name__)


_DISAGREEMENT_LOG = Path(
    "output_v2/text_extraction/_logs/fsm_llm_disagreements.jsonl"
)


@dataclass
class MergeResult:
    """Cross-validation outcome for one region."""
    assay_results: dict[str, list[AssayResult]] = field(default_factory=dict)
    n_fsm_only: int = 0
    n_llm_only: int = 0
    n_agreement: int = 0
    n_disagreement: int = 0


def cross_validate(
    fsm_rows: list[AlignedRow],
    llm_rows: list[RealignedRow],
    *,
    units: dict[str, str] | None = None,
    fingerprint: str = "",
    patent_id: str | None = None,
    page_offset: int = 0,
    tol_pct: float = 5.0,
) -> MergeResult:
    """Merge FSM and LLM extraction outputs into a single dict of
    `AssayResult` records keyed by compound_id.

    Args:
        fsm_rows: From `row_aligner.align_rows(...)`.
        llm_rows: From `llm_realigner.realign_region(...)`.
        units: Per-assay-name unit map from the LLM.
        fingerprint: Cache fingerprint of the region (for log provenance).
        patent_id: Source patent (for log provenance).
        page_offset: Region's start offset on the source page (for log).
        tol_pct: Relative tolerance for "values agree" check.

    Returns:
        MergeResult with merged AssayResult dict + disagreement counts.
    """
    units = units or {}
    fsm_idx = {_norm_id(r.compound_id): r for r in fsm_rows}
    llm_idx = {_norm_id(r.compound_id): r for r in llm_rows}

    out: dict[str, list[AssayResult]] = {}
    n_fsm_only = n_llm_only = n_agree = n_disagree = 0

    all_keys = set(fsm_idx.keys()) | set(llm_idx.keys())
    for key in all_keys:
        if not key:
            continue
        fsm_row = fsm_idx.get(key)
        llm_row = llm_idx.get(key)

        if llm_row and not fsm_row:
            n_llm_only += 1
            out[key] = _llm_to_results(llm_row, units)
            continue

        if fsm_row and not llm_row:
            n_fsm_only += 1
            out[key] = _fsm_to_results(fsm_row, units)
            continue

        # Both present — compare values
        if fsm_row and llm_row:
            disagreement = _check_disagreement(fsm_row, llm_row, tol_pct=tol_pct)
            if disagreement:
                n_disagree += 1
                _log_disagreement(
                    fingerprint=fingerprint,
                    patent_id=patent_id,
                    page_offset=page_offset,
                    compound_id=llm_row.compound_id,
                    detail=disagreement,
                )
            else:
                n_agree += 1
            # User-confirmed policy: prefer LLM on disagreement
            out[key] = _llm_to_results(llm_row, units)

    return MergeResult(
        assay_results=out,
        n_fsm_only=n_fsm_only,
        n_llm_only=n_llm_only,
        n_agreement=n_agree,
        n_disagreement=n_disagree,
    )


# ── Helpers ──────────────────────────────────────────────────────


def _norm_id(s: str | None) -> str:
    """Normalize a compound id for cross-side matching."""
    if not s:
        return ""
    return s.strip().lower().replace(" ", "")


def _llm_to_results(
    row: RealignedRow,
    units: dict[str, str],
) -> list[AssayResult]:
    out: list[AssayResult] = []
    for assay_name, value in (row.values or {}).items():
        try:
            v_num = float(value)
        except (TypeError, ValueError):
            continue
        unit = units.get(assay_name, "")
        qualifier = (row.qualifiers or {}).get(assay_name)
        n_runs = (row.n_runs or {}).get(assay_name)
        out.append(AssayResult(
            assay_name=assay_name,
            value_raw=str(value),
            value_numeric=v_num,
            qualifier=qualifier,
            unit=unit,
            n_runs=n_runs,
        ))
    return out


def _fsm_to_results(
    row: AlignedRow,
    units: dict[str, str],
) -> list[AssayResult]:
    """Convert an FSM AlignedRow into AssayResult records.

    The FSM row has *positional* cells (col 0, col 1, ...) without
    canonical assay names. We map them to "col_0", "col_1", etc.
    The LLM is normally the one that supplies canonical names — when
    the FSM stands alone (LLM truncated or failed), the column names
    will be generic. The cross_validator's "prefer-LLM" rule means
    these generic names only appear when the LLM didn't extract this
    compound at all.
    """
    out: list[AssayResult] = []
    for col_idx, cell in enumerate(row.cells):
        col_name = f"col_{col_idx}"
        if cell.is_null:
            out.append(AssayResult(
                assay_name=col_name,
                value_raw=cell.raw or "",
                value_numeric=None,
                qualifier=None,
                unit=units.get(col_name, ""),
                n_runs=cell.n_runs,
            ))
            continue
        if cell.value is None:
            continue
        out.append(AssayResult(
            assay_name=col_name,
            value_raw=cell.raw or str(cell.value),
            value_numeric=cell.value,
            qualifier=cell.qualifier,
            unit=units.get(col_name, ""),
            n_runs=cell.n_runs,
        ))
    return out


def _check_disagreement(
    fsm_row: AlignedRow,
    llm_row: RealignedRow,
    *,
    tol_pct: float,
) -> dict | None:
    """Return None if FSM and LLM agree on numeric content, else
    a dict describing the disagreement.

    Comparison strategy: compare the SET of numeric values present in
    each row (positional alignment is unreliable across FSM and LLM
    because the LLM may have a different column count). If every FSM
    value tolerance-matches some LLM value (and vice versa), they
    agree.
    """
    fsm_vals = sorted(c.value for c in fsm_row.cells if c.value is not None)
    llm_vals = sorted(
        v for v in (llm_row.values or {}).values()
        if isinstance(v, (int, float))
    )

    if len(fsm_vals) != len(llm_vals):
        return {
            "type": "cell_count_mismatch",
            "fsm_count": len(fsm_vals),
            "llm_count": len(llm_vals),
            "fsm_vals": list(fsm_vals)[:5],
            "llm_vals": list(llm_vals)[:5],
        }

    for fv, lv in zip(fsm_vals, llm_vals):
        # Tolerance check
        max_v = max(abs(fv), abs(lv), 1e-9)
        rel_diff = abs(fv - lv) / max_v * 100
        if rel_diff > tol_pct:
            return {
                "type": "value_disagreement",
                "fsm_val": fv,
                "llm_val": lv,
                "rel_diff_pct": round(rel_diff, 2),
            }
    return None


def _log_disagreement(
    *,
    fingerprint: str,
    patent_id: str | None,
    page_offset: int,
    compound_id: str,
    detail: dict,
) -> None:
    """Append one JSON line per disagreement. Append-only so multiple
    pipeline runs accumulate evidence over time."""
    try:
        _DISAGREEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        line = json.dumps({
            "ts": datetime.utcnow().isoformat() + "Z",
            "fp": fingerprint,
            "patent_id": patent_id,
            "page_offset": page_offset,
            "compound_id": compound_id,
            "detail": detail,
        }, ensure_ascii=False)
        with _DISAGREEMENT_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.warning("disagreement log write failed: %r", e)
