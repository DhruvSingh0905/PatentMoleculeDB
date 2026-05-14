"""Backwards-compatible shim around the new generic FSM pipeline.

Historically this module hosted a 1850-LOC cascade of hardcoded
patent-specific extraction logic (PI3K-delta-CD69 headers, Menin
binding patterns, SGK regexes, an HTML `<table>` pre-flight gate that
silently dropped plain-text assay tables, three separate `_*_via_adaptive_firer`
functions, etc.). All of that is now replaced by the generic FSM +
always-fire LLM pipeline at `core.assay_fsm`.

Public API (unchanged signature, drop-in for the legacy callers):

  - `extract_assays_for_patent(patent_id, ...)` — runs the new pipeline
    on every page of a patent, returns dict[cpd_id, list[AssayResult]]
    or `(dict, diagnostic_dict)` when `return_diagnostic=True`.

The `ASSAY_FSM_ENABLED` flag in `core.config` defaults to True. If a
caller sets `ASSAY_FSM=0` we currently raise — there is no legacy
fallback. The flag is preserved so we can wire a future alternative
backend without breaking the call sites.

Reusable helpers (`_parse_value`, `_sanitize_compound_id`,
`normalize_unit`, `_clean_assay_name`) historically defined here are
exposed for back-compat. New code should import them from
`core.assay_fsm.*` instead.
"""
from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from ..core import config
from ..core.models import AssayResult

logger = logging.getLogger(__name__)


# ── Reusable helpers (kept for back-compat with downstream tools) ──


def normalize_unit(value: float, unit: str, target_unit: str = "nM") -> float:
    """Convert assay values to a common unit. Used by audit + eval scripts."""
    conversions = {"nM": 1.0, "μM": 1000.0, "uM": 1000.0, "µM": 1000.0, "mM": 1e6}
    if unit in conversions and target_unit in conversions:
        return value * conversions[unit] / conversions[target_unit]
    return value


_VALUE_PATTERN = re.compile(r'^([<>≤≥~]?)\s*(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?$')


def _parse_value(raw: str) -> tuple[float | None, str | None]:
    """Parse an assay value string into (numeric, qualifier)."""
    raw = (raw or "").strip()
    raw = html.unescape(raw)
    if not raw or raw in ('—', '-', 'N/A', 'n/a', 'NT', 'nd', 'ND'):
        return None, None
    if 'no inhib' in raw.lower() or 'inactive' in raw.lower():
        return None, '>'
    m = _VALUE_PATTERN.match(raw)
    if m:
        return float(m.group(2)), m.group(1) or None
    nums = re.findall(r'(\d+\.?\d*)', raw)
    if nums:
        qualifier = '<' if ('<' in raw or '&lt;' in raw) else (
            '>' if ('>' in raw or '&gt;' in raw) else None
        )
        return float(nums[0]), qualifier
    return None, None


def _sanitize_compound_id(raw: str) -> str | None:
    """Normalize a compound-ID cell to a clean key, or None if invalid.
    Delegates to the canonical normalizer in `core.compound_id`.
    """
    from ..core.compound_id import normalize_cid_key
    return normalize_cid_key(raw)


def _clean_assay_name(raw_name: str) -> str:
    """Lightweight assay-name normalization (collapses whitespace, strips
    trailing/leading punctuation). The FSM pipeline already canonicalizes
    names via the LLM realigner; this helper is only here for callers
    that build their own assay strings."""
    s = re.sub(r"\s+", " ", (raw_name or "").strip())
    return s.strip(" ,;:")


# Re-export from the canonical home so downstream code can keep using
# `from .google_assays import _repair_mojibake` if it still does.
from ..core.assay_fsm.normalizer import repair_mojibake as _repair_mojibake  # noqa: E402,F401


# ── Public API ───────────────────────────────────────────────────


def extract_assays_for_patent(
    patent_id: str,
    data_dir: Path | None = None,
    return_diagnostic: bool = False,
):
    """Extract all assay data for a patent.

    Behavior:
      - When `config.ASSAY_FSM_ENABLED` is True (default), delegates
        to `core.assay_fsm.pipeline.extract_for_patent` — the generic
        FSM + always-fire LLM cascade.
      - When False, raises NotImplementedError. The legacy cascade was
        deleted in the Phase D cutover; if you need to re-enable a
        non-FSM backend, plug a new implementation in here.

    Args:
        patent_id: Patent identifier.
        data_dir: Root data directory. Defaults to `config.DATA_DIR`.
        return_diagnostic: If True, returns (assays, diagnostic_dict).

    Returns:
        Either dict[cpd_key, [AssayResult]] OR
        (dict[cpd_key, [AssayResult]], diagnostic_dict).
    """
    if not getattr(config, "ASSAY_FSM_ENABLED", True):
        raise NotImplementedError(
            "ASSAY_FSM_ENABLED=False but the legacy cascade was removed in "
            "the Phase D cutover. Set ASSAY_FSM=1 (the default) or wire a "
            "new backend in routes/google_assays.py."
        )
    from ..core.assay_fsm import extract_for_patent as _new_extract
    return _new_extract(
        patent_id, data_dir=data_dir, return_diagnostic=return_diagnostic,
    )


__all__ = [
    "extract_assays_for_patent",
    "normalize_unit",
    "_parse_value",
    "_sanitize_compound_id",
    "_clean_assay_name",
    "_repair_mojibake",
]
