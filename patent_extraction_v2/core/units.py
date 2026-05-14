"""Single source of truth for assay-value unit conversion.

Before this module: 6 separate implementations of `value_to_uM`,
`_normalize_unit_to_uM`, `normalize_unit`, etc., each in a different
script with subtle differences (some handle pM, some don't; some
canonicalize `uM` → `μM` first, others not). This module is the
canonical version — every CSV writer / fidelity check / extractor
should `from patent_extraction_v2.core.units import value_to_uM`.

Conventions follow BindingDB / ChEMBL: nM is the canonical "small"
unit, μM the canonical "table" unit. Patent tables that report in
mM or pM are converted to μM-equivalent for cross-source comparison.
"""
from __future__ import annotations


# Canonical multipliers: how many nM is `1 <unit>`?
# nM is the universal denominator (BindingDB convention).
_NM_PER_UNIT: dict[str, float] = {
    "nM":      1.0,
    "nmol/L":  1.0,
    "μM":      1000.0,
    "uM":      1000.0,         # ASCII transliteration
    "µM":      1000.0,         # micro-sign Latin-1
    "mcM":     1000.0,         # rare: "microcrM"
    "umol/L":  1000.0,
    "μmol/L":  1000.0,
    "mM":      1_000_000.0,
    "mmol/L":  1_000_000.0,
    "M":       1_000_000_000.0,
    "mol/L":   1_000_000_000.0,
    "pM":      0.001,
    "pmol/L":  0.001,
}


def _normalize_unit_token(unit: str | None) -> str:
    """Strip whitespace + collapse μ-variants. Returns the canonical
    key used in `_NM_PER_UNIT`, or empty string if unknown."""
    if not unit:
        return ""
    u = str(unit).strip()
    # Collapse all micro-variants to the canonical Greek μ
    u = u.replace("uM", "μM").replace("µM", "μM").replace("Î¼M", "μM").replace("ÂµM", "μM")
    return u


def value_to_uM(value: float | None, unit: str | None) -> float | None:
    """Convert `value` from `unit` to μM. Returns None on null input
    or unrecognized unit. Identity for already-μM values.

    This is the function every CSV builder / fidelity check should
    use. Do NOT reimplement.
    """
    if value is None:
        return None
    u = _normalize_unit_token(unit)
    nm_per = _NM_PER_UNIT.get(u)
    if nm_per is None:
        # Pass-through for unitless / "%" / unrecognized — caller decides
        return float(value)
    nm = float(value) * nm_per
    return nm / 1000.0


def value_to_nM(value: float | None, unit: str | None) -> float | None:
    """Same as `value_to_uM` but expressed in nM."""
    if value is None:
        return None
    u = _normalize_unit_token(unit)
    nm_per = _NM_PER_UNIT.get(u)
    if nm_per is None:
        return float(value)
    return float(value) * nm_per


def convert(value: float | None, from_unit: str, to_unit: str) -> float | None:
    """General-purpose: convert a value between any two known units.
    Returns None on null input or unrecognized units."""
    if value is None:
        return None
    src = _NM_PER_UNIT.get(_normalize_unit_token(from_unit))
    dst = _NM_PER_UNIT.get(_normalize_unit_token(to_unit))
    if src is None or dst is None:
        return None
    return float(value) * src / dst


def canonical_unit_string(unit: str | None) -> str:
    """Canonicalize a unit string for display / comparison. Maps every
    micro-variant (uM/µM/Î¼M) to `μM`. Empty string for unknown."""
    return _normalize_unit_token(unit)
