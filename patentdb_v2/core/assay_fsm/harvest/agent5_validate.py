"""HARVEST Agent 5 — Validation.

Cross-checks Agent 2's tuples against:
  - unit-conversion sanity (IC50 of 100 mM is suspicious, etc.)
  - source-quote presence (Agent 2 already verifies, we re-confirm)
  - target-assay match (assay_name should relate to an Agent 1 target)
  - intra-compound consistency (same cpd, same assay, same chunk →
    values should agree within 2x)

Output: same tuples with `confidence` re-graded + `validation_reason`
populated. Tuples are NEVER dropped — downstream filters by confidence.

Implementation choice: mostly deterministic. We use the LLM only when
the deterministic checks pass with ambiguity (e.g., a target name
that's PARTIALLY matched). For most validation, deterministic logic
is faster, cheaper, and more predictable than another LLM call.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from .agent1_targets import Target
from .agent2_activities import ActivityTuple

logger = logging.getLogger(__name__)


# Plausible value ranges by assay type, in nM (canonical unit).
# Outside the range → confidence downgraded.
# These are pharma-typical bounds, not hard cutoffs.
_PLAUSIBLE_RANGES_NM: dict[str, tuple[float, float]] = {
    # binding affinities — picomolar to high-micromolar
    "ic50": (0.001, 100_000),       # 1 pM to 100 μM
    "ki":   (0.001, 100_000),
    "kd":   (0.001, 100_000),
    # cellular potencies — wider window
    "ec50": (0.001, 1_000_000),     # 1 pM to 1 mM
    "ac50": (0.001, 1_000_000),
    "cc50": (0.001, 10_000_000),    # cytotoxicity often >1 mM
    # antiviral
    "ic90": (0.01, 1_000_000),
    "ec90": (0.01, 1_000_000),
}


def _canonical_assay_token(assay_name: str) -> str:
    """Map an assay name to its canonical short token (ic50/ec50/ki/kd/...).
    Returns "" if no token matches."""
    al = (assay_name or "").lower()
    for token in ("pic50", "pki", "pkd", "ic50", "ec50", "cc50", "ac50",
                  "ic90", "ec90", "ki", "kd", "potency"):
        if token in al:
            return token
    return ""


def _value_to_nM(value: float, unit: str) -> float | None:
    u = (unit or "").strip()
    if u in ("nM", "nmol/L"):
        return value
    if u in ("μM", "uM", "µM", "umol/L", "μmol/L"):
        return value * 1000.0
    if u in ("mM", "mmol/L"):
        return value * 1_000_000.0
    if u in ("pM", "pmol/L"):
        return value / 1000.0
    if u == "M":
        return value * 1_000_000_000.0
    return None


@dataclass
class Agent5Result:
    tuples: list[ActivityTuple] = field(default_factory=list)
    n_high: int = 0
    n_medium: int = 0
    n_low: int = 0


def validate(
    tuples: list[ActivityTuple],
    targets: list[Target],
    chunk_text: str,
) -> Agent5Result:
    """Run validation checks on every Agent 2 tuple. Re-grades
    `confidence` and populates `validation_reason` in-place."""
    target_names_lc = {t.target_name.lower() for t in targets if t.target_name}

    # Group by (compound_id, canonical_assay) for intra-group consistency
    grouped: dict[tuple[str, str], list[ActivityTuple]] = defaultdict(list)
    for t in tuples:
        key = (t.compound_id.lower(), _canonical_assay_token(t.assay_name) or t.assay_name.lower())
        grouped[key].append(t)

    n_high = n_medium = n_low = 0
    for t in tuples:
        reasons: list[str] = []
        if t.validation_reason:
            reasons.append(t.validation_reason)

        # 1. Source-quote presence (re-confirm)
        if t.source_quote and t.source_quote not in chunk_text:
            reasons.append("source_quote_not_in_chunk")

        # 2. Unit-conversion sanity
        v_nm = _value_to_nM(t.value, t.unit)
        token = _canonical_assay_token(t.assay_name)
        if v_nm is not None and token and token in _PLAUSIBLE_RANGES_NM:
            lo, hi = _PLAUSIBLE_RANGES_NM[token]
            if v_nm < lo:
                reasons.append(f"value_below_plausible_range({v_nm}_nM<{lo}_nM)")
            elif v_nm > hi:
                reasons.append(f"value_above_plausible_range({v_nm}_nM>{hi}_nM)")

        # 3. Target-assay match
        if target_names_lc:
            assay_lc = (t.assay_name or "").lower()
            if not any(tn in assay_lc or assay_lc in tn for tn in target_names_lc):
                if not token:
                    reasons.append("assay_not_in_target_list")

        # 4. Intra-group consistency (same cpd, same assay)
        key = (t.compound_id.lower(), token or t.assay_name.lower())
        peers = [p for p in grouped[key] if p is not t]
        if peers and v_nm is not None:
            for p in peers:
                p_nm = _value_to_nM(p.value, p.unit)
                if p_nm is None:
                    continue
                if max(v_nm, p_nm) / max(min(v_nm, p_nm), 1e-9) > 2.0:
                    reasons.append("intra_compound_inconsistent")
                    break

        # Re-grade confidence based on reasons
        original = t.confidence
        if reasons:
            # Any reason downgrades; multiple → low
            if len(reasons) >= 2 or "source_quote_not_in_chunk" in reasons:
                t.confidence = "low"
            else:
                t.confidence = "medium" if original == "high" else original
        # else: keep original confidence

        t.validation_reason = "; ".join(reasons)

        if t.confidence == "high":
            n_high += 1
        elif t.confidence == "medium":
            n_medium += 1
        else:
            n_low += 1

    if tuples:
        logger.info(
            "agent5: validated %d tuples — high=%d medium=%d low=%d",
            len(tuples), n_high, n_medium, n_low,
        )
    return Agent5Result(
        tuples=tuples, n_high=n_high, n_medium=n_medium, n_low=n_low,
    )
