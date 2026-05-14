"""Patent classification for route gating.

Classifies a patent into one of four classes based on the FormatAudit signals
and the layout-router output. The class determines which expensive routes
(image pipeline, Markush enumeration) are allowed to fire.

Patent-agnostic: rules are based purely on audit signals, no patent-specific
hardcoding. New patents are auto-classified at run_patent() entry.

Classes:
    A — clean text, no Markush, no big tables (most patents)
    B — Cpd.No tables with full chemical names (US10899738-style)
    C — substituent tables / Markush format (US9718825-style)
    D — image-heavy, macrocycles, non-IUPAC nomenclature (US11312727-style)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .google_patents import FormatAudit


class PatentClass(str, Enum):
    A = "tier_a"
    B = "tier_b"
    C = "tier_c"
    D = "tier_d"


# Route gates: which routes are allowed per class.
# Values: "always" | "if_incomplete" | "suppressed" | "rare"
# Pipeline.py interprets these to decide whether to invoke each step.
ROUTE_GATES: dict[PatentClass, dict[str, str]] = {
    PatentClass.A: {
        "text_routes":          "always",          # Routes 1a/1b/1c
        "synthesis_block":      "if_incomplete",   # Route 1d
        "image_pipeline":       "if_incomplete",   # Step 2.3 — only formula/dense pages
        "markush_enumeration":  "if_incomplete",   # Step 2.7 — skip if recall already high
    },
    PatentClass.B: {
        "text_routes":          "always",
        "synthesis_block":      "if_incomplete",
        "image_pipeline":       "if_incomplete",
        "markush_enumeration":  "if_incomplete",
    },
    PatentClass.C: {
        "text_routes":          "always",
        "synthesis_block":      "rare",            # cell-text usually not synthesis prose
        "image_pipeline":       "suppressed",      # crops are R-group fragments, not full mols
        "markush_enumeration":  "always",          # the right tool for this class
    },
    PatentClass.D: {
        "text_routes":          "always",
        "synthesis_block":      "if_incomplete",
        "image_pipeline":       "always",          # macrocycle figures are the source of truth
        "markush_enumeration":  "if_incomplete",
    },
}


def classify_patent(
    audit: "FormatAudit",
    format_info: dict,
    manifest=None,
) -> tuple[PatentClass, str]:
    """Return (PatentClass, reason_string) for the given audit + format info.

    Order of checks (first match wins):
      1. Substituent table  → C  (Markush is the right tool)
      2. Compound table     → B  (text route handles these; image is optional)
      3. Image-heavy        → D  (layout-router signal OR many formula/dense pages)
      4. Default            → A
    """
    if getattr(audit, 'has_substituent_table', False):
        return PatentClass.C, "audit detected substituent table (Ar/R1 columns)"

    if getattr(audit, 'has_compound_table', False):
        return PatentClass.B, "audit detected compound table (Cpd.No + Chemical Name)"

    # Layout-router signal — strong indicator of image-heavy patent
    image_score = format_info.get("image_score", 0)
    routing = format_info.get("routing", {}) or {}
    if routing.get("use_image_pipeline") or image_score > 30:
        return PatentClass.D, f"layout router flagged image pipeline (image_score={image_score})"

    # Manifest signal — image_dense_pages (heavy-image, low-text) is a strong
    # indicator of figure-driven patents like US11312727.
    # formula_pages alone is too noisy (claims mention "Formula (I)" repeatedly
    # without being figure pages), so we require image_dense_pages as the
    # primary signal here.
    if manifest is not None:
        n_image_dense = len(getattr(manifest, 'image_dense_pages', []))
        n_formula = len(getattr(manifest, 'formula_pages', []))
        if n_image_dense >= 10:
            return PatentClass.D, (
                f"manifest signals image-heavy "
                f"(image_dense_pages={n_image_dense}, formula_pages={n_formula})"
            )

    return PatentClass.A, "default class — clean text, no special tables"


def gate_for(patent_class: PatentClass, route: str) -> str:
    """Look up the gate value for a (class, route) pair.

    Returns one of: "always" | "if_incomplete" | "suppressed" | "rare" | "unknown".
    """
    return ROUTE_GATES.get(patent_class, {}).get(route, "unknown")


def should_run(
    patent_class: PatentClass,
    route: str,
    completeness: float,
    incomplete_threshold: float = 0.95,
) -> tuple[bool, str]:
    """Resolve the route gate against the current completeness.

    Returns (should_run, reason).
      - "always"        → always run
      - "suppressed"    → never run (returns False)
      - "if_incomplete" → run iff completeness < threshold
      - "rare"          → only run with much lower threshold (0.5)
    """
    gate = gate_for(patent_class, route)
    if gate == "always":
        return True, f"gate=always for {patent_class.value}"
    if gate == "suppressed":
        return False, f"gate=suppressed for {patent_class.value}"
    if gate == "rare":
        if completeness < 0.5:
            return True, f"gate=rare, completeness {completeness:.0%} < 50%"
        return False, f"gate=rare, completeness {completeness:.0%} ≥ 50% — skip"
    if gate == "if_incomplete":
        if completeness < incomplete_threshold:
            return True, f"gate=if_incomplete, completeness {completeness:.0%} < {incomplete_threshold:.0%}"
        return False, f"gate=if_incomplete, completeness {completeness:.0%} ≥ {incomplete_threshold:.0%}"
    return False, f"gate=unknown for ({patent_class.value}, {route})"
