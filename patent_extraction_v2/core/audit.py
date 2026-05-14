"""Automated pipeline-wiring audit.

Replaces the documentation-only "audit before reporting" discipline in
CLAUDE.md with code that runs at the end of every patent extraction.

For each route the pipeline could have invoked, we record:
- Did it fire?
- If skipped, why?
- How many compounds did it produce?
- How many were novel (not duplicates of upstream routes)?
- What did it cost?
- How long did it take?

We then validate against patent-class expectations: e.g., a Class-C patent
that ran the image route triggers a violation; a Markush step that fired
but spent money producing 0 novel compounds triggers a violation.

The result writes to `output/results/{patent_id}/audit.json` so a human
or CI script can grep for `wiring_violations`.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .patent_class import PatentClass, gate_for
from .smiles_utils import get_inchikey, get_connectivity_key

if TYPE_CHECKING:
    from .google_patents import FormatAudit
    from .models import Compound

logger = logging.getLogger(__name__)


class RouteRecord(BaseModel):
    """One row of the pipeline-wiring audit, per route invocation."""
    name: str
    fired: bool = False
    skipped_reason: str | None = None
    compounds_extracted: int = 0     # raw output from this route
    novel_connectivity: int = 0      # how many were new vs upstream union
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


class RouteTimer:
    """Context manager that builds a RouteRecord. Use with `with` blocks
    around each route invocation in pipeline.py.
    """
    def __init__(self, name: str, upstream_conn_keys: set[str]):
        self.record = RouteRecord(name=name)
        self.upstream_keys = upstream_conn_keys
        self._start: float = 0.0
        self._cost_at_start: float = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        # Best-effort: snapshot current cumulative cost so we can subtract later
        try:
            from .cost_tracker import cost_tracker
            self._cost_at_start = cost_tracker.total_cost
        except Exception:
            self._cost_at_start = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.record.duration_seconds = round(time.monotonic() - self._start, 3)
        try:
            from .cost_tracker import cost_tracker
            self.record.cost_usd = round(cost_tracker.total_cost - self._cost_at_start, 4)
        except Exception:
            pass

    def fired(self):
        self.record.fired = True

    def skipped(self, reason: str):
        self.record.fired = False
        self.record.skipped_reason = reason

    def add_compounds(self, compounds: list):
        """Record output count + novel-connectivity vs upstream."""
        self.record.compounds_extracted = len(compounds)
        novel = 0
        for c in compounds:
            smi = getattr(c, 'canonical_smiles', None)
            if not smi:
                continue
            ik = get_inchikey(smi)
            if ik:
                conn = get_connectivity_key(ik)
                if conn not in self.upstream_keys:
                    novel += 1
                    self.upstream_keys.add(conn)
        self.record.novel_connectivity = novel


class PatentAuditResult(BaseModel):
    """Per-patent audit summary, persisted to audit.json."""
    patent_id: str
    patent_class: str   # PatentClass.value (str)
    class_reason: str
    audit_signals: dict = Field(default_factory=dict)   # FormatAudit field-by-field
    routes: list[RouteRecord] = Field(default_factory=list)
    final_compound_count: int = 0
    final_with_smiles: int = 0
    estimated_compounds: int = 0
    completeness: float = 0.0
    bdb_recall: float | None = None
    markush_roi: dict | None = None
    # Compounds that arrived with canonical_smiles but no provenance set.
    # Populated by Compound's @model_validator → models.record_missing_provenance().
    # Surfaces as an UNATTRIBUTED wiring violation: a route call site
    # forgot to populate provenance, producing route="unknown" buckets
    # the audit module would otherwise treat as "no route fired".
    missing_provenance_count: int = 0
    missing_provenance_samples: list[dict] = Field(default_factory=list)
    # Assay-extraction cascade diagnostic. Populated by extract_assays_for_patent
    # with stage_counts (baseline/repaired/lm/failed) so the audit can detect
    # "LM blasting" (Sonnet firing too often) and "ASSAY_DEAD" (assay tables
    # exist but no compound got data).
    assay_extraction_diagnostic: dict | None = None
    wiring_violations: list[str] = Field(default_factory=list)

    def validate_wiring(self) -> list[str]:
        """Run all wiring checks, populate self.wiring_violations, return list."""
        violations: list[str] = []

        # Check 1: routes that fired but yielded ZERO compounds — almost
        # always a wiring bug (function returned empty list silently).
        for r in self.routes:
            if r.fired and r.compounds_extracted == 0:
                violations.append(
                    f"WIRING: {r.name} FIRED but yielded 0 compounds "
                    f"(cost ${r.cost_usd:.3f}, took {r.duration_seconds}s) — likely a wiring bug"
                )

        # Check 2: routes that fired with compounds but contributed 0 novel —
        # pure duplication. Not necessarily a bug, but worth flagging as cost/no-value.
        for r in self.routes:
            if r.fired and r.compounds_extracted > 0 and r.novel_connectivity == 0:
                violations.append(
                    f"DUPLICATION: {r.name} produced {r.compounds_extracted} compounds "
                    f"but 0 were novel (all duplicates of upstream); cost ${r.cost_usd:.3f}"
                )

        # Check 3: class-specific expectations
        try:
            cls_enum = PatentClass(self.patent_class)
        except Exception:
            cls_enum = None

        if cls_enum is not None:
            # Class C should suppress image route
            if cls_enum == PatentClass.C:
                for r in self.routes:
                    if r.name == "image_pipeline" and r.fired:
                        violations.append(
                            f"ROUTING: Class-C patent (substituent table) ran image_pipeline "
                            f"({r.compounds_extracted} compounds, ${r.cost_usd:.3f}) — "
                            f"should have been suppressed"
                        )

            # Image route should fire for Class D unless explicitly skipped
            if cls_enum == PatentClass.D:
                img_records = [r for r in self.routes if r.name == "image_pipeline"]
                if img_records and not any(r.fired for r in img_records):
                    violations.append(
                        f"ROUTING: Class-D patent (image-heavy) did NOT run image_pipeline — "
                        f"check why ({img_records[0].skipped_reason})"
                    )

            # Class A with very high recall shouldn't run Markush (waste of cost)
            if cls_enum == PatentClass.A and self.completeness >= 0.95:
                for r in self.routes:
                    if r.name == "markush_enumeration" and r.fired and r.cost_usd > 0:
                        violations.append(
                            f"OVERWORK: Class-A patent at {self.completeness:.0%} recall "
                            f"ran Markush enumeration costing ${r.cost_usd:.3f} — should have been gated"
                        )

        # Check 4: Markush ROI sanity
        if self.markush_roi and self.markush_roi.get("fired"):
            novel = self.markush_roi.get("novel_connectivity", 0)
            cost = self.markush_roi.get("lm_cost_usd", 0.0)
            if novel == 0 and cost > 0:
                violations.append(
                    f"MARKUSH ROI: spent ${cost:.3f} for 0 novel connectivity keys"
                )
            # Sub-check: ROI must be sourced from attribution, not a
            # patent_spend() diff. Diff-sourced ROI is unreliable when
            # async upstream calls land mid-Markush.
            if self.markush_roi.get("lm_cost_source") not in (None, "attributed"):
                violations.append(
                    f"ATTRIBUTION: Markush ROI cost is sourced from "
                    f"'{self.markush_roi.get('lm_cost_source')}' instead of "
                    f"attribution — wrap the Markush call in cost_tracker.attribute()"
                )

        # Check 5: any Compound that arrived with canonical_smiles but no
        # provenance means a call site forgot the wiring. The validator
        # already logged it; surface as an audit violation so --strict-audit
        # can fail CI.
        if self.missing_provenance_count > 0:
            sample_ids = [
                e.get("example_number") or e.get("smiles", "")[:30]
                for e in self.missing_provenance_samples[:3]
            ]
            violations.append(
                f"UNATTRIBUTED: {self.missing_provenance_count} compound(s) "
                f"arrived with canonical_smiles but provenance=None — "
                f"a route call site forgot to populate it. Examples: {sample_ids}"
            )

        # Check 6: assay-extraction cascade health
        # 6a) ASSAY_DEAD — patent has assay tables but extraction got nothing
        # 6b) ASSAY_LM_BLASTING — Sonnet firing too often (parser regression)
        if self.assay_extraction_diagnostic:
            d = self.assay_extraction_diagnostic
            expected_pages = d.get("expected_assay_pages", 0)
            n_compounds_w_assays = d.get("compounds_with_assays", 0)
            if expected_pages > 5 and n_compounds_w_assays == 0:
                violations.append(
                    f"ASSAY_DEAD: {expected_pages} pages have assay tables but 0 "
                    f"compounds got assay data — extraction completely broken "
                    f"(stages: {d.get('stage_counts')})"
                )
            n_lm = d.get("lm_call_count", 0)
            lm_ratio = d.get("lm_ratio", 0.0)
            if n_lm > 5 and lm_ratio > 0.20:
                lm_cost = d.get("lm_cost_usd", 0.0)
                violations.append(
                    f"ASSAY_LM_BLASTING: Sonnet fired on {n_lm} pages "
                    f"({lm_ratio:.0%} of attempts, ${lm_cost:.3f}) — local "
                    f"parser strategies failing too often. Likely a parser "
                    f"regression rather than genuinely garbled OCR."
                )

        self.wiring_violations = violations
        return violations


def build_audit_result(
    patent_id: str,
    patent_class: PatentClass,
    class_reason: str,
    audit: "FormatAudit",
    route_records: list[RouteRecord],
    final_compounds: list,
    enumerated_compounds: list,
    markush_roi: dict | None,
    bdb_recall: float | None = None,
    missing_provenance: list[dict] | None = None,
    assay_extraction_diagnostic: dict | None = None,
) -> PatentAuditResult:
    """Assemble a PatentAuditResult from collected route records + final state."""
    final_with_smiles = sum(1 for c in final_compounds if c.canonical_smiles)
    estimated = max(getattr(audit, 'estimated_compounds', 0), 1)
    completeness = final_with_smiles / estimated

    # Summarize audit signals
    audit_signals = {
        "total_example_markers": getattr(audit, 'total_example_markers', 0),
        "total_cpd_markers": getattr(audit, 'total_cpd_markers', 0),
        "has_compound_table": getattr(audit, 'has_compound_table', False),
        "has_substituent_table": getattr(audit, 'has_substituent_table', False),
        "has_claims_list": getattr(audit, 'has_claims_list', False),
        "estimated_compounds": getattr(audit, 'estimated_compounds', 0),
        "format_hints": getattr(audit, 'format_hints', []),
    }

    missing_prov = missing_provenance or []
    result = PatentAuditResult(
        patent_id=patent_id,
        patent_class=patent_class.value if hasattr(patent_class, 'value') else str(patent_class),
        class_reason=class_reason,
        audit_signals=audit_signals,
        routes=route_records,
        final_compound_count=len(final_compounds) + len(enumerated_compounds),
        final_with_smiles=final_with_smiles,
        estimated_compounds=estimated,
        completeness=round(completeness, 4),
        bdb_recall=bdb_recall,
        markush_roi=markush_roi,
        missing_provenance_count=len(missing_prov),
        missing_provenance_samples=missing_prov[:10],   # cap for JSON sanity
        assay_extraction_diagnostic=assay_extraction_diagnostic,
    )
    result.validate_wiring()
    return result
