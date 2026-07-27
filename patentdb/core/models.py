"""Data models for patent molecule extraction."""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ── Missing-provenance registry ────────────────────────────────────
# Process-global counter populated by Compound's validator whenever a
# Compound is constructed with canonical_smiles but no provenance. The
# audit module reads this to emit an UNATTRIBUTED violation, so any
# missed call site surfaces loudly instead of silently producing
# route="unknown" buckets.
#
# Indexed by patent_id; thread-safe because Compound construction can
# happen from background pool threads.
_PROVENANCE_LOCK = threading.Lock()
_MISSING_PROVENANCE: dict[str, list[dict]] = {}
# Some construction sites (e.g., loading from cache JSON, loading bench
# fixtures) legitimately rebuild Compounds without re-deriving provenance.
# Use this context flag to suspend the validator during those operations.
_VALIDATOR_SUSPENDED = threading.local()


def suspend_provenance_validator():
    """Context-manager-style flag for legitimate Compound rehydration.

    Use as:
        from .models import suspend_provenance_validator
        with suspend_provenance_validator():
            c = Compound(**cached_dict)

    This avoids spurious "missing provenance" entries when re-loading
    cached step results that were produced by a legitimate route call.
    """
    class _Ctx:
        def __enter__(self):
            _VALIDATOR_SUSPENDED.flag = True
            return self
        def __exit__(self, *a):
            _VALIDATOR_SUSPENDED.flag = False
    return _Ctx()


def reset_missing_provenance(patent_id: str) -> None:
    """Clear the registry for a patent (called at start of run_patent)."""
    with _PROVENANCE_LOCK:
        _MISSING_PROVENANCE[patent_id] = []


def record_missing_provenance(patent_id: str, smiles: str, example_number: str | None,
                              extraction_method: str | None) -> None:
    """Record a Compound that arrived with SMILES but no provenance."""
    with _PROVENANCE_LOCK:
        bucket = _MISSING_PROVENANCE.setdefault(patent_id, [])
        # Cap to avoid runaway memory if a route is broken
        if len(bucket) < 200:
            bucket.append({
                "smiles": smiles,
                "example_number": example_number,
                "extraction_method": extraction_method,
            })


def get_missing_provenance(patent_id: str) -> list[dict]:
    """Return the missing-provenance entries for a patent."""
    with _PROVENANCE_LOCK:
        return list(_MISSING_PROVENANCE.get(patent_id, []))


class ConfidenceTier(int, Enum):
    """Confidence tiers for extracted compounds."""
    DUAL_CONFIRMED = 1        # 2+ sources agree
    SINGLE_VALIDATED = 2      # 1 source, passed RDKit + drug-likeness
    SINGLE_UNVALIDATED = 3    # 1 source, RDKit only


class CompoundSource(str, Enum):
    EXEMPLIFIED = "exemplified"
    MARKUSH_ENUMERATED = "markush_enumerated"


class AssayResult(BaseModel):
    """A single assay measurement for a compound."""
    assay_name: str
    value_raw: str                              # "12", "A", ">2", "1-100"
    value_numeric: float | None = None
    qualifier: str | None = None                # ">", "<", "~", "range"
    unit: str = ""
    n_runs: int | None = None                   # Number of experimental replicates


class DrugLikenessResult(BaseModel):
    """Drug-likeness metrics for a compound."""
    mw: float
    logp: float
    hbd: int
    hba: int
    lipinski_passes: bool
    pains_clean: bool
    sa_score: float


class MarkushContext(BaseModel):
    """Markush scaffold context extracted from patent claims.

    Used as context for IUPAC→SMILES conversion, NOT for enumeration.
    """
    patent_id: str
    scaffold_smiles: str | None = None          # RDKit-validated scaffold SMILES
    scaffold_description: str = ""              # Natural language description
    r_group_definitions: dict[str, list[str]] = Field(default_factory=dict)
    position_mapping: dict[str, str] = Field(default_factory=dict)  # {R-label: placeholder_num}
    quality: Literal["high_confidence", "partial", "failed"] = "failed"
    # high_confidence: valid SMILES + complete R-group defs
    # partial: natural language only, SMILES invalid or absent
    # failed: couldn't extract coherent scaffold


class IupacSource(str, Enum):
    """How the IUPAC name was obtained."""
    PATENT_VERBATIM = "patent_verbatim"         # Directly extracted from patent text
    CORRECTED = "corrected"                     # Image SMILES won, IUPAC regenerated
    GENERATED = "generated"                     # From scaffold + substituent table


class SynthesisStep(BaseModel):
    """A single step in a compound's synthesis route."""
    step_number: int
    starting_materials: list[str] = Field(default_factory=list)  # IUPAC or common names
    reagents: list[str] = Field(default_factory=list)
    conditions: str | None = None               # "reflux, 2h, THF"
    product: str = ""                           # Intermediate name or "final product"


class CompoundProvenance(BaseModel):
    """Structured provenance for a single extraction event.

    A Compound carries one `provenance` (the primary source) and optionally
    a `provenance_chain` listing every other route/stage that also produced
    or merged into it. This separation lets `analyze_by_method` and the
    audit/joiner answer questions like "which routes contributed?" and
    "did Markush actually fill in this Example?" without losing data
    during connectivity-key dedup.
    """
    route: Literal[
        "google_patents_example",
        "google_patents_table",
        "adaptive",
        "synthesis_block",
        "image_pipeline",
        "markush_enumeration",
        "manifest_inline",
        "page_extraction",
        "table_extraction",
        "unknown",
    ] = "unknown"
    stage: str = ""                              # opsin_direct, llm_cleaned, decimer_local, etc.
    page: int | None = None
    image_bbox: list[int] | None = None
    image_path: str | None = None
    # Markush-specific
    scaffold_inchikey: str | None = None         # connectivity key (14-char) of the core scaffold
    scaffold_index: int | None = None            # which scaffold in the patent's list
    r_assignments: dict[str, str] | None = None
    enumeration_mode: str | None = None          # "scored" | "decomposition" | "exhaustive"
    scoring_rank: int | None = None              # 1-indexed position in scored output
    # Table-specific
    table_number: int | None = None
    table_row: int | None = None
    # Cascade chain — each upstream stage that touched this compound
    chain: list[str] = Field(default_factory=list)


class Compound(BaseModel):
    """A single extracted compound with all metadata."""
    patent_id: str
    example_number: str | None = None           # Full verbatim: "Example 1", not "1"
    iupac_name: str | None = None
    iupac_source: IupacSource = IupacSource.PATENT_VERBATIM

    # SMILES from different sources
    smiles_from_inline: str | None = None       # Regex-extracted SMILES/InChI
    smiles_from_text: str | None = None         # IUPAC→SMILES via Claude
    smiles_from_image: str | None = None        # Image→SMILES via Claude Vision

    # Validated output
    canonical_smiles: str | None = None         # Final canonical SMILES
    parent_smiles: str | None = None            # Salt-stripped parent
    inchikey: str | None = None
    parent_inchikey: str | None = None

    # Metadata
    source: CompoundSource = CompoundSource.EXEMPLIFIED
    confidence_tier: ConfidenceTier = ConfidenceTier.SINGLE_UNVALIDATED
    is_intermediate: bool = False
    markush_parent: str | None = None
    r_groups: dict[str, str] | None = None      # {"1": "methyl", "2": "F"}
    image_path: str | None = None
    source_page: int | None = None
    image_bbox: list[int] | None = None         # [x1, y1, x2, y2]

    # MS data for intermediate validation
    ms_mh_plus: float | None = None             # (M+H)+ value from MS/LCMS on same page
    mw_validated: bool | None = None             # True if ExactMolWt matches MS ±1.5 Da

    # Synthesis route (extracted from same page, for future retrosynthesis project)
    synthesis_route: list[SynthesisStep] = Field(default_factory=list)

    # Assay & drug-likeness
    assay_data: list[AssayResult] = Field(default_factory=list)
    drug_likeness: DrugLikenessResult | None = None

    # Provenance — how this compound was extracted
    extraction_method: str | None = None       # legacy single-string tag (kept for backward compat)
    provenance: CompoundProvenance | None = None  # NEW — structured primary provenance
    provenance_chain: list[CompoundProvenance] = Field(default_factory=list)  # NEW — every merge entry
    confidence_score: float | None = None       # 0.0-1.0
    inferred_stereochemistry: bool = False      # True if stereo was lost/inferred
    stereo_trusted: bool = True                 # False if stereo from LLM/OCR/cis-trans stripping
    lm_normalized: bool = False                 # True if name passed through any LM stage (3a/3b/synthesis_block)

    # Processing status
    processing_status: Literal[
        "pending", "text_done", "image_done", "validated", "failed"
    ] = "pending"
    failure_reason: str | None = None

    # ── Provenance integrity check ────────────────────────────────
    # If a Compound arrives with canonical_smiles but no provenance, that
    # means a route call site forgot to populate it. We don't want to raise
    # (cached data + bench fixtures can legitimately lack provenance), but
    # we DO want it to surface as an audit violation. Record into the
    # process-global registry; the audit module reads it at end of run.
    @model_validator(mode="after")
    def _check_provenance_present(self):
        if getattr(_VALIDATOR_SUSPENDED, "flag", False):
            return self
        if self.canonical_smiles and self.provenance is None:
            record_missing_provenance(
                self.patent_id,
                smiles=self.canonical_smiles,
                example_number=self.example_number,
                extraction_method=self.extraction_method,
            )
        return self


class RGroupDef(BaseModel):
    """Definition of a single R-group position in a Markush structure."""
    label: str                                  # "R1", "R2", "G", "X"
    options_text: list[str] = Field(default_factory=list)  # ["C1-C6 alkyl", "phenyl"]
    options_smiles: list[str] = Field(default_factory=list)  # ["C", "CC", "c1ccccc1"]
    option_type: str = "discrete"               # "discrete", "range", "nested"


class MarkushFormula(BaseModel):
    """Canonical Markush representation — ground truth for enumeration."""
    patent_id: str
    formula_name: str = "Formula I"             # "Formula I", "Formula II"
    core_smarts: str | None = None              # RDKit SMARTS with R-attachment points
    core_smiles: str | None = None              # Flat SMILES with [1*], [2*] placeholders
    r_groups: dict[str, RGroupDef] = Field(default_factory=dict)
    exceptions: list[str] = Field(default_factory=list)
    difficulty: str = "HIGH"                    # LOW/MED/HIGH
    source: str = "text"                        # "text", "image", "reconciled"


class MarkushMapping(BaseModel):
    """Result of mapping an IUPAC name to R-group assignments."""
    r_assignments: dict[str, str] = Field(default_factory=dict)  # {R1: "methyl"}
    instance_smiles: str | None = None          # Built from core + R-values
    confidence: str = "low"                     # "high", "medium", "low"
    alignment_method: str = "unknown"           # "symbolic", "lm", "reconciled"


class MarkushDefinition(BaseModel):
    """A parsed Markush structure definition from patent claims."""
    formula_name: str                           # "Formula I", "Formula II"
    core_scaffold: str                          # SMILES with [1*], [2*] attachment points
    variables: dict[str, list[str]]             # {"1": ["C", "CC", ...], "2": ["F", "Cl"]}
    claim_chain: list[str] = Field(default_factory=list)
    estimated_combinations: int = 0
    patent_id: str = ""


class PageManifest(BaseModel):
    """Detection results for a single patent."""
    patent_id: str
    image_pages: list[int] = Field(default_factory=list)
    example_pages: list[int] = Field(default_factory=list)
    markush_pages: list[int] = Field(default_factory=list)
    table_pages: list[int] = Field(default_factory=list)
    inline_smiles_pages: list[int] = Field(default_factory=list)
    inline_structures: list[dict] = Field(default_factory=list)
    # Page-targeting indices for the image-route cascade
    example_to_page: dict[str, int] = Field(default_factory=dict)   # "Example 5" → 29
    formula_pages: list[int] = Field(default_factory=list)          # pages with "Formula (I)", "Scheme N"
    image_dense_pages: list[int] = Field(default_factory=list)      # heavy-image / low-text pages


class PatentResult(BaseModel):
    """Complete extraction results for a patent."""
    patent_id: str
    exemplified_compounds: list[Compound] = Field(default_factory=list)
    markush_enumerated: list[Compound] = Field(default_factory=list)
    markush_definitions: list[MarkushDefinition] = Field(default_factory=list)
    manifest: PageManifest | None = None
    stats: dict = Field(default_factory=dict)


class ParsedElement(BaseModel):
    """A single parsed element from the markdown annotation format."""
    element_type: str                           # "text", "image", "table", etc.
    bbox: list[int]                             # [x1, y1, x2, y2]
    content: str = ""                           # Text content following the marker
