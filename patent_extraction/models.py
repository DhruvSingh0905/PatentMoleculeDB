"""Data models for patent molecule extraction."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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
    extraction_method: str | None = None       # "opsin_direct", "pubchem", "decimer_local", "opus_vision", etc.
    confidence_score: float | None = None       # 0.0-1.0
    inferred_stereochemistry: bool = False      # True if stereo was lost/inferred
    stereo_trusted: bool = True                 # False if stereo from LLM/OCR/cis-trans stripping

    # Processing status
    processing_status: Literal[
        "pending", "text_done", "image_done", "validated", "failed"
    ] = "pending"
    failure_reason: str | None = None


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
