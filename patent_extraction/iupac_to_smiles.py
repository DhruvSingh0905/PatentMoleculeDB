"""Step 2.2 — IUPAC → SMILES conversion with Markush context.

Separate from page extraction (Step 2.1) for debuggability:
if a SMILES is wrong, we know it's a conversion error, not a parsing error.

Uses Markush scaffold context to help Claude disambiguate complex names.
"""

from __future__ import annotations

import logging

from . import config
from .api_client import call_claude_text
from .models import Compound, MarkushContext
from .extract_markush_context import get_context_for_prompt
from .smiles_utils import validate_smiles, canonicalize_smiles, get_inchikey, strip_salt, compute_drug_likeness

logger = logging.getLogger(__name__)

CONVERSION_PROMPT_TEMPLATE = """{markush_context}

Convert this IUPAC chemical name to a canonical SMILES string.
Preserve ALL stereochemistry (R/S, E/Z, cis/trans — enantiomers must be distinguished).

Name: {iupac_name}

Return ONLY the SMILES string, nothing else."""

RETRY_PROMPT_TEMPLATE = """{markush_context}

Your previous SMILES conversion was invalid (failed RDKit validation).
Please re-examine this IUPAC name carefully and provide a corrected canonical SMILES.

Name: {iupac_name}

Return ONLY the SMILES string, nothing else."""


def convert_iupac_to_smiles(
    compound: Compound,
    markush_context: MarkushContext | None = None,
) -> Compound:
    """Convert a compound's IUPAC name to SMILES via Claude.

    Modifies the compound in-place and returns it.
    Sets smiles_from_text, canonical_smiles, inchikey, drug_likeness.

    Args:
        compound: Compound with iupac_name populated.
        markush_context: Optional Markush context for disambiguation.

    Returns:
        The compound with SMILES fields populated (or failure_reason set).
    """
    if not compound.iupac_name:
        compound.failure_reason = "no_iupac_name"
        return compound

    context_str = ""
    if markush_context:
        context_str = get_context_for_prompt(markush_context)

    # First attempt
    prompt = CONVERSION_PROMPT_TEMPLATE.format(
        markush_context=context_str,
        iupac_name=compound.iupac_name,
    )

    smiles = call_claude_text(
        prompt=prompt,
        patent_id=compound.patent_id,
        compound_id=compound.example_number or "unknown",
    )

    if smiles:
        smiles = smiles.strip().split("\n")[0].strip()  # Take first line only

    # Validate
    if smiles and validate_smiles(smiles):
        return _finalize_compound(compound, smiles)

    # Retry once
    logger.info(
        f"Patent {compound.patent_id} {compound.example_number}: "
        f"First SMILES attempt invalid ('{smiles}'), retrying"
    )

    retry_prompt = RETRY_PROMPT_TEMPLATE.format(
        markush_context=context_str,
        iupac_name=compound.iupac_name,
    )

    smiles = call_claude_text(
        prompt=retry_prompt,
        patent_id=compound.patent_id,
        compound_id=f"{compound.example_number or 'unknown'}_retry",
    )

    if smiles:
        smiles = smiles.strip().split("\n")[0].strip()

    if smiles and validate_smiles(smiles):
        return _finalize_compound(compound, smiles)

    # Permanent failure — flag as input quality issue
    compound.failure_reason = "input_quality_issue"
    compound.processing_status = "failed"
    logger.warning(
        f"Patent {compound.patent_id} {compound.example_number}: "
        f"IUPAC→SMILES failed after retry. Name: {compound.iupac_name[:100]}"
    )
    return compound


def _finalize_compound(compound: Compound, smiles: str) -> Compound:
    """Finalize a compound with a validated SMILES string."""
    compound.smiles_from_text = smiles
    compound.canonical_smiles = canonicalize_smiles(smiles)
    compound.inchikey = get_inchikey(compound.canonical_smiles or smiles)

    # Salt handling
    salt_result = strip_salt(compound.canonical_smiles or smiles)
    compound.parent_smiles = salt_result.get("parent_smiles")
    compound.parent_inchikey = salt_result.get("parent_inchikey")

    # Drug-likeness
    compound.drug_likeness = compute_drug_likeness(compound.canonical_smiles or smiles)

    compound.processing_status = "text_done"
    return compound


def convert_batch(
    compounds: list[Compound],
    markush_context: MarkushContext | None = None,
) -> list[Compound]:
    """Convert a batch of compounds' IUPAC names to SMILES.

    Args:
        compounds: List of Compounds with iupac_name populated.
        markush_context: Markush context for this patent.

    Returns:
        Same list with SMILES fields populated.
    """
    success = 0
    failed = 0

    for compound in compounds:
        if not compound.iupac_name:
            continue

        convert_iupac_to_smiles(compound, markush_context)

        if compound.canonical_smiles:
            success += 1
        else:
            failed += 1

    patent_id = compounds[0].patent_id if compounds else "unknown"
    logger.info(
        f"Patent {patent_id}: IUPAC→SMILES batch complete. "
        f"Success: {success}, Failed: {failed}"
    )

    return compounds
