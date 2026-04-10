"""Step 2.2 — IUPAC → SMILES conversion with Markush context.

Each IUPAC name gets its own API call (token-based pricing, no batch penalty).
Calls fire in PARALLEL via ThreadPoolExecutor for speed.

Separate from page extraction (Step 2.1) for debuggability.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .api_client import call_claude_text
from .models import Compound, MarkushContext
from .extract_markush_context import get_context_for_prompt
from .smiles_utils import validate_smiles, canonicalize_smiles, get_inchikey, strip_salt, compute_drug_likeness

logger = logging.getLogger(__name__)

MAX_PARALLEL = 10  # Concurrent API calls

CONVERSION_PROMPT = """{markush_context}

Convert this IUPAC name to a canonical SMILES string. Preserve all stereochemistry.

Name: {iupac_name}

ONLY output the SMILES string. Nothing else. No explanation."""

RETRY_PROMPT = """{markush_context}

Previous SMILES was invalid. Re-examine and output the correct SMILES.

Name: {iupac_name}

ONLY output the SMILES string. Nothing else."""


def _convert_single(compound: Compound, context_str: str) -> Compound:
    """Convert one compound's IUPAC name to SMILES. Called in parallel."""
    if not compound.iupac_name:
        compound.failure_reason = "no_iupac_name"
        return compound

    prompt = CONVERSION_PROMPT.format(
        markush_context=context_str,
        iupac_name=compound.iupac_name,
    )

    smiles = call_claude_text(
        prompt=prompt,
        patent_id=compound.patent_id,
        compound_id=compound.example_number or "unknown",
        max_tokens=200,
    )

    if smiles:
        smiles = smiles.strip().split("\n")[0].strip()

    if smiles and validate_smiles(smiles):
        return finalize_compound(compound, smiles)

    # Retry once
    retry_prompt = RETRY_PROMPT.format(
        markush_context=context_str,
        iupac_name=compound.iupac_name,
    )

    smiles = call_claude_text(
        prompt=retry_prompt,
        patent_id=compound.patent_id,
        compound_id=f"{compound.example_number or 'unknown'}_retry",
        max_tokens=200,
    )

    if smiles:
        smiles = smiles.strip().split("\n")[0].strip()

    if smiles and validate_smiles(smiles):
        return finalize_compound(compound, smiles)

    compound.failure_reason = "input_quality_issue"
    compound.processing_status = "failed"
    return compound


def finalize_compound(compound: Compound, smiles: str) -> Compound:
    """Finalize a compound with a validated SMILES string."""
    compound.smiles_from_text = smiles
    compound.canonical_smiles = canonicalize_smiles(smiles)
    compound.inchikey = get_inchikey(compound.canonical_smiles or smiles)

    salt_result = strip_salt(compound.canonical_smiles or smiles)
    compound.parent_smiles = salt_result.get("parent_smiles")
    compound.parent_inchikey = salt_result.get("parent_inchikey")

    compound.drug_likeness = compute_drug_likeness(compound.canonical_smiles or smiles)
    compound.processing_status = "text_done"
    return compound


def convert_iupac_to_smiles(
    compound: Compound,
    markush_context: MarkushContext | None = None,
) -> Compound:
    """Convert a single compound. For backward compat and tests."""
    context_str = get_context_for_prompt(markush_context) if markush_context else ""
    return _convert_single(compound, context_str)


def convert_batch(
    compounds: list[Compound],
    markush_context: MarkushContext | None = None,
) -> list[Compound]:
    """Convert all compounds' IUPAC names to SMILES in parallel.

    Fires up to MAX_PARALLEL individual API calls concurrently.
    """
    context_str = get_context_for_prompt(markush_context) if markush_context else ""
    patent_id = compounds[0].patent_id if compounds else "unknown"

    to_convert = [c for c in compounds if c.iupac_name]
    if not to_convert:
        return compounds

    logger.info(
        f"Patent {patent_id}: Converting {len(to_convert)} compounds "
        f"with {MAX_PARALLEL} parallel workers"
    )

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {
            executor.submit(_convert_single, c, context_str): c
            for c in to_convert
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                c = futures[future]
                logger.error(f"Conversion failed for {c.example_number}: {e}")
                c.failure_reason = str(e)
                c.processing_status = "failed"

    success = sum(1 for c in to_convert if c.canonical_smiles)
    failed = sum(1 for c in to_convert if c.processing_status == "failed")
    logger.info(
        f"Patent {patent_id}: IUPAC→SMILES complete. "
        f"Success: {success}, Failed: {failed}"
    )

    return compounds
