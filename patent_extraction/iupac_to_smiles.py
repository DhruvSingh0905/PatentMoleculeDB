"""Step 2.2 — IUPAC → SMILES conversion with Markush context.

Batched: sends 15 IUPAC names per API call (not 1-per-call).
Parallelized: uses concurrent.futures for multiple batches simultaneously.

Separate from page extraction (Step 2.1) for debuggability:
if a SMILES is wrong, we know it's a conversion error, not a parsing error.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .api_client import call_claude_text
from .models import Compound, MarkushContext
from .extract_markush_context import get_context_for_prompt
from .smiles_utils import validate_smiles, canonicalize_smiles, get_inchikey, strip_salt, compute_drug_likeness

logger = logging.getLogger(__name__)

BATCH_SIZE = 15  # IUPAC names per API call
MAX_PARALLEL = 5  # Concurrent API calls

BATCH_PROMPT_TEMPLATE = """{markush_context}

Convert each IUPAC chemical name below to a canonical SMILES string.
Preserve ALL stereochemistry (R/S, E/Z, cis/trans).

Return a JSON array of objects, one per compound, in the SAME ORDER as the input:
[{{"id": "Example 1", "smiles": "SMILES_STRING"}}]

If you cannot convert a name, use "smiles": null for that entry.
Return ONLY the JSON array. No explanations.

Compounds:
{compounds_list}"""

SINGLE_RETRY_PROMPT = """{markush_context}

Your previous SMILES for this compound was invalid (failed RDKit validation).
Re-examine carefully and provide the corrected canonical SMILES.
Preserve ALL stereochemistry.

Name: {iupac_name}

Return ONLY the SMILES string. Nothing else. No explanation."""


def _build_compounds_list(compounds: list[Compound]) -> str:
    """Format compound list for the batch prompt."""
    lines = []
    for c in compounds:
        lines.append(f'- ID: "{c.example_number}" | Name: {c.iupac_name}')
    return "\n".join(lines)


def _parse_batch_response(response: str, compounds: list[Compound]) -> dict[str, str]:
    """Parse Claude's batch SMILES response into {id: smiles} mapping."""
    if not response:
        return {}

    cleaned = response.strip()

    # Extract JSON array from response (handle code fences, surrounding text)
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                cleaned = part
                break

    if not cleaned.startswith("["):
        start = cleaned.find("[")
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "[":
                    depth += 1
                elif cleaned[i] == "]":
                    depth -= 1
                if depth == 0:
                    cleaned = cleaned[start:i+1]
                    break

    try:
        results = json.loads(cleaned)
        if not isinstance(results, list):
            return {}

        mapping = {}
        for item in results:
            cid = item.get("id", "")
            smiles = item.get("smiles")
            if cid and smiles:
                mapping[cid] = smiles
        return mapping

    except json.JSONDecodeError:
        logger.warning(f"Batch SMILES response not valid JSON: {cleaned[:200]}")
        return {}


def _convert_batch_chunk(
    compounds: list[Compound],
    context_str: str,
    patent_id: str,
) -> dict[str, str]:
    """Send a batch of compounds to Claude for SMILES conversion.

    Returns {compound_id: smiles} mapping for valid results.
    """
    compounds_list = _build_compounds_list(compounds)
    prompt = BATCH_PROMPT_TEMPLATE.format(
        markush_context=context_str,
        compounds_list=compounds_list,
    )

    response = call_claude_text(
        prompt=prompt,
        patent_id=patent_id,
        compound_id=f"batch_{compounds[0].example_number}_{len(compounds)}",
        max_tokens=800,  # 15 SMILES × ~50 chars each + JSON overhead
    )

    return _parse_batch_response(response, compounds)


def _retry_single(compound: Compound, context_str: str) -> str | None:
    """Retry a single failed SMILES conversion."""
    prompt = SINGLE_RETRY_PROMPT.format(
        markush_context=context_str,
        iupac_name=compound.iupac_name,
    )

    response = call_claude_text(
        prompt=prompt,
        patent_id=compound.patent_id,
        compound_id=f"{compound.example_number or 'unknown'}_retry",
        max_tokens=200,
    )

    if response:
        smiles = response.strip().split("\n")[0].strip()
        if validate_smiles(smiles):
            return smiles
    return None


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


# Keep backward-compatible single-call function for tests
def convert_iupac_to_smiles(
    compound: Compound,
    markush_context: MarkushContext | None = None,
) -> Compound:
    """Convert a single compound's IUPAC name to SMILES.

    For backward compatibility and retry logic. Production uses convert_batch.
    """
    if not compound.iupac_name:
        compound.failure_reason = "no_iupac_name"
        return compound

    context_str = get_context_for_prompt(markush_context) if markush_context else ""

    # Use batch of 1
    mapping = _convert_batch_chunk([compound], context_str, compound.patent_id)
    smiles = mapping.get(compound.example_number)

    if smiles and validate_smiles(smiles):
        return finalize_compound(compound, smiles)

    # Retry
    smiles = _retry_single(compound, context_str)
    if smiles:
        return finalize_compound(compound, smiles)

    compound.failure_reason = "input_quality_issue"
    compound.processing_status = "failed"
    return compound


def convert_batch(
    compounds: list[Compound],
    markush_context: MarkushContext | None = None,
) -> list[Compound]:
    """Convert all compounds' IUPAC names to SMILES using batched parallel calls.

    Sends BATCH_SIZE compounds per API call, MAX_PARALLEL calls concurrently.
    """
    context_str = get_context_for_prompt(markush_context) if markush_context else ""
    patent_id = compounds[0].patent_id if compounds else "unknown"

    # Filter to compounds that have IUPAC names
    to_convert = [c for c in compounds if c.iupac_name]
    if not to_convert:
        return compounds

    # Split into batches
    batches = [to_convert[i:i+BATCH_SIZE] for i in range(0, len(to_convert), BATCH_SIZE)]

    logger.info(
        f"Patent {patent_id}: Converting {len(to_convert)} compounds "
        f"in {len(batches)} batches of {BATCH_SIZE}, {MAX_PARALLEL} parallel"
    )

    # Process batches in parallel
    all_mappings: dict[str, str] = {}
    needs_retry: list[Compound] = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {
            executor.submit(_convert_batch_chunk, batch, context_str, patent_id): batch
            for batch in batches
        }

        for future in as_completed(futures):
            batch = futures[future]
            try:
                mapping = future.result()
                all_mappings.update(mapping)
            except Exception as e:
                logger.error(f"Batch failed: {e}")

    # Apply results to compounds
    success = 0
    for compound in to_convert:
        smiles = all_mappings.get(compound.example_number)

        if smiles and validate_smiles(smiles):
            finalize_compound(compound, smiles)
            success += 1
        else:
            # Queue for individual retry
            needs_retry.append(compound)

    # Retry failures individually (sequential — these are rare)
    retry_success = 0
    for compound in needs_retry:
        smiles = _retry_single(compound, context_str)
        if smiles:
            finalize_compound(compound, smiles)
            retry_success += 1
        else:
            compound.failure_reason = "input_quality_issue"
            compound.processing_status = "failed"

    total_failed = len(needs_retry) - retry_success
    logger.info(
        f"Patent {patent_id}: IUPAC→SMILES complete. "
        f"Batch success: {success}, Retry success: {retry_success}, Failed: {total_failed}"
    )

    return compounds
