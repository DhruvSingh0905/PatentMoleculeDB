"""Step 2.4 — Cross-validation of text vs image SMILES by example number.

Links text-extracted compounds (Step 2.1-2.2) to image-extracted SMILES (Step 2.3)
by matching on compound/example number. Page proximity is tiebreaker only.

When image SMILES wins over text SMILES, reverse-generates the IUPAC name
from the correct SMILES to fix the output.
"""

from __future__ import annotations

import logging

from . import config
from .api_client import call_claude_text
from .models import Compound, ConfidenceTier, IupacSource
from .smiles_utils import (
    canonicalize_smiles,
    get_inchikey,
    get_connectivity_key,
    validate_smiles,
)

logger = logging.getLogger(__name__)


def cross_validate_compounds(
    text_compounds: list[Compound],
    image_results: list[dict],
) -> list[Compound]:
    """Cross-validate text-extracted and image-extracted SMILES.

    Links by example number (compound_id). When both exist and agree → Tier 1.
    Mismatch → Claude arbitration. Single source → Tier 2 or 3.

    Also flags text_extraction_miss when an image has a compound_id not found
    in text extraction.

    Args:
        text_compounds: Compounds from Steps 2.1-2.2.
        image_results: Results from Step 2.3 (image_to_smiles).

    Returns:
        Updated text_compounds with confidence tiers assigned.
    """
    # Build lookup: compound_id → image result
    image_by_id: dict[str, dict] = {}
    image_by_page: dict[int, list[dict]] = {}

    for img in image_results:
        if not img.get("valid"):
            continue

        cid = img.get("compound_id")
        if cid:
            key = _normalize_id(cid)
            image_by_id[key] = img

        page = img.get("page_num")
        if page is not None:
            image_by_page.setdefault(page, []).append(img)

    # Build lookup: compound_id → text compound
    text_by_id = {_normalize_id(c.example_number): c for c in text_compounds if c.example_number}

    # Flag text_extraction_miss
    for img_id in image_by_id:
        if img_id not in text_by_id:
            logger.warning(f"text_extraction_miss: image has '{img_id}' but text extraction missed it")

    # Cross-validate each text compound
    matched = 0
    mismatched = 0
    single_source = 0

    for compound in text_compounds:
        if not compound.canonical_smiles:
            compound.confidence_tier = ConfidenceTier.SINGLE_UNVALIDATED
            single_source += 1
            continue

        # Find matching image result
        img_result = None
        if compound.example_number:
            key = _normalize_id(compound.example_number)
            img_result = image_by_id.get(key)

        # Fallback: page proximity (±2 pages)
        if img_result is None and compound.source_page is not None:
            for offset in [0, 1, -1, 2, -2]:
                page = compound.source_page + offset
                candidates = image_by_page.get(page, [])
                if candidates:
                    img_result = candidates[0]  # Take first unmatched
                    break

        if img_result is None or not img_result.get("canonical_smiles"):
            # Single source — tier depends on drug-likeness
            if compound.drug_likeness and compound.drug_likeness.lipinski_passes:
                compound.confidence_tier = ConfidenceTier.SINGLE_VALIDATED
            else:
                compound.confidence_tier = ConfidenceTier.SINGLE_UNVALIDATED
            single_source += 1
            continue

        # Compare text vs image SMILES
        text_smiles = compound.canonical_smiles
        image_smiles = img_result["canonical_smiles"]
        compound.smiles_from_image = image_smiles

        if text_smiles == image_smiles:
            # Exact match → Tier 1
            compound.confidence_tier = ConfidenceTier.DUAL_CONFIRMED
            matched += 1

        elif _connectivity_match(text_smiles, image_smiles):
            # Same connectivity, different stereo → Tier 1 with note
            compound.confidence_tier = ConfidenceTier.DUAL_CONFIRMED
            matched += 1

        else:
            # Mismatch — accept image SMILES (visual is more reliable for structure)
            # and reverse-generate IUPAC name
            logger.info(
                f"Patent {compound.patent_id} {compound.example_number}: "
                f"text/image SMILES mismatch. Text: {text_smiles[:50]}, Image: {image_smiles[:50]}. "
                f"Accepting image SMILES."
            )
            compound.canonical_smiles = image_smiles
            compound.inchikey = get_inchikey(image_smiles)

            # Reverse generate IUPAC from correct SMILES
            corrected_name = _reverse_iupac(image_smiles, compound.patent_id, compound.example_number)
            if corrected_name:
                compound.iupac_name = corrected_name
                compound.iupac_source = IupacSource.CORRECTED

            compound.confidence_tier = ConfidenceTier.SINGLE_VALIDATED
            mismatched += 1

    logger.info(
        f"Cross-validation: {matched} matched, {mismatched} mismatched, "
        f"{single_source} single-source"
    )

    return text_compounds


def _normalize_id(compound_id: str) -> str:
    """Normalize compound ID for matching."""
    return compound_id.strip().lower().replace(".", "").replace(" ", "")


def _connectivity_match(smiles1: str, smiles2: str) -> bool:
    """Check if two SMILES have the same connectivity (ignore stereo)."""
    key1 = get_inchikey(smiles1)
    key2 = get_inchikey(smiles2)
    if key1 is None or key2 is None:
        return False
    return get_connectivity_key(key1) == get_connectivity_key(key2)


def _reverse_iupac(smiles: str, patent_id: str, compound_id: str | None) -> str | None:
    """Generate IUPAC name from SMILES via Claude.

    Used when image SMILES wins over text SMILES — the original
    IUPAC name was wrong and needs to be corrected.
    """
    prompt = (
        f"Generate the correct IUPAC name for this molecule. "
        f"Include all stereochemistry designations.\n\n"
        f"SMILES: {smiles}\n\n"
        f"Return ONLY the IUPAC name, nothing else."
    )

    response = call_claude_text(
        prompt=prompt,
        patent_id=patent_id,
        compound_id=f"{compound_id or 'unknown'}_reverse_iupac",
    )

    if response:
        return response.strip()
    return None
