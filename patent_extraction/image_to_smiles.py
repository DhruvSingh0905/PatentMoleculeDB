"""Step 2.3b — Convert structure images to SMILES via Claude Vision.

Each image is sent individually to Claude Vision. Claude returns the SMILES
string and (if visible) the example number near the structure.

Results are used for cross-validation against text-extracted SMILES (Step 2.4).
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import config
from .api_client import call_claude_vision
from .smiles_utils import validate_smiles, canonicalize_smiles

logger = logging.getLogger(__name__)

IMAGE_PROMPT = """This is an image from a pharmaceutical patent.

If this is a chemical structure diagram, do the following:
1. Convert the structure to a canonical SMILES string. Preserve stereochemistry (wedge/dash bonds = R/S).
2. If an example number or compound identifier is visible near the structure, extract it.

Return a JSON object:
{"smiles": "canonical SMILES string", "compound_id": "Example 1" or null}

If this is NOT a chemical structure (e.g., a reaction scheme, flow chart, graph, or diagram):
Return: {"smiles": null, "compound_id": null}

Return ONLY the JSON object."""


def convert_image_to_smiles(
    image_path: str | Path,
    patent_id: str = "",
    page_num: int | None = None,
) -> dict:
    """Convert a structure image to SMILES via Claude Vision.

    Args:
        image_path: Path to the PNG image file.
        patent_id: Patent identifier for logging.
        page_num: Page number for logging.

    Returns:
        Dict with keys: smiles (str|None), compound_id (str|None),
        canonical_smiles (str|None), valid (bool)
    """
    result = {
        "smiles": None,
        "compound_id": None,
        "canonical_smiles": None,
        "valid": False,
        "image_path": str(image_path),
        "page_num": page_num,
    }

    response = call_claude_vision(
        prompt=IMAGE_PROMPT,
        image_path=image_path,
        patent_id=patent_id,
        compound_id=f"image_p{page_num}" if page_num else "image",
    )

    if response is None:
        return result

    # Parse response
    import json
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract SMILES directly if not valid JSON
        smiles = response.strip().split("\n")[0].strip()
        if validate_smiles(smiles):
            result["smiles"] = smiles
            result["canonical_smiles"] = canonicalize_smiles(smiles)
            result["valid"] = True
        return result

    smiles = data.get("smiles")
    result["compound_id"] = data.get("compound_id")

    if smiles and validate_smiles(smiles):
        result["smiles"] = smiles
        result["canonical_smiles"] = canonicalize_smiles(smiles)
        result["valid"] = True
    elif smiles:
        logger.info(
            f"Patent {patent_id} page {page_num}: "
            f"Image SMILES invalid: {smiles[:80]}"
        )

    return result


def convert_all_images(
    image_records: list[dict],
    patent_id: str,
) -> list[dict]:
    """Convert all extracted images to SMILES.

    Args:
        image_records: List from extract_images.extract_all_images().
        patent_id: Patent identifier.

    Returns:
        List of result dicts with smiles, compound_id, valid fields.
    """
    results = []
    valid_count = 0
    not_structure = 0

    for record in image_records:
        result = convert_image_to_smiles(
            image_path=record["image_path"],
            patent_id=patent_id,
            page_num=record.get("page_num"),
        )
        results.append(result)

        if result["valid"]:
            valid_count += 1
        elif result["smiles"] is None:
            not_structure += 1

    logger.info(
        f"Patent {patent_id}: Image→SMILES complete. "
        f"Valid: {valid_count}, Not structures: {not_structure}, "
        f"Invalid: {len(results) - valid_count - not_structure}"
    )

    return results
