"""Extract inline SMILES and InChI strings from patent text.

This module processes inline structures found during detection and converts
them into Compound objects with Tier 1 confidence (machine-readable, no
conversion needed — highest confidence source).
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import config
from .models import Compound, CompoundSource, ConfidenceTier, PageManifest
from .smiles_utils import (
    canonicalize_smiles,
    get_inchikey,
    strip_salt,
    compute_drug_likeness,
    validate_smiles,
)

logger = logging.getLogger(__name__)


def extract_inline_compounds(manifest: PageManifest) -> list[Compound]:
    """Convert inline SMILES/InChI from a manifest into Compound objects.

    All inline structures have already been RDKit-validated during detection.
    We canonicalize, compute InChIKey, strip salts, and assign Tier 1 confidence.

    Args:
        manifest: PageManifest with inline_structures populated.

    Returns:
        List of Compound objects.
    """
    compounds = []
    seen_inchikeys = set()

    for item in manifest.inline_structures:
        smiles = item.get("smiles")
        if not smiles or not validate_smiles(smiles):
            continue

        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            continue

        # Salt handling
        salt_result = strip_salt(canonical)
        parent_inchikey = salt_result.get("parent_inchikey")

        # Deduplicate within this patent
        if parent_inchikey and parent_inchikey in seen_inchikeys:
            continue
        if parent_inchikey:
            seen_inchikeys.add(parent_inchikey)

        inchikey = get_inchikey(canonical)
        drug_likeness = compute_drug_likeness(canonical)

        # Determine confidence tier
        # Inline SMILES/InChI are machine-readable — highest confidence
        if drug_likeness and drug_likeness.lipinski_passes:
            tier = ConfidenceTier.DUAL_CONFIRMED  # Tier 1: validated + drug-like
        else:
            tier = ConfidenceTier.SINGLE_VALIDATED  # Tier 2: valid but unusual properties

        compound = Compound(
            patent_id=manifest.patent_id,
            smiles_from_inline=item.get("raw"),
            canonical_smiles=canonical,
            parent_smiles=salt_result.get("parent_smiles"),
            inchikey=inchikey,
            parent_inchikey=parent_inchikey,
            source=CompoundSource.EXEMPLIFIED,
            confidence_tier=tier,
            source_page=item.get("page"),
            drug_likeness=drug_likeness,
            processing_status="validated",
        )
        compounds.append(compound)

    logger.info(
        f"Patent {manifest.patent_id}: "
        f"{len(compounds)} inline compounds extracted "
        f"(from {len(manifest.inline_structures)} candidates)"
    )
    return compounds
