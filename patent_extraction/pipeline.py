"""Main pipeline orchestrator — runs all extraction steps for a patent.

Usage:
    from patent_extraction.pipeline import run_patent
    result = run_patent("US10214537")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .detect_structures import detect_patent
from .extract_markush_context import extract_markush_context
from .extract_compounds import extract_all_compounds
from .parse_tables import extract_table_compounds
from .iupac_to_smiles import convert_batch
from .extract_images import extract_all_images
from .image_to_smiles import convert_all_images
from .cross_validate import cross_validate_compounds
from .models import PatentResult
from .progress import ProgressTracker
from .cost_tracker import cost_tracker

logger = logging.getLogger(__name__)


def run_patent(
    patent_id: str,
    data_dir: Path | None = None,
    skip_images: bool = False,
    output_version: str = "v1",
) -> PatentResult:
    """Run the full extraction pipeline for a single patent.

    Args:
        patent_id: Patent identifier (e.g., "US10214537").
        data_dir: Root data directory.
        skip_images: If True, skip image extraction and Vision calls (saves cost).
        output_version: Version tag for output directory (e.g., "v1", "v2").

    Returns:
        PatentResult with all extracted compounds.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    progress = ProgressTracker(patent_id)
    progress.update(status="started")

    logger.info(f"{'='*60}")
    logger.info(f"Starting pipeline for patent {patent_id}")
    logger.info(f"{'='*60}")

    # Step 1: Detection (local, no API)
    logger.info(f"Step 1: Detection")
    manifest = detect_patent(patent_id, data_dir)
    progress.update(status="detection_complete")

    # Step 2.0: Markush context extraction
    logger.info(f"Step 2.0: Markush context extraction")
    markush_context = extract_markush_context(patent_id, manifest, data_dir)
    progress.update(status="markush_context_complete")

    # Step 2.1: Page-level compound extraction (from iupacs_clean example pages)
    logger.info(f"Step 2.1: Page-level compound extraction")
    compounds = extract_all_compounds(patent_id, manifest, data_dir)

    # Step 2.5: Table compound extraction (from all_pages compound tables)
    logger.info(f"Step 2.5: Table compound extraction")
    table_compounds = extract_table_compounds(patent_id, data_dir)

    # Merge: add table compounds, preferring longer IUPAC names (final product > intermediate)
    compounds_by_id: dict[str, int] = {}
    for i, c in enumerate(compounds):
        key = (c.example_number or '').lower().replace(' ', '')
        if key in compounds_by_id:
            existing = compounds[compounds_by_id[key]]
            # Keep whichever has the longer IUPAC name (final products are longer)
            if len(c.iupac_name or '') > len(existing.iupac_name or ''):
                compounds[compounds_by_id[key]] = c  # Replace with longer name
        else:
            compounds_by_id[key] = i

    new_from_tables = 0
    for tc in table_compounds:
        tc_key = (tc.example_number or '').lower().replace(' ', '')
        if tc_key in compounds_by_id:
            existing = compounds[compounds_by_id[tc_key]]
            if len(tc.iupac_name or '') > len(existing.iupac_name or ''):
                compounds[compounds_by_id[tc_key]] = tc  # Table has better name
        else:
            compounds_by_id[tc_key] = len(compounds)
            compounds.append(tc)
            new_from_tables += 1

    logger.info(f"  Merged: {new_from_tables} new compounds from tables (total: {len(compounds)})")
    progress.update(
        status="compound_extraction_complete",
        compounds_found=len(compounds),
    )

    # Step 2.2: IUPAC → SMILES with Markush context
    logger.info(f"Step 2.2: IUPAC → SMILES conversion ({len(compounds)} compounds)")
    compounds = convert_batch(compounds, markush_context)
    smiles_count = sum(1 for c in compounds if c.canonical_smiles)
    progress.update(
        status="iupac_conversion_complete",
        smiles_from_text=smiles_count,
    )

    # Step 2.3: Image → SMILES (optional)
    image_results = []
    if not skip_images:
        logger.info(f"Step 2.3: Image extraction and Vision conversion")
        image_records = extract_all_images(patent_id, manifest, data_dir)
        if image_records:
            image_results = convert_all_images(image_records, patent_id)
        progress.update(
            status="image_conversion_complete",
            smiles_from_image=sum(1 for r in image_results if r.get("valid")),
        )
    else:
        logger.info(f"Step 2.3: Skipped (skip_images=True)")

    # Step 2.4: Cross-validation
    logger.info(f"Step 2.4: Cross-validation")
    compounds = cross_validate_compounds(compounds, image_results)

    # Count tiers
    from .models import ConfidenceTier
    tier_counts = {1: 0, 2: 0, 3: 0}
    for c in compounds:
        tier_counts[c.confidence_tier.value] = tier_counts.get(c.confidence_tier.value, 0) + 1

    progress.update(
        status="cross_validation_complete",
        tier_1=tier_counts[1],
        tier_2=tier_counts[2],
        tier_3=tier_counts[3],
    )

    # Build result
    result = PatentResult(
        patent_id=patent_id,
        exemplified_compounds=compounds,
        manifest=manifest,
        stats={
            "total_compounds": len(compounds),
            "with_smiles": smiles_count,
            "with_iupac": sum(1 for c in compounds if c.iupac_name),
            "images_processed": len(image_results),
            "tier_1": tier_counts[1],
            "tier_2": tier_counts[2],
            "tier_3": tier_counts[3],
            "markush_quality": markush_context.quality,
            "cost": cost_tracker.summary(),
        },
    )

    # Save output
    output_dir = config.OUTPUT_DIR / output_version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{patent_id}.json"

    with open(output_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)

    progress.update(status="complete", cost_usd=cost_tracker.total_cost)

    logger.info(f"Pipeline complete for {patent_id}")
    logger.info(f"  Compounds: {len(compounds)}")
    logger.info(f"  With SMILES: {smiles_count}")
    logger.info(f"  Tiers: T1={tier_counts[1]}, T2={tier_counts[2]}, T3={tier_counts[3]}")
    logger.info(f"  Cost so far: ${cost_tracker.total_cost:.2f}")
    logger.info(f"  Output: {output_path}")

    return result
