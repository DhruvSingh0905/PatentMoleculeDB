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
from .layout_router import detect_patent_format
from .image_pipeline import extract_image_compounds
from .google_patents import extract_patent_google
from .google_patents_tables import extract_table_compounds_from_gp
from .extract_assays import extract_assays_for_patent, attach_assays_to_compounds
from .models import PatentResult
from .progress import ProgressTracker
from .cost_tracker import cost_tracker
from .api_cache import cache_stats

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

    # Step 1b: Layout-aware routing
    format_info = detect_patent_format(patent_id, manifest, data_dir)
    logger.info(f"Step 1b: Format={format_info['format']} (text={format_info['text_score']}, image={format_info['image_score']})")
    progress.update(status="detection_complete")

    # Route 1: Google Patents — clean text, $0 cost, no OCR errors
    logger.info(f"Route 1: Google Patents clean text extraction")
    gp_compounds = []
    try:
        # Route 1a: Example-format extraction (claims lists, Example N:)
        gp_compounds = extract_patent_google(patent_id)
        # Route 1b: Table-format extraction (Cpd.No. tables with Chemical Name column)
        gp_table_compounds = extract_table_compounds_from_gp(patent_id)
        # Merge table compounds (avoid duplicates)
        gp_keys = {(c.example_number or '').lower().replace(' ', '') for c in gp_compounds}
        for tc in gp_table_compounds:
            tc_key = (tc.example_number or '').lower().replace(' ', '')
            if tc_key not in gp_keys:
                gp_compounds.append(tc)
                gp_keys.add(tc_key)
        logger.info(f"  Google Patents: {len(gp_compounds)} compounds extracted ($0)")
    except Exception as e:
        logger.warning(f"  Google Patents failed: {e}")

    progress.update(status="google_patents_complete", google_patents_compounds=len(gp_compounds))

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

    # Merge Google Patents compounds (already validated, skip IUPAC→SMILES for these)
    gp_already_validated = []
    new_from_gp = 0
    for gc in gp_compounds:
        gc_key = (gc.example_number or '').lower().replace(' ', '')
        if gc_key in compounds_by_id:
            existing = compounds[compounds_by_id[gc_key]]
            # Google Patents has cleaner text — prefer if existing has no SMILES yet
            if not existing.canonical_smiles and gc.canonical_smiles:
                compounds[compounds_by_id[gc_key]] = gc
                gp_already_validated.append(gc)
        else:
            compounds_by_id[gc_key] = len(compounds)
            compounds.append(gc)
            gp_already_validated.append(gc)
            new_from_gp += 1

    logger.info(f"  Google Patents added {new_from_gp} new compounds (total: {len(compounds)})")

    progress.update(
        status="compound_extraction_complete",
        compounds_found=len(compounds),
    )

    # Step 2.2: IUPAC → SMILES with Markush context
    # Skip compounds already validated by Google Patents route
    gp_validated_ids = {id(c) for c in gp_already_validated}
    compounds_needing_conversion = [c for c in compounds if id(c) not in gp_validated_ids]
    already_done = len(compounds) - len(compounds_needing_conversion)
    logger.info(f"Step 2.2: IUPAC → SMILES conversion ({len(compounds_needing_conversion)} compounds, {already_done} already done via Google Patents)")
    if compounds_needing_conversion:
        compounds_needing_conversion = convert_batch(compounds_needing_conversion, markush_context)
        # Rebuild full list: GP compounds + converted compounds
        compounds = list(gp_already_validated) + compounds_needing_conversion
    smiles_count = sum(1 for c in compounds if c.canonical_smiles)
    progress.update(
        status="iupac_conversion_complete",
        smiles_from_text=smiles_count,
    )

    # Step 2.3: Image → SMILES (optional, or auto if image-heavy patent)
    image_results = []
    use_image = not skip_images or format_info['format'] in ('image', 'hybrid')
    if use_image and format_info['routing']['use_image_pipeline']:
        logger.info(f"Step 2.3: Image pipeline (Layout-to-Local Hybrid)")
        try:
            image_compounds = extract_image_compounds(patent_id, data_dir)
            # Merge image compounds
            existing_ids = {(c.example_number or '').lower().replace(' ', '') for c in compounds}
            new_from_images = 0
            for ic in image_compounds:
                ic_key = (ic.example_number or '').lower().replace(' ', '')
                if ic_key not in existing_ids:
                    compounds.append(ic)
                    existing_ids.add(ic_key)
                    new_from_images += 1
            logger.info(f"  Image pipeline added {new_from_images} new compounds")
        except Exception as e:
            logger.warning(f"Image pipeline failed: {e}")
        progress.update(
            status="image_extraction_complete",
            smiles_from_image=sum(1 for c in compounds if c.smiles_from_image),
        )
    elif not skip_images:
        logger.info(f"Step 2.3: Standard image extraction")
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

    # Step 2.6: Assay data extraction (local, no API)
    logger.info(f"Step 2.6: Assay data extraction")
    assay_data = extract_assays_for_patent(patent_id, data_dir)
    assays_attached = attach_assays_to_compounds(compounds, assay_data)
    logger.info(f"  Attached assay data to {assays_attached}/{len(compounds)} compounds")
    progress.update(status="assay_extraction_complete", compounds_with_assays=assays_attached)

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
            "google_patents_compounds": len(gp_compounds),
            "compounds_with_assays": assays_attached,
            "total_assay_measurements": sum(len(c.assay_data) for c in compounds),
            "markush_quality": markush_context.quality,
            "patent_format": format_info['format'],
            "text_score": format_info['text_score'],
            "image_score": format_info['image_score'],
            "cost": cost_tracker.summary(),
            "cache": cache_stats(),
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
    logger.info(f"  Format: {format_info['format']}")
    logger.info(f"  Compounds: {len(compounds)}")
    logger.info(f"  With SMILES: {smiles_count}")
    logger.info(f"  Tiers: T1={tier_counts[1]}, T2={tier_counts[2]}, T3={tier_counts[3]}")
    logger.info(f"  Cost so far: ${cost_tracker.total_cost:.2f}")
    logger.info(f"  Cache: {cache_stats()}")
    logger.info(f"  Output: {output_path}")

    return result


def run_patents_parallel(
    patent_ids: list[str] | None = None,
    data_dir: Path | None = None,
    output_version: str = "v1",
    max_workers: int = 3,
    skip_images: bool = False,
) -> list[PatentResult]:
    """Run pipeline on multiple patents in parallel.

    Args:
        patent_ids: List of patent IDs. Defaults to all configured patents.
        data_dir: Root data directory.
        output_version: Version tag for output.
        max_workers: Number of parallel patent processors.
        skip_images: If True, skip image extraction.

    Returns:
        List of PatentResult objects.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if patent_ids is None:
        patent_ids = config.PATENT_IDS

    logger.info(f"Running {len(patent_ids)} patents with {max_workers} parallel workers")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_patent, pid, data_dir, skip_images, output_version
            ): pid
            for pid in patent_ids
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"  {pid}: {result.stats.get('total_compounds', 0)} compounds")
            except Exception as e:
                logger.error(f"  {pid}: FAILED — {e}")

    logger.info(f"All patents complete. Total: {sum(r.stats.get('total_compounds', 0) for r in results)} compounds")
    return results
