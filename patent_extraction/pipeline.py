"""Main pipeline orchestrator — runs all extraction steps for a patent.

Each step is cached based on its version (config.STEP_VERSIONS) and inputs.
If a step's code hasn't changed and its inputs are the same, cached results
are reused. Bump a step's version in config.py to force re-execution.

Usage:
    python3 -m patent_extraction US10899738
    python3 -m patent_extraction --force US10899738  # ignore cache
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .detect_structures import detect_patent
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
from .adaptive_extractor import extract_compounds_adaptive
from .extract_assays import extract_assays_for_patent, attach_assays_to_compounds
from .context_folding import scan_patent_sections
from .markush_agents import extract_markush_multiagent
from .models import PatentResult, Compound
from .progress import ProgressTracker
from .cost_tracker import cost_tracker
from .api_cache import cache_stats
from . import step_cache

logger = logging.getLogger(__name__)


def _cached_or_run(patent_id, step_name, input_data, run_fn, force=False, label=None):
    """Check step cache; if hit, return cached result. Otherwise run and cache."""
    if not force:
        cached = step_cache.get_cached(patent_id, step_name, input_data)
        if cached is not None:
            logger.info(f"  {label or step_name} [CACHED]")
            return cached

    result = run_fn()
    step_cache.store(patent_id, step_name, input_data, result)
    return result


def run_patent(
    patent_id: str,
    data_dir: Path | None = None,
    skip_images: bool = False,
    force: bool = False,
) -> PatentResult:
    """Run the full extraction pipeline for a single patent.

    Args:
        patent_id: Patent identifier (e.g., "US10214537").
        data_dir: Root data directory.
        skip_images: If True, skip image extraction and Vision calls.
        force: If True, ignore step cache and re-run everything.

    Returns:
        PatentResult with all extracted compounds.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    progress = ProgressTracker(patent_id)
    progress.update(status="started")

    logger.info(f"{'='*60}")
    logger.info(f"Pipeline: {patent_id}" + (" [FORCE]" if force else ""))
    logger.info(f"{'='*60}")

    # ── Step 1: Detection (local, no API) ──────────────────────────
    logger.info(f"Step 1: Detection")
    manifest = detect_patent(patent_id, data_dir)

    sections = scan_patent_sections(patent_id, data_dir)
    logger.info(
        f"  Context folding: {sections.total_pages} pages → "
        f"{len(sections.pages_to_process)} to process "
        f"({sections.savings_pct:.0f}% token savings)"
    )

    format_info = detect_patent_format(patent_id, manifest, data_dir)
    logger.info(f"  Format={format_info['format']} "
                f"(text={format_info['text_score']}, image={format_info['image_score']})")
    progress.update(status="detection_complete")

    # ── Route 1: Google Patents (free, clean text) ─────────────────
    logger.info(f"Route 1: Google Patents")
    gp_input = {"patent_id": patent_id}

    def _run_gp():
        compounds = extract_patent_google(patent_id)
        return [c.model_dump() for c in compounds]

    def _run_gp_tables():
        compounds = extract_table_compounds_from_gp(patent_id)
        return [c.model_dump() for c in compounds]

    gp_data = _cached_or_run(patent_id, "google_patents", gp_input,
                              _run_gp, force, "Route 1a: Examples")
    gp_compounds = [Compound(**c) for c in gp_data]

    gp_table_data = _cached_or_run(patent_id, "google_tables", gp_input,
                                    _run_gp_tables, force, "Route 1b: Tables")
    gp_table_compounds = [Compound(**c) for c in gp_table_data]

    # Merge table compounds (deduplicate by example_number)
    gp_keys = {(c.example_number or '').lower().replace(' ', '') for c in gp_compounds}
    for tc in gp_table_compounds:
        tc_key = (tc.example_number or '').lower().replace(' ', '')
        if tc_key not in gp_keys:
            gp_compounds.append(tc)
            gp_keys.add(tc_key)

    # Route 1c: Adaptive (if < 20 compounds from 1a/1b)
    if len(gp_compounds) < 20:
        def _run_adaptive():
            compounds = extract_compounds_adaptive(patent_id)
            return [c.model_dump() for c in compounds]

        adaptive_data = _cached_or_run(
            patent_id, "adaptive",
            {"patent_id": patent_id, "gp_count": len(gp_compounds)},
            _run_adaptive, force, "Route 1c: Adaptive"
        )
        for c_data in adaptive_data:
            ac = Compound(**c_data)
            ac_key = (ac.example_number or '').lower().replace(' ', '')
            if ac_key not in gp_keys:
                gp_compounds.append(ac)
                gp_keys.add(ac_key)

    logger.info(f"  Google Patents total: {len(gp_compounds)} compounds")
    progress.update(status="google_patents_complete", google_patents_compounds=len(gp_compounds))

    # ── Step 2.0: Markush context (multi-agent) ────────────────────
    logger.info(f"Step 2.0: Markush context")
    gp_smiles = [c.canonical_smiles for c in gp_compounds if c.canonical_smiles]

    def _run_markush():
        ctx = extract_markush_multiagent(patent_id, manifest, data_dir, example_smiles=gp_smiles)
        return ctx.model_dump()

    markush_data = _cached_or_run(
        patent_id, "markush_context",
        {"patent_id": patent_id, "n_examples": len(gp_smiles)},
        _run_markush, force, "Markush context"
    )
    from .models import MarkushContext
    markush_context = MarkushContext(**markush_data) if isinstance(markush_data, dict) else markush_data
    progress.update(status="markush_context_complete")

    # ── Step 2.1: Page-level compound extraction ───────────────────
    logger.info(f"Step 2.1: Page-level extraction")

    def _run_page():
        compounds = extract_all_compounds(patent_id, manifest, data_dir)
        return [c.model_dump() for c in compounds]

    page_data = _cached_or_run(
        patent_id, "page_extraction",
        {"patent_id": patent_id},
        _run_page, force, "Page extraction"
    )
    compounds = [Compound(**c) for c in page_data]

    # ── Step 2.5: Table compound extraction ────────────────────────
    logger.info(f"Step 2.5: Table extraction")

    def _run_tables():
        compounds = extract_table_compounds(patent_id, data_dir)
        return [c.model_dump() for c in compounds]

    table_data = _cached_or_run(
        patent_id, "table_extraction",
        {"patent_id": patent_id},
        _run_tables, force, "Table extraction"
    )
    table_compounds = [Compound(**c) for c in table_data]

    # Merge: add table compounds, preferring longer IUPAC names
    compounds_by_id: dict[str, int] = {}
    for i, c in enumerate(compounds):
        key = (c.example_number or '').lower().replace(' ', '')
        if key in compounds_by_id:
            existing = compounds[compounds_by_id[key]]
            if len(c.iupac_name or '') > len(existing.iupac_name or ''):
                compounds[compounds_by_id[key]] = c
        else:
            compounds_by_id[key] = i

    new_from_tables = 0
    for tc in table_compounds:
        tc_key = (tc.example_number or '').lower().replace(' ', '')
        if tc_key in compounds_by_id:
            existing = compounds[compounds_by_id[tc_key]]
            if len(tc.iupac_name or '') > len(existing.iupac_name or ''):
                compounds[compounds_by_id[tc_key]] = tc
        else:
            compounds_by_id[tc_key] = len(compounds)
            compounds.append(tc)
            new_from_tables += 1

    logger.info(f"  +{new_from_tables} from tables (total: {len(compounds)})")

    # Merge Google Patents compounds (already validated)
    gp_already_validated = []
    new_from_gp = 0
    for gc in gp_compounds:
        gc_key = (gc.example_number or '').lower().replace(' ', '')
        if gc_key in compounds_by_id:
            existing = compounds[compounds_by_id[gc_key]]
            if not existing.canonical_smiles and gc.canonical_smiles:
                compounds[compounds_by_id[gc_key]] = gc
                gp_already_validated.append(gc)
        else:
            compounds_by_id[gc_key] = len(compounds)
            compounds.append(gc)
            gp_already_validated.append(gc)
            new_from_gp += 1

    logger.info(f"  +{new_from_gp} from Google Patents (total: {len(compounds)})")
    progress.update(status="compound_extraction_complete", compounds_found=len(compounds))

    # ── Step 2.2: IUPAC → SMILES ──────────────────────────────────
    gp_validated_ids = {id(c) for c in gp_already_validated}
    compounds_needing_conversion = [c for c in compounds if id(c) not in gp_validated_ids]
    already_done = len(compounds) - len(compounds_needing_conversion)
    logger.info(f"Step 2.2: IUPAC→SMILES ({len(compounds_needing_conversion)} to convert, "
                f"{already_done} already done)")

    if compounds_needing_conversion:
        compounds_needing_conversion = convert_batch(compounds_needing_conversion, markush_context)
        compounds = list(gp_already_validated) + compounds_needing_conversion

    smiles_count = sum(1 for c in compounds if c.canonical_smiles)
    progress.update(status="iupac_conversion_complete", smiles_from_text=smiles_count)

    # ── Step 2.3: Image pipeline ──────────────────────────────────
    image_results = []
    use_image = not skip_images or format_info['format'] in ('image', 'hybrid')

    if use_image and format_info['routing']['use_image_pipeline']:
        logger.info(f"Step 2.3: Image pipeline (Layout-to-Local Hybrid)")
        try:
            image_compounds = extract_image_compounds(patent_id, data_dir)
            existing_ids = {(c.example_number or '').lower().replace(' ', '') for c in compounds}
            new_from_images = 0
            for ic in image_compounds:
                ic_key = (ic.example_number or '').lower().replace(' ', '')
                if ic_key not in existing_ids:
                    compounds.append(ic)
                    existing_ids.add(ic_key)
                    new_from_images += 1
            logger.info(f"  +{new_from_images} from images")
        except Exception as e:
            logger.warning(f"  Image pipeline failed: {e}")
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
        logger.info(f"Step 2.3: Skipped")

    # ── Step 2.4: Cross-validation ────────────────────────────────
    logger.info(f"Step 2.4: Cross-validation")
    compounds = cross_validate_compounds(compounds, image_results)

    from .models import ConfidenceTier
    tier_counts = {1: 0, 2: 0, 3: 0}
    for c in compounds:
        tier_counts[c.confidence_tier.value] = tier_counts.get(c.confidence_tier.value, 0) + 1

    progress.update(
        status="cross_validation_complete",
        tier_1=tier_counts[1], tier_2=tier_counts[2], tier_3=tier_counts[3],
    )

    # ── Step 2.4b: Escalation ─────────────────────────────────────
    tier_3_ratio = tier_counts[3] / max(len(compounds), 1)
    if tier_3_ratio > 0.3 and len(compounds) > 10:
        failed = [c for c in compounds if c.processing_status == "failed"]
        if failed:
            logger.warning(
                f"Step 2.4b: High Tier 3 ({tier_3_ratio:.0%}). "
                f"Re-parsing {min(len(failed), 10)} failed compounds."
            )
            from .iupac_to_smiles import _convert_single
            for c in failed[:10]:
                if c.iupac_name and c.processing_status == "failed":
                    c.processing_status = "pending"
                    _convert_single(c)
                    if c.processing_status == "validated":
                        logger.info(f"  Recovered: {c.example_number}")

    # ── Step 2.6: Assay extraction ────────────────────────────────
    logger.info(f"Step 2.6: Assay extraction")
    assay_data = extract_assays_for_patent(patent_id, data_dir)
    assays_attached = attach_assays_to_compounds(compounds, assay_data)
    logger.info(f"  Attached to {assays_attached}/{len(compounds)} compounds")
    progress.update(status="assay_extraction_complete", compounds_with_assays=assays_attached)

    # ── Build result and save ─────────────────────────────────────
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
            "step_versions": config.STEP_VERSIONS,
        },
    )

    # Save to canonical output location
    output_dir = config.RESULTS_DIR / patent_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined.json"

    with open(output_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)

    progress.update(status="complete", cost_usd=cost_tracker.total_cost)

    logger.info(f"\nPipeline complete: {patent_id}")
    logger.info(f"  Compounds: {len(compounds)} (SMILES: {smiles_count})")
    logger.info(f"  Tiers: T1={tier_counts[1]} T2={tier_counts[2]} T3={tier_counts[3]}")
    logger.info(f"  Cost: ${cost_tracker.total_cost:.2f}")
    logger.info(f"  Output: {output_path}")

    return result


def run_patents_parallel(
    patent_ids: list[str] | None = None,
    data_dir: Path | None = None,
    max_workers: int = 3,
    skip_images: bool = False,
    force: bool = False,
) -> list[PatentResult]:
    """Run pipeline on multiple patents in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if patent_ids is None:
        patent_ids = config.PATENT_IDS

    logger.info(f"Running {len(patent_ids)} patents ({max_workers} workers)")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_patent, pid, data_dir, skip_images, force): pid
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

    total = sum(r.stats.get('total_compounds', 0) for r in results)
    logger.info(f"All complete. Total: {total} compounds")
    return results
