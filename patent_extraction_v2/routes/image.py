"""Image extraction pipeline — Layout-to-Local Hybrid.

For patents where compounds are defined by structure drawings (Cpd.No. format):
1. Sonnet reads table page → extracts compound layout (Cpd.No. + bounding boxes)
2. Crop individual structure images from the PDF page
3. DECIMER converts cropped images → SMILES (free, local)
4. RDKit validates + drug-likeness check
5. Opus Vision as fallback for DECIMER failures

Uses each tool's strength:
- LLMs: reading messy table layouts
- DECIMER: converting clean structure images to SMILES
- No LLM-generated SMILES = no hallucination risk (except Opus fallback)
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from PIL import Image

from ..core import config
from ..core.api_client import call_claude_text, call_claude_vision
from ..core.cost_tracker import cost_tracker
from .extract_images import extract_image_from_pdf
from ..core.models import Compound, CompoundSource, IupacSource, ConfidenceTier
from ..core.smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey,
    strip_salt, compute_drug_likeness, molecular_weight,
)

logger = logging.getLogger(__name__)

MIN_SMILES_LENGTH = 10
MIN_SMILES_MW = 150
MAX_SMILES_MW = 1500

# OCR-bbox markers in patent markdown (consistent with extract_images.py format)
# Examples seen in the wild:
#   <|ref|>image<|/ref|><|det|>[[173, 90, 500, 480]]<|/det|>
#   <|ref|>image<|/ref|><|det|>[[10,20,30,40]]<|/det|>          (no spaces)
# Be lenient about whitespace and inner brackets to catch real variants.
_OCR_BBOX_RE = re.compile(
    r'<\|ref\|>image<\|/ref\|>\s*<\|det\|>\s*\[\[\s*([\d\s,]+?)\s*\]\]\s*<\|/det\|>'
)
# Coordinate range observed in markdown (matches extract_images._bbox_to_pdf_rect)
_OCR_BBOX_COORD_MAX = 1000.0


def _extract_ocr_bboxes(patent_id: str, page_num: int, data_dir: Path) -> list[list[int]]:
    """Parse OCR markdown for `<|ref|>image<|/ref|><|det|>[bbox]<|/det|>` markers.

    Returns list of [x1, y1, x2, y2] in 0-1000 normalized coords.
    Falls back to iupacs_clean if all_pages doesn't have the page.
    """
    bboxes: list[list[int]] = []
    for subdir in ("all_pages", "iupacs_clean"):
        page_file = data_dir / patent_id / subdir / f"page_{page_num:04d}.md"
        if not page_file.exists():
            continue
        try:
            text = page_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in _OCR_BBOX_RE.finditer(text):
            try:
                coords = [int(x.strip()) for x in m.group(1).split(",")]
                if len(coords) == 4:
                    bboxes.append(coords)
            except ValueError:
                continue
        if bboxes:
            break  # prefer the first directory that yielded results
    return bboxes


# JSONL logger for per-crop diagnostics (MW + crop-size calibration)
_IMAGE_LOG_PATH = config.LOGS_DIR / "image_pipeline.jsonl"


def _log_crop(record: dict):
    """Append a per-crop diagnostic record to image_pipeline.jsonl."""
    try:
        _IMAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_IMAGE_LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.debug(f"Image log write failed: {e}")

LAYOUT_PROMPT = """This is a page from a pharmaceutical patent showing a compound table.
For each compound visible on this page, extract:
1. The compound number (Cpd. No., Example number, or any identifier)
2. The approximate bounding box of its chemical structure drawing as [x1, y1, x2, y2]
   where coordinates are percentages of the page (0-100)
3. Any M+H, MS, or molecular weight value if shown
4. The chemical name if written in text

Return as JSON array:
[{"cpd_no": "148", "structure_bbox": [10, 20, 40, 50], "ms_mh": 450.3, "name": "IUPAC name or null"}]

Return ONLY the JSON array."""


def _extract_table_layout(page_image_path: str, patent_id: str, page_num: int) -> list[dict]:
    """Use Sonnet Vision to read table layout and identify compound locations."""
    response = call_claude_vision(
        prompt=LAYOUT_PROMPT,
        image_path=page_image_path,
        model=config.MODEL_SONNET,
        patent_id=patent_id,
        compound_id=f"layout_p{page_num}",
    )

    if not response:
        return []

    # Parse JSON
    cleaned = response.strip()
    if '```' in cleaned:
        parts = cleaned.split('```')
        for p in parts:
            p = p.strip()
            if p.startswith('json'):
                p = p[4:].strip()
            if p.startswith('['):
                cleaned = p
                break

    if not cleaned.startswith('['):
        start = cleaned.find('[')
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '[': depth += 1
                elif cleaned[i] == ']': depth -= 1
                if depth == 0:
                    cleaned = cleaned[start:i+1]
                    break

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse layout JSON for {patent_id} p{page_num}")
        return []


def _crop_structure(page_image: Image.Image, bbox_pct: list) -> Image.Image | None:
    """Crop a structure image from a page using percentage bounding box."""
    if not bbox_pct or len(bbox_pct) != 4:
        return None

    pw, ph = page_image.size
    x1 = int(bbox_pct[0] / 100 * pw)
    y1 = int(bbox_pct[1] / 100 * ph)
    x2 = int(bbox_pct[2] / 100 * pw)
    y2 = int(bbox_pct[3] / 100 * ph)

    # Validate dimensions
    if x2 - x1 < 50 or y2 - y1 < 50:
        return None
    if x1 < 0 or y1 < 0 or x2 > pw or y2 > ph:
        return None

    return page_image.crop((x1, y1, x2, y2))


def _decimer_to_smiles(image_path: str) -> str | None:
    """Convert structure image to SMILES using DECIMER."""
    try:
        from DECIMER import predict_SMILES
        raw = predict_SMILES(image_path)
        if not raw:
            return None

        # Take first fragment (DECIMER sometimes produces multi-fragment output)
        smiles = raw.split('.')[0].strip()

        if validate_smiles(smiles) and len(smiles) >= MIN_SMILES_LENGTH:
            mw = molecular_weight(smiles)
            if mw and MIN_SMILES_MW <= mw <= MAX_SMILES_MW:
                return smiles
    except Exception as e:
        logger.warning(f"DECIMER failed on {image_path}: {e}")

    return None


def _opus_vision_fallback(image_path: str, patent_id: str, cpd_no: str) -> str | None:
    """Fallback: use Opus Vision for structures DECIMER can't handle."""
    response = call_claude_vision(
        prompt="Convert this chemical structure to a canonical SMILES string. "
               "Preserve stereochemistry. Return ONLY the SMILES string.",
        image_path=image_path,
        model=config.MODEL_OPUS,
        patent_id=patent_id,
        compound_id=f"{cpd_no}_opus_vision",
    )

    if response:
        smiles = response.strip().split('\n')[0].strip()
        if validate_smiles(smiles) and len(smiles) >= MIN_SMILES_LENGTH:
            mw = molecular_weight(smiles)
            if mw and MIN_SMILES_MW <= mw <= MAX_SMILES_MW:
                return smiles
    return None


def _sonnet_vision_fallback(image_path: str, patent_id: str, cpd_no: str) -> str | None:
    """Budget fallback: Sonnet Vision for structures DECIMER can't handle (5x cheaper than Opus)."""
    from ..core.cost_tracker import cost_tracker
    # Budget guard: stay under 80% of ceiling
    if cost_tracker.total_cost > config.COST_CEILING * 0.8:
        return None

    response = call_claude_vision(
        prompt="Convert this chemical structure to a canonical SMILES string. "
               "Preserve stereochemistry. Return ONLY the SMILES string.",
        image_path=image_path,
        model=config.MODEL_SONNET,
        patent_id=patent_id,
        compound_id=f"{cpd_no}_sonnet_vision",
    )

    if response:
        smiles = response.strip().split('\n')[0].strip()
        if validate_smiles(smiles) and len(smiles) >= MIN_SMILES_LENGTH:
            mw = molecular_weight(smiles)
            if mw and MIN_SMILES_MW <= mw <= MAX_SMILES_MW:
                return smiles
    return None


def extract_image_compounds(
    patent_id: str,
    data_dir: Path | None = None,
    use_opus_fallback: bool = False,
    target_pages: list[int] | None = None,
    prefer_ocr_bboxes: bool = True,
    per_page_crop_cap: int = config.PER_PAGE_CROP_CAP_DEFAULT,
    formula_pages: list[int] | None = None,
    image_dense_pages: list[int] | None = None,
) -> list[Compound]:
    """Extract compounds from a patent using image-based pipeline.

    Cascading selection:
    - Auto-discovers "table" pages by keyword (existing behavior)
    - UNION with caller-provided `target_pages` (e.g. missed-example pages)
    - UNION with `formula_pages` and `image_dense_pages` from the manifest
      (patents where structures hide in figures rather than tables)

    Cascading crop strategy:
    - If `prefer_ocr_bboxes`, use OCR-markdown bboxes first (free). Only fall
      back to Sonnet-Vision layout detection if OCR yields nothing.
    - Per-page crop cap limits runaway DECIMER calls on dense schemes.
    - Per-patent image budget (cost_tracker.patent_image_exceeded) cuts off
      Vision fallbacks once config.PER_PATENT_IMAGE_CAP is hit.

    Per-crop SMILES cascade (unchanged):
    DECIMER → Sonnet-Vision fallback → (Opus-Vision if use_opus_fallback AND budget)

    Args:
        patent_id: Patent identifier.
        data_dir: Root data directory.
        use_opus_fallback: If True, use Opus Vision when DECIMER+Sonnet fail.
        target_pages: Explicit page numbers to process (e.g. missed-example pages).
            Unioned with auto-detected tables and formula/dense pages.
        prefer_ocr_bboxes: If True, try cheap OCR-markdown bboxes before Sonnet layout.
        per_page_crop_cap: Max crops per page to avoid runaway calls.
        formula_pages: Pages with "Formula (I)", "Scheme N" markers (from manifest).
        image_dense_pages: Heavy-image, low-text pages (from manifest).

    Returns:
        List of Compound objects from image extraction.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    pdf_path = data_dir / patent_id / f"{patent_id}.pdf"
    if not pdf_path.exists():
        logger.error(f"Patent {patent_id}: PDF not found")
        return []

    # Find table pages — check both iupacs_clean and all_pages
    table_pages = set()

    for subdir_name in ["iupacs_clean", "all_pages"]:
        pages_dir = data_dir / patent_id / subdir_name
        if not pages_dir.exists():
            continue
        for page_file in sorted(pages_dir.glob("page_*.md")):
            text = page_file.read_text(encoding="utf-8")
            text_lower = text.lower()
            # Pages with compound tables (multiple detection patterns)
            is_table_page = False
            if '<table>' in text_lower:
                if any(kw in text_lower for kw in [
                    'cpd', 'structure', 'chemical', 'compound',
                    'no.', 'name', 'table '
                ]):
                    is_table_page = True
            # Also catch pages with TABLE N headers and structure images
            if re.search(r'TABLE\s+\d+', text, re.IGNORECASE):
                if 'structure' in text_lower or 'compound' in text_lower or 'cpd' in text_lower:
                    is_table_page = True

            if is_table_page:
                match = re.search(r'page_(\d+)', page_file.stem)
                if match:
                    table_pages.add(int(match.group(1)))

    # Union with explicit target pages and manifest-derived formula/dense pages
    # so the image route also runs on macrocycle figures and missed-example pages.
    candidate_pages: set[int] = set(table_pages)
    if target_pages:
        candidate_pages.update(target_pages)
    if formula_pages:
        candidate_pages.update(formula_pages)
    if image_dense_pages:
        candidate_pages.update(image_dense_pages)

    if not candidate_pages:
        logger.info(f"Patent {patent_id}: No candidate pages for image extraction")
        return []

    logger.info(
        f"Patent {patent_id}: Image route — {len(candidate_pages)} candidate pages "
        f"({len(table_pages)} table + {len(target_pages or [])} targeted + "
        f"{len(formula_pages or [])} formula + {len(image_dense_pages or [])} image-dense)"
    )
    table_pages = candidate_pages  # downstream code expects `table_pages`

    doc = fitz.open(str(pdf_path))
    images_dir = config.IMAGES_DIR / patent_id
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Render pages + collect crop jobs ──────────────────
    # OCR-first cascade: try cheap OCR-markdown bboxes per page before
    # falling back to Sonnet-Vision layout calls.
    crop_jobs = []  # (cpd_no, crop_path, ms_mh, iupac_name, page_num)
    seen_ids: set[str] = set()
    pages_needing_sonnet: list[tuple[int, Image.Image, str]] = []  # (page_num, page_img, page_img_path)

    for page_num in sorted(table_pages):
        if page_num >= len(doc):
            continue
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_img_path = str(images_dir / f"page_{page_num:04d}.png")
        page_img.save(page_img_path)

        ocr_bboxes: list[list[int]] = []
        if prefer_ocr_bboxes:
            ocr_bboxes = _extract_ocr_bboxes(patent_id, page_num, data_dir)

        if ocr_bboxes:
            # Cheap path: crop directly from OCR bboxes (0-1000 normalized coords)
            # Cap to avoid runaway on dense schemes
            for idx, bbox_1000 in enumerate(ocr_bboxes[:per_page_crop_cap]):
                # Convert 0-1000 normalized → percentage for _crop_structure
                bbox_pct = [c / _OCR_BBOX_COORD_MAX * 100 for c in bbox_1000]
                crop = _crop_structure(page_img, bbox_pct)
                if crop is None:
                    continue
                # ID format: image-only crop has no Cpd.No yet; use page+idx anchor
                if page_num in (formula_pages or []):
                    cpd_no = f"Formula_p{page_num}_{idx}"
                else:
                    cpd_no = f"Image_p{page_num}_{idx}"
                if cpd_no in seen_ids:
                    continue
                seen_ids.add(cpd_no)
                crop_path = str(images_dir / f"ocr_p{page_num:04d}_{idx}.png")
                crop.save(crop_path)
                crop_jobs.append((cpd_no, crop_path, None, None, page_num))
        else:
            # No OCR bboxes for this page → defer to Sonnet layout
            pages_needing_sonnet.append((page_num, page_img, page_img_path))

    doc.close()

    logger.info(
        f"Patent {patent_id}: OCR-first phase yielded {len(crop_jobs)} crops; "
        f"{len(pages_needing_sonnet)} pages need Sonnet layout"
    )

    # ── Phase 2: Sonnet layout fallback (only for pages without OCR bboxes) ──
    if pages_needing_sonnet:
        def _get_layout(job):
            pn, pi, pip = job
            # Budget guard: skip if patent's image cap is exhausted
            if cost_tracker.patent_image_exceeded(patent_id):
                return pn, pi, []
            layouts = _extract_table_layout(pip, patent_id, pn)
            # Approximate Sonnet layout cost (~$0.005/page) tracked under image budget.
            # Real cost is logged under total_cost via api_client.
            cost_tracker.record_image(0.005, patent_id)
            return pn, pi, layouts

        layout_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for result in executor.map(_get_layout, pages_needing_sonnet):
                layout_results.append(result)

        for page_num, page_img, layouts in layout_results:
            if not layouts:
                continue
            crops_this_page = 0
            for layout in layouts:
                if crops_this_page >= per_page_crop_cap:
                    break
                cpd_no = str(layout.get('cpd_no', ''))
                if not cpd_no:
                    continue
                dedup_key = f"example{cpd_no}".lower().replace(' ', '')
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                bbox = layout.get('structure_bbox') or []
                crop = _crop_structure(page_img, bbox)
                if crop is None:
                    continue

                crop_path = str(images_dir / f"cpd_{cpd_no}_p{page_num}.png")
                crop.save(crop_path)
                # Sanitize ms_mh — Sonnet sometimes returns strings like "343 and 345"
                ms_mh_raw = layout.get('ms_mh')
                ms_mh_value = None
                if isinstance(ms_mh_raw, (int, float)):
                    ms_mh_value = float(ms_mh_raw)
                elif isinstance(ms_mh_raw, str):
                    # Take first numeric token if string like "343 and 345"
                    m = re.search(r'\d+(?:\.\d+)?', ms_mh_raw)
                    if m:
                        try:
                            ms_mh_value = float(m.group(0))
                        except ValueError:
                            ms_mh_value = None
                crop_jobs.append((cpd_no, crop_path, ms_mh_value, layout.get('name'), page_num))
                crops_this_page += 1

    logger.info(f"Patent {patent_id}: {len(crop_jobs)} structure crops to process with DECIMER")

    def _process_crop(job):
        cpd_no, crop_path, ms_mh, iupac_name, page_num = job

        # Per-crop diagnostic record for MW gate / crop size calibration
        crop_w = crop_h = 0
        try:
            with Image.open(crop_path) as _img:
                crop_w, crop_h = _img.size
        except Exception:
            pass

        smiles = _decimer_to_smiles(crop_path)
        extraction_method = "decimer_local"

        # Vision fallbacks gated by per-patent image budget
        if smiles is None and not cost_tracker.patent_image_exceeded(patent_id):
            # Try Sonnet Vision first (5x cheaper than Opus). ~$0.005/call.
            smiles = _sonnet_vision_fallback(crop_path, patent_id, cpd_no)
            cost_tracker.record_image(0.005, patent_id)
            extraction_method = "sonnet_vision_fallback"

        if smiles is None and use_opus_fallback and not cost_tracker.patent_image_exceeded(patent_id):
            # Last resort: Opus Vision. ~$0.025/call.
            smiles = _opus_vision_fallback(crop_path, patent_id, cpd_no)
            cost_tracker.record_image(0.025, patent_id)
            extraction_method = "opus_vision_fallback"

        # Log diagnostic before any return (includes failures)
        mw_val = molecular_weight(smiles) if smiles else None
        _log_crop({
            "patent_id": patent_id,
            "page": page_num,
            "cpd_no": cpd_no,
            "crop_path": crop_path,
            "crop_w_px": crop_w,
            "crop_h_px": crop_h,
            "decimer_smiles_preview": (smiles or "")[:80],
            "mw": mw_val,
            "extraction_method": extraction_method if smiles else "FAILED",
        })

        if smiles is None:
            return None

        canonical = canonicalize_smiles(smiles)
        if not canonical:
            return None

        inchikey = get_inchikey(canonical)
        salt_result = strip_salt(canonical)
        drug_likeness = compute_drug_likeness(canonical)

        # Preserve OCR-anchored IDs (Image_pX_Y / Formula_pX_Y) so downstream
        # joiners can map back to text routes via page+order. Numeric Cpd.No
        # gets the conventional "Cpd. No. N" format for table-pattern dedup.
        if cpd_no.startswith(("Image_p", "Formula_p")):
            example_number = cpd_no
        else:
            example_number = f"Cpd. No. {cpd_no}"

        # Structured provenance — image route doesn't go through _finalize
        from ..core.models import CompoundProvenance
        prov = CompoundProvenance(
            route="image_pipeline",
            stage=extraction_method,    # decimer_local / sonnet_vision_fallback / opus_vision_fallback
            page=page_num,
            image_bbox=None,            # bbox is in 0-1000 normalized; we don't carry through here
            image_path=crop_path,
            chain=[extraction_method],
        )

        return Compound(
            patent_id=patent_id,
            example_number=example_number,
            iupac_name=iupac_name,
            iupac_source=IupacSource.GENERATED if not iupac_name else IupacSource.PATENT_VERBATIM,
            smiles_from_image=smiles,
            canonical_smiles=canonical,
            inchikey=inchikey,
            parent_smiles=salt_result.get("parent_smiles"),
            parent_inchikey=salt_result.get("parent_inchikey"),
            ms_mh_plus=ms_mh,
            source=CompoundSource.EXEMPLIFIED,
            confidence_tier=ConfidenceTier.SINGLE_VALIDATED,
            extraction_method=extraction_method,
            provenance=prov,
            image_path=crop_path,
            source_page=page_num,
            drug_likeness=drug_likeness,
            processing_status="validated",
        )

    # Run DECIMER in parallel (4 workers — CPU-bound, don't overload)
    compounds: list[Compound] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(_process_crop, crop_jobs):
            if result is not None:
                compounds.append(result)

    logger.info(
        f"Patent {patent_id}: Image pipeline extracted {len(compounds)} compounds "
        f"from {len(table_pages)} table pages"
    )
    return compounds
