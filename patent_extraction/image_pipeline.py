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

from . import config
from .api_client import call_claude_text, call_claude_vision
from .models import Compound, CompoundSource, IupacSource, ConfidenceTier
from .smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey,
    strip_salt, compute_drug_likeness, molecular_weight,
)

logger = logging.getLogger(__name__)

MIN_SMILES_LENGTH = 10
MIN_SMILES_MW = 150

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
    if len(bbox_pct) != 4:
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
            if mw and mw >= MIN_SMILES_MW:
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
            if mw and mw >= MIN_SMILES_MW:
                return smiles
    return None


def extract_image_compounds(
    patent_id: str,
    data_dir: Path | None = None,
    use_opus_fallback: bool = False,
) -> list[Compound]:
    """Extract compounds from a patent using image-based pipeline.

    For each table page with structure drawings:
    1. Render page as image
    2. Sonnet reads layout → compound locations
    3. Crop individual structures
    4. DECIMER converts → SMILES
    5. Optional: Opus Vision fallback for DECIMER failures

    Args:
        patent_id: Patent identifier.
        data_dir: Root data directory.
        use_opus_fallback: If True, use Opus Vision when DECIMER fails.

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

    if not table_pages:
        logger.info(f"Patent {patent_id}: No compound table pages found for image extraction")
        return []

    logger.info(f"Patent {patent_id}: Processing {len(table_pages)} table pages via image pipeline")

    doc = fitz.open(str(pdf_path))
    images_dir = config.IMAGES_DIR / patent_id
    images_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Render all pages and extract layouts (parallelizable)
    page_jobs = []  # (page_num, page_img, page_img_path, layouts)
    for page_num in sorted(table_pages):
        if page_num >= len(doc):
            continue
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_img_path = str(images_dir / f"page_{page_num:04d}.png")
        page_img.save(page_img_path)
        page_jobs.append((page_num, page_img, page_img_path))

    doc.close()

    # Phase 2: Sonnet layout extraction (cached, parallel)
    def _get_layout(job):
        page_num, page_img, page_img_path = job
        layouts = _extract_table_layout(page_img_path, patent_id, page_num)
        return page_num, page_img, layouts

    layout_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for result in executor.map(_get_layout, page_jobs):
            layout_results.append(result)

    # Phase 3: Crop + DECIMER (parallelized per compound)
    crop_jobs = []  # (cpd_no, crop_path, ms_mh, iupac_name, page_num)
    seen_ids: set[str] = set()

    for page_num, page_img, layouts in layout_results:
        if not layouts:
            continue
        for layout in layouts:
            cpd_no = str(layout.get('cpd_no', ''))
            if not cpd_no:
                continue
            dedup_key = f"example{cpd_no}".lower().replace(' ', '')
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)

            bbox = layout.get('structure_bbox', [])
            crop = _crop_structure(page_img, bbox)
            if crop is None:
                continue

            crop_path = str(images_dir / f"cpd_{cpd_no}_p{page_num}.png")
            crop.save(crop_path)
            crop_jobs.append((cpd_no, crop_path, layout.get('ms_mh'), layout.get('name'), page_num))

    logger.info(f"Patent {patent_id}: {len(crop_jobs)} structure crops to process with DECIMER")

    def _process_crop(job):
        cpd_no, crop_path, ms_mh, iupac_name, page_num = job
        smiles = _decimer_to_smiles(crop_path)
        extraction_method = "decimer_local"

        if smiles is None and use_opus_fallback:
            smiles = _opus_vision_fallback(crop_path, patent_id, cpd_no)
            extraction_method = "opus_vision_fallback"

        if smiles is None:
            return None

        canonical = canonicalize_smiles(smiles)
        if not canonical:
            return None

        inchikey = get_inchikey(canonical)
        salt_result = strip_salt(canonical)
        drug_likeness = compute_drug_likeness(canonical)

        return Compound(
            patent_id=patent_id,
            example_number=f"Cpd. No. {cpd_no}",
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
