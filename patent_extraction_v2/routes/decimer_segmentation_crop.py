"""Chemistry-aware per-molecule cropping via DECIMER-Segmentation.

Why this exists
---------------
Geometric cropping (`structure_crop.py:geometric_crop_table`) assumes
fixed-width structure cells and uniform row heights. Real chemistry tables
have variable-size molecules (simple aromatics ~100 px wide vs peptide
macrocycles ~800+ px), variable row heights, and intermixed layouts. The
geometric approach cannot generalize across patents.

DECIMER-Segmentation v1.5.0 (Nature Communications 2023, same authors as
DECIMER) is purpose-built for "find each chemical structure on a scientific
PDF page and return a tight crop." It uses a Mask R-CNN trained on chemistry
literature pages — so it identifies molecule regions whatever their size or
layout.

This module is a thin wrapper around `decimer_segmentation.segment_chemical_structures`:
  - Render the PDF page to a PIL/numpy image
  - Run the segmentation model -> list of crops + bounding boxes
  - Sort in reading order (top-to-bottom, then left-to-right within rows)
  - Save crops + return CroppedStructure records

Compound-ID assignment is handled separately (see `assign_compound_ids` in
the cropping eval harness): we zip the row-sorted crops to a row-sorted list
of compound IDs from PaddleX text. This is the same row-order zip used by
the geometric strategy — DECIMER-Seg only solves the bounding-box problem,
not the labeling problem.

First-call cost: model weights (~200 MB) are downloaded on first invocation
and cached locally by the decimer_segmentation library.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from patent_extraction_v2.core import config as v2_config

logger = logging.getLogger(__name__)


def unload(force: bool = False) -> None:
    """Release the DECIMER-Seg Mask R-CNN model (~2-3 GB).

    decimer_segmentation keeps a module-level `_model` singleton; we
    null it out + run gc to free TF graph memory. Safe even if the
    model was never loaded.

    Honors `v2_config.UNLOAD_MODELS_AFTER_USE` — set False on cloud
    boxes to keep the model hot.
    """
    if not force and not v2_config.UNLOAD_MODELS_AFTER_USE:
        return
    try:
        import decimer_segmentation.decimer_segmentation as ds_mod
        if getattr(ds_mod, "_model", None) is not None:
            ds_mod._model = None
            import gc
            gc.collect()
            logger.info("DECIMER-Seg model unloaded")
    except Exception as e:
        logger.debug("DECIMER-Seg unload skipped: %s", e)

# Same DPI as the geometric path so crops are directly comparable
SEG_DPI = 300


@dataclass
class SegCrop:
    """One DECIMER-Seg-detected molecule on a page."""
    page_num: int
    bbox_px: tuple[int, int, int, int]   # (x1, y1, x2, y2) in page-image px
    image_path: Path
    crop_index: int                       # 0-based reading-order index


def render_pdf_page(pdf_path: Path, page_num: int, dpi: int = SEG_DPI) -> Image.Image:
    """Render a single PDF page to a PIL image at the given DPI.

    `page_num` is 1-indexed (matches PaddleX/MinerU page numbering).
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_num - 1)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def segment_page(
    pdf_path: Path,
    page_num: int,
    output_dir: Path,
    dpi: int = SEG_DPI,
    use_cache: bool = True,
) -> list[SegCrop]:
    """Segment all chemical structures on one PDF page.

    Returns a list of `SegCrop` records ordered top-to-bottom, then
    left-to-right within rows (reading order). Each crop is saved as
    `output_dir/seg_p{page}_{idx}.png`. A bbox JSON sidecar at
    `output_dir/seg_p{page}_meta.json` lets cached pages skip the heavy
    TF inference entirely on subsequent runs.

    DECIMER-Seg's TF model is ~2-3 GB; loading it just to re-derive
    crops we already have on disk is wasteful. The cache is keyed by
    (pdf path, page, dpi), assumed deterministic for a given input.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"seg_p{page_num:04d}_meta.json"

    # ── Cache hit: rebuild SegCrop list from sidecar without loading TF.
    if use_cache and meta_path.exists():
        try:
            import json
            meta = json.loads(meta_path.read_text())
            if meta.get("dpi") == dpi:
                cached: list[SegCrop] = []
                for entry in meta["crops"]:
                    p = Path(entry["image_path"])
                    if not p.exists():
                        # Sidecar references a missing PNG → cache invalid
                        cached = []
                        break
                    cached.append(SegCrop(
                        page_num=page_num,
                        bbox_px=tuple(entry["bbox_px"]),
                        image_path=p,
                        crop_index=entry["crop_index"],
                    ))
                if cached or meta.get("n_segments") == 0:
                    logger.info("DECIMER-Seg cache HIT: %d crops on %s p%d",
                                len(cached), pdf_path.name, page_num)
                    return cached
        except Exception as e:
            logger.warning("DECIMER-Seg cache read failed: %s — re-running", e)

    # Local import: heavy TF dependency, only paid when actually segmenting
    from decimer_segmentation import (
        segment_chemical_structures,
        sort_segments_bboxes,
    )

    page_image = render_pdf_page(pdf_path, page_num, dpi=dpi)
    page_array = np.array(page_image)

    segments, bboxes = segment_chemical_structures(
        page_array,
        expand=True,
        return_bboxes=True,
    )

    out: list[SegCrop] = []

    if segments:
        # Reading order (rows first, then x within each row)
        segments, bboxes = sort_segments_bboxes(
            segments, bboxes,
            same_row_pixel_threshold=int(0.04 * page_image.height),
        )
        for i, (seg, bbox) in enumerate(zip(segments, bboxes)):
            crop_img = Image.fromarray(seg)
            path = output_dir / f"seg_p{page_num:04d}_{i:02d}.png"
            crop_img.save(path)
            y1, x1, y2, x2 = bbox
            out.append(SegCrop(
                page_num=page_num,
                bbox_px=(int(x1), int(y1), int(x2), int(y2)),
                image_path=path,
                crop_index=i,
            ))

    # Write sidecar (always, even if 0 crops — caches the "no molecules" verdict)
    try:
        import json
        meta_path.write_text(json.dumps({
            "pdf": str(pdf_path),
            "page": page_num,
            "dpi": dpi,
            "n_segments": len(out),
            "crops": [{
                "crop_index": c.crop_index,
                "bbox_px": list(c.bbox_px),
                "image_path": str(c.image_path),
            } for c in out],
        }, indent=2))
    except Exception as e:
        logger.warning("DECIMER-Seg cache write failed: %s", e)

    logger.info("DECIMER-Seg: %d molecules on %s p%d (cached)",
                len(out), pdf_path.name, page_num)
    return out
