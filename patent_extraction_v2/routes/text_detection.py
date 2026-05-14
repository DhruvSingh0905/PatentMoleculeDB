"""Standalone text detection + recognition for patent pages.

Why this exists
---------------
PaddleX's `<patent>/all_pages/page_NNNN.md` runs a layout model first,
which classifies large figure regions as "image". Text inside those
image regions is NEVER OCR'd, so small compound-ID labels printed under
or beside structure drawings (e.g. "5a", "5b", "8A", "43A") are missing
from PaddleX's text blocks. Confirmed visually across 6 GT pages
(US10899738 p155, US11312727 p15/p17/p19/p63, etc.).

PaddleOCR 3.4.0 also exposes the text detection + recognition pipeline
WITHOUT the layout-model gate (`paddleocr.PaddleOCR`). Running this
directly on the rendered PDF page detects every text region on the
page — paragraphs, headings, AND the tiny standalone labels — with
their bboxes in the SAME pixel coord space we render at. That drops
straight into our canonical render-coord graph.

This wrapper:
  - Lazy-loads a singleton PaddleOCR pipeline (CPU)
  - Runs detect+recognize on a rendered PIL page image
  - Returns a list of CanonicalTextBlock-shaped tuples in render coords
  - Caches per (patent, page) hash to avoid re-inference

It does NOT filter by content. The caller (`mine_label_candidates`)
applies the compound-ID regex to keep paragraphs out of the candidate
pool. Keeping detection broad here means the same module also serves
future use cases (assay-value reading from tables — see plan deferred
follow-up #3).

Generic by construction. NO patent-specific code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Skip the model-hoster connectivity check; models are downloaded once and
# cached locally afterward.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from patent_extraction_v2.core import config as v2_config

logger = logging.getLogger(__name__)


@dataclass
class DetectedTextBlock:
    """One text region detected on a page.

    bbox_px is in the same coord space as the input image (which we
    feed in render-pixel coords, 300 DPI), so it drops directly into
    the canonical graph alongside DECIMER-Seg structure bboxes.
    """
    bbox_px: tuple[int, int, int, int]   # axis-aligned (x1, y1, x2, y2)
    content: str
    confidence: float


# ── Singleton + cache ──────────────────────────────────────────────

_PIPELINE = None
CACHE_DIR = v2_config.OUTPUT_DIR / "cache" / "text_detection"


def _get_pipeline():
    """Lazy-load the PaddleOCR pipeline. CPU; first call downloads
    the detection + recognition model weights (~100 MB total, cached).
    The loaded models occupy ~4-6 GB RAM — call `unload()` when done
    to release them.
    """
    global _PIPELINE
    if _PIPELINE is None:
        from paddleocr import PaddleOCR
        _PIPELINE = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _PIPELINE


def unload(force: bool = False) -> None:
    """Release the PaddleOCR pipeline + free its ~4-6 GB.

    Honors `v2_config.UNLOAD_MODELS_AFTER_USE`:
      - True (default, laptop): unload now.
      - False (cloud / big RAM): no-op so the next call hits the hot model.
    `force=True` overrides the config toggle (e.g. process is exiting).

    Safe to call even if the pipeline was never loaded.
    """
    global _PIPELINE
    if not force and not v2_config.UNLOAD_MODELS_AFTER_USE:
        return
    if _PIPELINE is not None:
        del _PIPELINE
        _PIPELINE = None
        import gc
        gc.collect()
        logger.info("PaddleOCR pipeline unloaded")


def _cache_path(patent_id: str, page: int, image_hash: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{patent_id}_p{page:04d}_{image_hash}.json"


def _hash_image(arr: np.ndarray) -> str:
    """Cheap content-hash of the image bytes — used as cache key so
    a re-render with different DPI gets a different cache entry.
    """
    return hashlib.sha1(arr.tobytes()).hexdigest()[:10]


# ── Public API ─────────────────────────────────────────────────────

def detect_text_blocks(
    page_image: Image.Image,
    patent_id: str | None = None,
    page: int | None = None,
    use_cache: bool = True,
) -> list[DetectedTextBlock]:
    """Run standalone OCR on a rendered page; return every detected
    text region with bbox + content.

    Args:
        page_image: rendered PDF page as PIL.Image. Coord space is
                    whatever DPI you rendered at — typically 300 DPI
                    to match DECIMER-Seg + cell detector.
        patent_id, page: used only for the disk cache key.
        use_cache: read/write the JSON cache.

    Returns:
        list of DetectedTextBlock, axis-aligned bboxes in input-image
        pixel coords. Empty list if PaddleOCR returns nothing or fails.
    """
    arr = np.array(page_image)
    img_hash = _hash_image(arr)

    # Cache lookup
    cache_file = None
    if use_cache and patent_id is not None and page is not None:
        cache_file = _cache_path(patent_id, page, img_hash)
        if cache_file.exists():
            try:
                raw = json.loads(cache_file.read_text())
                return [DetectedTextBlock(
                    bbox_px=tuple(b["bbox_px"]),
                    content=b["content"],
                    confidence=b["confidence"],
                ) for b in raw]
            except Exception:
                logger.warning("text-detection cache read failed at %s", cache_file)

    pipeline = _get_pipeline()
    try:
        results = pipeline.predict(arr)
    except Exception as e:
        logger.warning("PaddleOCR.predict failed for %s p%s: %s",
                       patent_id, page, e)
        return []

    out: list[DetectedTextBlock] = []
    # Result API (PaddleOCR 3.x): predict() returns an iterable of
    # OCRResult dict-likes per input image. Each has:
    #   "rec_polys": list of 4-point polygons (per detected text region)
    #   "rec_texts": list of recognized strings
    #   "rec_scores": list of confidences
    # We zip them and convert each polygon to an axis-aligned bbox.
    for result in results:
        try:
            polys = result.get("rec_polys") or result.get("dt_polys") or []
            texts = result.get("rec_texts") or []
            scores = result.get("rec_scores") or []
        except Exception:
            polys, texts, scores = [], [], []
        for poly, text, score in zip(polys, texts, scores):
            if not text:
                continue
            try:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bbox = (int(min(xs)), int(min(ys)),
                        int(max(xs)), int(max(ys)))
            except Exception:
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            out.append(DetectedTextBlock(
                bbox_px=bbox,
                content=str(text).strip(),
                confidence=float(score),
            ))

    if cache_file is not None:
        try:
            cache_file.write_text(json.dumps(
                [{"bbox_px": list(b.bbox_px), "content": b.content,
                  "confidence": b.confidence} for b in out],
                indent=2,
            ))
        except Exception:
            logger.warning("text-detection cache write failed at %s", cache_file)

    logger.info("PaddleOCR detected %d text regions on %s p%s",
                len(out), patent_id, page)
    return out


# ── CLI for visual debug ───────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Quick check: print every detected region for one (patent, page).

    Example:
        python3 -m patent_extraction_v2.routes.text_detection \\
            --patent US11312727 --page 15
    """
    import argparse
    from patent_extraction_v2.routes.decimer_segmentation_crop import (
        render_pdf_page,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--filter-short", action="store_true",
                    help="Show only short tokens (potential compound IDs).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    pdf = v2_config.DATA_DIR / args.patent / f"{args.patent}.pdf"
    if not pdf.exists():
        print(f"PDF not found: {pdf}")
        return 2
    page_img = render_pdf_page(pdf, args.page, dpi=args.dpi)
    blocks = detect_text_blocks(page_img, patent_id=args.patent, page=args.page,
                                 use_cache=not args.no_cache)
    print(f"\nDetected {len(blocks)} text regions on {args.patent} p{args.page}:")
    for b in blocks:
        if args.filter_short and len(b.content) > 6:
            continue
        print(f"  bbox={b.bbox_px}  conf={b.confidence:.2f}  text={b.content!r}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
