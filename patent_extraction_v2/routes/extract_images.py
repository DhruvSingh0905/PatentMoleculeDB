"""Step 2.3a — Extract structure images from PDFs at bounding box coordinates.

Uses PyMuPDF (fitz) to crop image regions from patent PDFs based on the
bounding boxes from the markdown annotations. Pre-filters blank and tiny images.

This module handles the LOCAL image extraction only (no API calls).
Image → SMILES conversion is in image_pipeline.py (DECIMER + vision fallback).
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from ..core import config
from ..core.markdown_parser import parse_markdown_file, extract_page_number
from ..core.models import PageManifest

logger = logging.getLogger(__name__)


def _bbox_to_pdf_rect(bbox: list[int], page_width: float, page_height: float) -> fitz.Rect:
    """Convert markdown bounding box coordinates to PDF coordinates.

    The markdown bbox uses a coordinate system where values range ~0-1000.
    Map proportionally to actual PDF page dimensions.

    Args:
        bbox: [x1, y1, x2, y2] from markdown annotations.
        page_width: Actual PDF page width in points.
        page_height: Actual PDF page height in points.

    Returns:
        fitz.Rect in PDF coordinate space.
    """
    # Observed bbox range is roughly 0-1000 (some patents go to ~860)
    # Normalize by assuming a 1000×1000 coordinate system
    COORD_MAX = 1000.0

    x1 = bbox[0] / COORD_MAX * page_width
    y1 = bbox[1] / COORD_MAX * page_height
    x2 = bbox[2] / COORD_MAX * page_width
    y2 = bbox[3] / COORD_MAX * page_height

    return fitz.Rect(x1, y1, x2, y2)


def _is_blank_image(image: Image.Image, threshold: float = None) -> bool:
    """Check if an image is blank (low pixel variance).

    Args:
        image: PIL Image to check.
        threshold: Minimum variance. Defaults to config value.

    Returns:
        True if the image appears blank.
    """
    if threshold is None:
        threshold = config.IMAGE_VARIANCE_THRESHOLD

    arr = np.array(image.convert("L"))  # Grayscale
    return float(np.var(arr)) < threshold


def _is_too_small(image: Image.Image, min_size: int = None) -> bool:
    """Check if image is too small to contain a useful structure.

    Args:
        image: PIL Image to check.
        min_size: Minimum dimension in pixels.

    Returns:
        True if either dimension is below minimum.
    """
    if min_size is None:
        min_size = config.IMAGE_MIN_SIZE

    return image.width < min_size or image.height < min_size


def extract_image_from_pdf(
    pdf_path: Path,
    page_num: int,
    bbox: list[int],
    dpi: int = None,
) -> Image.Image | None:
    """Extract a region from a PDF page as a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        page_num: 0-indexed page number.
        bbox: [x1, y1, x2, y2] bounding box from markdown.
        dpi: Resolution. Defaults to config value.

    Returns:
        PIL Image of the cropped region, or None if extraction fails.
    """
    if dpi is None:
        dpi = config.IMAGE_DPI_DEFAULT

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return None

    try:
        if page_num >= len(doc):
            logger.warning(f"Page {page_num} out of range for {pdf_path} ({len(doc)} pages)")
            return None

        page = doc[page_num]
        rect = _bbox_to_pdf_rect(bbox, page.rect.width, page.rect.height)

        # Render the clipped region at specified DPI
        zoom = dpi / 72  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        clip = rect

        pix = page.get_pixmap(matrix=mat, clip=clip)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return img

    except Exception as e:
        logger.error(f"Failed to extract image from {pdf_path} page {page_num}: {e}")
        return None
    finally:
        doc.close()


# NOTE: extract_all_images() was removed in the Phase-C cleanup.
# It belonged to the legacy "standard image extraction" branch in
# pipeline.py (paired with image_to_smiles.convert_all_images, since
# deleted). The new image route is image_pipeline.extract_image_compounds,
# which handles bbox extraction, OCR-first targeting, DECIMER, and
# vision fallback in one orchestrated flow.
#
# `extract_image_from_pdf` (the per-bbox utility above) is still used by
# image_pipeline.py and stays here.
