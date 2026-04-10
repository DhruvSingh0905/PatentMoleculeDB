"""Step 2.3a — Extract structure images from PDFs at bounding box coordinates.

Uses PyMuPDF (fitz) to crop image regions from patent PDFs based on the
bounding boxes from the markdown annotations. Pre-filters blank and tiny images.

This module handles the LOCAL image extraction only (no API calls).
Image → SMILES conversion is in image_to_smiles.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from . import config
from .markdown_parser import parse_markdown_file, extract_page_number
from .models import PageManifest

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


def extract_all_images(
    patent_id: str,
    manifest: PageManifest,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[dict]:
    """Extract all structure images from a patent's PDF.

    Scans iupacs_clean pages for image markers, extracts from PDF,
    filters blank/tiny images, saves to disk.

    Args:
        patent_id: Patent identifier.
        manifest: PageManifest with image_pages.
        data_dir: Root data directory.
        output_dir: Where to save extracted images.

    Returns:
        List of dicts: {page_num, bbox, image_path, example_number_hint}
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    if output_dir is None:
        output_dir = config.IMAGES_DIR / patent_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the PDF
    pdf_path = data_dir / patent_id / f"{patent_id}.pdf"
    if not pdf_path.exists():
        logger.error(f"Patent {patent_id}: PDF not found at {pdf_path}")
        return []

    # Scan iupacs_clean for image markers + bounding boxes
    iupacs_dir = data_dir / patent_id / "iupacs_clean"
    if not iupacs_dir.exists():
        logger.warning(f"Patent {patent_id}: No iupacs_clean directory")
        return []

    extracted = []
    skipped_blank = 0
    skipped_small = 0

    for page_file in sorted(iupacs_dir.glob("page_*.md")):
        page_num = extract_page_number(page_file)
        if page_num is None:
            continue

        elements = parse_markdown_file(page_file)

        for el in elements:
            if el.element_type != "image":
                continue

            bbox = el.bbox

            # Extract image from PDF
            img = extract_image_from_pdf(pdf_path, page_num, bbox)
            if img is None:
                continue

            # Filter blank images
            if _is_blank_image(img):
                skipped_blank += 1
                continue

            # Filter tiny images
            if _is_too_small(img):
                skipped_small += 1
                continue

            # Save image
            filename = f"p{page_num:04d}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"
            img_path = output_dir / filename
            img.save(str(img_path))

            extracted.append({
                "page_num": page_num,
                "bbox": bbox,
                "image_path": str(img_path),
            })

    logger.info(
        f"Patent {patent_id}: Extracted {len(extracted)} images "
        f"(skipped {skipped_blank} blank, {skipped_small} too small)"
    )

    return extracted
