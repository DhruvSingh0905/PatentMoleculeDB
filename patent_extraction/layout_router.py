"""Layout-aware router — detect patent format and route to best pipeline.

Two patent format types:
1. "text" — compounds have IUPAC names in Example sections → text pipeline (OPSIN)
2. "image" — compounds defined by Cpd.No. + structure drawings → image pipeline (DECIMER)

Detection heuristic:
- Count example pages with IUPAC-like text vs compound table pages with structure images
- If >60% of compounds are in tables with images → route to image pipeline
- Otherwise → text pipeline (with image fallback for stereo)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from . import config
from .models import PageManifest

logger = logging.getLogger(__name__)


def detect_patent_format(
    patent_id: str,
    manifest: PageManifest,
    data_dir: Path | None = None,
) -> dict:
    """Detect whether a patent is text-heavy or image-heavy.

    Returns:
        Dict with:
          format: "text" | "image" | "hybrid"
          text_score: 0-100 (how text-friendly)
          image_score: 0-100 (how image-dependent)
          routing: which pipeline steps to use
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    patent_dir = data_dir / patent_id

    # Count indicators
    iupacs_pages = 0
    iupacs_dir = patent_dir / "iupacs_clean"
    if iupacs_dir.exists():
        iupacs_pages = len(list(iupacs_dir.glob("page_*.md")))

    all_pages_count = 0
    all_pages_dir = patent_dir / "all_pages"
    if all_pages_dir.exists():
        all_pages_count = len(list(all_pages_dir.glob("page_*.md")))

    # Count pages with "Example N:" headers vs "Cpd. No." format
    example_headers = 0
    cpd_no_headers = 0
    image_heavy_pages = 0
    iupac_name_pages = 0

    if iupacs_dir.exists():
        for page_file in sorted(iupacs_dir.glob("page_*.md"))[:50]:  # Sample first 50
            text = page_file.read_text(encoding="utf-8")

            if re.search(r'Example\s+\d+', text, re.IGNORECASE):
                example_headers += 1
            if re.search(r'Cpd\.?\s*No\.?\s*\d+', text, re.IGNORECASE):
                cpd_no_headers += 1

            # Count image markers vs text content
            image_count = text.count('<|ref|>image<|/ref|>')
            text_blocks = len(re.findall(r'<\|ref\|>text<\|/ref\|>', text))

            if image_count > text_blocks:
                image_heavy_pages += 1

            # Check for IUPAC-like names (long chemical names with parentheses)
            if re.search(r'[a-z]{4,}-\d+-yl|pyrrolo|pyrazol|triazin|morpholin|piperidin', text, re.IGNORECASE):
                iupac_name_pages += 1

    sampled = min(50, iupacs_pages) if iupacs_pages > 0 else 1

    # Scoring
    text_score = int(100 * (example_headers + iupac_name_pages) / (2 * sampled))
    image_score = int(100 * (cpd_no_headers + image_heavy_pages) / (2 * sampled))

    # Determine format
    if text_score > 60 and image_score < 30:
        fmt = "text"
    elif image_score > 40 and text_score < 30:
        fmt = "image"
    else:
        fmt = "hybrid"

    # Routing instructions
    routing = {
        "use_text_pipeline": fmt in ("text", "hybrid"),
        "use_image_pipeline": fmt in ("image", "hybrid"),
        "use_claims_parser": True,  # Always try claims
        "use_table_parser": True,   # Always try tables
    }

    result = {
        "format": fmt,
        "text_score": min(text_score, 100),
        "image_score": min(image_score, 100),
        "example_headers": example_headers,
        "cpd_no_headers": cpd_no_headers,
        "iupac_name_pages": iupac_name_pages,
        "image_heavy_pages": image_heavy_pages,
        "iupacs_pages": iupacs_pages,
        "all_pages": all_pages_count,
        "routing": routing,
    }

    logger.info(
        f"Patent {patent_id}: format={fmt} "
        f"(text={text_score}, image={image_score}, "
        f"examples={example_headers}, cpd_no={cpd_no_headers})"
    )

    return result
