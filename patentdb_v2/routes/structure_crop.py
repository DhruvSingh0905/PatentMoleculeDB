"""Geometric per-cell structure cropping for compound tables.

Why this exists
---------------
PaddleX gives us per-table bboxes but NOT per-cell bboxes inside tables.
The OLD pipeline asked Sonnet Vision to subdivide a table into per-compound
regions (LAYOUT_PROMPT in image_pipeline.py); that was ~50% wrong (crops
shifted vertically, partial molecules, mis-labeled compound IDs).

This module replaces that with a deterministic geometric approach:
  - Take the table bbox from upstream OCR (PaddleX or MinerU; both agree)
  - Skip the title-row + header-row vertical band (auto-detected from rendered
    text density at the top of the table)
  - Divide the remaining vertical extent by the known row count
  - Crop the structure column for each row
  - Validate each crop has reasonable black-pixel density (real structure
    drawing, not a blank cell)

Compound IDs come from PaddleX's first-column text on the SAME page (filtered
to skip atom-label bleed rows like OH/O/R). Match by row position.

Free, deterministic, no LM calls. Works whenever rows are uniformly spaced
(true for compound tables in pharma patents).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


# ── Atom-label set (filter PaddleX bleed rows) ──────────────────────
ATOM_LABELS = {
    'OH', 'O', 'N', 'NH', 'NH2', 'R', 'S', 'F', 'CL', 'BR', 'I',
    'P', 'C', 'CH', 'CH2', 'CH3', 'COOH', 'CO', 'CN', 'CHO',
}


# ── Configuration ────────────────────────────────────────────────────

# DPI used for cropping. Higher = better DECIMER recognition; ~300 is the
# sweet spot for patent structure drawings.
CROP_DPI = 300

# Minimum black-pixel ratio for a crop to count as "has a structure"
# (filters blank cells, just-text cells). Compound structure drawings
# typically have 3–15% black pixels.
MIN_BLACK_PCT = 0.5

# Header-row band as fraction of total table height. Default 12% — the
# title row + column-header row of a typical pharma compound table.
DEFAULT_HEADER_BAND_PCT = 0.12


@dataclass
class CroppedStructure:
    """One per-compound crop with metadata for downstream DECIMER + audit."""
    compound_id: str
    page_num: int
    bbox_px: tuple[int, int, int, int]   # in page-image pixel coords
    image_path: Path
    black_pixel_pct: float
    has_content: bool
    notes: str = ""


# ── PaddleX-text → compound ID list ──────────────────────────────────

def extract_compound_ids_from_paddlex(text: str) -> list[str]:
    """Walk PaddleX's <table> first-column cells, filter atom-label bleed,
    return compound IDs in vertical row order.

    For non-bleed tables, returns the IDs from first column directly.
    For bleed tables (US11312727 style), filters out OH/O/R rows.
    """
    m = re.search(r'<table>(.*?)</table>', text, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    rows = re.findall(r'<tr>(.*?)</tr>', m.group(1), re.DOTALL | re.IGNORECASE)
    ids: list[str] = []
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
        if not cells:
            continue
        first = re.sub(r'<[^>]+>', '', cells[0]).strip()
        if not first:
            continue
        # Skip atom-label bleed rows
        if first.upper() in ATOM_LABELS:
            continue
        # Skip header-like cells
        if re.search(r'\b(?:compound|cpd|example|antiviral|number|table|continued)\b', first, re.I):
            continue
        # Compound ID must contain at least one digit
        if not re.search(r'\d', first):
            continue
        # Trim long strings (probably parser noise)
        if len(first) > 20:
            continue
        ids.append(first)
    return ids


# ── Geometric cropping ───────────────────────────────────────────────

def render_pdf_page(pdf_path: Path, page_num: int, dpi: int = CROP_DPI) -> Image.Image:
    """Render a single PDF page to a PIL image at the given DPI.
    page_num is 1-indexed (as in PaddleX output)."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_num - 1)  # PyMuPDF is 0-indexed
        zoom = dpi / 72  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def black_pixel_pct(image: Image.Image, threshold: int = 100) -> float:
    """Return percentage of pixels darker than `threshold` (0-255 grayscale).
    Higher = more "ink" = more likely to contain a structure drawing."""
    gray = image.convert('L')
    pixels = list(gray.getdata())
    n_dark = sum(1 for p in pixels if p < threshold)
    return 100.0 * n_dark / max(len(pixels), 1)


def detect_column_boundaries(table_image: Image.Image,
                               header_end_y: int) -> list[int]:
    """Find column-separator x-positions by looking for vertical strips of
    very-low ink density in the DATA region (below the header).

    Returns list of x-coordinates that mark column boundaries (left edges).
    First entry is always 0; last is table width.
    """
    w, h = table_image.size
    if h <= header_end_y:
        return [0, w]
    gray = table_image.convert('L')

    # Compute column-wise black-pixel density (one value per pixel-column)
    # over the DATA region only (skip header — its dense headers add noise)
    data_y_start = header_end_y
    data_y_end = h
    sample_step = max(1, (data_y_end - data_y_start) // 200)  # sample 200 rows
    col_densities = []
    for x in range(w):
        col_pixels = [gray.getpixel((x, y)) for y in range(data_y_start, data_y_end, sample_step)]
        n_dark = sum(1 for p in col_pixels if p < 100)
        col_densities.append(n_dark / max(len(col_pixels), 1))

    # Find strips of consecutive low-density columns (gap between cells)
    # These are the column boundaries.
    GAP_THRESHOLD = 0.005       # essentially zero ink
    MIN_GAP_WIDTH = 4           # need at least 4 px of consecutive low density
    boundaries = [0]
    in_gap = False
    gap_start = 0
    for x in range(w):
        if col_densities[x] < GAP_THRESHOLD:
            if not in_gap:
                in_gap = True
                gap_start = x
        else:
            if in_gap and (x - gap_start) >= MIN_GAP_WIDTH:
                # End of a gap — boundary at the middle of the gap
                boundaries.append((gap_start + x) // 2)
            in_gap = False
    boundaries.append(w)
    # Dedupe nearby boundaries (within 10 px → keep just the first)
    deduped = [boundaries[0]]
    for b in boundaries[1:]:
        if b - deduped[-1] >= 10:
            deduped.append(b)
    return deduped


def detect_header_band_height(table_image: Image.Image,
                               default_pct: float = DEFAULT_HEADER_BAND_PCT) -> int:
    """Find the bottom of the header band by looking for the FIRST row of
    very-low-content vertical strip (the gap between header and first data row).

    Returns Y-coordinate (in table-image pixels) where the data rows start.
    Falls back to default_pct of total height if no clear gap is found.
    """
    w, h = table_image.size
    gray = table_image.convert('L')
    # Compute row-wise black-pixel density (one value per pixel-row)
    row_densities = []
    for y in range(h):
        row_pixels = [gray.getpixel((x, y)) for x in range(0, w, max(1, w // 100))]  # sample 100 columns
        n_dark = sum(1 for p in row_pixels if p < 100)
        row_densities.append(n_dark / max(len(row_pixels), 1))
    # Find first sustained "low-density gap" after the top.
    # Scan downward starting at the default header position; look for a band
    # of >5 consecutive rows with density < 0.01 (essentially blank).
    start_search = int(h * 0.08)   # skip the first 8% (just-the-title region)
    consecutive_blank = 0
    GAP_THRESHOLD = 0.01
    GAP_LEN = 5
    for y in range(start_search, int(h * 0.30)):  # only look in first 30% of table
        if row_densities[y] < GAP_THRESHOLD:
            consecutive_blank += 1
            if consecutive_blank >= GAP_LEN:
                # Found the gap — data rows start just below
                return y + 2
        else:
            consecutive_blank = 0
    # Fallback
    return int(h * default_pct)


def geometric_crop_table(
    pdf_path: Path,
    page_num: int,
    table_bbox_pdf: tuple[float, float, float, float],
    compound_ids: list[str],
    structure_col_x_range_pct: tuple[float, float] | None = None,
    output_dir: Path | None = None,
) -> list[CroppedStructure]:
    """Crop each compound's structure cell from a table by geometric inference.

    Args:
        pdf_path: path to the patent PDF
        page_num: 1-indexed page number
        table_bbox_pdf: (x1, y1, x2, y2) in PDF coordinates (72 DPI)
                        from MinerU/PaddleX layout output
        compound_ids: list of compound IDs in vertical top-to-bottom order
        structure_col_x_range_pct: x-range as fraction of table width
                                   (default left 0–45% — typical structure column)
        output_dir: where to save crops

    Returns:
        list of CroppedStructure, one per compound_id (or fewer if rows
        couldn't be identified). Caller should verify .has_content.
    """
    if not compound_ids:
        return []

    # 1. Render the whole page at high DPI
    page_img = render_pdf_page(pdf_path, page_num, dpi=CROP_DPI)
    pw, ph = page_img.size
    # Scale table bbox from PDF coords (72 DPI) to image coords (CROP_DPI)
    scale = CROP_DPI / 72
    tx1 = int(table_bbox_pdf[0] * scale)
    ty1 = int(table_bbox_pdf[1] * scale)
    tx2 = int(table_bbox_pdf[2] * scale)
    ty2 = int(table_bbox_pdf[3] * scale)
    tx1, ty1 = max(0, tx1), max(0, ty1)
    tx2, ty2 = min(pw, tx2), min(ph, ty2)
    if tx2 <= tx1 or ty2 <= ty1:
        logger.warning(f"Invalid table bbox: ({tx1},{ty1})-({tx2},{ty2})")
        return []
    table_img = page_img.crop((tx1, ty1, tx2, ty2))
    tw, th = table_img.size

    # 2. Detect header band (auto-detected; skip the title + column header)
    header_end_y = detect_header_band_height(table_img)
    data_rows_y_start = header_end_y
    data_rows_y_end = th
    n_rows = len(compound_ids)
    row_height = (data_rows_y_end - data_rows_y_start) / n_rows
    logger.debug(
        f"page {page_num}: table {tw}×{th}, header_end={header_end_y}, "
        f"row_height={row_height:.1f}px for {n_rows} rows"
    )

    # 3. Define the structure-column x-range.
    # Use the caller-provided x-range when available (computed from MinerU's
    # column count). Otherwise, default to a safe leftmost-30% slice.
    # We deliberately AVOID image-based column detection here — it's too
    # fragile to layout edge cases (white border strips, header overflow).
    if structure_col_x_range_pct is not None:
        sx1 = int(tw * structure_col_x_range_pct[0])
        sx2 = int(tw * structure_col_x_range_pct[1])
    else:
        sx1, sx2 = 0, int(tw * 0.30)

    # 4. Crop each row's structure column
    if output_dir is None:
        output_dir = Path("output_v2/images") / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    out: list[CroppedStructure] = []
    for i, cpd_id in enumerate(compound_ids):
        row_y1 = int(data_rows_y_start + i * row_height)
        row_y2 = int(data_rows_y_start + (i + 1) * row_height)
        # No vertical padding — capturing tiny pieces of adjacent compounds
        # was a worse failure mode than clipping descenders. If clipping
        # turns out to matter, we can detect it from low-content edges
        # post-crop and re-extend.
        crop = table_img.crop((sx1, row_y1, sx2, row_y2))
        # Validate content
        bp = black_pixel_pct(crop)
        # Use page-image absolute coords for the bbox so audit can map back
        bbox_px = (tx1 + sx1, ty1 + row_y1, tx1 + sx2, ty1 + row_y2)
        # Sanitize compound id for filename
        safe_id = re.sub(r'[^\w\-]', '_', cpd_id)
        out_path = output_dir / f"p{page_num:04d}_cpd_{safe_id}.png"
        crop.save(out_path)
        out.append(CroppedStructure(
            compound_id=cpd_id,
            page_num=page_num,
            bbox_px=bbox_px,
            image_path=out_path,
            black_pixel_pct=bp,
            has_content=bp >= MIN_BLACK_PCT,
            notes=("low_content" if bp < MIN_BLACK_PCT else ""),
        ))
    return out


# ── MinerU-driven structure-column detection ────────────────────────

@dataclass
class TableLayout:
    """Per-page table layout inferred from MinerU's clean HTML."""
    n_data_rows: int                      # count of visible data rows on this page
    n_columns: int                        # column count from header
    has_structure_column: bool            # leftmost col contains structure cells (img refs or empty cells where bleed text was)
    structure_col_x_range_pct: tuple[float, float]  # (x_start, x_end) as table-width fraction


def detect_table_layout_from_mineru(mineru_md_path: Path) -> TableLayout | None:
    """Parse MinerU's <table> to determine row count, column count, and
    whether the leftmost column contains structures (vs pure data).

    Heuristic for "has structure column":
      - Leftmost column has any `<img>` reference, OR
      - Leftmost column has cells that are mostly empty in DATA rows (drawings
        that MinerU couldn't OCR became empty cells)
    """
    if not mineru_md_path or not mineru_md_path.exists():
        return None
    text = mineru_md_path.read_text(encoding='utf-8')
    tbl = re.search(r'<table>(.*?)</table>', text, re.DOTALL | re.IGNORECASE)
    if not tbl:
        return None
    rows = re.findall(r'<tr>(.*?)</tr>', tbl.group(1), re.DOTALL | re.IGNORECASE)
    if not rows:
        return None

    # Determine column count from the first row that has multiple non-title cells
    n_cols = 0
    for r in rows[:3]:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', r, re.DOTALL | re.IGNORECASE)
        if len(cells) >= 2:
            n_cols = len(cells)
            break
    if n_cols == 0:
        # Single-cell title-only table; no usable structure
        return TableLayout(0, 1, False, (0.0, 0.0))

    # Walk DATA rows to count + check the leftmost cell
    data_row_count = 0
    leftmost_empty_count = 0
    leftmost_img_count = 0
    leftmost_numeric_count = 0
    for r in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', r, re.DOTALL | re.IGNORECASE)
        if not cells:
            continue
        # Cleaned cells (preserve <img> as marker)
        cleaned = []
        for c in cells:
            if 'img' in c.lower():
                cleaned.append('|IMG|')
            else:
                cleaned.append(re.sub(r'<[^>]+>', '', c).strip())
        # Is this a data row? (any cell looks numeric)
        has_numeric = any(re.match(r'^[<>~≤≥]?\s*\d+(\.\d+)?\s*$', re.sub(r'&[a-z]+;', '', c)) for c in cleaned)
        if not has_numeric:
            continue
        data_row_count += 1
        first = cleaned[0] if cleaned else ''
        if first == '|IMG|':
            leftmost_img_count += 1
        elif first == '':
            leftmost_empty_count += 1
        elif re.match(r'^\d+[A-Za-z]*$', first):
            leftmost_numeric_count += 1

    # Decide: has_structure_column
    # If leftmost column is mostly numeric Cpd.No. → NO structure column (it's a data table)
    # If leftmost column has imgs OR mostly empty (where structures were OCR'd as nothing) → HAS structure column
    has_structure_col = (leftmost_img_count + leftmost_empty_count) > leftmost_numeric_count

    # Compute x-range: structure column is column 0 of n_cols total.
    # Empirical observation across pharma patents: when a table has a
    # structure column, that column is usually 1.5–2× wider than the value
    # columns (because structures take more visual space than a 4-digit
    # IC50 number). Using a multiplier of 1.6 over the equal-share width
    # captures the structure without bleeding into the next column.
    # This is GENERIC — derived from MinerU's column count, not patent-specific.
    if has_structure_col:
        col_width_pct = 1.0 / n_cols
        x_start = 0.0
        x_end = min(0.50, col_width_pct * 1.6)   # cap at 50% to avoid runaway
        x_range = (x_start, x_end)
    else:
        x_range = (0.0, 0.0)

    return TableLayout(
        n_data_rows=data_row_count,
        n_columns=n_cols,
        has_structure_column=has_structure_col,
        structure_col_x_range_pct=x_range,
    )


# ── Convenience: end-to-end on a page (PaddleX-text + MinerU-bbox + layout) ──

def crop_compounds_for_page(
    patent_id: str,
    page_num: int,
    pdf_path: Path,
    paddlex_md_path: Path,
    mineru_md_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[CroppedStructure]:
    """Combine MinerU layout (row count + structure column detection) +
    PaddleX text (compound IDs) + table bbox → cropped structures.

    Returns empty list (no cropping) if:
      - MinerU detects no structure column on this page (e.g. Cpd.No.+IC50 only)
      - No compound IDs found in PaddleX text
      - No table bbox could be determined
    """
    paddlex_text = paddlex_md_path.read_text(encoding='utf-8')
    cpd_ids = extract_compound_ids_from_paddlex(paddlex_text)

    # 1. Get layout from MinerU (preferred) — tells us row count + structure-col
    layout = detect_table_layout_from_mineru(mineru_md_path) if mineru_md_path else None

    # 2. SKIP if MinerU says no structure column
    if layout and not layout.has_structure_column:
        logger.info(
            f"{patent_id} p{page_num}: no structure column detected "
            f"({layout.n_columns} cols, leftmost is numeric Cpd.No.) — skipping"
        )
        return []

    if not cpd_ids:
        logger.info(f"{patent_id} p{page_num}: no compound IDs found in PaddleX text")
        return []

    # 3. Determine row count: prefer MinerU's data-row count (per-page-accurate);
    # fall back to PaddleX compound-ID count
    n_rows = layout.n_data_rows if (layout and layout.n_data_rows > 0) else len(cpd_ids)

    # 4. If MinerU saw more rows than PaddleX has IDs, label extras "unknown_pNNN_rowN"
    if n_rows > len(cpd_ids):
        cpd_ids = cpd_ids + [f"unknown_p{page_num}_r{i}"
                              for i in range(len(cpd_ids), n_rows)]
    elif n_rows < len(cpd_ids):
        # PaddleX over-counted (probably continuation table) — trim to actual visible
        cpd_ids = cpd_ids[:n_rows]

    # 5. Get table bbox — prefer MinerU's, fall back to PaddleX
    table_bbox = None
    if mineru_md_path and mineru_md_path.exists():
        content_list = mineru_md_path.parent / f"{mineru_md_path.stem}_content_list.json"
        if content_list.exists():
            import json
            for item in json.loads(content_list.read_text()):
                if item.get('type') == 'table' and item.get('bbox'):
                    table_bbox = tuple(item['bbox'])
                    break
    if table_bbox is None:
        m = re.search(
            r'<\|ref\|>table<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]',
            paddlex_text
        )
        if m:
            doc = fitz.open(str(pdf_path))
            try:
                page = doc.load_page(page_num - 1)
                pw, ph = page.rect.width, page.rect.height
            finally:
                doc.close()
            x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
            table_bbox = (x1 / 1000 * pw, y1 / 1000 * ph,
                          x2 / 1000 * pw, y2 / 1000 * ph)
    if table_bbox is None:
        logger.warning(f"{patent_id} p{page_num}: no table bbox found")
        return []

    # 6. Determine x-range: prefer MinerU-detected, else use a safe default
    x_range = layout.structure_col_x_range_pct if layout else (0.0, 0.30)

    return geometric_crop_table(
        pdf_path, page_num, table_bbox, cpd_ids,
        structure_col_x_range_pct=x_range,
        output_dir=output_dir,
    )
