"""Per-cell bbox detection inside table regions.

Why this exists
---------------
Verified during planning: MinerU and PaddleX both emit only the WHOLE-table
bbox in their layout output, no per-`<td>` coordinates. Without per-cell
bboxes, the table-cell-IoU branch of the label-association layer can't run
on compound-table pages (US10899738 p27, p28+, US9718825 substituent tables,
etc.) — the entire structure-table page class stays unlabel-able.

PaddleOCR 3.4.0 ships `TableCellsDetection` (RT-DETR-L_wired_table_cell_det)
in our existing venv. This is a thin wrapper that:

  1. Takes a rendered page image + the list of table-region bboxes that
     PaddleX/MinerU already gave us.
  2. For each table region, crops the region and runs TableCellsDetection
     to get per-cell bboxes in the crop frame.
  3. Translates cell bboxes back to page-image coordinates.
  4. Clusters cells into rows + cols by y-center / x-center to get
     (row_index, col_index). No fixed cell count assumed.

The output Cell objects have everything `label_association.py` needs to
do cell-IoU matching (Step 4a in the plan).

Generic by construction. NO patent_id checks. The same code runs against
any chemistry, biology, or unrelated table.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

# Network is optional; if PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK isn't set,
# PaddleOCR pings model hosters on import.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from patent_extraction_v2.core import config as v2_config

logger = logging.getLogger(__name__)


@dataclass
class Cell:
    """One detected table cell on a page.

    bbox_px is in the rendered-PDF coordinate space (300 DPI by default,
    matching DECIMER-Seg's coord space). table_id, row, col are derived
    by clustering cell centers within a single table region.
    """
    bbox_px: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    table_id: int                         # 0-based index of the table on page
    row: int                              # 0-based row within the table
    col: int                              # 0-based column within the table
    score: float                          # detector confidence


# ── Detector singletons ─────────────────────────────────────────────
# Both wired and wireless detectors. Wired (RT-DETR-L_wired_table_cell_det)
# is the PaddleOCR default; we try it first then fall back to wireless
# (RT-DETR-L_wireless_table_cell_det) if it returns nothing. Most chemistry
# compound tables in patents are WIRELESS — no visible cell borders, just
# whitespace separation — so wireless wins on the patents we care about.
# Trying wired first costs one quick inference but lets us handle either
# style without per-page classification overhead.

_DETECTORS: dict[str, object] = {}

_WIRED_MODEL    = "RT-DETR-L_wired_table_cell_det"
_WIRELESS_MODEL = "RT-DETR-L_wireless_table_cell_det"


def _get_detector(model_name: str):
    """Lazy-loaded singleton per detector model.
    Uses paddlex.create_model so we can pick the model by name."""
    if model_name not in _DETECTORS:
        from paddlex import create_model
        _DETECTORS[model_name] = create_model(model_name=model_name)
    return _DETECTORS[model_name]


def unload(force: bool = False) -> None:
    """Release all loaded cell-detection models (~2-4 GB).

    Honors `v2_config.UNLOAD_MODELS_AFTER_USE` — set to False on cloud
    instances with plenty of RAM to keep models hot across calls.
    """
    global _DETECTORS
    if not force and not v2_config.UNLOAD_MODELS_AFTER_USE:
        return
    if _DETECTORS:
        _DETECTORS.clear()
        import gc
        gc.collect()
        logger.info("Table cell-detection models unloaded")


def _extract_boxes(results) -> list[tuple[int, int, int, int, float]]:
    """Normalize the detector's `results` iterable into a list of
    (x1, y1, x2, y2, score) in CROP coordinates.

    PaddleX/PaddleOCR DetResult is dict-like with key "boxes" — a list of
    per-detection dicts containing `coordinate` / `bbox` / `box` for the
    bbox tuple and `score` for confidence. Different versions emit
    different keys; we accept all common variants.
    """
    out: list[tuple[int, int, int, int, float]] = []
    for result in results:
        boxes = None
        if isinstance(result, dict) or hasattr(result, "keys"):
            boxes = result.get("boxes") or result.get("bboxes")
        elif hasattr(result, "boxes"):
            boxes = result.boxes
        if not boxes:
            continue
        for b in boxes:
            if isinstance(b, dict) or hasattr(b, "keys"):
                coord = b.get("coordinate") or b.get("bbox") or b.get("box")
                score = float(b.get("score", 0.0))
            else:
                coord = list(b[:4])
                score = float(b[4]) if len(b) > 4 else 0.0
            if coord is None or len(coord) < 4:
                continue
            out.append(tuple(int(v) for v in coord[:4]) + (score,))
    return out


# ── Disk cache ──────────────────────────────────────────────────────

CACHE_DIR = v2_config.OUTPUT_DIR / "cache" / "table_cells"


def _cache_path(patent_id: str, page: int, table_bboxes_signature: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{patent_id}_p{page:04d}_{table_bboxes_signature}.json"


def _signature(table_bboxes_px: list[tuple[int, int, int, int]]) -> str:
    raw = json.dumps([list(b) for b in table_bboxes_px], sort_keys=True).encode()
    return hashlib.sha1(raw).hexdigest()[:10]


# ── Row / col clustering ────────────────────────────────────────────

def _cluster_1d(values: list[float], gap_threshold: float) -> list[int]:
    """Cluster sorted values where adjacent values within `gap_threshold`
    are in the same cluster. Returns the cluster index per input value
    (preserving the input order, not the sorted order)."""
    if not values:
        return []
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    cluster_idx = [0] * len(values)
    current = 0
    last_v = values[sorted_idx[0]]
    cluster_idx[sorted_idx[0]] = current
    for i in sorted_idx[1:]:
        if values[i] - last_v > gap_threshold:
            current += 1
        cluster_idx[i] = current
        last_v = values[i]
    return cluster_idx


# ── Public API ──────────────────────────────────────────────────────

def detect_table_cells(
    page_image: Image.Image,
    table_bboxes_px: list[tuple[int, int, int, int]] | None,
    patent_id: str | None = None,
    page: int | None = None,
    use_cache: bool = True,
) -> list[Cell]:
    """Detect per-cell bboxes within table regions on a page.

    Args:
        page_image: rendered PDF page (PIL.Image). Must be in the same
                    coord space as `table_bboxes_px` (typically 300 DPI).
        table_bboxes_px: list of (x1, y1, x2, y2) for table regions, e.g.,
                        from PaddleX `<|ref|>table<|/ref|>` block bboxes
                        scaled to page-image coords. If None or empty,
                        the detector runs on the WHOLE PAGE — handles
                        cases where PaddleX OCR'd a structure table as
                        individual image+text pairs without a `<|ref|>table<|/ref|>`
                        wrapper (US10899738 p27 etc.).
        patent_id, page: optional, used for caching only.
        use_cache: read/write the disk cache keyed by (patent_id, page,
                   table_bboxes signature).

    Returns:
        List of Cell records with row/col indices clustered within each
        table region. Empty if the detector finds nothing.
    """
    # Whole-page fallback: when no upstream table region is given, treat
    # the entire page as the search area. The cell detector itself
    # decides where cells exist; downstream code can then cluster them
    # into tables if needed.
    if not table_bboxes_px:
        page_w, page_h = page_image.size
        table_bboxes_px = [(0, 0, page_w, page_h)]

    # Cache lookup
    cache_file = None
    if use_cache and patent_id is not None and page is not None:
        sig = _signature(table_bboxes_px)
        cache_file = _cache_path(patent_id, page, sig)
        if cache_file.exists():
            try:
                raw = json.loads(cache_file.read_text())
                return [Cell(
                    bbox_px=tuple(c["bbox_px"]),
                    table_id=c["table_id"],
                    row=c["row"], col=c["col"],
                    score=c["score"],
                ) for c in raw]
            except Exception:
                logger.warning("Cell-detection cache read failed at %s", cache_file)

    all_cells: list[Cell] = []
    page_w, page_h = page_image.size

    for table_id, (tx1, ty1, tx2, ty2) in enumerate(table_bboxes_px):
        # Clamp + sanity-check the table bbox
        tx1 = max(0, int(tx1)); ty1 = max(0, int(ty1))
        tx2 = min(page_w, int(tx2)); ty2 = min(page_h, int(ty2))
        if tx2 <= tx1 or ty2 <= ty1:
            continue
        crop = page_image.crop((tx1, ty1, tx2, ty2))
        crop_arr = np.array(crop)

        # Try wired first; if 0 cells, fall back to wireless. Most chemistry
        # tables are wireless so wireless will usually win. Track which
        # model produced the cells (for debug/audit, not for scoring).
        crop_cells: list[tuple[int, int, int, int, float]] = []
        detector_used: str | None = None
        for model_name in (_WIRED_MODEL, _WIRELESS_MODEL):
            detector = _get_detector(model_name)
            try:
                results = detector.predict(crop_arr)
            except Exception as e:
                logger.warning("Cell detector %s failed on table %d: %s",
                               model_name, table_id, e)
                continue
            crop_cells = _extract_boxes(results)
            if crop_cells:
                detector_used = model_name
                break

        if not crop_cells:
            logger.info(
                "table %d (%s p%s): no cells found by either wired or wireless detector",
                table_id, patent_id, page,
            )
            continue
        logger.info(
            "table %d (%s p%s): %d cells found via %s",
            table_id, patent_id, page, len(crop_cells), detector_used,
        )

        # Translate back to page coords and cluster into rows/cols
        page_coord_cells = [
            (cx1 + tx1, cy1 + ty1, cx2 + tx1, cy2 + ty1, score)
            for (cx1, cy1, cx2, cy2, score) in crop_cells
        ]
        # Cluster gap = ~half the median cell height/width within this table
        cell_heights = [c[3] - c[1] for c in page_coord_cells]
        cell_widths = [c[2] - c[0] for c in page_coord_cells]
        row_gap = max(5.0, float(np.median(cell_heights)) * 0.4)
        col_gap = max(5.0, float(np.median(cell_widths)) * 0.4)

        y_centers = [(c[1] + c[3]) / 2.0 for c in page_coord_cells]
        x_centers = [(c[0] + c[2]) / 2.0 for c in page_coord_cells]
        rows = _cluster_1d(y_centers, row_gap)
        cols = _cluster_1d(x_centers, col_gap)

        for (x1, y1, x2, y2, score), r, c in zip(page_coord_cells, rows, cols):
            all_cells.append(Cell(
                bbox_px=(x1, y1, x2, y2),
                table_id=table_id, row=r, col=c, score=score,
            ))

    # Persist cache
    if cache_file is not None:
        try:
            cache_file.write_text(json.dumps(
                [asdict(c) | {"bbox_px": list(c.bbox_px)} for c in all_cells],
                indent=2,
            ))
        except Exception:
            logger.warning("Cell-detection cache write failed at %s", cache_file)

    logger.info(
        "TableCellsDetection: %s p%s -> %d cells across %d tables",
        patent_id, page, len(all_cells), len(table_bboxes_px),
    )
    return all_cells


# ── CLI for visual debugging ─────────────────────────────────────────

def _paddlex_table_bboxes_from_md(md_text: str, paddlex_to_render_scale: float) -> list[tuple[int, int, int, int]]:
    """Pull `<|ref|>table<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` bboxes
    out of PaddleX markdown and scale to render-DPI coordinate space.
    """
    block_re = re.compile(
        r"<\|ref\|>table<\|/ref\|><\|det\|>"
        r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]"
    )
    out: list[tuple[int, int, int, int]] = []
    for m in block_re.finditer(md_text):
        x1, y1, x2, y2 = (int(m.group(i)) for i in (1, 2, 3, 4))
        out.append((
            int(x1 * paddlex_to_render_scale),
            int(y1 * paddlex_to_render_scale),
            int(x2 * paddlex_to_render_scale),
            int(y2 * paddlex_to_render_scale),
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: detect cells on a single (patent, page), optionally write a
    debug visualization PNG with cell bboxes drawn on the rendered page.

    Example:
        python3 -m patent_extraction_v2.routes.table_cell_detection \\
            --patent US10899738 --page 27 --debug
    """
    import argparse
    from patent_extraction_v2.routes.decimer_segmentation_crop import render_pdf_page

    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--debug", action="store_true",
                    help="Write a PNG with detected cell bboxes drawn")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    pdf = v2_config.DATA_DIR / args.patent / f"{args.patent}.pdf"
    md = v2_config.DATA_DIR / args.patent / "all_pages" / f"page_{args.page:04d}.md"
    if not pdf.exists():
        print(f"PDF not found: {pdf}")
        return 2
    if not md.exists():
        print(f"PaddleX markdown not found: {md}")
        return 2

    page_img = render_pdf_page(pdf, args.page, dpi=args.dpi)
    md_text = md.read_text(errors="replace")

    # PaddleX renders pages at ~96 DPI internally; we render at 300.
    # Scale = 300/96 = 3.125. We could be smarter (probe md max coord)
    # but a simple ratio works because both use the same aspect ratio.
    scale = args.dpi / 96.0
    table_bboxes = _paddlex_table_bboxes_from_md(md_text, scale)
    if table_bboxes:
        print(f"Found {len(table_bboxes)} table region(s) in PaddleX md")
        for i, b in enumerate(table_bboxes):
            print(f"  table {i}: bbox_px={b}")
    else:
        print(f"No <|ref|>table<|/ref|> block in PaddleX md — running cell "
              f"detector on the whole page (whole-page fallback)")

    cells = detect_table_cells(
        page_img, table_bboxes,
        patent_id=args.patent, page=args.page,
        use_cache=not args.no_cache,
    )
    print(f"Detected {len(cells)} cells")
    by_table: dict[int, list[Cell]] = {}
    for c in cells:
        by_table.setdefault(c.table_id, []).append(c)
    for tid, cs in by_table.items():
        rows = max(c.row for c in cs) + 1 if cs else 0
        cols = max(c.col for c in cs) + 1 if cs else 0
        print(f"  table {tid}: {len(cs)} cells in {rows}x{cols} grid")

    if args.debug:
        from PIL import ImageDraw
        out_dir = v2_config.OUTPUT_DIR / "cropping_eval" / "_cell_detection"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.patent}_p{args.page:04d}_cells.png"
        img = page_img.copy()
        draw = ImageDraw.Draw(img)
        # Color by table_id for clarity; cycle through a small palette
        palette = ["red", "blue", "green", "purple", "orange"]
        for c in cells:
            color = palette[c.table_id % len(palette)]
            draw.rectangle(c.bbox_px, outline=color, width=3)
            draw.text(
                (c.bbox_px[0] + 4, c.bbox_px[1] + 4),
                f"r{c.row}c{c.col}", fill=color,
            )
        img.save(out_path)
        print(f"Wrote debug PNG: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
