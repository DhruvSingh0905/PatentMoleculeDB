"""Visual debug for the layout-graph label-association layer.

Source of truth = the page itself. This tool builds a CanonicalPage
(everything in render-pixel coords, asserted) and renders an overlay PNG
plus a JSON sidecar so you can hand-correct assignments and save them
as labelled training data for a future small layout model.

Layers in the overlay:
  - PaddleX table regions: thin DARK GRAY rectangles
  - Cells (PP-Structure):  GREEN, with `t{table_id}r{row}c{col}` text
  - Text candidates:       RED, with their compound-id text
  - Cell-anchored cands:   ORANGE
  - Structures:            BLUE (thick), with `s{idx}`
  - Assignment lines:      YELLOW from structure-center → assigned-candidate-center

The JSON sidecar contains everything in canonical coords, plus the
matcher's per-structure decision (top-3 candidates + reason). You can
edit the `correct_label` field for hand-correction; a future trainer
can read this directly.

Usage:
    python3 -m patentdb.scripts.eval.visualize_associations \\
        --patent US10899738 --page 28 --label-strategy layout_graph
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from patentdb.core import config as v2_config
from patentdb.routes import label_association as la
from patentdb.routes.decimer_segmentation_crop import (
    render_pdf_page, segment_page, SEG_DPI,
)
from patentdb.routes.page_geometry import (
    Bbox, CanonicalPage, build_canonical_page,
)
from patentdb.routes.page_regime import classify_page_regime
from patentdb.routes.table_cell_detection import (
    detect_table_cells, _paddlex_table_bboxes_from_md,
)
from patentdb.routes.text_detection import detect_text_blocks
from patentdb.scripts.eval.cropping_eval import (
    parse_paddlex_blocks, _paddlex_page_size,
)


_VIZ_DIR = v2_config.OUTPUT_DIR / "cropping_eval" / "_viz_associations"


def build_page(patent_id: str, page_num: int) -> CanonicalPage:
    """Run all upstream extractors and assemble a CanonicalPage."""
    pdf_path = v2_config.DATA_DIR / patent_id / f"{patent_id}.pdf"
    md_path = v2_config.DATA_DIR / patent_id / "all_pages" / f"page_{page_num:04d}.md"
    if not pdf_path.exists() or not md_path.exists():
        raise FileNotFoundError(
            f"missing inputs: pdf={pdf_path.exists()}, md={md_path.exists()}"
        )

    text = md_path.read_text(errors="replace")
    blocks = parse_paddlex_blocks(text)
    paddlex_w, paddlex_h = _paddlex_page_size(blocks)

    page_img = render_pdf_page(pdf_path, page_num, dpi=SEG_DPI)
    rendered_w, rendered_h = page_img.size

    # Upstream: structures (DECIMER-Seg) — render coords already
    seg_dir = (v2_config.OUTPUT_DIR / "cropping_eval" / "decimer_seg"
               / patent_id / f"page_{page_num:04d}")
    seg_crops = segment_page(pdf_path, page_num, seg_dir)
    structures_native = [
        (sc.bbox_px, str(sc.image_path), sc.crop_index) for sc in seg_crops
    ]

    # Upstream: PaddleX table regions (PaddleX coords)
    paddlex_table_regions_native = [b for b in _paddlex_table_bboxes_from_md(text, 1.0)]

    # Upstream: PP-Structure cells — render coords (we feed it the rendered image)
    paddlex_table_bboxes_render = _paddlex_table_bboxes_from_md(
        text, SEG_DPI / max(1, paddlex_w / paddlex_w),  # = identity for now
    )
    # Use the page-derived scale instead so it matches what build_canonical_page uses
    paddlex_to_render = rendered_w / max(1, paddlex_w)
    paddlex_table_bboxes_render = [
        (int(x1 * paddlex_to_render), int(y1 * paddlex_to_render),
         int(x2 * paddlex_to_render), int(y2 * paddlex_to_render))
        for (x1, y1, x2, y2) in paddlex_table_regions_native
    ]
    cells_objs = detect_table_cells(
        page_img, paddlex_table_bboxes_render,
        patent_id=patent_id, page=page_num,
    )
    cells_native = [
        (c.bbox_px, c.table_id, c.row, c.col, "ppstructure")
        for c in cells_objs
    ]

    # Upstream: PaddleX text-bearing blocks (PaddleX coords)
    LABEL_BLOCK_TYPES = {"text", "title", "table", "table_caption",
                         "figure_caption", "doc_title", "header"}
    text_blocks_native = [
        (b.bbox, b.content, b.type)
        for b in blocks if b.type in LABEL_BLOCK_TYPES
    ]

    # Upstream supplement: standalone PaddleOCR pass, catches small labels
    # PaddleX missed because they fell inside layout-classified "image"
    # regions. Output is in RENDER coords already (we feed it the rendered
    # page), so we tag block_type='text_detected' to signal "no PaddleX
    # scaling needed" downstream — build_canonical_page already handles
    # that via the canonical Bbox factories.
    detected_text_blocks = detect_text_blocks(
        page_img, patent_id=patent_id, page=page_num,
    )

    detected_text_native = [
        (b.bbox_px, b.content, "text_detected")
        for b in detected_text_blocks
    ]

    return build_canonical_page(
        patent_id=patent_id, page=page_num,
        page_width_px=rendered_w, page_height_px=rendered_h,
        paddlex_page_width_px=paddlex_w, paddlex_page_height_px=paddlex_h,
        structures_render_px=structures_native,
        cells_render_px=cells_native,
        text_blocks_paddlex_px=text_blocks_native,
        paddlex_table_regions_paddlex_px=paddlex_table_regions_native,
        text_blocks_render_px=detected_text_native,
    )


def _load_font(size: int = 22):
    """Best-effort large-ish font (PIL default is tiny on 300 DPI pages)."""
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, bbox: Bbox,
              color: str, label: str = "", width: int = 2,
              font=None, label_above: bool = False) -> None:
    """Draw a colored rectangle with an optional label.

    `label_above=True` places the label JUST ABOVE the bbox top-left
    instead of inside it. Used for cand:/cell: annotations that
    otherwise cover the page's actual compound-ID text and obstruct
    visual verification. Structures (S0/S1/...) keep the inside-the-box
    placement because the boxes are large and the text doesn't matter
    visually.
    """
    draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                   outline=color, width=width)
    if label:
        # Slim white pad behind the text so it stays readable over the page
        if label_above:
            # Sit the label clear of the box top edge so it doesn't
            # cover the text the box wraps. ~font height + small gap.
            text_xy = (bbox.x1 + 4, max(0, bbox.y1 - 42))
        else:
            text_xy = (bbox.x1 + 4, bbox.y1 + 4)
        try:
            tx, ty, tx2, ty2 = draw.textbbox(text_xy, label, font=font)
            draw.rectangle((tx - 2, ty - 1, tx2 + 2, ty2 + 1), fill="white")
        except Exception:
            pass
        draw.text(text_xy, label, fill=color, font=font)


def _draw_legend(draw: ImageDraw.ImageDraw, page_width_px: int, font) -> None:
    """In-image color legend at top-right so a fresh viewer doesn't have
    to remember what each color means. Sized for 300-DPI page."""
    rows = [
        ("#888888", "paddlex-table-N   PaddleX <|ref|>table<|/ref|> bbox"),
        ("#00aa00", "tT.rR.cC          PP-Structure cell + table/row/col index"),
        ("#cc0000", "cand:X            text-block candidate (mined from PaddleX text)"),
        ("#ff8800", "cell:X            cell-anchored candidate (from inline <table><td>)"),
        ("#0066ff", "Si                DECIMER-Seg structure i"),
        ("#ffcc00", "Si \u2192 X            matcher: assigned label X to structure Si"),
    ]
    pad = 24
    line_h = 56
    box_w = 1700
    swatch_w = 80
    swatch_h = 40
    box_h = pad * 2 + line_h * len(rows)
    x0 = page_width_px - box_w - 60
    y0 = 60
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h),
                   outline="black", fill="white", width=4)
    for i, (color, text) in enumerate(rows):
        y = y0 + pad + i * line_h
        draw.rectangle((x0 + pad, y + 6, x0 + pad + swatch_w, y + 6 + swatch_h),
                       outline=color, fill=color, width=2)
        draw.text((x0 + pad + swatch_w + 24, y), text, fill="black", font=font)


def visualize(
    page: CanonicalPage,
    provenances: list[la.AssociationProvenance],
    text_candidates_by_text: dict[str, Bbox],
    cell_candidates_by_text: dict[str, Bbox],
    label_strategy: str,
) -> Path:
    """Render the overlay PNG with every box labeled + legend."""
    pdf_path = v2_config.DATA_DIR / page.patent_id / f"{page.patent_id}.pdf"
    img = render_pdf_page(pdf_path, page.page, dpi=SEG_DPI).copy()
    draw = ImageDraw.Draw(img)

    # 300-DPI page = 2550x3300 px. Fonts at 20px are unreadable; bump way up.
    box_font = _load_font(36)
    line_font = _load_font(48)
    legend_font = _load_font(34)

    # PaddleX table regions — thin gray
    for ti, tr in enumerate(page.paddlex_table_regions):
        _draw_box(draw, tr, "#888888", f"paddlex-table-{ti}", width=2, font=box_font)

    # Cells — green
    for c in page.cells:
        _draw_box(draw, c.bbox, "#00aa00",
                  f"t{c.table_id}.r{c.row}.c{c.col}",
                  width=2, font=box_font)

    # Candidates: text-source = red, cell-anchored = orange (so the user
    # can tell at a glance whether this candidate came from PaddleX text
    # or was synthesized from cell-detector + inline <table>).
    for text, bbox in text_candidates_by_text.items():
        _draw_box(draw, bbox, "#cc0000", f"cand:{text}", width=2, font=box_font,
                  label_above=True)
    for text, bbox in cell_candidates_by_text.items():
        _draw_box(draw, bbox, "#ff8800", f"cell:{text}", width=2, font=box_font,
                  label_above=True)

    # Structures — blue, thick
    for i, s in enumerate(page.structures):
        _draw_box(draw, s.bbox, "#0066ff", f"S{i}", width=5, font=box_font)

    # Assignment lines — yellow, with mid-line "Si→X" label so the user
    # can read the assignment without zooming.
    for i, prov in enumerate(provenances):
        if prov.chosen_label is None:
            continue
        # Look up the candidate bbox in either pool
        target_bbox = (cell_candidates_by_text.get(prov.chosen_label.text)
                       or text_candidates_by_text.get(prov.chosen_label.text))
        if target_bbox is None:
            continue
        sb = page.structures[i].bbox
        x1, y1 = int(sb.cx), int(sb.cy)
        x2, y2 = int(target_bbox.cx), int(target_bbox.cy)
        draw.line([(x1, y1), (x2, y2)], fill="#ffcc00", width=5)
        # Mid-line label
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        line_label = f"S{i}\u2192{prov.chosen_label.text}"
        try:
            tx, ty, tx2, ty2 = draw.textbbox((mx, my), line_label, font=line_font)
            draw.rectangle((tx - 4, ty - 2, tx2 + 4, ty2 + 2),
                           fill="white", outline="#ffcc00", width=2)
        except Exception:
            pass
        draw.text((mx, my), line_label, fill="#cc8800", font=line_font)

    # Legend last so it's on top
    _draw_legend(draw, page.width_px, legend_font)

    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _VIZ_DIR / f"{page.patent_id}_p{page.page:04d}_{label_strategy}.png"
    img.save(out_path)
    # Half-size companion so the Read tool renders it readably without
    # zooming. Original PNG retains full 300 DPI for pixel-perfect inspection.
    half = img.resize((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
    half_path = _VIZ_DIR / f"{page.patent_id}_p{page.page:04d}_{label_strategy}_small.png"
    half.save(half_path)
    return out_path


def write_sidecar(page: CanonicalPage, provenances: list[la.AssociationProvenance],
                  candidates_by_text: dict[str, Bbox], label_strategy: str,
                  regime: str) -> Path:
    """Write a JSON sidecar that captures everything visible in the overlay,
    in canonical coords, with editable `correct_label` slots for hand-correction.

    This is the file you mark up. Keep it in version control if you want to
    use it as training data for a future small layout model.
    """
    payload = {
        "_meta": {
            "patent_id": page.patent_id,
            "page": page.page,
            "page_size_px": [page.width_px, page.height_px],
            "paddlex_page_size_px": list(page.paddlex_page_size_px),
            "label_strategy": label_strategy,
            "regime": regime,
            "coord_system": "render-pixel, top-left origin, 300 DPI",
            "instructions": (
                "For each `structures[i]` set `correct_label` to the compound "
                "ID the structure ACTUALLY shows on the page. Leave None if "
                "the crop is not a target compound (synthesis intermediate, "
                "scheme by-product, header artifact, etc.). The label-association "
                "trainer reads this file as-is."
            ),
        },
        "paddlex_table_regions": [b.as_tuple() for b in page.paddlex_table_regions],
        "cells": [
            {**asdict(c), "bbox": c.bbox.as_tuple()}
            for c in page.cells
        ],
        "text_blocks": [
            {**asdict(t), "bbox": t.bbox.as_tuple()}
            for t in page.text_blocks
        ],
        "candidates": [
            {"text": text, "bbox": bbox.as_tuple()}
            for text, bbox in candidates_by_text.items()
        ],
        "structures": [
            {
                "idx": i,
                "bbox": s.bbox.as_tuple(),
                "crop_index": s.crop_index,
                "crop_path": s.crop_path,
                "matcher_decision": {
                    "assigned_label": (prov.chosen_label.text
                                       if prov.chosen_label else None),
                    "confidence": (prov.chosen_label.confidence
                                   if prov.chosen_label else None),
                    "reason": prov.reason,
                    "scorer_used": prov.scorer_used,
                    "top_3": prov.top_candidates[:3],
                },
                # Editable: set to the true compound id (or null) when reviewing.
                "correct_label": None,
            }
            for i, (s, prov) in enumerate(zip(page.structures, provenances))
        ],
    }
    out_path = (_VIZ_DIR / f"{page.patent_id}_p{page.page:04d}_{label_strategy}.json")
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--label-strategy", default="layout_graph",
                    choices=["cell_iou", "proximity_direction", "layout_graph"])
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    page = build_page(args.patent, args.page)
    print(page.summary())

    # Build candidates the matcher will actually use
    text_blocks_for_la = [
        la.TextBlock(b.bbox.as_tuple(), b.content, b.block_type)
        for b in page.text_blocks
    ]
    paddlex_table_regions_for_la = [
        # mine_label_candidates expects PADDLEX coords for table regions; we
        # converted them to canonical, so scale back for that legacy API.
        # FIXME: a future iteration moves mine_label_candidates over to
        # canonical-only and removes this back-conversion.
        (b.x1, b.y1, b.x2, b.y2) for b in page.paddlex_table_regions
    ]
    cells_for_la = [
        la.Cell(c.bbox.as_tuple(), c.table_id, c.row, c.col)
        for c in page.cells
    ]
    structures_for_la = [
        la.StructureBox(s.bbox.as_tuple(), s.crop_path) for s in page.structures
    ]

    regime, _ = classify_page_regime(
        paddlex_text="",   # not needed when we already have parsed blocks
        text_blocks=[(b.bbox.as_tuple(), b.content) for b in page.text_blocks],
        cells=[c.bbox.as_tuple() for c in page.cells],
        structure_bboxes=[s.bbox.as_tuple() for s in page.structures],
        paddlex_table_bboxes=[b.as_tuple() for b in page.paddlex_table_regions],
    )
    # Re-derive regime using the original markdown text (the classifier needs
    # the raw text for "Cpd. No." / "Scheme N" markers). The empty call above
    # is a fallback; the visualizer always has the markdown handy.
    md_path = v2_config.DATA_DIR / args.patent / "all_pages" / f"page_{args.page:04d}.md"
    md_text = md_path.read_text(errors="replace")
    regime, signals = classify_page_regime(
        paddlex_text=md_text,
        text_blocks=[(b.bbox.as_tuple(), b.content) for b in page.text_blocks],
        cells=[c.bbox.as_tuple() for c in page.cells],
        structure_bboxes=[s.bbox.as_tuple() for s in page.structures],
        paddlex_table_bboxes=[b.as_tuple() for b in page.paddlex_table_regions],
    )
    print(f"regime={regime.value}")

    # IMPORTANT: text_blocks_for_la are in RENDER coords (build_canonical_page
    # canonicalized everything). Tell associate_labels both page sizes are
    # the render size so its internal paddlex_to_render_scale = 1.0
    # (no double-scaling).
    provenances = la.associate_labels(
        structures=structures_for_la,
        text_blocks=text_blocks_for_la,
        cells=cells_for_la,
        page_size_px=(page.width_px, page.height_px),
        paddlex_page_size_px=(page.width_px, page.height_px),
        strategy=args.label_strategy,
        regime=regime.value,
        paddlex_table_region_bboxes=paddlex_table_regions_for_la,
    )

    # Build SEPARATE {text -> canonical bbox} dicts for each candidate
    # source so the visualizer can color them differently.
    text_cands = la.mine_label_candidates(
        text_blocks_for_la, table_region_bboxes=paddlex_table_regions_for_la,
        page_width_px=page.width_px,
    )
    cell_anchored_cands = la.cell_anchored_table_candidates(
        cells_for_la, text_blocks_for_la,
    )
    # Candidates inherit the coord space of the text blocks they were
    # mined from. After build_canonical_page, our text_blocks_for_la are
    # in RENDER coords already, so all candidates are in render coords too
    # — no rescaling needed (and any rescaling here is a double-scale bug).
    text_candidates_by_text: dict[str, Bbox] = {}
    cell_candidates_by_text: dict[str, Bbox] = {}
    for cn in text_cands:
        text_candidates_by_text[cn.text] = Bbox.from_render_px(*cn.bbox_px)
    for cn in cell_anchored_cands:
        cell_candidates_by_text[cn.text] = Bbox.from_render_px(*cn.bbox_px)

    # Combined dict for the sidecar (single source of truth for the JSON)
    combined = {**text_candidates_by_text, **cell_candidates_by_text}

    out_png = visualize(page, provenances,
                        text_candidates_by_text=text_candidates_by_text,
                        cell_candidates_by_text=cell_candidates_by_text,
                        label_strategy=args.label_strategy)
    out_json = write_sidecar(page, provenances, combined,
                              args.label_strategy, regime.value)
    print(f"\nwrote overlay PNG:    {out_png}")
    print(f"wrote sidecar JSON:   {out_json}")

    print(f"\n{'idx':>4} {'bbox':<28} {'assigned':<14} {'reason':<18} {'top_3':<60}")
    for i, prov in enumerate(provenances):
        bbox_str = str(page.structures[i].bbox.as_tuple())
        assigned = prov.chosen_label.text if prov.chosen_label else "(none)"
        top = ", ".join(f"{t}={s:.2f}" for t, s in prov.top_candidates[:3])
        print(f"{i:>4} {bbox_str:<28} {assigned:<14} {prov.reason:<18} {top:<60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
