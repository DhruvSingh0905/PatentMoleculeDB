"""Head-to-head cropping-strategy evaluation.

For each (patent, page, compound_id, expected_inchikey_connectivity) in
`ground_truth_structures.json`:

  1. Run a cropping strategy on the page -> list of (compound_id, crop_path)
  2. Find the crop assigned to compound_id (or report "missing")
  3. Run DECIMER on the crop -> SMILES
  4. Compute InChIKey from SMILES -> compare connectivity (first 14 chars)

Strategies (selectable via --strategy, multiple allowed):
    geometric      — patentdb.routes.structure_crop (current v2)
    decimer_seg    — patentdb.routes.decimer_segmentation_crop
    sonnet_layout  — v1 patent_extraction.image_pipeline._extract_table_layout

Per-strategy report columns:
    patent       — ground-truth patent
    page         — page number
    n_gt         — ground-truth compounds on this page
    n_crops      — crops produced by the strategy
    n_decimer_ok — crops where DECIMER returned a valid SMILES
    n_match      — crops whose SMILES matched expected_inchikey_connectivity

Mandatory step before scoring:
    --visual-sample N   — print N random crop paths per strategy.
                          The user MUST manually `Read` these images and
                          confirm each is one complete molecule with the
                          structure not bleeding into adjacent compounds
                          (per Phase A.2 of the rebuild plan). Pixel-density
                          heuristics are not a substitute for vision.

Usage:
    python3 -m patentdb.scripts.eval.cropping_eval \\
        --strategy decimer_seg --visual-sample 5
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

# v2 package imports
from patentdb.core import config as v2_config
from patentdb.routes.structure_crop import (
    extract_compound_ids_from_paddlex,
    geometric_crop_table,
    render_pdf_page,
)
from patentdb.routes.decimer_segmentation_crop import (
    segment_page as decimer_seg_page,
    SegCrop,
)

logger = logging.getLogger("cropping_eval")

GT_PATH = Path(__file__).parent / "ground_truth_structures.json"
# All eval crops live under output_v2/cropping_eval/{strategy}/{patent}/page_XXXX/.
# Consolidated location so users always know where to look.
EVAL_OUT_ROOT = v2_config.OUTPUT_DIR / "cropping_eval"
EVAL_OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ── Crop type unifying every strategy's output ────────────────────────

@dataclass
class StrategyCrop:
    """Output of one cropping strategy for one compound on one page."""
    patent_id: str
    page: int
    assigned_compound_id: str       # what the strategy thinks this crop is
    crop_path: Path
    bbox_px: tuple[int, int, int, int] | None = None
    notes: str = ""


# ── Strategy 1: geometric ─────────────────────────────────────────────

def _md_path(patent_id: str, page: int) -> Path:
    return v2_config.DATA_DIR / patent_id / "all_pages" / f"page_{page:04d}.md"


def _pdf_path(patent_id: str) -> Path:
    return v2_config.DATA_DIR / patent_id / f"{patent_id}.pdf"


# Used by structure-page cropping. Different from extract_compound_ids_from_paddlex
# (which targets <table> rows in assay tables).
_HEADING_ON_PAGE = re.compile(
    r"(?:Example|Cpd\.?\s*No\.?|Compound)\s+([0-9]+[A-Za-z]{0,3})\b",
    re.IGNORECASE,
)

# PaddleX block annotations: <|ref|>type<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|> content
# `type` is one of: title, text, image, table (and a few rarer tags)
_PADDLEX_BLOCK = re.compile(
    r"<\|ref\|>(?P<type>\w+)<\|/ref\|><\|det\|>"
    r"\[\[\s*(?P<x1>\d+)\s*,\s*(?P<y1>\d+)\s*,\s*(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\s*\]\]"
    r"<\|/det\|>(?P<content>.*?)(?=<\|ref\||\Z)",
    re.DOTALL,
)


@dataclass
class PaddleBlock:
    """One PaddleX OCR block on a page."""
    type: str                     # 'title' | 'text' | 'image' | 'table' | etc
    bbox: tuple[int, int, int, int]
    content: str                  # cleaned text (whitespace-collapsed)


def parse_paddlex_blocks(text: str) -> list[PaddleBlock]:
    """Parse a PaddleX markdown page into ordered blocks with bbox metadata."""
    blocks: list[PaddleBlock] = []
    for m in _PADDLEX_BLOCK.finditer(text):
        blocks.append(PaddleBlock(
            type=m.group("type"),
            bbox=(int(m.group("x1")), int(m.group("y1")),
                  int(m.group("x2")), int(m.group("y2"))),
            content=re.sub(r"\s+", " ", m.group("content")).strip(),
        ))
    return blocks


def find_compound_id_mentions(blocks: list[PaddleBlock]) -> list[tuple[tuple[int, int, int, int], str]]:
    """Extract every (bbox, compound_id) where a compound label appears in
    PaddleX text/title blocks.

    Patents print "Cpd. No 350" / "Example 564" / "Compound 5A" as a label
    ABOVE the corresponding structure drawing — usually within a few mm.
    By matching each DECIMER-Seg crop to the nearest such label above it
    (and on the same horizontal column), we get the right compound ID even
    when (a) PaddleX miscounts image blocks, (b) synthesis schemes get
    detected as additional crops, or (c) the page has multiple compounds
    laid out side-by-side.

    Returns a list of (bbox, compound_id) preserving the original block
    order. The same compound_id may appear multiple times (e.g. label +
    NMR section) — the matcher will pick the closest one to each crop.
    """
    out: list[tuple[tuple[int, int, int, int], str]] = []
    for b in blocks:
        if b.type not in ("text", "title"):
            continue
        for m in _HEADING_ON_PAGE.finditer(b.content):
            out.append((b.bbox, m.group(1).strip()))
    return out


def assign_compound_ids_by_paddlex_position(
    crops: list[SegCrop],
    id_mentions: list[tuple[tuple[int, int, int, int], str]],
    page_height_px: int,
    paddlex_page_height_px: int,
    paddlex_page_width_px: int,
    page_width_px: int,
    max_distance_pct: float = 0.30,
) -> list[tuple[SegCrop, str | None]]:
    """For each DECIMER-Seg crop, attach the compound ID whose PaddleX text
    mention is the closest by 2D normalized distance.

    Why 2D-nearest (no above/same-column gating):
      Patent layouts vary widely. A compound label may appear above the
      structure (typical), below the structure (NMR section comes first),
      or on the side (multi-column layout). The simplest robust signal
      is "the closest text mention of a compound ID is the right one."
      We rely on patents not putting two unrelated compound IDs at the
      same physical position, which holds in practice.

    Coordinate spaces differ (PaddleX ~100 DPI, we render at 300 DPI);
    we normalize both to fractional page coordinates before comparing.

    Crops with no mention within `max_distance_pct` get None — likely
    synthesis intermediates or DECIMER-Seg false positives.
    """
    if not id_mentions:
        return [(c, None) for c in crops]

    pairs: list[tuple[SegCrop, str | None]] = []
    for crop in crops:
        x1, y1, x2, y2 = crop.bbox_px
        cx = ((x1 + x2) / 2) / max(1, page_width_px)
        cy = ((y1 + y2) / 2) / max(1, page_height_px)

        best_cid: str | None = None
        best_dist = float("inf")
        for (mx1, my1, mx2, my2), cid in id_mentions:
            mx = ((mx1 + mx2) / 2) / max(1, paddlex_page_width_px)
            my = ((my1 + my2) / 2) / max(1, paddlex_page_height_px)
            dx = cx - mx
            dy = cy - my
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_dist:
                best_dist = d
                best_cid = cid
        if best_dist > max_distance_pct:
            best_cid = None
        pairs.append((crop, best_cid))
    return pairs


def _paddlex_page_size(blocks: list[PaddleBlock]) -> tuple[int, int]:
    """Estimate PaddleX page (width_px, height_px) from max bbox extents."""
    if not blocks:
        return (850, 1100)     # US Letter @ ~100 DPI default
    return (
        max(b.bbox[2] for b in blocks) + 50,
        max(b.bbox[3] for b in blocks) + 50,
    )


def extract_structure_page_compound_ids(text: str) -> list[str]:
    """Backward-compatible: return compound IDs in document order from headings."""
    ids: list[str] = []
    for m in _HEADING_ON_PAGE.finditer(text):
        cid = m.group(1).strip()
        if cid not in ids:
            ids.append(cid)
    return ids


def _compound_ids_for_page(patent_id: str, page: int) -> list[str]:
    md = _md_path(patent_id, page)
    if not md.exists():
        return []
    text = md.read_text(errors="replace")
    table_ids = extract_compound_ids_from_paddlex(text)
    if table_ids:
        return table_ids
    return extract_structure_page_compound_ids(text)


def _mineru_table_bbox(patent_id: str, page: int) -> tuple[float, float, float, float] | None:
    """Pull the first table bbox (PDF coords) from MinerU's middle.json
    if available. Returns None if MinerU output isn't present for this page.
    """
    candidates = [
        v2_config.REPO_ROOT / "mineru_output" / f"{patent_id}_p{page:04d}" / "auto" / f"{patent_id}_p{page:04d}_middle.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for pdf_info in data.get("pdf_info", []):
            for block in pdf_info.get("preproc_blocks", []):
                if block.get("type") == "table":
                    bbox = block.get("bbox")
                    if bbox and len(bbox) == 4:
                        return tuple(bbox)
    return None


def run_geometric(patent_id: str, page: int, gt_compound_ids: list[str],
                  out_dir: Path) -> tuple[list[StrategyCrop], str]:
    """Run the v2 geometric cropper on one page.

    Requires a table bbox from MinerU. If unavailable, returns an empty
    list with a SKIPPED note — that itself is a failure mode worth
    surfacing in the report (geometric isn't generalizable without
    upstream table-layout info on every page).
    """
    paddle_ids = _compound_ids_for_page(patent_id, page)
    if not paddle_ids:
        return [], "PaddleX yielded 0 compound IDs (no table rows or Example/Cpd headings)"

    table_bbox = _mineru_table_bbox(patent_id, page)
    if table_bbox is None:
        # Fallback: use ~80% of page area centered. This is a deliberately
        # weak fallback — geometric needs real upstream bboxes and we want
        # the report to show that it can't cover all pages.
        return [], "no MinerU table bbox available"

    crops = geometric_crop_table(
        pdf_path=_pdf_path(patent_id),
        page_num=page,
        table_bbox_pdf=table_bbox,
        compound_ids=paddle_ids,
        output_dir=out_dir,
    )
    out: list[StrategyCrop] = []
    for c in crops:
        out.append(StrategyCrop(
            patent_id=patent_id, page=page,
            assigned_compound_id=c.compound_id,
            crop_path=c.image_path,
            bbox_px=c.bbox_px,
            notes=c.notes,
        ))
    return out, ""


# ── Strategy 2: decimer_seg ───────────────────────────────────────────

def run_decimer_seg(patent_id: str, page: int, gt_compound_ids: list[str],
                    out_dir: Path,
                    label_strategy: str = "legacy",
                    patent_gt_id_set: set[str] | None = None,
                    ) -> tuple[list[StrategyCrop], str]:
    """DECIMER-Seg cropper, with pluggable labeling strategy.

    label_strategy:
      legacy              — original 2D-nearest text-mention (baseline,
                            kept for reproducible head-to-head comparisons)
      cell_iou            — Step 4a: structure ↔ cell IoU matching
      proximity_direction — Step 4b: proximity + direction scoring
      layout_graph        — Step 4c: regime-routed combination
    """
    md = _md_path(patent_id, page)
    if not md.exists():
        return [], f"no PaddleX md at {md}"
    text = md.read_text(errors="replace")

    seg_crops = decimer_seg_page(_pdf_path(patent_id), page, out_dir)
    if not seg_crops:
        return [], "DECIMER-Seg detected 0 molecules"

    blocks = parse_paddlex_blocks(text)
    paddlex_w, paddlex_h = _paddlex_page_size(blocks)
    page_img = render_pdf_page(_pdf_path(patent_id), page, dpi=300)
    rendered_w, rendered_h = page_img.size

    if label_strategy == "legacy":
        # Original 2D-nearest text-mention
        id_mentions = find_compound_id_mentions(blocks)
        pairs = assign_compound_ids_by_paddlex_position(
            seg_crops, id_mentions,
            page_height_px=rendered_h,
            paddlex_page_height_px=paddlex_h,
            paddlex_page_width_px=paddlex_w,
            page_width_px=rendered_w,
        )
        out: list[StrategyCrop] = []
        for crop, cid in pairs:
            assigned = cid if cid else f"unknown_p{page}_r{crop.crop_index}"
            notes = "" if cid else "no nearby PaddleX text mention (legacy 2d-nearest)"
            out.append(StrategyCrop(
                patent_id=patent_id, page=page,
                assigned_compound_id=assigned,
                crop_path=crop.image_path,
                bbox_px=crop.bbox_px,
                notes=notes,
            ))
        return out, ""

    # New path: layout-graph matcher (Step 4)
    return _label_via_layout_graph(
        seg_crops, blocks, text, paddlex_w, paddlex_h, rendered_w, rendered_h,
        patent_id, page, page_img, label_strategy, patent_gt_id_set,
    )


def _label_via_layout_graph(
    seg_crops, blocks, text, paddlex_w, paddlex_h, rendered_w, rendered_h,
    patent_id, page, page_img, label_strategy, patent_gt_id_set,
) -> tuple[list[StrategyCrop], str]:
    """Run the new layout_graph matcher against existing DECIMER-Seg crops."""
    from patentdb.routes import label_association as la
    from patentdb.routes.page_regime import (
        classify_page_regime, PageRegime,
    )
    from patentdb.routes.table_cell_detection import (
        detect_table_cells, _paddlex_table_bboxes_from_md,
    )
    from patentdb.routes.text_detection import detect_text_blocks

    # Convert seg crops to the StructureBox dataclass the matcher expects
    structures = [la.StructureBox(bbox_px=c.bbox_px, crop_path=str(c.image_path))
                  for c in seg_crops]
    # PaddleX text blocks (paddlex coords). Include all text-bearing types.
    _LABEL_BLOCK_TYPES = {"text", "title", "table", "table_caption",
                          "figure_caption", "doc_title", "header"}
    paddlex_text_blocks = [la.TextBlock(bbox_px=b.bbox, content=b.content,
                                         block_type=b.type)
                            for b in blocks if b.type in _LABEL_BLOCK_TYPES]
    # PaddleOCR standalone text-detection pass — catches small standalone
    # labels PaddleX missed because they fell inside layout-classified
    # "image" regions. Output is in RENDER coords (we feed it the same
    # rendered page as DECIMER-Seg). We append these to the candidate pool
    # AFTER converting paddlex blocks to render coords (so all text blocks
    # share the render coord system).
    paddlex_to_render = rendered_w / max(1, paddlex_w)
    text_blocks = []
    for tb in paddlex_text_blocks:
        x1, y1, x2, y2 = tb.bbox_px
        text_blocks.append(la.TextBlock(
            bbox_px=(int(x1 * paddlex_to_render), int(y1 * paddlex_to_render),
                     int(x2 * paddlex_to_render), int(y2 * paddlex_to_render)),
            content=tb.content,
            block_type=tb.block_type,
        ))
    detected = detect_text_blocks(page_img, patent_id=patent_id, page=page)
    for d in detected:
        text_blocks.append(la.TextBlock(
            bbox_px=d.bbox_px, content=d.content, block_type="text_detected",
        ))

    # Get cells (Step 3) — only for table-ish regimes; we run detection always
    # so the regime classifier can use cell counts.
    paddlex_to_render_scale = rendered_w / max(1, paddlex_w)
    paddlex_table_bboxes = _paddlex_table_bboxes_from_md(text, paddlex_to_render_scale)
    # Same bboxes in PADDLEX coords (not render coords) — used by
    # mine_label_candidates to place row-derived candidates inside the
    # actual table region.
    paddlex_table_bboxes_paddlex_coords = _paddlex_table_bboxes_from_md(text, 1.0)
    cells_objs = detect_table_cells(
        page_img, paddlex_table_bboxes,
        patent_id=patent_id, page=page,
    )
    cells = [la.Cell(bbox_px=c.bbox_px, table_id=c.table_id, row=c.row, col=c.col)
             for c in cells_objs]
    structure_bboxes = [s.bbox_px for s in structures]

    regime, _signals = classify_page_regime(
        paddlex_text=text,
        text_blocks=[(b.bbox, b.content) for b in blocks if b.type in ("text", "title")],
        cells=[c.bbox_px for c in cells_objs],
        structure_bboxes=structure_bboxes,
        paddlex_table_bboxes=paddlex_table_bboxes,
    )

    # text_blocks are pre-canonicalized to render coords above. Tell
    # associate_labels both page sizes are the render size so its
    # internal paddlex_to_render_scale = 1.0 (no double-scaling).
    provenances = la.associate_labels(
        structures=structures,
        text_blocks=text_blocks,
        cells=cells,
        page_size_px=(rendered_w, rendered_h),
        paddlex_page_size_px=(rendered_w, rendered_h),
        strategy=label_strategy,
        regime=regime.value,
        patent_gt_id_set=patent_gt_id_set,
        paddlex_table_region_bboxes=paddlex_table_bboxes,  # render coords
    )

    out: list[StrategyCrop] = []
    for crop, prov in zip(seg_crops, provenances):
        if prov.chosen_label is not None:
            assigned = prov.chosen_label.text
            notes = (f"{prov.scorer_used} conf={prov.chosen_label.confidence:.2f} "
                     f"regime={regime.value}")
        else:
            assigned = f"unknown_p{page}_r{crop.crop_index}"
            notes = (f"{prov.scorer_used} reason={prov.reason} "
                     f"regime={regime.value} "
                     f"top={prov.top_candidates[:2]}")
        out.append(StrategyCrop(
            patent_id=patent_id, page=page,
            assigned_compound_id=assigned,
            crop_path=crop.image_path,
            bbox_px=crop.bbox_px,
            notes=notes,
        ))
    return out, ""


# ── Strategy 3: sonnet_layout ─────────────────────────────────────────

def run_sonnet_layout(patent_id: str, page: int, gt_compound_ids: list[str],
                      out_dir: Path) -> tuple[list[StrategyCrop], str]:
    """Use v1's Sonnet-Vision LAYOUT_PROMPT on the rendered page."""
    try:
        from patent_extraction.image_pipeline import (
            _extract_table_layout, _crop_structure,
        )
    except Exception as e:
        return [], f"v1 image_pipeline import failed: {e}"

    pdf = _pdf_path(patent_id)
    if not pdf.exists():
        return [], f"no PDF at {pdf}"

    out_dir.mkdir(parents=True, exist_ok=True)
    page_image_path = out_dir / f"page_{page:04d}_full.png"
    page_img = render_pdf_page(pdf, page, dpi=300)
    page_img.save(page_image_path)

    layouts = _extract_table_layout(str(page_image_path), patent_id, page)
    if not layouts:
        return [], "Sonnet LAYOUT_PROMPT returned 0 entries"

    out: list[StrategyCrop] = []
    for i, layout in enumerate(layouts):
        bbox_pct = layout.get("structure_bbox") or layout.get("bbox")
        cpd_no = str(layout.get("cpd_no") or layout.get("id") or f"unknown_p{page}_r{i}")
        if not bbox_pct:
            continue
        crop = _crop_structure(page_img, bbox_pct)
        if crop is None:
            continue
        crop_path = out_dir / f"sonnet_p{page:04d}_{i:02d}_{cpd_no}.png"
        crop.save(crop_path)
        # bbox_pct is [x1%, y1%, x2%, y2%]; convert to px for record
        pw, ph = page_img.size
        bbox_px = (
            int(bbox_pct[0] / 100 * pw), int(bbox_pct[1] / 100 * ph),
            int(bbox_pct[2] / 100 * pw), int(bbox_pct[3] / 100 * ph),
        )
        out.append(StrategyCrop(
            patent_id=patent_id, page=page,
            assigned_compound_id=cpd_no,
            crop_path=crop_path,
            bbox_px=bbox_px,
        ))
    return out, ""


STRATEGIES = {
    "geometric":    run_geometric,
    "decimer_seg":  run_decimer_seg,
    "sonnet_layout": run_sonnet_layout,
}


# ── DECIMER + scoring ────────────────────────────────────────────────

def _decimer_smiles(crop_path: Path) -> str | None:
    if not crop_path.exists():
        return None
    try:
        from DECIMER import predict_SMILES
        raw = predict_SMILES(str(crop_path))
    except Exception as e:
        logger.warning("DECIMER failed on %s: %s", crop_path, e)
        return None
    if not raw:
        return None
    smi = raw.split(".")[0].strip()
    return smi or None


def _connectivity_inchikey(smiles: str) -> str | None:
    """First 14 chars of full-stereo InChIKey for the given SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        ikey = MolToInchiKey(mol)
        return ikey[:14] if ikey and len(ikey) >= 14 else None
    except Exception:
        return None


def _flat_connectivity_inchikey(smiles: str) -> str | None:
    """First 14 chars of stereo-stripped InChIKey.

    DECIMER often loses stereo on small wedges even when connectivity is
    correct. The bench scores against this so connectivity-correct +
    stereo-loss is `OK`, not `DECIMER_MISS`.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        flat = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
        if flat is None:
            return None
        ikey = MolToInchiKey(flat)
        return ikey[:14] if ikey and len(ikey) >= 14 else None
    except Exception:
        return None


# ── Bucket classification (Step 2) ───────────────────────────────────

def classify_bucket(
    assigned_id: str,
    decimer_smiles: str | None,
    predicted_full_ikey14: str | None,
    predicted_flat_ikey14: str | None,
    expected_full_ikey14: str | None,
    expected_flat_ikey14: str | None,
    patent_gt_flat_ikey14_set: set[str],
) -> str:
    """Bucket each crop into one of the diagnostic categories.

    The flat InChIKey is used ONLY to determine if the LABEL is right
    (i.e., is this crop the molecule we think it is, ignoring stereo?).
    The full InChIKey separates `OK` (label + stereo correct) from
    `OK_STEREO_DIFF` (label correct, connectivity correct, stereo lost).

    NOTE: stereo is a real product requirement (Jie: enantiomers stay as
    separate entries with different binding data). `OK_STEREO_DIFF` is a
    DEFERRED problem to address in a later phase, not a permanent bucket.

    Buckets (per Step 2 of plan):
      OK                       — full_match_assigned_id (label + stereo correct)
      OK_STEREO_DIFF           — label correct + connectivity correct, stereo lost
                                 (DECIMER didn't reproduce wedges)
      LABEL_WRONG_DECIMER_OK   — flat doesn't match assigned_id, but flat matches
                                 SOME GT compound from this patent → labeling bug
      DECIMER_MISS             — assigned_id is in GT, DECIMER returned valid
                                 SMILES, but flat doesn't match any patent GT
                                 → real DECIMER recognition failure
      LABEL_NONE               — `unknown_*` AND flat doesn't match any patent GT
      LABEL_NONE_DECIMER_OK    — unlabeled crop but flat matches a patent GT
                                 compound (special case of LABEL_WRONG_DECIMER_OK)
      NOT_IN_GT                — labeled, label not in GT, flat doesn't match any
                                 patent GT compound (likely a real compound we
                                 just don't have ground truth for)
    """
    is_unlabeled = assigned_id.startswith("unknown_")
    decimer_in_patent_gt_flat = bool(
        predicted_flat_ikey14 and predicted_flat_ikey14 in patent_gt_flat_ikey14_set
    )
    in_gt = expected_flat_ikey14 is not None
    flat_match_assigned = bool(
        predicted_flat_ikey14 and expected_flat_ikey14
        and predicted_flat_ikey14 == expected_flat_ikey14
    )
    full_match_assigned = bool(
        predicted_full_ikey14 and expected_full_ikey14
        and predicted_full_ikey14 == expected_full_ikey14
    )

    if full_match_assigned:
        return "OK"
    if flat_match_assigned:
        # Label was right and connectivity was right; stereo differs.
        return "OK_STEREO_DIFF"
    if is_unlabeled:
        return "LABEL_NONE_DECIMER_OK" if decimer_in_patent_gt_flat else "LABEL_NONE"
    if decimer_in_patent_gt_flat:
        return "LABEL_WRONG_DECIMER_OK"
    if in_gt:
        return "DECIMER_MISS"
    return "NOT_IN_GT"


# ── Eval driver ──────────────────────────────────────────────────────

def load_ground_truth() -> list[dict]:
    if not GT_PATH.exists():
        sys.exit(f"Run build_cropping_ground_truth.py first ({GT_PATH} missing)")
    return json.loads(GT_PATH.read_text())["structures"]


def _norm_id(cid: str) -> str:
    return re.sub(r"\s+", "", str(cid)).upper()


def evaluate_strategy(strategy: str, gt: list[dict],
                      visual_sample: int,
                      label_strategy: str = "legacy") -> dict:
    runner = STRATEGIES[strategy]
    # Output dir is per (cropping_strategy, label_strategy) so we don't mix
    # crops/labels from different runs in the same folder.
    out_root = EVAL_OUT_ROOT / strategy / f"_label_{label_strategy}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Group ground truth by (patent, structure_page) — the page where the
    # molecule is drawn, NOT the assay-table page (those differ for Class B
    # patents like US10899738).
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in gt:
        by_page[(r["patent_id"], r["structure_page"])].append(r)

    # Per-patent set of all flat-connectivity InChIKeys in GT — used for
    # the LABEL_WRONG_DECIMER_OK bucket ("DECIMER recovered SOME patent
    # compound, just labeled it wrong").
    patent_gt_flat_keys: dict[str, set[str]] = defaultdict(set)
    for r in gt:
        if r.get("flat_connectivity_inchikey"):
            patent_gt_flat_keys[r["patent_id"]].add(r["flat_connectivity_inchikey"])

    # Per-patent set of compound IDs in GT — passed to layout_graph
    # `proximity_direction` scorer for the small `unique_id_in_gt` bonus.
    patent_gt_id_sets: dict[str, set[str]] = defaultdict(set)
    for r in gt:
        patent_gt_id_sets[r["patent_id"]].add(_norm_id(r["compound_id"]))

    page_results = []
    crop_records = []

    t0 = time.time()
    for (patent_id, page), gt_rows in sorted(by_page.items()):
        # Crops live in the original location (one place per cropping strategy)
        # so we don't duplicate PNGs per label-strategy.
        crop_out_dir = EVAL_OUT_ROOT / strategy / patent_id / f"page_{page:04d}"
        gt_ids = [r["compound_id"] for r in gt_rows]
        # Per-compound expected flat-connectivity InChIKey (for assigned-id match)
        gt_flat_lookup = {
            _norm_id(r["compound_id"]): r.get("flat_connectivity_inchikey")
            for r in gt_rows
        }
        # Full-stereo lookup so we can also report stereo-mismatch separately
        gt_full_lookup = {
            _norm_id(r["compound_id"]): r.get("connectivity_inchikey")
            for r in gt_rows
        }
        # Expected SMILES (for human-readable diverse-sample output)
        gt_smi_lookup = {
            _norm_id(r["compound_id"]): r.get("flat_canonical_smiles")
            for r in gt_rows
        }
        try:
            # Pass label_strategy through to the cropping runner.
            # decimer_seg understands it; geometric / sonnet_layout ignore extras.
            kwargs = {}
            if strategy == "decimer_seg":
                kwargs["label_strategy"] = label_strategy
                kwargs["patent_gt_id_set"] = patent_gt_id_sets.get(patent_id)
            crops, skip_reason = runner(patent_id, page, gt_ids, crop_out_dir, **kwargs)
        except Exception as e:
            logger.exception("strategy %s failed on %s p%d", strategy, patent_id, page)
            crops, skip_reason = [], f"exception: {e}"

        page_summary = {
            "patent_id": patent_id,
            "page": page,
            "n_gt": len(gt_rows),
            "n_crops": len(crops),
            "skip_reason": skip_reason,
            "buckets": defaultdict(int),
        }

        for crop in crops:
            smiles = _decimer_smiles(crop.crop_path)
            full_ikey14 = _connectivity_inchikey(smiles) if smiles else None
            flat_ikey14 = _flat_connectivity_inchikey(smiles) if smiles else None
            cid_norm = _norm_id(crop.assigned_compound_id)
            expected_flat = gt_flat_lookup.get(cid_norm)
            expected_full = gt_full_lookup.get(cid_norm)
            expected_smi = gt_smi_lookup.get(cid_norm)

            bucket = classify_bucket(
                assigned_id=crop.assigned_compound_id,
                decimer_smiles=smiles,
                predicted_full_ikey14=full_ikey14,
                predicted_flat_ikey14=flat_ikey14,
                expected_full_ikey14=expected_full,
                expected_flat_ikey14=expected_flat,
                patent_gt_flat_ikey14_set=patent_gt_flat_keys.get(patent_id, set()),
            )
            page_summary["buckets"][bucket] += 1

            # If the label is wrong but DECIMER recovered a patent compound,
            # find which GT compound's key matches → makes the diverse-sample
            # output show "labeled X but actually compound Y"
            actually_is = None
            if bucket in ("LABEL_WRONG_DECIMER_OK", "LABEL_NONE_DECIMER_OK", "OK_STEREO_DIFF"):
                for r in gt:
                    if r["patent_id"] != patent_id:
                        continue
                    if r.get("flat_connectivity_inchikey") == flat_ikey14:
                        actually_is = r["compound_id"]
                        break

            crop_records.append({
                "strategy": strategy,
                "patent_id": patent_id,
                "page": page,
                "assigned_id": crop.assigned_compound_id,
                "actually_is": actually_is,
                "crop_path": str(crop.crop_path),
                "decimer_smiles": smiles,
                "predicted_full_ikey14": full_ikey14,
                "predicted_flat_ikey14": flat_ikey14,
                "expected_full_ikey14": expected_full,
                "expected_flat_ikey14": expected_flat,
                "expected_smiles": expected_smi,
                "in_ground_truth": expected_flat is not None,
                "stereo_diff_only": bool(
                    full_ikey14 and expected_full
                    and full_ikey14 != expected_full
                    and flat_ikey14 == expected_flat
                ),
                "bucket": bucket,
                "notes": crop.notes,
            })

        # Convert defaultdict for JSON serialization
        page_summary["buckets"] = dict(page_summary["buckets"])
        page_results.append(page_summary)

    elapsed = time.time() - t0

    # Roll-up — bucket counts across all crops
    bucket_totals: dict[str, int] = defaultdict(int)
    for r in crop_records:
        bucket_totals[r["bucket"]] += 1
    n_gt = sum(p["n_gt"] for p in page_results)
    n_crops = sum(p["n_crops"] for p in page_results)
    n_ok        = bucket_totals.get("OK", 0)
    n_ok_stereo = bucket_totals.get("OK_STEREO_DIFF", 0)
    # Label-correct = OK + OK_STEREO_DIFF (the labeling-eval signal we care
    # about right now; stereo recovery is the deferred follow-up).
    n_label_correct = n_ok + n_ok_stereo
    # Stereo-mismatch diagnostic — DECIMER recovered the right connectivity
    # for SOME patent compound, just lost stereo. Counts cases across:
    # OK_STEREO_DIFF (label was right too) + LABEL_WRONG_DECIMER_OK +
    # LABEL_NONE_DECIMER_OK (label wrong/missing). This is the "deferred
    # problem #2" size — separately tracked, not a bench gate.
    n_decimer_recovered_with_stereo_loss = sum(
        1 for r in crop_records
        if r["stereo_diff_only"]
        or r["bucket"] in ("LABEL_WRONG_DECIMER_OK", "LABEL_NONE_DECIMER_OK")
    )

    rollup = {
        "strategy": strategy,
        "n_gt_total": n_gt,
        "n_crops_total": n_crops,
        "buckets": dict(bucket_totals),
        "label_correct_rate_vs_gt": (n_label_correct / n_gt) if n_gt else 0.0,
        "ok_rate_vs_gt": (n_ok / n_gt) if n_gt else 0.0,
        "deferred_stereo_mismatch_count": n_decimer_recovered_with_stereo_loss,
        "elapsed_seconds": elapsed,
        "per_page": page_results,
    }

    # ── Diverse-sample output (Hard rule #7) ─────────────────────────
    # ~3 crops per patent, stratified across buckets so the user can
    # hand-check every failure mode + every patent.
    sample = _select_diverse_sample(crop_records, per_patent=3)
    rollup["diverse_sample"] = sample

    sample_out = EVAL_OUT_ROOT / "_bench_samples" / strategy
    sample_out.mkdir(parents=True, exist_ok=True)
    (sample_out / "step2_sample.json").write_text(json.dumps(sample, indent=2))

    # Persist crop-level records for later inspection
    (out_root / "_crop_records.json").write_text(
        json.dumps(crop_records, indent=2)
    )
    (out_root / "_rollup.json").write_text(json.dumps(rollup, indent=2))

    return rollup


def _select_diverse_sample(
    crop_records: list[dict],
    per_patent: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Stratified sample for hands-on user verification.

    For each patent: pick `per_patent` crops, spread across distinct buckets
    so every failure mode the patent exhibits is visible. Seeded so reruns
    on the same data return the same sample.
    """
    by_patent: dict[str, list[dict]] = defaultdict(list)
    for r in crop_records:
        if Path(r["crop_path"]).exists():
            by_patent[r["patent_id"]].append(r)

    rng = random.Random(seed)
    out: list[dict] = []
    for patent_id in sorted(by_patent.keys()):
        records = by_patent[patent_id]
        # Group by bucket to stratify
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_bucket[r["bucket"]].append(r)
        buckets_present = sorted(by_bucket.keys())

        picks: list[dict] = []
        # Round-robin through buckets, picking one per bucket per pass
        bucket_idx = 0
        while len(picks) < per_patent and any(by_bucket[b] for b in buckets_present):
            b = buckets_present[bucket_idx % len(buckets_present)]
            if by_bucket[b]:
                pick = rng.choice(by_bucket[b])
                by_bucket[b].remove(pick)
                picks.append(pick)
            bucket_idx += 1
            if bucket_idx > 10 * len(buckets_present):
                break  # safety

        # Compact diverse-sample fields (drop the heavy GT metadata)
        for p in picks:
            out.append({
                "patent_id":            p["patent_id"],
                "page":                 p["page"],
                "crop_path":            p["crop_path"],
                "bucket":               p["bucket"],
                "assigned_label":       p["assigned_id"],
                "actually_is":          p.get("actually_is"),
                "decimer_smiles":       p.get("decimer_smiles"),
                "expected_smiles":      p.get("expected_smiles"),
                "predicted_full_ikey14":  p.get("predicted_full_ikey14"),
                "predicted_flat_ikey14":  p.get("predicted_flat_ikey14"),
                "expected_full_ikey14":   p.get("expected_full_ikey14"),
                "expected_flat_ikey14":   p.get("expected_flat_ikey14"),
                "stereo_diff_only":     p.get("stereo_diff_only"),
                "in_ground_truth":      p.get("in_ground_truth"),
                "notes":                p.get("notes"),
            })
    return out


_BUCKET_ORDER = [
    "OK", "OK_STEREO_DIFF",
    "LABEL_WRONG_DECIMER_OK", "LABEL_NONE_DECIMER_OK",
    "DECIMER_MISS", "LABEL_NONE", "NOT_IN_GT",
]


def print_report(rollups: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(" LABEL-ASSOCIATION EVAL — bucketed against BDB-derived ground truth")
    print("=" * 100)

    # Top-line: rate of correctly-labeled crops (OK + OK_STEREO_DIFF) / GT,
    # plus stereo-mismatch diagnostic count.
    print(f"{'strategy':<18}{'n_gt':>6}{'n_crops':>9}"
          f"{'label_ok%':>12}{'OK':>5}{'OK_S':>6}{'L?':>5}{'L0':>5}"
          f"{'DM':>5}{'L_':>5}{'NIG':>5}{'stereo_loss':>14}{'time_s':>10}")
    print("-" * 100)
    for r in rollups:
        b = r["buckets"]
        label_pct = f"{100 * r['label_correct_rate_vs_gt']:.1f}%"
        print(
            f"{r['strategy']:<18}{r['n_gt_total']:>6}{r['n_crops_total']:>9}"
            f"{label_pct:>12}"
            f"{b.get('OK', 0):>5}"
            f"{b.get('OK_STEREO_DIFF', 0):>6}"
            f"{b.get('LABEL_WRONG_DECIMER_OK', 0):>5}"
            f"{b.get('LABEL_NONE_DECIMER_OK', 0):>5}"
            f"{b.get('DECIMER_MISS', 0):>5}"
            f"{b.get('LABEL_NONE', 0):>5}"
            f"{b.get('NOT_IN_GT', 0):>5}"
            f"{r['deferred_stereo_mismatch_count']:>14}"
            f"{r['elapsed_seconds']:>10.1f}"
        )
    print()
    print("Bucket legend (sorted by 'this is good' → 'this is broken'):")
    print("  OK     = label correct + stereo correct (full success)")
    print("  OK_S   = OK_STEREO_DIFF: label correct + connectivity correct, stereo lost")
    print("           → DEFERRED problem #2 (Jie requires stereo for downstream docking)")
    print("  L?     = LABEL_WRONG_DECIMER_OK: DECIMER got SOME patent compound, label is wrong")
    print("           → CURRENT phase target — fixing labeling moves these to OK or OK_S")
    print("  L0     = LABEL_NONE_DECIMER_OK: unlabeled crop, DECIMER recovered patent compound")
    print("  DM     = DECIMER_MISS: assigned label in GT, DECIMER returned valid SMILES that")
    print("           doesn't match any patent compound → real DECIMER recognition fail")
    print("  L_     = LABEL_NONE: unlabeled crop, DECIMER didn't recover patent compound")
    print("  NIG    = NOT_IN_GT: labeled but label not in GT, DECIMER didn't recover GT match")
    print("  stereo_loss = count of crops where DECIMER's flat key matches some GT compound")
    print("                but full-stereo InChIKey differs (deferred problem size)")

    # Per-page detail using "label-correct/gt" rather than "match/gt"
    print("\n--- Per-page detail (label-correct/gt — OK + OK_STEREO_DIFF count as correct) ---")
    pages = set()
    for r in rollups:
        for p in r["per_page"]:
            pages.add((p["patent_id"], p["page"]))
    pages = sorted(pages)
    header = f"{'page':<25}" + "".join(f"{r['strategy']:>16}" for r in rollups)
    print(header)
    for pat, pg in pages:
        row_label = f"{pat}/p{pg}"
        cells = []
        for r in rollups:
            entry = next(
                (p for p in r["per_page"] if p["patent_id"] == pat and p["page"] == pg),
                None,
            )
            if entry is None:
                cells.append("-")
            elif entry["skip_reason"]:
                cells.append("SKIP")
            else:
                lc = (entry["buckets"].get("OK", 0)
                      + entry["buckets"].get("OK_STEREO_DIFF", 0))
                cells.append(f"{lc}/{entry['n_gt']}")
        print(f"{row_label:<25}" + "".join(f"{c:>16}" for c in cells))

    # Diverse-sample dump (mandatory, hands-on user-verify)
    for r in rollups:
        sample = r.get("diverse_sample", [])
        if not sample:
            continue
        print()
        print("=" * 100)
        print(f" DIVERSE SAMPLE FOR HAND-CHECK — strategy={r['strategy']} "
              f"({len(sample)} crops, ~3/patent, stratified by bucket)")
        print("=" * 100)
        print("Open each crop_path with the Read tool and confirm:")
        print("  (a) one complete molecule per crop")
        print("  (b) the assigned_label is the correct compound number for the molecule")
        print("  (c) the bucket assignment matches reality")
        print()
        last_patent = None
        for s in sample:
            if s["patent_id"] != last_patent:
                last_patent = s["patent_id"]
                print(f"\n  ## {last_patent}")
            print(f"    [p{s['page']:>4} {s['bucket']:<22}] "
                  f"label={s['assigned_label']:<22} "
                  f"actually={s.get('actually_is') or '-'}")
            if s.get("decimer_smiles"):
                print(f"      decimer:  {s['decimer_smiles'][:90]}")
            if s.get("expected_smiles"):
                print(f"      expected: {s['expected_smiles'][:90]}")
            if s.get("stereo_diff_only"):
                print(f"      ⚠ stereo_diff_only — flat InChIKey matches, full does not")
            print(f"      crop:     {s['crop_path']}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", action="append", default=None,
                    choices=list(STRATEGIES.keys()),
                    help="Cropping strategy to evaluate (repeatable). Default: all.")
    ap.add_argument(
        "--label-strategy", action="append", default=None,
        choices=["legacy", "cell_iou", "proximity_direction", "layout_graph"],
        help="Label-association strategy (repeatable; head-to-head when multiple). "
             "legacy = original 2D-nearest baseline; "
             "cell_iou (4a) = structure-cell IoU only; "
             "proximity_direction (4b) = proximity + above/below + heading specificity; "
             "layout_graph (4c) = regime-routed combination. "
             "Default: legacy + layout_graph (the most informative head-to-head).",
    )
    ap.add_argument("--visual-sample", type=int, default=5,
                    help="Legacy: include path-only diverse sample of N "
                         "crops/patent in stdout (the bucketed diverse sample "
                         "is always emitted regardless).")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    strategies = args.strategy or list(STRATEGIES.keys())
    label_strategies = args.label_strategy or ["legacy", "layout_graph"]
    gt = load_ground_truth()
    print(f"Loaded {len(gt)} ground-truth records "
          f"({len({(r['patent_id'], r['structure_page']) for r in gt})} structure pages, "
          f"{len({r['patent_id'] for r in gt})} patents)")

    rollups = []
    for cs in strategies:
        for ls in label_strategies:
            tag = f"{cs}+{ls}"
            print(f"\n>>> Running cropping={cs}  labeling={ls}")
            r = evaluate_strategy(cs, gt, args.visual_sample, label_strategy=ls)
            r["strategy"] = tag    # so the report distinguishes runs
            rollups.append(r)

    print_report(rollups)
    return 0


if __name__ == "__main__":
    sys.exit(main())
