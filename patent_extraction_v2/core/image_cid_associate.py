"""PDF-page structure extraction with proximity CID association.

The CID path the user specified (the eval-only `routes/label_association.py`
graph needs full-page OCR these patents lack; this is the lighter fallback):

  1. DECIMER-Seg (`segment_page`) → one bbox + crop per drawn structure. Splits
     multi-structure figures the Google per-figure PNGs lost (C00077/C00006).
  2. DECIMER `predict_SMILES` per crop → SMILES (complete vs markush via
     `image_recog.classify_decimer_output`).
  3. Find each structure's compound ID by growing a search box outward from its
     bbox in SMALL DIRECTIONAL STEPS (up/down a little, left/right a little,
     slowly increasing) until a "Example N" / "Cpd. No. N" / left-column number
     label is hit — each direction capped so we never reach a neighbour's label.
     The growth radius at which it's hit is stored as the DISTANCE, so when two
     structures resolve to the same CID the nearer one keeps it and the farther
     one re-resolves to its next candidate.

A CID is REQUIRED: a complete structure with no findable label is set aside
(`unresolved_no_cid`), never emitted with a fabricated id.

Layout grounding (US10544143 TABLE pages, verified by eye): the Ex. No. sits in
the LEFT column, row-aligned with the structure — so a bare integer in the left
margin is a candidate and the search reaches far LEFT but only a little up/down.

Noise filtering (ported from routes/label_association.py): reagent-time / atom /
intermediate tokens, patent line-number columns, and header/footer page numbers
are dropped before they can be mistaken for a CID.

MEMORY: DECIMER-Seg (~2-3 GB), DECIMER, and PaddleOCR are heavy. `run_patent`
runs in STAGES (segment → unload → decimer → unload → ocr → associate) so only
ONE big model is resident at a time. Everything is SEQUENTIAL — never threaded.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .models import (
    Compound, CompoundProvenance, CompoundSource, ConfidenceTier, IupacSource,
)
from .smiles_utils import canonicalize_smiles, get_inchikey, strip_salt
from .image_recog import classify_decimer_output, _decimer_predict

logger = logging.getLogger(__name__)

# ── Label patterns (mirror routes/label_association.py) ─────────────
_HEADING_CID_RE = re.compile(
    r"(?:Example|Cpd\.?\s*No\.?|Compound|Ex\.?\s*No\.?)\s*(\d{1,4}[A-Za-z]{0,3})",
    re.IGNORECASE,
)
_BARE_CID_RE = re.compile(r"^(\d{1,4}[A-Za-z]{0,2})$")
# Whole-block standalone heading (keyword + id alone) — inline prose mentions
# like "...described in Example 561, employing..." must NOT become candidates.
_STANDALONE_HEADING_RE = re.compile(
    r"^\s*#?\s*(?:Example|Cpd\.?\s*No\.?|Compound)\s*\d{1,4}[A-Za-z]{0,3}\s*:?\s*$",
    re.IGNORECASE,
)
# Chemistry-domain noise tokens — never a compound-ID label.
_NOISE_TOKEN_PATTERNS = [
    re.compile(r"^\d{1,3}\s*h$", re.IGNORECASE),     # "2h", "18 h" reagent times
    re.compile(r"^\d{1,3}\s*min$", re.IGNORECASE),   # "30 min"
    re.compile(r"^[Hh]rs?$"),                         # "h", "hr", "Hrs"
    re.compile(r"^S\d{1,3}$"),                        # "S19" synthesis intermediates
    re.compile(r"^[CNOSP]\d$"),                       # "C1", "N2" atom labels
    re.compile(r"^[CNOSP][a-z]$"),                    # "Cl", "Br"
    re.compile(r"^[A-Z][a-z]?$"),                     # single atom/element
    re.compile(r"^0+$"),                              # stray zeros
]

# ── Directional expansion caps (fractions of page dims) ─────────────
# Vertical is TIGHT (stay inside the structure's table row so we never reach the
# row above/below). Horizontal LEFT is generous (reach the Ex.No column). Right
# is tight (the right columns are MW/LCMS numbers, not CIDs). Growth proceeds in
# `_STEP_FRAC` increments until a candidate enters the box or a cap is reached.
_VCAP_FRAC = 0.055     # max up/down gap, ~half a typical row
_LEFT_CAP_FRAC = 0.42  # max leftward reach to the Ex.No column
_RIGHT_CAP_FRAC = 0.06 # max rightward reach (CIDs are rarely to the right)
_LEFT_COL_FRAC = 0.20  # x-center below this fraction of width = left label column


@dataclass
class _Label:
    cid: str
    text: str
    box: tuple[float, float, float, float]   # x1,y1,x2,y2
    heading: bool                            # matched "Example N" (preferred)


@dataclass
class StructureCID:
    crop_index: int
    bbox: tuple[int, int, int, int]
    smiles: str | None
    verdict: str                              # complete | markush | fail
    cid: str | None
    cid_distance: float | None                # growth radius (px) at which CID hit


@dataclass
class PageExtraction:
    page_num: int
    structures: list[StructureCID] = field(default_factory=list)
    unresolved_no_cid: list[int] = field(default_factory=list)  # crop indices
    stats: dict = field(default_factory=dict)


_paddle = None


def _get_ocr():
    global _paddle
    if _paddle is None:
        from paddleocr import PaddleOCR
        _paddle = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
    return _paddle


def _release_ocr() -> None:
    global _paddle
    _paddle = None
    import gc
    gc.collect()


def _box_xyxy(b) -> tuple[float, float, float, float]:
    b = np.array(b)
    if b.ndim == 1 and len(b) == 4:
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    xs, ys = b[:, 0], b[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _matches_noise_token(content: str) -> bool:
    plain = re.sub(r"\s+", "", (content or "").strip())
    if not plain:
        return False
    return any(p.match(plain) for p in _NOISE_TOKEN_PATTERNS)


def _line_number_token_boxes(raw: list[tuple[str, tuple]], page_h: int) -> set[int]:
    """Indices of tokens that form a patent LINE-NUMBER column (multiples of 5
    clustered at one x, spanning >=30% of page height). Ported from
    label_association._identify_line_number_column. Those are bare integers that
    would otherwise masquerade as left-column CIDs."""
    cand: list[tuple[int, int, float]] = []   # (idx, value, x_center)
    for i, (txt, box) in enumerate(raw):
        t = (txt or "").strip()
        if t.isdigit() and 5 <= int(t) <= 100 and int(t) % 5 == 0:
            cx = (box[0] + box[2]) / 2
            cand.append((i, int(t), cx))
    if len(cand) < 4:
        return set()
    cand.sort(key=lambda c: c[2])
    clusters: list[list[tuple[int, int, float]]] = []
    cur: list[tuple[int, int, float]] = []
    last_x = None
    for c in cand:
        if last_x is None or abs(c[2] - last_x) <= 60:
            cur.append(c)
        else:
            clusters.append(cur)
            cur = [c]
        last_x = c[2]
    if cur:
        clusters.append(cur)
    drop: set[int] = set()
    for cl in clusters:
        if len(cl) < 4:
            continue
        ys = [(raw[i][1][1] + raw[i][1][3]) / 2 for (i, _v, _x) in cl]
        if (max(ys) - min(ys)) / max(1, page_h) >= 0.30:
            drop.update(i for (i, _v, _x) in cl)
    return drop


def _mine_cid_labels(raw: list[tuple[str, tuple]], page_w: int, page_h: int) -> list[_Label]:
    """Turn raw (text, box) OCR tokens into CID label candidates, applying the
    ported noise filters. A token is a candidate if it is a STANDALONE
    `Example N` heading, or a bare integer in the left label column (and not a
    line-number / page-number / noise token)."""
    header_y = 0.13 * page_h
    footer_y = 0.92 * page_h
    line_no = _line_number_token_boxes(raw, page_h)
    labels: list[_Label] = []
    for i, (txt, box) in enumerate(raw):
        t = (txt or "").strip()
        if not t or i in line_no or _matches_noise_token(t):
            continue
        x1, y1, x2, y2 = box
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        # Heading-style: require it to be standalone (not inline prose).
        if _STANDALONE_HEADING_RE.match(t):
            m = _HEADING_CID_RE.search(t)
            if m:
                labels.append(_Label(m.group(1), t, (x1, y1, x2, y2), heading=True))
                continue
        # Bare integer: only in the left column, and not a header/footer page no.
        bm = _BARE_CID_RE.match(t)
        if bm and xc < _LEFT_COL_FRAC * page_w:
            if (yc < header_y or yc > footer_y) and t.isdigit() and len(t) <= 2:
                continue   # page number in header/footer band
            labels.append(_Label(bm.group(1), t, (x1, y1, x2, y2), heading=False))
    return labels


def _growth_radius(structure: tuple, label: tuple,
                   page_w: int, page_h: int) -> float | None:
    """The directional growth radius (px) at which `label` first enters a box
    grown out from `structure`, or None if it never enters within the caps.

    Grows up/down by the same vertical amount and left/right by their own
    horizontal amounts; the radius is how far we had to grow (normalized so the
    tightest cap dominates), so the NEAREST-in-its-direction label wins.
    """
    sx1, sy1, sx2, sy2 = structure
    lcx = (label[0] + label[2]) / 2
    lcy = (label[1] + label[3]) / 2
    # Directional gaps from the structure's edges to the label center.
    gap_up = sy1 - lcy if lcy < sy1 else 0.0
    gap_down = lcy - sy2 if lcy > sy2 else 0.0
    gap_left = sx1 - lcx if lcx < sx1 else 0.0
    gap_right = lcx - sx2 if lcx > sx2 else 0.0
    vgap = max(gap_up, gap_down)
    hgap = max(gap_left, gap_right)
    vcap = _VCAP_FRAC * page_h
    lcap = _LEFT_CAP_FRAC * page_w
    rcap = _RIGHT_CAP_FRAC * page_w
    if vgap > vcap:
        return None                       # different row → noise
    if gap_left > lcap or gap_right > rcap:
        return None                       # beyond horizontal cap → noise
    # Normalize each axis by its cap so a far-left in-column label (table case)
    # competes fairly with a small-vertical-gap label; radius = max normalized.
    hcap = lcap if gap_left >= gap_right else rcap
    r_norm = max(vgap / max(1.0, vcap), hgap / max(1.0, hcap))
    return r_norm


def _assign_cids(structures: list[tuple], labels: list[_Label],
                 page_w: int, page_h: int) -> dict[int, tuple[str, float]]:
    """Greedy nearest-growth-first assignment with distance-based conflict
    resolution. Each (structure, label) edge is the growth radius at which the
    label enters the structure's expanding box; process edges smallest-radius
    first, assigning a label to a structure only if neither is taken. So a
    contested CID goes to the nearer structure and the farther one falls through
    to its next candidate. Heading-style labels get a tiny radius discount so an
    explicit "Example N" is preferred over a bare integer at equal distance."""
    edges: list[tuple[float, int, int]] = []
    for si, s in enumerate(structures):
        for li, lab in enumerate(labels):
            r = _growth_radius(s, lab.box, page_w, page_h)
            if r is None:
                continue
            if lab.heading:
                r -= 0.05
            edges.append((r, si, li))
    edges.sort(key=lambda e: e[0])
    assigned: dict[int, tuple[str, float]] = {}
    used_s: set[int] = set()
    used_l: set[int] = set()
    for r, si, li in edges:
        if si in used_s or li in used_l:
            continue
        assigned[si] = (labels[li].cid, round(max(0.0, r), 4))
        used_s.add(si)
        used_l.add(li)
    return assigned


# ── Per-page orchestration (assumes models already loaded by caller) ──

def associate_page(
    page_arr: np.ndarray,
    crops: list,                 # list[SegCrop]
    decimer_raw: dict[int, str], # crop_index -> raw DECIMER SMILES
    patent_id: str,
    page_num: int,
    existing_inchikeys: set[str] | None = None,
) -> tuple[PageExtraction, list[Compound]]:
    """Pure assembly step (no model calls): given a page image, its segmented
    structure crops, and precomputed DECIMER outputs, OCR for CID labels,
    associate, and build Compounds for the COMPLETE molecules that got a CID."""
    existing = set(existing_inchikeys or set())
    page_h, page_w = page_arr.shape[0], page_arr.shape[1]

    res = _get_ocr().predict(page_arr)
    raw_tokens: list[tuple[str, tuple]] = []
    if res:
        r0 = res[0]
        texts = r0.get("rec_texts") or []
        boxes = r0.get("rec_boxes")
        if boxes is None:
            boxes = r0.get("rec_polys") or []
        for t, b in zip(texts, boxes):
            raw_tokens.append((t, _box_xyxy(b)))
    labels = _mine_cid_labels(raw_tokens, page_w, page_h)

    struct_boxes = [c.bbox_px for c in crops]
    assign = _assign_cids(struct_boxes, labels, page_w, page_h)

    page = PageExtraction(page_num=page_num)
    compounds: list[Compound] = []
    seen_ik: set[str] = set()
    n_complete = n_markush = n_fail = n_dup = 0
    for c in crops:
        raw = decimer_raw.get(c.crop_index, "")
        verdict, main = classify_decimer_output(raw)
        cid_pair = assign.get(c.crop_index)
        cid = cid_pair[0] if cid_pair else None
        cid_dist = cid_pair[1] if cid_pair else None
        page.structures.append(StructureCID(
            crop_index=c.crop_index, bbox=c.bbox_px, smiles=main or None,
            verdict=verdict, cid=cid, cid_distance=cid_dist,
        ))
        if verdict == "markush":
            n_markush += 1
            continue
        if verdict == "fail":
            n_fail += 1
            continue
        # verdict == complete
        if cid is None:
            page.unresolved_no_cid.append(c.crop_index)  # REQUIRE a CID — set aside
            continue
        canonical = canonicalize_smiles(main)
        if not canonical:
            n_fail += 1
            continue
        ik = get_inchikey(canonical)
        if not ik or ik in existing or ik in seen_ik:
            n_dup += 1
            continue
        seen_ik.add(ik)
        salt = strip_salt(canonical)
        compounds.append(Compound(
            patent_id=patent_id,
            example_number=f"Example {cid}",
            iupac_name=None,
            iupac_source=IupacSource.GENERATED,
            smiles_from_image=main,
            canonical_smiles=canonical,
            inchikey=ik,
            parent_smiles=salt.get("parent_smiles"),
            parent_inchikey=salt.get("parent_inchikey"),
            source=CompoundSource.EXEMPLIFIED,
            confidence_tier=ConfidenceTier.SINGLE_VALIDATED,
            extraction_method="image_decimer",
            provenance=CompoundProvenance(
                route="image_pipeline", stage="decimer_local",
                page=page_num, image_path=str(c.image_path), chain=["decimer_local"],
            ),
            image_path=str(c.image_path),
            source_page=page_num,
            processing_status="validated",
        ))
        n_complete += 1

    page.stats = {
        "page": page_num, "structures": len(crops), "labels": len(labels),
        "complete_with_cid": n_complete, "complete_no_cid": len(page.unresolved_no_cid),
        "markush": n_markush, "fail": n_fail, "dup": n_dup,
    }
    return page, compounds


# ── Overnight runner: memory-staged, sequential, resumable ──────────

def run_patent(
    patent_id: str,
    start_page: int = 1,
    end_page: int | None = None,
    dpi: int = 300,
    existing_inchikeys: set[str] | None = None,
) -> dict:
    """Process a whole patent overnight in MEMORY-STAGED passes so only one
    heavy model is resident at a time:

      Stage A: segment every page (DECIMER-Seg)  → crops + bboxes (disk-cached)
      Stage B: DECIMER every crop                → raw SMILES json (resumable)
      Stage C+D: OCR each page + associate       → compounds + per-page stats

    Sequential throughout. Writes results to output_v2/image_recog/{patent_id}/.
    Safe to re-run: Stage A uses segment_page's sidecar cache, Stage B caches
    raw SMILES to disk, so an interrupted run resumes cheaply.
    """
    from . import config
    from ..routes.decimer_segmentation_crop import segment_page, render_pdf_page, unload
    import fitz

    pdf = config.DATA_DIR / patent_id / f"{patent_id}.pdf"
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    out_dir = config.IMAGES_DIR.parent / "image_recog" / patent_id   # output_v2/image_recog/{pid}
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = out_dir / "_seg"
    raw_path = out_dir / "decimer_raw.json"

    doc = fitz.open(str(pdf))
    n_pages = len(doc)
    doc.close()
    end_page = end_page or n_pages
    pages = list(range(start_page, min(end_page, n_pages) + 1))

    # ── Stage A: segment all pages (only DECIMER-Seg resident) ─────
    logger.info("image_cid %s: STAGE A segment %d pages", patent_id, len(pages))
    page_crops: dict[int, list] = {}
    for pn in pages:
        page_crops[pn] = segment_page(pdf, pn, seg_dir, dpi=dpi)   # cached on re-run
    unload(force=True)   # free the ~2-3 GB Seg model before DECIMER

    # ── Stage B: DECIMER every crop (only DECIMER resident); resumable ─
    # Keys are strings (JSON round-trips dict keys as str); converted back to
    # int crop_index in the associate stage below.
    logger.info("image_cid %s: STAGE B decimer", patent_id)
    decimer_raw: dict[str, dict[str, str]] = {}
    if raw_path.exists():
        decimer_raw = json.loads(raw_path.read_text())
    for pn in pages:
        key = str(pn)
        done = decimer_raw.get(key, {})
        for c in page_crops[pn]:
            if str(c.crop_index) in done:
                continue
            try:
                done[str(c.crop_index)] = _decimer_predict(str(c.image_path))
            except Exception as e:
                logger.warning("decimer fail %s p%d c%d: %s", patent_id, pn, c.crop_index, e)
                done[str(c.crop_index)] = ""
        decimer_raw[key] = done
        raw_path.write_text(json.dumps(decimer_raw))   # checkpoint per page

    # ── Stage C+D: OCR + associate (only PaddleOCR resident) ───────
    logger.info("image_cid %s: STAGE C+D ocr+associate", patent_id)
    all_compounds: list[Compound] = []
    page_stats: list[dict] = []
    for pn in pages:
        if not page_crops[pn]:
            continue
        page_arr = np.array(render_pdf_page(pdf, pn, dpi=dpi))
        raw_for_page = {int(k): v for k, v in decimer_raw.get(str(pn), {}).items()}
        page, comps = associate_page(
            page_arr, page_crops[pn], raw_for_page, patent_id, pn, existing_inchikeys,
        )
        all_compounds.extend(comps)
        page_stats.append(page.stats)
    _release_ocr()

    result = {
        "patent_id": patent_id,
        "pages_processed": len(pages),
        "compounds": [c.model_dump() for c in all_compounds],
        "page_stats": page_stats,
        "n_compounds": len(all_compounds),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    logger.info("image_cid %s: DONE %d compounds → %s",
                patent_id, len(all_compounds), out_dir / "result.json")
    return result


if __name__ == "__main__":   # overnight CLI: python -m ...core.image_cid_associate US10544143 [start] [end]
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pid = sys.argv[1]
    sp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_patent(pid, start_page=sp, end_page=ep)
