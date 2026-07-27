"""Prime upstream-OCR caches for the GT bench pages, with explicit
model unloading between phases so peak RAM stays bounded.

The bench pipeline needs three model outputs per page:
  - DECIMER-Seg structure crops (~2-3 GB model)
  - PaddleOCR text-detection blocks (~4-6 GB models)
  - PP-Structure cell bboxes (~1-2 GB model)

Loading them simultaneously was peaking at 14 GB and crashing the
laptop. This script enforces strictly-sequential phases:

  PHASE 1: render every page → numpy array (cheap)
  PHASE 2: DECIMER-Seg on every page → cache → unload TF
  PHASE 3: PaddleOCR on every page → cache → unload paddle
  PHASE 4: PP-Structure cells on every page → cache → unload paddle

After each phase, peak RAM drops back to ~1 GB. Subsequent runs
(visualizer / bench) hit caches and load NO models at all.
"""
from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


GT_PAGES = [
    ("US10899738", 27),  ("US10899738", 155),
    ("US9718825",  89),  ("US9718825",  90),  ("US9718825",  91),
    ("US11312727", 15),  ("US11312727", 17),  ("US11312727", 18),
    ("US11312727", 19),  ("US11312727", 63),
]


def _log_mem(label: str) -> None:
    """Print resident memory size — quick debug for the unload pattern."""
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KB
        rss_mb = rss_kb / (1024 * 1024) if sys.platform == "darwin" else rss_kb / 1024
        print(f"  [mem after {label}] peak RSS = {rss_mb:.1f} MB", flush=True)
    except Exception:
        pass


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    from patentdb.core import config as v2_config

    pdf_paths = {pat: v2_config.DATA_DIR / pat / f"{pat}.pdf"
                  for pat in {p for p, _ in GT_PAGES}}

    # ── PHASE 1: render every page once, hold as numpy arrays ──────
    print("PHASE 1: render pages...", flush=True)
    from patentdb.routes.decimer_segmentation_crop import (
        render_pdf_page, SEG_DPI,
    )
    rendered: dict[tuple[str, int], object] = {}
    for pat, pg in GT_PAGES:
        rendered[(pat, pg)] = render_pdf_page(pdf_paths[pat], pg, dpi=SEG_DPI)
        print(f"  {pat} p{pg} rendered ({rendered[(pat,pg)].size})", flush=True)
    _log_mem("render")

    # ── PHASE 2: DECIMER-Seg ───────────────────────────────────────
    # Cached pages skip the TF load entirely. Force one pass to
    # populate any sidecars that haven't been written yet.
    print("\nPHASE 2: DECIMER-Seg structure detection...", flush=True)
    from patentdb.routes.decimer_segmentation_crop import segment_page
    for pat, pg in GT_PAGES:
        out_dir = (v2_config.OUTPUT_DIR / "cropping_eval" / "decimer_seg"
                   / pat / f"page_{pg:04d}")
        crops = segment_page(pdf_paths[pat], pg, out_dir)
        print(f"  {pat} p{pg}: {len(crops)} structures cached", flush=True)
    from patentdb.routes import decimer_segmentation_crop as ds
    ds.unload(force=True)
    _log_mem("DECIMER-Seg + unload")

    # ── PHASE 3: PaddleOCR text detection ──────────────────────────
    print("\nPHASE 3: PaddleOCR text-detection (the big one)...", flush=True)
    from patentdb.routes import text_detection as td
    for pat, pg in GT_PAGES:
        blocks = td.detect_text_blocks(rendered[(pat, pg)], patent_id=pat, page=pg)
        print(f"  {pat} p{pg}: {len(blocks)} text regions cached", flush=True)
    td.unload()
    gc.collect()
    _log_mem("PaddleOCR unload")

    # ── PHASE 4: PP-Structure cell detection ───────────────────────
    print("\nPHASE 4: PP-Structure table cells...", flush=True)
    from patentdb.routes import table_cell_detection as tcd
    from patentdb.routes.table_cell_detection import (
        _paddlex_table_bboxes_from_md,
    )
    for pat, pg in GT_PAGES:
        md_path = v2_config.DATA_DIR / pat / "all_pages" / f"page_{pg:04d}.md"
        if not md_path.exists():
            print(f"  {pat} p{pg}: no PaddleX md, skipping", flush=True)
            continue
        scale = SEG_DPI / 96.0   # PaddleX→render scale
        table_bboxes = _paddlex_table_bboxes_from_md(md_path.read_text(errors="replace"), scale)
        cells = tcd.detect_table_cells(rendered[(pat, pg)], table_bboxes,
                                        patent_id=pat, page=pg)
        print(f"  {pat} p{pg}: {len(cells)} cells cached", flush=True)
    tcd.unload()
    gc.collect()
    _log_mem("cell detector unload")

    print("\nDONE — all caches primed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
