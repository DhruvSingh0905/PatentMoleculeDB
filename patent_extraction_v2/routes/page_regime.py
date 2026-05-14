"""Page-type regime classifier (Step 3.5 of the label-association plan).

Why this exists
---------------
Different page layouts have stable, distinct geometric patterns. A
structure-table page wants cell-IoU matching; a synthesis-scheme page
wants proximity-direction; an inline-Example page wants heading-anchored
matching. We route the right matching algorithm to the right page based
on UPSTREAM signals — never patent IDs.

This classifier looks at three inputs:
  - PaddleX text/title/image/table blocks on the page (with bboxes)
  - PP-StructureV3 cell detector output (Step 3)
  - DECIMER-Seg structure detections on the page

and returns one of five regimes that drive the layout-graph matcher.

Hard rule: NO patent-specific code. Detection rules are derived from
text-content patterns + bbox geometry only. The same classifier runs
against any pharma patent without modification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class PageRegime(str, Enum):
    """The five regimes the layout-graph matcher knows how to route to."""

    # Structure-table page: PaddleX/PP-Structure detected cells, page has
    # a "Cpd. No." / "Compound No." / "Example No." header text block,
    # and ≥2 DECIMER-Seg structures fall inside cell bboxes.
    # Examples: US10899738 p27/p28 (Table 1 / Table 1-continued).
    TABLE_STRUCTURE = "table_structure"

    # Bleed-through table: PaddleX `<table>` markup contains compound-ID
    # `<td>` cells, but the structure cells were OCR'd as text-noise
    # (a known PaddleX failure on chemistry inside cells). DECIMER-Seg
    # still picks the structures up from the rendered page.
    # Examples: US11312727 p220, p235.
    BLEED_TABLE = "bleed_table"

    # Synthesis-scheme page: page contains "Scheme N" / "Formula (I)" /
    # "Step N — Synthesis" / "Synthesis of <name>" text patterns AND
    # ≥1 DECIMER-Seg structure. Multiple unrelated molecules per page.
    # Examples: US10899738 p155 (right column with synthesis schemes).
    SCHEME = "scheme"

    # Inline Example page: the page has "# Example N" or "# Cpd. No N"
    # heading-style text blocks AND structures cluster vertically with
    # roughly one structure per heading.
    # Examples: US9718825 p89-91 (Example 549/554/561/565 etc.)
    INLINE_EXAMPLE = "inline_example"

    # Anything that doesn't match — fall back to proximity-direction-only.
    # Includes assay-table pages (no structure column, nothing to label).
    UNKNOWN = "unknown"


# Patterns that indicate "this page is a compound table" — same regex set
# the GT builder uses, so the two stay consistent.
_TABLE_HEADER_RE = re.compile(
    r"(?<![A-Za-z])(?:Cpd\.?\s*No\.?|Compound\s*No\.?|Example\s*No\.?)",
)
_TABLE_CONTINUED_RE = re.compile(
    r"(?<![A-Za-z])TABLE\s*\d+(?:-continued|\s+continued)?",
    re.IGNORECASE,
)

# Synthesis / scheme markers
_SCHEME_RE = re.compile(
    r"(?:Scheme\s+\d+|Formula\s*\([IVX]+\)|Step\s+\d+\s*[—–\-]\s*Synthesis"
    r"|Synthesis\s+of\b|reaction\s+scheme)",
    re.IGNORECASE,
)

# Inline Example/Cpd heading markers (present if the page introduces
# compounds via per-Example sections rather than a table).
_INLINE_HEADING_RE = re.compile(
    r"#\s*(?:Example|Cpd\.?\s*No\.?|Compound)\s+\d+[A-Za-z]?\b",
    re.IGNORECASE,
)


@dataclass
class RegimeSignals:
    """Diagnostic info — useful for explaining why a page got a given regime."""
    n_structures: int                 # DECIMER-Seg structure detections
    n_cells: int                      # cells from Step 3 detector
    n_structures_inside_cells: int    # how many structures' centers lie in some cell
    has_table_header: bool            # "Cpd. No.", "Compound No.", "Example No." text
    has_table_continued: bool         # "TABLE N-continued"
    has_scheme_marker: bool           # "Scheme N", "Formula (I)", "Step N — Synthesis"
    n_inline_headings: int            # `# Example N` / `# Cpd. No N` markers
    has_paddlex_table_block: bool     # PaddleX emitted a `<|ref|>table<|/ref|>`


def _bbox_center_inside(structure_bbox: tuple[int, int, int, int],
                         cell_bbox: tuple[int, int, int, int]) -> bool:
    sx1, sy1, sx2, sy2 = structure_bbox
    cx1, cy1, cx2, cy2 = cell_bbox
    cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    return cx1 <= cx <= cx2 and cy1 <= cy <= cy2


def classify_page_regime(
    paddlex_text: str,
    text_blocks: list[tuple[tuple[int, int, int, int], str]],
    cells: list[tuple[int, int, int, int]],
    structure_bboxes: list[tuple[int, int, int, int]],
    paddlex_table_bboxes: list[tuple[int, int, int, int]],
) -> tuple[PageRegime, RegimeSignals]:
    """Decide the page regime + return the signals that drove the choice.

    Args:
        paddlex_text: full PaddleX markdown text (used for header/scheme markers).
        text_blocks: list of (bbox, content) for every PaddleX text/title block
                     on the page. Currently unused for routing (reserved for
                     bleed-table refinement) but kept in the signature so the
                     caller doesn't need to recompute it for Step 4.
        cells: list of cell bboxes from Step 3 (page-image coords).
        structure_bboxes: DECIMER-Seg structure bboxes (page-image coords).
        paddlex_table_bboxes: table-region bboxes from PaddleX `<|ref|>table<|/ref|>`
                              markdown blocks (page-image coords). Empty list
                              if PaddleX didn't emit a table region.

    Returns:
        (PageRegime, RegimeSignals). The signals object lets callers explain
        the decision in audit output.
    """
    n_structures = len(structure_bboxes)
    n_cells = len(cells)
    n_inside = sum(
        1 for s in structure_bboxes
        if any(_bbox_center_inside(s, c) for c in cells)
    )
    has_table_header = bool(_TABLE_HEADER_RE.search(paddlex_text))
    has_table_continued = bool(_TABLE_CONTINUED_RE.search(paddlex_text))
    has_scheme_marker = bool(_SCHEME_RE.search(paddlex_text))
    n_inline_headings = len(_INLINE_HEADING_RE.findall(paddlex_text))
    has_paddlex_table_block = bool(paddlex_table_bboxes)

    signals = RegimeSignals(
        n_structures=n_structures,
        n_cells=n_cells,
        n_structures_inside_cells=n_inside,
        has_table_header=has_table_header,
        has_table_continued=has_table_continued,
        has_scheme_marker=has_scheme_marker,
        n_inline_headings=n_inline_headings,
        has_paddlex_table_block=has_paddlex_table_block,
    )

    # --- BLEED_TABLE: PaddleX wrapped the area as `<|ref|>table<|/ref|>`
    # AND DECIMER-Seg picks structures up that PaddleX OCR'd as text noise.
    # Heuristic: PaddleX table block + ≥1 structure that overlaps the
    # table area. We check overlap by structure-center-inside-table-bbox.
    if has_paddlex_table_block:
        n_structures_in_table = sum(
            1 for s in structure_bboxes
            if any(_bbox_center_inside(s, t) for t in paddlex_table_bboxes)
        )
        if n_structures_in_table >= 1 and (has_table_header or has_table_continued):
            return (PageRegime.BLEED_TABLE, signals)

    # --- TABLE_STRUCTURE: cells exist, header is present, and structures
    # land inside cells (so the cell-IoU scorer has work to do).
    if (n_cells >= 2
            and (has_table_header or has_table_continued)
            and n_inside >= 2):
        return (PageRegime.TABLE_STRUCTURE, signals)

    # --- SCHEME: any scheme/formula marker + at least one structure.
    # Tested before INLINE_EXAMPLE so multi-compound synthesis pages
    # (which often also contain Example headings deeper in the text)
    # route to the right scorer.
    if has_scheme_marker and n_structures >= 1:
        return (PageRegime.SCHEME, signals)

    # --- INLINE_EXAMPLE: explicit `# Example N` / `# Cpd. No N` headings,
    # roughly one heading per detected structure (within ±1).
    if n_inline_headings >= 1 and n_structures >= 1 \
            and abs(n_inline_headings - n_structures) <= max(2, n_structures // 2):
        return (PageRegime.INLINE_EXAMPLE, signals)

    return (PageRegime.UNKNOWN, signals)


# ── CLI for visual bench ─────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Print regime + signals for one or more (patent, page) pairs.

    Example:
        python3 -m patent_extraction_v2.routes.page_regime \\
            --patent US10899738 --pages 27,28,155,179
    """
    import argparse
    from pathlib import Path
    from patent_extraction_v2.core import config as v2_config
    from patent_extraction_v2.routes.decimer_segmentation_crop import (
        render_pdf_page, segment_page, SEG_DPI,
    )
    from patent_extraction_v2.routes.table_cell_detection import (
        detect_table_cells, _paddlex_table_bboxes_from_md,
    )
    from patent_extraction_v2.scripts.eval.cropping_eval import (
        parse_paddlex_blocks,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", required=True)
    ap.add_argument("--pages", required=True,
                    help="Comma-separated page numbers, e.g. 27,28,155")
    ap.add_argument("--no-detector", action="store_true",
                    help="Skip structure + cell detectors (for fast regex-only check)")
    args = ap.parse_args(argv)

    pdf_path = v2_config.DATA_DIR / args.patent / f"{args.patent}.pdf"
    pages = [int(p.strip()) for p in args.pages.split(",") if p.strip()]

    print(f"\n{'page':<6}{'regime':<22}{'n_struct':>10}{'n_cells':>10}"
          f"{'inside':>8}{'hdr':>5}{'cont':>6}{'scm':>5}{'inl':>5}{'tbl':>5}")
    print("-" * 80)
    for page in pages:
        md_path = v2_config.DATA_DIR / args.patent / "all_pages" / f"page_{page:04d}.md"
        if not md_path.exists():
            print(f"  p{page}: no PaddleX md ({md_path})")
            continue
        md_text = md_path.read_text(errors="replace")
        # Parse text blocks (we don't need them for regime decisions yet)
        blocks = parse_paddlex_blocks(md_text)
        text_blocks = [(b.bbox, b.content) for b in blocks if b.type in ("text", "title")]

        # PaddleX table bboxes scaled to render DPI
        scale = SEG_DPI / 96.0
        paddlex_table_bboxes = _paddlex_table_bboxes_from_md(md_text, scale)

        if args.no_detector:
            cells = []
            structure_bboxes = []
        else:
            page_img = render_pdf_page(pdf_path, page)
            seg_dir = (v2_config.OUTPUT_DIR / "cropping_eval"
                       / "decimer_seg" / args.patent / f"page_{page:04d}")
            crops = segment_page(pdf_path, page, seg_dir)
            structure_bboxes = [c.bbox_px for c in crops]
            cells_objs = detect_table_cells(
                page_img, paddlex_table_bboxes,
                patent_id=args.patent, page=page,
            )
            cells = [c.bbox_px for c in cells_objs]

        regime, sig = classify_page_regime(
            paddlex_text=md_text,
            text_blocks=text_blocks,
            cells=cells,
            structure_bboxes=structure_bboxes,
            paddlex_table_bboxes=paddlex_table_bboxes,
        )
        print(f"p{page:<5d}{regime.value:<22}"
              f"{sig.n_structures:>10}{sig.n_cells:>10}"
              f"{sig.n_structures_inside_cells:>8}"
              f"{('Y' if sig.has_table_header else '-'):>5}"
              f"{('Y' if sig.has_table_continued else '-'):>6}"
              f"{('Y' if sig.has_scheme_marker else '-'):>5}"
              f"{sig.n_inline_headings:>5}"
              f"{('Y' if sig.has_paddlex_table_block else '-'):>5}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
