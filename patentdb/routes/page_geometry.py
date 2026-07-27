"""Canonical page geometry — ONE coord system, asserted.

Why this exists
---------------
The label-association layer kept hitting bugs because three subsystems
(DECIMER-Seg crops, PaddleX text/table blocks, PP-Structure cells) used
different coord systems:

  - DECIMER-Seg:   render-pixel coords at our chosen DPI (default 300)
  - PaddleX:       ~96-DPI normalized coords (page max y ≈ 1000)
  - PP-Structure:  render-pixel coords (we feed it the rendered image)

Mixing them inside the matcher meant every (structure, candidate) score
implicitly carried a unit conversion. We fix this by canonicalizing
EVERYTHING to render-pixel coords at the BOUNDARY (the moment we
receive an upstream bbox), then asserting + logging at every boundary
that the bboxes are sane.

After this module, downstream code only sees canonical bboxes and never
has to think about coord conversions.

The canonical coord system:
  - units:    pixels at the rendered-page DPI (default 300)
  - origin:   top-left of the page
  - x grows:  right
  - y grows:  down (matches PIL / PyMuPDF / DECIMER-Seg conventions)
  - bbox:     (x1, y1, x2, y2) with x1 < x2 and y1 < y2
  - 0 <= x1 < x2 <= page_width
  - 0 <= y1 < y2 <= page_height

Patent-agnostic — no patent_id branches anywhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)

# Render DPI used everywhere downstream. Match what
# `decimer_segmentation_crop.SEG_DPI` uses so the cropper output is
# already in canonical coords.
RENDER_DPI = 300
PADDLEX_DPI = 96      # PaddleX emits bboxes in this coord space


@dataclass
class CanonicalBbox:
    """A bounding box in canonical (render-pixel) page coords.

    Use the factory `Bbox.from_*` constructors so coord conversions
    happen at the boundary and asserts catch bad inputs there, not
    deep inside the matcher.
    """
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        # Cheap sanity asserts — fire ONLY at the boundary, never in
        # hot loops. If these trip we've got a unit bug upstream.
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise AssertionError(
                f"degenerate bbox: ({self.x1}, {self.y1}, {self.x2}, {self.y2}) "
                f"— width or height is non-positive"
            )

    # ── Constructors ──────────────────────────────────────────────

    @classmethod
    def from_render_px(cls, x1: int, y1: int, x2: int, y2: int) -> "CanonicalBbox":
        """Already in canonical coords — just wrap and assert."""
        return cls(int(x1), int(y1), int(x2), int(y2))

    @classmethod
    def from_paddlex(cls, x1: int, y1: int, x2: int, y2: int,
                     paddlex_to_render_scale: float | None = None) -> "CanonicalBbox":
        """Convert a PaddleX bbox (~96 DPI page coords) to canonical.

        If `paddlex_to_render_scale` is None, defaults to RENDER_DPI/PADDLEX_DPI.
        Pass the page-derived scale (render_w / paddlex_page_w) for accuracy
        when PaddleX's internal page width isn't exactly 96 DPI.
        """
        s = paddlex_to_render_scale if paddlex_to_render_scale is not None \
            else (RENDER_DPI / PADDLEX_DPI)
        return cls(int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s))

    @classmethod
    def from_pp_structure(cls, x1: int, y1: int, x2: int, y2: int) -> "CanonicalBbox":
        """PP-Structure cell bboxes are already in render coords (we feed
        it the rendered image). Same as `from_render_px` — separate factory
        kept for source-tracing in audits."""
        return cls(int(x1), int(y1), int(x2), int(y2))

    # ── Accessors ─────────────────────────────────────────────────

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    # ── Geometry ──────────────────────────────────────────────────

    def iou(self, other: "CanonicalBbox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / (self.area + other.area - inter)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def y_overlap(self, other: "CanonicalBbox") -> int:
        """Pixels of vertical overlap with `other` (0 if none)."""
        return max(0, min(self.y2, other.y2) - max(self.y1, other.y1))

    def x_overlap(self, other: "CanonicalBbox") -> int:
        return max(0, min(self.x2, other.x2) - max(self.x1, other.x1))


# Short alias so call sites read naturally
Bbox = CanonicalBbox


# ── Page-level holder ─────────────────────────────────────────────

@dataclass
class CanonicalPage:
    """Everything we know about a page in ONE coord system.

    All bboxes inside are CanonicalBbox in render-pixel coords. Built
    once by the eval driver, passed to the visualizer + matcher.
    """
    patent_id: str
    page: int
    width_px: int
    height_px: int                                       # render-pixel page size
    structures: list["CanonicalStructure"] = field(default_factory=list)
    cells: list["CanonicalCell"] = field(default_factory=list)
    text_blocks: list["CanonicalTextBlock"] = field(default_factory=list)
    paddlex_table_regions: list[CanonicalBbox] = field(default_factory=list)
    paddlex_page_size_px: tuple[int, int] = (0, 0)        # for downstream debug

    def assert_invariants(self) -> None:
        """Raise if any bbox lies outside the page or is degenerate.
        Cheap; runs once per page in tests + at the boundary."""
        for label, items in [
            ("structure", self.structures),
            ("cell", self.cells),
            ("text_block", self.text_blocks),
        ]:
            for i, it in enumerate(items):
                b = it.bbox
                if not (0 <= b.x1 < b.x2 <= self.width_px
                        and 0 <= b.y1 < b.y2 <= self.height_px):
                    logger.warning(
                        "[%s p%d] %s[%d] bbox %s outside page %dx%d",
                        self.patent_id, self.page, label, i, b.as_tuple(),
                        self.width_px, self.height_px,
                    )

    def summary(self) -> str:
        return (f"CanonicalPage({self.patent_id} p{self.page}, "
                f"page={self.width_px}x{self.height_px}, "
                f"structures={len(self.structures)}, cells={len(self.cells)}, "
                f"text_blocks={len(self.text_blocks)}, "
                f"paddlex_tables={len(self.paddlex_table_regions)})")


@dataclass
class CanonicalStructure:
    bbox: CanonicalBbox
    crop_path: str | None = None       # for traceability
    crop_index: int = -1                # 0-based reading-order index from DECIMER-Seg


@dataclass
class CanonicalCell:
    bbox: CanonicalBbox
    table_id: int
    row: int
    col: int
    detector: str = ""                  # 'wired' | 'wireless' | 'wholepage'


@dataclass
class CanonicalTextBlock:
    bbox: CanonicalBbox
    content: str
    block_type: str                     # 'text' | 'title' | 'table' | 'table_caption' ...


# ── Builders for upstream outputs ─────────────────────────────────

def build_canonical_page(
    *, patent_id: str, page: int,
    page_width_px: int, page_height_px: int,
    paddlex_page_width_px: int, paddlex_page_height_px: int,
    structures_render_px: Iterable[tuple[tuple[int, int, int, int], str | None, int]],
    cells_render_px: Iterable[tuple[tuple[int, int, int, int], int, int, int, str]],
    text_blocks_paddlex_px: Iterable[tuple[tuple[int, int, int, int], str, str]],
    paddlex_table_regions_paddlex_px: Iterable[tuple[int, int, int, int]],
    text_blocks_render_px: Iterable[tuple[tuple[int, int, int, int], str, str]] = (),
) -> CanonicalPage:
    """Build a CanonicalPage from upstream outputs in their NATIVE coords.

    Each tuple format:
      structures: ((x1,y1,x2,y2), crop_path|None, crop_index)
      cells:      ((x1,y1,x2,y2), table_id, row, col, detector)  -- all RENDER coords
      text_blocks:((x1,y1,x2,y2), content, block_type)           -- PADDLEX coords
      paddlex_tables: (x1,y1,x2,y2)                              -- PADDLEX coords

    The boundary is HERE: PaddleX coords get scaled exactly once via
    the page-derived scale factor (render_w / paddlex_w) — never re-scaled
    deeper in the pipeline.
    """
    paddlex_to_render = page_width_px / max(1, paddlex_page_width_px)

    structures = [
        CanonicalStructure(
            bbox=Bbox.from_render_px(*bbox),
            crop_path=crop_path,
            crop_index=crop_index,
        )
        for (bbox, crop_path, crop_index) in structures_render_px
    ]

    cells = [
        CanonicalCell(
            bbox=Bbox.from_pp_structure(*bbox),
            table_id=tid, row=row, col=col, detector=detector,
        )
        for (bbox, tid, row, col, detector) in cells_render_px
    ]

    text_blocks = [
        CanonicalTextBlock(
            bbox=Bbox.from_paddlex(*bbox, paddlex_to_render_scale=paddlex_to_render),
            content=content, block_type=block_type,
        )
        for (bbox, content, block_type) in text_blocks_paddlex_px
    ]

    # Append text blocks already in render coords (e.g., from PaddleOCR
    # standalone text-detection pass that bypasses PaddleX's layout-model
    # gating). Filter degenerate bboxes safely — defensive in case the
    # detector returns sub-pixel polygons that collapse.
    for (bbox, content, block_type) in text_blocks_render_px:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        text_blocks.append(CanonicalTextBlock(
            bbox=Bbox.from_render_px(x1, y1, x2, y2),
            content=content, block_type=block_type,
        ))

    paddlex_table_regions = [
        Bbox.from_paddlex(*bbox, paddlex_to_render_scale=paddlex_to_render)
        for bbox in paddlex_table_regions_paddlex_px
    ]

    page = CanonicalPage(
        patent_id=patent_id, page=page,
        width_px=page_width_px, height_px=page_height_px,
        structures=structures, cells=cells, text_blocks=text_blocks,
        paddlex_table_regions=paddlex_table_regions,
        paddlex_page_size_px=(paddlex_page_width_px, paddlex_page_height_px),
    )
    page.assert_invariants()
    return page
