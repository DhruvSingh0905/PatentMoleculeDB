"""Layout-graph label association (Step 4 of the label-association plan).

Why this exists
---------------
Treat each page as a layout graph:
  - Nodes:   structure bboxes (DECIMER-Seg) + label-candidate text blocks
             (PaddleX/MinerU + cell-derived labels)
  - Edges:   scored candidate-pairs between each structure and each label
  - Goal:    max-weight bipartite matching subject to a confidence floor,
             with each label consumed at most once

Three strategies, all using the same graph framing — they differ only in
which edge-weight terms are enabled. Each is independently selectable
via `strategy=` so the bench can attribute every percentage point of
improvement to a specific algorithm.

Strategies
----------
- `cell_iou` (4a): structure ↔ cell IoU matching for table regimes.
  For each structure, find the cell with maximum IoU; assign the label
  whose text block is in the same row but in the compound-ID column.

- `proximity_direction` (4b): proximity + above/below scoring for non-table
  regimes (synthesis schemes, inline Examples). Format-agnostic.

- `layout_graph` (4c): regime-routed combination — runs cell_iou where
  cells exist, proximity_direction where they don't, merges results.

Hard rule: NO patent-specific code. The same scoring runs everywhere;
cell signals just don't fire on non-table pages, direction signals
dominate there.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


# ── Public dataclasses ──────────────────────────────────────────────

@dataclass
class StructureBox:
    """One DECIMER-Seg-detected structure on a page (bbox in render coords)."""
    bbox_px: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    crop_path: str | None = None          # for traceability


@dataclass
class TextBlock:
    """One text/title/table block from PaddleX/MinerU (bbox in PaddleX-OCR coords)."""
    bbox_px: tuple[int, int, int, int]
    content: str
    block_type: str = "text"              # 'text' | 'title' | 'table' | etc.


@dataclass
class Cell:
    """One table cell from Step 3 detector (bbox in render coords)."""
    bbox_px: tuple[int, int, int, int]
    table_id: int
    row: int
    col: int


@dataclass
class Label:
    """A label assigned to one structure."""
    text: str                             # the compound ID, e.g. "5A" or "Example 67"
    source: str                           # which scorer produced it (for audit)
    confidence: float                     # 0..1, normalized score
    runner_up: tuple[str, float] | None = None  # (text, score) of 2nd-best, if any


@dataclass
class AssociationProvenance:
    """Per-structure decision trace — what was tried and why we picked (or didn't)."""
    chosen_label: Label | None
    reason: Literal["matched", "low_confidence", "no_candidate"]
    top_candidates: list[tuple[str, float]]  # (text, score) for the top N
    scorer_used: str
    notes: str = ""


# ── Candidate label mining (shared) ─────────────────────────────────

# The only "domain knowledge" — matches any patent's compound-numbering
# convention. Bare-id pattern catches table-cell IDs like "5a" / "100AA";
# heading-style pattern catches "Example 5", "Cpd. No. 148", "Compound 5A".
_BARE_ID_RE = re.compile(r"^\s*(\d{1,4}[A-Za-z]{0,3})\s*$")
_LETTER_DIGIT_RE = re.compile(r"^\s*([A-Z]\d{1,4})\s*$")
_HEADING_ID_RE = re.compile(
    r"(?:Example|Cpd\.?\s*No\.?|Compound)\s*(\d{1,4}[A-Za-z]{0,3})",
    re.IGNORECASE,
)
# Whole-block "this IS a standalone label" shape — heading keyword
# (Example|Cpd.No.|Compound) + id + optional trailing colon, nothing else.
# Optional leading `#` allows for markdown-formatted headings emitted by
# PaddleX/MinerU as `# Example 563`.
#
# This is the SOLE gate for accepting heading-style candidates (no
# bbox-width fallback). Per user direction: "if the number and preceding
# two words are not alone, it's not the label." Inline prose mentions
# like "procedure described in example 561, employing 5-chloro-" must
# not slip into the candidate pool — they poison proximity scoring by
# offering a geographically-close label that doesn't actually correspond
# to its column-mate structure.
#
# Whitespace between the keyword and the ID is OPTIONAL (`\s*`) — PaddleOCR
# sometimes outputs "Cpd. No.350" with no space (observed on US10899738
# p155 above the cpd 350 structure). Strict `\s+` would silently drop
# those captions and surface as Category B "label missing".
_STANDALONE_HEADING_RE = re.compile(
    r"^\s*#?\s*(?:Example|Cpd\.?\s*No\.?|Compound)\s*\d{1,4}[A-Za-z]{0,3}\s*:?\s*$",
    re.IGNORECASE,
)


@dataclass
class LabelCandidate:
    """A candidate compound-ID label mined from text blocks."""
    text: str                             # the normalized compound ID
    bbox_px: tuple[int, int, int, int]    # bbox (see coord_space)
    raw_content: str                      # original block content for debug
    is_heading_style: bool                # True if matched via "Example N" pattern
    coord_space: str = "paddlex"          # "paddlex" (default) | "render"
                                           # cell-anchored candidates are
                                           # already in render coords; the
                                           # cell_iou scorer skips its scale-up
                                           # for those.


def _extract_table_row_ids(table_html: str) -> list[str]:
    """Pull the first-cell content of each `<tr>` from inline HTML.

    Returns the FIRST `<td>` text per row, in document order. The first
    cell of a compound table is conventionally the compound ID column
    (Cpd. No. / Compound No. / Example No.). Returns an empty list if
    nothing parses.
    """
    out: list[str] = []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL | re.IGNORECASE)
        if not cells:
            continue
        # Strip nested tags from the first cell, normalize whitespace
        first = re.sub(r"<[^>]+>", " ", cells[0])
        first = re.sub(r"\s+", " ", first).strip()
        out.append(first)
    return out


def _identify_line_number_column(text_blocks: list[TextBlock]) -> set[int]:
    """Return the indices of text blocks whose content is a patent
    LINE NUMBER (the 5-step numeric sequence printed in the gutter
    between columns of patent prose). Those are noise candidates that
    we drop before they pollute proximity scoring.

    Detection (generic, no patent-specific code):
      1. Find all blocks whose content is a pure positive integer
         that's a multiple of 5 in [5, 100].
      2. Cluster them by x-center (within a 60px tolerance).
      3. A cluster is a "line-number column" if it has ≥ 4 such
         multiples-of-5 AND they span ≥ 30% of the page-y range.
      4. All blocks in such clusters get filtered.

    This catches the standard patent line-number pattern (5, 10, 15,
    20, ... at a fixed x) without dropping legitimate compound IDs
    that happen to be small numbers (cpd 5, 10, etc.) — real compound
    IDs rarely come in dense vertical sequences at a single x.
    """
    candidates: list[tuple[int, int, int]] = []   # (block_idx, value, x_center)
    for i, tb in enumerate(text_blocks):
        content = (tb.content or "").strip()
        if content.isdigit() and 5 <= int(content) <= 100 and int(content) % 5 == 0:
            cx = (tb.bbox_px[0] + tb.bbox_px[2]) / 2
            candidates.append((i, int(content), int(cx)))
    if len(candidates) < 4:
        return set()

    # Cluster by x within tolerance
    candidates.sort(key=lambda t: t[2])
    clusters: list[list[tuple[int, int, int]]] = []
    cluster: list[tuple[int, int, int]] = []
    last_x = None
    for cand in candidates:
        if last_x is None or abs(cand[2] - last_x) <= 60:
            cluster.append(cand)
        else:
            clusters.append(cluster)
            cluster = [cand]
        last_x = cand[2]
    if cluster:
        clusters.append(cluster)

    drop_indices: set[int] = set()
    for cl in clusters:
        if len(cl) < 4:
            continue
        # Get y-spread of this cluster
        ys = [(text_blocks[i].bbox_px[1] + text_blocks[i].bbox_px[3]) / 2
              for (i, _v, _x) in cl]
        y_spread = max(ys) - min(ys)
        # Estimate page height from max y across all blocks
        page_h = max(tb.bbox_px[3] for tb in text_blocks) + 1
        if y_spread / page_h >= 0.30:
            drop_indices.update(idx for (idx, _v, _x) in cl)
    return drop_indices


# Chemistry-domain noise tokens — when an entire block content is just
# one of these, it's not a compound-ID label and should not enter the
# candidate pool. Patterns are short and well-defined; no patent-specific
# logic. Applied by filter_noise_text_blocks below.
_NOISE_TOKEN_PATTERNS = [
    re.compile(r"^\d{1,3}\s*h$",     re.IGNORECASE),  # "2h", "18 h"  reagent times
    re.compile(r"^\d{1,3}\s*min$",   re.IGNORECASE),  # "30 min"
    re.compile(r"^[Hh]rs?$"),                          # "h", "hr", "Hrs"
    re.compile(r"^S\d{1,3}$"),                         # "S1", "S19" synthesis-intermediate IDs
                                                       # (compound-like but a separate numbering
                                                       # scheme; if we want to surface them, route
                                                       # through a distinct candidate channel)
    re.compile(r"^[CNOSP]\d$"),                        # "C1", "N2" atom abbreviations
    re.compile(r"^[CNOSP][a-z]$"),                     # "Cl", "Br" halogens
    re.compile(r"^[A-Z][a-z]?$"),                      # defensive single atom/element
    re.compile(r"^0+$"),                               # "0", "00" stray zeros
]


def _matches_noise_token(content: str) -> bool:
    """True iff the (whitespace-normalized) content is exactly one of the
    blacklisted chemistry-domain noise tokens (reagent times, synthesis
    intermediate IDs, atom abbreviations, stray zeros). Filtered out of
    the candidate pool BEFORE compound-ID mining so we don't waste a
    candidate slot on (e.g.) "2h" from a reaction-time annotation.
    """
    plain = re.sub(r"\s+", "", content.strip())
    if not plain:
        return False
    return any(p.match(plain) for p in _NOISE_TOKEN_PATTERNS)


def filter_noise_text_blocks(text_blocks: list[TextBlock]) -> list[TextBlock]:
    """Drop blocks that are obviously not compound-ID labels:
      - patent line-number columns (5, 10, 15, ... in column gutter)
      - SHORT pure-numeric content in header / footer regions (page numbers)
      - chemistry-domain noise tokens (reagent times, atom abbreviations,
        synthesis-intermediate IDs like "S19", stray zeros)

    Used by both `mine_label_candidates` and `cell_label_candidates`
    so the noise filter applies consistently across all candidate
    sources. Returns a filtered list (preserves order).

    Header/footer rule (refined 2026-04-22, D2.E): pure-numeric content
    in the header or footer zone is dropped only if it's ≤ 2 chars
    long. Compound IDs in our corpus span 1–~700, so 3+ digit numerics
    in the header zone are more likely to be compound labels printed at
    the top of the page (e.g., US9718825 cpd 562/566) than page numbers.
    The downstream proximity scorer rejects them anyway when no nearby
    structure exists.
    """
    drop_indices = _identify_line_number_column(text_blocks)
    page_h_est = max((tb.bbox_px[3] for tb in text_blocks), default=1) / 0.95
    header_y = 0.13 * page_h_est
    footer_y = 0.92 * page_h_est

    out: list[TextBlock] = []
    for i, tb in enumerate(text_blocks):
        if i in drop_indices:
            continue
        content = tb.content or ""
        # D2.C: chemistry-domain noise tokens (anywhere on the page).
        if _matches_noise_token(content):
            continue
        cy = (tb.bbox_px[1] + tb.bbox_px[3]) / 2
        if cy < header_y or cy > footer_y:
            content_norm = re.sub(r"\s+", "", content.strip())
            # D2.E: only drop SHORT pure-numeric content as page numbers.
            # 3+ digit pure numerics in the header zone are kept as
            # potential compound IDs (e.g., 562, 566).
            if content_norm.isdigit() and len(content_norm) <= 2:
                continue
        out.append(tb)
    return out


def mine_label_candidates(
    text_blocks: list[TextBlock],
    table_region_bboxes: list[tuple[int, int, int, int]] | None = None,
    page_width_px: int | None = None,
) -> list[LabelCandidate]:
    """Pull compound-ID candidates out of every block.

    A text block contributes candidates if its content matches one of:
      - bare alphanumeric ID like `41` or `5a` (table-cell labels)
      - letter+digits like `S19` (synthesis intermediates)
      - heading-style mention like `Example 5` / `Cpd. No 148`
      - inline `<table>...<tr><td>X</td>...` HTML — emits one candidate
        per row using the FIRST `<td>` content (the compound-ID column)

    Inline-HTML candidates are placed at row-derived y-positions inside
    the surrounding table region. We use `table_region_bboxes` (the
    PaddleX `<|ref|>table<|/ref|>` block bboxes) when available; if a
    block contains `<table>` HTML and no surrounding table region is
    given, we fall back to the block's own bbox.

    Heading-style matches are flagged so the scorer can prefer them
    (less ambiguous than bare-id matches that might be page numbers).

    `page_width_px` (render-pixel width) enables the bbox-width gate on
    heading-style candidates: paragraph-wide blocks that happen to
    mention "Compound N" in passing are rejected. Callers should pass
    `page.width_px` from CanonicalPage. If None, the width gate is
    skipped and only the strict content-shape fast path admits
    heading-style candidates.
    """
    out: list[LabelCandidate] = []
    table_region_bboxes = table_region_bboxes or []

    # Apply noise filter (line-number columns, page numbers in header/footer)
    text_blocks = filter_noise_text_blocks(text_blocks)

    for tb in text_blocks:
        content = (tb.content or "").strip()
        if not content:
            continue

        # ---- Inline HTML <table>...<tr><td>...</td></tr></table> ----
        # Match anywhere in the block content. Often emitted in a
        # `table_caption` or `table` block by PaddleX; the placement
        # bbox is the surrounding `<|ref|>table<|/ref|>` region if any,
        # else the block's own bbox.
        html_tables = re.findall(r"<table[^>]*>(.*?)</table>", content,
                                  re.DOTALL | re.IGNORECASE)
        if html_tables:
            # Find a table region bbox that covers / overlaps this block
            tx1, ty1, tx2, ty2 = tb.bbox_px
            cover_bbox = tb.bbox_px
            for trb in table_region_bboxes:
                if (trb[0] <= tx1 + 5 and trb[2] >= tx2 - 5
                        and trb[1] <= ty1 + 50 and trb[3] >= ty2 - 5):
                    cover_bbox = trb
                    break
            tx1, ty1, tx2, ty2 = cover_bbox
            for table_inner in html_tables:
                row_texts = _extract_table_row_ids(table_inner)
                n_rows = len(row_texts)
                if n_rows == 0:
                    continue
                row_h = (ty2 - ty1) / n_rows
                for ri, raw in enumerate(row_texts):
                    m = _BARE_ID_RE.match(raw) or _LETTER_DIGIT_RE.match(raw)
                    if not m:
                        continue
                    cy_top = int(ty1 + ri * row_h)
                    cy_bot = int(ty1 + (ri + 1) * row_h)
                    cx2 = int(tx1 + (tx2 - tx1) * 0.12)
                    out.append(LabelCandidate(
                        text=m.group(1).strip(),
                        bbox_px=(tx1, cy_top, cx2, cy_bot),
                        raw_content=raw,
                        is_heading_style=False,
                    ))
            # Continue on — a table block may also have a non-HTML
            # caption text we should regex for headings (rare but cheap).

        # ---- Bare-id (only if the ENTIRE block is just an ID, after
        # stripping any inline HTML so we don't pick up the wrapping
        # table content as bare-id) ----
        plain = re.sub(r"<[^>]+>", " ", content)
        plain = re.sub(r"\s+", " ", plain).strip()
        m = _BARE_ID_RE.match(plain)
        if m and not html_tables:
            out.append(LabelCandidate(
                text=m.group(1).strip(),
                bbox_px=tb.bbox_px,
                raw_content=content,
                is_heading_style=False,
            ))
            continue
        m = _LETTER_DIGIT_RE.match(plain)
        if m and not html_tables:
            out.append(LabelCandidate(
                text=m.group(1).strip(),
                bbox_px=tb.bbox_px,
                raw_content=content,
                is_heading_style=False,
            ))
            continue

        # ---- Heading-style — STRICT standalone-shape only.
        # The whole normalized block content must match
        # _STANDALONE_HEADING_RE — i.e., be exactly the heading keyword
        # plus the ID (with optional `#` markdown marker and `:` suffix).
        # No bbox-width fallback. Per user direction (2026-04-29): inline
        # prose like "procedure described in example 561, employing..."
        # is NEVER a label, even when its bbox is narrow, because the
        # block bbox-center sits inside the prose flow rather than at
        # the actual label position above the structure.
        if _STANDALONE_HEADING_RE.match(plain):
            for hm in _HEADING_ID_RE.finditer(plain):
                cid = hm.group(1).strip()
                out.append(LabelCandidate(
                    text=cid,
                    bbox_px=tb.bbox_px,
                    raw_content=content,
                    is_heading_style=True,
                ))
    return out


# ── Cells from Step-3-derived labels ────────────────────────────────

def cell_anchored_table_candidates(
    cells: list[Cell],
    text_blocks: list[TextBlock],
) -> list[LabelCandidate]:
    """For each detected cell (in y-order), pair it with the corresponding
    row in any inline `<table>` HTML found in the page's text blocks.

    This fixes the drift problem from y-uniform candidate distribution:
    by anchoring candidates directly to cell bboxes, the cell_iou scorer's
    "same row" comparison is exact (both sides see the same cell).

    Algorithm:
      1. Sort cells by y-center (top-to-bottom).
      2. Find the inline-HTML <table> in any text/title/table_caption
         block, extract the per-row first-<td> texts.
      3. Pair cells[i] ↔ html_row_texts[i] in order.
      4. Filter pairs whose row text matches the bare-id regex; emit
         a LabelCandidate with the cell's bbox (in render coords).

    Returns an empty list if no inline table HTML is found, in which case
    the caller falls back to text-distributed candidates.
    """
    if not cells:
        return []
    # D2.8 (consistency fix): apply the same noise filter that
    # cell_label_candidates and mine_label_candidates use, so chemistry
    # domain noise / page-number / line-number gutter blocks don't
    # leak through the inline-HTML path. The HTML extraction itself
    # then operates on the filtered set.
    text_blocks = filter_noise_text_blocks(text_blocks)
    # Find HTML rows in any block
    html_rows: list[str] = []
    for tb in text_blocks:
        for table_inner in re.findall(r"<table[^>]*>(.*?)</table>",
                                       tb.content or "", re.DOTALL | re.IGNORECASE):
            html_rows = _extract_table_row_ids(table_inner)
            if html_rows:
                break
        if html_rows:
            break
    if not html_rows:
        return []

    cells_by_y = sorted(cells, key=lambda c: (c.bbox_px[1] + c.bbox_px[3]) / 2)
    out: list[LabelCandidate] = []
    for i, raw in enumerate(html_rows):
        if i >= len(cells_by_y):
            break
        m = _BARE_ID_RE.match(raw) or _LETTER_DIGIT_RE.match(raw)
        if not m:
            continue
        cell = cells_by_y[i]
        out.append(LabelCandidate(
            text=m.group(1).strip(),
            bbox_px=cell.bbox_px,                 # render coords; cell-anchored
            raw_content=raw,
            is_heading_style=False,
            coord_space="render",
        ))
    return out


def cell_label_candidates(cells: list[Cell],
                           text_blocks: list[TextBlock],
                           paddlex_to_render_scale: float) -> list[LabelCandidate]:
    """For each cell, see if any PaddleX text block whose content is a
    bare-id sits inside the cell. Promote those to candidates with the
    cell's render-coord bbox so the cell_iou scorer can match against them.

    This is needed because PaddleX text-block bboxes are in ~96-DPI coords
    but cell bboxes are in render coords (300 DPI). We unify by scaling
    text-block bboxes up to render coords for the inside-cell test.
    """
    out: list[LabelCandidate] = []
    # Same noise filter as text-candidate mining — keeps line numbers /
    # page numbers from being promoted to cell-anchored candidates.
    filtered_blocks = filter_noise_text_blocks(text_blocks)
    for cell in cells:
        cx1, cy1, cx2, cy2 = cell.bbox_px
        for tb in filtered_blocks:
            content = (tb.content or "").strip()
            m = _BARE_ID_RE.match(content) or _LETTER_DIGIT_RE.match(content)
            if not m:
                continue
            tx1 = tb.bbox_px[0] * paddlex_to_render_scale
            ty1 = tb.bbox_px[1] * paddlex_to_render_scale
            tx2 = tb.bbox_px[2] * paddlex_to_render_scale
            ty2 = tb.bbox_px[3] * paddlex_to_render_scale
            tcx = (tx1 + tx2) / 2
            tcy = (ty1 + ty2) / 2
            if cx1 <= tcx <= cx2 and cy1 <= tcy <= cy2:
                out.append(LabelCandidate(
                    text=m.group(1).strip(),
                    bbox_px=(int(tx1), int(ty1), int(tx2), int(ty2)),
                    raw_content=content,
                    is_heading_style=False,
                ))
    return out


# ── Candidate pool dedup ────────────────────────────────────────────

def dedup_candidates_by_text(
    candidates: list[LabelCandidate],
    structures: list[StructureBox],
) -> list[LabelCandidate]:
    """Collapse candidates that share a normalized text to one survivor.

    PaddleOCR often emits the same compound ID twice on a page (once as a
    standalone caption, once inside a nearby paragraph fragment that
    survives the mine-gate, or simply as a duplicate detection). Both
    candidates carry the same `text` but different bboxes. Hungarian
    assignment then consumes each as a distinct column, so the same
    label can be assigned to two structures (observed on p15, p17, p27,
    p90).

    Tiebreak (refined 2026-04-29 after p90 visual review):

      Prefer the instance positioned JUST-ABOVE a structure in the same
      column (the patent-layout convention for compound-ID labels).
      Concretely: for each instance compute the smallest gap to any
      structure-top whose x-range overlaps the candidate's x-range,
      where the candidate sits above the structure (cy < structure_top).
      Pick the instance with the smallest such above-anchor gap.

      If no instance has an above-anchor relationship anywhere, fall
      back to min L2 distance to any structure center (legacy behaviour).

    Normalization is lowercase + whitespace stripped so "5A" and "5a"
    collapse into the same group (the matcher is case-insensitive on
    compound IDs in practice, and the GT scorer handles case separately).

    No patent-specific logic.
    """
    if not candidates:
        return []
    if not structures:
        return list(candidates)

    def _norm(text: str) -> str:
        return re.sub(r"\s+", "", text.strip().lower())

    def _x_in_search_band(cand: LabelCandidate, s: StructureBox) -> bool:
        """Strict pixel-column overlap between candidate and structure
        bboxes (matches the scorer's precompute). Used to determine
        which structures a candidate could legitimately anchor above
        when picking among same-text duplicate instances.
        """
        cx1, _, cx2, _ = cand.bbox_px
        sx1, _, sx2, _ = s.bbox_px
        return min(cx2, sx2) > max(cx1, sx1)

    def _smallest_above_anchor_gap(cand: LabelCandidate) -> float:
        """Smallest positive (s_top − cand_cy) over structures whose
        width-bounded search band includes the candidate. inf if no
        such relationship exists.
        """
        cy = (cand.bbox_px[1] + cand.bbox_px[3]) / 2
        best = float("inf")
        for s in structures:
            if not _x_in_search_band(cand, s):
                continue
            gap = s.bbox_px[1] - cy
            if gap >= 0 and gap < best:
                best = gap
        return best

    def _min_sq_dist_to_any_struct(cand: LabelCandidate) -> float:
        cx = (cand.bbox_px[0] + cand.bbox_px[2]) / 2
        cy = (cand.bbox_px[1] + cand.bbox_px[3]) / 2
        best = float("inf")
        for s in structures:
            sx = (s.bbox_px[0] + s.bbox_px[2]) / 2
            sy = (s.bbox_px[1] + s.bbox_px[3]) / 2
            d2 = (cx - sx) ** 2 + (cy - sy) ** 2
            if d2 < best:
                best = d2
        return best

    def _tiebreak_key(cand: LabelCandidate) -> tuple[float, float]:
        """(above_anchor_gap, min_dist) — lexicographic. Instances
        with a real above-anchor relationship beat everything; ties
        within that broken by min-distance fallback."""
        return (_smallest_above_anchor_gap(cand),
                _min_sq_dist_to_any_struct(cand))

    groups: dict[str, list[LabelCandidate]] = {}
    for cand in candidates:
        key = _norm(cand.text)
        if not key:
            continue
        groups.setdefault(key, []).append(cand)

    # Preserve input order of the kept candidates (stable dict iteration
    # in py3.7+; groups are keyed by first-encountered normalized text).
    out: list[LabelCandidate] = []
    for group in groups.values():
        if len(group) == 1:
            out.append(group[0])
        else:
            out.append(min(group, key=_tiebreak_key))
    return out


# ── Coord canonicalization ──────────────────────────────────────────

def _to_render_coords(cand: LabelCandidate,
                       paddlex_to_render_scale: float) -> LabelCandidate:
    """Return a copy of `cand` whose bbox is in render-pixel coords.
    No-op if the candidate is already in `render` coord_space.
    """
    if cand.coord_space == "render":
        return cand
    s = paddlex_to_render_scale
    x1, y1, x2, y2 = cand.bbox_px
    return LabelCandidate(
        text=cand.text,
        bbox_px=(int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s)),
        raw_content=cand.raw_content,
        is_heading_style=cand.is_heading_style,
        coord_space="render",
    )


# ── Shared geometry helpers ─────────────────────────────────────────

def _normalize_bbox(bbox: tuple[int, int, int, int],
                    page_w: int, page_h: int) -> tuple[float, float, float, float]:
    return (bbox[0] / max(1, page_w), bbox[1] / max(1, page_h),
            bbox[2] / max(1, page_w), bbox[3] / max(1, page_h))


def _bbox_iou(a: tuple[int, int, int, int],
              b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (a_area + b_area - inter)


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


# ── Hungarian assignment (shared) ───────────────────────────────────

def _hungarian_assignment(
    score_matrix: list[list[float]],
) -> list[tuple[int, int, float]]:
    """Solve a max-weight bipartite matching.

    Returns list of (structure_idx, candidate_idx, score) where each
    structure is matched to at most one candidate. Uses scipy if
    available; falls back to a greedy assignment otherwise.

    Padding handles unequal counts: candidates fewer than structures
    leaves some structures unmatched (returned with candidate_idx=-1).
    """
    if not score_matrix or not score_matrix[0]:
        return []
    n_struct = len(score_matrix)
    n_cand = len(score_matrix[0])

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = -np.array(score_matrix)
        # Pad to square if needed
        if n_struct != n_cand:
            size = max(n_struct, n_cand)
            padded = np.zeros((size, size))
            padded[:n_struct, :n_cand] = cost
            cost = padded
        row_idx, col_idx = linear_sum_assignment(cost)
        out: list[tuple[int, int, float]] = []
        for r, c in zip(row_idx, col_idx):
            if r >= n_struct:
                continue
            if c >= n_cand:
                out.append((int(r), -1, 0.0))
            else:
                out.append((int(r), int(c), float(score_matrix[r][c])))
        return out
    except ImportError:
        # Greedy fallback: pick highest-score pair, remove that row+col, repeat
        used_struct, used_cand = set(), set()
        pairs: list[tuple[int, int, float]] = []
        flat = sorted(
            [(s, ci, score_matrix[s][ci])
             for s in range(n_struct) for ci in range(n_cand)],
            key=lambda x: -x[2],
        )
        for s, c, score in flat:
            if s in used_struct or c in used_cand:
                continue
            pairs.append((s, c, score))
            used_struct.add(s); used_cand.add(c)
        for s in range(n_struct):
            if s not in used_struct:
                pairs.append((s, -1, 0.0))
        return pairs


# ── Strategy 4a: cell_iou ───────────────────────────────────────────

def _score_proximity_around_structure(
    structures: list[StructureBox],
    candidates: list[LabelCandidate],
    cells: list[Cell] | None,
    page_size_px: tuple[int, int],
) -> list[list[float]]:
    """For each structure, the score for each candidate is based on how
    close the candidate is to the structure's expanded bounding box.

    This matches the actual page layout: compound ID labels are printed
    ADJACENT to (just above, below, or beside) the structure they label.
    The matcher's job is to pick the closest qualifying label per structure.

    Algorithm (per structure):
      1. Define an expanded search region = structure.bbox padded by a
         margin in all directions. Margin starts small and the score
         decays smoothly to 0 outside it — no hard threshold.
      2. For each candidate compound-ID block, compute the gap-distance
         from the candidate's center to the structure's bbox edge
         (negative = inside the structure; positive = outside).
      3. Score = 1.0 - (gap / max_gap), clamped to [0, 1]. Cells with
         center INSIDE the structure get full credit (1.0). Cells far
         outside get 0.

    Coord assumption: ALL bboxes are in canonical render coords (the
    caller has canonicalized via _to_render_coords / build_canonical_page).

    `cells` is currently unused — kept in the signature for future
    refinements (column-aware proximity for table layouts), but the
    proximity-around-structure logic alone handles tables, schemes, and
    inline-Example pages because the COMPOUND ID is always physically
    adjacent to its structure regardless of regime.
    """
    if not structures or not candidates:
        return [[0.0] * len(candidates) for _ in structures]

    page_w, page_h = page_size_px
    max_x_gap_px = 0.12 * page_w

    # Pre-compute for each candidate, which structure it most likely
    # labels under "label-printed-above-structure" semantics: the
    # structure whose y_top is just BELOW the candidate's y_center.
    # This is robust to DECIMER's loose bboxes that extend past row
    # boundaries — what matters is the GAP between label and structure
    # TOP, not whether the label sits inside the previous structure's
    # bbox area.
    #
    # Mirror "label-printed-below-structure" handled in parallel.
    cand_y = [(cb := cand.bbox_px, (cb[1] + cb[3]) / 2)[1] for cand in candidates]
    s_top = [s.bbox_px[1] for s in structures]
    s_bot = [s.bbox_px[3] for s in structures]
    s_x_ranges = [(s.bbox_px[0], s.bbox_px[2]) for s in structures]

    def _x_in_search_band(c_x1: int, c_x2: int, s_x1: int, s_x2: int) -> bool:
        """Strict pixel-column overlap between candidate bbox and
        structure bbox. The actual p155 cpd 350 case ("caption to top-
        right of structure") already x-overlaps the structure under
        strict semantics — the caption's right edge (1224) overlaps the
        structure's right edge (1216) — so no relaxation is needed.
        Wider bands were tried (half-width margin) but introduced
        regressions on multi-structure pages with shifted-column layouts
        (US11312727 p19: S1 is x-shifted left of S0/S2; the wider band
        causes "8A" to lose its rightful S1 fallback by becoming an
        in-band non-anchor for the wider S0/S2 expanded bands).
        """
        return min(c_x2, s_x2) > max(c_x1, s_x1)

    # For each candidate, find the structure whose top is just below
    # (label-above) and just above (label-below). Distance to that top/bottom
    # is the "anchor gap" used to score the label-structure pair.
    #
    # Anchor only to structures whose horizontal column range OVERLAPS the
    # candidate's column range. Patents are typically two-column layouts;
    # without this gate, a candidate at top-of-right-column gets anchored
    # to a top-of-left-column structure when that structure happens to have
    # a smaller raw y-gap, which then poisons the right-column structure's
    # score for that candidate.
    #
    # Fallback (refined 2026-04-29 after p27 regression): if a candidate
    # has NO X-overlap with ANY structure (compound-table layout where the
    # ID column and the structure column are different X-bands by design),
    # the column-aware gate would zero out every score. In that case,
    # consider all structures (drop the column gate) so cell-anchored
    # candidates in side columns can still pair with their row's structure
    # by Y proximity. Up/down only — no side-to-side proximity contributes.
    cand_above_struct: list[tuple[int | None, float]] = []  # (s_idx, gap)
    cand_below_struct: list[tuple[int | None, float]] = []
    for c_idx, cy in enumerate(cand_y):
        c_x1, _, c_x2, _ = candidates[c_idx].bbox_px
        # Decide whether to apply the column gate for this candidate.
        # If the candidate has at least one X-overlapping structure, we're
        # in a multi-column layout (separate columns of structures with
        # their labels) and the column gate prevents cross-column pairing.
        # If NO structure X-overlaps, the candidate is in a side column
        # (compound-ID column of a table) and we fall back to all structures.
        any_in_search_band = any(_x_in_search_band(c_x1, c_x2, *s_x_ranges[s_i])
                            for s_i in range(len(structures)))
        # label-above-structure: smallest positive (s_top - cy)
        best_above = (None, float("inf"))
        for s_i, top in enumerate(s_top):
            if any_in_search_band and not _x_in_search_band(c_x1, c_x2, *s_x_ranges[s_i]):
                continue
            gap = top - cy
            if gap >= 0 and gap < best_above[1]:
                best_above = (s_i, gap)
        cand_above_struct.append(best_above)
        # label-below-structure: smallest positive (cy - s_bot)
        best_below = (None, float("inf"))
        for s_i, bot in enumerate(s_bot):
            if any_in_search_band and not _x_in_search_band(c_x1, c_x2, *s_x_ranges[s_i]):
                continue
            gap = cy - bot
            if gap >= 0 and gap < best_below[1]:
                best_below = (s_i, gap)
        cand_below_struct.append(best_below)

    # Score: per (structure, candidate). High when the candidate's
    # nearest-structure-below or nearest-structure-above is THIS structure
    # AND the gap is small relative to typical row spacing.
    typical_gap_px = 0.15 * page_h     # ~10mm at 300 DPI; well past most label-to-structure gaps

    matrix: list[list[float]] = []
    for s_idx, s in enumerate(structures):
        sx1, sy1, sx2, sy2 = s.bbox_px
        row: list[float] = []
        for c_idx, cand in enumerate(candidates):
            cx = (cand.bbox_px[0] + cand.bbox_px[2]) / 2

            # X proximity: penalize labels far from the structure's x-band
            dx_gap = max(sx1 - cx, 0, cx - sx2)
            x_prox = max(0.0, 1.0 - dx_gap / max_x_gap_px)

            # Y signal: this candidate "anchors" to a specific structure
            # via above/below logic. If THAT structure is us, full credit
            # decayed by gap. If a DIFFERENT structure, score 0.
            y_signal = 0.0
            ab_idx, ab_gap = cand_above_struct[c_idx]
            if ab_idx == s_idx and ab_gap < typical_gap_px:
                y_signal = max(y_signal, 1.0 - ab_gap / typical_gap_px)
            bl_idx, bl_gap = cand_below_struct[c_idx]
            if bl_idx == s_idx and bl_gap < typical_gap_px:
                # Slightly de-prioritized vs above (compound IDs are
                # typically printed above their structures in patents)
                y_signal = max(y_signal, 0.9 - bl_gap / typical_gap_px)

            score = y_signal * x_prox
            if cand.is_heading_style:
                score += 0.02
            row.append(score)
        matrix.append(row)
    return matrix


# Backward-compatible alias so the layout_graph caller doesn't change.
_score_cell_iou = _score_proximity_around_structure


# ── Strategy 4b: proximity_direction ────────────────────────────────

def _score_proximity_direction(
    structures: list[StructureBox],
    candidates: list[LabelCandidate],
    page_w: int, page_h: int,
    patent_gt_id_set: set[str] | None,
) -> list[list[float]]:
    """Proximity + above/below + heading-specificity scoring.

    All bboxes are in canonical render coords (caller has already
    canonicalized via `_to_render_coords`). Both structure and candidate
    bboxes are normalized by the same render-page size.

    Edge weight = sum of normalized terms:
      - proximity:      up to +1.0 (inverse of normalized 2D distance)
      - direction_below/above: +0.5 if label is just above/below structure
                              and horizontally aligned
      - regex_specificity: +0.3 if heading-style match
      - unique_id_in_gt: +0.2 if candidate text matches a known compound
    """
    matrix: list[list[float]] = []
    for s in structures:
        sx1, sy1, sx2, sy2 = _normalize_bbox(s.bbox_px, page_w, page_h)
        scx, scy = _center((sx1, sy1, sx2, sy2))
        row: list[float] = []
        for cand in candidates:
            # Same page-coord system on both sides — no second normalization
            cx1, cy1, cx2, cy2 = _normalize_bbox(cand.bbox_px, page_w, page_h)
            ccx, ccy = _center((cx1, cy1, cx2, cy2))

            dx = ccx - scx
            dy = ccy - scy
            dist = (dx * dx + dy * dy) ** 0.5

            # Proximity: 1 / (1 + 5*dist), clamped at 0 if dist > 0.3
            proximity = 0.0 if dist > 0.30 else 1.0 / (1.0 + 5.0 * dist)

            # Direction bonus: candidate just above (dy < 0) or just below
            # (dy > 0) the structure, with horizontal centers within 0.05.
            direction_bonus = 0.0
            if abs(dx) < 0.05:
                if 0 < -dy < 0.10:
                    direction_bonus = 0.5     # above
                elif 0 < dy < 0.10:
                    direction_bonus = 0.5     # below

            specificity_bonus = 0.3 if cand.is_heading_style else 0.0
            gt_bonus = (0.2 if patent_gt_id_set
                        and cand.text.upper() in patent_gt_id_set else 0.0)

            score = proximity + direction_bonus + specificity_bonus + gt_bonus
            row.append(score)
        matrix.append(row)
    return matrix


# ── Confidence gate (shared) ────────────────────────────────────────

# Proximity-around-structure scores decay linearly from 1.0 (label inside/
# touching the structure) to 0.0 (label at max_gap_px away). 0.10 means
# "label was within 90% of max_gap" — generous floor, since the Hungarian
# assignment is already picking the BEST option per structure. We mostly
# rely on the margin gate to suppress ambiguous cases.
_TAU_SCORE = 0.10
_TAU_MARGIN = 0.02       # winner must beat runner-up by this much


def _apply_confidence_gate(
    structures: list[StructureBox],
    candidates: list[LabelCandidate],
    score_matrix: list[list[float]],
    pairs: list[tuple[int, int, float]],
    scorer_name: str,
) -> list[AssociationProvenance]:
    """Apply the confidence gate: winning candidate must score ≥ TAU_SCORE
    AND beat its runner-up by ≥ TAU_MARGIN. Otherwise emit Label=None
    so step 6 (LM fallback) and audit can pick it up.

    Returns one AssociationProvenance per structure (in the order the
    structures were given to the scorer).
    """
    out: list[AssociationProvenance] = []
    for s_idx, _, s in zip(range(len(structures)), structures, structures):
        # Find the pair for this structure
        pair = next((p for p in pairs if p[0] == s_idx), None)
        if pair is None or pair[1] < 0:
            out.append(AssociationProvenance(
                chosen_label=None, reason="no_candidate",
                top_candidates=[], scorer_used=scorer_name,
            ))
            continue
        cand_idx, score = pair[1], pair[2]
        # Compute top-3 candidates by score for this structure
        scores_for_s = [(ci, score_matrix[s_idx][ci]) for ci in range(len(candidates))]
        scores_for_s.sort(key=lambda x: -x[1])
        top = [(candidates[ci].text, sc) for ci, sc in scores_for_s[:3]]

        winner_score = score
        winner_text = candidates[cand_idx].text
        # Margin gate compares against the next-best DIFFERENT label text.
        # When PaddleOCR detects the same label twice (e.g., "5a" written
        # twice on the page), the runner-up has identical text and a
        # raw-score margin gate would falsely reject — but the matcher
        # is correctly going to assign that label regardless of which
        # candidate it picks.
        next_diff_score = 0.0
        for cid, sc in scores_for_s[1:]:
            if candidates[cid].text != winner_text:
                next_diff_score = sc
                break
        margin = winner_score - next_diff_score
        top = [(candidates[ci].text, sc) for ci, sc in scores_for_s[:3]]

        if winner_score < _TAU_SCORE:
            out.append(AssociationProvenance(
                chosen_label=None, reason="low_confidence",
                top_candidates=top, scorer_used=scorer_name,
                notes=f"winner_score={winner_score:.3f} < TAU_SCORE={_TAU_SCORE}",
            ))
        elif margin < _TAU_MARGIN and next_diff_score > 0:
            out.append(AssociationProvenance(
                chosen_label=None, reason="low_confidence",
                top_candidates=top, scorer_used=scorer_name,
                notes=(f"margin vs next-different-label={margin:.3f} "
                       f"< TAU_MARGIN={_TAU_MARGIN}"),
            ))
        else:
            out.append(AssociationProvenance(
                chosen_label=Label(
                    text=candidates[cand_idx].text,
                    source=scorer_name,
                    confidence=min(1.0, winner_score),
                    runner_up=(top[1][0], top[1][1]) if len(top) > 1 else None,
                ),
                reason="matched",
                top_candidates=top,
                scorer_used=scorer_name,
            ))
    return out


# ── Public API ──────────────────────────────────────────────────────

ScorerName = Literal["cell_iou", "proximity_direction", "layout_graph"]


def associate_labels(
    structures: list[StructureBox],
    text_blocks: list[TextBlock],
    cells: list[Cell] | None,
    page_size_px: tuple[int, int],
    paddlex_page_size_px: tuple[int, int],
    *,
    strategy: ScorerName = "layout_graph",
    regime: str | None = None,
    patent_gt_id_set: set[str] | None = None,
    paddlex_table_region_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> list[AssociationProvenance]:
    """Layout-graph bipartite label-association.

    Args:
        structures: DECIMER-Seg detections (bboxes in render coords).
        text_blocks: PaddleX/MinerU text + title blocks (PaddleX coords).
        cells: TableCellsDetection output (render coords). Empty/None for
               non-table pages.
        page_size_px: (w, h) of the rendered PDF page (300 DPI).
        paddlex_page_size_px: (w, h) of PaddleX's coord space (~96 DPI).
        strategy: which scorer to run (4a, 4b, or 4c).
        regime: optional PageRegime value (only consulted by 'layout_graph').
        patent_gt_id_set: optional set of known compound IDs in this patent
                          (used by 'proximity_direction' for a small
                          "this label appears in the GT" bonus).

    Returns:
        List of AssociationProvenance — one per structure, in the same
        order. Each carries the chosen Label (or None) + audit info.
    """
    if not structures:
        return []

    page_w, page_h = page_size_px
    paddlex_page_w, paddlex_page_h = paddlex_page_size_px
    paddlex_to_render_scale = page_w / max(1, paddlex_page_w)

    # ── Structure-size pre-filter ─────────────────────────────────
    # DECIMER-Seg occasionally detects TEXT artifacts (e.g., small compound
    # number labels like "37") as "structures". Those are 60-100 px; real
    # chemistry drawings at 300 DPI are at least ~200 px wide or tall.
    # Matching with these false-positive structures in the pool steals
    # label slots from real structures (Hungarian consumes each candidate
    # once), shifting all assignments off-by-one.
    #
    # We PASS THROUGH all structures to the return list (keeping their
    # indices stable), but mark tiny ones as `filtered_too_small` so the
    # matcher skips them. The label pool stays available for real structures.
    MIN_STRUCTURE_AREA_PX = 150 * 100    # 15k px² at 300 DPI
    structure_mask: list[bool] = []      # True = real structure, consider it
    for s in structures:
        x1, y1, x2, y2 = s.bbox_px
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        structure_mask.append(area >= MIN_STRUCTURE_AREA_PX)
    real_structures = [s for s, ok in zip(structures, structure_mask) if ok]

    text_candidates = mine_label_candidates(
        text_blocks,
        table_region_bboxes=paddlex_table_region_bboxes,
        page_width_px=page_w,
    )
    # Canonicalize: convert any PaddleX-coord candidate bbox to render
    # coords HERE so all downstream scorers see one coord system.
    text_candidates = [_to_render_coords(c, paddlex_to_render_scale)
                       for c in text_candidates]

    # Cell-anchored candidates: snap inline-HTML rows directly onto detected
    # cells so the cell_iou scorer's row-comparison is exact (same cell list
    # used to derive both struct row and candidate row). Already in render
    # coords.
    cell_anchored = (cell_anchored_table_candidates(cells, text_blocks)
                     if cells else [])
    cell_inside = (cell_label_candidates(cells, text_blocks, paddlex_to_render_scale)
                   if cells else [])
    cell_candidates = cell_anchored or cell_inside

    # Helper to re-expand a per-real-structure provenance list back to one
    # entry per ORIGINAL structure (with `filtered_too_small` placeholders
    # for the dropped tiny ones).
    def _reexpand(real_provs: list[AssociationProvenance]) -> list[AssociationProvenance]:
        out: list[AssociationProvenance] = []
        ri = 0
        for ok in structure_mask:
            if not ok:
                out.append(AssociationProvenance(
                    chosen_label=None, reason="no_candidate",
                    top_candidates=[], scorer_used=f"{strategy}[filtered_too_small]",
                    notes="structure bbox area < MIN_STRUCTURE_AREA_PX (likely a "
                          "DECIMER-Seg false positive on a label/text crop)",
                ))
            else:
                out.append(real_provs[ri])
                ri += 1
        return out

    if strategy == "cell_iou":
        # Now uses proximity-around-structure scorer (kept under the
        # 'cell_iou' strategy name for backward-compat with the bench CLI).
        candidates = cell_candidates if cell_candidates else text_candidates
        candidates = dedup_candidates_by_text(candidates, real_structures)
        if candidates and real_structures:
            matrix = _score_proximity_around_structure(
                real_structures, candidates, cells, page_size_px,
            )
        else:
            matrix = [[0.0] * len(candidates) for _ in real_structures]
        pairs = _hungarian_assignment(matrix)
        provs = _apply_confidence_gate(real_structures, candidates, matrix, pairs,
                                       "cell_iou")
        return _reexpand(provs)

    if strategy == "proximity_direction":
        if not text_candidates or not real_structures:
            return _reexpand([AssociationProvenance(
                chosen_label=None, reason="no_candidate",
                top_candidates=[], scorer_used="proximity_direction",
            ) for _ in real_structures])
        text_candidates = dedup_candidates_by_text(text_candidates, real_structures)
        matrix = _score_proximity_direction(
            real_structures, text_candidates,
            page_w=page_w, page_h=page_h,
            patent_gt_id_set=patent_gt_id_set,
        )
        pairs = _hungarian_assignment(matrix)
        provs = _apply_confidence_gate(
            real_structures, text_candidates, matrix, pairs,
            "proximity_direction",
        )
        return _reexpand(provs)

    # layout_graph (4c) — regime-routed combination
    return _reexpand(_layout_graph(
        real_structures, text_candidates, cell_candidates, cells,
        page_size_px, paddlex_page_size_px,
        paddlex_to_render_scale, regime, patent_gt_id_set,
    ))


def _ordered_zip_provenances(
    structures: list[StructureBox],
    candidates: list[LabelCandidate],
    scorer_name: str,
) -> list[AssociationProvenance]:
    """Document-order rank-zip: sort both lists by y, pair them by index.

    This is the most robust signal across regime types when:
      - N structures detected on the page
      - ≥ N distinct compound IDs appear (deduped by text, in document order)
      - The first N IDs in document order map to the N structures in y order

    Holds in practice for:
      - BLEED_TABLE  (IDs come from inline `<table><td>`, in row order)
      - TABLE_STRUCTURE (cell-anchored candidates already in y-order)
      - SCHEME       (sequential "Compound 8 / 8A / 8B" headings paired
                      with sequential structure drawings on the same page)
      - INLINE_EXAMPLE (per-Example heading + structure block)

    When `len(candidates) < len(structures)`, the extras get None — better
    than guessing geometrically when we have no signal.
    """
    n_struct = len(structures)

    # Dedup candidates by text, preserving document order (first appearance wins).
    # We sort by y to define document order; for a single column this matches
    # reading order, for multi-column pages it's a rough proxy. The y-sort
    # also gives us the y-rank for the rank-zip below.
    seen: set[str] = set()
    deduped: list[LabelCandidate] = []
    for cand in sorted(candidates, key=lambda c: (c.bbox_px[1] + c.bbox_px[3]) / 2):
        if cand.text in seen:
            continue
        seen.add(cand.text)
        deduped.append(cand)

    # Sort structures by y too
    s_indexed = sorted(enumerate(structures),
                       key=lambda iv: (iv[1].bbox_px[1] + iv[1].bbox_px[3]) / 2)

    out: list[AssociationProvenance] = [None] * n_struct  # type: ignore
    top_3_for_audit = [(c.text, 1.0) for c in deduped[:3]]
    for rank, (orig_idx, _s) in enumerate(s_indexed):
        if rank < len(deduped):
            cand = deduped[rank]
            out[orig_idx] = AssociationProvenance(
                chosen_label=Label(
                    text=cand.text, source=scorer_name, confidence=1.0,
                ),
                reason="matched",
                top_candidates=[(cand.text, 1.0)] + top_3_for_audit[:2],
                scorer_used=scorer_name,
            )
        else:
            out[orig_idx] = AssociationProvenance(
                chosen_label=None, reason="no_candidate",
                top_candidates=top_3_for_audit,
                scorer_used=scorer_name,
                notes=f"rank {rank} >= {len(deduped)} candidates",
            )
    return out


def _layout_graph(
    structures: list[StructureBox],
    text_candidates: list[LabelCandidate],
    cell_candidates: list[LabelCandidate],
    cells: list[Cell] | None,
    page_size_px: tuple[int, int],
    paddlex_page_size_px: tuple[int, int],
    paddlex_to_render_scale: float,
    regime: str | None,
    patent_gt_id_set: set[str] | None,
) -> list[AssociationProvenance]:
    """Strategy 4c — DECIMER-anchored proximity matcher.

    Per the user's framing: DECIMER-Seg detections are the anchor (near
    always correct). For each detection, search around its bbox for the
    nearest qualifying compound-ID label. Closer = more likely correct.

    Algorithm:
      1. Pool ALL candidates from cell-anchored + text sources.
      2. Score every (structure, candidate) by gap-distance from the
         candidate's center to the structure's bbox edge. Inside or
         touching the bbox = score 1.0; ~12% of page-side away = 0.
      3. Hungarian assignment so each candidate is consumed once.
      4. Confidence gate (in _apply_confidence_gate) emits None when
         the winner is too far / too close to a runner-up.

    No rank-zip, no regime branches. Same algorithm runs on any layout —
    table pages, schemes, inline examples — because the compound ID is
    always physically adjacent to its structure regardless of regime.

    This works as long as PaddleX detected the small label NEAR the
    structure. When PaddleX missed the label (Case B in D1), no matcher
    can recover it — that's an upstream OCR fix.
    """
    n_struct = len(structures)

    # Combine candidates from both sources. Cell-anchored come first so
    # they win ties (more spatially precise) when dedup picks one.
    all_candidates = list(cell_candidates) + list(text_candidates)
    if not all_candidates:
        return [AssociationProvenance(
            chosen_label=None, reason="no_candidate",
            top_candidates=[], scorer_used="layout_graph",
        ) for _ in structures]

    # Drop duplicate-text candidates before scoring so Hungarian can't
    # assign the same label to two different structures (p15, p17, p90).
    all_candidates = dedup_candidates_by_text(all_candidates, structures)

    matrix = _score_proximity_around_structure(
        structures, all_candidates, cells, page_size_px,
    )
    pairs = _hungarian_assignment(matrix)
    provenances = _apply_confidence_gate(
        structures, all_candidates, matrix, pairs,
        "layout_graph[proximity_around_structure]",
    )
    return provenances
