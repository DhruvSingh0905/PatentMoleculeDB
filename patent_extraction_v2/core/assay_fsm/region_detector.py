"""Stage 1 — Region detection.

Find candidate assay-table regions in normalized page text using
*structural* signals only:
  1. An assay keyword is present (IC50/EC50/Ki/Kd/CC50/AC50/...)
  2. A unit annotation is present nearby ((nM)/(μM)/(mM)/...)
  3. Multiple compound-id-like + numeric-sequence rows are present,
     packed densely (median gap < 60 chars between consecutive
     row-id positions — distinguishes assay tables from
     structure-listing tables with long IUPAC names interleaved).

Returns 0+ regions per page. **NEVER early-exits with empty.** If no
candidate region is found, an empty list is returned and the
orchestrator simply has nothing to extract from this page — which is
fine for non-assay pages.

Patent-agnostic — uses only the structural signals above. No
hardcoded patent IDs, target names, or compound numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .vocabulary import AssayVocabulary


@dataclass(frozen=True)
class Region:
    """A candidate assay-table region in the source page."""
    start: int
    end: int
    text: str
    n_row_hits: int        # number of compound-id + number rows detected


# Compound-id + ≥2 numeric-cell pattern. Used to count candidate rows
# inside a region — NOT for final extraction (the tokenizer does that).
# Tolerant of qualifiers, n_runs annotations, and known null markers
# between values. Non-greedy {2,5}? on the value count to avoid
# swallowing the next compound's id; lookahead requires the next thing
# to be either a compound-id-shape, a known null marker, or end-of-text.
_VALUE_CELL = (
    r"(?:"
    r"[<>≤≥~≈]?\s*\d+(?:\.\d+)?"          # qualifier? + number
    r"(?:\s*\(\d+\))?"                     # optional (n_runs)
    r"(?:\s*±\s*\d+(?:\.\d+)?)?"           # optional ± stddev
    r"|(?:nt|nd|na|N\.?T\.?|N\.?D\.?|N/?A)"  # null markers
    r"|[—–\-*]"                            # dash placeholders
    r")"
)

_ROW_DETECT_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"([A-Z]?\d{1,4}[A-Za-z]{0,3})"        # compound id
    r"\s+"
    r"(" + _VALUE_CELL + r"(?:\s+" + _VALUE_CELL + r"){1,4}?)"
    r"(?=\s+[A-Z]?\d{1,4}[A-Za-z]{0,3}\s|\s+(?:nt|nd)\b|$|\n)",
    re.IGNORECASE,
)

# Quick keyword presence check. Pulled from canonical vocab classes
# but compiled here for speed.
_ASSAY_KEYWORD_RE = re.compile(
    r"\b(?:IC[\s_-]?50|EC[\s_-]?50|CC[\s_-]?50|AC[\s_-]?50|XC[\s_-]?50"
    r"|GI[\s_-]?50|MIC[\s_-]?50?|IC[\s_-]?90|EC[\s_-]?90"
    r"|K_?[id]|pIC[\s_-]?50|pK_?[id])\b",
    re.IGNORECASE,
)

# Unit annotation pattern — accepts μM/uM/µM/nM/mM/pM/M/percent
# inside parens or brackets. Must have already been mojibake-repaired.
_UNIT_RE = re.compile(
    r"[\(\[]\s*(?:nM|μM|uM|µM|mcM|mM|pM|M|%)\s*[\)\]]",
    re.IGNORECASE,
)


# Loose-fallback signals — used when strict row-shape detection finds
# zero clusters. The LLM is the safety net per the architectural
# commitment: "LLM ALWAYS fires when there's any reason to think
# there's data". These signals are deliberately broad.

# Direct value+unit pattern — captures "12 nM", "0.5 μM", etc.
_VALUE_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?\s*(?:nM|μM|uM|µM|mcM|mM|pM)(?![A-Za-z])",
    re.IGNORECASE,
)

# Letter-grade row pattern — used by patents that bin assays into
# A/B/C/D/E grades instead of reporting numeric values
# (e.g., "1. A 2. A 3. B 4. A ..." or "1 B 5 A 7 B ...").
_LETTER_GRADE_ROW_RE = re.compile(
    r"(?:^|[\s\n])(\d{1,4})\.?\s+([A-E])(?=\s|$|\n)",
    re.MULTILINE,
)


def detect_regions(
    text: str,
    vocab: AssayVocabulary | None = None,    # not currently used; reserved for future vocab-aware detection
    *,
    min_rows: int = 3,
    cluster_gap_threshold: int = 250,
    max_region_chars: int = 30000,
    structure_table_max_median_gap: int = 60,
    enable_loose_fallback: bool = True,
) -> list[Region]:
    """Find candidate assay-table regions in normalized page text.

    Args:
        text: Already-normalized page text (Stage 0 output).
        vocab: Reserved for future use; current detector relies only
               on the assay-keyword and unit regexes.
        min_rows: Minimum row hits to qualify a cluster.
        cluster_gap_threshold: Max chars between two row hits to keep
            them in the same cluster. 250 chars = ~3-4 short rows of
            slack to bridge interleaved noise.
        max_region_chars: Hard cap on a single emitted region — the
            chunker downstream subdivides for LLM calls.
        structure_table_max_median_gap: If the median gap between
            row hits in a cluster exceeds this, the cluster is
            disqualified (it's a structure-listing table with IUPAC
            names interleaved, not an assay table).

    Returns:
        List of Region triples. EMPTY LIST is a valid return — the
        orchestrator handles it as "no assay data on this page".
        Never raises, never returns None.
    """
    if not text:
        return []

    hits = [(m.start(1), m.end(0)) for m in _ROW_DETECT_RE.finditer(text)]
    if len(hits) < min_rows:
        # Strict path can't fire without enough row hits.
        # Fall through to the loose-fallback below — DON'T early-return
        # here. The architectural commitment is "LLM always fires when
        # there's any reason to think there's data".
        if enable_loose_fallback:
            return _detect_loose_regions(text, max_region_chars=max_region_chars)
        return []

    # Cluster row hits by proximity
    clusters: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for h in hits:
        if not current:
            current.append(h)
            continue
        if h[0] - current[-1][1] <= cluster_gap_threshold:
            current.append(h)
        else:
            if len(current) >= min_rows:
                clusters.append(current)
            current = [h]
    if len(current) >= min_rows:
        clusters.append(current)

    regions: list[Region] = []
    for cluster in clusters:
        cstart = cluster[0][0]
        cend = cluster[-1][1]

        # Unit annotation must exist within ~1500 chars before, 200 after
        win_start = max(0, cstart - 1500)
        win_end = min(len(text), cend + 200)
        window_text = text[win_start:win_end]
        if not _UNIT_RE.search(window_text):
            continue
        if not _ASSAY_KEYWORD_RE.search(window_text):
            continue

        # Disqualify structure-listing tables (sparse rows = IUPAC
        # names interleaved). Median-gap threshold is the right signal.
        if len(cluster) >= 3:
            id_starts = [h[0] for h in cluster]
            gaps = sorted(b - a for a, b in zip(id_starts[:-1], id_starts[1:]))
            median_gap = gaps[len(gaps) // 2]
            if median_gap > structure_table_max_median_gap:
                continue

        # Bound the region: include 800 chars header context + 200
        # chars trailing slack, capped at max_region_chars.
        rstart = max(0, cstart - 800)
        rend = min(len(text), cend + 200)
        if rend - rstart > max_region_chars:
            rend = rstart + max_region_chars

        regions.append(
            Region(
                start=rstart, end=rend,
                text=text[rstart:rend],
                n_row_hits=len(cluster),
            )
        )

    # ── Loose fallback ────────────────────────────────────────────
    # If strict row-shape detection found zero regions, try a much
    # looser detector. The LLM is the safety net — it always fires
    # when there's ANY indication of assay data, even if our FSM
    # can't tokenize the format. Patents that hit this path:
    #   - Letter-grade assay tables ("1. A 2. A 3. B ...") that don't
    #     match the numeric-value row regex
    #   - Sparse value+unit mentions in prose paragraphs without
    #     repeated row structure
    #   - New table formats we haven't seen yet
    #
    # The strict path is preferred because it's narrower; the loose
    # path widens the net only when strict produces nothing. Same
    # downstream — chunker + LLM realigner + cross-validator handle
    # both region types identically.
    if regions or not enable_loose_fallback:
        return regions

    return _detect_loose_regions(text, max_region_chars=max_region_chars)


def _detect_loose_regions(
    text: str,
    *,
    max_region_chars: int = 30000,
) -> list[Region]:
    """Fallback: emit regions wherever there's any assay-keyword +
    value/letter-grade signal. Used only when strict row-shape
    detection produces zero regions.

    Two trigger paths:
      1. Numeric path — clusters of `<num> <unit>` patterns near an
         assay keyword (IC50/EC50/Ki/Kd/CC50/...).
      2. Letter-grade path — clusters of `<id>. <A-E>` row hits near
         an assay keyword (e.g., the "TABLE A. 1. A 2. A 3. B ..."
         format used by some kinase patents).

    Either path emits a region. The chunker downstream will split it
    for LLM consumption. The LLM does the actual extraction; this
    function only decides "is there enough signal to fire the LLM?"
    """
    if not text or not _ASSAY_KEYWORD_RE.search(text):
        return []

    # Path 1: numeric value+unit cluster
    val_hits = [m.start() for m in _VALUE_UNIT_RE.finditer(text)]
    # Path 2: letter-grade row cluster
    grade_hits = [m.start() for m in _LETTER_GRADE_ROW_RE.finditer(text)]

    out: list[Region] = []
    out.extend(_cluster_to_regions(
        text, val_hits, kind="numeric_loose",
        min_hits=3, gap_threshold=400, max_region_chars=max_region_chars,
    ))
    out.extend(_cluster_to_regions(
        text, grade_hits, kind="letter_grade",
        min_hits=5, gap_threshold=120, max_region_chars=max_region_chars,
    ))

    # Deduplicate overlapping regions (a region from path 1 may
    # overlap one from path 2). Keep the larger of any overlap pair.
    out.sort(key=lambda r: (r.start, -r.end))
    deduped: list[Region] = []
    for r in out:
        if deduped and r.start < deduped[-1].end:
            # overlapping — keep the wider region
            if r.end - r.start > deduped[-1].end - deduped[-1].start:
                deduped[-1] = r
        else:
            deduped.append(r)
    return deduped


def _cluster_to_regions(
    text: str,
    hit_offsets: list[int],
    *,
    kind: str,
    min_hits: int,
    gap_threshold: int,
    max_region_chars: int,
) -> list[Region]:
    """Cluster hit offsets into Region objects."""
    if len(hit_offsets) < min_hits:
        return []

    clusters: list[list[int]] = []
    cur: list[int] = []
    for h in hit_offsets:
        if not cur:
            cur.append(h)
        elif h - cur[-1] <= gap_threshold:
            cur.append(h)
        else:
            if len(cur) >= min_hits:
                clusters.append(cur)
            cur = [h]
    if len(cur) >= min_hits:
        clusters.append(cur)

    out: list[Region] = []
    for cluster in clusters:
        rstart = max(0, cluster[0] - 800)
        rend = min(len(text), cluster[-1] + 800)
        if rend - rstart > max_region_chars:
            rend = rstart + max_region_chars
        # Require an assay keyword in the broader window so we don't
        # emit regions over pure synthesis prose
        if not _ASSAY_KEYWORD_RE.search(text[rstart:rend]):
            continue
        out.append(Region(
            start=rstart, end=rend,
            text=text[rstart:rend],
            n_row_hits=len(cluster),
        ))
    return out
