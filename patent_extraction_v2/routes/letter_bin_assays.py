"""Letter-grade potency-bin assay extractor (deterministic, no LLM).

Some patents report assay potency as *letter-grade bins* (``+``, ``++``,
``+++`` …) instead of precise numeric values, with a ``Key:`` legend mapping
each bin to an IC50 threshold/range, and per-bin lists of the Example IDs that
fall in it.  US11566007 is the canonical case: 13 tables (one per RAS-mutant
panel), ~7,600 (compound × table) records, only 3 compounds with a precise
numeric value — everything else is binned.  BindingDB curated those bins into
midpoint numbers; that is a *derivation*, not a measurement.

This module parses those tables and emits ``AssayResult``-shaped dicts that
carry the bin as an **honest range** — never a fabricated point:

    value_numeric = None                     # no invented midpoint
    value_raw     = "0.01-0.1" | "<0.01" | "≥10"
    qualifier     = "range" | "<" | ">"
    value_low     = 0.01  (literal patent threshold; None if open below)
    value_high    = 0.1   (literal patent threshold; None if open above)
    source        = "letter_bin"

The bin edges are read from the patent's own legends (not hard-coded).  Within
a table, the highest plus-level present is treated as open-ended (``≥edge``)
and ``+`` as open-below (``<edge``); intermediate levels are closed ranges —
exactly what the legends state.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── glyph / mojibake normalisation ──────────────────────────────────────────
# load_patent_description can hand back ≥/≤ as UTF-8-bytes-read-as-Latin-1.
_GLYPHS = {
    "â‰¥": "≥", "â‰¤": "≤",            # rendered form
    "â‰¥": "≥",          # raw 3-char form of ≥ (U+2265)
    "â‰¤": "≤",          # raw 3-char form of ≤ (U+2264)
}


def _fix(s: str) -> str:
    for k, v in _GLYPHS.items():
        s = s.replace(k, v)
    return s


# A bin marker is 1-5 '+' characters, standing alone as a token.
_BIN_RE = re.compile(r"^\++$")


def _norm_cid(cid: str) -> str:
    """Canonicalise an example ID to the pipeline's unpadded convention.

    Bin tables zero-pad for column alignment (``A028``) while example_index /
    the numeric assay tables use the unpadded form (``A28``).  Stripping the
    leading zeros lets bin records merge into the SAME assay_tables key as the
    compound's structure, so the workbook's per-cid lookup finds them.
    ``A001`` → ``A1``, ``A028`` → ``A28``, ``A89B`` → ``A89B``.
    """
    m = re.match(r"^([A-Za-z]+)0*(\d+)([A-Za-z]?)$", cid)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else cid
# Example IDs look like A208, A89A, AB12 — an upper-alpha prefix then digits.
_ID_RE = re.compile(r"^[A-Z]{1,2}\d{1,4}[A-Z]?$")
# Words that mark the end of a bin-list region inside a table body.
_STOP_RE = re.compile(
    r"\b(Key|Determination|Protocol|Note|TABLE|Synthesis|Example\b)", re.I
)


def _legend_boundaries(text: str) -> list[float]:
    """Collect the bin-edge thresholds (µM) from the legend definitions.

    The scale is read from the patent's own ``IC50 <sign> N uM`` threshold
    expressions (both ``IC50 ≥ N`` and ``N > IC50`` orderings), never assumed.
    Anchoring on IC50/EC50 avoids picking up buffer concentrations or other
    stray ``N uM`` figures in nearby protocol text.
    """
    text = _fix(text)
    edges: set[float] = set()
    pat_after = re.compile(r"[IE]C\s*50\s*[<>≥≤=]\s*(\d+(?:\.\d+)?)\s*[µu]?M")
    pat_before = re.compile(r"(\d+(?:\.\d+)?)\s*[µu]?M\s*[<>≥≤=]\s*[IE]C\s*50")
    for m in pat_after.finditer(text):
        edges.add(float(m.group(1)))
    for m in pat_before.finditer(text):
        edges.add(float(m.group(1)))
    return sorted(e for e in edges if 0 < e < 1e6)


def _bin_range(level: int, max_level: int, edges: list[float]):
    """Map a plus-level (1-indexed) to (low, high, qualifier, raw_str).

    edges = sorted bin boundaries, e.g. [0.01, 0.1, 1, 10].  Level *k* spans
    ``[edges[k-2], edges[k-1])``; level 1 is open-below (``<edges[0]``) and the
    table's top level is open-above (``≥edges[max_level-2]``).
    """
    def g(x):  # tidy number formatting
        return f"{x:g}"

    if not edges:
        return (None, None, None, "")
    # clamp indices into the edge list
    lo_i, hi_i = level - 2, level - 1
    if level == 1:
        hi = edges[0]
        return (None, hi, "<", f"<{g(hi)}")
    if level >= max_level:                      # top present bin → open above
        lo = edges[min(lo_i, len(edges) - 1)]
        return (lo, None, ">", f"≥{g(lo)}")
    lo = edges[min(lo_i, len(edges) - 1)]
    hi = edges[min(hi_i, len(edges) - 1)]
    if hi <= lo:                                # degenerate → open above
        return (lo, None, ">", f"≥{g(lo)}")
    return (lo, hi, "range", f"{g(lo)}-{g(hi)}")


def _caption(body: str) -> str:
    """Build a clean assay/target name from a table body's leading text.

    e.g. "Additional H358 Cell Viability assay data (K-Ras G12C, IC50, uM):
    IC50* Examples …"  →  "H358 Cell Viability K-Ras G12C IC50".
    """
    cut = len(body)
    for mark in ("Examples", "IC50*", "IC50 *"):
        i = body.find(mark)
        if 0 <= i < cut:
            cut = i
    seg = body[:cut]
    seg = seg.replace("(", " ").replace(")", " ")             # flatten parens
    seg = re.sub(r"\b(Additional|assay|data|results?|[µu]?M|nM)\b", " ", seg,
                 flags=re.I)
    seg = re.sub(r"[,:*—-]", " ", seg)
    seg = re.sub(r"\s+", " ", seg).strip()[:80].strip()
    if seg and not re.search(r"[IE]C\s*50", seg):
        seg = f"{seg} IC50"
    seg = re.sub(r"(IC\s*50)(\s+IC\s*50)+", r"\1", seg)        # collapse dups
    return seg or "potency_bin"


def extract_letter_bin_assays(
    patent_id: str, text: str | None = None
) -> dict[str, list[dict]]:
    """Parse letter-bin potency tables → ``{example_id: [record, …]}``.

    Returns an empty dict for patents that report numeric values (no bin
    legend), so it is safe to call unconditionally in the pipeline.
    """
    if text is None:
        from ..core.patent_text import load_patent_description

        text, _ = load_patent_description(patent_id, prefer_format="auto")
    text = _fix(text or "")

    edges = _legend_boundaries(text)
    if not edges:
        return {}

    # ── Pass 1: parse every bin table into {level: [ids]} ───────────────────
    tables: list[tuple[str, int, dict[int, list[str]]]] = []
    parts = re.split(r"(TABLE\s+\d+)", text)
    for i in range(1, len(parts), 2):
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # Cut the body at the first stop-word AFTER the example region begins,
        # so the legend / next-section text can't leak in.
        exm = re.search(r"Examples?\b", body)
        if not exm:
            continue
        region = body[exm.end():]
        stop = _STOP_RE.search(region)
        if stop:
            region = region[: stop.start()]
        # Walk tokens: '+'-runs set the current bin level; IDs attach to it.
        levels: dict[int, list[str]] = {}
        cur = 0
        for tok in re.split(r"[\s,]+", region):
            if not tok:
                continue
            if _BIN_RE.match(tok):
                cur = len(tok)
                levels.setdefault(cur, [])
            elif cur and _ID_RE.match(tok):
                levels[cur].append(tok)
        if sum(len(v) for v in levels.values()):
            tables.append((_caption(body), max(levels), levels))

    if not tables:
        return {}

    # ── Dominant example-ID prefix(es) ──────────────────────────────────────
    # _ID_RE also matches "IC50", RAS mutants ("G12C") and cell lines
    # ("SW1573").  Real example IDs share one alpha prefix ("A"); keep only
    # prefixes that are a non-trivial fraction of the most common one, which
    # drops the stray non-ID tokens without hard-coding domain vocabulary.
    from collections import Counter

    pref = Counter(
        m.group(1).upper()
        for _, _, levels in tables
        for ids in levels.values()
        for cid in ids
        if (m := re.match(r"([A-Za-z]{1,2})\d", cid))
    )
    if not pref:
        return {}
    dom_count = pref.most_common(1)[0][1]
    keep = {p for p, c in pref.items() if c >= max(2, 0.02 * dom_count)}

    # ── Pass 2: emit honest range records ───────────────────────────────────
    out: dict[str, list[dict]] = {}
    for name, max_level, levels in tables:
        seen_in_table: set[str] = set()        # one record per (cid, table)
        for level in sorted(levels):
            low, high, qual, raw = _bin_range(level, max_level, edges)
            if qual is None:
                continue
            for cid in levels[level]:
                m = re.match(r"([A-Za-z]{1,2})\d", cid)
                if not m or m.group(1).upper() not in keep:
                    continue
                ncid = _norm_cid(cid)
                if ncid in seen_in_table:
                    continue
                seen_in_table.add(ncid)
                out.setdefault(ncid, []).append(
                    {
                        "assay_name": name,
                        "value_numeric": None,     # honest: no fabricated point
                        "unit": "µM",
                        "qualifier": qual,
                        "n_runs": None,
                        "value_raw": raw,
                        "value_low": low,
                        "value_high": high,
                        "bin": "+" * level,
                        "source": "letter_bin",
                    }
                )

    n_rec = sum(len(v) for v in out.values())
    logger.info(
        "%s: letter-bin extractor — %d tables, %d compounds, %d records "
        "(edges=%s, prefixes=%s)",
        patent_id, len(tables), len(out), n_rec, edges, sorted(keep),
    )
    return out


# ── Geomean letter-grade de-fabrication ─────────────────────────────────────
# Some patents (US11292791) state a prose legend ("a value greater than 1 µM
# and less than 1000 µM is marked '+'") and a per-compound grade table. The FSM
# collapsed each grade to the geometric-mean midpoint of its bracket
# (sqrt(low·high)) and stored that as a precise number — a fabricated point. We
# reverse that here: parse the legend, recompute each bracket's geomean, and
# replace any matching value with the honest range the legend actually states.
_LEGEND_RANGE_RE = re.compile(
    r"value\s+(?:of\s+)?greater\s+than(?:\s+or\s+equal\s+to)?\s*([\d.]+)\s*[µμu]?M"
    r"\s+and\s+less\s+than(?:\s+or\s+equal\s+to)?\s*([\d.]+)\s*[µμu]?M"
    r"\s+is\s+marked\s*[^+\d]{0,6}(\++)",   # [^+\d]{0,6} skips mojibake quotes
    re.I,
)


def _parse_grade_legend(patent_id: str, text: str | None = None):
    """Parse a prose grade legend → list of (low, high, marker) in µM."""
    if text is None:
        from ..core.patent_text import load_patent_description

        text, _ = load_patent_description(patent_id, prefer_format="auto")
    text = _fix(text or "")
    out = []
    for m in _LEGEND_RANGE_RE.finditer(text):
        lo, hi, marker = float(m.group(1)), float(m.group(2)), m.group(3)
        if hi > lo:
            out.append((lo, hi, marker))
    return out


def dehallucinate_geomean_grades(
    patent_id: str, assay_tables: dict, text: str | None = None
) -> int:
    """In-place: replace fabricated geomean-midpoint values with honest ranges.

    For each grade bracket in the patent's prose legend, computes the geometric
    mean and rewrites any assay record whose ``value_numeric`` matches it (±5%)
    to ``value_numeric=None`` + ``value_low``/``value_high`` + ``value_raw`` +
    ``source="letter_bin"``. No-op for patents without such a legend (returns 0).
    """
    legend = _parse_grade_legend(patent_id, text)
    if not legend:
        return 0
    gmm = []
    for lo, hi, marker in legend:
        gmm.append(((lo * hi) ** 0.5, lo, hi, f"{lo:g}-{hi:g}", marker))
    n = 0
    for rows in assay_tables.values():
        for r in rows:
            v = r.get("value_numeric")
            if not v or r.get("source") == "letter_bin":
                continue
            for gm, lo, hi, raw, marker in gmm:
                if abs(v - gm) / gm < 0.05:
                    r["value_numeric"] = None
                    r["value_low"] = lo
                    r["value_high"] = hi
                    r["value_raw"] = raw
                    r["qualifier"] = "range"
                    r["bin"] = marker
                    r["source"] = "letter_bin"
                    n += 1
                    break
    if n:
        logger.info(
            "%s: de-fabricated %d geomean letter-grade values → honest ranges "
            "(legend=%s)",
            patent_id, n, [(l, h, mk) for l, h, mk in legend],
        )
    return n
