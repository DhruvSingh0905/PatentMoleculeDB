"""Stage 7 — Output validator.

HARVEST occasionally pulls values from synthesis-prose pages (NMR
coupling constants like `J=7.3 Hz`, MS m/z values, mass-balance
percentages) and attributes them to compounds as assay measurements.
This is patent-agnostic — the FSM/LLM region detector can grab
anything that looks like "compound_id + number" rows, even when
the surrounding text is clearly synthesis description.

This module is a deterministic post-validator that drops any
(cid, assay_name, value) triple unless (cid, value) co-occur in
EITHER a MinerU `<table>` block OR Google Patents' description
text (when available). Two sources matter because MinerU OCR
sometimes corrupts assay tables — collapsing several cid cells into
one (`<td>252627282930</td>`), losing cid cells entirely, or merging
adjacent rows — while GP's description carries the same data as
clean flat text the FSM can parse. Requiring presence in EITHER
source keeps NMR/MS leak detection working without throwing out
correct values rescued from MinerU OCR corruption.

Falls open (returns input unchanged) when:
  - No ``<table>`` blocks AND no GP description available
  - The pages directory doesn't exist AND no GP cache (operating
    on cached output where neither source is on disk)

Patent-agnostic. Catches NMR-leak / MS-leak / page-number-leak
without enumerating every possible noise pattern.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


_MINERU_MARKUP_RE = re.compile(
    r"<\|[a-z/_]+\|>|\[\[[^\]]*\]\]",
    re.IGNORECASE,
)
_TABLE_BLOCK_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)


def _load_table_blocks(patent_id: str, data_dir: Path) -> list[str]:
    """Read every ``<table>...</table>`` block from the patent's
    markdown pages, with MinerU artifact tags stripped."""
    pages_dir = data_dir / patent_id / "all_pages"
    if not pages_dir.exists():
        return []
    pieces: list[str] = []
    for p in sorted(pages_dir.glob("page_*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        text = _MINERU_MARKUP_RE.sub("", text)
        pieces.append(text)
    full = "\n".join(pieces)
    return _TABLE_BLOCK_RE.findall(full)


def _load_gp_description(patent_id: str) -> str:
    """Return the Google Patents description text for `patent_id`, or
    "" if the GP cache doesn't exist. The description is GP's clean-text
    rendering of the patent body — used as a second validation source
    when MinerU's HTML tables are corrupted."""
    from .. import config
    gp_cache = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
    if not gp_cache.exists():
        return ""
    try:
        return json.loads(gp_cache.read_text()).get("description", "") or ""
    except (OSError, json.JSONDecodeError):
        return ""


def _value_strings(value: float) -> set[str]:
    """Plausible string representations of `value` that might appear
    in a table cell. Includes the natural decimal form plus rounded
    forms at 1-4 places, but NEVER coarser than the value's own
    precision (else we'd round 0.025 to "0" and match every cell
    that contains a bare zero — catastrophic false positive).
    Doesn't try cross-unit conversions; the extractor already
    normalized units.
    """
    out: set[str] = {f"{value:g}"}
    # Generate fixed-precision forms ONLY at precisions that don't
    # truncate the value. e.g. for 0.025 we produce {"0.025", "0.0250"}
    # but skip "0" (.0f) and "0.03" (.2f) since both lose information.
    for p in (1, 2, 3, 4):
        s = f"{value:.{p}f}"
        if abs(float(s) - value) < 1e-9:
            out.add(s)
    # Strip trailing zeros from the .Xf forms so "1.40" matches "1.4"
    out.update({s.rstrip("0").rstrip(".") for s in list(out) if "." in s})
    out.discard("")
    # Final guard: never emit "0" (or empty) as a candidate for a
    # nonzero value — it'd match arbitrary "0" cells.
    if value != 0:
        out.discard("0")
    return out


_CID_PREFIX_RE = re.compile(
    r"^(?:cpd\.?\s*no\.?|compound|example|cpd|ex\.?|cmp\.?\s*no\.?|cmp\.?)\s*",
    re.IGNORECASE,
)


def _value_near_cid(cid: str, value: float, table: str, window: int = 800) -> bool:
    """Check if `value` appears within `window` chars of a `<td>` cell
    that names `cid`. Cid match is tolerant: bare ``<td>5</td>`` AND
    prefixed forms ``<td>Compound 5</td>`` / ``<td>Cpd. No. 5</td>``
    both count. Value match accepts any later `<td>` cell containing the
    value's text form (possibly with a leading qualifier like ``<``/``>``).
    """
    val_strs = _value_strings(value)
    if not val_strs:
        return False
    cid_str = str(cid)
    # Find every `<td>...</td>` whose stripped content equals cid (case-
    # insensitive, prefix-tolerant). Walking cells with finditer and
    # filtering in Python keeps the regex simple and avoids brittle
    # alternations for every possible prefix.
    for cm in re.finditer(r"<td[^>]*>(.*?)</td>", table, re.DOTALL | re.IGNORECASE):
        cell = cm.group(1).strip()
        cell_norm = _CID_PREFIX_RE.sub("", cell).strip()
        if cell_norm.lower() != cid_str.lower():
            continue
        # Window AROUND the cid cell — table rows put cid + values
        # in adjacent cells within a few hundred chars.
        s = max(0, cm.start() - 200)
        e = min(len(table), cm.end() + window)
        chunk = table[s:e]
        for v in val_strs:
            # Cell-bounded match so "1.6" doesn't match "11.6"
            if re.search(
                rf"<td[^>]*>\s*[<>~≤≥]?\s*{re.escape(v)}\s*(?:[^\d<].*?)?</td>",
                chunk, re.IGNORECASE,
            ):
                return True
    return False


def _value_near_cid_in_flat_text(
    cid: str, value: float, text: str, window: int = 800,
) -> bool:
    """Check if `value` appears within `window` chars of a cid token in
    `text` (a flat string, no HTML structure). Used to validate against
    GP description text where tables get flattened to "25 0.061 (2)
    3.12 (1) 26 0.0043 (4) ..." sequences. Cid token boundary is enforced
    by non-digit lookahead/lookbehind so "25" doesn't match inside "1259".

    Default window is 800 chars: when GP flattens a multi-column assay
    table, the cid header row can be hundreds of chars from the matching
    data cell. A 200-char window was over-dropping legitimate rows on
    patents without MinerU `<table>` structure.
    """
    val_strs = _value_strings(value)
    if not val_strs:
        return False
    cid_str = str(cid)
    # Cid must appear as a standalone token — preceded by non-alphanum
    # (or start), followed by whitespace (typical in flat tables).
    cid_pat = rf"(?:^|[^\w]){re.escape(cid_str)}\s"
    for m in re.finditer(cid_pat, text):
        s = m.end()
        e = min(len(text), s + window)
        chunk = text[s:e]
        for v in val_strs:
            # Value with optional qualifier, followed by whitespace/punct/EOL.
            # The (?:^|[^\d.]) lookbehind avoids matching "0.061" inside
            # "10.0610" — important when values share leading digits.
            if re.search(
                rf"(?:^|[^\d.])[<>~≤≥]?\s*{re.escape(v)}(?:[^\d.]|$)",
                chunk,
            ):
                return True
    return False


# Patterns that mark a row's `source_quote` as clearly synthesis-prose
# rather than assay data. Match any of these → drop the row outright.
_JUNK_QUOTE_PATTERNS = [
    # NMR shift / coupling annotations
    re.compile(r"\bNMR\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*MHz\b", re.IGNORECASE),
    re.compile(r"\bJ\s*=\s*\d", re.IGNORECASE),
    re.compile(r"δ\s*\d"),
    re.compile(r"\bppm\b", re.IGNORECASE),
    re.compile(r"\bDMSO[-]?d\s*6\b", re.IGNORECASE),
    re.compile(r"\bCDCl\s*3\b", re.IGNORECASE),
    re.compile(r"\bCD3OD\b|\bMeOD\b", re.IGNORECASE),
    # Mass-spec annotations (m/z, [M+H]+, calc'd / found, exact mass)
    re.compile(r"\bm/z\b", re.IGNORECASE),
    re.compile(r"\[M\s*[+\-]\s*H\]"),
    re.compile(r"\bM\+\s*Na\b", re.IGNORECASE),
    re.compile(r"\bcalc(?:ulate)?d\b", re.IGNORECASE),
    re.compile(r"\bESI\b", re.IGNORECASE),
    re.compile(r"\bexact\s+mass\b", re.IGNORECASE),
    # Synthesis-yield / mmol annotations
    re.compile(r"\b\d+\s*mmol\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*%\s*yield\b", re.IGNORECASE),
    # Spectroscopic peak descriptors (s, d, t, m, dd, td, br s, ...)
    re.compile(r"\(\s*\d+H\s*,"),
    re.compile(r"\b(?:singlet|doublet|triplet|quartet|multiplet)\b", re.IGNORECASE),
]


def _looks_like_synthesis_prose(quote: str) -> bool:
    """True if the source_quote carries unmistakable markers of NMR /
    mass-spec / synthesis-yield prose. These rows are almost never real
    assay data — the FSM grabbed a number that happened to live near a
    cid token in a synthesis paragraph.
    """
    if not quote:
        return False
    for pat in _JUNK_QUOTE_PATTERNS:
        if pat.search(quote):
            return True
    return False


def filter_assays_to_table_blocks(
    assay_tables: dict[str, list[dict]],
    patent_id: str,
    data_dir: Path,
) -> tuple[dict[str, list[dict]], int]:
    """Default-keep validator. Strategy:

      1. Rows whose `validation_reason` starts with `pattern_library:`
         came from a deterministic regex run on the full patent text by
         the assay-pattern feedback loop. They're already grounded in a
         structural row match; always keep.

      2. Rows whose `source_quote` carries NMR / m/z / synthesis-yield
         markers are clear synthesis-prose leak. Drop them.

      3. Everything else: keep.

    This is intentionally LOOSER than the prior version, which required
    proving (cid, value) co-occurrence and silently dropped 60%+ of
    pattern-library rows on Z-cid patents. Downstream consumers
    (the workbook builder, BDB matcher) handle the residual noise via
    per-assay-name and per-value sanity checks.

    The optional MinerU `<table>` / GP description proximity check is
    still computed for diagnostic logging but no longer gates dropping.
    """
    filtered: dict[str, list[dict]] = {}
    n_dropped = 0
    n_pattern_kept = 0
    n_prose_dropped = 0
    for cid, arr in assay_tables.items():
        kept: list[dict] = []
        for a in arr:
            reason = a.get("validation_reason", "") or ""
            quote = a.get("source_quote", "") or ""
            if reason.startswith("pattern_library:"):
                kept.append(a)
                n_pattern_kept += 1
                continue
            if _looks_like_synthesis_prose(quote):
                n_prose_dropped += 1
                n_dropped += 1
                continue
            kept.append(a)
        if kept:
            filtered[cid] = kept

    if n_dropped or n_pattern_kept:
        logger.info(
            "%s: output_validator (loose mode) — kept %d rows total, "
            "dropped %d as synthesis-prose (NMR/MS markers), "
            "protected %d pattern-library rows",
            patent_id,
            sum(len(v) for v in filtered.values()),
            n_dropped,
            n_pattern_kept,
        )
    return filtered, n_dropped
