"""Stage 5 — the LLM realigner, behind `config.ASSAY_REALIGN_ENABLED`.

It called itself the ALWAYS-FIRE realigner and meant it: one LLM call per
chunk of every region the detector emits, on every patent, with no feature
flag — bounded only by a fingerprint cache and `PER_PATENT_REALIGN_CAP`.
That commitment was made when the input was OCR of a picture of a document;
OCR left the tree in `43d037e`, and the same tables now arrive as USPTO
OASIS/CALS cells.

Measured before gating it (22 corpus patents, replayed from cache, zero paid
calls — see `tests/test_realign_gate.py` for the full trace): it costs $0.55
per new patent and 270 chunks corpus-wide, and gating it off LOSES 9,358
(cid, assay, value) triples of which 8,010 are reproduced by
`sources.uspto_assays.extract_from_patent`, and 1,615 compounds of which
1,610 are. Net: 5 compounds. Turning it off also GAINS 2,466 rows, because
its rows occupy (compound, assay) slots that `_merge_into(gap_fill_only=
True)` then denies to the free pattern library.

The gate lives INSIDE `realign_region` and again inside
`adaptive_extraction_cache.realign_table_via_llm`, not in
`pipeline._process_region` — `43d037e` removed two dead functions that had
carried the only `LLM_RECOVERY_ENABLED` check in their module, and a gate on
the caller is undone by the next caller.

Cache hits are gated too. A fingerprint hit is free, but it is still this
tier's output, and the measurement above says those free rows displace the
free pattern library's. OFF means off for both.

This module reuses two existing pieces of infrastructure:
  - `core.adaptive_extraction_cache.realign_table_via_llm` — the
    actual LLM call (Sonnet, structured-JSON output)
  - `core.adaptive_extraction_cache._parse_realign_json` — the
    truncation-tolerant JSON parser
  - `core.adaptive_extraction_cache.AdaptiveExtractionCache` — the
    JSON-backed fingerprint cache
  - `core.adaptive_extraction_cache.fingerprint_table_shape` — the
    fingerprint hash function (re-used to keep the cache compatible
    across pipeline versions)

Multi-chunking: regions over `max_chunk_chars` are split into
overlapping pieces (50 row hits each, 5-row overlap) and the LLM
is called per chunk. Chunks share a header context so column-name
inference is stable across calls.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from ..adaptive_extraction_cache import (
    AdaptiveExtractionCache,
    RealignedRow,
    fingerprint_table_shape,
    realign_table_via_llm,
)
from .region_detector import Region
from ..cost_tracker import cost_tracker

logger = logging.getLogger(__name__)


# ── Chunker ──────────────────────────────────────────────────────


# Row-detection regex for finding chunk boundaries (NOT extraction). Must
# count a row whose first data cell is EITHER a number ("Z1 5.2") OR a
# letter grade ("Z1 D" — A/B/C/D/E binned assays), and accept prefixed
# compound ids ("Z1", "I-0020"). Counting only numeric-first rows made
# US11254686's 645-row letter-grade table count as 36 rows → 1 chunk → the
# realigner sent all 16k chars in one LLM call and truncated at ~79 rows.
_CHUNK_ROW_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"([A-Z]{0,3}-?\d{1,4}[A-Za-z]{0,3})"
    r"\s+"
    r"(?:[A-E](?![A-Za-z])|[<>≤≥~≈]?\s*\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


# Unit-bearing line for header context. Anchor for prepending column
# schema to every chunk so the LLM has the full header even on chunk #2+.
_UNIT_RE = re.compile(
    r"[\(\[]\s*(?:nM|μM|uM|µM|mcM|mM|pM|M|%)\s*[\)\]]",
    re.IGNORECASE,
)


def chunk_region(
    region_text: str,
    *,
    rows_per_chunk: int = 50,
    overlap_rows: int = 5,
) -> list[str]:
    """Split a region into overlapping LLM-sized chunks.

    Each chunk has a HEADER context (the line bearing UNIT_PATTERN
    annotations, with surrounding context) prepended so the LLM can
    map column positions to canonical assay names regardless of which
    chunk it's processing.
    """
    rows = list(_CHUNK_ROW_RE.finditer(region_text))
    if len(rows) <= rows_per_chunk:
        return [region_text]

    # Locate header context — the line containing UNIT_PATTERN
    unit_match = _UNIT_RE.search(region_text)
    if unit_match:
        h_start = max(0, unit_match.start() - 300)
        h_end = min(len(region_text), unit_match.end() + 200)
        header_context = region_text[h_start:h_end]
    else:
        # Fall back to first 600 chars
        header_context = region_text[:min(600, rows[0].start())]

    chunks: list[str] = []
    step = max(1, rows_per_chunk - overlap_rows)
    for i in range(0, len(rows), step):
        chunk_rows = rows[i:i + rows_per_chunk]
        if not chunk_rows:
            break
        chunk_start = chunk_rows[0].start()
        chunk_end = min(len(region_text), chunk_rows[-1].end() + 100)
        chunks.append(header_context + "\n\n" + region_text[chunk_start:chunk_end])
        if i + rows_per_chunk >= len(rows):
            break
    return chunks


# ── Column-shift guard ──────────────────────────────────────────
#
# `chunk_region` locates chunk boundaries with `_CHUNK_ROW_RE`, which matches
# `<token> <number>`. Over a table flattened to one cell per line — the shape
# Google Patents renders, `...\n\n436\n\n8.9\n\n27\n\n8941\n\n437\n\n5.1...` —
# the non-overlapping scan matches VALUE PAIRS (`27 8941`) just as readily as
# row starts (`437 5.1`). Every 45th match is therefore an arbitrary cell, and
# a chunk beginning there opens on the tail of a row: the model reads the
# stream one cell early and every compound in that chunk is published against
# the wrong column.
#
# Measured on US10544143 (2026-08-04): 55 of 1,019 compounds in the TLR7/8/9
# tables shipped shifted, compound 437 among them — TLR7 8941 nM (compound
# 436's TLR9) where the patent prints 5.1 nM, a >1,700-fold error attributed
# to the wrong target. Two cached realigner responses cover those rows, one
# correct and one shifted; the shifted one came from the GP description, which
# `pipeline._merge_into` treats as authoritative, so it won.
#
# The guard needs positive PROOF of the shift, not a plausibility score: it
# drops a row only when the row's own values reproduce the cells starting one
# token BEFORE its compound id, and no occurrence of that id is followed by
# them. A row the source cannot confirm either way is left alone — rounding,
# reordered columns and `NT` cells all make a bare "values don't match" test
# reject correct rows, and this module's rule is that a missing assay is
# recoverable while a wrong one is not.

_NUM_CELL_RE = re.compile(
    r"(?<![A-Za-z0-9._-])"
    r"(?:&gt;|&lt;|&le;|&ge;|[<>≤≥~≈])?\s*"
    r"(\d+(?:\.\d+)?)"
    r"(?![A-Za-z0-9.])"
)


def _numeric_cells(text: str) -> tuple[list[float], list[str]]:
    """Every numeric cell of `text`, in reading order, as (values, raw)."""
    values: list[float] = []
    raws: list[str] = []
    for m in _NUM_CELL_RE.finditer(text):
        raws.append(m.group(1))
        values.append(float(m.group(1)))
    return values, raws


def _row_values(row: RealignedRow) -> list[float]:
    """The row's numeric values in the order the model emitted its columns."""
    out: list[float] = []
    for v in (row.values or {}).values():
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return []          # a non-numeric cell breaks positional proof
    return out


def drop_column_shifted_rows(
    rows: list[RealignedRow], region_text: str,
) -> list[RealignedRow]:
    """Drop rows proven to be read one cell early. See the note above."""
    if not rows or not region_text:
        return rows
    values, raws = _numeric_cells(region_text)
    if not values:
        return rows
    at: dict[str, list[int]] = {}
    for i, raw in enumerate(raws):
        at.setdefault(raw, []).append(i)

    kept: list[RealignedRow] = []
    for row in rows:
        cid = (row.compound_id or "").strip()
        vals = _row_values(row)
        # Two values minimum: with one, "shifted" and "the previous row's
        # value happens to repeat" are the same observation.
        if len(vals) < 2 or cid not in at:
            kept.append(row)
            continue
        k = len(vals)
        aligned = shifted = False
        for i in at[cid]:
            if values[i + 1:i + 1 + k] == vals:
                aligned = True
                break
            if i >= 1 and [values[i - 1]] + values[i + 1:i + k] == vals:
                shifted = True
        if aligned or not shifted:
            kept.append(row)
    n_dropped = len(rows) - len(kept)
    if n_dropped:
        logger.warning(
            "llm_realigner: dropped %d/%d realigned row(s) whose values "
            "reproduce the cells one token BEFORE their compound id — "
            "chunk boundary opened mid-row",
            n_dropped, len(rows),
        )
    return kept


# ── LLM call wrapper ────────────────────────────────────────────


@dataclass
class LLMResult:
    """Output of `realign_region(...)`."""
    rows: list[RealignedRow] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    fingerprint: str = ""
    cache_hit: bool = False     # True if returned from cache
    n_chunks: int = 0           # number of LLM calls made (0 on cache hit)


def realign_region(
    region: Region,
    *,
    patent_id: str | None = None,
    cache: AdaptiveExtractionCache | None = None,
) -> LLMResult:
    """LLM realigner with fingerprint-cached results. OFF unless asked.

    Args:
        region: A Region from `detect_regions(...)`.
        patent_id: For provenance in the cache (optional).
        cache: Reuse an existing cache object (saves load on hot path);
               creates a new one if None.

    Returns:
        LLMResult with rows + units. Empty rows when the tier is disabled
        or the LLM fails (caller proceeds with FSM-only extraction).
    """
    from .. import config as _cfg
    if not _cfg.ASSAY_REALIGN_ENABLED:
        # Including the cache-hit path — see the module docstring.
        return LLMResult(rows=[], units={}, notes="gated_off",
                         fingerprint="", cache_hit=False, n_chunks=0)

    if cache is None:
        cache = AdaptiveExtractionCache()

    fp = _build_fingerprint(region.text)
    # CROSS-PATENT CONTAMINATION FIX: include patent_id in the cache key.
    # Previously `f"plaintext:{fp}"` meant patent A's cached rows (with
    # patent A's compound IDs, patent A's assay names like RORγ Binding)
    # would be returned to patent B if their fingerprints collided. Verified
    # contamination: 14 different patents share the cache file, with
    # patent-specific row data stored under a "structural" fingerprint. The
    # tradeoff for correctness is more LM calls — accepted, since cross-
    # patent contamination silently poisons every downstream report.
    cache_key = f"plaintext:{patent_id or 'unknown'}:{fp}"
    # Hydration guard import (used below).
    from ..assay_name_guard import is_valid_assay_name

    # Cache hit?
    cached_entry = cache._cache.get(cache_key)
    if cached_entry:
        try:
            raw_rows = cached_entry["rule"].get("rows", [])
            rows = []
            for r in raw_rows:
                # Filter legacy/cached rows for invalid assay names — even
                # though the new realigner-output guard prevents writes,
                # the cache file may still contain pre-guard entries.
                values = {k: v for k, v in (r.get("values") or {}).items()
                          if is_valid_assay_name(k)}
                if not values:
                    continue
                quals = {k: v for k, v in (r.get("qualifiers") or {}).items()
                         if is_valid_assay_name(k)}
                n_runs = {k: v for k, v in (r.get("n_runs") or {}).items()
                          if is_valid_assay_name(k)}
                rows.append(RealignedRow(
                    compound_id=r.get("compound_id", ""),
                    values=values, qualifiers=quals, n_runs=n_runs,
                ))
            units = {k: v for k, v in (cached_entry["rule"].get("units", {}) or {}).items()
                     if is_valid_assay_name(k)}
            notes = cached_entry["rule"].get("notes", "")
            cached_entry["n_uses"] = cached_entry.get("n_uses", 0) + 1
            cached_entry["last_used"] = str(date.today())
            cache._save()
            logger.info(
                "llm_realigner: cache HIT fp=%s rows=%d patent=%s",
                fp[:8], len(rows), patent_id,
            )
            # Also on the cache-hit path: every shipped artifact replays
            # from here, so a guard that only ran on fresh calls would never
            # reach the rows that are already wrong on disk.
            rows = drop_column_shifted_rows(rows, region.text)
            return LLMResult(
                rows=rows, units=units, notes=notes,
                fingerprint=fp, cache_hit=True, n_chunks=0,
            )
        except Exception as e:
            logger.warning("llm_realigner: cache hydrate failed for %s: %r — re-firing", fp[:8], e)
            # Fall through to LLM call

    # Cache miss → fire LLM (one call per chunk)
    chunks = chunk_region(region.text)
    merged_rows: list[RealignedRow] = []
    merged_units: dict[str, str] = {}
    seen_cids: set[str] = set()

    for chunk_idx, chunk_text in enumerate(chunks):
        # Per-patent realigner budget gate (its own cap, separate from the LM
        # cap). A pathological table can't run the realigner past
        # PER_PATENT_REALIGN_CAP; cache hits above this loop stay free.
        if patent_id and cost_tracker.patent_realign_exceeded(patent_id):
            from .. import config as _cfg
            logger.warning(
                "llm_realigner: COST-GATED on %s at chunk %d/%d — realigner "
                "spend $%.3f >= cap $%.2f; %d chunks skipped (table partly "
                "extracted). Raise PER_PATENT_REALIGN_CAP to continue.",
                patent_id, chunk_idx + 1, len(chunks),
                cost_tracker.patent_realign_spend(patent_id),
                _cfg.PER_PATENT_REALIGN_CAP, len(chunks) - chunk_idx,
            )
            break
        result = realign_table_via_llm(
            chunk_text, max_text_chars=6000, max_tokens=8000,
            patent_id=patent_id or "",
        )
        if result is None:
            logger.warning(
                "llm_realigner: chunk %d/%d returned no rows for %s fp=%s",
                chunk_idx + 1, len(chunks), patent_id, fp[:8],
            )
            continue
        for r in result.rows:
            cid_key = (r.compound_id or "").strip().lower()
            if not cid_key or cid_key in seen_cids:
                continue
            seen_cids.add(cid_key)
            merged_rows.append(r)
        for k, v in result.units.items():
            merged_units.setdefault(k, v)
        logger.info(
            "llm_realigner: chunk %d/%d added %d rows (total=%d) for %s",
            chunk_idx + 1, len(chunks), len(result.rows), len(merged_rows), patent_id,
        )

    if merged_rows:
        cache._cache[cache_key] = {
            "rule": {
                "rows": [
                    {
                        "compound_id": r.compound_id,
                        "values": r.values,
                        "qualifiers": r.qualifiers,
                        "n_runs": r.n_runs,
                    }
                    for r in merged_rows
                ],
                "units": merged_units,
                "notes": f"{len(chunks)}-chunk extraction",
            },
            "first_observed": patent_id or "unknown",
            "n_uses": 1,
            "last_used": str(date.today()),
        }
        cache._save()
        logger.info(
            "llm_realigner: NEW cache entry fp=%s rows=%d chunks=%d patent=%s",
            fp[:8], len(merged_rows), len(chunks), patent_id,
        )

    return LLMResult(
        rows=drop_column_shifted_rows(merged_rows, region.text),
        units=merged_units,
        notes=f"{len(chunks)}-chunk extraction" if merged_rows else "llm_failed",
        fingerprint=fp,
        cache_hit=False,
        n_chunks=len(chunks),
    )


# ── Fingerprint builder ─────────────────────────────────────────


def _build_fingerprint(region_text: str) -> str:
    """Build a structural fingerprint for caching.

    Reuses `fingerprint_table_shape` from the existing adaptive
    cache module. Inputs are derived from the FIRST few row matches
    in the region: synthetic headers (compound id token + first 6
    surface tokens), row count, sample cell signatures.

    Includes a vocabulary-hash suffix so that when the LLM cid-prefix
    discovery (or any other discovery) appends entries to
    `data/assay_vocabulary.discoveries.json`, prior cached results
    are invalidated — otherwise the tokenizer would never re-parse
    the same chunk with the newly-recognized prefixes.

    Generic — independent of patent ids, target names, compound ids.
    """
    rows = list(_CHUNK_ROW_RE.finditer(region_text))
    if len(rows) < 3:
        # Fall back to a hash over the leading 4000 chars
        synth_headers = ["plain_text_table"]
        sample_cells = [region_text[:60]]
        base_fp = fingerprint_table_shape(synth_headers, len(rows), sample_cells)
    else:
        first = rows[0]
        synth_headers = ["plain_text_table"] + [
            t for t in re.split(r"\s+", first.group(0).strip()) if t
        ][:6]
        sample_cells = [r.group(0)[:40] for r in rows[:5]]
        base_fp = fingerprint_table_shape(synth_headers, len(rows), sample_cells)
    return f"{base_fp}_{_vocab_fingerprint()}"


def _vocab_fingerprint() -> str:
    """Hash the canonical + auto-loaded COMPOUND_ID_PREFIX surface forms.
    Stable across runs unless discoveries change. Cheap — 8-char prefix
    of the SHA-256 over a sorted, joined surface-form string.
    """
    try:
        from ..compound_id import _load_prefixes
        joined = "|".join(_load_prefixes())
    except Exception:
        joined = ""
    import hashlib
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]
