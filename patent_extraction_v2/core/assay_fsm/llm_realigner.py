"""Stage 5 — ALWAYS-FIRE LLM realigner.

Architectural commitment: this LLM call fires on every region the
detector emits, *the first time a fingerprint is seen*. Cache hits
return the cached result for free. Cache misses fire the LLM,
cache the result, and persist.

There is NO `_is_garbled_assay_page` heuristic gate, no "fire only
when baseline produced 0 rows" pattern. The LLM is co-equal with
the FSM, not a fallback.

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

logger = logging.getLogger(__name__)


# ── Chunker ──────────────────────────────────────────────────────


# Same row-detection regex as `region_detector._ROW_DETECT_RE`,
# kept in sync. Used here only to find chunk boundaries; not for
# extraction.
_CHUNK_ROW_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"([A-Z]?\d{1,4}[A-Za-z]{0,3})"
    r"\s+"
    r"(?:[<>≤≥~≈]?\s*\d+(?:\.\d+)?)",
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
    """Always-fire LLM realigner with fingerprint-cached results.

    Args:
        region: A Region from `detect_regions(...)`.
        patent_id: For provenance in the cache (optional).
        cache: Reuse an existing cache object (saves load on hot path);
               creates a new one if None.

    Returns:
        LLMResult with rows + units. Empty rows on LLM failure (caller
        proceeds with FSM-only extraction in that case).
    """
    if cache is None:
        cache = AdaptiveExtractionCache()

    fp = _build_fingerprint(region.text)
    cache_key = f"plaintext:{fp}"

    # Cache hit?
    cached_entry = cache._cache.get(cache_key)
    if cached_entry:
        try:
            rows = [RealignedRow(**r) for r in cached_entry["rule"].get("rows", [])]
            units = cached_entry["rule"].get("units", {})
            notes = cached_entry["rule"].get("notes", "")
            cached_entry["n_uses"] = cached_entry.get("n_uses", 0) + 1
            cached_entry["last_used"] = str(date.today())
            cache._save()
            logger.info(
                "llm_realigner: cache HIT fp=%s rows=%d patent=%s",
                fp[:8], len(rows), patent_id,
            )
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
        result = realign_table_via_llm(
            chunk_text, max_text_chars=6000, max_tokens=8000,
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
        rows=merged_rows,
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

    Generic — independent of patent ids, target names, compound ids.
    """
    rows = list(_CHUNK_ROW_RE.finditer(region_text))
    if len(rows) < 3:
        # Fall back to a hash over the leading 4000 chars
        synth_headers = ["plain_text_table"]
        sample_cells = [region_text[:60]]
        return fingerprint_table_shape(synth_headers, len(rows), sample_cells)

    first = rows[0]
    # Take a synthesized header from the first row's neighborhood
    synth_headers = ["plain_text_table"] + [
        t for t in re.split(r"\s+", first.group(0).strip()) if t
    ][:6]
    sample_cells = [r.group(0)[:40] for r in rows[:5]]
    return fingerprint_table_shape(synth_headers, len(rows), sample_cells)
