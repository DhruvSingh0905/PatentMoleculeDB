"""HARVEST burst orchestrator.

Sequences Agent 1 (Target Extraction) → Agent 2 (Activity Extraction)
→ Agent 5 (Validation) over a patent's text. Caches results by chunk
fingerprint so the same chunk is never re-extracted.

This is the SAFETY-NET tier — it fires only when the cheap pipeline's
gap detector says coverage is insufficient. After warmup, most patents
hit the cheap pipeline and never invoke this module.

Cost-bounded: max 5 chunks per burst, max 3 LLM calls per chunk
(Agent 1 + Agent 2; Agent 5 is deterministic). Max ~15 LLM calls
per cold patent.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from ...adaptive_extraction_cache import (
    AdaptiveExtractionCache,
    fingerprint_table_shape,
)
from ... import config
from ...cost_tracker import cost_tracker
from ...models import AssayResult
from ._compound_id import sanitize_compound_id
from .agent1_targets import Target, extract_targets
from .agent2_activities import ActivityTuple, extract_activities
from .agent5_validate import validate

logger = logging.getLogger(__name__)


@dataclass
class HarvestResult:
    """Top-level result of one HARVEST burst across a patent."""
    targets: list[Target] = field(default_factory=list)
    tuples: list[ActivityTuple] = field(default_factory=list)
    n_chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cost_estimate_usd: float = 0.0     # rough — for diagnostics
    n_chunks_with_tuples: int = 0      # HARVEST hit-rate numerator
    n_prelib_rows: int = 0             # rows from pre-applied pattern library

    def to_assay_results(self) -> dict[str, list[AssayResult]]:
        """Convert tuples to the standard `AssayResult` shape used by
        the rest of the pipeline. Sanitized compound IDs are used as
        dict keys.

        A tuple carrying `range_fields` (set by `_prelib_row_to_tuple` for a
        cell the patent published as a BIN rather than a number) ships that
        encoding verbatim: `value_numeric=None` plus `value_low`/`value_high`
        from the patent's own legend. `t.value` is not read for those — it
        only ever held a placeholder to satisfy `ActivityTuple.value: float`.
        """
        out: dict[str, list[AssayResult]] = {}
        for t in self.tuples:
            cid = sanitize_compound_id(t.compound_id)
            if not cid:
                continue
            rf = getattr(t, "range_fields", None) or {}
            out.setdefault(cid, []).append(AssayResult(
                assay_name=t.assay_name or "unknown",
                value_raw=rf.get("value_raw") or str(t.value),
                value_numeric=rf["value_numeric"] if rf else t.value,
                qualifier=rf.get("qualifier") if rf else t.qualifier,
                unit=(rf.get("unit") if rf else t.unit) or "",
                n_runs=rf.get("n_runs") if rf else t.n_runs,
                value_low=rf.get("value_low"),
                value_high=rf.get("value_high"),
                bin=rf.get("bin"),
                unit_source=rf.get("unit_source", ""),
                # `validation_reason` is where this tier records which arm
                # produced the tuple ("pattern_library:{key}", agent-5 verdicts).
                # Dropping it is what left 27,868 shipped rows unattributed.
                source=t.validation_reason or "",
            ))
        return out


# Approximate cost per LLM call (Sonnet-class model on ~6k input + 4k
# output tokens). Used only for the diagnostic field.
_AGENT_COST_USD = 0.025


def _harvest_cache(cache: AdaptiveExtractionCache | None) -> AdaptiveExtractionCache:
    return cache if cache is not None else AdaptiveExtractionCache()


# ── non-numeric cells: the range they are, never a fabricated point ─────────
#
# `assay_pattern_library` emits `value=None` + `value_categorical` for every
# cell that will not `float()`. Four different things land in that bucket, and
# for 2,931 shipped rows all four were flattened to the same `value_numeric:
# 0.0`, with the literal surviving only inside the `source` string:
#
#   `+` / `++`        a potency BIN      -> the legend's range (US11292791)
#   `>50000` `<0.004` a CENSORED number  -> the number and its qualifier
#   `1.6 ± 0.1`       a mean ± SD        -> the mean
#   `—` `NT` `No inhibition`             -> no measurement at all -> drop
#
# 0.0 μM reads as maximal potency, which is the exact inverse of what three of
# those four mean. This is the defect `f692dc8` fixed one tier over, where 109
# geometric-mean midpoints on US11292791 were reversed back to ranges because
# a bin does not contain a point value.
#
# Nothing new is invented here. `sources.uspto_assays.parse_value` is the cell
# parser behind the path that scores 99.9% exact against BindingDB, and
# `sources.bin_legend.parse_bin_key` is the legend reader CLAUDE.md names for
# `+`/`A-E` -> explicit numeric ranges. Both are imported lazily, like the
# `assay_pattern_library` import below, so `core` keeps no import-time edge
# to `sources`.


def _prelib_bin_key(text: str) -> dict:
    """`{symbol: BinRange}` from the patent's own legend; `{}` when absent.

    `parse_bin_key` never guesses a default scale — two patents in this corpus
    assign incompatible ranges to `++++`. A grade we cannot bind to a legend is
    therefore dropped rather than given a borrowed one.
    """
    from ....sources.bin_legend import parse_bin_key
    try:
        return parse_bin_key(text or "")
    except Exception as e:                                   # never break a run
        logger.warning("prelib: bin-key parse failed: %r", e)
        return {}


def _encode_prelib_value(raw: str, unit: str, bin_key: dict) -> dict | None:
    """One non-numeric cell literal -> the row fields it justifies, or None.

    None means "the patent published no measurement in this cell" — the caller
    drops the row. That is the `missing_fields` contract already in
    `sources/uspto_assays.py:452`: a grade with no key and no bounds is not a
    usable measurement.
    """
    import html

    from ....sources.uspto_assays import parse_value

    s = html.unescape(str(raw or "")).strip()
    if not s:
        return None
    parsed = parse_value(s)
    if parsed is None:
        # Covers the null vocabulary (`NT`, `nd`, `—`) and anything with no
        # number in it. US10214537's 375 `â` cells are `—` (U+2014) read as
        # Latin-1, so the null-vocabulary guard at assay_pattern_library.py:772
        # never saw them.
        return None

    grade = parsed.get("letter_grade")
    if grade:
        rng = bin_key.get(grade)
        if rng is None or (rng.lo is None and rng.hi is None):
            return None
        # Field-for-field what the `uspto_xml_table` row for the same cell
        # already ships (value_raw is the literal grade, qualifier stays null,
        # the unit comes from the legend). One measurement, one spelling.
        return {
            "value_numeric": None,
            "qualifier": None,
            "unit": rng.unit or unit or "",
            "unit_source": "bin_key",
            "n_runs": None,
            "value_raw": s,
            "value_low": rng.lo,
            "value_high": rng.hi,
            "bin": grade,
        }

    num = parsed.get("value_numeric")
    if num is None:
        return None
    return {
        "value_numeric": num,
        "qualifier": parsed.get("qualifier"),
        # The cell's own unit wins when it has one; otherwise keep the unit the
        # pattern library resolved from the column header.
        "unit": parsed.get("unit") or unit or "",
        "unit_source": "",
        "n_runs": parsed.get("n_runs"),
        "value_raw": s,
        "value_low": None,
        "value_high": None,
        "bin": None,
    }


def _prelib_row_to_tuple(row: dict, bin_key: dict) -> "ActivityTuple | None":
    """Pattern-library row -> ActivityTuple, with no fabricated value.

    `ActivityTuple.value` is typed `float` and `from_dict` rejects None, which
    is why this path used to pass `0.0`. The placeholder stays (the dataclass
    demands a float and `agent5_validate._value_to_nM` would raise on None),
    but it is now inert: `range_fields` carries the real encoding and
    `to_assay_results` reads that instead, so the 0.0 never reaches an
    artifact. Rows the encoder rejects are dropped here, not zeroed.
    """
    v = row.get("value")
    if v is not None:
        return ActivityTuple.from_dict(row)

    cat = row.get("value_categorical")
    if not cat:
        return None
    enc = _encode_prelib_value(cat, row.get("unit") or "", bin_key)
    if enc is None:
        return None
    t = ActivityTuple.from_dict({
        **row,
        "value": enc["value_numeric"] if enc["value_numeric"] is not None else 0.0,
    })
    if t is None:
        return None
    t.range_fields = enc
    t.unit = enc["unit"]
    t.qualifier = enc["qualifier"]
    t.n_runs = enc["n_runs"]
    t.validation_reason = (
        (t.validation_reason or "") + f" categorical={cat}"
    ).strip()
    return t


def prelib_tuples(text: str, patent_id: str | None = None) -> list[ActivityTuple]:
    """The pattern-library pass: regex only, ZERO LLM cost.

    Lifted out of `harvest_burst` so the pipeline can run it when the PAID
    tier is switched off. Every live reference to `apply_patterns_to_text`
    used to sit inside `harvest_burst`, whose only live caller is behind
    `HARVEST_BURST_ENABLED` — so `HARVEST_BURST=0`, set to stop a tier that
    cost $2.13 on one patent, also switched off a free regex pass that costs
    nothing. The flag's own docstring says it gates the safety-net tier; it
    was gating this too.

    Measured on the burst-off run that found it: US10214537 lost 2,305 rows
    (its untagged tier went 3,644 -> 1,339) while still spending $0.275 on an
    unrelated path. US9694016 stands to lose 4,310 rows / 1,100 compounds,
    100% of which survive `output_validator`.
    """
    from ..assay_pattern_library import apply_patterns_to_text as _apply_lib
    try:
        rows = _apply_lib(text, patent_id or "unknown", fresh_patterns=None)
    except Exception as e:
        logger.warning("prelib: pattern-library pass failed: %r", e)
        return []
    bin_key = _prelib_bin_key(text)
    out: list[ActivityTuple] = []
    n_dropped = 0
    for row in rows:
        t = _prelib_row_to_tuple(row, bin_key)
        if t is None:
            n_dropped += 1 if row.get("value") is None else 0
            continue
        out.append(t)
    if n_dropped:
        logger.info(
            "prelib: %s — dropped %d non-numeric cell(s) the patent published "
            "no measurement in (em-dash / NT / 'No inhibition'); they used to "
            "ship as value_numeric 0.0",
            patent_id or "unknown", n_dropped,
        )
    return out


def harvest_burst(
    text: str,
    *,
    patent_id: str | None = None,
    cache: AdaptiveExtractionCache | None = None,
    max_chunks: int | None = None,
    chunk_chars: int = 6000,
    chunk_overlap_chars: int = 400,
    early_stop_empty_streak: int = 4,
) -> HarvestResult:
    """Run the HARVEST burst on `text`. Cached per chunk fingerprint.

    Adaptive chunk budget:
      - When `max_chunks` is None (default), the budget scales with the
        signal-bearing portion of the patent. Cap is 5 chunks for short
        patents (≤10 ranked chunks total), bumped up to 30 for very long
        ones (≥200 ranked chunks), with linear interpolation in between.
      - Early-stop: once `early_stop_empty_streak` consecutive ranked
        chunks return zero tuples, we stop — further chunks are below
        the data threshold and would just burn budget. Default 4.

    Args:
        text: full patent text or a long region of interest.
        patent_id: provenance for cache entries.
        cache: existing AdaptiveExtractionCache or None (creates one).
        max_chunks: optional hard cap. Pass None for adaptive sizing.
        chunk_chars: chars per chunk passed to the LLM.
        chunk_overlap_chars: overlap between consecutive chunks.
        early_stop_empty_streak: stop after this many consecutive empty
            chunks (in ranked order) — dampens cost on patents whose
            ranked chunks fall off a cliff.

    Returns:
        HarvestResult with merged tuples + targets + diagnostics.
    """
    cache = _harvest_cache(cache)
    if not text:
        return HarvestResult()

    # Generate ALL chunks of the patent, then rank them by assay-signal
    # density. The cap on chunks scales with how much signal-bearing
    # text the patent has — short patents stop at 5, long ones extend
    # up to 30. Early-stop further trims the tail of zero-yield chunks.
    all_chunks = _chunk_text(
        text, chunk_chars=chunk_chars, overlap=chunk_overlap_chars,
    )
    # When caller explicitly asked for a large budget (escalation
    # path for `low_compound_density` patents), keep zero-score
    # chunks too — the regex signals undercount letter-grade tables
    # and we'd rather give the LLM more text than starve it.
    if max_chunks is not None and max_chunks >= 20:
        ranked = _rank_chunks_by_signal(all_chunks, min_score=0)
    else:
        ranked = _rank_chunks_by_signal(all_chunks)
    n_ranked = len(ranked)
    if max_chunks is None:
        # Linear interpolation: 5 chunks at ≤10 ranked, 30 at ≥200 ranked.
        if n_ranked <= 10:
            adaptive_cap = 5
        elif n_ranked >= 200:
            adaptive_cap = 30
        else:
            # 5 + (n_ranked - 10) * (30 - 5) / (200 - 10)
            adaptive_cap = 5 + int((n_ranked - 10) * 25 / 190)
        max_chunks = adaptive_cap
    chunks = ranked[:max_chunks]
    logger.info(
        "harvest_burst: %s — %d chunks selected from %d total ranked "
        "(adaptive cap %d, early-stop after %d empty streak)",
        patent_id, len(chunks), len(all_chunks), max_chunks, early_stop_empty_streak,
    )

    all_targets: list[Target] = []
    seen_target_keys: set[tuple[str, str, str]] = set()
    all_tuples: list[ActivityTuple] = []
    all_row_patterns: list[dict] = []
    cache_hits = cache_misses = 0
    cost_estimate = 0.0
    empty_streak = 0   # consecutive chunks that returned zero tuples
    n_chunks_with_tuples = 0  # HARVEST hit-rate numerator

    # ── Pre-apply existing pattern library to the FULL text ──
    # Patterns auto-promoted from previous patents fire here for free
    # (regex-only). If they extract enough rows, the chunks below may
    # need fewer LLM calls. This is the "discovery first, HARVEST last"
    # reordering the user requested.
    prelib_rows = prelib_tuples(text, patent_id)
    if prelib_rows:
        n_prelib_tuples = 0
        for t in prelib_rows:
            all_tuples.append(t)
            n_prelib_tuples += 1
        logger.info(
            "harvest_burst: %s — pre-library pass extracted %d rows (zero LLM cost) "
            "BEFORE Agent 1/2 fire",
            patent_id, n_prelib_tuples,
        )

    from ... import config as _cfg

    # ── HARVEST skip gate — pre-library was sufficient ──
    # When the pre-applied pattern library already extracted enough rows
    # AND the patent has registered patterns of its own (proxy for "we
    # know how to read this format"), skip the HARVEST burst entirely.
    # Saves ~$0.50 per patent on already-covered formats. Verified on
    # US11292791: 8,693 pre-library rows + 3 % HARVEST hit-rate → 75
    # batched LLM calls produced nothing the library hadn't already
    # caught. Threshold and toggle are env-tunable; see config.py.
    if (
        getattr(_cfg, "HARVEST_SKIP_ENABLED", True)
        and len(all_tuples) >= getattr(_cfg, "HARVEST_SKIP_MIN_PRELIB_ROWS", 500)
        and _patent_has_own_patterns(patent_id)
    ):
        logger.info(
            "harvest_burst: %s — SKIPPED (pre-library already extracted %d rows "
            "and patent has registered patterns; HARVEST_SKIP gate fired). "
            "Saves the Agent 1/2/2b LLM batches.",
            patent_id, len(all_tuples),
        )
        chunks = []  # short-circuits both batched and sequential paths

    # ── Batched-mode HARVEST ──
    # When config.LLM_BATCH_ENABLED, submit all Agent 1 then all Agent 2
    # calls as Message Batches (50% off) instead of the sequential loop.
    # The sequential loop below is the fallback path.
    if getattr(_cfg, "LLM_BATCH_ENABLED", False) and chunks:
        logger.info(
            "harvest_burst: %s — BATCHED mode, dispatching %d chunks "
            "(Anthropic Message Batches, 50%% discount)",
            patent_id, len(chunks),
        )
        batched_results = _run_chunks_batched(chunks, patent_id)
        for cidx, (chunk_targets, chunk_tuples, chunk_row_patterns) in enumerate(batched_results):
            # Dedup targets across chunks
            for t in chunk_targets:
                tk = (t.target_name.lower(), t.organism.lower(), t.isoform.lower())
                if tk in seen_target_keys:
                    continue
                seen_target_keys.add(tk)
                all_targets.append(t)
            all_tuples.extend(chunk_tuples)
            all_row_patterns.extend(chunk_row_patterns)
            if chunk_tuples:
                n_chunks_with_tuples += 1
        hit_rate = n_chunks_with_tuples / max(1, len(batched_results))
        logger.info(
            "harvest_burst: %s — batched results: %d chunks, %d targets, %d tuples, "
            "%d patterns; hit-rate %.0f%% (%d/%d chunks yielded tuples)",
            patent_id, len(batched_results), len(all_targets), len(all_tuples),
            len(all_row_patterns), 100 * hit_rate,
            n_chunks_with_tuples, len(batched_results),
        )
        # Skip to pattern feedback / return — bypasses sequential loop
        chunks = []  # short-circuits the for-loop below

    for cidx, chunk_text in enumerate(chunks):
        # The per-patent LM cap, which this path did not consult.
        #
        # `PER_PATENT_LM_CAP` is $0.20 and `api_client` records every call
        # against it, so the tracker knew the whole time — nothing asked. The
        # IUPAC sibling (`harvest/iupac_orchestrator`) checks it in four
        # places; this one checked it in none, and a measured probe on
        # US9718790 spent $2.13 across 55 calls before the batch it was
        # waiting on had even returned. A cap nothing consults is not a cap.
        #
        # Checked per chunk rather than up front because the spend accrues
        # chunk by chunk, and stopping at the boundary keeps every chunk
        # already paid for. Cache HITS are below this line deliberately: a
        # cached chunk costs nothing, and refusing to read one because an
        # earlier chunk was expensive would drop free coverage.
        if cost_tracker.patent_lm_exceeded(patent_id or ""):
            logger.warning(
                "harvest_burst: %s LM cap $%.2f reached — stopping at chunk %d/%d",
                patent_id or "-", config.PER_PATENT_LM_CAP, cidx + 1, len(chunks),
            )
            break
        fp = _chunk_fingerprint(chunk_text)
        cache_key = f"harvest:{fp}"

        cached_entry = cache._cache.get(cache_key)
        chunk_row_patterns: list[dict] = []
        if cached_entry:
            cache_hits += 1
            try:
                payload = cached_entry["rule"]
                chunk_targets = [Target.from_dict(t) for t in payload.get("targets", [])]
                chunk_tuples = [
                    ActivityTuple.from_dict(t) for t in payload.get("tuples", [])
                ]
                chunk_tuples = [t for t in chunk_tuples if t is not None]
                chunk_row_patterns = list(payload.get("row_patterns", []) or [])
                cached_entry["n_uses"] = cached_entry.get("n_uses", 0) + 1
                cached_entry["last_used"] = str(date.today())
                cache._save()
                logger.info(
                    "harvest_burst: chunk %d/%d cache HIT fp=%s (%d targets, %d tuples, %d patterns)",
                    cidx + 1, len(chunks), fp[:8],
                    len(chunk_targets), len(chunk_tuples), len(chunk_row_patterns),
                )
            except Exception as e:
                logger.warning(
                    "harvest_burst: cache hydrate failed for fp=%s: %r — re-firing",
                    fp[:8], e,
                )
                chunk_targets, chunk_tuples, chunk_row_patterns = _run_chunk(chunk_text, patent_id)
                cache_misses += 1
                cost_estimate += 2 * _AGENT_COST_USD
                _persist(cache, cache_key, chunk_targets, chunk_tuples, patent_id, chunk_row_patterns)
        else:
            cache_misses += 1
            cost_estimate += 2 * _AGENT_COST_USD
            chunk_targets, chunk_tuples, chunk_row_patterns = _run_chunk(chunk_text, patent_id)
            _persist(cache, cache_key, chunk_targets, chunk_tuples, patent_id, chunk_row_patterns)
            logger.info(
                "harvest_burst: chunk %d/%d MISS fp=%s — extracted %d targets, %d tuples, %d patterns",
                cidx + 1, len(chunks), fp[:8],
                len(chunk_targets), len(chunk_tuples), len(chunk_row_patterns),
            )

        # Dedup targets across chunks
        for t in chunk_targets:
            tk = (t.target_name.lower(), t.organism.lower(), t.isoform.lower())
            if tk in seen_target_keys:
                continue
            seen_target_keys.add(tk)
            all_targets.append(t)

        all_tuples.extend(chunk_tuples)
        all_row_patterns.extend(chunk_row_patterns)
        if chunk_tuples:
            n_chunks_with_tuples += 1

        # Adaptive early-stop: ranked chunks are already in descending
        # signal order, so a streak of empty results means we've passed
        # the data and further chunks won't recover. Cache hits don't
        # count toward the streak — they're free, keep walking until
        # we hit a tail of misses with no yield.
        if not cached_entry:
            if len(chunk_tuples) == 0:
                empty_streak += 1
                if empty_streak >= early_stop_empty_streak:
                    logger.info(
                        "harvest_burst: %s — early-stop at chunk %d/%d "
                        "(%d consecutive empty MISS chunks)",
                        patent_id, cidx + 1, len(chunks), empty_streak,
                    )
                    break
            else:
                empty_streak = 0

    # ─ Pattern feedback loop ─
    # The LLM extracted row-regexes from the chunks it saw. Apply them
    # across the FULL patent text to harvest the rest of those tables
    # for free (no extra LLM cost). Also register the patterns in the
    # discoveries library so future patents with the same table format
    # benefit immediately, before HARVEST burst would even fire.
    if all_row_patterns:
        from ..assay_pattern_library import add_pattern, apply_patterns_to_text
        # 1. Register the new patterns
        for rp in all_row_patterns:
            add_pattern(
                regex=rp.get("regex", ""),
                column_assays=rp.get("column_assays") or [],
                example_match=rp.get("example_match", ""),
                patent_id=patent_id or "unknown",
                header_text=rp.get("header_text", ""),
            )
        # 2. Apply ALL active patterns (library + fresh) across the
        #    full text. Returns ActivityTuple-shaped dicts.
        new_rows = apply_patterns_to_text(
            text, patent_id or "unknown",
            fresh_patterns=all_row_patterns,
        )
        # 3. Convert + dedup against tuples we already have. Dedup key
        #    is (cid, assay_name, value, unit) — same as downstream.
        seen = {
            (t.compound_id.lower(), t.assay_name.lower(), t.value, (t.unit or "").lower())
            for t in all_tuples
        }
        n_added_from_patterns = 0
        bin_key = _prelib_bin_key(text)
        for row in new_rows:
            v = row.get("value")
            if v is None:
                # Categorical/letter-grade — keep but use string for dedup
                key = (
                    str(row.get("compound_id", "")).lower(),
                    str(row.get("assay_name", "")).lower(),
                    str(row.get("value_categorical", "")),
                    "",
                )
            else:
                key = (
                    str(row.get("compound_id", "")).lower(),
                    str(row.get("assay_name", "")).lower(),
                    float(v),
                    str(row.get("unit", "")).lower(),
                )
            if key in seen:
                continue
            seen.add(key)
            # Same encoder as `prelib_tuples`: a bin becomes the legend's
            # range, a censored cell keeps its number and qualifier, and a
            # cell with no measurement in it is dropped rather than zeroed.
            t = _prelib_row_to_tuple(row, bin_key)
            if t is None:
                continue
            all_tuples.append(t)
            n_added_from_patterns += 1
        if n_added_from_patterns:
            logger.info(
                "harvest_burst: %s — pattern re-application added %d rows "
                "(zero LLM cost) across %d patterns",
                patent_id, n_added_from_patterns, len(all_row_patterns),
            )

    # n_chunks reflects the number we actually processed (sequential path
    # may early-stop; batched path completed all). When batched-mode
    # short-circuited the for-loop above, len(chunks) is 0 — fall back
    # to len(batched_results) by counting how many tuples-yielding chunks
    # we recorded.
    n_processed = max(len(chunks), n_chunks_with_tuples)
    return HarvestResult(
        targets=all_targets,
        tuples=all_tuples,
        n_chunks=n_processed,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cost_estimate_usd=round(cost_estimate, 4),
        n_chunks_with_tuples=n_chunks_with_tuples,
        n_prelib_rows=len(prelib_rows) if prelib_rows else 0,
    )


# ── Helpers ─────────────────────────────────────────────────────


def _run_chunk(
    chunk_text: str, patent_id: str | None,
) -> tuple[list[Target], list[ActivityTuple], list[dict]]:
    """Run Agent 1 → Agent 2 → Agent 5 on a single chunk. Also returns
    the row_patterns Agent 2 extracted so the orchestrator can register
    them with the pattern library and re-apply them across the full text.
    """
    a1 = extract_targets(chunk_text)
    targets = a1.targets

    a2 = extract_activities(chunk_text, targets)
    tuples = a2.tuples
    raw_patterns = [
        {
            "regex": rp.regex,
            "column_assays": rp.column_assays,
            "header_text": rp.header_text,
            "example_match": rp.example_match,
        }
        for rp in a2.row_patterns
    ]

    if tuples:
        a5 = validate(tuples, targets, chunk_text)
        tuples = a5.tuples

    return targets, tuples, raw_patterns


def _patent_has_own_patterns(patent_id: str | None) -> bool:
    """True when assay_patterns.discoveries.json lists this patent in
    `fingerprints_observed` for ≥ 1 active or pending pattern. Used by
    the HARVEST skip gate as the "we know this patent's format" signal —
    if the patent has its own discovered patterns, they were already
    applied in the pre-library pass and likely covered the assay tables.
    """
    if not patent_id:
        return False
    try:
        import json
        from pathlib import Path
        disc_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data" / "assay_patterns.discoveries.json"
        )
        if not disc_path.exists():
            return False
        d = json.loads(disc_path.read_text())
        for tok in d.get("tokens", []):
            if patent_id in tok.get("fingerprints_observed", []):
                return True
    except Exception:
        return False
    return False


def _run_chunks_batched(
    chunks: list[str], patent_id: str | None,
) -> list[tuple[list[Target], list[ActivityTuple], list[dict]]]:
    """Two-round batched HARVEST: one Message Batch for Agent 1 across
    ALL chunks, then one for Agent 2 (+ Agent 2b for chunks with row
    signal) across ALL chunks. Saves 50 % of the per-token cost via
    Anthropic's Message Batches discount and runs requests concurrently
    server-side.

    Agent 5 (validation) is skipped in batched mode — it only re-grades
    confidence and never drops tuples, so the small accuracy loss
    isn't worth a third round-trip.

    Returns a list parallel to ``chunks``: ``[(targets, tuples,
    row_patterns), …]`` — same shape as a sequential loop of
    ``_run_chunk`` calls.
    """
    from ...api_client import call_claude_text_batch
    from .agent1_targets import (
        AGENT1_TARGET_EXTRACTION, Agent1Result, Target,
        _parse_array_json as _parse_a1_json,
        summarize_targets,
    )
    from .agent2_activities import (
        AGENT2_ACTIVITY_EXTRACTION, AGENT2B_ROW_PATTERN,
        _parse_array_json as _parse_a2_json,
        _TABLE_ROW_SIGNAL_RE, RowPattern,
    )

    n = len(chunks)
    if n == 0:
        return []

    # ── Round 1: Agent 1 (target extraction) for all chunks ──
    a1_requests = [{
        "prompt": AGENT1_TARGET_EXTRACTION.format(chunk_text=c[:6000]),
        "max_tokens": 2000,
    } for c in chunks]
    a1_raw = call_claude_text_batch(a1_requests, patent_id=patent_id or "")

    targets_per_chunk: list[list[Target]] = []
    for raw in a1_raw:
        targets: list[Target] = []
        if raw:
            parsed = _parse_a1_json(raw)
            if parsed:
                for d in parsed:
                    try:
                        # Agent1Result.targets uses Target.from_dict — match it
                        t = Target.from_dict(d) if hasattr(Target, "from_dict") else Target(**d)
                        targets.append(t)
                    except Exception:
                        continue
        targets_per_chunk.append(targets)

    # ── Round 2: Agent 2 (tuple extraction) for all chunks ──
    a2_requests = [{
        "prompt": AGENT2_ACTIVITY_EXTRACTION.format(
            targets_summary=summarize_targets(targets_per_chunk[i]),
            chunk_text=chunks[i][:6000],
        ),
        "max_tokens": 4096,
    } for i in range(n)]
    a2_raw = call_claude_text_batch(a2_requests, patent_id=patent_id or "")

    tuples_per_chunk: list[list[ActivityTuple]] = []
    for raw in a2_raw:
        chunk_tuples: list[ActivityTuple] = []
        if raw:
            parsed = _parse_a2_json(raw)
            if parsed:
                for d in parsed:
                    t = ActivityTuple.from_dict(d)
                    if t is not None:
                        chunk_tuples.append(t)
        tuples_per_chunk.append(chunk_tuples)

    # ── Round 2b: Agent 2b (row-pattern extraction) — only on chunks
    # with row signal or extracted tuples, to keep batch size down ──
    a2b_indices: list[int] = []
    for i, chunk in enumerate(chunks):
        n_signal = len(_TABLE_ROW_SIGNAL_RE.findall(chunk))
        if tuples_per_chunk[i] or n_signal >= 10:
            a2b_indices.append(i)
    patterns_per_chunk: list[list[dict]] = [[] for _ in range(n)]
    if a2b_indices:
        a2b_requests = []
        for i in a2b_indices:
            sample_tuples = tuples_per_chunk[i][:5]
            tuples_summary = "\n".join(
                f"  - {t.compound_id} / {t.assay_name} = {t.value} {t.unit or ''}"
                for t in sample_tuples
            ) if sample_tuples else "  (no Agent 2 tuples; extract patterns by structure alone)"
            a2b_requests.append({
                "prompt": AGENT2B_ROW_PATTERN.format(
                    chunk_text=chunks[i][:6000],
                    tuples_summary=tuples_summary,
                ),
                "max_tokens": 2000,
            })
        a2b_raw = call_claude_text_batch(a2b_requests, patent_id=patent_id or "")
        for j, raw in enumerate(a2b_raw):
            if not raw:
                continue
            parsed = _parse_a2_json(raw)
            if not parsed:
                continue
            chunk_patterns: list[dict] = []
            for d in parsed:
                if not isinstance(d, dict):
                    continue
                rp = RowPattern.from_dict(d) if hasattr(RowPattern, "from_dict") else None
                if rp is None or not rp.regex:
                    continue
                chunk_patterns.append({
                    "regex": rp.regex,
                    "column_assays": rp.column_assays,
                    "example_match": rp.example_match,
                })
            patterns_per_chunk[a2b_indices[j]] = chunk_patterns

    return [
        (targets_per_chunk[i], tuples_per_chunk[i], patterns_per_chunk[i])
        for i in range(n)
    ]


def _persist(
    cache: AdaptiveExtractionCache,
    cache_key: str,
    targets: list[Target],
    tuples: list[ActivityTuple],
    patent_id: str | None,
    row_patterns: list[dict] | None = None,
) -> None:
    cache._cache[cache_key] = {
        "rule": {
            "targets": [t.__dict__ for t in targets],
            "tuples": [t.to_dict() for t in tuples],
            "row_patterns": row_patterns or [],
        },
        "first_observed": patent_id or "unknown",
        "n_uses": 1,
        "last_used": str(date.today()),
    }
    cache._save()


_ASSAY_KEYWORD_RE_LOCAL = re.compile(
    r"\b(?:IC[\s_-]?50|EC[\s_-]?50|CC[\s_-]?50|AC[\s_-]?50|XC[\s_-]?50"
    r"|GI[\s_-]?50|MIC[\s_-]?50?|IC[\s_-]?90|EC[\s_-]?90"
    r"|K_?[id]|pIC[\s_-]?50|pK_?[id])\b",
    re.IGNORECASE,
)
_VALUE_UNIT_RE_LOCAL = re.compile(
    r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?\s*(?:nM|μM|uM|µM|mcM|mM|pM)(?![A-Za-z])",
    re.IGNORECASE,
)
# Letter-grade row recognizer — supports three common bin systems:
#   - A-E single letters       (e.g. "1. A")
#   - +/++/+++/++++/+++++ pluses (e.g. "1 ++ +++") — used by Constellation
#     bromodomain patents like US11292791 to encode tiered IC50 ranges
#   - */**/***/****/***** stars  (less common but identical pattern)
# Without these the chunk ranker treats the table as prose, the burst
# picks the wrong chunks, and 100 % of LLM extraction returns empty.
_LETTER_GRADE_ROW_RE_LOCAL = re.compile(
    r"(?:^|[\s\n])\d{1,4}\.?\s+(?:[A-E]|\+{1,5}|\*{1,5})(?=\s|$|\n)",
    re.MULTILINE,
)
# A "table row" looks like: compound-id-shape token (letters? + optional
# dash + 1-4 digits + optional letter suffix), separated by whitespace
# from at least one short value token. Short value = letter grade A-E,
# or +-grade / *-grade, or qualified number, or bare integer/decimal.
# This is the strongest table-shape signal — NMR-prose chunks have lots
# of `\d nM`-style tokens but rarely have many DISTINCT cid-shape tokens
# leading rows.
_TABLE_ROW_RE_LOCAL = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-Z]{1,3}-?\d{1,4}[A-Za-z]{0,3})"   # cid-shape
    r"\s+"
    r"(?:[<>~≤≥≈]?\s*\d+(?:\.\d+)?|[A-E]|\+{1,5}|\*{1,5})"
    r"(?=\s|$|\n)",
    re.MULTILINE,
)
# Bare-digit numeric-table row detector — separate from
# `_TABLE_ROW_RE_LOCAL` so we don't loosen the cid-shape requirement
# (which would over-match prose like "in 5 minutes").  Requires a
# bare-digit cid (1-4 digits + optional letter suffix) followed by TWO
# numeric tokens — prose almost never has two consecutive bare numerics
# after an integer.  Accepts `>30` and HTML-entity-encoded `&gt;30` /
# `&lt;30` qualifiers (US10246453's flat-text activity table at offset
# 813K used `&gt;30` and the original regex scored that chunk 14/600 —
# rank 130/147 — so HARVEST never saw it).
_NUMERIC_TABLE_ROW_RE_LOCAL = re.compile(
    r"(?<![A-Za-z0-9])"
    r"\d{1,4}[A-Za-z]{0,3}"                       # bare-digit cid
    r"\s+"
    r"(?:[<>~≤≥≈]|&(?:gt|lt|le|ge);)?\s*\d+(?:\.\d+)?"  # value 1
    r"\s+"
    r"(?:[<>~≤≥≈]|&(?:gt|lt|le|ge);)?\s*\d+(?:\.\d+)?"  # value 2
    r"(?=\s|$|\n)",
    re.MULTILINE,
)
# NMR-noise penalty: "1H NMR" / "13C NMR" headers indicate a synthesis
# annotation block that's structurally similar to assay rows (lots of
# values + units) but contains no extractable data.
_NMR_HEADER_RE_LOCAL = re.compile(
    r"\b(?:1\s*H|13\s*C|19\s*F|31\s*P|2\s*D)\s*NMR\b|\bm/z\b|\b\[M\+H\]\+",
    re.IGNORECASE,
)
# Permissive table-row signal: cid-shape followed by ANY short non-
# whitespace token. This is the catch-all that lets the ranker find
# tables in novel grade systems we haven't hard-coded — `Active/
# Inactive`, `Red/Yellow/Green`, `Low/Med/High`, `NT/NA`, future
# patent conventions. Weight kept low because it false-positives on
# prose ("Compound 5 was reduced ..." matches "5 was"), so it only
# matters when the strict signals are silent — exactly the case where
# we'd otherwise miss the table entirely.
_PERMISSIVE_TABLE_ROW_RE_LOCAL = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Z]{0,3}-?\d{1,4}[A-Za-z]{0,3}"
    r"\s+"
    r"[^\s\d.,;:][^\s]{0,4}"   # 1-5 char non-numeric token
    r"(?=\s|$|\n)",
    re.MULTILINE,
)


def _signal_score(chunk: str) -> int:
    """How much assay-relevant signal is in this chunk?

    Used to rank chunks for the HARVEST burst — we want to spend the
    budget on chunks most likely to have measurements, not the patent
    cover page, claims section, or NMR-annotated synthesis prose.

    Patent-agnostic signals:
      - assay keywords (IC50/EC50/Ki/Kd/CC50/…)        weight 2
      - numeric value+unit hits (e.g. `12 nM`)          weight 1
      - letter-grade row hits (e.g. `1. A`)             weight 4
      - table-row hits (cid-shape followed by value)    weight 5  ← dominant
      - permissive row hits (cid-shape + any short tok) weight 1  ← catch-all
        Lets the ranker score novel-grade tables ("Active/Inactive",
        "Red/Green", colour codes, future conventions) without
        requiring a hard-coded value regex. False-positive prone on
        prose, so it only outranks NMR penalties when the chunk is
        DENSE in cid-shape rows (≥30 hits per 6KB chunk).
      - NMR-annotation penalty (1H NMR / m/z / [M+H]+)  weight −3
        (NMR blocks have lots of value+unit tokens but
         zero extractable assay data; without the penalty
         a chunk full of NMR couplings can out-score a
         chunk containing the actual assay table)
    """
    n_keywords = len(_ASSAY_KEYWORD_RE_LOCAL.findall(chunk))
    n_values   = len(_VALUE_UNIT_RE_LOCAL.findall(chunk))
    n_grades   = len(_LETTER_GRADE_ROW_RE_LOCAL.findall(chunk))
    n_rows     = len(_TABLE_ROW_RE_LOCAL.findall(chunk))
    n_rows_numeric = len(_NUMERIC_TABLE_ROW_RE_LOCAL.findall(chunk))
    n_rows_permissive = len(_PERMISSIVE_TABLE_ROW_RE_LOCAL.findall(chunk))
    n_nmr      = len(_NMR_HEADER_RE_LOCAL.findall(chunk))
    return (
        n_keywords * 2
        + n_values * 1
        + n_grades * 4
        + n_rows * 5
        + n_rows_numeric * 4
        + n_rows_permissive * 1
        - n_nmr * 3
    )


def _rank_chunks_by_signal(chunks: list[str], *, min_score: int = 1) -> list[str]:
    """Sort chunks descending by signal score. Ties broken by chunk
    position (earlier first) for determinism. Drops chunks below
    `min_score` to avoid burning LLM budget on pure prose — pass
    `min_score=0` to keep them when escalating coverage on a patent
    where the regex signals undercount the real assay density (e.g.,
    letter-grade activity bins).
    """
    indexed = [(i, _signal_score(c), c) for i, c in enumerate(chunks)]
    indexed.sort(key=lambda x: (-x[1], x[0]))
    return [c for (_, score, c) in indexed if score >= min_score]


def _chunk_text(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    """Split a long text into overlapping chunks. Tries to break on
    paragraph boundaries when possible to avoid mid-sentence cuts.

    After splitting, walks each chunk; if it contains numeric-row
    signals but lacks a column-header context (assay keyword + unit),
    prepends the nearest preceding header block from the full text.
    This fixes the case where a long activity table spans multiple
    chunks: the first chunk has the `IC50 (μM)` header AND the first
    rows, but downstream chunks contain only `<cid> <num> <num>` rows
    with no header in scope. Without propagation, Agent2 sees opaque
    rows of numbers and emits 0 tuples on every chunk after the first.
    """
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    raw_chunks: list[tuple[int, str]] = []  # (offset_in_full_text, chunk)
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + chunk_chars, n)
        if end < n:
            window = text[end - 400:end]
            best_break = window.rfind("\n\n")
            if best_break < 0:
                best_break = window.rfind(". ")
            if best_break >= 0:
                end = (end - 400) + best_break + 1
        raw_chunks.append((pos, text[pos:end]))
        if end >= n:
            break
        pos = max(end - overlap, pos + 1)

    return _propagate_table_headers(raw_chunks, full_text=text)


# Table headers look like an assay keyword + unit on the same or
# adjacent line. We look for ANY of these in the ~500 chars before the
# first data row to capture the column-name labels.
_HEADER_MARKER_RE_LOCAL = re.compile(
    r"\b(?:IC|EC|CC|AC|XC|GI|MIC)50\b"
    r"|\bK[id]\b"
    r"|\(\s*(?:μ|u|µ|p|n|m)M\s*\)"
    r"|\bIC\s*_?\s*\{?50\}?",
    re.IGNORECASE,
)


def _propagate_table_headers(
    raw_chunks: list[tuple[int, str]],
    *,
    full_text: str,
    header_window: int = 600,
) -> list[str]:
    """For each chunk with numeric-row signals but no header in scope,
    look back in full_text for the nearest preceding header marker
    (assay keyword + unit) and prepend a ``header_window`` slice as
    context. Keeps the original chunk content unchanged after the
    prepended block, so source-quote offsets remain matchable on a
    suffix-anchored basis.
    """
    augmented: list[str] = []
    for offset, chunk in raw_chunks:
        if (
            _NUMERIC_TABLE_ROW_RE_LOCAL.search(chunk)
            and not _HEADER_MARKER_RE_LOCAL.search(chunk)
        ):
            # Header is upstream — find the latest marker before this chunk
            ms = list(_HEADER_MARKER_RE_LOCAL.finditer(full_text, 0, offset))
            if ms:
                hm = ms[-1]
                ctx_start = max(0, hm.start() - 300)
                ctx_end = min(offset, hm.start() + 300)
                ctx = full_text[ctx_start:ctx_end]
                if ctx.strip():
                    chunk = (
                        "[TABLE HEADER CONTEXT — applies to the rows below]\n"
                        + ctx
                        + "\n\n[CONTINUED DATA ROWS — same columns as above]\n"
                        + chunk
                    )
        augmented.append(chunk)
    return augmented


def _chunk_fingerprint(chunk_text: str) -> str:
    """Build a structural fingerprint from a HARVEST chunk.

    Re-uses the existing `fingerprint_table_shape` for cache-key
    compatibility. Inputs derived from chunk SHAPE, not content:
      - row count proxy (number of newlines)
      - first 6 word-tokens of the chunk (after normalization) as
        synthetic "headers"
      - sample first 5 line-shapes (numeric runs replaced with N,
        word runs with X)
    """
    lines = [ln.strip() for ln in chunk_text.split("\n") if ln.strip()]
    n_rows = len(lines)
    first_tokens = [
        t for t in re.split(r"\s+", chunk_text[:200]) if t
    ][:6]
    sample = [
        re.sub(r"[A-Za-z]+", "X", re.sub(r"\d+", "N", ln[:40]))
        for ln in lines[:5]
    ]
    base = fingerprint_table_shape(
        ["harvest_chunk"] + first_tokens, n_rows, sample,
    )
    # Schema-version suffix:
    #   _v6 — Agent 2b became unconditional on row-rich chunks.
    #   _v7 — chunker now propagates the nearest preceding table
    #         header into chunks that contain only data rows; without
    #         this, multi-page activity tables produced 0 tuples on
    #         every chunk after the first (verified on US10246453
    #         offset ~1.84M containing cids 245-321 with no IC50 /
    #         Na V 1.6 header in scope).
    schema = "_v7"
    # Append the vocabulary fingerprint so newly-discovered cid prefixes
    # (e.g. patent-specific Z1/Z2 codes) invalidate prior cached results.
    try:
        from ...compound_id import _load_prefixes
        import hashlib
        vocab_fp = hashlib.sha256(
            "|".join(_load_prefixes()).encode("utf-8")
        ).hexdigest()[:8]
        return f"{base}{schema}_{vocab_fp}"
    except Exception:
        return f"{base}{schema}"
