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

    def to_assay_results(self) -> dict[str, list[AssayResult]]:
        """Convert tuples to the standard `AssayResult` shape used by
        the rest of the pipeline. Sanitized compound IDs are used as
        dict keys."""
        out: dict[str, list[AssayResult]] = {}
        for t in self.tuples:
            cid = sanitize_compound_id(t.compound_id)
            if not cid:
                continue
            out.setdefault(cid, []).append(AssayResult(
                assay_name=t.assay_name or "unknown",
                value_raw=str(t.value),
                value_numeric=t.value,
                qualifier=t.qualifier,
                unit=t.unit or "",
                n_runs=t.n_runs,
            ))
        return out


# Approximate cost per LLM call (Sonnet-class model on ~6k input + 4k
# output tokens). Used only for the diagnostic field.
_AGENT_COST_USD = 0.025


def _harvest_cache(cache: AdaptiveExtractionCache | None) -> AdaptiveExtractionCache:
    return cache if cache is not None else AdaptiveExtractionCache()


def harvest_burst(
    text: str,
    *,
    patent_id: str | None = None,
    cache: AdaptiveExtractionCache | None = None,
    max_chunks: int = 5,
    chunk_chars: int = 6000,
    chunk_overlap_chars: int = 400,
) -> HarvestResult:
    """Run the HARVEST burst on `text`. Cached per chunk fingerprint.

    Args:
        text: full patent text or a long region of interest. Caller
            decides scope; orchestrator chunks internally.
        patent_id: provenance for cache entries.
        cache: existing AdaptiveExtractionCache or None (creates one).
        max_chunks: hard cap on chunks per burst (cost guard).
        chunk_chars: chars per chunk passed to the LLM.
        chunk_overlap_chars: overlap between consecutive chunks.

    Returns:
        HarvestResult with merged tuples + targets + diagnostics.
    """
    cache = _harvest_cache(cache)
    if not text:
        return HarvestResult()

    # Generate ALL chunks of the patent, then rank them by assay-signal
    # density and take the top `max_chunks`. This way the burst spends
    # its budget on the parts of the patent most likely to have data —
    # critical for long patents where TABLE A might be at char 300k.
    all_chunks = _chunk_text(
        text, chunk_chars=chunk_chars, overlap=chunk_overlap_chars,
    )
    chunks = _rank_chunks_by_signal(all_chunks)[:max_chunks]
    logger.info(
        "harvest_burst: %s — %d chunks selected from %d total "
        "(by assay-signal density; max %d)",
        patent_id, len(chunks), len(all_chunks), max_chunks,
    )

    all_targets: list[Target] = []
    seen_target_keys: set[tuple[str, str, str]] = set()
    all_tuples: list[ActivityTuple] = []
    cache_hits = cache_misses = 0
    cost_estimate = 0.0

    for cidx, chunk_text in enumerate(chunks):
        fp = _chunk_fingerprint(chunk_text)
        cache_key = f"harvest:{fp}"

        cached_entry = cache._cache.get(cache_key)
        if cached_entry:
            cache_hits += 1
            try:
                payload = cached_entry["rule"]
                chunk_targets = [Target.from_dict(t) for t in payload.get("targets", [])]
                chunk_tuples = [
                    ActivityTuple.from_dict(t) for t in payload.get("tuples", [])
                ]
                chunk_tuples = [t for t in chunk_tuples if t is not None]
                cached_entry["n_uses"] = cached_entry.get("n_uses", 0) + 1
                cached_entry["last_used"] = str(date.today())
                cache._save()
                logger.info(
                    "harvest_burst: chunk %d/%d cache HIT fp=%s (%d targets, %d tuples)",
                    cidx + 1, len(chunks), fp[:8],
                    len(chunk_targets), len(chunk_tuples),
                )
            except Exception as e:
                logger.warning(
                    "harvest_burst: cache hydrate failed for fp=%s: %r — re-firing",
                    fp[:8], e,
                )
                chunk_targets, chunk_tuples = _run_chunk(chunk_text, patent_id)
                cache_misses += 1
                cost_estimate += 2 * _AGENT_COST_USD
                _persist(cache, cache_key, chunk_targets, chunk_tuples, patent_id)
        else:
            cache_misses += 1
            cost_estimate += 2 * _AGENT_COST_USD
            chunk_targets, chunk_tuples = _run_chunk(chunk_text, patent_id)
            _persist(cache, cache_key, chunk_targets, chunk_tuples, patent_id)
            logger.info(
                "harvest_burst: chunk %d/%d MISS fp=%s — extracted %d targets, %d tuples",
                cidx + 1, len(chunks), fp[:8],
                len(chunk_targets), len(chunk_tuples),
            )

        # Dedup targets across chunks
        for t in chunk_targets:
            tk = (t.target_name.lower(), t.organism.lower(), t.isoform.lower())
            if tk in seen_target_keys:
                continue
            seen_target_keys.add(tk)
            all_targets.append(t)

        all_tuples.extend(chunk_tuples)

    return HarvestResult(
        targets=all_targets,
        tuples=all_tuples,
        n_chunks=len(chunks),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cost_estimate_usd=round(cost_estimate, 4),
    )


# ── Helpers ─────────────────────────────────────────────────────


def _run_chunk(chunk_text: str, patent_id: str | None) -> tuple[list[Target], list[ActivityTuple]]:
    """Run Agent 1 → Agent 2 → Agent 5 on a single chunk."""
    a1 = extract_targets(chunk_text)
    targets = a1.targets

    a2 = extract_activities(chunk_text, targets)
    tuples = a2.tuples

    if tuples:
        a5 = validate(tuples, targets, chunk_text)
        tuples = a5.tuples

    return targets, tuples


def _persist(
    cache: AdaptiveExtractionCache,
    cache_key: str,
    targets: list[Target],
    tuples: list[ActivityTuple],
    patent_id: str | None,
) -> None:
    cache._cache[cache_key] = {
        "rule": {
            "targets": [t.__dict__ for t in targets],
            "tuples": [t.to_dict() for t in tuples],
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
_LETTER_GRADE_ROW_RE_LOCAL = re.compile(
    r"(?:^|[\s\n])\d{1,4}\.?\s+[A-E](?=\s|$|\n)",
    re.MULTILINE,
)


def _signal_score(chunk: str) -> int:
    """How much assay-relevant signal is in this chunk?

    Used to rank chunks for the HARVEST burst — we want to spend the
    budget on chunks most likely to have measurements, not the patent
    cover page or the claims section. Patent-agnostic signal:
      - assay keywords (IC50/EC50/Ki/Kd/CC50/...)
      - numeric value+unit hits
      - letter-grade row hits
    """
    n_keywords = len(_ASSAY_KEYWORD_RE_LOCAL.findall(chunk))
    n_values = len(_VALUE_UNIT_RE_LOCAL.findall(chunk))
    n_grades = len(_LETTER_GRADE_ROW_RE_LOCAL.findall(chunk))
    # Weight: value+unit hits are the strongest signal (they indicate
    # actual data, not just claim language)
    return n_keywords * 2 + n_values * 3 + n_grades


def _rank_chunks_by_signal(chunks: list[str]) -> list[str]:
    """Sort chunks descending by signal score. Ties broken by chunk
    position (earlier first) for determinism."""
    indexed = [(i, _signal_score(c), c) for i, c in enumerate(chunks)]
    indexed.sort(key=lambda x: (-x[1], x[0]))
    # Drop chunks with zero signal — no point sending them
    return [c for (_, score, c) in indexed if score > 0]


def _chunk_text(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    """Split a long text into overlapping chunks. Tries to break on
    paragraph boundaries when possible to avoid mid-sentence cuts."""
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    out: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + chunk_chars, n)
        if end < n:
            # Try to break on a paragraph or sentence boundary in the
            # last 400 chars of the chunk
            window = text[end - 400:end]
            best_break = window.rfind("\n\n")
            if best_break < 0:
                best_break = window.rfind(". ")
            if best_break >= 0:
                end = (end - 400) + best_break + 1
        out.append(text[pos:end])
        if end >= n:
            break
        pos = max(end - overlap, pos + 1)
    return out


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
    return fingerprint_table_shape(
        ["harvest_chunk"] + first_tokens, n_rows, sample,
    )
