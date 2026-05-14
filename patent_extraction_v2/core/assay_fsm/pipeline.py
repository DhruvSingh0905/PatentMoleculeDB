"""Orchestrator entry point for the new assay extraction pipeline.

`extract_page(text, patent_id)` is the public function called by the
shim in `routes/google_assays.py`. It runs Stages 0 → 6 for ONE page
of patent text and returns a dict[compound_id → list[AssayResult]].

`extract_for_patent(patent_id)` is the per-patent orchestrator that
loads cached page text, dispatches to `extract_page`, and aggregates
results. Drop-in replacement for the legacy
`extract_assays_for_patent`.

The pipeline NEVER returns "empty silently" — if a page has any
detectable assay-table region, the LLM realigner fires (cache miss)
or returns the cached result (cache hit). If a page has NO assay
region (truly non-assay content), the function returns {} with that
status reflected in the diagnostic dict.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..adaptive_extraction_cache import AdaptiveExtractionCache
from ..models import AssayResult
from .cross_validator import MergeResult, cross_validate
from .llm_realigner import LLMResult, realign_region
from .normalizer import normalize_page
from .region_detector import Region, detect_regions
from .row_aligner import align_rows, detect_n_columns
from .tokenizer import tokenize
from .vocab_learner import learn_from_llm_output
from .vocabulary import AssayVocabulary

logger = logging.getLogger(__name__)


# Module-level singleton for the loaded vocabulary. The first call
# to `extract_page` populates it; subsequent calls reuse with mtime-
# tracked reload so curator-promoted entries take effect mid-run.
_VOCAB: AssayVocabulary | None = None
_CACHE: AdaptiveExtractionCache | None = None


def _get_vocab() -> AssayVocabulary:
    global _VOCAB
    if _VOCAB is None:
        canonical = (
            Path(config.OUTPUT_DIR).parent
            / "patent_extraction_v2"
            / "data"
            / "assay_vocabulary.json"
        )
        # Fallback: search relative to the package root if config.OUTPUT_DIR
        # doesn't yield the right path.
        if not canonical.exists():
            canonical = Path(__file__).resolve().parent.parent.parent / "data" / "assay_vocabulary.json"
        _VOCAB = AssayVocabulary.load(canonical)
    else:
        _VOCAB.reload()
    return _VOCAB


def _get_cache() -> AdaptiveExtractionCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = AdaptiveExtractionCache()
    return _CACHE


@dataclass
class PageDiagnostic:
    """Per-page extraction stats, useful for audit."""
    n_regions: int = 0
    n_compounds: int = 0
    n_measurements: int = 0
    fingerprints: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    n_llm_chunks: int = 0
    n_disagreements: int = 0
    n_vocab_discoveries: int = 0


# ── Per-page entry point ────────────────────────────────────────


def extract_page(
    text: str,
    *,
    patent_id: str | None = None,
    return_diagnostic: bool = False,
):
    """Run Stages 0-6 on one page of patent text.

    Args:
        text: Raw page text (any format — HTML, plain, markdown).
        patent_id: For logging + cache provenance.
        return_diagnostic: If True, returns (results, diagnostic).

    Returns:
        dict[cpd_key, list[AssayResult]] or (dict, PageDiagnostic).
    """
    vocab = _get_vocab()
    cache = _get_cache()

    diag = PageDiagnostic()
    out: dict[str, list[AssayResult]] = {}

    # Stage 0
    norm = normalize_page(text)
    if not norm:
        return (out, diag) if return_diagnostic else out

    # Stage 1
    regions = detect_regions(norm, vocab)
    diag.n_regions = len(regions)

    if not regions:
        return (out, diag) if return_diagnostic else out

    for region in regions:
        merge = _process_region(
            region, vocab=vocab, cache=cache, patent_id=patent_id, diag=diag,
        )
        # Merge into out (different regions on the same page may
        # contribute to the same compound — keep both)
        for cid, results in merge.assay_results.items():
            out.setdefault(cid, []).extend(results)
        diag.n_disagreements += merge.n_disagreement

    diag.n_compounds = len(out)
    diag.n_measurements = sum(len(v) for v in out.values())
    return (out, diag) if return_diagnostic else out


def _process_region(
    region: Region,
    *,
    vocab: AssayVocabulary,
    cache: AdaptiveExtractionCache,
    patent_id: str | None,
    diag: PageDiagnostic,
) -> MergeResult:
    """Run Stages 2-6 on one region."""
    # Stage 2: tokenize
    stream = tokenize(region.text, vocab)

    # Stage 4: row align (two-pass — discover n_columns, then sharpen)
    rows1 = align_rows(stream)
    n_cols = detect_n_columns(rows1)
    fsm_rows = align_rows(stream, expected_n_columns=n_cols) if n_cols else rows1

    # Stage 5: ALWAYS-FIRE LLM realigner (fingerprint-cached)
    llm_result: LLMResult = realign_region(
        region, patent_id=patent_id, cache=cache,
    )
    if llm_result.cache_hit:
        diag.cache_hits += 1
    else:
        diag.cache_misses += 1
        diag.n_llm_chunks += llm_result.n_chunks
    diag.fingerprints.append(llm_result.fingerprint)

    # Stage 5b: cross-validate
    merge = cross_validate(
        fsm_rows=fsm_rows,
        llm_rows=llm_result.rows,
        units=llm_result.units,
        fingerprint=llm_result.fingerprint,
        patent_id=patent_id,
        page_offset=region.start,
    )

    # Stage 6: vocab learning (only fires on fresh LLM extractions —
    # cached results already contributed their discoveries last time)
    if not llm_result.cache_hit and llm_result.rows:
        from ..adaptive_extraction_cache import RealignedTable
        n_added = learn_from_llm_output(
            RealignedTable(
                rows=llm_result.rows,
                units=llm_result.units,
                notes=llm_result.notes,
            ),
            vocab=vocab,
            fingerprint=llm_result.fingerprint,
            patent_id=patent_id or "unknown",
        )
        diag.n_vocab_discoveries += n_added

    return merge


# ── Per-patent entry point ──────────────────────────────────────


def extract_for_patent(
    patent_id: str,
    *,
    data_dir: Path | None = None,
    return_diagnostic: bool = False,
):
    """Run the pipeline across all pages of a patent.

    Reads:
      - per-page markdown files under `data_dir/{patent_id}/all_pages/`
      - the Google Patents description from
        `output_v2/gpatents_cache/{patent_id}.json`

    Returns:
        dict[cpd_key, list[AssayResult]] aggregated across all sources,
        OR (dict, diagnostic_dict) if `return_diagnostic=True`.

    The dict keys are sanitized compound ids (lowercase, prefix-stripped).
    Backward-compatible with the legacy `extract_assays_for_patent`.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    all_assays: dict[str, list[AssayResult]] = {}
    aggregate_diag = {
        "n_pages_processed": 0,
        "n_regions": 0,
        "n_compounds": 0,
        "n_measurements": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "n_llm_chunks": 0,
        "n_disagreements": 0,
        "n_vocab_discoveries": 0,
        "fingerprints": set(),
    }

    # Walk per-page markdown files
    for subdir in ("all_pages", "iupacs_clean"):
        pages_dir = data_dir / patent_id / subdir
        if not pages_dir.exists():
            continue
        for page_file in sorted(pages_dir.glob("page_*.md")):
            text = page_file.read_text(encoding="utf-8")
            results, diag = extract_page(
                text, patent_id=patent_id, return_diagnostic=True,
            )
            aggregate_diag["n_pages_processed"] += 1
            _aggregate(aggregate_diag, diag)
            _merge_into(all_assays, results)

    # Process the Google Patents description text (often contains
    # tables not present as separate pages).
    gp_cache = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
    if gp_cache.exists():
        try:
            gp = json.loads(gp_cache.read_text())
            description = gp.get("description", "") or ""
            if description:
                results, diag = extract_page(
                    description, patent_id=patent_id, return_diagnostic=True,
                )
                aggregate_diag["n_pages_processed"] += 1
                _aggregate(aggregate_diag, diag)
                _merge_into(all_assays, results)
        except Exception as e:
            logger.warning("GP cache read failed for %s: %r", patent_id, e)

    # ── Tier 2 + Tier 3 — Coverage gap detection + HARVEST burst ──
    # The cheap pipeline has done its work. Now run the gap detector
    # against the FULL aggregated text + the cheap-tier diagnostic to
    # decide whether the HARVEST safety net should fire. The flag
    # `HARVEST_BURST_ENABLED` (env: `HARVEST_BURST`) gates the entire
    # tier so we can disable it during regression debugging without
    # editing code.
    aggregate_diag["harvest_burst_fired"] = False
    aggregate_diag["harvest_burst_n_added"] = 0
    aggregate_diag["harvest_signals"] = []
    if getattr(config, "HARVEST_BURST_ENABLED", True):
        full_text = _gather_full_text(patent_id, data_dir)
        if full_text:
            from .gap_detector import score as _gap_score
            from .harvest import harvest_burst

            gs = _gap_score(full_text, aggregate_diag, all_assays)
            aggregate_diag["harvest_signals"] = list(gs.signals)
            if gs.fire_burst:
                logger.info(
                    "harvest_burst: triggered on %s — signals %s",
                    patent_id, gs.signals,
                )
                try:
                    burst = harvest_burst(full_text, patent_id=patent_id)
                except Exception as e:
                    logger.warning(
                        "harvest_burst: failed for %s: %r", patent_id, e,
                    )
                    burst = None
                if burst is not None:
                    burst_results = burst.to_assay_results()
                    aggregate_diag["harvest_burst_fired"] = True
                    aggregate_diag["harvest_n_targets"] = len(burst.targets)
                    aggregate_diag["harvest_n_tuples"] = len(burst.tuples)
                    aggregate_diag["harvest_n_chunks"] = burst.n_chunks
                    aggregate_diag["harvest_cache_hits"] = burst.cache_hits
                    aggregate_diag["harvest_cache_misses"] = burst.cache_misses
                    aggregate_diag["harvest_cost_usd"] = burst.cost_estimate_usd
                    n_before = sum(len(v) for v in all_assays.values())
                    _merge_into(all_assays, burst_results)
                    n_after = sum(len(v) for v in all_assays.values())
                    aggregate_diag["harvest_burst_n_added"] = n_after - n_before
                    logger.info(
                        "harvest_burst: %s — added %d measurements (%d → %d)",
                        patent_id, n_after - n_before, n_before, n_after,
                    )

    # Final dedup pass — collapse (value, unit, qualifier) triples per
    # compound, normalizing μM/uM/µM unit-string variants.
    all_assays = _dedup(all_assays)

    aggregate_diag["n_compounds"] = len(all_assays)
    aggregate_diag["n_measurements"] = sum(len(v) for v in all_assays.values())
    aggregate_diag["fingerprints"] = sorted(aggregate_diag["fingerprints"])

    logger.info(
        "extract_for_patent(%s): %d compounds, %d measurements "
        "(%d pages, %d regions, %d cache_hits, %d cache_misses, "
        "%d LLM chunks, %d disagreements, %d discoveries, harvest=%s)",
        patent_id,
        aggregate_diag["n_compounds"],
        aggregate_diag["n_measurements"],
        aggregate_diag["n_pages_processed"],
        aggregate_diag["n_regions"],
        aggregate_diag["cache_hits"],
        aggregate_diag["cache_misses"],
        aggregate_diag["n_llm_chunks"],
        aggregate_diag["n_disagreements"],
        aggregate_diag["n_vocab_discoveries"],
        aggregate_diag.get("harvest_burst_fired", False),
    )

    if return_diagnostic:
        return all_assays, aggregate_diag
    return all_assays


def _gather_full_text(patent_id: str, data_dir: Path) -> str:
    """Concatenate the patent's full text in the same order the cheap
    pipeline processed it. Used by the gap detector + HARVEST burst.
    """
    pieces: list[str] = []
    for subdir in ("all_pages", "iupacs_clean"):
        pages_dir = data_dir / patent_id / subdir
        if not pages_dir.exists():
            continue
        for page_file in sorted(pages_dir.glob("page_*.md")):
            try:
                pieces.append(page_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    gp_cache = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
    if gp_cache.exists():
        try:
            data = json.loads(gp_cache.read_text())
            desc = data.get("description", "") or ""
            if desc:
                pieces.append(desc)
        except Exception:
            pass
    return "\n".join(pieces)


# ── Helpers ──────────────────────────────────────────────────────


def _aggregate(agg: dict, diag: PageDiagnostic) -> None:
    agg["n_regions"] += diag.n_regions
    agg["cache_hits"] += diag.cache_hits
    agg["cache_misses"] += diag.cache_misses
    agg["n_llm_chunks"] += diag.n_llm_chunks
    agg["n_disagreements"] += diag.n_disagreements
    agg["n_vocab_discoveries"] += diag.n_vocab_discoveries
    agg["fingerprints"].update(f for f in diag.fingerprints if f)


def _merge_into(
    target: dict[str, list[AssayResult]],
    source: dict[str, list[AssayResult]],
) -> None:
    for cid, results in source.items():
        target.setdefault(cid, []).extend(results)


def _canonical_unit(u: str | None) -> str:
    if not u:
        return ""
    return u.replace("uM", "μM").replace("µM", "μM").strip()


def _dedup(
    all_assays: dict[str, list[AssayResult]],
) -> dict[str, list[AssayResult]]:
    """Collapse (value_numeric, canonical_unit, qualifier) triples.

    Two AssayResults with the same value+unit+qualifier are treated as
    duplicate captures of the same measurement (e.g., HTML extractor
    finding it once and the LLM realigner finding it again). We keep
    the entry with the more canonical assay name.
    """
    canonical_assay_tokens = (
        "ic50", "ic 50", "ec50", "ki ", "kd ", "fret", "ec_50", "ic_50",
    )

    def name_score(name: str) -> tuple[int, int]:
        nl = (name or "").lower()
        return (
            1 if any(t in nl for t in canonical_assay_tokens) else 0,
            -len(name or ""),
        )

    out: dict[str, list[AssayResult]] = {}
    for cpd_key, arr in all_assays.items():
        by_key: dict[tuple, AssayResult] = {}
        for a in arr:
            key = (a.value_numeric, _canonical_unit(a.unit), a.qualifier)
            if key[0] is None:
                # Keep no-value entries individually (don't collapse)
                by_key[(id(a),) + key] = a   # type: ignore
                continue
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = a
            elif name_score(a.assay_name) > name_score(existing.assay_name):
                by_key[key] = a
        out[cpd_key] = list(by_key.values())
    return out
