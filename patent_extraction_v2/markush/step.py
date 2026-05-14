"""Pipeline-integrated Markush enumeration — patent-agnostic.

Drives the existing `markush_mapper.derive_multi_level_cores` +
`markush_enumerate.enumerate_markush` engine from a list of validated
text-extracted compounds. Designed to be called from `pipeline.py` so
every patent gets Markush coverage when conditions warrant it.

Per-scaffold caching: each scaffold's enumeration is cached separately
under `markush_enumeration_scaffold` step name with a key derived from
(patent_id, scaffold_inchikey, rgroup_libraries_hash). Re-running one
patent with a different R-library only invalidates the affected scaffold.
"""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

from ..core import config, step_cache
from .enumerate import (
    enumerate_markush,
    build_global_fragment_vocab,
)
from .mapper import (
    derive_multi_level_cores,
    derive_all_scaffolds_from_examples,
)
from ..core.models import Compound, MarkushFormula, RGroupDef
from ..core.smiles_utils import canonicalize_smiles, get_inchikey, get_connectivity_key

if TYPE_CHECKING:
    from .google_patents import FormatAudit

logger = logging.getLogger(__name__)


# ── Trigger logic — patent-agnostic ────────────────────────────────

def should_enumerate_markush(
    audit: "FormatAudit | None",
    validated_compounds: list[Compound],
    completeness: float,
) -> tuple[bool, str]:
    """Decide whether Markush enumeration would help this patent.

    Returns (should_run, reason).

    Generic policy (no patent-specific hardcoding):
    1. Need ≥10 validated compounds for scaffold derivation to be meaningful.
    2. Fire if either:
       a) audit detected a substituent-table format (Markush is the right tool)
       b) completeness is below 0.95 and we have enough scaffolding data
    3. Skip if we already have great recall (>95%) — no headroom to gain.
    """
    n = len(validated_compounds)
    if n < 10:
        return False, f"only {n} validated compounds (need ≥10 for scaffold derivation)"

    if completeness >= 0.95:
        return False, f"completeness already {completeness:.0%}, no headroom"

    has_substituent_table = bool(audit and audit.has_substituent_table)
    if has_substituent_table:
        return True, "audit detected substituent-table format (Markush territory)"

    if completeness < 0.95:
        return True, f"completeness {completeness:.0%} < 95% — Markush may fill the gap"

    return False, "no trigger condition met"


# ── Per-scaffold worker (top-level for ProcessPoolExecutor pickle) ─

def _enumerate_one_scaffold(args):
    """Enumerate one scaffold — designed for parallel execution.

    Returns (idx, match_count, n_rgroups, compounds_decomp, compounds_scored).
    """
    i, scaffold_info, patent_id = args

    scaf_smi = scaffold_info['scaffold_smiles']
    match_count = scaffold_info['match_count']
    rgroup_libs = scaffold_info['rgroup_libraries']
    decomp_results = scaffold_info['decomposition_results']

    # Adaptive scoring budget by scaffold support
    if match_count >= 50:
        budget = 500
    elif match_count >= 20:
        budget = 300
    elif match_count >= 10:
        budget = 200
    else:
        budget = 100

    r_groups = {
        label: RGroupDef(label=label, options_text=[], options_smiles=smiles_list)
        for label, smiles_list in rgroup_libs.items()
    }
    formula = MarkushFormula(
        patent_id=patent_id, core_smiles=scaf_smi, r_groups=r_groups
    )

    # Decomposition mode: reconstruct known examples (sanity check, no novel combos)
    compounds_decomp = enumerate_markush(
        formula, cap=len(decomp_results),
        mode="decomposition",
        decomposition_results=decomp_results,
        scaffold_index=i,
    )

    # Scored mode: novel combinations ranked by frequency + co-occurrence
    compounds_scored = enumerate_markush(
        formula, cap=budget,
        mode="scored",
        decomposition_results=decomp_results,
        scaffold_index=i,
    )

    return i, match_count, len(rgroup_libs), compounds_decomp, compounds_scored


# ── Main entry point ──────────────────────────────────────────────

def run_markush_enumeration(
    patent_id: str,
    validated_compounds: list[Compound],
    max_workers: int = 4,
) -> list[Compound]:
    """Generate enumerated compounds from a patent's validated examples.

    Patent-agnostic: derives multi-level cores + Murcko scaffolds from the
    SMILES set, then enumerates each scaffold with the existing scored engine.

    Args:
        patent_id: Patent identifier (for logging + Compound provenance).
        validated_compounds: Text-extracted compounds with canonical_smiles.
        max_workers: Parallel processes for per-scaffold enumeration.

    Returns:
        New unique Compound objects (SMILES not already in validated_compounds).
    """
    smiles_in = [c.canonical_smiles for c in validated_compounds if c.canonical_smiles]
    if len(smiles_in) < 10:
        logger.info(
            f"Markush enum {patent_id}: only {len(smiles_in)} validated compounds; "
            f"need ≥10 for scaffold derivation. Skipping."
        )
        return []

    logger.info(f"Markush enum {patent_id}: deriving scaffolds from {len(smiles_in)} compounds")

    # Build the global fragment vocab (cached across all patents)
    vocab = build_global_fragment_vocab()
    logger.info(f"  Global fragment vocab: {len(vocab)} fragments available")

    # Multi-level cores (shallow → deep) catch broad and specific scaffolds
    ml_cores = derive_multi_level_cores(smiles_in, min_match=3, max_cores=5)
    # Murcko gives sub-series coverage that multi-level may miss
    murcko = derive_all_scaffolds_from_examples(smiles_in, min_match=5, max_scaffolds=10)

    # Merge, preferring multi-level
    seen_cores = {s['scaffold_smiles'] for s in ml_cores}
    scaffolds = list(ml_cores)
    for ms in murcko:
        if ms['scaffold_smiles'] not in seen_cores:
            scaffolds.append(ms)
            seen_cores.add(ms['scaffold_smiles'])

    if not scaffolds:
        logger.info(f"Markush enum {patent_id}: no viable scaffolds derived. Skipping.")
        return []

    logger.info(
        f"  Scaffolds: {len(scaffolds)} total "
        f"({len(ml_cores)} multi-level + {len(scaffolds) - len(ml_cores)} Murcko)"
    )

    # Conn keys we already have — skip duplicates from enumeration
    existing_conn: set[str] = set()
    for c in validated_compounds:
        if c.canonical_smiles:
            ik = get_inchikey(c.canonical_smiles)
            if ik:
                existing_conn.add(get_connectivity_key(ik))

    # Build per-scaffold cache keys so a patent can re-use enumerated SMILES
    # across runs even if other scaffolds change.
    def _scaffold_cache_input(scaffold_info: dict) -> dict:
        scaf_smi = scaffold_info['scaffold_smiles']
        scaf_canon = canonicalize_smiles(scaf_smi) or scaf_smi
        scaf_ik = get_inchikey(scaf_canon) or "none"
        # Stable hash of the rgroup library keys + sorted SMILES per position
        rgroup_libs = scaffold_info['rgroup_libraries']
        lib_serialized = json.dumps(
            {label: sorted(opts)[:50] for label, opts in rgroup_libs.items()},
            sort_keys=True,
        )
        lib_hash = hashlib.sha256(lib_serialized.encode()).hexdigest()[:12]
        return {
            "patent_id": patent_id,
            "scaffold_inchikey": scaf_ik[:14],   # connectivity prefix
            "rgroup_lib_hash": lib_hash,
            "step_version": config.STEP_VERSIONS.get("markush_enumeration", "v1"),
        }

    args_list = [(i, sf, patent_id) for i, sf in enumerate(scaffolds)]
    new_compounds: list[Compound] = []
    seen_new_conn: set[str] = set()

    def _ingest(compounds_decomp, compounds_scored):
        for c in compounds_decomp + compounds_scored:
            if not c.canonical_smiles:
                continue
            ik = get_inchikey(c.canonical_smiles)
            if not ik:
                continue
            conn = get_connectivity_key(ik)
            if conn in existing_conn or conn in seen_new_conn:
                continue
            seen_new_conn.add(conn)
            # Ensure provenance is set
            c.extraction_method = c.extraction_method or "markush_enumeration"
            new_compounds.append(c)

    # Run enumeration with per-scaffold cache. We split the args into
    # (cached_hits, fresh_args) so the parallel pool only handles uncached.
    fresh_args = []
    cache_hits = 0
    for args in args_list:
        i, scaffold_info, _ = args
        cache_key_input = _scaffold_cache_input(scaffold_info)
        cached = step_cache.get_cached(
            patent_id, "markush_enumeration_scaffold", cache_key_input,
        )
        if cached is not None:
            # Cached payload: {"decomp": [Compound dicts], "scored": [Compound dicts], "stats": {...}}
            try:
                c_decomp = [Compound(**cd) for cd in cached.get("decomp", [])]
                c_scored = [Compound(**cs) for cs in cached.get("scored", [])]
                _ingest(c_decomp, c_scored)
                cache_hits += 1
                logger.info(
                    f"  Scaffold {i+1} [CACHED] → "
                    f"{len(c_decomp)} decomp + {len(c_scored)} scored "
                    f"({len(new_compounds)} unique-new total)"
                )
                continue
            except Exception as e:
                logger.warning(f"  Scaffold {i+1} cache hydrate failed ({e}); recomputing")
        fresh_args.append(args)

    if cache_hits:
        logger.info(f"  Markush per-scaffold cache: {cache_hits} hit / {len(args_list)} total")

    def _store_scaffold_result(scaffold_info, c_decomp, c_scored):
        """Persist per-scaffold enumeration to the step cache."""
        cache_key_input = _scaffold_cache_input(scaffold_info)
        payload = {
            "decomp": [c.model_dump() for c in c_decomp],
            "scored": [c.model_dump() for c in c_scored],
            "stats": {
                "scaffold_smiles": scaffold_info['scaffold_smiles'],
                "match_count": scaffold_info.get('match_count', 0),
                "n_decomp": len(c_decomp),
                "n_scored": len(c_scored),
            },
        }
        step_cache.store(
            patent_id, "markush_enumeration_scaffold", cache_key_input, payload,
        )

    if fresh_args:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map index → scaffold_info so we can store after each result
                future_to_args = {
                    executor.submit(_enumerate_one_scaffold, a): a
                    for a in fresh_args
                }
                for future in as_completed(future_to_args):
                    args = future_to_args[future]
                    idx, scaffold_info, _ = args
                    try:
                        i, match_count, n_rg, c_decomp, c_scored = future.result()
                        _ingest(c_decomp, c_scored)
                        _store_scaffold_result(scaffold_info, c_decomp, c_scored)
                        logger.info(
                            f"  Scaffold {i+1}: {match_count} matches, {n_rg} R-groups → "
                            f"{len(c_decomp)} decomp + {len(c_scored)} scored "
                            f"({len(new_compounds)} unique-new total)"
                        )
                    except Exception as e:
                        logger.warning(f"  Scaffold {idx+1} failed: {e}")
        except Exception as e:
            logger.warning(f"Markush enum {patent_id}: parallel pool failed ({e}); sequential")
            for args in fresh_args:
                idx, scaffold_info, _ = args
                try:
                    i, match_count, n_rg, c_decomp, c_scored = _enumerate_one_scaffold(args)
                    _ingest(c_decomp, c_scored)
                    _store_scaffold_result(scaffold_info, c_decomp, c_scored)
                except Exception as ee:
                    logger.warning(f"  Scaffold {idx+1} failed: {ee}")

    logger.info(
        f"Markush enum {patent_id}: {len(new_compounds)} new compounds "
        f"(from {len(scaffolds)} scaffolds, {cache_hits} cached)"
    )
    return new_compounds
