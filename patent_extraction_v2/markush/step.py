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
import re
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


# ── Explicit scaffold + substituent-table path ────────────────────────
# `run_markush_enumeration` (above) MINES scaffolds from ≥10 already-
# extracted FULL compounds, then recombines their decomposed R-groups.
# That fails for genus patents whose drawn examples are R-group FRAGMENTS,
# not whole molecules — there are no ≥10 full compounds to mine, so the
# < 10 guard trips and it returns []. US9718825 is exactly this case.
#
# This second path is the one the user described: DECIMER reads the drawn
# genus scaffold into a core SMILES carrying R-group attachment points,
# the patent's TEXT substituent table (routes.text_markush.extract_
# markush_dict — proven to parse 246 SMILES for $0 on US9718825) supplies
# the per-R option lists, and we enumerate species directly. No prior full
# compounds, no decomposition, no LM at enumerate time.
#
# DECIMER is the ONLY held dependency: it supplies `scaffold_core_smiles`.
# Pass that in (parameter) so the rest of the path runs today and DECIMER
# plugs into the same seam when it is un-held.

# Match a BRACKETED R-label pseudo-atom only — `[R1]`, `[R19a]`, `[R1:1]`.
# Bare unbracketed `R1` is NOT valid SMILES (R is not an element), and the
# greedy `[a-z]?` suffix would otherwise eat the following aromatic atom in
# `[R1]c1cc...`-style cores. The label number → attachment position;
# matching the engine's own `re.search(r'\d+', label)` mapping, a trailing
# letter (R19a/R19b) collapses to the numeric position (known limitation,
# consistent with enumerate_markush's pos_map).
_RLABEL_TO_STAR = re.compile(r"\[R(\d+)[a-z]?(?::\d+)?\]")


def normalize_scaffold_rlabels(scaffold_smiles: str) -> str:
    """Convert DECIMER-style bracketed R-group labels in a scaffold SMILES
    to the `[*:n]` attachment-point form that `enumerate_markush` expects.

    DECIMER (and image→SMILES tools) emit drawn R-groups as bracketed
    pseudo-atoms: `[R1]`, `[R19a]`. `_instantiate_core` keys substitution
    off `[*:n]` (or `[n*]`), so normalize `[R1]` → `[*:1]`. SMILES already
    in `[*:n]`/`[n*]` form is returned untouched. Patent-agnostic: matches
    the label SHAPE, no patent IDs.
    """
    if not scaffold_smiles:
        return scaffold_smiles
    return _RLABEL_TO_STAR.sub(lambda m: f"[*:{m.group(1)}]", scaffold_smiles)


def enumerate_from_scaffold_table(
    patent_id: str,
    scaffold_core_smiles: str,
    rgroup_options: "dict[str, list[str]]",
    cap: int = 500,
    existing_conn: "set[str] | None" = None,
    formula_name: str = "Formula I",
    mode: str = "sample",
) -> list[Compound]:
    """Enumerate specific compounds from a drawn scaffold + parsed table.

    The DECIMER-scaffold + text-table Markush path. Unlike
    `run_markush_enumeration`, this needs NO prior full compounds — it
    works for fragment-drawn genus patents (US9718825) where the mining
    path returns 0.

    Args:
        patent_id: for logging + Compound provenance.
        scaffold_core_smiles: genus core with R-group attachment points
            (`[*:1]`/`[1*]`/`R1`...). DECIMER supplies this; we normalize
            R-labels to `[*:n]`.
        rgroup_options: {label -> [resolved SMILES options]} from the text
            substituent table (decoupled from routes.text_markush so this
            module has no upward import).
        cap: max species to emit (sample/exhaustive are capped — the raw
            combinatorial space is astronomically large).
        existing_conn: connectivity keys we already have; emitted species
            with a matching key are dropped as duplicates.
        mode: "sample" (random product up to cap) or "exhaustive".

    Returns:
        New unique Compound objects with extraction_method tagged.
    """
    if not scaffold_core_smiles:
        logger.info(
            f"Markush table-enum {patent_id}: no scaffold core SMILES "
            f"(DECIMER held / not supplied); skipping"
        )
        return []

    core = normalize_scaffold_rlabels(scaffold_core_smiles)
    r_groups = {
        label: RGroupDef(
            label=label, options_text=[],
            options_smiles=[s for s in opts if s],
        )
        for label, opts in rgroup_options.items()
        if any(opts)
    }
    if not r_groups:
        logger.info(
            f"Markush table-enum {patent_id}: no resolvable R-group options "
            f"in table; skipping"
        )
        return []

    formula = MarkushFormula(
        patent_id=patent_id, formula_name=formula_name,
        core_smiles=core, r_groups=r_groups, source="image",
    )
    logger.info(
        f"Markush table-enum {patent_id}: scaffold={core[:60]} | "
        f"{len(r_groups)} R-groups | "
        f"{sum(len(d.options_smiles) for d in r_groups.values())} total options | "
        f"cap={cap}"
    )
    compounds = enumerate_markush(formula, cap=cap, mode=mode)

    # Dedup against connectivity keys we already have (and within this run).
    out: list[Compound] = []
    seen_new: set[str] = set()
    for c in compounds:
        if not c.canonical_smiles:
            continue
        ik = get_inchikey(c.canonical_smiles)
        conn = get_connectivity_key(ik) if ik else None
        if conn and existing_conn and conn in existing_conn:
            continue
        if conn and conn in seen_new:
            continue
        if conn:
            seen_new.add(conn)
        # Force the table-path route tag so the audit can distinguish this
        # (DECIMER-scaffold + text-table) path from run_markush_enumeration's
        # scaffold-mining path. The engine sets "markush_enumeration"; override.
        c.extraction_method = "markush_table_enumeration"
        out.append(c)

    logger.info(
        f"Markush table-enum {patent_id}: {len(out)} new species "
        f"({len(compounds)} enumerated, {len(compounds) - len(out)} dup/empty)"
    )
    return out
