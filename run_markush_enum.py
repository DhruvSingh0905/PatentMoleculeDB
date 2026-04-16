"""Run Markush enumeration for US10899738 and benchmark against BDB.

Strategy:
1. Derive scaffolds from text-extracted compounds
2. For each scaffold: decomposition-mode (reconstruct known) + scored enumeration (novel combos)
3. Benchmark against BindingDB ground truth
4. Save combined results
"""
import json
import logging
import sys
from pathlib import Path
from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdRGroupDecomposition
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import re

RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, str(Path(__file__).parent))
from patent_extraction import config
from patent_extraction.smiles_utils import (
    canonicalize_smiles, get_inchikey, get_connectivity_key,
    validate_smiles, molecular_weight,
)
from patent_extraction.markush_mapper import (
    derive_all_scaffolds_from_examples, derive_multi_level_cores,
)
from patent_extraction.markush_enumerate import (
    enumerate_markush, _instantiate_core, build_rgroup_stats,
    maxmin_diverse_subset, build_global_fragment_vocab,
)
from patent_extraction.models import MarkushFormula, RGroupDef, Compound

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PATENT_ID = "US10899738"


def load_bdb_ground_truth() -> dict[str, str]:
    """Load BDB {connectivity_key: SMILES}."""
    tsv = pd.read_csv('output/bindingdb/our_patents.tsv', sep='\t',
                       low_memory=False, on_bad_lines='skip')
    patent_col = [c for c in tsv.columns if 'patent' in c.lower()][0]
    mask = tsv[patent_col].astype(str).str.contains(
        '10899738|10,899,738', na=False, regex=True)
    bdb_rows = tsv[mask]

    smiles_col = [c for c in bdb_rows.columns
                  if 'smiles' in c.lower() and 'ligand' in c.lower()][0]
    result = {}
    for val in bdb_rows[smiles_col].dropna():
        smi = str(val).strip()
        can = canonicalize_smiles(smi)
        if can:
            ik = get_inchikey(can)
            if ik:
                result[get_connectivity_key(ik)] = can
    return result


def load_our_text_compounds() -> list[str]:
    """Load validated SMILES from pipeline combined output."""
    from patent_extraction.google_patents_tables import extract_table_compounds_from_gp

    smiles_set = set()

    # Try canonical output first (from pipeline.py)
    canonical = Path(f'output/results/{PATENT_ID}/combined.json')
    if canonical.exists():
        with open(canonical) as f:
            d = json.load(f)
        for c in d['exemplified_compounds']:
            if c.get('processing_status') == 'validated' and c.get('canonical_smiles'):
                smiles_set.add(c['canonical_smiles'])
        logger.info(f"Loaded {len(smiles_set)} compounds from {canonical}")
    else:
        # Fallback: load from final_v2 + table parser
        fallback = Path(f'output/final_v2/{PATENT_ID}.json')
        if fallback.exists():
            with open(fallback) as f:
                d = json.load(f)
            for c in d['exemplified_compounds']:
                if c.get('processing_status') == 'validated' and c.get('canonical_smiles'):
                    smiles_set.add(c['canonical_smiles'])

        table_compounds = extract_table_compounds_from_gp(PATENT_ID)
        for c in table_compounds:
            if c.canonical_smiles and len(c.canonical_smiles) > 10:
                smiles_set.add(c.canonical_smiles)

        logger.info(f"Fallback: {len(smiles_set)} compounds (final_v2 + table parser)")

    return list(smiles_set)


def _enumerate_one_scaffold(args):
    """Enumerate one scaffold — designed for parallel execution."""
    i, scaffold_info, patent_id = args

    scaf_smi = scaffold_info['scaffold_smiles']
    match_count = scaffold_info['match_count']
    rgroup_libs = scaffold_info['rgroup_libraries']
    decomp_results = scaffold_info['decomposition_results']

    # Adaptive budget
    if match_count >= 50:
        budget = 500
    elif match_count >= 20:
        budget = 300
    elif match_count >= 10:
        budget = 200
    else:
        budget = 100

    # Build MarkushFormula
    r_groups = {}
    for label, smiles_list in rgroup_libs.items():
        r_groups[label] = RGroupDef(
            label=label, options_text=[], options_smiles=smiles_list
        )

    formula = MarkushFormula(
        patent_id=patent_id, core_smiles=scaf_smi, r_groups=r_groups
    )

    # Mode 1: Decomposition-based (reconstruct known examples)
    compounds_decomp = enumerate_markush(
        formula, cap=len(decomp_results),
        mode="decomposition",
        decomposition_results=decomp_results,
    )

    # Mode 2: Scored enumeration (novel combinations)
    compounds_scored = enumerate_markush(
        formula, cap=budget,
        mode="scored",
        decomposition_results=decomp_results,
    )

    return i, match_count, len(rgroup_libs), compounds_decomp, compounds_scored


def enumerate_all_scaffolds(example_smiles: list[str],
                            budget_per_scaffold: int = 300) -> list[Compound]:
    """Derive scaffolds and enumerate R-group combinations using scored mode."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger.info(f"\n{'='*60}")
    logger.info(f"MULTI-LEVEL CORE ENUMERATION")
    logger.info(f"{'='*60}")

    # Build global fragment vocab first (one-time, cached)
    vocab = build_global_fragment_vocab()
    logger.info(f"Global fragment vocab: {len(vocab)} fragments")

    # Use multi-level cores (shallow → deep) instead of Murcko-only
    all_scaffolds = derive_multi_level_cores(
        example_smiles, min_match=3, max_cores=5
    )

    # Also add Murcko-derived scaffolds for sub-series coverage
    murcko_scaffolds = derive_all_scaffolds_from_examples(
        example_smiles, min_match=5, max_scaffolds=10
    )

    # Merge, preferring multi-level cores (they come first)
    seen_cores = {s['scaffold_smiles'] for s in all_scaffolds}
    for ms in murcko_scaffolds:
        if ms['scaffold_smiles'] not in seen_cores:
            all_scaffolds.append(ms)
            seen_cores.add(ms['scaffold_smiles'])

    logger.info(f"Total scaffolds: {len(all_scaffolds)} ({len(all_scaffolds) - len(murcko_scaffolds)} multi-level + {len(murcko_scaffolds)} Murcko)")

    # Prepare args for parallel execution
    args_list = [(i, sf, PATENT_ID) for i, sf in enumerate(all_scaffolds)]

    all_compounds = []
    seen_conn = set()

    # Run scaffolds in parallel (ProcessPool for CPU-bound RDKit work)
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_enumerate_one_scaffold, args): args[0]
                       for args in args_list}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    i, match_count, n_rg, compounds_decomp, compounds_scored = future.result()

                    for c in compounds_decomp + compounds_scored:
                        if c.canonical_smiles:
                            ik = get_inchikey(c.canonical_smiles)
                            if ik:
                                conn = get_connectivity_key(ik)
                                if conn not in seen_conn:
                                    seen_conn.add(conn)
                                    all_compounds.append(c)

                    logger.info(f"  Scaffold {i+1}: {match_count} matches, "
                                 f"{n_rg} R-groups → "
                                 f"{len(compounds_decomp)} decomp + "
                                 f"{len(compounds_scored)} scored, "
                                 f"{len(seen_conn)} total unique")
                except Exception as e:
                    logger.warning(f"  Scaffold {idx+1} failed: {e}")
    except Exception:
        # Fallback to sequential if multiprocessing fails
        logger.info("Parallel execution failed, falling back to sequential")
        for args in args_list:
            try:
                i, match_count, n_rg, compounds_decomp, compounds_scored = _enumerate_one_scaffold(args)
                for c in compounds_decomp + compounds_scored:
                    if c.canonical_smiles:
                        ik = get_inchikey(c.canonical_smiles)
                        if ik:
                            conn = get_connectivity_key(ik)
                            if conn not in seen_conn:
                                seen_conn.add(conn)
                                all_compounds.append(c)
                logger.info(f"  Scaffold {i+1}: {len(compounds_decomp)} decomp + "
                             f"{len(compounds_scored)} scored, "
                             f"{len(seen_conn)} total unique")
            except Exception as e:
                logger.warning(f"  Scaffold {args[0]+1} failed: {e}")

    return all_compounds


def benchmark(our_text_smiles: list[str], markush_compounds: list[Compound],
              bdb: dict[str, str]):
    """Report recall at each level."""
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK vs BindingDB ({len(bdb)} compounds)")
    logger.info(f"{'='*60}")

    # Text-only
    text_conn = set()
    for smi in our_text_smiles:
        ik = get_inchikey(smi)
        if ik:
            text_conn.add(get_connectivity_key(ik))
    text_matched = text_conn & set(bdb.keys())
    logger.info(f"\nText extraction: {len(text_matched)}/{len(bdb)} "
                 f"({100*len(text_matched)/len(bdb):.1f}%)")

    # Markush-only
    markush_conn = set()
    for c in markush_compounds:
        if c.canonical_smiles:
            ik = get_inchikey(c.canonical_smiles)
            if ik:
                markush_conn.add(get_connectivity_key(ik))
    markush_matched = markush_conn & set(bdb.keys())
    logger.info(f"Markush enumeration: {len(markush_matched)}/{len(bdb)} "
                 f"({100*len(markush_matched)/len(bdb):.1f}%) "
                 f"[from {len(markush_conn)} unique compounds]")

    # Combined
    combined = text_conn | markush_conn
    combined_matched = combined & set(bdb.keys())
    logger.info(f"Combined: {len(combined_matched)}/{len(bdb)} "
                 f"({100*len(combined_matched)/len(bdb):.1f}%)")

    still_missed = set(bdb.keys()) - combined
    logger.info(f"Still missed: {len(still_missed)}")

    # MW of still-missed
    missed_mws = []
    for conn in still_missed:
        mol = Chem.MolFromSmiles(bdb[conn])
        if mol:
            missed_mws.append(Descriptors.ExactMolWt(mol))
    if missed_mws:
        logger.info(f"Still-missed MW: {min(missed_mws):.0f}-{max(missed_mws):.0f} "
                     f"(mean {sum(missed_mws)/len(missed_mws):.0f})")

    return {
        'text_recall': len(text_matched),
        'markush_recall': len(markush_matched),
        'combined_recall': len(combined_matched),
        'bdb_total': len(bdb),
        'still_missed': len(still_missed),
        'markush_total': len(markush_conn),
    }


def main():
    bdb = load_bdb_ground_truth()
    logger.info(f"BDB ground truth: {len(bdb)} unique compounds")

    our_text = load_our_text_compounds()
    logger.info(f"Our text-extracted: {len(our_text)} compounds")

    # Run multi-scaffold enumeration with scoring
    markush_compounds = enumerate_all_scaffolds(our_text, budget_per_scaffold=300)
    logger.info(f"\nTotal Markush compounds: {len(markush_compounds)}")

    logger.info(f"Unique Markush compounds: {len(markush_compounds)}")

    # Benchmark
    results = benchmark(our_text, markush_compounds, bdb)

    # Save
    output = {
        'patent_id': PATENT_ID,
        'exemplified_compounds': [json.loads(
            Compound(patent_id=PATENT_ID, canonical_smiles=s, inchikey=get_inchikey(s),
                     processing_status='validated', extraction_method='text_extraction'
                     ).model_dump_json()
        ) for s in our_text if get_inchikey(s)],
        'markush_enumerated': [json.loads(c.model_dump_json()) for c in markush_compounds],
        'stats': results,
    }

    outpath = Path('output/v15/US10899738.json')
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
