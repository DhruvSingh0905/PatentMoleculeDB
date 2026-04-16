"""Diagnose why BDB compounds are missed by Markush enumeration.

Loads BDB ground truth, our extracted compounds, and the scaffold cache
to identify specific failure patterns in reconstruction.
"""
import json
import sys
from pathlib import Path
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
from patent_extraction import config
from patent_extraction.smiles_utils import (
    canonicalize_smiles, get_inchikey, get_connectivity_key,
    get_stereo_flattened_key,
)
from patent_extraction.benchmark import (
    load_bindingdb, filter_by_patents, extract_bindingdb_inchikeys,
)
from patent_extraction.markush_mapper import derive_all_scaffolds_from_examples

PATENT_ID = "US10899738"


def get_bdb_smiles(patent_id: str) -> dict[str, str]:
    """Return {connectivity_key: SMILES} for all BDB compounds."""
    import pandas as pd
    tsv_path = config.OUTPUT_DIR / "bindingdb" / "our_patents.tsv"
    df = pd.read_csv(tsv_path, sep='\t', low_memory=False, on_bad_lines='skip')

    # Filter to patent
    patent_col = [c for c in df.columns if 'patent' in c.lower()]
    if not patent_col:
        print(f"No patent column. Cols: {list(df.columns)[:10]}")
        return {}
    patent_col = patent_col[0]

    import re
    pid_base = re.sub(r'[AB]\d+$', '', patent_id)
    variants = [patent_id, pid_base, patent_id.replace('US', ''), pid_base.replace('US', '')]
    num = pid_base[2:] if pid_base.startswith('US') else pid_base
    if len(num) >= 7:
        formatted = f"{num[:-6]},{num[-6:-3]},{num[-3:]}"
        variants.extend([f"US {formatted}", f"US{formatted}"])

    mask = pd.Series(False, index=df.index)
    pvals = df[patent_col].astype(str)
    for v in variants:
        mask |= pvals.str.contains(v, na=False, regex=False)

    pdf = df[mask]
    print(f"BDB rows for {patent_id}: {len(pdf)}")

    smiles_col = [c for c in pdf.columns if 'smiles' in c.lower() and 'ligand' in c.lower()]
    if not smiles_col:
        smiles_col = [c for c in pdf.columns if 'smiles' in c.lower()]
    smiles_col = smiles_col[0] if smiles_col else None

    result = {}
    for val in pdf[smiles_col].dropna():
        smi = str(val).strip()
        can = canonicalize_smiles(smi)
        if can:
            ik = get_inchikey(can)
            if ik:
                conn = get_connectivity_key(ik)
                if conn:
                    result[conn] = can
    return result


def get_our_smiles(patent_id: str) -> dict[str, str]:
    """Return {connectivity_key: SMILES} for our extracted compounds."""
    # Try latest output version
    for v in ["v12", "v9", "v8", "v7", "v6", "v1"]:
        path = config.OUTPUT_DIR / v / f"{patent_id}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            compounds = data.get('exemplified_compounds', []) + data.get('markush_enumerated', [])
            result = {}
            for c in compounds:
                smi = c.get('canonical_smiles') or c.get('parent_smiles')
                if smi:
                    ik = get_inchikey(smi)
                    if ik:
                        result[get_connectivity_key(ik)] = smi
            print(f"Loaded {len(result)} compounds from {path}")
            return result
    return {}


def analyze_missed(bdb: dict, ours: dict) -> list[str]:
    """Return BDB SMILES we missed."""
    missed_keys = set(bdb.keys()) - set(ours.keys())
    matched_keys = set(bdb.keys()) & set(ours.keys())
    print(f"\nBDB total: {len(bdb)}")
    print(f"Our total: {len(ours)}")
    print(f"Matched (connectivity): {len(matched_keys)}")
    print(f"Missed: {len(missed_keys)}")

    missed_smiles = [bdb[k] for k in missed_keys]
    return missed_smiles


def scaffold_analysis(missed_smiles: list[str], our_smiles: list[str]):
    """Analyze what scaffolds the missed compounds belong to."""
    print("\n=== SCAFFOLD ANALYSIS OF MISSED COMPOUNDS ===")

    scaffold_counts = Counter()
    scaffold_examples = {}

    for smi in missed_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)
            scaffold_counts[scaffold_smi] += 1
            if scaffold_smi not in scaffold_examples:
                scaffold_examples[scaffold_smi] = []
            if len(scaffold_examples[scaffold_smi]) < 3:
                scaffold_examples[scaffold_smi].append(smi)
        except:
            pass

    print(f"\n{len(scaffold_counts)} unique scaffolds in missed compounds:")
    for scaffold, count in scaffold_counts.most_common(15):
        print(f"  [{count:3d}] {scaffold[:80]}")
        for ex in scaffold_examples[scaffold][:2]:
            mw = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(ex))
            print(f"        MW={mw:.0f}  {ex[:80]}")


def tanimoto_analysis(missed_smiles: list[str], our_smiles: list[str]):
    """Check how close missed compounds are to our extracted ones."""
    print("\n=== TANIMOTO SIMILARITY ANALYSIS ===")

    # Build fingerprints for our compounds
    our_fps = []
    for smi in our_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            our_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

    if not our_fps:
        print("No our fingerprints")
        return

    # For each missed compound, find max Tanimoto to any of ours
    sim_buckets = Counter()
    close_misses = []

    for smi in missed_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        max_sim = max(DataStructs.TanimotoSimilarity(fp, ofp) for ofp in our_fps)

        if max_sim >= 0.9:
            sim_buckets["≥0.9 (trivially close)"] += 1
            close_misses.append((max_sim, smi))
        elif max_sim >= 0.8:
            sim_buckets["0.8-0.9 (close analog)"] += 1
            close_misses.append((max_sim, smi))
        elif max_sim >= 0.6:
            sim_buckets["0.6-0.8 (related)"] += 1
        elif max_sim >= 0.4:
            sim_buckets["0.4-0.6 (distant)"] += 1
        else:
            sim_buckets["<0.4 (unrelated)"] += 1

    print("Similarity distribution of missed compounds to our nearest extracted:")
    for bucket, count in sorted(sim_buckets.items()):
        print(f"  {bucket}: {count}")

    if close_misses:
        print(f"\nTop 10 closest misses (Tanimoto ≥ 0.8):")
        for sim, smi in sorted(close_misses, reverse=True)[:10]:
            mw = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smi))
            print(f"  Tc={sim:.3f} MW={mw:.0f} {smi[:80]}")


def mw_analysis(missed_smiles: list[str], our_smiles: list[str]):
    """Compare MW distributions."""
    print("\n=== MW DISTRIBUTION ===")

    our_mws = []
    for smi in our_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            our_mws.append(Chem.Descriptors.ExactMolWt(mol))

    missed_mws = []
    for smi in missed_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            missed_mws.append(Chem.Descriptors.ExactMolWt(mol))

    if our_mws:
        print(f"Our MW range: {min(our_mws):.0f} - {max(our_mws):.0f} (mean {sum(our_mws)/len(our_mws):.0f})")
    if missed_mws:
        print(f"Missed MW range: {min(missed_mws):.0f} - {max(missed_mws):.0f} (mean {sum(missed_mws)/len(missed_mws):.0f})")

    # Check how many missed are outside our MW filter (150-1500)
    outside = [mw for mw in missed_mws if mw < 150 or mw > 1500]
    print(f"Missed outside MW 150-1500 filter: {len(outside)}")

    # Breakdown by MW range
    ranges = [(0, 300), (300, 400), (400, 500), (500, 600), (600, 700), (700, 1500)]
    print("\nMW breakdown (missed):")
    for lo, hi in ranges:
        n = sum(1 for mw in missed_mws if lo <= mw < hi)
        if n > 0:
            print(f"  {lo}-{hi}: {n}")


def decomposition_check(missed_smiles: list[str], our_smiles: list[str]):
    """Check if missed compounds can be decomposed against our scaffolds."""
    print("\n=== R-GROUP DECOMPOSITION CHECK ===")
    print("Can missed BDB compounds be decomposed against our scaffolds?")

    # Get scaffolds from our compounds
    all_our = [Chem.MolFromSmiles(s) for s in our_smiles if Chem.MolFromSmiles(s)]

    # Extract unique Murcko scaffolds from our compounds
    our_scaffolds = Counter()
    for mol in all_our:
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            our_scaffolds[Chem.MolToSmiles(scaf)] += 1
        except:
            pass

    print(f"Our top 5 scaffolds:")
    top_scaffolds = our_scaffolds.most_common(5)
    for smi, count in top_scaffolds:
        print(f"  [{count:3d}] {smi[:80]}")

    # Check each missed compound's scaffold
    missed_on_our_scaffold = 0
    missed_novel_scaffold = 0

    for smi in missed_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smi = Chem.MolToSmiles(scaf)
            if scaf_smi in our_scaffolds:
                missed_on_our_scaffold += 1
            else:
                missed_novel_scaffold += 1
        except:
            missed_novel_scaffold += 1

    print(f"\nMissed compounds on scaffolds we already have: {missed_on_our_scaffold}")
    print(f"Missed compounds on novel scaffolds: {missed_novel_scaffold}")

    return missed_on_our_scaffold, missed_novel_scaffold


if __name__ == "__main__":
    bdb = get_bdb_smiles(PATENT_ID)
    ours = get_our_smiles(PATENT_ID)

    missed_smiles = analyze_missed(bdb, ours)
    our_smiles_list = list(ours.values())

    mw_analysis(missed_smiles, our_smiles_list)
    scaffold_analysis(missed_smiles, our_smiles_list)
    tanimoto_analysis(missed_smiles, our_smiles_list)
    decomposition_check(missed_smiles, our_smiles_list)
