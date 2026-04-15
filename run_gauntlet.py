#!/usr/bin/env python3
"""Strategy gauntlet — A/B test pipeline strategies against mini benchmark.

Usage:
    python3 run_gauntlet.py                    # Run all strategies
    python3 run_gauntlet.py --strategy 1.1     # Test specific strategy
    python3 run_gauntlet.py --compare          # Print comparison table
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class StrategyResult:
    strategy_id: str
    name: str
    test_pass: int = 0
    test_total: int = 0
    recall_delta: float = 0.0
    time_seconds: float = 0.0
    cost_usd: float = 0.0
    notes: str = ""


def load_test_set():
    with open("test_set/mini_benchmark.json") as f:
        return json.load(f)


# ============================================================
# Strategy implementations
# ============================================================

def baseline(ts) -> StrategyResult:
    """Current pipeline baseline — no changes."""
    from patent_extraction.iupac_to_smiles import _try_opsin, rule_based_clean
    from patent_extraction.smiles_utils import validate_smiles

    passed = 0
    total = len(ts.get("opsin_failing_llm_needed", []))
    for c in ts.get("opsin_failing_llm_needed", []):
        name = c["iupac"]
        smiles, _ = _try_opsin(name)
        if not smiles:
            cleaned = rule_based_clean(name)
            smiles, _ = _try_opsin(cleaned)
        if smiles and validate_smiles(smiles):
            passed += 1

    return StrategyResult("baseline", "Current pipeline", passed, total)


def strategy_1_1_dspy(ts) -> StrategyResult:
    """DSPy CleanIUPACName on OPSIN failures."""
    from patent_extraction.dspy_modules import clean_iupac_dspy
    from patent_extraction.iupac_to_smiles import _try_opsin, rule_based_clean
    from patent_extraction.smiles_utils import validate_smiles

    passed = 0
    total = len(ts.get("opsin_failing_llm_needed", []))
    for c in ts.get("opsin_failing_llm_needed", []):
        name = c["iupac"]
        cleaned = rule_based_clean(name)
        _, error = _try_opsin(cleaned)

        fixed = clean_iupac_dspy(cleaned, error)
        if fixed:
            smiles, _ = _try_opsin(fixed)
            if smiles and validate_smiles(smiles):
                passed += 1

    return StrategyResult("1.1", "DSPy CleanIUPAC", passed, total, notes="vs LLM clean")


def strategy_1_2_stereo(ts) -> StrategyResult:
    """Stereo calibration — check if mismatches are Tier B (acceptable)."""
    from patent_extraction.smiles_utils import get_connectivity_key

    passed = 0
    total = len(ts.get("stereo_mismatches", []))
    for m in ts.get("stereo_mismatches", []):
        bdb_key = m["bdb_key"]
        our_key = m["our_key"]
        # Tier B: same connectivity, BDB has undefined stereo
        same_conn = get_connectivity_key(bdb_key) == get_connectivity_key(our_key)
        bdb_undefined = "UHFFFAOYSA" in bdb_key
        if same_conn and bdb_undefined:
            passed += 1  # Tier B: acceptable mismatch

    return StrategyResult("1.2", "Stereo calibration", passed, total, notes="Tier B = acceptable")


def strategy_2_2_enumerate(ts) -> StrategyResult:
    """Markush enumeration — can we generate BDB-matching molecules?"""
    from patent_extraction.markush_agents import _symbolic_r_group_parse
    from patent_extraction.smiles_utils import validate_smiles, canonicalize_smiles, get_inchikey
    from patent_extraction import config
    from py2opsin import py2opsin
    import pandas as pd

    r_groups = _symbolic_r_group_parse('US10899738', config.DATA_DIR)

    # Try to enumerate from simple R-groups
    simple_r = {k: v for k, v in r_groups.items() if len(v) <= 5}

    enumerated_keys = set()
    for label, options in simple_r.items():
        for opt in options[:3]:
            # Try OPSIN on the substituent name
            smi = py2opsin(opt)
            if smi and validate_smiles(smi):
                canonical = canonicalize_smiles(smi)
                if canonical:
                    key = get_inchikey(canonical)
                    if key:
                        enumerated_keys.add(key)

    # Check against BDB
    bdb = pd.read_csv(config.OUTPUT_DIR / 'bindingdb' / 'our_patents.tsv', sep='\t', low_memory=False)
    bdb_df = bdb[bdb['Patent Number'].astype(str).str.contains('US10899738', na=False)]
    ik_col = [c for c in bdb_df.columns if 'inchi' in c.lower() and 'key' in c.lower()][0]
    bdb_keys = set(bdb_df[ik_col].dropna().astype(str).str.strip())

    matched = enumerated_keys & bdb_keys

    return StrategyResult(
        "2.2", "Markush enumeration", len(matched), len(enumerated_keys) or 1,
        notes=f"{len(enumerated_keys)} enumerated, {len(matched)} BDB match"
    )


def strategy_3_2_roundtrip(ts) -> StrategyResult:
    """OCSR roundtrip — SMILES → image → DECIMER → compare."""
    from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles
    from patent_extraction.smiles_utils import validate_smiles, canonicalize_smiles
    import tempfile, os

    test_smiles = [c["smiles"] for c in ts.get("decimer_failing", []) if c.get("smiles")][:5]

    passed = 0
    total = len(test_smiles)

    try:
        from DECIMER import predict_SMILES
    except ImportError:
        return StrategyResult("3.2", "OCSR roundtrip", 0, total, notes="DECIMER not installed")

    for smi in test_smiles:
        mol = MolFromSmiles(smi)
        if not mol:
            continue

        # Generate image
        img = Draw.MolToImage(mol, size=(300, 300))
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            tmp_path = f.name

        try:
            # DECIMER roundtrip
            decoded = predict_SMILES(tmp_path)
            if decoded and validate_smiles(decoded):
                c1 = canonicalize_smiles(smi)
                c2 = canonicalize_smiles(decoded.split('.')[0])
                if c1 == c2:
                    passed += 1
        finally:
            os.unlink(tmp_path)

    return StrategyResult("3.2", "OCSR roundtrip", passed, total, notes="SMILES→img→DECIMER")


def strategy_4_1_assay_norm(ts) -> StrategyResult:
    """Assay unit normalization test."""
    passed = 0
    total = len(ts.get("assay_test_cases", []))

    from patent_extraction.extract_assays import extract_assays_for_patent

    for tc in ts.get("assay_test_cases", []):
        assays = extract_assays_for_patent(tc["patent"])
        ex_key = tc["example"]

        if ex_key in assays:
            for a in assays[ex_key]:
                if 'pi3k' in a.assay_name.lower() and a.value_numeric:
                    expected = tc.get("expected_pi3k_nM")
                    if expected and abs(a.value_numeric - expected) < 0.01:
                        passed += 1
                        break

    return StrategyResult("4.1", "Assay normalization", passed, total, notes="IC50 value accuracy")


# ============================================================
# Gauntlet runner
# ============================================================

def run_gauntlet(strategies=None):
    ts = load_test_set()

    all_strategies = {
        "baseline": baseline,
        "1.1": strategy_1_1_dspy,
        "1.2": strategy_1_2_stereo,
        "2.2": strategy_2_2_enumerate,
        "3.2": strategy_3_2_roundtrip,
        "4.1": strategy_4_1_assay_norm,
    }

    if strategies:
        run = {k: v for k, v in all_strategies.items() if k in strategies}
    else:
        run = all_strategies

    results = []
    for sid, func in run.items():
        start = time.time()
        try:
            result = func(ts)
            result.time_seconds = round(time.time() - start, 1)
            results.append(result)
        except Exception as e:
            results.append(StrategyResult(sid, f"ERROR: {e}", 0, 1, time_seconds=round(time.time() - start, 1)))

    # Print comparison table
    print("\n" + "=" * 75)
    print("STRATEGY GAUNTLET — RESULTS")
    print("=" * 75)
    print(f"\n{'ID':<8} {'Strategy':<25} {'Pass':<6} {'Total':<7} {'Rate':<8} {'Time':<7} {'Notes'}")
    print("-" * 75)
    for r in results:
        rate = f"{r.test_pass/max(r.test_total,1):.0%}"
        print(f"{r.strategy_id:<8} {r.name:<25} {r.test_pass:<6} {r.test_total:<7} {rate:<8} {r.time_seconds:<7.1f} {r.notes}")
    print("=" * 75)

    return results


if __name__ == "__main__":
    strategies = None
    if len(sys.argv) > 1 and sys.argv[1] == "--strategy":
        strategies = sys.argv[2].split(",") if len(sys.argv) > 2 else None

    run_gauntlet(strategies)
