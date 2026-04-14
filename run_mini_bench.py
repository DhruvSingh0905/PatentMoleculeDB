#!/usr/bin/env python3
"""Mini benchmark — scores pipeline changes in <2 minutes.

Tests against a curated set of 20 compounds covering:
- DECIMER failures (BDB compounds we missed)
- OPSIN failures (names needing LLM fallback)
- Stereo mismatches (right molecule, wrong stereochem)
- Markush table pages (scaffold + R-group enumeration)

Usage:
    python3 run_mini_bench.py                    # Run all tests
    python3 run_mini_bench.py --opsin-only       # Just OPSIN tests
    python3 run_mini_bench.py --markush-only     # Just Markush tests
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patent_extraction.smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey, get_connectivity_key,
)


def test_opsin_stage(test_set):
    """Test IUPAC→SMILES on names that previously needed LLM."""
    from patent_extraction.iupac_to_smiles import rule_based_clean, _try_opsin

    results = []
    for c in test_set.get("opsin_failing_llm_needed", []):
        name = c["iupac"]
        expected_smiles = c.get("smiles", "")

        # Stage 1: OPSIN direct
        smiles, err = _try_opsin(name)
        if smiles and validate_smiles(smiles):
            results.append({"example": c["example"], "stage": "opsin_direct", "match": True})
            continue

        # Stage 2: rule_based_clean
        cleaned = rule_based_clean(name)
        smiles, err = _try_opsin(cleaned)
        if smiles and validate_smiles(smiles):
            results.append({"example": c["example"], "stage": "rule_cleaned", "match": True})
            continue

        # Stage 2c: autocorrect (if integrated)
        try:
            from patent_extraction.ocr_autocorrect import autocorrect_iupac
            corrected = autocorrect_iupac(cleaned)
            if corrected != cleaned:
                smiles, err = _try_opsin(corrected)
                if smiles and validate_smiles(smiles):
                    results.append({"example": c["example"], "stage": "autocorrect", "match": True})
                    continue
        except ImportError:
            pass

        results.append({"example": c["example"], "stage": "failed", "match": False})

    passed = sum(1 for r in results if r["match"])
    print(f"\nOPSIN Stage: {passed}/{len(results)} resolved without LLM")
    for r in results:
        status = "pass" if r["match"] else "FAIL"
        print(f"  [{status}] {r['example']}: {r['stage']}")
    return passed, len(results)


def test_stereo(test_set):
    """Check if stereo mismatches are resolvable."""
    results = []
    for m in test_set.get("stereo_mismatches", []):
        bdb_key = m["bdb_key"]
        our_key = m["our_key"]
        conn = m["connectivity"]
        same_conn = get_connectivity_key(bdb_key) == get_connectivity_key(our_key)
        results.append({
            "connectivity": conn,
            "same_molecule": same_conn,
            "stereo_diff": bdb_key != our_key,
        })

    resolvable = sum(1 for r in results if r["same_molecule"])
    print(f"\nStereo: {resolvable}/{len(results)} are same molecule (stereo-only diff)")
    return resolvable, len(results)


def test_markush_pages(test_set):
    """Check if Markush pages can be parsed for R-group definitions."""
    results = []
    for page_info in test_set.get("markush_pages", []):
        page_path = Path(page_info["file"])
        if not page_path.exists():
            results.append({"page": page_info["page"], "has_rgroups": False, "cpd_count": 0})
            continue

        text = page_path.read_text()
        import re

        # Count compound entries
        cpd_matches = re.findall(r'(?:Cpd\.?\s*No\.?|No\.)\s*(\d+)', text, re.IGNORECASE)
        # Check for R-group definitions
        has_rgroups = bool(re.search(r'R\d+\s*(?:is|=|represents)', text, re.IGNORECASE))
        # Check for table structure
        has_table = '<table>' in text.lower()

        results.append({
            "page": page_info["page"],
            "has_rgroups": has_rgroups,
            "has_table": has_table,
            "cpd_count": len(cpd_matches),
            "cpd_ids": cpd_matches[:5],
        })

    print(f"\nMarkush Pages:")
    for r in results:
        rg = "R-groups" if r["has_rgroups"] else "no R-groups"
        tbl = "table" if r.get("has_table") else "no table"
        print(f"  Page {r['page']}: {r['cpd_count']} compounds, {rg}, {tbl}")
    return sum(r["cpd_count"] for r in results), len(results)


def test_decimer_crops(test_set):
    """Test DECIMER on specific failing crops (if installed)."""
    try:
        from DECIMER import predict_SMILES
    except ImportError:
        print("\nDECIMER: not installed (pip install decimer)")
        return 0, len(test_set.get("decimer_crops_to_test", []))

    bdb_keys = {c["inchikey"] for c in test_set.get("decimer_failing", [])}

    results = []
    for crop_path in test_set.get("decimer_crops_to_test", []):
        if not Path(crop_path).exists():
            results.append({"crop": crop_path, "valid": False, "bdb_match": False})
            continue

        raw = predict_SMILES(crop_path)
        if raw:
            smiles = raw.split('.')[0].strip()
            valid = validate_smiles(smiles)
            if valid:
                canonical = canonicalize_smiles(smiles)
                key = get_inchikey(canonical) if canonical else None
                conn = get_connectivity_key(key) if key else None
                bdb_conn = {get_connectivity_key(k) for k in bdb_keys if k}
                match = conn in bdb_conn if conn else False
                results.append({"crop": Path(crop_path).name, "valid": True, "bdb_match": match, "smiles": smiles[:40]})
            else:
                results.append({"crop": Path(crop_path).name, "valid": False, "bdb_match": False})
        else:
            results.append({"crop": Path(crop_path).name, "valid": False, "bdb_match": False})

    valid = sum(1 for r in results if r["valid"])
    matched = sum(1 for r in results if r["bdb_match"])
    print(f"\nDECIMER Crops: {valid}/{len(results)} valid SMILES, {matched} BDB connectivity match")
    for r in results:
        status = "match" if r["bdb_match"] else ("valid" if r["valid"] else "FAIL")
        print(f"  [{status}] {r['crop']}: {r.get('smiles', 'N/A')}")
    return matched, len(results)


def main():
    start = time.time()

    with open("test_set/mini_benchmark.json") as f:
        test_set = json.load(f)

    only = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("MINI BENCHMARK — Rapid Iteration Test")
    print("=" * 60)

    total_pass = total_tests = 0

    if not only or only == "--opsin-only":
        p, t = test_opsin_stage(test_set)
        total_pass += p; total_tests += t

    if not only or only == "--stereo-only":
        p, t = test_stereo(test_set)
        total_pass += p; total_tests += t

    if not only or only == "--markush-only":
        p, t = test_markush_pages(test_set)
        total_pass += p; total_tests += t

    if not only or only == "--decimer-only":
        p, t = test_decimer_crops(test_set)
        total_pass += p; total_tests += t

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_pass}/{total_tests} | Time: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
