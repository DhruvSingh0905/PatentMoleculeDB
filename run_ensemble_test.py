#!/usr/bin/env python3
"""Run Google Patents Route 1 on all 8 patents, then benchmark.

Tests the ensemble: Google Patents (free) as primary,
OCR pipeline as fallback for any patents that fail.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from patent_extraction import config
from patent_extraction.google_patents import extract_patent_google, fetch_patent_text
from patent_extraction.smiles_utils import get_inchikey, get_connectivity_key, canonicalize_smiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("ensemble_test")


def run_google_patents_all():
    """Run Google Patents extraction on all 8 patents."""
    output_dir = config.OUTPUT_DIR / "google_patents_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for patent_id in config.PATENT_IDS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Google Patents: {patent_id}")
        logger.info(f"{'='*60}")

        start = time.time()

        # Step 1: Fetch clean text
        text_data = fetch_patent_text(patent_id)
        if not text_data:
            logger.warning(f"  {patent_id}: No text from Google Patents")
            results[patent_id] = {
                "patent_id": patent_id,
                "status": "no_text",
                "compounds": 0,
                "validated": 0,
                "elapsed_s": round(time.time() - start, 1),
            }
            continue

        desc_len = len(text_data.get('description', ''))
        claims_len = len(text_data.get('claims', ''))
        logger.info(f"  Text: {desc_len:,} desc chars, {claims_len:,} claims chars")

        # Step 2: Extract compounds
        compounds = extract_patent_google(patent_id)
        elapsed = time.time() - start

        # Save results in same format as pipeline output
        compound_dicts = []
        for c in compounds:
            d = c.model_dump()
            # Convert enums to strings
            for k, v in d.items():
                if hasattr(v, 'value'):
                    d[k] = v.value
            compound_dicts.append(d)

        output_data = {
            "patent_id": patent_id,
            "exemplified_compounds": compound_dicts,
            "stats": {
                "total_compounds": len(compounds),
                "with_smiles": sum(1 for c in compounds if c.canonical_smiles),
                "extraction_method": "google_patents",
                "cost_usd": 0.0,
                "elapsed_s": round(elapsed, 1),
                "description_chars": desc_len,
                "claims_chars": claims_len,
            }
        }

        output_path = output_dir / f"{patent_id}.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        results[patent_id] = {
            "patent_id": patent_id,
            "status": "ok",
            "compounds": len(compounds),
            "validated": sum(1 for c in compounds if c.processing_status == "validated"),
            "elapsed_s": round(elapsed, 1),
            "desc_chars": desc_len,
            "claims_chars": claims_len,
        }

        logger.info(f"  Result: {len(compounds)} compounds, {results[patent_id]['validated']} validated in {elapsed:.1f}s")

    return results


def benchmark_against_bindingdb(version="google_patents_v1"):
    """Benchmark Google Patents results against BindingDB."""
    import pandas as pd

    # Load BindingDB
    bdb_path = config.OUTPUT_DIR / "bindingdb" / "our_patents.tsv"
    if not bdb_path.exists():
        # Try the big file
        bdb_path = Path("/Users/dhruvsingh/Downloads/Patent (1)/BindingDB_All.tsv")
        if not bdb_path.exists():
            logger.warning("BindingDB file not found — skipping benchmark")
            return {}

    logger.info(f"\nLoading BindingDB from {bdb_path}...")

    try:
        from patent_extraction.benchmark import load_bindingdb, filter_by_patents, benchmark_patent
        df = load_bindingdb(bdb_path)
        patent_data = filter_by_patents(df, config.PATENT_IDS)
    except Exception as e:
        logger.error(f"BindingDB load failed: {e}")
        return {}

    reports = []
    for patent_id in config.PATENT_IDS:
        results_path = config.OUTPUT_DIR / version / f"{patent_id}.json"
        if not results_path.exists():
            logger.info(f"  {patent_id}: No extraction results")
            continue

        if patent_id not in patent_data:
            reports.append({
                "patent_id": patent_id,
                "bindingdb_compounds": 0,
                "note": "No BindingDB coverage",
            })
            continue

        try:
            report = benchmark_patent(patent_id, results_path, patent_data[patent_id])
            reports.append(report)
        except Exception as e:
            logger.error(f"  {patent_id} benchmark failed: {e}")

    return reports


def print_summary(gp_results, benchmark_reports):
    """Print a clean summary table."""
    print("\n" + "="*80)
    print("GOOGLE PATENTS ENSEMBLE TEST — RESULTS")
    print("="*80)

    print(f"\n{'Patent ID':<25} {'Status':<10} {'Compounds':<12} {'Validated':<12} {'Time (s)':<10}")
    print("-"*69)

    total_compounds = 0
    total_validated = 0
    for pid in config.PATENT_IDS:
        r = gp_results.get(pid, {})
        status = r.get('status', 'skipped')
        compounds = r.get('compounds', 0)
        validated = r.get('validated', 0)
        elapsed = r.get('elapsed_s', 0)
        total_compounds += compounds
        total_validated += validated
        print(f"{pid:<25} {status:<10} {compounds:<12} {validated:<12} {elapsed:<10}")

    print("-"*69)
    print(f"{'TOTAL':<25} {'—':<10} {total_compounds:<12} {total_validated:<12}")

    if benchmark_reports:
        print(f"\n{'='*80}")
        print("BINDINGDB BENCHMARK")
        print(f"{'='*80}")

        print(f"\n{'Patent ID':<25} {'Ours':<8} {'BDB':<8} {'Matched':<10} {'Precision':<12} {'Recall':<10}")
        print("-"*73)

        for report in benchmark_reports:
            pid = report.get('patent_id', '?')
            if 'note' in report:
                print(f"{pid:<25} {'—':<8} {report.get('bindingdb_compounds', 0):<8} {'—':<10} {'N/A':<12} {'N/A':<10}")
                continue
            ours = report.get('our_validated', 0)
            bdb = report.get('bindingdb_compounds', 0)
            matched = report.get('matched_full_inchikey', 0)
            precision = report.get('precision_full', 0)
            recall = report.get('recall_full', 0)
            print(f"{pid:<25} {ours:<8} {bdb:<8} {matched:<10} {precision:<12.1%} {recall:<10.1%}")

    print(f"\nTotal cost: $0.00 (Google Patents = free)")
    print(f"{'='*80}")


if __name__ == "__main__":
    print("Running Google Patents extraction on all 8 patents...")
    print("Cost: $0 (no API calls)\n")

    gp_results = run_google_patents_all()

    print("\nRunning BindingDB benchmark...")
    benchmark_reports = benchmark_against_bindingdb()

    print_summary(gp_results, benchmark_reports)
