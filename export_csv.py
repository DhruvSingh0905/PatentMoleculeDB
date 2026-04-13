#!/usr/bin/env python3
"""Export extracted compounds in Jie's required CSV format.

Runs assay extraction live from patent markdown pages, attaches to compounds.
Enantiomers kept as SEPARATE entries (R/S not collapsed).
Markush used as context during extraction, not independently enumerated.
"""

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patent_extraction import config
from patent_extraction.extract_assays import extract_assays_for_patent


def export_patent_csv(patent_ids=None, output_path=None, versions=None):
    if patent_ids is None:
        patent_ids = config.PATENT_IDS
    if output_path is None:
        output_path = config.OUTPUT_DIR / "PatentMoleculeDB_export.csv"
    if versions is None:
        versions = ["decimer_v1", "google_patents_v1", "final_v2", "v13"]

    # Fixed columns matching Jie's format
    headers = [
        "patent_id",
        "compound_id",
        "iupac_name",
        "canonical_smiles",
        "inchikey",
        "target",
        # Primary assay (varies per patent — populated dynamically)
        "primary_assay_name",
        "primary_assay_value",
        "primary_assay_unit",
        "primary_assay_qualifier",
        # Secondary assay
        "secondary_assay_name",
        "secondary_assay_value",
        "secondary_assay_unit",
        "secondary_assay_qualifier",
        # All assay data as semicolon-separated string
        "all_assays",
        # Extraction metadata
        "extraction_method",
        "confidence_tier",
        "processing_status",
        "MW",
    ]

    all_rows = []
    seen_keys = {}
    stats = {}

    for pid in patent_ids:
        # Find results file
        data = None
        source_ver = None
        for ver in versions:
            path = config.OUTPUT_DIR / ver / f"{pid}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                source_ver = ver
                break

        if data is None:
            continue

        # Extract assay data live from markdown pages
        assay_data = extract_assays_for_patent(pid)

        compounds = data.get("exemplified_compounds", [])
        patent_stats = {
            "total": len(compounds),
            "validated": 0,
            "with_assays": 0,
            "assay_measurements": 0,
            "source": source_ver,
        }

        for c in compounds:
            status = c.get("processing_status", "pending")
            if status != "validated":
                continue

            smiles = c.get("canonical_smiles", "")
            if not smiles:
                continue

            patent_stats["validated"] += 1
            inchikey = c.get("inchikey", "")
            ex = c.get("example_number", "") or ""

            # Match assay data by example number
            assays = []
            keys_to_try = [
                ex.lower().replace(' ', ''),
                re.sub(r'[^0-9]', '', ex),
                f"example{re.sub(r'[^0-9]', '', ex)}",
            ]
            for key in keys_to_try:
                if key in assay_data:
                    assays = assay_data[key]
                    break

            # Also check compound's own assay_data field (from prior pipeline runs)
            if not assays and c.get("assay_data"):
                assays = c["assay_data"]
                # Convert dicts to usable format
                if assays and isinstance(assays[0], dict):
                    pass  # Already dict format

            # Format assay columns
            primary_name = primary_val = primary_unit = primary_qual = ""
            secondary_name = secondary_val = secondary_unit = secondary_qual = ""
            all_assays_str = ""

            if assays:
                patent_stats["with_assays"] += 1
                patent_stats["assay_measurements"] += len(assays)

                # Format all assays as string
                assay_strs = []
                for i, a in enumerate(assays):
                    if isinstance(a, dict):
                        name = a.get("assay_name", "")
                        val = a.get("value_raw", "")
                        unit = a.get("unit", "")
                        qual = a.get("qualifier", "") or ""
                    else:
                        name = a.assay_name
                        val = a.value_raw
                        unit = a.unit
                        qual = a.qualifier or ""

                    assay_strs.append(f"{name}={val} {unit}".strip())

                    if i == 0:
                        primary_name = name
                        primary_val = val
                        primary_unit = unit
                        primary_qual = qual
                    elif i == 1:
                        secondary_name = name
                        secondary_val = val
                        secondary_unit = unit
                        secondary_qual = qual

                all_assays_str = "; ".join(assay_strs)

            # Target from assay name
            target = ""
            if primary_name:
                name_l = primary_name.lower()
                if "pi3k" in name_l:
                    target = "PI3K delta"
                elif "menin" in name_l:
                    target = "Menin"
                elif "sgk" in name_l:
                    target = "SGK-1"
                elif "antiviral" in name_l or "a549" in name_l:
                    target = "Influenza"
                elif "fret" in name_l:
                    target = "Protease"

            # MW
            dl = c.get("drug_likeness")
            mw = ""
            if dl and isinstance(dl, dict):
                mw = dl.get("mw", "")
            if not mw and c.get("ms_mh_plus"):
                mw = round(c["ms_mh_plus"] - 1.008, 2)

            # Confidence tier
            tier = c.get("confidence_tier", "")
            if isinstance(tier, int):
                tier_map = {1: "dual_confirmed", 2: "single_validated", 3: "single_unvalidated"}
                tier = tier_map.get(tier, str(tier))

            row = {
                "patent_id": pid,
                "compound_id": ex,
                "iupac_name": c.get("iupac_name", "") or "",
                "canonical_smiles": smiles,
                "inchikey": inchikey,
                "target": target,
                "primary_assay_name": primary_name,
                "primary_assay_value": primary_val,
                "primary_assay_unit": primary_unit,
                "primary_assay_qualifier": primary_qual,
                "secondary_assay_name": secondary_name,
                "secondary_assay_value": secondary_val,
                "secondary_assay_unit": secondary_unit,
                "secondary_assay_qualifier": secondary_qual,
                "all_assays": all_assays_str,
                "extraction_method": c.get("extraction_method", ""),
                "confidence_tier": tier,
                "processing_status": status,
                "MW": mw,
            }

            if inchikey in seen_keys:
                existing = seen_keys[inchikey]
                if len(row["iupac_name"]) > len(existing["iupac_name"]):
                    seen_keys[inchikey] = row
            else:
                seen_keys[inchikey] = row

        stats[pid] = patent_stats

    all_rows = sorted(seen_keys.values(), key=lambda r: (r["patent_id"], r["compound_id"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary
    print(f"\nExported {len(all_rows)} compounds to {output_path}")
    print(f"\n{'='*70}")
    print(f"PATENT COMPOUND EXTRACTION — SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Patent':<25} {'Validated':<12} {'With Assays':<14} {'Measurements':<14} {'Source'}")
    print(f"{'-'*75}")

    total_validated = 0
    total_assays = 0
    total_measurements = 0
    for pid in patent_ids:
        s = stats.get(pid, {})
        v = s.get("validated", 0)
        a = s.get("with_assays", 0)
        m = s.get("assay_measurements", 0)
        src = s.get("source", "—")
        total_validated += v
        total_assays += a
        total_measurements += m
        if v > 0:
            print(f"{pid:<25} {v:<12} {a:<14} {m:<14} {src}")

    print(f"{'-'*75}")
    print(f"{'TOTAL':<25} {total_validated:<12} {total_assays:<14} {total_measurements:<14}")
    print(f"\nEnantiomers: kept as separate entries (R/S not collapsed)")
    print(f"Markush: used as context, not independently enumerated")
    print(f"Assay data: extracted from patent tables (no API calls)")
    print(f"{'='*70}")

    return all_rows


if __name__ == "__main__":
    export_patent_csv()
