#!/usr/bin/env python3
"""Export extracted compounds in Jie's required CSV format.

Output columns:
  patent_id, compound_id, iupac_name, canonical_smiles, inchikey,
  target, flap_binding_ki_uM, flap_binding_ki_qualifier, flap_binding_ki_n_runs,
  hwb_ltb4_ic50_uM, hwb_ltb4_ic50_qualifier, hwb_ltb4_ic50_n_runs,
  flap_binding_assay, hwb_assay,
  extraction_method, confidence_tier, processing_status, MW

Enantiomers: kept as SEPARATE entries (R/S not collapsed).
Markush: context used during extraction, not enumerated independently.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patent_extraction import config


def export_patent_csv(patent_ids=None, output_path=None, versions=None):
    """Export compounds from one or more patents to Jie's CSV format.

    Args:
        patent_ids: List of patent IDs. Defaults to all.
        output_path: Output CSV path. Defaults to output/PatentMoleculeDB_export.csv.
        versions: List of output versions to check, in priority order.
    """
    if patent_ids is None:
        patent_ids = config.PATENT_IDS
    if output_path is None:
        output_path = config.OUTPUT_DIR / "PatentMoleculeDB_export.csv"
    if versions is None:
        versions = ["decimer_v1", "google_patents_v1", "final_v2", "v13"]

    headers = [
        "patent_id",
        "compound_id",
        "iupac_name",
        "canonical_smiles",
        "inchikey",
        "target",
        "flap_binding_ki_uM",
        "flap_binding_ki_qualifier",
        "flap_binding_ki_n_runs",
        "hwb_ltb4_ic50_uM",
        "hwb_ltb4_ic50_qualifier",
        "hwb_ltb4_ic50_n_runs",
        "flap_binding_assay",
        "hwb_assay",
        "extraction_method",
        "confidence_tier",
        "processing_status",
        "MW",
    ]

    all_rows = []
    seen_keys = {}  # inchikey -> row (for dedup, keep best)

    for pid in patent_ids:
        # Find results file (check versions in priority order)
        data = None
        for ver in versions:
            path = config.OUTPUT_DIR / ver / f"{pid}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                break

        if data is None:
            continue

        compounds = data.get("exemplified_compounds", [])

        for c in compounds:
            status = c.get("processing_status", "pending")
            if status != "validated":
                continue

            smiles = c.get("canonical_smiles", "")
            if not smiles:
                continue

            inchikey = c.get("inchikey", "")

            # Extract assay data if available
            assay_data = c.get("assay_data", [])
            flap_ki = ""
            flap_qual = ""
            flap_n = ""
            hwb_ic50 = ""
            hwb_qual = ""
            hwb_n = ""
            target = ""

            for assay in assay_data:
                name = (assay.get("assay_name", "") or "").lower()
                if "flap" in name or "binding" in name:
                    flap_ki = assay.get("value_numeric", assay.get("value_raw", ""))
                    flap_qual = assay.get("qualifier", "")
                    flap_n = assay.get("n_runs", "")
                    target = target or "FLAP"
                elif "hwb" in name or "ltb4" in name or "ic50" in name:
                    hwb_ic50 = assay.get("value_numeric", assay.get("value_raw", ""))
                    hwb_qual = assay.get("qualifier", "")
                    hwb_n = assay.get("n_runs", "")

            # Drug-likeness MW
            dl = c.get("drug_likeness")
            mw = ""
            if dl:
                if isinstance(dl, dict):
                    mw = dl.get("mw", "")
                else:
                    mw = getattr(dl, "mw", "")

            # MS data as MW fallback
            if not mw and c.get("ms_mh_plus"):
                mw = round(c["ms_mh_plus"] - 1.008, 2)  # M+H → MW

            # Confidence tier
            tier = c.get("confidence_tier", "")
            if isinstance(tier, int):
                tier_map = {1: "dual_confirmed", 2: "single_validated", 3: "single_unvalidated"}
                tier = tier_map.get(tier, str(tier))

            row = {
                "patent_id": pid,
                "compound_id": c.get("example_number", ""),
                "iupac_name": c.get("iupac_name", "") or "",
                "canonical_smiles": smiles,
                "inchikey": inchikey,
                "target": target,
                "flap_binding_ki_uM": flap_ki,
                "flap_binding_ki_qualifier": flap_qual or "",
                "flap_binding_ki_n_runs": flap_n or "",
                "hwb_ltb4_ic50_uM": hwb_ic50,
                "hwb_ltb4_ic50_qualifier": hwb_qual or "",
                "hwb_ltb4_ic50_n_runs": hwb_n or "",
                "flap_binding_assay": "",
                "hwb_assay": "",
                "extraction_method": c.get("extraction_method", ""),
                "confidence_tier": tier,
                "processing_status": status,
                "MW": mw,
            }

            # Dedup: keep separate enantiomers (different full InChIKey)
            # But collapse duplicates with identical InChIKey (prefer longer IUPAC)
            if inchikey in seen_keys:
                existing = seen_keys[inchikey]
                if len(row["iupac_name"]) > len(existing["iupac_name"]):
                    seen_keys[inchikey] = row
            else:
                seen_keys[inchikey] = row

    # Sort by patent_id then compound_id
    all_rows = sorted(seen_keys.values(), key=lambda r: (r["patent_id"], r["compound_id"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Exported {len(all_rows)} compounds to {output_path}")

    # Summary
    from collections import Counter
    patents = Counter(r["patent_id"] for r in all_rows)
    for pid, count in patents.most_common():
        print(f"  {pid}: {count} compounds")

    return all_rows


if __name__ == "__main__":
    export_patent_csv()
