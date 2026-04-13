#!/usr/bin/env python3
"""Export extracted compounds in Jie's CSV format.

- Comma-delimited
- compound_id as bare integers (strip "Example ", "Cpd. No. " prefixes)
- Assay columns pivoted per-patent (assay name baked into column header)
- Only compounds WITH assay data in output
- "—" values → empty cells
- No internal metadata (extraction_method, confidence_tier, processing_status)
- SMILES and InChIKey kept as useful extra columns
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patent_extraction import config
from patent_extraction.extract_assays import extract_assays_for_patent


def _strip_compound_prefix(raw: str) -> str:
    """'Example 101' → '101', 'Cpd. No. 148' → '148'."""
    s = re.sub(r'^(Example|Cpd\.?\s*No\.?|Compound)\s*', '', raw, flags=re.IGNORECASE).strip()
    return s


def _clean_value(v: str) -> str:
    """Replace dash placeholders with empty string."""
    if v in ('—', '-', 'N/A', 'n/a', 'NT', 'nd', 'ND', 'No inhibition'):
        return ''
    return v


def _assay_col_name(assay_name: str, unit: str) -> str:
    """Build Jie-style column name: 'pi3k_delta_ic50_nM'."""
    name = assay_name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = name.strip('_')
    # Shorten common patterns
    name = re.sub(r'_value$', '', name)
    name = re.sub(r'_values$', '', name)
    # Remove duplicate fragments (e.g. "pi3k_delta_ic50_cd69_ic50" from merged headers)
    # If it contains two assay names, it's a bad merge — skip it
    ic50_count = name.count('ic50')
    if ic50_count > 1:
        return None  # Signal to skip this column
    if unit:
        name += f'_{unit}'
    return name


def export_patent_csv(patent_ids=None, output_path=None, versions=None):
    if patent_ids is None:
        patent_ids = config.PATENT_IDS
    if output_path is None:
        output_path = config.OUTPUT_DIR / "PatentMoleculeDB_export.csv"
    if versions is None:
        versions = ["decimer_v1", "google_patents_v1", "final_v2", "v13"]

    # First pass: collect all rows + discover assay columns per patent
    raw_rows = []
    assay_columns_ordered = []  # Maintain insertion order
    assay_columns_set = set()

    for pid in patent_ids:
        # Find results file
        data = None
        for ver in versions:
            path = config.OUTPUT_DIR / ver / f"{pid}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                break
        if data is None:
            continue

        # Extract assay data live
        assay_data = extract_assays_for_patent(pid)

        compounds = data.get("exemplified_compounds", [])
        for c in compounds:
            if c.get("processing_status") != "validated":
                continue
            smiles = c.get("canonical_smiles", "")
            if not smiles:
                continue

            ex = c.get("example_number", "") or ""

            # Match assay data
            assays = []
            keys_to_try = [
                ex.lower().replace(' ', ''),
                re.sub(r'[^0-9]', '', ex),
            ]
            for key in keys_to_try:
                if key in assay_data:
                    assays = assay_data[key]
                    break

            # Skip compounds without assay data
            if not assays:
                continue

            # Build assay dict with pivoted columns
            assay_dict = {}
            for a in assays:
                if isinstance(a, dict):
                    name, val, unit, qual = a.get("assay_name", ""), a.get("value_raw", ""), a.get("unit", ""), a.get("qualifier", "") or ""
                    n_runs = a.get("n_runs", "")
                else:
                    name, val, unit, qual = a.assay_name, a.value_raw, a.unit, a.qualifier or ""
                    n_runs = a.n_runs or ""

                col_base = _assay_col_name(name, unit)
                if col_base is None:
                    continue  # Skip bad merged headers
                val = _clean_value(val)

                # Add columns: value, qualifier, n_runs
                assay_dict[col_base] = val
                assay_dict[f"{col_base}_qualifier"] = qual
                assay_dict[f"{col_base}_n_runs"] = str(n_runs) if n_runs else ""

                for col in [col_base, f"{col_base}_qualifier", f"{col_base}_n_runs"]:
                    if col not in assay_columns_set:
                        assay_columns_set.add(col)
                        assay_columns_ordered.append(col)

            # Target from assay names
            target = ""
            for a in assays:
                n = (a.assay_name if hasattr(a, 'assay_name') else a.get("assay_name", "")).lower()
                if "pi3k" in n: target = "PI3K delta"; break
                elif "menin" in n: target = "Menin"; break
                elif "sgk" in n: target = "SGK-1"; break
                elif "antiviral" in n or "a549" in n: target = "Influenza"; break
                elif "fret" in n: target = "Protease"; break

            # MW
            dl = c.get("drug_likeness")
            mw = ""
            if dl and isinstance(dl, dict):
                mw = dl.get("mw", "")

            raw_rows.append({
                "patent_id": pid,
                "compound_id": _strip_compound_prefix(ex),
                "iupac_name": c.get("iupac_name", "") or "",
                "target": target,
                "canonical_smiles": smiles,
                "inchikey": c.get("inchikey", ""),
                "MW": mw,
                **assay_dict,
            })

    # Build final headers: fixed cols + dynamic assay cols
    fixed_cols = ["patent_id", "compound_id", "iupac_name", "target"]
    # Group assay columns: value, qualifier, n_runs together
    trailing_cols = ["canonical_smiles", "inchikey", "MW"]
    headers = fixed_cols + assay_columns_ordered + trailing_cols

    # Sort rows by patent_id, then compound_id (numeric)
    def sort_key(r):
        try:
            num = int(re.sub(r'[^0-9]', '', r.get("compound_id", "0")) or "0")
        except ValueError:
            num = 0
        return (r["patent_id"], num)

    raw_rows.sort(key=sort_key)

    # Write CSV (comma-delimited)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=",", extrasaction="ignore")
        writer.writeheader()
        for row in raw_rows:
            # Fill missing assay columns with empty string
            for col in assay_columns_ordered:
                if col not in row:
                    row[col] = ""
            writer.writerow(row)

    # Print summary
    print(f"Exported {len(raw_rows)} compounds to {output_path}")
    from collections import Counter
    patents = Counter(r["patent_id"] for r in raw_rows)
    for pid, count in patents.most_common():
        assay_cols_with_data = set()
        for r in raw_rows:
            if r["patent_id"] == pid:
                for col in assay_columns_ordered:
                    if not col.endswith("_qualifier") and not col.endswith("_n_runs") and r.get(col):
                        assay_cols_with_data.add(col)
        print(f"  {pid}: {count} compounds, assays: {', '.join(sorted(assay_cols_with_data))}")

    return raw_rows


if __name__ == "__main__":
    export_patent_csv()
