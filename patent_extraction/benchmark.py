"""BindingDB benchmark — compare extracted compounds against ground truth.

Matches by InChIKey (full for stereo-aware, 14-char for connectivity-only).
Reports precision, recall, and logs unmatched compounds for investigation.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd

from . import config
from .smiles_utils import canonicalize_smiles, get_inchikey, get_connectivity_key

logger = logging.getLogger(__name__)

BINDINGDB_DIR = config.OUTPUT_DIR / "bindingdb"
BINDINGDB_ZIP = BINDINGDB_DIR / "BindingDB_Patents.tsv.zip"


def load_bindingdb(tsv_path: Path | None = None) -> pd.DataFrame:
    """Load BindingDB patents TSV.

    Args:
        tsv_path: Path to TSV file or zip. Auto-detects zip.

    Returns:
        DataFrame with BindingDB data.
    """
    if tsv_path is None:
        tsv_path = BINDINGDB_ZIP

    if not tsv_path.exists():
        raise FileNotFoundError(f"BindingDB file not found: {tsv_path}")

    # Handle zip
    if str(tsv_path).endswith('.zip'):
        with zipfile.ZipFile(tsv_path) as zf:
            tsv_names = [n for n in zf.namelist() if n.endswith('.tsv')]
            if not tsv_names:
                raise ValueError(f"No TSV file found in {tsv_path}")
            with zf.open(tsv_names[0]) as f:
                df = pd.read_csv(f, sep='\t', low_memory=False, on_bad_lines='skip')
    else:
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False, on_bad_lines='skip')

    logger.info(f"Loaded BindingDB: {len(df)} rows, {len(df.columns)} columns")
    return df


def find_patent_column(df: pd.DataFrame) -> str | None:
    """Find the column containing patent numbers."""
    candidates = [c for c in df.columns if 'patent' in c.lower()]
    if candidates:
        return candidates[0]
    # Also check for columns containing US patent-like values
    for col in df.columns:
        sample = df[col].dropna().head(100).astype(str)
        if sample.str.contains('US\\d{7,}', regex=True).any():
            return col
    return None


def filter_by_patents(df: pd.DataFrame, patent_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Filter BindingDB to rows matching our patent IDs.

    Tries multiple format variants for each patent ID.

    Returns:
        Dict of patent_id -> filtered DataFrame.
    """
    patent_col = find_patent_column(df)
    if patent_col is None:
        logger.error(f"No patent column found. Columns: {list(df.columns)[:20]}")
        return {}

    logger.info(f"Using patent column: '{patent_col}'")

    results = {}
    patent_values = df[patent_col].astype(str)

    for pid in patent_ids:
        # Try format variants
        # Strip A1/B1/B2 suffix for application patents
        pid_base = re.sub(r'[AB]\d+$', '', pid)
        variants = [
            pid,                              # US20240335431A1
            pid_base,                         # US20240335431
            pid.replace('US', ''),            # 20240335431A1
            pid_base.replace('US', ''),       # 20240335431
            f"US {pid[2:]}" if pid.startswith('US') else pid,  # US 20240335431A1
            f"US {pid_base[2:]}" if pid_base.startswith('US') else pid_base,
        ]
        # Also try with commas: US 10,214,537
        if pid.startswith('US') and len(pid) > 5:
            num = pid_base[2:] if pid_base.startswith('US') else pid_base
            if len(num) >= 7:
                formatted = f"{num[:-6]},{num[-6:-3]},{num[-3:]}"
                variants.append(f"US {formatted}")
                variants.append(f"US{formatted}")

        mask = pd.Series(False, index=df.index)
        for variant in variants:
            mask |= patent_values.str.contains(variant, na=False, regex=False)

        matched = df[mask]
        if len(matched) > 0:
            results[pid] = matched
            logger.info(f"  {pid}: {len(matched)} BindingDB entries")
        else:
            logger.info(f"  {pid}: no BindingDB entries found")

    return results


def extract_bindingdb_inchikeys(df: pd.DataFrame) -> set[str]:
    """Extract InChIKeys from BindingDB rows.

    Tries the InChIKey column first, falls back to generating from SMILES.
    """
    keys = set()

    # Try InChIKey column
    inchikey_cols = [c for c in df.columns if 'inchi' in c.lower() and 'key' in c.lower()]
    if inchikey_cols:
        for val in df[inchikey_cols[0]].dropna():
            val = str(val).strip()
            if len(val) >= 14:
                keys.add(val)

    # Also generate from SMILES for completeness
    smiles_cols = [c for c in df.columns if 'smiles' in c.lower() and 'ligand' in c.lower()]
    if not smiles_cols:
        smiles_cols = [c for c in df.columns if 'smiles' in c.lower()]

    if smiles_cols:
        for val in df[smiles_cols[0]].dropna():
            smiles = str(val).strip()
            canonical = canonicalize_smiles(smiles)
            if canonical:
                key = get_inchikey(canonical)
                if key:
                    keys.add(key)

    return keys


def benchmark_patent(
    patent_id: str,
    our_results_path: Path,
    bindingdb_df: pd.DataFrame,
) -> dict:
    """Benchmark one patent's extraction against BindingDB.

    Args:
        patent_id: Patent identifier.
        our_results_path: Path to our extraction JSON.
        bindingdb_df: BindingDB rows for this patent.

    Returns:
        Dict with precision, recall, matched/unmatched counts.
    """
    # Load our results
    with open(our_results_path) as f:
        our_data = json.load(f)

    our_compounds = our_data['exemplified_compounds']
    our_validated = [c for c in our_compounds if c['processing_status'] == 'validated']

    # Generate our InChIKeys
    our_keys = {}
    for c in our_validated:
        key = c.get('inchikey')
        if key:
            our_keys[key] = c
        # Also use parent InChIKey for salt-form matching
        parent_key = c.get('parent_inchikey')
        if parent_key and parent_key != key:
            our_keys[parent_key] = c

    # Get BindingDB InChIKeys
    bdb_keys = extract_bindingdb_inchikeys(bindingdb_df)

    # Full InChIKey matching
    our_key_set = set(our_keys.keys())
    matched_full = our_key_set & bdb_keys

    # Stereo-flattened matching (PRIMARY METRIC — accurate molecular recall)
    from .smiles_utils import get_stereo_flattened_key
    smiles_col = [c for c in bindingdb_df.columns if 'smiles' in c.lower() and 'ligand' in c.lower()]
    smiles_col = smiles_col[0] if smiles_col else None

    our_flat = set()
    for c in our_validated:
        smi = c.get('canonical_smiles', '')
        if smi:
            fk = get_stereo_flattened_key(smi)
            if fk:
                our_flat.add(fk)

    bdb_flat = set()
    if smiles_col:
        for val in bindingdb_df[smiles_col].dropna():
            smi = str(val).strip()
            fk = get_stereo_flattened_key(smi)
            if fk:
                bdb_flat.add(fk)
    else:
        bdb_flat = bdb_keys  # Fallback to InChIKey if no SMILES column

    matched_flat = our_flat & bdb_flat

    # Connectivity-only matching (14 chars)
    our_conn = {get_connectivity_key(k): k for k in our_key_set if k}
    bdb_conn = {get_connectivity_key(k): k for k in bdb_keys if k}
    matched_conn = set(our_conn.keys()) & set(bdb_conn.keys())

    # Metrics
    our_total = len(set(c.get('inchikey') for c in our_validated if c.get('inchikey')))
    bdb_total = len(bdb_keys)
    bdb_flat_total = len(bdb_flat) if bdb_flat else bdb_total

    precision_full = len(matched_full) / max(our_total, 1)
    recall_full = len(matched_full) / max(bdb_total, 1)

    recall_flattened = len(matched_flat) / max(bdb_flat_total, 1)

    precision_conn = len(matched_conn) / max(len(our_conn), 1)
    recall_conn = len(matched_conn) / max(len(bdb_conn), 1)

    # Unmatched on our side (possible extraction errors)
    our_unmatched = our_key_set - bdb_keys
    # Unmatched on BindingDB side (compounds we missed)
    bdb_unmatched = bdb_keys - our_key_set

    report = {
        'patent_id': patent_id,
        'our_validated': our_total,
        'bindingdb_compounds': bdb_total,
        'matched_full_inchikey': len(matched_full),
        'matched_flattened': len(matched_flat),
        'matched_connectivity': len(matched_conn),
        'precision_full': round(precision_full, 3),
        'recall_full': round(recall_full, 3),
        'recall_flattened': round(recall_flattened, 3),
        'precision_connectivity': round(precision_conn, 3),
        'recall_connectivity': round(recall_conn, 3),
        'our_unmatched': len(our_unmatched),
        'bdb_unmatched': len(bdb_unmatched),
    }

    return report


def run_benchmark(
    patent_ids: list[str] | None = None,
    results_version: str = "v7",
) -> list[dict]:
    """Run full BindingDB benchmark across patents.

    Args:
        patent_ids: Patents to benchmark. Defaults to all.
        results_version: Which output version to benchmark.

    Returns:
        List of per-patent benchmark reports.
    """
    if patent_ids is None:
        patent_ids = config.PATENT_IDS

    # Load BindingDB
    df = load_bindingdb()

    # Check columns
    logger.info(f"BindingDB columns: {list(df.columns)[:10]}...")

    # Filter to our patents
    patent_data = filter_by_patents(df, patent_ids)

    reports = []
    for pid in patent_ids:
        results_path = config.OUTPUT_DIR / results_version / f"{pid}.json"
        if not results_path.exists():
            logger.info(f"  {pid}: No extraction results found at {results_path}")
            continue

        if pid not in patent_data:
            logger.info(f"  {pid}: No BindingDB data — skipping benchmark")
            reports.append({
                'patent_id': pid,
                'bindingdb_compounds': 0,
                'note': 'No BindingDB coverage (patent too recent or not curated)',
            })
            continue

        report = benchmark_patent(pid, results_path, patent_data[pid])
        reports.append(report)

        print(f"\n{pid}:")
        print(f"  Our compounds: {report['our_validated']}")
        print(f"  BindingDB compounds: {report['bindingdb_compounds']}")
        print(f"  Matched (full InChIKey): {report['matched_full_inchikey']}")
        print(f"  Matched (connectivity): {report['matched_connectivity']}")
        print(f"  Precision (full): {report['precision_full']:.1%}")
        print(f"  Recall (full): {report['recall_full']:.1%}")
        print(f"  Precision (connectivity): {report['precision_connectivity']:.1%}")
        print(f"  Recall (connectivity): {report['recall_connectivity']:.1%}")

    return reports
