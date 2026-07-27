"""Fidelity check: text_index extraction vs BindingDB ground truth.

For each patent, compares the v2 text_index output (example_index.json
+ assay_tables.json) against BindingDB's known compound + assay records
for that same patent. Reports:

    coverage   = fraction of BDB compounds found in v2 example_index
    smiles_ok  = of those found, fraction with matching InChIKey
    assays_ok  = fraction of BDB assay values matched by any v2 row
                 for the same compound (within 5 % numeric tolerance)

Target: ≥ 95 % on text-primary patents (Class A/B). Lower is acceptable
for image-heavy / Markush patents whose primary route isn't text.

Patent-agnostic — no hardcoded patent IDs or compound IDs.

Usage
-----
    python3 -m patentdb.scripts.eval.fidelity_check \\
        --patent US10899738 [--patent US10214537 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import inchi

from ...core import config

logger = logging.getLogger(__name__)


# ── BDB loader ──────────────────────────────────────────────────────


@dataclass
class BdbCompound:
    """One compound extracted from output/bindingdb/our_patents.tsv.

    BDB sometimes has MULTIPLE SMILES per compound_id (the patent
    discusses several analogs under one Cpd. No., or BDB has both the
    trivial-name row and IUPAC-named rows pointing at structural
    variants). We keep all of them in `smiles_variants` and match v2
    against ANY — it's "found" if v2's InChIKey matches any variant.
    """
    compound_id: str                # patent's compound id (e.g., "350", "5A")
    raw_name: str                   # raw BDB ligand name (first occurrence)
    smiles: str                     # primary SMILES (first row's)
    full_inchikey: str | None       # primary, RDKit-derived
    flat_inchikey: str | None       # primary, stereo-stripped
    flat_inchikey_variants: set     # ALL flat InChIKeys observed for this cpd_id
    assays: list[dict]              # [{assay: 'IC50', value: 24.0, unit: 'nM'}, ...]


# Header indices (BDB TSV)
_BDB_NAME = 5
_BDB_SMILES = 1
_BDB_KI = 8
_BDB_IC50 = 9
_BDB_KD = 10
_BDB_EC50 = 11
_BDB_NAME_RE = re.compile(
    r"(?:Cpd\.?\s*No\.?|Compound|Example|Cpd)\s*([A-Za-z]?\d{1,5}[A-Za-z]{0,3})\b",
    re.IGNORECASE,
)


def _parse_compound_id(raw_name: str) -> str | None:
    """Extract the compound_id token from a BDB ligand name. None on miss."""
    m = _BDB_NAME_RE.search(raw_name or "")
    if m:
        return m.group(1).strip()
    return None


def load_bdb_compounds(patent_id: str) -> dict[str, BdbCompound]:
    """Build {normalized_compound_id: BdbCompound} from BDB TSV.

    Multiple BDB rows for the same compound (different assay readouts)
    are merged into a single BdbCompound with `assays` list aggregated.
    """
    tsv_path = config.REPO_ROOT / "output" / "bindingdb" / "our_patents.tsv"
    if not tsv_path.exists():
        return {}

    out: dict[str, BdbCompound] = {}
    with open(tsv_path) as f:
        rd = csv.reader(f, delimiter="\t")
        for row in rd:
            if len(row) < 12:
                continue
            name = row[_BDB_NAME] or ""
            if patent_id not in name:
                continue
            cid = _parse_compound_id(name)
            if not cid:
                continue
            cid_norm = cid.lower().strip()

            # Per-row InChIKey (so we can collect ALL variants per cpd_id)
            row_smiles = (row[_BDB_SMILES] or "").strip()
            row_flat_ik = None
            row_full_ik = None
            if row_smiles:
                try:
                    mol = Chem.MolFromSmiles(row_smiles)
                    if mol is not None:
                        row_full_ik = inchi.MolToInchiKey(mol)
                        flat = Chem.MolFromSmiles(
                            Chem.MolToSmiles(mol, isomericSmiles=False))
                        if flat:
                            row_flat_ik = inchi.MolToInchiKey(flat)
                except Exception:
                    pass

            # Build / get the existing entry
            entry = out.get(cid_norm)
            if entry is None:
                entry = BdbCompound(
                    compound_id=cid, raw_name=name, smiles=row_smiles,
                    full_inchikey=row_full_ik, flat_inchikey=row_flat_ik,
                    flat_inchikey_variants=set(), assays=[],
                )
                out[cid_norm] = entry
            if row_flat_ik:
                entry.flat_inchikey_variants.add(row_flat_ik)

            # Append assay readouts present on this row
            for col_idx, assay_name in [
                (_BDB_KI, "Ki"), (_BDB_IC50, "IC50"),
                (_BDB_KD, "Kd"), (_BDB_EC50, "EC50"),
            ]:
                v_raw = (row[col_idx] or "").strip() if col_idx < len(row) else ""
                if not v_raw:
                    continue
                # Strip qualifiers
                qual = None
                if v_raw[0] in "<>":
                    qual = v_raw[0]
                    v_raw = v_raw[1:].strip()
                try:
                    v_num = float(v_raw)
                except ValueError:
                    continue
                entry.assays.append({
                    "assay": assay_name, "value": v_num,
                    "unit": "nM", "qualifier": qual,
                })
    return out


# ── v2 loader ───────────────────────────────────────────────────────


def _norm_key(k: str) -> str:
    """Same normalization used by text_index: lowercase, strip non-alphanumeric."""
    s = k.strip().lower()
    s = re.sub(r"cpd\.?\s*no\.?", "", s)
    s = re.sub(r"compound|example", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def load_v2_extraction(patent_id: str) -> tuple[dict, dict]:
    """Return (example_index_by_norm_key, assay_tables_by_norm_key)."""
    base = config.OUTPUT_DIR / "text_extraction" / patent_id
    ex_path = base / "example_index.json"
    as_path = base / "assay_tables.json"
    if not ex_path.exists() or not as_path.exists():
        return {}, {}
    ex = json.loads(ex_path.read_text())
    ay = json.loads(as_path.read_text())
    ex_norm = {_norm_key(k): v for k, v in ex.items()}
    ay_norm = {_norm_key(k): v for k, v in ay.items()}
    return ex_norm, ay_norm


# ── Comparison helpers ──────────────────────────────────────────────


def _values_match(v_bdb: float, unit_bdb: str, v_v2: float, unit_v2: str,
                  tol_pct: float = 5.0) -> bool:
    """Compare two assay measurements with unit normalization (everything → nM)."""
    def to_nM(v: float, u: str) -> float:
        u = (u or "").lower()
        if u in ("μm", "um", "µm"): return v * 1000.0
        if u in ("mm",): return v * 1_000_000.0
        if u in ("nm",): return v
        if u in ("pm",): return v / 1000.0
        return v   # unknown unit — compare as-is
    a, b = to_nM(v_bdb, unit_bdb), to_nM(v_v2, unit_v2)
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    rel = abs(a - b) / max(abs(a), abs(b))
    return rel <= (tol_pct / 100.0)


@dataclass
class FidelityScore:
    patent_id: str
    bdb_total: int
    found_in_v2: int                  # compounds with norm_key match
    smiles_match: int                 # subset where SMILES InChIKey matches
    assays_total_bdb: int             # total BDB assay readouts
    assays_matched_v2: int            # how many BDB assays we found in v2
    coverage_pct: float
    smiles_pct: float
    assay_pct: float
    sample_misses: list


def compute_fidelity(patent_id: str) -> FidelityScore:
    bdb = load_bdb_compounds(patent_id)
    v2_ex, v2_ay = load_v2_extraction(patent_id)

    if not bdb:
        return FidelityScore(patent_id, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, [])

    found = 0
    found_with_smiles = 0   # subset of `found` where v2 has a non-stub entry
    structure_only = 0      # subset of `found` flagged is_structure_only
    smiles_match = 0
    assays_total = 0
    assays_matched = 0
    misses = []

    for cid_norm, bc in bdb.items():
        v2_rec = v2_ex.get(cid_norm)
        if v2_rec is None:
            if len(misses) < 5:
                misses.append(("not_in_v2", bc.compound_id, None))
            continue
        found += 1
        if v2_rec.get("is_structure_only"):
            structure_only += 1
        else:
            found_with_smiles += 1

        # SMILES match via flat InChIKey (stereo-tolerant). The v2
        # example_index may have `inchikey=None` even when SMILES is
        # populated (extract_table_compounds_from_gp doesn't always run
        # the full IUPAC-to-SMILES cascade that computes the InChIKey).
        # Compute on-the-fly from canonical_smiles when missing.
        v2_flat_prefix = ""
        v2_ik = v2_rec.get("inchikey")
        if v2_ik:
            v2_flat_prefix = v2_ik.split("-")[0]
        elif v2_rec.get("canonical_smiles"):
            try:
                mol = Chem.MolFromSmiles(v2_rec["canonical_smiles"])
                if mol is not None:
                    flat = Chem.MolFromSmiles(
                        Chem.MolToSmiles(mol, isomericSmiles=False))
                    if flat is not None:
                        v2_flat_prefix = inchi.MolToInchiKey(flat).split("-")[0]
            except Exception:
                pass
        # Match against ANY of the BDB SMILES variants for this cpd_id
        # (BDB sometimes lists multiple analogs under one compound_id).
        bdb_flat_prefixes = {ik.split("-")[0] for ik in bc.flat_inchikey_variants}
        if v2_flat_prefix and v2_flat_prefix in bdb_flat_prefixes:
            smiles_match += 1
        elif v2_flat_prefix and bdb_flat_prefixes:
            if len(misses) < 5:
                misses.append(("smiles_mismatch", bc.compound_id,
                               f"v2={v2_flat_prefix} bdb={sorted(bdb_flat_prefixes)[:3]}"))

        # Assay match
        v2_assays = v2_ay.get(cid_norm, [])
        for bdb_a in bc.assays:
            assays_total += 1
            matched = False
            for v2_a in v2_assays:
                v2_val = v2_a.get("value_numeric")
                if v2_val is None:
                    continue
                if _values_match(bdb_a["value"], bdb_a["unit"],
                                 float(v2_val), v2_a.get("unit") or ""):
                    matched = True
                    break
            if matched:
                assays_matched += 1

    return FidelityScore(
        patent_id=patent_id,
        bdb_total=len(bdb),
        found_in_v2=found,
        smiles_match=smiles_match,
        assays_total_bdb=assays_total,
        assays_matched_v2=assays_matched,
        coverage_pct=100.0 * found / max(1, len(bdb)),
        # SMILES match denominator excludes structure-only stubs — they're
        # known text-route gaps where IUPAC isn't extractable, not parser
        # failures. Counting them would unfairly penalize the SMILES metric.
        smiles_pct=100.0 * smiles_match / max(1, found_with_smiles),
        assay_pct=100.0 * assays_matched / max(1, assays_total),
        sample_misses=misses,
    )


def _classify_patent(fs: FidelityScore) -> str:
    """Heuristic class label based on extraction profile."""
    if fs.bdb_total == 0:
        return "no-BDB-data"
    if fs.coverage_pct >= 80 and fs.smiles_pct >= 80:
        return "TEXT-PRIMARY ✓"
    if fs.coverage_pct < 10:
        return "image-primary or Markush (text route insufficient)"
    if fs.smiles_pct < 50:
        return "text route partial — name extraction needs work"
    return "mixed"


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", action="append", required=True)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    print(f"{'patent':>16}  {'BDB':>4}  {'cov':>5}  {'smi':>5}  {'asy':>5}  "
          f"{'cov%':>5}  {'smi%':>5}  {'asy%':>5}  class")
    print("-" * 100)
    fs_all: list[FidelityScore] = []
    for pat in args.patent:
        fs = compute_fidelity(pat)
        fs_all.append(fs)
        cls = _classify_patent(fs)
        print(f"{pat:>16}  {fs.bdb_total:>4}  {fs.found_in_v2:>5}  "
              f"{fs.smiles_match:>5}  {fs.assays_matched_v2:>5}  "
              f"{fs.coverage_pct:>4.0f}%  {fs.smiles_pct:>4.0f}%  "
              f"{fs.assay_pct:>4.0f}%  {cls}")
    print()
    # Per-patent miss samples
    for fs in fs_all:
        if fs.sample_misses and fs.bdb_total > 0:
            print(f"--- {fs.patent_id} sample misses ---")
            for kind, cid, info in fs.sample_misses:
                print(f"   {kind}: cpd={cid} {info or ''}")
    print()
    # Summary: text-primary patents at 95%+
    text_primary = [fs for fs in fs_all
                    if fs.coverage_pct >= 80 and fs.bdb_total > 0]
    if text_primary:
        print("=== Text-primary 95% targets ===")
        for fs in text_primary:
            cov_ok = "✓" if fs.coverage_pct >= 95 else "✗"
            smi_ok = "✓" if fs.smiles_pct >= 95 else "✗"
            asy_ok = "✓" if fs.assay_pct >= 95 else "✗"
            print(f"  {fs.patent_id:>16}  cov={fs.coverage_pct:.1f}% {cov_ok}  "
                  f"smi={fs.smiles_pct:.1f}% {smi_ok}  asy={fs.assay_pct:.1f}% {asy_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
