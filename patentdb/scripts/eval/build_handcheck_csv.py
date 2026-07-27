"""Build hand-checkable side-by-side CSVs for US10899738, US10214537, and
US8952177 — three axes per row: compound presence, SMILES match, assay
value match. Each row is one (patent, compound, assay) tuple.

For US10899738 + US10214537: comparison is against BindingDB.
For US8952177: comparison is against Jie's hand-curated reference CSV.

Output columns (all CSVs):
    patent_id
    compound_id
    our_iupac_name
    our_smiles
    reference_smiles      (BDB primary OR Jie's IUPAC name)
    smiles_match          (Y / N / no_v2_smiles / no_ref_smiles)
    smiles_inchikey_v2
    smiles_inchikey_ref
    assay_name
    our_value_uM
    our_qualifier
    our_n_runs
    reference_value_uM
    reference_qualifier
    reference_n_runs
    value_match_5pct      (Y / N / null_either)
    relative_diff_pct
    source                (BDB / Jie)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from patentdb.scripts.eval import fidelity_check as fc

try:
    from rdkit import Chem
    RDKIT = True
except ImportError:
    RDKIT = False


def smiles_to_inchikey(s: str | None) -> tuple[str | None, str | None]:
    """Returns (full_inchikey, flat_inchikey) — flat strips stereo."""
    if not s or not RDKIT:
        return None, None
    try:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            return None, None
        full = Chem.MolToInchiKey(mol)
        # Flat = first 14 chars (skeletal, ignores stereo + isotope)
        flat = full.split("-")[0] if full else None
        return full, flat
    except Exception:
        return None, None


# Canonical unit converter (was reimplemented; centralized in core/units.py)
from patentdb.core.units import value_to_uM


def find_assay(arr: list[dict], kind: str) -> dict | None:
    """`kind`: 'ki' / 'ic50' / 'ec50' / etc. Returns first matching."""
    if not arr:
        return None
    for a in arr:
        an = (a.get("assay_name") or "").lower()
        if kind in an:
            return a
        if kind == "ic50" and any(k in an for k in ("ltb", "hwb", "menin", "pi3k", "cd69", "fret", "cc50", "molm", "mv4")):
            return a
        if kind == "ki" and "flap" in an:
            return a
    return None


# ─── BDB-comparison CSV (US10899738, US10214537) ────────────────


def build_bdb_handcheck(patent_id: str, out_path: Path) -> dict:
    bdb = fc.load_bdb_compounds(patent_id)
    _, v2_ay = fc.load_v2_extraction(patent_id)

    # Load v2 IUPAC + SMILES (example_index)
    ex_path = Path(f"output_v2/text_extraction/{patent_id}/example_index.json")
    v2_ex = {}
    if ex_path.exists():
        try:
            v2_ex = json.loads(ex_path.read_text())
        except Exception:
            pass

    rows = []
    n_total_meas = n_smi_match = n_val_match = 0
    for cid, bc in bdb.items():
        if not bc.assays:
            continue
        v2_arr = v2_ay.get(cid, [])
        v2_rec = v2_ex.get(cid) or v2_ex.get(cid.lower())
        our_iupac = (v2_rec.get("iupac_name") if v2_rec else "") or ""
        our_smiles = (v2_rec.get("canonical_smiles") if v2_rec else "") or ""
        our_full_ik, our_flat_ik = smiles_to_inchikey(our_smiles)

        bdb_smiles = bc.smiles or ""
        bdb_full_ik, bdb_flat_ik = smiles_to_inchikey(bdb_smiles)
        # BDB stores multiple variants — match against any
        bdb_iks = bc.flat_inchikey_variants or set()
        if bdb_flat_ik:
            bdb_iks = bdb_iks | {bdb_flat_ik}

        if our_smiles and bdb_iks:
            smiles_match = "Y" if (our_flat_ik and our_flat_ik in bdb_iks) else "N"
        elif not our_smiles:
            smiles_match = "no_v2_smiles"
        else:
            smiles_match = "no_ref_smiles"

        # Build per-assay rows (one row per BDB measurement)
        for bdb_assay in bc.assays:
            assay = bdb_assay.get("assay") or ""
            bdb_val = bdb_assay.get("value")
            bdb_unit = bdb_assay.get("unit") or "nM"
            bdb_qual = bdb_assay.get("qualifier") or ""
            bdb_runs = bdb_assay.get("n_runs")
            bdb_uM = value_to_uM(bdb_val, bdb_unit)

            # Find best-matching v2 measurement
            our_match = None
            our_val_uM = None
            for a in (v2_arr or []):
                v2_val = a.get("value_numeric")
                v2_uM = value_to_uM(v2_val, a.get("unit") or "")
                if v2_uM is None or bdb_uM is None:
                    continue
                if abs(v2_uM - bdb_uM) / max(bdb_uM, 1e-9) < 0.05:
                    our_match = a
                    our_val_uM = v2_uM
                    break

            if our_match and bdb_uM:
                value_match = "Y"
                rel_diff = abs(our_val_uM - bdb_uM) / bdb_uM * 100
            elif bdb_uM is None:
                value_match = "null_ref"
                rel_diff = None
            elif not v2_arr:
                value_match = "no_v2_data"
                rel_diff = None
            else:
                value_match = "N"
                rel_diff = None
                # Pick the closest-magnitude v2 value to show
                if v2_arr:
                    best = min(
                        ((value_to_uM(a.get("value_numeric"), a.get("unit","")), a) for a in v2_arr),
                        key=lambda t: abs((t[0] or 0) - bdb_uM),
                    )
                    our_val_uM = best[0]
                    our_match = best[1]

            n_total_meas += 1
            if smiles_match == "Y": n_smi_match += 1
            if value_match == "Y": n_val_match += 1

            rows.append({
                "patent_id": patent_id,
                "compound_id": cid,
                "our_iupac_name": our_iupac[:200],
                "our_smiles": our_smiles[:120],
                "reference_smiles": bdb_smiles[:120],
                "smiles_match": smiles_match,
                "smiles_inchikey_v2": our_flat_ik or "",
                "smiles_inchikey_ref": bdb_flat_ik or "",
                "assay_name": assay,
                "our_value_uM": our_val_uM if our_val_uM is not None else "",
                "our_qualifier": (our_match.get("qualifier") if our_match else "") or "",
                "our_n_runs": (our_match.get("n_runs") if our_match else "") or "",
                "reference_value_uM": bdb_uM if bdb_uM is not None else "",
                "reference_qualifier": bdb_qual,
                "reference_n_runs": bdb_runs if bdb_runs is not None else "",
                "value_match_5pct": value_match,
                "relative_diff_pct": f"{rel_diff:.2f}" if rel_diff is not None else "",
                "source": "BDB",
            })

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        # Metadata header lines
        f.write(f"# {patent_id} — hand-check CSV (vs BindingDB)\n")
        f.write(f"# {n_total_meas} BDB measurements; "
                f"{n_smi_match} SMILES-match rows ({100*n_smi_match/max(1,n_total_meas):.1f}%); "
                f"{n_val_match} value-match rows ({100*n_val_match/max(1,n_total_meas):.1f}%)\n")
        f.write("# value_match_5pct: Y=value within 5%, N=mismatch, no_v2_data=we missed compound, null_ref=BDB has no numeric\n")
        f.write("# smiles_match: Y=our InChIKey matches a BDB variant (stereo-stripped), N=mismatch, no_v2_smiles=we lack SMILES\n")
        if fieldnames:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    return {"n_meas": n_total_meas, "n_smi_match": n_smi_match, "n_val_match": n_val_match}


# ─── Jie-comparison CSV (US8952177) ─────────────────────────────


def build_jie_handcheck(out_path: Path) -> dict:
    jie_path = Path.home() / "Downloads" / "US8952177 Binding IUPAC Final (1).csv"
    if not jie_path.exists():
        return {"missing": True}

    jie = {}
    with jie_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            cid = row["compound_id"].strip()
            jie[cid] = {
                "iupac": row.get("iupac_name", "").strip(),
                "ki": float(row["flap_binding_ki_uM"]) if row.get("flap_binding_ki_uM") else None,
                "ki_qual": (row.get("flap_binding_ki_qualifier") or "").strip(),
                "ki_runs": int(row["flap_binding_ki_n_runs"]) if row.get("flap_binding_ki_n_runs") else None,
                "ic50": float(row["hwb_ltb4_ic50_uM"]) if row.get("hwb_ltb4_ic50_uM") else None,
                "ic50_qual": (row.get("hwb_ltb4_ic50_qualifier") or "").strip(),
                "ic50_runs": int(row["hwb_ltb4_ic50_n_runs"]) if row.get("hwb_ltb4_ic50_n_runs") else None,
            }

    v2_ay = json.loads(open("output_v2/text_extraction/US8952177/assay_tables.json").read())
    ex_path = Path("output_v2/text_extraction/US8952177/example_index.json")
    v2_ex = json.loads(ex_path.read_text()) if ex_path.exists() else {}

    rows = []
    n_total = n_match = 0
    for cid, ref in jie.items():
        v2_arr = v2_ay.get(cid, []) or v2_ay.get(cid.lower(), [])
        v2_rec = v2_ex.get(cid) or v2_ex.get(cid.lower())
        our_iupac = (v2_rec.get("iupac_name") if v2_rec else "") or ""
        our_smiles = (v2_rec.get("canonical_smiles") if v2_rec else "") or ""

        for assay_kind, ref_v, ref_q, ref_r in [
            ("Ki",   ref["ki"],   ref["ki_qual"],   ref["ki_runs"]),
            ("IC50", ref["ic50"], ref["ic50_qual"], ref["ic50_runs"]),
        ]:
            our_a = find_assay(v2_arr, "ki" if assay_kind == "Ki" else "ic50")
            our_uM = value_to_uM(our_a.get("value_numeric") if our_a else None,
                                 our_a.get("unit") if our_a else "")
            our_q = (our_a.get("qualifier") if our_a else "") or ""
            our_r = (our_a.get("n_runs") if our_a else "") or ""

            if ref_v is not None and our_uM is not None:
                rel_diff = abs(our_uM - ref_v) / max(ref_v, 1e-9) * 100
                value_match = "Y" if rel_diff < 5 else "N"
            elif ref_v is None and our_uM is None:
                value_match = "both_blank"   # patent says nt, both sides blank
                rel_diff = None
            elif ref_v is None:
                value_match = "ref_blank"   # Jie left blank but we have a value
                rel_diff = None
            else:
                value_match = "no_v2_value"   # Jie has it, we don't
                rel_diff = None

            n_total += 1
            if value_match in ("Y", "both_blank"):
                n_match += 1

            rows.append({
                "patent_id": "US8952177",
                "compound_id": cid,
                "our_iupac_name": our_iupac[:200],
                "jie_iupac_name": ref["iupac"][:200],
                "our_smiles": our_smiles[:120],
                "assay_kind": assay_kind,
                "our_value_uM": our_uM if our_uM is not None else "",
                "our_qualifier": our_q,
                "our_n_runs": our_r,
                "jie_value_uM": ref_v if ref_v is not None else "",
                "jie_qualifier": ref_q,
                "jie_n_runs": ref_r if ref_r is not None else "",
                "value_match_5pct": value_match,
                "relative_diff_pct": f"{rel_diff:.2f}" if rel_diff is not None else "",
                "source": "Jie",
            })

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# US8952177 — hand-check CSV (vs Jie's hand-curated reference)\n")
        f.write(f"# {len(jie)} compounds × 2 assays = {n_total} cells; "
                f"{n_match} match (Y or both_blank) ({100*n_match/max(1,n_total):.1f}%)\n")
        f.write("# value_match_5pct: Y=within 5%, N=mismatch, both_blank=patent shows 'nt' both sides, ref_blank=Jie blank but we have value\n")
        if fieldnames:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    return {"n_total": n_total, "n_match": n_match, "match_pct": 100*n_match/max(1,n_total)}


def main() -> None:
    DOWNLOADS = Path.home() / "Downloads"

    print("Building hand-check CSVs...")
    print()

    out1 = DOWNLOADS / "US10899738_handcheck_vs_BDB.csv"
    s1 = build_bdb_handcheck("US10899738", out1)
    print(f"  {out1.name}: {s1['n_meas']} BDB measurements, "
          f"SMILES match {s1['n_smi_match']} ({100*s1['n_smi_match']/max(1,s1['n_meas']):.1f}%), "
          f"value match {s1['n_val_match']} ({100*s1['n_val_match']/max(1,s1['n_meas']):.1f}%)")

    out2 = DOWNLOADS / "US10214537_handcheck_vs_BDB.csv"
    s2 = build_bdb_handcheck("US10214537", out2)
    print(f"  {out2.name}: {s2['n_meas']} BDB measurements, "
          f"SMILES match {s2['n_smi_match']} ({100*s2['n_smi_match']/max(1,s2['n_meas']):.1f}%), "
          f"value match {s2['n_val_match']} ({100*s2['n_val_match']/max(1,s2['n_meas']):.1f}%)")

    out3 = DOWNLOADS / "US8952177_handcheck_vs_Jie.csv"
    s3 = build_jie_handcheck(out3)
    print(f"  {out3.name}: {s3['n_total']} cells (190 cpds × 2 assays), "
          f"match {s3['n_match']} ({s3['match_pct']:.1f}%)")


if __name__ == "__main__":
    main()
