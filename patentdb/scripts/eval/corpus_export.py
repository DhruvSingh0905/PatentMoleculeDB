"""Dump the corpus to CSV and compute the coverage metrics from those files.

    python3 -m patentdb.scripts.eval.corpus_export
    python3 -m patentdb.scripts.eval.corpus_export --patents US8952177,US9302989
    python3 -m patentdb.scripts.eval.corpus_export --out docs/reports

Three files, because three different questions are being asked and mixing them
is how a coverage number stops meaning anything:

  records.csv    one row per assay measurement — the raw extraction
  compounds.csv  one row per (patent, compound) — what we know about each
  patents.csv    one row per patent — the per-document rollup

Assay records come from BOTH reading tiers, union by compound id, for the
reason `pipeline_bench` exists: `extract_from_patent` never calls `apply_rule`,
so counting it alone omits every layout the 84 learned rules were bought for.

STRUCTURES ARE A SEPARATE PASS AND MOST PATENTS HAVE NOT HAD IT. `example_index
.json` — IUPAC, SMILES, InChIKey — is written by `process_patent`, and only 22
of 103 patents in this corpus have one. BDB coverage is an InChIKey comparison,
so it is computable for those 22 and NOT for the other 81. Those patents are
reported as `structures=no` and excluded from the coverage denominator rather
than counted as misses, because "we never tried to resolve it" and "we tried
and failed" are different facts and averaging them together would understate
the resolver and overstate the corpus in one step.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _pipeline_records(pid: str, xml: str) -> list:
    """Every assay record for this patent, both tiers, de-duplicated."""
    os.environ.setdefault("REPAIR", "0")
    from ...repair.loop import repair_patent
    from ...sources.uspto_assays import extract_from_patent

    out, seen = [], set()
    for tier, fn in (("raw", lambda: extract_from_patent(xml)),
                     ("rules", lambda: repair_patent(pid, xml)[0])):
        try:
            recs = list(fn())
        except Exception as e:
            logger.warning("export: %s %s raised %r", pid, tier, e)
            continue
        for r in recs:
            if not getattr(r, "is_usable", False):
                continue
            key = (r.cid, r.assay_name, r.value_numeric, r.unit, r.table_id)
            if key in seen:
                continue
            seen.add(key)
            out.append((tier, r))
    return out


def _structures(pid: str) -> dict[str, dict]:
    """`example_index.json` for this patent, or {} when it was never resolved."""
    from ...core import config

    p = config.OUTPUT_DIR / "text_extraction" / pid / "example_index.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _norm(cid: str) -> str:
    return str(cid or "").strip().lstrip("0").upper() or str(cid or "").strip().upper()


def export(pids: list[str] | None, out_dir) -> dict:
    from ...core import config

    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    every = sorted(p.stem for p in xml_dir.glob("*.xml"))
    pids = [p for p in (pids or every) if (xml_dir / f"{p}.xml").exists()]

    try:
        from .reference_bench import load_bindingdb
        bdb_by_patent, _ = load_bindingdb()
    except Exception as e:
        logger.warning("export: BindingDB unavailable (%r); coverage omitted", e)
        bdb_by_patent = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    rec_f = open(out_dir / "records.csv", "w", newline="")
    cmp_f = open(out_dir / "compounds.csv", "w", newline="")
    pat_f = open(out_dir / "patents.csv", "w", newline="")
    rw = csv.writer(rec_f); cw = csv.writer(cmp_f); pw = csv.writer(pat_f)
    rw.writerow(["patent", "tier", "compound_id", "assay_name", "value_numeric",
                 "unit", "qualifier", "n_runs", "range_lo", "range_hi",
                 "letter_grade", "table_id", "column_header", "value_text"])
    cw.writerow(["patent", "compound_id", "n_records", "n_distinct_assays",
                 "has_n_runs", "has_iupac", "has_smiles", "has_inchikey",
                 "inchikey", "in_bindingdb"])
    pw.writerow(["patent", "n_compounds", "n_records", "n_with_n_runs",
                 "pct_records_with_n_runs", "n_distinct_assays", "structures",
                 "n_structures_resolved", "bdb_ligands", "bdb_matched",
                 "bdb_coverage_pct"])

    totals = {"patents": 0, "records": 0, "compounds": 0, "n_runs": 0,
              "zero_patents": [], "struct_patents": 0, "bdb_ligands": 0,
              "bdb_matched": 0, "compounds_with_ik": 0}
    for pid in pids:
        xml = (xml_dir / f"{pid}.xml").read_text(errors="ignore")
        rows = _pipeline_records(pid, xml)
        struct = _structures(pid)
        by_cid: dict[str, list] = {}
        for tier, r in rows:
            by_cid.setdefault(str(r.cid), []).append(r)
            rw.writerow([pid, tier, r.cid, r.assay_name, r.value_numeric, r.unit,
                         getattr(r, "qualifier", ""), getattr(r, "n_runs", ""),
                         getattr(r, "range_lo", ""), getattr(r, "range_hi", ""),
                         getattr(r, "letter_grade", ""), r.table_id,
                         getattr(r, "column_header", ""),
                         getattr(r, "value_text", "")])

        # Structures are keyed by the patent's own compound id, which the assay
        # side may write differently ("07" vs "7"), so match on the normalised
        # form as well as verbatim.
        s_by_norm = {_norm(k): v for k, v in struct.items()}
        bdb_iks = set(bdb_by_patent.get(pid, set()))
        ours_iks, matched = set(), set()
        n_runs_rows = 0
        for cid, rs in sorted(by_cid.items()):
            info = struct.get(cid) or s_by_norm.get(_norm(cid)) or {}
            ik = (info.get("inchikey") or "").strip()
            has_n = any(getattr(r, "n_runs", None) for r in rs)
            n_runs_rows += sum(1 for r in rs if getattr(r, "n_runs", None))
            in_bdb = ""
            if ik:
                ours_iks.add(ik)
                if bdb_iks:
                    in_bdb = "yes" if ik in bdb_iks else "no"
                    if ik in bdb_iks:
                        matched.add(ik)
            cw.writerow([pid, cid, len(rs), len({r.assay_name for r in rs}),
                         "yes" if has_n else "no",
                         "yes" if info.get("iupac_name") else "no",
                         "yes" if info.get("canonical_smiles") else "no",
                         "yes" if ik else "no", ik, in_bdb])

        n_rec = len(rows)
        cov = round(100.0 * len(matched) / len(bdb_iks), 1) if bdb_iks else ""
        pw.writerow([pid, len(by_cid), n_rec, n_runs_rows,
                     round(100.0 * n_runs_rows / n_rec, 1) if n_rec else 0.0,
                     len({r.assay_name for _t, r in rows}),
                     "yes" if struct else "no", len(struct),
                     len(bdb_iks), len(matched), cov])
        totals["patents"] += 1
        totals["records"] += n_rec
        totals["compounds"] += len(by_cid)
        totals["n_runs"] += n_runs_rows
        totals["compounds_with_ik"] += len(ours_iks)
        if not by_cid:
            totals["zero_patents"].append(pid)
        if struct:
            totals["struct_patents"] += 1
            totals["bdb_ligands"] += len(bdb_iks)
            totals["bdb_matched"] += len(matched)
    for f in (rec_f, cmp_f, pat_f):
        f.close()
    return totals


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", help="comma-separated; default = every cached XML")
    ap.add_argument("--out", default="docs/reports/corpus", help="output directory")
    a = ap.parse_args()
    from ...core import config

    pids = [p.strip().upper() for p in a.patents.split(",")] if a.patents else None
    out_dir = config.REPO_ROOT / a.out
    t = export(pids, out_dir)

    n_p, n_r = t["patents"], t["records"]
    print(f"\npatents                    : {n_p}")
    print(f"assay records              : {n_r}")
    print(f"distinct compounds         : {t['compounds']}")
    print(f"patents yielding nothing   : {len(t['zero_patents'])} {t['zero_patents']}")
    print(f"records carrying n_runs    : {t['n_runs']} "
          f"({100.0 * t['n_runs'] / n_r:.1f}%)" if n_r else "")
    print(f"patents with structures    : {t['struct_patents']} of {n_p} "
          f"(the rest never had the resolver run — NOT failures)")
    print(f"compounds with an InChIKey : {t['compounds_with_ik']}")
    if t["bdb_ligands"]:
        print(f"BDB coverage (those {t['struct_patents']:>2} only) : "
              f"{t['bdb_matched']}/{t['bdb_ligands']} "
              f"({100.0 * t['bdb_matched'] / t['bdb_ligands']:.1f}%)")
    print(f"\nwrote {out_dir}/records.csv, compounds.csv, patents.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
