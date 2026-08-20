"""Bulk-load the staging tables over PostgREST.

    python3 -m patentdb3.stage_supabase

Writing needs an INSERT policy, which is opened and closed by two migrations
either side of a load — see `data/supabase_schema.sql`. With the read-only
policies in place this script can only read, which is the resting state.


Runs LOCALLY and streams JSON straight from the artifacts, so none of the
170,000 rows passes through a tool call. The key used is the PUBLISHABLE one —
it is designed to be public — and the write window it needs is opened and
closed by two migrations either side of this script.
"""
import csv
import json
import os
import pathlib
import re
import sys
import urllib.request

URL = os.environ.get("SUPABASE_URL",
                     "https://lucersomrohoehbcemxp.supabase.co") + "/rest/v1"
# The PUBLISHABLE key — designed to be public, and read-only under the row
# level security policies. Override with SUPABASE_URL / SUPABASE_KEY.
KEY = os.environ.get("SUPABASE_KEY",
                     "sb_publishable_aueDcFcErU09Bchmn5O4vQ_tfUuuoBT")
BASE = pathlib.Path("/Users/dhruvsingh/Projects/Patent (1)")

TO_UM = {"nM": 1e-3, "uM": 1.0, "µM": 1.0, "μM": 1.0, "mM": 1e3, "pM": 1e-6,
         "M": 1e6, "mol/L": 1e6, "mol/l": 1e6, "nmol/L": 1e-3,
         "umol/L": 1.0, "µmol/L": 1.0, "μmol/L": 1.0}

_METRIC = [("pIC50", r"\bpIC\s*-?\s*50\b"), ("pKi", r"\bpK\s*-?\s*i\b"),
           ("pKd", r"\bpK\s*-?\s*d\b"), ("IC50", r"\bIC\s*-?\s*50\b"),
           ("EC50", r"\bEC\s*-?\s*50\b"), ("IC90", r"\bIC\s*-?\s*90\b"),
           ("EC90", r"\bEC\s*-?\s*90\b"), ("GI50", r"\bGI\s*-?\s*50\b"),
           ("CC50", r"\bCC\s*-?\s*50\b"), ("ED50", r"\bED\s*-?\s*50\b"),
           ("MIC", r"\bMIC\b"), ("Ki", r"\bK\s*-?\s*i\b"),
           ("Kd", r"\bK\s*-?\s*d\b"), ("Emax", r"\bEmax\b"),
           ("percent", r"%|\bpercent\b|\binhibition\b")]
_UNIT_IN_NAME = re.compile(
    r"\(\s*(?:n|u|µ|μ|m|p)?M\s*\)|\(\s*%\s*\)|\bnM\b|\buM\b|\bµM\b|\bμM\b", re.I)


def axes(name):
    metric = next((c for c, p in _METRIC if re.search(p, name, re.I)), None)
    t = name
    if metric:
        for _c, p in _METRIC:
            t = re.sub(p, " ", t, flags=re.I)
    t = _UNIT_IN_NAME.sub(" ", t)
    t = re.sub(r"[()\[\]]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -_,;:")
    return metric, (t or None)


def num(v):
    if v is None or str(v).strip() == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if f != f else f


def post(table, rows, per=2000):
    sent = 0
    for i in range(0, len(rows), per):
        body = json.dumps(rows[i:i + per]).encode()
        req = urllib.request.Request(
            f"{URL}/{table}", data=body, method="POST",
            headers={"apikey": KEY, "Authorization": f"Bearer {KEY}",
                     "Content-Type": "application/json",
                     "Prefer": "return=minimal,resolution=ignore-duplicates"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
            sent += len(rows[i:i + per])
        except urllib.error.HTTPError as e:
            print(f"  {table} batch {i//per}: HTTP {e.code} {e.read()[:300]!r}")
            return sent
        print(f"\r  {table}: {sent:,}/{len(rows):,}", end="", flush=True)
    print()
    return sent


def main():
    struct = list(csv.DictReader((BASE / "patentdb3/out/structures.tsv").open(),
                                 delimiter="\t"))
    dump = list(csv.DictReader((BASE / "patentdb3/out/reader_dump.tsv").open(),
                               delimiter="\t"))

    best = {}
    for r in struct:
        if not r.get("cid"):
            continue
        k = (r["patent_id"], r["cid"])
        if k not in best or (not best[k].get("inchikey") and r.get("inchikey")):
            best[k] = r
    for r in dump:
        best.setdefault((r["patent_id"], r["cid"]),
                        {"patent_id": r["patent_id"], "cid": r["cid"]})
    mz = {}
    for r in dump:
        v = (r.get("reported_mz") or "").strip()
        if v:
            mz.setdefault((r["patent_id"], r["cid"]), v)

    compounds = [{
        "patent_id": pid, "cid": cid,
        "name": r.get("name") or None, "smiles": r.get("smiles") or None,
        "inchikey": r.get("inchikey") or None, "source": r.get("source") or None,
        "reported_mz": num(mz.get((pid, cid))),
        "mass_check": r.get("mass_check") or None,
        "mass_delta": num(r.get("mass_delta")),
        "markush": r.get("markush") == "True",
        "drawn_only": bool(r.get("drawn_ref")),
    } for (pid, cid), r in best.items()]

    seen, assays = {}, []
    for r in dump:
        k = (r["patent_id"], r["assay_name"], r["table_id"])
        if k in seen:
            continue
        seen[k] = len(seen) + 1
        m, t = axes(r["assay_name"])
        assays.append({"assay_id": seen[k], "patent_id": r["patent_id"],
                       "assay_name": r["assay_name"], "metric": m,
                       "target_raw": t, "unit": r.get("unit") or None,
                       "table_id": r["table_id"]})

    meas = []
    for r in dump:
        f = TO_UM.get((r.get("unit") or "").strip())
        v = num(r.get("value_numeric"))
        lo, hi = num(r.get("range_lo")), num(r.get("range_hi"))
        meas.append({
            "patent_id": r["patent_id"], "cid": r["cid"],
            "assay_id": seen[(r["patent_id"], r["assay_name"], r["table_id"])],
            "value_numeric": v, "qualifier": r.get("qualifier") or None,
            "unit": r.get("unit") or None,
            "value_um": (v * f) if (f and v is not None) else None,
            "letter_grade": r.get("letter_grade") or None,
            "range_lo_um": (lo * f) if (f and lo is not None) else None,
            "range_hi_um": (hi * f) if (f and hi is not None) else None,
            "n_runs": int(num(r.get("n_runs"))) if num(r.get("n_runs")) else None,
            "table_id": r["table_id"],
            "column_header": r.get("column_header") or None})

    print(f"compounds {len(compounds):,}  assays {len(assays):,}  "
          f"measurements {len(meas):,}")
    post("compounds", compounds)
    post("assays", assays)
    post("measurements", meas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
