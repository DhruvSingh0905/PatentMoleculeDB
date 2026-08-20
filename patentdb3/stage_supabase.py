"""Stage the corpus on Supabase as ONE joined table.

    python3 -m patentdb3.stage_supabase

The pipeline's normalised shape — compounds, assays, measurements — is how the
extractor works, not how anyone reads the result. What ships is one row per
compound with its assays nested as JSONB, plus the two things a reader needs to
judge a row before using it:

    route     xml | molscribe | markush | gp — where the structure came from
    status    whether it is finished, and if not, what is holding it up

A drawn compound is not a failure. It is work queued behind image recognition,
and `image_file` carries the drawing so someone can fill it in later without
going back to the XML to find out which picture to read.

Writing needs an INSERT policy, opened and closed by two migrations either side
of a load. The resting state is read-only.
"""
import csv
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

URL = os.environ.get("SUPABASE_URL",
                     "https://lucersomrohoehbcemxp.supabase.co") + "/rest/v1"
KEY = os.environ.get("SUPABASE_KEY",
                     "sb_publishable_aueDcFcErU09Bchmn5O4vQ_tfUuuoBT")
BASE = pathlib.Path(__file__).resolve().parent

TO_UM = {"nM": 1e-3, "uM": 1.0, "µM": 1.0, "μM": 1.0, "mM": 1e3, "pM": 1e-6,
         "M": 1e6, "mol/L": 1e6, "mol/l": 1e6, "nmol/L": 1e-3,
         "umol/L": 1.0, "µmol/L": 1.0, "μmol/L": 1.0}

# The metric a heading names, longest form first so `pIC50` is not read as
# `IC50`. Lifted out of the heading because every patent writes its own and
# 553 distinct ones cannot each be a column.
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
    """(metric, target) for one heading. Neither is invented."""
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


def post(table, rows, per=1000):
    sent = 0
    for i in range(0, len(rows), per):
        body = json.dumps(rows[i:i + per]).encode()
        req = urllib.request.Request(
            f"{URL}/{table}", data=body, method="POST",
            headers={"apikey": KEY, "Authorization": f"Bearer {KEY}",
                     "Content-Type": "application/json",
                     "Prefer": "return=minimal,resolution=ignore-duplicates"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                r.read()
            sent += len(rows[i:i + per])
        except urllib.error.HTTPError as e:
            print(f"\n  {table} batch {i // per}: HTTP {e.code} "
                  f"{e.read()[:300]!r}")
            return sent
        print(f"\r  {table}: {sent:,}/{len(rows):,}", end="", flush=True)
    print()
    return sent


def build():
    struct = list(csv.DictReader((BASE / "out/structures.tsv").open(),
                                 delimiter="\t"))
    dump = list(csv.DictReader((BASE / "out/reader_dump.tsv").open(),
                               delimiter="\t"))

    # one structure row per compound; prefer the one that resolved something
    best = {}
    for r in struct:
        if not r.get("cid"):
            continue
        k = (r["patent_id"], r["cid"])
        if k not in best or (not best[k].get("inchikey") and r.get("inchikey")):
            best[k] = r
    for r in dump:                                  # measured but never named
        best.setdefault((r["patent_id"], r["cid"]),
                        {"patent_id": r["patent_id"], "cid": r["cid"]})

    mz, nested = {}, {}
    for r in dump:
        k = (r["patent_id"], r["cid"])
        v = (r.get("reported_mz") or "").strip()
        if v:
            mz.setdefault(k, v)
        f = TO_UM.get((r.get("unit") or "").strip())
        val = num(r.get("value_numeric"))
        lo, hi = num(r.get("range_lo")), num(r.get("range_hi"))
        metric, target = axes(r["assay_name"])
        nested.setdefault(k, []).append({
            "assay": r["assay_name"], "metric": metric, "target": target,
            "qualifier": r.get("qualifier") or None,
            "value_um": (val * f) if (f and val is not None) else None,
            "value": val, "unit": r.get("unit") or None,
            "grade": r.get("letter_grade") or None,
            "lo_um": (lo * f) if (f and lo is not None) else None,
            "hi_um": (hi * f) if (f and hi is not None) else None,
            "n_runs": int(num(r.get("n_runs"))) if num(r.get("n_runs")) else None,
            "table": r["table_id"]})

    rows = []
    for (pid, cid), r in best.items():
        a = sorted(nested.get((pid, cid), []), key=lambda d: d["assay"] or "")
        ums = [d["value_um"] for d in a if d["value_um"] is not None]
        drawn = (r.get("drawn_ref") or "").strip()
        has_structure = bool(r.get("inchikey"))
        markush = r.get("markush") == "True"
        if has_structure and markush:
            route, status = "markush", "resolved"
        elif has_structure:
            route, status = "xml", "resolved"
        elif markush:
            route, status = "markush", "scaffold only, not assembled"
        elif drawn:
            route, status = "molscribe", "drawn, not yet enumerated"
        else:
            route, status = None, "no structure found"
        rows.append({
            "patent_id": pid, "cid": cid,
            "name": r.get("name") or None, "smiles": r.get("smiles") or None,
            "inchikey": r.get("inchikey") or None,
            "route": route, "status": status,
            "image_file": (r.get("drawn_file") or None) if not has_structure else None,
            "image_ref": (drawn or None) if not has_structure else None,
            "reported_mz": num(mz.get((pid, cid))),
            "mass_check": r.get("mass_check") or None,
            "mass_delta": num(r.get("mass_delta")),
            "n_assays": len({d["assay"] for d in a}),
            "n_measurements": len(a),
            "best_um": min(ums) if ums else None,
            "metrics": sorted({d["metric"] for d in a if d["metric"]}) or None,
            "targets": sorted({d["target"] for d in a if d["target"]}) or None,
            "assays": a})

    # Google Patents structures: no compound number and no assays, so they get
    # a synthetic cid. `route` is what tells them apart, not a separate table.
    gp = BASE / "out/gp_names.tsv"
    if gp.exists():
        seen = set()
        for r in csv.DictReader(gp.open(), delimiter="\t"):
            if r.get("finished") != "True" or not r.get("inchikey"):
                continue
            k = (r["patent_id"], r["inchikey"])
            if k in seen:
                continue
            seen.add(k)
            rows.append({
                "patent_id": r["patent_id"], "cid": f"gp:{r['inchikey']}",
                "name": r.get("gp_name") or None,
                "smiles": r.get("smiles") or None, "inchikey": r["inchikey"],
                "route": "gp", "status": "resolved, no compound number",
                "image_file": None, "image_ref": None,
                "reported_mz": None, "mass_check": None, "mass_delta": None,
                "n_assays": 0, "n_measurements": 0, "best_um": None,
                "metrics": None, "targets": None, "assays": []})
    return rows


def main():
    rows = build()
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"{len(rows):,} compounds")
    for k, v in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"   {v:7,}  {k}")
    post("compounds", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
