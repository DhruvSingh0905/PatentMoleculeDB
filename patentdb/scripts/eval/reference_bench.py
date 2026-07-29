"""Benchmark extraction against BindingDB and ChEMBL.

Two questions, deliberately kept apart because they fail differently:

  COVERAGE   — of the compounds a reference database records for this patent,
               how many did we extract at all? Misses are recall failures.
  CORRECTNESS— where the reference and we both name the same compound (same
               `Example N`), do the structures agree? Mismatches are
               attribution or chemistry failures, and they are far more
               damaging than a miss: a missing row is absent, a wrong row is
               a lie the database can't distinguish from truth.

Reporting one blended "accuracy" number hides exactly the failure this
codebase has: 97.6% structure coverage alongside a route that is correct
0.2% of the time.

References
----------
BindingDB  — `output/bindingdb/our_patents.tsv` (or the full dump). ~43% of its
             rows carry a patent number and it is the largest open source of
             patent-derived bioactivity. Semi-automated curation, so treat
             disagreement as a flag, not proof we are wrong.
ChEMBL     — 7,145 PATENT-type documents via the public API, cached locally.
             High precision, low recall: it is downstream of SureChEMBL and
             BindingDB and curates a deliberately narrow slice, so use it to
             confirm true positives, never to measure recall.

Neither is an audited gold standard. Both are the best available.

Usage:
    python3 -m patentdb.scripts.eval.reference_bench
    python3 -m patentdb.scripts.eval.reference_bench --patents US8952177,US9718825
    python3 -m patentdb.scripts.eval.reference_bench --refs bindingdb --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from patentdb.core import config

csv.field_size_limit(10 ** 9)

BDB_TSV = config.REPO_ROOT / "output" / "bindingdb" / "our_patents.tsv"
CHEMBL_CACHE = config.OUTPUT_DIR / "_cache" / "chembl_patents.json"
EXTRACTIONS = config.OUTPUT_DIR / "text_extraction"

# How BindingDB attributes a ligand to a patent compound. The word "Example"
# is OPTIONAL, because it is often simply absent:
#
#   US8952177, Example 1
#   US8722692, 1                              <- no keyword at all
#   US9303033, N47, Table 58A, Compound 11    <- the patent's code, then BDB's
#
# Requiring it scored three whole patents at zero that BindingDB holds in full:
# US9303033 has 2,239 rows there, US8722692 732, US9708336 837. Across the 53
# patents with no structures of our own, reading only the strict form found
# 10,147 of 16,222 compounds; reading this form finds 13,453.
#
# The id is ALWAYS the token straight after the patent number, and never the
# trailing "Compound N" — that is BindingDB's own within-table numbering.
# Measured on US9303033: the first token hits our extracted ids 2,491 times and
# misses 12; the trailing one hits ZERO times and misses 2,482. Taking the
# wrong one would have attached 1,237 structures to the wrong compounds.
_REF_STOP = (r"(?!(?:table|compound|scheme|fig(?:ure)?|claim|page|para|"
             r"col(?:umn)?|entry|item|no)\b)")
_EXAMPLE_REF = re.compile(
    r"\b(US\d{7,11}[A-Z]?\d?)\s*,\s*(?:Example\s+)?" + _REF_STOP
    + r"([0-9A-Za-z][0-9A-Za-z\-]{0,9})\b", re.I)


def _norm_cid(cid: str) -> str:
    """'007' / 'Example 7' / 'Cpd. No. 7' → '7'. One canonical form.

    Delegates to the extractor's own `normalize_cid`, because "one canonical
    form" has to mean ONE. This function used to re-implement it with
    `s.lstrip("0")`, which only strips zeros at position 0 — so BindingDB's
    `I-0117` stayed `I-0117` while the patent's `I-117` normalised to `I-117`
    and the two never met. On US9718790 that was 1,119 compounds scored as
    missing that we had extracted correctly all along: a benchmark measuring
    its own normaliser rather than the extraction.
    """
    from patentdb.sources.uspto_assays import normalize_cid
    return normalize_cid(str(cid)).upper()


def _skeleton(ik: str) -> str:
    """First InChIKey block — connectivity only, stereo-insensitive."""
    return ik.split("-")[0] if ik else ""


def patent_owns_cid(patent_id: str, cid: str) -> bool | None:
    """Does this patent actually contain the identifier a reference cites?

    BindingDB records a compound's provenance as free text in the ligand name
    ("US9303033, Example 275"), and those citations are not always right. On
    US9303033 four cited Example numbers appear nowhere in the document — the
    patent numbers its compounds `A1`, `A2`, … and its description's Example
    numbers never reach 275. Scored naively that reads as a 0/4 extraction
    failure, and we spent real time chasing it.

    A reference point is only usable as ground truth if the identifier it
    names exists in the source. Returns None when the patent's text cannot be
    fetched, so an unverifiable citation is excluded rather than assumed good.
    """
    from patentdb.sources import uspto_assays, uspto_xml
    try:
        xml = uspto_xml.fetch_grant_xml(patent_id)
    except Exception:
        return None
    want = uspto_assays.normalize_cid(cid).upper()
    if not want:
        return False
    for t in uspto_xml.parse_tables(xml):
        for row in t.body_rows:
            for c in row:
                if uspto_assays.normalize_cid(c.text.strip()).upper() == want:
                    return True
    desc = uspto_xml.description_text(xml)
    return bool(re.search(rf"\bExamples?\s+0*{re.escape(want)}\b", desc, re.I))


# ── references ────────────────────────────────────────────────────

def load_bindingdb() -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """Return (patent → all InChIKeys, patent → {example → InChIKeys})."""
    by_patent: dict[str, set[str]] = defaultdict(set)
    by_example: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if not BDB_TSV.exists():
        print(f"  ! BindingDB subset not found at {BDB_TSV}", file=sys.stderr)
        return by_patent, by_example
    with open(BDB_TSV, newline="", encoding="utf-8", errors="ignore") as fh:
        rdr = csv.reader(fh, delimiter="\t")
        hdr = next(rdr)
        try:
            ik_i = hdr.index("Ligand InChI Key")
            nm_i = hdr.index("BindingDB Ligand Name")
        except ValueError:
            return by_patent, by_example
        pat_i = hdr.index("Patent Number") if "Patent Number" in hdr else None
        for row in rdr:
            if len(row) <= max(ik_i, nm_i):
                continue
            ik = row[ik_i].strip()
            if not ik:
                continue
            name = row[nm_i] or ""
            for m in _EXAMPLE_REF.finditer(name):
                pid = m.group(1).upper()
                by_patent[pid].add(ik)
                by_example[pid][_norm_cid(m.group(2))].add(ik)
            if pat_i is not None and len(row) > pat_i:
                p = re.sub(r"[^A-Z0-9]", "", (row[pat_i] or "").upper())
                if p:
                    by_patent[p if p.startswith("US") else "US" + p].add(ik)
    return by_patent, by_example


def _get_json(url: str, tries: int = 3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def load_chembl(patent_ids: list[str], *, refresh: bool = False) -> dict[str, set[str]]:
    """Return patent → InChIKeys, via ChEMBL's PATENT-type documents.

    ChEMBL writes patent ids as `US-20130089624-A1`; we normalize to `US...`.
    Cached, because this is a slow multi-request walk per patent.
    """
    cache: dict[str, list[str]] = {}
    if CHEMBL_CACHE.exists() and not refresh:
        cache = json.loads(CHEMBL_CACHE.read_text())
    base = "https://www.ebi.ac.uk/chembl/api/data"
    out: dict[str, set[str]] = {}
    wanted = [p for p in patent_ids if p.upper() not in cache]
    for pid in wanted:
        keys: set[str] = set()
        try:
            bare = re.sub(r"^US", "", pid.upper())
            bare = re.sub(r"[A-Z]\d?$", "", bare)
            docs = _get_json(
                f"{base}/document.json?doc_type=PATENT"
                f"&patent_id__contains={bare}&limit=20"
            ) or {}
            for doc in docs.get("documents", []):
                did = doc.get("document_chembl_id")
                if not did:
                    continue
                acts = _get_json(
                    f"{base}/activity.json?document_chembl_id={did}"
                    f"&limit=1000&only=molecule_chembl_id"
                ) or {}
                mols = {a.get("molecule_chembl_id")
                        for a in acts.get("activities", []) if a.get("molecule_chembl_id")}
                for chunk in [list(mols)[i:i + 40] for i in range(0, len(mols), 40)]:
                    m = _get_json(
                        f"{base}/molecule.json?molecule_chembl_id__in="
                        f"{','.join(chunk)}&limit=40"
                        f"&only=molecule_structures"
                    ) or {}
                    for mol in m.get("molecules", []):
                        st = mol.get("molecule_structures") or {}
                        if st.get("standard_inchi_key"):
                            keys.add(st["standard_inchi_key"])
        except Exception as e:
            print(f"  ! chembl {pid}: {e}", file=sys.stderr)
        cache[pid.upper()] = sorted(keys)
    CHEMBL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CHEMBL_CACHE.write_text(json.dumps(cache))
    for pid in patent_ids:
        out[pid.upper()] = set(cache.get(pid.upper(), []))
    return out


# ── our extraction ────────────────────────────────────────────────

def load_ours(pid: str) -> dict[str, dict]:
    f = EXTRACTIONS / pid / "example_index.json"
    if not f.exists():
        return {}
    try:
        data = json.loads(f.read_text())
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


# ── scoring ───────────────────────────────────────────────────────

def score(pid: str, ours: dict, ref_all: set[str],
          ref_by_ex: dict[str, set[str]]) -> dict:
    our_keys = {(r.get("inchikey") or "").strip()
                for r in ours.values() if r.get("inchikey")}
    our_skel = {_skeleton(k) for k in our_keys}
    ref_skel = {_skeleton(k) for k in ref_all}

    covered = ref_all & our_keys
    covered_skel = ref_skel & our_skel

    # Correctness: only where reference and we both label the same example.
    checked = exact = skel_ok = wrong_id = 0
    per_method: dict[str, Counter] = defaultdict(Counter)
    for cid, rec in ours.items():
        gold = ref_by_ex.get(_norm_cid(cid))
        if not gold:
            continue
        ik = (rec.get("inchikey") or "").strip()
        if not ik:
            continue
        method = rec.get("extraction_method") or "(unset)"
        checked += 1
        per_method[method]["n"] += 1
        if ik in gold:
            exact += 1
            per_method[method]["exact"] += 1
        elif _skeleton(ik) in {_skeleton(g) for g in gold}:
            skel_ok += 1
            per_method[method]["skeleton"] += 1
        elif ik in ref_all:
            # Right molecule for this patent, attached to the wrong example.
            wrong_id += 1
            per_method[method]["wrong_id"] += 1
    return {
        "patent": pid,
        "ours_total": len(ours),
        "ours_with_structure": len(our_keys),
        "ref_total": len(ref_all),
        "coverage_exact": len(covered),
        "coverage_skeleton": len(covered_skel),
        "checked": checked,
        "exact": exact,
        "skeleton_only": skel_ok,
        "wrong_id": wrong_id,
        "per_method": {k: dict(v) for k, v in per_method.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", help="comma-separated; default = everything extracted")
    ap.add_argument("--refs", default="bindingdb",
                    help="bindingdb,chembl (chembl hits the network)")
    ap.add_argument("--json", help="write full results here")
    args = ap.parse_args()

    pids = ([p.strip().upper() for p in args.patents.split(",")] if args.patents
            else sorted(d.name.upper() for d in EXTRACTIONS.iterdir()
                        if d.is_dir() and not d.name.startswith("_")))
    refs = {r.strip() for r in args.refs.split(",")}

    bdb_all: dict[str, set[str]] = {}
    bdb_ex: dict[str, dict[str, set[str]]] = {}
    if "bindingdb" in refs:
        print("loading BindingDB ...", file=sys.stderr)
        bdb_all, bdb_ex = load_bindingdb()
    chembl: dict[str, set[str]] = {}
    if "chembl" in refs:
        print("loading ChEMBL ...", file=sys.stderr)
        chembl = load_chembl(pids)

    rows, agg = [], Counter()
    method_tot: dict[str, Counter] = defaultdict(Counter)
    for pid in pids:
        ours = load_ours(pid)
        if not ours:
            continue
        ref_all = set(bdb_all.get(pid, set())) | set(chembl.get(pid, set()))
        if not ref_all:
            continue
        r = score(pid, ours, ref_all, bdb_ex.get(pid, {}))
        rows.append(r)
        for k in ("ours_with_structure", "ref_total", "coverage_exact",
                  "coverage_skeleton", "checked", "exact", "skeleton_only", "wrong_id"):
            agg[k] += r[k]
        for m, c in r["per_method"].items():
            for k, v in c.items():
                method_tot[m][k] += v

    if not rows:
        print("no patents had reference data — nothing to score")
        return 1

    print(f"\n{'patent':<18}{'ref':>6}{'covered':>9}{'cov%':>7}"
          f"{'checked':>9}{'exact':>7}{'ok%':>7}{'wrongID':>9}")
    for r in sorted(rows, key=lambda x: -x["ref_total"]):
        cov = r["coverage_exact"] / max(r["ref_total"], 1) * 100
        acc = r["exact"] / max(r["checked"], 1) * 100
        print(f"{r['patent']:<18}{r['ref_total']:>6}{r['coverage_exact']:>9}{cov:>6.1f}%"
              f"{r['checked']:>9}{r['exact']:>7}{acc:>6.1f}%{r['wrong_id']:>9}")

    print(f"\n{'TOTAL':<18}{agg['ref_total']:>6}{agg['coverage_exact']:>9}"
          f"{agg['coverage_exact']/max(agg['ref_total'],1)*100:>6.1f}%"
          f"{agg['checked']:>9}{agg['exact']:>7}"
          f"{agg['exact']/max(agg['checked'],1)*100:>6.1f}%{agg['wrong_id']:>9}")

    print("\nCOVERAGE   reference compounds we found : "
          f"{agg['coverage_exact']}/{agg['ref_total']} "
          f"({agg['coverage_exact']/max(agg['ref_total'],1)*100:.1f}% exact, "
          f"{agg['coverage_skeleton']/max(agg['ref_total'],1)*100:.1f}% ignoring stereo)")
    print("CORRECTNESS same-example structure agreement: "
          f"{agg['exact']}/{agg['checked']} "
          f"({agg['exact']/max(agg['checked'],1)*100:.1f}%)")
    print(f"           of which right-molecule-wrong-ID : {agg['wrong_id']} "
          f"({agg['wrong_id']/max(agg['checked'],1)*100:.1f}%) — an attribution bug, "
          f"not a chemistry one")

    print(f"\n{'extraction_method':<44}{'n':>7}{'exact%':>9}{'wrongID%':>10}")
    for m, c in sorted(method_tot.items(), key=lambda kv: -kv[1]["n"]):
        if c["n"] < 5:
            continue
        print(f"{m[:43]:<44}{c['n']:>7}{c['exact']/c['n']*100:>8.1f}%"
              f"{c['wrong_id']/c['n']*100:>9.1f}%")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"per_patent": rows, "totals": dict(agg),
             "per_method": {k: dict(v) for k, v in method_tot.items()}}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
