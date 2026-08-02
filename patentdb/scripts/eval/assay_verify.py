"""Verify the ASSAY payload on every patent: values, units, and n_runs.

Coverage answers "did we find the compound". This answers the question that
actually makes a row useful: is the measurement attached to it complete and
right. They fail independently — a patent can be at 100% compound coverage
while every value carries a fabricated unit, and that has happened here.

Three things are reported per patent, and they are deliberately NOT blended:

  COMPLETENESS  of the records we produced, how many carry a number, a unit,
                and a replicate count. `n_runs` is sparse by nature — most
                patents never print one — so it is reported as "of the cells
                that stated one, how many did we keep", never as a share of
                all records. A metric that scores us against data the document
                does not contain reads as a failure we cannot fix.

  AGREEMENT     for the values BindingDB also publishes, do ours match. 5%
                tolerance for BDB's three-significant-figure rounding, then
                bucketed: variance (2x), disagree (10x), wrong_scale beyond.

  SILENCE       records that exist but are unusable, by the field they lack.
                This is where a patent hides: 242 records all missing `unit`
                is invisible to a compound count and fatal to every one of them.

Nothing here gates anything. It reports.

    python3 -m patentdb.scripts.eval.assay_verify
    python3 -m patentdb.scripts.eval.assay_verify --patents US9018217,US10266548
    python3 -m patentdb.scripts.eval.assay_verify --json docs/reports/assays.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys

from ...core import config

# A cell that STATES a replicate count, read off the RAW table rather than off
# a parsed record. Both halves of that matter and both were wrong first time.
#
# RAW, because a denominator taken from `value_text` only sees cells that
# already PARSED — so every cell whose replicate count we failed to read was
# also missing from the population we scored ourselves against, and the corpus
# reported 41 such cells when it holds 606. A denominator computed from the
# parsed view cannot measure the parser.
#
# And ANCHORED to a measurement, because `\(\d+\)$` alone matches
# `Tris(dibenzylideneacetone)dipalladium(0)`, `C(10)`, `O(2)` and
# crystallographic `9211(1)` — 1,597 hits, almost all reagent names and atom
# labels. A replicate count follows a number.
_STATES_N = re.compile(
    r"^[<>~≈≥≤]?\s*\d[\d.,]*\s*(?:\(\s*\d{1,3}\s*\)|.*?\bn\s*=\s*\d{1,3})\s*$",
    re.I)
# ...and a bare `n = 3` in its OWN column, which holds no measurement at all.
# `parse_value` returns None for it, correctly; `extract_from_tables` reads it
# from the neighbouring cell. Counted separately so the two paths stay legible.
_BARE_N = re.compile(r"^[^()]*\bn\s*=\s*\d{1,3}\b\s*$", re.I)


def _n_stating_cells(xml: str) -> tuple[int, int]:
    """(cells stating a replicate count, of which are bare `n = N` columns)."""
    from ...sources.uspto_assays import _header_rows_of
    from ...sources.uspto_xml import assemble_blocks, parse_tables

    total = bare = 0
    for t in assemble_blocks(parse_tables(xml)):
        _, data = _header_rows_of(t)
        for row in data:
            for c in row:
                s = c.text.strip()
                if not s:
                    continue
                if _STATES_N.match(s):
                    total += 1
                elif _BARE_N.match(s):
                    total += 1
                    bare += 1
    return total, bare


def audit_patent(patent_id: str, xml: str) -> dict:
    from ...repair.loop import repair_patent
    from ...sources.uspto_assays import extract_from_patent

    recs = list(extract_from_patent(xml))
    try:
        extra, _ = repair_patent(patent_id, xml, max_calls=0)
        recs += list(extra)
    except Exception:                            # the loop is additive here too
        pass

    usable = [r for r in recs if r.is_usable]
    stated, bare = _n_stating_cells(xml)
    missing = collections.Counter(
        f for r in recs if not r.is_usable for f in r.missing_fields())

    return {
        "patent": patent_id,
        "records": len(recs),
        "usable": len(usable),
        "compounds": len({r.cid for r in usable if r.cid}),
        "with_value": sum(1 for r in usable if r.value_numeric is not None),
        "with_range": sum(1 for r in usable
                          if r.value_numeric is None and r.range_lo is not None),
        # Against ALL records, not just usable ones: `is_usable` REQUIRES a
        # unit, so "unit, as a share of usable" is 100% by construction and
        # measures nothing. The interesting number is how many records we
        # produced at all that never got one.
        "with_unit": sum(1 for r in recs if r.unit),
        "with_assay_name": sum(1 for r in recs if r.assay_name),
        # n_runs against the cells that STATED one, read from the raw table.
        "n_runs_stated": stated,
        "n_runs_bare_column": bare,
        "n_runs_kept": sum(1 for r in recs if r.n_runs is not None),
        "unusable_missing": dict(missing),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", help="comma-separated; default = all cached")
    ap.add_argument("--json", help="write full per-patent results here")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    a = ap.parse_args()

    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    pids = ([p.strip().upper() for p in a.patents.split(",")] if a.patents
            else sorted(p.stem for p in xml_dir.glob("*.xml")))

    rows, failed = [], []
    for pid in pids:
        f = xml_dir / f"{pid}.xml"
        if not f.exists():
            continue
        try:
            rows.append(audit_patent(pid, f.read_text(errors="ignore")))
        except Exception as e:                   # a CODE error; never silent
            failed.append({"patent": pid, "error": repr(e)[:200]})

    # Values against BindingDB. Separate pass — it needs the reference loaded
    # once, and a patent absent from BDB is not a patent we got wrong.
    agree: dict[str, dict] = {}
    try:
        from ...repair.value_check import check_corpus
        agree = check_corpus([r["patent"] for r in rows]).get("per_patent", {})
    except Exception as e:
        print(f"(value check unavailable: {e!r})", file=sys.stderr)

    if not a.quiet:
        print(f"{'patent':15s} {'rec':>6s} {'usable':>7s} {'cmpd':>6s} "
              f"{'unit':>6s} {'n=':>7s} {'BDB ok':>8s}  top missing")
        for r in sorted(rows, key=lambda x: x["usable"]):
            b = (agree.get(r["patent"]) or {}).get("buckets") or {}
            refs = b.get("refs") or 0
            ok = f"{(b.get('agree', 0) / refs):.0%}" if refs else "-"
            nr = (f"{r['n_runs_kept']}/{r['n_runs_stated']}"
                  if r["n_runs_stated"] else "-")
            miss = ", ".join(f"{k}:{v}" for k, v in
                             sorted(r["unusable_missing"].items(),
                                    key=lambda kv: -kv[1])[:2]) or "-"
            print(f"{r['patent']:15s} {r['records']:6d} {r['usable']:7d} "
                  f"{r['compounds']:6d} {r['with_unit']:6d} {nr:>7s} "
                  f"{ok:>8s}  {miss}")

    t = collections.Counter()
    for r in rows:
        for k in ("records", "usable", "compounds", "with_value", "with_range",
                  "with_unit", "n_runs_stated", "n_runs_kept",
                  "n_runs_bare_column"):
            t[k] += r[k]
    miss_tot = collections.Counter()
    for r in rows:
        miss_tot.update(r["unusable_missing"])

    print(f"\n=== {len(rows)} patents ===")
    print(f"  records            {t['records']}")
    print(f"  usable             {t['usable']}"
          f"  ({t['usable'] / max(t['records'], 1):.1%})")
    print(f"  compounds          {t['compounds']}")
    print(f"  with a number      {t['with_value']}")
    print(f"  with a range only  {t['with_range']}")
    print(f"  with a unit        {t['with_unit']}"
          f"  ({t['with_unit'] / max(t['records'], 1):.1%} of ALL records)")
    print(f"  n_runs             {t['n_runs_kept']} captured against "
          f"{t['n_runs_stated']} cells that state one"
          f"  ({t['n_runs_kept'] / max(t['n_runs_stated'], 1):.1%})")
    print(f"                     of those, {t['n_runs_bare_column']} are a bare "
          f"`n = N` column carrying no measurement")
    print("\n  unusable records, by the field they lack:")
    for k, v in miss_tot.most_common():
        print(f"     {v:7d}  {k}")

    ab = collections.Counter()
    for v in agree.values():
        ab.update(v.get("buckets") or {})
    if ab.get("refs"):
        # EVERY bucket, including `no_record` and `no_value`. Printing only the
        # comparison outcomes showed 14,273 agreeing out of 19,547 and left the
        # other 5,000 unaccounted for — those are references we produced nothing
        # for, which is the largest bucket after `agree` and the one a reader
        # most needs to see. A report that lists only the rows it can score
        # flatters itself.
        print(f"\n  values vs BindingDB ({ab['refs']} references):")
        for k, v in sorted(ab.items(), key=lambda kv: -kv[1]):
            if k == "refs" or not v:
                continue
            print(f"     {v:7d}  {k:14s} ({v / ab['refs']:.1%})")
        scored = ab["refs"] - ab.get("no_record", 0) - ab.get("no_value", 0)
        if scored:
            print(f"     of the {scored} we could actually compare, "
                  f"{ab.get('agree', 0) / scored:.1%} agree")

    if failed:
        print(f"\n  CODE ERRORS on {len(failed)} patent(s):")
        for f in failed:
            print(f"     {f['patent']:15s} {f['error']}")

    if a.json:
        dest = config.REPO_ROOT / a.json
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(
            {"per_patent": rows, "agreement": agree, "failed": failed,
             "totals": dict(t), "missing": dict(miss_tot)}, indent=1, default=str))
        print(f"\nwrote {dest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
