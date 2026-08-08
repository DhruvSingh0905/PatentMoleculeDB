"""Are our numbers right? Checked against BindingDB, at 5% tolerance.

    python3 -m patentdb.scripts.eval.value_check_cli
    python3 -m patentdb.scripts.eval.value_check_cli --patents US11254686

Coverage says we found the compound; this says whether we read its value. The
two fail in opposite directions, and only this one can see a dimensionless
ratio recorded as a nanomolar potency — 99 such records existed while every
count went up.

5% is BindingDB's rounding: it stores three significant figures, so our
`0.023456 uM` and its `23.5 nM` are the same measurement. Beyond that the
buckets separate assay variability (up to 2x) from a different quantity
(over 10x). Exit code is non-zero when any value disagrees.
"""
from __future__ import annotations

import argparse
import json
import sys

from ...repair.value_check import TOLERANCE, check_corpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", default=None, help="comma-separated ids")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = check_corpus(a.patents.split(",") if a.patents else None)
    if a.json:
        print(json.dumps(r, indent=2, default=str))
        return 1 if r["bad_patents"] else 0
    t = r["total"]
    n = t.get("refs", 0)
    print(f"value check vs BindingDB at {TOLERANCE:.0%} tolerance — "
          f"{len(r['per_patent'])} patents, {n} reference values\n")
    for k, label in (("agree", "agree (within 5%)"),
                     ("range_contains", "range contains reference"),
                     ("variance", "variance (5%-2x)"),
                     ("disagree", "DISAGREE (2x-10x)"),
                     ("wrong_scale", "WRONG SCALE (>10x)"),
                     ("range_misses", "range misses reference"),
                     ("no_record", "compound not extracted"),
                     ("no_value", "extracted, no usable value")):
        if t.get(k):
            print(f"   {label:28s} {t[k]:6d}  {100 * t[k] / n:5.1f}%")
    if not r["bad_patents"]:
        print("\nno value disagreements")
        return 0
    print(f"\n{len(r['bad_patents'])} patent(s) with a disagreeing value:")
    for p, cnt in sorted(r["bad_patents"].items(), key=lambda x: -x[1]):
        print(f"\n  {p}: {cnt}")
        for e in r["per_patent"][p]["examples"][:4]:
            print(f"     [{e['bucket']}] {e['cid']}: {e['detail']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
