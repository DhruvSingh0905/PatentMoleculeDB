"""Layouts no RULE can express — find them, patch the code, verify, apply.

    python3 -m patentdb.scripts.eval.capability_repair              # scan, free
    python3 -m patentdb.scripts.eval.capability_repair --repair     # patch + APPLY
    python3 -m patentdb.scripts.eval.capability_repair --repair --limit 1
    python3 -m patentdb.scripts.eval.parser_health --history        # shared journal
    python3 -m patentdb.scripts.eval.parser_health --revert 7       # shared revert

A capability gap is defined by outcome: a gap that had a rule available and
produced no records. Not a bad rule — a shape the rule vocabulary cannot say.
US9302989 is the case that motivated it: 1,561 rows of `0.0125, nd`, two
measurements in one cell, behind a column map that is entirely correct.

The scan is free and deterministic. `--repair` costs ONE model call per
distinct layout fingerprint, and every candidate is verified against the whole
corpus and the full test suite in a scratch tree before it can touch anything.

Exit code is non-zero while any capability gap is open, so it works as a CI
signal: it means the loop has found something it cannot currently fix.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from ...repair.capability import collect_gaps, repair_capabilities


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repair", action="store_true",
                    help="buy a patch per gap, verify, and APPLY")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the N largest gaps")
    ap.add_argument("--patents", default=None,
                    help="comma-separated patent ids (default: all cached)")
    ap.add_argument("--model", default=None, help="override the synthesis model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pids = args.patents.split(",") if args.patents else None

    if not args.repair:
        gaps = collect_gaps(pids)
        if args.json:
            print(json.dumps(gaps, indent=2, default=str))
            return 1 if gaps else 0
        if not gaps:
            print("capability: OK — every gap with a rule available produces records")
            return 0
        rows = sum(g["rows_at_stake"] for g in gaps)
        print(f"capability: {len(gaps)} layout(s) no rule can express, "
              f"{rows} rows at stake\n")
        for g in gaps:
            print(f"  {g['fingerprint']}  {g['patent']} {g['table']}  "
                  f"{g['rows_at_stake']} rows")
            print(f"     tried  : {g['rule_kind']} {json.dumps(g['rule_payload'])[:110]}")
            print(f"     why    : {g['why'][:150]}")
            if g.get("unparsed_examples"):
                print(f"     cells  : {g['unparsed_examples'][:4]}")
            print()
        print("Run with --repair to patch the code (one call per layout).")
        return 1

    report = repair_capabilities(limit=args.limit, patent_ids=pids, model=args.model)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["declined"] == 0 else 1

    if not report["gaps"]:
        print("capability: nothing to repair")
        return 0
    for r in report["results"]:
        state = "APPLIED" if r.get("ok") else "DECLINED"
        print(f"\n  {r.get('journal_id', '—')} {state}: {r.get('fingerprint')} "
              f"→ {r.get('target', '?')}  ({r.get('rows_at_stake')} rows)")
        if r.get("diagnosis"):
            print(f"    diagnosis: {r['diagnosis'][:300]}")
        if not r.get("ok"):
            print(f"    why      : {r.get('why')}")
            continue
        print(f"    verified : corpus clean, tests pass, {r.get('total_usable')} usable")
        print(f"    RECOVERED: +{r.get('gap_rows_recovered')} records on the gap's own patent")
        moved = r.get("coverage_moved") or {}
        if moved:
            shown = list(moved.items())[:6]
            print("    moved    : " + ", ".join(f"{k} {v[0]}->{v[1]}" for k, v in shown)
                  + (f" (+{len(moved) - len(shown)} more)" if len(moved) > 6 else ""))
        jid = str(r.get("journal_id", "")).split("-")[0]
        print(f"    revert   : parser_health --revert {jid}")
    print(f"\napplied {report['applied']}, declined {report['declined']} "
          f"— `parser_health --history` for the full record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
