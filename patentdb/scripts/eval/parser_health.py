"""Does our reader agree with the patents' own XML? Corpus-wide, one command.

    python3 -m patentdb.scripts.eval.parser_health              # scan (free, CI gate)
    python3 -m patentdb.scripts.eval.parser_health --repair     # diagnose, patch, APPLY
    python3 -m patentdb.scripts.eval.parser_health --history    # what has been changed
    python3 -m patentdb.scripts.eval.parser_health --revert 3   # undo entry 0003
    python3 -m patentdb.scripts.eval.parser_health --force 4    # apply a declined one

`--repair` heals without asking. That is the point: a fix that waits for someone
to flip a switch is a queue, and the defect is corrupting every patent while it
waits. What makes it safe is not permission but the record — every proposal,
applied or declined, is journaled with its full before/after source and the
per-patent coverage it moved, so any state is recoverable.

Exit code is non-zero when the reader disagrees with any source, so the bare
scan works as a CI gate: a parsed cell must exist for every `<entry>` the patent
declares, and nothing downstream can be trusted while that is violated.

The scan is deterministic and free. `--repair` costs ONE model call per distinct
defect, not per patent — a reader bug is global, so paying per patent to work
around it is waste. Measured on the `<entry/>` defect: 63 patents, 1,261 tables,
49,818 cells, reduced to a single 44-byte reproduction and one question.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from ...repair.parser_repair import (
    apply_journaled, corpus_defects, journal_read, repair_reader, revert,
)


def _print_history() -> int:
    entries = journal_read()
    if not entries:
        print("no parser repairs recorded")
        return 0
    print(f"{len(entries)} journal entr(ies)\n")
    for e in entries:
        state = ("APPLIED" if e.get("applied") else
                 "declined" if e.get("action") == "patch" else e.get("action", "?"))
        print(f"  {e.get('id')}  {state:9s} {e.get('action')}  "
              f"{e.get('signature', '')}")
        if e.get("blast_radius"):
            print(f"      radius: {e['blast_radius']}")
        if e.get("why"):
            print(f"      why   : {str(e['why'])[:150]}")
        before, after = e.get("total_usable_before"), e.get("total_usable_after")
        if before is not None and after is not None:
            print(f"      usable: {before} -> {after}")
        moved = e.get("coverage_moved") or {}
        if moved:
            shown = list(moved.items())[:6]
            print(f"      moved : " + ", ".join(f"{k} {v[0]}->{v[1]}" for k, v in shown)
                  + (f" (+{len(moved) - len(shown)} more)" if len(moved) > 6 else ""))
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repair", action="store_true",
                    help="diagnose, patch and apply (one model call per defect)")
    ap.add_argument("--history", action="store_true", help="show the repair journal")
    ap.add_argument("--revert", metavar="ID", help="undo a journaled patch")
    ap.add_argument("--force", metavar="ID",
                    help="apply a journaled proposal the check declined")
    ap.add_argument("--limit", type=int, default=None,
                    help="scan only the first N cached patents")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.history:
        return _print_history()
    if args.revert:
        r = revert(args.revert)
        print(json.dumps(r, indent=2))
        return 0 if r.get("ok") else 1
    if args.force:
        r = apply_journaled(args.force)
        print(json.dumps(r, indent=2))
        return 0 if r.get("ok") else 1

    defects = corpus_defects(limit=args.limit)
    if not defects:
        print("parser health: OK — every <entry> in every cached patent has a cell")
        return 0

    print(f"parser health: {len(defects)} distinct defect(s)\n")
    for d in defects:
        print(f"  signature {d.signature!r} — {d.blast_radius}")
        print(f"    repro: {d.repro[:200]}")
        print(f"    patents: {', '.join(d.patents[:8])}"
              f"{' …' if len(d.patents) > 8 else ''}\n")

    if not args.repair:
        print("Run with --repair to fix them (one call per defect, not per patent).")
        return 1

    report = repair_reader()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["declined"] == 0 else 1

    for res in report["results"]:
        state = "APPLIED" if res.get("ok") else "DECLINED"
        print(f"\n  {res.get('journal_id', '?')} {state}: {res.get('signature')!r}")
        if res.get("diagnosis"):
            print(f"    diagnosis: {res['diagnosis'][:300]}")
        if not res.get("ok"):
            print(f"    why      : {res.get('why')}")
            jid = str(res.get("journal_id", "")).split("-")[0]
            print(f"    override : parser_health --force {jid}")
            continue
        print(f"    fidelity clean, tests pass, {res.get('total_usable')} usable")
        moved = res.get("coverage_moved") or {}
        if moved:
            print(f"    coverage moved on {len(moved)} patent(s): "
                  + ", ".join(f"{k} {v[0]}->{v[1]}" for k, v in list(moved.items())[:6]))
        if res.get("changed_on_corrupt_baseline"):
            # Not a regression — those baselines were measured with the broken
            # reader, so they cannot be a floor. Surfaced because it is worth
            # a look, not because it blocks anything.
            print(f"    REVIEW   : counts moved on patents whose baseline was "
                  f"itself corrupt: {res['changed_on_corrupt_baseline']}")
        jid = str(res.get("journal_id", "")).split("-")[0]
        print(f"    revert   : parser_health --revert {jid}")

    print(f"\napplied {report['applied']}, declined {report['declined']} "
          f"— `--history` for the full record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
