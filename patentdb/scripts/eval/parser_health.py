"""Does our reader agree with the patents' own XML? Corpus-wide, one command.

    python3 -m patentdb.scripts.eval.parser_health              # scan only
    python3 -m patentdb.scripts.eval.parser_health --repair     # + propose a patch
    PARSER_REPAIR_APPLY=1 ... --repair                          # + apply if verified

Exit code is non-zero when the reader disagrees with any source, so this works
as a CI gate — the invariant is that a parsed cell exists for every `<entry>`
the patent declares, and nothing downstream can be trusted while it is violated.

The scan is deterministic and free. `--repair` costs ONE model call per distinct
defect, not per patent: the whole point of this tier is that a reader bug is
global, so paying per-patent to work around it is waste. Measured on the
`<entry/>` defect: 63 patents, 1,261 tables, 49,818 cells lost, reduced to a
single 44-byte reproduction and one question.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from ...core import config
from ...repair.parser_repair import corpus_defects, repair_reader


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repair", action="store_true",
                    help="propose and verify a patch for each defect found")
    ap.add_argument("--limit", type=int, default=None,
                    help="scan only the first N cached patents")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
        print("Run with --repair to buy one patch per defect (not per patent).")
        return 1

    report = repair_reader()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["rejected"] == 0 else 1

    for res in report["results"]:
        print(f"\n  {res.get('signature')!r}: "
              f"{'ACCEPTED' if res.get('ok') else 'REJECTED'}")
        if res.get("diagnosis"):
            print(f"    diagnosis: {res['diagnosis'][:300]}")
        if not res.get("ok"):
            print(f"    why: {res.get('why')}")
            continue
        print(f"    fidelity clean, tests pass, {res.get('total_usable')} usable")
        if res.get("changed_on_corrupt_baseline"):
            # Not a regression — those baselines were measured with the broken
            # reader, so they cannot be a floor. Surfaced because a human should
            # still look at what moved.
            print(f"    REVIEW — counts moved on patents whose baseline was "
                  f"itself corrupt: {res['changed_on_corrupt_baseline']}")
        if res.get("patch"):
            print("    patch verified but NOT applied "
                  "(set PARSER_REPAIR_APPLY=1 to write it)")
    if report["applied"]:
        print(f"\napplied {report['applied']} patch(es); re-run to confirm clean")
    elif not config.PARSER_REPAIR_APPLY:
        print("\nnothing written — PARSER_REPAIR_APPLY is off")
    return 0 if report["rejected"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
