"""Corpus coverage as the PIPELINE sees it: the union of both reading tiers.

    python3 -m patentdb.scripts.eval.pipeline_bench
    python3 -m patentdb.scripts.eval.pipeline_bench --json docs/reports/bench.json

This exists because measuring one tier produced two different wrong answers in
one session, and each looked entirely plausible on its own:

    extract_from_patent only   32,505 compounds, 16 zeros
    repair_patent only          4,410 compounds, 78 zeros
    UNION                      35,355 compounds,  4 zeros

`uspto_assays.extract_from_patent` never calls `apply_rule` — the 84 learned
rules live in `repair/loop` — so the first figure omits every layout the rule
tier was bought to read, and twelve patents that produce 2,407 compounds
between them read as total failures. Acting on that, the capability tier spent
real money being asked to patch the parser so it could read documents another
tier already read; on US10030020 it correctly answered "nothing to select
from", because there was no gap.

`repair_patent` returns only the rule-recovered SUPPLEMENT, not the whole
result, so measuring it alone is wrong in the opposite direction: US10266548
reads as zero while the raw parser is yielding 118 compounds for it.

A compound counts if EITHER tier produces it. `--zero-only` prints the patents
neither tier can read, which is the list the repair loop should ever be pointed
at — `greedy.measure` still scores patches on the raw path, which is fine for
scoring (same metric before and after) but is NOT a coverage number.

`REPAIR=0` is forced: this reports what the code can already do, and a
benchmark that buys new rules while measuring is measuring itself.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def measure(xml_dir=None) -> dict:
    """Per-patent distinct usable compounds, counting both tiers."""
    os.environ["REPAIR"] = "0"
    from ...core import config
    from ...repair.loop import repair_patent
    from ...sources.uspto_assays import extract_from_patent

    xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")
    per: dict[str, int] = {}
    crashed: list[str] = []
    for f in sorted(xml_dir.glob("*.xml")):
        xml = f.read_text(errors="ignore")
        cids: set[str] = set()
        for tier, fn in (("raw", lambda: extract_from_patent(xml)),
                         ("rules", lambda: repair_patent(f.stem, xml)[0])):
            try:
                cids |= {r.cid for r in fn()
                         if getattr(r, "is_usable", False) and r.cid}
            except Exception as e:
                crashed.append(f"{f.stem}:{tier}")
                logger.warning("bench: %s %s raised %r", f.stem, tier, e)
        per[f.stem] = len(cids)
    zeros = sorted(p for p, n in per.items() if n == 0)
    return {"per_patent": per, "compounds": sum(per.values()),
            "n_patents": len(per), "n_zero": len(zeros), "zeros": zeros,
            "crashed": sorted(set(crashed))}


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="also write the full per-patent result here")
    ap.add_argument("--zero-only", action="store_true",
                    help="print only the patents neither tier can read")
    a = ap.parse_args()

    out = measure()
    if a.zero_only:
        for p in out["zeros"]:
            print(p)
        return 0
    print(f"patents   : {out['n_patents']}")
    print(f"compounds : {out['compounds']}")
    print(f"zeros     : {out['n_zero']}  {out['zeros']}")
    if out["crashed"]:
        print(f"CRASHED   : {out['crashed']}")
    if a.json:
        from ...core import config
        dest = config.REPO_ROOT / a.json
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1))
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
