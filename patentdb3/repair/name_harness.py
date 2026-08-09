"""Sweep the name heal-loop's knobs and report YIELD AGAINST COST.

Not a pipeline and not a test. A bench: it runs `repair_names` over a fixed set
of patents under several configurations, each with its OWN scratch library and
journal, and prints compounds-recovered against dollars-spent for each.

WHY A SCRATCH LIBRARY PER CONFIGURATION
-----------------------------------------
The loop's whole economic argument is that a rule bought once is free
afterwards. That makes runs ORDER-DEPENDENT: configuration B run second
inherits everything A bought and looks free by comparison. Every configuration
therefore starts from an empty synthesized library, so the numbers are
comparable — and the shipped `data/name_rules.json` is never touched, which
also means running this bench cannot silently change what the pipeline does.

WHAT IS ACTUALLY BEING TRADED
-------------------------------
  BATCH        failures shown per call. Larger = fewer calls over the same
               gaps and more chance the model sees the shared form, but a
               longer user turn on every call and more names for one rule to
               satisfy.
  MAX_TURNS    model turns per proposal.
  SHOW_CLEAN   already-parsing names included as contrast. Pure input cost;
               the question is whether it buys accuracy.
  opsin_error  whether OPSIN's own diagnosis is attached to each failure. It
               is free to obtain (one subprocess per patent) but costs input
               tokens on every call, so it has to earn them.

COST IS READ FROM THE TRACKER, NOT ESTIMATED. `cost_tracker` prices the
prompt-cache buckets at Anthropic's real multipliers (reads 0.1x, writes
1.25x), which `usage.input_tokens` alone does not include — an estimate built
from input+output would report a cached run as free.

    python3 -m patentdb3.repair.name_harness              # default sweep
    python3 -m patentdb3.repair.name_harness --patents US10155002,US10710980
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config
from ..core.cost_tracker import cost_tracker
from . import name_synthesize as SYN
from .name_gap import find_name_gaps
from .name_loop import repair_names
from .name_rules import NameRuleLibrary

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """One point in the sweep. `name` is what gets printed."""
    name: str
    batch: int = SYN.BATCH
    max_turns: int = SYN.MAX_TURNS
    show_clean: int = SYN.SHOW_CLEAN
    opsin_errors: bool = True
    max_calls: int = 4


@dataclass
class Result:
    config: str
    gaps: int = 0
    fixed_library: int = 0
    fixed_new: int = 0
    rules: list[str] = field(default_factory=list)
    escalated: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    rejected: list[str] = field(default_factory=list)

    @property
    def fixed(self) -> int:
        return self.fixed_library + self.fixed_new

    @property
    def usd_per_fix(self) -> float:
        return self.usd / self.fixed_new if self.fixed_new else float("inf")


# The default sweep. Each entry changes ONE thing from `baseline` so a
# difference is attributable — a sweep where two knobs move at once reports
# which CONFIGURATION won and nothing about which knob did it.
SWEEP: list[Config] = [
    Config("baseline           b8 t2 c3 +err"),
    Config("no-opsin-error     b8 t2 c3 -err", opsin_errors=False),
    Config("no-clean-contrast  b8 t2 c0 +err", show_clean=0),
    Config("small-batch        b3 t2 c3 +err", batch=3),
    Config("large-batch       b16 t2 c3 +err", batch=16),
]


def _run_one(cfg: Config, patents: list[str], xmls: dict[str, str]) -> Result:
    """One configuration over every patent, from an empty library."""
    SYN.BATCH, SYN.MAX_TURNS, SYN.SHOW_CLEAN = cfg.batch, cfg.max_turns, cfg.show_clean

    tmp = Path(tempfile.mkdtemp(prefix="nameharness_"))
    lib = NameRuleLibrary(tmp / "lib.json")
    res = Result(config=cfg.name)
    t0 = time.time()
    spent0 = cost_tracker.total_cost

    for pid in patents:
        if not cfg.opsin_errors:
            # Strip the annotation rather than skipping detection, so the two
            # arms see the IDENTICAL gap set and differ only in what the prompt
            # carries. Re-detecting without it would also change which gaps
            # exist if detection ever starts depending on the error text.
            gaps = find_name_gaps(xmls[pid], pid, with_opsin_errors=False)
            if not gaps:
                continue
        try:
            r = repair_names(xmls[pid], pid, library=lib,
                             journal=tmp / "journal.jsonl",
                             max_calls=cfg.max_calls)
        except Exception as e:
            logger.warning("harness: %s failed under %s: %r", pid, cfg.name, e)
            continue
        res.gaps += r.gaps_found
        res.fixed_library += r.fixed_by_library
        res.fixed_new += r.fixed_by_new_rule
        res.rules += r.rules_adopted
        res.escalated += r.escalated
        res.rejected += r.rejected[:2]

    res.usd = round(cost_tracker.total_cost - spent0, 5)
    res.seconds = round(time.time() - t0, 1)
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patents", default="",
                    help="comma-separated; default = the worst gap-bearing ones")
    ap.add_argument("--n", type=int, default=3, help="how many patents")
    ap.add_argument("--only", default="", help="run one named config")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.patents:
        patents = [p.strip() for p in args.patents.split(",") if p.strip()]
    else:
        # Pick the patents that actually HAVE gaps — a sweep over patents with
        # nothing to fix measures nothing and still costs the system prefix.
        print("scanning for gap-bearing patents ...", flush=True)
        scored: list[tuple[int, str]] = []
        for p in sorted(config.XML_INPUT_DIR.glob("*.xml")):
            try:
                n = len(find_name_gaps(p.read_text(errors="replace"), p.stem,
                                       with_opsin_errors=False))
            except Exception:
                continue
            if n:
                scored.append((n, p.stem))
        scored.sort(reverse=True)
        patents = [pid for _, pid in scored[: args.n]]
        print(f"  using {patents}")

    xmls = {pid: (config.XML_INPUT_DIR / f"{pid}.xml").read_text(errors="replace")
            for pid in patents}

    sweep = [c for c in SWEEP if not args.only or c.name.startswith(args.only)]
    results: list[Result] = []
    for cfg in sweep:
        print(f"\n=== {cfg.name} ===", flush=True)
        r = _run_one(cfg, patents, xmls)
        results.append(r)
        print(f"  gaps {r.gaps}  fixed {r.fixed} (lib {r.fixed_library}, "
              f"new {r.fixed_new})  rules {len(r.rules)}  ${r.usd:.4f}  "
              f"{r.seconds}s", flush=True)

    print("\n" + "=" * 86)
    print(f"{'configuration':<34}{'gaps':>6}{'fixed':>7}{'new':>6}"
          f"{'rules':>7}{'USD':>9}{'$/new':>9}{'sec':>7}")
    for r in results:
        per = "—" if r.usd_per_fix == float("inf") else f"{r.usd_per_fix:.4f}"
        print(f"{r.config:<34}{r.gaps:>6}{r.fixed:>7}{r.fixed_new:>6}"
              f"{len(r.rules):>7}{r.usd:>9.4f}{per:>9}{r.seconds:>7.1f}")
    print("=" * 86)
    print("\nrules adopted per configuration:")
    for r in results:
        print(f"  {r.config:<34} {r.rules}")
    print("\nfirst rejections (what the gate said):")
    for r in results:
        for x in r.rejected[:2]:
            print(f"  [{r.config.split()[0]}] {x[:150]}")

    out = config.OUTPUT_DIR / "name_harness.json"
    out.write_text(json.dumps([r.__dict__ for r in results], indent=1, default=str))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
