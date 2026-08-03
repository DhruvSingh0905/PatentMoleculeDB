"""Propose patches for every open capability gap, then keep the best SET.

    python3 -m patentdb.scripts.eval.greedy_repair --dry-run
    python3 -m patentdb.scripts.eval.greedy_repair --patents US10189840,US9018217
    python3 -m patentdb.scripts.eval.greedy_repair --apply

The old tier bought one patch per gap and judged each against the whole corpus
independently, which is why four correct patches in a row were declined for
what they did to patents they were never about. Two things change here:

  FROZEN PATENTS ARE OUT OF SCOPE. A patch is scored on the patent it targets
  plus those not yet frozen — see `repair/snapshot`. Finished work stops being
  a floor.

  THE SET IS CHOSEN GREEDILY. Candidates are tried against the tree as it
  stands, best-first, re-measuring after every accept, because patch effects
  compose through header promotion and are not additive.

Rejected candidates are retried in later rounds: a patch can become viable once
another has landed, and that is exactly the interaction the old one-shot
verifier could not represent.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ...core import config
from ...repair import snapshot
from ...repair.greedy import Candidate, select

logger = logging.getLogger(__name__)


def _candidates(pids: list[str] | None, limit: int,
                rounds: int | None = None) -> list[Candidate]:
    """One candidate per open capability gap — REFINED, not asked once.

    Each gap goes through `iterate.refine`: propose a patch, splice it, run a
    real extraction, feed the measured result back, propose again. What arrives
    here is the best attempt of that conversation, which `greedy` then scores
    against the corpus exactly as before. The two loops answer different
    questions — `refine` asks "does this patch read the patent it was written
    for", `greedy` asks "is it worth having in the tree" — and neither
    substitutes for the other.
    """
    from ...repair.capability import _attach_oracle, collect_gaps
    from ...repair.iterate import refine
    from ...sources.uspto_xml import assemble_blocks, parse_tables

    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    gaps = collect_gaps(pids)[:limit]
    out: list[Candidate] = []
    for g in gaps:
        xml = (xml_dir / f"{g['patent']}.xml").read_text(errors="ignore")
        table = {t.table_id: t for t in assemble_blocks(parse_tables(xml))}.get(g["table"])
        if table is None:
            continue
        g["xml"] = xml
        _attach_oracle(g, xml)
        best, attempts = refine(g, table, rounds=rounds)
        for a in attempts:
            print(f"  round {a.round_no}: {g['patent']} "
                  f"{a.target_before} -> {a.target_after} compounds  "
                  f"edited={a.names or '-'} noop={a.noop or '-'}"
                  + (f"  BROKEN {a.broken[:60]}" if a.broken else ""),
                  file=sys.stderr)
        if best is None:
            logger.warning("greedy: %s — %d round(s), nothing runnable",
                           g["patent"], len(attempts))
            continue
        out.append(Candidate(
            label=f"{g['patent']}:{','.join(best.names)}",
            target_patent=g["patent"], edits=best.edits,
            diagnosis=best.diagnosis,
            fingerprint=str(g.get("fingerprint")),
            meta={"rows_at_stake": g.get("rows_at_stake"),
                  "rounds": len(attempts), "round_kept": best.round_no,
                  "target_after": best.target_after}))
    return out


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", help="comma-separated; default = every open gap")
    ap.add_argument("--limit", type=int, default=8, help="max candidates to buy")
    ap.add_argument("--rounds", type=int, default=None,
                    help="observe-and-adjust rounds per gap (default "
                         f"{__import__('os').environ.get('CAPABILITY_ROUNDS', 3)})")
    ap.add_argument("--apply", action="store_true",
                    help="write the winning set to the tree (default: measure only)")
    ap.add_argument("--json", default="docs/reports/greedy_repair.json")
    a = ap.parse_args()

    pids = [p.strip().upper() for p in a.patents.split(",")] if a.patents else None
    frozen = snapshot.frozen_ids()
    print(f"frozen patents (out of scope): {len(frozen)}", file=sys.stderr)

    cands = _candidates(pids, a.limit, a.rounds)
    print(f"candidates: {len(cands)}", file=sys.stderr)
    for c in cands:
        print(f"  {c.label:44s} rows_at_stake={c.meta.get('rows_at_stake')}",
              file=sys.stderr)
    if not cands:
        print("nothing to select from")
        return 0

    outcomes = select(cands, apply=a.apply)

    print(f"\n{'candidate':46s} {'ok':>4s} {'tgt':>5s} {'corpus':>7s}  why")
    for o in outcomes:
        print(f"{o.label:46s} {'YES' if o.accepted else 'no':>4s} "
              f"{o.target_gain:+5d} {o.corpus_gain:+7d}  {o.why[:70]}")
    kept = [o for o in outcomes if o.accepted]
    print(f"\naccepted {len(kept)} of {len(cands)}; "
          f"net corpus {sum(o.corpus_gain for o in kept):+d} compounds"
          + ("" if a.apply else "  (NOT applied — pass --apply)"))

    dest = config.REPO_ROOT / a.json
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"applied": a.apply, "frozen": sorted(frozen),
         "candidates": [{"label": c.label, "target": c.target_patent,
                         "diagnosis": c.diagnosis,
                         "modules": sorted(Path(m).name for m in c.edits)}
                        for c in cands],
         "outcomes": [o.__dict__ for o in outcomes]}, indent=1, default=str))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
