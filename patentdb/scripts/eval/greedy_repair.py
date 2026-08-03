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


def _candidates(pids: list[str] | None, limit: int) -> list[Candidate]:
    """One proposal per open capability gap, as an editable candidate."""
    from ...repair.capability import (
        MAX_TARGETS, PATCHABLE, _attach_oracle, _function_source, collect_gaps,
        propose_capability_patch,
    )
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
        prop = propose_capability_patch(g, table)
        patches = (prop or {}).get("patches") or []
        if not patches:
            logger.warning("greedy: %s — model proposed nothing (%s)",
                           g["patent"], str((prop or {}).get("diagnosis"))[:120])
            continue
        # Splice every target into its module IN MEMORY: two targets can live in
        # one file, so edits must accumulate per module rather than each being
        # applied to the on-disk text.
        edits: dict[str, str] = {}
        names = []
        for patch in patches[:MAX_TARGETS]:
            name = patch.get("target")
            body = (patch.get("function_source") or "").rstrip()
            if name not in PATCHABLE or not body:
                continue
            module = PATCHABLE[name][0]
            src = edits.get(str(module)) or module.read_text()
            old = _function_source(module, name)
            if not old or old not in src:
                continue
            edits[str(module)] = src.replace(old, body)
            names.append(name)
        # The splice must PARSE. One proposal produced
        # `.replace('\u2266', '<=').n    s_norm = ...` — a newline collapsed
        # into the letter `n` — and the candidate reached `measure()`, threw
        # SyntaxError on import, and was scored as a patch that found nothing.
        # A candidate that cannot be imported is not a bad patch, it is not a
        # patch, and it must never cost a measurement round.
        import ast
        broken = []
        for mod, text in edits.items():
            try:
                ast.parse(text)
            except SyntaxError as e:
                broken.append(f"{Path(mod).name}:{e.lineno} {e.msg}")
        if broken:
            logger.warning("greedy: %s — splice does not parse (%s); dropped",
                           g["patent"], "; ".join(broken)[:160])
            edits = {}

        if edits:
            out.append(Candidate(
                label=f"{g['patent']}:{','.join(names)}",
                target_patent=g["patent"], edits=edits,
                diagnosis=str(prop.get("diagnosis", ""))[:400],
                fingerprint=str(g.get("fingerprint")),
                meta={"rows_at_stake": g.get("rows_at_stake")}))
    return out


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", help="comma-separated; default = every open gap")
    ap.add_argument("--limit", type=int, default=8, help="max candidates to buy")
    ap.add_argument("--apply", action="store_true",
                    help="write the winning set to the tree (default: measure only)")
    ap.add_argument("--json", default="docs/reports/greedy_repair.json")
    a = ap.parse_args()

    pids = [p.strip().upper() for p in a.patents.split(",")] if a.patents else None
    frozen = snapshot.frozen_ids()
    print(f"frozen patents (out of scope): {len(frozen)}", file=sys.stderr)

    cands = _candidates(pids, a.limit)
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
