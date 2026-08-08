"""Run the self-healing loop three times — Haiku, Sonnet, Opus — and compare.

    python3 -m patentdb.scripts.eval.model_bakeoff --patents US11649247,US9302989
    python3 -m patentdb.scripts.eval.model_bakeoff --patents US9302989 --dry-run
    python3 -m patentdb.scripts.eval.model_bakeoff --models haiku,opus --patch

Why this exists: acceptance is now ONE condition — does the patched code pick
up more compounds than before. A fixed rule about what a good repair looks like
is the wrong premise for an extractor whose job is adapting to layouts nobody
anticipated, and every such rule here eventually blocked something correct.

Removing the rules does not remove the judgement; it moves it to us. Choosing
the model IS the remaining control, so it has to be measured rather than
assumed. Haiku diagnosed a two-assay column correctly and then wrote code
calling a helper that does not exist; Sonnet fixed it in two functions; Opus
wrote a correct patch the harness wrongly declined. One anecdote each is not a
basis for a default.

EVERY MODEL STARTS FROM THE SAME STATE. The code tier rewrites live source
files and the rule tier writes a shared library, so a second model would
otherwise inherit the first one's repairs and look better or worse for reasons
that have nothing to do with it. `_Snapshot` captures every patchable module,
the rule library and the journal before the first run and restores them before
each subsequent one — and again at the end, so the tree is exactly as it was.

What is logged, per model: every rule proposed and its kind, what was adopted,
rejected, escalated or crashed, every code patch with its target functions and
diagnosis, the compounds gained, and the BindingDB value agreement — recorded,
never enforced. Cost is taken from the tracker, not estimated.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ...core import config

MODELS = {
    "haiku": config.MODEL_HAIKU,
    "sonnet": config.MODEL_SONNET,
    "opus": config.MODEL_OPUS,
}


class _Snapshot:
    """Everything a repair run can mutate, saved and restorable.

    Not git: the working tree is routinely dirty here, and a bake-off must not
    depend on the user having committed. These are the only paths the loop
    writes — the patchable modules, the learned rules, and the patch journal.
    """

    def __init__(self) -> None:
        from ...repair.capability import PATCHABLE
        from ...repair.parser_repair import _journal_path

        self.paths: list[Path] = sorted(
            {mod for mod, _ in PATCHABLE.values()}
            | {config.PACKAGE_ROOT / "sources" / "uspto_xml.py",
               config.PACKAGE_ROOT / "data" / "layout_rules.json",
               Path(_journal_path())})
        self.dir = Path(tempfile.mkdtemp(prefix="bakeoff-snapshot-"))
        self.saved: dict[Path, Path] = {}
        for p in self.paths:
            if p.exists():
                dst = self.dir / p.name
                shutil.copy2(p, dst)
                self.saved[p] = dst

    def restore(self) -> list[str]:
        changed = []
        for p, dst in self.saved.items():
            if not p.exists() or p.read_bytes() != dst.read_bytes():
                shutil.copy2(dst, p)
                changed.append(p.name)
        # A file the run CREATED (a first journal) is not in `saved`.
        for p in self.paths:
            if p.exists() and p not in self.saved:
                p.unlink()
                changed.append(f"{p.name} (removed)")
        return changed

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def _reload_pipeline() -> None:
    """Re-import the modules a patch may have rewritten.

    A code patch edits a source file on disk, but this process already holds
    the old module object — so without this, run two would measure run one's
    code. Order matters: the sources first, then everything that imported them.
    """
    import importlib

    for name in ("patentdb.sources.bin_legend", "patentdb.sources.uspto_xml",
                 "patentdb.sources.uspto_assays", "patentdb.repair.gap",
                 "patentdb.repair.rules", "patentdb.repair.loop",
                 "patentdb.repair.value_check", "patentdb.repair.capability"):
        mod = sys.modules.get(name)
        if mod is not None:
            importlib.reload(mod)


def _forget(pids: list[str]) -> int:
    """Drop the library's answers for these patents' layouts.

    Without this the bake-off is VACUOUS, and silently so. Rules are keyed by
    layout fingerprint and persist forever by design — that is the whole cost
    advantage over HARVEST — so a corpus that has already been through the loop
    has an answer for every layout the test patents contain. All three models
    are then asked nothing, every run reports `asked 0, adopted N`, and the
    table compares three identical replays of the FIRST model's work.

    Measured on batch 2: 0 asked, 6 adopted from cache. A harness whose null
    result looks exactly like a real result is the failure mode this repo keeps
    finding, so the forgetting is explicit rather than left to the operator.

    `_Snapshot` still holds the original library and restores it between models
    and at the end, so this is scoped to the run.
    """
    from ...repair.gap import gaps_for_patent
    from ...repair.rules import RuleLibrary

    lib = RuleLibrary()
    fingerprints: set[str] = set()
    for pid in pids:
        f = config.OUTPUT_DIR / "uspto_xml" / f"{pid}.xml"
        if not f.exists():
            continue
        try:
            for g in gaps_for_patent(pid, f.read_text(errors="ignore")):
                fingerprints.add(g.fingerprint)
        except Exception as e:
            print(f"  (could not enumerate gaps for {pid}: {e!r})"[:120])
    dropped = sum(1 for fp in fingerprints if lib._rules.pop(fp, None) is not None)
    if dropped:
        lib.save()
    return dropped


def _compounds(pids: list[str]) -> dict[str, int]:
    """Distinct compounds per patent — the one acceptance metric."""
    from ...sources.uspto_assays import extract_from_patent

    out = {}
    for pid in pids:
        f = config.OUTPUT_DIR / "uspto_xml" / f"{pid}.xml"
        if not f.exists():
            continue
        try:
            recs = extract_from_patent(f.read_text(errors="ignore"))
            out[pid] = len({r.cid for r in recs if r.is_usable and r.cid})
        except Exception:
            out[pid] = -1
    return out


def _values(pids: list[str]) -> dict:
    """BindingDB agreement. RECORDED, never a gate — see the module docstring."""
    from ...repair.value_check import check_corpus

    try:
        r = check_corpus(pids)
        return {"total": r["total"], "bad": r["bad_patents"]}
    except Exception as e:
        return {"error": repr(e)[:120]}


def run_one(label: str, model: str, pids: list[str], *, patch: bool,
            dry_run: bool, forget: bool = False) -> dict:
    """One model, from a clean tree. Returns everything it did."""
    from ...core.cost_tracker import cost_tracker
    from ...repair.loop import repair_patent

    started = time.time()
    before = _compounds(pids)
    # `total_cost`, not `total_usd`. The tracker has never had a `total_usd`
    # attribute, so `getattr(..., "total_usd", 0.0)` returned the DEFAULT on
    # every run and this column reported $0.0000 whatever happened — a meter
    # that reads zero when it is working and zero when it is not.
    spend0 = cost_tracker.total_cost
    calls0 = cost_tracker.call_count
    forgotten = _forget(pids) if forget else 0
    if forget:
        print(f"  forgot {forgotten} cached rule(s) for these layouts")

    per_patent = []
    for pid in pids:
        f = config.OUTPUT_DIR / "uspto_xml" / f"{pid}.xml"
        if not f.exists():
            continue
        xml = f.read_text(errors="ignore")
        try:
            _, rep = repair_patent(pid, xml, max_calls=4, dry_run=dry_run,
                                   model=model)
        except Exception as e:                      # should not happen; recorded if it does
            per_patent.append({"patent": pid, "uncaught": repr(e)[:160]})
            continue
        per_patent.append({
            "patent": pid, "gaps": rep.gaps_found, "known": rep.already_known,
            "asked": rep.proposed, "adopted": rep.adopted,
            "rejected": rep.rejected, "escalated": rep.escalated,
            "over_objection": rep.adopted_over_objection,
            "capability_gaps": len(rep.capability_gaps),
            "crashed": rep.crashed, "rows": rep.rows_recovered,
            "escalations": [{"capability": e.get("capability"),
                             "note": str(e.get("note"))[:220]}
                            for e in rep.escalations],
            "rejections": [{"kind": r.get("proposed"),
                            "why": str(r.get("why_rejected"))[:220],
                            "contract": r.get("contract", False)}
                           for r in rep.rejections],
        })

    patches = []
    if patch and not dry_run:
        from ...repair.capability import repair_capabilities
        try:
            # `--patch` is the operator asking; the bakeoff restores the tree
            # between rungs itself. See repair/guard.
            rep = repair_capabilities(patent_ids=pids, model=model, apply=True)
            for r in rep["results"]:
                patches.append({
                    "fingerprint": r.get("fingerprint"), "target": r.get("target"),
                    "applied": bool(r.get("ok")), "why": str(r.get("why"))[:200],
                    "diagnosis": str(r.get("diagnosis"))[:400],
                    "recovered": r.get("gap_rows_recovered"),
                    "objections": r.get("objections") or [],
                })
        except Exception as e:
            patches.append({"uncaught": repr(e)[:200]})
        _reload_pipeline()

    after = _compounds(pids)
    from ...repair.rules import RuleLibrary
    lib = RuleLibrary()
    # A $0 here is ambiguous unless we also say whether the API was reached:
    # answers are cached per (fingerprint, model) and persist across sessions,
    # so a genuine $0 (every question already answered) looks exactly like a
    # broken meter. `api_calls` is what separates them.
    api_calls = cost_tracker.call_count - calls0
    return {
        "model": label, "model_id": model, "seconds": round(time.time() - started, 1),
        "cost_usd": round(cost_tracker.total_cost - spend0, 4),
        "api_calls": api_calls,
        "rules_forgotten": forgotten,
        "compounds_before": before, "compounds_after": after,
        "compounds_gained": {p: after.get(p, 0) - before.get(p, 0)
                             for p in before if after.get(p, 0) != before.get(p, 0)},
        "per_patent": per_patent,
        "patches": patches,
        "library_by_kind": dict(collections.Counter(
            r.kind for r in lib._rules.values())),
        "values": _values(pids),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", required=True, help="comma-separated test patents")
    ap.add_argument("--models", default="haiku,sonnet,opus")
    ap.add_argument("--patch", action="store_true",
                    help="also run the code-patch tier (writes source files)")
    ap.add_argument("--dry-run", action="store_true",
                    help="count what WOULD be asked; no API calls, no writes")
    ap.add_argument("--forget", action="store_true",
                    help="drop cached rules for these layouts before each model, "
                         "so every model is asked the same questions (without "
                         "this a settled corpus asks nothing and the comparison "
                         "measures nothing)")
    ap.add_argument("--out", default="docs/reports/model_bakeoff.json")
    a = ap.parse_args()

    pids = a.patents.split(",")
    labels = [m.strip() for m in a.models.split(",")]
    unknown = [m for m in labels if m not in MODELS]
    if unknown:
        print(f"unknown model(s): {unknown}; choose from {list(MODELS)}", file=sys.stderr)
        return 2

    snap = _Snapshot()
    print(f"snapshot: {len(snap.saved)} file(s) — "
          f"{', '.join(p.name for p in snap.saved)}\n")
    runs = []
    try:
        for i, label in enumerate(labels):
            if i:
                changed = snap.restore()
                _reload_pipeline()
                print(f"  reverted before {label}: "
                      f"{', '.join(changed) if changed else 'nothing had changed'}")
            print(f"\n=== {label} ({MODELS[label]}) ===")
            r = run_one(label, MODELS[label], pids, patch=a.patch,
                        dry_run=a.dry_run, forget=a.forget)
            runs.append(r)
            asked = sum(p.get("asked", 0) for p in r["per_patent"])
            print(f"  asked {asked}, gained {sum(r['compounds_gained'].values())} "
                  f"compounds, {len(r['patches'])} patch(es), ${r['cost_usd']}")
    finally:
        changed = snap.restore()
        snap.cleanup()
        _reload_pipeline()
        print(f"\nrestored: {', '.join(changed) if changed else 'nothing had changed'}")

    dest = config.REPO_ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"patents": pids, "runs": runs}, indent=1,
                               default=str))
    print(f"\nwrote {dest}\n")

    print(f"{'model':8s} {'asked':>6s} {'adopt':>6s} {'rej':>5s} {'esc':>5s} "
          f"{'crash':>6s} {'patch':>6s} {'cmpd+':>6s} {'api':>5s} {'$':>8s}")
    for r in runs:
        pp = r["per_patent"]
        print(f"{r['model']:8s} {sum(p.get('asked',0) for p in pp):6d} "
              f"{sum(p.get('adopted',0) for p in pp):6d} "
              f"{sum(p.get('rejected',0) for p in pp):5d} "
              f"{sum(p.get('escalated',0) for p in pp):5d} "
              f"{sum(len(p.get('crashed',[])) for p in pp):6d} "
              f"{sum(1 for x in r['patches'] if x.get('applied')):6d} "
              f"{sum(r['compounds_gained'].values()):6d} "
              f"{r.get('api_calls', 0):5d} "
              f"{r['cost_usd']:8.4f}")
    if all(r.get("api_calls", 0) == 0 for r in runs):
        print("\n  api=0 everywhere: every answer replayed from the "
              "(fingerprint, model) cache. The comparison is real and it cost "
              "nothing; the $ column is not evidence the meter works.")
    print("\nWhat each model chose to do — the part a table cannot show:")
    for r in runs:
        print(f"\n  {r['model']}:")
        for p in r["patches"]:
            if p.get("diagnosis"):
                print(f"    patch {p.get('target')}: {p['diagnosis'][:190]}")
        for pp in r["per_patent"]:
            for e in pp.get("escalations", [])[:2]:
                print(f"    escalated [{pp['patent']}] {e.get('capability')}: "
                      f"{e.get('note','')[:150]}")
            for x in pp.get("rejections", [])[:2]:
                print(f"    rejected  [{pp['patent']}] {x.get('kind')}"
                      f"{' (contract)' if x.get('contract') else ''}: {x.get('why','')[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
