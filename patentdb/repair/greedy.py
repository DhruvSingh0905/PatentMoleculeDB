"""Choose the SET of patches that yields most, one at a time, re-measuring.

The old rule was "a patch is declined if the corpus picks up fewer compounds".
That rule is why nothing lands. It asks each patch to be non-destructive across
103 patents at once, and the functions worth patching — `_is_namelike`,
`_opens_with_id`, `assemble_block` — decide for every block in the corpus. A
patch that fixes US10189840's stereoisomer rows necessarily changes how every
headerless block derives its header. Measured, four patches in a row were
correct about their own patent and declined for what they did elsewhere.

Two changes make the search well-posed.

FROZEN PATENTS ARE NOT RE-DERIVED. `snapshot` pins a patent's answer when it is
processed, so a patch is scored on the patent it targets plus those not yet
frozen — the population it can still affect. Work already finished stops being
a floor that every later patch must clear.

SCORING IS GREEDY AND RE-MEASURED. Candidates are tried one at a time against
the CURRENT tree, best-first, and the tree is kept only when a candidate wins.
Precomputing a score per patch and then taking the top-k would be wrong: patch
effects are not additive — they compose through header promotion, which is
exactly how one `_is_namelike` change cost US10660877 all 860 of its compounds
without touching a row of it.

No Pareto front, no combinatorics. One loop, one sort, re-measure between
accepts. What stops a patch that trades many real compounds for slightly more
elsewhere is not the total but `MAX_PATENT_LOSS`: no single unfrozen patent may
lose more than that fraction of what it had. 860 -> 0 is refused even if the
corpus sum rises, because a total is an average over patents and an average
cannot see one being annihilated.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..core import config
from . import snapshot

logger = logging.getLogger(__name__)

# No unfrozen patent may lose more than this share of its compounds, whatever
# the total does. Set from the failures this exists to stop: the four declined
# patches cost 100%, 17%, 7% and 5% of a patent each. A tenth is generous
# enough for a genuine re-interpretation and nowhere near an annihilation.
MAX_PATENT_LOSS = 0.10


@dataclass
class Candidate:
    """One patch proposal, with everything needed to apply and judge it."""
    label: str
    target_patent: str
    edits: dict                                  # Path -> new source text
    diagnosis: str = ""
    fingerprint: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Outcome:
    label: str
    accepted: bool
    why: str
    target_gain: int = 0
    corpus_gain: int = 0
    worst_patent: str = ""
    worst_loss: float = 0.0


@contextmanager
def applied(edits: dict):
    """Run the body with `edits` spliced into the tree, then put it back.

    Every measurement in this loop is a real extraction against real modules —
    there is no way to evaluate a patch to `_is_namelike` short of importing
    it. So the tree is written, reloaded, measured and restored, and the
    restore is in a `finally` because a candidate that raises must not leave
    the working tree patched.

    Shared with `iterate`, which needs exactly this to let a model observe its
    own patch. Two copies of a write/reload/restore is one copy too many.
    """
    from pathlib import Path

    saved = {Path(p): Path(p).read_text() for p in edits}
    try:
        for path, text in edits.items():
            Path(path).write_text(text)
        _reload()
        yield
    finally:
        for path, text in saved.items():
            path.write_text(text)
        _reload()


def measure(pids: list[str], *, xml_dir=None) -> dict[str, int]:
    """Distinct usable compounds per patent, live — never from a snapshot."""
    from ..sources.uspto_assays import extract_from_patent

    xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")
    out: dict[str, int] = {}
    for pid in pids:
        f = xml_dir / f"{pid}.xml"
        if not f.exists():
            continue
        try:
            recs = extract_from_patent(f.read_text(errors="ignore"))
            out[pid] = len({r.cid for r in recs if r.is_usable and r.cid})
        except Exception:
            out[pid] = -1                        # a crash is a loss, not a skip
    return out


def scorable_patents(target: str, *, xml_dir=None) -> list[str]:
    """The patents a patch can still help or harm: the target, plus unfrozen.

    A frozen patent keeps its answer whatever the code does next, so including
    it here would re-impose the very floor freezing removes.
    """
    xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")
    frozen = snapshot.frozen_ids()
    every = sorted(p.stem for p in xml_dir.glob("*.xml"))
    live = [p for p in every if p.upper() not in frozen]
    if target and target not in live:
        live.append(target)
    return live


def judge(before: dict[str, int], after: dict[str, int],
          target: str) -> tuple[bool, str, int, int, str, float]:
    """Accept? Returns (ok, why, target_gain, corpus_gain, worst_pid, worst_loss)."""
    tg = after.get(target, 0) - before.get(target, 0)
    cg = sum(after.get(p, 0) for p in before) - sum(before.values())

    crashed = [p for p, v in after.items() if v < 0]
    if crashed:
        return (False, f"extraction RAISES on {len(crashed)} patent(s): "
                f"{', '.join(crashed[:4])}", tg, cg, crashed[0], 1.0)

    worst_pid, worst_loss = "", 0.0
    for p, was in before.items():
        if was <= 0:
            continue
        loss = (was - after.get(p, 0)) / was
        if loss > worst_loss:
            worst_pid, worst_loss = p, loss
    if worst_loss > MAX_PATENT_LOSS:
        return (False, f"{worst_pid} loses {worst_loss:.0%} of its compounds "
                f"({before[worst_pid]} -> {after.get(worst_pid, 0)}); a corpus "
                f"total cannot see one patent being emptied", tg, cg,
                worst_pid, worst_loss)
    if tg <= 0 and cg <= 0:
        return (False, f"no gain: target {tg:+d}, corpus {cg:+d}",
                tg, cg, worst_pid, worst_loss)
    return (True, f"target {tg:+d}, corpus {cg:+d}", tg, cg, worst_pid, worst_loss)


def select(candidates: list[Candidate], *, apply: bool = True,
           xml_dir=None) -> list[Outcome]:
    """Greedy forward selection. Applies winners to the tree as it goes.

    Each candidate is tried against the tree AS IT STANDS, so a patch is judged
    on what it adds to the ones already accepted rather than to the tree it was
    written against. That is the whole reason this re-measures instead of
    ranking once — and it means a candidate can win here having lost earlier,
    or the reverse.
    """
    from pathlib import Path

    remaining = list(candidates)
    outcomes: list[Outcome] = []
    round_no = 0
    while remaining:
        round_no += 1
        pids = sorted({p for c in remaining
                       for p in scorable_patents(c.target_patent, xml_dir=xml_dir)})
        before = measure(pids, xml_dir=xml_dir)
        best: tuple | None = None

        for cand in remaining:
            try:
                with applied(cand.edits):
                    after = measure(pids, xml_dir=xml_dir)
            except Exception as e:
                after, _ = {}, logger.warning("greedy: %s raised: %r", cand.label, e)
            ok, why, tg, cg, wp, wl = judge(before, after, cand.target_patent)
            logger.info("greedy: %s -> %s (%s)", cand.label,
                        "accept" if ok else "reject", why)
            if ok and (best is None or (tg, cg) > (best[1], best[2])):
                best = (cand, tg, cg, why)
            if not ok:
                outcomes.append(Outcome(cand.label, False, why, tg, cg, wp, wl))

        if best is None:
            break
        cand, tg, cg, why = best
        if apply:
            saved_tree = {Path(p): Path(p).read_text() for p in cand.edits}
            for path, text in cand.edits.items():
                Path(path).write_text(text)
            _reload()
            # Confirm from COLD before keeping it. In-process measurement has
            # now twice reported a gain a fresh interpreter could not
            # reproduce, and the second time it wrote a module that raised
            # NameError on import for the whole corpus.
            fresh = verify_fresh(sorted(before), xml_dir=xml_dir)
            broke = [p for p, v in fresh.items() if v < 0]
            lost = sum(max(0, before[p] - fresh.get(p, 0)) for p in before)
            if broke or (fresh.get(cand.target_patent, 0) <= before.get(
                    cand.target_patent, 0) and lost):
                for path, text in saved_tree.items():
                    Path(path).write_text(text)
                _reload()
                why = ("REVERTED: a fresh interpreter cannot import it "
                       f"({len(broke)} patent(s) raise)" if broke else
                       "REVERTED: a fresh interpreter does not reproduce the gain")
                outcomes.append(Outcome(cand.label, False, why, tg, cg))
                logger.warning("greedy: %s reverted — %s", cand.label, why)
                remaining = [c for c in remaining if c is not cand]
                continue
        outcomes.append(Outcome(cand.label, True, why, tg, cg))
        logger.warning("greedy: ACCEPTED %s — %s", cand.label, why)
        # Everything rejected this round is retried next round against the new
        # tree: a patch can become viable once another has landed.
        remaining = [c for c in remaining if c is not cand]
        outcomes = [o for o in outcomes if o.accepted or o.label != cand.label]
    return outcomes


_RELOADABLE = ("patentdb.sources.bin_legend", "patentdb.sources.uspto_xml",
               "patentdb.sources.uspto_assays")


def _reload() -> None:
    """PURGE and re-import. `importlib.reload` is not safe for this.

    reload() re-executes the new source INTO THE EXISTING module dict, so any
    name the patch DELETED stays bound from the previous execution. Measured:
    a patch to `_ID_CELL` removed the `_HEADER_POTENCY` constant, reload left
    the old binding in place, extraction ran, the candidate measured +53 and
    was accepted — and a fresh process then raised `NameError` on every patent
    in the corpus with 64 tests failing.

    A measurement that a fresh interpreter cannot reproduce is not a
    measurement. Dropping the modules from `sys.modules` first means the
    re-import builds a new namespace containing only what the patched source
    actually defines, which is what the next process will see.
    """
    import importlib
    import sys
    for name in _RELOADABLE:
        sys.modules.pop(name, None)
    for name in _RELOADABLE:
        importlib.import_module(name)


def verify_fresh(pids: list[str], *, xml_dir=None) -> dict[str, int]:
    """Re-measure in a SEPARATE INTERPRETER. The only fully trustworthy count.

    Even with the purge above, this process has already executed the unpatched
    source once and holds references handed out by earlier imports. A
    subprocess shares none of that, so it is the check that a patch about to be
    kept can actually be imported from cold.
    """
    import json
    import subprocess
    import sys

    xml_dir = str(xml_dir or (config.OUTPUT_DIR / "uspto_xml"))
    code = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "from patentdb.sources.uspto_assays import extract_from_patent\n"
        "out={}\n"
        "for pid in sys.argv[2:]:\n"
        "    f=Path(sys.argv[1])/(pid+'.xml')\n"
        "    if not f.exists(): continue\n"
        "    try:\n"
        "        rs=extract_from_patent(f.read_text(errors='ignore'))\n"
        "        out[pid]=len({r.cid for r in rs if r.is_usable and r.cid})\n"
        "    except Exception:\n"
        "        out[pid]=-1\n"
        "print(json.dumps(out))\n")
    try:
        p = subprocess.run([sys.executable, "-c", code, xml_dir, *pids],
                           capture_output=True, text=True, timeout=1800,
                           cwd=str(config.REPO_ROOT))
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        logger.warning("greedy: fresh-process verification failed: %r", e)
        return {p: -1 for p in pids}
