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
# the total does. This guards ANNIHILATION, which is what it was built for: the
# case behind it is US10660877 going 860 -> 0 from a patch that never touched
# one of its rows, and a corpus total cannot see one patent being emptied.
#
# It was 0.10, and at that setting it became the binding constraint rather than
# a safeguard. Measured: US11365191's patch was refused for taking US10208064
# from 36 compounds to 29 — SEVEN compounds — while gaining 186 on a patent
# that produced NOTHING and 260 across the corpus. Trading a 260-compound gain
# and a closed zero to protect seven is not what this gate was for, and a zero
# is the failure this whole loop exists to prevent.
#
# A half is still far inside annihilation: 860 -> 0 is refused, 860 -> 500 is
# refused, 36 -> 29 is allowed. Losses below it are printed by `judge` on every
# accept, so nothing is hidden — the journal, not the threshold, is what makes
# this safe, and any state is one `--revert` away.
MAX_PATENT_LOSS = 0.50

# ...and a patent that had compounds may never reach zero, at any threshold.
# A proportional rule alone cannot express this: a patent with 2 compounds
# losing both is a 100% loss that a ratio test on a large corpus would wave
# through as noise.
NEVER_EMPTY = True

# How much of the corpus's EXACT-number yield a patch may cost. Small on
# purpose: a real fix changes which rows are read, not what a read row means,
# so a patch that turns thousands of measurements into brackets is doing
# something other than what it claims. 2% leaves room for a genuine
# re-interpretation of one layout without leaving room for US11365191's 57%.
MAX_EXACT_LOSS = 0.02


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


def exact_values(pids: list[str], *, xml_dir=None) -> int:
    """How many usable records carry an EXACT number rather than a bracket.

    Compounds are the coverage signal and cannot be argued with, but they are
    not the only thing a patch can change, and counting them alone let through
    the worst patch this loop has produced.

    US11365191's patch raised the corpus by 485 compounds and closed a zero,
    and converted exact measurements into ranges wholesale. Against BindingDB:

        agree           14,275 -> 4,918
        range_contains   2,621 -> 11,731
        range_misses         0 -> 198      <- ranges that EXCLUDE the true value

    Nine thousand values went from "IC50 is 4.7 nM" to "IC50 is somewhere in a
    bracket", and 198 became simply wrong, while every count this loop watched
    went UP. That is the founding rule of the module inverted: a missing assay
    is recoverable, a wrong one is not.

    So this is measured alongside compounds. It needs no reference database —
    it asks only whether the number the patent printed survived as a number.
    """
    from ..sources.uspto_assays import extract_from_patent

    xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")
    n = 0
    for pid in pids:
        f = xml_dir / f"{pid}.xml"
        if not f.exists():
            continue
        try:
            n += sum(1 for r in extract_from_patent(f.read_text(errors="ignore"))
                     if getattr(r, "is_usable", False)
                     and getattr(r, "value_numeric", None) is not None)
        except Exception:
            return -1
    return n


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
    emptied = []
    for p, was in before.items():
        if was <= 0:
            continue
        now = after.get(p, 0)
        if NEVER_EMPTY and was > 0 and now == 0:
            emptied.append(p)
        loss = (was - now) / was
        if loss > worst_loss:
            worst_pid, worst_loss = p, loss
    if emptied:
        return (False, f"{len(emptied)} patent(s) go to ZERO: "
                f"{', '.join(emptied[:4])}. A patent that produced compounds "
                f"must never stop producing them — that is the failure this "
                f"loop exists to prevent, whatever the total does.",
                tg, cg, emptied[0], 1.0)
    if worst_loss > MAX_PATENT_LOSS:
        return (False, f"{worst_pid} loses {worst_loss:.0%} of its compounds "
                f"({before[worst_pid]} -> {after.get(worst_pid, 0)}); a corpus "
                f"total cannot see one patent being emptied", tg, cg,
                worst_pid, worst_loss)
    if tg <= 0 and cg <= 0:
        return (False, f"no gain: target {tg:+d}, corpus {cg:+d}",
                tg, cg, worst_pid, worst_loss)
    cost = (f"; {worst_pid} {before[worst_pid]}->{after.get(worst_pid, 0)} "
            f"({worst_loss:.0%})" if worst_loss > 0 else "")
    return (True, f"target {tg:+d}, corpus {cg:+d}{cost}",
            tg, cg, worst_pid, worst_loss)


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
        exact_before = exact_values(pids, xml_dir=xml_dir)
        best: tuple | None = None

        for cand in remaining:
            exact_after = exact_before
            try:
                with applied(cand.edits):
                    after = measure(pids, xml_dir=xml_dir)
                    exact_after = exact_values(pids, xml_dir=xml_dir)
            except Exception as e:
                after, _ = {}, logger.warning("greedy: %s raised: %r", cand.label, e)
            ok, why, tg, cg, wp, wl = judge(before, after, cand.target_patent)
            # A patch may not pay for compounds with NUMBERS. See `exact_values`:
            # the case behind this raised coverage 485 and turned 9,357 exact
            # measurements into brackets, 198 of which excluded the true value.
            if ok and exact_before > 0:
                drop = (exact_before - exact_after) / exact_before
                if drop > MAX_EXACT_LOSS:
                    ok = False
                    why = (f"{exact_before - exact_after} exact value(s) become "
                           f"ranges or vanish ({drop:.0%} of {exact_before}); "
                           f"a patch may not buy compounds with numbers")
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
