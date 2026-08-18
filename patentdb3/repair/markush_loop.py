"""Assemble a substituent table: try free first, measure, buy a plan only if needed.

    gaps -> deterministic plan -> apply -> MEASURE -> keep
                                              |
                                          negative -> buy a plan -> apply ->
                                          MEASURE -> keep, or feed the
                                          measurement back (x MAX_ATTEMPTS)

THE DETERMINISTIC PLAN RUNS FIRST AND IS MEASURED LIKE ANY OTHER
-----------------------------------------------------------------
Most of a substituent table needs no model at all. The heading `R2` states its
own attachment point, because a recogniser writes `R2` in a drawing as the
dummy `[2*]`; OPSIN resolves the substituent names; `markush`'s condensed
formula grammar reads `CH3` and `NH2` under a written-hydrogen gate. What is
left over is one genuinely open question — a heading like `Ar` or `RL` that
names a position the drawing marks some other way — and that is the only thing
worth paying for.

So attempt zero costs nothing and is scored by exactly the gate a paid attempt
is scored by. A tier that calls a model before trying the free answer spends
its budget proving it did not need to.

WHAT IS BOUGHT IS A PLAN, NOT A MOLECULE
-----------------------------------------
A molecule costs one call per row and is worth nothing on the next table. A
plan — which column fills which attachment point — costs one call per LAYOUT
and is reused wherever that layout appears, which is the same economics that
made `repair/gap.py` ask for a rule instead of data.

STRUCTURES COME IN, THEY ARE NOT FETCHED HERE
----------------------------------------------
`structures` maps a `<chemistry>` id to SMILES and is supplied by the caller
from the image pipeline. This module does no image work: recognition runs on a
GPU, in batches, on its own schedule, and a repair loop that blocked on it
could not be tested without one. A gap whose scaffold is not in `structures`
is reported as blocked, by name, rather than assembled from a guess.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from ..core import config
from ..sources import markush as MK
from .markush_gap import MarkushGap, find_gaps
from .markush_outcome import MarkushOutcome, measure, summarise

logger = logging.getLogger(__name__)

# Paid attempts at ONE layout before it is recorded as an escalation. Same
# number and same reason as `loop.MAX_ATTEMPTS`: a model that has been shown
# its own measurement twice and still misses is not going to find it on the
# fourth try, and the layout is cheaper to look at by hand.
MAX_ATTEMPTS = 3

BLOCK_NO_SCAFFOLD = "scaffold_drawing_not_recognised"
BLOCK_NO_FRAGMENT = "fragment_drawing_not_recognised"
BLOCK_NO_REFEREE = "nothing_in_the_document_can_check_this"


@dataclass
class Plan:
    """Which column fills which attachment point on the scaffold."""
    slot_map: dict = field(default_factory=dict)     # heading -> isotope
    source: str = "deterministic"                    # or "model"
    note: str = ""


@dataclass
class TableReport:
    patent_id: str
    table_id: str
    fingerprint: str
    n_rows: int
    adopted: bool = False
    plan: Plan | None = None
    outcome: MarkushOutcome | None = None
    blocked: str = ""
    structures: dict = field(default_factory=dict)   # cid -> smiles, if adopted
    attempts: int = 0
    # Why each row that produced no molecule produced none, by cause. A table
    # that builds nothing must say what stopped it, or the summary describes
    # some other problem — see `apply_plan`.
    refusals: dict = field(default_factory=dict)


def deterministic_plan(gap: MarkushGap) -> Plan:
    """The plan the headings already state. `slot_map` covers only what is certain.

    A heading carrying a number needs no entry — `markush.build_text_group`
    reads it from the heading itself. This exists to say, explicitly, that a
    heading WITHOUT one is not being guessed at.
    """
    unnumbered = [h for h in gap.headings if not MK._SLOT_NUMBER.match(h)]
    return Plan(slot_map={}, source="deterministic",
                note=("every slot heading carries its own number"
                      if not unnumbered
                      else f"no attachment point stated for: {', '.join(unnumbered)}"))


def apply_plan(gap: MarkushGap, plan: Plan,
               structures: dict) -> tuple[dict, Counter]:
    """Run one plan over one table. `({cid -> smiles}, {reason -> count})`.

    THE REFUSALS ARE RETURNED, NOT LOGGED. They were a `logger.debug` line,
    and the result was a report reading `built 0; refused 0; NOTHING COULD
    CHECK THIS TABLE` for a 426-row table that prints 426 masses. Every one of
    those rows was refused for one nameable reason — the fragment drawing
    carried no attachment point — and the report said the table was
    uncheckable instead. That is the silent-block shape: a whole table
    yielding nothing while the summary describes a different problem.
    """
    why: Counter = Counter()
    scaf = structures.get(gap.scaffold_ref, "")
    if not scaf:
        why[BLOCK_NO_SCAFFOLD] = len(gap.rows)
        return {}, why
    out: dict[str, str] = {}
    for row in gap.rows:
        frag = structures.get(row.fragment_ref, "") if row.fragment_ref else ""
        if row.fragment_ref and not frag:
            why[BLOCK_NO_FRAGMENT] += 1
            continue
        if row.route == MK.ROUTE_IMAGE_ONLY and not row.varying:
            smi, err = MK.build_image_only(scaf, frag)
        else:
            smi, err = MK.build_text_group(scaf, row, frag,
                                           names=gap.names,
                                           slot_map=plan.slot_map or None)
        if smi:
            out[row.cid] = smi
        else:
            # Keep the SHAPE of the reason, not the row's own text: 426 rows
            # refused for one cause must read as one cause with a count, or a
            # reader sees 426 different problems.
            why[(err or "refused").split(":")[0].strip()[:60]] += 1
    return out, why


def repair_table(gap: MarkushGap, structures: dict, *,
                 propose=None) -> TableReport:
    """One table, start to finish. `propose(gap, feedback)` may return a Plan."""
    rep = TableReport(patent_id=gap.patent_id, table_id=gap.table_id,
                      fingerprint=gap.fingerprint, n_rows=gap.n_rows)
    if gap.scaffold_ref not in structures:
        rep.blocked = BLOCK_NO_SCAFFOLD
        return rep
    if not gap.referees:
        # Refused BEFORE any work: an assembly nothing can check is not a
        # result, and building it would only make a wrong answer look finished.
        rep.blocked = BLOCK_NO_REFEREE
        return rep

    plan = deterministic_plan(gap)
    feedback = ""
    for attempt in range(MAX_ATTEMPTS + 1):
        rep.attempts = attempt
        built, why = apply_plan(gap, plan, structures)
        oc = measure(gap, built) if built else MarkushOutcome()
        rep.plan, rep.outcome, rep.refusals = plan, oc, why
        if oc.positive:
            rep.adopted, rep.structures = True, built
            logger.info("markush %s %s: ADOPTED (%s) via %s",
                        gap.patent_id, gap.table_id, summarise(oc), plan.source)
            return rep
        feedback = summarise(oc) if built else _why(why, gap.n_rows)
        logger.info("markush %s %s attempt %d: %s",
                    gap.patent_id, gap.table_id, attempt, feedback)
        if propose is None or attempt >= MAX_ATTEMPTS:
            break
        nxt = propose(gap, feedback)
        if nxt is None:
            break
        plan = nxt
    return rep


def repair_patent(patent_id: str, xml: str, structures: dict | None = None,
                  *, propose=None) -> list[TableReport]:
    """Every substituent table in one patent.

    `structures` defaults to whatever `recognise` already has for this patent —
    `{}` when the backend is off, which is the default. So this is safe to call
    unconditionally: with no GPU anywhere it reports every table blocked and
    costs nothing.
    """
    if not config.MARKUSH_ASSEMBLY:
        return []
    if structures is None:
        from ..recognise import structures as _recognised
        structures = _recognised(patent_id)
    return [repair_table(g, structures, propose=propose)
            for g in find_gaps(patent_id, xml)]


def _why(why: dict, n_rows: int) -> str:
    """The refusal causes, biggest first. Never "nothing happened"."""
    if not why:
        return f"no molecule from {n_rows} row(s), and no reason recorded"
    top = sorted(why.items(), key=lambda kv: -kv[1])
    return "; ".join(f"{v} x {k}" for k, v in top[:3])


def summarise_patent(reports: list[TableReport]) -> str:
    if not reports:
        return "no substituent tables"
    ok = [r for r in reports if r.adopted]
    blocked: dict[str, int] = {}
    for r in reports:
        if r.blocked:
            blocked[r.blocked] = blocked.get(r.blocked, 0) + 1
    bits = [f"{len(ok)}/{len(reports)} tables adopted",
            f"{sum(len(r.structures) for r in ok)} molecules"]
    for k, v in sorted(blocked.items()):
        bits.append(f"{k}={v}")
    return "; ".join(bits)
