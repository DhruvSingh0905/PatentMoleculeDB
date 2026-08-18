"""Ask a model for an ASSEMBLY PLAN, never for a molecule.

The `repair/synthesize.py` of the assembly tier. Same contract, same reason.

WHY A PLAN AND NOT THE COMPOUNDS
---------------------------------
Asking "what is compound 451" asks for DATA. The model will answer, fluently,
and a plausible SMILES that parses is exactly what this pipeline's gates exist
to reject — nothing about it is checkable, and it is worth nothing on row 452.
A plan is two small facts about a LAYOUT:

    slot_map      which numbered attachment point the column `Ar` fills
    fragment_cut  the group the recogniser invented where a cut mark was

Both are then applied to every row of the table and judged by what the
assembled molecules weigh against the masses the patent itself printed. One
call buys 632 rows on US9718825, and buys them checkably.

WHAT THE MODEL IS AND IS NOT ASKED
-----------------------------------
It is shown the scaffold with its points numbered, the column headings, a few
real rows, and the masses those rows print. It is NOT shown the answer, and it
is not asked to be confident: `MarkushOutcome` will contradict it against
hundreds of rows, and a wrong plan costs one measurement rather than one wrong
molecule in the artifact.

THE POINT NUMBERS CARRY NO MEANING
-----------------------------------
`markush.number_open_points` labels the scaffold's dummies 1..N in the
recogniser's own atom order. That order is stable for one image and arbitrary
across images, so the mapping genuinely cannot be derived — which is precisely
why it is worth paying for, and why it must be measured rather than trusted.
"""
from __future__ import annotations

import json
import logging

from ..core import config
from ..core.cost_tracker import cost_tracker
from ..sources import markush as MK

logger = logging.getLogger(__name__)

MODEL = config.SYNTH_MODEL
MAX_TOKENS = 1200
# Rows shown. Enough for the model to see what varies and what does not,
# few enough that the call stays cheap on a 426-row table.
SHOW_ROWS = 6
# Fragment reads shown. The invented cut group is the same in every one, so a
# handful establishes the pattern and 600 would only cost tokens.
SHOW_FRAGS = 5

SYSTEM = """You map a patent substituent table onto a scaffold drawing.

You are given a scaffold whose attachment points are numbered [1*], [2*], ...
and a table whose columns supply what varies. Decide which column fills which
numbered point.

Two things matter more than being right:

1. A wrong join produces a legal, clean-looking molecule that nothing
   downstream can detect. If you cannot tell which point a column fills, say
   so in `note` and leave it out of `slot_map`. An incomplete plan is refused
   safely; a confident wrong one is not.

2. Patent drawings mark a cut bond with a wavy line, which no recogniser
   reads. Recognisers instead CAP the stub with an invented group. If the
   fragment reads all share a group that cannot be part of the real
   substituent, give its SMARTS as `fragment_cut` and it will be removed and
   replaced by an attachment point. If they do not, leave it empty.

Your plan will be applied to every row and weighed against the mass the patent
prints for each. You will be shown the result if it disagrees."""

TOOL = {
    "name": "assembly_plan",
    "description": "One plan for one table layout.",
    "input_schema": {
        "type": "object",
        "properties": {
            "slot_map": {
                "type": "object",
                "description": ("Column heading -> attachment point number. "
                                "Omit any heading you are not sure of."),
                "additionalProperties": {"type": "integer"},
            },
            "fragment_cut": {
                "type": "string",
                "description": ("SMARTS of the invented group at the cut bond, "
                                "or an empty string if the fragments need no cut."),
            },
            "note": {"type": "string",
                     "description": "What you were unsure of, if anything."},
        },
        "required": ["slot_map", "fragment_cut", "note"],
    },
}


def _prompt(gap, structures: dict, feedback: str) -> str:
    scaf = MK.number_open_points(structures.get(gap.scaffold_ref, ""))
    lines = [f"SCAFFOLD (shared by every row of {gap.table_id}):",
             f"    {scaf}", "",
             f"SLOT COLUMNS: {', '.join(gap.headings)}", ""]

    lines.append("ROWS (cid | slots | drawn group | mass the patent prints):")
    for r in gap.rows[:SHOW_ROWS]:
        slots = "  ".join(f"{k}={v!r}" for k, v in r.slots.items())
        mass = gap.printed_mass.get(r.cid)
        lines.append(f"    {r.cid}: {slots}"
                     + (f"  drawing={r.fragment_ref}" if r.fragment_ref else "")
                     + (f"  m/z {mass}" if mass else ""))

    frags = [(r.fragment_ref, structures.get(r.fragment_ref, ""))
             for r in gap.rows if r.fragment_ref
             and structures.get(r.fragment_ref)][:SHOW_FRAGS]
    if frags:
        lines += ["", "HOW THE DRAWN GROUPS WERE READ:"]
        lines += [f"    {i}: {s}" for i, s in frags]

    named = {k: v for k, v in list(gap.names.items())[:6]}
    if named:
        lines += ["", "SLOT TEXT ALREADY RESOLVED (by name, not by you):"]
        lines += [f"    {k!r} -> {v}" for k, v in named.items()]

    if feedback:
        lines += ["", "YOUR LAST PLAN WAS APPLIED AND MEASURED:",
                  f"    {feedback}",
                  "Propose a different one, or leave slot_map empty to stop."]
    return "\n".join(lines)


def propose(gap, structures: dict, feedback: str = "", *, model: str = MODEL):
    """One model call -> a `Plan`, or None. Never raises into the loop."""
    from .markush_loop import Plan

    if not config.ANTHROPIC_API_KEY:
        logger.info("markush_synth: no API key; cannot propose a plan")
        return None
    # `PER_PATENT_LM_CAP` is read straight off the tracker's own ledger.
    # `config.py:178` and `cost_tracker.py:215` both refer to a
    # `patent_lm_exceeded()` guard; no such function exists, in this package or
    # anywhere else. Two comments describing a check nobody wrote.
    if cost_tracker.per_patent.get(gap.patent_id, 0.0) >= config.PER_PATENT_LM_CAP:
        logger.info("markush_synth: %s at PER_PATENT_LM_CAP ($%.2f)",
                    gap.patent_id, cost_tracker.per_patent[gap.patent_id])
        return None

    from ..core.api_client import resilient_client
    client = resilient_client()
    try:
        resp = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, temperature=0,
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[TOOL], tool_choice={"type": "any"},
            messages=[{"role": "user",
                       "content": _prompt(gap, structures, feedback)}])
    except Exception as e:
        logger.warning("markush_synth: call failed for %s %s: %r",
                       gap.patent_id, gap.table_id, e)
        return None

    u = resp.usage
    # Cache buckets are billed separately — reads at 0.1x, writes at 1.25x.
    # Recording only `input_tokens` makes cached tokens free in our accounting
    # while costing real money, and a spend cap cannot trip on what it never sees.
    cost_tracker.record(
        u.input_tokens, u.output_tokens, model, patent_id=gap.patent_id,
        cost_category="lm",
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)

    call = next((b for b in resp.content
                 if getattr(b, "type", "") == "tool_use"), None)
    if call is None:
        return None
    data = call.input if isinstance(call.input, dict) else json.loads(call.input)
    plan = Plan(slot_map={str(k): int(v)
                          for k, v in (data.get("slot_map") or {}).items()},
                fragment_cut=(data.get("fragment_cut") or "").strip(),
                source="model", note=(data.get("note") or "")[:200])
    err = ground(plan, gap, structures)
    if err:
        logger.info("markush_synth: %s %s proposal rejected — %s",
                    gap.patent_id, gap.table_id, err)
        plan.note = f"{plan.note} [rejected: {err}]"[:200]
        return None
    logger.info("markush_synth: %s %s -> slot_map=%s cut=%r",
                gap.patent_id, gap.table_id, plan.slot_map, plan.fragment_cut)
    return plan


def ground(plan, gap, structures: dict) -> str:
    """Cheap checks BEFORE the plan is run. `""` when it is worth running.

    Not a judgement of whether the plan is right — that is the measurement's
    job and nothing here can anticipate it. These only catch a plan that
    cannot mean anything: a point number the scaffold does not have, a heading
    the table does not have, a SMARTS that will not compile.
    """
    from rdkit import Chem

    scaf = MK.number_open_points(structures.get(gap.scaffold_ref, ""))
    m = Chem.MolFromSmiles(scaf) if scaf else None
    if m is None:
        return "scaffold did not parse"
    points = {a.GetIsotope() for a in m.GetAtoms() if a.GetAtomicNum() == 0}
    for head, n in plan.slot_map.items():
        if head not in gap.headings:
            return f"heading {head!r} is not a column of this table"
        if n not in points:
            return (f"{head} -> point {n}, but the scaffold has "
                    f"{sorted(points)}")
    if len(set(plan.slot_map.values())) != len(plan.slot_map):
        return "two headings claim the same attachment point"
    if plan.fragment_cut and Chem.MolFromSmarts(plan.fragment_cut) is None:
        return f"fragment_cut is not valid SMARTS: {plan.fragment_cut!r}"
    return ""
