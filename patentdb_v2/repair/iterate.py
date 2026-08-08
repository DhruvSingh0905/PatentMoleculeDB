"""Write a patch, RUN it, look at what it did, write the next one.

Every capability patch until now was one-shot: one prompt, one answer, scored
once by `greedy` and discarded. The model never found out what its own code
did. That is not a small omission — it is the difference between a program that
can debug and one that can only guess, and the failure it produces is specific
and repeatable:

    US10266548  the model correctly identified BOTH halves of the defect and
                named `merge_header` and `_unit_from` as its targets. Its
                `merge_header` rewrite is genuinely correct — it strips the
                scientific-notation values that had been concatenated into the
                header, turning `median IC50 [mol/l] 2.6e-009 1.2e-007` back
                into `median IC50 [mol/l]`. Its `_unit_from` rewrite is
                byte-identical to the original. Cleaning the header is
                necessary and not sufficient: the unit vocabulary still has no
                `mol/l`, every record stays unusable, and the patent stays at 0.

One more turn is all that shape needs. `_where_it_stops` recomputed after the
patch says, in the same words it said before, that the assay columns still
carry no unit — and the model that wrote half the fix is the one best placed to
write the other half. It cannot do that without being told, and there was no
channel to tell it.

WHAT IS FED BACK IS MEASURED, NEVER NARRATED. Huang et al. (2310.01798) is
direct about this: LLMs asked to self-correct from their own review of their
own work degrade. Self-correction works when the feedback is external and
factual. So every line of the observation here is a number this process
computed by importing the patched code and extracting a real patent:

  * compounds and usable records on the target, before and after;
  * the SAME `_where_it_stops` diagnostic, recomputed against a block
    re-assembled by the patched reader — because a patch to `assemble_block`
    changes what the block even is, and diffing against the old object would
    describe a table that no longer exists;
  * which edits were no-ops, byte-compared;
  * an import or syntax failure, verbatim;
  * what it cost the other patents it can still affect.

The loop stops the moment the target gains. It is not asked to keep improving a
patch that worked.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config
from . import greedy

logger = logging.getLogger(__name__)

# Three rounds, not more. The first is a diagnosis from the table alone; the
# second is the first real edit-observe cycle and is where the two-part fixes
# land; a third exists for the case where round two broke something and the
# model needs to see that. Beyond that the observations stop changing — a
# model that has been told twice that the unit is still missing and has not
# fixed it is not going to on the fourth telling, and each round is a full
# extraction pass over every unfrozen patent.
MAX_ROUNDS = int(__import__("os").environ.get("CAPABILITY_ROUNDS", "3"))


@dataclass
class Attempt:
    """One round: what the model wrote, and what running it actually did."""
    round_no: int
    names: list[str]
    edits: dict
    diagnosis: str = ""
    noop: list[str] = field(default_factory=list)
    broken: str = ""
    target_before: int = 0
    target_after: int = 0
    usable_after: int = 0
    records_after: int = 0
    stops: str = ""
    worst_pid: str = ""
    worst_loss: float = 0.0

    @property
    def gained(self) -> bool:
        return self.target_after > self.target_before


# Escapes a model writes for characters it could have typed. Inside a string
# literal `±` and `±` are the same thing; inside a COMMENT the escape is
# six literal characters and the sentence is corrupted.
# The characters a model escapes instead of typing. Built at RUNTIME from
# the codepoints, never written as literal pairs: an editor that helpfully
# turns "\\u00b1" into the character collapses such a table into an identity
# map that silently does nothing, which is exactly what happened here on the
# first attempt at this function.
_ESCAPED = "\u00b1\u2014\u2013\u2248\u2265\u2264\u00b5\u03bc\u2212" \
           "\u2266\u2267\u2a7e\u2a7d\u00b0\u00d7\u2032\u2033"
_ESCAPES = {"\\u%04x" % ord(c): c for c in _ESCAPED}


def deescape_comments(src: str) -> str:
    """Turn `\\uXXXX` back into the character it names — COMMENT LINES ONLY.

    PATCH_SYSTEM has forbidden this in words since the second occurrence and
    it has now happened a third time, on `'3.0 \\u00b1 1.0*'` in a comment the
    model wrote to explain the very footnote-marker fix it was making. An
    instruction that has failed three times is not a control.

    Scoped to comments on purpose. In a string literal the two forms are
    identical to the interpreter, and rewriting a regex nobody asked us to
    touch is a behaviour change smuggled in as tidying; in a comment the
    escape is not an escape at all, just six characters where a symbol should
    be, and no test can ever see it.
    """
    out = []
    for line in src.split("\n"):
        if line.lstrip().startswith("#"):
            for esc, ch in _ESCAPES.items():
                line = line.replace(esc, ch)
        out.append(line)
    return "\n".join(out)


def splice(patches: list, *, max_targets: int | None = None) -> tuple[dict, list, list, str]:
    """Patches -> (edits per module, names, no-op names, parse error).

    Two targets can live in one file, so edits accumulate per module rather
    than each being applied to the on-disk text.

    A no-op is reported rather than silently dropped. `_unit_from` came back
    byte-identical while being listed as a target, and with no channel to say
    so the candidate was measured, scored `+0`, and read as a model that had
    tried and failed — when in fact it had not edited the function at all.
    """
    from .capability import MAX_TARGETS, _function_source, all_targets

    max_targets = MAX_TARGETS if max_targets is None else max_targets
    targets = all_targets()
    edits: dict[str, str] = {}
    names: list[str] = []
    noop: list[str] = []
    for patch in (patches or [])[:max_targets]:
        name = patch.get("target")
        body = deescape_comments((patch.get("function_source") or "").rstrip())
        if name not in targets or not body:
            continue
        module = targets[name][0]
        src = edits.get(str(module)) or module.read_text()
        old = _function_source(module, name)
        if not old or old not in src:
            continue
        if old.rstrip() == body:
            noop.append(name)
            continue
        edits[str(module)] = src.replace(old, body)
        names.append(name)

    # The splice must PARSE. One proposal produced `.replace('≦', '<=').n
    # s_norm = ...` — a newline collapsed into the letter `n` — and the
    # candidate reached measurement, threw SyntaxError on import, and was
    # scored as a patch that found nothing. A candidate that cannot be imported
    # is not a bad patch; it is not a patch.
    broken = []
    for mod, text in edits.items():
        try:
            ast.parse(text)
        except SyntaxError as e:
            broken.append(f"{Path(mod).name} line {e.lineno}: {e.msg}")
    if broken:
        return {}, names, noop, "; ".join(broken)
    return edits, names, noop, ""


def _live(patent_id: str, xml: str) -> tuple[int, int, int]:
    """(distinct usable compounds, usable records, records) under current code."""
    from ..sources.uspto_assays import extract_from_patent

    recs = list(extract_from_patent(xml))
    usable = [r for r in recs if r.is_usable]
    return len({r.cid for r in usable if r.cid}), len(usable), len(recs)


def _stops_now(xml: str, table_id: str) -> str:
    """`_where_it_stops` against a block RE-ASSEMBLED by the patched reader.

    Re-assembly is the point. A patch to `assemble_block` or `_is_namelike`
    changes which rows are header and which are body, so the pre-patch table
    object describes a layout that no longer exists — reporting against it
    would tell the model its patch did nothing when it may have done a great
    deal.
    """
    from ..sources.uspto_xml import assemble_blocks, parse_tables
    from .capability import _where_it_stops

    try:
        block = {t.table_id: t for t in assemble_blocks(parse_tables(xml))}.get(table_id)
        return _where_it_stops(block) if block is not None else (
            f"\n!! block {table_id} no longer exists after your patch — "
            f"`assemble_block` did not produce it.\n")
    except Exception as e:
        return f"\n!! recomputing the diagnostic RAISED: {e!r}\n"


def observe(attempt: Attempt, patent_id: str, table_id: str, xml: str,
            *, xml_dir=None) -> Attempt:
    """Run the patch for real and fill in what it did. No judgement, only facts."""
    attempt.target_before, _, _ = _live(patent_id, xml)
    if attempt.broken or not attempt.edits:
        return attempt

    pids = greedy.scorable_patents(patent_id, xml_dir=xml_dir)
    before = greedy.measure(pids, xml_dir=xml_dir)
    try:
        with greedy.applied(attempt.edits):
            after = greedy.measure(pids, xml_dir=xml_dir)
            attempt.target_after, attempt.usable_after, attempt.records_after = \
                _live(patent_id, xml)
            attempt.stops = _stops_now(xml, table_id)
    except Exception as e:
        attempt.broken = f"importing the patched module raised {e!r}"
        return attempt

    crashed = [p for p, v in after.items() if v < 0]
    if crashed:
        attempt.worst_pid, attempt.worst_loss = crashed[0], 1.0
        return attempt
    for p, was in before.items():
        if was <= 0 or p == patent_id:
            continue
        loss = (was - after.get(p, 0)) / was
        if loss > attempt.worst_loss:
            attempt.worst_pid, attempt.worst_loss = p, loss
    return attempt


def feedback(a: Attempt, patent_id: str) -> str:
    """The observation, as the model will read it. Numbers, then the same lens."""
    if a.broken:
        return (f"YOUR PATCH DOES NOT RUN: {a.broken}\n\n"
                f"That is a syntax or import error in the code you returned, "
                f"not a fact about the patent. Return the same fix, written so "
                f"it parses. Watch for newlines lost inside string literals.\n\n"
                f"Send the corrected patch.")
    if not a.edits:
        return ("YOUR PATCH CHANGED NOTHING: every function you returned was "
                "byte-identical to the source you were shown"
                + (f" ({', '.join(a.noop)})" if a.noop else "")
                + ". Returning a function unchanged is not a patch. Either "
                  "write a real edit, or set `patches` to [] and say in "
                  "`diagnosis` why no function on the candidate list can read "
                  "this table.\n\nSend a real patch.")

    lines = [f"I APPLIED YOUR PATCH AND RAN IT. Measured, not predicted:\n",
             f"  {patent_id}: {a.target_before} -> {a.target_after} distinct "
             f"compounds ({a.usable_after} usable records of {a.records_after} "
             f"extracted)."]
    if a.noop:
        lines.append(
            f"  NOTE: you listed {', '.join(a.noop)} as target(s) but returned "
            f"{'them' if len(a.noop) > 1 else 'it'} BYTE-IDENTICAL to the "
            f"source you were shown, so {'those' if len(a.noop) > 1 else 'that'} "
            f"function did not change at all. Only {', '.join(a.names)} was "
            f"actually edited. If the fix needs both halves, the other half has "
            f"not been written yet.")
    if a.worst_loss > 0:
        lines.append(f"  COLLATERAL: {a.worst_pid} lost {a.worst_loss:.0%} of "
                     f"its compounds. A patch that empties another patent is "
                     f"refused however much it gains here.")
    if a.gained:
        lines.append("\nThat is an improvement.")
        return "\n".join(lines)

    lines.append(
        f"\nSo the patent still yields nothing. Here is the SAME diagnostic you "
        f"were given before, recomputed against the block as YOUR patched code "
        f"now assembles it:\n{a.stops}\n"
        f"Read what changed and what did not. If your edit moved the failure to "
        f"a later gate, patch that gate. If it moved nothing, the function you "
        f"chose is not the one that decides. You may target any function on the "
        f"candidate list, including ones you have already edited — return the "
        f"complete set of functions you want in the tree, since your previous "
        f"patch has been reverted.")
    return "\n".join(lines)


def refine(gap_info: dict, table, *, rounds: int | None = None,
           model: str | None = None, xml_dir=None) -> tuple[Attempt | None, list[Attempt]]:
    """Propose, run, observe, re-propose. Returns (best attempt, every attempt).

    Best is the first attempt that gains on the target; failing that, the one
    that got furthest, so a partial fix is still available to `greedy` to score
    against the corpus. A round that produces nothing runnable does not end the
    loop — an unparseable answer is exactly the case another turn repairs.
    """
    from .capability import MODEL_LADDER, propose_capability_patch

    rounds = MAX_ROUNDS if rounds is None else rounds
    # Sonnet, not the module default of Haiku. This tier's economics are the
    # rule tier's inverted — a capability is bought a handful of times in the
    # whole corpus, and every round costs a full extraction pass over every
    # unfrozen patent, so the token difference between models is noise beside
    # the compute. Measured: Haiku diagnosed the shape correctly and then wrote
    # code calling a helper that does not exist.
    model = model or MODEL_LADDER[0]
    patent_id, table_id = gap_info["patent"], gap_info["table"]
    xml = gap_info.get("xml") or ""
    if not xml:
        xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")
        xml = (Path(xml_dir) / f"{patent_id}.xml").read_text(errors="ignore")

    history: list[dict] = []
    attempts: list[Attempt] = []
    for i in range(1, rounds + 1):
        # An API failure ENDS THE ROUNDS, it does not end the run. Measured:
        # a 400 on the fifth gap's second round propagated out of `refine`,
        # out of `_candidates`, and killed the process before `select` had
        # applied anything — discarding a verified +122 that a previous gap
        # had already won. Rounds already observed are real work and are
        # returned; the caller decides what to do with a short conversation.
        try:
            prop = propose_capability_patch(gap_info, table, model=model,
                                            history=history or None)
        except Exception as e:
            logger.warning("iterate: %s round %d — asking failed (%s); keeping "
                           "the %d round(s) already observed",
                           patent_id, i, type(e).__name__, len(attempts))
            break
        if not prop:
            logger.warning("iterate: %s round %d — no answer", patent_id, i)
            break
        patches = prop.get("patches") or []
        edits, names, noop, broken = splice(patches)
        a = Attempt(round_no=i, names=names, edits=edits, noop=noop, broken=broken,
                    diagnosis=str(prop.get("diagnosis", ""))[:400])
        observe(a, patent_id, table_id, xml, xml_dir=xml_dir)
        attempts.append(a)
        logger.warning("iterate: %s round %d — %s -> %s compounds%s%s",
                       patent_id, i, a.target_before, a.target_after,
                       f" [{', '.join(names)}]" if names else " [no edit]",
                       f" BROKEN: {a.broken}" if a.broken else "")
        if a.gained:
            return a, attempts
        if i == rounds:
            break
        # The assistant turn is what it wrote; the user turn is what that did.
        history.append({"role": "assistant", "content":
                        f"diagnosis: {a.diagnosis}\n\n" + "\n\n".join(
                            f"### {p.get('target')}\n```python\n"
                            f"{p.get('function_source', '')}\n```"
                            for p in patches[:3])})
        history.append({"role": "user", "content": feedback(a, patent_id)})

    runnable = [a for a in attempts if a.edits and not a.broken]
    if not runnable:
        return None, attempts
    return max(runnable, key=lambda a: (a.target_after, -a.worst_loss)), attempts
