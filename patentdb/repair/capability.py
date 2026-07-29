"""The third repair tier: when no RULE can express the fix, patch the CODE.

The loop has three things it can conclude about a failing table, and until now
only two of them could act:

  the DOCUMENT is unusual      → buy a rule                 (`loop` + `rules`)
  the READER lost cells        → patch `uspto_xml`          (`parser_repair`)
  our VOCABULARY is too narrow → *nothing*                  ← this module

The third is the one that has cost the most. US9302989 TABLE-US-00001 holds
1,561 rows of `Example | probe 1, probe 2` with cells reading `0.0125, nd` —
two measurements in one cell. The column map a model proposes for it is
*correct*, and it yields zero, because `parse_value` reads one number per cell
and no rule kind can say otherwise. The loop bought that rule, cached it, and
the layout was marked answered forever.

So a capability gap is defined by OUTCOME, not by opinion: a gap that had a
rule available and produced no records. That is observed, needs no validator,
and cannot be argued with — which matters, because every judgement-based gate
in this system has been wrong at least as often as right.

The patch machinery is `parser_repair`'s, unchanged: a candidate is written
into a scratch copy of the tracked tree, run over every cached patent and the
full test suite, and discarded unless fidelity is clean, tests pass, total
usable records do not fall, and no fidelity-clean patent loses rows. Same
journal, so the same `--revert`.

What differs is the QUESTION. `parser_repair` asks "this row declares 9 cells
and you produced 7"; here we ask "this table holds N rows the parser cannot
read, here is the cell, here is the function that reads it". The model picks
which function is responsible from a fixed list — it may not invent a target,
because a patch is only as safe as the blast radius we can verify.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..core import config
from .parser_repair import journal_append, verify_patch

logger = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parent.parent / "sources"

# The only functions a capability patch may target. Each is the single place
# one class of "we cannot read this" is decided, and each is small enough that
# a whole-function rewrite is reviewable. Anything outside this list is refused:
# the verifier bounds the damage a patch can do, and an unbounded target is a
# patch we cannot reason about.
PATCHABLE: dict[str, tuple[Path, str]] = {
    "parse_value": (_SRC / "uspto_assays.py",
                    "reads ONE cell into a number, qualifier, unit and run count"),
    "classify_column": (_SRC / "uspto_assays.py",
                        "decides what a column holds from its header and values"),
    "_header_rows_of": (_SRC / "uspto_assays.py",
                        "decides which rows of a table are header rather than data"),
    "build_columns": (_SRC / "uspto_assays.py",
                      "builds the Column list for a table, id column included"),
    "extract_from_tables": (_SRC / "uspto_assays.py",
                            "walks rows and EMITS AssayRecords from assay columns"),
    "parse_bin_key": (_SRC / "bin_legend.py",
                      "turns a legend's grade symbols into numeric ranges"),
}

# How many functions one proposal may rewrite. Three, because the shapes that
# defeated the single-function tier need two or three cooperating changes
# (see US9302989: header recognition + record emission) and nothing observed
# has needed more — while every extra function widens a blast radius the
# verifier has to cover and a human has to read.
MAX_TARGETS = 3

# Bump whenever the tool schema, the system prompt or the candidate list
# changes. The API cache is keyed by fingerprint and model, and neither moves
# when the QUESTION does: widening the tool from one target to a list of three
# replayed a cached single-target answer, which now parses as an empty
# `patches` list and reads as the model declining. Same failure `SYNTH_EPOCH`
# exists to prevent one tier up — a stale answer to a question we no longer ask.
PATCH_EPOCH = "v3-preserve-comments"

# Tried in order until one patch VERIFIES. Deliberately not Haiku-first, and
# the reason is that this tier's economics are the opposite of the rule tier's.
#
# A rule is bought per layout and there are hundreds of layouts, so a cheap
# model that is right most of the time wins. A capability patch is bought per
# CAPABILITY — three in the whole corpus — and every attempt costs a full
# verification run: the entire corpus re-extracted plus the test suite, minutes
# of compute per candidate. The token difference between models is noise beside
# that, and a wrong patch costs a verification run whether it cost $0.002 or
# $0.05 to ask for.
#
# Measured on the one gap that has been through it: Haiku diagnosed the shape
# correctly — "two assays in one column, two comma-separated measurements per
# cell" — and then wrote code calling a helper that does not exist and invented
# a `multi_value` return shape; the suite caught it on `1,234.5`, a thousands
# separator its comma-split read as two measurements. Sonnet, given the same
# prompt and three targets, wrote `classify_column` + `extract_from_tables` and
# recovered 1,628 records. Diagnosis is easy here; writing code against a live
# codebase is not.
MODEL_LADDER = (config.MODEL_SONNET, config.MODEL_OPUS)


def _function_source(module: Path, name: str) -> str | None:
    src = module.read_text()
    m = re.search(rf"^def {re.escape(name)}\(.*?(?=\n(?:def |@|# ──|\Z))",
                  src, re.S | re.M)
    return m.group(0).rstrip() if m else None


PATCH_SYSTEM = """You are widening an extraction CODEBASE, not describing a table.

You are shown a patent table our parser reads as empty or unusable, the rule \
that was tried and produced nothing, and the current source of the functions \
that decide how such a table is read. Return corrected versions of the ones \
that need to change — up to three.

The failure is that our code has no way to express what this table does. Do not \
propose something specific to this patent: name the general shape and handle it. \
A cell holding `0.0125, nd` is two measurements in one cell; a column headed \
`probe 1, probe 2` names two assays. Those are conventions, not accidents, and \
they recur.

Constraints:
  - Return the COMPLETE function, from `def` through its final return, indented \
at module level. Not a diff, not a fragment.
  - Keep the name, the signature and the return TYPE. Callers are not being \
patched with you; a function that starts returning a different shape breaks \
everything upstream of it and will be rejected.
  - Change as little as possible. You are widening one thing.
  - KEEP EVERY EXISTING COMMENT AND DOCSTRING, verbatim, in place. They record \
why each guard exists — which patent broke, what it cost, what was tried and \
rejected — and that reasoning cannot be recovered from the code. Deleting a \
comment is a silent regression no test can catch, so it is treated as one. Add \
to the docstring to describe your change; do not rewrite or prune what is \
already there. If a comment describes a line you are moving, move the comment \
with it.
  - Never read LESS than the current code. Your patch runs against every patent \
in the corpus and is discarded if total records fall or any healthy patent \
loses rows. Satisfying a count by parsing less is the failure mode being \
guarded against, and it will not pass.
  - Give EVERY function the fix needs, not the smallest edit you can defend. \
They are applied and verified together, and a half-fix is not a smaller risk — \
it is a patch that changes nothing, which is discarded. A shape where a column \
header names two assays needs the classifier to see that AND the emitter to \
produce two records; either alone is inert.
  - If none of the offered functions is the right place, say so in `diagnosis` \
and return an empty `patches` list. An honest refusal is cheap; a patch to the \
wrong function wastes a verification run and teaches us nothing.

Your patch is applied to a scratch copy, run over the whole corpus and the full \
test suite, and discarded unless every check passes. Describe the fix; the \
harness decides."""

PATCH_TOOL = {
    "name": "propose_capability_patch",
    "description": ("Widen one to three functions so a currently-unreadable "
                    "layout parses."),
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": ("One or two sentences: what shape this table uses "
                                "that the current code cannot express."),
            },
            "patches": {
                "type": "array",
                "maxItems": MAX_TARGETS,
                "description": ("The functions to rewrite. Give every function the "
                                "fix needs — they are applied and verified "
                                "TOGETHER, so a half-fix is not a smaller risk, "
                                "it is a patch that changes nothing and is "
                                "discarded. Empty to decline."),
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "enum": [*PATCHABLE]},
                        "function_source": {
                            "type": "string",
                            "description": ("The complete corrected function, from "
                                            "`def` through its final return, "
                                            "indented at module level."),
                        },
                    },
                    "required": ["target", "function_source"],
                },
            },
        },
        "required": ["diagnosis", "patches"],
    },
}


def _sample_of_table(table) -> str:
    from ..sources.uspto_assays import build_columns, merge_header
    lines = [f"HEADER: {merge_header(table)}"]
    lines.append("COLUMNS AS WE CLASSIFY THEM: "
                 + ", ".join(f"[{c.index}] {c.kind}" for c in build_columns(table)))
    if table.caption:
        lines.append(f"CAPTION: {table.caption[:220]}")
    prev = (getattr(table, "preceding", "") or "").strip()
    if prev:
        lines.append(f"TEXT BEFORE TABLE: ...{prev[-320:]}")
    for r in table.body_rows[:6]:
        lines.append("ROW: " + repr([c.text.strip()[:40] for c in r]))
    return "\n".join(lines)


def propose_capability_patch(gap_info: dict, table, *,
                             model: str | None = None) -> dict | None:
    """One paid question per LAYOUT FINGERPRINT, cached like every other."""
    import anthropic

    from ..core.api_cache import get_cached, store_cached
    from ..core.cost_tracker import cost_tracker

    model = model or config.MODEL_HAIKU
    offered = []
    for name, (module, what) in PATCHABLE.items():
        src = _function_source(module, name)
        if src:
            offered.append(f"### `{name}` in {module.name} — {what}\n"
                           f"```python\n{src}\n```")
    if not offered:
        return None

    prompt = (
        f"TABLE {gap_info['table']} of {gap_info['patent']} holds "
        f"{gap_info['rows_at_stake']} rows we cannot turn into records.\n\n"
        f"{_sample_of_table(table)}\n\n"
        f"A `{gap_info['rule_kind']}` rule was already tried on this layout:\n"
        f"  {json.dumps(gap_info.get('rule_payload'))[:400]}\n"
        f"It produced NOTHING. {gap_info.get('why', '')}\n\n"
        f"CANDIDATE FUNCTIONS:\n\n" + "\n\n".join(offered))

    key = f"capability::{PATCH_EPOCH}::{gap_info['fingerprint']}::{model}"
    cached = get_cached(model, key)
    if cached is not None:
        try:
            return json.loads(cached)
        except ValueError:
            pass
    if not config.ANTHROPIC_API_KEY:
        logger.info("capability: no API key; cannot propose a patch")
        return None

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=model, max_tokens=4000, temperature=0,
        system=[{"type": "text", "text": PATCH_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[PATCH_TOOL], tool_choice={"type": "tool", "name": PATCH_TOOL["name"]},
        messages=[{"role": "user", "content": prompt}])
    cost_tracker.record(resp.usage.input_tokens, resp.usage.output_tokens, model,
                        patent_id=gap_info.get("patent", ""), cost_category="lm")
    call = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
    if call is None:
        return None
    out = dict(call.input)
    store_cached(model, key, json.dumps(out),
                 input_tokens=resp.usage.input_tokens,
                 output_tokens=resp.usage.output_tokens)
    return out


def collect_gaps(patent_ids: list[str] | None = None) -> list[dict]:
    """Every capability gap in the corpus, biggest first. Free — no model calls."""
    from .loop import repair_patent

    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    pids = patent_ids or sorted(p.stem for p in xml_dir.glob("*.xml"))
    out: list[dict] = []
    for pid in pids:
        try:
            _, rep = repair_patent(pid, (xml_dir / f"{pid}.xml").read_text(errors="ignore"),
                                   max_calls=0)
        except Exception as e:                       # a broken patent is not a gap
            logger.warning("capability: %s raised %r", pid, e)
            continue
        out.extend(rep.capability_gaps)
    # One question per FINGERPRINT, not per patent — the whole economic argument
    # of this loop. Keep the instance with the most rows riding on it.
    best: dict[str, dict] = {}
    for g in out:
        cur = best.get(g["fingerprint"])
        if cur is None or g["rows_at_stake"] > cur["rows_at_stake"]:
            best[g["fingerprint"]] = g
    return sorted(best.values(), key=lambda g: -g["rows_at_stake"])


def repair_capabilities(*, apply: bool | None = None, limit: int | None = None,
                        patent_ids: list[str] | None = None,
                        model: str | None = None) -> dict:
    """Find capability gaps, buy one patch each, verify, APPLY, journal.

    Applies without asking, for the same reason `repair_reader` does: a fix that
    waits on a switch is a queue, and the gap is costing records while it waits.
    Safety is the verifier and the journal, not permission — every proposal,
    taken or declined, is recorded with its full before/after source and the
    per-patent coverage it moved.
    """
    from .parser_repair import baseline_counts
    from ..sources.uspto_xml import assemble_blocks, parse_tables

    do_apply = config.PARSER_REPAIR_APPLY if apply is None else apply
    gaps = collect_gaps(patent_ids)
    if limit:
        gaps = gaps[:limit]
    if not gaps:
        return {"gaps": 0, "applied": 0, "declined": 0, "results": []}

    base = baseline_counts()
    results = []
    applied = declined = 0
    xml_dir = config.OUTPUT_DIR / "uspto_xml"

    for g in gaps:
        xml = (xml_dir / f"{g['patent']}.xml").read_text(errors="ignore")
        table = {t.table_id: t for t in assemble_blocks(parse_tables(xml))}.get(g["table"])
        if table is None:
            continue
        ladder = (model,) if model else MODEL_LADDER
        for attempt, use_model in enumerate(ladder, 1):
            outcome = _try_one(g, table, use_model, base, do_apply,
                               last=attempt == len(ladder))
            if outcome is None:
                continue                       # declined; climb the ladder
            results.append(outcome)
            if outcome.get("ok"):
                applied += 1
            else:
                declined += 1
            break
        else:
            declined += 1
    return {"gaps": len(gaps), "applied": applied, "declined": declined,
            "results": results}


def _try_one(g: dict, table, model: str, base: dict, do_apply: bool,
             *, last: bool) -> dict | None:
    """One model's attempt at one gap. None means "declined, try the next".

    A rung that fails returns None rather than a result, so the caller climbs.
    The LAST rung always returns a result — otherwise a gap every model refused
    would vanish from the report instead of being visible as still open.
    """
    from .parser_repair import journal_append as _journal
    prop = propose_capability_patch(g, table, model=model)
    patches = (prop or {}).get("patches") or []
    if not prop or not patches:
        if not last:
            return None
        return {"ok": False, "fingerprint": g["fingerprint"], "model": model,
                "rows_at_stake": g["rows_at_stake"],
                "why": "model declined: " + (prop or {}).get("diagnosis", "no proposal"),
                "diagnosis": (prop or {}).get("diagnosis", "")}

    # Splice every function into its module IN MEMORY first. Two targets can
    # live in one file, so edits accumulate per module and are written once —
    # patching the same file twice from its on-disk text would drop the first.
    edited: dict[Path, str] = {}
    parts: list[dict] = []
    for patch in patches[:MAX_TARGETS]:
        target = patch.get("target")
        body = (patch.get("function_source") or "").rstrip()
        if target not in PATCHABLE or not body:
            return None if not last else {
                "ok": False, "fingerprint": g["fingerprint"], "model": model,
                "rows_at_stake": g["rows_at_stake"],
                "why": f"unknown or empty target {target!r}",
                "diagnosis": prop.get("diagnosis", "")}
        mod, _ = PATCHABLE[target]
        base_text = edited.get(mod, mod.read_text())
        current = _function_source(mod, target)
        if current is None or current not in base_text:
            return None if not last else {
                "ok": False, "fingerprint": g["fingerprint"], "model": model,
                "rows_at_stake": g["rows_at_stake"],
                "why": f"could not locate {target} in {mod.name}",
                "diagnosis": prop.get("diagnosis", "")}
        edited[mod] = base_text.replace(current, body, 1)
        parts.append({"module": str(mod), "target": target,
                      "before_source": current, "after_source": body})

    module, patched = next(iter(edited.items()))
    also = {m: t for m, t in edited.items() if m != module}
    targets = ", ".join(p["target"] for p in parts)

    verdict = verify_patch(module, patched, baseline=base, also=also)

    # ...and it must FIX THE GAP IT WAS BOUGHT FOR.
    #
    # `verify_patch` asks "did anything get worse". Nothing asks "did anything
    # get better", so a patch that changes no behaviour at all sails through:
    # Sonnet's first single-target `classify_column` proposal was applied clean
    # — corpus fine, tests green, 70,051 usable — and US9302989 still produced
    # 30 records with the gap still open. That is the same defect this module
    # exists to fix, one tier up: an answer that does nothing being recorded as
    # an answer.
    if verdict.get("ok"):
        before = base.get(g["patent"], 0)
        after = verdict.get("per_patent", {}).get(g["patent"], 0)
        if after <= before:
            verdict["ok"] = False
            verdict["why"] = (
                f"patch is inert: {g['patent']} still yields {after} usable "
                f"records (was {before}). It broke nothing and fixed nothing, and "
                f"the {g['rows_at_stake']} rows it was bought for are still unread.")
        else:
            verdict["gap_rows_recovered"] = after - before

    # A rung that failed climbs, and leaves no journal entry: the record that
    # matters is what was applied and what the last model could not do.
    if not verdict.get("ok") and not last:
        logger.info("capability: %s declined %s (%s) — escalating",
                    model, g["fingerprint"], verdict.get("why", "")[:90])
        return None

    entry = {
        "action": "capability_patch", "fingerprint": g["fingerprint"],
        "patent": g["patent"], "table": g["table"],
        "rows_at_stake": g["rows_at_stake"], "target": targets,
        "signature": f"{targets}::{g['fingerprint']}",
        "diagnosis": prop.get("diagnosis", ""), "model": model,
        # The group, and a single-patch view of its first member so the shared
        # journal reader keeps working on old and new entries alike.
        "patches": parts,
        "module": parts[0]["module"],
        "before_source": parts[0]["before_source"],
        "after_source": parts[0]["after_source"],
        "applied": False, "why": verdict.get("why", ""),
        "gap_rows_recovered": verdict.get("gap_rows_recovered"),
        "total_usable_before": sum(v for k, v in base.items() if k != "_clean"),
        "total_usable_after": verdict.get("total_usable"),
        "coverage_moved": {
            q: (base[q], verdict.get("per_patent", {}).get(q, 0))
            for q in base if q != "_clean"
            and verdict.get("per_patent", {}).get(q, 0) != base[q]},
    }
    if verdict.get("ok") and do_apply:
        for mod, text in edited.items():
            mod.write_text(text)
        entry["applied"] = True
    jid = _journal(entry)
    return {"ok": bool(verdict.get("ok")), "journal_id": jid, "model": model,
            "fingerprint": g["fingerprint"], "target": targets,
            "rows_at_stake": g["rows_at_stake"],
            "diagnosis": prop.get("diagnosis", ""), "why": verdict.get("why", ""),
            "gap_rows_recovered": verdict.get("gap_rows_recovered"),
            "total_usable": verdict.get("total_usable"),
            "coverage_moved": entry["coverage_moved"]}
