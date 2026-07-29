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
    "parse_bin_key": (_SRC / "bin_legend.py",
                      "turns a legend's grade symbols into numeric ranges"),
}


def _function_source(module: Path, name: str) -> str | None:
    src = module.read_text()
    m = re.search(rf"^def {re.escape(name)}\(.*?(?=\n(?:def |@|# ──|\Z))",
                  src, re.S | re.M)
    return m.group(0).rstrip() if m else None


PATCH_SYSTEM = """You are widening an extraction CODEBASE, not describing a table.

You are shown a patent table our parser reads as empty or unusable, the rule \
that was tried and produced nothing, and the current source of the functions \
that decide how such a table is read. Return a corrected version of ONE of them.

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
  - Never read LESS than the current code. Your patch runs against every patent \
in the corpus and is discarded if total records fall or any healthy patent \
loses rows. Satisfying a count by parsing less is the failure mode being \
guarded against, and it will not pass.
  - If none of the offered functions is the right place, say so in `diagnosis` \
and set `target` to `none`. An honest refusal is cheap; a patch to the wrong \
function wastes a verification run and teaches us nothing.

Your patch is applied to a scratch copy, run over the whole corpus and the full \
test suite, and discarded unless every check passes. Describe the fix; the \
harness decides."""

PATCH_TOOL = {
    "name": "propose_capability_patch",
    "description": "Widen one function so a currently-unreadable layout parses.",
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": ("One or two sentences: what shape this table uses "
                                "that the current code cannot express."),
            },
            "target": {
                "type": "string",
                "enum": [*PATCHABLE, "none"],
                "description": "Which function to widen, or `none` if it is not one of these.",
            },
            "function_source": {
                "type": "string",
                "description": ("The complete corrected function. Empty when "
                                "target is `none`."),
            },
        },
        "required": ["diagnosis", "target", "function_source"],
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

    key = f"capability::{gap_info['fingerprint']}::{model}"
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
        prop = propose_capability_patch(g, table, model=model)
        if not prop or prop.get("target") in (None, "none") or not prop.get("function_source"):
            declined += 1
            results.append({"ok": False, "fingerprint": g["fingerprint"],
                            "rows_at_stake": g["rows_at_stake"],
                            "why": "model declined: " + (prop or {}).get("diagnosis", "no proposal"),
                            "diagnosis": (prop or {}).get("diagnosis", "")})
            continue

        target = prop["target"]
        module, _ = PATCHABLE[target]
        current = _function_source(module, target)
        if current is None:
            declined += 1
            continue
        patched = module.read_text().replace(current, prop["function_source"].rstrip(), 1)
        if patched == module.read_text():
            declined += 1
            results.append({"ok": False, "fingerprint": g["fingerprint"],
                            "why": f"could not splice {target} back into {module.name}"})
            continue

        verdict = verify_patch(module, patched, baseline=base)

        # ...and it must FIX THE GAP IT WAS BOUGHT FOR.
        #
        # `verify_patch` asks "did anything get worse". Nothing asks "did
        # anything get better", so a patch that changes no behaviour at all
        # sails through: Sonnet's first `classify_column` proposal was applied
        # clean — corpus fine, tests green, 70,051 usable — and US9302989 still
        # produced 30 records with the gap still open. That is the same defect
        # this whole module exists to fix, one tier up: an answer that does
        # nothing being recorded as an answer.
        #
        # Scoped to the gap's own patent, because that is the claim being
        # tested. A patch may legitimately move nothing elsewhere.
        if verdict.get("ok"):
            before = base.get(g["patent"], 0)
            after = verdict.get("per_patent", {}).get(g["patent"], 0)
            if after <= before:
                verdict["ok"] = False
                verdict["why"] = (
                    f"patch is inert: {g['patent']} still yields {after} usable "
                    f"records (was {before}). It broke nothing and fixed nothing, "
                    f"and the {g['rows_at_stake']} rows it was bought for are "
                    f"still unread.")
            else:
                verdict["gap_rows_recovered"] = after - before
        entry = {
            "action": "capability_patch", "fingerprint": g["fingerprint"],
            "patent": g["patent"], "table": g["table"],
            "rows_at_stake": g["rows_at_stake"], "target": target,
            # Absolute, and named `before_source`/`after_source`: this shares
            # `parser_repair`'s journal, so it must share its contract or
            # `--revert` fails on the entry it was meant to undo.
            "module": str(module),
            "signature": f"{target}::{g['fingerprint']}",
            "diagnosis": prop.get("diagnosis", ""), "model": model or config.MODEL_HAIKU,
            "before_source": current, "after_source": prop["function_source"].rstrip(),
            "applied": False, "why": verdict.get("why", ""),
            "total_usable_before": sum(v for k, v in base.items() if k != "_clean"),
            "total_usable_after": verdict.get("total_usable"),
            "coverage_moved": {
                p: (base[p], verdict.get("per_patent", {}).get(p, 0))
                for p in base if p != "_clean"
                and verdict.get("per_patent", {}).get(p, 0) != base[p]},
        }
        if verdict.get("ok") and do_apply:
            module.write_text(patched)
            entry["applied"] = True
            applied += 1
        elif not verdict.get("ok"):
            declined += 1
        jid = journal_append(entry)
        results.append({"ok": bool(verdict.get("ok")), "journal_id": jid,
                        "fingerprint": g["fingerprint"], "target": target,
                        "rows_at_stake": g["rows_at_stake"],
                        "diagnosis": prop.get("diagnosis", ""),
                        "why": verdict.get("why", ""),
                        "gap_rows_recovered": verdict.get("gap_rows_recovered"),
                        "total_usable": verdict.get("total_usable"),
                        "coverage_moved": entry["coverage_moved"]})

    return {"gaps": len(gaps), "applied": applied, "declined": declined,
            "results": results}
