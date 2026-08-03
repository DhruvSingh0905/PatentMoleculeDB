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

The patch machinery is `parser_repair`'s: a candidate is written into a scratch
copy of the tracked tree and run over every cached patent and the full test
suite. It is discarded for exactly ONE reason — it reads fewer compounds than
before. Fidelity discrepancies, test failures and a patch that recovers nothing
are recorded as `objections` and applied anyway, because a gate that guesses is
how a correct column_map scored 0/23 and a patch recovering 1,238 rows was
declined for looking inert. Safety here is the journal, not permission: same
`--revert`, and every applied patch carries its coverage delta.

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
from . import value_check
from .oracle import diff, sample_patent, target_note
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
    # ── the ASSEMBLER ────────────────────────────────────────────────
    # Absent until now, and their absence was visible in the model's own
    # answers: on US10189840 and US9018217 it diagnosed the defect exactly
    # ("example numbers 1 2 3 ... 35 are crammed into the same cell as
    # `Ex. No.`") and then declined, because every function it was permitted
    # to rewrite lives downstream of the damage. It could see the bug and not
    # reach it.
    #
    # A patent that yields NOTHING is usually broken here rather than in the
    # readers above — by the time a row reaches `parse_value`, assembly has
    # already decided whether it is data at all. Higher blast radius, so the
    # same protection as everything else and nothing more: a patch that picks
    # up fewer compounds is declined, and the journal makes any state one
    # `--revert` away.
    "assemble_block": (_SRC / "uspto_xml.py",
                       "stitches every tgroup of one <tables> block into the "
                       "grid it encodes, choosing which fragments are data and "
                       "which are header or interleaved annotation"),
    "_opens_with_id": (_SRC / "uspto_xml.py",
                       "decides whether a row begins with a compound identifier "
                       "— the test that separates a data fragment from an "
                       "annotation fragment"),
    "_is_namelike": (_SRC / "uspto_xml.py",
                     "decides whether a row is column NAMES rather than data; a "
                     "false positive here files data rows as header"),
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
PATCH_EPOCH = "v6-raw-cals-in-prompt"

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

Your patch is applied to a scratch copy and run over the whole corpus and the \
full test suite. It is rejected for one reason only: if it reads FEWER \
compounds than the current code. Describe the fix; the harness decides.
NEVER write a non-ASCII character as a `\\uXXXX` escape. Emit the literal
character. Inside a string literal the two are equivalent, but inside a COMMENT
`\\u2212` is not an escape — it is the six characters, and the comment is
corrupted. This has happened twice on the same line, where the comment
explaining that `float` cannot read a typesetter's minus had that very minus
turned into `\\u2212`. Behaviour survived, the reasoning did not, and no test
can see it.
"""

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


def _bad_values_now(patent_id: str) -> int:
    """How many of this patent's values already disagree with BindingDB.

    The baseline for the delta. Computed on the unpatched tree, through the
    same full path the sandbox measures — parse plus cached rules — so the two
    numbers are comparable.
    """
    from ..sources.uspto_assays import extract_from_patent
    from .loop import repair_patent
    xml = (config.OUTPUT_DIR / "uspto_xml" / f"{patent_id}.xml")
    if not xml.exists():
        return 0
    try:
        text = xml.read_text(errors="ignore")
        base = [r for r in extract_from_patent(text) if r.is_usable]
        extra, _ = repair_patent(patent_id, text, max_calls=0)
        return value_check.check_patent(patent_id, base + list(extra))["bad"]
    except Exception as e:
        logger.warning("value_check baseline failed for %s: %r", patent_id, e)
        return 0


def _sample_of_table(table, xml: str = "") -> str:
    """What we make of the block, beside what the block actually says.

    Every field here except the last was OUR OUTPUT, and on the patents that
    matter our output is the defect. Measured across the six patents still at
    zero, this showed the model:

        US9018217     2 rows of 254   (assembly had destroyed the rest)
        US10189840    5 rows of 94    (89 were collapsed into the header STRING)
        US9695181     3 rows of 9     (all three were footnotes)

    — between 681 and 1,078 characters, against an 8,129-character block. The
    models still diagnosed it correctly, from almost nothing, and then wrote
    patches against the damage. That is this repo's own rule — never diagnose
    from the parsed view — enforced for humans and broken for the model.

    So the parsed view is still shown, because "what we currently believe" is
    genuinely useful, but it is labelled as ours and the SOURCE follows it.
    """
    from ..sources.uspto_assays import build_columns, merge_header
    from ..sources.uspto_xml import parse_tables

    lines = ["=== WHAT OUR PARSER CURRENTLY MAKES OF THIS BLOCK ===",
             "(this is our OUTPUT, not the document — where it looks wrong, it is)"]
    lines.append(f"HEADER (our merge): {merge_header(table)}")
    lines.append("COLUMNS AS WE CLASSIFY THEM: "
                 + ", ".join(f"[{c.index}] {c.kind}" for c in build_columns(table)))
    if table.caption:
        lines.append(f"CAPTION: {table.caption[:220]}")
    prev = (getattr(table, "preceding", "") or "").strip()
    if prev:
        lines.append(f"TEXT BEFORE TABLE: ...{prev[-320:]}")

    body = [r for r in table.body_rows if any(c.text.strip() for c in r)]
    # UNTRUNCATED. The 40-character cap cut US10227341's inverted table mid
    # compound-list — `'2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, '` — which is
    # precisely the payload that names the layout.
    for r in body[:12]:
        lines.append("ROW: " + repr([c.text.strip()[:200] for c in r]))
    lines.append(f"(assembled: {len(table.header_rows)} header row(s), "
                 f"{len(body)} body row(s))")

    if xml:
        raw = [t for t in parse_tables(xml) if t.table_id == table.table_id]
        raw_rows = sum(len([r for r in t.body_rows if any(c.text.strip() for c in r)])
                       for t in raw)
        if raw_rows and raw_rows != len(body):
            lines.append(
                f"\n!! THE READER DISAGREES WITH ITS OWN SOURCE: the document "
                f"declares {raw_rows} populated body row(s) across {len(raw)} "
                f"tgroup(s), and assembly kept {len(body)}. The rows are not "
                f"missing from the patent; they are missing from what you are "
                f"shown above. Assume the reader is wrong before the layout is.")
        from .oracle import _raw_block
        block = _raw_block(xml, table.table_id)
        if block:
            lines.append("\n=== THE BLOCK AS THE PATENT PUBLISHES IT (OASIS/CALS) ===")
            lines.append(block)
    return "\n".join(lines)


def _retry_note(gap_info: dict) -> str:
    """What the LAST patch for this layout did, when there was one.

    A patch is applied on corpus coverage, which is the right condition — it is
    the only one that cannot be argued with. But it means a patch can land,
    raise the corpus, and leave the patent that ASKED for it at zero: measured,
    US10266548 and US9801872 both had a patch applied and stayed at 0.

    That is not a reason to block the patch. It is a reason not to call the gap
    answered. The model is told what its predecessor did so the next attempt is
    a different attempt rather than the same one re-derived.
    """
    prior = gap_info.get("prior_attempts") or []
    if not prior:
        return ""
    lines = ["A PREVIOUS patch for this same layout was already APPLIED and did "
             "NOT fix this patent:"]
    for a in prior[-3:]:
        lines.append(f"  - targeted {a.get('target')}: corpus coverage moved "
                     f"{a.get('total_usable_before')} -> {a.get('total_usable_after')}, "
                     f"but {gap_info['patent']} went "
                     f"{a.get('before', '?')} -> {a.get('after', '?')} usable.")
        if a.get("diagnosis"):
            lines.append(f"    it reasoned: {str(a['diagnosis'])[:300]}")
    lines.append("Do not repeat that patch. Either target a DIFFERENT function, "
                 "or explain in `diagnosis` why this patent cannot be read by "
                 "any function on the candidate list.\n\n")
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
        f"{_sample_of_table(table, gap_info.get('xml') or '')}\n\n"
        f"A `{gap_info['rule_kind']}` rule was already tried on this layout:\n"
        f"  {json.dumps(gap_info.get('rule_payload'))[:400]}\n"
        f"It produced NOTHING. {gap_info.get('why', '')}\n\n"
        + (gap_info.get("oracle_note") or "")
        + _retry_note(gap_info)
        + f"CANDIDATE FUNCTIONS:\n\n" + "\n\n".join(offered))

    # The retry count is IN THE KEY. Without it a re-run replays the same
    # cached answer, so a patch that applied and left its own patent at zero
    # would be proposed again, verbatim, for ever. `retry` is derived from the
    # journal, so it survives the process that recorded it.
    retry = int(gap_info.get("retry") or 0)
    key = (f"capability::{PATCH_EPOCH}::{gap_info['fingerprint']}::{model}"
           + (f"::retry{retry}" if retry else "")
           + ("::oracle" if gap_info.get("oracle_note") else ""))
    cached = get_cached(model, key)
    if cached is not None:
        try:
            return json.loads(cached)
        except ValueError:
            pass
    if not config.ANTHROPIC_API_KEY:
        logger.info("capability: no API key; cannot propose a patch")
        return None

    from ..core.api_client import resilient_client
    client = resilient_client()
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


def _gap_from_a_silent_patent(patent_id: str, report) -> dict | None:
    """A gap for a patent that failed too completely to produce one.

    Every entry in `capability_gaps` is raised by a TABLE-level detector, and a
    detector needs a parsed table to measure. When the defect is large enough to
    destroy the parsed view, it destroys the evidence too: US9018217 filed its
    data rows as header, produced zero records, raised zero gaps, and therefore
    could not be handed to the tier built to fix exactly that. It reached
    `maybe_escalate`, spent a budget slot, and `collect_gaps` returned nothing.

    So when a patent yields nothing and no table can say why, the gap is the
    PATENT. Pick the block carrying the most measurement-shaped cells — read off
    raw cell text, so no classifier we might have broken is consulted — and let
    the model see the raw CALS beside the functions that failed on it. That is
    the stated fallback for this whole design: give it the source and our code.

    Returns None whenever the patent produced anything at all, so this can never
    compete with the precise signal.
    """
    from ..sources.uspto_assays import extract_from_patent
    from ..sources.uspto_xml import assemble_blocks, parse_tables
    from .gap import layout_fingerprint
    from .loop import _GRADE, _SHAPED
    from ..sources.uspto_assays import _header_rows_of, merge_header

    blank = [e for e in getattr(report, "escalations", [])
             if e.get("capability") == "PATENT YIELDED NOTHING"]
    if not blank:
        return None
    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    try:
        xml = (xml_dir / f"{patent_id}.xml").read_text(errors="ignore")
        if any(r.is_usable for r in extract_from_patent(xml)):
            return None                      # not silent; the precise signal owns it
        raw = parse_tables(xml)
        tables = {t.table_id: t for t in assemble_blocks(raw)}
    except Exception:
        return None

    # Counted on the RAW tgroups, across header AND body rows.
    #
    # The first version counted `body_rows` of the ASSEMBLED block, which is the
    # damaged view — the one the defect produced. On US10189840, whose data rows
    # were filed as HEADER rows, that scored the real table at 5 rows and picked
    # a 35-cell decoy instead of the 94-row block actually holding the assay
    # data. Choosing which table to show the model from the view that failed is
    # the mistake this file's own docstring warns about.
    per_block: dict[str, int] = {}
    for t in raw:
        n = sum(1 for row in (t.header_rows + t.body_rows) for c in row
                if c.text.strip()
                and (_SHAPED.match(c.text.strip()) or _GRADE.match(c.text.strip())))
        per_block[t.table_id] = per_block.get(t.table_id, 0) + n
    if not per_block:
        return None
    best_id = max(per_block, key=lambda k: per_block[k])
    best_n = per_block[best_id]
    best = tables.get(best_id)
    # NO patent-level floor here. There was one — the same constant the caller
    # uses — and it is the wrong UNIT at this point, not a wrong number.
    #
    # `loop` sums shaped cells across the whole patent to decide the document is
    # worth reporting. This picks the single biggest BLOCK. US9695181 holds 18
    # cells as 6 + 6 + 6, so it cleared the patent-level floor of 10, fired the
    # invariant, reached `autoheal` — and then failed a per-block comparison
    # against that same 10 and produced no gap at all. A patent that raises and
    # cannot be collected is worse than one that never raised.
    #
    # By the time we are here the caller has already decided this document is
    # worth looking at. The only question left is which block to show, and the
    # answer is the biggest one, whatever its absolute size.
    if best is None or best_n < 1:
        return None
    hdrs = merge_header(best, _header_rows_of(best)[0])
    return {
        "fingerprint": layout_fingerprint(best, hdrs),
        "patent": patent_id, "table": best.table_id,
        "rows_at_stake": blank[0].get("rows_at_stake") or best_n,
        "rule_kind": None, "rule_payload": {},
        "why": (f"{patent_id} produced NO usable measurement from any block, and "
                f"no table-level detector could say why — the failure was large "
                f"enough to destroy the evidence a detector reads. "
                f"{best.table_id} carries {best_n} measurement-shaped cells and "
                f"is the largest such block. Assume the reader, not the layout: "
                f"compare the raw CALS against what our functions make of it."
                + (" The assembler also reports that this patent's header "
                   "outgrew its own body."
                   if any(e.get("capability", "").startswith("ASSEMBLY DEFECT")
                          for e in getattr(report, "escalations", [])) else "")),
        "unparsed_examples": [
            " | ".join(c.text.strip() for c in row)[:80]
            for row in best.body_rows[:6] if any(c.text.strip() for c in row)],
    }


def _annotate_prior_attempts(gaps: list[dict]) -> None:
    """Attach prior applied-but-inert attempts, per fingerprint, from the journal."""
    from .parser_repair import journal_read

    try:
        entries = journal_read()
    except Exception:                            # a missing journal is not an error
        return
    by_fp: dict[str, list[dict]] = {}
    for e in entries:
        if e.get("action") != "capability_patch" or not e.get("applied"):
            continue
        if (e.get("gap_rows_recovered") or 0) > 0:
            continue                             # it worked; nothing to warn about
        by_fp.setdefault(str(e.get("fingerprint")), []).append({
            "target": e.get("target"), "diagnosis": e.get("diagnosis"),
            "total_usable_before": e.get("total_usable_before"),
            "total_usable_after": e.get("total_usable_after"),
            "before": (e.get("coverage_moved") or {}).get(e.get("patent"), [None, None])[0],
            "after": (e.get("coverage_moved") or {}).get(e.get("patent"), [None, None])[1],
        })
    for g in gaps:
        prior = by_fp.get(str(g.get("fingerprint")))
        if prior:
            g["prior_attempts"] = prior
            g["retry"] = len(prior)


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
        if not rep.capability_gaps:
            synth = _gap_from_a_silent_patent(pid, rep)
            if synth is not None:
                out.append(synth)
    # A gap with nothing at stake is not a gap. Once the code tier fixes a
    # layout, the deterministic parser reads it directly and the rule bought
    # for it becomes redundant — `apply_rule` returns nothing because there is
    # nothing left to recover, not because anything is broken. US9302989 sat in
    # this list at 0 rows after its own patch landed.
    out = [g for g in out if g["rows_at_stake"] > 0]
    # Annotate each gap with what has ALREADY been tried on its layout and
    # failed to move the patent that raised it. Read from the journal, so it
    # survives the process — a gap whose patch applied without fixing its own
    # patent must come back as a DIFFERENT question, not the same one replayed
    # from cache.
    _annotate_prior_attempts(out)
    # One question per FINGERPRINT, not per patent — the whole economic argument
    # of this loop. Keep the instance with the most rows riding on it.
    best: dict[str, dict] = {}
    for g in out:
        cur = best.get(g["fingerprint"])
        if cur is None or g["rows_at_stake"] > cur["rows_at_stake"]:
            best[g["fingerprint"]] = g
    return sorted(best.values(), key=lambda g: -g["rows_at_stake"])


def _attach_oracle(g: dict, xml: str) -> None:
    """Read the failing block independently, and diff it against what we got.

    Only for a patent that yields NOTHING, because that is the case where the
    model has twice declined to write a patch — it could describe the defect and
    had no way to check a fix. One block, one call, cached by fingerprint.

    It also settles the question the tier could not previously ask: is there
    anything in here at all. `holds_assay_data: false` ends the gap for free
    rather than buying a patch for a formulation table.
    """
    from ..sources.uspto_assays import extract_from_patent

    try:
        records = list(extract_from_patent(xml))
        if any(r.is_usable for r in records):
            return                               # not a silent patent
        # BREADTH. `sample_patent` reads several blocks shallowly rather than
        # one block deeply — the failing block is rarely the only broken one,
        # and a defect is far easier to name from three structures than from
        # ninety rows of one. The gap's own block is read first regardless.
        samples = sample_patent(g["patent"], xml)
        if not samples:
            return
        # Put the gap's own block at the front if the sampler reached it.
        samples.sort(key=lambda o: 0 if o.get("table_id") == g["table"] else 1)
        g["oracle_samples"] = samples

        notes, total_missing = [], 0
        for o in samples:
            tid = o.get("table_id")
            if not o.get("holds_assay_data"):
                notes.append(f"[{tid}] an independent read finds NO assay data: "
                             f"{str(o.get('why_we_might_miss_them'))[:200]}")
                continue
            d = diff(o, records, tid)
            total_missing += d["n_missing_row"] + d["n_missing_value"] + d["n_missing_unit"]
            if tid == g["table"]:
                g["oracle"] = o
                g["oracle_diff"] = d
            notes.append(target_note(o, d).rstrip()
                         + f"\n  (block {tid}"
                         + (f", same shape as {len(o.get('shares_shape_with') or [])} other(s)"
                            if o.get("shares_shape_with") else "") + ")")
            logger.warning("oracle: %s %s — %d row(s) present, we produce %d; "
                           "missing row/value/unit = %d/%d/%d",
                           g["patent"], tid, d["oracle_rows"], d["we_produced"],
                           d["n_missing_row"], d["n_missing_value"],
                           d["n_missing_unit"])
        if not notes:
            return
        g["oracle_note"] = (
            f"INDEPENDENT READS OF {len(samples)} BLOCK(S) IN THIS PATENT.\n"
            f"These are a TARGET and a description of SHAPE — never data to "
            f"store. Where several blocks fail the same way, fix the cause "
            f"once.\n\n" + "\n\n".join(notes) + "\n\n")
    except Exception as e:                       # never break a run
        logger.warning("oracle: skipped for %s (%r)", g.get("patent"), e)


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
        _attach_oracle(g, xml)
        table = {t.table_id: t for t in assemble_blocks(parse_tables(xml))}.get(g["table"])
        if table is None:
            continue
        g["xml"] = xml                           # so the prompt can show the SOURCE
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

    verdict = verify_patch(module, patched, baseline=base, also=also,
                           repair_pid=g["patent"])

    # The BindingDB value delta is RECORDED, not enforced.
    #
    # It was the second blocking condition for one commit. Taking it out is
    # deliberate: a fixed rule about what a good patch looks like is the wrong
    # premise for an extractor whose whole job is adapting to layouts nobody
    # anticipated. Every such rule here has eventually blocked something
    # correct — a column_map at 0/23, a 49% floor on a reader bug, an
    # inert-check on a patch recovering 1,238 rows — and each cost more than
    # the fabrications it caught, which the value check finds afterwards
    # anyway. So there is ONE condition, and it is the only one that cannot be
    # an opinion: does the patched code pick up more compounds than before?
    #
    # Everything else — fidelity, the suite, value agreement, inertness — is
    # measured, journaled, printed, and applied regardless. The journal is what
    # makes that safe: any state is one `--revert` away.
    if verdict.get("bad_values") is not None:
        before_bad = _bad_values_now(g["patent"])
        after_bad = verdict["bad_values"]
        verdict["bad_values_before"] = before_bad
        if after_bad > before_bad:
            verdict.setdefault("objections", []).append(
                f"introduces {after_bad - before_bad} value(s) disagreeing with "
                f"BindingDB ({before_bad} -> {after_bad})")

    # Recorded, NOT enforced. An inert patch adds dead code; it does not lose
    # a compound, and this tier does not block on anything but that.
    #
    # It was a blocker for one commit and it declined a patch that recovers
    # 1,238 rows: Opus promoted US11286268's `FP` column of `+`/`++` to a named
    # assay, which `extract_from_patent` scores as 0 usable because the bin_key
    # rule that turns each grade into a range runs in the loop, not the parse.
    # Judged on the parse it looked like it did nothing. The measurement is
    # fixed now — `repaired_usable` reads the full path — but the lesson is
    # that the gate was the mistake, not the metric behind it.
    #
    # Escalation still uses it: a rung whose patch recovers nothing climbs to
    # the next model, and only the last rung's attempt is applied. That is a
    # preference between candidates, not a veto over all of them.
    before = base.get(g["patent"], 0)
    after = verdict.get("repaired_usable")
    if after is None or after < 0:
        after = verdict.get("per_patent", {}).get(g["patent"], 0)
    verdict["gap_rows_recovered"] = after - before
    if verdict.get("ok") and after <= before:
        verdict.setdefault("objections", []).append(
            f"inert: {g['patent']} still reaches {after} usable records "
            f"(was {before}); the {g['rows_at_stake']} rows it was bought for "
            f"are still unread")

    # A rung that failed climbs, and leaves no journal entry: the record that
    # matters is what was applied and what the last model could not do.
    if (not verdict.get("ok") or verdict["gap_rows_recovered"] <= 0) and not last:
        logger.info("capability: %s recovered nothing on %s (%s) — escalating",
                    model, g["fingerprint"],
                    (verdict.get("why") or "; ".join(verdict.get("objections", [])))[:90])
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
        "objections": verdict.get("objections") or [],
        "gap_rows_recovered": verdict.get("gap_rows_recovered"),
        "bad_values": verdict.get("bad_values"),
        "bad_values_before": verdict.get("bad_values_before"),
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
