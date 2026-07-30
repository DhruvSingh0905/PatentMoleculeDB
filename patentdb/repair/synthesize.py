"""Ask a model for a RULE, never for data. One paid call per distinct layout.

The economics
-------------
HARVEST costs ~$8.81/patent (25.7M input tokens over the 22-patent corpus)
because it re-reads every document in 6,000-char chunks and asks for data
tuples — output nothing can verify, paid for again on every patent.

This asks a different question. Given a ~60-120 token sample of ONE failing
table, return a rule that we then execute deterministically over every patent
sharing that layout. Cost is per *distinct table shape*, not per patent and not
per chunk, and the answer is verifiable by running it.

This is the architecture production self-healing scrapers converged on (Kadoa,
Anansi): cache a per-layout artifact, $0 on hit, one small repair call on miss,
persist the result.

Why tool-use rather than a parsing library
------------------------------------------
Outlines / Guidance / XGrammar / GBNF all constrain decoding by masking logits,
which requires running the model yourself. Against a hosted API they cannot
give any guarantee. Anthropic implements the same technique server-side, so a
`strict` tool schema is the hosted-API equivalent and needs no extra
dependency.

The guarantee is structural, not semantic: the model will return well-formed
JSON, but whether the rule is *correct* is unknowable from the response. That
is entirely the validator's job — `rules.validate()` re-runs the proposal
against real rows and an adversarial battery, and a rule that fails is rejected
no matter how confident the model claimed to be.
"""
from __future__ import annotations

import json
import logging
import re

from ..core import config
from ..core.api_cache import get_cached, store_cached
from ..core.cost_tracker import cost_tracker
from .gap import Gap
from .rules import (BIN_KEY, COLUMN_MAP, ESCALATE, NOT_ASSAY, ROW_REGEX,
                    SYNTH_EPOCH, VALUE_PATTERN, Rule)

logger = logging.getLogger(__name__)

# Haiku by default: this is a small, narrow, highly-constrained task and the
# validator catches errors, so paying Sonnet rates for it is waste.
SYNTH_MODEL = config.MODEL_HAIKU

# Bump when SYSTEM, TOOL or the sample rendering changes. The cache key is
# the layout fingerprint, which is content-independent by design — so without
# a version here, improving the prompt silently replays answers produced by
# the old one. Exactly the stale-cache trap the repo's CLAUDE.md warns about.
#
# One string for both caches. They were separate, and only this one was
# honoured: a bumped prompt bought a fresh API call whose answer was then
# discarded in favour of the stale `escalate` already sitting in the rule
# library. Two version keys for one question is a drift hazard, so there is now
# one, defined next to the library it also governs.
PROMPT_VERSION = SYNTH_EPOCH

# Frozen prefix — identical on every call so it can be prompt-cached (cache
# reads bill at ~0.1x). Everything patent-specific goes in the user turn.
SYSTEM = """You read one fragment of a table from a US patent and describe how \
to extract bioassay measurements from it. You do NOT extract the data itself.

A bioassay measurement is a potency or activity value for a compound: IC50, \
EC50, Ki, Kd, percent inhibition, clearance. The following are NOT assay \
values and must never be mapped as one:
  - NMR shifts and coupling constants (1H NMR, delta, ppm, J = ... Hz)
  - mass spectrometry m/z, [M+H]+, LCMS, HRMS, "found", "calc'd"
  - molecular weight, exact mass
  - retention time, purity, method names
  - the compound identifier itself

You are shown TWO views of the same table: the patent's own CALS XML, and our \
reading of it. The XML is the source and our reading is derived, so where they \
disagree the fault is ours. Say so — a note that our header, our column count \
or our cell boundaries do not match what the source declares is worth more \
than any rule, because a rule written against a damaged view encodes the \
damage. `colspec` widths in particular state how a header tgroup maps onto a \
body tgroup of a different width, and we have got that wrong before.

Prefer `column_map` whenever the table has usable cell structure; it is safer \
than a regex because there is nothing to over-match. Use `row_regex` only when \
cell boundaries are lost. Anchor any regex and keep it specific — a pattern \
that matches "some number somewhere in the row" will be rejected.

If the columns are already identified correctly and the cells plainly contain \
numbers we are failing to read, the problem is our cell parser, not the layout \
— return `value_pattern` with a regex for ONE CELL. It must be a strict \
superset of what already works: it will be rejected if it changes the value of \
any cell we currently read correctly, or if it matches non-measurements.

If the cells hold grade symbols (A-E, +/++/+++) and the patent's text defines \
what they mean, return `bin_key`. If that text defines MORE THAN ONE scale over \
the same symbols — commonly a potency scale and a hERG/selectivity scale, which \
usually run in OPPOSITE directions — that is not a reason to escalate: return \
`scales`, one entry per scale, each with a `match` regex naming the assay \
columns it governs. Escalating a multi-scale legend throws away data we can \
extract; merging the scales, or picking one, is a silent 50-fold error.

If the table is characterisation data rather than assay data, return \
`not_assay`.

YOUR JOB IS TO DESCRIBE, NOT TO DECIDE. Every rule you return is re-run \
against real rows before anything is adopted: it must beat the coverage the \
current parser achieves, generalise to rows you were never shown, and survive \
an adversarial battery of NMR, MS, molecular-weight and retention-time lines. \
A rule that fails is discarded and cannot reach the database. Judging safety \
is that check's job — it can test an answer against thousands of rows and you \
cannot see past the fragment in front of you.

So there is no option here for "this is awkward" or "I am unsure". Report what \
the fragment shows. If two readings are possible, return the more likely one. \
If part of the layout defeats you, describe the part you DO understand — a \
rule covering three of five columns is worth more than nothing, and the check \
downstream will keep whichever part holds up."""

# At most this many model turns per gap: an ask, a look, an answer. Bounded so
# a model that keeps requesting context cannot spend without end.
MAX_TURNS = 3

MORE_TOOL = {
    "name": "request_more_context",
    "description": (
        "Call this INSTEAD of proposing a rule when the fragment you were shown "
        "is not enough to decide safely — a header you cannot align to columns, "
        "a legend that appears cut off, too few rows to see the pattern. You "
        "will be shown a larger view of the same table and can then answer. "
        "The original CALS XML is already in your first prompt, so ask for "
        "`more_rows` or `full_legend` when the obstacle is scope rather than "
        "fidelity. "
        "Prefer this over `escalate` whenever the obstacle is that you cannot "
        "SEE enough: escalate means the capability is missing, not the data."),
    "input_schema": {
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "enum": ["more_rows", "full_legend", "raw_headers",
                         "surrounding_text", "raw_source", "everything"],
                "description": "Which part is insufficient.",
            },
            "why": {
                "type": "string",
                "description": "One sentence: what you could not determine.",
            },
        },
        "required": ["what", "why"],
    },
}

TOOL = {
    "name": "propose_extraction_rule",
    "description": "Describe how to extract assay measurements from this table layout.",
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                # ESCALATE is deliberately NOT offered. While it was in this
                # enum it was a free exit from any awkward layout, and awkward
                # layouts are exactly where the data we are missing lives: on
                # US10172859 the model escalated eight times while its own note
                # showed it had read the table correctly. Escalation is now
                # DERIVED — a gap escalates when the validator rejects real
                # proposals against real rows, carrying that evidence, rather
                # than when the model feels unsure.
                "enum": [COLUMN_MAP, ROW_REGEX, VALUE_PATTERN, BIN_KEY, NOT_ASSAY],
                "description": "Which sort of rule applies to this layout.",
            },
            "cid_column": {
                "type": "integer",
                "description": "column_map only: 0-based index of the compound-id column.",
            },
            "assay_columns": {
                "type": "array",
                "description": "column_map only: the measurement columns.",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "name": {"type": "string",
                                 "description": "assay name, e.g. 'hERG IC50'"},
                        "unit": {"type": "string",
                                 "description": "nM, uM, mM, pM or % — omit if unstated"},
                    },
                    "required": ["index", "name"],
                },
            },
            "pattern": {
                "type": "string",
                "description": ("row_regex only: a Python regex applied to one row "
                                "rendered as 'cell | cell | cell'. Must capture a "
                                "named group `cid` and at least one `value1`, "
                                "`value2`, ... group."),
            },
            "value_pattern": {
                "type": "string",
                "description": (
                    "value_pattern only: a Python regex matched against ONE CELL. "
                    "It MUST use named groups. Example: "
                    r"^\s*(?P<qual>[<>~])?\s*(?P<num>\d+(?:\.\d+)?)\s*"
                    r"(?P<unit>nM|uM)?\s*$"
                    " — `num` is required; `qual` and `unit` are optional. Use "
                    "this when the columns are already identified correctly but "
                    "our parser fails on cells that plainly contain a value."),
            },
            "value_columns": {
                "type": "array", "items": {"type": "integer"},
                "description": "value_pattern only: which column indices it applies to.",
            },
            "bins": {
                "type": "array",
                "description": ("bin_key only: each grade symbol and the numeric "
                                "range it denotes, read from the patent's own text."),
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "e.g. A, B, ++"},
                        "lo": {"type": "number", "description": "lower bound, omit if unbounded"},
                        "hi": {"type": "number", "description": "upper bound, omit if unbounded"},
                        "unit": {"type": "string", "description": "nM, uM, mM or pM"},
                    },
                    "required": ["symbol", "unit"],
                },
            },
            "scales": {
                "type": "array",
                "description": (
                    "bin_key only, INSTEAD of `bins`, when the legend defines "
                    "more than one scale over the same symbols — e.g. one A-D "
                    "scale for potency and another for hERG, often running in "
                    "opposite directions. One entry per scale."),
                "items": {
                    "type": "object",
                    "properties": {
                        "match": {
                            "type": "string",
                            "description": (
                                "Regex matched case-insensitively against the "
                                "assay/column name this scale governs, e.g. "
                                "'herg|kv11' or 'dna-?pk'. Omit on at most one "
                                "entry to make it the default."),
                        },
                        "bins": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "lo": {"type": "number"},
                                    "hi": {"type": "number"},
                                    "unit": {"type": "string"},
                                },
                                "required": ["symbol", "unit"],
                            },
                        },
                    },
                    "required": ["bins"],
                },
            },
            "because": {
                "type": "string",
                "description": "not_assay only: what this table actually contains.",
            },
            "capability": {
                "type": "string",
                "description": ("escalate only: the missing capability, e.g. 'needs "
                                "figure/plot reading' or 'needs a bin-key legend parser'."),
            },
            "note": {"type": "string", "description": "One short sentence of reasoning."},
        },
        "required": ["kind", "note"],
    },
}


# Evidence in the sample that a table really does carry potency data. Used to
# VETO a `not_assay` verdict, never to create one.
_ASSAY_EVIDENCE = re.compile(
    r"IC\s*50|EC\s*50|\bKi\b|\bKd\b|inhibit|potenc|%\s*inh", re.I)


def _veto_not_assay(kind: str, sample: str) -> tuple[str, str | None]:
    """Refuse a `not_assay` verdict that contradicts the visible text.

    `not_assay` is the only outcome that is both permanent and silent: it is
    cached as a negative rule and suppresses the layout everywhere, forever.
    Measured on 12 real assay tables, Haiku 4.5 wrongly returned `not_assay`
    for 10 of them — inverted bin tables whose own legend reads `IC50*
    Examples ++ A183, ...`. Sonnet got 0 wrong, but no model should be trusted
    with an irreversible, unverifiable "there is nothing here".

    So the veto is deterministic and model-independent: if the sample the model
    just read contains potency language, its dismissal is downgraded to an
    escalation, which is recoverable. The asymmetry is deliberate — a spurious
    queue item costs a human a minute, a wrong suppression costs the database
    real measurements with no trace.
    """
    if kind == NOT_ASSAY and _ASSAY_EVIDENCE.search(sample or ""):
        return ESCALATE, ("model said not_assay but the sample contains potency "
                          "language — downgraded to escalation rather than "
                          "silently suppressing this layout")
    return kind, None


# What each rule kind needs in order to be that kind. Order matters only for a
# response carrying two payloads at once, which is rare and resolved by
# specificity: a bin legend is a stronger signal than a lone `pattern` string.
_PAYLOAD_SIGNATURES = (
    (BIN_KEY, ("scales", "bins")),
    (COLUMN_MAP, ("assay_columns",)),
    (VALUE_PATTERN, ("value_pattern",)),
    (ROW_REGEX, ("pattern",)),
    (NOT_ASSAY, ("because",)),
)

_KNOWN_KINDS = {COLUMN_MAP, ROW_REGEX, VALUE_PATTERN, BIN_KEY, NOT_ASSAY, ESCALATE}


def _infer_kind(out: dict) -> str | None:
    """Which rule this response actually is, judged by what it contains.

    The `kind` enum in the tool schema is ADVISORY — the Messages API does not
    constrain a tool argument to its enum — and our own prompt uses the word
    `scales` as if it named an outcome ("return `scales`, one entry per
    scale"). So Haiku answered `kind: "scales"` on US10172859 and carried a
    flawless three-scale legend underneath: every bound matching the patent
    text, both directions of the opposite-running hERG scale intact. The old
    catch-all wrote that to the library as `escalate / capability:
    unspecified`, and eight tables' worth of data went with it.

    Reading the payload instead of the label is not leniency. It is safe
    precisely because it decides nothing: whatever we route to still faces
    `validate()` against real rows, so a wrong guess becomes a rejection
    carrying evidence — which is strictly more use than an escalation carrying
    none.
    """
    for kind, keys in _PAYLOAD_SIGNATURES:
        if any(out.get(k) for k in keys):
            return kind
    return None


def _to_rule(fingerprint: str, out: dict, model: str, sample: str = "") -> Rule:
    kind = out.get("kind")
    if kind not in _KNOWN_KINDS:
        inferred = _infer_kind(out)
        if inferred is not None:
            logger.info("repair: %s returned kind=%r; routing as %s on payload shape",
                        fingerprint, kind, inferred)
            kind = inferred
    kind, veto = _veto_not_assay(kind, sample)
    if veto:
        return Rule(fingerprint=fingerprint, kind=ESCALATE,
                    payload={"capability": "contested not_assay", "note": veto},
                    source="llm", model=model)
    if kind == COLUMN_MAP:
        payload = {
            "cid": out.get("cid_column"),
            "assays": [
                {"index": a.get("index"), "name": a.get("name"),
                 "unit": (a.get("unit") or None)}
                for a in (out.get("assay_columns") or [])
            ],
        }
    elif kind == ROW_REGEX:
        payload = {"pattern": out.get("pattern", "")}
    elif kind == BIN_KEY:
        def _bins(items):
            return {b["symbol"].upper(): {"lo": b.get("lo"), "hi": b.get("hi"),
                                          "unit": b.get("unit")}
                    for b in (items or []) if b.get("symbol")}

        if out.get("scales"):
            payload = {"scales": [
                {"match": s.get("match") or "", "bins": _bins(s.get("bins"))}
                for s in out["scales"] if s.get("bins")
            ]}
        else:
            payload = {"bins": _bins(out.get("bins"))}
    elif kind == VALUE_PATTERN:
        payload = {"pattern": out.get("value_pattern", ""),
                   "columns": out.get("value_columns") or None}
    elif kind == NOT_ASSAY:
        payload = {"because": out.get("because") or out.get("note", "")}
    else:
        # Nothing recognisable came back. That is a real escalation — but say
        # what arrived. "capability: unspecified" was the single least useful
        # string in the queue, and it was what a correct three-scale legend got
        # recorded as for eight tables.
        kind = ESCALATE
        keys = sorted(k for k, v in out.items() if v not in (None, "", [], {}))
        payload = {
            "capability": out.get("capability")
            or f"response carried no usable rule payload (keys: {keys})",
            "note": out.get("note", ""),
        }
    return Rule(fingerprint=fingerprint, kind=kind, payload=payload,
                source="llm", model=model)


# Characters of raw CALS sent per gap. 24k is ~6k tokens, under a cent at
# Haiku rates, and covers the median block (7.6k) whole. The corpus maximum is
# 428k — one block of 699 rows — so the largest are head-and-tailed: a rule is
# inferred from a table's SHAPE, and the 400th identical row adds nothing the
# first thirty did not.
_RAW_BUDGET = 24_000


def propose(gap: Gap, *, model: str = SYNTH_MODEL, patent_id: str = "") -> Rule | None:
    """One call. Returns an UNVALIDATED proposal — the caller must gate it."""
    # Lead with the diagnosis and the cells that actually failed. Buried at the
    # bottom, this information was ignored: shown a healthy-looking table the
    # model proposed a column_map every time, because nothing on screen told it
    # WHICH cells we could not read.
    failing = ""
    if gap.unparsed_examples:
        failing = (
            "\nCELLS WE COULD NOT READ (the columns are already identified; "
            "our cell parser rejected these):\n  "
            + "\n  ".join(repr(c) for c in gap.unparsed_examples)
            + "\nIf these are plainly measurements, the fault is our cell parser "
              "and the fix is `value_pattern`, not `column_map`.\n")
    legends = ""
    if gap.legend_candidates:
        legends = ("\nTEXT FOUND ELSEWHERE IN THIS PATENT THAT MAY DEFINE THESE "
                   "GRADES:\n  " + "\n  ".join(c[:800] for c in gap.legend_candidates)
                   # Transcription, not adjudication. Asking "is it safe to
                   # encode this legend" got `escalate` every time, with a note
                   # proving the legend had been read correctly. Asking "write
                   # down what it says" got the right answer on the first try.
                   # Phrased for ANY number of scales so nothing here decides
                   # what shape the legend has — one scale in, one scale out.
                   + "\nTRANSCRIBE it into `bin_key`. Do not judge whether "
                     "transcribing is wise; that is already decided. Write down "
                     "every scale the text defines, using `scales` with one "
                     "entry per scale and a `match` regex naming the assay "
                     "columns that scale governs — one entry if it defines one "
                     "scale, three if it defines three. Use `bins` alone only "
                     "when a single scale governs the whole table. Copy each "
                     "bound in the direction the text states it, even where "
                     "scales run opposite ways.\n")
    # The SOURCE goes in the first prompt, not behind a tool call.
    #
    # It used to be offered via `request_more_context` and it was never once
    # taken: zero of 7,472 cached responses contain that call, and for two of
    # the three Gap construction sites `raw_source` was empty anyway, so asking
    # would have returned nothing. A capability nobody used, that mostly did
    # not work.
    #
    # The cost that justified withholding it does not exist. Median raw block
    # is 7,575 characters against a 1,362-character sample — 6x — and sending
    # raw for EVERY gap in the corpus, all 44, costs $0.44 once and is cached
    # by fingerprint forever.
    #
    # What withholding it did cost is the whole class of bug this session was
    # spent on. `gap.sample` is a view WE built; asking a model to diagnose it
    # asks about our parse, not the patent. That is the repo's own rule —
    # never diagnose from the parsed view — applied one level up, and it is
    # why a model answered `not_assay` about an interleaved NMR fragment and
    # why a correct `column_map` was proposed for a table whose real problem
    # was one layer below any rule.
    #
    # Both views are shown, in that order, and the prompt says they must
    # agree. A disagreement between them is a parser defect, which is worth
    # more than any rule the model could return.
    raw = gap.raw_source or ""
    if len(raw) > _RAW_BUDGET:
        head = raw[: _RAW_BUDGET * 2 // 3]
        tail = raw[-(_RAW_BUDGET // 3):]
        raw = (f"{head}\n\n... [{len(gap.raw_source) - _RAW_BUDGET} characters of "
               f"middle rows omitted] ...\n\n{tail}")
    source_block = (
        f"THE PATENT'S OWN XML FOR THIS TABLE — the source of truth:\n"
        f"{raw}\n\n"
        f"OUR READING OF IT — derived, and possibly where the fault is:\n"
        if raw else "OUR READING OF THIS TABLE:\n")
    prompt = (
        f"PROBLEM: {gap.reason}\n{legends}"
        f"{failing}\n"
        f"Table fragment (patent {gap.patent_id}, {gap.n_cols} columns, "
        f"{gap.n_data_rows} data rows).\n"
        f"Columns currently classified as: {gap.column_kinds}\n\n"
        f"{source_block}"
        f"{gap.sample}\n"
    )
    cache_key = f"synth::{PROMPT_VERSION}::{gap.fingerprint}::{model}"
    cached = get_cached(model, cache_key)
    if cached is not None:
        try:
            return _to_rule(gap.fingerprint, json.loads(cached), model, gap.sample)
        except (json.JSONDecodeError, TypeError):
            pass

    if not config.ANTHROPIC_API_KEY:
        logger.info("repair: no API key; cannot synthesize a rule for %s", gap.fingerprint)
        return None

    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Let it look again before it answers.
    #
    # This was a single forced tool call: `tool_choice` pinned to the rule tool
    # over one small fixed sample. A model that judged the sample insufficient
    # had exactly one way to say so — `escalate` — so "I need more" and "this
    # cannot be done" were the same output, and we read every one of them as
    # the latter. On US10172859 it escalated eight times while its own note
    # showed it had understood the table completely.
    #
    # So offer `request_more_context` alongside the rule tool. The model decides
    # whether it has enough, and only then judges. The expanded view is built at
    # detection time and costs nothing to prepare; we pay for it only when asked.
    tools = [TOOL, MORE_TOOL]
    messages: list = [{"role": "user", "content": prompt}]
    out = None
    for turn in range(MAX_TURNS):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=900,
                temperature=0,
                system=[{"type": "text", "text": SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                tool_choice={"type": "any"},
                messages=messages,
            )
        except Exception as e:
            logger.warning("repair: synthesis call failed for %s: %r", gap.fingerprint, e)
            return None

        usage = resp.usage
        cost_tracker.record(usage.input_tokens, usage.output_tokens, model,
                            patent_id=patent_id or gap.patent_id, cost_category="lm")
        logger.info(
            "repair: %s turn=%d in=%d out=%d cached_read=%d",
            gap.fingerprint, turn, usage.input_tokens, usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

        call = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
        if call is None:
            return None
        if call.name != MORE_TOOL["name"]:
            out = call.input
            break

        # It asked. Serve the wider view once; on the last turn it must answer
        # from what it has, and the rule tool is then the only thing offered.
        want = (call.input or {}).get("what") or "everything"
        logger.info("repair: %s asked for more context (%s): %s",
                    gap.fingerprint, want, (call.input or {}).get("why", "")[:120])
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": call.id,
            "content": _more_context(gap, want),
        }]})
        if turn == MAX_TURNS - 2:
            tools = [TOOL]

    if not isinstance(out, dict):
        return None
    store_cached(model, cache_key, json.dumps(out),
                 input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
    return _to_rule(gap.fingerprint, out, model, gap.sample)


def _more_context(gap: Gap, what: str) -> str:
    """Serve the wider view of the gap the model just asked for."""
    parts: list[str] = []
    if what in ("raw_source", "everything"):
        # No truthiness guard. `raw_block` is a lookup for the very `<tables>`
        # element the Table was parsed out of, so in production it is never
        # legitimately empty — there is always something to hand back. Guarding
        # on it turned a wiring bug into a silent no-op: the model asked for the
        # source, got a reply with no source in it, and had no way to tell that
        # from a table with nothing more to say. If it is empty now, that is a
        # defect and it says so, out loud, where the model can report it.
        parts.append(
            "ORIGINAL CALS XML FOR THIS TABLE (the source of truth — everything "
            "else you were shown is derived from this):\n"
            + (gap.raw_source[:6000] if gap.raw_source else
               "!! UNAVAILABLE — this is a bug in the caller, not a property of "
               "the patent. Say so in `escalate`: the source could not be shown.")
            # Truncated from the FRONT of the block: the thead is what
            # disagreements are usually about, and it is the part our merge most
            # often damages. `colspec`/`colwidth` sit there too, and they state
            # outright how a header tgroup maps onto a body tgroup of a
            # different width — which is what US10376513 turned on.
        )
    if what in ("full_legend", "everything") and gap.legend_candidates:
        parts.append("LEGEND TEXT FOUND ELSEWHERE IN THIS PATENT (untruncated):\n  "
                     + "\n  ".join(c for c in gap.legend_candidates))
    if what in ("more_rows", "raw_headers", "surrounding_text", "everything"):
        parts.append("EXPANDED VIEW OF THE SAME TABLE:\n" + (gap.expanded_sample or gap.sample))
    if gap.unparsed_examples:
        parts.append("CELLS WE COULD NOT READ:\n  "
                     + "\n  ".join(str(u) for u in gap.unparsed_examples[:12]))
    parts.append("You now have the wider view. Answer with `propose_extraction_rule`. "
                 "Use `escalate` ONLY if a capability is genuinely missing, not "
                 "because the layout is awkward.")
    return "\n\n".join(parts)


def estimate_tokens(gap: Gap) -> int:
    """Rough input size for one proposal, for budgeting before spending."""
    return (len(SYSTEM) + len(gap.sample) + 300) // 4
