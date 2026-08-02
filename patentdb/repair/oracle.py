"""What SHOULD this table yield? Asked once, of one block, to localise a defect.

For a patent we extract nothing from, the capability tier shows a model one
table and the functions that read it, and asks for a patch. Measured, it
declines: on US10189840 it answered "the example numbers are packed into a
single cell alongside `Ex. No.`" and offered no code, and on US11059825 "the
IDs are embedded in NMR prose rather than a dedicated column". Both diagnoses
are correct. Neither is a patch, because the fix is not a single function and
the model has no way to check itself.

So ask a different question first — not "fix this" but "what is IN here".

An ORACLE is the rows a model reads out of the raw CALS of ONE block. It gives
three things nothing else can:

  PROOF        the block really does hold measurements. "We extract nothing"
               and "there is nothing here" look identical from our side, and
               have been confused repeatedly. A patent with no assay data is
               not a defect and must stop costing verification runs.
  A TARGET     the patch is no longer judged on corpus coverage alone. It has
               rows it must reproduce, which is a far easier question than
               "is this code correct".
  THE DIAGNOSIS  diffing the oracle against our records says exactly where the
               pipeline drops them — no id, no value, no unit, or no row at all.

**AN ORACLE IS A TEST FIXTURE, NEVER EXTRACTION OUTPUT.** Not one of these rows
reaches `example_index.json` or any database. That is the same line the rule
tier draws — ask for a rule, never for data — and it is drawn here for the same
reason: a model's reading of a table is unverifiable, and unverifiable data that
looks like a measurement is worse than a missing one. What the oracle is allowed
to do is say a patch is wrong.

Cost is bounded by construction: one block per patent, only for patents that
yield NOTHING, cached by layout fingerprint so a shape is read once ever.
Measured at ~4-8k input tokens — cents. The expensive part of this tier was
never the tokens; it is `verify_patch` re-extracting 103 patents and running the
suite per candidate, and an oracle is what lets us reject a hopeless patch
before paying for that.
"""
from __future__ import annotations

import json
import logging
import re

from ..core import config

logger = logging.getLogger(__name__)

# Bump when the prompt or the schema changes — same discipline as SYNTH_EPOCH
# and PATCH_EPOCH. A stale answer to a question we no longer ask reads as the
# model being wrong.
ORACLE_EPOCH = "v3-bounded-rows"

# Enough of the block to read, not so much that a 1,700-row table is billed in
# full. The head is where the header and the first data rows live, which is all
# a reader needs to establish the pattern.
_RAW_BUDGET = 14_000

# An oracle is a TARGET, not an extraction: thirty rows prove the data is there
# and give a patch something to reproduce, and the thirty-first proves nothing
# further. Bounding the ASK is what keeps the answer inside `_MAX_TOKENS`.
_MAX_ROWS = 30
# 4,096 was not enough. The first real call ran out mid-`rows` on a 94-row
# table, the tool JSON truncated after its first key, and the caller read that
# as "this block contains nothing" — a truncated answer and an empty table are
# the same object unless someone checks. `stop_reason` is now checked; this
# ceiling is set so it should not be reached.
_MAX_TOKENS = 16_000

_TOOL = {
    "name": "report_rows",
    "description": "Report the assay measurements this table block contains.",
    "input_schema": {
        "type": "object",
        "properties": {
            "holds_assay_data": {
                "type": "boolean",
                "description": ("False if this block is not assay data at all — "
                                "NMR, mass spec, retention times, formulations, "
                                "crystallography. Say so plainly; a block with no "
                                "measurements is not a defect."),
            },
            "rows": {
                "type": "array",
                "description": (f"One entry per measurement actually printed in "
                                f"the block shown, up to {_MAX_ROWS} — this is a "
                                f"TARGET for a parser, not an extraction, so a "
                                f"representative sample of the first rows is "
                                f"exactly as useful as all of them. Do NOT infer, "
                                f"interpolate or continue a pattern beyond what "
                                f"is written."),
                "items": {
                    "type": "object",
                    "properties": {
                        "cid": {"type": "string",
                                "description": "compound identifier as printed"},
                        "assay": {"type": "string",
                                  "description": "assay name from the header"},
                        "value": {"type": "string",
                                  "description": ("the value exactly as printed. "
                                                  "An ORDINAL GRADE — `A`, `B`, "
                                                  "`+++`, `**` — IS a value: it "
                                                  "is a potency band whose legend "
                                                  "sits elsewhere in the patent. "
                                                  "Report it as printed.")},
                        "unit": {"type": "string",
                                 "description": "unit, or '' if the block states none"},
                        "n_runs": {"type": "string",
                                   "description": "replicate count if printed, else ''"},
                    },
                    "required": ["cid", "assay", "value"],
                },
            },
            "why_we_might_miss_them": {
                "type": "string",
                "description": ("One or two sentences: what is structurally "
                                "unusual about how this block encodes its rows. "
                                "If you are returning NO rows, this must say "
                                "why — an empty answer with no reason is "
                                "indistinguishable from an empty table."),
            },
        },
        # `why` is REQUIRED. It was optional, and the first real call came back
        # `holds_assay_data: true` with zero rows and no explanation — for a
        # block holding 40 graded compounds. A silent empty answer reads exactly
        # like a block with nothing in it, which is the one confusion this whole
        # module exists to end.
        "required": ["holds_assay_data", "rows", "why_we_might_miss_them"],
    },
}

_SYSTEM = """You are reading ONE table out of a patent, in its original OASIS/CALS XML.

Report only what is PRINTED. Never infer a value, never continue a numbering
pattern, never carry a unit from your own knowledge of the assay. If a compound
id is written once and the rows beneath it are blank, those rows belong to that
same compound — say so by repeating the id.

A VALUE IS NOT ALWAYS A NUMBER. Patents routinely grade potency in ordinal
bands — `A`/`B`/`C`, `+`/`++`/`+++`, `*`/`**` — and define them in a legend that
may sit anywhere in the document, often nowhere near the table. A column of
those letters IS assay data and each letter IS a value: report it as printed and
leave `unit` empty. Returning no rows because the cells "have no numbers" is the
single most likely way to be wrong here.

If the block is not assay data — NMR shifts, mass spec, retention times, a
formulation recipe, crystallographic parameters — set `holds_assay_data` false
and return no rows. That answer is as useful as any other and costs nothing to
be wrong about.

Your rows are a TEST FIXTURE. They will be used to check whether our parser is
broken, and they will never be stored as data. Accuracy on a few rows is worth
far more than volume."""


def _raw_block(xml: str, table_id: str) -> str:
    m = re.search(r"<tables\b[^>]*id=\"" + re.escape(table_id) + r"\".*?</tables>",
                  xml, re.S)
    if not m:
        return ""
    raw = m.group(0)
    return raw if len(raw) <= _RAW_BUDGET else raw[:_RAW_BUDGET] + "\n<!-- truncated -->"


def read_block(patent_id: str, table_id: str, xml: str, fingerprint: str,
               *, model: str | None = None) -> dict | None:
    """Ask a model what one block contains. Cached by layout fingerprint."""
    import anthropic

    from ..core.api_cache import get_cached, store_cached
    from ..core.cost_tracker import cost_tracker

    model = model or config.MODEL_SONNET
    raw = _raw_block(xml, table_id)
    if not raw:
        return None

    key = f"oracle::{ORACLE_EPOCH}::{fingerprint}::{model}"
    cached = get_cached(model, key)
    if cached is not None:
        try:
            return json.loads(cached)
        except ValueError:
            pass

    if not config.ANTHROPIC_API_KEY:
        return None
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        resp = client.messages.create(
            model=model, max_tokens=_MAX_TOKENS, system=_SYSTEM,
            tools=[_TOOL], tool_choice={"type": "tool", "name": "report_rows"},
            messages=[{"role": "user", "content":
                       f"Patent {patent_id}, block {table_id}:\n\n{raw}"}],
        )
    except Exception as e:
        logger.warning("oracle: %s %s failed: %r", patent_id, table_id, e)
        return None
    cost_tracker.record(resp.usage.input_tokens, resp.usage.output_tokens,
                        model, patent_id=patent_id)
    # A TRUNCATED answer is not an empty table. The tool JSON is emitted as a
    # stream, so hitting the ceiling leaves a partial object that parses fine
    # and reports whatever keys arrived first — on the first real call, exactly
    # `{"holds_assay_data": true}` for a block holding 40 graded compounds.
    # Read as data that says "there is nothing here", which is the one answer
    # this module exists to distinguish from "we cannot read it".
    if resp.stop_reason == "max_tokens":
        logger.warning("oracle: %s %s hit the output ceiling (%d tokens) — "
                       "discarding the partial answer rather than reading it "
                       "as an empty block", patent_id, table_id, _MAX_TOKENS)
        return None
    call = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
    if call is None:
        return None
    out = dict(call.input)
    if out.get("holds_assay_data") and not (out.get("rows") or []):
        logger.warning("oracle: %s %s says it holds assay data and returned no "
                       "rows — treating as no answer. why=%r", patent_id,
                       table_id, str(out.get("why_we_might_miss_them"))[:160])
        return None
    store_cached(model, key, json.dumps(out),
                 input_tokens=resp.usage.input_tokens,
                 output_tokens=resp.usage.output_tokens)
    return out


def _norm(s: str) -> str:
    return re.sub(r"[\s.\-–—]", "", (s or "").strip().lower())


def diff(oracle: dict, records, table_id: str) -> dict:
    """Where the pipeline loses the rows the oracle says are there.

    Reported per FIELD, because that is what names the function to fix: a row
    we never produced is a different defect from one produced without a unit,
    and they live in different places.
    """
    rows = oracle.get("rows") or []
    ours = [r for r in records if r.table_id == table_id]
    by_cid: dict[str, list] = {}
    for r in ours:
        by_cid.setdefault(_norm(r.cid or ""), []).append(r)

    missing_row, missing_value, missing_unit, present = [], [], [], []
    for row in rows:
        got = by_cid.get(_norm(row.get("cid", "")))
        if not got:
            missing_row.append(row)
            continue
        if not any(g.value_numeric is not None or g.range_lo is not None
                   for g in got):
            missing_value.append(row)
        elif not any(g.unit for g in got):
            missing_unit.append(row)
        else:
            present.append(row)
    return {
        "oracle_rows": len(rows),
        "we_produced": len(ours),
        "matched": len(present),
        "missing_row": missing_row[:8],
        "missing_value": missing_value[:8],
        "missing_unit": missing_unit[:8],
        "n_missing_row": len(missing_row),
        "n_missing_value": len(missing_value),
        "n_missing_unit": len(missing_unit),
    }


def target_note(oracle: dict, d: dict) -> str:
    """The oracle and the diff, rendered for the patch prompt."""
    if not oracle.get("holds_assay_data"):
        return ("An independent read of this block says it holds NO assay data "
                f"({oracle.get('why_we_might_miss_them', '')}). If that is right, "
                "the correct answer is that there is nothing to fix here.\n\n")
    lines = [
        f"AN INDEPENDENT READ of this block found {d['oracle_rows']} measurement(s) "
        f"where our parser produced {d['we_produced']} record(s). "
        f"{d['matched']} agree.",
        f"  rows we never produced at all : {d['n_missing_row']}",
        f"  produced with no value        : {d['n_missing_value']}",
        f"  produced with no unit         : {d['n_missing_unit']}",
    ]
    if oracle.get("why_we_might_miss_them"):
        lines.append(f"  it noted: {oracle['why_we_might_miss_them'][:300]}")
    sample = (d["missing_row"] or d["missing_value"] or d["missing_unit"])[:6]
    if sample:
        lines.append("\nRows the patch MUST produce (these are printed in the "
                     "block above — they are a target, not data to trust):")
        for r in sample:
            lines.append(f"  {r.get('cid')} | {r.get('assay')} | "
                         f"{r.get('value')} {r.get('unit', '')}".rstrip())
    lines.append("")
    return "\n".join(lines) + "\n"
