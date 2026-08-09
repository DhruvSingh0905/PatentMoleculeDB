"""Ask a model for a name-repair RULE, never for the corrected name.

The `repair/synthesize.py` of the name tier. Same contract, same reason.

WHY A RULE AND NOT THE ANSWER
------------------------------
Asking "what is the correct name for `...1,2-dihydropuridin-3-yl...`" asks for
DATA. The model will answer, fluently, and nothing about the answer is
checkable — a plausible name that parses is exactly what this pipeline's whole
gate structure exists to reject. Asking for a TRANSFORMATION gives something
that can be ground, run, measured, and reused: `s/puridin/pyridin/` is
verifiable against the document and then applies to every other occurrence for
free.

The economics were measured before this file was written, by running
`name_repair`'s six hand-written patterns over all 137 patents:

    pattern                  confirms  patents
    stray_opening_bracket          75       43
    stray_closing_bracket          70       49
    dropped_close_bracket           6        5
    digit_for_paren                 2        1

4 rules serve 98 patent-instances. Buying the answer instead would have been
153 separate purchases, none reusable.

BATCHING IS THE COST LEVER
---------------------------
One call per failing NAME would pay for the frozen system prefix once per
compound. Failures cluster inside a document — one typesetting defect produces
many corrupted names — so the model is shown up to `BATCH` failures from the
SAME patent in one call and asked for a rule that explains as many as it can.
That makes the unit of purchase the defect FORM rather than the compound, which
is the same shift the table tier makes by keying on layout rather than patent.

`BATCH`, `MAX_TURNS`, `MAX_TOKENS` and `SHOW_CLEAN` are module constants rather
than buried literals specifically so `repair/name_harness.py` can sweep them and
report cost against yield.
"""
from __future__ import annotations

import hashlib
import json
import logging

from ..core import config
from ..core.api_cache import get_cached, store_cached
from ..core.cost_tracker import cost_tracker
from .name_gap import NameGap
from .name_rules import NAME_SYNTH_EPOCH, NamePattern

logger = logging.getLogger(__name__)

SYNTH_MODEL = config.SYNTH_MODEL

# Bump on ANY change to SYSTEM, TOOL, or the rendering below — the cache key
# carries it. Defined in `name_rules` beside the library it also governs, for
# the same reason `synthesize.PROMPT_VERSION` is `rules.SYNTH_EPOCH`: two
# version keys for one question is a drift hazard.
PROMPT_VERSION = NAME_SYNTH_EPOCH

# ── tunable knobs (swept by name_harness.py) ────────────────────────────────
BATCH = 8              # failing names shown per call
MAX_TURNS = 2          # model turns per batch; 2 = ask, answer
MAX_TOKENS = 700
SHOW_CLEAN = 3         # already-parsing names shown as contrast

# Frozen prefix — identical on every call so it prompt-caches (reads bill at
# ~0.1x). Everything patent-specific goes in the user turn.
SYSTEM = """You are shown compound names copied out of a US patent's own \
headings. The patent states, in its own voice, that each names a real compound \
of its own — `Example 43: <name>` — but our IUPAC parser (OPSIN) rejects them.

The patent's grant XML is typeset text, and it is corrupted in small, specific \
ways: a bracket dropped or doubled, two letters substituted, a space injected \
mid-token where a printed line wrapped, a superscript footnote digit welded \
onto the end of a word when its tag was stripped.

YOUR JOB IS TO RETURN A RULE, NOT A NAME. Never write out what you think the \
compound is called. Return a regular expression that locates the corruption, \
and the short literal string that should replace what it matches. We then run \
that rule ourselves.

A rule is judged by running it. THREE conditions, all measured, none of them \
an opinion:

  1. OPSIN parses the repaired name.
  2. The repaired name still covers at least 90% of the name the patent \
printed. This exists because most truncations of a valid chemical name are \
themselves valid names — deleting `-2-yl` from `pyridin-2-yl` yields `pyridin`, \
which parses as pyridine, a DIFFERENT MOLECULE. Measured on this corpus, \
trimming characters off the ends made 11,446 rejected names parse, and \
essentially all of them were amputations of this kind. A rule that shortens a \
name into a smaller molecule fails.
  3. The repaired fragment appears VERBATIM somewhere else in this same patent. \
A corrupted name is usually spelled correctly elsewhere in the document, and \
that other occurrence is what confirms your correction is the right one rather \
than merely a spelling that happens to parse. If you cannot see the correct \
form in the text you were given, your rule will not confirm.

Your rule is ALSO checked, before it runs, against the names in this patent \
that already parse correctly. A pattern that rewrites more than a couple of \
percent of them is rejected outright, however well it fixes the broken one — \
so anchor it to the corruption, not to a common substring.

Prefer the SMALLEST edit that explains the failure. Your replacement may be at \
most 4 characters; a longer one means you are writing chemistry into the name, \
which is data, and we do not buy data.

Look for the form these names have IN COMMON. You are shown several failures \
from one document because one typesetting defect usually produces many \
corrupted names, and a rule covering five of them is worth five times a rule \
covering one. If they genuinely share nothing, write the rule for the clearest \
single case.

There is no option here for "unsure" and no penalty for being wrong: a wrong \
rule costs one more attempt and is told exactly what it did, while a missing \
rule costs the compound permanently."""

TOOL = {
    "name": "propose_name_repair",
    "description": (
        "A regex that locates a text corruption in a compound name, and the "
        "literal string that replaces what it matches."),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": ("short snake_case name for the corruption form, "
                                "e.g. 'doubled_close_paren' or 'u_for_y_in_ring'"),
            },
            "site": {
                "type": "string",
                "description": (
                    "Python regex matched against the WHOLE name. What it "
                    "matches is what gets replaced, so keep it tight — use "
                    "lookaround for context you need to see but not consume. "
                    r"Example: 'puridin' or '(?<=[a-z])\\d$' or '\\)\\)(?=[a-z])'."),
            },
            "replacement": {
                "type": "string",
                "description": ("Literal text the matched span becomes. Empty "
                                "string deletes it. At most 4 characters."),
            },
            "covers": {
                "type": "integer",
                "description": "How many of the names shown you think this fixes.",
            },
            "note": {"type": "string",
                     "description": "One short sentence: what the corruption is."},
        },
        "required": ["id", "site", "replacement", "note"],
    },
}


def _render(gaps: list[NameGap], digest: str, attempts: list[dict]) -> str:
    """The user turn. Everything patent-specific lives here so SYSTEM caches."""
    g0 = gaps[0]
    lines = [f"PATENT {g0.patent_id} — {len(gaps)} compounds it names that we "
             f"cannot parse.\n"]

    for i, g in enumerate(gaps, 1):
        lines.append(f"  [{i}] compound id {g.cid}")
        lines.append(f"      name as printed: {g.name_text}")
        if g.opsin_error:
            # OPSIN's OWN diagnosis, verbatim. It distinguishes "unmatched
            # opening bracket" from "uninterpretable" from "unable to assign
            # all locants" — three different repairs — and it is the one party
            # that actually knows why the parse failed.
            lines.append(f"      OPSIN says: {g.opsin_error}")
    lines.append("")

    if SHOW_CLEAN and g0.clean_names:
        lines.append("NAMES FROM THIS SAME PATENT THAT PARSE CORRECTLY — the "
                     "corrupted ones above should end up looking like these:")
        for n in g0.clean_names[:SHOW_CLEAN]:
            lines.append(f"  {n[:150]}")
        lines.append("")

    if digest:
        lines.append(digest)

    if attempts:
        lines.append(
            f"\nYOU HAVE ALREADY TRIED {len(attempts)} RULE(S) ON THESE NAMES. "
            f"Each was run and measured. Do not repeat one that failed:")
        for i, a in enumerate(attempts, 1):
            lines.append(f"\n  ATTEMPT {i} — s/{a.get('site')}/{a.get('replacement')}/")
            lines.append(f"    what it did: {a.get('why')}")
        lines.append(
            "\nThat is a measurement of your rule, not an opinion about it. "
            "Diagnose it yourself and answer again.\n")

    return "\n".join(lines)


def propose(gaps: list[NameGap], *, model: str = SYNTH_MODEL,
            attempts: list[dict] | None = None,
            digest: str = "") -> NamePattern | None:
    """One call. Returns an UNGATED proposal — the caller grounds and measures it.

    Returns `None` when there is no API key, so `repair_names` degrades to
    "apply the library, adopt nothing new" rather than crashing — the same
    tolerance `synthesize.propose` has.
    """
    if not gaps:
        return None
    attempts = attempts or []
    prompt = _render(gaps, digest, attempts)

    # The attempt history is part of the QUESTION, so it is part of the key —
    # without it every retry reads attempt one's answer back at $0 and the loop
    # spins, which looks exactly like a model that will not change its mind.
    # The gap set is keyed by CONTENT so two patents with identical failures
    # share one purchase.
    body = json.dumps([g.name_text for g in gaps], sort_keys=True)
    key_src = body + json.dumps([a.get("why") for a in attempts], default=str)
    cache_key = (f"namesynth::{PROMPT_VERSION}::{model}::"
                 f"{hashlib.sha256(key_src.encode()).hexdigest()[:16]}")
    cached = get_cached(model, cache_key)
    if cached is not None:
        try:
            return _to_pattern(json.loads(cached), model, gaps[0].patent_id)
        except (json.JSONDecodeError, TypeError):
            pass

    if not config.ANTHROPIC_API_KEY:
        logger.info("name_synth: no API key; cannot synthesize a name rule")
        return None

    from ..core.api_client import resilient_client
    client = resilient_client()

    messages: list = [{"role": "user", "content": prompt}]
    out = None
    usage = None
    for turn in range(MAX_TURNS):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=0,
                system=[{"type": "text", "text": SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                tools=[TOOL],
                tool_choice={"type": "any"},
                messages=messages,
            )
        except Exception as e:
            logger.warning("name_synth: call failed for %s: %r",
                           gaps[0].patent_id, e)
            return None

        usage = resp.usage
        # Cache buckets are SEPARATE from `input_tokens` and are billed — reads
        # at 0.1x, writes at 1.25x. Passing only `input_tokens` makes every
        # cached token free in our accounting while costing real money, and a
        # spend cap cannot trip on tokens it never sees.
        cost_tracker.record(
            usage.input_tokens, usage.output_tokens, model,
            patent_id=gaps[0].patent_id, cost_category="lm",
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0)
        logger.info("name_synth: %s turn=%d in=%d out=%d cached_read=%d",
                    gaps[0].patent_id, turn, usage.input_tokens,
                    usage.output_tokens,
                    getattr(usage, "cache_read_input_tokens", 0) or 0)

        call = next((b for b in resp.content
                     if getattr(b, "type", "") == "tool_use"), None)
        if call is None:
            return None
        out = call.input
        break

    if not isinstance(out, dict) or usage is None:
        return None
    store_cached(model, cache_key, json.dumps(out),
                 input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
    return _to_pattern(out, model, gaps[0].patent_id)


def _to_pattern(out: dict, model: str, patent_id: str) -> NamePattern | None:
    site = (out.get("site") or "").strip()
    if not site:
        return None
    pid = (out.get("id") or "synth").strip().replace(" ", "_")[:48] or "synth"
    return NamePattern(
        id=f"{pid}",
        site=site,
        replacement=out.get("replacement") or "",
        note=(out.get("note") or "")[:200],
        model=model,
        epoch=PROMPT_VERSION,
        bought_for=patent_id,
    )
