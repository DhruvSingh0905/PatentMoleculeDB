"""Model-authored REPAIR CODE, for failure classes no regex could express.

WHY THIS TIER EXISTS, AND WHY IT DID NOT BEFORE
------------------------------------------------
`name_rules` buys a strict subset — one site regex, one replacement template —
and its docstring states the reason plainly: "a synthesized rule has to be
checkable by `ground()` without executing model-authored code". That was a
deliberate limit, and it holds for most defects: a character substitution, a
stray bracket, a footnote digit are all regex-shaped.

It does not hold for the residue. Measured over the 30-patent sample after the
escalation budget was fixed, 604 gaps produced 71 repairs and left
`unmatched_opening_bracket` (107) and `unmatched_closing_bracket` (71) — **178
gaps, 30% of the corpus** — with no rule after three genuine attempts each. The
shape is already documented one module over: `name_repair.dropped_close_bracket`
"tries a ')' at every offset in a 60-char window, WHICH NO REGEX EXPRESSES".
A tier that can only propose regexes cannot reach that, however many times it
is asked, so the attempts are spent rather than invested.

So this tier lifts the constraint for exactly those classes, and pays for it
with gates rather than with trust.

WHAT THE MODEL IS ALLOWED TO WRITE
------------------------------------
The SAME interface the seven hand-written patterns already implement, and
nothing else:

    def site(name: str) -> list[tuple[int, int]]   # candidate spans
    def fix(span: str)  -> list[str]               # candidate replacements

That is the whole surface. It is enough for the bracket case — `site` can walk
the string keeping a depth counter, which is precisely what a regex cannot —
and it is small enough to check. There is no import, no file access, no
network: the sandbox rejects the source before it runs if it contains an
import statement, an attribute access on a dunder, or a call to `eval`/`exec`/
`open`/`compile`/`__import__`.

WHY IT IS DATA, NOT A SOURCE PATCH
------------------------------------
v2 patched `sources/*.py` directly (`repair/capability.py`, `MAX_TARGETS=3`,
a Sonnet->Opus ladder). That is strictly more powerful and strictly less
recoverable: reverting means `git checkout`, which takes uncommitted work with
it, and a bad patch can break an import so that nothing runs at all. Here the
pattern is stored as source TEXT in the rule library, adopted or not adopted,
and reverted by deleting one entry. The pipeline's own files are never
modified, so a catastrophic proposal costs a rejected adoption rather than a
broken tree.

THE GATES, IN ORDER, ALL OF THEM BEFORE ADOPTION
--------------------------------------------------
  1. STATIC   the source parses, defines exactly `site` and `fix`, and
              contains no import / dunder / eval / exec / open.
  2. SANDBOX  it runs in a SUBPROCESS with a hard timeout. A model-authored
              `while True` is a hang, and a hang in-process would take the
              corpus run with it.
  3. CONTRACT `site` returns in-bounds, non-inverted spans; `fix` returns
              strings. A pattern that returns garbage is rejected here rather
              than corrupting a name downstream.
  4. COLLATERAL it must not alter ANY name that already parses. This is the
              check that cannot be done after the fact — a pattern that
              mangles 100 good names while fixing 1 still shows a confirmed
              repair for that 1. The population is this patent's own resolved
              names.
  5. OUTCOME  it must actually recover at least one escalated gap, with OPSIN
              accepting the result and the coverage floor met — the same
              `name_outcome` gate a bought regex faces, unchanged.

A pattern that fails any gate is journalled with the reason and discarded.
Nothing here decides whether a repair is a GOOD idea; every gate is a
measurement.
"""
from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config
from .name_gap import NameGap

logger = logging.getLogger(__name__)

# Bump on any change to the prompt or to this contract, exactly as
# `NAME_SYNTH_EPOCH` does for the regex tier.
CAPABILITY_EPOCH = "c1"

# Wall-clock ceiling for one sandboxed evaluation of a whole batch of names.
# A model-authored scan is O(len(name)) per name in every sane implementation;
# anything that needs longer is looping.
SANDBOX_TIMEOUT_S = 20

# Classes worth this tier at all. A code patch is the expensive answer, so it
# is offered only where the cheap one provably failed.
MAX_TARGETS = 3

# Builtins that make model-authored source unsafe to run even in a subprocess.
# CHECKED AGAINST THE PARSE TREE, NEVER AGAINST THE TEXT. The first version of
# this gate was a substring scan of the lowercased source, and it rejected FOUR
# OF SIX real proposals for containing "open" — as `open_count`, `open_pos`,
# `opening_bracket`. That is the natural vocabulary of bracket balancing, which
# is the one thing this tier was built to reach, so the gate was refusing
# exactly the code it was meant to admit and the run looked like the model
# could not write a bracket fixer.
#
# An AST check says what was actually meant: is `open` CALLED, or is it part of
# an identifier. The sandbox's builtins whitelist is the real containment
# anyway — this is the readable early refusal, and it has to be precise or it
# is worse than nothing.
_FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "open", "compile", "globals", "locals", "getattr",
    "setattr", "delattr", "input", "breakpoint", "vars", "dir", "memoryview"})


@dataclass
class CodePattern:
    """Model-authored `site`/`fix` source for one failure class."""
    id: str
    source: str                     # the Python, verbatim, as adopted
    note: str = ""
    model: str = ""
    epoch: str = CAPABILITY_EPOCH
    bought_for: str = ""
    confirms: int = 0
    signatures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "note": self.note,
                "model": self.model, "epoch": self.epoch,
                "bought_for": self.bought_for, "confirms": self.confirms,
                "signatures": self.signatures}

    @staticmethod
    def from_dict(d: dict) -> "CodePattern":
        return CodePattern(
            id=d["id"], source=d["source"], note=d.get("note", ""),
            model=d.get("model", ""), epoch=d.get("epoch", ""),
            bought_for=d.get("bought_for", ""),
            confirms=int(d.get("confirms", 0)),
            signatures=list(d.get("signatures", []) or []))


# ── gate 1: static ───────────────────────────────────────────────────────

def static_check(source: str) -> str:
    """`""` if the source is admissible, else the reason it is not."""
    if not source.strip():
        return "empty source"
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"source does not parse: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "source imports a module"
        # A NAME USED AS A VALUE, not a name that merely contains the word.
        # `open_count` is an identifier; `open(...)` is a call; `f = open` is a
        # reference. Only the last two are refused.
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_CALLS:
            return f"source references the builtin {node.id!r}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"source touches the dunder attribute {node.attr!r}"

    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if funcs != {"site", "fix"}:
        return (f"source must define exactly `site` and `fix`, "
                f"found {sorted(funcs) or 'nothing'}")
    if any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        return "source may contain nothing but the two function definitions"
    return ""


# ── gate 2+3: sandbox and contract ───────────────────────────────────────

_DRIVER = r'''
import json, sys
src = open(sys.argv[1]).read()
names = json.load(open(sys.argv[2]))
ns = {}
exec(compile(src, "<pattern>", "exec"), {"__builtins__": {
    "len": len, "range": range, "enumerate": enumerate, "list": list,
    "str": str, "int": int, "min": min, "max": max, "abs": abs,
    "sorted": sorted, "reversed": reversed, "any": any, "all": all,
    "tuple": tuple, "set": set, "dict": dict, "zip": zip, "True": True,
    "False": False, "None": None}}, ns)
site, fix = ns["site"], ns["fix"]
out = []
for n in names:
    spans = site(n)
    cands = []
    for (a, b) in spans:
        for rep in fix(n[a:b]):
            cands.append([a, b, rep])
    out.append(cands)
json.dump(out, open(sys.argv[3], "w"))
'''


def run_sandboxed(source: str, names: list[str]) -> tuple[list[list] | None, str]:
    """Apply the pattern to `names` in a SUBPROCESS. `(results, error)`.

    A subprocess rather than an in-process `exec` because the failure mode
    being defended against is a hang, and an in-process hang takes the corpus
    run with it. The timeout is the only thing that can stop `while True`.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "p.py").write_text(source)
        (d / "n.json").write_text(json.dumps(names))
        (d / "drv.py").write_text(_DRIVER)
        try:
            r = subprocess.run(
                [sys.executable, str(d / "drv.py"), str(d / "p.py"),
                 str(d / "n.json"), str(d / "o.json")],
                capture_output=True, text=True, timeout=SANDBOX_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None, f"pattern did not finish in {SANDBOX_TIMEOUT_S}s"
        if r.returncode != 0:
            return None, f"pattern raised: {(r.stderr or '').strip()[:200]}"
        try:
            res = json.loads((d / "o.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            return None, f"pattern produced no readable output: {e}"

    if not isinstance(res, list) or len(res) != len(names):
        return None, "pattern did not return one result per name"
    for cands, name in zip(res, names):
        if not isinstance(cands, list):
            return None, "pattern returned a non-list for a name"
        for c in cands:
            if (not isinstance(c, list) or len(c) != 3
                    or not isinstance(c[2], str)):
                return None, "a candidate is not (start, end, replacement:str)"
            a, b = c[0], c[1]
            if not isinstance(a, int) or not isinstance(b, int):
                return None, "a span bound is not an integer"
            if a < 0 or b > len(name) or a > b:
                return None, f"span ({a},{b}) is out of bounds for this name"
    return res, ""


def apply_pattern(source: str, names: list[str]) -> tuple[list[list[str]], str]:
    """Candidate repaired STRINGS per name. `([[cand, ...], ...], error)`."""
    res, err = run_sandboxed(source, names)
    if res is None:
        return [], err
    out: list[list[str]] = []
    for cands, name in zip(res, names):
        seen: list[str] = []
        for a, b, rep in cands:
            cand = name[:a] + rep + name[b:]
            if cand != name and cand not in seen:
                seen.append(cand)
        out.append(seen)
    return out, ""


# ── gate 4: collateral ───────────────────────────────────────────────────

def collateral(source: str, clean_names: list[str]) -> tuple[int, str]:
    """How many ALREADY-PARSING names this pattern would alter.

    Must be zero. A pattern that damages good names while fixing a broken one
    still shows a confirmed repair for the broken one, so this cannot be
    measured from the output — it has to be blocked before the fact. Same
    reasoning as `name_rules.ground`'s `_MAX_COLLATERAL`, held to a stricter
    bound because this tier's patterns are not constrained to a single site.
    """
    if not clean_names:
        return 0, ""
    cands, err = apply_pattern(source, clean_names)
    if err:
        return -1, err
    return sum(1 for c in cands if c), ""


# ── the ask ──────────────────────────────────────────────────────────────

SYSTEM = """You are repairing chemical compound NAMES copied out of US patent \
XML. The text is damaged by typesetting and OCR, never by the chemistry being \
wrong.

You are being asked ONLY because a simple find-and-replace rule has already \
been tried on this failure class and could not express the repair. Write Python \
instead.

Define EXACTLY two functions and nothing else:

    def site(name: str) -> list[tuple[int, int]]:
        # every span in `name` that might be the damage, left to right
    def fix(span: str) -> list[str]:
        # candidate replacements for one such span, best first

Rules that are checked mechanically and will reject your answer:
  - no imports, no dunder attributes, no eval/exec/open/getattr
  - `site` returns in-bounds (start, end) pairs; `fix` returns strings
  - it must not alter any name that is ALREADY correct — you are shown \
examples of correct names from the same document, and a pattern that touches \
one is discarded
  - repair the TYPESETTING. Never add or remove chemistry to make a name \
parse: deleting a substituent makes a smaller molecule parse, which is a \
wrong answer that looks like a right one.

You may scan the string, count bracket depth, look at neighbouring characters \
— anything a regex cannot do. That freedom is the entire reason you are here."""

TOOL = {
    "name": "propose_code_pattern",
    "description": "Propose Python site/fix functions for this failure class.",
    "input_schema": {
        "type": "object",
        "required": ["id", "source", "note"],
        "properties": {
            "id": {"type": "string",
                   "description": "short snake_case id for this repair"},
            "source": {"type": "string",
                       "description": "the two function definitions, verbatim"},
            "note": {"type": "string",
                     "description": "one sentence: what the damage is"},
        },
    },
}


def _render(gaps, signature: str, clean_names: list[str],
            attempts: list[dict]) -> str:
    lines = [f"FAILURE CLASS: {signature}",
             f"{len(gaps)} compound name(s) in this class could not be parsed.",
             ""]
    for g in gaps[:6]:
        lines.append(f"  name  : {g.name_text[:300]}")
        if g.opsin_error:
            lines.append(f"  parser: {g.opsin_error[:200]}")
        lines.append("")
    if clean_names:
        lines.append("Names from the SAME document that are already correct — "
                     "your pattern must not alter any of these:")
        for c in clean_names[:5]:
            lines.append(f"  {c[:200]}")
        lines.append("")
    for a in attempts:
        lines.append(f"REJECTED: {a.get('id', '?')} — {a.get('why', '')}")
    return "\n".join(lines)


def propose(gaps, signature: str, *, clean_names=None, attempts=None,
            model: str = "") -> CodePattern | None:
    """One call. Returns an UNGATED proposal — the caller runs every gate."""
    import hashlib
    from ..core.api_cache import get_cached, store_cached
    from ..core.cost_tracker import cost_tracker

    if not gaps:
        return None
    model = model or config.CAPABILITY_MODEL
    prompt = _render(gaps, signature, clean_names or [], attempts or [])
    key = (f"namecap::{CAPABILITY_EPOCH}::{model}::"
           f"{hashlib.sha256(prompt.encode()).hexdigest()[:16]}")
    cached = get_cached(model, key)
    if cached is not None:
        try:
            return CodePattern.from_dict(json.loads(cached))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    if not config.ANTHROPIC_API_KEY:
        logger.info("name_cap: no API key; cannot author a code pattern")
        return None

    from ..core.api_client import resilient_client
    try:
        resp = resilient_client().messages.create(
            model=model, max_tokens=2000, temperature=0,
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[TOOL], tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}])
    except Exception as e:
        logger.warning("name_cap: call failed: %r", e)
        return None

    u = resp.usage
    cost_tracker.record(
        u.input_tokens, u.output_tokens, model,
        patent_id=gaps[0].patent_id, cost_category="lm",
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)
    call = next((b for b in resp.content
                 if getattr(b, "type", "") == "tool_use"), None)
    if call is None or not isinstance(call.input, dict):
        return None
    p = CodePattern(
        id=(call.input.get("id") or "code").strip().replace(" ", "_")[:48],
        source=call.input.get("source") or "",
        note=(call.input.get("note") or "")[:200],
        model=model, bought_for=gaps[0].patent_id)
    store_cached(model, key, json.dumps(p.to_dict()),
                 input_tokens=u.input_tokens, output_tokens=u.output_tokens)
    return p


# ── the authoring loop: gates as TOOLS, not as silent rejections ─────────
#
# THE FIRST DESIGN PROPOSED ONCE AND REJECTED, AND IT THREW THE WORK AWAY.
# Each attempt was a fresh call that saw only a one-line reason ("recovered no
# gap in this class"), so the model could not tell a pattern that fired on
# nothing from one that fired everywhere and repaired wrongly. Measured: on
# `unmatched_closing_bracket` the first two attempts each passed every safety
# gate, recovered 0, and were discarded whole — three calls to reach one
# answer, with two complete pieces of analysis destroyed on the way.
#
# Only ONE gate is a blocker, and it is a blocker because it is about SAFETY
# rather than quality: source that imports, touches dunders, calls `open`, or
# hangs cannot be run at all, so there is nothing to give feedback about.
# Everything else — how many gaps it repairs, how many good names it damages,
# whether the repaired spelling occurs in the patent — is a MEASUREMENT the
# model can ask for and act on. It gets the tools and the turns to converge.
#
# Nothing is discarded: every attempt is kept with its measurements, and the
# BEST one by outcome is adopted rather than the last one tried.

MAX_TURNS = 10

AUTHOR_SYSTEM = """You are repairing chemical compound NAMES copied out of US \
patent XML. The damage is typesetting and OCR; the chemistry underneath is \
correct.

A find-and-replace rule has already failed on this failure class, so write \
Python instead. Define EXACTLY two functions and nothing else:

    def site(name: str) -> list[tuple[int, int]]:   # spans that may be damaged
    def fix(span: str) -> list[str]:                # candidate replacements

YOU ARE EXPECTED TO ITERATE. Do not answer in one shot. Use the tools:

  test_pattern       run your code against the real failing names. Returns how
                     many it repairs, how many ALREADY-CORRECT names it damages,
                     and real before/after examples. Call this every time you
                     change anything.
  check_corroboration for names you repaired, ask whether the repaired spelling
                     actually occurs elsewhere in this patent. A repair the
                     document itself confirms is strong evidence; one that
                     appears nowhere may be a plausible-looking invention.
  submit             only when the numbers stop improving.

WHAT A GOOD ANSWER LOOKS LIKE, in priority order:
  1. It damages ZERO already-correct names. A pattern that mangles 100 good
     names to fix 1 still reports a repair for that 1 — you cannot see this
     without calling the tool.
  2. Its repairs are corroborated by the document where possible.
  3. It repairs as many of the failing names as it honestly can.

DELETION IS THE TRAP. Removing a bracket or a substituent makes a SMALLER \
molecule parse. That is a wrong answer wearing the costume of a right one — \
OPSIN will accept it and the compound will be silently wrong. Prefer inserting \
what is missing over deleting what is present, and when you must delete, \
corroborate."""

AUTHOR_TOOLS = [
    {"name": "test_pattern",
     "description": "Run site/fix against the failing names and the correct "
                    "names. Returns repairs, collateral damage, and examples.",
     "input_schema": {"type": "object", "required": ["source"],
                      "properties": {"source": {"type": "string"}}}},
    {"name": "check_corroboration",
     "description": "For repaired names, report whether the repaired spelling "
                    "occurs elsewhere in this patent as published.",
     "input_schema": {"type": "object", "required": ["source"],
                      "properties": {"source": {"type": "string"}}}},
    {"name": "submit",
     "description": "Adopt this pattern. Call only when results stop improving.",
     "input_schema": {"type": "object", "required": ["id", "source", "note"],
                      "properties": {"id": {"type": "string"},
                                     "source": {"type": "string"},
                                     "note": {"type": "string"}}}},
]


def _corroborated(repaired: list[str], doc: str) -> tuple[int, list[str]]:
    """How many repaired spellings occur in the patent AS PUBLISHED.

    The gate `name_repair` applies to every untrusted hand-written form, wired
    in here because it is the only check that distinguishes a repair the
    document confirms from a plausible invention. It matters most for the
    deletion shape: removing a bracket makes a SMALLER molecule parse, OPSIN
    accepts it, retention scores ~0.99 because one character moved — and none
    of that is evidence the compound is the one the patent meant. If the
    corrected spelling appears elsewhere in the document, it is.
    """
    if not doc:
        return 0, []
    norm = " ".join(doc.split()).lower()
    hits = [r for r in repaired if " ".join(r.split()).lower() in norm]
    return len(hits), hits[:3]


def _evaluate(source: str, gap_names: list[str], clean_names: list[str],
              doc: str, opsin) -> dict:
    """Every measurement at once. Returns a dict the model reads as a tool result."""
    why = static_check(source)
    if why:
        return {"ok": False, "blocked": why}
    cands, err = apply_pattern(source, gap_names)
    if err:
        return {"ok": False, "blocked": err}

    flat, owner = [], []
    for i, cs in enumerate(cands):
        for c in cs:
            flat.append(c); owner.append(i)
    smis = opsin(flat, "SMILES") if flat else []
    repaired_idx, repaired_str = set(), []
    for j, s in enumerate(smis):
        if s and owner[j] not in repaired_idx:
            repaired_idx.add(owner[j]); repaired_str.append(flat[j])

    n_coll, coll_err = collateral(source, clean_names)
    n_corr, corr_ex = _corroborated(repaired_str, doc)

    ex = []
    for j, s in enumerate(smis):
        if not s or len(ex) >= 3:
            continue
        o, r = gap_names[owner[j]], flat[j]
        d = next((k for k in range(min(len(o), len(r))) if o[k] != r[k]),
                 min(len(o), len(r)))
        ex.append({"before": o[max(0, d - 40):d + 30],
                   "after": r[max(0, d - 40):d + 30]})
    return {"ok": True, "repaired": len(repaired_idx), "of": len(gap_names),
            "damages_correct_names": n_coll if not coll_err else coll_err,
            "corroborated_by_document": n_corr,
            "corroborated_examples": corr_ex, "edits": ex}


def author_pattern(gaps, signature: str, *, clean_names, doc: str,
                   opsin, model: str = "", max_turns: int = MAX_TURNS):
    """Let the model ITERATE with the measurements as tools.

    Returns `(best_pattern | None, attempts)` where `attempts` holds every
    evaluation made, so a run that adopts nothing still shows its working.
    The BEST attempt by (no damage, corroboration, repairs) is returned — not
    the last, and not only the one the model chose to submit.
    """
    from ..core.cost_tracker import cost_tracker
    from ..core.api_client import resilient_client

    model = model or config.CAPABILITY_MODEL
    if not config.ANTHROPIC_API_KEY:
        return None, []

    gap_names = [g.name_text for g in gaps]
    prompt = _render(gaps, signature, clean_names, [])
    messages: list = [{"role": "user", "content": prompt}]
    attempts: list[dict] = []
    submitted = None

    for turn in range(max_turns):
        try:
            resp = resilient_client().messages.create(
                model=model, max_tokens=4000, temperature=0,
                system=[{"type": "text", "text": AUTHOR_SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                tools=AUTHOR_TOOLS, messages=messages)
        except Exception as e:
            logger.warning("name_cap: call failed: %r", e)
            break
        u = resp.usage
        cost_tracker.record(
            u.input_tokens, u.output_tokens, model,
            patent_id=gaps[0].patent_id, cost_category="lm",
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)

        calls = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if not calls:
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for c in calls:
            src = (c.input or {}).get("source", "")
            if c.name == "submit":
                submitted = CodePattern(
                    id=(c.input.get("id") or "code")[:48].replace(" ", "_"),
                    source=src, note=(c.input.get("note") or "")[:200],
                    model=model, bought_for=gaps[0].patent_id)
                res = _evaluate(src, gap_names, clean_names, doc, opsin)
                attempts.append({"turn": turn, "tool": "submit",
                                 "source": src, **res})
                results.append({"type": "tool_result", "tool_use_id": c.id,
                                "content": json.dumps(res)})
            elif c.name in ("test_pattern", "check_corroboration"):
                res = _evaluate(src, gap_names, clean_names, doc, opsin)
                attempts.append({"turn": turn, "tool": c.name,
                                 "source": src, **res})
                results.append({"type": "tool_result", "tool_use_id": c.id,
                                "content": json.dumps(res)})
        messages.append({"role": "user", "content": results})
        if submitted is not None:
            break

    # BEST, NOT LAST. The model may submit an earlier idea after a worse
    # experiment, and a run that never submits can still have found something.
    scored = [a for a in attempts if a.get("ok")
              and isinstance(a.get("damages_correct_names"), int)]
    if not scored:
        return None, attempts
    best = max(scored, key=lambda a: (a["damages_correct_names"] == 0,
                                      a["corroborated_by_document"],
                                      a["repaired"]))
    if best["repaired"] == 0:
        return None, attempts
    pat = submitted if (submitted and submitted.source == best["source"]) else \
        CodePattern(id=f"{signature.split('|')[0]}_authored", source=best["source"],
                    note="best measured attempt", model=model,
                    bought_for=gaps[0].patent_id)
    pat.confirms = best["repaired"]
    return pat, attempts
