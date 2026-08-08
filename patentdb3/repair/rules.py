"""The learned rule library: propose → ground → apply → MEASURE → keep or retry.

A rule is keyed by LAYOUT FINGERPRINT, never by patent. The whole economic
argument for this loop is amortisation: one paid question per distinct table
shape, reused free on every patent that shares it. Keying by patent would
reduce it to HARVEST with extra steps.

Five rule kinds, because "emit a regex or give up" is too narrow for what
actually goes wrong:

  column_map  — say which column is the id, which are assays, their units.
                Covers most failures; no regex, nothing to over-match.
  row_regex   — a pattern applied per row when the cell structure is lost.
  value_pattern — how to read an individual CELL. This is what makes the loop
                self-healing for parser bugs rather than only layout bugs. When
                `_VALUE_PAT` rejected any number >= 1000, the columns were
                mapped perfectly and the cells were plainly numeric; nothing in
                the other rule kinds could say "your number regex is too narrow".
  bin_key     — a legend's grade symbols (`A-E`, `+/++/+++`) as explicit numeric
                ranges, transcribed from the patent's own text.
  not_assay   — a NEGATIVE rule. US11566007's `A1 | 944.5` table is mass-spec.
                The correct outcome is "never treat this shape as assay data",
                and recording it stops us paying to rediscover that forever.
  escalate    — no rule could be made to work here. Carries what capability is
                missing so a human reads a queue, not a log.

WHAT THIS FILE NO LONGER DOES
-----------------------------
It used to hold `validate()`: a coverage floor, an anti-deletion guard, an
adversarial probe battery and an id-shape test, all run BEFORE the rule was
applied, all trying to predict whether it would do harm. They were suspended by
default because they had been wrong as often as right, and the 88 rules adopted
over their objection were then measured — 28,281 rows, zero contradicting the
deterministic reader, and the anti-deletion guard found to be comparing a
held-out row count against a whole-table record count, so it could never have
been right.

Prediction is replaced by observation. `repair/outcome.py` runs the rule on the
patent it was bought for and reports what came out; `loop.py` keeps it or hands
the failure back for another attempt. What survives here is the half that was
never a judgement:

  `ground()` — can this rule RUN (compile, index in range, capture a number),
  and does every name and unit in it appear in THIS document. The second is the
  one corruption no measurement of the output can catch, because a fabricated
  unit produces perfectly usable records.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core import config

logger = logging.getLogger(__name__)

COLUMN_MAP = "column_map"
ROW_REGEX = "row_regex"
VALUE_PATTERN = "value_pattern"
BIN_KEY = "bin_key"
NOT_ASSAY = "not_assay"
ESCALATE = "escalate"

_LIBRARY = config.PACKAGE_ROOT / "data" / "layout_rules.json"

# Bump whenever the prompt, the tool schema, the sample rendering or the way a
# response is routed changes — anything that could make us answer a question
# better than we did last time.
#
# This governs BOTH caches. The API response cache was already versioned; the
# rule library was not, and that asymmetry is why eight US10172859 gaps were
# frozen at a stale `escalate` through every subsequent improvement. `lib.get()`
# short-circuits before the model is called, so a persisted escalation pins the
# layout to the capability we had the day it was written.
SYNTH_EPOCH = "v20-measured-outcome"

# Regex constructs that make catastrophic backtracking possible. A synthesized
# pattern runs over thousands of rows, so a quadratic blowup is a hang, and the
# model has no idea it wrote one.
_REDOS = re.compile(r"\((?=[^)]*[+*])[^)]*\)\s*[+*]|\(\?[:=!][^)]*[+*][^)]*\)[+*]")


@dataclass
class Rule:
    fingerprint: str
    kind: str
    # column_map: {"cid": 0, "assays": [{"index": 1, "name": ..., "unit": ...}]}
    # row_regex:  {"pattern": ..., "fields": {...}}
    # not_assay:  {"because": "mass-spec characterisation"}
    # escalate:   {"capability": ..., "note": ...}
    payload: dict = field(default_factory=dict)
    source: str = "llm"                  # llm | human | builtin
    model: str = ""
    created: str = ""
    # The fingerprint's PREIMAGE — `n_cols|shapes|header words`. The hash keys
    # the library and cannot be compared; this can, and it is what makes
    # `RuleLibrary.digest` possible. Empty on rules adopted before it existed:
    # they simply do not appear in a digest, which degrades to today's
    # behaviour rather than breaking.
    signature: str = ""
    # Which synthesis epoch produced this. Only negatives consult it — see
    # RuleLibrary.get.
    epoch: str = ""
    # Evidence, kept so a bad rule can be traced and revoked.
    validated_on: dict = field(default_factory=dict)
    times_applied: int = 0
    rows_yielded: int = 0

    def to_json(self) -> dict:
        return asdict(self)


class RuleLibrary:
    """Persistent, append-only-ish store of validated layout rules."""

    def __init__(self, path: Path | None = None):
        self.path = path or _LIBRARY
        self._rules: dict[str, Rule] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for r in raw.get("rules", []):
            try:
                self._rules[r["fingerprint"]] = Rule(**r)
            except TypeError:
                continue

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"schema": 1, "rules": [r.to_json() for r in self._rules.values()]},
            indent=2))
        tmp.replace(self.path)

    def get(self, fingerprint: str) -> Rule | None:
        """The stored answer, unless it is an escalation we have outgrown.

        A validated rule is EVIDENCE: it was re-run against real rows and it
        held, so it stays true no matter what changes here. An escalation is
        not an answer at all — it is a record that our capability ran out on
        this layout, on a particular day, with a particular prompt. Treating
        the two the same froze eight US10172859 gaps at "capability:
        unspecified" while the fixes that would have resolved them landed one
        after another and were never consulted.

        `not_assay` deliberately does NOT expire. It is a positive claim about
        what the table contains, already guarded by the veto in synthesize, and
        re-litigating it every epoch would pay to rediscover mass-spec tables
        forever.
        """
        rule = self._rules.get(fingerprint)
        if rule is not None and rule.kind == ESCALATE and rule.epoch != SYNTH_EPOCH:
            return None
        return rule

    def add(self, rule: Rule) -> None:
        rule.created = rule.created or time.strftime("%Y-%m-%dT%H:%M:%S")
        rule.epoch = rule.epoch or SYNTH_EPOCH
        self._rules[rule.fingerprint] = rule

    def record_use(self, fingerprint: str, rows: int) -> None:
        r = self._rules.get(fingerprint)
        if r:
            r.times_applied += 1
            r.rows_yielded += rows

    # ── what the model is allowed to know about the library ──────────
    #
    # The library is 172 rules / 165 KB and grows with every new layout, so
    # "show the model what it already knows" cannot mean sending it. It also
    # must not mean sending nothing, which is the state this replaces: every
    # gap is asked in isolation, so a shape solved three patents ago is
    # re-derived from scratch and the loop never compounds.
    #
    # The middle is retrieval on the layout SIGNATURE — the fingerprint before
    # hashing, which is the one comparable form of a shape we have. Two or
    # three nearest neighbours, rendered as one line each, is ~80 tokens per
    # rule against 165 KB for the library.
    #
    # Safe to show because it decides nothing. A model that copies a
    # neighbouring rule that does not fit produces a rule that fails on this
    # patent, which the outcome gate observes and hands back for another
    # attempt. Anchoring costs a retry; it cannot ship anything.

    def similar(self, signature: str, k: int = 3) -> list[tuple[float, Rule]]:
        """The k most structurally similar rules that have actually WORKED.

        `rows_yielded > 0` is the filter, and it is the only quality signal
        here that needs no judgement: it is observed. 36 of 88 rules measured
        on 2026-08-07 had never produced a row in any application, and showing
        one of those as an exemplar teaches the shape of a rule that does
        nothing.
        """
        if not signature:
            return []
        want = _parse_signature(signature)
        scored = []
        for r in self._rules.values():
            if not r.signature or r.rows_yielded <= 0 or r.kind in (ESCALATE, NOT_ASSAY):
                continue
            if r.signature == signature:
                continue          # the exact layout is `get()`, not a neighbour
            s = _signature_similarity(want, _parse_signature(r.signature))
            if s > 0.4:
                scored.append((s, r))
        scored.sort(key=lambda t: (-t[0], -t[1].rows_yielded))
        return scored[:k]

    def digest(self, signature: str, k: int = 3) -> str:
        """Those neighbours as prompt text. Empty string when there are none."""
        near = self.similar(signature, k)
        if not near:
            return ""
        lines = ["RULES ALREADY LEARNED FOR THE MOST SIMILAR LAYOUTS. These are "
                 "for reference only — this table is NOT one of them, and "
                 "copying a rule that does not fit will simply fail. Use them "
                 "to see what a working rule for this sort of shape looks "
                 "like."]
        for score, r in near:
            lines.append(f"  [{score:.0%} similar] {r.signature[:150]}")
            lines.append(f"    -> {_render_payload(r)}"
                         f"  ({r.rows_yielded:,} rows over {r.times_applied} use(s))")
        return "\n".join(lines) + "\n"


# ── layout similarity, for the digest above ───────────────────────

def _parse_signature(sig: str) -> tuple[int, list[str], list[str]]:
    """`"8|cid,num,num|compound,ic+nm,herg"` -> (8, [shapes], [header words])."""
    parts = (sig or "").split("|")
    if len(parts) != 3:
        return 0, [], []
    try:
        n = int(parts[0])
    except ValueError:
        return 0, [], []
    return n, parts[1].split(","), parts[2].split(",")


def _signature_similarity(a: tuple, b: tuple) -> float:
    """How alike two layouts are, in [0, 1]. Cheap, deterministic, no model.

    COLUMN COUNT IS A HARD FILTER, not a weighted term. Every rule kind here
    addresses columns by POSITION — `column_map` names indices outright,
    `value_pattern` carries a `columns` list, `row_regex` counts `|` separators
    — so a rule written for a 3-column table says nothing usable about an
    8-column one however alike they read. Scored as a weight it came out at
    0.47 for exactly that pair, which is above any sensible threshold, and the
    consequence would be showing a model a worked example whose indices cannot
    fit the table in front of it.

    Within one column count, the per-column value SHAPES carry more than the
    header WORDS: shapes say where the ids and the numbers physically sit,
    which is what a rule is written against, while the same assay is spelled a
    dozen ways across patents and drafting firms.
    """
    (na, sa, wa), (nb, sb, wb) = a, b
    if not na or not nb or na != nb:
        return 0.0
    width = min(len(sa), len(sb))
    shape = (sum(1 for i in range(width) if sa[i] == sb[i]) / max(len(sa), len(sb))
             if width else 0.0)
    ta = {t for w in wa for t in w.split("+") if t}
    tb = {t for w in wb for t in w.split("+") if t}
    words = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    return 0.6 * shape + 0.4 * words


def _render_payload(rule: Rule) -> str:
    """A rule as one short line. The digest sends this, never the raw JSON."""
    p = rule.payload or {}
    if rule.kind == COLUMN_MAP:
        cols = ", ".join(
            f"{a.get('index')}:{(a.get('name') or '?')[:28]!r}"
            + (f" {a['unit']}" if a.get("unit") else "")
            for a in (p.get("assays") or [])[:6])
        return f"column_map: cid={p.get('cid')}, assays=[{cols}]"
    if rule.kind in (ROW_REGEX, VALUE_PATTERN):
        return f"{rule.kind}: {str(p.get('pattern', ''))[:110]}"
    if rule.kind == BIN_KEY:
        scales = p.get("scales") or []
        if scales:
            return (f"bin_key: {len(scales)} column-scoped scale(s) — "
                    + "; ".join(f"{(s.get('match') or 'default')[:22]} -> "
                                f"{sorted((s.get('bins') or {}))}" for s in scales[:3]))
        return f"bin_key: {sorted(p.get('bins') or {})}"
    return f"{rule.kind}: {str(p)[:100]}"


# ── the contract: can this rule RUN, and does it speak about THIS document ──

class Rejected(Exception):
    """A proposed rule failed its gate. Carries why, for the escalation queue."""


class Invalid(Rejected):
    """The rule cannot be EXECUTED at all — a contract violation, not a verdict.

    The suspension of the judgement gates was right and this is why it needed a
    second category. `validate` asks two different kinds of question:

      "is this rule GOOD?"   — coverage floors, the adversarial battery,
                               grounding. Opinions, and this codebase's have
                               been wrong at least as often as right: a correct
                               `column_map` scored 0/23, a 49% floor blamed a
                               layout for a regex in the reader. SUSPENDED.

      "can this rule RUN?"   — does the regex compile, does a column_map name a
                               cid, can a value_pattern ever yield a number.
                               Not opinions. A rule failing these is not
                               low-quality, it is not a rule.

    Suspending the first suspended the second with it, and three structurally
    unusable rules entered the library over the objection "value_pattern must
    capture a named group `num`" — then crashed two whole patents out of the
    loop at apply time. This subclasses `Rejected` so nothing that catches the
    old type stops working, and `loop` catches it FIRST so it always blocks.
    """


try:                                    # `regex` supports a match timeout; `re` does not
    import regex as _re_engine
    _HAS_TIMEOUT = True
except ImportError:                     # pragma: no cover - fallback path
    _re_engine = re
    _HAS_TIMEOUT = False

MATCH_TIMEOUT_S = 0.25


def _safe_regex(pattern: str):
    """Compile a synthesized pattern, refusing the obviously dangerous shapes.

    The static check is a filter, NOT a guarantee. Automata-based ReDoS
    detectors are documented to miss polynomial blowup and anything involving
    lookaround or backreferences, so the real backstop is the per-match timeout
    applied in `_search` — a synthesized pattern runs over thousands of rows,
    and a quadratic one is an outage, not a slow query.
    """
    if len(pattern) > 400:
        raise Rejected("regex implausibly long; likely memorised the sample")
    if _REDOS.search(pattern):
        raise Rejected("regex has nested quantifiers (catastrophic backtracking risk)")
    try:
        return _re_engine.compile(pattern)
    except Exception as e:
        # `(?P&lt;cid&gt;\d+)` — "bad character in group name at position 21".
        # The model escaped its own output; the sample it read contained no
        # entities (US11613531's source XML has zero `&lt;`). A pattern one
        # unescape away from valid is malformed, not wrong, and rejecting it
        # cost a whole layout. Repaired only AFTER a genuine failure, so a
        # pattern matching a literal `&` is never rewritten.
        import html
        repaired = html.unescape(pattern)
        if repaired != pattern:
            try:
                return _re_engine.compile(repaired)
            except Exception:
                pass
        raise Rejected(f"regex does not compile: {e}") from e


def _search(pat, text: str):
    """Run a synthesized pattern under a hard wall-clock cap."""
    try:
        if _HAS_TIMEOUT:
            return pat.search(text, timeout=MATCH_TIMEOUT_S)
        return pat.search(text)
    except TimeoutError as e:
        raise Rejected(
            f"regex exceeded {MATCH_TIMEOUT_S}s on a single row — catastrophic "
            f"backtracking the static check did not catch") from e


def _states_unit(source: str, unit: str) -> bool:
    """Does this document state this unit, as text? No vocabulary consulted.

    Three normalisations, each because a spelling difference is not a
    disagreement about meaning:

    WHITESPACE is ignored. US9233167 heads a column `AUCall (min · ng/mL)` and
    a model proposing `min·ng/mL` is quoting it, not inventing it.

    MICRO is one character with three spellings — `µ` (U+00B5 micro sign), `μ`
    (U+03BC greek mu) and plain ASCII `u`. All three fold together. US11649247
    heads sixteen columns `IC50 (μM)` and the model answered `uM`; folding only
    the two non-ASCII forms left those as different strings and dropped every
    one of them.

    LITRE CASE is folded, and NOTHING ELSE IS. `mL` and `ml` are the same
    litre; `nM` and `nm` are nanomolar and nanometre, a concentration and a
    length. The old check ran `re.I` over everything, and both failure modes
    are live in this corpus: US9221791's only `nM` is `monitor 240 and 254 nm`
    in a purification step, and US10172859's three are all `λ=220 nm`
    chromatography wavelengths — under `re.I` each one certified a proposed
    nanomolar that the block never states. Case-insensitivity on the molar M is
    a hole in the direction that ships wrong data by 1000x.

    Folding micro to `u` is safe for the same reason: the M stays capital. The
    19 lowercase `um` strings in US10172859's block are all inside `sodium`,
    `pyrimidin` and `tetrahydrofuran`, and none of them survives `(?![A-Za-z])`
    or the capital M.

    Bounded by LETTERS only, not by digits: `10nM` and `10 nM` both state nM,
    while `min` inside `determination` states nothing.
    """
    def fold(s: str) -> str:
        return s.replace("µ", "u").replace("μ", "u").replace("L", "l")

    chars = [c for c in fold(unit) if not c.isspace()]
    if not chars:
        return False
    pattern = r"(?<![A-Za-z])" + r"\s*".join(re.escape(c) for c in chars) + r"(?![A-Za-z])"
    try:
        return re.search(pattern, fold(source)) is not None
    except re.error:                     # a unit string that will not compile
        return False


def _id_style(values: list[str]) -> str:
    """`code`, `name`, or `""` — how (or whether) this column identifies rows.

    `_CID_PAT` matches `12`, `I-2300`, `Z1`, `5a` and is deliberately strict,
    because without a shape test any text column could be claimed as the id and
    a molecular-weight table would key on nothing. The cost of that strictness
    was invisible until US9233167: it names its compounds
    (`α-6-mPEG1-O-Morphine`) rather than numbering them, so a CORRECT
    column_map scored 0/23 rows and escalated as "it describes the sample, not
    the layout" — blaming the model for a limit in our own gate. A patent that
    names its compounds could never be repaired.

    A name is a perfectly good identifier; it just is not a code. What actually
    matters is that the column identifies: distinct per row, not a measurement,
    not prose. Those are checkable, so check them instead of the shape.
    """
    from ..sources.uspto_assays import _CID_PAT, _VALUE_PAT

    vals = [v.strip() for v in values if v and v.strip()]
    if len(vals) < 3:
        return ""
    if sum(bool(_CID_PAT.match(v)) for v in vals) >= 0.6 * len(vals):
        return "code"
    # An identifier is UNIQUE. NMR text repeats its structure row to row and is
    # the thing that must never be keyed on.
    if len({v.lower() for v in vals}) < 0.8 * len(vals):
        return ""
    # ...is not a measurement (a column of numbers is data, not a key)...
    if any(_VALUE_PAT.match(v) for v in vals):
        return ""
    # ...carries letters, and is short enough to be a name rather than a
    # sentence. Chemical names run long; NMR blocks and prose run longer.
    if not all(re.search(r"[A-Za-z]", v) and 2 <= len(v) <= 90 for v in vals):
        return ""
    return "name"


def _check_bin_key(rule: Rule) -> None:
    """A symbol -> range mapping must be internally COHERENT. Nothing else.

    This is what lets the loop learn a legend grammar it has never seen: rather
    than a parser per format — `A <= 10 nM; 10 nM < B <= 100 nM`, `++++: IC50
    >= 1 uM`, `is marked "+++"` — the model reads whichever form the patent used
    and returns the mapping, so any future format is covered for free.

    What is checked here is only what makes the mapping EXECUTABLE: a bin with
    no bounds cannot produce a range, `lo >= hi` is not an interval, a range
    with no unit is not a measurement, and two unscoped scales leave it
    undefined which one governs a column.

    What is NOT checked here any more is whether the key explains the symbols
    this table actually uses. That was `Rejected("it belongs to a different
    table")`, and it is a prediction about what applying the rule would do —
    now simply observed: a key that explains none of the grades present yields
    zero records, which `outcome.measure` reports as negative with the counts
    attached, and the model gets a retry instead of a rejection.
    """
    scales = rule.payload.get("scales") or []
    if scales:
        for s in scales:
            if not (s.get("bins") or {}):
                raise Invalid("a scale carries no bins")
            pat = (s.get("match") or "").strip()
            if pat:
                try:
                    re.compile(pat)
                except re.error as e:
                    raise Invalid(f"scale match {pat!r} is not a regex: {e}")
        if sum(1 for s in scales if not (s.get("match") or "").strip()) > 1:
            raise Invalid("more than one unscoped scale — which governs which "
                          "column is then undefined")
        bins = {k: v for s in scales for k, v in (s.get("bins") or {}).items()}
    else:
        bins = rule.payload.get("bins") or {}
    if not bins:
        raise Invalid("bin_key carries no bins")

    for sym, b in bins.items():
        lo, hi = b.get("lo"), b.get("hi")
        if lo is None and hi is None:
            raise Invalid(f"bin {sym!r} has neither bound")
        if lo is not None and hi is not None and lo >= hi:
            raise Invalid(f"bin {sym!r} has lo >= hi ({lo} >= {hi})")
        if not b.get("unit"):
            raise Invalid(f"bin {sym!r} has no unit; a range without one is unusable")


def first_number(m) -> str | None:
    """The number a value pattern captured, whatever the model called it.

    `num` is the documented contract, and three adopted rules do not honour
    it: a model describing a two-value cell writes `num1`/`num2`, and one
    returned a pattern with no groups at all. The suspended gate let all three
    into the library over its own objection, and `m.group("num")` then raised
    `IndexError('no such group')` — crashing two whole patents out of the loop.

    That is the distinction the suspension did not draw. A gate that asks "is
    this rule GOOD" is a judgement and has been wrong repeatedly; a gate that
    asks "does this rule have the shape the applier requires" is a contract.
    Suspending the first suspended the second with it.

    Rather than re-gate, the applier is made total: prefer `num`, else the
    first participating group named `num*`, else nothing. `num1`/`num2` then
    WORK instead of crashing, which is what the model meant by them.
    """
    if m is None:
        return None
    names = m.re.groupindex or {}
    if "num" in names:
        return m.group("num")
    for name in sorted(names, key=lambda n: names[n]):
        if name.lower().startswith("num") and m.group(name):
            return m.group(name)
    return None


def _check_value_pattern(rule: Rule) -> None:
    """A cell parser must be able to capture a NUMBER. That is the whole check.

    Everything else this used to do is now measured instead of predicted:

      RESCUES (`rescued < 3`) — a pattern that rescues nothing produces no
        records, which `outcome.measure` reports as negative, with counts.
      NO REGRESSION — impossible by construction. `apply_rule` skips every cell
        `parse_value` already reads (`if not s or parse_value(s): continue`),
        so a value_pattern cannot reinterpret a cell we read correctly. This
        gate was scoring a rule the applier never ships.
      NO OVER-REACH — five synthetic probe strings, replaced by
        `outcome.NOT_A_MEASUREMENT`, which asks whether the rule pulled a value
        out of a column of THIS patent that the reader identified as MW/MS/NMR/
        RT. The real columns beat five invented rows.
      NOTHING TO ATTACH TO — `apply_rule` already refuses to run unless a
        column is NAMED as a compound id (`_HEADER_CID`), so this rejected a
        rule for a condition the applier enforces anyway.
    """
    raw = rule.payload.get("pattern", "")
    pat = _safe_regex(raw)
    if "num" in (pat.groupindex or {}):
        return
    # A single unnamed capture group is unambiguous — it can only be the number
    # — so promote it rather than failing the call over syntax.
    if pat.groups == 1 and not pat.groupindex:
        raw = raw.replace("(", "(?P<num>", 1)
        _safe_regex(raw)
        rule.payload["pattern"] = raw
        return
    if not any(n.lower().startswith("num") for n in (pat.groupindex or {})):
        raise Invalid(
            "value_pattern captures no number: no group named `num`, no "
            "single unnamed group to promote, and nothing named `num*` "
            f"(groups={pat.groups}, named={list(pat.groupindex)}). Such a "
            "rule cannot produce a value on any input.")


def ground(rule: Rule, table) -> dict:
    """Make a proposal EXECUTABLE and TRUTHFUL about this document, or refuse.

    This is everything that survives of the old `validate()`, and it is
    deliberately not a quality gate. It asks two questions, neither of which is
    a matter of taste:

      CAN IT RUN?  Does the regex compile, is the column index in range, can a
        value_pattern ever capture a number, is `lo < hi`. A rule failing these
        is not low-quality — it is not a rule, and three such entered the
        library while the judgement gates were suspended and then crashed two
        whole patents at apply time. Raises `Invalid`.

      DOES IT SPEAK ABOUT THIS DOCUMENT?  A name or a unit the patent does not
        contain is a claim about chemistry, not about this table, and it is the
        one corruption no measurement of the output can catch: US9221791's
        columns are headed `CYP2C9 IC50` with no unit anywhere near them, a
        model supplied `nM`, and 440 records shipped 1000x low with every count
        healthy. So ungrounded names and units are DROPPED — the column stays,
        the invented part does not. Dropping rather than rejecting is what
        honours the prompt's own instruction that a rule covering three of five
        columns beats nothing.

    Everything that used to live between those two — the 50% coverage floor,
    the anti-deletion guard, the adversarial probe battery, `bad_cid > hits` —
    is gone. They predicted what applying the rule would do; `outcome.measure`
    observes it instead, on this patent, after `apply_rule` has run. Measured
    over the 88 rules adopted over their objection: zero contradicted the
    deterministic reader across 28,281 rows.

    Returns evidence to store on the rule. Never returns a verdict.
    """
    if rule.kind in (NOT_ASSAY, ESCALATE):
        return {"checked": "none required", "kind": rule.kind}
    if rule.kind == VALUE_PATTERN:
        _check_value_pattern(rule)
        return {"kind": VALUE_PATTERN, "pattern": rule.payload.get("pattern", "")[:120]}
    if rule.kind == BIN_KEY:
        _check_bin_key(rule)
        return {"kind": BIN_KEY,
                "symbols": sorted(rule.payload.get("bins") or {})
                or [s.get("match") or "default"
                    for s in (rule.payload.get("scales") or [])]}

    if rule.kind == ROW_REGEX:
        pat = _safe_regex(rule.payload.get("pattern", ""))
        if "cid" not in (pat.groupindex or {}):
            raise Invalid("row_regex must capture a named group `cid`; without "
                          "one there is no compound to attach a value to")
        if not any(g.startswith("value") for g in pat.groupindex):
            raise Invalid("row_regex must capture at least one `value…` group; "
                          "without one it can never produce a measurement")
        return {"kind": ROW_REGEX, "groups": sorted(pat.groupindex)}

    if rule.kind != COLUMN_MAP:
        raise Invalid(f"unknown rule kind {rule.kind!r}")

    from ..sources.uspto_assays import _header_rows_of, merge_header

    cid_i = rule.payload.get("cid")
    assays = rule.payload.get("assays") or []
    if not isinstance(cid_i, int) or not assays:
        raise Invalid("column_map needs an integer `cid` and ≥1 assay column")
    idxs = [cid_i] + [a.get("index") for a in assays]
    if any(not isinstance(i, int) or i < 0 or i >= table.n_cols for i in idxs):
        raise Invalid(f"column index out of range for a {table.n_cols}-column table")
    if cid_i in [a.get("index") for a in assays]:
        raise Invalid("the id column is also listed as an assay column")

    hdr_rows, data = _header_rows_of(table)
    headers = merge_header(table, hdr_rows)

    # Grounding: a proposed name may REGROUP the patent's words, never add new
    # ones.
    #
    # Shown US11254686's raw header stacks, Haiku correctly recovered `A2B cAMP`
    # and `A2A cAMP` from cells our merge had fused into `CYP Ave A2A cAMP` —
    # and then named two more columns `CYP3A4 % inhibition` and `CYP2D6 %
    # inhibition`. The patent contains neither "3A4" nor "2D6", and the second
    # of those columns is liver-microsome clearance, not a CYP inhibition assay
    # at all. Plausible domain knowledge, applied to the wrong columns, and
    # indistinguishable from a correct answer once written to the database.
    #
    # DIGITS are the load-bearing part: CYP3A4 vs CYP2D6, A2A vs A2B, ROCK1 vs
    # ROCK2 differ only there. So a token carrying a digit must appear in the
    # patent's own text for this table verbatim; a purely alphabetic token may
    # be an expansion of an abbreviation the table does use (`% INH` → `%
    # inhibition`), which prefix matching allows.
    from ..sources.uspto_assays import table_legend as _legend
    # BODY CELLS ARE IN HERE, and they were the second-largest source of false
    # "inventions". `_header_rows_of` decides where the header stops, and when
    # it stops early the unit row lands in the body: US9682141's merged header
    # is `mTO RC | PI3K α | …` while body rows 0-1 read `IC50 | IC50 | …` and
    # `(nM) | (nM) | …`. US12065407 splits one unit ACROSS the boundary —
    # header `Clearance CL int (μL/min/mg`, body row 0 `protein)`. In both the
    # model reassembled the header correctly and we called it invention,
    # because this text excluded exactly the rows our own assembler had
    # misfiled. Self-inflicted twice: mis-split, then refuse to look where it
    # was put.
    source_raw = " ".join(
        [" ".join(c.text for r in hdr_rows for c in r), table.caption or "",
         getattr(table, "preceding", "") or "", _legend(table) or "",
         " ".join(c.text for r in data for c in r)])
    source_text = source_raw.lower()
    source_toks = set(re.findall(r"[a-z0-9]+", source_text))

    def _ungrounded(name: str) -> str | None:
        for tok in re.findall(r"[a-z0-9]+", (name or "").lower()):
            if len(tok) < 3 and not any(ch.isdigit() for ch in tok):
                continue
            if any(ch.isdigit() for ch in tok):
                if tok in source_toks:
                    continue
            elif any(s.startswith(tok) or tok.startswith(s)
                     for s in source_toks if len(s) >= 3):
                continue
            return tok
        return None

    kept, dropped = [], []
    for a in assays:
        bad = _ungrounded(a.get("name") or "")
        (dropped if bad else kept).append((a, bad))
    if dropped:
        logger.info("repair: dropped %d ungrounded column name(s): %s",
                    len(dropped), "; ".join(f"{a.get('name')!r} introduces {b!r}"
                                            for a, b in dropped))
    if not kept:
        raise Invalid(
            "every proposed name introduces a word that appears nowhere in this "
            "table's header, caption or surrounding text (e.g. "
            f"{dropped[0][0].get('name')!r} introduces {dropped[0][1]!r}) — a "
            "repair may regroup the patent's own words, not supply new ones")
    assays = [a for a, _ in kept]

    # The same rule applies to UNITS, and it is asked the same way: is this
    # text in the document. Nothing more.
    #
    # US9221791 heads its columns `CYP2C9 IC50` with no unit anywhere near them;
    # the caption names ug/mL for a different column. A model proposed `nM`. The
    # patent says uM, mM and ug/mL and never once says nM — its single `nM` is a
    # `254 nm` HPLC wavelength — and BindingDB puts those compounds at 42,000 nM.
    # A 1000x understatement on 440 records, arriving with coverage going UP.
    # Nothing but the value check saw it.
    #
    # THERE IS NO UNIT VOCABULARY. There was: `nM|pM|mM|uM|ug/mL|ng/mL|%`, seven
    # spellings, and a proposal outside it was called invented whatever the
    # patent said. Traced over 208 proposed units, that allowlist was the single
    # largest cause of a correct reading being discarded — 36 of 55 drops. Every
    # PK and clearance unit failed by construction: US9233167's header reads
    # `AUCall (min · ng/mL)` verbatim and the unit was still dropped. A list of
    # units we happen to know is a claim about chemistry; whether a string is in
    # this document is a fact, and only the fact is checkable.
    ungrounded_units = []
    for a in assays:
        u = (a.get("unit") or "").strip()
        if u and not _states_unit(source_raw, u):
            logger.info("repair: dropped ungrounded unit %r on %r — that text "
                        "appears nowhere in this table's header, caption, "
                        "legend or cells", u, a.get("name"))
            ungrounded_units.append(f"{a.get('name')}={u}")
            a["unit"] = None

    # An unlabelled column is the ambiguous case: we cannot read its header
    # because it has none. Kept only when the surrounding prose says this table
    # is assay data — the same caption gate used elsewhere. Dropped rather than
    # rejected, so the columns that DO have headers still ship.
    from ..sources.uspto_assays import caption_assay_hint, table_legend
    unlabelled = [a for a in assays
                  if not (headers[a["index"]] if a["index"] < len(headers) else "").strip()]
    if unlabelled and not caption_assay_hint(f"{table.caption} {table_legend(table)}")[0]:
        logger.info("repair: dropped %d unheaded column(s) — no assay language in "
                    "the caption or legend to confirm they are measurements",
                    len(unlabelled))
        assays = [a for a in assays if a not in unlabelled]
        if not assays:
            raise Invalid(
                f"columns {[a['index'] for a in unlabelled]} have no header and "
                f"no assay language in the caption or legend — nothing confirms "
                f"they are measurements, and no other column was proposed")

    # Can this column KEY the rows at all? Decided once, from the whole column,
    # rather than testing each cell against one hard-coded shape.
    #
    # This is a contract, not a quality score: `apply_rule` will mint one record
    # per (row, assay) using this column as the compound id, and if the column
    # is a PEG chain length running 0,1,2,3… it mints records whose compound is
    # a polymer length. `outcome.measure` cannot see that — those records are
    # usable and contradict nothing.
    #
    # `_id_style` is itself the REPAIR for the one time this gate was wrong:
    # US9233167 names its compounds (`α-6-mPEG1-O-Morphine`) rather than
    # numbering them, and a correct column_map scored 0/23 against `_CID_PAT`.
    # A name is a perfectly good identifier; it just is not a code.
    style = _id_style([r[cid_i].text for r in data if len(r) > cid_i])
    if not style:
        sample = next((r[cid_i].text.strip() for r in data
                       if len(r) > cid_i and r[cid_i].text.strip()), "")
        raise Invalid(
            f"column {cid_i} cannot identify rows — its values are not distinct "
            f"non-measurement labels (e.g. {sample[:60]!r}). Name the column "
            f"that holds the compound id: without one there is nothing to "
            f"attach a measurement to")

    rule.payload["assays"] = assays
    return {"kind": COLUMN_MAP, "id_style": style, "assay_columns": len(assays),
            "dropped_names": [a.get("name") for a, _ in dropped],
            "dropped_units": ungrounded_units}
