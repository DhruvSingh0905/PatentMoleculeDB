"""Synthesized name-repair patterns: the kind, the contract checks, the library.

The `repair/rules.py` of the name tier. Same three jobs — define what a rule
IS, block contract violations before one is ever run, and persist what survives
— against a different unit of work: a corrupted compound NAME rather than a
table layout.

WHY A SEPARATE LIBRARY FROM `sources/name_repair.PATTERNS`
-----------------------------------------------------------
`name_repair.PATTERNS` is six HAND-WRITTEN forms, each with a `site`/`fix` pair
that may be arbitrary Python — `dropped_close_bracket`'s fix tries a `)` at
every offset in a 60-char window, which no regex expresses. Those stay where
they are, authored and reviewed.

This library holds what a MODEL proposed. It is deliberately a strict subset:
one site regex, one replacement template. That subset is not a limitation
discovered late, it is the point — a synthesized rule has to be checkable by
`ground()` without executing model-authored code, and "does this regex compile,
does it change the string, does it fire on the whole corpus" are answerable
about a regex and not about a callable.

Both libraries feed the same consumer. `as_corruption_patterns()` grounds every
entry here into the exact `CorruptionPattern` shape `name_repair` already
consumes, so the OPSIN + corroboration gate applies identically to a
hand-written form and a bought one.

WHY RULES ARE KEYED GLOBALLY, NOT PER PATENT
----------------------------------------------
The table tier keys a rule by layout fingerprint because layouts repeat. The
name tier's equivalent question — does a repair bought for one patent fire on
others — was measured over the 137-patent corpus before this module was
written, running each of `name_repair`'s six patterns against every failing
seed:

    pattern                  confirms  patents  ratio
    stray_opening_bracket          75       43    1.7
    stray_closing_bracket          70       49    1.4
    dropped_close_bracket           6        5    1.2
    digit_for_paren                 2        1    2.0

Three of four fire across dozens of documents; 4 purchases serve 98
patent-instances, so a library-first retry avoids 94 re-purchases. Hence: the
library is global, it is tried in full at $0 before any call is made, and a
pattern is stored by its own id rather than by the patent that bought it.

`digit_for_paren` firing in exactly one patent is the honest counter-example
and it is why a narrow rule is not treated as a failure — it was deliberately
scoped to one confirmed shape. A rule that never fires again costs its own
storage and nothing else.

WHAT `ground()` BLOCKS, AND WHAT IT DOES NOT
----------------------------------------------
Contract violations only, exactly as `rules.ground()` does for the table tier.
It has no opinion about whether a rule is a GOOD idea — that is
`name_outcome.measure`'s job, and it is answered by running the thing.

  - the site regex must compile, under a timeout-capable engine where available
  - it must actually match the name it was proposed for
  - applying it must CHANGE the string (a no-op rule is a wasted retry that
    looks like a considered answer)
  - the replacement must not be longer than `_MAX_INSERT` characters — a rule
    that inserts a whole fragment is writing chemistry, which is data, and this
    tier buys rules
  - the site regex must not be so general it fires on most of the document's
    own well-formed names. That check is the analogue of `rules.ground()`'s
    document-grounding: it cannot be seen in the output (a rule that mangles
    100 good names and fixes 1 still produces a confirmed repair for that 1)
    and so has to be blocked before the fact.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config

logger = logging.getLogger(__name__)

try:                                        # same optional engine as rules.py
    import regex as _re_engine              # per-match timeout
    _HAS_TIMEOUT = True
except ImportError:                         # pragma: no cover
    _re_engine = re                         # type: ignore[assignment]
    _HAS_TIMEOUT = False

# Bump on ANY change to `name_synthesize.SYSTEM`, its tool schema, or the
# rendering of a gap. Same contract as `rules.SYNTH_EPOCH`: the cache key and
# the escalation key both carry it, so a prompt change re-asks every layout
# that previously could not be solved, while an ADOPTED rule never expires.
# Forgetting to bump it replays a stale answer as if nothing had changed.
NAME_SYNTH_EPOCH = "n1"

# Failed attempts at ONE failure class before it is written off for this
# epoch. Mirrors `loop.MAX_ATTEMPTS`, which bounds paid attempts at one LAYOUT
# before that layout is escalated — the bound this tier was missing when the
# memory was lifted to corpus scale. Lives here rather than in config.py for
# the same reason `MAX_ATTEMPTS` lives in loop.py: it bounds the loop, not the
# corpus.
ESCALATE_AFTER = 3

LIBRARY_PATH = config.PACKAGE_ROOT / "data" / "name_rules.json"

# THE BOUND IS ON THE SIZE OF THE EDIT, NOT THE LENGTH OF THE REPLACEMENT.
#
# This was `len(replacement) <= 4`, and it was measuring the wrong thing. The
# first real sweep produced ZERO adopted rules across five configurations, all
# rejected identically:
#
#     puridin_to_pyridin: replacement 'pyridin' is 7 characters
#
# `puridin` -> `pyridin` is a ONE CHARACTER error (`u` for `y`) in US10155002,
# and the model had diagnosed it correctly. It wrote the natural spelling
# `s/puridin/pyridin/` rather than the tight `s/ur/yr/`, and the length rule
# threw it away — then threw the retry away for the same reason, because
# nothing about the feedback made a tighter spelling obviously available.
#
# What the bound is actually FOR is "do not let the model write chemistry into
# the name": the difference between a typesetting repair and supplying data.
# Levenshtein distance between the matched text and its replacement says that
# directly, and says it about every shape at once:
#
#     puridin -> pyridin      1   substitution, kept
#     ')' -> ''               1   stray bracket, kept
#     '' -> '('               1   dropped bracket, kept
#     '-2-yl' -> ''           5   AMPUTATION, refused
#     ... -> 20 chars        20   writing chemistry, refused
#
# The amputation case failing out at 5 is not a coincidence to be tuned around
# — it is the same boundary the coverage gate enforces downstream, reached
# independently.
_MAX_EDIT = 4


def _levenshtein(a: str, b: str) -> int:
    """Edit distance. Small enough to inline; no dependency for four lines."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

# A site regex may not match more than this fraction of the document's own
# ALREADY-PARSEABLE names. See the module docstring: a rule that damages many
# good names while fixing one still shows a confirmed repair, so this cannot be
# measured after the fact.
_MAX_COLLATERAL = 0.02


# ── the failure fingerprint ──────────────────────────────────────────────
#
# THE ASSAY TIER'S DESIGN, ON THE ONE AXIS THAT TRANSFERS. `gap.layout_signature`
# hashes a table's SHAPE and deliberately excludes its content, "otherwise every
# table is unique and nothing is ever reused". The equivalent move for names is
# NOT to hash the name: `name_gap`'s docstring records that shape clustering was
# measured and failed, because a name's shape is dominated by what kind of
# chemistry it is while a defect is a small local perturbation — the largest
# clusters came out as `1-methylpiperidin-4-yl` (1,048) and
# `1,1,1,3,3,3-Hexafluoropropan-2-yl` (735), which are clusters of chemistry.
#
# What IS content-independent is OPSIN's own account of the failure — the same
# source `sources/opsin.errors` feeds `name_gap`, and for the same stated
# reason: use the parser's diagnosis rather than guess from the string.
# Measured over
# the asserted-name gaps of 5 patents, 96 distinct raw error strings collapse
# to a handful of families once the offending token is replaced by its CLASS:
#
#     uninterpretable: 3 / : 2 / : 4 / : 5          133   one defect, four digits
#     uninterpretable: (frompeak1)5 / (peak2)3       16   annotation + footnote
#     Invalid fusion / Lambda convention / hydrogen   24   OPSIN grammar limits
#     Unmatched opening bracket found in :<name>      12   bracket faults
#
# Those 133 are a single family — confirmed independently, since ONE rule
# (`[0-9]$` -> ``) took 113 of them. A fingerprint that kept the digit would
# have split them four ways and bought four rules.
_ERR_KIND = [
    (re.compile(r"unmatched (opening|closing) bracket", re.I),
     lambda m: f"unmatched_{m.group(1).lower()}_bracket"),
    (re.compile(r"invalid fusion descriptor", re.I), lambda m: "invalid_fusion"),
    (re.compile(r"lambda convention", re.I), lambda m: "lambda_convention"),
    (re.compile(r"hydrogen addition at locant", re.I), lambda m: "hydrogen_addition"),
    (re.compile(r"could not find atom that|<stereoChemistry", re.I),
     lambda m: "stereo_unsupported"),
    (re.compile(r"unsure of the meaning of the words", re.I),
     lambda m: "substituent_only"),
    (re.compile(r"uninterpretable:\s*(.*?)(?:\s+The following|$)", re.I | re.S),
     lambda m: "uninterpretable"),
]

# The offending token, reduced to its class. `3` and `5` are the same defect;
# `(from peak 1)5` is a different one; `yloro` is a third.
_TOK_CLASSES = [
    (re.compile(r"^\d+$"), "digit"),
    (re.compile(r"^\([^)]*\)\d+$"), "paren+digit"),
    (re.compile(r"^[;,]"), "trailing_annotation"),
    (re.compile(r"^[A-Za-z]+$"), "word"),
]


def name_signature(opsin_error: str, name_text: str = "") -> str:
    """The fingerprint's PREIMAGE — the failure, before it is hashed.

    `kind|token class|position`, short and readable. Split from the hash for
    exactly the reason `gap.layout_signature` is: whatever goes into the hash
    has to be the same thing similarity is measured on, and a human or a model
    needs a form it can actually compare.

    The name itself never enters. OPSIN echoes the whole input into several of
    its messages (`Unmatched opening bracket found in :N-((4,6-dimethyl...`),
    which would make every failure unique — the precise trap the layout
    signature's content exclusion exists to avoid.
    """
    err = (opsin_error or "").strip()
    if not err:
        return "no_error|-|-"

    kind, tok = "other", ""
    for pat, name in _ERR_KIND:
        m = pat.search(err)
        if m:
            kind = name(m)
            if kind == "uninterpretable":
                tok = re.sub(r"\s+", "", m.group(1) or "")[:24]
            break

    tok_class = "-"
    if tok:
        tok_class = "text"
        for pat, cls in _TOK_CLASSES:
            if pat.match(tok):
                tok_class = cls
                break
        # `text` IS A CATCH-ALL AND HAD TO BE SPLIT. It held 258 of 604 gaps
        # (42.7%) in a single class — worse than the real catch-all, which is
        # only 4.8% — so one write-off discarded 258 unrelated compounds.
        # Reading the actual tokens shows why they cannot share a class:
        # `#` (27), `cyclohex-1-ol` (14), `Example49`, `Step1:2-bromo-4-...`,
        # `(racemic)`, `(MixtureofIsomers)`, a J. Med. Chem. citation, and
        # genuine chemistry like `5-methoxpyrimidine-2-carboxamide`.
        #
        # 290 of those 344 tokens are DISTINCT, so keying on the token itself
        # gives ~1 gap per class and no reuse at all. The skeleton below —
        # character classes, runs collapsed, plus a length bucket — was
        # measured to give 77 buckets over the same 344, ~4.5 gaps each, which
        # is the granularity the layout tier runs at (306 fingerprints across
        # 63 patents). It describes the DEFECT's shape, not the compound's:
        # `name_gap`'s docstring records that clustering whole NAMES fails
        # because chemistry dominates their shape, and the offending token is
        # a few characters rather than a molecule.
        if tok_class == "text":
            sk = re.sub(r"[a-z]+", "a", tok)
            sk = re.sub(r"[A-Z]+", "A", sk)
            sk = re.sub(r"\d+", "9", sk)
            tok_class = f"text:{min(len(tok), 9)}:{sk[:12]}"

    # Where the defect sits. A trailing footnote and a mid-name corruption need
    # different rules even when OPSIN reports them the same way.
    pos = "-"
    if tok and name_text:
        flat = re.sub(r"\s+", "", name_text)
        pos = ("end" if flat.endswith(tok) else
               "start" if flat.startswith(tok) else "mid")
    return f"{kind}|{tok_class}|{pos}"


def name_fingerprint(opsin_error: str, name_text: str = "") -> str:
    """Stable id for a FAILURE CLASS, independent of the compound involved.

    Two gaps share a fingerprint when a rule written for one should work on the
    other. Same contract as `gap.layout_fingerprint`, same 16 hex characters.
    """
    return hashlib.sha256(
        name_signature(opsin_error, name_text).encode()).hexdigest()[:16]


@dataclass
class NamePattern:
    """One model-proposed corruption form. Regex in, replacement out."""
    id: str
    site: str                     # a Python regex, matched against the name
    replacement: str              # what the matched span becomes ("" = delete)
    note: str = ""
    model: str = ""
    epoch: str = NAME_SYNTH_EPOCH
    # Never True for a synthesized rule. `name_repair`'s `trusted` means "OPSIN
    # acceptance alone is sufficient confirmation", which was granted to exactly
    # one hand-reviewed form. A bought rule always faces corroboration.
    trusted: bool = False
    # Bookkeeping, written by the loop after the gate passes.
    bought_for: str = ""          # patent id that paid for it
    confirms: int = 0             # repairs it has confirmed, lifetime
    # Failure-class SIGNATURES this pattern has actually resolved (readable
    # preimages, not hashes — they are what `solved_signatures` shows a model).
    # A pattern is still tried against every gap; this records what it is known
    # to be FOR, which is what makes the library legible as it grows.
    signatures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "site": self.site, "replacement": self.replacement,
                "note": self.note, "model": self.model, "epoch": self.epoch,
                "bought_for": self.bought_for, "confirms": self.confirms,
                "signatures": self.signatures}

    @staticmethod
    def from_dict(d: dict) -> "NamePattern":
        return NamePattern(
            id=d["id"], site=d["site"], replacement=d.get("replacement", ""),
            note=d.get("note", ""), model=d.get("model", ""),
            epoch=d.get("epoch", ""), bought_for=d.get("bought_for", ""),
            confirms=int(d.get("confirms", 0)),
            # Enumerated rather than `**d` so an unknown key in a future schema
            # cannot crash the load; a schema-1 file simply has no signatures.
            signatures=list(d.get("signatures", []) or []))


@dataclass
class Grounded:
    """`ground()`'s verdict. `ok` is the only thing the caller may act on."""
    ok: bool
    why: str = ""
    compiled: object = None


def _compile(pattern: str):
    """Compile under the timeout-capable engine when it is installed.

    Same reasoning as `rules.py`: a synthesized regex is untrusted input and
    `regex`'s per-match timeout is the only backstop against catastrophic
    backtracking. Without the package the code still runs, minus that net.
    """
    return _re_engine.compile(pattern)


def _search(pat, text: str):
    if _HAS_TIMEOUT:
        try:
            return pat.search(text, timeout=1.0)
        except TimeoutError:
            return None
    return pat.search(text)


def apply_pattern(p: NamePattern, name: str, compiled=None) -> list[tuple[str, tuple[int, int]]]:
    """Every repair `p` proposes for `name`, deterministic order.

    Returns `(repaired, span)` where `span` locates the REPLACEMENT inside the
    repaired string — the shape `name_repair._probe_window` needs to build a
    corroboration probe.

    ALL-OCCURRENCES COMES FIRST, and it is not an optimisation. A per-match
    version of this function was written first and could not repair a name
    carrying the SAME defect twice, which is the common case rather than the
    exotic one: `[1,1&#x2032;-biphenyl]` names carry the corruption once in the
    `4&#x2032;-` locant and again in the ring name, and fixing either alone
    still leaves OPSIN a string it rejects. The loop then read that as "the
    rule does not work", fed the failure back, and asked again — paying twice
    to be told the same thing about a rule that was correct.

    The per-match candidates are still generated after it, because a rule can
    be right about one site and wrong about another; OPSIN and the coverage
    gate decide which reading survives, exactly as everywhere else here.
    """
    pat = compiled if compiled is not None else _compile(p.site)
    out: list[tuple[str, tuple[int, int]]] = []
    seen: set[str] = set()

    matches = list(pat.finditer(name))
    if not matches:
        return out

    if len(matches) > 1:
        whole = pat.sub(p.replacement.replace("\\", "\\\\"), name)
        if whole != name:
            first = matches[0].start()
            out.append((whole, (first, first + len(p.replacement))))
            seen.add(whole)

    for m in matches:
        repaired = name[: m.start()] + p.replacement + name[m.end():]
        if repaired != name and repaired not in seen:
            seen.add(repaired)
            out.append((repaired, (m.start(), m.start() + len(p.replacement))))
    return out


def ground(p: NamePattern, names: list[str], clean_names: list[str]) -> Grounded:
    """Contract checks. Never a judgement about whether the rule is a good idea.

    `names` is EVERY broken name the rule was shown, not just the first. The
    rule is bought for a BATCH and may legitimately target the third name in it
    while leaving the first alone — checking only `names[0]` would reject a
    correct rule for the one thing this tier is built to exploit, which is that
    one typesetting defect produces many corrupted names.

    `clean_names` is this document's own names that ALREADY parse — the
    collateral-damage population. See the module docstring for why that check
    has to happen here rather than in the outcome gate.
    """
    if not p.site:
        return Grounded(False, "no site regex")
    if not names:
        return Grounded(False, "no names to ground against")
    try:
        pat = _compile(p.site)
    except Exception as e:
        return Grounded(False, f"site regex does not compile: {e}")

    matched = [n for n in names if _search(pat, n)]
    if not matched:
        return Grounded(
            False,
            f"the site regex matches none of the {len(names)} names it was "
            f"shown — it cannot repair any of them")

    if not any(apply_pattern(p, n, pat) for n in matched):
        return Grounded(False, "applying the rule leaves every name unchanged")

    # THE EDIT-SIZE BOUND. Measured per SITE — between the text the regex
    # matched and what replaces it — so a rule fixing the same one-character
    # defect five times in one name is judged as five edits of size 1, not one
    # edit of size 5. See `_MAX_EDIT` for what this replaced and why.
    worst = 0
    worst_pair = ("", "")
    for n in matched:
        for m in pat.finditer(n):
            d = _levenshtein(m.group(0), p.replacement)
            if d > worst:
                worst, worst_pair = d, (m.group(0), p.replacement)
    if worst > _MAX_EDIT:
        return Grounded(
            False,
            f"replacing {worst_pair[0]!r} with {worst_pair[1]!r} is an edit of "
            f"distance {worst}; a rule may change at most {_MAX_EDIT} "
            f"characters at one site. A larger change is not a typesetting "
            f"repair — it either writes chemistry into the name or removes a "
            f"whole fragment of it")

    # Collateral: how many already-good names does this rule also rewrite?
    if clean_names:
        hits = sum(1 for c in clean_names if _search(pat, c))
        frac = hits / len(clean_names)
        if frac > _MAX_COLLATERAL:
            return Grounded(
                False,
                f"the site regex also matches {hits} of {len(clean_names)} names "
                f"in this patent that ALREADY parse ({frac:.0%}); a rule that "
                f"rewrites correct names is not a repair")
    return Grounded(True, "", pat)


class NameRuleLibrary:
    """Every synthesized pattern, global across patents. Tried before spending."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or LIBRARY_PATH
        self.patterns: list[NamePattern] = []
        # fingerprint -> {"epoch", "signature", "note"}. See `is_escalated`.
        self.escalations: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for d in raw.get("patterns", []):
            try:
                self.patterns.append(NamePattern.from_dict(d))
            except (KeyError, TypeError):
                logger.warning("name_rules: skipping malformed entry %r", d)
        esc = raw.get("escalations", {})
        if isinstance(esc, dict):
            self.escalations = {k: v for k, v in esc.items() if isinstance(v, dict)}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic, like `rules.RuleLibrary.save` — a truncated library read back
        # as "no rules" would silently re-buy the entire corpus.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"schema": 2,
             "patterns": [p.to_dict() for p in self.patterns],
             "escalations": self.escalations}, indent=1))
        tmp.replace(self.path)

    # ── escalation memory ────────────────────────────────────────────────
    #
    # THE ASYMMETRY IS THE WHOLE POINT, and it is `rules.RuleLibrary.get`'s:
    # an adopted pattern is EVIDENCE — it was run against real names and it
    # held, so it stays true regardless of what changes here. An escalation is
    # not an answer; it is a record that our capability ran out on this failure
    # class, on a particular day, with a particular prompt. Treating the two
    # alike is what froze eight US10172859 layout gaps at a stale verdict while
    # the fixes that would have resolved them landed one after another.
    #
    # Without this the name tier had no memory of failure at all: measured, a
    # second run over US10376513 fixed 132 gaps from the library, bought
    # nothing new, and still spent $0.0764 re-attempting the same 34 gaps the
    # first run had already given up on.

    def is_escalated(self, fingerprint: str) -> bool:
        """Have we given up on this failure class, under THIS prompt?"""
        rec = self.escalations.get(fingerprint)
        return (bool(rec) and rec.get("epoch") == NAME_SYNTH_EPOCH
                and rec.get("failures", 0) >= ESCALATE_AFTER)

    def escalate(self, fingerprint: str, signature: str, note: str = "") -> None:
        """Record ONE failure against this class; give up only at
        `ESCALATE_AFTER`.

        THE FIRST VERSION GAVE UP AFTER A SINGLE FAILED BATCH, AND IT WAS A
        NET LOSS. Measured over 30 patents: 604 gaps, 537 of them SKIPPED
        without ever being attempted, 17 fixed, 2 rules bought. Whole patents
        went untouched — US20250163061A1 162 gaps / 162 skipped, US9133148
        135 / 135 — because a class had been closed by one failed batch on an
        earlier patent, in one case a patent with a single gap in it.
        `repair/loop.py` escalates a layout after `MAX_ATTEMPTS` PAID ATTEMPTS
        AT THAT LAYOUT; lifting the memory to corpus scale without lifting the
        attempt budget with it is what produced the veto.

        Counting is per class and per epoch, so a class gets `ESCALATE_AFTER`
        genuine attempts spread across whatever patents raise it, and bumping
        `NAME_SYNTH_EPOCH` resets the count as well as the verdict.
        """
        rec = self.escalations.get(fingerprint)
        if not rec or rec.get("epoch") != NAME_SYNTH_EPOCH:
            rec = {"epoch": NAME_SYNTH_EPOCH, "signature": signature,
                   "failures": 0, "note": ""}
        rec["failures"] = int(rec.get("failures", 0)) + 1
        rec["note"] = note[:200] or rec.get("note", "")
        self.escalations[fingerprint] = rec
        self.save()

    def solved_signatures(self, limit: int = 6) -> list[tuple[str, NamePattern]]:
        """`(signature, pattern)` for classes already solved.

        The other half of the assay tier's design: a model asked about a new
        failure is otherwise shown nothing about what has been learned, so it
        re-derives from scratch and cannot notice that a near-identical class
        was solved three patents ago. Cheap to send — a signature is ~20
        characters, not the 165 KB the library would be.
        """
        out: list[tuple[str, NamePattern]] = []
        for p in self.patterns:
            for sig in p.signatures:
                out.append((sig, p))
                if len(out) >= limit:
                    return out
        return out

    def add(self, p: NamePattern) -> None:
        self.patterns = [x for x in self.patterns if x.id != p.id] + [p]
        # A RULE OUTRANKS AN ESCALATION FOR THE SAME CLASS. When a batch fails
        # every gap in it is escalated, so one heterogeneous failed batch marks
        # classes that a later purchase then solves — observed directly:
        # `uninterpretable|digit|end` was recorded as escalated in the same run
        # where two rules resolved 118 of its members. Leaving both would let a
        # stale failure suppress a class we demonstrably handle.
        #
        # This is `rules.RuleLibrary.get`'s asymmetry applied at write time: the
        # rule is EVIDENCE (it ran against real names and held); the escalation
        # was only a record of where we stopped that day.
        solved = set(p.signatures)
        if solved:
            for fp, rec in list(self.escalations.items()):
                if rec.get("signature") in solved:
                    del self.escalations[fp]
        self.save()

    def has(self, pattern_id: str) -> bool:
        return any(p.id == pattern_id for p in self.patterns)

    def digest(self) -> str:
        """A few lines naming what has already been bought.

        The name tier's `RuleLibrary.digest()`. Sent in the user turn so the
        model does not re-propose a form the library already holds — which is
        the cheapest possible way to avoid paying twice for one rule.
        """
        if not self.patterns:
            return ""
        lines = ["\nRULES ALREADY IN THE LIBRARY (do not re-propose these; they "
                 "have already been tried on this name and did not fix it):"]
        for p in sorted(self.patterns, key=lambda x: -x.confirms)[:12]:
            lines.append(f"  {p.id}: s/{p.site}/{p.replacement}/  — {p.note[:70]}")
        return "\n".join(lines) + "\n"
