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

LIBRARY_PATH = config.PACKAGE_ROOT / "data" / "name_rules.json"

# Longest replacement a synthesized rule may insert. One or two characters is a
# typesetting repair — a dropped bracket, a substituted letter pair. Anything
# longer is the model writing chemistry into the name, which is DATA, and this
# tier does not buy data. `ch` (the longest hand-written fix in
# `name_repair.PATTERNS`) is 2.
_MAX_INSERT = 4

# A site regex may not match more than this fraction of the document's own
# ALREADY-PARSEABLE names. See the module docstring: a rule that damages many
# good names while fixing one still shows a confirmed repair, so this cannot be
# measured after the fact.
_MAX_COLLATERAL = 0.02


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

    def to_dict(self) -> dict:
        return {"id": self.id, "site": self.site, "replacement": self.replacement,
                "note": self.note, "model": self.model, "epoch": self.epoch,
                "bought_for": self.bought_for, "confirms": self.confirms}

    @staticmethod
    def from_dict(d: dict) -> "NamePattern":
        return NamePattern(
            id=d["id"], site=d["site"], replacement=d.get("replacement", ""),
            note=d.get("note", ""), model=d.get("model", ""),
            epoch=d.get("epoch", ""), bought_for=d.get("bought_for", ""),
            confirms=int(d.get("confirms", 0)))


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
    if len(p.replacement) > _MAX_INSERT:
        return Grounded(
            False,
            f"replacement {p.replacement!r} is {len(p.replacement)} characters; "
            f"a rule may insert at most {_MAX_INSERT}. Longer than that is "
            f"writing chemistry into the name, which is data, not a rule")
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

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"schema": 1, "patterns": [p.to_dict() for p in self.patterns]},
            indent=1))

    def add(self, p: NamePattern) -> None:
        self.patterns = [x for x in self.patterns if x.id != p.id] + [p]
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
