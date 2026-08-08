"""Compound-id anchoring: does the patent's OWN number sit next to this name?

WHY A SEPARATE MODULE
----------------------
This logic started as five private functions inline in `iupac_names.py`
(`_anchor_text`, `_find_cid`, `_CID_LEFT`, `_CID_RIGHT`, `_ANCHOR_BOUND`).
Anchoring is a distinct problem from name extraction — it never touches
OPSIN, never touches candidate-span generation, and needs none of that
machinery to be exercised. Coupling it to `extract_names` meant the only way
to check whether a proximity rule worked was to run a whole patent through
OPSIN first. Here it is a pure function of two strings (`anchor_text`,
`name`) and is called FROM `extract_names`, not merged into it — see the
one-line call site in `iupac_names.py::extract_names`, step 5.

THE RULE
--------
`description_text()` (`uspto_xml.py`) is built only from `<p>` tags and has
no `<heading>` content, so it never contains "Example 1" as a standalone
line — a full-text search for one on US8952177 returns zero hits despite 190
such headings in the raw XML. `anchor_text()` below is a second, heading-
INCLUSIVE flattening of the same XML, used only here; `description_text()`
and every `NamedCompound.start` offset into it are untouched.

Given that text, the rule is: for every occurrence of a name, look BOTH
directions for a compound id sitting immediately next to it —

    <id> <name>            "Example 1\\nracemic cis-2-{...}..."     (left)
    <id>: <name>           "Intermediate 1D: tert-butyl 5-(3-..."   (left)
    <name> (<id>)          "...triazin-4-amine (544); ..."          (right)

— and take whichever occurrence gives the id closest to the name, bounded so
a coincidental number far away never wins. Measured against 190 hand-checked
rows of US8952177 (`US8952177 Binding IUPAC Final (2).csv`), restricted to
extracted names that match a CSV row: the character distance from id to name,
when the assigned id was RIGHT, is 2-12 chars; when a nearby number existed
but was WRONG (a stray quantity, or an earlier compound's number in a bulleted
embodiment list), the distance was 162+ chars — two clean orders of magnitude
away, no overlap. `_ANCHOR_BOUND = 25` sits between those clusters with
headroom for the "Label ID: " shape (13-14 chars) without reaching anywhere
near the 162-char false cluster. A sweep at 25/30/40/50/80 (this module,
`docs`-less — the numbers are the record) shows 25 is not just "a bound that
still passes": widening it PAYS OFF IN CLASHES, NOT ANCHORS, past this point
— e.g. US10544143 goes 116 anchored @ 25 -> 107 @ 30 -> 87 @ 40 -> 75 @ 50,
because a wider window catches a second, unrelated id before it catches
nothing, and the clash guard (below) correctly refuses to pick between them.

CLASHES SURFACE, THEY DO NOT VANISH
------------------------------------
Generic ring/substituent fragments (`thiomorpholine-1,1-dione`,
`[1,2,4]triazin-4-amine`) are themselves OPSIN-valid names, so they appear as
their own "distinct structure" AND as a substring at the tail of dozens of
different full compound names — each occurrence can point at a different id.
Picking "smallest distance" among those would be picking arbitrary among
wrong answers, so `find_cid` does not: when a name's occurrences disagree
about the id, `AnchorResult.cid` is `None` and `clashed` is `True`, but
`candidates` carries every id anyone proposed, closest evidence first — never
a silent drop. A real example, US10214537: "morpholine-3-carboxamide" is a
tail fragment of several full names. One occurrence sits right after
"Intermediate 402B: ", another right before " (402)" — two DIFFERENT ids for
the same fragment, both real headings, and the honest answer is "this
fragment's occurrences don't agree," not a guess at 402 or 402B. (Under the
old digit-only rule this specific fragment never SAW "402B" at all — an
alphanumeric id — so it silently kept 402 as if uncontested. The clash is not
a regression the alnum support introduced; it is evidence the digit-only rule
was blind to.)

WHAT MOVED THE RATE, MEASURED (not assumed) — one ablation, three patents
---------------------------------------------------------------------------
Anchor rate = anchored / distinct OPSIN-resolved structures. `SELF_HEAL=0`,
no LLM, no network, `python3 -m patentdb3.verify` never invoked — this is
`extract_names()` called directly against cached `output_v3/uspto_xml/*.xml`.

| change                                   | US8952177 | US10214537 | US10544143 |
|-------------------------------------------|-----------|------------|------------|
| baseline (digit-only, both directions)     | 65.1%     | 76.9%      | 36.9%      |
| + right-side id may be alphanumeric        | 65.1%     | 77.0%      | 36.9%      |
| + left-side "ID:"/"ID;" may be alphanumeric, blacklist-guarded | 65.1% | 77.6% | 52.3% |
| + both (shipped)                           | 65.1%     | 77.7%      | **52.3%**  |

US10544143's headline number lives almost entirely in one change. Its
synthesis section numbers INTERMEDIATES as "Intermediate 1D:", "1E:", "2A:"
— alphanumeric, colon-separated, 133 such headings (vs. 92 "Example N").
The pre-existing rule required a bare digit run followed directly by
whitespace (`re.compile(r"(\\d+)\\s")`) and could never match "1D:" at all —
not a distance problem, a character-class problem. That is the corpus-wide
"alphanumeric ids ... 10.8% of example labels" limitation named going in.

The naive fix (any alphanumeric token immediately before a colon, anywhere)
was tried first and REJECTED by measurement: it also matched "Method 1:",
"Step 2:", "N2 sparged" (nitrogen gas, not a compound), turning a correct
digit-only anchor on US8952177 into a wrong one twice (65.1% -> 64.3%, one
new WRONG value: "2-methyltetrahydrofuran" — a solvent name that happens to
sit near "N2" — anchored to id "N2" instead of correctly finding nothing).
The shipped rule keeps that recall by gating on a closed blacklist
(`_NON_ID_WORD`) of the words that immediately precede a colon in this corpus
without being a compound label — "method", "step", "stage", "table", etc. —
checked against the text right before the candidate id, not a whitelist of
label words to require. A whitelist (`Example|Intermediate|Compound` before
the id) was also tried and produced a SMALLER gain on US10544143 (36.9% to
39.6%, not 52.3%) because OPSIN's accepted candidate span often starts a few
words into the true name ("tert-butyl 5-(3-..." keeps only "5-(3-..."), which
pushes the label word itself outside the 25-char window even though the id
is still well within it. The blacklist does not need the label in-window, so
it is not sensitive to that boundary loss.

Right-side alphanumeric support (`(I-0020)`, `(Z1)`, not just `(544)`) is
kept even though its measured effect on these three patents is small
(+1 on US10214537) — it is the same character-class gap on the other
convention, and the "10.8% corpus-wide" alphanumeric-id figure was not
scoped to these three patents.

PRECISION / RECALL AGAINST GROUND TRUTH
-----------------------------------------
`US8952177 Binding IUPAC Final (2).csv`, 190 rows. Of 238 distinct structures
`extract_names` resolves for US8952177, 67 have text that matches a CSV row
under stereo-prefix/salt-suffix normalization (this module's own matching,
not a shipped scorer — see `tests/test_anchor.py::test_csv_ground_truth` for
the exact rule). On that subset, digit-only and the shipped rule score
IDENTICALLY — 100% precision (63/63 anchored, 0 wrong), 94.0% recall
(63/67) — because every CSV-matched name in THIS patent uses the plain
"Example N" convention; the alphanumeric change earns nothing here and loses
nothing. Against all 190 CSV rows (not just the 67 matched — most of the
gap is unmatched EXTRACTION, a separate limitation, not anchoring):
recall = 63/190 = 33.2%.

WHAT IS STILL UNANCHORED, AND WHY THAT IS CORRECT
----------------------------------------------------
Run against US10544143's 106 remaining unanchored structures: 43 have no id
candidate at all, and inspection of every one sampled is a reagent or solvent
named in synthesis prose ("1,3-butanediol", "N-methylpyrrolidine",
"6-bromopyridin-3-amine") — never a patent-numbered compound, so finding
nothing is the right answer. One further shape this module does not solve:
"Intermediate 5A-1: 5-bromo-2-(...)... and 5-bromo-3-(...)..." — ONE id
covering TWO names in a "mixture of X and Y" heading. The second name's
nearest text is "... and ", not the id, which sits 80+ characters back,
outside any bound this module tested without flooding other patents with
false proximity matches. That is a real, known gap, left as `None` with no
candidates — not a clash, because nothing was found nearby at all.

CONFIGURATION
--------------
No feature flag lives here or is added to `core/config.py` — this module is
pure and always runs; whether `iupac_names.extract_names` calls it is gated
once, at that call's own boundary, by `config.IUPAC_NAMES` (already the
case before this module existed). The one tunable constant with a measured
value is `_ANCHOR_BOUND` (see above); the character-class and blacklist
constants below carry their own justification next to their definition.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tunables — each one's value is a measurement, not a default. See the module
# docstring for the sweep / ablation that produced it.

# Chars of window searched on the LEFT of a name, and chars looked ahead on
# the RIGHT. Sweeping 25/30/40/50/80 on all three headline patents showed
# monotonically WORSE anchor counts past 25 (more clashes, not more correct
# anchors) — see "THE RULE" above.
_ANCHOR_BOUND = 25
# The right-side shape is tighter by construction — "(id)" sits immediately
# after the name with no room for prose — so its lookahead can be smaller.
# 12 covers the longest realistic token (3 letters + 5 digits + 2 letters +
# "()" + up to 3 leading spaces = 14 in the worst case admitted by
# `_ID_TOKEN`; 12 is what every real corpus example measured needed).
_RIGHT_LOOKAHEAD = 12
# Safety valve on repeated occurrences of one name in a huge patent. Never
# observed to bind in this corpus.
_MAX_OCCURRENCES = 200

# A compound id, in either of this corpus's two conventions: bare digits
# ("1", "402"), or alphanumeric ("1D", "12a", "I-0020", "Z1", "5A-1"). At
# least one digit is required — a pure-letter token is never this corpus's
# compound id. Deliberately permissive on letters (0-3 prefix, 0-2 suffix):
# it is filtered by PROXIMITY and by the blacklist below, not by the shape of
# the token alone.
_ID_TOKEN = r"[A-Za-z]{0,3}-?\d{1,5}-?[A-Za-z]{0,2}"

# LEFT, shape 1: "<id> <name>" — a bare digit run immediately followed by
# whitespace, no separator. This is the ORIGINAL rule, unchanged: measured at
# 100% precision / 97.1% recall on the matched CSV subset when it was the
# only rule in play. Left digit-only (not routed through `_ID_TOKEN`) on
# purpose — widening this ONE shape to letters is what temporarily
# regressed US8952177 (see "WHAT MOVED THE RATE" above); the letters-in-play
# shape below is what replaced that attempt.
_CID_LEFT_PLAIN = re.compile(r"(\d+)\s")

# LEFT, shape 2: "<id>[:;] <name>" — an alphanumeric id, a colon or
# semicolon, then the name. This is the "Intermediate 1D: <name>" /
# "Example 746: <name>" combined-heading shape, and the only place
# alphanumeric ids get a chance on the left. The character class before the
# id token is deliberately narrow (whitespace/`.,;:()`/start-of-window) so a
# match cannot start in the middle of an unrelated alnum run.
_CID_LEFT_SEP = re.compile(rf"(?:^|[\s.,;:()])({_ID_TOKEN})\s*[:;]\s+")

# Words that legitimately precede "<number>:" without the number being a
# compound id — a closed list, the same idiom `iupac_names._TAIL_WORDS` and
# `_LEAD_JUNK` already use for the same reason (unbounded prose is not a
# vocabulary to enumerate; these specific false positives were MEASURED:
# "Method 1:" and "Step 2:" both cost a correct anchor before this existed).
_NON_ID_WORD = re.compile(
    r"\b(?:method|step|stage|note|scheme|procedure|part|section|table|"
    r"figure|equation|formula|claim|item|page|paragraph|entry|group|"
    r"embodiment|aspect|reaction)\s*$", re.I)

# RIGHT: "<name> (<id>)", immediately adjacent — the
# "...triazin-4-amine (544); ..." semicolon-list shape. Alphanumeric from the
# start (unlike the left plain rule, this shape was never digit-only in a
# way worth preserving separately — the parenthesized-immediately-after
# convention is unambiguous regardless of what characters the id itself uses).
_CID_RIGHT = re.compile(rf"^\s{{0,3}}\(({_ID_TOKEN})\)")

_TAG = re.compile(r"<[^>]+>")


def _clean_fragment(s: str) -> str:
    if "<" in s:
        s = _TAG.sub("", s)
    if "&" in s:
        s = html.unescape(s)
    return " ".join(s.split())


def anchor_text(xml: str) -> str:
    """`description_text()`, but with `<heading>` text put back in.

    This is the only place a compound's own number and its name are ever
    adjacent — see "THE RULE" in the module docstring. Tables are still
    dropped (same reason `description_text` drops them); this corpus is used
    for id anchoring ONLY, never for name/SMILES extraction, and callers
    must not treat its offsets as interchangeable with `description_text`'s.
    """
    m = re.search(r"<description\b[^>]*>(.*?)</description>", xml, re.S)
    if not m:
        return ""
    body = m.group(1)
    body = re.sub(r"<tables\b.*?</tables>", "\n", body, flags=re.S)
    parts = re.findall(r"<(?:heading|p)\b[^>]*>(.*?)</(?:heading|p)>", body, re.S)
    return "\n".join(t for t in (_clean_fragment(p) for p in parts) if t)


@dataclass(frozen=True)
class Candidate:
    """One id some occurrence of a name points to, and the closest evidence.

    `distance` is on the scale of ITS OWN direction only — LEFT counts chars
    from where the id starts to where the name starts; RIGHT counts chars
    from where the name ends to where the id starts. Both mean the same
    thing ("how much unrelated text sits between id and name") and are used
    the same way (smaller wins), but they are never compared as if the same
    ruler produced both — a left candidate and a right candidate for the same
    id are two separate pieces of evidence for one conclusion, not two
    measurements to average.
    """
    cid: str
    distance: int
    direction: str          # "left" or "right"
    context: str            # the matched text, trimmed — for a human to read


@dataclass(frozen=True)
class AnchorResult:
    """What was learned about one name's compound id — never a silent drop.

    Exactly one of three states, all visible to the caller:
      - `cid` set, `clashed=False`, `candidates` has exactly one entry: every
        occurrence that found an id agreed.
      - `cid=None`, `clashed=True`, `candidates` has 2+ entries, closest
        first: occurrences disagreed. This is the case the old inline
        version discarded outright; here the ids and their evidence are
        still here for a human or a later pass to resolve.
      - `cid=None`, `clashed=False`, `candidates=()`: no occurrence of this
        name had any id within reach. Not an error and not a clash — just
        nothing found (see "WHAT IS STILL UNANCHORED" in the module
        docstring for what this looks like in practice).
    """
    name: str
    cid: str | None
    clashed: bool
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)


def _consider(best_by_id: dict[str, Candidate], cid: str, distance: int,
              direction: str, context: str) -> None:
    cur = best_by_id.get(cid)
    if cur is None or distance < cur.distance:
        best_by_id[cid] = Candidate(cid=cid, distance=distance, direction=direction,
                                     context=" ".join(context.split())[:48])


def find_cid(text: str, name: str, *, bound: int = _ANCHOR_BOUND) -> AnchorResult:
    """The patent's own id for `name`, or the clash if its occurrences disagree.

    Every occurrence of `name` in `text` is a candidate — a name stated once
    in a SUMMARY embodiment list and again at its own Example heading is
    common, and only the latter has an id nearby. Each occurrence is checked
    in both directions (see the `_CID_LEFT_*` / `_CID_RIGHT` patterns above);
    the occurrence+direction with the smallest distance wins PER ID, so a
    genuine adjacent anchor always beats a coincidental one, but if two
    DIFFERENT ids each have their own close evidence, both are kept and
    surfaced as a clash rather than one being picked arbitrarily.
    """
    best_by_id: dict[str, Candidate] = {}
    start = 0
    seen = 0
    while seen < _MAX_OCCURRENCES:
        pos = text.find(name, start)
        if pos < 0:
            break
        seen += 1
        start = pos + 1
        end = pos + len(name)
        window = text[max(0, pos - bound):pos]

        m = None
        for m in _CID_LEFT_PLAIN.finditer(window):
            pass                                # last match = closest to `pos`
        if m is not None:
            _consider(best_by_id, m.group(1), len(window) - m.start(1), "left", window)

        m = None
        for cand in _CID_LEFT_SEP.finditer(window):
            if _NON_ID_WORD.search(window[:cand.start(1)]):
                continue                         # "Method 1:", "Step 2:", ...
            m = cand
        if m is not None:
            _consider(best_by_id, m.group(1), len(window) - m.start(1), "left", window)

        lookahead = text[end:end + _RIGHT_LOOKAHEAD]
        rm = _CID_RIGHT.match(lookahead)
        if rm is not None:
            _consider(best_by_id, rm.group(1), rm.start(1), "right", lookahead)

    if not best_by_id:
        return AnchorResult(name=name, cid=None, clashed=False, candidates=())
    candidates = tuple(sorted(best_by_id.values(), key=lambda c: c.distance))
    if len(candidates) > 1:
        return AnchorResult(name=name, cid=None, clashed=True, candidates=candidates)
    return AnchorResult(name=name, cid=candidates[0].cid, clashed=False, candidates=candidates)
