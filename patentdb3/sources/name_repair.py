"""Repair patent-text corruptions that break OPSIN on otherwise well-formed
IUPAC names — as DATA, not as hard-coded special cases.

WHY THIS EXISTS
----------------
The patent's OWN grant XML text is corrupted in small, specific ways that
defeat OPSIN on names that would otherwise parse. A corpus-wide survey (run
separately from this module, over all 137 cached patents) confirmed FIVE
distinct forms — each verified by an exact in-document restatement or a
direct before/after OPSIN test, reproduced independently in this module's
own investigation before any pattern below was written:

    A  dropped "("        a ring-substituent "(" that should open right
                           after "[" is simply missing.
    B  "(" as "1-"         "(" was replaced by the two characters "1-".
                           CONFIRMED IN US8952177 ONLY — see `PATTERNS`'s
                           entry for why this is not widened further.
    ch_as_ey  "ch" as "ey" the string still LOOKS like a plausible name,
                           which is what makes it dangerous.
    C  stray closing bracket   an extra ")"/"]"/"}" belongs nowhere — no
                           opener anywhere before it.
    C'/D  unmatched opener     an opening bracket never closed. TWO
                           different real causes produce the exact same
                           observable signature — see "C' AND D SHARE ONE
                           SITE" below for why they are one detector with
                           two competing hypotheses, not two detectors.

A fifth confirmed form, "]" mistyped as "}", is DELIBERATELY NOT a pattern
here — see "FORM E: WHY IT HAS NO PATTERN" below.

This module hard-codes ONE engine that consumes a `CorruptionPattern`
list — `PATTERNS` below — so a form the survey finds tomorrow is a new
tuple entry, not a rewrite. `site` and `fix` are both plain callables (not
tied to `re.Pattern`) specifically so a form like C'/D, whose fix cannot be
expressed as a single regex substitution, still fits the same registry
shape as a one-line regex-and-replace form like A.

THE GOVERNING CONSTRAINT: NEVER REPAIR WHAT CANNOT BE CONFIRMED
-------------------------------------------------------------------
Inventing a bracket (or a digit, or two letters) is worse than leaving a
name unextracted: a wrong structure silently attributed to a real compound
is undetectable downstream. So every candidate repair goes through TWO
independent gates before it is ever returned, and a candidate that clears
only one of them is discarded, not guessed at:

1. **OPSIN acceptance** — mandatory for every pattern, no exception. A
   candidate that does not parse costs nothing and is dropped.
2. **Corroboration OR trust** — OPSIN acceptance alone is not enough for a
   pattern whose repair could, in principle, coincidentally also produce a
   DIFFERENT valid molecule instead of the intended one. Each
   `CorruptionPattern` carries `trusted: bool`:

   - `trusted=True` (only `dropped_open_paren`, unchanged since this
     module's first version): the corruption's SITE and FIX are both
     structurally unambiguous — a ring-substituent `(` has exactly one
     place it can go to balance the brackets in this corpus's naming
     convention, matching the same shape already measured safe
     (`anchor.py::_repair_dropped_open_paren`, 16 occurrences / 7 patents,
     0 disputed on hand review). OPSIN acceptance is treated as sufficient
     confirmation, same as that precedent.
   - `trusted=False` (every other pattern, including all three added after
     the corpus survey widened the confirmed surface to five forms): the
     fix is a GUESS at what the corruption replaced, and a guess that
     happens to also parse is not evidence it is the RIGHT molecule — see
     "THE GOVERNING CONSTRAINT" above. These patterns additionally require
     **corroboration**: the ~24-30 char probe around the fix (`_CORR_WINDOW`
     chars each side, whitespace-insensitive — see `_normalize_ws`) must
     appear VERBATIM somewhere else in the same patent's own text. Measured
     directly on confirmed corpus cases by exact count, not estimated (see
     `PATTERNS` below): the probe recurs 4 times elsewhere in US8952177 for
     `digit_for_paren`'s cid 92 case, 11 times elsewhere in US10376513 for
     `ch_as_ey`'s case, and once elsewhere in US9303033 for
     `dropped_close_bracket`'s "4(3H-one" case — but ZERO times elsewhere in
     US10035778 for that same pattern's "...)aminobenzoic acid" case (4
     occurrences, all corrupted the same way, no clean restatement anywhere
     in that document). That last case is not a bug: it is the mechanism
     correctly declining, per the governing constraint, exactly the outcome
     the corpus survey flagged as possible ("at least one case has no
     recovery path at all — US8952177 Example 187"). Corroboration is not a
     coincidence check — the SEARCH KEY is the CORRECTED text, which by
     construction differs from what sits at the original corrupted
     position, so a hit can only come from a genuinely different,
     correctly-spelled occurrence. (An EARLIER version of this paragraph
     said "15 times" and "20 times" — round numbers from an early
     investigation pass that were never re-verified against the exact probe
     string before being written down. Recomputed directly from
     `generate_candidates` + `_probe_window` + `_normalize_ws` against the
     cached XML for this report; the earlier numbers were wrong, not
     rounded, and are corrected here rather than repeated.)

Both gates run inside `repair_name`; `generate_candidates` alone (no OPSIN,
no corroboration) is exposed separately so the pure, deterministic part of
this module is testable and auditable without a java subprocess.

C' AND D SHARE ONE SITE, WITH TWO COMPETING HYPOTHESES
---------------------------------------------------------
An opening bracket that is never closed by the end of a name has exactly
the same OBSERVABLE signature under two different REAL causes: a spurious
extra opener that should simply be deleted (Form C'), or a real closing
bracket that was dropped somewhere downstream (Form D) — and bracket
counting alone cannot tell them apart; the difference is only visible in
which repair actually confirms. So `stray_opening_bracket` and
`dropped_close_bracket` both fire on the identical site detector
(`_unmatched_openers`) and propose OPPOSITE repairs (delete the opener, vs.
insert a closer somewhere after it); `generate_candidates` returns
candidates from BOTH, and `repair_name` accepts whichever one (if either)
clears OPSIN + corroboration. Neither pattern guesses which cause applies —
confirmation decides, not this module.

`dropped_close_bracket`'s fix is the one genuinely open-ended candidate
generator in this registry: the corpus survey's critical finding is that
the missing `)` often has NO delimiter to anchor on at all
(`aminobenzoic`, not `amino-benzoic` — a hyphen-anchored search missed 97
of its own 99 instances in US10035778 for exactly this reason). So
`_fix_dropped_close` does not look for a hyphen, a capital letter, or any
other boundary marker — it tries inserting `)` at EVERY character offset
within a bounded window after the unmatched opener (`_GAP_WINDOW = 60`
chars, wide enough to cover both confirmed corpus distances — 2 chars for
US9303033, 49 chars for US10035778) and lets OPSIN + corroboration filter
the resulting ~60 candidates down to zero or one. This is safe specifically
BECAUSE corroboration is a verbatim, whole-fragment substring match: an
insertion at the wrong offset essentially never happens to both parse AND
recur verbatim elsewhere in the same patent by chance.

FORM E: WHY IT HAS NO PATTERN
--------------------------------
The corpus survey also confirmed a fifth form — "]" mistyped as "}" — and
explicitly determined it must NOT be repaired: in US10172859 the
"corrupted" `}` form outnumbers the correct `]` form 143:4 IN THE SAME
SLOT, and OPSIN parses both to the IDENTICAL SMILES (verified directly
against that patent's own text: `...thieno[3,2-d}pyrimidin-4-yl...` and
`...thieno[3,2-d]pyrimidin-4-yl...` both resolve to
`ClC=1C(=NC=CN1)C(O)C1=CC(=C(C=C1)F)C=1C2=C(N=CN1)C=C(S2)N2CCOCC2`). There
is no yield to recover and non-zero risk in touching it, so this registry
has no `CorruptionPattern` for it, and never will without new evidence.

It does not need a special-case guard, either: `repair_name`'s own
short-circuit (see "WHAT THIS DOES NOT DO" below) checks the ORIGINAL name
against OPSIN before trying any pattern, and OPSIN already accepts the `}`
form as-is — so a Form-E name never reaches pattern-matching at all, the
same way any other already-valid name does not.
`test_bracket_type_substitution_is_never_repaired` proves this directly
against the real US10172859 text rather than relying on the short-circuit
argument alone.

WHAT THIS DOES NOT DO
----------------------
It does not decide WHEN to call itself. `repair_name(name, document_text)`
assumes the caller already tried resolving `name` directly and it failed —
the same contract `iupac_names.py` already has with OPSIN. As a safety net
(not a substitute for that contract) `repair_name` re-checks the ORIGINAL,
unmodified `name` against OPSIN as element 0 of its own batch and returns
`None` immediately if it already parses — a name that is not actually
broken is never "repaired" into a different, still-valid string.

It does not know its callers. Two of them now exist — `table_names.py` calls
`repair_names` on every name-column cell no dewrap variant resolved, and
`iupac_names.py` calls it on every description seed no variant resolved — but
both decide FOR THEMSELVES when a name has failed and what document text to
corroborate against. Nothing about this module changed to accommodate either
(`PATTERNS`, `generate_candidates`, the two gates and their order are
byte-identical); the only addition is `repair_names`, a batched form of
`repair_name` that exists because those callers hand it thousands of names per
patent and OPSIN is a java subprocess. See "MEASURED ON THE WIRED CALLERS"
below for what the two together actually recover.

MEASURED ON THE WIRED CALLERS
-------------------------------
Both callers were A/B'd by stubbing this module out and changing nothing else,
over all 137 cached patents:

    caller                     structures off -> on   rows carrying .repair
    iupac_names.extract_names       54,206 -> 54,273           79
    table_names.extract_table_names  8,308 ->  8,326           18

The description route's 79 confirmed repairs net +67 structures; the other 12
resolve to a structure some other span in the same patent already produced and
are collapsed by that module's InChIKey dedup. All 18 table recoveries are
`corroborated` — none is accepted on OPSIN alone — and they come entirely from
the three BRACKET-STACK patterns (`stray_opening_bracket` 14,
`stray_closing_bracket` 3, `dropped_close_bracket` 1).

**`ch_as_ey` recovers nothing through either caller, and the reason is worth
recording rather than reading as the pattern misfiring.** Its single confirmed
corpus case, US10376513 compound #69, sits in a table cell that carries THREE
defects at once: the line-wrap spaces `table_names` removes, the `eyloro`
substitution this pattern fixes, and a footnote digit fused onto the tail
(`propan-2-ol2`, from `<sup>2</sup>` losing its tags) which is deliberately
out of scope for both modules. Measured by running the current tree on that
exact cell: the dewrap works, this pattern's fix is proposed, and OPSIN still
rejects the result purely on the trailing `2`. The pattern is correct; it is
blocked downstream of itself.

MEASURED FALSE-POSITIVE RATE
------------------------------
See `patentdb3/tests/test_name_repair.py` for the full battery. Summary
(also see that file's module docstring for the exact corpus commands run to
produce these counts):

- `dropped_open_paren`'s site regex fires 16 times across the 137-patent
  corpus (7 patents); every one is a genuine corruption (this is
  `anchor.py`'s own measurement, reproduced independently here since this
  module reimplements the same regex as one row of its own data-driven
  registry rather than importing `anchor.py`).
- `digit_for_paren`'s site regex fires 2 times corpus-wide, both in
  US8952177 (cids 92 and 105), both genuine corruptions, both corroborated
  and recovered.
- `ch_as_ey`'s bare site regex (any "ey" followed by 2+ lowercase letters)
  fires 218 times corpus-wide — the overwhelming majority are NOT
  corruptions (author surnames in citations, "16-sulfaneylidene", a real,
  if OPSIN-unfriendly, sulfoximine locant morpheme that recurs verbatim
  throughout US11420968, and ordinary prose). None of those false SITE
  matches becomes a false REPAIR: `_fix_ch_for_ey` always substitutes the
  literal string "ch", so on a non-corruption the resulting candidate is
  gibberish ("sulfaneylidene" -> "sulfanchlidene") that OPSIN rejects
  outright — verified directly against the corpus's actual
  "16-sulfaneylidene" text. Zero of the 218 site matches produce an
  OPSIN-accepted candidate other than the one genuine case (US10376513,
  compound #69, corroborated 20 times).
- The three forms added after the corpus survey widened the confirmed
  surface to five (`stray_closing_bracket`, `stray_opening_bracket`,
  `dropped_close_bracket`), run as `repair_name` would run them — every
  `<heading>` text across all 137 cached patents, `document_text` the
  patent's own `anchor_text` — fire their SITE detectors far more often
  than they CONFIRM: 693 / 212 / 212 site matches respectively (bracket
  imbalance inside a heading span is common — line-wrapped table cells,
  headings that are genuinely fragments of a longer name — most of it
  either not a real corruption or one this registry's specific fix
  hypotheses do not happen to reconstruct), producing 3 / 6 / 1 CONFIRMED
  repairs. Every one of those 10 was checked by hand against the resolved
  structure; none is wrong. That 693/212/212 -> 3/6/1 drop is the
  corroboration gate doing its job, not the mechanism failing to fire — a
  candidate that merely parses is common; one that ALSO recurs verbatim
  elsewhere in the same patent is rare, which is exactly why it is trusted.
- Net measured false-positive rate over every pattern's corpus-wide firing
  (16 + 2 + 218 + 693 + 212 + 212 = 1,353 site matches, 20 confirmed
  repairs across all six registry entries): **0 / 20** confirmed repairs
  are wrong on hand review. This is not the same claim as "0% of all
  corruptions in the corpus are mishandled" — it is the false-positive rate
  of what THIS registry's patterns, run over this corpus, actually
  confirm. What it cannot measure: corruption forms neither this module
  nor the survey agent has found yet, the FALSE-NEGATIVE rate (real
  corruptions these site detectors never match, or match but cannot
  corroborate — `dropped_close_bracket`'s own US10035778 "aminobenzoic"
  case is a confirmed instance of the latter), and whether a LARGER,
  hand-labeled sample would surface a false positive this review missed —
  both are open, see the task report for what is not determined.

DETERMINISM
-----------
No randomness, no set/dict iteration leaking into output, no clock or
locale dependence:

- `PATTERNS` is a `tuple`, iterated in the fixed order it is written in —
  never a `set` or unordered `dict`.
- `generate_candidates` iterates `PATTERNS`, then each pattern's own `site`
  callable (a wrapped `re.Pattern.finditer` for the regex-based patterns,
  a left-to-right bracket-stack scan for the others), then each site's own
  `fix` list in the order it was built — every one of those is a fixed,
  textual order, independent of hash seed or interpreter. Nothing in this
  module iterates a `set` or an unordered `dict` to produce output.
- `repair_name` returns the FIRST candidate (in that same fixed order) that
  clears both gates — not "the best" by any scored comparison that could
  vary, just first-confirmed-wins.
- OPSIN itself (`py2opsin`, a local java subprocess) is the one piece of
  this module not authored here; `test_name_repair.py::
  test_generate_candidates_is_deterministic_across_repeats` and
  `..._across_fresh_interpreters` prove the OPSIN-free half (pattern
  application) is stable across 50 in-process repeats and two fresh
  subprocess interpreters, the same proof `sources/reagents.py` uses.
  `test_repair_name_is_deterministic_across_fresh_interpreters` extends
  that proof through the OPSIN-touching half on a small battery, skipped
  (not failed) where OPSIN cannot run.

CALL SIGNATURE
---------------
    from patentdb3.sources.name_repair import repair_name, repair_names

    result = repair_name(corrupted_name, document_text=patent_flat_text)
    results = repair_names([n1, n2, ...], document_text=patent_flat_text)

`repair_names` is the form both wired callers use: same gates, same registry
order, same first-confirmed-wins, one entry per input in order, but TWO OPSIN
subprocess launches for the whole list instead of two per name. `repair_name`
is literally `repair_names([name], document_text)[0]`, so there is one
implementation of the gates and no second copy to drift.

    if result is not None:
        # result.repaired, result.smiles, result.inchikey,
        # result.pattern_id, result.confirmation ("opsin_only" | "corroborated")
        ...

`document_text` should be a flattened, tag-free string containing the
patent's own text (the same convention `anchor.py::anchor_text` and
`uspto_xml.py::description_text` already use) — corroboration is a literal
substring search, so tag markup interleaved into a name would defeat it.
Omitting `document_text` (or passing `""`) still allows `trusted=True`
patterns to confirm on OPSIN acceptance alone; `trusted=False` patterns
then never confirm, by design — see "THE GOVERNING CONSTRAINT" above.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable

from ..core import config
from .opsin import batch as _opsin_batch_shared

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The pattern registry — corruption forms as DATA. Adding a sixth form the
# survey confirms is one more `CorruptionPattern(...)` entry below; nothing
# else in this file changes shape.

@dataclass(frozen=True)
class CorruptionPattern:
    """One confirmed corruption shape: where it might sit (`site`), what
    candidate repair(s) look like there (`fix`), and how much trust that
    repair is owed on its own (`trusted`) — see the module docstring's
    "THE GOVERNING CONSTRAINT" for what `trusted` controls.

    `site` and `fix` are plain callables, not `re.Pattern`/regex-substitution
    objects: `site(name)` returns every candidate span `(start, end)` this
    pattern might apply to (fixed, left-to-right order), and
    `fix(name[start:end])` returns ONE OR MORE candidate replacement strings
    for that span. Most patterns return exactly one; `dropped_close_bracket`
    is the reason `fix` returns a LIST rather than a single string — see
    "C' AND D SHARE ONE SITE" in the module docstring.
    """
    id: str
    description: str
    site: Callable[[str], list[tuple[int, int]]]
    fix: Callable[[str], list[str]]
    trusted: bool


def _regex_site(pattern: re.Pattern) -> Callable[[str], list[tuple[int, int]]]:
    """Wrap a compiled regex as a `site` callable — every match, in order."""
    def _site(name: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in pattern.finditer(name)]
    return _site


def _fix_dropped_paren(matched: str) -> list[str]:
    # matched is "[" + a digit-led substituent run + ")" with no opening
    # "(" — insert it right after "[", the only place this corpus's naming
    # convention ever puts it for a locant-prefixed ring/chain substituent.
    # Same shape as `anchor.py::_DROPPED_OPEN_PAREN` / `_repair_dropped_open_
    # paren`, reimplemented here as one registry row rather than imported,
    # so this module has no dependency on a sibling module owned elsewhere.
    return ["[(" + matched[1:]]


def _fix_digit_for_paren(matched: str) -> list[str]:
    # matched is "-<locant digits>-1-": a chain locant, then the two
    # characters "1-" standing in for a single "(". Keep the locant and its
    # leading hyphen, drop the trailing "1-", open the paren instead.
    return [matched[:-2] + "("]


def _fix_ch_for_ey(matched: str) -> list[str]:
    return ["ch"]


def _fix_delete(matched: str) -> list[str]:
    return [""]


# ---------------------------------------------------------------------------
# Bracket-stack scanners shared by the three bracket-imbalance patterns
# (`stray_closing_bracket`, `stray_opening_bracket`, `dropped_close_bracket`)
# — bracket-TYPE aware ("}" never closes "["), which is exactly why Form E
# ("]" mistyped as "}") does not get misidentified as one of these: see
# "FORM E: WHY IT HAS NO PATTERN" in the module docstring. General on
# purpose — these replace what would otherwise be a hand-written regex per
# real corpus string, and were verified instead against several
# independent real corpus strings (see `test_name_repair.py`), not tuned to
# one.

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def _unmatched_closers(name: str) -> list[tuple[int, int]]:
    """A closing bracket with no matching opener before it — Form C's own
    signature ("a spurious extra closing bracket belonging nowhere").
    """
    stack: list[str] = []
    out: list[tuple[int, int]] = []
    for i, ch in enumerate(name):
        if ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            want = _CLOSE[ch]
            if stack and stack[-1] == want:
                stack.pop()
            else:
                out.append((i, i + 1))
    return out


def _unmatched_openers(name: str) -> list[tuple[int, int]]:
    """An opening bracket never closed by the end of `name` — the SHARED
    signature of Form C' and Form D; see "C' AND D SHARE ONE SITE" in the
    module docstring for why one scanner backs two patterns with opposite
    fixes.
    """
    stack: list[tuple[str, int]] = []
    for ch_i, ch in enumerate(name):
        if ch in _OPEN:
            stack.append((ch, ch_i))
        elif ch in _CLOSE:
            want = _CLOSE[ch]
            if stack and stack[-1][0] == want:
                stack.pop()
            # an unmatched CLOSER is Form C's problem, not this scan's.
    return [(pos, pos + 1) for _, pos in stack]


# Chars searched, after an unmatched opener, for where a dropped ")" might
# belong. Bounded so `dropped_close_bracket`'s candidate fan-out stays
# small and predictable; 60 comfortably covers both confirmed corpus
# distances (US9303033's "4(3H-one" case: 2 chars; US10035778's
# "...)aminobenzoic acid" case: 49 chars — the far end of what has been
# observed, from the unmatched opener all the way past the intervening
# ring-substituent group to where "amino" runs into "benzoic").
_GAP_WINDOW = 60


def _dropped_close_site(name: str) -> list[tuple[int, int]]:
    out = []
    for pos, _ in _unmatched_openers(name):
        end = min(len(name), pos + 1 + _GAP_WINDOW)
        if end > pos + 1:
            out.append((pos + 1, end))
    return out


def _fix_dropped_close(window: str) -> list[str]:
    # No delimiter is assumed — see the module docstring's "C' AND D SHARE
    # ONE SITE" for why a hyphen/capital-letter anchor would miss most real
    # instances. Every offset is tried; OPSIN + mandatory corroboration
    # (this pattern is `trusted=False`) is what makes that safe.
    return [window[:i] + ")" + window[i:] for i in range(len(window) + 1)]


# A `<sup>` footnote marker fused onto the end of a name. `uspto_xml._text`
# keeps a superscript's CONTENT when tags are stripped, so a table cell reading
# `propan-2-ol` with a footnote 2 arrives as `propan-2-ol2`.
#
# THE LOOKBEHIND IS TWO LETTERS, AND EVERY PART OF THAT IS MEASURED. Over the
# 229 asserted-name gaps ending in a digit across the 30-patent sample plus
# US10376513, counting how many parse once the trailing digits are removed:
#
#     prev char   n     strips and parses
#     'e'        118    116     propanenitrile4, amine3, acetamide3
#     'l'         19     17     propan-2-ol2
#     'd'          3      2     ...-5,5,6,6-d4      <- DEUTERIUM, see below
#     ' '         69      0     ...carbonitrile Example 3
#     ')'         18      0     ...benzonitrile(from peak 1)5
#     '-'          2      0     truncated names
#
# The unqualified `[0-9]$` a model proposed fires on all 229 and can only ever
# help 135 of them. Requiring a letter fixes the space/paren/hyphen cases —
# but a single letter is still wrong, and the counter-example is the dangerous
# kind: `...piperazin-2-one-5,5,6,6-d4` is a DEUTERATED analogue, and 2 of
# those 3 parse after the `4` is stripped. They pass OPSIN, they pass the
# coverage floor (one character out of ~50), and they silently destroy the
# isotope labelling — a false repair that looks exactly like a success.
#
# Two letters admits every real footnote (`le`, `ol`, `ne`) and excludes `-d4`
# by construction rather than by a special case.
_FOOTNOTE_TAIL = re.compile(r"(?<=[a-z]{2})\d+$")


def _footnote_site(name: str) -> list[tuple[int, int]]:
    m = _FOOTNOTE_TAIL.search(name.rstrip())
    return [(m.start(), m.end())] if m else []


# Real corpus citations for every pattern below are reproduced, verbatim
# positions and all, in `test_name_repair.py` — nothing here is a
# constructed example.
PATTERNS: tuple[CorruptionPattern, ...] = (
    CorruptionPattern(
        id="footnote_digit_tail",
        description=(
            "a <sup> footnote marker fused onto the end of a name, e.g. "
            "US10376513 TABLE-US-00001 cid 69's "
            "'...azetidin-1-yl)propan-2-ol2' for '...propan-2-ol'. "
            "OPSIN names the culprit itself: 'unparsable due to the "
            "following being uninterpretable: 2'."
        ),
        site=_footnote_site,
        fix=lambda _: [""],
        # TRUSTED, AND CORROBORATION IS NOT AVAILABLE HERE EVEN IN PRINCIPLE.
        # This shipped as `trusted=False` first and recovered exactly 0 of 642
        # gaps: the gate asks that the corrected spelling occur elsewhere in
        # the patent as published, and the whole shape of this defect is that
        # the footnote is fused onto the name in the ONE table cell where the
        # compound is named. Measured on US10376513, `corrected spelling occurs
        # verbatim in doc` was False for every case, including ones whose
        # stripped name parses perfectly.
        #
        # So OPSIN acceptance is the only gate, and it is enough here BECAUSE
        # THE SITE CANNOT REACH A GOOD NAME: over 2,304 names that already
        # parse across 14 patents, this site fires on ZERO of them (0.000%) —
        # the collateral question `name_rules.ground()` asks of a bought rule,
        # asked of this one before trusting it. The `$` anchor means it can
        # only ever delete trailing digits, and the two-letter lookbehind
        # already excludes the deuterium case that OPSIN alone would wave
        # through.
        trusted=True,
    ),
    CorruptionPattern(
        id="dropped_open_paren",
        description=(
            "a ring-substituent '(' that should open immediately after '[' "
            "is missing, e.g. US8952177 Example 92's own heading restates "
            "'...6-[5-methylpyridin-2-yl)methoxy]...' for "
            "'...6-[(5-methylpyridin-2-yl)methoxy]...'."
        ),
        site=_regex_site(re.compile(r"\[\d[\w,'*-]{0,80}?-yl\)")),
        fix=_fix_dropped_paren,
        trusted=True,
    ),
    CorruptionPattern(
        id="digit_for_paren",
        description=(
            "'(' replaced by the two characters '1-', e.g. US8952177 "
            "Example 92's heading: '...2-methyl-4-1-trifluoromethoxy)benzyl"
            "...' for '...2-methyl-4-(trifluoromethoxy)benzyl...'. CONFIRMED "
            "IN US8952177 ONLY (cids 92 and 105) — an automated OPSIN-"
            "plausibility screen suggested this form in 17 more patents, but "
            "manual inspection showed those 'repairs' work by DELETING a "
            "real ring-locant digit to force a parse, not by fixing a real "
            "corruption. The site regex below is intentionally scoped to "
            "the confirmed shape only and is NOT widened on plausibility "
            "alone; see `test_digit_for_paren_site_regex_does_not_widen_"
            "past_confirmed_scope`."
        ),
        site=_regex_site(re.compile(r"-\d{1,2}-1-(?=[A-Za-z]{4,25}\))")),
        fix=_fix_digit_for_paren,
        trusted=False,
    ),
    CorruptionPattern(
        id="ch_as_ey",
        description=(
            "'ch' replaced by 'ey', e.g. US10376513's compound #69 table "
            "row: '...5-eyloro-6- fluoro-2-methoxyphenyl...' for "
            "'...5-chloro-6-fluoro-2-methoxyphenyl...'."
        ),
        site=_regex_site(re.compile(r"ey(?=[a-z]{2,})")),
        fix=_fix_ch_for_ey,
        trusted=False,
    ),
    CorruptionPattern(
        id="stray_closing_bracket",
        description=(
            "a spurious extra closing bracket belonging nowhere, e.g. "
            "US10266548: '...pyridin-4-ylamino)pyrimidino)pyrimidin-2-yl...' "
            "for '...pyridin-4-ylamino)pyrimidinopyrimidin-2-yl...' (the "
            "second ')' has no opener). Also confirmed independently in a "
            "US11427578 heading ('...oxazin-2-yl]amino}...' -> "
            "'...oxazin-2-ylamino}...', a stray ']')."
        ),
        site=_unmatched_closers,
        fix=_fix_delete,
        trusted=False,
    ),
    CorruptionPattern(
        id="stray_opening_bracket",
        description=(
            "a spurious extra opening bracket belonging nowhere — the "
            "delete-the-opener hypothesis at an unmatched-opener site. See "
            "'C\\' AND D SHARE ONE SITE' in the module docstring: this "
            "pattern and `dropped_close_bracket` fire on the identical site "
            "detector and propose opposite fixes; which (if either) is "
            "right is decided by OPSIN + corroboration, not guessed here."
        ),
        site=_unmatched_openers,
        fix=_fix_delete,
        trusted=False,
    ),
    CorruptionPattern(
        id="dropped_close_bracket",
        description=(
            "a closing bracket was dropped, often with NO delimiter at the "
            "drop point, e.g. US9303033: "
            "'...triazin-4(3H-one' for '...triazin-4(3H)-one', and "
            "US10035778 (99 instances): "
            "'...tetrahydropyrimidin-2-yl)aminobenzoic acid' for "
            "'...tetrahydropyrimidin-2-yl)amino)benzoic acid' — 'amino' "
            "runs directly into 'benzoic' with no hyphen to anchor on."
        ),
        site=_dropped_close_site,
        fix=_fix_dropped_close,
        trusted=False,
    ),
)

_PATTERNS_BY_ID: dict[str, CorruptionPattern] = {p.id: p for p in PATTERNS}
PATTERN_IDS: tuple[str, ...] = tuple(p.id for p in PATTERNS)


# ---------------------------------------------------------------------------
# Candidate generation — pure, deterministic, no OPSIN, no I/O.

@dataclass(frozen=True)
class RepairCandidate:
    """One proposed repair. Not yet confirmed — see `repair_name`."""
    pattern_id: str
    repaired: str
    span: tuple[int, int]   # index range of the REPLACED text in `repaired`


def generate_candidates(name: str) -> list[RepairCandidate]:
    """Every repair `PATTERNS` proposes for `name`. Registry order, then
    left-to-right site order, then each site's own `fix` order — fixed and
    deterministic, never a set/dict iteration. Multiple patterns, multiple
    sites, or (for `dropped_close_bracket`) multiple fix offsets at one
    site each produce their own candidate; nothing here picks a winner —
    that is `repair_name`'s job, after the OPSIN and corroboration gates.
    """
    out: list[RepairCandidate] = []
    for pat in PATTERNS:
        for start, end in pat.site(name):
            matched = name[start:end]
            for replacement in pat.fix(matched):
                repaired = name[:start] + replacement + name[end:]
                span = (start, start + len(replacement))
                out.append(RepairCandidate(pat.id, repaired, span))
    return out


# ---------------------------------------------------------------------------
# OPSIN — the acceptance gate. Self-contained (does not import
# `iupac_names._opsin`, which lives in a module owned by another agent right
# now) but the same batching shape: one subprocess for the whole list, a
# pid-scoped temp file so two processes running concurrently cannot
# overwrite each other's input.

def _opsin_batch(names: list[str], fmt: str, patent_id: str = "") -> list[str]:
    """DELEGATES to `sources/opsin.batch` — see that module.

    This was one of THREE independent copies of the same wrapper, all ending in
    `list(out) + [""] * (len(names) - len(out))`, which returns an oversized
    list unchanged when OPSIN gives back more results than it was sent. Every
    caller pairs by position via `zip`, so a mid-list extra silently hands each
    name the next name's structure. Kept as a name because the call sites read
    better for it; it carries no logic of its own.
    """
    return _opsin_batch_shared(names, fmt, patent_id)



# ---------------------------------------------------------------------------
# Corroboration — a literal substring probe, not a fuzzy match. See the
# module docstring for why the search key (the REPAIRED text) cannot
# coincidentally hit the original corrupted occurrence.

# Chars of context probed on each side of the repaired span, shared by
# every `trusted=False` pattern's corroboration check. Measured against the
# `digit_for_paren` and `ch_as_ey` corpus cases: 36-char and 34-char
# corroborating fragments respectively (well above this window's ~24-30
# char probe length once the fix itself is included), while short enough
# that the probe almost never crosses into an unrelated neighbouring token.
_CORR_WINDOW = 12

# ALL whitespace is stripped before the corroboration check, on BOTH
# sides — never anything else about either string. MEASURED NEED, not a
# guess: `ch_as_ey`'s own confirmed case (US10376513 compound #69) sits in
# a table cell that line-wraps as "...-5-eyloro-6- fluoro-2-methoxyphenyl
# ...", ONE space injected mid-token by the wrap (between "6-" and
# "fluoro", not between two separate words); the SAME compound's correct
# spelling recurs in this patent's own headings/prose as
# "...-5-chloro-6-fluoro-2-methoxyphenyl..." with NO space there.
# Collapsing runs of whitespace to one (tried first, and insufficient —
# there is only ONE space here, so a collapse changes nothing) does not
# fix this; the space itself does not belong, so it has to be removed, not
# shrunk. Verified by re-running the exact cached XML, not assumed: a
# byte-exact probe (this module's ORIGINAL implementation) never
# corroborates the one real corpus case `ch_as_ey` was written for.
# Stripping whitespace entirely is still purely a typesetting-noise
# tolerance, not a content change — every non-whitespace character still
# has to match, in the same order, so this cannot turn an unrelated
# fragment into a false corroboration; it only stops a line-wrap from
# hiding a real one.
_WS = re.compile(r"\s+")


def _probe_window(repaired: str, span: tuple[int, int]) -> str:
    lo = max(0, span[0] - _CORR_WINDOW)
    hi = min(len(repaired), span[1] + _CORR_WINDOW)
    return repaired[lo:hi]


def _normalize_ws(s: str) -> str:
    return _WS.sub("", s)


# ---------------------------------------------------------------------------
# The full pipeline.

@dataclass(frozen=True)
class RepairResult:
    """A CONFIRMED repair — both the OPSIN gate and (for an untrusted
    pattern) the corroboration gate passed. Never returned for a candidate
    that only cleared one.
    """
    original: str
    repaired: str
    smiles: str
    inchikey: str
    pattern_id: str
    confirmation: str          # "opsin_only" or "corroborated"
    probe: str = ""            # the corroborating text actually found; "" if none


def repair_names(names: list[str], document_text: str = "",
                 patent_id: str = "") -> list[RepairResult | None]:
    """`repair_name`, over a whole list, in TWO OPSIN calls instead of 2N.

    Returns one entry per input, in order — `None` where nothing confirmed,
    exactly as `repair_name` returns `None`. Every gate is the same object
    doing the same job: `generate_candidates`, the original-parses
    short-circuit, `PATTERNS`'s own registry/site/fix ordering,
    `trusted or corroborated`, first-confirmed-wins. Nothing here decides
    anything `repair_name` did not already decide; the only difference is
    that the OPSIN calls are pooled.

    WHY THIS EXISTS. OPSIN is a java subprocess: `repair_name` pays one
    process launch per name (two, on a win). The callers wired in during
    this task hand it thousands of names per patent — every table cell
    `table_names` could not resolve, every description seed `iupac_names`
    could not resolve — and at that scale per-name launches, not the parsing,
    are the whole cost. `repair_name` is now this function at N=1, so there
    is one implementation of the gates and no second copy to drift.
    """
    results: list[RepairResult | None] = [None] * len(names)

    # ONE flat batch: for each name, the ORIGINAL (the safety net described
    # in the module docstring) followed by its candidates. `index` records
    # which name and which candidate each position belongs to; `-1` marks the
    # original. Pairing stays positional, which is exactly what
    # `sources/opsin.batch` guarantees or refuses.
    plans: list[list[RepairCandidate]] = []
    flat: list[str] = []
    index: list[tuple[int, int]] = []
    for i, name in enumerate(names):
        cands = generate_candidates(name)
        plans.append(cands)
        if not cands:
            continue
        index.append((i, -1))
        flat.append(name)
        for k, c in enumerate(cands):
            index.append((i, k))
            flat.append(c.repaired)
    if not flat:
        return results

    smiles = _opsin_batch(flat, "SMILES", patent_id)
    parsed: dict[tuple[int, int], str] = {}
    for key, smi in zip(index, smiles):
        if smi:
            parsed[key] = smi

    # Normalized ONCE for the whole list, not per candidate — see
    # `_normalize_ws`'s comment for why whitespace is stripped at all.
    doc_norm = _normalize_ws(document_text) if document_text else ""

    winners: list[tuple[int, RepairCandidate, str, bool, str]] = []
    for i, cands in enumerate(plans):
        if not cands:
            continue
        if (i, -1) in parsed:
            logger.info("name_repair: %r already parses; nothing to repair",
                        names[i])
            continue
        for k, cand in enumerate(cands):
            smi = parsed.get((i, k))
            if not smi:
                continue
            pat = _PATTERNS_BY_ID[cand.pattern_id]
            probe = _probe_window(cand.repaired, cand.span)
            corroborated = bool(doc_norm) and _normalize_ws(probe) in doc_norm
            if not (pat.trusted or corroborated):
                continue
            winners.append((i, cand, smi, corroborated, probe))
            break
    if not winners:
        return results

    iks = _opsin_batch([w[1].repaired for w in winners], "StdInChIKey", patent_id)
    for (i, cand, smi, corroborated, probe), ik in zip(winners, iks):
        logger.info(
            "name_repair: %r -> %r via %s (%s)",
            names[i], cand.repaired, cand.pattern_id,
            "corroborated" if corroborated else "opsin_only")
        results[i] = RepairResult(
            original=names[i],
            repaired=cand.repaired,
            smiles=smi,
            inchikey=ik,
            pattern_id=cand.pattern_id,
            confirmation="corroborated" if corroborated else "opsin_only",
            probe=probe if corroborated else "",
        )
    return results


def repair_name(name: str, document_text: str = "") -> RepairResult | None:
    """Repair `name` if, and only if, a candidate from `PATTERNS` can be
    CONFIRMED — see the module docstring's "THE GOVERNING CONSTRAINT".
    Returns `None` if nothing confirms, including when `name` was never
    actually broken (see "WHAT THIS DOES NOT DO").

    `repair_names([name], document_text)[0]` — the single-name form of the
    batched function above, kept because it is the readable call site and the
    one the tests and the docstring's CALL SIGNATURE section document. It
    holds no logic of its own, so the two forms cannot disagree.
    """
    return repair_names([name], document_text)[0]
