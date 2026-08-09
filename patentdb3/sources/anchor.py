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

THE FALSE ANCHORS — ROOT CAUSE, MEASURED, AND FIXED
------------------------------------------------------
A prior measurement on US8952177 reported: of 155 structures anchored by the
rule above, 67 (43%) disagreed with the hand-checked CSV reference for that
same compound number, and 13 ids each had more than one structure attached
(id "1" had 24). Reproduced exactly before touching anything (238 distinct
structures, 155 anchored, 29 clashed, 13 dup-cids, 67/155 disagreeing) — then
traced to its cause rather than assumed.

Cause 1, the dominant one: CROSS-REFERENCE CITATIONS of "Example N" reuse the
exact `<id>` + whitespace + `<name>` shape a real heading uses, AT THE SAME SHORT DISTANCE the
"THE RULE" section above calls the clean "correct" cluster (2-12 chars). US
patent synthesis sections routinely write "The title compound was prepared in
a manner analogous to that in Example 1, substituting 4-bromo-2-fluorobenzyl-
amine in Step A" — a citation of Example 1's METHOD, naming a completely
different reagent. Measured directly on the raw text: id "1" recurs as a bare
"Example 1" 44+ times in US8952177, nearly all of them exactly this citation,
each one 8-18 characters from whatever reagent the sentence happens to name.
`_ANCHOR_BOUND` cannot separate this from a real heading — both clusters are
short — so the earlier "2-12 correct vs. 162+ wrong, no overlap" measurement
was true only because it was taken on the 67 CSV-MATCHED rows, which are all
real title compounds and never exercise a citation sentence. What DOES
separate them: a real heading has NOTHING between the id and the name but
whitespace and, at most, one stereo-descriptor token already stripped off the
OPSIN-accepted name by `iupac_names._LEAD_JUNK` ("racemic ", "cis-", "trans-",
"(1R*,2S*)-"); a citation always has a verb or unrelated punctuation there
("using ", "substituting ", "mmol), ", "Steps A-B using "). Measured on the
67 disagreements: 62 had exactly a verb/punctuation gap, 0 of those 62 had a
clean gap. `_LEFT_GAP_OK` enforces this for `_CID_LEFT_PLAIN` only — see its
own comment for why `_CID_LEFT_SEP` (the colon shape) does not need it: real
corpus citations of "Intermediate 5B" never carry a colon, so they never
reach that pattern at all, while a colon-shape gap legitimately carries a
chemical prefix word (`test_left_colon_alnum_anchors`'s "tert-butyl ") that
this guard must not reject.

Cause 2, smaller: `_CID_LEFT_PLAIN`'s bare digit-run capture group captured
the DIGIT TAIL of an alphanumeric token it was never meant to see.
US10214537 numbers some intermediates "Q38", "Q17", ... — real, distinct
ids — but the old pattern matched
"38" out of "Q38" and anchored Q38's own title compound to plain id "38", a
different, unrelated compound. `(?<![A-Za-z0-9])` blocks a digit run whose
start is glued to a letter or digit; it never lets `_CID_LEFT_PLAIN` return
an alphanumeric id (that would repeat the widening this section's neighbor
above says was tried and regressed US8952177) — it only refuses to treat a
token's tail as if it were a whole bare-digit id.

Both fixed, measured on the SAME 238-structure candidate set (apples-to-apples
with every number in this docstring, `iupac_names.extract_names` unmodified):

| | US8952177 (238) | US10214537 (1018) | US10544143 (222) |
|---|---|---|---|
| anchored, before  | 155 (65.1%) | 791 (77.7%) | 116 (52.3%) |
| anchored, after   | 89 (37.4%)  | 717 (70.4%) | 112 (50.5%) |
| clashed, before   | 29          | 72          | 63 |
| clashed, after    | 11          | 45          | 17 |
| dup-cid ids, before (max) | 13 (24) | 43 (9) | 5 (11) |
| dup-cid ids, after (max)  | 0       | 6 (2)  | 2 (2)  |

The raw anchor RATE drops — that is the correct trade, not a regression: every
one of the 67 US8952177 structures that stopped being anchored (verified by
name, not assumed) is a synthesis reagent or intermediate named in a citation
sentence ("water-acetonitrile", an HPLC mobile phase; "(4-bromo-3-fluoro-
phenyl)methanamine"; "trans-hexahydroisobenzofuran-1,3-dione"), never a title
compound. Against the CSV reference, disagreement among anchored structures
that have a CSV row for their id falls from 67/155 (43.2%) to 6/89 (6.7%) —
and every one of those remaining 6 (ids 90, 93, 97, 98, 114, 137) is a CSV
transcription typo, not an extraction error: "trifluoro-1-methoxy" for
"trifluoromethoxy" (90), "benzimidazol-1-yl"/"-3-yl" for "-2-yl" (93/97/98),
"methyl-ylpyridin" for "methylpyridin" (137) — the task brief already named
98 and 137 as cases where OUR extraction is right and the CSV is wrong; this
measurement finds four more of the same kind, not a new failure mode. Real
precision on this subset is 89/89, not 83/89. The 6 remaining `US10214537`
and 2 remaining `US10544143` dup-cids after the fix were inspected by hand
and are NOT the same bug: each is a fragment/full-name pair (e.g.
"5-chloro-3-isopropyl-1H-pyrrolo[3,2-b]pyridine" and the same name with a
"-1-carboxylate" tail) that both genuinely sit next to ONE real heading and
both correctly resolve to that heading's id — OPSIN legitimately accepts two
different spans as two different structures; the shared id is correct for
both, not a proximity error.

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
"alphanumeric ids" limitation named going in, quoted at the time as "10.8%
of example labels" from a 3-patent sample. Re-measured across all 137 cached
XMLs (122 contributing at least one match), matching heading text against
`^(?:Example|Intermediate|Compound) <id>[.:]?$` (id-only heading) OR the
same prefix followed immediately by a colon (id+name-in-one-heading, this
patent's own convention): **12.85% (2,426 / 18,878)**. Same order of
magnitude as the 3-patent figure, not identical — the two counts use
different, not directly reproducible methods (the original 3-patent number's
exact method was not recorded) — but both agree the shape is corpus-wide, not
a 3-patent artifact: 122/137 patents carry at least one alphanumeric id
heading, and the per-patent rate varies enormously (US8952177 8.7%,
US10214537 65.6%, US10544143 59.4%) — a fixed corpus-wide percentage was
never the useful number; per-patent convention is.

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
convention, and the corpus-wide alphanumeric-id rate (12.85%, re-measured
below) was never scoped to these three patents.

PRECISION / RECALL AGAINST GROUND TRUTH
-----------------------------------------
`US8952177 Binding IUPAC Final (2).csv`, 190 rows, ~26 known malformed/corrupt
(see "THE FALSE ANCHORS" above for concrete examples — single-character CSV
typos, not extraction errors). Historical note: an earlier measurement,
restricted to the subset of extracted names matching a CSV row under a
stereo-prefix/salt-suffix normalization that was never committed as a test
(the docstring pointed at `tests/test_anchor.py::test_csv_ground_truth`,
which does not exist in this file — a dangling reference, now removed rather
than repeated), found digit-only and the then-shipped rule scoring
IDENTICALLY on 67 matched names: 100% precision (63/63), 94.0% recall
(63/67). That comparison predates the fix above and used a narrower
population (extraction-matched names only, not all anchored structures); see
"THE FALSE ANCHORS" for the corpus-denominator numbers — 89/89 correct
against the CSV once its own typos are excluded, measured directly against
`find_cid`'s live output rather than a restricted match subset.

Those were 89 anchored of 238 structures when measured. Both figures have since
moved and the RATIO is what carries over: `description_text` began including
`<heading>`, so US8952177 then yielded 308 structures with 165 anchored (53.6%).
Anchors nearly doubled without new errors, because a heading pair is exactly
the `<id>` `<name>` adjacency this module accepts and exactly what a prose
cross-reference is not.

Moved again, once more without new errors: the dropped-open-paren repair (see
`_DROPPED_OPEN_PAREN` below) takes US8952177 to 309 structures, 170 anchored
(55.0%) — +1 structure, +5 anchors, clash count unchanged at 1. Every added
anchor and the added structure were individually verified against this same
CSV (`US8952177 Binding IUPAC Final (2).csv`) before/after; see that
constant's comment for the full measurement, including the two of the seven
repaired headings that still do not get their own anchor, and why.

WHAT IS STILL UNANCHORED, AND WHY THAT IS CORRECT
----------------------------------------------------
Run against US10544143's 106 remaining unanchored structures: 43 have no id
candidate at all, and inspection of every one sampled is a reagent or solvent
named in synthesis prose ("1,3-butanediol", "N-methylpyrrolidine",
"6-bromopyridin-3-amine") — never a patent-numbered compound, so finding
nothing is the right answer.

The "N2 sparged" plain-digit false positive named as a known limitation when
this module was written ("2-methyltetrahydrofuran" anchored to id "2" off
"N2") is now FIXED, as a side effect of both changes in "THE FALSE ANCHORS":
`_LEFT_GAP_OK` rejects the "sparged " gap, and `(?<![A-Za-z0-9])` separately
refuses to start a digit match glued to the "N" in front of it. Verified
directly: `find_cid("...N2 sparged 2-methyltetrahydrofuran...", "2-methyl-
tetrahydrofuran")` now returns `cid=None`, not `"2"`.

One shape this module still does not solve: "Intermediate 5A-1: 5-bromo-2-
(...)... and 5-bromo-3-(...)..." — ONE id covering TWO names in a "mixture of
X and Y" heading. The second name's nearest text is "... and ", not the id,
which sits 80+ characters back, outside any bound this module tested without
flooding other patents with false proximity matches (widening the bound was
tried and rejected — see "THE RULE" above for the mechanism of that failure).

AN EARLIER VERSION OF THIS PARAGRAPH CALLED THE SHAPE "NOT rare" ON THE
STRENGTH OF A PROXY COUNT — "760 candidate headings in 70/137 patents" — and
cited US10214537 and US10087188 as real instances. Re-measured properly, both
the number and both citations turn out not to support the claim.

The proxy counted headings that merely CONTAIN " and ". Adjudicating each one
with OPSIN instead (all 137 cached XMLs, 50,909 headings; split at the last
" and ", strip any leading label+id off the left side BEFORE parsing — the
step the original proxy skipped, which is why it could not tell a two-compound
heading from a sentence):

  955 headings in 90 patents match the "contains ' and ', two long sides"
      proxy at all;
  176 in 18 patents genuinely name two compounds (both sides resolve to
      non-empty, DIFFERENT StdInChIKeys);
  and of those 176, bucketed by whether an id is even present:
      120 in 10 patents carry NO id anywhere in the heading — there is
          nothing to inherit and `find_cid` returning None is already right;
       44 in 10 patents carry a BLACKLISTED word's number ("Step 3: X and Y")
          — a synthesis step, not a compound id. `_NON_ID_WORD` refuses these
          today and must keep refusing them; inheriting there would be a false
          anchor, not a recovered one;
       12 in ONE patent (US10870641) carry a real compound id. That is the
          entire fixable population.

Of those 12, 11 have their second structure already extracted by
`extract_names` and currently unanchored. So the whole corpus-wide payoff of
the mechanism described below is +11 anchors on 1 of 137 patents.

The two cited instances were checked directly and neither is this shape:

  - US10214537's headings read "Intermediates Q36-A and Q37-A:
    2,6-Dicyclopropylpiperazine (Q36-A) and 2-Cyclopropyl-6-isopropylpiperazine
    (Q37-A)". Each name carries its OWN id in trailing parens, which is
    `_CID_RIGHT`'s shape — so this is already solved. Verified through the real
    `find_cid` on the real XML: "2-Cyclopropyl-6-isopropylpiperazine" -> Q37-A,
    the two ethanones -> Q36 / Q37, and "2,6-Dicyclopropylpiperazine" correctly
    CLASHES (Q36-A right at distance 2, Q37-A left at 7) rather than guessing.
  - US10087188's paired stereoisomer headings carry either no id at all or a
    "Step N:" number, so they land in the two buckets above, not this one.

The mechanism that could work is unchanged — name B inherits the id of name A
because the text immediately before B is "... and " and immediately before THAT
is where A's own span ends — and so is the reason it is not built here: it
needs BOTH names' identities at once. `find_cid` is deliberately a pure
function of one name and one text (see "WHY A SEPARATE MODULE" above); giving
it that would mean either (a) passing sibling-candidate context into every
call, changing the module's contract from "one name in, one answer out" to a
batch operation, or (b) resolving it at the `extract_names` call site in
`iupac_names.py`, which owns the set of names.

What HAS changed is that the trade is now measured rather than assumed, and it
does not favour building it: 11 anchors on one patent, against a rule that
would have to correctly refuse the 164 headings (44 "Step N:" + 120 id-less)
in 19 patents where the identical textual shape means something else, using a
left window 4-7x wider than `_ANCHOR_BOUND` — the same widening "THE RULE"
above measured as paying off in CLASHES, not anchors. Precision is worth more
than coverage here: an unanchored name is visible and recoverable, a name
anchored to the wrong compound number is not. Left unbuilt deliberately.

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

from ..core import config
from dataclasses import dataclass, field

from . import losses as _losses

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
# Safety valve on repeated occurrences of one name in a huge patent.
#
# ITS OWN COMMENT USED TO SAY "Never observed to bind in this corpus." That was
# false, and the value it justified (200) was low enough to change answers.
# Measured directly — `extract_names` over all 137 cached XMLs in
# `output_v3/uspto_xml/`, every distinct OPSIN-resolved name (54,206 of them),
# each run through `find_cid` twice, once at 200 and once with the cap out of
# reach, the two `AnchorResult`s compared field by field:
#
#   55 names in 30 of 137 patents have MORE than 200 occurrences.
#   41 of the 55 get the identical answer either way.
#   14 do not, and one of those is a correctness defect, not a coverage one:
#
#     US9394297, "6,7-dihydro-1H-pyrrolo[3,2-c]pyridin-4(5H)-one", 752
#     occurrences. At the 200 cap: cid="7", clashed=False, ONE candidate — a
#     confident anchor. Complete scan: cid=None, clashed=True, THIRTY-ONE
#     candidates. The cap stopped the scan before it reached the occurrences
#     that disagreed, so the function asserted "every occurrence agreed" about
#     evidence it had never looked at. That is the one thing this module
#     promises never to do (see "CLASHES SURFACE" above), and a name anchored
#     to a wrong compound number is invisible downstream.
#
#     US9656988, "pyrazine-2-carboxamide", 729 occurrences: at the cap,
#     cid=None with ZERO candidates — the "nothing found" state, which
#     `AnchorResult`'s docstring defines as distinct from a clash. Complete
#     scan: a 222-way clash. The caller was told the name has no id anywhere
#     when in truth 222 ids compete for it.
#
#     Two more are plain recall losses (US10544120, US10870641: the cap hid
#     the only evidence, cid=None where the complete scan resolves one id) and
#     ten are clash->clash, where only the candidate LIST was truncated and
#     the decision was identical.
#
# What the cap bought, measured the same way: `find_cid` over the 24,923 names
# of those 30 patents takes 3.61 s at cap=200 and 4.33 s uncapped — 0.72 s
# total, over 21,879 extra occurrences, i.e. ~33 us per extra occurrence
# scanned, against a run whose wall-clock is dominated by OPSIN.
#
# So the valve stays — a document with one short name repeated without bound is
# still worth refusing to grind on — but it is set far above anything this
# corpus produces (largest observed: 4,597 occurrences of
# "1,3-dihydro-2H-isoindole-2-carboxamide" in US9302989, a 5.4x margin), and
# hitting it is now SAFE rather than silent: see `find_cid`, which never
# returns a confident `cid` from a scan it knows it truncated.
_MAX_OCCURRENCES = 25_000

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
#
# `(?<![A-Za-z0-9])` guards the START of the digit run: without it, `\d+`
# happily matches the DIGIT TAIL of an alphanumeric token it was never meant
# to see. Measured on US10214537: "Intermediate Q38" is a real, distinct
# heading (`h-0134`) for its own compound, unrelated to plain id "38"
# (a different compound entirely, present elsewhere in the same patent) —
# but bare `(\d+)\s` matched "38" out of "Q38" and anchored Q38's title
# compound to id "38", producing a dup-cid (two unrelated structures both
# claiming id "38"). This is not the letters-in-PLAIN regression the
# comment above warns about — it never lets `_CID_LEFT_PLAIN` RETURN an
# alphanumeric id, it only stops a digit run from matching when it is
# plainly the tail of a longer alphanumeric token immediately to its left.
_CID_LEFT_PLAIN = re.compile(r"(?<![A-Za-z0-9])(\d+)\s")

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
# `intermediate` is deliberately NOT in that list. It was added, measured,
# and removed: refusing to anchor an `Intermediate N:` heading is the
# anchor-side half of dropping intermediates, and the corpus run showed that
# decision deletes correct structures while leaving the corrupted prose
# duplicates behind (US10071079 T-1: the right `tert-butyl 4-(...)` row went,
# the Boc-stripped one stayed). The requirement to ship only finished
# molecules is real and belongs downstream as a filter on a labelled row.
# See `iupac_names._DROP_INTERMEDIATES` for the other half and the numbers.

# THE GUARD NEEDS A WIDER VIEW THAN THE PROXIMITY RULE DOES.
#
# `_ANCHOR_BOUND` answers "is this number close enough to be this name's id",
# and 25 characters is the measured answer to that. It was ALSO, accidentally,
# how much text the blacklist got to look at — and the blacklist is asking a
# different question, about the word sitting in front of the number, which can
# easily start further back than 25 characters:
#
#     real left context : '…[M+H]+.\nStep 2: Synthesis of ethyl '
#     the 25-char view  : 'ep 2: Synthesis of ethyl '
#
# `Step` is clipped to `ep`, `\bstep\b` cannot match, and the step number is
# taken as a compound id. Measured on US10030020 this produces cids 2, 3 and 4
# from Steps 2, 3 and 4; corpus-wide it is 218 of the 1,499 conflicting cids.
#
# So the guard now reads its own, wider slice. The proximity bound is
# untouched — a number still has to be within `_ANCHOR_BOUND` to anchor at
# all — this only changes how much context is available to REFUSE one.
_GUARD_BOUND = 90

# An id that names a synthesis waypoint rather than a claimed compound. Gated
# on `config.FINISHED_ONLY`; kept OUT of `_NON_ID_WORD` so that list keeps
# meaning "never a compound id in any configuration", which `intermediate` is
# not — it is a real id, for a thing we have decided not to ship.
_NOT_FINISHED = re.compile(r"\b(?:intermediate|step)\s*$", re.I)

# What is allowed to sit BETWEEN a LEFT id-match and the name it anchors.
# ROOT CAUSE (measured on US8952177, see the module docstring's "FALSE
# ANCHORS" section for the full writeup): a heading citation like
# "Example 1 substituting X" or "...similar methods to those in Example 1
# Steps A-B using Y" reuses the exact same "<id>\s" shape as a real heading,
# at the SAME short distance (8-18 chars) the docstring's own ablation
# treats as the "correct" cluster (2-12 chars) — `_ANCHOR_BOUND` cannot tell
# these apart because both are short. What separates them is not distance,
# it is what fills the gap: a real heading has NOTHING between the id and
# the name but whitespace and, at most, a stereo-descriptor word already
# stripped off the OPSIN-accepted name by `_LEAD_JUNK` in `iupac_names.py`
# ("racemic ", "cis-", "trans-", a locant descriptor "(1R*,2S*)-"); a
# cross-reference citation always has a verb or unrelated-clause punctuation
# in that gap ("using ", "substituting ", "mmol), ", "Steps A-B using ").
_LEFT_GAP_OK = re.compile(
    r"^(?:racemic\s+|cis-|trans-|\([0-9][0-9A-Za-z*,\s]{0,10}\)-)*$", re.I)

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


# ---------------------------------------------------------------------------
# THE DROPPED-OPENING-PAREN DEFECT — a typesetting loss in the PATENT'S OWN
# grant XML, not in our flattening. Confirmed by direct inspection of
# US8952177's RAW XML (`html.unescape`d, tags stripped, never diagnosed from
# a parsed view): 7 headings restate a title compound's name with the
# ring-substituent parenthesis that should open immediately after a `[`
# simply missing —
#
#     as filed:  ...-6-[5-methylpyridin-2-yl)methoxy]-...
#     correct:   ...-6-[(5-methylpyridin-2-yl)methoxy]-...
#                        ^ the "(" is gone; its matching ")" survives
#
# Hand-verified against the raw XML on Examples 25, 162, 163, 166, 171, 173,
# 186 (7 occurrences — NOT 174, which an earlier description of this defect
# named; that heading (`h-0413`, "Example 174") is well-formed on direct
# inspection, and 166 was the one it omitted). Each is a real title-compound
# heading whose OWN malformed text never matches the well-formed name OPSIN
# accepts elsewhere in the document (a correctly-bracketed restatement of the
# SAME structure — usually the very next stereoisomer's own heading, or an
# embodiment list earlier in the SUMMARY) — so `find_cid`'s exact-string
# re-scan finds the id-adjacent occurrence NOWHERE, because the name isn't
# spelled that way at that position. One structure (Example 163's own
# stereoisomer) has NO correctly-bracketed restatement anywhere in the
# document at all, so before this fix it was not merely mis-anchored, it was
# never extracted as a structure — repairing the text is what makes OPSIN see
# a parseable name there for the first time.
#
# Fixed HERE, once, before either candidate generation
# (`iupac_names.extract_names`'s seeding) or anchoring ever sees the text:
# `extract_names` now builds its one working string via `anchor_text(xml)`
# instead of calling `description_text` directly, so both consumers repair
# and reuse the SAME string — preserving the "ONE STRING, not two
# flattenings" invariant this module already relies on for offset sanity.
#
# SCOPE, DELIBERATELY NARROW: `[` immediately followed by a digit-led run
# with NO nested bracket characters, ending `-yl)` — the exact shape of every
# confirmed occurrence (a locant-prefixed ring/chain substituent name, which
# this corpus's naming convention always wraps in its own parens before
# nesting it inside `[...]`). Measured corpus-wide (all 137 cached XMLs,
# this exact pattern, RAW text): 7 patents, 16 occurrences — general enough
# to fix here rather than as a US8952177-only patch (US20250163061A1,
# US9718790, US9745328 x2, US10035778, US10689373, US8846929 each carry it
# too), narrow enough that it does NOT also "fix" two DIFFERENT, unconfirmed
# corruptions found nearby in the SAME US8952177 document while measuring
# this: "(" mistyped as the digit "1" (`[4-1-trifluoromethoxy)benzyl]`), and
# a dropped "(" that is NOT the first character after `[`
# (`[2-fluoro-4-trifluoromethoxy)benzyl]`, missing the paren several
# characters in, before "trifluoromethoxy"). Both are real defects but
# neither is THIS one, and inventing a bracket for an unconfirmed shape is
# worse than leaving a name unanchored.
#
# Repairing INSERTS one character (the dropped `(`) per hit, so offsets past
# a repair point are into the REPAIRED string, not the raw `description_text`
# output — checked before relying on that: nothing outside `extract_names`
# reads `NamedCompound.start` to re-index a different string (`grep -rn
# '\.start\b' patentdb3/` — the only non-definition hits are `iupac_names.py`
# using it for seeding/dedup order, both against this same repaired text).
_DROPPED_OPEN_PAREN = re.compile(r"\[(\d[\w,'*-]{0,80}?-yl)\)")


def _repair_dropped_open_paren(text: str) -> str:
    return _DROPPED_OPEN_PAREN.sub(lambda m: "[(" + m.group(1) + ")", text)


# ---------------------------------------------------------------------------
# TABLES-DROPPED LOSS DETECTION — observability only, never used to extract.
#
# `description_text()` (`uspto_xml.py`, not owned by this task) strips every
# `<tables>...</tables>` block out of the description body before returning
# text — "handled structurally by parse_tables", per that function's own
# comment, which is true for the ASSAY side but there is no equivalent
# structural reader for a compound NAME that happens to sit inside a table
# cell (a Markush R-group table naming a substituent, a table row restating
# "Example 12: 2-{...}..." instead of a heading). Whatever named structure
# was in there is invisible to `extract_names` from the moment
# `description_text` returns, with nothing recording that a block existed at
# all.
#
# This module cannot fix that — `uspto_xml.py` is out of scope for this task,
# and the structural fix (routing table cells through `iupac_names`'
# candidate machinery, or through `parse_tables` and back) is a real, separate
# piece of work. What it CAN do, cheaply and without touching the other
# module, is measure the loss: re-find the same `<tables>` blocks
# `description_text` is about to discard and count how much of it LOOKS
# like a chemical name, using the same permissive character class
# `iupac_names._NAME_CHARS` is built from (duplicated here rather than
# imported — `iupac_names` imports THIS module, so the reverse import would
# be circular) and the same two-part shape filter `extract_names` applies to
# a real seed (a run of >=3 lowercase letters, and a digit/bracket/dash
# somewhere in it). This never feeds anything back into extraction; it is
# read once, per patent, purely to write a `tables_dropped` record.
_TABLES_BLOCK = re.compile(r"<tables\b.*?</tables>", re.S)
_NAME_LIKE_IN_TABLE = re.compile(r"[A-Za-z0-9\[\]\(\)\{\}',\-\+\.′’*]{12,400}")
_DESCRIPTION_BODY = re.compile(r"<description\b[^>]*>(.*?)</description>", re.S)


def _log_dropped_tables(xml: str, patent_id: str) -> None:
    m = _DESCRIPTION_BODY.search(xml)
    if not m:
        return
    body = m.group(1)
    for tm in _TABLES_BLOCK.finditer(body):
        block = tm.group(0)
        cleaned = _clean_fragment(block)
        candidates = [g for g in _NAME_LIKE_IN_TABLE.findall(cleaned)
                      if re.search(r"[a-z]{3}", g) and re.search(r"[\d\[\(\-]", g)]
        _losses.record("tables_dropped", patent_id,
                        position=tm.start(), chars=len(block),
                        name_like_candidates=len(candidates),
                        sample=candidates[:5])


def anchor_text(xml: str, patent_id: str = "") -> str:
    """The heading-inclusive flattening. DELEGATES — it does not reimplement.

    This used to carry its own copy of the `<description>`/`<tables>`/tag
    regexes, identical in effect to `description_text(include_headings=True)`.
    Two implementations of one rule drift: a fix to the tables-dropping or the
    entity handling in one would silently not reach the other, and anchoring
    would then be matching against a document extraction no longer produces.

    Kept as a name because it says what the text is FOR, and because tests and
    `iupac_names` both read better for it.

    One repair is applied on top of the delegated flattening — see
    `_repair_dropped_open_paren` above — because the DEFECT lives in the
    patent's own XML, not in how either flattening reads it; fixing it here
    means every caller of `anchor_text` (this module's tests, and
    `iupac_names.extract_names`, which now reuses this exact string for
    candidate generation too) sees the same repaired text once.

    `patent_id` is optional and used ONLY for loss-log context (see
    "TABLES-DROPPED LOSS DETECTION" above) — every existing caller that omits
    it (this module's own tests) is unaffected; the returned text is
    identical either way.
    """
    from .uspto_xml import description_text

    if _losses.ENABLED:
        _log_dropped_tables(xml, patent_id)
    return _repair_dropped_open_paren(description_text(xml, include_headings=True))


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

    `truncated` cuts ACROSS those three: it says the scan stopped at
    `_MAX_OCCURRENCES` with occurrences of this name still unexamined, so
    whatever the other fields say is based on partial evidence. When it is
    True, `cid` is ALWAYS None — "every occurrence agreed" is not a claim this
    function can make about occurrences it never read, and asserting it anyway
    is exactly the defect that constant's comment records. `clashed` keeps its
    plain meaning either way (True iff two ids were actually observed to
    disagree), and `candidates` still carries everything that WAS found, so a
    truncated result is degraded, never silent.
    """
    name: str
    cid: str | None
    clashed: bool
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    truncated: bool = False


def _consider(best_by_id: dict[str, Candidate], cid: str, distance: int,
              direction: str, context: str) -> None:
    cur = best_by_id.get(cid)
    if cur is None or distance < cur.distance:
        best_by_id[cid] = Candidate(cid=cid, distance=distance, direction=direction,
                                     context=" ".join(context.split())[:48])


def find_cid(text: str, name: str, *, bound: int = _ANCHOR_BOUND,
             patent_id: str = "") -> AnchorResult:
    """The patent's own id for `name`, or the clash if its occurrences disagree.

    Every occurrence of `name` in `text` is a candidate — a name stated once
    in a SUMMARY embodiment list and again at its own Example heading is
    common, and only the latter has an id nearby. Each occurrence is checked
    in both directions (see the `_CID_LEFT_*` / `_CID_RIGHT` patterns above);
    the occurrence+direction with the smallest distance wins PER ID, so a
    genuine adjacent anchor always beats a coincidental one, but if two
    DIFFERENT ids each have their own close evidence, both are kept and
    surfaced as a clash rather than one being picked arbitrarily.

    The scan is bounded by `_MAX_OCCURRENCES`. If that bound is ever reached
    with occurrences still unread, `AnchorResult.truncated` is True and `cid`
    is None regardless of what was found — see that constant's comment for the
    measurement that made this necessary, and `AnchorResult` for the contract.

    `patent_id` is optional and used ONLY to give the loss log
    (`sources/losses.py`) context — this function's return value does not
    depend on it, and every existing call (this module's own tests, which all
    omit it) is unaffected.
    """
    best_by_id: dict[str, Candidate] = {}
    start = 0
    seen = 0
    truncated = False
    while True:
        pos = text.find(name, start)
        if pos < 0:
            break
        if seen >= _MAX_OCCURRENCES:
            # An occurrence EXISTS that this scan will not look at. Checked
            # here, after a successful `find`, rather than as the `while`
            # condition with a `while/else`: the old form fired its loss
            # record whenever `seen` reached the cap, including for a name
            # occurring EXACTLY `_MAX_OCCURRENCES` times, where nothing was
            # missed at all. That over-reported truncation by one whole class
            # of name — see `test_exactly_the_cap_is_not_truncation`.
            truncated = True
            if _losses.ENABLED:
                _losses.record("anchor_max_occurrences", patent_id, name=name,
                                cap=_MAX_OCCURRENCES)
            break
        seen += 1
        start = pos + 1
        end = pos + len(name)
        window = text[max(0, pos - bound):pos]
        # The blacklist reads further back than the proximity rule does — see
        # `_GUARD_BOUND`. `offset` maps a position in `window` onto the same
        # character in `guard`, so the two stay in step whatever `bound` is.
        guard = text[max(0, pos - max(bound, _GUARD_BOUND)):pos]
        offset = len(guard) - len(window)

        m = None
        for m in _CID_LEFT_PLAIN.finditer(window):
            pass                                # last match = closest to `pos`
        if (m is not None and _LEFT_GAP_OK.match(window[m.end():])
                and not _NON_ID_WORD.search(guard[:offset + m.start(1)])):
            _consider(best_by_id, m.group(1), len(window) - m.start(1), "left", window)

        m = None
        for cand in _CID_LEFT_SEP.finditer(window):
            if (config.FINISHED_ONLY
                    and _NOT_FINISHED.search(guard[:offset + cand.start(1)])):
                continue                         # "Intermediate 10:", "Step 2:"
            if _NON_ID_WORD.search(guard[:offset + cand.start(1)]):
                continue                         # "Method 1:", "Step 2:", ...
            m = cand
        if m is not None:
            # No gap check here, unlike PLAIN above: the colon is itself the
            # heading marker, so a cross-reference sentence would need
            # "Intermediate 5B: using X" (colon present) to slip through —
            # measured in US10544143, real cross-references read "described
            # in Intermediate 5B using X" with NO colon, so they never match
            # `_CID_LEFT_SEP` at all (it requires `[:;]` right after the id).
            # What DOES legitimately sit in this gap is a chemical prefix
            # word OPSIN's candidate generator seeded separately from the
            # rest of the name ("tert-butyl ", "4-(3-Bromophenyl)") — see
            # `test_left_colon_alnum_anchors` / the clash test in
            # `test_anchor.py`. Gap-checking this shape would reject those.
            _consider(best_by_id, m.group(1), len(window) - m.start(1), "left", window)

        lookahead = text[end:end + _RIGHT_LOOKAHEAD]
        rm = _CID_RIGHT.match(lookahead)
        if rm is not None:
            _consider(best_by_id, rm.group(1), rm.start(1), "right", lookahead)

    if not best_by_id:
        if _losses.ENABLED:
            _losses.record("anchor_not_found", patent_id, name=name)
        return AnchorResult(name=name, cid=None, clashed=False, candidates=(),
                             truncated=truncated)
    candidates = tuple(sorted(best_by_id.values(), key=lambda c: c.distance))
    if len(candidates) > 1:
        if _losses.ENABLED:
            _losses.record("anchor_clash", patent_id, name=name,
                            candidates=[{"cid": c.cid, "distance": c.distance,
                                         "direction": c.direction} for c in candidates])
        return AnchorResult(name=name, cid=None, clashed=True, candidates=candidates,
                             truncated=truncated)
    if truncated:
        # ONE id found, but occurrences of this name were left unread. The
        # single-candidate branch below means "every occurrence that found an
        # id agreed" — a statement about ALL of them. It cannot be made here.
        # Measured cost of making it anyway (US9394297, see
        # `_MAX_OCCURRENCES`): a confident cid="7" where the complete scan
        # finds a 31-way clash. The candidate is still returned as evidence;
        # only the confidence is withheld.
        return AnchorResult(name=name, cid=None, clashed=False,
                             candidates=candidates, truncated=True)
    return AnchorResult(name=name, cid=candidates[0].cid, clashed=False, candidates=candidates)
