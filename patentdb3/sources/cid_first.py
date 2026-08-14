"""Names found by starting from the compound ids that matter, not from the names.

THE INVERSION
-------------
`iupac_names.extract_names` finds every OPSIN-parseable span in a patent and
then asks `anchor.find_cid` which compound number each one belongs to. That
guess is the weak link: measured on the shipped artifacts
(`patentdb3/out/structures.tsv`, 63,727 rows, manifest `structures_rows`
agreeing), the two routes that READ an id out of the document beside the name
— the heading route and the table route — carry a cid on essentially every row
they emit, while the prose route has to infer one.

This module runs the join the other way round. `uspto_assays.extract_from_patent`
already knows exactly which compounds matter — the ones carrying an assay
value — so those ids are the starting set, and the search is outward from each
id to the text next to it.

THE CID IS USED EXACTLY AS THE ASSAY READER PRODUCED IT
-------------------------------------------------------
No `normalize_cid` on the search token. Canonicalising gains a handful of
compounds corpus-wide and blanks hundreds of others, and it mangles real ids
(`EM09912` -> `EM9912`, `CAP01551` -> `CAP1551`) into tokens that occur
nowhere in the document. A token that does not occur cannot be searched for.

Two cid shapes are unsearchable and are recorded rather than silently skipped:
a cid longer than `_MAX_CID_LEN`, and a cid containing whitespace. Both come
from the same upstream defect `normalize_cid`'s own docstring describes — a
`Name` column scoring as the id column, so a compound's 70-344 character IUPAC
name lands in `AssayRecord.cid`. US9018217 in the 20-patent sample is entirely
this case: 122 assay "cids", every one of them a wrapped chemical name, zero
of them a token.

WHAT AN OCCURRENCE IS — AND WHY IT IS NOT "THE TOKEN APPEARS HERE"
-------------------------------------------------------------------
The obvious occurrence index — one regex alternation over every cid, whole
tokens only, `(?<![A-Za-z0-9])` / `(?![A-Za-z0-9])` on both ends — was built
first and MEASURED, and it does not work. On US10155002 it finds **32,169
occurrences of 187 cids in 769,303 characters**, a mean of one every 24
characters; 25 sampled uniformly at random contained **zero** compound-id
assertions. They are NMR shifts (`1.64-1.67 (m, 2H)` -> cid 1), reagent
quantities (`2.2 g, 4.50 mmol` -> cid 4), chromatography settings (`Col.
Temp.: 30 C.` -> cid 30) and — most often — plain locants inside other
compounds' names (`...dihydropyridin-3-yl)...` -> cid 3). Same shape on
US10544143 (19,211 occurrences, one per 22 chars) and US9718825 (5,651).

A bare digit is not an assertion, and a span bounded by neighbouring bare
digits is ~20 characters wide — narrower than any chemical name — so the span
rule applied to that index yields nothing at all.

So an occurrence here is a place where the document DECLARES the id, in one of
three shapes, all of them already characterised in `anchor.py` from the other
direction:

    label   `Example 24:` / `Compound 71` / `Cpd. No. 7`   a label word within
            `_LABEL_WINDOW` chars to the left, nothing but `no.`/`#` between
    colon   `1D: tert-butyl ...`                            line-initial id
            followed by `:`/`.`
    paren   `...triazin-4-amine (544)`                      parenthesised, the
            `_CID_RIGHT` shape

`anchor._NON_ID_WORD` is imported, not re-listed: "Method 1:", "Table 3:",
"Scheme 2:" are the same false ids from either direction, and two copies of one
closed list drift. Measured by running THIS file's `_occurrence_index` and
`_assertion` over the cached XML, token occurrences -> assertions ->
assertions surviving `config.FINISHED_ONLY`:

    US10155002    32,169 ->  579 -> 280
    US10544143    19,211 ->  100 ->  99
    US9718825      5,651 ->   70 ->  70

THE SPAN RULE
-------------
For an occurrence of a cid at `[s, e)`, the region searched for its name is
bounded by the NEIGHBOURING occurrences — any cid's, not just this one — and
capped:

    forward   text[e : min(next_occurrence_start, e + _SPAN_CAP)]
    backward  text[max(prev_occurrence_end, s - 1 - _SPAN_CAP) : s - 1], then
              cut again at the last `\n` or `;` inside it (`_BACK_BOUNDARY`)

A name separated from cid X by an assertion of cid Y belongs to Y, and the
bound says so structurally rather than by scoring proximity. It also handles
`I-20` followed by `I-20a` with no special case: the whole-token boundaries
mean `I-20` never matches inside `I-20a`, and `I-20a`'s own occurrence closes
`I-20`'s span.

`_SPAN_CAP = 500` characters. That is a measurement, not a round number: the
longest name in the 63,727-row shipped structures dump is 343 characters, p99
is 188, and nothing exceeds 400. 500 clears the longest observed name with
headroom and still refuses to run a document region with no other id in it out
to the end of the paragraph.

The forward region is read as LINES, not as a blob (`anchor_text` preserves the
element boundaries as newlines), because the three real layouts differ only in
which line the name is on:

    Example 24: Synthesis of N-((4,6-dimethyl-...   same line   US10155002
    Example 743 \n 1-(6-(2-(7,8-dimethyl-...        next line   US10544143
    Example 94 \n Sequence: ... \n [(6S,9S,...      two down    US20240010684A1

`_LINES_AHEAD = 3` covers all three. What stops the scan running on into the
synthesis paragraph — whose first sentence names the STARTING MATERIAL, a real
molecule that OPSIN parses perfectly — is `_NAME_GAP_MAX = 120`, the character
distance from the end of the id token to the start of the name text, plus the
coverage gate below.

NAMES WITHIN THE SPAN — THE HEADING ROUTE'S MACHINERY, NOT A NEW ONE
----------------------------------------------------------------------
A name text is framing-stripped with `iupac_names`' own `_FRAMING_LEAD` /
`_HEADING_ID` / `_FRAMING_TAIL` / `_FRAMING_TAIL_CODE`, then handed the same
four transformations `iupac_names.heading_resolution` uses, least invasive
first: `as_is_whole`, `dewrap_whole`, `dewrap_seeded`, `stereo_stripped`. One
`sources.opsin.batch` call per patent for SMILES over the deduplicated
candidate strings, one more for StdInChIKey over the winners.

`iupac_names._coverage` at `_COVERAGE_MIN` is the precision gate, and it is
what makes the line scan safe: a prose line ("A solution of 6-(3-isopropyl-...,
TFA (15.93 mg) in methanol was...") does contain a parseable name, but that
name covers a small fraction of the line's alphanumerics, so it is refused.
`stereo_stripped` is exempt for the same reason it is exempt in the heading
route — it deletes out of the MIDDLE, so containment cannot hold — and pays
for the exemption the same way, with `markush=True` and no InChIKey.

Seeds are pre-filtered by the coverage floor before `_variants` ever runs: a
seed whose alphanumeric length is below `_SEED_COVERAGE_FLOOR` of the name
text's cannot produce a candidate that clears `_COVERAGE_MIN`, since every
variant only shrinks the seed (the one exception, a closed-list tail word
pulled across a space, only ever lengthens it — hence the floor sits below
`_COVERAGE_MIN`, not at it). This is an exact bound, not a heuristic, and it
is what keeps the candidate count per patent in the thousands rather than the
hundreds of thousands.

THE MARKUSH INVARIANT IS THE SAME ONE, NOT A SECOND COPY OF IT
----------------------------------------------------------------
A name carrying a relative-stereo descriptor (`_RELATIVE_STEREO`), or one that
only parsed after `_STEREO_UNSUPPORTED` was deleted, denotes a SET of
stereoisomers. It gets `markush=True` and **no InChIKey**, exactly as in
`iupac_names.extract_names` and `_heading_structures`, so nothing downstream
can join on an identity this module did not establish.

WHAT THIS DOES NOT SEE, AND THE MEASURED RESULT
-------------------------------------------------
`anchor_text` delegates to `uspto_xml.description_text`, which DELETES every
`<tables>` block before returning. So this route cannot read a name published
in a table cell — that is `sources/table_names.py`'s job, and the two are
complementary rather than competing. It is also the single largest reason a cid
the current route resolves is missed here.

Measured on a fixed 20-patent sample, against the shipped artifacts
(`patentdb3/out/reader_dump.tsv` for the assay cids, `structures.tsv` for the
current route; `latest.json` `structures_rows: 63727` verified against the
file's own row count before either was trusted):

    distinct assay cids                            7,322
    resolved by the current route (all 3 routes)   3,424   46.8%
    resolved by cid_first alone                    3,668   50.1%
    resolved by either                             4,409   60.2%
    cid_first only                                   985
    current only                                     741   (714 table-sourced)

Where the two disagree about the structure for the SAME cid: 98 of the 2,683
cids both resolve (3.7%). In 88 of the 98 the current route's name is a strict
substring of this route's — the prose route amputating a multi-word name at a
`iupac_names._TAIL_WORDS` boundary ("...carboxylic acid" for "...carboxylic
acid ethyl ester", US8669266 x31) or keeping only a terminal fragment
("1,3,4-oxadiazole", "2-ethylbutan-1-one"). The remaining 10 are different
molecules; 5 of those are documents contradicting THEMSELVES (US11254686
Z131/Z274/Z33, US11708332 146g/150f: the `Example ZNNN.` heading and the
`Synthesis of ...` line immediately below it name different regiochemistry),
and this route follows the heading because that is where the id is declared.

Wall clock 1.5 s per patent (max 3.2 s), $0, no network.

`config.FINISHED_ONLY`
----------------------
Applied at the CID level, not per row: an occurrence introduced by
`Intermediate` / `Step` / `Preparation` is not an eligible assertion, so a cid
whose only assertions are of that kind resolves to nothing at all. This is
deliberately the weaker of the two readings available — the stronger one
("drop the cid outright if ANY occurrence introduces it as an intermediate")
is measurable from `_resolve`'s stats (`cid_not_finished`) and is NOT applied,
because `Step 7:` recurs in every worked example in this corpus and would
delete the low-numbered Examples along with the steps.

ONE ROW PER CID, NOT ONE PER STRUCTURE
----------------------------------------
`extract_names` deduplicates on structure and keeps the longest name.  This
module deduplicates on CID: two Examples that resolve to the same InChIKey are
two deliverable compounds with two sets of assay values, and collapsing them
is the defect `NamedCompound.key`'s markush namespacing was added to stop. The
returned list therefore has at most one entry per assay cid, and never two
entries for one cid.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..core import config
from . import losses as _losses
from .anchor import _NON_ID_WORD
from .anchor import anchor_text as _anchor_text
from .dewrap import targeted as _dewrap_targeted
from .iupac_names import (
    _ALNUM,
    _COVERAGE_MIN,
    _FRAMING_LEAD,
    _FRAMING_TAIL,
    _FRAMING_TAIL_CODE,
    _HEADING_ID,
    _HEADING_MIN,
    _NAME_CHARS,
    _RELATIVE_STEREO,
    _SEED,
    _STEREO_UNSUPPORTED,
    NamedCompound,
    _coverage,
    _unescape,
    _variants,
)
from .gp_images import image_urls as _gp_image_urls
from .opsin import batch as _opsin_batch
from .reagents import ReagentVerdict as _ReagentVerdict
from .reagents import classify as _classify_reagent
from .uspto_assays import CID, SUBSTITUENT, build_columns, normalize_cid
from .uspto_assays import extract_from_patent as _extract_assays
from .uspto_xml import assemble_blocks, parse_tables

logger = logging.getLogger(__name__)

SOURCE = "cid_first"

# ---------------------------------------------------------------------------
# Tunables. Each carries the measurement that set it — see the module
# docstring for the run each number came from.

# Longest cid worth searching for as a token. `uspto_assays._CID_MAX_LEN` is
# 40 for the same reason and this deliberately matches it: past that the
# string is a compound NAME that reached the cid field through the
# `build_columns` defect `normalize_cid`'s docstring records, not an id.
_MAX_CID_LEN = 40

# Chars to the left of a cid token in which a label word may sit. "Compound
# No. 12" is the longest real shape in this corpus (13 chars of framing).
_LABEL_WINDOW = 24

# Region bound, both directions. Longest name in the 63,727-row shipped
# structures dump is 343 chars; p99 is 188; none exceed 400.
_SPAN_CAP = 500

# Lines after the id in which the name may sit — see the three layouts in the
# module docstring.
_LINES_AHEAD = 3

# Chars from the end of the id token to the start of the name text. This is
# what stops the line scan reaching the synthesis paragraph, whose first
# sentence names the starting material.
_NAME_GAP_MAX = 120

# A seed below this fraction of the name text's alphanumeric length cannot
# produce a candidate that clears `_COVERAGE_MIN`. Set below `_COVERAGE_MIN`
# rather than at it because `_variants` may pull a closed-list tail word in
# across a space, lengthening the candidate past its seed.
_SEED_COVERAGE_FLOOR = 0.75

# Hard ceiling on candidates generated for one name text, so a bracket-heavy
# line cannot dominate the batch.
_MAX_CANDIDATES_PER_TEXT = 24

# Label words that introduce a compound id. `steps?` / `intermediates?` /
# `preparations?` are matched SO THAT THEY CAN BE REFUSED under
# `config.FINISHED_ONLY` — the same reason `iupac_names._HEADING_ID` matches
# them. Not matching them would mean an unlabelled fall-through, which is
# indistinguishable from "we did not recognise this".
_LABEL_WORD = (r"examples?|compounds?|cpds?|entry|entries|ex|intermediates?|"
               r"preparations?|steps?")
_LABEL_BEFORE = re.compile(
    rf"(?:^|[^A-Za-z])({_LABEL_WORD})\s*(?:nos?\.?|#)?\s*$", re.I)

# The subset of the above that does not name a finished molecule. Same set as
# `iupac_names._NOT_A_FINISHED_COMPOUND`, singular and plural.
_NOT_FINISHED_LABEL = {"intermediate", "intermediates", "step", "steps",
                       "preparation", "preparations"}

# THE WORD NEXT TO THE ID CAN SAY THE OPPOSITE OF WHAT THE PHRASE SAYS.
#
#     Preparation of Intermediate for Example 534. Ethyl 3-hydroxy-1,4-...
#                                   ^^^^^^^ _LABEL_BEFORE reads this
#
# `_LABEL_BEFORE` anchors at `$`, so it captures the ONE word touching the id.
# Here that word is `Example`, which is not in `_NOT_FINISHED_LABEL`, so
# `FINISHED_ONLY` never refuses it — and the route then anchors the
# INTERMEDIATE's name to compound 534. Measured against the patent's own
# `MS (ESI) 561 (M+H)`: the anchored structure weighs 185 Da. Ten compounds on
# US10730863 shipped a reagent this way, every one labelled `compound`.
#
# WHY THIS IS SPELLED SO NARROWLY. Four phrase shapes in this corpus put a
# not-finished word before an id, and only ONE of them misdirects:
#
#   Preparation of Intermediate for Example 524.  -> a NAME follows; the name
#   Intermediate for the Synthesis of Example 366:   belongs to the intermediate
#   Step G for the preparation of Example 159:    -> yield data follows; the id
#   Preparation as described for example 30 using    IS the compound described
#
# The head noun before `for` is what separates them: an INTERMEDIATE *for* an
# example is a different compound, while a STEP *for the preparation of* an
# example is that example's own synthesis. So only `intermediate` heads this
# pattern, and the gap to `for` admits lowercase words only — that is what
# keeps `Intermediate 519E according to methods described for ... Example 130`
# out, where a digit and a second id sit in between.
_REFERRED_WINDOW = 72
_INTERMEDIATE_FOR = re.compile(
    rf"\bintermediates?\b[a-z ]{{0,12}}?\bfor\b[A-Za-z ]{{0,20}}?"
    rf"(?:{_LABEL_WORD})\s*(?:nos?\.?|#)?\s*$", re.I)

# `<id>:` or `<id>.` at the start of a line, with no label word — the
# combined-heading shape a patent writes when it has already said "Example"
# in a section title.
_COLON_AFTER = re.compile(r"^\s{0,2}[:.]\s")

# Separator characters that may sit between the id and the name text on the
# SAME line. Not a general strip: a name never starts with any of these, and
# allowing more would let the scan skip over prose.
_SAME_LINE_SEP = re.compile(r"^[\s:.;,)\-–—]{0,4}")

# Where a BACKWARD name text may start: the nearest newline or semicolon
# before the `(<id>)`. NOT the trailing run of name-legal characters, which
# was tried first and is wrong — `_NAME_CHARS` has no space in it, so a
# line-wrap space inside the name truncates it. Measured on US10478424's
# SUMMARY embodiment list: `...-[1,2,4] triazolo[1,5-a]pyridine (94);` yields
# the 22-character fragment `triazolo[1,5-a]pyridine`, which OPSIN accepts as
# a real (wrong) molecule and which scores coverage 1.00 because a truncated
# name text is its own denominator. Cutting at the SEPARATOR instead gives
# `dewrap_whole` the whole name to repair and gives `_coverage` something to
# measure against.
#
# `;` and `\n` only. A comma in an IUPAC name is a locant separator and a
# period ends `[1,2,4]`-style bracket groups often enough that neither can be
# a boundary — the same rule `iupac_names._coverage`'s docstring records after
# comma-splitting recovered `nicotinamide` out of a full name.
_BACK_BOUNDARY = re.compile(r"[\n;]")

# Transformation order, least invasive first. Identical to
# `iupac_names.heading_resolution`'s, and the tie-break below is identical
# too: the first kind that produces anything wins, longest within that kind.
_KIND_ORDER = {"as_is_whole": 0, "dewrap_whole": 1, "dewrap_seeded": 2,
               "stereo_stripped": 3}

# Tie-break between two occurrences of ONE cid that are otherwise equally
# close. A label word is the patent DECLARING a compound number ("Example
# Z33."); a trailing parenthetical is the patent REFERRING to one, and a
# sub-synthesis inside a worked example can carry the parent example's number
# while naming an intermediate (US11254686 Z33 — see the report's disagreement
# list, where the document contradicts itself and this ordering follows the
# heading). Used only as a tie-break, after line and gap.
_SHAPE_RANK = {"label": 0, "colon": 1, "paren": 2}


@dataclass
class _Occurrence:
    """One place the document DECLARES a compound id."""
    cid: str
    start: int
    end: int
    shape: str                  # "label" | "colon" | "paren"
    label: str = ""             # the introducing word, for FINISHED_ONLY


@dataclass
class _NameText:
    """A span of document text that might be one cid's name."""
    cid: str
    text: str
    offset: int                 # absolute offset into the anchor text
    line: int                   # 0 = same line as the id (or the paren shape)
    gap: int                    # chars between the id token and this text
    shape: str = "label"        # the shape of the occurrence that produced it


@dataclass
class Stats:
    """What happened to every cid, so a zero is never ambiguous.

    Every count below is over DISTINCT assay cids, and they partition the
    total: `unsearchable` + `no_occurrence` + `no_name_text` + `opsin_reject`
    + `coverage_reject` + `resolved` == `assay_cids`. `cid_not_finished` cuts
    across (it is the subset whose only assertions were intermediate/step
    introduced, already counted in `no_occurrence`).

    `cid_as_name` and `drawn_marked` are SUBSETS OF `resolved`, not further
    partition members — the first is a cid that needed no search because the
    string was the name, the second a cid that carries a `drawn_ref` marker
    and asserts no structure at all.
    """
    assay_cids: int = 0
    unsearchable: int = 0
    no_occurrence: int = 0
    cid_not_finished: int = 0
    no_name_text: int = 0
    opsin_reject: int = 0
    coverage_reject: int = 0
    resolved: int = 0
    occurrences_raw: int = 0
    occurrences_asserted: int = 0
    candidates: int = 0
    # Both were previously set with `getattr(st, ..., 0) + 1` on an undeclared
    # attribute, which type-checks as an error and, worse, reads as 0 for any
    # caller that constructs a `Stats` and never runs the branch. Declared so
    # the field exists whether or not the branch fires.
    cid_as_name: int = 0
    drawn_marked: int = 0
    # Compounds defined by a scaffold + substituents, whose row picture is
    # a SUBSTITUENT rather than the molecule. Emitted with `markush=True`
    # and an EMPTY `drawn_ref` — see `_markush_cids` for why claiming they
    # are drawn is a false statement rather than a coarse one.
    markush_marked: int = 0
    # Subset of `drawn_marked` that also carries an openable image URL. Zero
    # whenever `config.GP_ENABLED` is off, which is the default — so a zero
    # here means "GP was not consulted", not "GP had nothing".
    drawn_with_url: int = 0
    by_shape: dict = field(default_factory=dict)
    by_kind: dict = field(default_factory=dict)
    by_line: dict = field(default_factory=dict)


def _alnum_len(s: str) -> int:
    return len(_ALNUM.sub("", s))


def _occurrence_index(text: str, cids: list[str]) -> list[re.Match]:
    """Every whole-token occurrence of any cid, ONE regex, ONE pass.

    Longest alternative first so `I-200` is tried before `I-20`; the trailing
    boundary makes the order immaterial for correctness, but it keeps the
    match text equal to the cid it matched. Never `search()` in a loop: a
    patent in this corpus carries over 2,000 cids and the per-cid loop is the
    difference between one pass and two thousand.
    """
    if not cids:
        return []
    alt = "|".join(re.escape(c) for c in sorted(cids, key=len, reverse=True))
    rx = re.compile(rf"(?<![A-Za-z0-9])(?:{alt})(?![A-Za-z0-9])")
    return list(rx.finditer(text))


def _assertion(text: str, m: re.Match) -> _Occurrence | None:
    """Is this occurrence the document DECLARING the id, or just the token?

    See "WHAT AN OCCURRENCE IS" in the module docstring for the measurement
    that makes this filter necessary rather than optional.
    """
    s, e = m.start(), m.end()
    cid = m.group(0)
    left = text[max(0, s - _LABEL_WINDOW):s]
    lm = _LABEL_BEFORE.search(left)
    if lm is not None:
        label = lm.group(1).lower()
        # READ THE PHRASE, NOT THE ADJACENT WORD. See `_INTERMEDIATE_FOR`:
        # `Intermediate for Example 534` declares an intermediate FOR 534, and
        # the label touching the id says the opposite. Correcting the LABEL
        # rather than returning None keeps one refusal path — `FINISHED_ONLY`
        # already refuses this value and already records why.
        if _INTERMEDIATE_FOR.search(text[max(0, s - _REFERRED_WINDOW):s]):
            label = "intermediate"
        return _Occurrence(cid=cid, start=s, end=e, shape="label", label=label)
    # `(<id>)`, with the opening paren not glued to a word — "Acquity(4)" is a
    # column model number, not a compound id.
    if (s >= 1 and text[s - 1] == "(" and e < len(text) and text[e] == ")"
            and (s < 2 or not text[s - 2].isalnum())):
        return _Occurrence(cid=cid, start=s, end=e, shape="paren")
    # Line-initial `<id>:`. Guarded by `_NON_ID_WORD` even though nothing
    # precedes it on the line, because the previous line's tail can carry the
    # disqualifying word ("... according to\nMethod 3: ...").
    line_start = text.rfind("\n", 0, s) + 1
    if (not text[line_start:s].strip() and _COLON_AFTER.match(text[e:e + 3])
            and not _NON_ID_WORD.search(text[max(0, s - 90):s])):
        return _Occurrence(cid=cid, start=s, end=e, shape="colon")
    return None


def _strip_framing(s: str) -> str:
    """`iupac_names._heading_texts`'s framing peel, applied to a line.

    Same regexes, same loop, same order — imported rather than restated. A
    second id token is framing too (`Example 1: Synthesis of Compound 1: 5-...`
    leaves `Compound 1: 5-...` once the lead clause goes), which is why
    `_HEADING_ID` is inside the loop.
    """
    while True:
        stripped = _HEADING_ID.sub("", _FRAMING_LEAD.sub("", s, count=1), count=1)
        if stripped == s:
            break
        s = stripped
    s = _FRAMING_TAIL.sub("", s)
    return _FRAMING_TAIL_CODE.sub("", s).strip().strip(":;,. ")


def _name_texts(text: str, occ: _Occurrence, prev_end: int,
                next_start: int) -> list[_NameText]:
    """The spans in which `occ`'s name may sit — span rule applied here.

    Forward: up to `_LINES_AHEAD` lines, region truncated at the next
    occurrence of ANY cid and at `_SPAN_CAP` chars. Backward: only for the
    `(<id>)` shape, and only the trailing run of name characters.
    """
    out: list[_NameText] = []

    stop = min(next_start, occ.end + _SPAN_CAP, len(text))
    region = text[occ.end:stop]
    pos = 0
    for line_idx, raw in enumerate(region.split("\n")):
        if line_idx > _LINES_AHEAD:
            break
        lead = (_SAME_LINE_SEP.match(raw).end() if line_idx == 0
                else len(raw) - len(raw.lstrip()))
        body = raw[lead:]
        gap = pos + lead
        if gap > _NAME_GAP_MAX:
            break
        framed = _strip_framing(body)
        if len(framed) >= _HEADING_MIN:
            # Where the framed text actually begins, so `.start` points at the
            # name rather than at the framing that was peeled off it.
            at = body.find(framed)
            out.append(_NameText(cid=occ.cid, text=framed,
                                 offset=occ.end + gap + (at if at >= 0 else 0),
                                 line=line_idx, gap=gap, shape=occ.shape))
        pos += len(raw) + 1

    if occ.shape == "paren":
        back_start = max(prev_end, occ.start - 1 - _SPAN_CAP, 0)
        pre = text[back_start:max(back_start, occ.start - 1)]
        cut = 0
        for bm in _BACK_BOUNDARY.finditer(pre):
            cut = bm.end()
        body = pre[cut:]
        lead = len(body) - len(body.lstrip())
        framed = _strip_framing(body[lead:])
        if len(framed) >= _HEADING_MIN:
            at = body.find(framed)
            out.append(_NameText(
                cid=occ.cid, text=framed,
                offset=back_start + cut + (at if at >= 0 else lead),
                line=0, gap=len(pre) - len(pre.rstrip()) + 1,
                shape=occ.shape))
    return out


def _candidates(nt: _NameText, patent_id: str) -> list[tuple[str, str]]:
    """`[(kind, candidate), ...]` for one name text — the heading route's four
    transformations, least invasive first.

    Seeds are pre-filtered by `_SEED_COVERAGE_FLOOR`: a seed too short to ever
    clear `_COVERAGE_MIN` cannot win, so asking OPSIN about its twelve
    variants is pure cost. Exact, not a heuristic — every variant is a
    substring of its seed except for the closed-list tail word, which only
    lengthens it.
    """
    out: list[tuple[str, str]] = [("as_is_whole", nt.text)]
    flat = _dewrap_targeted(nt.text)
    if flat != nt.text:
        out.append(("dewrap_whole", flat))
    floor = _SEED_COVERAGE_FLOOR * _alnum_len(flat)
    for m in _SEED.finditer(flat):
        g = m.group(0)
        if not (re.search(r"[a-z]{3}", g) and re.search(r"[\d\[\(\-]", g)):
            continue
        if _alnum_len(g) < floor:
            continue
        for v in _variants(flat, m.start(), m.end(), patent_id):
            out.append(("dewrap_seeded", v))
    bare = _STEREO_UNSUPPORTED.sub("", flat)
    if bare != flat:
        out.append(("stereo_stripped", bare))
    return out[:_MAX_CANDIDATES_PER_TEXT]



# ── drawn-not-named ──────────────────────────────────────────────────────

# A compound the patent DREW instead of naming. Confirmed structurally: the
# cid's OWN table `<row>` contains a `<chemistry>`/`<img>`. Not proximity —
# containment — because a character-distance threshold cannot tell a picture
# belonging to this row from the previous row's.
#
# MEASURED, and the measurement is why this is a separate marker rather than
# an assumption about everything we cannot find. Over 8 patents, 2,783 assay
# compounds with no structure split:
#
#     2,088 (75%)  own row carries a drawing   -> unreachable by any text route
#       695 (25%)  no drawing anywhere         -> UNEXPLAINED, not drawn
#
# The 25% matters: US9018217 has 0.25 images per unresolved compound, so
# drawings cannot account for its 122, and US9763922 has TEN images per
# unresolved compound yet none of them sit in those cids' rows. Calling either
# "image-limited" would have turned our own defects into a ceiling.
_ROW = re.compile(r"<row>.*?</row>", re.S)
_ENTRY = re.compile(r"<entry[^>]*>(.*?)</entry>", re.S)
_CHEM_ID = re.compile(r'<chemistry[^>]*id="([^"]+)"')
_IMG_FILE = re.compile(r'<img[^>]*file="([^"]+)"')

# A column header naming the compound in full. Its PRESENCE is what separates
# a substituent table the patent already enumerated (`cid | Name | R2 | R4`,
# which `table_names` reads directly) from one it did not.
_NAME_HDR = re.compile(r"\bname\b", re.I)

def _header_scaffolds(xml: str) -> dict[str, tuple[str, str]]:
    """`chemistry id in a row -> (table id, the scaffold's chemistry id)`.

    A table whose HEADER holds a `<chemistry>` states a shared scaffold, and
    its rows then supply what varies. US10125101 TABLE-US-00012 is the shape:
    the header reads `Ex- R L ... (coupling partner is R L -Br)` and holds
    `CHEM-US-00050`; the row for compound 11 holds `CHEM-US-00051`, which is
    the coupling partner, not the molecule.

    So the row picture must NOT be offered to an image-to-structure step as
    the compound. It is still wanted — it is one half of an enumeration — so
    the caller records both halves rather than dropping the compound.
    """
    out: dict[str, tuple[str, str]] = {}
    for tb in re.finditer(r'<tables id="([^"]+)".*?</tables>', xml, re.S):
        tid, body = tb.group(1), tb.group(0)
        thead = re.search(r"<thead>.*?</thead>", body, re.S)
        if not thead:
            continue
        scaf = _CHEM_ID.findall(thead.group(0))
        if not scaf:
            continue
        for row in _ROW.findall(body):
            for rid in _CHEM_ID.findall(row):
                out.setdefault(rid, (tid, scaf[0]))
    return out


def _markush_cids(xml: str) -> dict[str, str]:
    """`cid -> table id`, for cids defined by a scaffold PLUS substituents.

    WHY THIS EXISTS: `_drawing_refs` finds a `<chemistry>` in a compound's own
    table row and concludes the compound is drawn. In a substituent table that
    conclusion is FALSE, and confirmed false by eye — US9718825's rows read
    `cid | Ar (text) | R1 (text) | <chemistry> | Synthesis | Yield | MS`, and
    the per-row image is not the molecule but the value of one substituent
    column whose own header is the graphic `-Z-R3`. Compounds 8 and 9 carry
    MD5-IDENTICAL images (a morpholine fragment on a wavy attachment bond)
    and differ only in `Ar`, which is text.

    Marking those as drawn asserts something untrue in the shipped artifact:
    it says "the structure is this picture" when the picture is one shared
    fragment. 644 compounds across 2 patents (US9718825 593, US10626094 51)
    — 2.8% of everything marked drawn corpus-wide.

    A table with a Name column is NOT included: the patent enumerated those
    itself and the name is already read.

    KNOWN OVER-REACH, MEASURED AND LEFT ALONE. `build_columns` reads a
    crystallography table — `x | y | z | U (eq)` over rows like
    `H(21A) | 11662 | 5142 | 2184 | 33` — as a compound id with two substituent
    columns, so US10376513 TABLE-US-00007 yields 28 "markush cids" that are
    atom labels. It reaches nothing: `_resolve` only ever asks about cids the
    assay reader produced, and the overlap there is ZERO
    (`markush_marked == 0` on that patent).

    A filter on "substituent values that are only numbers" was tried and
    removed. It does not hold — crystallography writes a Unicode minus
    (`\u22121277`) and an estimated standard deviation (`5640(60)`), so the
    numeric test needs a special case per convention, which is patching the
    patch. The defect is one level up, in `build_columns` calling a coordinate
    axis a substituent column, and that is where it should be fixed.
    """
    out: dict[str, str] = {}
    try:
        blocks = assemble_blocks(parse_tables(xml))
    except Exception as e:
        logger.warning("markush cid scan failed: %r", e)
        return out
    for t in blocks:
        try:
            cols = build_columns(t)
        except Exception:
            continue
        if not any(c.kind == SUBSTITUENT for c in cols):
            continue
        if any(_NAME_HDR.search(c.header or "") for c in cols):
            continue
        ci = next((c.index for c in cols if c.kind == CID), None)
        if ci is None:
            continue

        for row in t.body_rows:
            if ci >= len(row):
                continue
            raw = row[ci].text.strip()
            if raw:
                out.setdefault(normalize_cid(raw) or raw, t.table_id)
    return out


def _drawing_refs(xml: str) -> dict[str, str]:
    """`cid -> a reference to the drawing in that cid's own table row`.

    The reference is the `<chemistry>` id, falling back to the image filename,
    so a consumer can find the picture in the source without re-parsing.

    THE KEY MUST BE UNESCAPED, and it was not. A patent that indents its
    compound numbers writes the cell as `&#x2002;12` (EN SPACE), and this
    function used that raw text as the dict key while every cid it is looked
    up with has already been through `normalize_cid` and reads `12`. The two
    can never meet, so the drawing marker was silently not emitted and the
    compound fell through to "no structure, no marker" — indistinguishable in
    the artifact from a compound we genuinely could not place.

    Measured on the corpus when this was found: US10730877 had 87 of its 1,065
    drawing rows keyed this way and contributed exactly 87 unresolved
    compounds, 84 of which (97%) are these. US10870641 had 68. It is the
    single largest identified cause inside the unexplained bucket, and it was
    never a gap in the source — only in this dictionary.
    """
    out: dict[str, str] = {}
    for row in _ROW.finditer(xml):
        body = row.group(0)
        if "<img" not in body and "<chemistry" not in body:
            continue
        cells = _ENTRY.findall(body)
        if not cells:
            continue
        first = re.sub(r"\s+", " ",
                       _unescape(re.sub(r"<[^>]+>", "", cells[0]))).strip()
        if not first or len(first) > _MAX_CID_LEN:
            continue
        m = _CHEM_ID.search(body) or _IMG_FILE.search(body)
        if m and first not in out:
            out[first] = m.group(1)
    return out


# A cid long enough, and punctuated enough, to be a chemical name rather than
# an identifier. Same shape test `table_names._looks_chem_shaped` applies to a
# table cell, and deliberately the same numbers — a string that would not be
# believed as a name in a table cell is not believed as one here either.
_NAME_SHAPED_MIN = 20
_NAME_SHAPED_PUNCT = 3


def _cids_that_are_names(cids: list[str],
                         patent_id: str = "") -> list[NamedCompound]:
    """Assay cids that ARE compound names, read by OPSIN. See `_resolve`.

    OPSIN is the only acceptance gate, exactly as it is on every other route:
    the shape test below decides what is worth ASKING about, never what is
    accepted. A cid that is really an identifier (`I-133`, `7`) fails the
    shape test; one that is prose fails OPSIN.
    """
    cands = [c for c in cids
             if len(c) >= _NAME_SHAPED_MIN
             and (c.count("-") + c.count("[") + c.count("(")) >= _NAME_SHAPED_PUNCT]
    if not cands:
        return []

    # As-is first, then the targeted dewrap — the assay reader's own row-join
    # leaves the source's wrap space behind, and `_dewrap_targeted` is what
    # removes a space whose neighbour is punctuation.
    plan: list[tuple[str, str, str]] = []          # (cid, kind, text)
    for c in cands:
        plan.append((c, "cid_as_name", c))
        d = _dewrap_targeted(c)
        if d != c:
            plan.append((c, "cid_as_name_dewrap", d))

    uniq = sorted({t for _, _, t in plan})
    smiles = dict(zip(uniq, _opsin_batch(uniq, "SMILES", patent_id)))

    best: dict[str, tuple[str, str, str]] = {}     # cid -> (kind, name, smiles)
    for cid, kind, text in plan:
        smi = smiles.get(text, "")
        if smi and cid not in best:
            best[cid] = (kind, text, smi)
    if not best:
        return []

    order = sorted(best)
    keys = _opsin_batch([best[c][1] for c in order], "StdInChIKey", patent_id)

    out: list[NamedCompound] = []
    for cid, ik in zip(order, keys):
        kind, name, smi = best[cid]
        stereo = _RELATIVE_STEREO.findall(name)
        if stereo:                                  # the same invariant
            ik = ""
        try:
            verdict = _classify_reagent(name, smi)
        except Exception as e:
            if _losses.ENABLED:
                _losses.record("reagent_classify_exception", patent_id,
                               name=name, smiles=smi, error=repr(e))
            verdict = _ReagentVerdict(label="compound", reason="")
        out.append(NamedCompound(
            patent_id=patent_id, name=name, smiles=smi, inchikey=ik,
            start=-1, source=SOURCE, cid=cid,
            label=verdict.label, reason=verdict.reason,
            markush=bool(stereo),
            markush_kind="relative_stereo" if stereo else "",
            markush_reason=("relative_stereo:" + ",".join(stereo)) if stereo else "",
            heading_transform=kind))
    return out


def _resolve(xml: str, patent_id: str = "") -> tuple[list[NamedCompound], Stats]:
    """`extract_by_cid` plus the accounting. See `Stats` for the partition.

    Split out so a measurement harness never has to re-derive from the
    returned rows what the function already knew — the shape of mistake
    CLAUDE.md's "Backtrace every result" section exists to catch.
    """
    st = Stats()

    records = _extract_assays(xml)
    all_cids = sorted({r.cid for r in records if r.cid})
    st.assay_cids = len(all_cids)

    searchable = [c for c in all_cids
                  if len(c) <= _MAX_CID_LEN and not re.search(r"\s", c)]
    st.unsearchable = len(all_cids) - len(searchable)
    unsearchable = [c for c in all_cids if c not in set(searchable)]

    # THE UNSEARCHABLE CID IS SOMETIMES THE ANSWER, NOT THE PROBLEM.
    # `_MAX_CID_LEN`'s own comment already says what these strings are: a
    # compound NAME that reached the cid field through the `build_columns`
    # defect. That diagnosis was right and the response was incomplete —
    # the name was recorded as a loss and thrown away, while the patent had
    # in fact stated the identity of the compound plainly, in the same cell
    # as its assay value. US9018217 is the whole shape: no example numbers
    # anywhere, 122 assay records keyed by name, and 0 structures from this
    # route before this branch existed.
    #
    # Nothing is searched here and no proximity is guessed — the string is
    # handed to OPSIN exactly as the assay reader stored it. It is worth
    # noting that `extract_from_patent` has ALREADY rejoined this text across
    # `<row>` boundaries (`...imidazol-2- ylthio)methyl...` is rows 8 and 9 of
    # TABLE-US-00001 joined), so the same wrap space the table route dewraps
    # is present here and `_dewrap_targeted` removes it.
    named_cids = _cids_that_are_names(unsearchable, patent_id)
    st.cid_as_name = len(named_cids)

    if st.unsearchable and _losses.ENABLED:
        resolved_as_name = {nc.cid for nc in named_cids}
        for c in unsearchable:
            if c in resolved_as_name:
                continue
            _losses.record("cid_first_unsearchable_cid", patent_id,
                           cid=c[:120], length=len(c),
                           reason=("too_long" if len(c) > _MAX_CID_LEN
                                   else "contains_whitespace"))
    if not searchable:
        st.resolved = len(named_cids)
        return named_cids, st

    text = _anchor_text(xml, patent_id)
    if not text:
        if _losses.ENABLED:
            _losses.record("cid_first_no_text", patent_id,
                           reason="anchor_text returned empty")
        st.no_occurrence = len(searchable)
        return [], st

    matches = _occurrence_index(text, searchable)
    st.occurrences_raw = len(matches)

    occs: list[_Occurrence] = []
    not_finished: set[str] = set()
    for m in matches:
        o = _assertion(text, m)
        if o is None:
            continue
        if config.FINISHED_ONLY and o.label in _NOT_FINISHED_LABEL:
            not_finished.add(o.cid)
            if _losses.ENABLED:
                _losses.record("cid_first_not_finished", patent_id, cid=o.cid,
                               label=o.label,
                               context=" ".join(
                                   text[max(0, o.start - 30):o.end + 30].split()))
            continue
        occs.append(o)
        st.by_shape[o.shape] = st.by_shape.get(o.shape, 0) + 1
    st.occurrences_asserted = len(occs)

    # THE SPAN RULE'S NEIGHBOURS ARE EVERY ASSERTION, not this cid's own —
    # a name separated from cid X by an assertion of cid Y belongs to Y.
    texts: list[_NameText] = []
    for i, o in enumerate(occs):
        prev_end = occs[i - 1].end if i else 0
        next_start = occs[i + 1].start if i + 1 < len(occs) else len(text)
        texts.extend(_name_texts(text, o, prev_end, next_start))

    with_occ = {o.cid for o in occs}
    with_text = {t.cid for t in texts}
    st.no_occurrence = len(searchable) - len(with_occ)
    st.cid_not_finished = len(not_finished - with_occ)
    st.no_name_text = len(with_occ - with_text)

    # ONE OPSIN BATCH FOR THE PATENT. Candidates are deduplicated by string
    # first — the same name text is generated for every occurrence of the same
    # cid, and OPSIN is a java subprocess whose cost is per candidate.
    plan: list[tuple[int, str, str]] = []      # (name-text index, kind, candidate)
    for i, nt in enumerate(texts):
        for kind, cand in _candidates(nt, patent_id):
            plan.append((i, kind, cand))
    st.candidates = len(plan)
    if not plan:
        return [], st

    uniq = sorted({c for _, _, c in plan})
    smiles = dict(zip(uniq, _opsin_batch(uniq, "SMILES", patent_id)))

    # Per name text: first kind that parses AND covers wins, longest within
    # that kind — `heading_resolution`'s own tie-break, unchanged.
    best_text: dict[int, tuple[str, str, str]] = {}     # i -> (kind, name, smiles)
    parsed_any: set[str] = set()
    covered_any: set[str] = set()
    for i, kind, cand in plan:
        smi = smiles.get(cand, "")
        if not smi:
            continue
        parsed_any.add(texts[i].cid)
        if kind != "stereo_stripped" and _coverage(cand, texts[i].text) < _COVERAGE_MIN:
            if _losses.ENABLED:
                _losses.record("cid_first_coverage_too_low", patent_id,
                               cid=texts[i].cid, name_text=texts[i].text[:200],
                               candidate=cand[:200], transformation=kind,
                               coverage=round(_coverage(cand, texts[i].text), 3),
                               floor=_COVERAGE_MIN)
            continue
        covered_any.add(texts[i].cid)
        cur = best_text.get(i)
        if cur is None or (_KIND_ORDER[kind] < _KIND_ORDER[cur[0]]
                           or (kind == cur[0] and len(cand) > len(cur[1]))):
            best_text[i] = (kind, cand, smi)

    st.opsin_reject = len(with_text - parsed_any)
    st.coverage_reject = len(parsed_any - covered_any)

    # Per cid: nearest line, then smallest gap, then longest name. A cid's
    # occurrences are competing pieces of evidence for one answer; the
    # ordering prefers the one where the document put the name closest to the
    # number it wrote.
    best_cid: dict[str, tuple[tuple, int]] = {}
    for i, (kind, cand, smi) in best_text.items():
        nt = texts[i]
        rank = (nt.line, nt.gap, _SHAPE_RANK[nt.shape], -len(cand),
                _KIND_ORDER[kind])
        cur = best_cid.get(nt.cid)
        if cur is None or rank < cur[0]:
            best_cid[nt.cid] = (rank, i)

    order = sorted(best_cid, key=lambda c: texts[best_cid[c][1]].offset)
    winners = [(c, texts[best_cid[c][1]], best_text[best_cid[c][1]]) for c in order]
    keys = _opsin_batch([w[2][1] for w in winners], "StdInChIKey", patent_id)

    out: list[NamedCompound] = []
    for (cid, nt, (kind, name, smi)), ik in zip(winners, keys):
        stereo = _RELATIVE_STEREO.findall(name)
        is_markush = bool(stereo) or kind == "stereo_stripped"
        dropped = (_STEREO_UNSUPPORTED.search(nt.text)
                   if kind == "stereo_stripped" else None)
        # THE INVARIANT, and it is the same one, not a second copy: a name
        # that denotes a SET of stereoisomers carries no single-structure
        # identifier. See `iupac_names.extract_names`.
        if is_markush:
            ik = ""
        try:
            verdict = _classify_reagent(name, smi)
        except Exception as e:                       # RDKit absent, odd SMILES
            if _losses.ENABLED:
                _losses.record("reagent_classify_exception", patent_id,
                               name=name, smiles=smi, error=repr(e))
            verdict = _ReagentVerdict(label="compound", reason="")
        out.append(NamedCompound(
            patent_id=patent_id, name=name, smiles=smi, inchikey=ik,
            start=nt.offset, source=SOURCE, cid=cid,
            label=verdict.label, reason=verdict.reason,
            markush=is_markush,
            markush_reason=(
                ("relative_stereo:" + ",".join(stereo)) if stereo else
                (f"stereo_stripped:{dropped.group(0).strip('-')}" if dropped else
                 ("stereo_stripped" if kind == "stereo_stripped" else ""))),
            heading_transform=kind))
        st.by_kind[kind] = st.by_kind.get(kind, 0) + 1
        st.by_line[nt.line] = st.by_line.get(nt.line, 0) + 1

    # MARK THE DRAWN ONES. A cid we could not name, whose own row carries a
    # picture, is not a failure to be rediscovered later — it is a compound
    # this route structurally cannot reach. The row asserts NO structure
    # (`name`/`smiles`/`inchikey` all empty) and carries `drawn_ref` so the
    # picture can be found. Without it, "drawn" and "we missed it" are the
    # same blank, and they need completely different work.
    resolved_cids = {nc.cid for nc in out}
    refs = _drawing_refs(xml)
    # `<chemistry>` id -> the image file named inside that same block, so a
    # marker can carry an openable URL as well as an XML-internal pointer. One
    # pass over the document; `{}` when GP is disabled, which costs the marker
    # nothing (see `gp_images` for why GP's structures are not read at all).
    chem_files = {m.group(1): m.group(2) for m in re.finditer(
        r'<chemistry[^>]*id="([^"]+)"(?:(?!</chemistry>).)*?'
        r'<img[^>]*file="([^"]+)"', xml, re.S)}
    chem_img = {m.group(1): m.group(2).rsplit(".", 1)[0]
                for m in re.finditer(
                    r'<chemistry[^>]*id="([^"]+)"(?:(?!</chemistry>).)*?'
                    r'<img[^>]*file="([^"]+)"', xml, re.S)}
    gp_urls = _gp_image_urls(patent_id)
    markush = _markush_cids(xml)
    scaffolds = _header_scaffolds(xml)
    for cid in searchable:
        if cid in resolved_cids or cid not in refs:
            continue
        ref = refs[cid]
        if ref in scaffolds:
            # THE ROW PICTURE IS NOT THE COMPOUND — the table header holds a
            # shared scaffold and this row supplies a part. Recorded with both
            # halves so enumeration can run after the image work, per the
            # decision to defer it rather than drop these compounds.
            tid, scaf = scaffolds[ref]
            out.append(NamedCompound(
                patent_id=patent_id, name="", smiles="", inchikey="",
                start=-1, source=SOURCE, cid=cid,
                markush=True, markush_kind="header_scaffold",
                markush_reason=f"header_scaffold:{tid}",
                markush_parts=(
                    f"scaffold={scaf};scaffold_file={chem_files.get(scaf, '')};"
                    f"fragment={ref};fragment_file={chem_files.get(ref, '')}")))
            st.markush_marked += 1
            continue
        if cid in markush:
            # NOT DRAWN — see `_markush_cids`. The row's picture is one
            # substituent's value, so `drawn_ref` stays EMPTY rather than
            # pointing at a fragment and calling it the structure. The row is
            # still emitted, carrying the reason: this compound exists only as
            # a scaffold plus substituents that nothing in this tree composes.
            out.append(NamedCompound(
                patent_id=patent_id, name="", smiles="", inchikey="",
                start=-1, source=SOURCE, cid=cid,
                markush=True, markush_kind="substituent_table",
                markush_reason=f"substituent_table:{markush[cid]}",
                markush_parts=(f"fragment={ref};"
                               f"fragment_file={chem_files.get(ref, '')}")))
            st.markush_marked += 1
            continue
        out.append(NamedCompound(
            patent_id=patent_id, name="", smiles="", inchikey="",
            start=-1, source=SOURCE, cid=cid, drawn_ref=ref,
            drawn_file=chem_files.get(ref, ""),
            drawn_url=gp_urls.get(chem_img.get(ref, ref), "")))
        st.drawn_marked += 1
    st.drawn_with_url = sum(1 for nc in out if nc.drawn_url)

    # The cid-as-name rows join the search-based ones here, not earlier: the
    # drawn marker above must not fire for a cid this branch already named.
    out.extend(nc for nc in named_cids if nc.cid not in resolved_cids)

    st.resolved = len(out)
    logger.info(
        "cid_first: %s — %d assay cids (%d unsearchable), %d/%d token "
        "occurrences are assertions, %d candidates, %d resolved",
        patent_id, st.assay_cids, st.unsearchable, st.occurrences_asserted,
        st.occurrences_raw, st.candidates, st.resolved)
    return out, st


def extract_by_cid(xml: str, patent_id: str = "") -> list[NamedCompound]:
    """Every assay compound id this patent names, resolved to a structure.

    Returns `sources.iupac_names.NamedCompound` — the same dataclass every
    other identity route returns, so `verify.dump`, `STRUCT_FIELDS` and every
    consumer are unchanged — with `source == "cid_first"` and at most one
    entry per cid.

    No feature flag is consulted here, deliberately, for the same reason
    `extract_names` consults none: whether to RUN this route is the caller's
    decision, and a pure function that returns `[]` because of a global reads
    as a broken extractor rather than a disabled one.
    """
    return _resolve(xml, patent_id)[0]
