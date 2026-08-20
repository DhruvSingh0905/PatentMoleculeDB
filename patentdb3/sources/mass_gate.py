"""Weigh a resolved structure against the mass the patent prints for it.

WHY THIS EXISTS
----------------
Every acceptance test in this package is a test of the NAME: OPSIN parses it,
`_coverage` says the seed covers the cell, `reagents.classify` labels it. None
of them can tell you the structure belongs to the compound number it was filed
under. A wrong anchor produces a valid name, a valid SMILES and a valid
InChIKey — it is shaped exactly like a success.

It was found by accident. `sources/image_ocr.py` read the caption printed
inside compound 534's own drawing on US10730863 and disagreed with the text
track. The patent settles it in the same table row:

    MS (ESI) 561 (M+H)
    caption      561.17   3-((4-((5-cyclopropyl-3-(3,5-dichloropyridin-4-yl)...
    text track   185.09   Ethyl 3-hydroxy-1,4-dimethyl-1H-pyrazole-5-carboxylate

The text track had anchored the synthesis INTERMEDIATE to the compound number
(`Preparation of Intermediate for Example 534.` — see
`cid_first._INTERMEDIATE_FOR`, which now refuses that shape). Ten compounds on
that patent shipped a reagent this way, every one labelled `compound`.

This module is the general form of that check. It does not care what caused
the wrong anchor, which is the point: the cause above is fixed, and this still
catches the next one.

WHAT IT IS NOT
---------------
Not a correction. It never edits a name, never picks between two candidates,
and never deletes a row — it returns a verdict and the caller records it.
BindingDB is not involved and neither is any external reference; the only
evidence used is the number the patent itself printed in that compound's row.

Not a coverage measure either, and its reach is NARROWER THAN THE PATENT COUNT
SUGGESTS. Two conditions must both hold for one row: the compound must have a
resolved structure, AND its own table row must print an MS. Measured over the
full 137-patent corpus (`verify --all --dump --no-heal`, 38,671 structures):

    rows weighed                    74   (0.2%)
    patents printing an MS anywhere  5   of 137

Those two lines describe different things and confusing them is easy. Only 5
patents print an MS inside an assay row at all, and on THREE of those five the
set of compounds with a mass and the set with a resolved structure do not
intersect:

    US10125101       7 resolved     30 with a mass    overlap 0
    US10329273       1 resolved     23 with a mass    overlap 0
    US20240166635    0 resolved    195 with a mass    overlap 0

So "5 patents" is not "5 patents' worth of checking". Nearly all 74 weighable
rows are on US10730863, and no second patent has yet been found where the gate
does real work. A silent verdict means "not checkable", and it must never be
read as "checked and fine".

THE TOLERANCE
--------------
The patent prints a NOMINAL integer (`561`) and this computes an exact mass.
`tolerance()` is FLAT at 1.5 Da and its own docstring carries the measurement
that set it, including what it cannot see. Do not restate the number here —
one of the two copies will drift, and it has already happened once.

THE ADDUCT
-----------
Not always `[M+H]`. US10125101 prints 20 rows as `[M-H]-` beside 44 as
`[M+H]+`, and adding a proton to a negative-mode row makes a CORRECT structure
read 2.015 Da light — a confident wrong verdict, on a row that `images.emit`
then discards from the truth set. `reported_shifts` reads the sign from the
same row as the m/z, off the raw markup, and `verdict` takes it as an
argument. Its default is `[M+H]` because most rows print that, and that
default is now a stated fallback rather than an unexamined assumption.
"""
from __future__ import annotations

import html
import logging
import re

from .uspto_assays import normalize_cid as _normalize_cid

logger = logging.getLogger(__name__)

# ONE ANCHOR: THE ADDUCT. NOT THE INSTRUMENT NAME.
#
# Two vendor-enumerating patterns used to live here, `MS\s*\(ESI...\)` and a
# list of instrument prefixes, and each new document added a spelling. A census
# of all 58,273 mass-like statements in this corpus ended that approach:
#
#     the instrument name VARIES     MS (ESI), MS (apci), MS (ES+), MS(ES),
#                                    MS obsd. (ESI+), m/z (ESI, +ve), ESI+:,
#                                    HRMS(A), ESI-MS, LC/MS
#     the ADDUCT does not            [M+H], (M+H), [M+Na], MH+, m/z, m/e
#
#     statements carrying an adduct or m/z marker  44,858   77.0%
#     the instrument patterns read                 27,652   47.5%
#
# `MS (apci) m/z=N (M+H).` was invisible 745 times and `MS (ES + ): m/e=N.`
# 644 times, for no better reason than word order.
#
# The isotope label inside the bracket is a number and is not a mass —
# `[M(37Cl) + H]+ = 453` gave 37 — so the pattern steps over it.
_ADDUCT_ANCHOR = re.compile(
    r"[\[(]\s*M\s*(?:\([^)]{0,8}\)\s*)?[+\-−–]\s*(?:H|Na|K|1|NH4)\s*[\])]"
    r"|\bm\s*/\s*[ez]\b"
    # `[MH]`, `[MH]+`, `MH+`, `(MH+)`. The bracket alone is the statement —
    # requiring the `+` immediately after `MH` missed `[MH] + : 620.`
    r"|\[\s*MH\s*\]|\bMH\s*\+", re.I)

# The instrument is a FALLBACK anchor, for a statement that names one and
# states no adduct: `HRMS (ESI) 350.1`. Where both appear the adduct wins —
# an instrument read walks forward blind and steps into the adduct bracket.
_INSTRUMENT = re.compile(
    r"MS\s*\(ESI[^)]*\)|\b(?:ESI|APCI|LC|HR|HPLC)[\s/-]*MS\b", re.I)

# WHAT THE ANCHOR INTRODUCES. `[M+H]=477.2` and `HRMS (ESI) 350.1` introduce
# their number; `617 (M + H). Ex. 276` states it BEFORE and introduces nothing,
# so reading forward there walks into the next sentence. The gap may hold the
# ionisation mode in brackets and nothing else — anything with words in it is
# a different statement.
_GAP_OK = re.compile(r"^[\s+\-–−\])]*(?:\([^)]{0,14}\)[\s:,]*)?[=:]?\s*$")

# A MOLECULAR FORMULA'S SUBSCRIPT IS A NUMBER, AND IT SITS EXACTLY WHERE THE
# MASS SITS. `LCMS calculated for C 12 H 18 ClIN 3 OSi (M+H) + m/z=410.0` gave
# 12 — the carbon count — because a `[^0-9]` run reaches the first digit it
# finds and that digit belongs to the formula. Every compound in US10280164
# (16 of 16) and US10722495 (48 of 51) read as contradicting on nothing but
# this. 4,772 of 26,332 instrument mentions in the corpus print `calculated
# for`, so it is not one vendor's habit.
#
# Same family as the adduct digits this file already steps over, and as
# `&#x2212;` being read as 2,212: THE GAP BETWEEN THE INSTRUMENT AND THE MASS
# HAS NUMBERS IN IT, AND THEY ARE NOT MASSES. Enumerating the things that can
# appear in the gap is what failed twice; this asks about the number instead.
#
# A subscript follows its ELEMENT SYMBOL. A mass follows `=`, `:`, `,`, a
# bracket, or `m/z`. So a candidate whose preceding text ends in an element
# symbol is not a mass, and the scan moves to the next number.
_ELEMENT_TAIL = re.compile(r"(?:^|[^A-Za-z])[A-Z][a-z]?\s*$")

# EVERY NUMBER, WHOLE. This read `\d{2,4}(?:\.\d+)?`, which can start in the
# MIDDLE of a number: `t_R=0.69 min` has one digit before the point, so the
# pattern could not match at the `0` and matched the `69` instead. US12011444
# writes `LC-MS A: t_R=0.69 min; [M+H]+=426.97` and every one of its 88
# weighable structures was weighed against a RETENTION TIME. US10730877,
# US10544143 and US11053244 print the same boilerplate.
#
# So the number is matched whole and judged afterwards, on the count of digits
# before the point: an m/z in this corpus has two to four, and a retention
# time, a yield or a step number has one. Judging a token is safe where
# matching a prefix of one is not.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_MZ_DIGITS = (2, 4)

# How far past the instrument name the mass may sit. A formula plus an adduct
# is long — `calculated for C 23 H 33 ClN 5 OSi (M+H) + m/z=` is 46 characters
# — so the window has to clear one, and stopping short is what made the old
# patterns settle for the formula's first digit.
_MASS_WINDOW = 80

# How close a mass sits when NO formula is in the way. `MS (ESI) 485 (M+H)`,
# `LC-MS: 412 (M+1)`, `m/z=477.2` — the number follows the statement almost
# immediately. See `printed_mass` for why the far window has to be earned.
_MASS_NEAR = 24

# A COLUMN, not a sentence: `MS (m/e)` over `481.0 (M + H), CP`, or
# `[m/z (M+H)]` over `450.24`. 22 of 137 patents print their mass only in this
# shape, including every markush substituent table — which is why the wiki's
# claim that markush tables cannot be verified against the document was wrong.
# Which columns those are is `uspto_assays.build_columns`'s answer, not a
# second header regex here; this is only the CELL under one.
#
# A number, optionally followed by the adduct and any trailing note
# (`481.0 (M + H), CP`). It must START with the number: a cell reading
# `not determined 481` states no mass for this row.
_MASS_CELL = re.compile(r"^\s*(\d{2,4}(?:\.\d+)?)\s*(?:[(\[]|,|\s|$)")

# The row's own compound number: the first cell, digits with an optional
# letter suffix. Same shape `uspto_assays` reads, and deliberately strict —
# a row whose first cell is prose states no compound number.
_ROW_CID = re.compile(r"\s*(\d+[A-Za-z]?)\s")

# CALS table markup, in either spelling. A heading section may CONTAIN a table,
# and a mass printed inside one belongs to that table's own row — the first
# shape below already owns it. Cutting the tables out of the section leaves
# only the prose the heading itself introduces.
_TABLE_BLOCK = re.compile(r"<tables\b.*?</tables>|<table\b.*?</table>",
                          re.S | re.I)

PROTON = 1.00728

# THE ADDUCT IS NOT ALWAYS [M+H], AND ASSUMING IT WAS PRODUCED WRONG VERDICTS.
# This constant's comment used to read "the only adduct this corpus prints".
# It is not: US10125101 prints 20 rows as `[M - H]-` (negative mode) beside 44
# as `[M + H]+`. Adding a proton to a negative-mode row makes it read 2.015 Da
# light — outside the 1.5 Da window, so a CORRECT structure is reported as
# contradicting, and `images.emit` then discards it from the truth set. 22
# rows corpus-wide are affected and all of them would have been thrown away.
#
# The minus sign may be ASCII, U+2212, an en dash, or an XML entity for any of
# those, and it may sit inside a `<sup>` tag. So this is matched against the
# RAW markup, not a tag-stripped string — the same reason `cid_first`
# `_drawing_refs` reads raw XML.
# AND THE BRACKET IS NOT ALWAYS SQUARE. This spelled four ways to write a
# minus sign and exactly one way to write a bracket. US9718825 prints its
# adduct in ROUND parentheses — `(M &#x2212; H)` 30 times, `(M&#x2212;H)` 5
# times, square brackets zero times — so the gate saw no negative-mode row in
# that document at all, added a proton to every one, and reported 28 correctly
# assembled molecules as contradicting by exactly 2 x PROTON. Re-weighed with
# the sign it prints, 30 of 30 agree, worst residual 0.27 Da.
#
# Corpus-wide the old pattern read 278 of 688 negative-mode occurrences: 40%.
# No document mixes the two styles, so a patent is invisible or it is not.
#
# Same family as `\besi\b` matching "Synthesis" and `REPORTED` requiring the
# literal `MS (ESI`: a pattern that enumerates one axis of variation
# exhaustively and freezes another without noticing it had one.
_ADDUCT_MINUS = re.compile(
    r"[\[(]\s*M\s*(?:&#x2212;|&#8722;|&minus;|[-−–])\s*(?:H|1)\s*[\])]", re.I)

# A THIRD ADDUCT, AND IT IS NOT RARE. `[M+Na]` appears 674 times over 37
# patents. US9670157 prints `[M+H]+=421.05 (M+Na)` — the tag says protonated
# and the trailing marker says sodiated, and the VALUE is the sodiated one.
# Judged as `[M+H]` those rows read 21.98 Da light, which is exactly Na - H,
# and six of them contradicted on nothing else.
#
# The sodium marker wins wherever it appears, because it is the more specific
# statement: a row that says `(M+Na)` at all is reporting a sodium adduct, and
# the `[M+H]` beside it is boilerplate the author did not clear.
_ADDUCT_SODIUM = re.compile(r"[\[(]\s*M\s*\+\s*Na\s*[\])]", re.I)

# Sodium ION, not the neutral atom: 22.98977 less one electron.
SODIUM = 22.98922

# AN INSTRUMENT NAME IS ALSO A PURIFICATION METHOD. `purified by preparative
# LCMS (Waters Xbridge C18, 19x150 mm)` states no mass at all — and the gate
# read 19, the column's DIAMETER IN MILLIMETRES, on 7 rows over two patents.
# A statement introduced by `preparative`, `prep-`, `purified` or `column` is
# describing how the compound was separated, not what it weighs.
_PURIFICATION = re.compile(
    r"\b(?:prep(?:arative|\.)?|purif\w*|column|chromatograph\w*)\b[\s-]*$", re.I)


def _valid(tok: str) -> bool:
    """Could this token be an m/z? Two to four digits before the point.

    A retention time (`0.69`), a yield and a step number have one. Matching a
    PREFIX of a number instead of the whole one is how `t_R=0.69 min` became
    69 on all 88 of US12011444's weighable rows.
    """
    lo, hi = _MZ_DIGITS
    return lo <= len(tok.split(".")[0]) <= hi


def _forward(flat: str, at: int):
    """The number this anchor introduces, or None. `(value, index)`."""
    window = flat[at:at + _MASS_WINDOW]
    skipped = False
    for n in _NUMBER.finditer(window):
        if _ELEMENT_TAIL.search(window[:n.start()]):
            skipped = True                    # a formula subscript
            continue
        if not _valid(n.group(0)):
            # NOT a reason to look further. `skipped` exists for a molecular
            # FORMULA standing between the anchor and its mass; a stray digit
            # does not push the mass anywhere. Letting this set it meant the
            # `1` of `1 H NMR (400 MHz` earned the wide window and the gate
            # read the spectrometer's FREQUENCY as the mass — 400.0 on 50
            # US9745328 rows, 500.0 and 600.0 on others.
            continue
        if not (_GAP_OK.match(window[:n.start()]) or skipped):
            return None                       # not introduced by this anchor
        return float(n.group(0)), at + n.start()
    return None


def _backward(flat: str, at: int):
    """The nearest mass BEFORE `at`, or None.

    The number sits on either side of the adduct and the corpus uses both:
    `[M+H]+=477.2` after, `MS (ESI) 485 (M+H)` before, and
    `Calcd: 335.14 Found: 336.0 [M + H]+` puts the FOUND value immediately
    before — which is the one wanted, and nearest gets it.
    """
    start = max(0, at - _MASS_NEAR)
    window = flat[start:at]
    best = None
    for n in _NUMBER.finditer(window):
        if _ELEMENT_TAIL.search(window[:n.start()]) or not _valid(n.group(0)):
            continue
        best = (float(n.group(0)), start + n.start())
    return best


def _mass_hits(flat: str):
    """Every m/z the text states, as `(mass, index in flat)`, in order.

    `flat` is tag-stripped and entity-unescaped. The adduct anchors first; the
    instrument anchors only where no adduct sits in its window.
    """
    out: dict[int, float] = {}
    for m in _ADDUCT_ANCHOR.finditer(flat):
        if _PURIFICATION.search(flat[max(0, m.start() - 24):m.start()]):
            continue
        got = _forward(flat, m.end()) or _backward(flat, m.start())
        if got:
            out.setdefault(got[1], got[0])
    for m in _INSTRUMENT.finditer(flat):
        if _PURIFICATION.search(flat[max(0, m.start() - 24):m.start()]):
            continue
        if _ADDUCT_ANCHOR.search(flat[m.end():m.end() + _MASS_WINDOW]):
            continue                          # the adduct already read it
        got = _forward(flat, m.end())
        if got:
            out.setdefault(got[1], got[0])
    for pos in sorted(out):
        yield out[pos], pos


# THE WIDEST TWO MASSES ONE COMPOUND MAY LEGITIMATELY SHOW IN ONE STATEMENT.
# `[M+H]` and `[M+Na]` of the same molecule are 21.98 Da apart, and a calcd
# printed beside a found differ by rounding. Anything wider is a SECOND
# MOLECULE, and a block stating two of those cannot say which one its heading
# names.
_ONE_COMPOUND_SPREAD = 22.5


def _unambiguous(masses: list) -> "float | None":
    """The one mass a block states, or None if it states more than one.

    REFUSING IS THE POINT. This gate's job is to check a structure someone
    else produced — a drawing MolScribe read, a markush row assembled from a
    scaffold — and for that a WRONG reference is far worse than no reference:
    it discards a correct answer and shrinks the only truth set a recogniser
    can be scored against.

    Picking one of several was tried both ways and both are wrong. Taking the
    FIRST gives Step 1's intermediate, a constant 243 Da light on US9694016.
    Taking the LAST gives a co-isolated byproduct on US11254686, or the next
    compound's product where a heading sits after its own synthesis paragraph
    on US9694016. There is no third choice that reads the author's intent, so
    the block yields nothing and says so.
    """
    if not masses:
        return None
    if max(masses) - min(masses) > _ONE_COMPOUND_SPREAD:
        return None
    return masses[-1]


def printed_mass(flat: str) -> "float | None":
    """The one m/z this text states, or None if it states none or several."""
    return _unambiguous([v for v, _ in _mass_hits(flat)])


def _shift(row_markup: str) -> float:
    """Mass to ADD to a neutral structure to compare it with this row.

    `+1.007` for `[M+H]`, `-1.007` for `[M-H]`. Defaults to `[M+H]`, which is
    what 331 of the 353 adduct-bearing rows in this corpus print — but the
    default is now a stated fallback rather than an assumption nobody checked.
    """
    if _ADDUCT_SODIUM.search(row_markup):
        return SODIUM
    return -PROTON if _ADDUCT_MINUS.search(row_markup) else PROTON

VERDICT_AGREES = "agrees"
VERDICT_CONTRADICTS = "contradicts"
VERDICT_UNCHECKED = ""    # no MS in the row, or the structure would not weigh
# A FOURTH VALUE, AND THE REASON IT IS NOT BLANK. Without rdkit nothing can be
# weighed, and a blank column would be indistinguishable from the ordinary
# "this row prints no mass" — the gate would be switched off and every
# artifact would still look normal. A row that HAD a mass to check against and
# could not be checked says so in the data, where a reader cannot miss it.
VERDICT_UNAVAILABLE = "gate_unavailable"


def available() -> bool:
    """Can this gate run at all? rdkit is an optional dependency."""
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


def tolerance(nominal: float) -> float:
    """Half-width of the accepted window, in Da. Flat, and deliberately tight.

    THIS USED TO SCALE WITH MASS AND THAT WAS WRONG. The scaling term was
    absorbing a difference the gate can now handle properly — a patent quotes
    the MONOISOTOPIC mass on some rows and the AVERAGE on others, and the two
    diverge by ~1.4 Da at 600 Da. Widening the window to swallow both let a
    real error through: on US10730863 the old window was 1.31 Da at 596, and
    DECIMER's characteristic failure is +2.016 Da (two hydrogens, from reading
    an aromatic ring as partly saturated). Three of its four wrong answers on
    that patent sat just outside the truth and just inside the window.

    `verdict` now compares against BOTH masses and keeps the nearer, so this
    only has to cover rounding.

    WHAT THIS GATE CANNOT SEE, STATED RATHER THAN TUNED AWAY. Measured over
    US10730863, where 70 structures have a printed mass to check against:

        correct structures, worst |delta|          1.20 Da   (cid 460)
        DECIMER two-hydrogen error, |delta|        1.19 Da   (cid 140)

    Those overlap, so NO threshold separates them. Set at 0.9 the gate catches
    all four DECIMER errors and falsely flags four correct structures; set at
    1.5 it flags nothing correct and catches two of the four. There is no
    third option, and pretending otherwise by picking a number in between
    would hide the overlap rather than resolve it.

    1.5 is chosen because a FALSE POSITIVE HERE IS EXPENSIVE: `images.emit`
    demotes a flagged row out of the truth set, so a wrong flag discards a
    good answer and shrinks the only population any recogniser can be scored
    against. The gate's demonstrated job is the large error — a 376 Da
    anchored reagent, a 4 Da atom substitution — and a two-hydrogen
    misreading at 600 Da is simply below its resolution. Catching that class
    needs a different instrument, not a tighter window.
    """
    return 1.5


def reported_masses(xml: str) -> dict[str, float]:
    """`{compound number -> the m/z the patent prints in that compound's row}`.

    THE VALUE IS THE PRINTED NUMBER, NOT A NEUTRAL MASS. Which adduct it
    represents is a separate question — see `reported_shifts`, and pass both
    to `verdict`.

    Read per `<row>` rather than per document, so a mass can only ever be
    attributed to the row that carries it. A row with no id, or no MS, is
    absent from the result — never guessed at from a neighbour.

    First occurrence wins. A compound number repeated across tables states the
    same compound, and the alternative (last wins) would silently prefer
    whichever table the parser happened to reach last.

    FOUR SHAPES, IN PRIORITY ORDER. A sentence naming the instrument
    (`REPORTED`, then `REPORTED_ALT`) beats a bare column, because a sentence
    says what it is measuring and a column says it only in its header. The
    column shape is read last and only for rows the sentences did not answer.
    A sentence in a table row and a sentence in a heading section are the same
    kind of statement; the row is tried first only because it is the narrower
    scope.
    """
    return {cid: mass for cid, (mass, _shift_) in _scan(xml).items()}


def _scan(xml: str) -> dict[str, tuple[float, float]]:
    """`{compound number -> (printed m/z, the adduct shift for that row)}`.

    ONE SCAN FOR BOTH ANSWERS, and that is the point. `reported_masses` and
    `reported_shifts` used to walk the document separately with different
    patterns, so broadening one and not the other gave every row the other
    could not see a silent `[M+H]` default. On a negative-mode row that makes a
    CORRECT structure read 2.015 Da light — a confident wrong verdict, and
    `images.emit` then discards that row from the truth set. It cost 9 correct
    structures the first time this file was broadened, all of them landing in a
    tight cluster at 2.1-2.3 Da that is the signature of exactly this mistake.

    A shape either yields both numbers or neither.

    EVERY KEY IS NORMALISED, ON BOTH SIDES. `check` looks up `r.cid`, and the
    heading route stores a cid that has been through `normalize_cid`, so a
    document numbering its examples `007` produced a dict keyed `007` and a
    lookup asking for `7`. Zero overlap, and the gate simply reported those
    rows unchecked — the same shape as `cid_first`'s raw-cell dict, which cost
    US12065407 every drawn marker it had.
    """
    out: dict[str, tuple[float, float]] = {}
    for m in re.finditer(r"<row>.*?</row>", xml, re.S):
        raw = m.group(0)
        # UNESCAPE BEFORE MATCHING. Stripping tags leaves `&#x2212;` intact,
        # and `REPORTED_ALT` then reads its DIGITS as the mass: US11485738
        # prints `LC-MS [M &#x2212; 1]: 453.3` and the gate recorded 2212.0 —
        # the character code of a minus sign — for cids 221 and 226. Both
        # structures were correct and both were reported as contradicting.
        # A referee that can invent a 2,212 Da compound is worse than no
        # referee, because it discredits right answers.
        flat = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        mass = printed_mass(flat)
        if mass is None:
            continue
        cid = _ROW_CID.match(flat)
        if cid and (key := _normalize_cid(cid.group(1))):
            out.setdefault(key, (mass, _shift(raw)))
    for cid, pair in _heading_masses(xml).items():
        out.setdefault(cid, pair)
    for cid, pair in _column_masses(xml).items():
        if key := _normalize_cid(cid):
            out.setdefault(key, pair)
    return out


def _heading_masses(xml: str) -> dict[str, tuple[float, float]]:
    """The heading-section shape. `{compound number -> (printed m/z, shift)}`.

    `<heading>Example 418: <name></heading>`, then the characterisation data in
    the `<p>` elements beneath it, up to the next heading:

        <heading>Example 418: N-(4-cyano-3-(...)phenyl)-2-(...)isonicotinamide
        <p><chemistry><img .../></chemistry></p>
        <p>1H NMR (400 MHz, cd3od) ... LCMS (m/z) (M+H)=477.2, Rt=0.78 min.</p>

    THE GATE COULD NOT SEE ANY OF THIS. `_scan` read `<row>` elements and
    nothing else, so a document that prints its masses in prose — which is most
    of them — had no reference at all. 3,735 of 38,402 structures carried a
    verdict; this brings 7,326 more compounds into range, and the documents it
    reaches are the ones that need it: US10245267 states the SAME NAME under
    Examples 415, 418 and 419 while printing 484.2, 477.2 and 473.2 beneath
    them. The name is repeated, the masses are not, and the paragraph disproves
    the name it sits under.

    Attribution is bounded exactly as the row shape is. The section ends at the
    next heading, so a mass can only ever reach the compound whose heading
    introduces it, never a neighbour's.

    The id comes from `iupac_names._HEADING_ID` — the SAME cue that names the
    compound — rather than a second pattern of this module's own. A referee
    that disagreed with the route it referees about which compound a heading
    names would be worse than no referee.
    """
    # Imported here, not at module scope: this file is otherwise a leaf, and
    # `iupac_names` pulls in OPSIN, the anchor index and the repair library.
    from .iupac_names import (_HEADING_EL, _HEADING_ID,
                              _NOT_A_FINISHED_COMPOUND)

    out: dict[str, tuple[float, float]] = {}
    heads = list(_HEADING_EL.finditer(xml))
    for i, h in enumerate(heads):
        title = html.unescape(re.sub(r"<[^>]+>", " ", h.group(1))).strip()
        idm = _HEADING_ID.match(title)
        if not idm:
            continue
        # `Preparation 16` AND `Example 16` ARE TWO DIFFERENT COMPOUNDS. They
        # are separate numbering series, and normalising both gives the cid
        # `16` twice — so an intermediate's mass lands on the example's
        # structure and reports a correct molecule as contradicting. 89 of
        # US20250163061A1's 167 structures failed on exactly that.
        #
        # The name route already refuses these headings, so the referee refuses
        # the same ones, by the same set. Two rules for which headings assert a
        # compound would eventually disagree, and the referee would lose.
        if (idm.group("label") or "").lower() in _NOT_A_FINISHED_COMPOUND:
            continue
        cid = _normalize_cid(idm.group("cid"))
        if not cid:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(xml)
        raw = _TABLE_BLOCK.sub(" ", xml[h.end():end])
        flat = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        # THE LAST MASS, NOT THE FIRST. A heading section is a whole synthesis:
        #
        #     Example 1: Synthesis of N-(4-methyl-3-(6-morpholinopyrimidin-4-yl)
        #                phenyl)-3-(trifluoromethyl)benzamide
        #     ... Step 1 ... LCMS (m/z) (M+H)=200.0/201.8 ...
        #     ... Step 2 ... LCMS (m/z) (M+H)=443.2 ...
        #
        # 200.0 is Step 1's intermediate. The heading names what the section
        # PRODUCES, so the mass that belongs to it is the one stated last.
        # Taking the first reported every multi-step example as contradicting
        # by the mass of everything the last step still had to add — 243 Da on
        # US9694016 cids 1, 3 and 6 alike, a constant offset that is the
        # signature of comparing against the wrong molecule rather than of a
        # wrong structure.
        hits = list(_mass_hits(flat))
        mass = _unambiguous([v for v, _ in hits])
        if mass is not None:
            at = next(a for v, a in hits if v == mass)
            # The adduct for THAT statement, not for whichever step ran in
            # negative mode. Same both-or-neither rule as `_scan`.
            out.setdefault(cid, (mass, _shift(flat[max(0, at - 60):at + 60])))
    return out


def _column_masses(xml: str) -> dict[str, tuple[float, float]]:
    """The bare-column shape. `{compound number -> (printed m/z, adduct shift)}`.

    The header names the measurement once and every cell below it is a plain
    number. `uspto_assays` already solves the hard half of this — which header
    governs which column, across split headers and sibling tgroups — so this
    reads the assembled table rather than re-deriving it from flat text, which
    is what made the header invisible to the sentence patterns.

    THE COLUMN MUST BE A MASS COLUMN, NOT A NUMERIC ONE. The header has to say
    so. An IC50 column of integers is indistinguishable from an m/z column by
    its values alone — that mistake, made one layer down in `build_columns`,
    is what let an assay column win the compound-id role on US10253019.

    So the column kinds come from `uspto_assays.build_columns`, which already
    decides which header governs which column across split headers and sibling
    tgroups. Reading `Table.header_rows` directly instead re-derives that badly:
    US11548900 spells its header over two lines (`LCMS` above
    `m/z = (M + H)+`) and only the first is classed as a header row, so a
    hand-rolled scan sees `LCMS` over a blank and finds nothing.
    """
    from .uspto_assays import MS, build_columns, CID
    from .uspto_xml import assemble_blocks, parse_tables

    out: dict[str, tuple[float, float]] = {}
    try:
        tables = assemble_blocks(parse_tables(xml))
    except Exception as e:                       # a parse failure is not a mass
        logger.debug("column mass scan skipped: %r", e)
        return out
    for t in tables:
        try:
            cols = build_columns(t)
        except Exception as e:
            logger.debug("column mass scan skipped on %s: %r", t.table_id, e)
            continue
        mass_cols = [c.index for c in cols if c.kind == MS]
        if not mass_cols:
            continue
        id_col = next((c.index for c in cols if c.kind == CID), 0)
        # The adduct is stated in the HEADER of a column, not in every cell:
        # `m/z = (M + H)+` over bare numbers. A header is one string per table,
        # so it is read once — and the cell is still checked, because a table
        # printing `[M+H]` in its header may print `[M-H]` in a given row.
        #
        # BOTH ASK `_shift`, RATHER THAN RE-DERIVING IT. This block used to test
        # `_ADDUCT_MINUS` inline, in two places, so when `_shift` learned to
        # read `[M+Na]` this shape did not: US9670157 prints its sodium masses
        # in a COLUMN, and its six sodiated rows kept being judged as `[M+H]`
        # and kept contradicting by Na - H. One rule, called twice, is the only
        # arrangement where teaching it something teaches every caller.
        head_shift = {i: _shift(next((c.header or "" for c in cols
                                      if c.index == i), ""))
                      for i in mass_cols}
        for row in t.body_rows:
            if len(row) <= id_col:
                continue
            cid = (row[id_col].text or "").strip().split()
            if not cid or not re.fullmatch(r"\d+[A-Za-z]?", cid[0]):
                continue
            for i in mass_cols:
                if i >= len(row):
                    continue
                cell = (row[i].text or "").strip()
                hit = _MASS_CELL.match(cell)
                if hit:
                    # The cell wins only when it states an adduct of its own.
                    stated = (_ADDUCT_MINUS.search(cell)
                              or _ADDUCT_SODIUM.search(cell))
                    shift = (_shift(cell) if stated
                             else head_shift.get(i, PROTON))
                    out.setdefault(cid[0], (float(hit.group(1)), shift))
                    break
    return out


def reported_shifts(xml: str) -> dict[str, float]:
    """`{compound number -> mass to add to the neutral structure}`.

    Read from the SAME `<row>` as the m/z, on the raw markup so a `<sup>`
    around the sign cannot hide it. Absent from the result when the row prints
    no mass at all; `verdict` then falls back to `[M+H]`.
    """
    return {cid: sh for cid, (_m, sh) in _scan(xml).items()}


def _mass(smiles: str) -> "tuple[float, float] | None":
    """`(monoisotopic, average)` mass of `smiles`, or None if it will not build.

    BOTH, because the patent may have printed either and does not say which.
    On US10730863 compound 140 the row prints 596: the correct structure is
    595.18 monoisotopic and 596.53 average, so demanding monoisotopic alone
    would call a right answer wrong. Compound 438 on the same patent is the
    reverse — 684.14 monoisotopic against a printed 684, average 685.61.

    rdkit is imported here, not at module scope: it is a heavy dependency and
    a tree without it must still extract, just without this gate.
    """
    if not smiles:
        return None
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Descriptors import ExactMolWt, MolWt
    except ImportError:
        return None
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # A SALT IS WEIGHED AS ITS FREE BASE, because that is what the patent
    # printed. `...azetidin-3-ol trifluoroacetic acid salt` resolves to
    # `FC(C(=O)O)(F)F.<amine>` — two disconnected fragments — and the document
    # reports `[M+H]+` for the amine alone. A counterion does not carry the
    # charge in positive-mode ESI, so it never appears in that number.
    #
    # Weighing the whole SMILES added the counterion to our side only. 168 rows
    # contradicted on nothing else, and on 155 of them (92%) the delta IS the
    # counterion: +115.0 for trifluoroacetate, +36.98 for chloride, +228.0 for
    # a bis-TFA salt. Neither the structure nor the printed mass was wrong —
    # the comparison was.
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    return (ExactMolWt(mol), MolWt(mol))


def verdict(smiles: str, reported: "float | None",
            shift: float = PROTON) -> tuple[str, float | None]:
    """`(verdict, delta)` for one structure against one reported m/z.

    `shift` is what the row's adduct adds to the neutral mass: `+PROTON` for
    `[M+H]`, `-PROTON` for `[M-H]`. Defaulting it to `[M+H]` is safe only
    because that is what most rows print — pass the real value from
    `reported_shifts`, or a negative-mode row is judged 2.015 Da light and a
    correct structure is reported as contradicting.

    `delta` is signed and in Da — positive means the structure is heavier than
    the patent says. It is carried so a reader can tell a lost methyl (14 Da)
    from an anchored reagent (376 Da) without recomputing anything.
    """
    if reported is None:
        return VERDICT_UNCHECKED, None
    m = _mass(smiles)
    if m is None:
        return VERDICT_UNCHECKED, None
    # The patent printed ONE of these and did not say which, so the nearer one
    # is the fair comparison. Reporting the nearer delta also keeps the number
    # a reader can act on: an anchored reagent is hundreds of Da out either
    # way, and a two-hydrogen recognition error is 2.0 either way.
    deltas = [mass + shift - reported for mass in m]
    delta = min(deltas, key=abs)
    if abs(delta) <= tolerance(reported):
        return VERDICT_AGREES, delta
    return VERDICT_CONTRADICTS, delta


def check(rows: list, xml: str, patent_id: str = "") -> dict[str, int]:
    """Stamp `mass_check` and `mass_delta` on every row that can be weighed.

    Mutates in place and returns the tally. Rows are `NamedCompound` or
    `TableName`; both carry `cid`, `smiles` and the two fields set here.

    NOTHING IS DROPPED. A contradicting row keeps its structure and ships with
    the verdict attached, because this gate is new and its false-positive rate
    against the corpus is not yet known. Making the defect VISIBLE is the
    change; deciding what to do about a flagged row is a separate call, and
    one the artifact now carries enough information to make.
    """
    from . import losses as _losses

    ms = reported_masses(xml)
    shifts = reported_shifts(xml)

    # THE GATE ANNOUNCES ITS OWN ABSENCE. Silently returning "everything is
    # unchecked" is how a safety check stays off for a week without anyone
    # noticing: the column is blank on 99.8% of rows anyway, so blank carries
    # no information. Rows that had something to check against are stamped
    # `gate_unavailable`, and one record per patent lands in the loss log.
    if not available():
        n = sum(1 for r in rows
                if ms.get(_normalize_cid(str(getattr(r, "cid", "") or ""))))
        for r in rows:
            try:
                r.mass_check = (VERDICT_UNAVAILABLE
                                if ms.get(_normalize_cid(
                                    str(getattr(r, "cid", "") or "")))
                                else VERDICT_UNCHECKED)
                r.mass_delta = ""
            except AttributeError:
                continue
        logger.warning("mass_gate: %s — rdkit is not installed; %d row(s) that "
                       "could have been weighed were NOT checked", patent_id, n)
        if _losses.ENABLED:
            _losses.record("mass_gate_unavailable", patent_id,
                           reason="rdkit not installed", weighable_rows=n)
        return {VERDICT_AGREES: 0, VERDICT_CONTRADICTS: 0, "unchecked": 0,
                VERDICT_UNAVAILABLE: n}

    tally = {VERDICT_AGREES: 0, VERDICT_CONTRADICTS: 0, "unchecked": 0}
    for r in rows:
        cid = getattr(r, "cid", None)
        key = _normalize_cid(str(cid)) if cid else ""
        v, d = verdict(getattr(r, "smiles", ""), ms.get(key),
                       shifts.get(key, PROTON))
        try:
            r.mass_check = v
            r.mass_delta = round(d, 2) if d is not None else ""
        except AttributeError:                 # a row type without the fields
            continue
        tally[v if v else "unchecked"] += 1
        if v == VERDICT_CONTRADICTS and _losses.ENABLED:
            _losses.record(
                "mass_contradicts_patent", patent_id, cid=cid,
                name=getattr(r, "name", "")[:120], source=getattr(r, "source", ""),
                reported_mh=ms.get(key), delta=round(d or 0, 2))
    if tally[VERDICT_CONTRADICTS]:
        logger.info("mass_gate: %s — %d of %d weighable structures contradict "
                    "the mass their own row prints", patent_id,
                    tally[VERDICT_CONTRADICTS],
                    tally[VERDICT_CONTRADICTS] + tally[VERDICT_AGREES])
    return tally
