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

logger = logging.getLogger(__name__)

# `MS (ESI) 561 (M+H)`, `MS (ESI+) m/z 561.2 (M+H)+`, `MS(ESI) 561`. The mass
# is the FIRST number after the ionisation mode — a `+` charge marker, `m/z`
# and a space may sit between, and nothing else observed does.
#
# THE DECIMAL IS PART OF THE NUMBER. Every pattern here captured `(\d{2,4})`
# and dropped `(?:\.\d+)?` on the floor, so `414.3` was compared as `414` —
# up to 0.99 Da of a 1.5 Da budget thrown away before any chemistry happened,
# and the docstring above says the value IS the printed number. Keeping it
# turned 6 corpus contradictions back into agreements.
#
# THIS PATTERN ALONE READ 5 OF 137 PATENTS. It requires the literal order
# `MS (ESI`, and the corpus does not agree on it: US8722692 prints `ESI-MS m/z
# 499.2 (MH+)` 500 times and `MS (ESI` zero times, so `reported_masses`
# returned `{}` for that whole document and six structures wrong by 40-145 Da
# were never weighed. The gate did not pass them — it never saw them.
REPORTED = re.compile(r"MS\s*\(ESI[^)]*\)[^0-9]{0,12}(\d{2,4}(?:\.\d+)?)", re.I)

# The same statement with the words in the other order, or another instrument.
# `ESI-MS m/z 499.2 (MH+)`, `LC-MS: 412 (M+1)`, `HRMS (ESI) m/z 350.1`.
# Anchored on `m/z` or on the mode marker so a bare `MS` in prose cannot match.
# THE ADDUCT SITS BETWEEN THE INSTRUMENT AND THE MASS, AND IT HAS DIGITS IN IT.
# `LC-MS [M - 1]: 453.3` puts `[M - 1]` in the gap, so a `[^0-9]` run stops on
# the `1` and never reaches 453.3. Before entities were unescaped this matched
# anyway — on the digits of `&#x2212;`, recording a 2,212 Da compound. So the
# gap explicitly steps over one bracketed group.
REPORTED_ALT = re.compile(
    r"\b(?:ESI|APCI|LC|HR|HPLC)[\s/-]*MS\b"
    r"[^0-9]{0,10}(?:[\[(][^\])]{0,14}[\])])?[^0-9]{0,10}"
    r"(\d{2,4}(?:\.\d+)?)", re.I)

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


def _shift(row_markup: str) -> float:
    """Mass to ADD to a neutral structure to compare it with this row.

    `+1.007` for `[M+H]`, `-1.007` for `[M-H]`. Defaults to `[M+H]`, which is
    what 331 of the 353 adduct-bearing rows in this corpus print — but the
    default is now a stated fallback rather than an assumption nobody checked.
    """
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

    THREE SHAPES, IN PRIORITY ORDER. A sentence naming the instrument
    (`REPORTED`, then `REPORTED_ALT`) beats a bare column, because a sentence
    says what it is measuring and a column says it only in its header. The
    column shape is read last and only for rows the sentences did not answer.
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
        hit = REPORTED.search(flat) or REPORTED_ALT.search(flat)
        if not hit:
            continue
        cid = _ROW_CID.match(flat)
        if cid:
            out.setdefault(cid.group(1), (float(hit.group(1)), _shift(raw)))
    for cid, pair in _column_masses(xml).items():
        out.setdefault(cid, pair)
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
        head_shift = {i: -PROTON if _ADDUCT_MINUS.search(
            next((c.header or "" for c in cols if c.index == i), "")) else PROTON
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
                    shift = (-PROTON if _ADDUCT_MINUS.search(cell)
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
    return None if mol is None else (ExactMolWt(mol), MolWt(mol))


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
        n = sum(1 for r in rows if ms.get(str(getattr(r, "cid", "") or "")))
        for r in rows:
            try:
                r.mass_check = (VERDICT_UNAVAILABLE
                                if ms.get(str(getattr(r, "cid", "") or ""))
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
        key = str(cid) if cid else ""
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
                reported_mh=ms.get(str(cid)), delta=round(d or 0, 2))
    if tally[VERDICT_CONTRADICTS]:
        logger.info("mass_gate: %s — %d of %d weighable structures contradict "
                    "the mass their own row prints", patent_id,
                    tally[VERDICT_CONTRADICTS],
                    tally[VERDICT_CONTRADICTS] + tally[VERDICT_AGREES])
    return tally
