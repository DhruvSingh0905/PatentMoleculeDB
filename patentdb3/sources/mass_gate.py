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

Not a coverage measure either. It can only judge a compound whose row prints
an MS, and most do not: over a 10-patent bench, 81 of 2,596 resolved
structures (3.1%) carry one. A silent verdict here means "not checkable", and
it must never be read as "checked and fine".

THE TOLERANCE
--------------
The patent prints a NOMINAL integer (`561`) and this computes an EXACT
monoisotopic mass (`561.17`). The two drift apart as a molecule grows, because
every carbon adds ~3.4 mDa of defect and every hydrogen ~7.8 mDa. A flat
window therefore fails at one end or the other: 0.5 Da rejects correct large
structures, and 2.0 Da accepts a wrong small one. `tolerance()` scales, and it
is deliberately generous — this gate exists to catch a 300 Da error, not to
adjudicate an isotope.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# `MS (ESI) 561 (M+H)`, `MS (ESI+) m/z 561.2 (M+H)+`, `MS(ESI) 561`. The mass
# is the FIRST number after the ionisation mode — a `+` charge marker, `m/z`
# and a space may sit between, and nothing else observed does.
REPORTED = re.compile(r"MS\s*\(ESI[^)]*\)[^0-9]{0,12}(\d{2,4})(?:\.\d+)?", re.I)

# The row's own compound number: the first cell, digits with an optional
# letter suffix. Same shape `uspto_assays` reads, and deliberately strict —
# a row whose first cell is prose states no compound number.
_ROW_CID = re.compile(r"\s*(\d+[A-Za-z]?)\s")

PROTON = 1.00728          # M+H, the only adduct this corpus prints

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


def reported_masses(xml: str) -> dict[str, int]:
    """`{compound number -> M+H the patent prints in that compound's row}`.

    Read per `<row>` rather than per document, so a mass can only ever be
    attributed to the row that carries it. A row with no id, or no MS, is
    absent from the result — never guessed at from a neighbour.

    First occurrence wins. A compound number repeated across tables states the
    same compound, and the alternative (last wins) would silently prefer
    whichever table the parser happened to reach last.
    """
    out: dict[str, int] = {}
    for m in re.finditer(r"<row>.*?</row>", xml, re.S):
        flat = re.sub(r"<[^>]+>", " ", m.group(0))
        hit = REPORTED.search(flat)
        if not hit:
            continue
        cid = _ROW_CID.match(flat)
        if cid:
            out.setdefault(cid.group(1), int(hit.group(1)))
    return out


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


def verdict(smiles: str, reported: "int | None") -> tuple[str, float | None]:
    """`(verdict, delta)` for one structure against one reported M+H.

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
    deltas = [mass + PROTON - reported for mass in m]
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
        v, d = verdict(getattr(r, "smiles", ""), ms.get(str(cid)) if cid else None)
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
