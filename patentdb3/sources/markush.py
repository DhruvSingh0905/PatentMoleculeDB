"""Classify a substituent table row, then send it to the routine that fits.

WHY CLASSIFY BEFORE ENUMERATING
--------------------------------
A Markush table is not one problem. It is several, mixed together in the same
88 rows, and they need different machinery:

    Entry | R1 (image)      | R2      | R3 | IC50
    1     | CHEM-US-00080   | 3-CH3   | H  | 7.12      <- image + a locant
    4     | CHEM-US-00083   | H       | H  | 17.0      <- image only
    ...

Row 4 needs one picture read and one bond made. Row 1 needs that PLUS a rule
for turning "position 3 of the phenyl ring" into an atom, which no library
does and which we have not written. Treating them as one population means the
easy 32 rows wait on the hard 6, and a single failure mode reads as "Markush
does not work".

So intake tags every row with the route it needs and with anything KNOWN to
block it. A route with no blocker runs. A route with a blocker is recorded and
skipped — visibly, by name, so the size of each remaining problem is a number
rather than an impression.

THE ROUTES, AND WHY THESE
--------------------------
`image_only`         every text slot is a default (H or blank). One picture to
                     read, one bond to make. The whole answer is in the
                     fragment image. MEASURED: 29 of 29 on US10125101, every
                     molecule legal and every mass matching the patent's own.

`image_and_locant`   as above, plus a text slot like `3-CH3` that names a ring
                     POSITION. The position exists only as a number; nothing
                     maps it to an atom. Blocked until that is written.

`text_only`          no fragment picture, substituents given entirely as text.

`nothing_varies`     no image, no non-default text. Nothing to enumerate.

WHAT IS A DEFAULT
------------------
`H`, an empty cell, or a dash. All three mean "no substituent here". Hydrogen
is implicit in every SMILES, so a default slot needs NO action at all — not a
hydrogen added, not a placeholder removed from the product. It simply is not a
variable. 32 of the 40 populated rows on US10626094 read `H` for R2, which is
why this is the route worth building first.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# A cell that means "no substituent here".
_DEFAULT_SLOT = {"", "-", "h", "—", "–", "none", "na"}

# A column heading that names a variable position: R1, R2, R', RL, X, Y, Z.
_SLOT_HEADING = re.compile(r"^(R\s*[0-9A-Za-z']{0,3}|[XYZQ])$", re.I)

# A text slot that names a ring POSITION before the group: `3-CH3`, `4-O(i-Pr)`,
# `2-F`. The leading integer is an IUPAC locant, and resolving it to an atom is
# the piece with no library behind it.
_LOCANT_VALUE = re.compile(r"^(\d+)\s*-\s*(\S.*)$")

ROUTE_IMAGE_ONLY = "image_only"
ROUTE_IMAGE_AND_LOCANT = "image_and_locant"
ROUTE_TEXT_ONLY = "text_only"
ROUTE_NOTHING_VARIES = "nothing_varies"

# Blockers. A route says what the row NEEDS; a blocker says what stops it now.
# Kept separate because they move independently: `image_only` will not change,
# while `wavy_cut_bond` disappears the day a recogniser reads that mark.
BLOCK_NO_LOCANT_MAP = "no_locant_to_atom_map"
BLOCK_SCAFFOLD_SPLIT = "scaffold_disconnected"
BLOCK_FRAGMENT_NO_POINT = "fragment_has_no_attachment_point"


@dataclass
class MarkushRow:
    """One compound in a substituent table, tagged with how to build it."""
    cid: str
    route: str
    fragment_ref: str = ""              # `<chemistry>` id of the R-group image
    # {slot heading -> cell text}, defaults INCLUDED so a reader can see that
    # the patent said `H` rather than that we failed to read the cell.
    slots: dict = field(default_factory=dict)
    # Only the slots that actually vary: {heading -> (locant, group text)}.
    varying: dict = field(default_factory=dict)
    blockers: list = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return not self.blockers


def _slot_columns(rows: list) -> tuple[int, dict]:
    """`(header_index, {column -> heading})` for the R-group columns.

    The header is found by CONTENT, not position: a substituent table can open
    with a title row, a caption row and a spanning image row before the real
    headings, and their number varies by document.
    """
    for i, r in enumerate(rows):
        heads = {j: c.text.strip() for j, c in enumerate(r)
                 if _SLOT_HEADING.match(c.text.strip())}
        if heads:
            return i, heads
    return -1, {}


def classify(body_rows: list, header_rows: list | None = None) -> list[MarkushRow]:
    """Tag every data row of one parsed substituent table.

    `body_rows` / `header_rows` are `uspto_xml.Table`'s. The image reference
    comes from `Cell.chemistry_id`, which is why that field exists — before
    it, an image cell parsed to an empty string and a row carrying a drawn
    substituent was indistinguishable from one with a blank column.

    NOT EVERY SUBSTITUENT TABLE HAS TEXT SLOTS. US10626094 has three columns
    of them (`R1 | R2 | R3`). US10125101 has none at all: its rows are
    `compound number | drawn substituent | analytical data`, and the single
    variable position is named only in the scaffold image. A table with no
    slot column cannot have a non-default slot, so every row of it is
    `image_only` by construction — and returning nothing for such a table,
    which the first version of this function did, hides 29 buildable
    compounds behind a header-detection failure.
    """
    hdr, headings = _slot_columns(body_rows)
    if hdr < 0 and header_rows:
        _h, headings = _slot_columns(header_rows)
        hdr = -1                      # headings live above; all body is data
    out: list[MarkushRow] = []
    for r in body_rows[hdr + 1:]:
        if not r:
            continue
        cid = r[0].text.strip()
        if not cid or cid in _DEFAULT_SLOT or not re.match(r"^\d", cid):
            continue
        cid = cid.split()[0]                     # `1 Screen hit` -> `1`
        frag = next((c.chemistry_id for c in r if c.chemistry_id), "")
        slots, varying = {}, {}
        for col, name in headings.items():
            if col >= len(r):
                continue
            v = r[col].text.strip()
            slots[name] = v
            if v.lower() in _DEFAULT_SLOT or r[col].chemistry_id:
                continue                          # default, or the image slot
            m = _LOCANT_VALUE.match(v)
            varying[name] = (m.group(1), m.group(2)) if m else (None, v)

        if frag and not varying:
            route = ROUTE_IMAGE_ONLY
        elif frag:
            route = ROUTE_IMAGE_AND_LOCANT
        elif varying:
            route = ROUTE_TEXT_ONLY
        else:
            route = ROUTE_NOTHING_VARIES

        blockers = []
        if any(loc for loc, _g in varying.values()):
            blockers.append(BLOCK_NO_LOCANT_MAP)
        out.append(MarkushRow(cid=cid, route=route, fragment_ref=frag,
                              slots=slots, varying=varying, blockers=blockers))
    return out


def strip_open_points(mol, keep: int = 1):
    """Drop the scaffold's UNUSED attachment points. Returns `(mol, dropped)`.

    A scaffold drawn with three variable positions comes back from the
    recogniser with three dummy atoms — and on a floating attachment, as a
    DISCONNECTED fragment (`*C . *C . *c1nc(...)`), because the bond in the
    drawing crosses a ring instead of landing on an atom.

    When every text slot is a default there is nothing to attach at those
    positions, so they are not holes to fill: they are noise. Removing them
    leaves the plain scaffold, which is what the compound actually is.

    `keep` is how many attachment points the row still needs — one, for the
    image fragment. The LARGEST connected piece is kept, because the scaffold
    is the molecule and the strays are single atoms hanging off a dummy.
    """
    from rdkit import Chem

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    dropped = len(frags) - 1
    core = max(frags, key=lambda m: m.GetNumHeavyAtoms()) if frags else mol
    dummies = [a.GetIdx() for a in core.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) <= keep:
        return core, dropped
    # Keep the FIRST dummy in atom order and delete the rest. Order is the
    # recogniser's own, which is stable for one image but carries no meaning —
    # so this is only safe when the extras are known to be unused, which is
    # exactly the `image_only` route's precondition.
    rw = Chem.RWMol(core)
    for idx in sorted(dummies[keep:], reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol(), dropped + (len(dummies) - keep)


def build_image_only(scaffold_smiles: str, fragment_smiles: str):
    """The `image_only` route. `(smiles, error)` — one of the two is empty.

    Every text slot is a default, so the whole compound is the scaffold plus
    the one drawn substituent. No locant is resolved and none is needed.
    """
    from rdkit import Chem

    s = Chem.MolFromSmiles(scaffold_smiles) if scaffold_smiles else None
    f = Chem.MolFromSmiles(fragment_smiles) if fragment_smiles else None
    if s is None:
        return "", "scaffold did not parse"
    if f is None:
        return "", "fragment did not parse"
    s, _ = strip_open_points(s, keep=1)
    sd = [a for a in s.GetAtoms() if a.GetAtomicNum() == 0]
    fd = [a for a in f.GetAtoms() if a.GetAtomicNum() == 0]
    if not sd:
        return "", "scaffold has no attachment point"
    if len(fd) != 1:
        return "", (f"fragment has {len(fd)} attachment points, expected 1"
                    if fd else BLOCK_FRAGMENT_NO_POINT)
    for a in sd[:1] + fd:
        a.SetAtomMapNum(1)
    try:
        joined = Chem.molzip(s, f, Chem.rdmolops.MolzipParams())
        smi = Chem.MolToSmiles(joined)
    except Exception as e:
        return "", f"molzip failed: {e!r}"[:120]
    # A MISSING FRAGMENT LEAVES A DUMMY AND RAISES NOTHING — verified against
    # rdkit directly. The product must be checked, not assumed.
    if "*" in smi:
        return "", "join left an unfilled attachment point"
    if Chem.MolFromSmiles(smi) is None:
        return "", "joined molecule does not sanitize"
    return smi, ""
