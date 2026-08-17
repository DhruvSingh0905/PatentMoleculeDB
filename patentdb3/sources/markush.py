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

WHAT BLOCKS US10626094, MEASURED
---------------------------------
Its 32 runnable rows were run end to end. The scaffold half worked and the
fragment half did not, and the split is total:

    scaffold   `*C . *C . *c1nc(-c2ccccc2)nc2ccccc12`   3 pieces, 3 dummies
    after `strip_open_points`  ->  `*c1nc(-c2ccccc2)nc2ccccc12`   clean, 1 point

    fragments  attachment points per fragment:  {0: 32}
    build      fragment_has_no_attachment_point:    32

ZERO of 32 fragments carried an attachment point. Not most — all. Those
images mark the cut bond with a WAVY LINE drawn across it, and the recogniser
reads that as an ordinary bond end and invents a group there: one was
returned as a plain tert-butyl. Nothing in the output says a mark was missed.

So exactly one failure stands between this patent and 32 compounds, and it is
a recogniser limitation rather than anything this module can repair. The
build refuses these rather than joining at an arbitrary atom, because a join
made at a guessed position produces a clean-looking wrong molecule that
nothing downstream can detect.

US10125101's fragments use an asterisk instead, which the same recogniser
reads correctly — 29 of 29. The two patents differ only in house style.

WHAT IS A DEFAULT
------------------
`H`, an empty cell, or a dash. All three mean "no substituent here". Hydrogen
is implicit in every SMILES, so a default slot needs NO action at all — not a
hydrogen added, not a placeholder removed from the product. It simply is not a
variable. 32 of the 40 populated rows on US10626094 read `H` for R2, which is
why this is the route worth building first.

A GROUP WITH NO LOCANT
-----------------------
Over the eleven patents with a real substituent table the varying slots are
222 groups with no locant against 8 with one, so the group is the population
and `build_text_group` is its builder. Two questions have to be answered and
NEITHER may be guessed:

WHICH ATOM. The heading is the answer. The recogniser writes `R1` in a drawing
as a dummy of isotope 1 (`core/config.py`, the RECOGNISER note), so heading
`R2` means the atom carrying `[2*]` and no other. A heading with no number
(`X`, `RL`) names nothing on the scaffold and is refused.

WHAT STRUCTURE. The slot text is a condensed formula (`CH3`, `NH2`,
`CH(CH3)2`) — not a name. OPSIN resolves none of it: measured, 0 of 22, and
its own message says why ("a name is just a substituent"). `rdAbbreviations`
holds 2 of the 13 measured tokens and reads `NC` as an isocyanide. So the
formula is parsed here, under a grammar where every bond is SINGLE, and the
WRITTEN HYDROGEN COUNT IS THE GATE: build the skeleton, then require its
computed formula to equal the one on the page. That check is what makes the
grammar safe to run on text nobody has looked at. `CH2CH3` -> C2H5 passes;
`CN` under single bonds would be aminomethyl, C1H4N1 against a written C1N1,
and is refused — it survives only as a table entry, gated by the same check.
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

# A heading that names a NUMBERED position: `R1`, `R 2`, `R66`. The number is
# the entire correspondence between a column and an atom.
_SLOT_NUMBER = re.compile(r"^R\s*(\d+)$", re.I)

# A slot value may open with an em dash: `—CH3`. U+2014, not a hyphen, and a
# `str.strip("-")` therefore leaves it in place.
_LEAD_DASH = re.compile(r"^[\s\-‐‑‒–—―−]+")

# One item of a condensed formula: an element symbol or a bracket, each with an
# optional count. `CH(CH3)2` -> `C` `H` `(` `C` `H3` `)2`.
_FORMULA_ITEM = re.compile(r"([A-Z][a-z]?|\(|\))(\d*)")

# The elements the grammar admits. Every bond it makes is single; anything
# needing a double or triple bond fails the hydrogen count and is refused.
_FORMULA_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}

# Never a chain atom, whatever its position. `CH2Cl` is C-Cl and must not be
# read as C-Cl-<next>.
_MONOVALENT = {"F", "Cl", "Br", "I"}

# The one measured token whose bond order the written formula does not state.
# A table entry, not a grammar rule, and checked like every other.
_CONVENTIONAL = {"CN": "*C#N"}

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
BLOCK_UNKNOWN_GROUP = "group_text_not_a_condensed_formula"
BLOCK_FORMULA_MISMATCH = "written_formula_does_not_match_structure"
BLOCK_SLOT_NOT_MARKED = "slot_not_marked_on_scaffold"
BLOCK_UNMAPPED_POINT = "attachment_point_matches_no_slot"


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
    # The heading of the slot whose cell holds the picture, when a slot column
    # holds it. Without this a drawn slot is stored as `""` and is
    # indistinguishable from a blank one — and a builder that treats it as
    # blank caps that position with hydrogen instead of the drawn group.
    image_slot: str = ""

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
        slots, varying, image_slot = {}, {}, ""
        for col, name in headings.items():
            if col >= len(r):
                continue
            v = r[col].text.strip()
            slots[name] = v
            if r[col].chemistry_id:
                image_slot = name
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
                              slots=slots, varying=varying, blockers=blockers,
                              image_slot=image_slot))
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


def _formula_items(text: str):
    """`text` as nested `(symbol, count, children)`, or `None`.

    `None` means "this is not a condensed formula" — a word (`phenyl`), a
    retention time (`1.45`), a stereo prefix (`(R)-CH3`). Refusing the whole
    string is the point: a partial read of a substituent is a wrong molecule.
    """
    stack: list[list] = [[]]
    pos = 0
    while pos < len(text):
        m = _FORMULA_ITEM.match(text, pos)
        if not m:
            return None
        sym, num, pos = m.group(1), m.group(2), m.end()
        if sym == "(":
            if num:
                return None
            stack.append([])
        elif sym == ")":
            if len(stack) == 1 or not stack[-1]:
                return None
            kids = stack.pop()
            stack[-1].append(("()", int(num) if num else 1, kids))
        elif sym in _FORMULA_ELEMENTS:
            stack[-1].append((sym, int(num) if num else 1, []))
        else:
            return None
    return stack[0] if len(stack) == 1 and stack[0] else None


def _written_counts(items) -> dict:
    """The atom counts the PAGE states, hydrogen included."""
    out: dict = {}
    for sym, count, kids in items:
        if sym == "()":
            for k, v in _written_counts(kids).items():
                out[k] = out.get(k, 0) + v * count
        else:
            out[sym] = out.get(sym, 0) + count
    return out


def _mol_counts(mol) -> dict:
    """The atom counts the STRUCTURE has. The attachment point is not one."""
    out: dict = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            continue
        out[a.GetSymbol()] = out.get(a.GetSymbol(), 0) + 1
        if a.GetTotalNumHs():
            out["H"] = out.get("H", 0) + a.GetTotalNumHs()
    return out


def _grow(items, rw, anchor):
    """Hang `items` off atom `anchor` in `rw`. Every bond made is single.

    Reading rule, and the only one: an item extends the chain when it can be
    a chain atom, and branches when it cannot. A count above one means the
    copies hang off the current atom (`CF3`), and so does a halogen at any
    count, because a halogen is an end.
    """
    from rdkit import Chem

    current = anchor
    for sym, count, kids in items:
        if sym == "()":
            for _ in range(count):
                _grow(kids, rw, current)
        elif sym == "H":
            pass                       # implicit in the skeleton; counted only
        elif sym in _MONOVALENT or count > 1:
            for _ in range(count):
                idx = rw.AddAtom(Chem.Atom(sym))
                rw.AddBond(current, idx, Chem.BondType.SINGLE)
        else:
            idx = rw.AddAtom(Chem.Atom(sym))
            rw.AddBond(current, idx, Chem.BondType.SINGLE)
            current = idx


def _group_fragment(text: str, isotope: int) -> tuple[str, str]:
    """One slot value -> a one-point fragment SMILES. `(smiles, error)`.

    The returned dummy carries `isotope`, which is how the caller says WHERE
    this group goes. Nothing here knows the scaffold.
    """
    from rdkit import Chem

    items = _formula_items(text)
    if items is None:
        return "", f"{BLOCK_UNKNOWN_GROUP}: {text!r}"
    known = _CONVENTIONAL.get(text)
    if known:
        mol = Chem.MolFromSmiles(known)
    else:
        rw = Chem.RWMol()
        star = rw.AddAtom(Chem.Atom(0))
        _grow(items, rw, star)
        mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:                        # a valence the page implies
        return "", f"{text!r} does not sanitize: {e!r}"[:120]
    want, got = _written_counts(items), _mol_counts(mol)
    if want != got:
        return "", f"{BLOCK_FORMULA_MISMATCH}: {text!r} states {want}, built {got}"
    d = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(d) != 1:
        return "", f"{text!r} built {len(d)} attachment points, expected 1"
    d[0].SetIsotope(isotope)
    return Chem.MolToSmiles(mol), ""


def build_text_group(scaffold_smiles: str, row: MarkushRow,
                     fragment_smiles: str = "") -> tuple[str, str]:
    """The group-with-no-locant route. `(smiles, error)` — one is empty.

    `fragment_smiles` is the drawn substituent, needed only when the row has
    an image slot as well. Without it such a row is refused rather than
    capped with hydrogen, because capping produces a legal molecule that is
    not the compound.

    EVERY attachment point on the scaffold must be accounted for before
    anything is built: filled by a slot, or dropped because its slot is a
    default. One left over means the scaffold carries a variable position
    this table does not explain, and the row is refused.
    """
    from rdkit import Chem

    s = Chem.MolFromSmiles(scaffold_smiles) if scaffold_smiles else None
    if s is None:
        return "", "scaffold did not parse"
    if any(loc for loc, _g in row.varying.values()):
        return "", BLOCK_NO_LOCANT_MAP
    if not row.varying:
        return "", "no varying text slot"

    by_iso: dict[int, list[int]] = {}
    for a in s.GetAtoms():
        if a.GetAtomicNum() == 0:
            by_iso.setdefault(a.GetIsotope(), []).append(a.GetIdx())
    if not by_iso:
        return "", "scaffold has no attachment point"

    def _iso(head: str):
        m = _SLOT_NUMBER.match(head or "")
        return int(m.group(1)) if m else None

    fill: dict[int, str] = {}
    drop: set = set()
    for head, (_loc, text) in sorted(row.varying.items()):
        n = _iso(head)
        if n is None:
            return "", f"{BLOCK_SLOT_NOT_MARKED}: heading {head!r} has no number"
        if n not in by_iso:
            return "", f"{BLOCK_SLOT_NOT_MARKED}: {head}"
        if len(by_iso[n]) > 1:
            return "", f"scaffold marks {head} at {len(by_iso[n])} atoms"
        group = _LEAD_DASH.sub("", text).strip()
        if group.lower() in _DEFAULT_SLOT:        # `—H`: a default a dash hid
            drop.add(n)
            continue
        frag, err = _group_fragment(group, n)
        if err:
            return "", err
        fill[n] = frag

    for head, _v in row.slots.items():            # the defaults
        if head == row.image_slot:
            continue
        n = _iso(head)
        if n in by_iso and n not in fill:
            drop.add(n)

    if row.image_slot or fragment_smiles:
        n = _iso(row.image_slot)
        if n is None:                             # `RL`, or no slot column
            free = [i for i in by_iso if i not in fill and i not in drop]
            if len(free) != 1:
                return "", ("cannot tell which point the drawn substituent "
                            f"fills — {len(free)} are unclaimed")
            n = free[0]
        if n not in by_iso:
            return "", f"{BLOCK_SLOT_NOT_MARKED}: {row.image_slot}"
        if not fragment_smiles:
            return "", f"image slot {row.image_slot or '?'} has no structure"
        f = Chem.MolFromSmiles(fragment_smiles)
        if f is None:
            return "", "fragment did not parse"
        fd = [a for a in f.GetAtoms() if a.GetAtomicNum() == 0]
        if len(fd) != 1:
            return "", (f"fragment has {len(fd)} attachment points, expected 1"
                        if fd else BLOCK_FRAGMENT_NO_POINT)
        fd[0].SetIsotope(n)
        fill[n] = Chem.MolToSmiles(f)
        drop.discard(n)

    left = sorted(i for i in by_iso if i not in fill and i not in drop)
    if left:
        return "", (f"{BLOCK_UNMAPPED_POINT}: "
                    + ",".join(f"[{i}*]" for i in left))

    rw = Chem.RWMol(s)
    for idx in sorted((i for n in drop for i in by_iso[n]), reverse=True):
        rw.RemoveAtom(idx)                        # a default IS hydrogen
    core = rw.GetMol()
    try:
        Chem.SanitizeMol(core)
    except Exception as e:
        return "", f"scaffold does not sanitize: {e!r}"[:120]
    if len(Chem.GetMolFrags(core)) != 1:
        return "", BLOCK_SCAFFOLD_SPLIT
    if not fill:                                  # every slot read as default
        return Chem.MolToSmiles(core), ""

    parts = Chem.MolFromSmiles(".".join(fill[n] for n in sorted(fill)))
    if parts is None:
        return "", "assembled fragments did not parse"
    params = Chem.rdmolops.MolzipParams()
    params.label = Chem.rdmolops.MolzipLabel.Isotope
    try:
        joined = Chem.molzip(core, parts, params)
        smi = Chem.MolToSmiles(joined)
    except Exception as e:                        # a repeated label raises here
        return "", f"molzip failed: {e!r}"[:120]
    # Same check as `build_image_only`, same reason: a point left open raises
    # nothing at all.
    if "*" in smi:
        return "", "join left an unfilled attachment point"
    if Chem.MolFromSmiles(smi) is None:
        return "", "joined molecule does not sanitize"
    return smi, ""
