"""Routing a substituent table row, and building the case that is solved.

Every table shape here is real. US10626094 has three text slot columns;
US10125101 has none at all, and getting that second shape wrong hid 29
buildable compounds behind a header-detection failure.
"""
import pytest

from patentdb3.core import config
from patentdb3.sources import markush as MK
from patentdb3.sources.uspto_xml import Cell

requires_xml = pytest.mark.skipif(
    not (config.XML_INPUT_DIR / "US10626094.xml").exists(),
    reason="US10626094.xml not cached")


def _row(*vals):
    """Cells from `(text, chemistry_id)` pairs or plain strings."""
    out = []
    for v in vals:
        if isinstance(v, tuple):
            out.append(Cell(v[0], 1, -1, v[1]))
        else:
            out.append(Cell(v))
    return out


HEADER = _row("Entry", "P/S*", "R1", "R2", "R3", "IC50")


# ── routing ───────────────────────────────────────────────────────────────

def test_every_text_slot_default_routes_to_image_only():
    """`H` means no substituent. Hydrogen is implicit in every SMILES, so a
    default slot needs no action at all — not even a placeholder removed."""
    rows = MK.classify([HEADER, _row("4", "CP", ("", "CHEM-US-00083"),
                                     "H", "H", "17.0")])
    assert len(rows) == 1
    assert rows[0].route == MK.ROUTE_IMAGE_ONLY
    assert rows[0].fragment_ref == "CHEM-US-00083"
    assert rows[0].varying == {}
    assert rows[0].runnable


def test_a_locant_slot_routes_differently_and_is_blocked():
    """`3-CH3` names a ring POSITION. Nothing maps that number to an atom."""
    rows = MK.classify([HEADER, _row("1 Screen hit", "CP P",
                                     ("", "CHEM-US-00080"), "3-CH3", "H", "7.12")])
    r = rows[0]
    assert r.route == MK.ROUTE_IMAGE_AND_LOCANT
    assert r.varying == {"R2": ("3", "CH3")}
    assert MK.BLOCK_NO_LOCANT_MAP in r.blockers
    assert not r.runnable
    # the compound id is the FIRST token: `1 Screen hit` is compound 1
    assert r.cid == "1"


def test_a_blank_dash_or_H_all_mean_no_substituent():
    rows = MK.classify([HEADER,
                        _row("5", "CP", ("", "CHEM-1"), "", "-", "1.0"),
                        _row("6", "CP", ("", "CHEM-2"), "H", "h", "1.0")])
    assert [r.route for r in rows] == [MK.ROUTE_IMAGE_ONLY] * 2


def test_a_table_with_no_slot_column_is_image_only_by_construction():
    """US10125101's shape: `compound | drawn substituent | analytical data`.

    No slot column can hold a non-default value, so every such row is
    buildable. Returning nothing here — which the first version did — hides
    29 real compounds behind a header that was never found.
    """
    body = [_row("11", ("", "CHEM-US-00051"), "LC: tR = 0.84 min"),
            _row("12", ("", "CHEM-US-00052"), "LC: tR = 0.93 min")]
    rows = MK.classify(body, header_rows=[_row("Ex-", "RL", "Retention time")])
    assert len(rows) == 2
    assert all(r.route == MK.ROUTE_IMAGE_ONLY and r.runnable for r in rows)


def test_rows_that_are_not_compounds_are_skipped():
    """Captions, spanning image rows and blank separators are not data."""
    rows = MK.classify([HEADER, _row("TABLE 1"), _row(""), _row("-"),
                        _row("SAR explorations of the core")])
    assert rows == []


# ── the open attachment points ────────────────────────────────────────────

def test_unused_floating_points_are_dropped_not_filled():
    """A scaffold drawn with floating attachments comes back DISCONNECTED.

    Real MolScribe output for US10626094's scaffold:
        `*C . *C . *c1nc(-c2ccccc2)nc2ccccc12`
    R2 and R3 are bonds that cross a ring in the drawing, so there is no atom
    to land on and the recogniser emits them as separate fragments. When both
    text slots are defaults those pieces are noise, not holes.
    """
    from rdkit import Chem
    m = Chem.MolFromSmiles("*C.*C.*c1nc(-c2ccccc2)nc2ccccc12")
    core, dropped = MK.strip_open_points(m, keep=1)
    assert dropped >= 2
    assert len(Chem.GetMolFrags(core)) == 1
    assert sum(1 for a in core.GetAtoms() if a.GetAtomicNum() == 0) == 1


# ── building ──────────────────────────────────────────────────────────────

# Real MolScribe output for US10125101's scaffold and compound 11's fragment.
SCAFFOLD = "*c1ccc(F)c2c1CC[C@H]2Nc1ccc([C@H]2C[C@@H]2C(=O)O)nc1"
FRAGMENT = "*c1c(C)cc(C(N)=O)cc1C"


def test_the_image_only_route_builds_the_compound():
    """Compound 11. The patent prints `MS (ESI-): m/z = 458 [M - H]-`."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    smi, err = MK.build_image_only(SCAFFOLD, FRAGMENT)
    assert not err, err
    assert "*" not in smi
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None                      # legal, sanitizes
    assert abs(Descriptors.ExactMolWt(mol) - 1.00728 - 458) < 1.5


def test_a_fragment_with_no_attachment_point_is_refused_not_guessed():
    """The wavy-cut-bond failure. MolScribe read one such fragment as a plain
    tert-butyl — no dummy atom anywhere. Joining anyway would attach at an
    arbitrary atom and produce a confident wrong molecule."""
    smi, err = MK.build_image_only(SCAFFOLD, "CC(C)(C)NCCCN1CCOCC1")
    assert smi == ""
    assert MK.BLOCK_FRAGMENT_NO_POINT in err or "attachment" in err


def test_an_unparseable_input_is_reported_not_swallowed():
    assert MK.build_image_only("not a smiles", FRAGMENT)[1] == "scaffold did not parse"
    assert MK.build_image_only(SCAFFOLD, "not a smiles")[1] == "fragment did not parse"


# ── a group with no locant ────────────────────────────────────────────────

# A scaffold as the recogniser emits one: two variable positions, each an
# isotope-tagged dummy. `R1` in the drawing becomes `[1*]`. Verified against
# rdkit directly — `[2*]` parses to a dummy whose `GetIsotope()` is 2, and
# `MolzipLabel.Isotope` joins it to the fragment carrying the same number.
TWO_POINT = "[1*]c1nc(-c2ccc([2*])cc2)nc2ccccc12"

# Every distinct value the eleven substituent tables give for a slot with no
# locant. The leading character on the last group is U+2014, an em dash.
VOCABULARY = ["CH3", "H", "NH2", "CF3",
              "—CN", "—Br", "—Cl", "—CH2CH3", "—CH3", "—CH(CH3)2",
              "—H", "—I", "—F"]


def _mk(**kw):
    kw.setdefault("cid", "1")
    kw.setdefault("route", MK.ROUTE_TEXT_ONLY)
    return MK.MarkushRow(**kw)


def _one_slot(value, heading="R2"):
    """A row carrying `value` at `heading`. The other position is a default —
    it has to be named, because a point no column explains is refused."""
    other = "R1" if heading == "R2" else "R2"
    return _mk(slots={other: "", heading: value},
               varying={heading: (None, value)})


def test_a_group_with_no_locant_carries_no_blocker():
    """`R2 = CH3` names no ring position, so nothing about it is blocked."""
    rows = MK.classify([HEADER, _row("7", "CP", "", "CH3", "H", "1.0")])
    r = rows[0]
    assert r.varying == {"R2": (None, "CH3")}
    assert r.route == MK.ROUTE_TEXT_ONLY
    assert r.runnable


def test_the_drawn_slot_is_recorded_by_name():
    """A slot holding the picture reads as an empty cell. Which heading held
    it has to be kept, or a builder caps that position with hydrogen and
    returns a legal molecule that is not the compound."""
    rows = MK.classify([HEADER, _row("8", "CP", ("", "CHEM-9"), "CH3", "H", "1.0")])
    assert rows[0].image_slot == "R1"
    assert rows[0].slots["R1"] == ""


@pytest.mark.parametrize("value", VOCABULARY)
def test_every_measured_slot_value_builds(value):
    """All 13. `H` and `—H` are defaults, so they build the bare scaffold."""
    from rdkit import Chem
    smi, err = MK.build_text_group(TWO_POINT, _one_slot(value))
    assert not err, f"{value!r}: {err}"
    assert "*" not in smi
    assert Chem.MolFromSmiles(smi) is not None


def test_the_group_lands_on_the_position_its_heading_names():
    """`R2` means the atom carrying `[2*]`, and no other. The scaffold's two
    positions are distinguishable, so a join at the wrong one is visible."""
    from rdkit import Chem
    at_2 = MK.build_text_group(TWO_POINT, _one_slot("CH3", "R2"))[0]
    at_1 = MK.build_text_group(TWO_POINT, _one_slot("CH3", "R1"))[0]
    assert at_2 == Chem.CanonSmiles("Cc1ccc(-c2ncc3ccccc3n2)cc1")
    assert at_1 == Chem.CanonSmiles("Cc1nc(-c2ccccc2)nc2ccccc12")
    assert at_1 != at_2


def test_two_slots_are_filled_in_one_join():
    from rdkit import Chem
    r = _mk(slots={"R1": "NH2", "R2": "CH3"},
            varying={"R1": (None, "NH2"), "R2": (None, "CH3")})
    smi, err = MK.build_text_group(TWO_POINT, r)
    assert not err, err
    assert smi == Chem.CanonSmiles("Cc1ccc(-c2nc(N)c3ccccc3n2)cc1")


def test_a_default_slot_leaves_hydrogen_not_a_placeholder():
    from rdkit import Chem
    smi, err = MK.build_text_group(TWO_POINT, _one_slot("—H"))
    assert not err, err
    assert smi == Chem.CanonSmiles("c1ccc(-c2ncc3ccccc3n2)cc1")


def test_CN_is_the_nitrile_and_not_what_single_bonds_would_give():
    """The one measured token whose bond order the page does not state."""
    smi, err = MK.build_text_group(TWO_POINT, _one_slot("—CN"))
    assert not err, err
    assert "C#N" in smi or "N#C" in smi


# ── what the group route refuses, and why ─────────────────────────────────

@pytest.mark.parametrize("value,reason", [
    ("phenyl", MK.BLOCK_UNKNOWN_GROUP),         # a name, not a formula
    ("Chloro", MK.BLOCK_UNKNOWN_GROUP),
    ("1.45", MK.BLOCK_UNKNOWN_GROUP),           # a retention time in a slot
    ("(R)-CH3", MK.BLOCK_UNKNOWN_GROUP),        # a stereo prefix
    ("C", MK.BLOCK_FORMULA_MISMATCH),           # states C1, a skeleton is CH3
    ("C(O)CH2NH(CH3)", MK.BLOCK_FORMULA_MISMATCH),   # needs a double bond
])
def test_a_group_the_grammar_cannot_prove_is_refused(value, reason):
    """The written hydrogen count is the gate. A formula that disagrees with
    the structure built from it is not read a second, looser way."""
    smi, err = MK.build_text_group(TWO_POINT, _one_slot(value))
    assert smi == ""
    assert reason in err


def test_a_heading_that_names_no_position_is_refused():
    r = _mk(slots={"R1": "", "X": "CH3"}, varying={"X": (None, "CH3")})
    assert MK.build_text_group(TWO_POINT, r)[1].startswith(MK.BLOCK_SLOT_NOT_MARKED)


def test_a_heading_the_scaffold_does_not_mark_is_refused():
    r = _mk(slots={"R1": "", "R3": "CH3"}, varying={"R3": (None, "CH3")})
    smi, err = MK.build_text_group(TWO_POINT, r)
    assert smi == "" and err == f"{MK.BLOCK_SLOT_NOT_MARKED}: R3"


def test_an_attachment_point_no_slot_explains_is_refused():
    """`[1*]` belongs to no column of this row. Filling only R2 would cap it
    with hydrogen, which is an answer to a question the table never asked."""
    r = _mk(slots={"R2": "CH3"}, varying={"R2": (None, "CH3")})
    smi, err = MK.build_text_group(TWO_POINT, r)
    assert smi == ""
    assert err == f"{MK.BLOCK_UNMAPPED_POINT}: [1*]"


def test_a_locant_slot_is_refused_by_this_route_too():
    r = _mk(slots={"R1": "", "R2": "3-CH3"}, varying={"R2": ("3", "CH3")})
    assert MK.build_text_group(TWO_POINT, r) == ("", MK.BLOCK_NO_LOCANT_MAP)


def test_a_drawn_slot_without_its_structure_is_refused_not_capped():
    r = _mk(slots={"R1": "", "R2": "CH3"}, varying={"R2": (None, "CH3")},
            image_slot="R1", fragment_ref="CHEM-9")
    assert MK.build_text_group(TWO_POINT, r)[0] == ""
    assert "no structure" in MK.build_text_group(TWO_POINT, r)[1]


def test_a_drawn_slot_and_a_text_slot_build_together():
    from rdkit import Chem
    r = _mk(slots={"R1": "", "R2": "CH3"}, varying={"R2": (None, "CH3")},
            image_slot="R1", fragment_ref="CHEM-9")
    smi, err = MK.build_text_group(TWO_POINT, r, "*c1ccccc1")
    assert not err, err
    assert smi == Chem.CanonSmiles("Cc1ccc(-c2nc(-c3ccccc3)c3ccccc3n2)cc1")
    # and the wavy-cut-bond fragment is refused here for the same reason it is
    # refused by the image_only route
    assert MK.build_text_group(TWO_POINT, r, "CC(C)(C)N")[1] == \
        MK.BLOCK_FRAGMENT_NO_POINT


def test_a_scaffold_that_falls_apart_when_a_default_is_dropped_is_refused():
    """`*C . *c1ccccc1` — the recogniser's shape when a bond crosses a ring.
    Dropping the default's dummy leaves a stray carbon, not a molecule."""
    r = _one_slot("CH3")
    assert MK.build_text_group("[1*]C.[2*]c1ccccc1", r)[1] == MK.BLOCK_SCAFFOLD_SPLIT


def test_the_group_route_does_not_answer_the_image_only_question():
    r = _mk(slots={"R1": "", "R2": "H"}, varying={})
    assert MK.build_text_group(TWO_POINT, r) == ("", "no varying text slot")


@requires_xml
def test_the_real_tables_route_as_measured():
    """The counts this module was built against, read from the cached XML."""
    from patentdb3.sources.uspto_xml import parse_tables
    xml = (config.XML_INPUT_DIR / "US10626094.xml").read_text(errors="replace")
    t = {x.table_id: x for x in parse_tables(xml)}["TABLE-US-00002"]
    rows = MK.classify(t.body_rows, t.header_rows)
    runnable = [r for r in rows if r.runnable]
    assert len(rows) == 40
    assert len(runnable) == 32                  # 32 of 40 need no locant
    assert all(r.route == MK.ROUTE_IMAGE_ONLY for r in runnable)
