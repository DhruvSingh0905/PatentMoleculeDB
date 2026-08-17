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
