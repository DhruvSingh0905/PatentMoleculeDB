"""A substituent cell is not the compound's structure.

`_drawing_refs` finds a `<chemistry>` in a compound's own table row and the
drawn marker concludes the compound is drawn. In a Markush substituent table
that conclusion is FALSE, and it was shipping as a positive claim in the
artifact until `_markush_cids` was added.

The evidence these tests encode, from US9718825 TABLE-US-00001:

    cid | Ar (text)               | R1 (text) | <chemistry> | Synthesis | Yield
    8   | 5-chloro-2-fluoro-phenyl| H         | CHEM-US-00022 | 2       | 2
    9   | 2,3-dichloro-phenyl     | H         | CHEM-US-00023 | 2       | 49

Those two chemistry elements have DIFFERENT ids and render to MD5-IDENTICAL
images — a morpholine on a wavy attachment bond — because they are the value of
one substituent column (`-Z-R3`, whose own header is a graphic), not two
molecules. The patent states the convention itself: "the line crossed with the
symbol represents the free bond via which the group -Z-R3 is bonded to the
carbon".

Measured when this landed: 644 compounds over 2 of 137 patents (US9718825 593,
US10626094 51), 2.8% of everything marked drawn corpus-wide.

A table that DOES carry a Name column is excluded — the patent enumerated
those itself and `table_names` already reads the name, so suppressing them
would lose real structures.
"""
from __future__ import annotations

import re

import pytest

from patentdb3.core import config
from patentdb3.sources.cid_first import _drawn_table_ids, _markush_cids, _resolve
from patentdb3.sources.uspto_assays import build_columns, extract_from_patent
from patentdb3.sources.uspto_xml import assemble_blocks, parse_tables

PID = "US9718825"          # 593 substituent-table compounds
NAMED = "US10376513"       # substituent tables WITH a Name column
PLAIN = "US10730877"       # drawn compounds, no substituent tables


def _xml(pid: str) -> str:
    p = config.XML_INPUT_DIR / f"{pid}.xml"
    if not p.exists():
        pytest.skip(f"{pid}.xml not cached")
    return p.read_text(errors="replace")


def test_substituent_table_cids_are_identified():
    mk = _markush_cids(_xml(PID))
    assert mk, "US9718825's substituent tables produced no cids"
    assert all(t.startswith("TABLE-US-") for t in mk.values())


def test_a_table_with_a_name_column_is_not_treated_as_markush():
    """`cid | Name | R2 | R4` was already enumerated BY THE PATENT. Suppressing
    those would throw away structures `table_names` reads today — the guard
    that keeps this fix from costing more than it saves.

    THIS ASSERTED `== {}` AND THE PREMISE WAS STALE. US10376513 holds Name-column
    substituent tables AND a crystallography table (TABLE-US-00007: `x | y | z |
    U (eq)` over `H(21A) | 11662 | 5142 | 2184 | 33`), which has no Name column
    and which `build_columns` reads as a compound id with two substituent
    columns. So `_markush_cids` returned 28 atom labels from it.

    That surfaced only when the header assembly was fixed: those coordinate
    rows used to be promoted INTO the header, and the resulting garbage stopped
    the column classifier cold. A correct header exposed a misclassification
    that had always been there.

    The 28 are gone now — that table draws nothing, so the drawing gate skips
    it; see `test_a_table_that_draws_nothing_is_never_claimed`. The column
    misclassification underneath is NOT fixed and still belongs to
    `build_columns`, so this test keeps asserting the two things the marker
    actually depends on rather than a count: no NAME-COLUMN table is claimed,
    and nothing the assay reader measured is wrongly marked.
    """
    xml = _xml(NAMED)
    mk = _markush_cids(xml)

    # 1. no cid from a Name-column table may appear
    named_tables = set()
    for t in assemble_blocks(parse_tables(xml)):
        try:
            cols = build_columns(t)
        except Exception:
            continue
        if any(re.search(r"\bname\b", c.header or "", re.I) for c in cols):
            named_tables.add(t.table_id)
    assert named_tables, "fixture no longer has a Name-column table"
    assert not (set(mk.values()) & named_tables), (
        f"a Name-column table was claimed as markush: "
        f"{set(mk.values()) & named_tables}")

    # 2. THE ONE THAT REACHES THE ARTIFACT: nothing measured is marked
    measured = {r.cid for r in extract_from_patent(xml) if r.cid}
    assert not (set(mk) & measured), (
        f"{len(set(mk) & measured)} measured compound(s) wrongly marked markush")
    _, st = _resolve(xml, NAMED)
    assert st.markush_marked == 0


def test_a_markush_compound_never_claims_a_drawing():
    """THE INVARIANT. `drawn_ref` asserts "this compound's structure is that
    picture". For a substituent cell that is a false statement, so the field
    stays empty and the row carries its reason instead."""
    out, st = _resolve(_xml(PID), PID)
    mk = [n for n in out if n.markush_reason.startswith("substituent_table:")]
    assert mk, "no markush markers emitted for US9718825"
    assert all(not n.drawn_ref for n in mk)
    assert all(not n.drawn_url for n in mk)
    assert all(n.markush for n in mk)
    # and it asserts no structure either — this is a recorded gap, not a result
    assert all(not n.smiles and not n.inchikey for n in mk)


def test_the_compound_is_still_emitted_rather_than_dropped():
    """Suppressing the false claim must not make the compound vanish. A silent
    drop would put these back in the same undifferentiated blank the marker
    exists to prevent — see `Stats.markush_marked`."""
    out, st = _resolve(_xml(PID), PID)
    assert st.markush_marked > 0
    assert st.markush_marked == sum(
        1 for n in out if n.markush_reason.startswith("substituent_table:"))
    assert st.markush_marked + st.drawn_marked <= st.resolved


def test_a_patent_with_no_substituent_tables_is_untouched():
    """The fix is targeted. US10730877's drawn compounds are real drawn
    compounds and must keep their refs."""
    assert _markush_cids(_xml(PLAIN)) == {}
    out, st = _resolve(_xml(PLAIN), PLAIN)
    assert st.markush_marked == 0
    assert st.drawn_marked > 0


# ── the drawing gate ───────────────────────────────────────────────────────
#
# `_markush_cids` exists to overturn `_drawing_refs`, and `_drawing_refs` only
# claims anything about a row holding a `<chemistry>`. A table that draws
# nothing makes no claim to overturn, so flagging its rows only costs: the
# compound loses its InChIKey and `images.emit` drops it from the recognition
# work list.
#
# Census over the 137 cached patents, `_markush_cids` rows per table:
#
#     before   18 tables   1,241 rows
#     after    13 tables   1,152 rows
#
# The 89 that went were five tables with ZERO `<chemistry>` between them —
# three of them not chemistry at all, two of them real substituent tables that
# simply print no picture. The 13 that stayed all draw. See `NO_DRAW` below.

# (patent, table id, rows it used to contribute) — the full 89.
NO_DRAW = [
    ("US10376513", "TABLE-US-00007", 28),   # `x | y | z | U (eq)`
    ("US11649247", "TABLE-US-00011", 37),   # `ATOM | X | Y | Z`
    ("US12011444", "TABLE-US-00001", 2),    # HPLC gradient program
    ("US9718825", "TABLE-US-00005", 6),     # real substituents, no picture
    ("US9718825", "TABLE-US-00006", 16),    # the same
]


@pytest.mark.parametrize("pid,tid,was", NO_DRAW,
                         ids=[f"{p}-{t[-5:]}" for p, t, _ in NO_DRAW])
def test_a_table_that_draws_nothing_is_never_claimed(pid, tid, was):
    """Each of the 89. The premise is asserted first — if a fixture ever gains
    a drawing, this test must fail loudly rather than pass for a new reason."""
    xml = _xml(pid)
    block = re.search(rf'<tables\b[^>]*id="{tid}"[^>]*>(.*?)</tables>', xml, re.S)
    assert block, f"{pid} no longer holds {tid}"
    assert "<chemistry" not in block.group(1), (
        f"{pid} {tid} now draws something — this fixture no longer tests the gate")
    assert tid not in set(_markush_cids(xml).values()), (
        f"{pid} {tid} is claimed as a substituent table again ({was} rows)")


@pytest.mark.parametrize("pid", sorted({p for p, _, _ in NO_DRAW} | {PID, NAMED}))
def test_every_claimed_table_holds_a_drawing(pid):
    """The gate, stated as the invariant rather than as a list. A claimed table
    must contain the `<chemistry>` whose reading the claim overturns."""
    xml = _xml(pid)
    drawn = _drawn_table_ids(xml)
    claimed = set(_markush_cids(xml).values())
    assert not (claimed - drawn), f"claimed but draws nothing: {claimed - drawn}"


def test_the_real_substituent_tables_are_all_still_claimed():
    """THE OTHER DIRECTION, and the one that costs structures if it breaks.
    US9718825 is 5 of the 13 surviving tables and 639 of the 1,152 rows."""
    mk = _markush_cids(_xml(PID))
    per: dict[str, int] = {}
    for tid in mk.values():
        per[tid] = per.get(tid, 0) + 1
    assert per == {
        "TABLE-US-00001": 429, "TABLE-US-00002": 10, "TABLE-US-00003": 81,
        "TABLE-US-00007": 116, "TABLE-US-00012": 3,
    }, per


# ── the gate on its own, with the real row shapes ──────────────────────────

def _tbl(rows: str, *, cols: int, header: str, table_id: str = "TABLE-US-00001") -> str:
    return (f'<tables id="{table_id}"><tgroup cols="{cols}">'
            f"<thead><row>{header}</row></thead>"
            f"<tbody>{rows}</tbody></tgroup></tables>")


def _cells(*vals: str) -> str:
    return "".join(f"<entry>{v}</entry>" for v in vals)


# US11649247 TABLE-US-00011, verbatim including the Unicode minus.
COORDS = _cells("ATOM", "X", "Y", "Z"), [
    ("O2", "48.364", "1.353", "−2.648"),
    ("C7", "47.181", "0.982", "−3.310"),
    ("N1", "46.055", "1.628", "−2.925"),
]


def test_a_coordinate_table_that_draws_nothing_is_not_markush():
    """`ATOM | X | Y | Z` passes the column test — `_HEADER_SUBST` matches a
    bare X/Y/Z — and it must still not be claimed, because it draws nothing."""
    hdr, rows = COORDS
    xml = _tbl("".join(f"<row>{_cells(*r)}</row>" for r in rows),
               cols=4, header=hdr)
    assert _markush_cids(xml) == {}


def test_the_same_table_IS_markush_once_it_draws():
    """THE REGRESSION GUARD FOR THE FILTER THAT WAS ALREADY REJECTED. A test on
    "the substituent values are only numbers" would throw this table out too,
    and this table is exactly the shape `_markush_cids` must keep: a shared
    picture beside per-row values. The gate is about the drawing, never the
    values — so numeric substituents, a Unicode minus and an e.s.d. form all
    stay claimed."""
    hdr, rows = COORDS
    body = "".join(
        f"<row>{_cells(*r)}"
        f'<entry><chemistry id="CHEM-US-0000{i}"/></entry></row>'
        for i, r in enumerate(rows))
    xml = _tbl(body, cols=5, header=hdr + _cells("Structure"))
    assert set(_markush_cids(xml)) == {"O2", "C7", "N1"}


def test_esd_and_unicode_minus_survive_the_gate():
    """`5640(60)` and `−1277` are the two forms that killed the numeric
    filter. They are values, and the gate never reads values."""
    xml = _tbl(
        "<row>" + _cells("C12", "5640(60)", "−1277", "33")
        + '<entry><chemistry id="CHEM-US-00001"/></entry></row>',
        cols=5, header=_cells("ATOM", "X", "Y", "Z", "Structure"))
    assert set(_markush_cids(xml)) == {"C12"}


def test_a_real_substituent_table_with_no_picture_is_not_claimed():
    """US9718825 TABLE-US-00005, verbatim. It IS substituent data — and there
    is no drawing in it to be mistaken for the molecule, so the claim would
    buy nothing and would cost the row its place in the image work list."""
    xml = _tbl(
        "".join(f"<row>{_cells(*r)}</row>" for r in (
            ("554", "5-chloro-2-cyano-phenyl", "425.3 (M + H), CP"),
            ("555", "5-chloro-2,4-difluoro-phenyl", "436.1 (M + H), CP"),
            ("556", "2,5-dichloro-phenyl", "434.1 (M + H), CP"))),
        cols=3, header=_cells("Example no.", "Ar", "MS (m/e)"))
    assert _markush_cids(xml) == {}


def test_an_hplc_gradient_is_not_claimed():
    """US12011444 TABLE-US-00001. `x` is the unknown in `100-x`, not an axis
    and not a substituent — three columns of it head the table."""
    xml = _tbl(
        "<row>" + _cells("t (min)", "0", "0.01", "4.0", "6.0") + "</row>"
        + "<row>" + _cells("Eluent B (%)", "100-x", "100-x", "5", "5") + "</row>",
        cols=5, header=_cells("Eluent A (%)", "x", "x", "95", "95"))
    assert _markush_cids(xml) == {}


def test_the_gate_reads_the_block_not_the_row():
    """A substituent table prints the fragment once and shares it down the
    column, so most of its rows hold no `<chemistry>` of their own —
    US9670157 TABLE-US-00006 flags 20 rows over 9 drawings. One drawing
    anywhere in the block admits every row in it."""
    xml = _tbl(
        '<row>' + _cells("1", "phenyl") + '<entry><chemistry id="C1"/></entry></row>'
        + "<row>" + _cells("2", "pyridyl") + "<entry/></row>"
        + "<row>" + _cells("3", "thienyl") + "<entry/></row>",
        cols=3, header=_cells("Example", "Ar", "Structure"))
    assert _drawn_table_ids(xml) == {"TABLE-US-00001"}
    assert set(_markush_cids(xml)) == {"1", "2", "3"}


def test_drawn_table_ids_keys_on_the_block_id():
    """It must return the id `Table.table_id` carries, so the two can meet.
    A block with no drawing is absent, not present-and-empty."""
    drawing = '<entry><chemistry id="CHEM-US-00001"/></entry>'
    xml = (_tbl("<row>" + _cells("1", "phenyl") + drawing + "</row>",
                cols=3, header=_cells("Example", "Ar", "Structure"),
                table_id="TABLE-US-00004")
           + _tbl("<row>" + _cells("2", "pyridyl") + "</row>",
                  cols=2, header=_cells("Example", "Ar"),
                  table_id="TABLE-US-00009"))
    assert _drawn_table_ids(xml) == {"TABLE-US-00004"}
    ids = {t.table_id for t in assemble_blocks(parse_tables(xml))}
    assert _drawn_table_ids(xml) <= ids
