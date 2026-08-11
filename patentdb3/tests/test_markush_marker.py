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

import pytest

from patentdb3.core import config
from patentdb3.sources.cid_first import _markush_cids, _resolve

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
    that keeps this fix from costing more than it saves."""
    assert _markush_cids(_xml(NAMED)) == {}


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
