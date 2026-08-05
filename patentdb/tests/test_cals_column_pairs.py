"""Repeating `(id, value)` column groups in one CALS row.

US9718790 TABLE-US-00569 through TABLE-US-00580 lay out THREE
`(Compound No., P2X3 IC50 (μM))` pairs per row — the compact-table idiom a
patent uses to pour one long sorted list into N columns instead of printing a
page of two-column rows. The reader read each row as ONE compound with three
values:

    row:  I-0268  0.861  I-0943  0.061  I-1607  0.035
    read: I-268 = {0.861, 0.061, 0.035}
    true: I-268 = 0.861,  I-943 = 0.061,  I-1607 = 0.035

Two harms, and the second is the larger one. The value is wrong — the scorer
takes the median, 0.061, against BindingDB's 861 nM, which is the `wrong_scale`
line `repair/value_check.py` printed for this patent. And I-943 and I-1607 get
no record at all: they are silently swallowed as values of their neighbour.

The tests below are in two halves, and the second half is the more important
one. A false positive here SPLITS a correct row into fabricated compounds,
which is strictly worse than the under-read it replaces — a missing assay is
recoverable, an invented compound is not. So every ordinary layout that has
ever been near this code is pinned as MUST-NOT-SPLIT.
"""
import pathlib

import pytest

from patentdb.sources import uspto_assays as A
from patentdb.sources.uspto_xml import Cell, Table

_XML_DIR = pathlib.Path(__file__).resolve().parents[2] / "output_v2" / "uspto_xml"


def _t(n_cols, header_rows, body_rows, table_id="T1"):
    def mk(rows):
        return [[c if isinstance(c, Cell) else Cell(str(c)) for c in r] for r in rows]
    return Table(table_id=table_id, n_cols=n_cols,
                 header_rows=mk(header_rows), body_rows=mk(body_rows))


# The exact header US9718790 publishes, as two stacked `thead` rows that
# `merge_header` joins column-wise into "Compound No." / "P2X3 IC50 (μM)".
_US9718790_HEADER = [
    ["Compound", "P2X3", "Compound", "P2X3", "Compound", "P2X3"],
    ["No.", "IC50 (μM)", "No.", "IC50 (μM)", "No.", "IC50 (μM)"],
]

# Real rows from TABLE-US-00569, verbatim from the grant XML.
_US9718790_ROWS = [
    ["I-0020", "0.384", "I-0897", "0.025", "I-1555", "0.016"],
    ["I-0258", "0.683", "I-0942", "0.022", "I-1606", "0.015"],
    ["I-0268", "0.861", "I-0943", "0.061", "I-1607", "0.035"],
    ["I-0269", "0.415", "I-0945", "0.173", "I-1608", "0.100"],
]


# ── the defect ────────────────────────────────────────────────────

def test_three_pairs_per_row_become_three_compounds():
    """Each `(id, value)` pair is its own compound, not a value of the first."""
    t = _t(6, _US9718790_HEADER, _US9718790_ROWS)
    by_cid = {}
    for r in A.extract_from_tables([t]):
        by_cid.setdefault(r.cid, []).append(r.value_numeric)

    # The compounds swallowed as values of their neighbour.
    assert by_cid["I-943"] == [0.061]
    assert by_cid["I-1607"] == [0.035]
    # ...and the one that carried their values now carries only its own.
    assert by_cid["I-268"] == [0.861]
    assert len(by_cid) == len(_US9718790_ROWS) * 3


def test_every_group_keeps_the_assay_name_and_unit_of_its_own_column():
    t = _t(6, _US9718790_HEADER, _US9718790_ROWS)
    recs = A.extract_from_tables([t])
    assert recs, "no records at all"
    assert {r.unit for r in recs} == {"uM"}
    assert all("P2X3" in r.assay_name for r in recs)


def test_two_pairs_per_row_us11136320_shape():
    """US11136320 TABLE-US-00011: two pairs, and a trailing reference compound.

    The last rows spell "Reference compound (roblitinib)" down the second id
    column across three XML rows. Those cells are not compound ids and must not
    become records — but they must not veto the split either, which is what an
    id-purity floor measured over ALL non-empty cells did (11 of 14 = 0.79).
    """
    t = _t(4, [["Compound", "FGFR4 (IC50 nM)", "Compound", "FGFR4 (IC50 nM)"]],
           [["Compound 1", "1.5", "Compound 59", "2.1"],
            ["Compound 4", "1.4", "Compound 60", "3.0"],
            ["Compound 5", "2.0", "Compound 62", "1.7"],
            ["Compound 56", "1.6", "Reference ", "2.9"],
            ["", "", "compound", ""],
            ["", "", "(roblitinib)", ""]])
    by_cid = {r.cid: r.value_numeric for r in A.extract_from_tables([t])}
    assert by_cid == {"1": 1.5, "4": 1.4, "5": 2.0, "56": 1.6,
                      "59": 2.1, "60": 3.0, "62": 1.7}


@pytest.mark.skipif(not (_XML_DIR / "US9718790.xml").exists(),
                    reason="cached grant XML not present")
def test_us9718790_real_xml_block_yields_one_compound_per_pair():
    from patentdb.sources.uspto_xml import parse_tables

    xml = (_XML_DIR / "US9718790.xml").read_text(errors="replace")
    tables = A._best_per_block(parse_tables(xml))
    t = next(t for t in tables if t.table_id == "TABLE-US-00569")
    recs = A.extract_from_tables([t])
    by_cid = {}
    for r in recs:
        by_cid.setdefault(r.cid, []).append(r.value_numeric)
    # 55 data rows x 3 pairs = 165 id cells, none blank — but only 164 distinct
    # compounds, because the PATENT prints I-0949 twice, with different values:
    #   `... I-0256 0.867 I-0949 0.008 I-1604 ...`
    #   `... I-0273 0.038 I-0949 0.021 I-1612 ...`
    # Verified by flattening the raw XML block, not read off the parsed view.
    # Both readings are kept: the duplicate is the patent's, and picking one is
    # a judgement this module has no basis to make.
    assert sum(len(v) for v in by_cid.values()) == 165
    assert len(by_cid) == 164
    assert sorted(by_cid["I-949"]) == [0.008, 0.021]
    assert by_cid["I-268"] == [0.861]
    assert by_cid["I-943"] == [0.061]
    assert by_cid["I-1607"] == [0.035]


# ── what must NEVER be split ──────────────────────────────────────

def test_ordinary_single_group_table_is_untouched():
    """The common case: one id column, several DIFFERENT assay columns.

    Nothing about this row is a repeating group, and the guarantee this test
    pins is that the pair reader cannot reach it — one compound, three values.
    """
    t = _t(4, [["Cpd No.", "hERG IC50 (nM)", "CYP3A4 IC50 (nM)", "Ki (nM)"]],
           [["I-0020", "12.0", "34.0", "5.6"],
            ["I-0021", "13.0", "35.0", "5.7"]])
    by_cid = {}
    for r in A.extract_from_tables([t]):
        by_cid.setdefault(r.cid, []).append(r.value_numeric)
    assert set(by_cid) == {"I-20", "I-21"}
    assert sorted(by_cid["I-20"]) == [5.6, 12.0, 34.0]


def test_repeated_assay_header_without_a_second_id_column_is_untouched():
    """One id, the SAME assay measured twice. Two values, one compound.

    The header repeats and the values repeat, so header-shape alone would call
    this a group. It is not: there is no second id column to split on.
    """
    t = _t(3, [["Cpd No.", "P2X3 IC50 (μM)", "P2X3 IC50 (μM)"]],
           [["I-0020", "0.384", "0.377"]])
    recs = A.extract_from_tables([t])
    assert {r.cid for r in recs} == {"I-20"}
    assert sorted(r.value_numeric for r in recs) == [0.377, 0.384]


def test_second_id_column_holding_non_ids_does_not_split():
    """The header tiles but the body does not agree — trust the body.

    A header-merge artifact or a mis-inherited header can make two columns read
    the same when the second holds no ids at all. Splitting there fabricates
    compounds out of prose, so the body has the last word.
    """
    t = _t(4, [["Compound", "IC50 (nM)", "Compound", "IC50 (nM)"]],
           [["1", "12.0", "not tested", "34.0"],
            ["2", "13.0", "see above", "35.0"],
            ["3", "14.0", "n/a", "36.0"]])
    by_cid = {}
    for r in A.extract_from_tables([t]):
        by_cid.setdefault(r.cid, []).append(r.value_numeric)
    assert set(by_cid) == {"1", "2", "3"}
    assert sorted(by_cid["1"]) == [12.0, 34.0]


def test_prior_art_comparison_table_does_not_split():
    """US10125101 TABLE-US-00004/5 pairs this patent's examples against another's.

    `Example in this invention | ... | Example in WO 2013/178575 | ...` is a
    repeating group by SHAPE, and splitting it would file WO 2013/178575's
    example numbers as this patent's compounds — the second column's `17` is a
    different document's compound 17. The headers are not identical, and that
    is exactly the signal that says so. Requiring an EXACT header repeat is
    what keeps this table whole.
    """
    t = _t(4, [["Example in this invention", "IC50 [μM]",
                "Example in WO 2013/178575", "IC50 [μM]"]],
           [["2", "20", "17", "6"],
            ["3", "21", "18", "7"],
            ["4", "22", "19", "8"]])
    cids = {r.cid for r in A.extract_from_tables([t])}
    assert cids == {"2", "3", "4"}


def test_id_families_must_match_across_groups():
    """`I-0020` and `SM A-1-35` are not the same kind of label.

    US9493446 heads intermediate columns `SM A-1` / `SM B-3` beside `Ex. No.`;
    those cells are starting materials, not compounds under test. Here the
    header is forced to repeat exactly so ONLY the family check can decline it.
    """
    t = _t(4, [["Compound", "IC50 (nM)", "Compound", "IC50 (nM)"]],
           [["I-0020", "12.0", "A-1-35", "34.0"],
            ["I-0021", "13.0", "A-1-36", "35.0"],
            ["I-0022", "14.0", "A-1-37", "36.0"]])
    cids = {r.cid for r in A.extract_from_tables([t])}
    assert cids == {"I-20", "I-21", "I-22"}


def test_too_few_rows_to_infer_a_layout():
    """One row is not evidence of a repeating layout."""
    t = _t(4, [["Compound", "IC50 (nM)", "Compound", "IC50 (nM)"]],
           [["1", "12.0", "59", "34.0"]])
    by_cid = {}
    for r in A.extract_from_tables([t]):
        by_cid.setdefault(r.cid, []).append(r.value_numeric)
    assert set(by_cid) == {"1"}


def test_group_without_an_assay_column_is_not_a_pair_table():
    """`No. | Structure | No. | Structure` carries no measurement to split."""
    t = _t(4, [["Compound No.", "Structure", "Compound No.", "Structure"]],
           [["1", "", "59", ""],
            ["2", "", "60", ""],
            ["3", "", "62", ""]])
    assert A.extract_from_tables([t]) == []


# ── the detector, directly ────────────────────────────────────────

def test_detector_declines_a_blank_header():
    """A headerless table tiles trivially. Trivial is not evidence."""
    t = _t(4, [], [["1", "12.0", "59", "34.0"],
                   ["2", "13.0", "60", "35.0"],
                   ["3", "14.0", "62", "36.0"]])
    hdr_rows, data_rows = A._header_rows_of(t)
    cols = A.build_columns(t, data_rows=data_rows)
    assert A._column_groups(cols, data_rows) is None
