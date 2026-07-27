"""Tests for gap detection — especially the value-level signal.

Every parser bug found by hand during the BindingDB hunt was SILENT at row
level: `_VALUE_PAT` rejected any number >= 1000, and rejected `24 (*)` where
the parenthetical is a footnote rather than a replicate count. In both cases
the rows were simply never counted, coverage looked fine, and no gap was
raised. These tests pin the alarm that would have caught them.
"""
import pytest

from patentdb.repair.gap import find_gaps, layout_fingerprint
from patentdb.sources.uspto_assays import parse_value
from patentdb.sources.uspto_xml import Cell, Table


def _t(n_cols, header_rows, body_rows, table_id="T1", caption=""):
    def mk(rows):
        return [[c if isinstance(c, Cell) else Cell(str(c)) for c in r] for r in rows]
    return Table(table_id=table_id, n_cols=n_cols, header_rows=mk(header_rows),
                 body_rows=mk(body_rows), caption=caption)


# ── the value-level alarm ─────────────────────────────────────────

def test_value_level_gap_fires_when_ids_parse_but_values_do_not():
    """The branch that row-level coverage cannot see.

    Half the rows extract, half have a cell shape the value parser rejects.
    Row coverage looks acceptable; the values are being thrown away.
    """
    rows = [[str(i), f"{i}.5"] for i in range(1, 9)]          # parse fine
    rows += [[str(i), f"{i} ‡‡"] for i in range(9, 21)]  # unparseable cell
    t = _t(2, [["Ex", "IC50 (nM)"]], rows)
    gaps = find_gaps("USTEST", [t], {"T1": 8})     # 8 of 20 rows produced records
    assert gaps, "a table discarding 12 populated assay cells must raise a gap"
    assert "cannot be parsed" in gaps[0].reason


def test_null_markers_are_not_mistaken_for_unparseable_values():
    """`ND` / `—` mean 'not tested'. They are absence, not a parser failure."""
    rows = [[str(i), "ND"] for i in range(1, 21)]
    t = _t(2, [["Ex", "IC50 (nM)"]], rows)
    assert find_gaps("USTEST", [t], {"T1": 0}) == []


def test_a_fully_read_table_raises_nothing():
    rows = [[str(i), f"{i}.5"] for i in range(1, 21)]
    t = _t(2, [["Ex", "IC50 (nM)"]], rows)
    assert find_gaps("USTEST", [t], {"T1": 20}) == []


def test_mostly_read_tables_are_not_worth_asking_about():
    """Measured: proposals for these always yielded fewer rows than the parser."""
    rows = [[str(i), f"{i}.5"] for i in range(1, 21)]
    t = _t(2, [["Ex", "IC50 (nM)"]], rows)
    assert find_gaps("USTEST", [t], {"T1": 18}) == []


# ── the two silent parser bugs, as regressions ───────────────────

@pytest.mark.parametrize("cell,expected", [
    ("1511.5", 1511.5), ("8618", 8618.0), ("1412", 1412.0), ("1,234.5", 1234.5),
])
def test_values_of_a_thousand_or_more_parse(cell, expected):
    """`\\d{1,3}` silently dropped the entire weak-activity tail of the corpus."""
    got = parse_value(cell)
    assert got and got["value_numeric"] == expected


@pytest.mark.parametrize("cell,value,runs,annot", [
    ("24 (*)", 24.0, None, "*"),
    ("0.83 (A)", 0.83, None, "A"),
    ("12 (8)", 12.0, 8, None),
])
def test_parenthetical_is_a_run_count_only_when_it_is_digits(cell, value, runs, annot):
    got = parse_value(cell)
    assert got["value_numeric"] == value
    assert got["n_runs"] == runs
    assert got.get("annotation") == annot


# ── fingerprinting ────────────────────────────────────────────────

def test_fingerprint_ignores_content_so_rules_are_reusable():
    a = _t(2, [["Ex", "IC50 (nM)"]], [["1", "5.0"], ["2", "6.0"]])
    b = _t(2, [["Ex", "IC50 (nM)"]], [["941", "88.1"], ["942", "91.4"]])
    assert layout_fingerprint(a, ["Ex", "IC50 (nM)"]) == \
           layout_fingerprint(b, ["Ex", "IC50 (nM)"])


def test_fingerprint_separates_genuinely_different_shapes():
    a = _t(2, [["Ex", "IC50 (nM)"]], [["1", "5.0"]])
    b = _t(3, [["Ex", "MW", "RT"]], [["1", "400.1", "1.2"]])
    assert layout_fingerprint(a, ["Ex", "IC50 (nM)"]) != \
           layout_fingerprint(b, ["Ex", "MW", "RT"])


def test_a_header_only_tgroup_still_registers_its_header():
    """The bug that made the detector blind to the largest table in a patent.

    A `<tables>` block often splits into a header-only tgroup (zero data rows)
    plus continuations. `min_rows` discarded the header-only one BEFORE its
    header was recorded, so the continuation inherited a stale same-width
    header from an unrelated block, classified as [cid, structure, ms, rt, rt],
    found no assay column, and the value-level check never ran.
    """
    header_only = _t(3, [["Ex No.", "TLR7 IC50 (nM)", "TLR8 IC50 (nM)"]], [],
                     table_id="BLOCK-A")
    data = _t(3, [], [[str(i), f"{i}.5", f"{i}.9"] for i in range(1, 21)],
              table_id="BLOCK-A")
    # An unrelated 3-column block earlier in the document, whose header would be
    # inherited if BLOCK-A's own header were dropped.
    decoy = _t(3, [["Cpd", "MW", "RT (min)"]], [["1", "400.1", "1.2"]],
               table_id="BLOCK-DECOY")
    gaps = find_gaps("USTEST", [decoy, header_only, data], {"BLOCK-A": 0})
    a = [g for g in gaps if g.table_id == "BLOCK-A"]
    assert a, "BLOCK-A must be visible to the detector"
    assert "assay" in a[0].column_kinds, (
        f"inherited the wrong header: {a[0].headers}")


# ── value_pattern: self-healing for parser bugs ──────────────────

def _cellstable(cells):
    return _t(2, [["Ex", "IC50 (nM)"]], [[str(i), c] for i, c in enumerate(cells, 1)])


def test_value_pattern_must_not_change_cells_the_parser_already_reads():
    """The dangerous failure mode: rescue a few cells, silently reinterpret many.

    A pattern capturing the wrong span would sail through a naive
    'did coverage go up' check while corrupting everything else.
    """
    from patentdb.repair.rules import VALUE_PATTERN, Rejected, Rule, validate
    t = _cellstable(["5.1", "37.6", "1412\u2020", "8618\u2020", "9015\u2020"])
    bad = Rule(fingerprint="f", kind=VALUE_PATTERN,
               payload={"pattern": r"^\s*(?P<num>\d)"})   # first digit only
    with pytest.raises(Rejected, match="already read"):
        validate(bad, t)


def test_value_pattern_must_actually_rescue_something():
    from patentdb.repair.rules import VALUE_PATTERN, Rejected, Rule, validate
    t = _cellstable(["5.1", "37.6", "2.2", "0.7"])       # all already parse
    r = Rule(fingerprint="f", kind=VALUE_PATTERN,
             payload={"pattern": r"^\s*(?P<num>\d+(?:\.\d+)?)\s*$"})
    with pytest.raises(Rejected, match="rescues only"):
        validate(r, t)


def test_value_pattern_must_still_refuse_non_measurements():
    from patentdb.repair.rules import VALUE_PATTERN, Rejected, Rule, validate
    t = _cellstable(["1412\u2020", "8618\u2020", "9015\u2020", "1912\u2020"])
    greedy = Rule(fingerprint="f", kind=VALUE_PATTERN,
                  payload={"pattern": r"(?P<num>\d+(?:\.\d+)?)"})   # unanchored
    with pytest.raises(Rejected, match="matching text"):
        validate(greedy, t)


def test_a_lone_unnamed_group_is_promoted_rather_than_failing_the_call():
    """Models write `^(\\d+)$`; the single group can only be the number."""
    from patentdb.repair.rules import VALUE_PATTERN, Rule, validate
    t = _cellstable(["1412\u2020", "8618\u2020", "9015\u2020", "1912\u2020"])
    r = Rule(fingerprint="f", kind=VALUE_PATTERN,
             payload={"pattern": r"^(\d+)\u2020$", "columns": [1]})
    ev = validate(r, t)
    assert ev["rescued"] >= 3
    assert "?P<num>" in r.payload["pattern"]
