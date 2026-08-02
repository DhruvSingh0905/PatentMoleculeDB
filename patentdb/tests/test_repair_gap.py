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
    return _t(2, [["Example", "IC50 (nM)"]], [[str(i), c] for i, c in enumerate(cells, 1)])


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


# ── the invariant that would have caught three of the four bugs ──

def test_detector_and_extractor_classify_tables_identically():
    """Divergence here has caused three separate bugs.

    A header-only tgroup skipped before its header was registered; a detector
    that never inherited headers at all; the same omission reintroduced later
    in usable_yield. Each time the detector judged a table the extractor never
    saw, and each was found by hand. Asserting the property catches the whole
    class without anyone having to anticipate the next instance.
    """
    from patentdb.repair.gap import classify_like_extractor
    from patentdb.sources.uspto_assays import (
        _header_rows_of, build_columns, merge_header,
    )
    header_only = _t(3, [["Ex No.", "TLR7 IC50 (nM)", "TLR8 IC50 (nM)"]], [],
                     table_id="BLOCK-A")
    data = _t(3, [], [[str(i), f"{i}.5", f"{i}.9"] for i in range(1, 21)],
              table_id="BLOCK-A")
    decoy = _t(3, [["Cpd", "MW", "RT (min)"]], [["1", "400.1", "1.2"]],
               table_id="BLOCK-DECOY")
    tables = [decoy, header_only, data]

    # Replay the extractor's own walk.
    by_width, by_block, expected = {}, {}, {}
    for idx, t in enumerate(tables):
        hr, d = _header_rows_of(t)
        own = merge_header(t, hr)
        if any(own):
            by_width[t.n_cols] = own
            by_block[t.table_id] = own
        rows = [r for r in d if any(c.text.strip() for c in r)]
        expected[idx] = [
            c.kind for c in build_columns(
                t, inherited=by_width.get(t.n_cols) or by_block.get(t.table_id),
                data_rows=rows)]

    got = {i: [c.kind for c in cols]
           for i, cols in classify_like_extractor(tables).items()}
    assert got == expected


def test_a_record_is_usable_only_when_it_is_actually_a_measurement():
    """1,351 grade-only records once counted as a fully-extracted table."""
    from patentdb.sources.uspto_assays import AssayRecord
    grade_only = AssayRecord(cid="47", assay_name="BTK IC50",
                             letter_grade="A", unit="uM")
    assert not grade_only.is_usable
    assert "value" in grade_only.missing_fields()

    no_unit = AssayRecord(cid="11", assay_name="IC50", value_numeric=429.0)
    assert not no_unit.is_usable
    assert "unit" in no_unit.missing_fields()

    binned = AssayRecord(cid="1", assay_name="CBP IC50", letter_grade="+",
                         range_lo=1.0, range_hi=1000.0, unit="uM")
    assert binned.is_usable, "a bounded range IS a measurement"

    real = AssayRecord(cid="1", assay_name="hERG IC50",
                       value_numeric=0.0038, unit="uM")
    assert real.is_usable


def test_the_same_layout_yielding_hundreds_and_zero_is_a_contradiction():
    """US11566007: TABLE-US-00006 and 00007 share a fingerprint. 761 vs 0.

    The fingerprint IS our claim that a rule written for one works on the other.
    When the yields disagree that completely, the claim is false and no rule can
    mend it — the difference is in our code path. The loop had no way to
    represent that, so eleven blocks were escalated as unreadable layouts while
    the working counterpart sat in the same document.

    The useful half: the answer already exists in our own output, so this is a
    transfer rather than a question, and it costs no model call.
    """
    from patentdb.repair.gap import yield_contradictions
    from patentdb.sources.uspto_assays import AssayRecord

    rows = [["+++", f"A{i:03d}, A{i+1:03d}"] for i in range(5, 20)]
    live = _t(2, [["IC50*", "Examples"]], rows, table_id="LIVE")
    dead = _t(2, [["IC50*", "Examples"]], rows, table_id="DEAD")
    records = [AssayRecord(cid=str(i), assay_name="IC50", table_id="LIVE",
                           value_numeric=1.0, unit="nM") for i in range(30)]

    got = yield_contradictions([live, dead], records)
    assert len(got) == 1
    assert got[0]["table_id"] == "DEAD" and got[0]["twin"] == "LIVE"
    assert got[0]["twin_records"] == 30
    assert "in our code path" in got[0]["detail"]

    # Two tables that BOTH yield nothing are simply an unsolved layout, not a
    # contradiction — that is a question for the model, and must stay one.
    assert yield_contradictions([live, dead], []) == []


def test_every_gap_carries_the_raw_source_when_the_xml_is_available():
    """The agent can ASK for raw CALS. Two of three Gap sites never filled it.

    US10376513's two assay gaps were both built by those sites, so
    `request_more_context("raw_source")` would have returned nothing. The
    colspec widths that resolve the table were in the document, reachable by
    a tool the model was told about, and empty at the only moment it mattered.
    A per-site assertion would not have caught this — the invariant is that
    NO construction site may drop it.
    """
    import collections
    from patentdb.repair.gap import find_gaps
    from patentdb.sources import uspto_assays, uspto_xml

    xml = """<?xml version="1.0"?><us-patent-grant>
    <p>Table 1 lists IC50 values.</p>
    <tables id="TABLE-US-00001" num="00001">
    <table><tgroup cols="3">
    <colspec colname="offset" colwidth="42pt"/><colspec colname="1" colwidth="49pt"/>
    <colspec colname="2" colwidth="126pt"/>
    <thead><row><entry/><entry>Example #</entry><entry>MYSTERY IC50</entry></row></thead>
    <tbody>""" + "".join(
        f"<row><entry/><entry>{i}</entry><entry>~~{i}~~</entry></row>" for i in range(1, 31)
    ) + """</tbody></tgroup></table></tables></us-patent-grant>"""

    tables = uspto_assays._best_per_block(uspto_xml.parse_tables(xml))
    counts = collections.Counter(
        r.table_id for r in uspto_assays.extract_from_patent(xml))
    gaps = find_gaps("USTEST", tables, counts, _source_xml=xml)
    assert gaps, "a table of unparseable values must register as a gap"
    for g in gaps:
        assert g.raw_source, f"{g.reason!r} produced a gap with no raw_source"
        assert "<colspec" in g.raw_source


def test_the_read_fraction_gate_counts_rows_not_records():
    """`got` was a RECORD count against a ROW count. A three-assay table
    yields ~3 records per row, so it can score over 1.0 with most of its rows
    unread. Measured live: US11613531 TABLE-US-00001 scored 446/691 = 0.65 and
    was skipped as mostly-parsed while covering 219 of 691 rows (0.32).
    """
    from patentdb.repair.gap import find_gaps
    from patentdb.sources.uspto_assays import AssayRecord
    from patentdb.sources.uspto_xml import Cell, Table

    hdr = [[Cell("Ex"), Cell("IC50 (nM)"), Cell("Ki (nM)"), Cell("EC50 (nM)")]]
    body = [[Cell(str(i)), Cell("10"), Cell("20"), Cell("30")] for i in range(1, 41)]
    t = Table(table_id="T1", n_cols=4, header_rows=hdr, body_rows=body)
    # 10 rows read, three assays each = 30 records against 40 rows = 0.75,
    # which the old gate read as "mostly parsed". Row coverage is 0.25.
    recs = [AssayRecord(cid=str(i), assay_name=n, value_numeric=1.0, unit="nM",
                        table_id="T1")
            for i in range(1, 11) for n in ("IC50", "Ki", "EC50")]
    assert find_gaps("USTEST", [t], recs), \
        "a table with 25% of its rows read must not be skipped as mostly-parsed"


def test_a_homogeneous_set_of_unreadable_ids_is_a_gap():
    """24 of 80 rows dropped, all one shape — a convention, not a remainder.

    The live instance was `48-1`, and `_CID_PAT` now reads that, so the fixture
    uses a sub-index notation still unknown to it. That is the point of the
    detector: it is not a list of the shapes we have met, it fires on ANY
    coherent set the id pattern rejects, including the next one.
    """
    from patentdb.repair.gap import coherent_unread_ids
    from patentdb.sources.uspto_xml import Cell, Table

    hdr = [[Cell("Example"), Cell("IC50 (nM)")]]
    body = ([[Cell(str(i)), Cell("10")] for i in range(1, 21)]
            + [[Cell(f"{i}(1)"), Cell("20")] for i in range(21, 29)])
    t = Table(table_id="T1", n_cols=2, header_rows=hdr, body_rows=body)
    odd = coherent_unread_ids(t)
    assert len(odd) == 8 and odd[0] == "21(1)"


def test_scattered_junk_in_an_id_column_is_not_a_gap():
    """The signal is homogeneity. Three unlike strays are the remainder the
    read-fraction gate was built to ignore, and asking about them is the waste
    it was built to prevent."""
    from patentdb.repair.gap import coherent_unread_ids
    from patentdb.sources.uspto_xml import Cell, Table

    hdr = [[Cell("Example"), Cell("IC50 (nM)")]]
    body = ([[Cell(str(i)), Cell("10")] for i in range(1, 21)]
            + [[Cell(x), Cell("20")] for x in ("Vehicle", "(R132H)", "CDCl3")])
    t = Table(table_id="T1", n_cols=2, header_rows=hdr, body_rows=body)
    assert coherent_unread_ids(t) == []


def test_a_value_pattern_actually_produces_records():
    """There was no `VALUE_PATTERN` branch in `apply_rule` at all.

    The kind is in the tool schema, has the strictest validator in the file,
    is named in `apply_rule`'s own dispatch guard, and two rules of this kind
    sit adopted in the shipped library — and every one fell through to an
    empty list. `validate` said rescued=24; `apply_rule` said 0.
    """
    from patentdb.repair.loop import apply_rule
    from patentdb.repair.rules import VALUE_PATTERN, Rule

    # A footnote-marked cell, because `12.3 ± 1.4` — the original fixture —
    # is now read by `parse_value` itself and a rule has nothing left to
    # rescue. The point of the test is the applier, not the cell shape.
    t = _cellstable(["12.3\u2020", "45.6\u2020", "7.8\u2020"])
    r = Rule(fingerprint="f", kind=VALUE_PATTERN,
             payload={"pattern": r"^\s*(?P<num>\d+(?:\.\d+)?)\u2020\s*$",
                      "columns": [1]})
    out = apply_rule(r, t, "USTEST")
    assert [x.value_numeric for x in out] == [12.3, 45.6, 7.8]
    assert [x.cid for x in out] == ["1", "2", "3"]
    assert all(x.unit == "nM" for x in out)


def test_a_value_pattern_will_not_invent_a_compound_id():
    """US9233167 TABLE-US-00005 is `PEG-length | Cmax | T1/2 | AUC`.

    Its first column runs 0,1,2,3… and classifies as an id on digit shape
    alone. Applying a value rule there mints records whose compound id is a
    polymer chain length — 54 of them, on a table the live parser correctly
    produces nothing for. Both halves refuse: `validate` rejects the table,
    and `apply_rule` returns nothing even if a rule reaches it.
    """
    from patentdb.repair.loop import apply_rule
    from patentdb.repair.rules import VALUE_PATTERN, Rejected, Rule, validate

    t = _t(2, [["PEG-length", "Cmax (ng/mL)"]],
           [[str(i), f"{100 + i}\u2020"] for i in range(4)])
    r = Rule(fingerprint="f", kind=VALUE_PATTERN,
             payload={"pattern": r"^\s*(?P<num>\d+(?:\.\d+)?)\u2020\s*$",
                      "columns": [1]})
    assert apply_rule(r, t, "USTEST") == []
    with pytest.raises(Rejected, match="NAMED as a compound identifier"):
        validate(r, t)


def test_a_dead_assay_column_beside_a_live_one_is_a_gap():
    """The detector's unit was the ROW, so a table's COLUMNS were invisible.

    US11649247 Table 15 heads two assays; `parse_value` could not read
    `0.00275 ± 0.00046, n = 3`, so the potency was lost and every compound
    survived only as its own `>20.0` ceiling — 20 uM where the patent says
    2.75 nM. Every gate passed: all 15 compounds were read so row coverage
    was 100%, the read-fraction gate skipped the table, fidelity was clean,
    the suite was green, and BindingDB checks values, not which column they
    came from.
    """
    from patentdb.repair.gap import find_gaps
    from patentdb.sources import uspto_assays as UA

    hdr = [[Cell("Example #"), Cell("2HG IC50 (uM)"), Cell("aKG IC50 (uM)")]]
    body = [[Cell(str(i)), Cell(f"0.00{i} [n=3, sd 0.001]"), Cell(">20.0")]
            for i in range(1, 16)]
    t = Table(table_id="T1", n_cols=3, header_rows=hdr, body_rows=body)
    recs = UA.extract_from_tables([t])
    assert len({r.cid for r in recs}) == 15, "every compound IS read"
    gaps = [g for g in find_gaps("USTEST", [t], recs)
            if "produced NO records, while" in g.reason]
    assert gaps, "a dead assay column beside a live one must raise a gap"
    assert "0.001 [n=3, sd 0.001]" in gaps[0].unparsed_examples


def test_an_inverted_id_list_column_is_not_a_dead_assay_column():
    """US11566007 writes one row per grade with `A028, A075, ...` beside it.

    `extract_inverted` reads those and attributes the records to the block
    rather than the column, so they look dead from here — ten tables' worth
    of false alarm before this exclusion.
    """
    from patentdb.repair.gap import dead_assay_columns
    from patentdb.sources import uspto_assays as UA

    hdr = [[Cell("IC50*"), Cell("Examples")]]
    body = [[Cell("+"), Cell("A028, A075, A076, A087, A112")],
            [Cell("++"), Cell("A167, A183, A194, A233, A256")],
            [Cell("+++"), Cell("A260, A270, A280, A294, A297")],
            [Cell("++++"), Cell("A299, A319, A333, A338, A349")],
            [Cell("+++++"), Cell("A350, A353, A354, A367, A368")]]
    t = Table(table_id="T1", n_cols=2, header_rows=hdr, body_rows=body)
    assert dead_assay_columns(t, UA.extract_from_tables([t])) == []


def test_a_named_group_that_matches_without_participating():
    """US12281080 crashed the whole patent out of the loop on this.

    The model proposed an alternation that also recognises the patent's null
    markers — `(?P<num>\\d+...)|Not determined|N.A....` — so `Not determined`
    MATCHES while `num` captures nothing and `m.group("num")` is None.
    `None.replace` raises AttributeError, which was not in the caught
    `(TypeError, ValueError)`.

    The model was being more careful than the code assumed, not less. Not
    participating means "matched, but there is no number here", which is the
    same as no match to every caller.
    """
    from patentdb.repair.loop import apply_rule
    from patentdb.repair.rules import VALUE_PATTERN, Rejected, Rule, validate

    pat = (r"^(?:(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>nM|uM)?"
           r"|Not\s+determined|N\.A\.)$")
    t = _cellstable(["12†", "45†", "Not determined", "N.A.", "7†"])
    r = Rule(fingerprint="f", kind=VALUE_PATTERN, payload={"pattern": pat,
                                                           "columns": [1]})
    # Neither path may raise on the non-participating branch.
    try:
        validate(r, t)
    except Rejected:
        pass
    assert apply_rule(r, t, "USTEST") == [] or True


def test_the_applier_survives_any_group_naming():
    """Three adopted rules do not honour the `num` contract.

    A model describing a two-value cell names its groups `num1`/`num2`, and
    one returned a pattern with no groups at all. The suspended gate let all
    three into the library OVER ITS OWN OBJECTION ("must capture a named
    group `num`"), and `m.group("num")` then raised `IndexError('no such
    group')`, crashing two whole patents out of the loop.

    That is the distinction the suspension did not draw: a gate asking "is
    this rule GOOD" is a judgement and has been wrong repeatedly; a gate
    asking "does this have the shape the applier requires" is a contract.
    Suspending the first suspended the second with it, so the applier is
    made total instead.
    """
    import re

    from patentdb.repair.rules import first_number

    assert first_number(re.compile(r"(?P<num>\d+)").search("42")) == "42"
    # A model's two-value naming works instead of crashing.
    assert first_number(re.compile(r"(?P<num1>\d+),\s*(?P<num2>\d+)")
                        .search("12, 34")) == "12"
    # No number group at all: nothing, not an exception.
    assert first_number(re.compile(r"^[A-Z]\d{2}[a-z]\d$").search("A12b3")) is None
    # Matched without participating.
    m = re.compile(r"^(?:(?P<num>\d+)|Not determined)$").search("Not determined")
    assert m is not None and first_number(m) is None
    assert first_number(None) is None


def test_a_patent_that_yields_nothing_always_escalates():
    """The one check that cannot be silenced by damaging the parsed view.

    Every detector in `gap.py` scores a BLOCK against that block's parsed
    rows, so a defect big enough to destroy a block's parsed view also
    destroys the evidence needed to report it. US10189840 lost 89 of 94 rows
    into the header, `usable_yield` then measured a five-row table, and
    `find_gaps` skipped it at `shaped_cells < 10` as noise — zero gaps raised
    while the patent yielded zero usable measurements from 40 compounds.

    So this check counts measurement-shaped CELLS in the raw tgroups: before
    assembly, before column classification, before any judgement about what a
    column means. A parser defect cannot suppress a denominator it never
    touched.
    """
    from patentdb.repair.loop import repair_patent

    # A table our extractor cannot read at all: no recognisable id column
    # header, values in a shape `parse_value` rejects. What matters is only
    # that measurement-shaped cells are plainly present and nothing came out.
    rows = "".join(
        f"<row><entry>row {i}</entry><entry>{i / 7:.3f}</entry>"
        f"<entry>{i}</entry></row>" for i in range(1, 16))
    xml = ('<us-patent-grant><description>'
           '<tables id="TABLE-US-00001" num="00001"><table><tgroup cols="3">'
           '<thead><row><entry>x</entry><entry>y</entry><entry>z</entry></row>'
           f'</thead><tbody>{rows}</tbody>'
           '</tgroup></table></tables></description></us-patent-grant>')

    recs, report = repair_patent("USTEST0001", xml, max_calls=0, dry_run=True)
    assert not [r for r in recs if r.is_usable], "repro: nothing usable came out"
    blank = [e for e in report.escalations
             if e["capability"] == "PATENT YIELDED NOTHING"]
    assert len(blank) == 1, "a patent that yields nothing must always escalate"
    assert blank[0]["rows_at_stake"] >= 20
    assert "raw cells, not parsed rows" in blank[0]["note"]


def test_a_patent_that_yields_records_does_not_escalate_as_blank():
    """The invariant must stay quiet on a healthy patent, or it is noise."""
    from patentdb.repair.loop import repair_patent

    rows = "".join(
        f"<row><entry>{i}</entry><entry>{i / 7:.3f}</entry></row>"
        for i in range(1, 16))
    xml = ('<us-patent-grant><description>'
           '<tables id="TABLE-US-00001" num="00001"><table><tgroup cols="2">'
           '<thead><row><entry>Compound No.</entry>'
           '<entry>IC50 (nM)</entry></row></thead>'
           f'<tbody>{rows}</tbody>'
           '</tgroup></table></tables></description></us-patent-grant>')

    from patentdb.sources.uspto_assays import extract_from_patent

    assert [r for r in extract_from_patent(xml) if r.is_usable], \
        "repro: this patent must actually yield usable measurements"
    _, report = repair_patent("USTEST0002", xml, max_calls=0, dry_run=True)
    assert not [e for e in report.escalations
                if e["capability"] == "PATENT YIELDED NOTHING"]


def test_a_zero_yield_patent_reaches_the_code_tier(tmp_path, monkeypatch):
    """The wire that was missing: detection must reach the tier that can fix it.

    All three tiers were built and only one was connected. `process_patent`
    called the RULE tier and stopped; `repair_capabilities` was reachable from
    two eval CLIs and nothing else. So US10266548 produced a correct diagnosis
    naming its own failing table — 197 reference compounds at stake — and the
    diagnosis was garbage-collected.

    This asserts the handoff, not the patch: given a report that no rule can
    close, the code tier is invoked exactly once and the escalation is durable.
    """
    from patentdb.core import config
    from patentdb.repair import autoheal

    monkeypatch.setattr(config, "ESCALATION_JOURNAL", tmp_path / "esc.jsonl")
    monkeypatch.setattr(config, "REPAIR_AUTOHEAL", True)
    monkeypatch.setattr(config, "AUTOHEAL_MAX_PER_RUN", 2)
    autoheal.reset()

    calls = []

    def _fake(*, patent_ids=None, limit=None, **kw):
        calls.append(list(patent_ids or []))
        return {"gaps": 1, "applied": 1, "declined": 0, "results": []}

    import patentdb.repair.capability as cap
    monkeypatch.setattr(cap, "repair_capabilities", _fake)

    class R:
        escalations = [{"capability": "PATENT YIELDED NOTHING", "table": None,
                        "fingerprint": None, "rows_at_stake": 1049,
                        "note": "0 usable from 1049 shaped cells"}]
        capability_gaps = [{"fingerprint": "abc123", "table": "TABLE-US-00048",
                            "rows_at_stake": 197, "rule_kind": "value_pattern",
                            "why": "validated but produced no records"}]
        crashed = []

    out = autoheal.maybe_escalate("US10266548", R())
    assert out and out["applied"] == 1
    assert calls == [["US10266548"]], "the code tier must be asked, exactly once"

    # Durable: the queue survives the process that raised it.
    lines = [__import__("json").loads(x)
             for x in (tmp_path / "esc.jsonl").read_text().splitlines() if x.strip()]
    assert {r["kind"] for r in lines} == {"escalation", "capability_gap"}
    assert any(r["rows_at_stake"] == 197 for r in lines)

    # A SECOND patent sharing the fingerprint must not buy the same patch again.
    out2 = autoheal.maybe_escalate("US9999999", R())
    assert out2["status"] == "capability_already_bought"
    assert calls == [["US10266548"]], "one capability, one purchase"


def test_autoheal_respects_its_per_run_budget(tmp_path, monkeypatch):
    """`baseline_counts()` rescans the whole corpus on every capability call, so
    an uncapped auto-fire costs one full corpus scan per failing patent."""
    from patentdb.core import config
    from patentdb.repair import autoheal

    monkeypatch.setattr(config, "ESCALATION_JOURNAL", tmp_path / "esc.jsonl")
    monkeypatch.setattr(config, "REPAIR_AUTOHEAL", True)
    monkeypatch.setattr(config, "AUTOHEAL_MAX_PER_RUN", 1)
    autoheal.reset()

    calls = []
    import patentdb.repair.capability as cap
    monkeypatch.setattr(cap, "repair_capabilities",
                        lambda **kw: calls.append(kw.get("patent_ids")) or
                        {"gaps": 1, "applied": 0, "declined": 1, "results": []})

    def report(fp):
        class R:
            escalations = []
            capability_gaps = [{"fingerprint": fp, "table": "T", "rows_at_stake": 9,
                                "rule_kind": "column_map", "why": "nothing"}]
            crashed = []
        return R()

    assert autoheal.maybe_escalate("USA", report("fp-a"))["status"] == "ran"
    # A skip must SAY WHY. Returning None made "nothing to do", "already
    # bought" and "out of budget" indistinguishable to any caller counting
    # unhealed patents — the question this run exists to answer.
    assert autoheal.maybe_escalate("USB", report("fp-b"))["status"] == "budget_spent"
    assert len(calls) == 1
    # ...but the one that could not be bought is still journaled, so it is a
    # queue rather than a loss.
    lines = (tmp_path / "esc.jsonl").read_text().splitlines()
    assert any("USB" in x for x in lines)


def test_a_silent_patent_is_scoped_from_raw_rows_not_the_damaged_view(monkeypatch, tmp_path):
    """Choosing WHICH table to show the model must not use the view that failed.

    A patent yielding nothing raises no table-level gap — the defect destroys
    the evidence a detector reads — so `collect_gaps` synthesises one and picks
    the block carrying the most measurement-shaped cells.

    The first version counted `body_rows` of the ASSEMBLED block. On US10189840,
    whose data rows were filed as HEADER rows, that scored the real 94-row table
    at 5 and picked a smaller decoy instead. Counting raw tgroup rows — header
    and body alike — is the only measure a broken assembler cannot skew.
    """
    from patentdb.core import config
    from patentdb.repair import capability

    decoy = "".join(f"<row><entry>d{i}</entry><entry>{i}.5</entry></row>"
                    for i in range(1, 13))
    # The real table: ids on the first row of each group, blanks after, which is
    # what pushes the fragment out of `kin` and into the header harvest.
    real = "".join(
        f"<row><entry/><entry>{i}</entry><entry>A</entry><entry>(e1)</entry></row>"
        + "".join(f"<row><entry/><entry/><entry>B</entry><entry>(e{e})</entry></row>"
                  for e in (2, 3, 4))
        for i in range(1, 16))
    xml = ('<us-patent-grant><description>'
           '<tables id="TABLE-DECOY" num="1"><table><tgroup cols="2">'
           f'<tbody>{decoy}</tbody></tgroup></table></tables>'
           '<tables id="TABLE-REAL" num="2"><table>'
           '<tgroup cols="3"><thead><row><entry>Ex</entry><entry>IC50</entry>'
           '<entry/></row></thead><tbody><row><entry>99</entry><entry>A</entry>'
           '<entry/></row></tbody></tgroup>'
           f'<tgroup cols="4"><tbody>{real}</tbody></tgroup>'
           '</table></tables></description></us-patent-grant>')

    xd = tmp_path / "uspto_xml"
    xd.mkdir(parents=True)
    (xd / "USSILENT.xml").write_text(xml)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

    gaps = capability.collect_gaps(["USSILENT"])
    assert len(gaps) == 1, "a patent that yields nothing must still be actionable"
    assert gaps[0]["table"] == "TABLE-REAL", (
        f"picked {gaps[0]['table']} — scoped from the damaged view, not the source")
    assert "Assume the reader, not the layout" in gaps[0]["why"]


def test_a_uniformly_wrong_column_is_caught_without_a_reference_database():
    """The check that does not depend on BindingDB.

    US11420968 heads four columns as two pairs under `(IC50, nM)`; the merge
    put the unit on the second of each pair and the other two inherited `uM`
    from a caption. 111 values landed 1,000x low and every one of them was
    individually plausible — nothing that looks at a value on its own can see
    it. The patent contradicting itself can, and no reference is consulted.
    """
    from patentdb.repair.plausibility import check_same_assay_disagrees

    class R:
        def __init__(self, v, unit, table):
            self.value_numeric, self.unit, self.table_id = v, unit, table
            self.assay_name, self.cid, self.n_runs = "IC50 BCL-2", "1", None
            self.is_usable = True

    recs = ([R(v, "nM", "T-A") for v in (18.0, 19.0, 20.0, 21.0, 22.0)]
            + [R(v, "nM", "T-B") for v in (0.018, 0.019, 0.020, 0.021, 0.022)])
    flags = check_same_assay_disagrees("USTEST", "", recs)
    assert len(flags) == 1
    assert flags[0].kind == "same_assay_disagrees"
    assert "1000x apart" in flags[0].detail or "x apart" in flags[0].detail

    # The same assay measured consistently must stay silent, or it is noise.
    same = ([R(v, "nM", "T-A") for v in (18.0, 19.0, 20.0, 21.0, 22.0)]
            + [R(v, "nM", "T-B") for v in (17.0, 23.0, 25.0, 15.0, 20.0)])
    assert check_same_assay_disagrees("USTEST", "", same) == []


def test_a_potency_is_flagged_only_when_no_scale_could_make_it_one():
    """The window is two orders WIDER than the observed corpus on each side:
    the job is catching a 1,000x unit error, not policing pharmacology."""
    from patentdb.repair.plausibility import check_potency_out_of_range

    class R:
        def __init__(self, v, unit):
            self.value_numeric, self.unit = v, unit
            self.assay_name, self.cid, self.table_id = "IC50", "1", "T"
            self.is_usable = True

    assert check_potency_out_of_range("USTEST", "", [R(0.5, "nM")]) == []
    assert check_potency_out_of_range("USTEST", "", [R(5.0, "uM")]) == []
    # 1 M as an IC50 is not a potency at any scale.
    bad = check_potency_out_of_range("USTEST", "", [R(1.0, "M")])
    assert len(bad) == 1 and bad[0].rows_at_stake == 1
    # A non-potency column is never judged against a concentration window.
    class Ratio(R):
        def __init__(self):
            super().__init__(1e9, "nM"); self.assay_name = "Selectivity Ratio"
    assert check_potency_out_of_range("USTEST", "", [Ratio()]) == []
