"""Tests for deterministic assay extraction from USPTO CALS tables.

The failure this module exists to prevent is not "missed an assay" — it is
"recorded a molecular weight as an IC50". Most of these tests are therefore
about what must NOT be extracted.
"""
import pytest

from patentdb.sources import uspto_assays as A
from patentdb.sources.uspto_xml import Cell, Table


def _t(n_cols, header_rows, body_rows, table_id="T1"):
    def mk(rows):
        return [[c if isinstance(c, Cell) else Cell(str(c)) for c in r] for r in rows]
    return Table(table_id=table_id, n_cols=n_cols,
                 header_rows=mk(header_rows), body_rows=mk(body_rows))


# ── what must not be extracted ────────────────────────────────────

def test_nmr_column_is_never_read():
    """US11292791's third column is `1H NMR ... δ (ppm)` — pure numeric noise."""
    t = _t(3, [["Example Number", "Structure", "1H NMR, LCMS"]],
           [["414", "", "1H NMR (CD3OD, 400 MHz) 7.35"]])
    assert A.extract_from_tables([t]) == []


def test_mass_and_retention_columns_are_never_read():
    """US10544143's LCMS table: MW, found mass, RT — none are assays."""
    t = _t(5, [["Cpd", "MW calc", "MS found", "RT (min)", "Method"]],
           [["12", "403.53", "404.3", "1.2", "QC-ACN-AA-XB"]])
    assert A.extract_from_tables([t]) == []


def test_a_column_we_cannot_classify_is_skipped_not_guessed():
    t = _t(2, [["Cpd", "Lot reference"]], [["7", "12.5"]])
    assert A.extract_from_tables([t]) == []


# ── what must be extracted ────────────────────────────────────────

def test_basic_assay_row():
    t = _t(2, [["Cpd No.", "P2X3 IC50 (μM)"]], [["I-2300", "0.003"]])
    recs = A.extract_from_tables([t])
    assert len(recs) == 1
    r = recs[0]
    assert r.cid == "I-2300"
    assert r.value_numeric == 0.003
    assert r.unit == "uM"
    assert "P2X3" in r.assay_name


def test_qualifiers_and_run_counts_survive():
    t = _t(3, [["Cmp No.", "Ki (μM)", "(n)"]], [["3", "~0.83", "(8)"], ["4", ">30", "(2)"]])
    recs = A.extract_from_tables([t])
    by_cid = {r.cid: r for r in recs}
    assert by_cid["3"].qualifier == "~" and by_cid["3"].value_numeric == 0.83
    assert by_cid["3"].n_runs == 8
    assert by_cid["4"].qualifier == ">" and by_cid["4"].n_runs == 2


def test_letter_grade_bins_are_kept_as_grades_not_numbers():
    """US11254686 reports potency as A-E bins; inventing a number would be a lie."""
    t = _t(2, [["Compound", "A2B cAMP IC50"]], [["Z1", "D"]])
    r = A.extract_from_tables([t])[0]
    assert r.letter_grade == "D"
    assert r.value_numeric is None


def test_null_markers_produce_no_record():
    t = _t(2, [["Cpd", "IC50 (nM)"]], [["1", "ND"], ["2", "—"], ["3", "n.d."], ["4", "5.0"]])
    recs = A.extract_from_tables([t])
    assert [r.cid for r in recs] == ["4"]


@pytest.mark.parametrize("cid", ["12", "I-2300", "Z1", "A1", "5a"])
def test_compound_id_shapes_seen_in_real_grants(cid):
    t = _t(2, [["Compound", "IC50 (nM)"]], [[cid, "1.0"]])
    assert len(A.extract_from_tables([t])) == 1


# ── header reconstruction ─────────────────────────────────────────

def test_stacked_header_rows_merge_into_one_name():
    """`Compound`/`No.` + `P2X3`/`IC50 (μM)` — reading one row loses the assay."""
    t = _t(2, [["Compound", "P2X3"], ["No.", "IC50 (μM)"]], [["I-1", "0.5"]])
    assert A.merge_header(t) == ["Compound No.", "P2X3 IC50 (μM)"]


def test_table_title_is_not_folded_into_the_assay_name():
    t = _t(2, [["TABLE 569"], ["Compound No.", "P2X3 IC50 (μM)"]], [["I-1", "0.5"]])
    assert "TABLE" not in A.extract_from_tables([t])[0].assay_name


def test_header_living_in_tbody_is_promoted():
    """US8952177 puts its whole header in tbody; thead is empty."""
    t = _t(2, [], [["Cmp No.", "FLAP Ki (μM)"], ["1", "0.0038"]])
    recs = A.extract_from_tables([t])
    assert len(recs) == 1
    assert "FLAP" in recs[0].assay_name
    assert recs[0].value_numeric == 0.0038


def test_headerless_continuation_inherits_the_previous_header():
    head = _t(2, [["Cmp No.", "hERG IC50 (nM)"]], [["1", "10"]])
    cont = _t(2, [], [["2", "20"]])
    recs = A.extract_from_tables([head, cont])
    assert len(recs) == 2
    assert all("hERG" in r.assay_name for r in recs)


def test_inherited_header_skips_unlabelled_run_count_columns():
    """US8952177: header declares 3 columns, data has 5 (each value + `(n)`).

    Assigning positionally would put the second assay's name on a run-count
    column and mislabel every value in it.
    """
    head = _t(3, [["Cmp No.", "FLAP Ki (μM)", "LTB4 IC50 (μM)"]], [])
    data = _t(5, [], [["1", "0.0038", "(8)", "0.4", "(3)"]])
    recs = A.extract_from_tables([head, data])
    names = {r.assay_name for r in recs}
    assert any("FLAP" in n for n in names)
    assert any("LTB4" in n for n in names)
    vals = {r.value_numeric for r in recs}
    assert vals == {0.0038, 0.4}
    assert {r.n_runs for r in recs} == {8, 3}


def test_short_header_row_alignment_is_chosen_by_coherence():
    """`FLAP Binding wild`/`Human Whole Blood` belong to columns 1-2, not 0-1.

    Left-aligning shifts every assay name one column, which mislabels values
    rather than dropping them — the worse failure.
    """
    t = _t(3,
           [["FLAP Binding wild", "Human Whole Blood"],
            ["Cmp No.", "type HTRF Ki (μM)", "LTB4 IC50 (μM)"]],
           [["1", "0.0038", "0.4"]])
    headers = A.merge_header(t)
    assert "Cmp No." in headers[0]
    assert "FLAP" in headers[1] and "HTRF" in headers[1]
    assert "LTB4" in headers[2]


def test_spacer_rows_do_not_break_extraction():
    t = _t(2, [["Cpd", "IC50 (nM)"]], [["1", "5"], ["", ""], ["2", "6"]])
    assert len(A.extract_from_tables([t])) == 2


def test_grouping_to_assay_tables_shape():
    t = _t(2, [["Cpd", "IC50 (nM)"]], [["1", "5"], ["1", "6"], ["2", "7"]])
    grouped = A.to_assay_tables(A.extract_from_tables([t]))
    assert set(grouped) == {"1", "2"}
    assert len(grouped["1"]) == 2
    assert grouped["1"][0]["unit"] == "nM"


# ── regressions found by benchmarking against BindingDB ───────────

@pytest.mark.parametrize("raw,expected", [
    ("Example 1", "1"), ("Example 314", "314"), ("Cpd. No. 5", "5"),
    ("Ex. 7", "7"), ("No. 12", "12"), ("007", "7"),
    ("1", "1"), ("I-2300", "I-2300"), ("Z1", "Z1"), ("A1", "A1"), ("5a", "5a"),
])
def test_labelled_compound_ids_are_recognised_and_normalised(raw, expected):
    """`Example 1` in the id column cost a whole 1,108-row table on US10245267.

    Both value columns classified correctly as assays, but the id column read
    as `unknown`, so the table produced nothing at all.
    """
    assert A._CID_PAT.match(raw)
    assert A.normalize_cid(raw) == expected


@pytest.mark.parametrize("bad", ["0.0038", "IC50 (nM)", "Structure", "QC-ACN-AA-XB", "(8)"])
def test_cid_pattern_still_rejects_non_ids(bad):
    assert not A._CID_PAT.match(bad)


def test_labelled_ids_extract_end_to_end():
    t = _t(3, [], [["Ex. No.", "C-Raf IC50", "B-Raf IC50"],
                   ["Example 1", "0.00030", "0.00010"]])
    recs = A.extract_from_tables([t])
    assert {r.cid for r in recs} == {"1"}
    assert {r.value_numeric for r in recs} == {0.0003, 0.0001}


@pytest.mark.parametrize("word,unit", [
    ("micromolar", "uM"), ("nanomolar", "nM"),
    ("millimolar", "mM"), ("picomolar", "pM"),
])
def test_spelled_out_units_are_understood(word, unit):
    """US10245267 states the unit only as `IC50's are micromolar.`"""
    assert A._unit_from(f"IC50's are {word}.") == unit


def test_wrapped_legend_inside_the_table_yields_its_unit():
    """The legend is line-wrapped across rows; per-row it is invisible."""
    t = _t(2, [],
           [["Selected compound structures and Raf inhibition data: numbering corresponds"],
            ["to the Examples above, most structures are found in the Examples. IC50's are"],
            ["micromolar."],
            ["Ex. No.", "C-Raf IC50"],
            ["Example 1", "0.000145"]])
    assert A._unit_from(A.table_legend(t)) == "uM"
    recs = A.extract_from_tables([t])
    assert len(recs) == 1
    assert recs[0].unit == "uM" and recs[0].value_numeric == 0.000145


def test_a_legend_unit_never_overrides_a_column_unit():
    t = _t(2, [["Cpd", "IC50 (nM)"]], [["1", "5.0"]])
    assert A.extract_from_tables([t])[0].unit == "nM"


# ── potency bins → ranges (the "unaddressable" patents) ──────────

def test_bin_key_prose_form():
    from patentdb.sources.bin_legend import parse_bin_key
    key = parse_bin_key(
        'an IC 50 value of greater than or equal to 0.001 μM and less than or '
        'equal to 0.01 μM is marked "++++"; a value greater than 0.01 μM and '
        'less than or equal to 0.1 μM is marked "+++";')
    assert (key["++++"].lo, key["++++"].hi, key["++++"].unit) == (0.001, 0.01, "uM")
    assert (key["+++"].lo, key["+++"].hi) == (0.01, 0.1)


def test_bin_key_compact_form():
    from patentdb.sources.bin_legend import parse_bin_key
    key = parse_bin_key("*Key: ++++: IC50 ≥ 1 uM +++: 1 uM > IC50 ≥ 0.1 uM")
    assert (key["++++"].lo, key["++++"].hi) == (1.0, None)
    assert (key["+++"].lo, key["+++"].hi) == (0.1, 1.0)


def test_bin_keys_are_never_shared_between_patents():
    """US11292791 and US11566007 give `++++` incompatible ranges.

    A global symbol→range table would silently corrupt one of them, so keys are
    always resolved per `<tables>` block.
    """
    from patentdb.sources.bin_legend import parse_bin_key
    a = parse_bin_key('a value greater than or equal to 0.001 μM and less than '
                      'or equal to 0.01 μM is marked "++++";')["++++"]
    b = parse_bin_key("*Key: ++++: IC50 ≥ 1 uM")["++++"]
    assert a != b
    assert a.hi == 0.01 and b.lo == 1.0


def test_no_key_means_no_range_rather_than_a_guess():
    from patentdb.sources.bin_legend import parse_bin_key
    assert parse_bin_key("Table 7 shows the results.") == {}


def test_range_midpoint_is_geometric():
    """Potency spans orders of magnitude; the middle of 0.01-0.1 is 0.032."""
    from patentdb.sources.bin_legend import BinRange
    assert abs(BinRange("+", 0.01, 0.1, "uM").midpoint - 0.0316) < 1e-3


def test_inverted_bin_table_yields_one_record_per_compound():
    """US11566007: one row per bin, hundreds of ids inside a single cell."""
    from patentdb.sources.bin_legend import parse_bin_key
    key = parse_bin_key("*Key: ++: IC50 ≥ 1 uM +: IC50 < 1 uM")
    t = _t(2, [], [["++", "A1, A2, A3, A4"], ["+", "A10, A11, A12"]])
    recs = A.extract_inverted([t], key, assay_name="H358 IC50", unit="uM")
    assert {r.cid for r in recs} == {"A1", "A2", "A3", "A4", "A10", "A11", "A12"}
    assert all(r.value_numeric is None for r in recs)
    assert {r.range_lo for r in recs if r.letter_grade == "++"} == {1.0}


def test_inverted_table_carries_the_bin_across_wrapped_rows():
    """The symbol appears once; the id list wraps across following rows."""
    from patentdb.sources.bin_legend import parse_bin_key
    key = parse_bin_key("*Key: ++: IC50 ≥ 1 uM")
    t = _t(2, [], [["++", "A1, A2, A3"], ["", "A4, A5, A6"], ["", "A7, A8, A9"]])
    recs = A.extract_inverted([t], key, assay_name="x", unit="uM")
    assert len(recs) == 9
    assert all(r.letter_grade == "++" for r in recs)


def test_not_tested_bins_are_dropped():
    from patentdb.sources.bin_legend import parse_bin_key
    t = _t(2, [], [["NT", "A1, A2, A3"]])
    assert A.extract_inverted([t], parse_bin_key("*Key: ++: IC50 ≥ 1 uM"),
                              assay_name="x", unit="uM") == []


def test_a_prose_cell_is_not_mistaken_for_a_compound_list():
    from patentdb.sources.bin_legend import parse_bin_key
    t = _t(2, [], [["+", "prepared as described above, see Example 1, and purified"]])
    assert A.extract_inverted([t], parse_bin_key("*Key: +: IC50 < 1 uM"),
                              assay_name="x", unit="uM") == []


def test_stacked_header_lines_of_equal_width_share_one_offset():
    """US10172859 Table 6, where two assay columns merged to the SAME name.

    The header is three physical lines::

        IC50   IC50   Ki
        DNA-   pDNA-  [Kv1.11
        No. | Structural formula | Name | PK | PK | hERG]

    The two short lines are the same three labels stacked, so they occupy the
    same columns by construction. Choosing an offset per row independently sent
    line 1 to columns 3-5 and line 2 to columns 0-2, and both assay columns
    came out named `IC50 PK`.

    That collision is not cosmetic: the patent measures DNA-PK in **nM** and
    pDNA-PK in **μM**, so one identical name over both columns means half those
    records get a scale that is 166-fold wrong, with nothing to distinguish
    them afterwards.
    """
    t = _t(6,
           [["IC50", "IC50", "Ki"],
            ["DNA-", "pDNA-", "[Kv1.11"],
            ["No.", "Structural formula", "Name", "PK", "PK", "hERG]"]],
           [["207", "", "[4-Fluoro-2-methyl]thiazol-2-ylmethanol", "A", "B", "B"],
            ["208", "", "[4-Fluoro-2-methyl]oxazol-2-ylmethanol", "B", "C", "A"],
            ["209", "", "[5-Chloro-2-methyl]thiazol-2-ylmethanol", "A", "A", "C"]])
    hdr, _ = A._header_rows_of(t)
    merged = A.merge_header(t, hdr)

    assert merged[3] != merged[4], (
        f"the two assay columns must not collapse to one name: {merged!r}")
    assert "DNA-" in merged[3] and "PK" in merged[3]
    assert "pDNA-" in merged[4]
    # The label lines must not land on the identifier/structure columns.
    assert "pDNA-" not in merged[1] and "Kv1.11" not in merged[2]


def test_a_line_break_hyphen_joins_without_a_space():
    """`DNA-` stacked over `PK` is `DNA-PK`, not `DNA- PK`."""
    assert A._join_header_lines(["IC50", "DNA-", "PK"]) == "IC50 DNA-PK"
    assert A._join_header_lines(["Ki", "[Kv1.11", "hERG]"]) == "Ki [Kv1.11 hERG]"
    assert A._join_header_lines(["IC50", "(nM)"]) == "IC50 (nM)"


def test_four_separated_stereoisomers_are_four_distinct_compounds():
    """US11312727 labels them 100AA / 100AB / 100BA / 100BB.

    `_CID_CORE` allowed ONE trailing letter, so all four failed the id test and
    the whole example was dropped — 135 of that patent's 382 compounds, and 131
    of the 171 compounds missing across the entire BindingDB reference corpus.
    The same document also uses single-letter ids (`102A`), which is why the
    gap was invisible: most of the patent extracted fine.
    """
    for cid in ("100AA", "100AB", "100BA", "100BB", "97AA"):
        assert A._CID_PAT.match(cid), cid
        assert A.normalize_cid(cid) == cid
    # Single-letter and bare forms must keep working.
    for cid in ("102A", "488-B", "12a", "I-2300", "Z1", "7"):
        assert A._CID_PAT.match(cid), cid
    # ...and a three-letter tail is a word, not a stereo label.
    assert not A._CID_PAT.match("12abc")


# ── vocabulary widenings found on 10 unseen patents (2026-07-28) ──

def test_a_bin_key_may_define_its_symbols_with_a_verb():
    """US10376513 writes "+ refers to <=10 nM", never "+:". 348 compounds."""
    from patentdb.sources.bin_legend import looks_like_key, parse_bin_key
    key = ("*column symbols: + refers to ≤10 nM ++ refers to >10 nM to 50 nM "
           "+++ refers to >50 nM to 200 nM +++++ refers to >500 nM")
    assert looks_like_key(key)
    k = parse_bin_key(key)
    assert (k["+"].lo, k["+"].hi) == (None, 10.0)
    # The upper bound must survive: ">10 nM to 50 nM" is a bounded bin, and
    # reading it as ">10 nM" silently widens every record it labels.
    assert (k["++"].lo, k["++"].hi) == (10.0, 50.0)
    assert (k["+++++"].lo, k["+++++"].hi) == (500.0, None)
    assert {b.unit for b in k.values()} == {"nM"}


def test_a_minus_sign_separates_the_bounds_of_a_bin():
    """US10626094: "A: IC 50 >200 nM−<800 nM" — U+2212, not a hyphen."""
    from patentdb.sources.bin_legend import parse_bin_key
    k = parse_bin_key("A: IC 50 >200 nM−<800 nM B: IC 50 >801 nM−<5000 nM "
                      "C: IC 50 >5001 nM")
    assert (k["A"].lo, k["A"].hi) == (200.0, 800.0)
    assert (k["B"].lo, k["B"].hi) == (801.0, 5000.0)
    assert (k["C"].lo, k["C"].hi) == (5001.0, None)


def test_the_older_compact_key_forms_still_parse_the_same():
    """The two incompatible `++++` scales this corpus contains, unchanged."""
    from patentdb.sources.bin_legend import parse_bin_key
    a = parse_bin_key("++++: IC50 >= 1 uM   +++: 1 uM > IC50 >= 0.1 uM")
    assert (a["++++"].lo, a["++++"].hi) == (1.0, None)
    assert (a["+++"].lo, a["+++"].hi) == (0.1, 1.0)
    b = parse_bin_key("+++++: IC50 >= 10 uM  ++++: 10 uM > IC50 >= 1 uM")
    assert (b["++++"].lo, b["++++"].hi) == (1.0, 10.0)


def test_a_potency_metric_with_a_unit_outranks_the_lcms_exclusion():
    """US10329273 heads its only assay column "h-MGAT LCMS IC50 (nM)"."""
    c = A.classify_column("h-MGAT LCMS IC50 (nM)", ["27", "5", "340"])
    assert c.kind == A.ASSAY and c.unit == "nM"


def test_a_metric_named_without_a_unit_is_still_excluded():
    """Both conditions are required: "LCMS IC50 method" names no scale."""
    assert A.classify_column("LCMS IC50 method", ["A", "B"]).kind != A.ASSAY


def test_a_bin_range_carries_the_key_s_unit_not_the_column_s():
    """US11485738 defines "A=<100 nM"; its column resolved to mM — 10^6 out."""
    from patentdb.sources.bin_legend import parse_bin_key
    key = parse_bin_key("Key: A=<100 nM; B=>100 nM-<500 nM; and C=>500 nM.")
    assert key["A"].unit == "nM"
    assert (key["B"].lo, key["B"].hi) == (100.0, 500.0)


def test_a_hyphenated_sub_index_is_an_identifier():
    """`48-1` is the first separated atropisomer of example 48, not a range."""
    assert A._CID_PAT.fullmatch("48-1")
    assert A._CID_PAT.fullmatch("323-2")
    assert A.normalize_cid("48-1") == "48-1"
    # Still an identifier, still two trailing letters, still not a measurement.
    assert A._CID_PAT.fullmatch("100AA")
    assert not A._CID_PAT.fullmatch("0.5-1.0")
    assert not A._CID_PAT.fullmatch("100-200 nM")


def test_scientific_notation_is_a_measurement():
    """US9765018 reports its most potent compounds as `6.49E−03`.

    Nine cells, all in the sub-nanomolar tail — the compounds anyone reads
    first — parsed as nothing at all. The minus may be ASCII, U+2212 or an en
    dash; typesetting must not decide whether a measurement survives.
    """
    assert A.parse_value("6.49E-03")["value_numeric"] == 0.00649
    assert A.parse_value("6.49E−03")["value_numeric"] == 0.00649
    assert A.parse_value("1.2e-5")["value_numeric"] == 1.2e-5
    # ...and the exponent must not turn text into a number.
    assert A.parse_value("Method F") is None
    assert A.parse_value("10E") is None


def test_a_bin_key_is_taken_from_the_nearest_preceding_run():
    """The last MATCH is one grade; the last RUN is the whole key.

    US11566007 prints ten bin tables whose keys sit past a wall of inline
    compound ids. Taking the final match alone returns `+: IC50 < 0.01 uM` —
    a one-grade scale that leaves three quarters of the table unbinned.
    """
    from patentdb.sources.bin_legend import nearest_key_before, parse_bin_key
    text = ("A643, A644, A646, A648, A649, A657, A663 "
            "*Key: ++++: IC50 >= 1 uM +++: 1 uM > IC50 >= 0.1 uM "
            "++: 0.1 uM > IC50 >= 0.01 uM +: IC50 < 0.01 uM")
    k = parse_bin_key(nearest_key_before(text))
    assert len(k) == 4 and (k["++++"].lo, k["++++"].hi) == (1.0, None)


def test_the_nearest_key_wins_when_two_scales_precede_a_table():
    """US11566007 defines a four-grade scale and then a five-grade one.

    `++++` means `>= 1 uM` in the first and `1-10 uM` in the second. Reaching
    back past the nearer definition would rewrite an upper-bounded bin as
    unbounded — a 10x overstatement presented as free recall.
    """
    from patentdb.sources.bin_legend import nearest_key_before, parse_bin_key
    text = ("*Key: ++++: IC50 >= 1 uM +++: 1 uM > IC50 >= 0.1 uM "
            "A100, A101, A102, A103 "
            "+++++: IC50 >= 10 uM ++++: 10 uM > IC50 >= 1 uM "
            "+++: 1 uM > IC50 >= 0.1 uM")
    k = parse_bin_key(nearest_key_before(text))
    assert (k["++++"].lo, k["++++"].hi) == (1.0, 10.0)


def test_a_symbol_defined_by_a_prose_clause():
    """US11752149: `A = IC50 of less than 10 nM` — words, not an operator.

    Form 1 needs a comparison symbol or a bare number straight after the
    separator and finds neither, so the whole key parsed as {} and all 47 of
    the patent's graded records came back with no value.
    """
    from patentdb.sources.bin_legend import parse_bin_key
    k = parse_bin_key(
        "A = IC 50 of less than 10 nM; B = IC 50 less than 100 nM but greater "
        "than or equal to 10 nM; C = IC 50 less than 1 μM (1,000 nM) but "
        "greater than or equal to 100 nM")
    assert (k["A"].lo, k["A"].hi) == (None, 10.0)
    assert (k["B"].lo, k["B"].hi) == (10.0, 100.0)
    # C states its bounds in DIFFERENT units. Applying the first unit to both
    # returned lo=100, hi=1 — an interval inverted and 1,000x wrong.
    assert (k["C"].lo, k["C"].hi, k["C"].unit) == (100.0, 1000.0, "nM")


def test_the_other_less_than_or_equal_glyphs():
    """US9670210 uses U+2266 `≦` throughout and never U+2264 `≤`.

    Its `+` grade resolved to nothing and `++` to "anything above 100 nM"
    when the patent bounds it at 500 — a silent widening decided by which
    glyph a typesetter picked.
    """
    from patentdb.sources.bin_legend import parse_bin_key
    k = parse_bin_key("(+ refers to IC 50 ≦100 nM; ++ refers to IC 50 >100 nM "
                      "and ≦500 nM)")
    assert (k["+"].lo, k["+"].hi) == (None, 100.0)
    assert (k["++"].lo, k["++"].hi) == (100.0, 500.0)


def test_a_prose_clause_may_only_tighten_a_bin_never_widen_it():
    """The prose reader runs after Form 1 and can refine it — one direction.

    Its body is also barred from crossing into the next symbol's clause: with
    no semicolons, `+ refers to ≤10 nM ++ refers to >10 nM to 50 nM` gave `+`
    a body running through `++`'s definition and `+` came out as 10..10.
    """
    from patentdb.sources.bin_legend import parse_bin_key
    k = parse_bin_key("+ refers to ≤10 nM ++ refers to >10 nM to 50 nM "
                      "+++ refers to >50 nM to 200 nM")
    assert (k["+"].lo, k["+"].hi) == (None, 10.0)
    assert (k["++"].lo, k["++"].hi) == (10.0, 50.0)
    assert (k["+++"].lo, k["+++"].hi) == (50.0, 200.0)
    # ...and a scale Form 1 already read correctly is left alone.
    a = parse_bin_key("++++: IC50 >= 1 uM   +++: 1 uM > IC50 >= 0.1 uM")
    assert (a["++++"].lo, a["++++"].hi) == (1.0, None)


def test_a_replicate_count_column_is_not_a_second_assay():
    """US8952177 writes the run count in its OWN column beside the value.

        ['1', '0.0038', '(8)', '0.4', '(3)']

    and lets each assay's header SPAN both columns, so the count column
    inherits "FLAP Binding wild type HTRF Ki (μM)" and classifies as a second
    assay. That both invents a duplicate assay column and costs every value
    its `n` — the attach step only fires on a neighbour typed NRUNS/UNKNOWN.
    Jie's curated CSV carries n=8 and n=3 for this compound, so the count is
    data someone asked for, not decoration.
    """
    hdr = [[A.Cell("Cmp No."), A.Cell("FLAP Ki (uM)"), A.Cell("FLAP Ki (uM)"),
            A.Cell("HWB IC50 (uM)"), A.Cell("HWB IC50 (uM)")]] \
        if hasattr(A, "Cell") else None
    from patentdb.sources.uspto_xml import Cell, Table
    hdr = [[Cell("Cmp No."), Cell("FLAP Ki (uM)"), Cell("FLAP Ki (uM)"),
            Cell("HWB IC50 (uM)"), Cell("HWB IC50 (uM)")]]
    body = [[Cell("1"), Cell("0.0038"), Cell("(8)"), Cell("0.4"), Cell("(3)")],
            [Cell("2"), Cell("0.0014"), Cell("(10)"), Cell("0.22"), Cell("(10)")]]
    t = Table(table_id="T1", n_cols=5, header_rows=hdr, body_rows=body)
    kinds = [c.kind for c in A.build_columns(t)]
    assert kinds == [A.CID, A.ASSAY, A.NRUNS, A.ASSAY, A.NRUNS]
    recs = {(r.cid, r.assay_name): r for r in A.extract_from_tables([t])}
    assert recs[("1", "FLAP Ki (uM)")].n_runs == 8
    assert recs[("1", "HWB IC50 (uM)")].n_runs == 3
    assert recs[("2", "FLAP Ki (uM)")].n_runs == 10


def test_the_benchmark_normalises_ids_the_way_the_extractor_does():
    """"One canonical form" has to mean ONE.

    `reference_bench._norm_cid` re-implemented normalisation with
    `s.lstrip("0")`, which only strips zeros at position 0 — so BindingDB's
    `I-0117` stayed `I-0117` while the patent's `I-117` normalised to `I-117`
    and the two never met. 1,119 compounds on US9718790 scored as missing that
    had been extracted correctly all along.
    """
    from patentdb.scripts.eval.reference_bench import _norm_cid
    for raw in ("I-0117", "I-0020", "A-0005", "007", "Example 0012", "N47",
                "48-1", "100AA"):
        assert _norm_cid(raw) == A.normalize_cid(raw).upper(), raw
    assert _norm_cid("I-0117") == _norm_cid("I-117")


def test_a_bindingdb_attribution_without_the_word_example():
    """BDB writes the patent's compound id straight after the patent number.

    The trailing "Compound N" is BindingDB's own within-table numbering, NOT
    the patent's id. Measured on US9303033: the first token hits our extracted
    ids 2,491 times and misses 12; the trailing one hits ZERO and misses 2,482.
    """
    from patentdb.scripts.eval.reference_bench import _EXAMPLE_REF
    def ids(s):
        return [m.group(2) for m in _EXAMPLE_REF.finditer(s)]
    assert ids("US8952177, Example 1") == ["1"]
    assert ids("US8722692, 1") == ["1"]
    assert ids("US9303033, N47, Table 58A, Compound 11") == ["N47"]
    assert ids("BDBM220085::US9303033, J48, Table 58A, Compound 33") == ["J48"]
    # A LABEL immediately after the patent number is part of the id, not a
    # reason to skip: US11286268 numbers its compounds 1..1837 and writes
    # "US11286268, Compound 1". A flat stop-list on the word `compound` reads
    # US9303033 right and US11286268 wrong — it cost that patent all 1,828 of
    # its reference values, so its patch could not be value-checked at all.
    # Position separates them, not vocabulary.
    assert ids("US11286268, Compound 1") == ["1"]
    assert ids("US1234567, Cmpd. No. 42") == ["42"]
    # ...and a bare section word is still never an identifier.
    assert ids("US1234567, Table 5") == []


def test_a_grade_a_colon_and_a_number_is_not_a_key():
    """Every part of Form 1 except the symbol is optional, so `A: 4` matched it.

    A patent's chemistry prose is full of those — US20230365584A1 alone yields
    97 Form-1 matches, including `A: 4`, `B: 4`, `E=20`. Harmless while the
    harvest text was a few hundred characters of local legend; the preceding
    window is now 6,000, so junk is one widening away from becoming a scale.

    A real key carries a unit, a comparison, the metric name, or a RANGE. The
    range had to be learned: US11254686 writes `A=<10 nM  B=10-50 nM ...
    D=100-500`, stating the unit early and dropping it later, and requiring
    only the first three cost that patent 372 records.
    """
    from patentdb.sources.bin_legend import parse_bin_key
    for junk in ("A: 4", "B: 4", "E=20", "E=30", "a: 5",
                 "Ring A is a C5-C7 cycloalkyl", "Step B: 4-methoxythiophene"):
        assert parse_bin_key(junk) == {}, junk
    k = parse_bin_key("A=<10 nM B=10-50 nM C=50-100 nM D=100-500")
    assert (k["D"].lo, k["D"].hi, k["D"].unit) == (100.0, 500.0, "nM")
    d = parse_bin_key("A denotes IC50 < 1 nM; B denotes 1 nM ≤ IC50 < 10 nM")
    assert (d["A"].hi, d["B"].lo, d["B"].hi) == (1.0, 1.0, 10.0)


def test_a_block_unit_is_not_stamped_on_a_dimensionless_column():
    """US11254686 TABLE-US-00003 puts `A=<10 nM; B=10-50 nM; ...` in its
    caption and heads nine columns, only some of which are concentrations.

    Every assay column took `nM`, including a dimensionless selectivity
    `Ratio` and two clearances whose own units are printed in their headers.
    Checked against BindingDB that was 99 records reading `2.24 nM` where the
    reference says 300 nM — not a near miss, a different quantity, and no
    coverage check can see it because the count goes UP.
    """
    from patentdb.sources.uspto_xml import Cell, Table
    cap = "The legend is as follows: A=<10 nM; B=10-50 nM; C=50-100 nM"
    hdr = [[Cell("Compound"), Cell("Ave A2B cAMP IC50"), Cell("Ratio"),
            Cell("LM CLint (uL/ min/mg/ protein)")]]
    body = [[Cell("Z1"), Cell("A"), Cell("0.55"), Cell("27.86")]]
    t = Table(table_id="T1", n_cols=4, header_rows=hdr, body_rows=body,
              caption=cap)
    by = {c.header: c for c in A.build_columns(t)}
    assert by["Ave A2B cAMP IC50"].unit == "nM"      # the legend describes it
    assert by["Ratio"].unit is None                  # dimensionless
    assert by["LM CLint (uL/ min/mg/ protein)"].unit is None   # states its own


def test_the_value_check_forgives_rounding_and_not_a_wrong_scale():
    """BindingDB stores three significant figures, so 23.456 nM and 23.5 nM
    are the same measurement. A ratio read as a potency is not."""
    from patentdb.repair.value_check import TOLERANCE, _bucket
    assert TOLERANCE == 0.05
    assert _bucket(23.456, 23.5) == "agree"          # rounding
    assert _bucket(1.0, 1.049) == "agree"
    assert _bucket(1.0, 1.9) == "variance"           # assay to assay
    assert _bucket(1.0, 5.0) == "disagree"
    assert _bucket(2.24, 300.0) == "wrong_scale"     # the Ratio class


def test_a_mean_with_a_standard_deviation_is_a_measurement():
    """`0.00275 ± 0.00046, n = 3` is a potency and a spread, not a reject.

    Rejecting the whole cell lost the measurement too: on US11649247 it kept
    the `>20.0` ceiling from the neighbouring column and dropped the real
    value, so the compound read as inactive where the patent reports 2.75 nM.
    147 assay cells across four patents.
    """
    got = A.parse_value("0.00275 ± 0.00046, n = 3")
    assert got["value_numeric"] == 0.00275 and got["n_runs"] == 3
    got = A.parse_value("1680 ± 150 (n = 4)")
    assert got["value_numeric"] == 1680.0 and got["n_runs"] == 4
    assert A.parse_value("5 ± 2 nM")["unit"] == "nM"
    # ...and the spread is dropped, not averaged into the value.
    assert A.parse_value("0.00275 ± 0.00046")["value_numeric"] == 0.00275


def test_a_family_wide_bindingdb_row_is_not_evidence_about_one_patent():
    """BindingDB names whole families in a single row:

        US9079866, 114::US9884878, Compound 114::US9745328, Compound 114

    46,000 of its 84,000 patent-bearing rows cite two or more, up to 17. The
    affinity was measured once and which family member's table it came from
    is not recorded, so attributing it to all of them scored 21 of our 24
    corpus value disagreements — every one against a patent whose own table
    says something different and is right.
    """
    import inspect

    from patentdb.repair import value_check

    src = inspect.getsource(value_check.load_reference)
    assert "single_patent_only" in src
    assert 'len({p for p, _ in hits}) > 1' in src
    assert inspect.signature(
        value_check.load_reference).parameters["single_patent_only"].default is True


def test_a_comma_inside_brackets_does_not_split_an_assay_name():
    """`probe 1, probe 2` is two assays. `HT1080 (R132C, 2-hydroxyglutarate)
    IC50 (uM)` is ONE — the comma names a mutant and a metabolite.

    Splitting it filed 13 records under the assay name `HT1080 (R132C` and
    paired the other half with `n = 7` out of the cell, so the run count
    vanished too. It was intermittent: the sibling column `HT1080 (R132C,
    aKG) IC50 (uM)` kept its full name only because its cells read `>20.0`
    with no comma to split on — the same header parsed two ways depending on
    what sat under it.
    """
    from patentdb.sources.uspto_assays import split_top_level as sp
    assert sp("probe 1, probe 2") == ["probe 1", "probe 2"]
    assert sp("A2A, A2B") == ["A2A", "A2B"]
    assert sp("IC50 (nM), Ki (nM)") == ["IC50 (nM)", "Ki (nM)"]
    assert sp("HT1080 (R132C, 2-hydroxyglutarate) IC50 (uM)") == \
        ["HT1080 (R132C, 2-hydroxyglutarate) IC50 (uM)"]
    assert sp("0.000698 ± 0.000352, n = 7") == ["0.000698 ± 0.000352", "n = 7"]


def test_the_table_s_own_header_outranks_its_caption_for_units():
    """US11420968 heads four assay columns as two pairs under a spanning
    `(IC50, nM)`, and our merge lands it on the second column of each pair.

    The two that missed out had no unit of their own and fell through to the
    caption — a paragraph opening "The Bcl-2 family proteins are central
    regulators of apoptosis" that mentions uM somewhere — so 111 values were
    recorded 1000x low with every count healthy. A caption is prose about the
    biology and may name any unit for any reason; a header is a statement
    about these columns.
    """
    from patentdb.sources.uspto_xml import Cell, Table
    cap = ("The Bcl-2 family proteins are central regulators of apoptosis, "
           "assayed at 10 uM in our hands.")
    hdr = [[Cell(""), Cell("Biochemical activity"), Cell("")],
           [Cell(""), Cell("(IC50, nM)"), Cell("")],
           [Cell("Example"), Cell("BCL-2"), Cell("Bcl-xl")]]
    body = [[Cell("A1"), Cell("43"), Cell("4570")]]
    t = Table(table_id="T1", n_cols=3, header_rows=hdr, body_rows=body,
              caption=cap)
    units = {c.header: c.unit for c in A.build_columns(t) if c.kind == A.ASSAY}
    assert units and set(units.values()) == {"nM"}, units


def test_a_table_that_cannot_agree_on_its_own_units_is_a_gap():
    """Not "a unit was guessed" — guessing is usually right and often the only
    option. The flag is a guess that CONTRADICTS what the same table states
    elsewhere, which does not happen in a well-formed header.

    Corpus-wide that distinction is the difference between 4 gaps and 0, and
    it would still have caught US11420968 saying uM and nM at once.
    """
    from patentdb.repair.gap import guessed_units
    from patentdb.sources.uspto_xml import Cell, Table

    hdr = [[Cell("Example"), Cell("Potency (nM)"), Cell("Cellular activity")]]
    body = [[Cell("1"), Cell("43"), Cell("582")]]
    t = Table(table_id="T1", n_cols=3, header_rows=hdr, body_rows=body,
              caption="Assays were run at 10 uM.")
    # The header's own nM wins, so the table agrees with itself.
    assert all(g["agrees"] for g in guessed_units(t))
    # A single-assay table has no sibling to disagree with.
    solo = Table(table_id="T2", n_cols=2,
                 header_rows=[[Cell("Example"), Cell("Activity")]],
                 body_rows=[[Cell("1"), Cell("5")]],
                 caption="Assays were run at 10 uM.")
    assert guessed_units(solo) == []
