"""The potency-bin key: the forms a legend takes, and what must not be one.

Every legend string here is copied out of a real patent in the cached corpus,
including its punctuation and its typos. That matters more than usual for this
file: each of the seven forms below was in the corpus the whole time and read
as no key at all, so a synthetic legend would have tested only the shapes we
already handled.

The failure this file guards against is not "no range" — it is a WRONG range
that looks right. A bin key is applied uniformly to every graded row under it,
so a bound on the wrong side, or a scale on the wrong column, mislabels
thousands of records at once with nothing in the output to show for it.
"""
import pytest

from patentdb3.core import config
from patentdb3.sources import bin_legend
from patentdb3.sources.uspto_assays import (
    extract_from_patent,
    _redistribute_shared_prefix,
)


def _xml(pid: str) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="ignore")


def _bins(text: str):
    return {s: (b.lo, b.hi, b.unit) for s, b in bin_legend.parse_bin_key(text).items()}


# ── A. the forms a legend is written in ──────────────────────────────────
#
# The separator is the whole story. Every string below reached a parser that
# could read it and was turned away one step earlier, by a pre-filter whose
# accepted separators were `:`, `=` and five verbs.

@pytest.mark.parametrize("pid,text,expected", [
    # A colon, and the symbol in a DIFFERENT CELL from its range. Joined here
    # the way `_legend_lines` joins a legend row across its cells.
    ("US10172859", "A: IC50 < 3 nM", {"A": (None, 3.0, "nM")}),
    # A parenthesis as the separator, and no colon anywhere.
    ("US9682141",
     "+ (greater than 10 microMolar), ++ (less than 10 microMolar), "
     "+++ (less than 1 microMolar), and ++++ (less than 100 nM).",
     {"+": (10.0, None, "uM"), "++": (None, 10.0, "uM"),
      "+++": (None, 1.0, "uM"), "++++": (None, 100.0, "nM")}),
    # No separator at all — the comparison follows the symbol directly. The
    # interior grades are bracketed from BOTH sides, and the bound on the left
    # is the one a left-to-right read drops.
    ("US9656988",
     "IC50: A ≦ 10 nM; 10 nM < B ≦ 100 nM; 100 nM < C ≦ 1 μM; "
     "1 μM < D ≦ 10 μM; E > 10 μM",
     {"A": (None, 10.0, "nM"), "B": (10.0, 100.0, "nM"),
      "C": (100.0, 1000.0, "nM"),          # stated in nM and μM; see test_D
      "D": (1.0, 10.0, "uM"), "E": (10.0, None, "uM")}),
    # `*` and `#` are grade symbols, not footnote markers, and the verb is the
    # bare `is`. Two scales in one string, told apart by symbol alphabet.
    ("US10030020",
     "*** is less than 100 nM; ** is between 1000 and 100 nM; "
     "* is greater than 1000 nM ### is ≥75%; "
     "## is equal to or greater than 25% but less than 75%; # is <25%",
     {"***": (None, 100.0, "nM"), "*": (1000.0, None, "nM")}),
    # The range is stated first and the symbol assigned to it by a verb.
    ("US11286268",
     'In Table 4A, IC50 values of less than 0.05 μM are labelled as "+++", '
     'from 1 to 0.05 μM are labelled as "++", and greater than 1 μM are '
     'labelled as "+".',
     {"+++": (None, 0.05, "uM"), "+": (1.0, None, "uM")}),
    # A quoted symbol, and filler words between the verb and the comparison.
    ("US20240166635",
     "“A” represents a calculated IC50 value of less than 10 nM; “B” represents "
     "a calculated IC50 value of greater than or equal to 10 nM and less than "
     "100 nM",
     {"A": (None, 10.0, "nM"), "B": (10.0, 100.0, "nM")}),
    # Reversed: the range comes first and `=` assigns the symbol to it. Every
    # other form reads left-to-right from the symbol and reads this backwards.
    ("US9688680",
     "for D816V activity, the following designations are used: <1.00 nM=A; "
     "1.01-10.0 nM=B; 10.01-100.0 nM=C; >100 nM=D; and ND=not determined.",
     {"A": (None, 1.0, "nM"), "B": (1.01, 10.0, "nM"),
      "C": (10.01, 100.0, "nM"), "D": (100.0, None, "nM")}),
])
def test_every_form_in_the_corpus_parses_with_the_right_bounds(pid, text, expected):
    got = _bins(text)
    for sym, want in expected.items():
        assert got.get(sym) == want, f"{pid} {sym}: got {got.get(sym)}, want {want}"


@pytest.mark.parametrize("text,expected", [
    # The bound sits BEFORE the metric and nothing follows it. While the
    # trailing bound was mandatory the engine backtracked, released `10` as the
    # upper bound and re-matched it as the lower — `Ki <= 10 uM` read as
    # `Ki >= 10 uM`, inverted, on every row written that way.
    ("D: 10 μM ≥ Ki", {"D": (None, 10.0, "uM")}),
    ("A: Ki > 25 μM", {"A": (25.0, None, "uM")}),
    # Both bounds present, upper stated first.
    ("B: 25 μM ≥ Ki > 15 μM", {"B": (15.0, 25.0, "uM")}),
])
def test_a_bound_before_the_metric_keeps_its_side(text, expected):
    assert _bins(text) == expected


def test_two_bounds_on_different_scales_are_reconciled():
    """`100 nM < C ≦ 1 μM` states its bounds 1,000x apart. Taking one unit for
    the pair reported C as `100..1 uM` — a lower bound above its upper, still
    shaped like a valid answer."""
    assert _bins("100 nM < C ≦ 1 μM") == {"C": (100.0, 1000.0, "nM")}


# ── B. what is NOT a key ─────────────────────────────────────────────────

def test_a_grade_glued_to_a_word_is_not_a_grade():
    """`the following designations are used: <1.00 nM=A` defines A. It also
    matched `d` — the last letter of "used", a colon, and a range. Case-folding
    the grade for the sake of the English verbs around it turns the tail of any
    word into a symbol."""
    got = _bins("the following designations are used: <1.00 nM=A")
    assert set(got) == {"A"}, f"a grade inside a word was read: {set(got) - {'A'}}"


def test_a_grade_after_a_digit_is_still_a_grade():
    """The mirror of the test above, and the reason the guard excludes letters
    only. US11485738 runs its sentence into its key: `the results are shown in
    Table 15A=<250 nM`. Excluding digits too drops that key and 117 records."""
    assert _bins("the results are shown in Table 15A=<250 nM") == {
        "A": (None, 250.0, "nM")}


def test_a_bare_number_assigned_to_a_symbol_states_no_interval():
    """`100 nM=B` names an edge without a side. Choosing one invents the bin."""
    assert _bins("the key is: 100 nM=B") == {}


@pytest.mark.parametrize("text", [
    # US9221791 rates fungicides in a two-column table, value then rating.
    # Flattened to prose it reads `C` followed by `<40` — which is verbatim the
    # value printed on D's row. Every grade took the next grade's number.
    "% Disease Control @ 50 ppm Rating 80-100 A 60-79 B 40-59 C <40 D Not tested E",
    "MIC (μg/mL) Rating ≦0.5 A >1.5-1.5 B >1.5-4 C >4 D Not tested E",
])
def test_a_value_printed_before_a_symbol_is_not_its_definition(text):
    assert _bins(text) == {}


def test_a_value_pointing_at_a_symbol_still_is_its_definition():
    """The mirror of the test above, and the reason the guard tests for an
    operator. `10 nM < B` also puts a value before the symbol — but with a
    comparison aimed AT it, which is a real lower bound."""
    assert _bins("10 nM < B ≦ 100 nM") == {"B": (10.0, 100.0, "nM")}


def test_a_symbol_two_scales_define_differently_yields_nothing():
    """US9688680 states two scales in one paragraph, `for D816V activity` and
    `For wild-type Kit activity`, and `A` is `<1.00 nM` in one and `<10 nM` in
    the other. Taking the first and applying it to both columns is a silent 10x
    on half the records."""
    both = ("for D816V activity, the following designations are used: "
            "<1.00 nM=A; 1.01-10.0 nM=B. For wild-type Kit activity, the "
            "following designations are used: <10 nM=A; 11-100 nM=B.")
    assert _bins(both) == {}
    one = ("for D816V activity, the following designations are used: "
           "<1.00 nM=A; 1.01-10.0 nM=B.")
    assert _bins(one) == {"A": (None, 1.0, "nM"), "B": (1.01, 10.0, "nM")}


def test_a_percent_scale_does_not_inherit_a_concentration_unit():
    """US10030020 grades potency with `*` in nM and inhibition with `#` in
    percent, in one legend. `%` was not a unit here, so the `#` bins parsed
    unitless and the backfill stamped `nM` on them — 1,243 records claiming a
    nanomolar potency for a column headed `% Inh 1 μM (mouse)`."""
    got = _bins("*** is less than 100 nM; ** is between 1000 and 100 nM; "
                "* is greater than 1000 nM ### is ≥75%; ## is equal to or "
                "greater than 25% but less than 75%; # is <25%")
    assert got["###"] == (75.0, None, "%")
    assert got["#"] == (None, 25.0, "%")
    assert got["***"] == (None, 100.0, "nM")


@pytest.mark.parametrize("column_unit,key_unit,ok", [
    ("uM", "fold", False),      # US12351648: a selectivity ratio on a Ki column
    ("nM", "%", False),
    ("nM", "uM", True),         # same dimension, converted elsewhere
    (None, "fold", True),       # a column that states no unit takes the key's
    ("uM", None, True),
])
def test_a_bin_may_not_change_what_is_being_measured(column_unit, key_unit, ok):
    assert bin_legend.compatible(column_unit, key_unit) is ok


# ── C. one scale per column ──────────────────────────────────────────────
#
# The case this section exists for costs three orders of magnitude when it is
# got wrong, and looks completely normal in the output either way.

LEGEND_10172859 = [
    "DNA-PK (enzymatic):",
    "A: IC50 < 3 nM", "B: 3 nM ≤ IC50 < 7 nM",
    "C: 7 nM ≤ IC50 < 30 nM", "D: 30 nM ≤ IC50",
    "pDNA-PK (cellular):",
    "A: IC50 < 0.5 μM", "B: 0.5 μM ≤ IC50 < 5 μM",
    "C: 5 μM ≤ IC50 < 10 μM", "D: 10 μM ≤ IC50 < 30 μM",
    "Kv11.1 hERG:",
    "A: Ki > 25 μM", "B: 25 μM ≥ Ki > 15 μM",
    "C: 15 μM ≥ Ki > 10 μM", "D: 10 μM ≥ Ki",
]


def test_the_same_letter_means_three_different_things():
    """US10172859 defines A-D three times over, once per assay. `B` is 3-7 nM,
    0.5-5 uM or 15-25 uM depending only on which column it sits in. Flattening
    the three and parsing once returns the first, because the parser takes the
    first definition of a symbol by design — so every hERG bin would come out
    on a nM scale."""
    secs = bin_legend.parse_sectioned_key(LEGEND_10172859)
    assert set(secs) == {"DNA-PK (enzymatic)", "pDNA-PK (cellular)", "Kv11.1 hERG"}
    b = {h: (k["B"].lo, k["B"].hi, k["B"].unit) for h, k in secs.items()}
    assert b["DNA-PK (enzymatic)"] == (3.0, 7.0, "nM")
    assert b["pDNA-PK (cellular)"] == (0.5, 5.0, "uM")
    assert b["Kv11.1 hERG"] == (15.0, 25.0, "uM")


@pytest.mark.parametrize("column,heading", [
    # `DNA-PK` is a substring of `pDNA-PK`, so this column matches BOTH
    # headings and the shorter one is wrong by exactly the margin that makes
    # it look right. Longest match wins.
    ("IC50 pDNA-PK", "pDNA-PK (cellular)"),
    ("IC50 DNA-PK", "DNA-PK (enzymatic)"),
    # The patent transposes its own digits — `Kv11.1` in the legend, `Kv1.11`
    # in the column. Both normalise to the same letters and digits.
    ("Ki [Kv1.11 hERG]", "Kv11.1 hERG"),
    # A column no heading describes gets no scale rather than a guessed one.
    ("MS m/z", None),
    ("IC50", None),
])
def test_a_column_takes_the_scale_that_names_it(column, heading):
    secs = bin_legend.parse_sectioned_key(LEGEND_10172859)
    assert bin_legend.section_for_column(column, secs) == heading


def test_an_unreadable_definition_row_does_not_become_a_heading():
    """A definition the parser cannot read must fail closed. Promoted to a
    heading it would open a section that does not exist and push every row
    below it out of the scale it belongs to, so one unreadable grade would
    mislabel a whole column rather than costing its own row."""
    secs = bin_legend.parse_sectioned_key(
        ["hERG:", "A: Ki > 25 μM", "B: gibberish with no number at all",
         "C: 15 μM ≥ Ki > 10 μM"])
    assert set(secs) == {"hERG"}
    assert set(secs["hERG"]) == {"A", "C"}


# ── D. end to end, against the patents themselves ────────────────────────

def test_the_three_scales_land_on_their_own_columns():
    """The whole point, measured on the output rather than the parser: every
    graded record in US10172859 carries a range, and the ranges differ by
    column even though the letters do not."""
    recs = [r for r in extract_from_patent(_xml("US10172859")) if r.letter_grade]
    assert recs, "no graded records"
    assert all(r.range_lo is not None or r.range_hi is not None for r in recs)
    b = {(r.assay_name, r.letter_grade): (r.range_lo, r.range_hi, r.unit)
         for r in recs}
    assert b[("IC50 DNA-PK", "B")] == (3.0, 7.0, "nM")
    assert b[("IC50 pDNA-PK", "B")] == (0.5, 5.0, "uM")
    assert b[("Ki [Kv1.11 hERG]", "B")] == (15.0, 25.0, "uM")


def test_a_data_row_is_not_a_legend():
    """US20250163063 grades four kinase columns and prints a literal `>10 uM`
    beside each grade, so a row joined across its cells reads `... A >10 uM
    ...` — the exact shape of `A ≦ 10 nM`. Every data row then defined every
    grade, all four collapsed onto `>10 uM`, and 2,977 records took whichever
    came first. A legend row carries a symbol and its meaning; a data row
    carries a compound and its results."""
    recs = [r for r in extract_from_patent(_xml("US20250163063")) if r.letter_grade]
    assert recs, "no graded records"
    bins = {r.letter_grade: (r.range_lo, r.range_hi) for r in recs}
    assert bins.get("B") == (0.1, 1.0), f"B is {bins.get('B')}, patent says 0.1-1.0 uM"
    distinct = {v for v in bins.values() if v != (None, None)}
    assert len(distinct) > 1, f"every grade collapsed onto one range: {bins}"


def test_each_block_keeps_its_own_scale():
    """US9133148 grades three different quantities with overlapping symbols:
    IC50 in nM across Tables 6 and 7, an ED50 in mg/kg in Table 7, and percent
    inhibition in Table 8 — `***` meaning `<100 nM` in one and `>75%` in
    another. One flat key for the document puts a nanomolar potency on a
    percentage column."""
    recs = [r for r in extract_from_patent(_xml("US9133148"))
            if r.letter_grade and r.range_lo is not None or
            (r.letter_grade and r.range_hi is not None)]
    by_col = {}
    for r in recs:
        by_col.setdefault((r.table_id, r.assay_name), {})[r.letter_grade] = r.unit
    units = {col: set(g.values()) for col, g in by_col.items()}
    for (table, name), seen in units.items():
        if "IC50" in (name or ""):
            assert seen <= {"nM"}, f"{table} {name} got {seen}"
        elif "ED50" in (name or ""):
            assert seen <= {"mg/kg"}, f"{table} {name} got {seen}"
    assert any(u == {"%"} for u in units.values()), "the percent scale vanished"


# ── E. a legend laid out as a table, in either column order ──────────────

SEPTORIA = [["", "≦0.5", "A"], ["", ">1.5-1.5", "B"],
            ["", ">1.5-4", "C"], ["", ">4", "D"], ["", "Not tested", "E"]]
PUCCINIA = [["", "80-100", "A"], ["", "60-79", "B"],
            ["", "40-59", "C"], ["", "<40", "D"], ["", "Not tested", "E"]]


def test_a_legend_table_reads_in_either_column_order():
    """US10172859 prints `A: | IC50 < 3 nM`; US9221791 prints `≦0.5 | A`.
    Whichever cell is nothing but a symbol is the symbol."""
    forward = bin_legend.parse_bin_table(
        [["", "A:", "IC50 < 3 nM"], ["", "B:", "3 nM ≤ IC50 < 7 nM"]])
    assert {s: (b.lo, b.hi) for s, b in forward.items()} == {
        "A": (None, 3.0), "B": (3.0, 7.0)}
    reverse = bin_legend.parse_bin_table(PUCCINIA)
    assert {s: (b.lo, b.hi) for s, b in reverse.items()} == {
        "A": (80.0, 100.0), "B": (60.0, 79.0), "C": (40.0, 59.0),
        "D": (None, 40.0)}


def test_a_legend_table_takes_the_unit_from_its_own_heading():
    """The rows carry bare numbers; the heading carries the unit. US9221791
    heads them `MIC (μg/mL` — the bracket is lost in the CALS split — and
    `% Disease Control @ 50 ppm`."""
    sep = bin_legend.parse_bin_table(SEPTORIA, unit_hint="MIC (μg/mL Rating")
    puc = bin_legend.parse_bin_table(
        PUCCINIA, unit_hint="% Disease Control @ 50 ppm Rating")
    assert {b.unit for b in sep.values()} == {"ug/mL"}
    assert {b.unit for b in puc.values()} == {"%"}


# ── F. one scale per column, when the prose says which ───────────────────

US9688680_PROSE = (
    "In the Table below, for D816V activity, the following designations are "
    "used: <1.00 nM=A; 1.01-10.0 nM=B; 10.01-100.0 nM=C; >100 nM=D; and "
    "ND=not determined. For wild-type Kit activity, the following "
    "designations are used: <10 nM=A; 11-100 nM=B; 100-1000 nM=C; "
    ">1000 nM=D; and ND=not determined.")


def test_prose_that_states_two_scales_is_split_before_it_is_parsed():
    secs = dict(bin_legend.split_prose_sections(US9688680_PROSE))
    assert list(secs) == ["D816V activity", "wild-type Kit activity"]
    a = bin_legend.parse_bin_key(secs["D816V activity"])
    b = bin_legend.parse_bin_key(secs["wild-type Kit activity"])
    assert (a["A"].lo, a["A"].hi) == (None, 1.0)
    assert (b["A"].lo, b["A"].hi) == (None, 10.0)


def test_one_scale_stated_alone_is_not_split():
    """A single scale keeps the ordinary path — splitting is only for the
    ambiguous case."""
    assert bin_legend.split_prose_sections(
        "the following designations are used: <10 nM=A; 11-100 nM=B") == []


def test_the_last_column_is_settled_by_counting_when_names_run_out():
    """`WT` shares not one character with `wild-type`, so no amount of name
    matching pairs them. `D816V` pins the first column and excludes the
    second, which leaves the second pairing determined rather than guessed."""
    sections = ["D816V activity", "wild-type Kit activity"]
    assert bin_legend.section_for_column("D816V IC50 (nM)", sections) == \
        "D816V activity"
    assert bin_legend.section_for_column("WT IC50 (nM)", sections) is None
    assert bin_legend.assign_sections(
        ["D816V IC50 (nM)", "WT IC50 (nM)"], sections) == {
            "D816V IC50 (nM)": "D816V activity",
            "WT IC50 (nM)": "wild-type Kit activity"}


def test_counting_settles_nothing_when_the_counts_disagree():
    """Two scales and three columns is not a pairing. Guessing one would put a
    scale on a column the document never assigned it to."""
    got = bin_legend.assign_sections(
        ["D816V IC50 (nM)", "WT IC50 (nM)", "Phospho IC50 (nM)"],
        ["D816V activity", "wild-type Kit activity"])
    assert got == {"D816V IC50 (nM)": "D816V activity"}


def test_two_scales_land_on_their_own_columns_end_to_end():
    """US9688680 Table 2 grades D816V and wild-type Kit in one table with the
    same letters. `A` is <1.00 nM in one column and <10 nM in the other."""
    recs = [r for r in extract_from_patent(_xml("US9688680")) if r.letter_grade]
    b = {(r.assay_name, r.letter_grade): (r.range_lo, r.range_hi)
         for r in recs if r.range_lo is not None or r.range_hi is not None}
    assert b[("D816V IC50 (nM)", "B")] == (1.01, 10.0)
    assert b[("WT IC50 (nM)", "B")] == (11.0, 100.0)


def test_a_caption_names_the_column_a_whole_legend_block_governs():
    """US9221791 publishes its two scales as separate `value | Rating` tables,
    captioned `... the Septoria rating scale is as follows:` and the same for
    Puccinia. Nothing inside either legend says which column it means."""
    recs = [r for r in extract_from_patent(_xml("US9221791")) if r.letter_grade]
    b = {(r.assay_name, r.letter_grade): (r.range_lo, r.range_hi, r.unit)
         for r in recs if r.range_lo is not None or r.range_hi is not None}
    assert b[("Puccinia Rating", "C")] == (40.0, 59.0, "%")
    assert b[("Septoria Rating", "C")] == (1.5, 4.0, "ug/mL")


def test_a_selectivity_ratio_never_lands_on_a_concentration_column():
    """US12351648 defines `*`-`****` twice — once for Ki in μM and once for a
    MASP-2-versus-thrombin selectivity in fold. The fold scale reached three
    columns headed `Ki (μM)`. The column header said μM the whole time."""
    recs = [r for r in extract_from_patent(_xml("US12351648")) if r.letter_grade]
    assert recs, "no graded records"
    for r in recs:
        if r.unit == "fold":
            assert "selectivity" in (r.assay_name or "").lower(), \
                f"a fold range landed on {r.assay_name!r}"


# ── G. what the column measures, versus what it was measured AT ──────────
#
# These live beside the bin tests because they are the same question asked of
# the other half of the record. A bin key states a dimension and a column
# states a dimension, and the reader has to get both right before it can
# compare them — see `compatible`.

@pytest.mark.parametrize("header,percent", [
    # The `%` stands alone: it is the unit of the value.
    ("MAGL % Inh 1 μM (mouse)", True),
    ("BACE1 inhibition at 10 μM (%)", True),
    ("CYP 450 % INH @ 10 μM", True),
    ("micro-somal stability Cl [% Qh]", True),
    ("ROCK2 % inh. @ [conc]", True),
    # The `%` is bound to a number: it states a THRESHOLD, and the value is
    # the concentration at which that threshold is reached. US9987276's
    # `>50% occupancy` column holds nM and is right to.
    (">50% occupancy", False),
    (">90% occupancy", False),
    # The patent itself says the column holds either. Neither answer fits
    # every row, so its own hedge stands.
    ("FKBP12 Ki (μM) or %", False),
    # No percent at all.
    ("BACE1 Ki (nM)", False),
])
def test_a_percent_sign_is_a_unit_only_when_it_is_not_a_threshold(header, percent):
    from patentdb3.sources.uspto_assays import _percent_header
    assert _percent_header(header) is percent


def test_a_percentage_column_is_not_labelled_with_the_concentration_it_ran_at():
    """`at 10 μM (%)` holding 104.0 read as `104 uM` — a dead compound rather
    than complete inhibition, from the same digits. The first unit token in
    the header is the assay's CONDITION, not the value's unit."""
    recs = [r for r in extract_from_patent(_xml("US10004738"))
            if "(%)" in (r.assay_name or "")]
    assert recs, "no percent column found"
    assert {r.unit for r in recs} == {"%"}


def test_a_percentage_column_is_still_an_assay():
    """The assay gate reads `is_assay or (unit and unit != "%")`, so a column
    whose only claim was a unit now correctly read as `%` falls through it.
    That cost 113 records — `% Effect at 30 μM relative to 2'3'-cGAMP` and
    `% amount of pSer376-SLP-76 @ 20 μM` are assay results by any reading."""
    for pid, needle in [("US10730849", "% Effect"),
                        ("US11427578", "% amount of")]:
        recs = [r for r in extract_from_patent(_xml(pid))
                if needle in (r.assay_name or "")]
        assert recs, f"{pid}: {needle!r} column produced nothing"
        assert {r.unit for r in recs} == {"%"}


# ── H. a label the document never wrote ──────────────────────────────────

def test_a_column_named_only_by_its_data_says_so():
    """A headerless column of grades is almost certainly an assay, and what it
    MEASURED is not knowable from its cells. The marker is what turns that into
    a question for the heal loop instead of an assertion."""
    from patentdb3.sources.uspto_assays import classify_column
    bare = classify_column("", ["E"] * 12 + ["A"])
    assert bare.label_source == "shape"
    named = classify_column("BACE1 IC50 (nM)", ["1.2", "3.4"])
    assert named.label_source == "header"


def test_the_column_cache_does_not_drop_fields():
    """The memoised classifier rebuilt its result field by field, so a field
    added to `Column` was silently replaced by its default on every cached
    call. `label_source` was set correctly by the classifier and read as
    `header` by every caller, which made the flag above do nothing at all."""
    from patentdb3.sources.uspto_assays import classify_column
    samples = ["E"] * 12 + ["A"]
    first = classify_column("", samples)          # computed
    second = classify_column("", samples)         # served from the cache
    assert first.label_source == second.label_source == "shape"
    assert second is not first, "callers must not share one mutable column"


# ── I. the header rows a column is named from ────────────────────────────

@pytest.mark.parametrize("parts,expected", [
    # A hyphenated NAME keeps its hyphen: the continuation starts a new word.
    (["DNA-", "PK"], "DNA-PK"),
    (["BACE-", "1 Ki"], "BACE-1 Ki"),
    # A word the typesetter BROKE across rows loses it: the continuation is
    # lower-case. `Meth-od` is invisible to a pattern matching `\bmethod\b`,
    # so US9611261's synthesis column read as an assay for 288 records.
    (["Meth-", "od"], "Method"),
    (["Ex-", "ample", "#"], "Example #"),
    (["Inter-", "me-", "diate"], "Intermediate"),
    (["prolifer-", "ation"], "proliferation"),
    # No hyphen: stacked rows join with a space.
    (["AAK1 IC50", "(nM)"], "AAK1 IC50 (nM)"),
])
def test_a_broken_word_is_rejoined_and_a_hyphenated_name_is_not(parts, expected):
    from patentdb3.sources.uspto_assays import _join_header_lines
    assert _join_header_lines(parts) == expected


@pytest.mark.parametrize("text,is_title", [
    # A designator is not always a number, and a pattern that reads only
    # digits lets the rest through as though it named a column — which put
    # `TABLE 4A` on the front of an assay name and, on US10870641, replaced
    # `IC50` outright.
    ("TABLE 1", True), ("TABLE A", True), ("TABLE 1A", True),
    ("TABLE A-1", True), ("TABLE 37B", True), ("TABLE-US-00003", True),
    # Real column names, including one that opens with the word.
    ("AAK1 IC50", False), ("elF4E", False), ("IC50 (nM)", False),
    ("TABLE 1A FRET_ _IC50 (uM", False),
])
def test_a_table_title_is_not_a_column_name(text, is_title):
    from patentdb3.sources.uspto_xml import _ROW_TITLE
    assert bool(_ROW_TITLE.match(text)) is is_title


def test_a_header_row_naming_one_column_is_kept():
    """A multi-row header puts the assay on one row and its unit on the next,
    filling only the cell above the column it describes:

        ['', '',        'AAK1 IC50']
        ['', 'Example', '(nM)'     ]

    The row test required two populated cells, so the first row was dropped
    and the column was named `(nM)` — the unit kept, the assay lost."""
    from patentdb3.sources.uspto_assays import (
        ASSAY, _best_per_block, build_columns, merge_header)
    from patentdb3.sources.uspto_xml import parse_tables
    for t in _best_per_block(parse_tables(_xml("US10544120"))):
        if t.table_id != "TABLE-US-00001":
            continue
        assert merge_header(t)[2] == "AAK1 IC50 (nM)"
        assay = [c for c in build_columns(t) if c.kind == ASSAY]
        assert [c.assay_name for c in assay] == ["AAK1 IC50 (nM)"]
        return
    pytest.skip("US10544120 TABLE-US-00001 not present")


def test_a_synthesis_method_column_is_not_an_assay():
    """US9611261 heads a column `Meth-`/`od` over single letters naming which
    general procedure made each compound. 288 records of synthesis metadata
    presented as measurements."""
    from patentdb3.sources.uspto_assays import extract_from_patent as ex
    labels = {r.assay_name for r in ex(_xml("US9611261"))}
    assert not any("meth" in (a or "").lower() for a in labels), \
        f"a method column is still an assay: {labels}"


# ── J. legend forms found by the corpus-wide audit ───────────────────────

@pytest.mark.parametrize("text,expected", [
    # The whole INTERVAL first, the symbol assigned after it. US11229631 states
    # its scale this way in a footer; Form 4 reads `<value> = <symbol>` and
    # stops at the first number, so a bound on each side left it nothing.
    ("1000 nM < IC50 ≤ 10000 nM: +++",
     {"+++": (1000.0, 10000.0, "nM")}),
    # The symbol, then an interval whose middle names an ARBITRARY metric —
    # here the assay's own name rather than one of IC50/EC50/Ki/Kd.
    ("A 0 < PI3K Delta Activity < 50 nM", {"A": (0.0, 50.0, "nM")}),
    # A bound whose direction is stated AFTER the value.
    ("“D” represents a calculated IC50 value of 1 μM or greater",
     {"D": (1.0, None, "uM")}),
    # A span written high-to-low. Read in order it inverts.
    ('from 1 to 0.05 μM are labelled as "++"', {"++": (0.05, 1.0, "uM")}),
    # A verb this file did not know. US9987276 defines two scales this way.
    ('"A" provided an IC50 ≤10 nM', {"A": (None, 10.0, "nM")}),
])
def test_the_forms_the_audit_found(text, expected):
    assert _bins(text) == expected


def test_a_minus_grade_does_not_swallow_the_grade_before_it():
    """U+2212 MINUS is a grade: a two-level scale writes `+` and `−`. Leaving
    it out cost twice — its own rows were dropped, and the prose body for `+`
    ran straight through `− indicates ≥10 μm`, read BOTH bounds, and overwrote
    `≤10 μM` with the point interval `10..10`. US10953012 shipped 270 records
    saying a compound is exactly 10 μM."""
    got = _bins("IC50 Kinase Domain + indicates ≤10 μm − indicates ≥10 μm")
    assert got["+"] == (None, 10.0, "uM")
    assert got["−"] == (10.0, None, "uM")


@pytest.mark.parametrize("text,expected", [
    # The ASCII hyphen and U+2212 are still range separators between numbers.
    ("A: 10−50 nM", {"A": (10.0, 50.0, "nM")}),
    ("A: IC 50 >200 nM−<800 nM", {"A": (200.0, 800.0, "nM")}),
])
def test_a_minus_between_two_numbers_is_still_a_range(text, expected):
    assert _bins(text) == expected


def test_no_compound_is_given_a_point_interval_by_a_bin_key():
    """A bin is an interval. `lo == hi` is not a measurement the patent made —
    it is two bounds read out of two different grades' clauses."""
    from patentdb3.core import config
    from patentdb3.sources.uspto_assays import extract_from_patent
    for pid in ("US10953012", "US10172859", "US9656988"):
        path = config.XML_INPUT_DIR / f"{pid}.xml"
        if not path.exists():
            continue
        for r in extract_from_patent(path.read_text(errors="ignore")):
            if r.letter_grade and r.range_lo is not None and r.range_hi is not None:
                assert r.range_lo < r.range_hi, \
                    f"{pid} {r.cid} {r.letter_grade}: {r.range_lo}-{r.range_hi}"


# ── K. one physical column, two assays: the header's prefix is shared ────

def test_a_two_row_header_gives_its_prefix_to_every_sub_assay():
    """US9302989 TABLE-US-00001 spans its header over two rows and merges to
    `TR-FRET Binding IC50 (uM) probe 1, probe 2`. The cell below it is also
    comma-separated (`0.00309, 0.00252`), so this is one physical column
    naming TWO assays. `split_top_level` splits correctly on the top-level
    comma, but the shared prefix sits ahead of the FIRST sub-name only, so
    the second assay came out named bare `probe 2` — a reader of the output
    could not tell what it measured. The fix reads the shared prefix off the
    first part's tail (both parts end in "probe <digit>", only the digit
    differs) and gives it back to every part after it.
    """
    got = _redistribute_shared_prefix(
        ["TR-FRET Binding IC50 (μM) probe 1", "probe 2"]
    )
    assert got == [
        "TR-FRET Binding IC50 (μM) probe 1",
        "TR-FRET Binding IC50 (μM) probe 2",
    ]


def test_the_prefix_is_left_alone_when_the_shape_does_not_match():
    """Two independent assay names, not one column split in two — nothing
    about `IC50` and `EC50` says one is a truncated copy of the other, so the
    function must not invent a prefix for either. Same for a header whose
    tail varies by a WORD (`mutant A` / `mutant B`) rather than a digit: a
    letter could be a real word change, not an index, so the safe move is to
    skip the rename rather than guess it.
    """
    assert _redistribute_shared_prefix(["IC50", "EC50"]) == ["IC50", "EC50"]
    assert _redistribute_shared_prefix(["Ki (nM) mutant A", "mutant B"]) == \
        ["Ki (nM) mutant A", "mutant B"]


# ── L. a grade assignment is data, even without its scale ────────────────

def test_a_chemical_name_is_not_a_list_of_compound_ids():
    """`_cid_list` pulls ids out of any text, which is what it is for. A
    chemical name carries a locant run, and
    `[4-Fluoro-3-[7-(2,2,3,3,5,5,6,6-octadeuterio-morpholin-4-...` yields six
    "compounds" that are ring positions. Beside a graded cell in the same row
    that satisfied the inverted-table shape and minted 54 records for an
    ordinary row-per-compound table. An inverted table's cell IS the list."""
    from patentdb3.sources.uspto_assays import _is_id_list_cell
    assert _is_id_list_cell("A028, A075, A076, A087, A112")
    assert _is_id_list_cell("1, 2, 3, 4, 5")
    assert not _is_id_list_cell(
        "[4-Fluoro-3-[7-(2,2,3,3,5,5,6,6-octadeuterio-morpholin-4-yl)quinazolin")


def test_an_inverted_table_needs_the_grade_beside_the_list():
    """The grade and the compounds it applies to are printed side by side.
    Looking for a symbol anywhere in a block and an id list anywhere else
    matches a normal graded table too."""
    from patentdb3.sources.uspto_assays import _best_per_block, _is_inverted_block
    from patentdb3.sources.uspto_xml import parse_tables
    blocks = {}
    for t in parse_tables(_xml("US11566007")):
        blocks.setdefault(t.table_id, []).append(t)
    assert _is_inverted_block(blocks["TABLE-US-00009"])
    normal = {}
    for t in parse_tables(_xml("US10172859")):
        normal.setdefault(t.table_id, []).append(t)
    assert not _is_inverted_block(normal["TABLE-US-00009"]), \
        "a row-per-compound table is not an inverted one"


def test_a_block_whose_scale_is_out_of_reach_still_yields_its_grades():
    """US11566007's inverted tables carry hundreds of compound ids inline, so
    for the later blocks the key — printed before the PREVIOUS table — sits
    beyond the 6,000-character look-back and is never seen. TABLE-US-00008
    finds it and yielded 825 records; TABLE-US-00009 has the identical layout,
    does not find it, and yielded 0.

    The repair loop diagnosed that itself and refused to buy a rule for it:
    'INCONSISTENT HANDLING — not a layout gap'. The range is a SECOND fact
    about a grade, not a precondition for admitting the assignment."""
    recs = extract_from_patent(_xml("US11566007"))
    for tid in ("TABLE-US-00009", "TABLE-US-00010", "TABLE-US-00012"):
        block = [r for r in recs if r.table_id == tid]
        assert len(block) > 500, f"{tid} yielded {len(block)}"
        assert all(r.letter_grade for r in block), "every record carries its grade"


# ── M. a flag must cover every way the thing it flags is made ────────────

def test_every_placeholder_name_the_reader_mints_is_one_it_flags():
    """`label_source` marked the placeholders minted by the column classifier
    and missed `assay (binned)`, minted by the inverted-table path — so a block
    of 818 records with no name at all raised nothing, while a single unnamed
    column raised a gap. Keying a flag on WHICH CODE PATH produced something is
    how it silently covers only some of them.

    This asserts the reverse: every literal the module uses as a fallback assay
    name is in the vocabulary the detector flags on. A new fallback added at an
    eighth call site fails here rather than going unnoticed."""
    import re as _re
    from pathlib import Path

    from patentdb3.sources.uspto_assays import PLACEHOLDER_ASSAY_NAMES
    src = Path(__file__).resolve().parents[1] / "sources" / "uspto_assays.py"
    text = src.read_text()
    minted = set(_re.findall(r'assay_name=(?:[^,\n]*?or\s+)?"([^"]+)"', text))
    unflagged = {m for m in minted
                 if ("unnamed" in m.lower() or "binned" in m.lower())
                 and m not in PLACEHOLDER_ASSAY_NAMES}
    assert not unflagged, (
        f"minted as a fallback but not in PLACEHOLDER_ASSAY_NAMES: {unflagged}")


def test_a_block_of_unnamed_records_is_raised_for_the_loop():
    """The column-based detector asks this of COLUMNS, and columns are not the
    only way a name is minted — the inverted path names a whole block at once.
    The signal is the name, not the code path."""
    from patentdb3.repair.gap import find_gaps
    from patentdb3.sources.uspto_xml import assemble_blocks, parse_tables
    xml = _xml("US11566007")
    gaps = find_gaps("US11566007", assemble_blocks(parse_tables(xml)),
                     extract_from_patent(xml), _source_xml=xml)
    hit = [g for g in gaps
           if g.table_id == "TABLE-US-00005" and "placeholder name" in (g.reason or "")]
    assert hit, "a block whose every record is unnamed must reach the loop"
    assert hit[0].asks == "column_names", \
        "it must ask for names, or the loop answers a scale question instead"


def test_a_caption_naming_another_table_is_not_used():
    """The caption is the last <p> before the block, which is a guess. 802 of
    the 1,820 blocks that state their own title were captioned with a different
    table's text, every one off by one. It supplies the assay name and the unit
    hint, so US11566007 TABLE-US-00008 — `KRAS G13C FRET data` — was captioned
    `TABLE 7 KRAS G12D FRET data`: a different mutant, which nothing downstream
    can catch."""
    from patentdb3.sources.uspto_xml import parse_tables
    names = {r.table_id: r.assay_name
             for r in extract_from_patent(_xml("US11566007"))}
    assert "G13C" in (names.get("TABLE-US-00008") or ""), names.get("TABLE-US-00008")
    assert "G12D" not in (names.get("TABLE-US-00008") or "")
    assert "G12D" in (names.get("TABLE-US-00007") or ""), names.get("TABLE-US-00007")


# ── N. what survives when a repair and the reader both read a row ────────

def test_a_repair_that_adds_a_range_replaces_the_copy_without_one():
    """The merge kept a usable copy over an unusable one, which holds only
    when the repair turns unusable into usable. It does not hold for the
    commonest repair of all — a `bin_key` takes a grade the reader already read
    and ADDS its numeric range. Both copies are usable, so neither was dropped
    and every record shipped twice."""
    from patentdb3.repair.loop import _informative
    from patentdb3.sources.uspto_assays import AssayRecord
    bare = AssayRecord(cid="1", assay_name="IC50", letter_grade="B", unit="nM")
    ranged = AssayRecord(cid="1", assay_name="IC50", letter_grade="B", unit="nM",
                         range_lo=3.0, range_hi=7.0)
    assert _informative(ranged) > _informative(bare)


def test_the_same_fact_twice_is_one_fact():
    """Ranking settles which copy to prefer; it does not settle two copies that
    say the SAME thing, and that is the common case once the reader and a
    bought rule agree. Collapsed on the FACT — value, grade, interval, unit —
    never on which produced it, so two genuinely different readings both ship.
    """
    import tempfile, shutil
    from pathlib import Path
    from patentdb3.repair.loop import repair_patent
    from patentdb3.repair.rules import RuleLibrary
    lib = RuleLibrary()
    if not lib._rules:
        pytest.skip("no rule library on disk")
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(lib.path, tmp / "r.json")       # never mutate the tracked file
    recs, _ = repair_patent("US11547697", _xml("US11547697"),
                            library=RuleLibrary(path=tmp / "r.json"),
                            max_calls=0, journal=tmp / "j.jsonl")
    block = [r for r in recs if r.table_id == "TABLE-US-00002"]
    facts = {(r.cid, r.assay_name, r.value_numeric, r.letter_grade,
              r.range_lo, r.range_hi, r.unit) for r in block}
    assert len(facts) == len(block), \
        f"{len(block) - len(facts)} identical facts shipped more than once"
