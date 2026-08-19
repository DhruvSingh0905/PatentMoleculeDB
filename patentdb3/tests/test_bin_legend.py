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
from patentdb3.sources.uspto_assays import extract_from_patent


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
