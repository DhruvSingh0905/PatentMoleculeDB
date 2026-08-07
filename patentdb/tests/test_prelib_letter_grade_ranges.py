"""A letter grade is a RANGE. The pattern-library tier used to ship it as 0.0.

`core/assay_fsm/assay_pattern_library.py:800-814` emits a row with
``value=None`` + ``value_categorical`` whenever a table cell will not
``float()`` — a `+`/`++` potency bin, a censored `>50000`, a `1.6 ± 0.1`
mean±SD, or the patent's own em-dash placeholder.  `ActivityTuple.from_dict`
(`harvest/agent2_activities.py:62-64`) returns None for a null value, so
`harvest/orchestrator.py` substituted **0.0** to keep the row alive and
appended the literal to `validation_reason`, which ships as `source`:

    {"assay_name": "CBP IC50 (μM gmean)", "value_numeric": 0.0, "unit": "μM",
     "source": "pattern_library:86590369654b717b:letter_grade categorical=+"}

Measured on the Aug-6 corpus artifacts (22 patents, 84,517 rows): 2,931 rows
carry that shape.  A consumer reading `value_numeric` sees `CBP IC50 = 0.0 μM`
— maximal potency — where the patent said "somewhere in 1-1000 μM".  For the
same cell of the same compound the `uspto_xml_table` row already ships
``value_low=1.0, value_high=1000.0, value_raw="+"``.

The project's own encoding for a bin is `routes/letter_bin_assays.py` and
`sources/uspto_assays.parse_value` + `sources/bin_legend.parse_bin_key`
(CLAUDE.md:16).  These tests pin the pattern-library tier to it.

Commit `f692dc8` reversed 109 fabricated geometric-mean midpoints on
US11292791 for exactly this reason — a bin does not contain a point value.
"""
from __future__ import annotations

import pytest

from patentdb.core.assay_fsm.harvest.orchestrator import (
    HarvestResult,
    _encode_prelib_value,
    _prelib_bin_key,
)
from patentdb.core.assay_fsm.harvest.agent2_activities import ActivityTuple


# The legend US11292791 actually prints, verbatim from the GP flat
# description (`_gather_full_text("US11292791")`).  The bin edges must be
# read from THIS, never hard-coded — two patents in the corpus assign
# incompatible ranges to `++++` (see sources/bin_legend.parse_bin_key).
US11292791_LEGEND = (
    "As set forth in Table 19 below, an IC 50 value of greater than or equal "
    "to 0.001 μM and less than or equal to 0.01 μM is marked "
    "“++++”; a value greater than 0.01 μM and less than or "
    "equal to 0.1 μM is marked “+++”; a value greater than 0.1 "
    "μM and less than or equal to 1 μM is marked “++”; and "
    "a value greater than 1 μM and less than 1000 μM is marked "
    "“+.” Compounds that were not tested in a particular assay are "
    "marked “NT.”"
)


@pytest.fixture(scope="module")
def bin_key():
    return _prelib_bin_key(US11292791_LEGEND)


# ── the defect itself ───────────────────────────────────────────────────────

def test_plus_bin_is_the_legend_range_not_zero(bin_key):
    """`+` on US11292791 means 1-1000 uM. It shipped as 0.0."""
    enc = _encode_prelib_value("+", "μM", bin_key)
    assert enc is not None
    assert enc["value_numeric"] is None, "a bin has no point value"
    assert enc["value_low"] == 1.0
    assert enc["value_high"] == 1000.0
    assert enc["value_raw"] == "+", "the literal grade, as uspto_xml ships it"
    assert enc["bin"] == "+"


def test_every_grade_in_the_legend_maps_to_its_own_bracket(bin_key):
    expect = {
        "+": (1.0, 1000.0),
        "++": (0.1, 1.0),
        "+++": (0.01, 0.1),
        "++++": (0.001, 0.01),
    }
    for grade, (lo, hi) in expect.items():
        enc = _encode_prelib_value(grade, "μM", bin_key)
        assert enc is not None, grade
        assert (enc["value_low"], enc["value_high"]) == (lo, hi), grade
        assert enc["value_numeric"] is None, grade


def test_bin_row_matches_the_uspto_xml_row_field_for_field(bin_key):
    """The correct representation already ships beside the fabricated one.

    Verbatim from output_v2/text_extraction/US11292791/assay_tables.json,
    cid "1", source uspto_xml_table.  The pattern-library row must not
    invent a second, contradictory spelling of the same measurement.
    """
    enc = _encode_prelib_value("+", "μM", bin_key)
    for field, want in (
        ("value_numeric", None),
        ("qualifier", None),
        ("n_runs", None),
        ("value_low", 1.0),
        ("value_high", 1000.0),
        ("value_raw", "+"),
        ("bin", "+"),
        ("unit", "uM"),
        ("unit_source", "bin_key"),
    ):
        assert enc[field] == want, field


# ── the other three shapes hiding in the same 2,931 rows ────────────────────

@pytest.mark.parametrize("raw,num,qual", [
    (">30", 30.0, ">"),          # US10246453, 266 rows, no other row for the pair
    (">10", 10.0, ">"),          # US9745328 / US10899738
    ("<0.004", 0.004, "<"),      # US9718790 — a POTENT compound shipped as 0.0
    (">50000", 50000.0, ">"),    # US10544143
    ("~2.49", 2.49, "~"),
])
def test_censored_value_keeps_its_number_and_qualifier(raw, num, qual, bin_key):
    enc = _encode_prelib_value(raw, "nM", bin_key)
    assert enc is not None, raw
    assert enc["value_numeric"] == num
    assert enc["qualifier"] == qual
    assert enc["value_raw"] == raw


@pytest.mark.parametrize("raw,num", [("&gt;50000", 50000.0), ("&gt;3125", 3125.0)])
def test_html_entity_qualifier_is_unescaped_before_parsing(raw, num, bin_key):
    """GP flat text ships `&gt;`. 84 corpus rows are this shape."""
    enc = _encode_prelib_value(raw, "nM", bin_key)
    assert enc is not None, raw
    assert enc["value_numeric"] == num
    assert enc["qualifier"] == ">"


@pytest.mark.parametrize("raw,num", [("1.6 ± 0.1", 1.6), ("0.009 ± 0.0004", 0.009)])
def test_mean_sd_keeps_the_mean(raw, num, bin_key):
    """60 US10899738 rows read `X ± Y`; every one shipped as 0.0 μM."""
    enc = _encode_prelib_value(raw, "μM", bin_key)
    assert enc is not None, raw
    assert enc["value_numeric"] == num


@pytest.mark.parametrize("raw", [
    "â",            # US10214537: UTF-8 em-dash read as Latin-1 (375 rows)
    "—",            # the em-dash itself — the patent's "not tested"
    "No inhibition",     # US10899738, 37 rows
    "NT", "nd", "nt",
])
def test_a_non_value_is_dropped_not_zeroed(raw, bin_key):
    """`—` is the patent saying "no measurement". 0.0 nM says the opposite."""
    assert _encode_prelib_value(raw, "nM", bin_key) is None


def test_grade_without_a_legend_is_dropped(bin_key):
    """`missing_fields` (sources/uspto_assays.py:452): a grade with no key
    and no bounds is not a usable measurement. Guessing a scale is worse
    than dropping — parse_bin_key refuses to for the same reason."""
    assert _encode_prelib_value("+", "μM", {}) is None
    assert _encode_prelib_value("B", "μM", {}) is None


# ── what actually ships ─────────────────────────────────────────────────────

def test_to_assay_results_ships_the_range_not_a_zero(bin_key):
    t = ActivityTuple(compound_id="1", assay_name="CBP IC50 (μM gmean)",
                      value=0.0, unit="μM")
    t.range_fields = _encode_prelib_value("+", "μM", bin_key)
    out = HarvestResult(tuples=[t]).to_assay_results()
    (row,) = out["1"]
    assert row.value_numeric is None
    assert row.value_low == 1.0
    assert row.value_high == 1000.0
    assert row.value_raw == "+"
    assert row.bin == "+"


def test_a_plain_numeric_row_is_untouched():
    t = ActivityTuple(compound_id="7", assay_name="IC50", value=12.5, unit="nM")
    out = HarvestResult(tuples=[t]).to_assay_results()
    (row,) = out["7"]
    assert row.value_numeric == 12.5
    assert row.value_raw == "12.5"
    assert row.value_low is None and row.value_high is None and row.bin is None


def test_the_artifact_row_carries_the_range(bin_key):
    """Producer → artifact. `_run_text_dominant` rebuilt each row with six
    keys and dropped `value_raw`/`value_low`/`value_high`, so even a correct
    AssayResult would have shipped as a bare `value_numeric`.
    """
    import json

    from patentdb.routes.process_patent import _harvest_row_dict

    t = ActivityTuple(compound_id="1", assay_name="CBP IC50 (μM gmean)",
                      value=0.0, unit="μM")
    t.range_fields = _encode_prelib_value("+", "μM", bin_key)
    (res,) = HarvestResult(tuples=[t]).to_assay_results()["1"]
    row = json.loads(json.dumps(_harvest_row_dict(res)))
    assert row["value_numeric"] is None
    assert (row["value_low"], row["value_high"]) == (1.0, 1000.0)
    assert row["value_raw"] == "+" and row["bin"] == "+"
    assert row["unit_source"] == "bin_key"


def test_a_plain_numeric_artifact_row_keeps_the_six_key_shape():
    """41,526 shipped rows are this shape. Adding bin keys to all of them
    would be a schema change no consumer asked for."""
    from patentdb.routes.process_patent import _harvest_row_dict

    t = ActivityTuple(compound_id="7", assay_name="IC50", value=12.5, unit="nM")
    (res,) = HarvestResult(tuples=[t]).to_assay_results()["7"]
    assert set(_harvest_row_dict(res)) == {
        "assay_name", "value_numeric", "unit", "qualifier", "n_runs", "source",
    }


def test_no_shipped_row_pairs_a_categorical_source_with_a_zero(bin_key):
    """The regression sentinel: the 2,931-row shape must not come back.

    Any row whose provenance says `letter_grade` must carry either a
    number the patent printed or a range, never `value_numeric == 0.0`.
    """
    tuples = []
    for raw in ("+", "++", ">30", "&gt;50000", "1.6 ± 0.1", "â", "NT"):
        enc = _encode_prelib_value(raw, "μM", bin_key)
        if enc is None:
            continue
        t = ActivityTuple(compound_id="1", assay_name="IC50", value=0.0,
                          unit="μM",
                          validation_reason=f"pattern_library:k:letter_grade categorical={raw}")
        t.range_fields = enc
        tuples.append(t)
    rows = [r for arr in HarvestResult(tuples=tuples).to_assay_results().values()
            for r in arr]
    assert rows, "encoder dropped everything — check the fixture"
    for r in rows:
        if "letter_grade" not in r.source:
            continue
        assert not (r.value_numeric == 0.0), f"fabricated zero: {r}"
        assert (r.value_numeric is not None
                or r.value_low is not None
                or r.value_high is not None), f"no value at all: {r}"
