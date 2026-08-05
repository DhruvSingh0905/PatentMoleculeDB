"""A NATIVE pattern stamping its label onto a different table of its own patent.

`test_pattern_library_leak.py` covers the FOREIGN case: a label learned on
patent A fired on patent B. The gate it installed is scoped to foreign entries
and consults `first_seen_patent`, so it cannot see this one — here the label IS
the patent's own, read off a header this document really prints. What travels
wrongly is not the label's provenance but its REACH: the row-regex is purely
structural (`I-\\d+` then a decimal) and matches every "id, then number" table
in the document, including the ones that publish retention times and masses.

Measured on this checkout (2026-08-04) over the 6 corpus patents whose library
patterns fire — 62,566 native rows, graded against each patent's OWN CALS
tables via `sources.uspto_assays.extract_from_patent`:

    row sits under a header that…      rows    corroborated   contradicted
    names this assay                 40,017        32,382          2,678   92.4%
    names a DIFFERENT measurement    18,422           888          8,545    9.4%
    (retention time / [M+H] / m/z /
     MW / QC method / NMR)

Two concrete fabrications, both native, both from the largest emitters:

    US9718790   TABLE 141  "Retention Compound Time Structure No. (min)
                            [M + H] Method"
                I-0687 1.86 580 2   ->  I-0687, P2X3 IC50 = 1.86 μM
                (1.86 is MINUTES of retention time)

    US10214537  TABLE 3    "Ex. LCMS No. R Name (M + H) +"
                4 21 2-(4-acetyl-…  ->  compound 4, CD69 IC50 = 2 nM
                (the "2" is the first character of an IUPAC NAME)

WHY DISTANCE IS THE WRONG YARDSTICK, and why this file does not test one.
The obvious gate — "a row must sit within 20,000 chars of its anchor", the
threshold the foreign gate already uses — was measured against the same ground
truth and rejected:

    gate                       drops    verified WRONG   verified CORRECT lost
    locality <= 20k           20,886           7,560                   4,366
    non-assay local header    15,112           6,135                       4

US9694016 is the counter-example that settles it: 2,689 of its rows sit more
than 20,000 chars from their anchor — up to 28,844 — and 2,688 of them are
CORRECT. They are the tail of one enormous `Example NNN 0.00280 0.00050` table
that simply runs longer than 20 KB. A distance gate deletes all of them; the
header rule costs that patent nothing, because nothing else is printed in
between.

The four "correct" rows the header rule loses are all US10214537 name-table
matches whose captured value is the leading digit of the IUPAC name that
follows; they score correct only because CD69 IC50 readings are frequently
single digits. Verified real collateral: zero.

NOT the same defect, and measured here so it is not conflated: US9718790's 161
BindingDB-bad compounds do not come from this module. `repair/value_check.py`
scores `sources.uspto_assays.extract_from_patent` on the XML — the CALS path —
and never sees a pattern-library row. Its mechanism is `TABLE-US-00569`, three
`(Compound No., P2X3 IC50)` column PAIRS per row, read as one compound with
three values: `I-0268 0.861 I-0943 0.061 I-1607 0.035` becomes I-268 = {0.861,
0.061, 0.035}, the scorer takes the median 0.061, and BindingDB says 861 nM.
I-943 and I-1607 get no record at all. That is a column-pair defect in
`sources/uspto_assays.py`.

So the test below fixes the DEFECT (a row read out of a mass-spec table) and
deliberately pins the NON-defect (a correct row 25,000 chars downstream) so a
future distance-shaped fix fails here.
"""
from __future__ import annotations

import json

import pytest

from patentdb.core.assay_fsm import assay_pattern_library as lib


# ── fixtures ──────────────────────────────────────────────────────
#
# Hermetic. Each is the measured shape of one real case: the same regex from
# `patentdb/data/assay_patterns.discoveries.json`, the same `column_assays`,
# and header + row text copied from the patent that exhibited it.

# US9718790's highest-yield entry (5,129 rows).
_P2X3_PATTERN = {
    "key": "1f3ceacc4dd0c6d0",
    "regex": r"(?P<cid>I-\d+)\s+(?P<value0>[\d.]+)",
    "column_assays": ["P2X3 IC50 (μM)"],
    "header_text": "",
    "example_match": "I-2300 0.003",
    "status": "pending",
    "fingerprints_observed": ["US9718790"],
    "first_seen_patent": "US9718790",
    "n_observations": 4,
}

# US10214537's highest-yield entry.
_CD69_PATTERN = {
    "key": "f74170a7e3509c03",
    "regex": r"(?P<cid>\d+)\s+(?P<value0>[\d.]+)\s+(?P<value1>[\d.]+)",
    "column_assays": ["PI3K delta IC50 (nM)", "CD69 IC50 (nM)"],
    "header_text": "",
    "example_match": "638 2 68",
    "status": "pending",
    "fingerprints_observed": ["US10214537"],
    "first_seen_patent": "US10214537",
    "n_observations": 3,
}

# US9718790's own potency table, then — 60 KB later — its LC/MS tables.
# Both blocks are verbatim shapes from the patent.
_P2X3_ASSAY_BLOCK = (
    "TABLE 569\n"
    "Compound P2X3 Compound P2X3 Compound P2X3\n"
    "No. IC50 (μM) No. IC50 (μM) No. IC50 (μM)\n"
    "I-2653 0.009 I-2654 0.009 I-2655 0.005\n"
    "I-2656 0.006 I-2657 0.007 I-2658 0.005\n"
)
_P2X3_LCMS_BLOCK = (
    "TABLE 141\n"
    "Retention Compound Time Structure No. (min) [M + H] Method\n"
    "I-0686 1.91 459 3\n"
    "I-0687 1.86 580 2\n"
    "I-0688 2.51 622 3\n"
)
_FILLER = "The compounds of the invention were prepared as described. " * 1100


def _install(tmp_path, monkeypatch, *entries: dict) -> None:
    path = tmp_path / "assay_patterns.discoveries.json"
    path.write_text(json.dumps(
        {"schema_version": "1.0", "tokens": list(entries)}, indent=1))
    monkeypatch.setattr(lib, "_PATTERNS_PATH", path)


@pytest.fixture
def library(tmp_path, monkeypatch):
    def _f(*entries: dict) -> None:
        _install(tmp_path, monkeypatch, *entries)
    return _f


# ── the defect ────────────────────────────────────────────────────

def test_retention_time_is_not_a_micromolar_potency(library):
    """US9718790: `I-0687 1.86` under an LC/MS header ships as 1.86 μM.

    The pattern is NATIVE — `P2X3 IC50 (μM)` is this patent's own column, off
    this patent's own TABLE 569 — so the foreign gate never looks at it. The
    row-regex then matches the retention-time tables too, and 14,293 of the
    patent's 38,666 pattern rows are read out of them. 1.86 is minutes.
    """
    library(_P2X3_PATTERN)
    text = _P2X3_ASSAY_BLOCK + _FILLER + _P2X3_LCMS_BLOCK

    rows = lib.apply_patterns_to_text(text, "US9718790")

    fabricated = [r for r in rows if r["compound_id"].startswith("I-068")]
    assert not fabricated, (
        "a retention time is being reported as a micromolar potency: "
        f"{fabricated[0]}"
    )


def test_the_first_character_of_a_name_is_not_a_potency(library):
    """US10214537: `4 21 2-(4-acetyl-…` under `Ex. LCMS No. R Name (M + H) +`.

    Three columns of a NAME table — an [M+H] mass, an example number, and the
    leading digit of the IUPAC name that follows — read as two potencies.
    150 of this patent's rows are this shape.
    """
    library(_CD69_PATTERN)
    text = (
        "TABLE 47\n"
        "Ex. No. PI3K delta IC50 (nM) CD69 IC50 (nM)\n"
        "638 2 68\n"
        "639 5 91\n"
        + "The following compounds were prepared analogously. " * 900
        + "TABLE 3\n"
        "Ex. LCMS No. R Name (M + H) +\n"
        "4 21 2-(4-acetyl-3,3-dimethyl-2-oxopiperazin-1-yl)-4-(4-amino"
        "pyrrolo[2,1-f][1,2,4]triazin-7-yl)-N-isopropylbenzamide 388.3\n"
    )

    rows = lib.apply_patterns_to_text(text, "US10214537")

    fabricated = [r for r in rows if r["compound_id"] == "4"]
    assert not fabricated, (
        "an IUPAC name's leading digit is being reported as a potency: "
        f"{fabricated[0]}"
    )


# ── what the fix must not break ───────────────────────────────────

def test_a_correct_row_far_from_its_header_survives(library):
    """US9694016's 2,689 far rows are 100% correct — distance is not guilt.

    Its `Example NNN 0.00280 0.00050` table runs past 28,000 chars, so every
    row in the tail sits outside the 20,000-char window the foreign gate uses.
    Nothing else is printed in between: the header the row belongs to is still
    the last one printed. A distance-shaped fix deletes 2,688 verified-correct
    rows here; this test is what makes that fix fail.
    """
    library({
        "key": "braf0000000000ff",
        "regex": r"(?P<cid>Example \d+)\s+(?P<value0>\d+\.\d+)",
        "column_assays": ["B-Raf IC50 (μM)"],
        "header_text": "",
        "example_match": "Example 873 0.00280",
        "status": "pending",
        "fingerprints_observed": ["US9694016"],
        "first_seen_patent": "US9694016",
        "n_observations": 3,
    })
    rows_block = "".join(
        f"Example {i} 0.00{i % 9 + 1}0\n" for i in range(1, 1400)
    )
    text = "TABLE 8\nEx. No. B-Raf IC50 (μM)\n" + rows_block
    assert len(text) > 25_000

    rows = lib.apply_patterns_to_text(text, "US9694016")

    far = [r for r in rows if r["source_offset"] > 25_000]
    assert far, "rows in the tail of one long table were dropped"
    assert {r["assay_name"] for r in far} == {"B-Raf IC50 (μM)"}


def test_rows_under_their_own_header_are_untouched(library):
    """The assay block of the mixed document still yields its rows.

    Guards the cheapest wrong fix: dropping every row in a document that
    contains an LC/MS table anywhere.
    """
    library(_P2X3_PATTERN)
    text = _P2X3_ASSAY_BLOCK + _FILLER + _P2X3_LCMS_BLOCK

    rows = lib.apply_patterns_to_text(text, "US9718790")

    good = {r["compound_id"] for r in rows}
    assert "I-2655" in good and "I-2658" in good, (
        f"the patent's own potency table lost its rows: {sorted(good)}"
    )


def test_a_table_with_no_caption_is_not_penalised(library):
    """No `TABLE n` marker anywhere -> the rule has nothing to judge on.

    Google Patents renders some tables as flat prose with the caption gone;
    US10246453's 446 far rows and US10273259's 881 corroborated rows are in
    exactly that state. Absence of evidence about the local header must read
    as "keep", or the rule deletes the patents whose captions the renderer
    dropped.
    """
    library(_P2X3_PATTERN)
    text = (
        "Activity against P2X3 IC50 (μM) was measured. "
        + "I-2653 0.009 I-2654 0.009 I-2655 0.005 I-2656 0.006\n"
    )

    rows = lib.apply_patterns_to_text(text, "US9718790")

    assert {r["compound_id"] for r in rows} >= {"I-2653", "I-2655"}
