"""A row that prints THREE (compound, value) column pairs is three compounds.

`TABLE-US-00569` of US9718790 is one table, sixty rows, six columns:

    Compound   P2X3        Compound   P2X3        Compound   P2X3
    No.        IC50 (μM)   No.        IC50 (μM)   No.        IC50 (μM)
    I-0020     0.384       I-0897     0.025       I-1555     0.016

Five entries in `patentdb/data/assay_patterns.discoveries.json` capture that
shape correctly — `52836cae04c24c40`, `718980c5cc0bbb8d`, `8c49be14f829bf77`,
`d38bd730f305c2a7`, `7fd29cd64583a92d` all declare `cid`, `cid1` AND `cid2`.
`apply_patterns_to_text` read only `cid`, then walked `column_assays` and
attached `value0`, `value1` and `value2` to it, so I-0020 shipped 0.384 (its
own) plus 0.025 (I-0897's) plus 0.016 (I-1555's), and I-0897 and I-1555
shipped nothing at all.

Measured on this checkout, `apply_patterns_to_text` over the GP description
of US9718790, graded against that patent's own CALS via ElementTree — a row
is CORRECT when (compound_id, value) are adjacent cells of one CALS row:

    pattern             rows    correct    wrong    %
    52836cae04c24c40    2238        772     1466   34.5%
    d38bd730f305c2a7    2238        772     1466   34.5%
    718980c5cc0bbb8d    2256        781     1475   34.6%
    7fd29cd64583a92d    2256        781     1475   34.6%
    8c49be14f829bf77    2256        781     1475   34.6%
    every other pattern on this patent                100.0%

One in three, which is what a three-column-pair table read as one column pair
scores by construction. The other seven patterns firing on US9718790 declare
a single `cid` and score 2,261/2,262.

The sibling `180ac3078101462a` — `(?P<cid>I-\\d+)\\s+(?P<value0>[\\d.]+)\\s*$`,
same table, one column pair — is NOT the fix. Its `$` never binds: Google
Patents renders TABLE 569 as flat prose, so it takes 0 raw regex matches on
the text the pipeline loads and emits 0 rows. It cannot rescue I-0897.

The rule installed here is positional and reduces to the old behaviour
whenever a pattern declares one id: a `valueN` group belongs to the id group
that most recently PRECEDES it in the regex source. `test_one_id_pattern…`
below is what pins that reduction.
"""
from __future__ import annotations

import json

import pytest

from patentdb.core.assay_fsm import assay_pattern_library as lib


# Verbatim from the library (key 52836cae04c24c40, first_seen US9718790).
_THREE_PAIR = {
    "key": "52836cae04c24c40",
    "regex": (r"(?P<cid>I-\d+)\s+(?P<value0>[\d.]+)\s+"
              r"(?P<cid1>I-\d+)\s+(?P<value1>[\d.]+)\s+"
              r"(?P<cid2>I-\d+)\s+(?P<value2>[\d.]+)"),
    "column_assays": ["P2X3 IC50 (μM)", "P2X3 IC50 (μM)", "P2X3 IC50 (μM)"],
    "header_text": "",
    "example_match": "I-0610 0.035 I-1262 0.163 I-1991 0.006",
    "status": "pending",
    "fingerprints_observed": ["US9718790"],
    "first_seen_patent": "US9718790",
    "n_observations": 9,
}

# The same table's single-pair sibling: one id, several value columns. Its
# values all belong to the one compound and must keep doing so.
_ONE_ID_TWO_VALUES = {
    "key": "0000000000000001",
    "regex": r"(?P<cid>\d+)\s+(?P<value0>[\d.]+)\s+(?P<value1>[\d.]+)",
    "column_assays": ["PI3K delta IC50 (nM)", "CD69 IC50 (nM)"],
    "header_text": "",
    "example_match": "638 2 68",
    "status": "pending",
    "fingerprints_observed": ["US10214537"],
    "first_seen_patent": "US10214537",
    "n_observations": 3,
}

# Rows 1-3 of TABLE-US-00569, and its real caption.
_TABLE_569 = (
    "TABLE 569\n"
    "Compound P2X3 Compound P2X3 Compound P2X3\n"
    "No. IC50 (μM) No. IC50 (μM) No. IC50 (μM)\n"
    "I-0020 0.384 I-0897 0.025 I-1555 0.016\n"
    "I-0021 0.197 I-0898 0.020 I-1556 0.011\n"
    "I-0029 0.595 I-0899 0.023 I-1558 0.008\n"
)


@pytest.fixture
def library(tmp_path, monkeypatch):
    def _f(*entries: dict) -> None:
        p = tmp_path / "assay_patterns.discoveries.json"
        p.write_text(json.dumps(
            {"schema_version": "1.0", "tokens": list(entries)}, indent=1))
        monkeypatch.setattr(lib, "_PATTERNS_PATH", p)
    return _f


def _by_cid(rows):
    out: dict[str, set] = {}
    for r in rows:
        out.setdefault(r["compound_id"], set()).add(r["value"])
    return out


# ── the defect ────────────────────────────────────────────────────

def test_the_second_and_third_column_pairs_keep_their_own_compounds(library):
    """I-0020 owns 0.384. 0.025 is I-0897's and 0.016 is I-1555's."""
    library(_THREE_PAIR)

    got = _by_cid(lib.apply_patterns_to_text(_TABLE_569, "US9718790"))

    assert got.get("I-0020") == {0.384}, (
        f"I-0020 was given its neighbours' readings: {got.get('I-0020')}")
    assert got.get("I-0897") == {0.025}, (
        f"I-0897's own reading was never emitted: {got.get('I-0897')}")
    assert got.get("I-1555") == {0.016}, (
        f"I-1555's own reading was never emitted: {got.get('I-1555')}")


def test_every_compound_the_row_names_gets_a_record(library):
    """Nine compounds are printed across three rows; nine must ship.

    The old applier emitted three compounds and nine values. Compound
    coverage, not record count, is what a column-pair table costs us:
    I-0897 and I-1555 had no record at all.
    """
    library(_THREE_PAIR)

    rows = lib.apply_patterns_to_text(_TABLE_569, "US9718790")

    assert {r["compound_id"] for r in rows} == {
        "I-0020", "I-0021", "I-0029",
        "I-0897", "I-0898", "I-0899",
        "I-1555", "I-1556", "I-1558",
    }


def test_no_value_is_attached_to_a_compound_the_row_does_not_print(library):
    """Every emitted (cid, value) is a pair the table really prints."""
    library(_THREE_PAIR)
    printed = {
        ("I-0020", 0.384), ("I-0897", 0.025), ("I-1555", 0.016),
        ("I-0021", 0.197), ("I-0898", 0.020), ("I-1556", 0.011),
        ("I-0029", 0.595), ("I-0899", 0.023), ("I-1558", 0.008),
    }

    rows = lib.apply_patterns_to_text(_TABLE_569, "US9718790")

    invented = [(r["compound_id"], r["value"]) for r in rows
                if (r["compound_id"], r["value"]) not in printed]
    assert not invented, f"pairs the patent never prints: {invented}"


# ── what the fix must not break ───────────────────────────────────

def test_one_id_pattern_still_fans_every_value_onto_it(library):
    """The 111 patterns that declare a single `cid` must not change.

    `638 2 68` under `Ex. No. | PI3K delta IC50 (nM) | CD69 IC50 (nM)` is one
    compound with two readings, and both belong to compound 638.
    """
    library(_ONE_ID_TWO_VALUES)
    text = (
        "TABLE 11\n"
        "Ex. No. PI3K delta IC50 value (nM) CD69 IC50 value (nM)\n"
        "638 2 68\n"
    )

    rows = lib.apply_patterns_to_text(text, "US10214537")

    assert _by_cid(rows) == {"638": {2.0, 68.0}}
    assert {r["assay_name"] for r in rows} == {
        "PI3K delta IC50 (nM)", "CD69 IC50 (nM)"}


def test_an_id_group_that_matched_nothing_does_not_drop_the_row(library):
    """`cid2` optional and absent — the pairs that DID match still ship.

    A pattern whose trailing column pair is optional (the last row of a
    three-up table is routinely short) must not lose the two pairs in front
    of it.
    """
    library({
        **_THREE_PAIR,
        "key": "0000000000000002",
        "regex": (r"(?P<cid>I-\d+)\s+(?P<value0>[\d.]+)\s+"
                  r"(?P<cid1>I-\d+)\s+(?P<value1>[\d.]+)"
                  r"(?:\s+(?P<cid2>I-\d+)\s+(?P<value2>[\d.]+))?"),
    })
    text = (
        "TABLE 569\n"
        "Compound P2X3 Compound P2X3 Compound P2X3\n"
        "No. IC50 (μM) No. IC50 (μM) No. IC50 (μM)\n"
        "I-0031 0.400 I-0900 0.018\n"
    )

    got = _by_cid(lib.apply_patterns_to_text(text, "US9718790"))

    assert got == {"I-0031": {0.400}, "I-0900": {0.018}}
