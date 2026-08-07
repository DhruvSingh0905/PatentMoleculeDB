"""A recovered pair is only usable if its KEY is a compound id this patent owns.

`iupac_burst_targeted` is asked about a specific list of cids and returns
whatever `(compound_id, iupac_name)` pairs the model read out of the window.
The model is under no obligation to answer with the id it was asked about, and
on US11566007 it mostly does not: of 19 pairs replayed from cache, 17 are keyed
on prose it invented —

    "Step 1 product"                        "Intermediate 1 (Step 9 product)"
    "Intermediate 1 Alt Step 2 product A"   "Intermediate 1"

— and `_targeted_fill_missing_cids` merged every one of them verbatim. Once in
`example_index` such a record is indistinguishable from a real compound: it has
an `iupac_name`, a `canonical_smiles`, an `inchikey`, and a `source`. Nothing
downstream can tell it apart, and the trade this project is built on is that a
missing compound can be recovered later while a fabricated one cannot.

The guard is membership, not shape, because shape does not separate the two
populations: `normalize_cid_key("Intermediate 1")` and
`canonical_cid("Intermediate 1")` both accept it. What actually distinguishes a
real id is that the patent's own tables use it — which is exactly the
`missing_cids` list this call was built from. Ids are compared through
`compound_id.canonical_cid`, the codebase's single dedup key, so a pair keyed
`Compound 5a` still merges under the tables' `5A` rather than opening a second
namespace; and the record is stored under the TABLES' spelling, never the
model's.

Corpus effect, measured over the five patents that return anything at all:
54 pairs in, 7 kept, 47 dropped.
"""
from __future__ import annotations

import pytest

from patentdb.routes import process_patent as PP


NAME = "2-[4-(3-chlorophenyl)piperazin-1-yl]-N-(pyridin-3-ylmethyl)acetamide"


@pytest.fixture
def burst(monkeypatch):
    """Replace the recovery burst with a canned answer; record what it asked."""
    box = {"pairs": {}, "asked": None}

    def fake(*, patent_id, text, missing_cids, cost_tracker, window_chars=8000):
        box["asked"] = list(missing_cids)
        return dict(box["pairs"])

    import patentdb.core.assay_fsm.harvest.iupac_orchestrator as IO
    monkeypatch.setattr(IO, "iupac_burst_targeted", fake)
    return box


def _run(burst, pairs, example_index, assay_tables):
    burst["pairs"] = pairs
    return PP._targeted_fill_missing_cids(
        "USTEST", "some patent text", example_index, assay_tables,
    )


def test_a_cid_the_patent_owns_is_merged(burst):
    """The baseline this guard must not break."""
    idx = _run(burst, {"A71": NAME}, {}, {"A71": [{"value": 1.0}]})
    assert idx["A71"]["canonical_smiles"]
    assert idx["A71"]["extraction_method"] == "iupac_harvest_targeted"


@pytest.mark.parametrize("key", [
    "Step 1 product",
    "Intermediate 1",
    "Intermediate 1 Alt Step 2 product A",
    "Intermediate 1 (Step 9 product)",
])
def test_a_prose_key_is_never_merged(burst, key):
    """Verbatim keys from US11566007's cached burst output."""
    idx = _run(burst, {key: NAME}, {}, {"A71": [{"value": 1.0}]})
    assert idx == {}, f"{key!r} was merged as if it were a compound"


def test_an_amino_acid_abbreviation_is_not_a_compound_id(burst):
    """US20240010684A1's burst answered with residue codes — `Abu`, `Orn`,
    `PEG2`, `(N-Me)C`. `PEG2` even survives `canonical_cid`; only membership
    in the patent's own table ids rejects it."""
    pairs = {k: NAME for k in ("Abu", "Orn", "PEG2", "(N-Me)C", "TTDS")}
    idx = _run(burst, pairs, {}, {"60": [{"value": 1.0}]})
    assert idx == {}


def test_a_differently_spelled_cid_merges_under_the_tables_spelling(burst):
    """`canonical_cid` is the join, so the model's spelling does not create a
    second namespace — the record lands on the key the assay rows use."""
    idx = _run(burst, {"Compound 5a": NAME}, {}, {"5A": [{"value": 1.0}]})
    assert "5A" in idx and "Compound 5a" not in idx and "5a" not in idx


def test_a_real_id_we_were_not_missing_is_left_alone(burst):
    """US8952177's burst answered about 157-165 while only 162 and 166 lacked
    a structure. Those are real ids — and they already have one. This call
    exists to FILL empty cids; overwriting a resolved compound with an
    LLM-recovered name is not what it was asked to do."""
    resolved = {
        "compound_id": "Example 157", "iupac_name": "propan-2-ol",
        "canonical_smiles": "CC(C)O",
        "inchikey": "KFZMGEQAYNKOFK-UHFFFAOYSA-N",
        "source": "examples", "extraction_method": "explicit_example_header",
    }
    idx = _run(
        burst, {"157": NAME, "162": NAME},
        {"157": dict(resolved)},
        {"157": [{"value": 1.0}], "162": [{"value": 2.0}]},
    )
    assert idx["157"] == resolved, "a resolved compound was overwritten"
    assert idx["162"]["extraction_method"] == "iupac_harvest_targeted"


def test_a_structureless_stub_is_still_fillable(burst):
    """A cid present but empty IS missing — the guard must not read
    'already in example_index' as 'already found'."""
    idx = _run(
        burst, {"162": NAME},
        {"162": {"compound_id": "Example 162", "iupac_name": "",
                 "canonical_smiles": "", "inchikey": "", "source": "ms_stub"}},
        {"162": [{"value": 1.0}]},
    )
    assert idx["162"]["canonical_smiles"]


def test_nothing_is_dropped_when_every_key_is_a_missing_cid(burst):
    """The guard must be inert on a well-behaved answer."""
    idx = _run(
        burst, {"A71": NAME, "A194": NAME}, {},
        {"A71": [{"value": 1.0}], "A194": [{"value": 2.0}]},
    )
    assert set(idx) == {"A71", "A194"}
