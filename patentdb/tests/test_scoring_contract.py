"""What the SCORING layer promises, graded against what it actually does.

Every other test in this package grades the EXTRACTOR. These grade the thing
that grades the extractor, and they are here because a scorer's defects are
invisible in exactly the way an extractor's are not: a broken parser prints
zero rows, a broken scorer prints a number that looks like success. Three of
the four defects below make the reported score go UP.

Each test is `xfail(strict=True)`. It fails against the tree as it stands —
that is the proof it grades something real and not a restatement of current
behaviour. When the defect is fixed the test XPASSes, strict turns that into a
FAILURE, and whoever fixed it is forced to come here and delete the marker.
A characterization test (one that passes today, against the broken code) would
have locked the defect in instead.

Nothing here reads `output_v2/bindingdb/our_patents.tsv` (271 MB) or reaches a
network or a model. S2 monkeypatches the reference path to a four-line
synthetic TSV; S1 passes its reference dict in directly, so `load_reference`
is never called at all.

Measured on this tree, 2026-08:

  S1  a compound with 1 correct record scores `agree`; the SAME compound with
      10 additional records that are each 50x wrong still scores `agree`,
      bad=0. Identical score, 91% of the evidence wrong.
  S2  19,550 `(patent, cid)` keys carry what 37,573 `(patent, cid, measure,
      target)` keys distinguish — 48.0% of the reference's resolving power
      discarded before a single comparison happens. 92.8% of the finer keys
      hold exactly ONE value, so the finer key is almost always decisive.
  S3  `{_norm_key(k): v for k, v in ay.items()}` is a many-to-one map written
      as a dict comprehension: `"1-2"` and `"12"` both normalise to `"12"` and
      one compound's entire row list replaces the other's, silently.

S4 (the `flap` target-name collision) is NOT in this file. It does not
reproduce — see the note at the bottom.
"""
import csv
import json

import pytest

from patentdb.repair import value_check
from patentdb.repair.value_check import check_patent
from patentdb.sources.uspto_assays import AssayRecord


@pytest.fixture(autouse=True)
def _no_repair(monkeypatch):
    """No rule synthesis, no model, no network — these are pure scoring tests."""
    monkeypatch.setenv("REPAIR", "0")


# ── S1: the scorer cannot fail ──────────────────────────────────────
#
# `check_patent` (repair/value_check.py:144-174) takes the BEST bucket over the
# CROSS-PRODUCT of (our records for a compound) x (BDB's values for it). Its
# own comment says "Deliberately optimistic per compound ... a compound counts
# as agreeing if any pairing agrees". The justification is real — a patent
# reports several assays, BDB may hold several, and we do not know which pairs.
# The consequence is that the metric is MONOTONE IN RECORD COUNT: 20 rows
# against 5 references is 100 chances to land within 5%, and no wrong record
# can ever cost a compound its `agree`.
#
# That is the precise failure mode the module's own docstring says it exists to
# catch: "a coverage gate ... waves through a patch that reads MORE of the
# wrong thing". A patch that triples the row count and gets two thirds of them
# wrong scores identically to one that reads every value correctly, so
# value_check cannot tell those two patches apart — which is what it is for.

_REF_NM = 100.0          # what BindingDB publishes for this compound
_CORRECT_NM = 100.0      # our record, on the nose
_WRONG_NM = 5000.0       # 50x out — `wrong_scale`, not assay variability


def _rec(value_nm):
    return AssayRecord(cid="1", assay_name="ALOX5AP IC50",
                       value_numeric=value_nm, unit="nM")


def _score(records):
    return check_patent("USTEST01", records,
                        reference={("USTEST01", "1"): {_REF_NM}})


@pytest.mark.xfail(strict=True, reason=(
    "check_patent takes the best bucket over the cross-product of our records "
    "x BDB's values, so adding WRONG records can never lower the score. "
    "Measured: 1 correct record scores agree/bad=0; that same record plus ten "
    "50x-wrong ones also scores agree/bad=0. 91% wrong, identical score."))
def test_wrong_records_cannot_be_hidden_by_one_right_one():
    """Adding wrong data must not be free, and must not be a way to score.

    A metric that improves-or-holds when you add wrong records is not measuring
    correctness, it is measuring effort. This asserts the one thing that makes
    it a measurement: a compound whose evidence is MAJORITY WRONG must not read
    as agreement.
    """
    only_right = _score([_rec(_CORRECT_NM)])
    only_wrong = _score([_rec(_WRONG_NM) for _ in range(10)])

    # Sanity — the scorer does work in the two unambiguous cases. Both hold
    # today; neither is what this test grades.
    assert only_right["buckets"].get("agree") == 1, "a correct record must agree"
    assert only_wrong["bad"] == 1, "ten 50x-wrong records must score bad"

    # 1 right + 10 wrong. 10 of 11 records for this compound are 50x out.
    mostly_wrong = _score([_rec(_CORRECT_NM)] + [_rec(_WRONG_NM) for _ in range(10)])

    # THE GRADE. Not "is it worse than perfect" — is it *anything other than*
    # a clean pass. Today `mostly_wrong` == `only_right` exactly: agree=1,
    # bad=0. Ten wrong records cost nothing and are invisible in the output.
    assert mostly_wrong["buckets"].get("agree", 0) == 0, (
        "a compound with 10 wrong records and 1 right one scored `agree` — "
        "the scorer is monotone in record count")

    # Corollary, stated separately so a fix cannot satisfy the above by simply
    # refusing to bucket anything: adding wrong records must be able to make
    # the score WORSE, never only better-or-equal.
    assert mostly_wrong["bad"] >= only_right["bad"] + 1, (
        "adding ten wrong records did not increase `bad`")


# ── S2: the reference discards half its resolving power ─────────────
#
# `load_reference` (repair/value_check.py:56-111) returns
# `(patent, cid) -> set[float]`. `Target Name` (column 6) is never read, and
# the Ki/IC50/Kd/EC50 columns (8-11) are located only so their numbers can be
# poured into one flat list (`:89-100`) and stored under a key that mentions
# neither which measurement nor which protein (`:110`).
#
# Measured over the reference file config actually resolves to
# (output_v2/bindingdb/our_patents.tsv, 95,245 rows): 19,550 coarse keys where
# the same rows support 37,573 `(patent, cid, measure, target)` keys — 48.0% of
# the resolving power thrown away, and 92.8% of the finer keys hold exactly one
# value, i.e. the finer key almost always names a unique number.
#
# It compounds S1 rather than merely adding to it. US9745328 is the live case:
# its BDB rows carry BOTH `Arachidonate 5-lipoxygenase-activating protein`
# (1,619 rows) and `Flap endonuclease 1` (898 rows). Flattened, an IC50 against
# one protein and a Ki against the other become interchangeable targets for any
# record we produce — so a value we read against the WRONG protein scores
# `agree` against the right one's number.

_BDB_HEADER = [
    "BindingDB Reactant_set_id", "Ligand SMILES", "Ligand InChI",
    "Ligand InChI Key", "BindingDB MonomerID", "BindingDB Ligand Name",
    "Target Name", "Target Source Organism According to Curator or DataSource",
    "Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)",
]


def _bdb_row(ligand_name, target, ki="", ic50="", kd="", ec50=""):
    return ["1", "C", "InChI=1S/C", "AAAAAAAAAAAAAA-A", "50000",
            ligand_name, target, "Homo sapiens", ki, ic50, kd, ec50]


@pytest.fixture
def synthetic_bdb(tmp_path, monkeypatch):
    """A four-line stand-in for the 271 MB dump, shaped like US9745328.

    ONE compound, TWO measurements, TWO proteins — the pattern that makes the
    flattening lossy. Hermetic on purpose: a scoring test that needs an 8 GB
    file to run is a test nobody runs.
    """
    path = tmp_path / "our_patents.tsv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(_BDB_HEADER)
        w.writerow(_bdb_row("US9999999, Compound 1",
                            "Arachidonate 5-lipoxygenase-activating protein",
                            ic50="10.0"))
        w.writerow(_bdb_row("US9999999, Compound 1",
                            "Flap endonuclease 1",
                            ki="500.0"))
    monkeypatch.setattr(value_check, "_BDB", path)
    return path


@pytest.mark.xfail(strict=True, reason=(
    "load_reference keys on (patent, cid) only — Target Name (col 6) is never "
    "read and Ki/IC50/Kd/EC50 (cols 8-11) are flattened into one set. "
    "Measured: 19,550 coarse keys vs 37,573 (patent, cid, measure, target) "
    "keys = 48.0% of resolving power discarded; 92.8% of the finer keys hold a "
    "single value."))
def test_reference_distinguishes_ic50_from_ki_and_target_from_target(synthetic_bdb):
    """A 10 nM IC50 and a 500 nM Ki are not two guesses at one number.

    They are two different measurements against two different proteins. Poured
    into one set they become alternative right answers, and any record within
    5% of EITHER scores `agree` — including one that read the wrong protein's
    column.
    """
    ref = value_check.load_reference({"US9999999"})

    # Sanity — the fixture parsed and neither measurement was dropped on the
    # floor. Guards against a "fix" that raises the key count by discarding.
    all_values = {v for values in ref.values() for v in values}
    assert all_values == {10.0, 500.0}, f"fixture did not load: {ref!r}"

    # THE GRADE. Deliberately shape-agnostic: it does not prescribe what the
    # key becomes, only that the two measurements stop being interchangeable.
    # Today: one key, ('US9999999', '1') -> {10.0, 500.0}.
    collapsed = [k for k, values in ref.items() if len(values) > 1]
    assert not collapsed, (
        f"an IC50 vs ALOX5AP and a Ki vs FEN1 collapsed into one "
        f"undifferentiated set under {collapsed!r} — either can now "
        f"corroborate a record that matched neither")


# ── S3: fidelity_check loses a whole compound, silently ─────────────
#
# `load_v2_extraction` (scripts/eval/fidelity_check.py:180-181):
#
#     ay_norm = {_norm_key(k): v for k, v in ay.items()}
#
# `_norm_key` is many-to-one — it strips every non-alphanumeric character, so
# `"1-2"` and `"12"` are both `"12"` (verified against the live function, along
# with `"Example 12"`, `"Compound 12"`, `"Cpd. No. 12"` and `"1 2"`). Written
# as a dict comprehension, the collision resolves by LAST-WINS: one compound's
# entire row list is replaced by another's. No exception, no warning, no count
# — the denominator just gets smaller and `assay_pct` goes up.
#
# Hyphenated compound ids are not hypothetical here; US10246453's live
# assay_tables.json carries `K-30K` alongside plain numeric ids. The collision
# does not currently fire on the 22 cached patents (measured: 0 collisions
# across both example_index.json and assay_tables.json), so this grades the
# CONTRACT rather than a today-failure — which is what makes it worth pinning.
# A lossy map behind a silent last-wins is a defect whether or not this week's
# corpus happens to trip it, and the day it trips there is nothing to see.


@pytest.fixture
def v2_output(tmp_path, monkeypatch):
    """A text_extraction/{pid} tree with two cids that normalise identically."""
    from patentdb.core import config

    base = tmp_path / "text_extraction" / "USTEST02"
    base.mkdir(parents=True)
    # "1-2" and "12" both -> "12". Three rows on one, two on the other.
    assay_tables = {
        "12":  [{"assay_name": "ALOX5AP IC50", "value_numeric": 10.0, "unit": "nM"},
                {"assay_name": "ALOX5AP IC50", "value_numeric": 11.0, "unit": "nM"},
                {"assay_name": "hWB IC50", "value_numeric": 250.0, "unit": "nM"}],
        "1-2": [{"assay_name": "ALOX5AP IC50", "value_numeric": 900.0, "unit": "nM"},
                {"assay_name": "hWB IC50", "value_numeric": 4000.0, "unit": "nM"}],
    }
    example_index = {
        "12":  {"canonical_smiles": "CCO", "inchikey": None},
        "1-2": {"canonical_smiles": "CCC", "inchikey": None},
    }
    (base / "assay_tables.json").write_text(json.dumps(assay_tables))
    (base / "example_index.json").write_text(json.dumps(example_index))
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    return assay_tables


@pytest.mark.xfail(strict=True, reason=(
    "fidelity_check.py:181 normalises cid keys through a many-to-one _norm_key "
    "inside a dict comprehension. '1-2' and '12' both normalise to '12', so "
    "last-wins silently drops one compound's entire row list: 5 rows in, "
    "2 rows out, 3 lost with no error and no counter."))
def test_cid_normalisation_never_drops_a_compounds_rows(v2_output):
    """Normalising a key must never be able to delete a measurement.

    Asserted on ROWS, not keys, so any honest fix passes — suffixing the
    colliding key, merging the lists, or raising. What must not pass is the
    current behaviour, where the rows simply cease to exist and every
    downstream percentage is computed over a quietly smaller denominator.
    """
    from patentdb.scripts.eval import fidelity_check

    raw_rows = sum(len(rows) for rows in v2_output.values())
    assert raw_rows == 5, "fixture drifted"

    _ex_norm, ay_norm = fidelity_check.load_v2_extraction("USTEST02")

    # Sanity — the fixture was found and parsed at all.
    assert ay_norm, "load_v2_extraction returned nothing; fixture path is wrong"

    # THE GRADE. Today: {'12': [<the two rows from '1-2'>]} — 2 rows, 3 lost.
    kept_rows = sum(len(rows) for rows in ay_norm.values())
    assert kept_rows == 5, (
        f"cid normalisation dropped {raw_rows - kept_rows} of {raw_rows} assay "
        f"rows: '1-2' and '12' both normalise to '12' and one compound's whole "
        f"row list replaced the other's")


# ── S4: the `flap` target-name collision — NOT REPRODUCIBLE, no test ─
#
# The premise checks out on both sides and the defect still does not exist.
#
#   - the collision is real in the INDEX:  patentdb/data/assay_target_index.json
#     maps token `flap` -> 'Flap endonuclease 1' (verified: it is in `tokens`).
#   - the collision is real in the DATA:   US9745328's BDB rows carry BOTH
#     'Arachidonate 5-lipoxygenase-activating protein' (1,619 rows) and
#     'Flap endonuclease 1' (898 rows), so scoping a lookup to the targets the
#     patent actually has could not disambiguate it either.
#   - but the RESOLVER already refuses to answer. `assay_reconciler`
#     `_AMBIGUOUS_TOKENS == {'adenosine', 'flap', 'a1'}` and `_target_tokens`
#     skips those tokens outright, with the reason written at :77 —
#     "flap -> Flap endonuclease 1 (vs ALOX5AP = 5-LO-activating protein)".
#     Verified: reconcile() returns None for 'FLAP', 'FLAP IC50',
#     'FLAP IC50 (nM)', 'human FLAP binding IC50', '5-LO/FLAP',
#     'FLAP-mediated LTB4' and lowercase 'flap'. `reconcile` is the only reader
#     of the index (sole caller: scripts/eval/build_handcheck_xlsx.py:568), and
#     it takes no patent argument, so there is no second path in.
#
# An `xfail(strict=True)` test here would XPASS on the day it was written,
# which strict correctly reports as a failure. The guard is what should be
# pinned, and pinning it is a characterization test of working code — a
# different job from this file's.
