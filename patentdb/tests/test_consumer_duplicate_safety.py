"""Grade downstream consumers for duplicate-row safety ahead of the v2
assay-tier merge.

Context: `assay_tables` is becoming `dict[cid -> list[row]]` fed by TWO
extraction tiers (existing + USPTO XML), both tagged with a `source`
field and never deduplicated — every merge point already does
`setdefault(cid, []).append/extend`, so the container itself is fine
with duplicates. ~59% of the XML tier's rows have a value-equal
counterpart already shipped by the existing tier, so a `(cid, value)`
appearing twice under two different `source` tags is the common case,
not an edge case.

The risk is entirely downstream: any consumer that assumes "one row
per (compound, assay)" will silently produce a different number, a
different classification, or a different selected value the moment a
second tier's row lands next to an equivalent one from the first tier.

Each test here:
  1. builds a small `assay_tables`-shaped fixture,
  2. builds a second version with ONE (or, where the target code's own
     comment specifies duplicating a whole class of rows, that class
     of) row duplicated under a different `source` tag,
  3. runs the REAL consumer function (imported, never reimplemented)
     on both,
  4. asserts the reported OUTPUT is unchanged.

Every test is `xfail(strict=True)`: the consumer is not fixed here.
`strict=True` means an accidental fix flips the test to XPASS, which
pytest reports as a failure — that is the intended tripwire forcing
the marker's removal once the consumer is actually made duplicate-safe.
Verified by running each scenario standalone before writing the
assertion (see task history); none of these are hypothetical.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest


# ── C1 — scripts/eval/assay_completeness_audit.py: diff_one ──────────────
#
# `diff_one` greedy-matches each BDB row against at most one v2 row. A
# duplicated v2 row (same value, second `source`) can never be consumed
# by the same BDB row twice, so it falls, unmatched, into `n_v2_only` —
# and the report's own docstring frames that bucket as a POSITIVE
# finding ("compounds where v2 has MORE assays than BDB, multi-experiment
# captures"). A second-tier duplicate therefore reads as evidence of
# richer coverage instead of the corruption it is.
#
# Verified directly: bdb=[10.0 nM], v2=[10.0 nM] -> n_v2_only=0,
# n_matched=1. Duplicating the v2 row (same value, source="uspto_xml")
# -> n_v2_only=1, n_matched still 1. The duplicate is invisible to
# n_matched but visible (and misread as "extra data") in n_v2_only.

from patentdb.scripts.eval.assay_completeness_audit import diff_one


@pytest.mark.xfail(
    strict=True,
    reason=(
        "diff_one's greedy 1:1 matcher can only consume a BDB row once, so "
        "a value-equal duplicate v2 row (second source tag) always falls "
        "into n_v2_only — a merge-tier duplicate is reported as a genuine "
        "'v2 has MORE assays than BDB' finding instead of being flagged."
    ),
)
def test_c1_diff_one_n_v2_only_unchanged_by_duplicate():
    bdb_assays = [{"assay": "IC50", "value": 10.0, "unit": "nM"}]
    v2_assays = [
        {"assay_name": "IC50", "value_numeric": 10.0, "unit": "nM", "source": "gp"},
    ]
    v2_assays_dup = v2_assays + [
        {"assay_name": "IC50", "value_numeric": 10.0, "unit": "nM", "source": "uspto_xml"},
    ]

    baseline = diff_one("C1", bdb_assays, v2_assays)
    dup = diff_one("C1", bdb_assays, v2_assays_dup)

    assert dup.n_v2_only == baseline.n_v2_only


# ── C2 — scripts/eval/build_handcheck_xlsx.py: build_assay_row_stats ─────
#
# `bdb_row_coverage_pct = 100 * v2_rows_for_matched / bdb_rows` counts
# every v2 row for a matched compound, uncapped by how many BDB rows
# that compound actually has — a colour-highlighted "quality" signal in
# the handcheck xlsx. `build_assay_row_stats` takes only a `patent_id`
# and resolves its own file paths via `Path(__file__).resolve()....` and
# a hardcoded `output/bindingdb/our_patents.tsv`, so it isn't a pure
# function of an `assay_tables` dict — it is exercised here by
# monkeypatching the MODULE's `__file__` attribute (a plain global
# lookup inside the function body) to point at a throwaway fixture tree,
# so the real repo's `output_v2/` and `output/bindingdb/` are never
# touched.
#
# Verified directly: 1 BDB row + 1 matching v2 row -> pct=100.0.
# Duplicating that v2 row (same value, source="uspto_xml") with the SAME
# single BDB row -> pct=200.0.

import patentdb.scripts.eval.build_handcheck_xlsx as bhx


def _fake_repo_for_row_stats(
    root: Path, patent_id: str, assay_tables: dict, example_index: dict, bdb_rows: list[dict]
) -> str:
    """Lay out a throwaway `{root}/output_v2/...` + `{root}/output/...`
    tree matching what `build_assay_row_stats` / `_letter_grade_status`
    expect, and return the fake `__file__` to install on the module so
    its internal `Path(__file__).resolve()....` chain lands in `root`.
    """
    fake_file = root / "patentdb" / "scripts" / "eval" / "build_handcheck_xlsx.py"
    out_dir = root / "output_v2" / "text_extraction" / patent_id
    out_dir.mkdir(parents=True)
    (out_dir / "assay_tables.json").write_text(json.dumps(assay_tables))
    (out_dir / "example_index.json").write_text(json.dumps(example_index))
    bdb_dir = root / "output" / "bindingdb"
    bdb_dir.mkdir(parents=True)
    with (bdb_dir / "our_patents.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["Patent Number", "Ligand InChI Key", "Target Name", "PMID", "Article DOI"],
        )
        writer.writeheader()
        for row in bdb_rows:
            writer.writerow(row)
    return str(fake_file)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "bdb_row_coverage_pct = 100 * v2_rows_for_matched / bdb_rows counts "
        "every v2 row for a matched compound uncapped by BDB's row count, so "
        "a duplicate v2 row inflates the percentage (100% -> 200% in this "
        "fixture) even though BDB's own row count didn't change."
    ),
)
def test_c2_bdb_row_coverage_pct_unchanged_by_duplicate(tmp_path, monkeypatch):
    patent_id = "US_TEST_C2_0001"
    ik = "ABCDEFGHIJKLMNO-UHFFFAOYSA-N"
    example_index = {"C1": {"inchikey": ik}}
    bdb_rows = [
        {"Patent Number": patent_id, "Ligand InChI Key": ik, "Target Name": "TargetX",
         "PMID": "", "Article DOI": ""},
    ]
    row = {"assay_name": "IC50", "value_numeric": 10.0, "unit": "nM", "source": "gp"}
    assay_tables = {"C1": [row]}
    assay_tables_dup = {"C1": [row, dict(row, source="uspto_xml")]}

    root_baseline = tmp_path / "baseline"
    monkeypatch.setattr(
        bhx, "__file__",
        _fake_repo_for_row_stats(root_baseline, patent_id, assay_tables, example_index, bdb_rows),
    )
    baseline = bhx.build_assay_row_stats(patent_id)

    root_dup = tmp_path / "dup"
    monkeypatch.setattr(
        bhx, "__file__",
        _fake_repo_for_row_stats(root_dup, patent_id, assay_tables_dup, example_index, bdb_rows),
    )
    dup = bhx.build_assay_row_stats(patent_id)

    assert dup["bdb_row_coverage_pct"] == baseline["bdb_row_coverage_pct"]


# ── C3 — scripts/eval/build_handcheck_xlsx.py: _letter_grade_status ──────
#
# Letter-grade detection short-circuits to True when
# `n_bin / n_all >= 0.30`, where `n_bin` counts rows with
# `source == "letter_bin"` and `n_all` counts every row in the patent's
# `assay_tables.json`. A second extraction tier that ships duplicates of
# only the NON-bin rows (it has no reason to duplicate the honest,
# value_numeric=None bin rows) grows the denominator without growing the
# numerator, diluting the ratio and silently flipping a genuinely
# letter-graded patent to look numeric.
#
# Verified directly: 31 bin rows / 100 total = 0.31 -> True. Duplicating
# all 69 non-bin rows (their own `source` tag) -> 31 / 169 = 0.183 ->
# False. Classification flips on data that says nothing new about
# whether the patent uses letter grades.

@pytest.mark.xfail(
    strict=True,
    reason=(
        "n_bin / n_all >= 0.30 is diluted by duplicate non-bin rows from a "
        "second tier: 31/100 (0.31, True) becomes 31/169 (0.18, False) once "
        "the 69 non-bin rows are each duplicated under a new source tag, "
        "even though nothing about the patent's letter-grade usage changed."
    ),
)
def test_c3_letter_grade_status_unchanged_by_nonbin_duplicates(tmp_path, monkeypatch):
    patent_id = "US_TEST_C3_0001"
    bin_rows = [
        {"assay_name": "IC50", "value_numeric": None, "source": "letter_bin"}
        for _ in range(31)
    ]
    nonbin_rows = [
        {"assay_name": "IC50", "value_numeric": 100.0 + i, "unit": "uM", "source": "gp"}
        for i in range(69)
    ]
    dup_nonbin_rows = [dict(r, source="uspto_xml") for r in nonbin_rows]

    baseline_ay = {"C1": bin_rows + nonbin_rows}
    dup_ay = {"C1": bin_rows + nonbin_rows + dup_nonbin_rows}

    root_baseline = tmp_path / "baseline"
    fake_file = root_baseline / "patentdb" / "scripts" / "eval" / "build_handcheck_xlsx.py"
    out_dir = root_baseline / "output_v2" / "text_extraction" / patent_id
    out_dir.mkdir(parents=True)
    (out_dir / "assay_tables.json").write_text(json.dumps(baseline_ay))
    monkeypatch.setattr(bhx, "__file__", str(fake_file))
    baseline_status = bhx._letter_grade_status(patent_id)

    root_dup = tmp_path / "dup"
    fake_file_dup = root_dup / "patentdb" / "scripts" / "eval" / "build_handcheck_xlsx.py"
    out_dir_dup = root_dup / "output_v2" / "text_extraction" / patent_id
    out_dir_dup.mkdir(parents=True)
    (out_dir_dup / "assay_tables.json").write_text(json.dumps(dup_ay))
    monkeypatch.setattr(bhx, "__file__", str(fake_file_dup))
    dup_status = bhx._letter_grade_status(patent_id)

    assert dup_status == baseline_status


# ── C4 — scripts/eval/export_assay_csv.py: _classify_patent ──────────────
#
# `n_measurements / n_compounds >= 1.5` (with `n_compounds >= 50`) labels
# a patent `table_primary`. Duplicate rows from a second tier inflate
# `n_measurements` without adding compounds, so a patent sitting just
# under the ratio threshold crosses it purely from merge-tier duplication.
#
# Verified directly: 50 compounds / 74 measurements (ratio 1.48) ->
# "small_table". Duplicating ONE row (75 measurements, same 50
# compounds, ratio 1.50) -> "table_primary".

from patentdb.scripts.eval.export_assay_csv import _classify_patent, export_csv


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_classify_patent's n_measurements/n_compounds >= 1.5 threshold "
        "crosses from a single duplicated row with no new compounds: 50 "
        "compounds / 74 rows (1.48, 'small_table') becomes 50 / 75 (1.50, "
        "'table_primary') after ONE row is duplicated under a new source."
    ),
)
def test_c4_classify_patent_label_unchanged_by_duplicate():
    assays: dict[str, list[dict]] = {}
    for i in range(50):
        n_rows = 2 if i < 24 else 1
        assays[f"C{i}"] = [
            {"assay_name": "IC50", "value_numeric": 10.0, "unit": "nM", "source": "gp"}
            for _ in range(n_rows)
        ]
    n_meas = sum(len(v) for v in assays.values())
    assert n_meas == 74  # sanity check on the fixture, not the thing under test
    baseline_label = _classify_patent(len(assays), n_meas)

    dup_assays = copy.deepcopy(assays)
    dup_assays["C0"].append(
        {"assay_name": "IC50", "value_numeric": 10.0, "unit": "nM", "source": "uspto_xml"}
    )
    n_meas_dup = sum(len(v) for v in dup_assays.values())
    dup_label = _classify_patent(len(dup_assays), n_meas_dup)

    assert dup_label == baseline_label


# ── C5 — scripts/eval/export_assay_csv.py: export_csv fieldnames ─────────
#
# The long-form CSV's header has no `source` column, so once a second
# tier's row is merged in, its line in the export is byte-identical to
# the original's (verified: both rows print as the exact same 12-field
# tuple) — a human reviewer, or any downstream aggregation over the CSV,
# cannot tell "one measurement, reported twice by two tiers" from "two
# independent measurements".

@pytest.mark.xfail(
    strict=True,
    reason=(
        "export_csv's fieldnames omit `source`; a duplicate row from a "
        "second extraction tier is written as a byte-identical extra CSV "
        "line with no column that could distinguish it from the original."
    ),
)
def test_c5_export_csv_has_source_column(tmp_path):
    extractions_root = tmp_path / "extraction"
    patent_id = "US_TEST_C5_0001"
    out_dir = extractions_root / patent_id
    out_dir.mkdir(parents=True)
    (out_dir / "assay_tables.json").write_text(json.dumps({
        "C1": [
            {"assay_name": "IC50", "value_numeric": 10.0, "unit": "nM",
             "qualifier": None, "source": "gp_table"},
            {"assay_name": "IC50", "value_numeric": 10.0, "unit": "nM",
             "qualifier": None, "source": "uspto_xml"},
        ]
    }))
    output_path = tmp_path / "out.csv"
    export_csv(output_path, [patent_id], extractions_root)

    with output_path.open() as f:
        header = next(csv.reader(f))

    assert "source" in header


# ── C6 — scripts/eval/build_comparison_csv_and_plot.py: build_comparison_csv ─
#
# `build_comparison_csv` is hardcoded to US8952177 via two module-level
# globals (`EXTRACTION`, `JIE_CSV`) rather than taking a patent_id
# parameter, so it's exercised here by monkeypatching those globals (bare
# names resolved at call time inside `load_v2_us8952177` /
# `load_jie_us8952177`) at a throwaway fixture tree — same mechanism as
# the `__file__` patch above, applied to plain module attributes instead.
#
# `v2_ki = ki_arr[0]["value_numeric"]` (line ~70) is first-element-wins:
# whichever tier's row happens to sit first in the list is the one
# compared against the hand-curated reference AND plotted as
# `ki_match_5pct`. Verified directly: two rows for the same compound
# (10.0 uM vs 50.0 uM, matching ref=10.0 uM) flip `claude_ki_uM` between
# "10.0" and "50.0", and `ki_match_5pct` between 1 and 0, purely from
# list order — i.e. from which tier the merge happened to place first.

import patentdb.scripts.eval.build_comparison_csv_and_plot as bcp


def _run_build_comparison_csv(tmp_path: Path, order: str) -> tuple[dict, dict]:
    extraction_root = tmp_path / f"extraction_{order}"
    us_dir = extraction_root / "US8952177"
    us_dir.mkdir(parents=True)
    good = {"assay_name": "FLAP binding Ki", "value_numeric": 10.0, "unit": "uM", "source": "gp_table"}
    bad = {"assay_name": "FLAP binding Ki", "value_numeric": 50.0, "unit": "uM", "source": "uspto_xml"}
    rows = [good, bad] if order == "good_first" else [bad, good]
    (us_dir / "assay_tables.json").write_text(json.dumps({"C1": rows}))

    jie_csv = tmp_path / f"jie_{order}.csv"
    with jie_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "compound_id", "iupac_name",
            "flap_binding_ki_uM", "flap_binding_ki_qualifier", "flap_binding_ki_n_runs",
            "hwb_ltb4_ic50_uM", "hwb_ltb4_ic50_qualifier", "hwb_ltb4_ic50_n_runs",
        ])
        writer.writeheader()
        writer.writerow({
            "compound_id": "C1", "iupac_name": "test compound",
            "flap_binding_ki_uM": "10.0", "flap_binding_ki_qualifier": "",
            "flap_binding_ki_n_runs": "",
            "hwb_ltb4_ic50_uM": "", "hwb_ltb4_ic50_qualifier": "", "hwb_ltb4_ic50_n_runs": "",
        })

    bcp.EXTRACTION = extraction_root
    bcp.JIE_CSV = jie_csv
    out_path = tmp_path / f"comparison_{order}.csv"
    stats = bcp.build_comparison_csv(out_path)
    with out_path.open() as f:
        row = next(csv.DictReader(f))
    return stats, row


@pytest.mark.xfail(
    strict=True,
    reason=(
        "build_comparison_csv selects ki_arr[0]['value_numeric'] — the "
        "first-listed tier's value wins. Two disagreeing rows for the same "
        "compound (10.0 uM vs 50.0 uM) select a different claude_ki_uM and "
        "flip ki_match_5pct purely by which tier the merge put first."
    ),
)
def test_c6_build_comparison_csv_selection_independent_of_order(tmp_path, monkeypatch):
    stats_good_first, row_good_first = _run_build_comparison_csv(tmp_path, "good_first")
    stats_bad_first, row_bad_first = _run_build_comparison_csv(tmp_path, "bad_first")

    assert row_good_first["claude_ki_uM"] == row_bad_first["claude_ki_uM"]
    assert stats_good_first["ki_match_5pct"] == stats_bad_first["ki_match_5pct"]


# ── C7 — routes/process_patent.py: _bridge_gp_to_harvest_cids ────────────
#
# The GP-to-HARVEST cid rename is gated by an MS `[M+H]+` agreement
# check that takes the FIRST row in `assay_tables[new_cid]` whose
# assay_name mentions "ms"/"m+h" (`break` on first match, lines
# ~1116-1120) and never looks at the rest. Whether the rename — which
# controls whether assay rows get attached to the right compound at all
# — is trusted therefore depends on which tier's MS row a merge happens
# to place first.
#
# Verified directly with a molecule (benzene) whose MW makes one MS
# candidate agree (mass-verified) and a second, 50 Da off, disagree, with
# no other alignment signal trusted (single GP entry, single digit-cid
# assay row — all of positional/prefix/digit alignment stay below their
# thresholds): [good, bad] renames GP107 -> "107"; [bad, good] leaves it
# as the orphaned "GP107". Same two rows, same molecule, opposite outcome.

from patentdb.core.smiles_utils import molecular_weight
from patentdb.routes.process_patent import _bridge_gp_to_harvest_cids


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_bridge_gp_to_harvest_cids picks the FIRST assay row whose name "
        "mentions MS/[M+H]+ and never checks the rest, so the rename "
        "decision (trusted vs skipped) depends on which of two "
        "disagreeing tiers' MS rows a merge places first: [good, bad] "
        "renames GP107 -> '107', [bad, good] leaves it as 'GP107'."
    ),
)
def test_c7_bridge_rename_decision_independent_of_ms_row_order():
    smiles = "c1ccccc1"  # benzene — cheap, deterministic RDKit MW
    mw = molecular_weight(smiles)
    good_ms = mw + 1.008   # exact [M+H]+ agreement
    bad_ms = good_ms + 50.0  # a different molecule's worth of mass, not noise

    example_index_template = {
        "GP107": {"canonical_smiles": smiles, "inchikey": "FAKEIKAAAAAAAA-UHFFFAOYSA-N"},
    }
    good_row = {"assay_name": "MS (M+H)+", "value_numeric": good_ms, "source": "uspto_xml"}
    bad_row = {"assay_name": "MS (M+H)+", "value_numeric": bad_ms, "source": "gp_table"}

    # A patent_id guaranteed absent from data/assay_vocabulary.discoveries.json
    # so _detect_patent_cid_prefixes returns [] and only the bare-digit
    # candidate path fires — keeps the MS check as the sole trust signal
    # (single GP entry / single digit assay key puts positional, prefix,
    # and digit alignment all below their own trust thresholds).
    patent_id = "US_TEST_NONEXISTENT_C7_0001"

    example_index_good_first = copy.deepcopy(example_index_template)
    assay_tables_good_first = {"107": [good_row, bad_row]}
    result_good_first = _bridge_gp_to_harvest_cids(
        patent_id, example_index_good_first, assay_tables_good_first
    )

    example_index_bad_first = copy.deepcopy(example_index_template)
    assay_tables_bad_first = {"107": [bad_row, good_row]}
    result_bad_first = _bridge_gp_to_harvest_cids(
        patent_id, example_index_bad_first, assay_tables_bad_first
    )

    renamed_good_first = "107" in result_good_first
    renamed_bad_first = "107" in result_bad_first
    assert renamed_good_first == renamed_bad_first
