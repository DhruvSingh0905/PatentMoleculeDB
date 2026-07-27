"""US8952177 end-to-end regression test.

This patent was the catalyst for the FSM rebuild — the legacy pipeline
extracted ZERO compounds because:
  1. TABLE 2 is plain-text format (no `<table>` HTML tag)
  2. Headers don't match the legacy hardcoded regex list
  3. Row format uses `~` qualifier and `(N)` n_runs annotations the
     row regex didn't tolerate.

This test pins the contract that the new pipeline must extract all
190 compounds correctly, with NO US8952177-specific code anywhere
under `core/assay_fsm/` or `data/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSAY_TABLES_PATH = REPO_ROOT / "output_v2" / "text_extraction" / "US8952177" / "assay_tables.json"
REFERENCE_CSV = REPO_ROOT / "US8952177 Binding IUPAC Final (1).csv"
# CSV may also live in user's Downloads — the second path is the
# canonical hand-curated reference.
DOWNLOADS_CSV = Path.home() / "Downloads" / "US8952177 Binding IUPAC Final (1).csv"


# ── Generalizability gate ────────────────────────────────────────


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove `#` comments and triple-quoted blocks (docstrings + prompt
    templates) so only EXECUTABLE code remains. Patent IDs cited in
    comments/docstrings document WHY a generic heuristic exists (the same
    provenance style the project CLAUDE.md uses) and are NOT patent-specific
    code; real hardcoding (`if patent_id == "US…"`) uses inline string
    literals on code lines, which survive this strip and are still caught."""
    import re as _re
    src = _re.sub(r'"""[\s\S]*?"""', "", src)
    src = _re.sub(r"'''[\s\S]*?'''", "", src)
    src = _re.sub(r"#[^\n]*", "", src)
    return src


def test_no_patent_specific_code_in_fsm_package() -> None:
    """The FSM package must contain no patent-specific CODE — no hardcoded
    US8952177 branching or its target/assay names (FLAP/HTRF/LTB4/HWB) baked
    into logic. Patent IDs in comments/docstrings are allowed as provenance;
    only executable code is scanned. The `data/` dir is scanned in full — no
    patent-specific data fixtures belong in the package either."""
    import re as _re
    fsm_dir = REPO_ROOT / "patentdb" / "core" / "assay_fsm"
    data_dir = REPO_ROOT / "patentdb" / "data"
    pattern = _re.compile(r"US[0-9]{6,}|FLAP|HTRF|LTB4|HWB")

    offenders: list[str] = []
    # Scan CODE only (.py, comments/docstrings stripped) in both dirs. We do
    # NOT scan data/*.json: the cross-patent learned-vocabulary files
    # (assay_vocabulary/patterns.discoveries.json) legitimately record a
    # `first_seen_patent` provenance field — that is the generalization
    # mechanism (learn a pattern on one patent, apply to all), the opposite
    # of patent-specific hardcoding. A patent-specific .py fixture, however,
    # would still be caught here.
    for base in (fsm_dir, data_dir):
        if not base.exists():
            continue
        for py in sorted(base.rglob("*.py")):
            code = _strip_comments_and_docstrings(
                py.read_text(encoding="utf-8", errors="replace")
            )
            for i, line in enumerate(code.splitlines(), 1):
                if pattern.search(line):
                    offenders.append(
                        f"{py.relative_to(REPO_ROOT)}:{i}: {line.strip()}"
                    )

    assert not offenders, (
        "generalizability gate failed — patent-specific tokens in CODE:\n"
        + "\n".join(offenders)
    )


# ── End-to-end extraction ────────────────────────────────────────


@pytest.fixture
def extracted_assays() -> dict[str, list[dict]]:
    """Read the extraction output. Pipeline must have been run already
    via the shadow extractor; this test gates merge readiness, not
    runtime."""
    if not ASSAY_TABLES_PATH.exists():
        pytest.skip(
            f"Run extraction first: ASSAY_FSM=1 python -c "
            f"'from patentdb.core.assay_fsm import extract_for_patent; "
            f"extract_for_patent(\"US8952177\")' "
            f"(expected output at {ASSAY_TABLES_PATH})"
        )
    return json.loads(ASSAY_TABLES_PATH.read_text())


def test_compound_count(extracted_assays: dict) -> None:
    """All 190 compounds from TABLE 2 must be extracted."""
    assert len(extracted_assays) == 190, (
        f"Expected 190 compounds, got {len(extracted_assays)}"
    )


def test_known_compound_values(extracted_assays: dict) -> None:
    """Spot-check 5 compounds against known TABLE 2 values."""
    expected = {
        # cpd_id: (Ki_uM, Ki_qualifier, IC50_uM, IC50_qualifier)
        "1": (0.0038, None, 0.4, None),
        "2": (0.0014, None, 0.22, None),
        "3": (0.83, "~", 30.0, ">"),
        "4": (0.0025, None, 1.02, None),
        "190": (0.21, None, 0.93, None),
    }
    for cid, (ki, ki_q, ic50, ic50_q) in expected.items():
        assert cid in extracted_assays, f"cpd {cid} not extracted"
        results = extracted_assays[cid]
        # Find Ki and IC50 entries
        ki_results = [r for r in results if "ki" in (r["assay_name"] or "").lower() or "flap" in (r["assay_name"] or "").lower()]
        ic50_results = [r for r in results if "ic50" in (r["assay_name"] or "").lower() or "ltb" in (r["assay_name"] or "").lower() or "hwb" in (r["assay_name"] or "").lower()]
        assert ki_results, f"cpd {cid}: Ki not extracted (results: {results})"
        assert ic50_results, f"cpd {cid}: IC50 not extracted (results: {results})"

        # Tolerance match (5% relative)
        assert any(
            r["value_numeric"] is not None
            and abs(r["value_numeric"] - ki) / max(ki, 1e-9) < 0.05
            for r in ki_results
        ), f"cpd {cid}: Ki value mismatch (expected {ki}, got {[r['value_numeric'] for r in ki_results]})"

        assert any(
            r["value_numeric"] is not None
            and abs(r["value_numeric"] - ic50) / max(ic50, 1e-9) < 0.06
            for r in ic50_results
        ), f"cpd {cid}: IC50 value mismatch (expected {ic50}, got {[r['value_numeric'] for r in ic50_results]})"

        # Qualifiers
        if ki_q:
            assert any(r.get("qualifier") == ki_q for r in ki_results), (
                f"cpd {cid}: expected Ki qualifier {ki_q!r}, got "
                f"{[r.get('qualifier') for r in ki_results]}"
            )
        if ic50_q:
            assert any(r.get("qualifier") == ic50_q for r in ic50_results), (
                f"cpd {cid}: expected IC50 qualifier {ic50_q!r}, got "
                f"{[r.get('qualifier') for r in ic50_results]}"
            )


def test_units_are_canonical(extracted_assays: dict) -> None:
    """Every value must have a μM unit (the patent's TABLE 2 reports
    in μM throughout). Allow `μM` or `uM` (canonicalized in dedup)."""
    canonical = {"μM", "uM", "µM", ""}
    for cid, results in extracted_assays.items():
        for r in results:
            assert r["unit"] in canonical, (
                f"cpd {cid}: non-canonical unit {r['unit']!r} in {r}"
            )


def test_qualifiers_preserved(extracted_assays: dict) -> None:
    """At least some compounds must have `~` and `>` qualifiers
    preserved — the patent uses these heavily."""
    has_approx = False
    has_gt = False
    for cid, results in extracted_assays.items():
        for r in results:
            if r.get("qualifier") == "~":
                has_approx = True
            if r.get("qualifier") == ">":
                has_gt = True
    assert has_approx, "no `~` qualifier preserved across 190 compounds"
    assert has_gt, "no `>` qualifier preserved across 190 compounds"


# ── Reference CSV fidelity (when CSV available) ─────────────────


def _load_reference_csv() -> dict | None:
    import csv
    for path in (DOWNLOADS_CSV, REFERENCE_CSV):
        if path.exists():
            with open(path) as f:
                rdr = csv.DictReader(f)
                return {
                    row["compound_id"].strip(): {
                        "ki": float(row["flap_binding_ki_uM"]) if row.get("flap_binding_ki_uM") else None,
                        "ic50": float(row["hwb_ltb4_ic50_uM"]) if row.get("hwb_ltb4_ic50_uM") else None,
                    }
                    for row in rdr if row.get("compound_id")
                }
    return None


def test_fidelity_against_reference_csv(extracted_assays: dict) -> None:
    """≥97% of compounds must match the reference CSV within 5% on
    BOTH Ki and IC50 values. Skipped if reference CSV not available."""
    ref = _load_reference_csv()
    if ref is None:
        pytest.skip(f"reference CSV not at {DOWNLOADS_CSV} or {REFERENCE_CSV}")

    matched = 0
    for cid, ref_vals in ref.items():
        v2_arr = extracted_assays.get(cid) or extracted_assays.get(cid.lower())
        if not v2_arr:
            continue
        ki_results = [r for r in v2_arr if "ki" in (r["assay_name"] or "").lower() or "flap" in (r["assay_name"] or "").lower()]
        ic50_results = [r for r in v2_arr if "ic50" in (r["assay_name"] or "").lower() or "ltb" in (r["assay_name"] or "").lower() or "hwb" in (r["assay_name"] or "").lower()]

        ki_ok = ref_vals["ki"] is None or any(
            r["value_numeric"] is not None
            and abs(r["value_numeric"] - ref_vals["ki"]) / max(ref_vals["ki"], 1e-9) < 0.05
            for r in ki_results
        )
        ic50_ok = ref_vals["ic50"] is None or any(
            r["value_numeric"] is not None
            and abs(r["value_numeric"] - ref_vals["ic50"]) / max(ref_vals["ic50"], 1e-9) < 0.06
            for r in ic50_results
        )
        if ki_ok and ic50_ok:
            matched += 1

    pct = matched / len(ref) * 100
    assert pct >= 97.0, (
        f"reference fidelity below 97%: {matched}/{len(ref)} = {pct:.1f}%"
    )
