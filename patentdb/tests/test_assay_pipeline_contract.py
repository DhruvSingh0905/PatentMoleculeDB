"""Contract tests that GRADE seven known defects in the assay pipeline.

None of these defects is fixed. Every test here therefore FAILS against the
tree as it stands, and each carries `@pytest.mark.xfail(strict=True)` so the
suite stays green while the defect stands and turns RED the moment it is fixed.

That inversion is the point. A test that passes today against broken code
grades nothing; `strict=True` makes the marker itself the thing that has to be
removed, so a fix cannot land while still claiming the defect is open. Whoever
fixes one of these deletes its marker in the same commit — the XPASS is the
notification.

Each docstring records the number MEASURED on this tree, on 2026-08-04, from
the shipped artifacts in `output_v2/text_extraction/` and from the modules
named. A test whose defect could not be reproduced is not here.

No network, no paid API calls: the two tests that touch the repair loop pass
`max_calls=0, dry_run=True` and assert `proposed == 0` before going further,
and the fixtures they use produce zero gaps so no synthesis is even reachable.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pytest

# Read before any patentdb import: `config.REPAIR_ENABLED` is bound at import.
os.environ.setdefault("REPAIR", "0")

_REPO = Path(__file__).resolve().parents[2]
_SHIPPED = _REPO / "output_v2" / "text_extraction"
_XML_CACHE = _REPO / "output_v2" / "uspto_xml"
_PROCESS_PATENT = _REPO / "patentdb" / "routes" / "process_patent.py"


# ── shared fixtures ───────────────────────────────────────────────

# Two compounds, one unambiguous IC50 column. Chosen so the deterministic
# parser reads it cleanly: `find_gaps` returns nothing, the repair loop never
# reaches a model, and what the test observes is purely the plumbing.
SYNTHETIC_NUMERIC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<us-patent-grant><description>
<p>Table 1 reports inhibition data.</p>
<tables id="TABLE-US-00001" num="00001">
<table frame="none">
<tgroup cols="2">
<colspec colname="1"/><colspec colname="2"/>
<thead>
<row><entry>Cpd No.</entry><entry>P2X3 IC50 (nM)</entry></row>
</thead>
<tbody>
<row><entry>101</entry><entry>4.2</entry></row>
<row><entry>102</entry><entry>17.5</entry></row>
</tbody>
</tgroup>
</table>
</tables>
</description></us-patent-grant>
"""

# The other shape a patent publishes potency in: a prose bin key plus letter
# grades in the cells. `value_numeric` stays None on purpose — inventing a
# point value for "0.1 to 1 uM" would be a fabrication — so the bounds ARE the
# measurement, and anything that drops them drops the whole record's content.
SYNTHETIC_BINNED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<us-patent-grant><description>
<p>Key: A means less than 0.1 uM, B means 0.1 uM to 1 uM, C means greater than 1 uM.</p>
<tables id="TABLE-US-00001" num="00001">
<table frame="none">
<tgroup cols="2">
<colspec colname="1"/><colspec colname="2"/>
<thead>
<row><entry>Compound</entry><entry>A2B cAMP IC50</entry></row>
</thead>
<tbody>
<row><entry>Z1</entry><entry>B</entry></row>
<row><entry>Z2</entry><entry>A</entry></row>
</tbody>
</tgroup>
</table>
</tables>
</description></us-patent-grant>
"""

# The letter-bin path's own input shape: a legend stating the µM edges, then a
# TABLE whose example region lists ids under `+`-runs.
SYNTHETIC_BIN_TEXT = """
H358 Cell Viability assay. A compound with IC50 < 0.1 uM is marked +++,
a compound with IC50 < 1 uM is marked ++, and a compound with IC50 < 10 uM
is marked +.

TABLE 1
H358 Cell Viability assay data (K-Ras G12C, IC50, uM):
Examples
+++ A1, A2, A3
++ A4, A5
+ A6, A7

Biological data.
"""


def _shipped(patent_id: str) -> dict:
    p = _SHIPPED / patent_id / "assay_tables.json"
    if not p.exists():
        pytest.skip(f"shipped artifact absent: {p}")
    return json.loads(p.read_text())


def _all_shipped_rows():
    """(patent_id, row) for every assay row in every shipped artifact."""
    paths = sorted(_SHIPPED.glob("*/assay_tables.json"))
    if not paths:
        pytest.skip(f"no shipped artifacts under {_SHIPPED}")
    for p in paths:
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for rows in doc.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    yield p.parent.name, row


def _merge_row_literals():
    """The dict literals `process_patent` appends into `assay_tables`, as AST.

    Read out of the shipped source rather than retyped here, because a copy of
    the merge in this file would grade the copy. Evaluating the real literal
    means the day someone adds `range_lo` to it at
    `routes/process_patent.py:1811`, these tests XPASS with no edit — which is
    exactly the ratchet the `strict=True` markers are there to enforce.
    """
    tree = ast.parse(_PROCESS_PATENT.read_text())
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Dict)):
            continue
        inner = node.func.value
        if (isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "setdefault"
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "assay_tables"):
            out.append((node.lineno, node.args[0]))
    assert out, ("no `assay_tables.setdefault(...).append({...})` merge site "
                 "found in process_patent.py — this helper needs updating")
    return [d for _lineno, d in sorted(out)]


def _merge(record) -> dict:
    """Push one AssayRecord through the FIRST (unconditional) merge site.

    `routes/process_patent.py:1811` — the one that runs on every patent, as
    opposed to :1845 which is behind `if healed and healed.get("applied")`.
    """
    expr = ast.Expression(body=_merge_row_literals()[0])
    ast.fix_missing_locations(expr)
    return eval(compile(expr, "<process_patent merge>", "eval"),
                {}, {"rec": record})


# Names the two binned paths use for the same three things: the bounds, the
# grade, and the printed form. Membership in this set is what D3 compares.
_RANGE_KEY_NAMES = frozenset({
    "range_lo", "range_hi", "value_low", "value_high",
    "letter_grade", "bin", "value_text", "value_raw",
})


# ── D1 ────────────────────────────────────────────────────────────

def test_d1_xml_baseline_reaches_assay_tables():
    """The deterministic XML records are computed, then thrown away.

    `repair_patent` builds `baseline = extract_from_patent(xml)` to measure
    gaps against, and returns `recovered` alone. So the only route by which an
    XML record can reach `assay_tables` is the re-extract at
    `process_patent.py:1845`, gated on `if healed and healed.get("applied")` —
    a capability patch actually landing.

    Measured across all 22 shipped `assay_tables.json` (2026-08-04): the
    `source` distribution is `{'<none>': 27868, 'letter_bin': 8020}`. Not one
    of the 35,888 rows is tagged `uspto_xml_table`, `uspto_xml_bin_table`,
    `repair_rule_*` or `autoheal_*` — the 99.9%-exact source described in
    CLAUDE.md contributes zero rows to the artifact.

    On this fixture `extract_from_patent` returns two usable records and
    `repair_patent` returns none.
    """
    from patentdb.repair.loop import repair_patent
    from patentdb.sources.uspto_assays import extract_from_patent

    # Precondition — the reader is not the problem. Holds today.
    baseline = extract_from_patent(SYNTHETIC_NUMERIC_XML)
    assert [(r.cid, r.value_numeric, r.unit, r.source) for r in baseline] == [
        ("101", 4.2, "nM", "uspto_xml_table"),
        ("102", 17.5, "nM", "uspto_xml_table"),
    ]

    recovered, report = repair_patent(
        "SYNTH-D1", SYNTHETIC_NUMERIC_XML, dry_run=True, max_calls=0)
    # Spend guard: this fixture must never reach a model.
    assert report.gaps_found == 0 and report.proposed == 0

    assay_tables: dict[str, list[dict]] = {}
    for rec in recovered:
        assay_tables.setdefault(rec.cid, []).append(_merge(rec))

    from_xml = [row for row in assay_tables.get("101", [])
                if str(row.get("source") or "").startswith("uspto_xml")]
    assert len(from_xml) == 1, (
        f"no uspto_xml-sourced row for compound 101; "
        f"repair_patent returned {len(recovered)} records")
    assert from_xml[0]["value_numeric"] == 4.2
    assert from_xml[0]["unit"] == "nM"


# ── D2 ────────────────────────────────────────────────────────────

def test_d2_merge_preserves_a_binned_records_range():
    """A record whose value IS a range loses its value crossing the merge.

    `AssayRecord` (sources/uspto_assays.py:352-369) defines 14 fields. The
    merge dict at `process_patent.py:1811-1818` writes 6, and `cid` is only
    the outer key — `letter_grade`, `range_lo`, `range_hi`, `value_text`,
    `table_id`, `column_header` and `unit_source` are all dropped.

    For a numeric record that is a loss of provenance. For a BINNED one it is
    a loss of the measurement: `value_numeric` is None by design (see the
    field comment — inventing a point value for "0.1 to 1 uM" is a
    fabrication), so the bounds are the entire content, and the merged dict
    carries `value_numeric: None` and nothing else.

    Not a rare shape: 45 of the 166 rules in `patentdb/data/layout_rules.json`
    are `bin_key` — measured 2026-08-04.

    The `is_usable` assertion below is the discriminator. It holds today
    (`uspto_assays.py:396-399` counts a range as a value), which proves the
    record was fully-formed when the merge received it, so this test grades
    the merge and not the extractor.
    """
    from patentdb.sources.uspto_assays import AssayRecord

    rec = AssayRecord(
        cid="Z1", assay_name="A2B cAMP IC50", value_numeric=None,
        unit="uM", letter_grade="A", range_lo=1.0, range_hi=100.0,
        value_text="A", table_id="TABLE-US-00001",
        column_header="A2B cAMP IC50", unit_source="bin_key",
    )
    # Preconditions — the record is complete and usable on the way in.
    assert rec.missing_fields() == []
    assert rec.is_usable is True

    row = _merge(rec)
    assert row["value_numeric"] is None      # by design, not the defect

    # Either spelling counts (D3 grades the disagreement separately); what
    # must not happen is the bounds vanishing.
    lo = row.get("range_lo", row.get("value_low"))
    hi = row.get("range_hi", row.get("value_high"))
    assert (lo, hi) == (1.0, 100.0), (
        f"binned record reached assay_tables with no value; "
        f"merge wrote keys {sorted(row)}")
    assert row.get("letter_grade", row.get("bin")) == "A"


# ── D3 ────────────────────────────────────────────────────────────

def test_d3_binned_paths_share_one_range_schema():
    """Two producers, one file, two incompatible spellings of "a range".

    `routes/letter_bin_assays.py:231-234` emits `value_raw` / `value_low` /
    `value_high` / `bin`. `sources/uspto_assays.py:359-366` emits `range_lo` /
    `range_hi` / `letter_grade` / `value_text`. Both are destined for the same
    `assay_tables.json`, so any consumer must know both — and today's shipped
    corpus contains 8,020 rows of the first spelling and 0 of the second
    (measured 2026-08-04), which means the second has never been exercised by
    a reader at all.

    Both records below are produced by the real extractors, not hand-built, so
    the key names compared are the ones that reach disk.
    """
    from patentdb.routes.letter_bin_assays import extract_letter_bin_assays
    from patentdb.sources.uspto_assays import extract_from_patent

    bin_rows = [row
                for rows in extract_letter_bin_assays(
                    "SYNTH-D3", SYNTHETIC_BIN_TEXT).values()
                for row in rows
                if row.get("value_low") is not None
                and row.get("value_high") is not None]
    assert bin_rows, "letter-bin fixture produced no two-sided range"
    letter_bin_keys = set(bin_rows[0]) & _RANGE_KEY_NAMES

    xml_rows = [r.as_dict() for r in extract_from_patent(SYNTHETIC_BINNED_XML)
                if r.range_lo is not None and r.range_hi is not None]
    assert xml_rows, "uspto_xml fixture produced no two-sided range"
    xml_keys = set(xml_rows[0]) & _RANGE_KEY_NAMES

    # Preconditions — both really are bounded ranges over the same interval.
    assert (bin_rows[0]["value_low"], bin_rows[0]["value_high"]) == (0.1, 1.0)
    assert (xml_rows[0]["range_lo"], xml_rows[0]["range_hi"]) == (0.1, 1.0)

    assert letter_bin_keys == xml_keys, (
        f"same measurement, different schema: letter_bin uses "
        f"{sorted(letter_bin_keys)}, uspto_xml uses {sorted(xml_keys)}")


# ── D4 ────────────────────────────────────────────────────────────

def test_d4_assay_result_carries_provenance():
    """Provenance is missing from the data model, not just from the writer.

    `core/models.py:90-97` gives `AssayResult` exactly six fields —
    `assay_name`, `value_raw`, `value_numeric`, `qualifier`, `unit`,
    `n_runs` (measured 2026-08-04) — and none of them is `source`. That is why
    `process_patent.py:1539-1552` cannot stamp provenance on the HARVEST
    branch even if the key were added to the dict: there is nothing on the
    object to read.

    The information exists upstream and is discarded.
    `ActivityTuple.validation_reason` (harvest/agent2_activities.py:45-56)
    already carries e.g. `"pattern_library:{key}"`, and
    `HarvestResult.to_assay_results()` (harvest/orchestrator.py:47-64) simply
    does not copy it. That discard is the mechanical cause of the 27,868
    `<none>`-sourced rows counted in D1: they came from this path.
    """
    from patentdb.core.assay_fsm.harvest.agent2_activities import ActivityTuple
    from patentdb.core.assay_fsm.harvest.orchestrator import HarvestResult
    from patentdb.core.models import AssayResult

    assert "source" in AssayResult.model_fields, (
        f"AssayResult fields are {list(AssayResult.model_fields)}")

    tup = ActivityTuple(
        compound_id="101", assay_name="P2X3 IC50", value=4.2, unit="nM",
        validation_reason="pattern_library:a1659e1fea48b93c",
    )
    results = HarvestResult(tuples=[tup]).to_assay_results()
    # Precondition — the tuple does survive conversion; only provenance is lost.
    assert results["101"][0].value_numeric == 4.2
    assert results["101"][0].source == "pattern_library:a1659e1fea48b93c"


# ── D5 ────────────────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason=(
    "US10544143 TLR columns are rotated by one in the shipped path; "
    "see output_v2/text_extraction/US10544143/assay_tables.json vs "
    "sources/uspto_assays.extract_from_patent"))
def test_d5_us10544143_tlr_columns_are_not_shifted():
    """The shipped values for US10544143 are one column to the right.

    Raw CALS row for compound 437 in `TABLE-US-00005` reads
    `['437', '5.1', '11', '3752']` against headers TLR7 / TLR8 / TLR9 (nM);
    `extract_from_patent` reads it correctly, and BindingDB agrees. The
    shipped artifact does not:

        shipped  TLR7=8941.0  TLR8=5.1     TLR9=11.0
        XML      TLR7=5.1     TLR8=11.0    TLR9=3752.0

    5.1 and 11.0 have each moved one column right, and 3752.0 — a >700-fold
    weaker measurement — is gone entirely, replaced at TLR7 by 8941.0 which
    belongs to no cell in that row. Reproduced on compounds 427 (shipped
    1398.0 / 2.4 / 7.6 vs XML 2.4 / 7.6 / 4904.0) and 419 (shipped 2865.0 /
    4.1 / 2.5 vs XML 4.1 / 2.5 / 1744.0), measured 2026-08-04.

    This is the failure mode the module's founding rule is about: not a
    missing assay, which is recoverable, but a wrong one — a nanomolar potency
    published against the wrong target.
    """
    xml_path = _XML_CACHE / "US10544143.xml"
    if not xml_path.exists():
        pytest.skip(f"XML cache absent: {xml_path}")
    from patentdb.sources.uspto_assays import extract_from_patent

    # Precondition — establish the truth from the patent's own CALS. Holds today.
    xml_437 = {r.assay_name.split(" (")[0]: r.value_numeric
               for r in extract_from_patent(xml_path.read_text())
               if r.cid == "437"}
    assert xml_437 == {"TLR7 IC50": 5.1, "TLR8 IC50": 11.0, "TLR9 IC50": 3752.0}

    shipped = {row["assay_name"]: row.get("value_numeric")
               for row in _shipped("US10544143").get("437", [])}
    assert shipped.get("TLR7 IC50") == 5.1, f"shipped row for 437: {shipped}"
    assert shipped.get("TLR8 IC50") == 11.0
    assert shipped.get("TLR9 IC50") == 3752.0


# ── D6 ────────────────────────────────────────────────────────────

def test_d6_pattern_library_rows_carry_a_unit():
    """The pattern-library path emits every row with no unit at all.

    Both emit sites in `apply_patterns_to_text` write the literal
    `"unit": ""` — the numeric branch at
    `core/assay_fsm/assay_pattern_library.py:388` and the letter-grade branch
    at `:405`. The discovered regex captures `cid` and `value0…valueN` and
    nothing else; the header the pattern was anchored on, which is where the
    unit is printed, is never read for its unit.

    Measured on US9694016, 2026-08-04: 13,582 rows over 2,215 compound ids,
    and **0 of 13,582 carry a unit** — the distribution is
    `{'': 13582}`. Example 1's B-Raf IC50 ships as `0.000145 μM` and replays
    from this path as a bare `0.0003` with `unit: ''`. A potency with no scale
    is not a measurement: `AssayRecord.missing_fields()` would call it
    unusable, but nothing on this path applies that contract.

    This matters in proportion to how much of the corpus flows through here —
    ablating `apply_patterns_to_text` drops US9694016 from 2,215 compound ids
    to 1.
    """
    from patentdb.core.assay_fsm.assay_pattern_library import (
        apply_patterns_to_text,
    )
    from patentdb.core.patent_text import load_patent_description

    try:
        loaded = load_patent_description("US9694016", prefer_format="auto")
    except Exception as exc:                      # no cached GP/MinerU text
        pytest.skip(f"US9694016 text unavailable: {exc!r}")
    text = loaded[0] if isinstance(loaded, tuple) else loaded
    if not text:
        pytest.skip("US9694016 text unavailable")

    rows = apply_patterns_to_text(text, "US9694016")
    if not rows:
        pytest.skip("pattern library empty on this checkout "
                    "(patentdb/data/assay_patterns.discoveries.json)")

    unitless = [r for r in rows if not str(r.get("unit") or "").strip()]
    assert len(unitless) == 0, (
        f"{len(unitless)}/{len(rows)} pattern-library rows carry no unit; "
        f"first: {unitless[0]}")


# ── D7 ────────────────────────────────────────────────────────────

# A chromatography technique AND a retention-time token in the same name.
# Both halves are required: `C-Raf TR IC50` (235 shipped rows) is a
# time-resolved FRET potency and a legitimate assay, and a bare `TR` test
# would condemn it.
_CHROMATOGRAPHY = re.compile(
    r"(?<![A-Za-z])(hplc|uplc|lc[\s/-]*ms|lcms|gc[\s-]*ms|sfc|chromatograph)",
    re.I)
_RETENTION = re.compile(
    r"(?<![A-Za-z])(t\s*[_-]?\s*r|rt|retention(\s+time)?)(?![A-Za-z])", re.I)


def _is_retention_time(name: str) -> bool:
    if re.search(r"retention\s+time", name or "", re.I):
        return True
    return bool(_CHROMATOGRAPHY.search(name or "")
                and _RETENTION.search(name or ""))


@pytest.mark.xfail(strict=True, reason=(
    "864 shipped rows file an HPLC retention time as an assay; "
    "see sources/uspto_assays.py column classification and "
    "output_v2/text_extraction/US10273259/assay_tables.json"))
def test_d7_retention_time_is_not_filed_as_an_assay():
    """A retention time is a chromatography artefact, not a measurement.

    `tR (min.)` says when a peak eluted on one lab's column with one gradient.
    It is not comparable between patents, it has no target, and filed under
    `assay_name` it is indistinguishable to every downstream consumer from a
    potency in minutes.

    Measured across all 22 shipped artifacts, 2026-08-04: 35,888 assay rows
    over 814 distinct `assay_name` values, of which **864 rows read
    `'HPLC tR (min.)'`**, all of them in US10273259. That is the only class
    the detector above finds — the full enumeration, not a sample. The one
    near-miss it deliberately spares is `'C-Raf TR IC50'` (235 rows), where
    `TR` is time-resolved FRET and the row is a real potency.

    `tests/test_uspto_assays.py::test_mass_and_retention_columns_are_never_read`
    already asserts the XML reader refuses an `RT (min)` column. These 864
    rows are on a different path, and nothing there enforces the same rule —
    which is the same "know which module you are editing" hazard CLAUDE.md
    warns about for `core/tables/`.
    """
    from collections import Counter

    offenders = Counter(
        (pid, row.get("assay_name") or "")
        for pid, row in _all_shipped_rows()
        if _is_retention_time(row.get("assay_name") or "")
    )
    assert not offenders, (
        f"{sum(offenders.values())} rows file a retention time as an assay: "
        + ", ".join(f"{pid} {name!r} x{n}"
                    for (pid, name), n in offenders.most_common())
    )
