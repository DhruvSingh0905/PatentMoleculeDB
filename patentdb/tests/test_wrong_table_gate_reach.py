"""How far down a `TABLE <n>` caption's authority reaches.

`_under_foreign_header` decides whether a pattern-library row sits under a
caption that names a DIFFERENT measurement. It answered that question only for
rows in the first `_HEADER_SCAN` (2,000) characters below the caption and
returned False — "no evidence" — for everything past it.

Patent tables are not 2,000 characters long. US10544143's `TABLE 1`
(`Ex. No. | Structure | Mol. Wt. | LCMS M+ | Ret Time (min) | HPLC Method`)
runs 64,587 chars in the grant XML and another 61,289 in the Google Patents
rendering of the same document. The first data row that survives the regex sits
2,007 chars below the caption — seven characters past the budget — so on the
patent this gate exists for, it blocked 714 of 10,605 candidate rows and let
the rest through as `TLR7/TLR8/TLR9 IC50 (nM)`.

These tests pin the boundary rule, not the constant:

  * a caption governs everything up to the NEXT caption, and
  * its authority ends early where the pattern's OWN header text appears,
    which is what a Google-Patents-flattened table looks like when the
    `TABLE <n>` markup is lost (US10273259's RORγ table).

No network, no paid calls. The two real-data tests skip when their cache is
absent.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pytest

# Read before any patentdb import: `config.REPAIR_ENABLED` binds at import.
# Nothing else is set here — pytest imports every test module into ONE process,
# so an `os.environ.setdefault` in this file is a process-wide default. Setting
# `PARSER_REPAIR_APPLY=0` here failed 7 tests in `test_repair_write_gate.py` and
# `test_uspto_xml.py`, which are about that switch.
os.environ.setdefault("REPAIR", "0")

_REPO = Path(__file__).resolve().parents[2]
_XML_CACHE = _REPO / "output_v2" / "uspto_xml"


# ── hermetic fixtures ─────────────────────────────────────────────
#
# `fresh_patterns` entries are stamped with the caller's patent id, so these
# run against a pattern this test owns rather than whatever the library
# happens to hold on this checkout.

_MASS_PATTERN = {
    "regex": r"^\s*(?P<cid>\d+)\s+(?P<value0>[\d.]+)\s+(?P<value1>[\d.]+)"
             r"\s+(?P<value2>[\d.]+)\s*$",
    "column_assays": ["TLR7 IC50 (nM)", "TLR8 IC50 (nM)", "TLR9 IC50 (nM)"],
}


def _mass_table(n_rows: int) -> str:
    """A `Mol. Wt. / LCMS M+ / Ret Time` table, one cell per line — the shape
    the grant XML's CALS markup flattens to."""
    head = "TABLE 1\n\nRet\n\nEx.\n\nMol.\n\nLCMS\n\nTime\n\nHPLC\n\n" \
           "No.\n\nStructure\n\nWt.\n\nM + \n\n(min)\n\nMethod\n"
    rows = "".join(
        f"\n{i}\n\n{400 + i}.51\n\n{401 + i}.3\n\n0.7{i % 10}\n\nQC- ACN- AA-XB\n"
        for i in range(1, n_rows + 1)
    )
    return head + rows


def _assay_table(n_rows: int) -> str:
    """The real thing: a caption that names the assay the pattern describes."""
    head = "TABLE 4\n\nTLR7/8/9 Reporter Assay Data\n\nTLR7\n\nTLR8\n\nTLR9\n\n" \
           "Ex.\n\nIC 50 \n\nIC 50 \n\nIC 50 \n\nNo.\n\n(nM)\n\n(nM)\n\n(nM)\n"
    rows = "".join(
        f"\n{i}\n\n{i}.5\n\n{i}.7\n\n{700 + i}\n"
        for i in range(1, n_rows + 1)
    )
    return head + rows


def _run(text: str, patent_id: str = "USTEST0001") -> list[dict]:
    from patentdb.core.assay_fsm.assay_pattern_library import apply_patterns_to_text
    return apply_patterns_to_text(
        text, patent_id, fresh_patterns=[dict(_MASS_PATTERN)],
    )


def _mine(rows: list[dict]) -> list[dict]:
    """Only the rows this test's own pattern produced."""
    return [r for r in rows
            if "TLR" in (r.get("assay_name") or "")
            and r.get("validation_reason", "").startswith("pattern_library:")]


# ── 1. the caption reaches the whole table ────────────────────────

def test_a_mass_table_is_blocked_past_the_first_2000_chars():
    """Every row of a `Mol. Wt. / LCMS M+ / Ret Time` table is a wrong-table
    row, however far below the caption it is printed.

    The table below is ~7,000 chars; the caption names masses and retention
    times and does not name TLR7/8/9. Before the fix the gate stopped looking
    2,000 chars in and shipped the tail as nanomolar potencies.
    """
    text = _mass_table(250) + "\n\n" + _assay_table(3)
    assert len(text) > 6_000, "fixture must outrun the old 2,000-char budget"
    rows = _mine(_run(text))
    from_mass_table = [
        r for r in rows if r["source_offset"] < text.index("TABLE 4")
    ]
    assert from_mass_table == [], (
        f"{len(from_mass_table)} row(s) escaped the mass table, e.g. "
        f"{from_mass_table[:3]}"
    )


def test_the_assay_table_itself_still_reports():
    """The other half of the contract: the widened caption must not silence
    the table the pattern was learned from."""
    text = _mass_table(120) + "\n\n" + _assay_table(30)
    rows = _mine(_run(text))
    assert rows, "the pattern's own table produced nothing"
    cids = {r["compound_id"] for r in rows}
    assert len(cids) >= 25, f"only {len(cids)} compounds read from TABLE 4"


def test_an_uncaptioned_table_of_its_own_kind_ends_the_captions_authority():
    """Google Patents renders some tables as flat prose with the `TABLE <n>`
    caption gone. The nearest caption above such a table then belongs to a
    different table entirely, and reading it as this table's header deletes
    real rows (US10273259's RORγ binding data sits 2,787 chars under
    `TABLE 19 LCMS m/z HPLC ... t R (min) method`).

    The signal that the caption no longer governs is the pattern's OWN header
    printed in between.
    """
    text = (
        _mass_table(60)
        + "\n\nIC 50 values for compounds of the invention are provided below.\n\n"
        + "Ex.\n\nTLR7\n\nTLR8\n\nTLR9\n\nNo.\n\nIC 50 (nM)\n\nIC 50 (nM)\n\nIC 50 (nM)\n"
        + "".join(f"\n{i}\n\n{i}.5\n\n{i}.7\n\n{700 + i}\n" for i in range(1, 31))
    )
    rows = _mine(_run(text))
    tail = [r for r in rows if r["source_offset"] > text.index("IC 50 values")]
    assert len({r["compound_id"] for r in tail}) >= 25, (
        "the uncaptioned assay table was swallowed by the mass table's caption: "
        f"{len(tail)} row(s) survived"
    )


# ── 2. the patent this gate exists for ────────────────────────────

def _table_us_00002_cells() -> dict[str, tuple[float, float, float]]:
    """{Ex. No. → (Mol. Wt., LCMS M+, Ret Time)} straight out of the raw CALS.

    Read with a local regex rather than the assay parser on purpose — the
    assay parser correctly refuses this table, which is the whole point.
    """
    xml = (_XML_CACHE / "US10544143.xml").read_text()
    m = re.search(r'<tables id="TABLE-US-00002".*?</tables>', xml, re.S)
    assert m, "TABLE-US-00002 not found in the cached grant XML"
    out: dict[str, tuple[float, float, float]] = {}
    for row in re.findall(r"<row>(.*?)</row>", m.group(0), re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<entry[^>]*>(.*?)</entry>", row, re.S)]
        if len(cells) >= 5 and re.fullmatch(r"\d+", cells[0]):
            try:
                out[cells[0]] = (float(cells[2]), float(cells[3]), float(cells[4]))
            except ValueError:
                pass
    return out


_TLR_COL = {"TLR7 IC50 (nM)": 0, "TLR8 IC50 (nM)": 1, "TLR9 IC50 (nM)": 2}


def _us10544143_rows():
    from patentdb.core import config
    from patentdb.core.assay_fsm.assay_pattern_library import apply_patterns_to_text
    from patentdb.core.assay_fsm.pipeline import _gather_full_text

    if not (_XML_CACHE / "US10544143.xml").exists():
        pytest.skip("XML cache absent")
    text = _gather_full_text("US10544143", config.DATA_DIR)
    if not text:
        pytest.skip("US10544143 text unavailable")
    return apply_patterns_to_text(text, "US10544143")


def test_us10544143_no_compound_reports_its_property_row_as_potencies():
    """`TABLE-US-00002` is US10544143's property table: `Ex. No. | Structure |
    Mol. Wt. | LCMS M+ | Ret Time (min) | HPLC Method`. Compound 79's row reads
    `405.51 | 406.1 | 0.65`.

    Measured on the shipped artifact of 2026-08-06 18:39
    (`output_v2/text_extraction/US10544143/assay_tables.json`): 1,770 rows are
    literally one of those cells, filed under `TLR7/TLR8/TLR9 IC50` with
    `unit: nM`, all from `pattern_library:7f9fc5d030c7e9e0`. Compound 79 ships
    `TLR7 = 405.51 nM, TLR8 = 406.1 nM, TLR9 = 0.65 nM` beside the CALS
    reader's correct `373 / 355 / 9724`. A molecular weight is not a potency,
    and a retention time in MINUTES is not a nanomolar potency.

    The assertion is on a compound reporting its WHOLE property row across the
    three columns, in order. One column matching by itself proves nothing —
    a real 1.6 nM potency and a real 1.6 min retention time collide — but
    (Mol. Wt., M+, Ret Time) landing in (TLR7, TLR8, TLR9) is that row and
    nothing else. Measured on this text: 634 compounds before, 0 after.

    Asserts on the live extraction, not on the artifact, so it stays true
    across re-runs.
    """
    cells = _table_us_00002_cells()
    assert cells, "no numeric rows read from TABLE-US-00002"
    hit = defaultdict(set)
    for r in _us10544143_rows():
        i = _TLR_COL.get(r.get("assay_name") or "")
        c = cells.get(r.get("compound_id") or "")
        if i is None or c is None or r.get("value") is None:
            continue
        if abs(r["value"] - c[i]) < 1e-9:
            hit[r["compound_id"]].add(i)
    offenders = sorted(cid for cid, cols in hit.items() if cols == {0, 1, 2})
    assert not offenders, (
        f"{len(offenders)} compound(s) report their Mol.Wt / LCMS M+ / Ret Time "
        f"row as TLR7 / TLR8 / TLR9 potencies in nM, e.g. {offenders[:5]}"
    )


@pytest.mark.xfail(strict=True, reason=(
    "OPEN, and NOT the wrong-table defect: an unanchored row-regex slides one "
    "field out of frame inside the CORRECT table. See the docstring."
))
def test_us10544143_no_pattern_row_contradicts_the_patents_own_cals():
    """The residue after the caption-reach fix, and the defect D5 documents.

    `pattern_library:fabd55b27b1eae19` carries no line anchors:
    `(?P<cid>\\d+)\\s+(?P<value0>...)\\s+(?P<value1>...)\\s+(?P<value2>...)`.
    The grant XML flattens one CALS cell per line, so every token looks like a
    row start; once a cell the value alternation cannot match goes by
    (`686 >3125 >3125 >50000` — the regex accepts `&gt;` but not a bare `>`)
    the scan resynchronises one field late and stays there:

        text   … 276  8.9  9.0  309 │ 277  0.40  1.6  1378 …
        match                  ^cid=309, 277, 0.40, 1.6

    So compound 309 is issued `TLR9 = 1.6 nM` — which is Ex 277's TLR8, and
    also, by coincidence, compound 309's own retention time in TABLE-US-00002.
    Its true TLR9 is 268. Compound 437 gets `694 / 2309 / 2850` the same way,
    read off Ex 693's tail; its true row is `5.1 / 11 / 3752`.

    Widening the caption's reach does not touch this: these matches are inside
    `TABLE 4 TLR7/8/9 Reporter Assay Data`, the pattern's own table. Neither
    line anchoring nor a preceding-token test separates them, because every
    cell sits on its own line and every row starts with a bare integer. No
    measured fix exists yet, so this grades the defect rather than hiding it.
    """
    from patentdb.sources.uspto_assays import extract_from_patent

    truth: dict[str, dict[str, float]] = defaultdict(dict)
    for rec in extract_from_patent((_XML_CACHE / "US10544143.xml").read_text()):
        if rec.value_numeric is not None:
            truth[rec.cid][rec.assay_name] = rec.value_numeric
    bad = []
    for r in _us10544143_rows():
        t = truth.get(r.get("compound_id") or "", {}).get(r.get("assay_name") or "")
        v = r.get("value")
        if t is None or v is None:
            continue
        if abs(v - t) > max(0.05 * abs(t), 1e-9):
            bad.append((r["compound_id"], r["assay_name"], v, t))
    assert not bad, (
        f"{len(bad)} pattern row(s) contradict the patent's own CALS, "
        f"e.g. {sorted(set(bad))[:4]}"
    )


def test_us10273259_rorgamma_binding_rows_survive():
    """The regression guard for widening the caption's reach.

    US10273259's RORγ binding table is printed with no `TABLE <n>` caption of
    its own — the nearest one above is `TABLE 19 LCMS m/z HPLC HPLC Ex. #
    Structure observed t R (min) method`, 2,787 chars up. That patent is
    titled "Tricyclic sulfones as RORγ modulators"; these rows are its primary
    assay, and a caption-reach rule that convicts them is worse than the
    defect it fixes.
    """
    if not (_XML_CACHE / "US10273259.xml").exists():
        pytest.skip("XML cache absent")
    from patentdb.core import config
    from patentdb.core.assay_fsm.assay_pattern_library import apply_patterns_to_text
    from patentdb.core.assay_fsm.pipeline import _gather_full_text

    text = _gather_full_text("US10273259", config.DATA_DIR)
    if not text or "ROR" not in text:
        pytest.skip("US10273259 text unavailable")
    rows = [r for r in apply_patterns_to_text(text, "US10273259")
            if (r.get("assay_name") or "").startswith("ROR")]
    if not rows:
        pytest.skip("no RORγ pattern in the library on this checkout "
                    "(patentdb/data/assay_patterns.discoveries.json)")
    got = {(r["compound_id"], r["value"]) for r in rows}
    # Straight off the flattened text: `... 3 0.065 4 0.107 ... 39 0.024 ...`
    for cid, val in (("3", 0.065), ("4", 0.107), ("39", 0.024), ("62", 0.033)):
        assert (cid, val) in got, (
            f"compound {cid}'s RORγ Binding IC50 {val} μM was dropped; "
            f"{len(got)} RORγ pairs survived"
        )
