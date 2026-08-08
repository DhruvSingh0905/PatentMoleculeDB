"""Tests for the STRUCTURES artifact's wiring — `verify.dump`'s second file.

What this file locks in, and why each assertion exists:

  - `verify.STRUCT_FIELDS` is the UNION of `NamedCompound` and `TableName`,
    `NamedCompound`'s order first. Both identity routes write into ONE file
    (`config.STRUCTURES`), so the header has to cover both shapes; pinning the
    leading order means an existing consumer's column positions do not move
    when the table route is added behind it.
  - Both routes actually reach the file, distinguished by `source`. An import
    is not evidence of wiring — v3 has no reachability audit (CLAUDE.md,
    "Audit wiring before reporting any number"), so this runs `dump()` and
    reads the bytes back.
  - The manifest tells "route ran, found nothing" apart from "route never
    ran", for the table route and the repair tier as well as for
    `iupac_names`.

EVERY PATH IS REDIRECTED TO `tmp_path`. `dump()` writes `config.DUMP`,
`config.STRUCTURES`, `config.MANIFEST` and the loss log — the real, canonical
artifacts a human reads. A test that overwrote them would silently replace a
measurement run's output with a two-row fixture, which is precisely the
"which run produced this file" failure the manifest exists to prevent. The
module-level path constants are monkeypatched instead, and `losses.reset()`
is pointed at `tmp_path` too.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from patentdb3 import verify
from patentdb3.core import config
from patentdb3.sources import losses
from patentdb3.sources.iupac_names import NamedCompound
from patentdb3.sources.table_names import TableName

PID = "US10214537"     # both routes are non-empty here: prose AND name columns


def _have_xml(pid: str = PID):
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path


# ── the header, pure ─────────────────────────────────────────────────────

def test_struct_fields_is_the_union_of_both_dataclasses():
    named = [f.name for f in dataclasses.fields(NamedCompound)]
    table = [f.name for f in dataclasses.fields(TableName)]
    assert set(verify.STRUCT_FIELDS) == set(named) | set(table)
    # no duplicates — a repeated column would make the TSV unreadable by name
    assert len(verify.STRUCT_FIELDS) == len(set(verify.STRUCT_FIELDS))


def test_struct_fields_keeps_named_compound_field_order_in_front():
    named = [f.name for f in dataclasses.fields(NamedCompound)]
    assert list(verify.STRUCT_FIELDS[:len(named)]) == named


def test_both_dataclasses_carry_the_repair_provenance_field():
    """`name_repair` is the only thing that sets `.repair`, on either route,
    and `verify` counts it for the manifest. If it is dropped from one of the
    two dataclasses the count silently under-reports rather than failing."""
    for cls in (NamedCompound, TableName):
        names = {f.name for f in dataclasses.fields(cls)}
        assert "repair" in names, cls.__name__
        assert "source" in names, cls.__name__


def test_the_two_routes_declare_different_sources():
    """`source` is the ONLY thing separating the two routes inside one file —
    the manifest's `structures_sources` tally and every consumer's filter
    depend on the defaults differing."""
    assert NamedCompound.source == "description"
    assert TableName.source == "table"


# ── end to end through `dump()` ──────────────────────────────────────────

@pytest.fixture()
def redirected(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "DUMP_PATH", tmp_path / "reader_dump.tsv")
    monkeypatch.setattr(verify, "STRUCT_PATH", tmp_path / "structures.tsv")
    monkeypatch.setattr(verify, "MANIFEST_PATH", tmp_path / "latest.json")
    losses.reset(tmp_path / "loss_log.jsonl")
    yield tmp_path
    losses.reset(losses.LOSS_LOG)


def _read_struct(path):
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return header, [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def test_dump_writes_both_routes_into_the_one_structures_file(redirected, monkeypatch):
    pytest.importorskip("py2opsin")
    _have_xml()
    monkeypatch.setattr(config, "IUPAC_NAMES", True)
    verify.dump([PID], heal=False)

    header, rows = _read_struct(redirected / "structures.tsv")
    assert header == list(verify.STRUCT_FIELDS)
    if not rows:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    sources = {r["source"] for r in rows}
    # `heading` is `extract_names`'s own second producer (whole-heading names)
    # and rides in on the description route; `table` is the separate
    # extractor. Both of the two ROUTES must be present.
    assert {"description", "table"} <= sources, (
        f"one of the two identity routes did not reach the artifact: {sources}")
    assert sources <= {"description", "heading", "table"}, sources

    manifest = json.loads((redirected / "latest.json").read_text())
    assert manifest["iupac_names"] is True
    assert manifest["structures_rows"] == len(rows)
    assert manifest["structures_fields"] == list(verify.STRUCT_FIELDS)
    assert set(manifest["structures_sources"]) == sources
    assert sum(manifest["structures_sources"].values()) == len(rows)
    # `structures_repaired` counts rows carrying a confirmed `name_repair`
    # pattern id, on either route — the manifest's only view of that tier.
    assert manifest["structures_repaired"] == sum(1 for r in rows if r["repair"])


def test_a_table_row_carries_table_provenance_a_description_row_does_not(
        redirected, monkeypatch):
    """The union header is not cosmetic: each route fills its OWN columns and
    leaves the other's empty. If they ever collapsed to one shape, a row's
    provenance — the whole reason both routes are kept unmerged — would be
    unreadable off the file."""
    pytest.importorskip("py2opsin")
    _have_xml()
    monkeypatch.setattr(config, "IUPAC_NAMES", True)
    verify.dump([PID], heal=False)
    _, rows = _read_struct(redirected / "structures.tsv")
    if not rows:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")

    tbl = [r for r in rows if r["source"] == "table"]
    desc = [r for r in rows if r["source"] == "description"]
    assert tbl and desc
    assert all(r["table_id"] and r["raw_cell"] for r in tbl)
    assert all(not r["table_id"] and not r["raw_cell"] for r in desc)
    # `start` is the description route's own offset; a table row has none.
    assert all(r["start"] == "" for r in tbl)
    # a heading row is neither: no table coordinates, no text offset (-1),
    # and it names which whole-heading transformation produced it
    for r in [r for r in rows if r["source"] == "heading"]:
        assert not r["table_id"] and r["start"] == "-1"
        assert r["heading_transform"] in {
            "as_is_whole", "dewrap_whole", "dewrap_seeded"}
        assert r["cid"], "a heading row must carry the id the heading stated"


def test_disabled_route_still_writes_the_file_and_says_so(redirected, monkeypatch):
    """`IUPAC_NAMES=0` must produce an EMPTY structures file plus a manifest
    that reads `iupac_names: false` — "the routes did not run" and "the routes
    found nothing" are different facts and the artifact alone cannot tell them
    apart."""
    _have_xml()
    monkeypatch.setattr(config, "IUPAC_NAMES", False)
    verify.dump([PID], heal=False)
    header, rows = _read_struct(redirected / "structures.tsv")
    assert header == list(verify.STRUCT_FIELDS)
    assert rows == []
    manifest = json.loads((redirected / "latest.json").read_text())
    assert manifest["iupac_names"] is False
    assert manifest["structures_rows"] == 0
    assert manifest["structures_sources"] == {}
    assert manifest["structures_repaired"] == 0


def test_a_crashing_table_route_costs_neither_other_artifact(redirected, monkeypatch):
    """`extract_table_names` is the newest thing in this path. Its failure
    must not be able to cost the assay dump OR the description route their
    rows — the same guarantee `extract_names` already has."""
    _have_xml()
    monkeypatch.setattr(config, "IUPAC_NAMES", True)

    def boom(xml, pid=""):
        raise RuntimeError("table route exploded")

    monkeypatch.setattr(verify, "extract_table_names", boom)
    monkeypatch.setattr(verify, "extract_names",
                        lambda xml, pid="": [NamedCompound(
                            patent_id=pid, name="benzene", smiles="c1ccccc1",
                            inchikey="UHOVQNZJYSORNB-UHFFFAOYSA-N", start=0)])
    verify.dump([PID], heal=False)

    _, rows = _read_struct(redirected / "structures.tsv")
    assert [r["source"] for r in rows] == ["description"]
    assert (redirected / "reader_dump.tsv").read_text().count("\n") > 1
    manifest = json.loads((redirected / "latest.json").read_text())
    assert manifest["loss_counts"].get("extract_table_names_exception") == 1
