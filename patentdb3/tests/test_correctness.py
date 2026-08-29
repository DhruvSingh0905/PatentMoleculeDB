"""Is the OUTPUT true to the document? Six checks, no mocks, no fixtures.

`test_end_to_end.py` proves the pipeline is wired: the manifest agrees with
its artifact, a crash in one route costs another nothing, a healed dump is
reproducible. That is plumbing, and plumbing can be perfect while every number
in the artifact is wrong.

This file asks the other question. Each check reads a real cached patent,
takes what the pipeline produced, and holds it against what the document
itself says — a value against the row it was printed in, a letter grade
against the legend that defines it, a drawing marker against the picture in
that compound's own row. Nothing here is compared to a reference database:
every oracle is the patent.

WHY SIX AND NOT SIXTY. A count of passing tests measures nothing. What matters
is whether a check can fail on a defect that has actually happened, so each of
these is anchored to a real one — the failure it would have caught is named in
its docstring, with the patent and the row count. A check that cannot fail is
worse than no check, because it makes the suite look thorough.

The patents are chosen for what they can prove, not for convenience:

  US10125101   two compound-id columns, the second numbering ANOTHER PATENT's
               examples — the layout where a value can migrate to a compound
               that never had it.
  US11566007   grades only, no numbers at all: 8,652 records whose entire
               meaning comes from a legend elsewhere in the document.
  US9718790    2,930 drawings and a `Structure | Compound No. | ...` layout —
               the compound number is not in the first cell.
  US9303033    drawings and data in SEPARATE blocks, paired by title.
  US9987276    one prose legend governing tables that never restate it.
"""
from __future__ import annotations

import collections
import re
import warnings

import pytest

from patentdb3.core import config

warnings.filterwarnings("ignore")

_ROW = re.compile(r"<row>.*?</row>", re.S)
_ENTRY = re.compile(r"<entry[^>]*>(.*?)</entry>", re.S)


def _xml(pid: str) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="replace")


def _rows_of(xml: str, table_id: str) -> list[list[str]]:
    """Every `<row>` of one block, as plain cell text."""
    i = xml.find(table_id)
    if i < 0:
        return []
    block = xml[i:xml.find("</tables>", i)]
    out = []
    for r in _ROW.finditer(block):
        out.append([re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip()
                    for c in _ENTRY.findall(r.group(0))])
    return out


# ─────────────────────────────────────────────────────────────────────
# 1. A value belongs to the compound whose row it was printed in.
# ─────────────────────────────────────────────────────────────────────
def test_no_value_migrates_to_a_compound_that_never_had_it():
    """THE defect class that cannot be spotted downstream.

    US10125101 TABLE-US-00005 heads four columns —
    `Example in this invention | IC50 | Example in WO 2013/178575 | IC50` —
    its own compounds beside a prior-art patent's. The reader took the
    leftmost id and attached BOTH IC50 columns to it, so compound 2 shipped
    its own 20 μM and the 6 μM belonging to the WO compound beside it: a real
    number under a real compound, indistinguishable from a measurement.

    The assertion is not "compound 2 has one value" — that is the bug's
    signature, not its definition. It is that EVERY value we emit for a
    compound appears in a row that also states that compound's number.
    """
    from patentdb3.sources.uspto_assays import extract_from_patent

    xml = _xml("US10125101")
    recs = [r for r in extract_from_patent(xml)
            if r.table_id in ("TABLE-US-00005", "TABLE-US-00006")]
    assert recs, "fixture must produce records or the check is vacuous"

    rows = {tid: _rows_of(xml, tid)
            for tid in ("TABLE-US-00005", "TABLE-US-00006")}
    orphans = []
    for r in recs:
        if r.value_numeric is None:
            continue
        printed = f"{r.value_numeric:g}"
        home = [row for row in rows[r.table_id]
                if any(c.strip() == r.cid for c in row)]
        if not any(any(printed in c for c in row) for row in home):
            orphans.append((r.cid, printed, r.table_id))
    assert not orphans, (
        f"{len(orphans)} value(s) are filed under a compound whose own row "
        f"does not print them, e.g. {orphans[:3]}")


# ─────────────────────────────────────────────────────────────────────
# 2. A letter grade means what its own document says it means.
# ─────────────────────────────────────────────────────────────────────
def test_a_grade_carries_the_range_its_own_legend_defines():
    """A key is applied to thousands of rows at once, so a wrong one is silent.

    US11566007 publishes 8,652 records that are grades and nothing else — no
    numbers at all — and states its scale once, in a footer:

        +++++: IC50 ≥ 10 uM   ++++: 10 uM > IC50 ≥ 1 uM

    5,753 of them carried a letter and no range at all, and separately a
    thousands separator turned `≥ 10,000 nM` into `≥ 10 nM` elsewhere in the
    corpus. Both shipped plausible output.

    This reads the legend out of the document and checks the records against
    it, rather than against numbers typed into this file — the string a test
    author writes is not the string the extractor sees.
    """
    from patentdb3.sources.uspto_assays import extract_from_patent

    xml = _xml("US11566007")
    graded = [r for r in extract_from_patent(xml) if r.letter_grade]
    assert len(graded) > 8_000, "fixture must be grade-driven"

    unranged = [r for r in graded
                if r.range_lo is None and r.range_hi is None]
    assert not unranged, (
        f"{len(unranged)} graded record(s) carry no range, though the document "
        f"defines every symbol it uses")

    # The document's own words, read from the XML — not retyped here. Entities
    # must be resolved first: the patent writes `≥` as `&#x2265;`, so a regex
    # over the raw text finds nothing and the check would pass vacuously.
    from patentdb3.sources.cid_first import _unescape

    text = _unescape(re.sub(r"<[^>]+>", " ", xml))
    stated = re.search(r"\+{5}\s*:\s*IC\s*50\s*[≥>=]+\s*([\d.,]+)\s*([a-zµμ]+)",
                       text, re.I)
    assert stated, "the legend this check depends on is no longer in the file"
    lo = float(stated.group(1).replace(",", ""))

    five = {(r.range_lo, r.range_hi) for r in graded if r.letter_grade == "+++++"}
    assert five == {(lo, None)}, (
        f"`+++++` must be the open interval above {lo} the document states, "
        f"got {five}")

    # An interval is an interval: no bin may be inverted or a point.
    bad = [(r.letter_grade, r.range_lo, r.range_hi) for r in graded
           if r.range_lo is not None and r.range_hi is not None
           and r.range_lo >= r.range_hi]
    assert not bad, f"{len(bad)} bin(s) are not intervals, e.g. bad[:3]={bad[:3]}"


# ─────────────────────────────────────────────────────────────────────
# 3. A drawing marker points at the picture in that compound's own row.
# ─────────────────────────────────────────────────────────────────────
def test_a_drawn_marker_points_at_a_drawing_in_that_compounds_own_row():
    """`_drawing_refs` read `cells[0]` and nothing else.

    US9718790 lays its rows out `Structure | Compound No. | RT | [M+H]`, so
    the first cell is the picture and tag-strips to `""`. All 2,711 of its
    drawing rows were skipped — 2,930 drawings, not one reference emitted,
    while 2,248 of its compounds resolved to nothing. The artifact recorded
    them as "no structure found", which is what a compound we genuinely
    cannot place looks like too.

    A count alone would not catch a wrong pairing, so this verifies the
    EVIDENCE: for each marker, the compound number and the `<chemistry>` it
    was given must be in the same physical row.
    """
    from patentdb3.sources.cid_first import (
        _drawing_refs, _extract_assays, normalize_cid,
    )

    xml = _xml("US9718790")
    cids = {normalize_cid(r.cid) for r in _extract_assays(xml) if r.cid}
    refs = _drawing_refs(xml, cids)
    reached = {c: refs[c] for c in refs if c in cids}
    assert len(reached) > 2_000, (
        f"only {len(reached)} of this document's 2,930 drawings reach a "
        f"measured compound")

    rows = [r.group(0) for r in _ROW.finditer(xml)]
    wrong = []
    for cid, ref in list(reached.items())[:200]:
        row = next((r for r in rows if f'id="{ref}"' in r), None)
        if row is None:
            wrong.append((cid, ref, "ref not in any row"))
            continue
        text = " ".join(re.sub(r"<[^>]+>", " ", c) for c in _ENTRY.findall(row))
        if not any(normalize_cid(t) == cid for t in text.split()):
            wrong.append((cid, ref, "cid absent from that row"))
    assert not wrong, f"{len(wrong)} marker(s) point elsewhere, e.g. {wrong[:3]}"

    # No picture may be claimed by two compounds.
    dupes = [r for r, n in collections.Counter(reached.values()).items() if n > 1]
    assert not dupes, f"{len(dupes)} drawing(s) claimed by more than one compound"


# ─────────────────────────────────────────────────────────────────────
# 4. A positional pairing is believed only on the document's own say-so.
# ─────────────────────────────────────────────────────────────────────
def test_a_split_table_is_joined_only_on_the_documents_own_evidence():
    """US9303033 prints `TABLE 37A` (354 drawings, no numbers) then
    `TABLE 37B` (354 numbers, no drawings). Row position is the only
    correspondence such a table has, which makes it the easiest place to
    invent one: a wrong pairing hands 931 compounds the wrong picture and
    nothing downstream can tell.

    So the check is two-sided. The join must happen where the document states
    it — adjacent blocks, A/B halves of one table number, equal counts, every
    id a compound we measured — and must NOT happen anywhere else. A gate that
    fires on every patent is matching shape, not evidence.
    """
    from patentdb3.sources.cid_first import (
        _extract_assays, _split_table_refs, normalize_cid,
    )

    xml = _xml("US9303033")
    known = {normalize_cid(r.cid) for r in _extract_assays(xml) if r.cid}
    refs = _split_table_refs(xml, known)
    assert len(refs) > 900, f"the pairing this document states is lost ({len(refs)})"
    assert set(refs) <= known, "a pairing may only name compounds we measured"
    assert len(set(refs.values())) == len(refs), "one drawing, one compound"

    for pid in ("US9718790", "US10125101", "US11566007"):
        other = _xml(pid)
        cids = {normalize_cid(r.cid) for r in _extract_assays(other) if r.cid}
        assert not _split_table_refs(other, cids), (
            f"{pid} states no A/B split and must produce no positional pairing")


# ─────────────────────────────────────────────────────────────────────
# 5. A compound has one identity.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pid", ["US10125101", "US9987276", "US11566007"])
def test_one_compound_number_resolves_to_one_structure(pid):
    """Two structures under one number is a silent corruption: a consumer
    joining on `(patent, cid)` gets whichever row it happens to read first.

    Checked on three documents with different identity routes rather than one,
    because the failure would be introduced by whichever route is weakest and
    a single-patent check would miss it.
    """
    from patentdb3.sources.cid_first import extract_by_cid

    rows = [r for r in extract_by_cid(_xml(pid), pid) if r.cid]
    keys = collections.defaultdict(set)
    for r in rows:
        if r.inchikey:
            keys[r.cid].add(r.inchikey)
    clashes = {c: k for c, k in keys.items() if len(k) > 1}
    assert not clashes, (
        f"{len(clashes)} compound number(s) carry two structures, e.g. "
        f"{list(clashes.items())[:2]}")

    # A markush row denotes a SET of stereoisomers, so it must never assert a
    # single key. This is a design invariant, not a preference.
    assert not [r for r in rows if getattr(r, "markush", False) and r.inchikey], \
        "a markush row carries no InChIKey"


# ─────────────────────────────────────────────────────────────────────
# 6. What ships is what was extracted.
# ─────────────────────────────────────────────────────────────────────
def test_the_staged_row_is_faithful_to_the_artifact_it_came_from():
    """The staging step reshapes 119,000 measurements into one row per
    compound. Reshaping is where data goes missing quietly — a percent band
    was dropped for a whole release because `%` is not a concentration and the
    code kept only the converted copy.

    This re-derives the staged totals from the artifact and requires them to
    agree, and requires every assay object to carry the patent's own wording
    alongside whatever was lifted out of it.
    """
    import csv
    import json

    from patentdb3.stage_supabase import build

    out = config.PACKAGE_ROOT / "out"
    dump, manifest = out / "reader_dump.tsv", out / "latest.json"
    if not dump.exists() or not manifest.exists():
        pytest.skip("no dump artifact; run `verify --all --dump` first")

    # THE ARTIFACT MUST BE WHOLE. `verify --dump` streams this file, so a run
    # in progress leaves a valid-looking TSV with fewer rows than the manifest
    # claims, and comparing against it would report a staging bug that is
    # really a race. The manifest is written last, so it is the authority.
    declared = json.loads(manifest.read_text()).get("rows")
    on_disk = sum(1 for _ in dump.open()) - 1
    if declared is not None and on_disk != declared:
        pytest.skip(f"dump is mid-write ({on_disk:,} of {declared:,} rows)")

    rows = build()
    staged = [a for r in rows for a in r["assays"]]
    raw = list(csv.DictReader(dump.open(), delimiter="\t"))
    assert len(staged) == len(raw), (
        f"staging changed the measurement count: {len(raw):,} -> {len(staged):,}")

    # Every object keeps the document's own words, and every key it carries is
    # filled — a written-but-null key is the thing that made these unreadable.
    assert all(a.get("assay") for a in staged), "the patent's own heading is lost"
    assert not [a for a in staged if any(v is None for v in a.values())], \
        "an object may not carry a key it has nothing to put in"

    # A grade is only useful with its band; where the reader resolved one, the
    # staged row must still say so.
    graded = [a for a in staged if "grade" in a]
    banded = [a for a in graded if "band" in a]
    from_raw = sum(1 for r in raw if (r.get("letter_grade") or "").strip()
                   and ((r.get("range_lo") or "").strip()
                        or (r.get("range_hi") or "").strip()))
    assert len(banded) == from_raw, (
        f"{from_raw:,} measurements have a range in the artifact but "
        f"{len(banded):,} carry a band after staging")
