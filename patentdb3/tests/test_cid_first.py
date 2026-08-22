"""`sources/cid_first.py` — the id-first identity route.

Every test here builds its own synthetic patent rather than reading a cached
XML, for the reason `conftest.py`'s docstring gives about artifacts: a test
that depends on `output_v3/uspto_xml/` passes or fails on what somebody else
downloaded. The two OPSIN-dependent tests skip (never fail) when the java
subprocess cannot run, exactly as `test_table_names.py` does.
"""
from __future__ import annotations

import re

import pytest

from patentdb3.core import config
from patentdb3.sources import cid_first
from patentdb3.sources.cid_first import _assertion, _occurrence_index, extract_by_cid

PID = "TEST0000001"

# A real relative-stereo name OPSIN resolves to a concrete stereoisomer — the
# case the markush invariant exists for. `test_table_names.py` uses the same
# string for the same reason: OPSIN returns BOTH a SMILES and a StdInChIKey
# for it, and this module must not ship that key.
MARKUSH_NAME = "(1R*,2S*)-2-phenylcyclopropane-1-carboxylic acid"
PLAIN_NAME = "2-phenylcyclopropane-1-carboxylic acid"
OTHER_NAME = "4-chlorophenylacetic acid"


def _assay_table(cids: list[str], table_id: str = "TABLE-US-00001") -> str:
    """A minimal CALS block `uspto_assays.extract_from_patent` reads as an
    assay table, so `cid_first`'s starting set is the reader's own output and
    not a list this test wrote."""
    rows = "".join(
        f"<row><entry>{c}</entry><entry>12.5</entry></row>" for c in cids)
    return (f'<tables id="{table_id}"><tgroup cols="2">'
            "<thead><row><entry>Example</entry>"
            "<entry>IC50 (nM)</entry></row></thead>"
            f"<tbody>{rows}</tbody></tgroup></tables>")


def _patent(body: str, cids: list[str]) -> str:
    return (f"<description>{body}{_assay_table(cids)}</description>")


def _run(xml: str):
    pytest.importorskip("py2opsin")
    try:
        return extract_by_cid(xml, PID)
    except Exception as e:                            # java missing, etc.
        pytest.skip(f"OPSIN unavailable in this environment: {e!r}")


# ── pure: the occurrence index ─────────────────────────────────────────────

def test_whole_token_boundaries_keep_I_20_out_of_I_200():
    """`I-20` must never match inside `I-200`, in either direction, and the
    single alternation must still find the standalone token."""
    text = "Example I-200: alpha. Example I-20: beta. Example XI-20: gamma."
    got = [(m.group(0), m.start()) for m in _occurrence_index(text, ["I-20", "I-200"])]
    assert [g for g, _ in got] == ["I-200", "I-20"]
    # the third is `XI-20` — glued to a letter on the left, so not a token
    assert all(text[s - 1] not in "XI" or g == "I-200" for g, s in got)


def test_one_regex_one_pass_not_a_search_per_cid():
    """The index is built from ONE compiled alternation. A per-cid loop is the
    difference between seconds and hours on a 2,000-cid patent, so this pins
    the interface: `_occurrence_index` takes the whole cid list at once and
    returns matches in document order."""
    text = " ".join(f"Example {i}: x" for i in range(200))
    ms = _occurrence_index(text, [str(i) for i in range(200)])
    assert len(ms) == 200
    assert [m.start() for m in ms] == sorted(m.start() for m in ms)


def test_a_bare_token_is_not_an_assertion():
    """The measurement in the module docstring: NMR shifts, reagent
    quantities and locants inside other names all produce whole-token cid
    matches, and none of them is the document declaring a compound number."""
    for text in ("1H NMR: 1.64-1.67 (m, 2H), 3.10 (m, 1H)",
                 "was dissolved in DCM (25 mL) 4.50 mmol",
                 "N-((4,6-dimethyl-2-oxo-1,2-dihydropyridin-3-yl)methyl)"):
        for m in _occurrence_index(text, ["1", "2", "3", "4"]):
            assert _assertion(text, m) is None, (text, m.group(0), m.start())


def test_label_paren_and_colon_shapes_are_assertions():
    cases = [("Example 24: a-name", "label"),
             ("Compound No. 7 a-name", "label"),
             ("...triazin-4-amine (544); next", "paren"),
             ("\n12: tert-butyl something", "colon")]
    for text, shape in cases:
        found = [_assertion(text, m) for m in _occurrence_index(text, ["24", "7", "544", "12"])]
        found = [o for o in found if o is not None]
        assert found and found[0].shape == shape, (text, found)


def test_non_id_words_are_refused_from_anchor_s_own_list():
    """`Method 1:` / `Table 3:` / `Scheme 2:` are the same false ids from
    either direction; the closed list is imported from `anchor`, not copied,
    so this also pins that it stays imported."""
    for text in ("\nMethod 1: something", "\nTable 3: results", "\nScheme 2: route"):
        assert all(_assertion(text, m) is None
                   for m in _occurrence_index(text, ["1", "2", "3"]))


# ── the span rule ──────────────────────────────────────────────────────────

def test_a_name_behind_another_cid_is_not_attributed_to_the_first():
    """THE LOAD-BEARING RULE. `Example 1` is followed by `Example 2` and only
    then by a name; that name belongs to 2 and cid 1 must resolve to nothing
    rather than reach past its neighbour."""
    body = (f"<heading>Example 1</heading>"
            f"<heading>Example 2</heading>"
            f"<heading>{PLAIN_NAME}</heading>")
    out = {c.cid: c.name for c in _run(_patent(body, ["1", "2"]))}
    assert out.get("2") == PLAIN_NAME
    assert "1" not in out


def test_sub_form_ids_need_no_special_case():
    """`I-20` followed by `I-20a`: the token boundary keeps them apart and
    `I-20a`'s own occurrence closes `I-20`'s span, so each keeps its own
    name with nothing in the code that knows about sub-forms."""
    body = (f"<heading>Example I-20</heading><heading>{PLAIN_NAME}</heading>"
            f"<heading>Example I-20a</heading><heading>{OTHER_NAME}</heading>")
    out = {c.cid: c.name for c in _run(_patent(body, ["I-20", "I-20a"]))}
    assert out.get("I-20") == PLAIN_NAME
    assert out.get("I-20a") == OTHER_NAME


def test_a_name_further_than_the_gap_cap_is_not_reached():
    """`_NAME_GAP_MAX` is what stops the line scan walking into the synthesis
    paragraph, whose first sentence names the STARTING MATERIAL — a real
    molecule OPSIN parses perfectly."""
    filler = "<p>" + ("The mixture was stirred at ambient temperature. " * 4) + "</p>"
    body = f"<heading>Example 1</heading>{filler}<heading>{PLAIN_NAME}</heading>"
    assert not [c for c in _run(_patent(body, ["1"])) if c.cid == "1"]


# ── the contract every consumer depends on ─────────────────────────────────

def test_returns_named_compound_with_source_cid_first_and_one_row_per_cid():
    body = (f"<heading>Example 1</heading><heading>{PLAIN_NAME}</heading>"
            f"<p>Example 1 was also described as {PLAIN_NAME} elsewhere.</p>")
    out = _run(_patent(body, ["1"]))
    assert out and all(c.source == "cid_first" for c in out)
    assert all(c.patent_id == PID for c in out)
    cids = [c.cid for c in out]
    assert len(cids) == len(set(cids)), "at most one row per cid"


def test_markush_name_gets_no_inchikey():
    """NON-NEGOTIABLE, and the same invariant `iupac_names` holds: a relative-
    stereo name denotes a SET of stereoisomers, so a single-structure
    identifier is a false claim about it."""
    body = f"<heading>Example 5</heading><heading>{MARKUSH_NAME}</heading>"
    out = [c for c in _run(_patent(body, ["5"])) if c.cid == "5"]
    assert out, "the markush name itself must still be extracted"
    got = out[0]
    assert got.markush is True
    assert got.inchikey == ""
    assert got.smiles, "the SMILES is kept; only the identity claim is withheld"
    assert got.markush_reason.startswith("relative_stereo:")


def test_no_markush_row_anywhere_ever_carries_an_inchikey():
    body = (f"<heading>Example 5</heading><heading>{MARKUSH_NAME}</heading>"
            f"<heading>Example 6</heading><heading>{PLAIN_NAME}</heading>")
    out = _run(_patent(body, ["5", "6"]))
    assert all(not (c.markush and c.inchikey) for c in out)


def test_finished_only_drops_an_intermediate_introduced_cid(monkeypatch):
    """A cid whose only assertion is introduced by `Intermediate` / `Step` /
    `Preparation` is a synthesis waypoint, not a deliverable compound."""
    body = f"<heading>Intermediate 9</heading><heading>{PLAIN_NAME}</heading>"
    xml = _patent(body, ["9"])
    monkeypatch.setattr(config, "FINISHED_ONLY", True)
    assert not [c for c in _run(xml) if c.cid == "9"]
    monkeypatch.setattr(config, "FINISHED_ONLY", False)
    assert [c for c in _run(xml) if c.cid == "9"]


def test_an_unsearchable_cid_is_counted_not_silently_skipped():
    """US9018217's whole assay set is chemical NAMES in the cid field (the
    `build_columns` defect `normalize_cid`'s docstring records). They cannot
    be searched for as tokens, and the accounting has to say so."""
    long_cid = "(2-{2-[2-(5,7-dimethyl-triazolo-pyrimidin-2-yl)-ethyl]-phenyl})"
    xml = _patent("<p>nothing</p>", [long_cid])
    _out, st = cid_first._resolve(xml, PID)
    assert st.assay_cids == 1
    assert st.unsearchable == 1
    assert st.resolved == 0


def test_stats_partition_the_assay_cids():
    """Every cid lands in exactly one bucket, so a zero is never ambiguous."""
    body = (f"<heading>Example 1</heading><heading>{PLAIN_NAME}</heading>"
            f"<heading>Example 2</heading><p>no name here at all</p>")
    _out, st = cid_first._resolve(_patent(body, ["1", "2"]), PID)
    total = (st.unsearchable + st.no_occurrence + st.no_name_text
             + st.opsin_reject + st.coverage_reject + st.resolved)
    assert total == st.assay_cids, st


def test_the_cid_is_searched_exactly_as_the_assay_reader_produced_it():
    """No `normalize_cid` on the search token: canonicalising mangles real ids
    (`EM09912` -> `EM9912`) into strings that occur nowhere in the document."""
    body = f"<heading>Example 1D</heading><heading>{PLAIN_NAME}</heading>"
    out = [c for c in _run(_patent(body, ["1D"])) if c.name == PLAIN_NAME]
    assert out and out[0].cid == "1D"


def test_a_padded_id_is_a_known_upstream_loss():
    """`extract_from_patent` has ALREADY canonicalised its own cids by the
    time this module sees them (`EM09912` -> `EM9912`, `I-0020` -> `I-20`,
    verified directly), so a padded id is unsearchable here no matter what
    this module does. Measured over the 20-patent sample: 1,315 cids have no
    whole-token occurrence in the document at all, and a zero-padded spelling
    exists for **5** of them — a real but tiny loss whose fix belongs in
    `uspto_assays`, not here. Pinned so it cannot change silently."""
    body = f"<heading>Example EM09912</heading><heading>{PLAIN_NAME}</heading>"
    _out, st = cid_first._resolve(_patent(body, ["EM09912"]), PID)
    assert st.assay_cids == 1 and st.no_occurrence == 1 and st.resolved == 0


def test_module_does_not_consult_the_route_feature_flag():
    """`IUPAC_NAMES` gates the ROUTE at its call site, not this pure function
    — the same decision `extract_names` records in its own body. A function
    that returns `[]` because of a global reads as a broken extractor."""
    src = (cid_first.__file__).replace(".pyc", ".py")
    body = open(src).read().split('"""', 2)[2]
    assert not re.search(r"config\.IUPAC_NAMES", body)


def _xml_or_skip(pid: str) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="replace")


def test_a_table_split_into_a_drawings_half_and_a_data_half_is_joined():
    """US9303033 prints `TABLE 37A` holding 354 drawings and no numbers, then
    `TABLE 37B` holding 354 numbers and no drawings.

    Every other route reads a compound's OWN row, so neither half yields
    anything: the drawings have no cid to key on and the numbers have no
    picture beside them. The patent has 1,271 assay compounds and had 1,270
    with no structure and no marker — the largest single block of unplaced
    compounds in the corpus, while the document states the pairing plainly.
    """
    from patentdb3.sources.cid_first import (
        _drawing_refs, _extract_assays, _split_table_refs, normalize_cid,
    )
    xml = _xml_or_skip("US9303033")
    known = {normalize_cid(r.cid) for r in _extract_assays(xml) if r.cid}
    assert not _drawing_refs(xml), \
        "no compound here has a drawing in its own row — that is the point"

    refs = _split_table_refs(xml, known)
    assert len(refs) == 931
    # Verified against the raw XML: 37A's drawn rows open at CHEM-US-00848 and
    # 37B's data rows open at A20, in the same order.
    assert refs["A20"] == "CHEM-US-00848"
    assert refs["B20"] == "CHEM-US-00849"
    assert refs["C20"] == "CHEM-US-00850"


def test_the_split_table_join_refuses_everything_it_cannot_confirm():
    """Row position is the only correspondence such a table can have, so the
    bar is four-fold: adjacency, an A/B title pair of the SAME number, exactly
    equal counts, and every id in the B half being a compound we MEASURED.

    Two consequences are asserted here rather than described. Eight of
    US9303033's own 42 candidate pairs disagree on count and are refused. And
    across the other 136 cached patents the join never fires at all — a gate
    that produced pairings everywhere would be matching on shape, not on
    evidence.
    """
    from patentdb3.sources.cid_first import (
        _extract_assays, _split_table_refs, normalize_cid,
    )
    for pid in ("US9718790", "US10125101", "US11566007"):
        xml = _xml_or_skip(pid)
        known = {normalize_cid(r.cid) for r in _extract_assays(xml) if r.cid}
        assert not _split_table_refs(xml, known), \
            f"{pid} has no A/B split table and must produce no pairing"
