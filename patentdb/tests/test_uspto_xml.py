"""Offline tests for the USPTO grant-XML source. No network required."""
import pytest

from patentdb.sources import uspto_xml as U

# A cut-down but structurally faithful sample: CALS tgroup, a multi-row header
# living in tbody (as real patents do), and a name split at a typeset break.
SAMPLE = """<us-patent-grant>
<description>
<p id="p-0001">Compounds were assayed as follows.</p>
<tables id="TABLE-US-00001" num="00001">
<table><tgroup cols="3">
<colspec colname="1"/><colspec colname="2"/><colspec colname="3"/>
<thead><row><entry>Cmp No.</entry><entry>Ki (&#x3bc;M)</entry><entry>n</entry></row></thead>
<tbody>
<row><entry>1</entry><entry>0.0038</entry><entry>(8)</entry></row>
<row><entry>3</entry><entry>~0.83</entry><entry>(8)</entry></row>
<row><entry>4</entry><entry>&#x3e;30</entry><entry>(2)</entry></row>
</tbody>
</tgroup></table>
</tables>
<tables id="TABLE-US-00002" num="00002">
<table><tgroup cols="1">
<colspec colname="1"/>
<tbody>
<row><entry>racemic cis-2-{1-(4-Bromobenzyl)-6-[(5-methylpyridin-2-yl)methoxy]-1H-benzimidazol-2-</entry></row>
<row><entry>yl}cyclohexanecarboxylic acid,</entry></row>
</tbody>
</tgroup></table>
</tables>
<p id="p-0002">End of description.</p>
</description>
</us-patent-grant>"""


def test_parses_cals_tables():
    tables = U.parse_tables(SAMPLE)
    assert len(tables) == 2
    t = tables[0]
    assert t.n_cols == 3
    assert t.table_id == "TABLE-US-00001"
    assert len(t.body_rows) == 3


def test_cell_values_and_qualifiers_survive():
    """Qualifiers are the whole point — `>30` must not become `30`."""
    rows = U.parse_tables(SAMPLE)[0].body_rows
    assert [c.text for c in rows[0]] == ["1", "0.0038", "(8)"]
    assert [c.text for c in rows[1]] == ["3", "~0.83", "(8)"]
    assert [c.text for c in rows[2]] == ["4", ">30", "(2)"]   # &#x3e; unescaped


def test_header_entities_unescape():
    assert "Ki (μM)" in U.parse_tables(SAMPLE)[0].header_text


def test_description_excludes_table_noise():
    text = U.description_text(SAMPLE)
    assert "Compounds were assayed" in text
    assert "End of description." in text
    # Table numerics must not leak into prose — that is what makes flat text
    # unusable for the Example-N slicing downstream.
    assert "0.0038" not in text


def test_detects_a_wrapped_cell():
    assert U.looks_wrapped("...1H-benzimidazol-2-")
    assert U.looks_wrapped("2-{1-(4-Bromobenzyl)")     # unclosed brace
    assert not U.looks_wrapped("cyclohexanecarboxylic acid")


def test_join_keeps_the_semantic_hyphen():
    """`benzimidazol-2-` + `yl}` must be `-2-yl}`, never `-2yl}`.

    The hyphen before a locant is part of the name; the typesetter breaks at
    one rather than inserting one. Dropping it yields a name OPSIN rejects.
    """
    joined = U.join_wrapped_cells([
        "...1H-benzimidazol-2-", "yl}cyclohexanecarboxylic acid,",
    ])
    assert joined == ["...1H-benzimidazol-2-yl}cyclohexanecarboxylic acid,"]


def test_join_reassembles_the_sample_name():
    cells = [c.text for r in U.parse_tables(SAMPLE)[1].body_rows for c in r]
    joined = U.join_wrapped_cells(cells)
    assert len(joined) == 1
    assert joined[0].endswith("cyclohexanecarboxylic acid,")
    assert "benzimidazol-2-yl}" in joined[0]


def test_join_candidates_offers_both_hyphen_readings():
    cands = U.join_candidates("benzimidazol-2-yl")
    assert "benzimidazol-2-yl" in cands
    assert len(cands) > 1        # the de-hyphenated alternative, for OPSIN


def test_publication_numbers_route_to_appxml_not_grant_xml():
    """Pre-grant publications are a different product, not an unsupported case.

    They are ~15% of BindingDB's patent-linked rows, so refusing them forfeits
    a large slice of the corpus. `_normalize_patent_number` must decline them
    (they are not grants) while `_normalize_publication_number` claims them.
    """
    assert U._normalize_patent_number("US20230365584A1") is None
    assert U._normalize_publication_number("US20230365584A1") == "20230365584"
    assert U._normalize_publication_number("US20240335431A1") == "20240335431"
    # A grant number must NOT be mistaken for a publication.
    assert U._normalize_publication_number("US8952177") is None


def test_unusable_identifiers_still_raise():
    with pytest.raises(U.UsptoUnavailable):
        U.fetch_grant_xml("EP1234567")


@pytest.mark.parametrize("raw,expected", [
    ("US8952177", "8952177"),
    ("8952177", "8952177"),
    ("US8952177B2", "8952177"),
    ("US10899738", "10899738"),
])
def test_patent_number_normalization(raw, expected):
    assert U._normalize_patent_number(raw) == expected


def test_racemic_prefix_is_stripped_by_the_cleaner():
    """One word blocked 16 of 35 names on US8952177's compound table."""
    from patentdb.core.iupac_to_smiles import rule_based_clean
    out = rule_based_clean("racemic cis-2-{1-(4-Bromobenzyl)}cyclohexane")
    assert not out.lower().startswith("racemic")
    assert out.startswith("cis-") or out.startswith("2-")


# ── block assembly ───────────────────────────────────────────────
# A `<tables>` block is fragmented into many tgroups and no single one of them
# is the table. These pin the three ways the old max-by-rows pick went wrong.

def _tg(cols, header, body):
    return U.Table(
        table_id="T1", n_cols=cols,
        header_rows=[[U.Cell(c) for c in r] for r in header],
        body_rows=[[U.Cell(c) for c in r] for r in body],
    )


def test_assembly_takes_the_header_from_a_sibling_fragment():
    """US9656988: the header tgroup holds no data, the data tgroup no header."""
    frags = [
        _tg(3, [], [["Example #", "BTK IC50", ""]]),
        _tg(3, [], [["1", "D"], ["2", "A"], ["3", "B"]]),
    ]
    b = U.assemble_block(frags, "T1")
    assert b.header_text.startswith("Example #")
    assert len(b.body_rows) == 3


def test_assembly_ignores_interleaved_annotation_fragments():
    """US10172859: the NMR/MS rows outnumber the assay rows, so row count is
    the wrong discriminator — the compound id is the right one."""
    frags = [
        _tg(6, [], [["202", "", "name", "C", "B", "B"]]),
        _tg(3, [], [["MS: 453.2 (M + H+); Rt 73.58 min", "see racemate", ""]]),
        _tg(6, [], [["203", "", "name", "D", "D", "A"]]),
        _tg(3, [], [["MS: 452.2 (M + H+); Rt 24.50 min", "see racemate", ""]]),
    ]
    b = U.assemble_block(frags, "T1")
    assert b.n_cols == 6
    assert [r[0].text for r in b.body_rows] == ["202", "203"]


def test_assembly_keeps_short_rows_that_are_real_compounds():
    """US8952177: a 4-col row is a compound with a trailing value omitted, not
    a different table. Selecting one width silently drops it."""
    frags = [
        _tg(5, [], [["1", "0.0038", "(8)", "0.4", "(3)"]]),
        _tg(4, [], [["12", "~2", "(2)", "nt"]]),
    ]
    b = U.assemble_block(frags, "T1")
    assert [r[0].text for r in b.body_rows] == ["1", "12"]


def test_promotion_stops_at_the_first_compound_id():
    """A graded row ["1", "D"] is name-like by shape; promoting it as a header
    continuation eats real data off the top of the table."""
    frags = [_tg(3, [["Compd", "Btk"]], [["ID", "(IC50)"], ["1", "D"], ["2", "A"]])]
    b = U.assemble_block(frags, "T1")
    assert "(IC50)" in b.header_text
    assert [r[0].text for r in b.body_rows] == ["1", "2"]


def test_empty_entries_hold_their_column_position():
    """US11254686's header is fully determined in the source. We destroyed it.

    CALS marks an unoccupied header cell with a self-closing `<entry/>`, which
    is how a 9-column table writes a label that sits over columns 6-8 only. The
    row is still nine entries wide, so nothing about the alignment is ambiguous.

    The parser's alternation tried `<entry\\b([^>]*)>(.*?)</entry>` FIRST, and
    `[^>]*` matches the `/` of a self-closing tag — so `<entry/>` matched the
    paired branch and `(.*?)</entry>` ran forward to the NEXT closing tag,
    swallowing the following cell. Empty cells did not merely vanish: they
    consumed their neighbour and shifted the rest of the row left.

    Nine entries came back as five, every `col_start` was -1, and the offset
    search then had to guess positions the patent had stated outright.
    """
    from patentdb.sources.uspto_xml import _parse_row

    row = ("<row><entry/><entry>Ave</entry><entry>Ave</entry><entry/>"
           "<entry/><entry/><entry>450</entry><entry>CLint</entry>"
           "<entry>CLint</entry></row>")
    cells = _parse_row(row)
    assert len(cells) == 9, [c.text for c in cells]
    assert [c.text for c in cells] == [
        "", "Ave", "Ave", "", "", "", "450", "CLint", "CLint"]

    # A spanning rule row still parses, and keeps its explicit position.
    span = _parse_row('<row><entry namest="1" nameend="9" align="center"/></row>')
    assert len(span) == 1 and span[0].colspan == 9 and span[0].col_start == 0


def test_parse_fidelity_reports_cells_lost_before_extraction_ran():
    """The check that would have ended a three-session hunt in one run.

    Every gap signal in this repo is computed from the PARSED view, so a defect
    in the parser is invisible to all of them — it presents as a hard layout and
    the repair loop buys rules to describe damage we inflicted ourselves.
    US11613531 lost 2,359 cells this way and was reported twice as "fires on
    336/687 held-out rows (49%), just under the 50% floor".

    This compares the two views directly: `<entry>` elements in, Cells out.
    No model, no reference data, no judgement.
    """
    from patentdb.sources.uspto_xml import parse_fidelity

    xml = ('<tables id="TABLE-US-00001" num="00001"><table><tgroup cols="3">'
           '<thead><row><entry/><entry>A</entry><entry>B</entry></row></thead>'
           '<tbody><row><entry>1</entry><entry>2</entry><entry>3</entry></row>'
           '</tbody></tgroup></table></tables>')
    assert parse_fidelity(xml) == [], "a faithfully parsed block reports nothing"

    # Simulate the defect: a reader that silently drops empty cells.
    import patentdb.sources.uspto_xml as U
    real = U._parse_row
    try:
        U._parse_row = lambda rx: [c for c in real(rx) if c.text.strip()]
        bad = parse_fidelity(xml)
        assert len(bad) == 1
        assert bad[0]["source_entries"] == 6 and bad[0]["parsed_cells"] == 5
        assert "lost before any extraction logic ran" in bad[0]["detail"]
    finally:
        U._parse_row = real


def test_a_reader_defect_reduces_to_one_repro_for_the_whole_corpus():
    """One bug must ask one question, not sixty.

    Grouping defects by the raw shape of the failing row fragmented a single
    regex bug into 60 "distinct" signatures — `eE`, `eE2`, `e2E`, `e2E3` — which
    would have been 60 paid questions for one line of code. Delta-debugging each
    row down to the smallest fragment that still breaks collapses them onto one
    motif: on the real defect, `<entry/>` followed by any paired entry.
    """
    import re

    from patentdb.repair.parser_repair import _reduce, _row_shape
    import patentdb.sources.uspto_xml as U

    row = ("<row><entry/><entry/><entry>LCMS m/z</entry><entry>HPLC</entry>"
           "<entry>HPLC</entry></row>")
    real = U._parse_row

    def defective(row_xml):
        """The original alternation: paired branch first, so `<entry/>` matches
        it and `(.*?)</entry>` swallows the next cell."""
        return [U.Cell(U._text(m.group(2) or ""))
                for m in re.finditer(
                    r"<entry\b([^>]*)>(.*?)</entry>|<entry\b([^>]*)/>",
                    row_xml, re.S)]

    try:
        U._parse_row = defective
        small = _reduce(row)
        assert _row_shape(small) == "eE", small
        assert len(small) < len(row)
    finally:
        U._parse_row = real

    # A healthy reader has nothing to reduce.
    assert _reduce(row) == row


def test_a_patch_can_be_reverted_from_its_journal_entry_alone(tmp_path, monkeypatch):
    """Authority to heal is only safe if every state is recoverable.

    The loop applies reader patches without asking, so the record IS the safety
    mechanism. Revert must work from the journal alone — no git, no clean tree,
    no requirement that the patch is still the newest thing in the file.
    """
    from patentdb.core import config
    from patentdb.repair import parser_repair as P

    monkeypatch.setattr(config, "PARSER_REPAIR_JOURNAL", tmp_path / "j.jsonl")
    module = tmp_path / "reader.py"
    module.write_text("def read():\n    return 'old'\n")

    entry_id = P.journal_append({
        "action": "patch", "module": str(module), "signature": "eE",
        "before_source": "    return 'old'", "after_source": "    return 'new'",
        "applied": True, "coverage_moved": {"US1": [6, 446]},
    })
    module.write_text(module.read_text().replace("    return 'old'",
                                                 "    return 'new'"))
    assert "new" in module.read_text()

    # Revert by the short numeric prefix — what a human actually types.
    assert P.revert("0001")["ok"], "revert must accept the id prefix"
    assert module.read_text() == "def read():\n    return 'old'\n"

    # The revert is itself journaled, so history is append-only and auditable.
    hist = P.journal_read()
    assert len(hist) == 2 and hist[1]["action"] == "revert"
    assert hist[1]["reverted"] == entry_id

    # And a declined patch is never refused forever.
    assert P.apply_journaled("0001")["ok"]
    assert "new" in module.read_text()


def test_revert_refuses_when_the_file_has_moved_on(tmp_path, monkeypatch):
    """Silently mangling a hand-edited file would be worse than refusing."""
    from patentdb.core import config
    from patentdb.repair import parser_repair as P

    monkeypatch.setattr(config, "PARSER_REPAIR_JOURNAL", tmp_path / "j.jsonl")
    module = tmp_path / "reader.py"
    module.write_text("def read():\n    return 'new'\n")
    P.journal_append({"action": "patch", "module": str(module), "signature": "x",
                      "before_source": "    return 'old'",
                      "after_source": "    return 'new'", "applied": True})
    module.write_text("def read():\n    return 'hand edited'\n")
    r = P.revert("0001")
    assert not r["ok"] and "edited" in r["why"]
