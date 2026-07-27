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
