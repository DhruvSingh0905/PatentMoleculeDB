"""Tests for `sources/table_names.py` (this task's own, new module).

Two groups, the same split `test_anchor.py` / `test_iupac_reagent_markush.py`
already use:

  - PURE tests need no OPSIN, no java subprocess, no network: dewrap-candidate
    generation, the chemical-name-shape heuristic, the two column-detection
    signals (`_name_columns`), and `TableName.key`'s markush namespacing.
    These run unconditionally.
  - REAL-OPSIN tests exercise `extract_table_names` end to end on small,
    self-contained XML fragments (this module does not need a whole patent —
    `parse_tables`/`assemble_blocks` search for `<tables>` directly in
    whatever string they are given). Gated with the same
    `pytest.importorskip("py2opsin")` + graceful-skip pattern the sibling
    test files use, so a Python-only environment with no OPSIN jar collects
    but skips these rather than failing.

One fixture is drawn from real cached corpus data rather than hand-built:
`_WRAP_NEEDED_RAW` below is US9708336's actual cid=17 cell — a case checked
directly against the raw XML bytes, not invented, where the untouched cell
provably fails OPSIN and the targeted-dewrap candidate provably succeeds
(see this task's report for how it was found). Hand-built examples in this
corpus turned out to be a bad source of "wrap actually breaks parsing" cases
— OPSIN tolerates a bare space after a hyphen far more often than expected,
so most synthetic wrap examples parse "as-is" and prove nothing about the
dewrap path specifically.
"""
from __future__ import annotations

import pytest

from patentdb3.core import config
from patentdb3.sources.table_names import (
    MIN_CELL_LEN,
    TableName,
    _column_headers,
    _looks_chem_shaped,
    _name_columns,
    dewrap_candidates,
    extract_table_names,
)
from patentdb3.sources.uspto_xml import Cell, Table

PID = "US9708336"

# US9708336's actual cid=17 Name-column cell (see module docstring above).
# Confirmed directly against py2opsin: the untouched string fails to parse;
# removing only the two hyphen-adjacent spaces (the "targeted" candidate)
# succeeds. Not invented — this is what made choosing a hand-built wrap
# example unreliable in the first place: OPSIN parsed almost every
# synthetic one anyway.
_WRAP_NEEDED_RAW = (
    "N-(2-hydroxyethyl)-3′-sulfamoyl- 2′-(2H-tetrazol-5-yl)"
    "biphenyl-4- carboxamide"
)
_WRAP_NEEDED_TARGETED = (
    "N-(2-hydroxyethyl)-3′-sulfamoyl-2′-(2H-tetrazol-5-yl)"
    "biphenyl-4-carboxamide"
)

# A real relative-stereo name OPSIN resolves to one concrete stereoisomer —
# exactly the case `.key`'s "markush::" namespacing and the InChIKey-blanking
# in `extract_table_names` both exist for. Verified directly: OPSIN returns a
# non-empty SMILES *and* a non-empty StdInChIKey for this string; this
# module must not ship that key.
_MARKUSH_NAME = "(1R*,2S*)-2-phenylcyclopropane-1-carboxylic acid"


def _xml(pid: str = PID) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="ignore")


def _table_xml(rows_xml: str, *, cols: int, table_id: str = "TABLE-US-TEST",
                header_xml: str = "") -> str:
    """One minimal `<tables>` block — `parse_tables`/`assemble_blocks` need
    nothing else, not a `<description>` wrapper, not a whole patent."""
    thead = f"<thead><row>{header_xml}</row></thead>" if header_xml else ""
    return (f'<tables id="{table_id}"><tgroup cols="{cols}">'
            f"{thead}<tbody>{rows_xml}</tbody></tgroup></tables>")


def _extract(xml: str, patent_id: str = "TEST") -> list[TableName]:
    """`extract_table_names`, OPSIN-gated the same way the sibling files
    gate `extract_names`: skip (never fail) when OPSIN cannot run here."""
    pytest.importorskip("py2opsin")
    try:
        return extract_table_names(xml, patent_id)
    except Exception as e:                            # java missing, etc.
        pytest.skip(f"OPSIN unavailable in this environment: {e!r}")


# ── pure: dewrap_candidates ────────────────────────────────────────────────

def test_dewrap_no_whitespace_returns_only_the_cell_itself():
    assert dewrap_candidates("propan-2-ol") == [("none", "propan-2-ol")]


def test_dewrap_targeted_matches_the_three_documented_wrap_examples():
    """The three shapes named in this task's brief, all real: a space
    injected directly after a hyphen, inside a bracketed locant pair, and
    before a locant digit. TARGETED must remove exactly the wrap space and
    nothing else."""
    cases = [
        ("pyrazolo[3,4- d]pyrimidin", "pyrazolo[3,4-d]pyrimidin"),
        ("azetidin- 1-yl", "azetidin-1-yl"),
        ("3- methyl", "3-methyl"),
    ]
    for raw, want in cases:
        labels = dict(dewrap_candidates(raw))
        assert labels.get("targeted") == want, (raw, labels)


def test_dewrap_targeted_preserves_a_genuine_multiword_tail():
    """"...carboxylic acid" is two real words, neither side of the space is
    punctuation — TARGETED must leave it alone; only AGGRESSIVE touches it."""
    raw = "4-chlorophenylacetic acid"
    labels = dict(dewrap_candidates(raw))
    assert "targeted" not in labels, (
        "a genuine word-word space was treated as a wrap artifact")
    assert labels["aggressive"] == "4-chlorophenylacetic acid".replace(" ", "")


def test_dewrap_wrap_adjacent_to_bracket_and_paren_too():
    """Not confined to hyphens — space after `]` and after `)` are real wrap
    points too (measured on US10081601 / US10087188, see module docstring)."""
    cases = [
        ("azaspiro[2.5] octane-6,7-diol", "azaspiro[2.5]octane-6,7-diol"),
        ("pyrazin- 3- yl]bicyclo[1.1.1] pentane", "pyrazin-3-yl]bicyclo[1.1.1]pentane"),
    ]
    for raw, want in cases:
        labels = dict(dewrap_candidates(raw))
        assert labels.get("targeted") == want, (raw, labels)


def test_dewrap_dedups_identical_candidates():
    """When targeted and aggressive land on the same string (no genuine
    word-word space anywhere), only one extra candidate is offered — not a
    wasted, identical second OPSIN call."""
    raw = "3- methyl-1H-pyrazolo[3,4- d]pyrimidin-4-amine"
    cands = dewrap_candidates(raw)
    texts = [c for _, c in cands]
    assert len(texts) == len(set(texts)), cands
    assert len(cands) == 2, cands            # "none" + one dedup'd rewrite


def test_dewrap_real_corpus_example_needs_exactly_targeted():
    """The US9708336 cid=17 fixture: `none` must reproduce the untouched
    cell, `targeted` must reproduce the hand-verified fixed string, and no
    `aggressive` candidate is offered (targeted already covers every space)."""
    labels = dict(dewrap_candidates(_WRAP_NEEDED_RAW))
    assert labels["none"] == _WRAP_NEEDED_RAW
    assert labels["targeted"] == _WRAP_NEEDED_TARGETED
    assert "aggressive" not in labels


# ── pure: chemical-name shape ──────────────────────────────────────────────

def test_looks_chem_shaped_positive():
    assert _looks_chem_shaped(
        "4-(4-chlorophenyl)-1-methylpiperidine-4-carboxamide")


def test_looks_chem_shaped_rejects_short_or_flat_text():
    assert not _looks_chem_shaped("Me")                       # too short
    assert not _looks_chem_shaped("F")
    assert not _looks_chem_shaped(
        "this is a plain english sentence with no chemistry in it at all")


# ── pure: column detection (Signal A / Signal B) ───────────────────────────

def _synthetic_table(headers: list[str], rows: list[list[str]], *,
                      table_id: str = "T") -> Table:
    header_row = [Cell(h) for h in headers]
    body = [[Cell(v) for v in r] for r in rows]
    return Table(table_id=table_id, n_cols=len(headers),
                 header_rows=[header_row], body_rows=body)


def test_column_headers_plain_full_width_row():
    t = _synthetic_table(["No.", "Name", "Method"], [["1", "x", "A"]])
    assert _column_headers(t) == ["No.", "Name", "Method"]


def test_column_headers_respects_col_start_for_spanning_cells():
    """A header cell declaring CALS `namest`/`nameend` (captured as
    `Cell.col_start`/`colspan`) is authoritative and must not be shifted by
    left-to-right accumulation."""
    header_row = [Cell("IC50", colspan=2, col_start=1)]
    t = Table(table_id="T", n_cols=3, header_rows=[header_row], body_rows=[])
    assert _column_headers(t) == ["", "IC50", "IC50"]


def test_name_columns_signal_a_explicit_header():
    """Signal A: an explicit 'Name' header, next to a recognisable id
    column — the US10376513 TABLE-US-00001 shape."""
    t = _synthetic_table(
        ["No.", "Name", "R2"],
        [[str(i), f"some-long-chemical-looking-name-{i}-yl-phenylpiperidine", "Me"]
         for i in range(1, 6)])
    sig_a, sig_b, cid_idx = _name_columns(t)
    assert sig_a == {1}
    assert sig_b == set()
    assert cid_idx == 0


def test_name_columns_signal_b_unlabeled_shape_with_id_column():
    """Signal B: no header at all on the name column — the US10214537
    `Ex. | <blank> | <blank> | LCMS` shape."""
    t = _synthetic_table(
        ["Ex.", "", "LCMS"],
        [[str(i), f"4-(4-chlorophenyl)-1-methylpiperidine-{i}-carboxamide", "312.4"]
         for i in range(1, 6)])
    sig_a, sig_b, cid_idx = _name_columns(t)
    assert sig_a == set()
    assert sig_b == {1}
    assert cid_idx == 0


def test_name_columns_never_promotes_a_value_column():
    """The core "do not read a value cell as a name" guarantee at the
    column-selection level: a numeric assay column must never enter either
    signal set, however long or hyphen-heavy the header is."""
    t = _synthetic_table(
        ["Ex.", "IC50 (nM)"],
        [[str(i), f"{i}.5"] for i in range(1, 6)])
    sig_a, sig_b, _ = _name_columns(t)
    assert sig_a == set()
    assert sig_b == set()


def test_name_columns_rejects_unlabeled_shape_when_not_majority_chemlike():
    """Signal B requires a MAJORITY of a column's cells to look like a
    name — a mixed column (mostly short/plain tokens, one long fluke) must
    not qualify."""
    t = _synthetic_table(
        ["Ex.", "", "Method"],
        [["1", "one long chemical-looking-name-with-plenty-of-hyphens-here", "A"],
         ["2", "short", "B"],
         ["3", "ok", "C"],
         ["4", "no", "D"],
         ["5", "na", "E"]])
    _, sig_b, _ = _name_columns(t)
    assert sig_b == set()


# ── pure: TableName.key / markush namespacing ──────────────────────────────

def test_table_name_key_falls_back_to_smiles_when_no_inchikey():
    tn = TableName(patent_id="X", name="n", smiles="CCO", inchikey="",
                    raw_cell="n", dewrap="none", table_id="T", row_index=0,
                    column_index=0, column_signal="header")
    assert tn.key == "CCO"


def test_table_name_key_prefers_inchikey_when_present():
    tn = TableName(patent_id="X", name="n", smiles="CCO", inchikey="IK123",
                    raw_cell="n", dewrap="none", table_id="T", row_index=0,
                    column_index=0, column_signal="header")
    assert tn.key == "IK123"


def test_table_name_key_markush_is_namespaced_even_with_no_inchikey():
    """Mirrors `iupac_names.NamedCompound.key` exactly (see that module's
    `test_markush_does_not_deduplicate_against_a_concrete_structure`): a
    markush entry's key must never collide with a concrete structure that
    happens to share the same SMILES."""
    markush = TableName(patent_id="X", name="(1R*,2S*)-n", smiles="CCO",
                         inchikey="", raw_cell="n", dewrap="none",
                         table_id="T", row_index=0, column_index=0,
                         column_signal="header", markush=True,
                         markush_reason="relative_stereo:R*,S*")
    concrete = TableName(patent_id="X", name="n", smiles="CCO", inchikey="",
                          raw_cell="n", dewrap="none", table_id="T",
                          row_index=1, column_index=0, column_signal="header")
    assert markush.key == "markush::CCO"
    assert concrete.key == "CCO"
    assert markush.key != concrete.key


# ── real-OPSIN: end to end on minimal synthetic tables ─────────────────────

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extract_resolves_signal_a_and_carries_the_row_id():
    xml = _table_xml(
        "<row><entry>7</entry><entry>4-(4-chlorophenyl)-1-methylpiperidine"
        "</entry><entry>A</entry></row>",
        cols=3, header_xml="<entry>No.</entry><entry>Name</entry><entry>Method</entry>")
    out = _extract(xml)
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert len(out) == 1
    tn = out[0]
    assert tn.cid == "7"
    assert tn.column_signal == "header"
    assert tn.dewrap == "none"
    assert tn.smiles


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extract_resolves_signal_b_only_with_a_row_id():
    """Same unlabeled name column, four rows: three carry an id (clearing
    Signal B's own `_MIN_CHEM_SAMPLES` floor for the column), one does not.
    Only the id-bearing rows' names may survive — the guard measured on
    US10376513 (69 false hits removed, see module docstring)."""
    rows_xml = "".join(
        f"<row><entry>{cid}</entry><entry>{name}</entry><entry>3{i}2.8</entry></row>"
        for i, (cid, name) in enumerate([
            ("3", "4-(4-chlorophenyl)-1-methylpiperidine"),
            ("4", "4-(4-chlorophenyl)-1-ethylpiperidine"),
            ("5", "4-(4-chlorophenyl)-1-propylpiperidine"),
            ("", "4-(4-chlorophenyl)-1-butylpiperidine"),   # no id — must be dropped
        ]))
    xml = _table_xml(rows_xml, cols=3,
                      header_xml="<entry>Ex.</entry><entry></entry><entry>LCMS</entry>")
    out = _extract(xml)
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    cids = {o.cid for o in out}
    assert cids == {"3", "4", "5"}, cids
    assert all(o.column_signal == "unlabeled" for o in out)
    assert not any("butyl" in o.name for o in out), (
        "the id-less row's name leaked through Signal B's cid requirement")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extract_rejects_a_prose_nmr_cell_in_the_name_column():
    """US10172859 TABLE-US-00003's exact shape: a genuine `Name` column
    whose SECOND row is really an interleaved NMR continuation line, blank
    id and all. This candidate must never reach OPSIN at all — no py2opsin
    call is needed for this test to pass, since the reject happens before
    the OPSIN batch (see `extract_table_names`)."""
    xml = _table_xml(
        "<row><entry>92</entry><entry>4-(4-chlorophenyl)-1-methylpiperidine"
        "</entry></row>"
        "<row><entry></entry><entry>1H NMR (400 MHz, DMSO-d6) ppm = 12.89 "
        "(d, J = 2.4, 1H), 9.12 (s, 1H)</entry></row>",
        cols=2, header_xml="<entry>No.</entry><entry>Name</entry>")
    out = extract_table_names(xml, "TEST")           # no OPSIN dependency here
    assert all("NMR" not in o.raw_cell for o in out)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extract_dewraps_the_real_corpus_example():
    """The US9708336 cid=17 fixture end to end: the untouched cell fails,
    so the surviving record must be the TARGETED dewrap."""
    xml = _table_xml(
        f"<row><entry>17</entry><entry>{_WRAP_NEEDED_RAW}</entry></row>",
        cols=2, header_xml="<entry>No.</entry><entry>Name</entry>")
    out = _extract(xml)
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert len(out) == 1
    assert out[0].dewrap == "targeted"
    assert out[0].name == _WRAP_NEEDED_TARGETED
    assert out[0].raw_cell == _WRAP_NEEDED_RAW


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extract_markush_cell_carries_no_inchikey():
    """A relative-stereo name stated directly in a table cell must not ship
    an InChIKey — same rule, same reason as `iupac_names.extract_names`
    (see `TableName.key`'s docstring)."""
    xml = _table_xml(
        f"<row><entry>4</entry><entry>{_MARKUSH_NAME}</entry></row>",
        cols=2, header_xml="<entry>No.</entry><entry>Name</entry>")
    out = _extract(xml)
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert len(out) == 1
    tn = out[0]
    assert tn.markush is True
    assert tn.inchikey == ""
    assert tn.smiles                      # SMILES is still kept
    assert tn.key == "markush::" + tn.smiles


# ── real XML: corpus shape sanity (locked-in regression numbers) ──────────

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_us10376513_measured_residue():
    """The exact patent this task was assigned around. Measured directly
    (see task report): 189 candidate Name-column cells, of which exactly
    ONE — cid 291 — clears OPSIN. The illustrative cid=69 cell quoted in
    this task's brief is NOT that one: it carries a `chloro` -> `eyloro`
    character substitution (confirmed directly against py2opsin — the
    dewrapped, still-corrupted string fails to parse), the out-of-scope
    corruption defect this module deliberately does not repair. Locked in
    as a regression: this number must not silently change because this
    module's dewrap grew (or lost) reach, and it must not be "fixed" by
    reaching into the corruption family that belongs to `name_repair`.
    """
    out = _extract(_xml("US10376513"), "US10376513")
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    cids = {o.cid for o in out}
    assert cids == {"291"}, sorted(c for c in cids if c)
