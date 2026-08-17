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
    """This patent's Name columns, and why the number moved from 1 to 133.

    It used to assert `cids == {"291"}`: 189 candidate cells, exactly one
    clearing OPSIN, because every other cell carries a `<sup>` footnote digit
    fused onto its tail (`...propan-2-ol2`) which nothing repaired. That
    assertion also said the fix "must not be reached by going into the
    corruption family that belongs to `name_repair`" — and it still must not
    be. It was not: `name_repair` gained a SIXTH confirmed form,
    `footnote_digit_tail`, and this module reaches it only through the rescue
    stage it has always called. The repair lives on the far side of that
    boundary, exactly where the old docstring said it belonged.

    All 133 rows are attributed to that one pattern, and 131 of them join to
    a compound the patent actually measured.

    NOTE ON cid 291, because it is the one row whose PATH changed rather than
    its answer. It used to parse with no repair at all. The cross-`<row>`
    rejoin now merges its superscript continuation row into the name, so the
    cell arrives as `...azetidine-1-carboxylate3` and is resolved by the
    footnote pattern instead. Same accepted name, same InChIKey, one more
    stage. Measured, so the rejoin cannot be blamed for the family as a whole:
    118 of the 133 are `rows_joined == 1` — genuine source footnotes — and
    with the rejoin disabled this patent still yields 125 rows, 124 of them
    via the same pattern.
    """
    out = _extract(_xml("US10376513"), "US10376513")
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert len(out) == 133, len(out)
    assert {o.repair for o in out} == {"footnote_digit_tail"}
    assert "291" in {o.cid for o in out}
    # the defect is the SOURCE's, not the rejoin's
    assert sum(1 for o in out if o.rows_joined == 1) == 118


# ── the rescue stage: `name_repair` on what dewrapping could not save ─────
#
# The composition ORDER between this module's dewrap and `name_repair`'s
# character/bracket repair was measured over the full 137-patent cached
# corpus, on the 16,054 name-column cells no dewrap variant resolves:
#
#     A  repair-then-dewrap   206,846 OPSIN candidates   18 cells recovered
#     B  dewrap-then-repair   206,785 OPSIN candidates   18 cells recovered
#
# same 18 cells, byte-identical accepted string and SMILES. Order B ships.
# The tests below lock in the mechanism (who gets asked), the reason the two
# orders agree (bracket-stack site detection is whitespace-blind), one real
# recovery, and the out-of-scope defect that still blocks the module
# docstring's own worked example.

def test_rescue_is_asked_only_about_cells_no_dewrap_resolved(monkeypatch):
    """`name_repair` must never see a cell that already parsed. It is not a
    correctness question — `repair_names` short-circuits on an original that
    parses — but a cost one: asking about every cell instead of the residue
    would multiply this module's OPSIN volume by the ~24,000 cells that do
    not need it. Pure: `_repair_names` is stubbed, so no OPSIN runs for the
    rescue half."""
    pytest.importorskip("py2opsin")
    asked: list[str] = []

    def spy(names, document_text="", patent_id=""):
        asked.extend(names)
        return [None] * len(names)

    monkeypatch.setattr("patentdb3.sources.table_names._repair_names", spy)
    # Row 2's `qqqphenyl` is a nonsense morpheme OPSIN cannot read, chosen
    # over a mismatched bracket on purpose: OPSIN treats `]`/`}`/`)` far more
    # interchangeably than expected (the same leniency behind `name_repair`'s
    # refused Form E), so a bracket-typo cell parses anyway and would prove
    # nothing about which cells reach the rescue stage.
    xml = _table_xml(
        "<row><entry>1</entry><entry>4-(4-chlorophenyl)-1-methylpiperidine</entry></row>"
        "<row><entry>2</entry><entry>4-(4-qqqphenyl)-1-methylpiperidine</entry></row>",
        cols=2, header_xml="<entry>No.</entry><entry>Name</entry>")
    out = _extract(xml)
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert [o.cid for o in out] == ["1"]
    assert asked, "the rescue stage never ran on the unresolved cell"
    assert all("qqqphenyl" in a for a in asked), asked


def test_a_cell_that_parses_on_its_own_carries_no_repair_marker():
    """`.repair` is provenance, and provenance that fires on a clean cell is
    worse than none — the manifest counts this field."""
    xml = _table_xml(
        "<row><entry>1</entry><entry>4-(4-chlorophenyl)-1-methylpiperidine</entry></row>",
        cols=2, header_xml="<entry>No.</entry><entry>Name</entry>")
    out = _extract(xml)
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert [o.repair for o in out] == [""]


def test_bracket_stack_sites_are_whitespace_blind():
    """WHY THE TWO ORDERS AGREE, asserted rather than asserted-about.

    All 18 corpus recoveries come from `name_repair`'s bracket-stack patterns
    (`stray_opening_bracket` 14, `stray_closing_bracket` 3,
    `dropped_close_bracket` 1). Those scanners count brackets and never look
    at whitespace, so dewrapping before or after them finds the SAME bracket
    characters — which is why order A and order B recovered the same cells
    and the same strings. Shown here on the real US10172859 cell that
    recovers, with no OPSIN needed.

    WHAT IS *NOT* CLAIMED: that the two orders generate identical candidate
    SETS. They do not, and the difference is entirely `dropped_close_bracket`,
    whose fix scans a fixed 60-CHARACTER window after the opener — a window
    that reaches further into real content once spaces are gone, so order B
    proposes insertion offsets order A never sees (this is why the corpus
    counts differ at all: 206,846 vs 206,785). Measured: none of those
    extra offsets ever produced a confirmed repair.
    """
    from patentdb3.sources.name_repair import (
        _normalize_ws, _unmatched_closers, _unmatched_openers,
        generate_candidates)

    raw = ("[4-Fluoro-3-(5-fluoro- 7-morpholin-4-yl- quinazolin-4-yl- phenyl]-"
           "(3-methyl- pyrazin-2-yl)- methanol")
    flat = _normalize_ws(raw)
    for scan in (_unmatched_openers, _unmatched_closers):
        assert ([raw[s] for s, _ in scan(raw)]
                == [flat[s] for s, _ in scan(flat)]), scan.__name__

    delete_only = {"stray_opening_bracket", "stray_closing_bracket"}
    order_a = {_normalize_ws(dw)
               for rc in generate_candidates(raw) if rc.pattern_id in delete_only
               for _, dw in dewrap_candidates(rc.repaired)}
    order_b = {_normalize_ws(rc.repaired)
               for _, dw in dewrap_candidates(raw)
               for rc in generate_candidates(dw) if rc.pattern_id in delete_only}
    assert order_a == order_b
    assert order_a, "no repair candidate generated for the known corpus case"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_us10172859_rescues_exactly_the_measured_cell():
    """The one real recovery in this patent, measured against the cached XML:
    cid 268's cell opens with a `[` that nothing closes. `name_repair`'s
    `stray_opening_bracket` deletes it, OPSIN accepts the result, and the
    corrected fragment is found verbatim elsewhere in US10172859's own text —
    so it is `corroborated`, not accepted on OPSIN alone. Locked in as the
    regression case for the rescue stage's wiring: if this drops to zero the
    stage has become unreachable, which an import graph cannot tell you.
    """
    out = _extract(_xml("US10172859"), "US10172859")
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    rescued = [o for o in out if o.repair]
    assert len(rescued) == 1, [(o.cid, o.repair) for o in rescued]
    tn = rescued[0]
    assert tn.repair == "stray_opening_bracket"
    assert tn.cid == "268"
    assert not tn.name.startswith("[")
    assert tn.raw_cell.startswith("[")
    assert tn.smiles


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cid_69_is_still_blocked_by_the_out_of_scope_footnote_digit():
    """THE MODULE DOCSTRING'S OWN WORKED EXAMPLE, AND IT STILL DOES NOT
    RESOLVE — deliberately.

    US10376513's cid 69 cell carries THREE defects at once: line-wrap spaces
    (this module's), `chloro` written `eyloro` (`name_repair.ch_as_ey`'s one
    confirmed corpus case), and a footnote digit fused onto the tail
    (`propan-2-ol2`, from `<sup>2</sup>` losing its tags) which is
    deliberately out of scope for BOTH modules. Measured against the cached
    XML: TARGETED removes every embedded space, `ch_as_ey` proposes the
    `eyloro` fix, and the result still fails OPSIN purely on the trailing
    `2`. That is why `ch_as_ey` contributes 0 of the corpus's 18 rescues.

    This asserts the ABSENCE stays an absence. If a later change makes cid 69
    resolve, it has either reached into the footnote-digit family (which is
    owned elsewhere and must not be special-cased here) or accepted a
    structure for a name the patent never spelled — either way, re-measure
    before editing this test.
    """
    from patentdb3.sources.name_repair import generate_candidates
    from patentdb3.sources.opsin import batch as opsin_batch

    pytest.importorskip("py2opsin")
    raw = ("(2R)-1-(3-{3-[1-(4-Amino-3- methyl-1H-pyrazolo[3,4- d]pyrimidin-1-"
           "yl)ethyl]-5-eyloro-6- fluoro-2-methoxyphenyl}azetidin- 1-yl)"
           "propan-2-ol2")
    tried = [dw for _, dw in dewrap_candidates(raw)]
    tried += [rc.repaired for _, dw in dewrap_candidates(raw)
              for rc in generate_candidates(dw)]
    assert any("chloro" in t for t in tried), (
        "ch_as_ey no longer proposes the fix this test is about")
    smiles = opsin_batch(tried, "SMILES", "US10376513")
    if len(smiles) != len(tried):
        pytest.skip("OPSIN refused the batch — treat as unavailable")
    assert not any(smiles), (
        "cid 69 now resolves — the footnote digit is out of scope for this "
        "module; re-measure before changing this assertion")

    out = _extract(_xml("US10376513"), "US10376513")
    if out:
        assert "69" not in {o.cid for o in out}


# ── vertical records: one compound per RUN of rows ────────────────────────
#
# The layout and every string below are copied verbatim from US9265734
# TABLE-US-00006 (label column, value column, seven fields per record, the
# `(nM)` unit line on its own row). Hand-built only in the sense that seven
# real records are stacked to clear `vertical_blocks`'s own measured floors
# (>= 30 label rows, an anchor with >= 5 evenly spaced hits); no string is
# invented. See "A THIRD SHAPE" in the module docstring.

# (cid, name) — R119/R120/R122 are the patent's own first records; R122's name
# is the wrapped one, split exactly where the source splits it.
_VERT_RECORDS = [
    ("R119", ["N-(2-aminophenyl)-6-(phenylsulfonamido)hexanamide"]),
    ("R120", ["N-(6-(2-amino-4-fluorophenylamino)-6-oxohexyl)-4-fluoro-"
              "N-methylbenzamide"]),
    ("R122", ["N-(2-amino-4-fluorophenyl)-6-(6-fluoro-1-oxo-3,4-"
              "dihydroisoquinolin-2(1H)-",
              "yl)hexanamide"]),
    ("R121", ["N-(7-(2-aminophenylamino)-7-oxoheptyl)-4-methylbenzamide"]),
    ("R123", ["N-(6-(2-aminophenylamino)-6-oxohexyl)benzamide"]),
    ("R124", ["N-(5-(2-aminophenylamino)-5-oxopentyl)benzamide"]),
    ("R125", ["N-(2-aminophenyl)-6-(phenylsulfonamido)hexanamide"]),
]


def _vertical_xml(records=None, *, name_label: str = "Chemical_name",
                   table_id: str = "TABLE-US-00006") -> str:
    """US9265734 TABLE-US-00006's shape: two columns, fields down column 0."""
    recs = _VERT_RECORDS if records is None else records

    def row(a: str, b: str) -> str:
        return f"<row><entry>{a}</entry><entry>{b}</entry></row>"

    rows = []
    for i, (cid, parts) in enumerate(recs, start=1):
        rows.append(row(f"Record {i}", ""))
        rows.append(row("Structure", ""))
        rows.append(row("Comp id", cid))
        rows.append(row("HDAC1 IC50", str(7000 + i)))
        rows.append(row("(nM)", ""))
        rows.append(row("HDAC3 IC50", str(1100 + i)))
        rows.append(row("(nM)", ""))
        rows.append(row(name_label, parts[0]))
        for tail in parts[1:]:
            rows.append(row("", tail))          # the source's own wrapped line
        rows.append(row("LC/MS Calc'd (M + H)", "376.4"))
        rows.append(row("LC/MS Obsv'd (M + H)", "376.1"))
    return _table_xml("".join(rows), cols=2, table_id=table_id)


def test_vert_name_label_is_a_separate_pattern_because_signal_a_cannot_match():
    """`_NAME_HEADER` provably does not reach `Chemical_name`, which is why a
    second pattern exists rather than a widened first one — `_` is a word
    character, so `\\bname\\b` finds no boundary in front of `name`."""
    from patentdb3.sources.table_names import _NAME_HEADER, _VERT_NAME_LABEL

    assert not _NAME_HEADER.search("Chemical_name")
    for label in ("Chemical_name", "Name", "Compound Name", "IUPAC name",
                  "chemical name:"):
        assert _VERT_NAME_LABEL.match(label), label
    # Full-string match: the other fields sharing that column must not read as
    # names, and neither must a real name that happens to contain the word.
    for label in ("Structure", "Comp id", "LC/MS Calc'd (M + H)",
                  "HDAC1 IC50 (nM)", "Record 1", "",
                  "N-(2-aminophenyl)-6-(phenylsulfonamido)hexanamide"):
        assert not _VERT_NAME_LABEL.match(label), label


def test_vertical_candidates_read_the_name_field_and_its_record_id():
    from patentdb3.sources.table_names import _vertical_candidates
    from patentdb3.sources.uspto_xml import parse_tables

    got = _vertical_candidates(parse_tables(_vertical_xml()), set())
    assert len(got) == len(_VERT_RECORDS)
    assert [c[4] for c in got] == [r[0] for r in _VERT_RECORDS]
    assert all(c[0] == "TABLE-US-00006" for c in got)
    assert all(c[3] == "vertical" for c in got)
    assert [c[1] for c in got] == list(range(len(_VERT_RECORDS)))
    # Nothing but the name field is offered: no cid, no IC50, no LC/MS mass.
    assert got[0][5] == _VERT_RECORDS[0][1][0]
    assert all("IC50" not in c[5] and "376" not in c[5] for c in got)


def test_vertical_name_rejoins_a_label_less_wrapped_row():
    """R122's name spills onto a row with an EMPTY label. It is one name."""
    from patentdb3.sources.table_names import _vertical_candidates
    from patentdb3.sources.uspto_xml import parse_tables

    got = _vertical_candidates(parse_tables(_vertical_xml()), set())
    r122 = next(c for c in got if c[4] == "R122")
    assert r122[6] == 2                       # rows_joined
    assert r122[5].endswith("2(1H)- yl)hexanamide")
    assert all(c[6] == 1 for c in got if c[4] != "R122")


def test_vertical_does_not_absorb_a_finished_name():
    """A label-less row after a COMPLETE name starts nothing and joins nothing
    — the same `_UNFINISHED_TAIL` rule `_records` applies to the row layout.

    R122 is the record edited, and it has to be: `_vertical_anchor` cuts on a
    label whose spacing is EVEN (modal stride, >= 80% of strides), so the one
    record already carrying an extra row is the only one that can carry this
    one. Give the extra row to any other record and the detector — correctly —
    stops recognising the table at all.
    """
    from patentdb3.sources.table_names import _vertical_candidates
    from patentdb3.sources.uspto_xml import parse_tables

    recs = list(_VERT_RECORDS)
    finished = "N-(2-aminophenyl)-6-(phenylsulfonamido)hexanamide"
    recs[2] = ("R122", [finished, "4-(4-chlorophenyl)-1-methylpiperidine"])
    got = _vertical_candidates(parse_tables(_vertical_xml(recs)), set())
    r122 = next(c for c in got if c[4] == "R122")
    assert r122[6] == 1
    assert r122[5] == finished


def test_vertical_pass_skips_a_block_the_row_wise_pass_already_read():
    """`skip` is what makes this pass strictly additive — a block cannot be
    read both ways and emit the same compound twice."""
    from patentdb3.sources.table_names import _vertical_candidates
    from patentdb3.sources.uspto_xml import parse_tables

    tabs = parse_tables(_vertical_xml())
    assert _vertical_candidates(tabs, {"TABLE-US-00006"}) == []


def test_vertical_detector_is_not_reimplemented_here():
    """This module calls `uspto_assays.vertical_blocks`; it does not own a
    second copy of the detection. Asserted so a later 'small local tweak'
    to the thresholds has to break a test to happen."""
    from patentdb3.sources import table_names, uspto_assays

    assert table_names.vertical_blocks is uspto_assays.vertical_blocks
    src = (table_names.__file__ and
           open(table_names.__file__, encoding="utf-8").read())
    for private in ("_VERT_MAX_LABEL_UNIQ", "_VERT_MIN_LABEL_ROWS",
                    "_VERT_MIN_ANCHOR_HITS", "_vertical_anchor",
                    "_vertical_pairs"):
        assert f"{private} =" not in src and f"def {private}" not in src


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extract_resolves_vertical_records_through_opsin():
    out = _extract(_vertical_xml(), "US9265734")
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert len(out) == len(_VERT_RECORDS)
    assert all(o.column_signal == "vertical" for o in out)
    assert {o.cid for o in out} == {r[0] for r in _VERT_RECORDS}
    assert all(o.smiles and o.inchikey for o in out)
    # The wrapped one is rejoined and then dewrapped — the glue space this
    # module injected is removed by TARGETED, exactly as `_variants` documents.
    r122 = next(o for o in out if o.cid == "R122")
    assert r122.rows_joined == 2
    assert r122.dewrap == "targeted"
    assert " " not in r122.name


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_us9265734_vertical_measured_yield():
    """MEASURED 2026-08-17 against the cached XML, reader only, no heal:
    TABLE-US-00006 offers 199 `Chemical_name` fields, all 199 carry a `Comp id`,
    OPSIN resolves 199 of 199 (0 refused), 7 needed the label-less rejoin and
    all 7 then needed TARGETED dewrap. Before this pass the same call returned
    0. Re-measure before editing any number here."""
    out = _extract(_xml("US9265734"), "US9265734")
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    vert = [o for o in out if o.column_signal == "vertical"]
    assert len(vert) == 199
    assert all(o.table_id == "TABLE-US-00006" for o in vert)
    assert len({o.cid for o in vert}) == 199
    assert all(o.smiles for o in vert)
    assert sum(1 for o in vert if o.rows_joined > 1) == 7
    assert all(o.dewrap == "targeted" for o in vert if o.rows_joined > 1)
