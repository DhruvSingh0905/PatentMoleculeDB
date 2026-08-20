"""The anchor check, and the phrase read that stopped feeding it wrong rows.

Every string here is real. The shapes come from US10730863 and US20250163063,
and the numbers come from those patents' own table rows.
"""
import json

import pytest

from patentdb3.core import config
from patentdb3.sources import images, losses, mass_gate
from patentdb3.sources.cid_first import _INTERMEDIATE_FOR, _LABEL_WINDOW
from patentdb3.sources.iupac_names import NamedCompound

requires_opsin = pytest.mark.skipif(
    not (config.XML_INPUT_DIR / "US10730863.xml").exists(),
    reason="US10730863.xml not cached")


# ── A. the phrase read ────────────────────────────────────────────────────

@pytest.mark.parametrize("framing,refuse", [
    # THE DEFECT. A name follows the id, and it is the intermediate's name.
    ("Preparation of Intermediate for Example ", True),
    ("Intermediate for the Synthesis of Example ", True),
    ("Preparation of Intermediate for Compound No. ", True),
    # NOT THE DEFECT. Yield data follows, and the id IS the compound whose
    # synthesis is being described — a step FOR THE PREPARATION OF an example
    # is that example's own work, unlike an intermediate FOR an example.
    ("Step G for the preparation of Example ", False),
    ("Preparation as described for example ", False),
    ("step-1 of scheme 208 for preparation of compound ", False),
    # A second id and a digit sit between the noun and `for`, so `Intermediate`
    # is not heading this phrase at all.
    ("Intermediate 519E according to methods described for the synthesis of Example ",
     False),
    ("Example ", False),
    ("Preparation of Example ", False),
])
def test_the_framing_decides_not_the_adjacent_word(framing, refuse):
    assert bool(_INTERMEDIATE_FOR.search(framing[-72:])) is refuse


def test_the_adjacent_word_alone_cannot_see_the_defect():
    """Why the wider window is necessary, stated as a test.

    `_LABEL_BEFORE` reads 24 characters. `Intermediate` sits outside them.
    """
    framing = "Preparation of Intermediate for Example "
    assert "intermediate" not in framing[-_LABEL_WINDOW:].lower()
    assert _INTERMEDIATE_FOR.search(framing[-72:])


@requires_opsin
def test_the_reagent_no_longer_ships_under_the_compound_number():
    """US10730863 compound 534 is the worked case.

    The document writes `Preparation of Intermediate for Example 534.` and
    then the intermediate's name. That heading is the ONLY place the number
    appears, so there is no correct text answer to find and the route must
    refuse rather than take the one that is there.
    """
    from patentdb3.sources.cid_first import extract_by_cid
    xml = (config.XML_INPUT_DIR / "US10730863.xml").read_text(errors="replace")
    rows = {n.cid: n for n in extract_by_cid(xml, "US10730863") if n.cid}
    for cid in ("524", "528", "534"):
        got = rows.get(cid)
        assert not (got and got.inchikey), (
            f"cid={cid} still resolves to {got.name!r}")


# ── B. the gate ───────────────────────────────────────────────────────────

def test_reported_masses_are_read_per_row_never_from_a_neighbour():
    xml = ("<table><row><entry>534</entry><entry>MS (ESI) 561 (M + H)</entry>"
           "</row><row><entry>535</entry><entry>no mass here</entry></row>"
           "<row><entry>536</entry><entry>MS (ESI) 608 (M+H)</entry></row>"
           "</table>")
    assert mass_gate.reported_masses(xml) == {"534": 561, "536": 608}


def test_an_nmr_solvent_is_not_a_mass():
    """`DMSO-d 6` contains `MS`. The pattern requires the ionisation mode."""
    xml = "<row><entry>12</entry><entry>1H NMR (DMSO-d 6) 8.75</entry></row>"
    assert mass_gate.reported_masses(xml) == {}


def test_the_window_clears_every_correct_structure_measured():
    """WHAT THE WINDOW IS FOR, AND WHAT IT CANNOT DO.

    Measured over the 70 structures on US10730863 that have a printed mass:
    the worst |delta| on a CORRECT structure is 1.20 Da, and DECIMER's
    two-hydrogen error reads as 1.19 Da once average mass is allowed. Those
    overlap, so no threshold separates them and this one does not pretend to.

    It is set above every correct structure because a false flag here is
    expensive — `images.emit` demotes a flagged row out of the truth set, and
    the truth set is the only thing any recogniser can be scored against.
    """
    for m in (200, 561, 596, 900):
        assert 1.20 < mass_gate.tolerance(m) < 2.016


def test_a_negative_mode_row_is_not_judged_as_if_it_were_positive():
    """US10125101 prints 20 `[M-H]-` rows beside 44 `[M+H]+`.

    Adding a proton to a negative-mode row makes a CORRECT structure read
    2.015 Da light — outside the window, so it is reported as contradicting,
    and `images.emit` then discards it from the truth set. 22 rows corpus-wide
    were affected.

    The markup is the real shape from that patent: a Unicode minus, as an XML
    entity, inside a `<sup>` tag.
    """
    xml = ('<table>'
           '<row><entry>1</entry><entry>MS (ESI<sup>+</sup>): m/z = 181 '
           '[M + H]<sup>+</sup></entry></row>'
           '<row><entry>2</entry><entry>MS (ESI&#x2212;): m/z = 179 '
           '[M &#x2212; H]&#x2212;</entry></row></table>')
    assert mass_gate.reported_masses(xml) == {"1": 181, "2": 179}
    shifts = mass_gate.reported_shifts(xml)
    assert shifts["1"] == pytest.approx(mass_gate.PROTON)
    assert shifts["2"] == pytest.approx(-mass_gate.PROTON)

    # ASPIRIN is 180.04 neutral: 181.05 as [M+H], 179.03 as [M-H]. Both rows
    # describe the same correct structure and both must agree.
    for cid in ("1", "2"):
        v, _ = mass_gate.verdict(ASPIRIN, mass_gate.reported_masses(xml)[cid],
                                 shifts[cid])
        assert v == mass_gate.VERDICT_AGREES, cid
    # and the bug this replaced: the same row judged as [M+H]
    assert mass_gate.verdict(ASPIRIN, 179, mass_gate.PROTON)[0] == \
        mass_gate.VERDICT_CONTRADICTS


def test_a_large_error_is_still_caught_at_this_window():
    """The class the gate demonstrably resolves: whole-molecule substitution.

    Compound 96's DECIMER answer read a fluorine as a methyl — 3.97 Da, well
    clear. Compound 534's text answer was a 376 Da reagent.
    """
    assert mass_gate.verdict(ASPIRIN, 185)[0] == mass_gate.VERDICT_CONTRADICTS
    assert mass_gate.verdict(ASPIRIN, 561)[0] == mass_gate.VERDICT_CONTRADICTS


def test_either_monoisotopic_or_average_may_be_what_the_patent_printed():
    """US10730863 prints both conventions, and never says which.

    Compound 140's correct structure is 595.18 monoisotopic / 596.53 average
    against a printed 596 — average fits. Compound 438's is 684.14 / 685.61
    against a printed 684 — monoisotopic fits. Demanding either one alone
    calls a right answer wrong.
    """
    c140 = ("O=C(O)c1ccc2c(c1)CN(C(=O)C13CCC(CC1)(CC3)COc1c(C3CC3)onc1-c1c(Cl)"
            "cccc1Cl)CC2")
    assert mass_gate.verdict(c140, 596)[0] == mass_gate.VERDICT_AGREES


ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"          # 180.04, M+H 181.05


def test_a_structure_that_matches_its_row_agrees():
    v, d = mass_gate.verdict(ASPIRIN, 181)
    assert v == mass_gate.VERDICT_AGREES
    assert abs(d) < 0.1


def test_an_anchored_reagent_contradicts():
    """The real magnitude: compound 534 printed 561 and carried a 185 Da row."""
    v, d = mass_gate.verdict(ASPIRIN, 561)
    assert v == mass_gate.VERDICT_CONTRADICTS
    assert d < -300


def test_no_reported_mass_is_unchecked_and_never_a_pass():
    assert mass_gate.verdict(ASPIRIN, None)[0] == mass_gate.VERDICT_UNCHECKED
    assert mass_gate.verdict("", 181)[0] == mass_gate.VERDICT_UNCHECKED
    assert mass_gate.verdict("not a smiles", 181)[0] == mass_gate.VERDICT_UNCHECKED


def _row(cid, smiles, name="x"):
    return NamedCompound(patent_id="US1", name=name, smiles=smiles,
                         inchikey="K", start=-1, cid=cid)


def test_check_stamps_every_row_and_records_only_the_contradictions(tmp_path):
    losses.reset(tmp_path / "loss.jsonl")
    xml = ("<table><row><entry>1</entry><entry>MS (ESI) 181 (M+H)</entry></row>"
           "<row><entry>2</entry><entry>MS (ESI) 561 (M+H)</entry></row>"
           "<row><entry>3</entry><entry>no mass</entry></row></table>")
    rows = [_row("1", ASPIRIN), _row("2", ASPIRIN), _row("3", ASPIRIN)]
    tally = mass_gate.check(rows, xml, "US1")
    losses.flush()
    assert [r.mass_check for r in rows] == ["agrees", "contradicts", ""]
    assert tally["agrees"] == 1 and tally["contradicts"] == 1
    recorded = [json.loads(l) for l in
                (tmp_path / "loss.jsonl").read_text().splitlines() if l.strip()]
    assert [r["cid"] for r in recorded] == ["2"]
    assert recorded[0]["reported_mh"] == 561


def test_a_gate_that_cannot_run_says_so_in_the_data(tmp_path, monkeypatch):
    """THE WEEK-LONG-BLINDNESS TEST.

    Without rdkit nothing can be weighed. If that produced a blank column the
    gate would be switched off and every artifact would still look normal —
    blank is the value 99.8% of rows carry anyway, so it carries no
    information. A row that HAD a mass to check against must say the check
    could not run, and the loss log must hold a record.
    """
    losses.reset(tmp_path / "loss.jsonl")
    monkeypatch.setattr(mass_gate, "available", lambda: False)
    xml = ("<table><row><entry>1</entry><entry>MS (ESI) 181 (M+H)</entry></row>"
           "<row><entry>2</entry><entry>no mass</entry></row></table>")
    rows = [_row("1", ASPIRIN), _row("2", ASPIRIN)]
    tally = mass_gate.check(rows, xml, "US1")
    losses.flush()
    assert rows[0].mass_check == mass_gate.VERDICT_UNAVAILABLE
    assert rows[1].mass_check == mass_gate.VERDICT_UNCHECKED
    assert tally[mass_gate.VERDICT_UNAVAILABLE] == 1
    recorded = [json.loads(l) for l in
                (tmp_path / "loss.jsonl").read_text().splitlines() if l.strip()]
    assert [r["loss_type"] for r in recorded] == ["mass_gate_unavailable"]


def test_gate_unavailable_is_never_confused_with_a_pass():
    """Four values, and only one of them means the row was checked and fine."""
    assert mass_gate.VERDICT_UNAVAILABLE not in (
        mass_gate.VERDICT_AGREES, mass_gate.VERDICT_UNCHECKED)
    assert mass_gate.VERDICT_UNAVAILABLE != ""


def test_the_gate_never_drops_a_row():
    """It doubts a row. It does not delete one — see the module docstring."""
    xml = "<row><entry>1</entry><entry>MS (ESI) 561 (M+H)</entry></row>"
    rows = [_row("1", ASPIRIN)]
    mass_gate.check(rows, xml, "US1")
    assert len(rows) == 1 and rows[0].smiles == ASPIRIN


# ── C. the truth set ──────────────────────────────────────────────────────

def test_a_flagged_answer_is_demoted_not_used_as_truth(tmp_path, monkeypatch):
    """A VALIDATE row is the only thing an image reader is scored against.

    Scoring against a flagged answer inverts the measurement: a correct read
    disagrees with a reagent and is recorded as an error.
    """
    struct = tmp_path / "structures.tsv"
    dump = tmp_path / "dump.tsv"
    man = tmp_path / "latest.json"
    struct.write_text(
        "patent_id\tcid\tinchikey\tmass_check\n"
        "US1\t10\tGOODKEY-UHFFFAOYSA-N\tagrees\n"
        "US1\t11\tBADKEY-UHFFFAOYSA-N\tcontradicts\n")
    dump.write_text("patent_id\tcid\tassay_name\tvalue_numeric\tunit\n"
                    "US1\t10\tFXR\t1\tnM\nUS1\t11\tFXR\t2\tnM\n")
    man.write_text(json.dumps({"structures": str(struct), "dump": str(dump)}))
    monkeypatch.setattr(config, "MANIFEST", man)
    monkeypatch.setattr(config, "XML_INPUT_DIR", tmp_path)   # no XML -> no rows
    out = images.emit(path=tmp_path / "wl.tsv")
    assert out["demoted_mass_contradicts"] == 1


# ── D. the prose shape ────────────────────────────────────────────────────
#
# Every string below is the real shape from the patent named in its docstring.
# The gate read `<row>` elements and nothing else, so a document that states
# its masses in prose had no reference at all: 3,735 of 38,402 structures
# carried a verdict, and the documents with the most defects carried none.


def test_a_mass_stated_under_a_heading_is_read():
    """US10245267. The id is in the heading, the mass in the `<p>` below it."""
    xml = ('<heading>Example 418: N-(4-cyanophenyl)benzamide</heading>'
           '<p><chemistry><img file="C00536.TIF"/></chemistry></p>'
           '<p>1H NMR (400 MHz) 7.17-7.32 (m, 2H). '
           'LCMS (m/z) (M+H)=477.2, Rt=0.78 min.</p>'
           '<heading>Example 419: N-(4-cyanophenyl)pyridine-4-carboxamide'
           '</heading>'
           '<p>1H NMR (400 MHz) 8.84 (d, 1H). '
           'LCMS (m/z) (M+H)=473.2, Rt=0.76 min.</p>')
    assert mass_gate.reported_masses(xml) == {"418": 477.2, "419": 473.2}


def test_a_molecular_formula_subscript_is_not_a_mass():
    """US10280164 and US10722495.

    `LCMS calculated for C 12 H 18 ClIN 3 OSi (M+H) + m/z=410.0` gave 12 — the
    carbon count — because the gap between the instrument and the mass has
    digits in it. Every compound in both documents read as contradicting.
    """
    xml = ('<heading>Example 7. 6-Chloro-3-iodopyrazolo[4,3-c]pyridine</heading>'
           '<p>The crude product was purified by chromatography (3.20 g, 60%). '
           'LCMS calculated for C 12 H 18 ClIN 3 OSi (M+H) '
           '<sup>+</sup> m/z=410.0; found 410.0.</p>')
    assert mass_gate.reported_masses(xml) == {"7": 410.0}


def test_an_instrument_word_used_as_a_verb_states_no_mass():
    """US20250163061A1 writes `monitored by LCMS control). Then a water
    solution ... was stirred for 30 minutes`. A bare window reads the 30.

    Distance from the instrument has to be earned by a formula in the way.
    """
    xml = ('<heading>Example 9. tert-Butyl piperazine-1-carboxylate</heading>'
           '<p>The mixture was stirred (LCMS control). Then a water solution '
           'of sodium bicarbonate was added and stirred for 30 minutes.</p>')
    assert mass_gate.reported_masses(xml) == {}


def test_the_last_mass_in_a_synthesis_is_the_one_the_heading_names():
    """US9694016. A heading section is a whole synthesis, and the first mass
    belongs to Step 1's intermediate.

    Reading the first reported every multi-step example as contradicting by
    the mass of everything the last step still had to add — a constant 243 Da
    on cids 1, 3 and 6 alike, which is the signature of comparing against the
    wrong molecule rather than of a wrong structure.
    """
    xml = ('<heading>Example 1: Synthesis of N-(4-methylphenyl)benzamide</heading>'
           '<p>Step 1. The residue gave a white solid in 93% yield. '
           'LCMS (m/z) (M+H)=200.0/201.8, Rt=0.35 min.</p>'
           '<p>Step 2. The title compound was obtained. '
           'LCMS (m/z) (M+H)=443.2, Rt=0.88 min.</p>')
    assert mass_gate.reported_masses(xml) == {"1": 443.2}


def test_a_preparation_does_not_donate_its_mass_to_an_example():
    """US20250163061A1 numbers `Preparation 16` and `Example 16` separately.

    Both normalise to the cid `16`, so the intermediate's mass landed on the
    example's structure and reported a correct molecule as contradicting. The
    name route already refuses these headings; the referee refuses the same
    set, so the two cannot disagree about which headings assert a compound.
    """
    xml = ('<heading>Preparation 16. tert-Butyl 4-cyanopiperazine-1-carboxylate'
           '</heading><p>LCMS (ESI) [MH]<sup>+</sup>: 727.</p>'
           '<heading>Example 16. N-(4-cyanophenyl)benzamide</heading>'
           '<p>LCMS (ESI) [MH]<sup>+</sup>: 620.</p>')
    assert mass_gate.reported_masses(xml) == {"16": 620.0}


def test_the_key_is_normalised_on_both_sides():
    """A document numbering its examples `007` produced a dict keyed `007`
    while every lookup asked for `7`. Zero overlap, and the gate reported
    those rows unchecked — the same shape as `cid_first`'s raw-cell dict.
    """
    xml = ('<heading>Example 007. N-(4-cyanophenyl)benzamide</heading>'
           '<p>LCMS (ESI) m/z 238.1 (M+H).</p>')
    assert mass_gate.reported_masses(xml) == {"7": 238.1}

    rows = [NamedCompound(patent_id="US1", cid="7", name="benzamide",
                          smiles="O=C(N)c1ccccc1", inchikey="", start=0)]
    mass_gate.check(rows, xml, "US1")
    assert rows[0].mass_check in (mass_gate.VERDICT_AGREES,
                                  mass_gate.VERDICT_CONTRADICTS,
                                  mass_gate.VERDICT_UNAVAILABLE)


def test_a_mass_inside_a_table_belongs_to_its_row_not_to_the_heading():
    """A heading section may CONTAIN a table. The row shape already owns
    anything printed inside one, so the section is read with tables cut out.
    """
    xml = ('<heading>Example 5. N-(4-cyanophenyl)benzamide</heading>'
           '<tables><table><row><entry>91</entry>'
           '<entry>MS (ESI) 561 (M + H)</entry></row></table></tables>'
           '<p>LCMS (ESI) m/z 238.1 (M+H).</p>')
    assert mass_gate.reported_masses(xml) == {"91": 561.0, "5": 238.1}


# ── E. a number must be read whole, and a salt weighed as its free base ───

def test_a_retention_time_is_not_a_mass():
    """US12011444 writes `LC-MS A: t_R=0.69 min; [M+H]+=426.97`.

    `_NUMBER` read `\\d{2,4}(?:\\.\\d+)?`, which can start in the MIDDLE of a
    number: `0.69` has one digit before the point, so the pattern could not
    match at the `0` and matched the `69` instead. Every one of that patent's
    88 weighable structures was weighed against a retention time. US10730877,
    US10544143 and US11053244 print the same boilerplate.

    A number is now matched whole and judged on its digits before the point.
    """
    assert mass_gate.printed_mass(
        "LC-MS A: tR=0.69 min; [M+H]+=426.97.") == 426.97
    assert mass_gate.printed_mass(
        "LC-MS A: t R =0.67 min; [M+H]+=502.1") == 502.1
    # the shapes that already worked must keep working
    assert mass_gate.printed_mass("MS (ESI) 485 (M+H)") == 485.0
    assert mass_gate.printed_mass("LCMS (m/z) (M+H)=477.2, Rt=0.78 min.") == 477.2
    assert mass_gate.printed_mass(
        "LCMS calculated for C 12 H 18 ClIN 3 OSi (M+H) + m/z=410.0") == 410.0


def test_a_salt_is_weighed_as_its_free_base():
    """`...azetidin-3-ol trifluoroacetic acid salt` resolves to two
    disconnected fragments, and the patent reports `[M+H]+` for the amine
    alone — a counterion does not carry the charge in positive-mode ESI.

    Weighing the whole SMILES added the counterion to our side only. 168 rows
    contradicted on nothing else, and on 155 of them the delta IS the
    counterion: +115.0 for trifluoroacetate, +36.98 for chloride, +228.0 for a
    bis-TFA salt.
    """
    pytest.importorskip("rdkit")
    base = "Nc1ncc(-c2ccc(-c3ccccc3S(=O)(=O)N3CC(O)C3)c(F)c2)cn1"
    salt = "FC(C(=O)O)(F)F." + base
    mono_base, _ = mass_gate._mass(base)
    mono_salt, _ = mass_gate._mass(salt)
    assert mono_salt == pytest.approx(mono_base), (
        "the trifluoroacetate was weighed into the compound")

    # and the verdict follows: the patent prints the free base's [M+H]
    reported = mono_base + mass_gate.PROTON
    assert mass_gate.verdict(salt, reported)[0] == mass_gate.VERDICT_AGREES
