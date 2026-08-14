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
