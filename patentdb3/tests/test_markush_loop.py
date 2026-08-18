"""The markush assembly tier: does its GATE work.

The loop's value is not that it can join two molecules — `markush.py` already
does that and is tested. Its value is that it refuses to keep a join the
document contradicts, and refuses to keep one the document cannot check at all.
So every test here is about the gate.

The scaffold is `[1*]c1ccc([2*])cc1` — para-disubstituted benzene, one dummy
per slot — and the substituents are text, so no recogniser is involved and the
expected molecule can be written down by hand:

    R1=CH3  R2=NH2   ->  Cc1ccc(N)cc1        108.08 (M+H)
    R1=CH3  R2=F     ->  Cc1ccc(F)cc1        111.06 (M+H)

Both verified against rdkit before being written here.
"""
from __future__ import annotations

import pytest

from patentdb3.repair import markush_loop as ML
from patentdb3.repair.markush_gap import MarkushGap
from patentdb3.repair.markush_outcome import measure
from patentdb3.sources.markush import MarkushRow, ROUTE_TEXT_ONLY

SCAFFOLD = "[1*]c1ccc([2*])cc1"
SCAF_REF = "CHEM-TEST-0001"


def _row(cid: str, r1: str, r2: str) -> MarkushRow:
    return MarkushRow(cid=cid, route=ROUTE_TEXT_ONLY,
                      slots={"R1": r1, "R2": r2},
                      varying={"R1": (None, r1), "R2": (None, r2)})


def _gap(rows, masses=None, held=None, **kw) -> MarkushGap:
    return MarkushGap(
        patent_id="USTEST", table_id="TABLE-US-00001", fingerprint="deadbeef",
        signature="slots=2|heads=r#|image=0|mass=1|scaffold=banner",
        scaffold_ref=SCAF_REF, rows=rows, headings=["R1", "R2"],
        printed_mass=masses or {}, held_out=held or {}, n_rows=len(rows), **kw)


def test_a_correct_plan_is_adopted_and_the_mass_says_so():
    rows = [_row("1", "CH3", "NH2"), _row("2", "CH3", "F")]
    gap = _gap(rows, masses={"1": 108, "2": 111})
    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD})
    assert rep.adopted, ML.summarise(rep.outcome) if rep.outcome else rep.blocked
    assert rep.outcome.mass_agrees == 2
    assert rep.outcome.mass_contradicts == 0
    assert rep.structures["1"] == "Cc1ccc(N)cc1"


def test_a_mass_the_patent_contradicts_is_never_adopted():
    """ONE contradicted row kills the plan. Not a rate — the patent has said
    this molecule is not that compound, and a plan is one rule for the table."""
    rows = [_row("1", "CH3", "NH2"), _row("2", "CH3", "F")]
    gap = _gap(rows, masses={"1": 108, "2": 400})       # 111 built vs 400 printed
    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD})
    assert not rep.adopted
    assert rep.outcome.mass_contradicts == 1
    assert rep.outcome.contradicted


def test_a_table_nothing_can_check_is_blocked_not_adopted():
    """US10626094's 31 rows print no mass and no name. An assembly of them is
    unfalsifiable, and shipping an unfalsifiable answer is how a wrong
    structure ships looking like a right one."""
    gap = _gap([_row("1", "CH3", "NH2")])               # no mass, no name, 1 row
    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD})
    assert not rep.adopted
    assert rep.blocked == ML.BLOCK_NO_REFEREE


def test_a_held_out_name_can_referee_on_its_own():
    rows = [_row("1", "CH3", "NH2"), _row("2", "CH3", "F")]
    gap = _gap(rows, held={"1": "4-methylaniline"})
    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD})
    assert rep.outcome.name_agrees == 1
    assert rep.outcome.name_disagrees == 0
    assert rep.adopted


def test_a_held_out_name_that_disagrees_kills_the_plan():
    rows = [_row("1", "CH3", "NH2"), _row("2", "CH3", "F")]
    gap = _gap(rows, held={"1": "aniline"})             # built is 4-methylaniline
    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD})
    assert rep.outcome.name_disagrees == 1
    assert not rep.adopted


def test_every_row_identical_is_a_failure_not_a_success():
    """The failure that looks most like success: a slot left unwired makes
    every row of the table come out as the same molecule."""
    rows = [_row("1", "CH3", "NH2"), _row("2", "CH3", "NH2")]
    gap = _gap(rows, masses={"1": 108, "2": 108})
    oc = measure(gap, {"1": "Cc1ccc(N)cc1", "2": "Cc1ccc(N)cc1"})
    assert oc.mass_agrees == 2                # the mass is perfectly happy
    assert oc.all_identical
    assert not oc.positive                    # and it is still refused


def test_an_unrecognised_scaffold_blocks_rather_than_guesses():
    gap = _gap([_row("1", "CH3", "NH2")], masses={"1": 108})
    rep = ML.repair_table(gap, {})            # no structures at all
    assert not rep.adopted
    assert rep.blocked == ML.BLOCK_NO_SCAFFOLD
    assert not rep.structures


def test_a_heading_with_no_number_is_not_guessed_at():
    """`Ar` names a position the drawing marks some other way. Refusing is the
    point: a guessed join builds a clean-looking wrong molecule."""
    row = MarkushRow(cid="1", route=ROUTE_TEXT_ONLY,
                     slots={"Ar": "phenyl", "R2": "NH2"},
                     varying={"Ar": (None, "phenyl"), "R2": (None, "NH2")})
    gap = _gap([row], masses={"1": 169})
    gap.headings = ["Ar", "R2"]
    gap.names = {"phenyl": "*c1ccccc1"}
    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD})
    assert not rep.adopted
    assert "no attachment point stated for: Ar" in rep.plan.note


def test_a_plan_supplies_what_the_heading_does_not():
    """And with the slot map the model would propose, the same table builds."""
    row = MarkushRow(cid="1", route=ROUTE_TEXT_ONLY,
                     slots={"Ar": "phenyl", "R2": "NH2"},
                     varying={"Ar": (None, "phenyl"), "R2": (None, "NH2")})
    gap = _gap([row], masses={"1": 171})       # 4-aminobiphenyl, verified
    gap.headings = ["Ar", "R2"]
    gap.names = {"phenyl": "*c1ccccc1"}

    def propose(_gap, _feedback):
        return ML.Plan(slot_map={"Ar": 1}, source="model")

    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD}, propose=propose)
    assert rep.adopted, ML.summarise(rep.outcome)
    assert rep.plan.source == "model"
    assert rep.attempts == 1                   # attempt 0 free, attempt 1 paid


def test_the_loop_stops_after_max_attempts():
    row = _row("1", "CH3", "NH2")
    gap = _gap([row], masses={"1": 999})       # can never agree
    calls = []

    def propose(_gap, feedback):
        calls.append(feedback)
        return ML.Plan(slot_map={}, source="model")

    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD}, propose=propose)
    assert not rep.adopted
    assert len(calls) == ML.MAX_ATTEMPTS
    assert all(c for c in calls), "the model must be shown its own measurement"


def test_the_free_plan_runs_first_and_costs_nothing():
    """A tier that calls a model before trying the free answer spends its
    budget proving it did not need to."""
    rows = [_row("1", "CH3", "NH2"), _row("2", "CH3", "F")]
    gap = _gap(rows, masses={"1": 108, "2": 111})
    called = []
    rep = ML.repair_table(gap, {SCAF_REF: SCAFFOLD},
                          propose=lambda g, f: called.append(f))
    assert rep.adopted
    assert rep.plan.source == "deterministic"
    assert called == []


@pytest.mark.parametrize("pid,tid,rows,masses", [
    ("US9718825", "TABLE-US-00001", 426, 426),
    ("US9718825", "TABLE-US-00003", 81, 81),
    ("US10626094", "TABLE-US-00002", 40, 0),
])
def test_the_real_tables_are_found_with_the_referees_the_page_gives(
        pid, tid, rows, masses):
    """The corpus half of the claim: these tables print their own m/z, which
    `CLAUDE.md` recorded as impossible for markush tables. It was measured with
    a mass finder that read one vendor's phrasing."""
    from patentdb3.core import config
    from patentdb3.repair.markush_gap import find_gaps

    p = config.XML_INPUT_DIR / f"{pid}.xml"
    if not p.exists():
        pytest.skip(f"{pid}.xml not cached")
    gaps = {g.table_id: g for g in find_gaps(pid, p.read_text(errors="replace"))}
    assert tid in gaps, f"{tid} was not found as a substituent table"
    g = gaps[tid]
    assert g.n_rows == rows
    assert len(g.printed_mass) == masses
    assert g.scaffold_ref, "the shared scaffold drawing was not located"


def test_one_layout_serves_every_table_of_that_shape():
    """The fingerprint is the LAYOUT, never the patent — so a plan bought for
    US9718825's Table 1 is reused on its Tables 2, 3 and 7 without paying
    again. Four tables, 632 rows, one question."""
    from patentdb3.core import config
    from patentdb3.repair.markush_gap import find_gaps

    p = config.XML_INPUT_DIR / "US9718825.xml"
    if not p.exists():
        pytest.skip("US9718825.xml not cached")
    gaps = find_gaps("US9718825", p.read_text(errors="replace"))
    with_mass = [g for g in gaps if g.printed_mass]
    assert len({g.fingerprint for g in with_mass}) == 1
    assert sum(g.n_rows for g in with_mass) >= 600
