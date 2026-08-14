"""The name heal-loop tier: `name_rules`, `name_gap`, `name_outcome`, `name_loop`.

Every OPSIN-touching test skips itself if java is unavailable rather than
failing the suite, the same convention `test_anchor.py` uses. Nothing here
makes an API call — `name_synthesize.propose` is either monkeypatched or
exercised through its no-key path, so the suite stays free.

Two of these are locked REGRESSIONS for bugs found by running the loop for the
first time, both of which produced a plausible-looking wrong answer rather than
a crash:

  - `apply_pattern` replaced one match at a time, so a name carrying the same
    defect twice could never be repaired. The loop read that as "the rule does
    not work", fed it back, and paid again to be told the same thing about a
    correct rule.
  - `ground` checked the site regex against the FIRST name in a batch only,
    rejecting a correct rule aimed at the third — which is the case the whole
    batching design exists to exploit.
"""
from __future__ import annotations

import pytest

from patentdb3.core import config
from patentdb3.repair.name_gap import NameGap, find_name_gaps
from patentdb3.repair.name_outcome import COVERAGE_MIN, measure
from patentdb3.repair.name_rules import (NamePattern, NameRuleLibrary,
                                         apply_pattern, ground)

PID = "US8952177"


def _xml(pid: str = PID) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="ignore")


def _p(**kw) -> NamePattern:
    base = dict(id="t", site="x", replacement="")
    base.update(kw)
    return NamePattern(**base)          # type: ignore[arg-type]


# ── apply_pattern ────────────────────────────────────────────────────────

def test_all_occurrences_candidate_comes_first():
    """THE REGRESSION. A name carrying the same defect twice must be fully
    repairable in one candidate.

    Real shape: `[1,1&#x2032;-biphenyl]` names carry the corruption once in the
    `4&#x2032;-` locant and again in the ring name. Repairing either alone
    still leaves OPSIN a string it rejects.
    """
    p = _p(site="Q", replacement="'")
    out = apply_pattern(p, "aQb-[1,1Q-biphenyl]")
    assert out, "no candidate at all"
    assert out[0][0] == "a'b-[1,1'-biphenyl]", "all-occurrences must be first"
    # the per-match readings are still offered
    assert "a'b-[1,1Q-biphenyl]" in [c for c, _ in out]
    assert "aQb-[1,1'-biphenyl]" in [c for c, _ in out]


def test_single_match_still_produces_one_candidate():
    out = apply_pattern(_p(site="Q", replacement="'"), "aQb")
    assert [c for c, _ in out] == ["a'b"]


def test_span_locates_the_replacement_not_the_match():
    """`name_repair._probe_window` slices the REPAIRED string by this span."""
    (rep, span), = apply_pattern(_p(site="XY", replacement="z"), "aXYb")
    assert rep == "azb"
    assert rep[span[0]:span[1]] == "z"


def test_no_match_yields_nothing():
    assert apply_pattern(_p(site="zzz", replacement="q"), "abc") == []


# ── ground: contract checks only ─────────────────────────────────────────

def test_ground_accepts_a_rule_aimed_at_a_later_name_in_the_batch():
    """THE OTHER REGRESSION. A rule is bought for a BATCH and may legitimately
    target the third name while leaving the first alone."""
    names = ["clean-one-aaaa", "clean-two-bbbb", "broken-dihydropuridin-3-yl"]
    # `ur` -> `yr`, not `puridin` -> `pyridin`: the tight spelling is what
    # `_MAX_INSERT` forces, and it is the smaller edit anyway.
    g = ground(_p(site="puridin", replacement="pyri"), names, [])
    assert g.ok, g.why


def test_ground_rejects_a_rule_matching_none_of_the_batch():
    g = ground(_p(site="zzzz", replacement="q"), ["abc", "def"], [])
    assert not g.ok
    assert "matches none" in g.why


def test_ground_rejects_a_large_edit():
    """A rule may repair typesetting; writing 20 characters of chemistry into
    a name is supplying DATA, which this tier does not buy."""
    g = ground(_p(site="a", replacement="x" * 20), ["abc"], [])
    assert not g.ok
    assert "distance" in g.why


def test_ground_accepts_a_long_replacement_that_is_a_small_edit():
    """THE THIRD REGRESSION, and it cost a whole sweep.

    The bound was `len(replacement) <= 4`, which rejected `puridin` ->
    `pyridin` — a ONE CHARACTER substitution in US10155002 that the model had
    diagnosed correctly — because the natural spelling of the rule writes seven
    characters. Five configurations adopted zero rules for this reason alone.
    What the bound is for is "do not write chemistry into the name", and edit
    distance says that directly.
    """
    g = ground(_p(site="puridin", replacement="pyridin"),
               ["...dihydropuridin-3-yl..."], [])
    assert g.ok, g.why


def test_ground_rejects_deleting_a_substituent_suffix():
    """`-2-yl` -> `` is distance 5. The amputation boundary falls out of the
    edit bound independently of the coverage gate downstream."""
    g = ground(_p(site="-2-yl$", replacement=""),
               ["1-methylpiperidin-2-yl"], [])
    assert not g.ok
    assert "distance" in g.why


def test_ground_rejects_an_uncompilable_regex():
    g = ground(_p(site="(unclosed", replacement=""), ["abc"], [])
    assert not g.ok
    assert "compile" in g.why


def test_ground_rejects_a_rule_that_changes_nothing():
    g = ground(_p(site="b", replacement="b"), ["abc"], [])
    assert not g.ok
    assert "unchanged" in g.why


def test_ground_rejects_collateral_damage_to_names_that_already_parse():
    """Invisible to the outcome gate — the damaged names are not in this gap
    and are never measured there. So it has to be blocked before the fact."""
    # A SMALL edit that fires everywhere. The edit-size bound is checked first,
    # so a fixture with a large edit would be rejected for that reason instead
    # and this test would pass without ever reaching the collateral check.
    clean = [f"methyl-{i}-benzamide" for i in range(100)]
    g = ground(_p(site="e", replacement="a"), ["methyl-brokene"], clean)
    assert not g.ok
    assert "ALREADY parse" in g.why


def test_ground_tolerates_collateral_below_the_threshold():
    clean = [f"compound-{i}" for i in range(100)] + ["rare-qq-here"]
    g = ground(_p(site="qq", replacement="q"), ["rare-qq-broken"], clean)
    assert g.ok, g.why


# ── the outcome gate ─────────────────────────────────────────────────────

def _opsin_ok() -> bool:
    try:
        from patentdb3.sources.opsin import batch
        return bool(batch(["benzene"], "SMILES")[0])
    except Exception:
        return False


requires_opsin = pytest.mark.skipif(
    not _opsin_ok(), reason="OPSIN/java unavailable in this environment")


@requires_opsin
def test_amputation_is_refused_even_though_it_parses():
    """THE CENTRAL CLAIM OF THIS TIER, on the real counter-example.

    Deleting the trailing `-2-yl` from a substituent yields a shorter name that
    OPSIN accepts happily — and it is a DIFFERENT MOLECULE. Corpus-measured:
    generic character trimming made 11,446 of 82,398 rejected seeds parse, and
    the most common text removed was `yl` (3,899) and `2-yl` (1,305).
    """
    name = "1-methylpiperidin-4-yl"
    gap = NameGap(patent_id="T", cid="1", name_text=name, doc_text=name)
    oc = measure(_p(site=r"-4-yl$", replacement=""), gap)
    assert not oc.positive
    assert oc.parsed_any, "precondition: the amputated form does parse"
    assert oc.coverage < COVERAGE_MIN or "covers only" in oc.detail


@requires_opsin
def test_comma_split_fragment_is_refused():
    """The other measured amputation: median coverage 0.14 over 106 cases."""
    full = ("5-(4-(3-azabicyclo[3.1.0]hexan-1-yl)phenyl)-2-amino-"
            "N-(4-hydroxycyclohexyl)nicotinamide")
    gap = NameGap(patent_id="T", cid="1", name_text=full, doc_text=full)
    oc = measure(_p(site=r"^.*\)", replacement=""), gap)
    assert not oc.positive, "a terminal fragment must never count as a repair"


@requires_opsin
def test_repair_without_corroboration_is_refused():
    """OPSIN acceptance plus full coverage is still not enough: nothing in the
    document confirms the correction is the RIGHT one."""
    name = "1-methylpiperidine-4-carboxamidee"
    gap = NameGap(patent_id="T", cid="1", name_text=name,
                  doc_text="unrelated text with no confirming occurrence")
    oc = measure(_p(site="ee$", replacement="e"), gap)
    assert oc.parsed_any, "precondition: the repaired form does parse"
    assert not oc.positive
    assert "nowhere else" in oc.detail


@requires_opsin
def test_a_corroborated_full_coverage_repair_is_kept():
    name = "1-methylpiperidine-4-carboxamidee"
    gap = NameGap(
        patent_id="T", cid="1", name_text=name,
        doc_text="elsewhere the patent writes 1-methylpiperidine-4-carboxamide "
                 "in full")
    oc = measure(_p(site="ee$", replacement="e"), gap)
    assert oc.positive, oc.detail
    assert oc.repaired == "1-methylpiperidine-4-carboxamide"
    assert oc.smiles
    assert oc.coverage >= COVERAGE_MIN
    assert oc.corroborated


# ── gap detection ────────────────────────────────────────────────────────

@requires_opsin
def test_gaps_carry_the_patents_own_compound_ids():
    gaps = find_name_gaps(_xml("US10155002"), "US10155002",
                          with_opsin_errors=False)
    if not gaps:
        pytest.skip("no gaps on this patent in the current tree")
    assert all(g.cid for g in gaps)
    assert all(g.name_text for g in gaps)
    assert all(g.doc_text for g in gaps)


@requires_opsin
def test_phane_and_conjunction_headings_are_not_offered_as_gaps():
    """Neither is repairable by a character rule: OPSIN 2.9.0 does not
    implement IUPAC P-26 phane nomenclature (it rejects `[2.2]paracyclophane`
    itself), and a two-compound heading needs a split, not a character fix."""
    from patentdb3.repair.name_gap import _CONJUNCTION, _PHANE
    assert _PHANE.search("...triazina-3(1,2)-benzenacyclononaphane-14,16-dione")
    assert _CONJUNCTION.search("...dione (Compound 5A) and (18S,Z)-12-hydroxy...")
    assert not _PHANE.search("2-methyl-4-(trifluoromethoxy)benzyl acetate")


@requires_opsin
def test_every_excluded_compound_is_recorded_not_just_skipped(tmp_path):
    """LEAVING THE LOOP IS NOT THE SAME AS LEAVING THE ARTIFACT.

    Three populations used to exit `find_name_gaps` through a bare `continue`:
    a name blocked only by stereochemistry, a phane, and two compounds sharing
    one heading. All three were then indistinguishable from a compound nothing
    ever looked at. Each now carries a reason code so a later pass can act on
    a list instead of re-deriving one.

    US10227341 is the fixture because its headings carry the conjunction shape
    (`Example 42 and Example 43: ...`) — verified against the cached XML, not
    constructed.
    """
    import json

    from patentdb3.sources import losses, opsin
    losses.reset(tmp_path / "loss.jsonl")
    find_name_gaps(_xml("US10227341"), "US10227341", with_opsin_errors=False)
    losses.flush()
    rows = [json.loads(l) for l in
            (tmp_path / "loss.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["loss_type"] == "asserted_name_unparsed"]
    assert rows, "every exclusion was silent again"
    assert {r["reason"] for r in rows} <= {
        opsin.REASON_STEREO, opsin.REASON_GRAMMAR,
        "two_compounds_one_assertion"}
    assert all(r["cid"] and r["name"] for r in rows)
    assert all(r["assertion"] in ("heading", "table") for r in rows)


# ── library ──────────────────────────────────────────────────────────────

def test_library_round_trips_and_replaces_by_id(tmp_path):
    lib = NameRuleLibrary(tmp_path / "lib.json")
    lib.add(_p(id="a", site="x", replacement="y", note="first"))
    lib.add(_p(id="a", site="x", replacement="z", note="second"))
    assert len(lib.patterns) == 1, "same id must replace, not accumulate"
    assert NameRuleLibrary(tmp_path / "lib.json").patterns[0].replacement == "z"


def test_digest_is_small_enough_to_send_every_call(tmp_path):
    lib = NameRuleLibrary(tmp_path / "lib.json")
    for i in range(40):
        lib.add(_p(id=f"r{i}", site=f"s{i}", replacement="q", note="x" * 200))
    assert len(lib.digest()) < 2000, "digest must stay a summary, not the library"


def test_empty_library_has_no_digest(tmp_path):
    assert NameRuleLibrary(tmp_path / "lib.json").digest() == ""


# ── degradation ──────────────────────────────────────────────────────────

def test_no_api_key_means_no_rule_and_no_crash(tmp_path, monkeypatch):
    """The same tolerance `repair_patent` has: absent credentials make the tier
    deterministic rather than broken."""
    from patentdb3.repair import name_loop, name_synthesize
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(name_synthesize, "propose",
                        lambda *a, **k: None)
    lib = NameRuleLibrary(tmp_path / "lib.json")
    rep = name_loop.repair_names("<doc></doc>", "TEST", library=lib,
                                 journal=tmp_path / "j.jsonl")
    assert rep.gaps_found == 0
    assert rep.rules_adopted == []
    assert rep.usd_spent == 0.0
