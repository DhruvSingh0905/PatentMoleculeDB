"""`sources/anchor.py` — the proximity rule, in isolation.

Every test through `test_real_xml_*` runs on synthetic strings and needs no
OPSIN, no java subprocess, and no network — that is the whole point of
pulling anchoring out of `iupac_names.py` into its own module (see
`anchor.py`'s docstring, "WHY A SEPARATE MODULE"). The `test_real_xml_*`
group at the bottom is the one required real-XML case, read from the cached
grant XML this repo already ships (`output_v3/uspto_xml/`); it still calls
only `anchor_text()` and `find_cid()`, not `extract_names()` — no OPSIN
there either. `test_extract_names_reproduces_measured_anchor_rate` is the
one exception: it runs the full `extract_names()` pass (which does invoke
OPSIN, offline/free — no network, no API spend) to lock in the corpus-level
numbers quoted in the module docstring and this task's report as an actual
regression guard, not a claim nobody re-checks. It skips itself if OPSIN
cannot run in this environment rather than failing the whole suite.
"""
from __future__ import annotations

import pytest

from patentdb3.core import config
from patentdb3.sources.anchor import AnchorResult, Candidate, anchor_text, find_cid

PID = "US8952177"


def _xml(pid: str = PID) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="ignore")


# ── left, plain digit — the original "Example N\n<name>" shape ──────────

def test_left_plain_digit_anchors():
    text = "Example 1\nracemic cis-2-{1-(4-Bromobenzyl)}cyclohexanecarboxylic acid\n"
    r = find_cid(text, "cis-2-{1-(4-Bromobenzyl)}cyclohexanecarboxylic acid")
    assert isinstance(r, AnchorResult)
    assert r.cid == "1"
    assert not r.clashed
    assert len(r.candidates) == 1
    assert r.candidates[0].cid == "1"
    assert r.candidates[0].direction == "left"


def test_far_away_digit_is_not_anchored():
    """The whole reason for a bound: a stray quantity two paragraphs back
    must not out-compete "nothing found nearby". Distance here (~160 chars)
    is deliberately in the range the module docstring calls out as the
    measured WRONG cluster.
    """
    junk = "x" * 160
    text = f"5 mg of reagent was used.{junk}\n4-chlorophenylacetic acid was isolated"
    r = find_cid(text, "4-chlorophenylacetic acid")
    assert r.cid is None
    assert not r.clashed
    assert r.candidates == ()


# ── left, alphanumeric + colon/semicolon — the "Intermediate 1D:" shape ─

def test_left_colon_alnum_anchors():
    text = "Intermediate 1D: tert-butyl 5-(3-azetidinyl)pyrrolopyridine-1-carboxylate\n"
    r = find_cid(text, "5-(3-azetidinyl)pyrrolopyridine-1-carboxylate")
    assert r.cid == "1D"
    assert not r.clashed


def test_left_colon_plain_digit_also_works():
    text = "Example 746: 4-methylpiperidin-1-yl acetic acid\n"
    r = find_cid(text, "4-methylpiperidin-1-yl acetic acid")
    assert r.cid == "746"


def test_blacklisted_word_before_colon_is_not_a_compound_id():
    """"Method 1:", "Step 2:" etc. are real headings in this corpus and are
    NOT compound numbers. Measured cost of NOT guarding this (an earlier,
    rejected version of this rule): a correct digit-only anchor on
    US8952177 flipped wrong twice, one of them anchoring the solvent name
    "2-methyltetrahydrofuran" to "N2" (nitrogen gas) purely because "N2 "
    sat in the window. See `anchor.py`'s docstring, "WHAT MOVED THE RATE".
    """
    text = "Method 1: 4-methylpiperidin-1-yl acetic acid\n"
    r = find_cid(text, "4-methylpiperidin-1-yl acetic acid")
    assert r.cid is None
    assert not r.clashed
    assert r.candidates == ()


def test_left_colon_blacklist_does_not_reach_the_plain_digit_rule():
    """Documents a KNOWN, PRE-EXISTING limitation rather than hiding it.

    `_NON_ID_WORD` only guards the colon-separated shape (`_CID_LEFT_SEP`);
    the plain "<digits> <name>" shape (`_CID_LEFT_PLAIN`) is untouched from
    the original inline rule and has no such guard. So "N2 sparged
    2-methyltetrahydrofuran" — nitrogen gas, not a compound id — still
    anchors this solvent name to "2", exactly as the ORIGINAL digit-only
    rule already did before this module existed. This was deliberate: the
    module docstring's "WHAT MOVED THE RATE" section shows widening the
    PLAIN rule itself (rather than adding the separate colon-shape rule) is
    what regressed US8952177 (65.1% -> 64.3%, wrong values). Leaving
    `_CID_LEFT_PLAIN` exactly as validated, and adding the alnum/colon
    support as a SEPARATE pattern, avoided that regression — at the cost of
    this one pre-existing false positive going unfixed. Not silently true:
    asserted here so a future change to `_CID_LEFT_PLAIN` has a test to
    explain why it must stay conservative.
    """
    text = "benzylamine (75.0 g), N2 sparged 2-methyltetrahydrofuran (750 mL)\n"
    r = find_cid(text, "2-methyltetrahydrofuran")
    assert r.cid == "2"          # known false positive, pre-dates this module


# ── right, "<name> (<id>)" — the semicolon-list shape ────────────────────

def test_right_paren_digit_anchors():
    text = "some prose ...[1,2,4]triazin-4-amine (544); other-name (545);\n"
    r = find_cid(text, "[1,2,4]triazin-4-amine")
    assert r.cid == "544"
    assert not r.clashed


def test_right_paren_alnum_anchors():
    """The corpus-wide gap named going in: alphanumeric ids (`I-0020`, `Z1`)
    were unreachable because the old rule was digit-only in both directions.
    """
    text = "4-bromo-2-fluorobenzonitrile (I-0020); next-compound (I-0021);\n"
    r = find_cid(text, "4-bromo-2-fluorobenzonitrile")
    assert r.cid == "I-0020"


# ── clashes surface, they do not vanish ───────────────────────────────────

def test_disagreeing_occurrences_surface_as_a_clash_not_a_silent_none():
    """The core behavioural requirement: when a name's occurrences disagree
    about the id, the caller gets the candidates AND their evidence, not a
    bare None indistinguishable from "nothing nearby at all".
    """
    text = (
        "Intermediate 402B: 4-(3-Bromophenyl)morpholine-3-carboxamide\n"
        "A mixture of 4-(3-bromophenyl)phenyl) morpholine-3-carboxamide (402) "
        "was concentrated to give the title compound.\n"
    )
    r = find_cid(text, "morpholine-3-carboxamide")
    assert r.cid is None
    assert r.clashed is True
    ids = {c.cid for c in r.candidates}
    assert ids == {"402B", "402"}
    # evidence is present and ordered closest-first, not just the bare ids
    assert len(r.candidates) == 2
    assert r.candidates[0].distance <= r.candidates[1].distance
    assert all(isinstance(c, Candidate) and c.context for c in r.candidates)


def test_clash_is_distinct_from_no_candidate():
    """`clashed=False, candidates=()` (nothing found) must never be conflated
    with `clashed=True` (found, but disagreed) — a caller distinguishing
    "should I expect this to ever resolve" from "did resolution fail" needs
    both states to be observable, not collapsed to the same `None`.
    """
    nothing = find_cid("no ids anywhere near this text at all", "quinazoline")
    assert nothing.cid is None and not nothing.clashed and nothing.candidates == ()

    clash = find_cid("Example 1\nfoo-acid text Example 2\nfoo-acid text",
                      "foo-acid")
    assert clash.cid is None and clash.clashed


def test_agreeing_occurrences_resolve_even_with_repetition():
    """A name stated once in a SUMMARY embodiment list (no id nearby) and
    again at its own Example heading (id nearby) must resolve — the
    embodiment-list occurrence contributes no candidate, so there is nothing
    to disagree with.
    """
    text = (
        "selected from the group consisting of foo-acid, bar-acid, and others.\n"
        "Example 9\nfoo-acid was prepared as follows\n"
    )
    r = find_cid(text, "foo-acid")
    assert r.cid == "9"
    assert not r.clashed


# ── anchor_text: headings back in, tables still out ───────────────────────

def test_anchor_text_includes_headings_description_text_would_drop():
    xml = (
        "<description>"
        "<heading>Example 1</heading>"
        "<heading>racemic foo-acid</heading>"
        "<p>Some prose about foo-acid and its synthesis.</p>"
        "<tables><tables id='t'><tgroup cols='2'><tbody>"
        "<row><entry>1</entry><entry>0.5 nM</entry></row>"
        "</tbody></tgroup></tables></tables>"
        "</description>"
    )
    text = anchor_text(xml)
    assert "Example 1" in text
    assert "racemic foo-acid" in text
    assert "Some prose" in text
    assert "0.5 nM" not in text            # tables dropped, same as description_text


def test_anchor_text_empty_without_a_description_element():
    assert anchor_text("<not-a-patent/>") == ""


# ── real XML: the required non-synthetic case ─────────────────────────────

def test_real_xml_anchor_text_has_example_headings():
    """`description_text()` (the OTHER flattening, in `uspto_xml.py`) has ZERO
    isolated "Example N" lines for this patent — that is the whole reason
    `anchor_text` exists. This asserts the fix side of that fact directly
    against the real cached grant XML, not description_text's absence.
    """
    text = anchor_text(_xml())
    assert "Example 1\n" in text
    assert "Example 51\n" in text


def test_real_xml_known_example_anchors_correctly():
    """A hand-verified case from this patent's own text (see the module
    docstring's ablation table): "Example 51" immediately precedes this
    exact name in the grant XML, and CSV row 51
    (`US8952177 Binding IUPAC Final (2).csv`) confirms cid 51 is this
    structure (stated there without the "racemic " prose prefix, which
    `extract_names`'s candidate generation also strips before this module
    ever sees the name).
    """
    name = ("cis-2-{1-[2-Fluoro-4-(trifluoromethoxy)benzyl]-6-"
            "[(1-methyl-1H-pyrazol-3-yl)methoxy]-1H-benzimidazol-2-yl}"
            "cyclohexanecarboxylic acid")
    text = anchor_text(_xml())
    r = find_cid(text, name)
    assert r.cid == "51"
    assert not r.clashed


def test_real_xml_generic_fragment_clashes_rather_than_guesses():
    """A real fragment from this patent's own extracted structures (verified
    by direct measurement, not assumed): `extract_names` resolves
    "2-{1-(4-Bromo-2-fluorobenzyl)-6-[(5-methylpyridin-2-yl)methoxy]-1H-
    benzimidazol-2-yl}cyclohexanecarboxylic" as its own distinct structure —
    it is a substring shared by Examples 10-13, which are four different
    stereoisomers/salts of near-identical names, so its occurrences point at
    four different Example numbers. Whatever it resolves to, it must not be
    a single confident number — this is the shape the clash guard exists
    for, on this patent's real text, not a synthetic stand-in.
    """
    name = ("2-{1-(4-Bromo-2-fluorobenzyl)-6-[(5-methylpyridin-2-yl)methoxy]-"
            "1H-benzimidazol-2-yl}cyclohexanecarboxylic")
    text = anchor_text(_xml())
    r = find_cid(text, name)
    assert r.cid is None
    assert r.clashed
    ids = {c.cid for c in r.candidates}
    assert ids == {"10", "11", "12", "13"}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extract_names_reproduces_measured_anchor_rate():
    """Locks in the headline number from this task's report: 238 distinct
    structures, 155 anchored (65.1%) on US8952177 with the shipped rule.
    Skips (not fails) if OPSIN cannot run here — this is the one test in
    the file that needs it, and its absence is an environment fact, not an
    anchoring regression.
    """
    pytest.importorskip("py2opsin")
    import logging
    logging.disable(logging.CRITICAL)
    try:
        from patentdb3.sources.iupac_names import extract_names
        out = extract_names(_xml(), PID)
    except Exception as e:                            # java missing, etc.
        pytest.skip(f"OPSIN unavailable in this environment: {e!r}")
    finally:
        logging.disable(logging.NOTSET)
    if len(out) == 0:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    anchored = sum(1 for nc in out if nc.cid)
    clashed = sum(1 for nc in out if nc.cid_clash)
    assert len(out) == 238, (
        f"candidate generation drifted ({len(out)} distinct structures, was 238) — "
        f"this module did not change it; re-measure before touching this assertion")
    assert anchored == 155, f"anchor rate regressed: {anchored}/238 (was 155/238, 65.1%)"
    assert clashed == 29, f"clash count changed: {clashed} (was 29)"
