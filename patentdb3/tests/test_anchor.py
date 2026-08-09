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
    """The colon shape anchors an alphanumeric id — on a CLAIMED compound."""
    text = "Example 1D: tert-butyl 5-(3-azetidinyl)pyrrolopyridine-1-carboxylate\n"
    r = find_cid(text, "5-(3-azetidinyl)pyrrolopyridine-1-carboxylate")
    assert r.cid == "1D"
    assert not r.clashed


def test_an_intermediate_id_is_refused_under_finished_only():
    """The same shape, labelled `Intermediate`, must NOT anchor.

    This test used to assert `cid == "1D"` on exactly this string — it was
    written before the deliverable's scope was settled. An intermediate is a
    real id for a real thing; it is simply not a molecule we ship, so
    `config.FINISHED_ONLY` refuses it. Measured cost of the whole rule over a
    fixed 20-patent sample: 6 joined compounds out of 2,705, against 159 fewer
    conflicting cids — intermediates are rarely assayed, so they were never
    going to join.
    """
    text = "Intermediate 1D: tert-butyl 5-(3-azetidinyl)pyrrolopyridine-1-carboxylate\n"
    r = find_cid(text, "5-(3-azetidinyl)pyrrolopyridine-1-carboxylate")
    assert r.cid is None
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


@pytest.mark.parametrize("label", ["Method 1", "Step 2", "Table 3"])
def test_blacklisted_words_before_colon_do_not_anchor(label):
    """Independently exercises three DIFFERENT alternatives of `_NON_ID_WORD`
    (`method|step|...|table|...`), not just the one the test above covers.

    `test_blacklisted_word_before_colon_is_not_a_compound_id` only asserts on
    "Method 1:" — proving the FIRST alternative of the blacklist regex works
    says nothing about the others; a future edit could narrow the alternation
    (e.g. drop `|step|` or `|table|`) and that single test would keep passing
    while "Step 2:" or "Table 3:" silently started anchoring again. The task
    brief names exactly these three labels for a reason: "Method 1:",
    "Step 2:" and "Table 3:" are the three shapes distinguished by name in
    this module's own docstring history, so each gets its own assertion here.
    """
    text = f"{label}: 4-methylpiperidin-1-yl acetic acid\n"
    r = find_cid(text, "4-methylpiperidin-1-yl acetic acid")
    assert r.cid is None
    assert not r.clashed
    assert r.candidates == ()


@pytest.mark.parametrize("label", ["Method", "Step", "Table", "Scheme"])
@pytest.mark.parametrize("sep", [" : ", " ; ", "  "])
def test_left_colon_blacklist_does_not_reach_the_plain_digit_rule(label, sep):
    """`_NON_ID_WORD` guards ONE of the two left rules. The other must refuse
    the same number independently — or the blacklist is decorative.

    `find_cid` runs `_CID_LEFT_PLAIN` and `_CID_LEFT_SEP` as two separate
    `finditer` passes over the SAME window, and the blacklist check lives
    inside the `_CID_LEFT_SEP` loop only (`continue`). So when the colon rule
    refuses "Method 1:" as a compound id, nothing in that code path stops the
    plain-digit rule from matching the very same "1" and anchoring to it.

    The tests above cannot see this: they all spell the label "Method 1:",
    digit immediately followed by a colon, and `_CID_LEFT_PLAIN` requires
    `(\\d+)\\s` — a digit run followed by WHITESPACE — so it never matches that
    spelling and the plain rule is never actually exercised. The spellings
    here are the ones where it IS: "Method 1 : X" and "Step 2 ; X" (space
    before the separator) both make `_CID_LEFT_PLAIN` match "1 "/"2 ", and
    "Method 1  X" has no separator at all, so `_CID_LEFT_SEP` — and with it
    the blacklist — never even applies. Verified directly against the shipped
    regexes before this test was written: in all three spellings
    `_CID_LEFT_PLAIN.finditer(window)` yields the number.

    What refuses it is `_LEFT_GAP_OK`: the gap left between the id and the
    name (": ", "; ", " ") is not the bare whitespace/stereo-prefix gap a real
    heading leaves. That coupling is the thing under test — widening
    `_LEFT_GAP_OK` to tolerate a stray separator would silently turn every
    "Step 2 : <name>" in the corpus into an anchor on id "2".

    Named for the citation in `test_iupac_reagent_markush.py`'s docstring,
    which referred to this test before it existed.
    """
    name = "4-methylpiperidin-1-yl acetic acid"
    r = find_cid(f"{label} 1{sep}{name}\n", name)
    assert r.cid is None, f"{label!r} + {sep!r} leaked through to the plain-digit rule"
    assert not r.clashed
    assert r.candidates == ()


@pytest.mark.parametrize("sep", [" : ", " ; "])
def test_a_non_blacklisted_label_still_anchors_in_the_same_spelling(sep):
    """The positive control for the test above — without it, that test would
    still pass if `find_cid` simply stopped anchoring anything with a space
    before the separator.

    "Example 5 : X" is the identical punctuation shape with a label that is
    NOT in `_NON_ID_WORD`, and it must still resolve to 5 via
    `_CID_LEFT_SEP` (which allows `\\s*[:;]\\s+` around the separator).
    """
    name = "4-methylpiperidin-1-yl acetic acid"
    r = find_cid(f"Example 5{sep}{name}\n", name)
    assert r.cid == "5"
    assert not r.clashed


def test_n2_sparged_no_longer_a_false_positive():
    """FIXED — this used to document a known, pre-existing limitation.

    "N2 sparged 2-methyltetrahydrofuran" (nitrogen gas, not a compound id)
    used to anchor the solvent name to id "2", purely because a bare digit
    run ("2" from "N2") sat within `_ANCHOR_BOUND` of the name. Two
    independent guards now both refuse this, for different reasons — see
    `anchor.py`'s "THE FALSE ANCHORS" section:

    1. `(?<![A-Za-z0-9])` on `_CID_LEFT_PLAIN` refuses to start a digit match
       glued to the "N" in front of it — "2" is the tail of "N2", not a bare
       id, the same reasoning that fixes the unrelated "Q38" collision on
       US10214537.
    2. `_LEFT_GAP_OK` refuses the match anyway even if (1) did not exist:
       the gap between the id and the name ("sparged ") is not a bare
       whitespace/stereo-prefix gap.

    Either guard alone is sufficient here; both are asserted together
    because this text exercises both.
    """
    text = "benzylamine (75.0 g), N2 sparged 2-methyltetrahydrofuran (750 mL)\n"
    r = find_cid(text, "2-methyltetrahydrofuran")
    assert r.cid is None
    assert not r.clashed
    assert r.candidates == ()


def test_digit_run_glued_to_a_letter_is_not_a_bare_id():
    """The "Q38" collision, isolated from the N2 case above and from OPSIN.

    Measured on US10214537: "Intermediate Q38" is a real, distinct heading
    for its own title compound. Before `(?<![A-Za-z0-9])`, bare `(\\d+)\\s`
    matched "38" out of "Q38" and anchored Q38's compound to unrelated plain
    id "38" — a dup-cid, not a clash, because nothing else in the window
    disagreed with the (wrong) match.
    """
    text = "Intermediate Q38\n1-(4-(3-Bromophenyl)-3-(2-fluorophenyl)piperazin-1-yl)ethanone\n"
    r = find_cid(text, "1-(4-(3-Bromophenyl)-3-(2-fluorophenyl)piperazin-1-yl)ethanone")
    assert r.cid is None          # "Q38" is alphanumeric; PLAIN must not see "38"
    assert not r.clashed
    assert r.candidates == ()


def test_citation_of_a_prior_example_does_not_anchor_a_different_reagent():
    """The dominant root cause behind the 43%/13-cid false-anchor finding.

    "prepared ... analogous to that in Example 1, substituting X" cites
    Example 1's METHOD while naming X, a reagent for THIS (later, unlabeled)
    example — not Example 1's own title compound. The old rule anchored X to
    "1" anyway because the digit-plus-whitespace shape and the short distance
    (here: "substituting ") look identical to a real heading's "Example
    1\\nracemic ...". `_LEFT_GAP_OK` rejects it: nothing but a bare
    stereo-prefix word is allowed between the id and the name.
    """
    text = (
        "Example 1\nracemic cis-2-{1-(4-Bromobenzyl)}cyclohexanecarboxylic acid\n"
        "The title compound was prepared in a manner analogous to that in "
        "Example 1 substituting trans-hexahydroisobenzofuran-1,3-dione in Step C.\n"
    )
    r = find_cid(text, "trans-hexahydroisobenzofuran-1,3-dione")
    assert r.cid is None
    assert not r.clashed
    assert r.candidates == ()
    # the real heading occurrence is untouched by the same fix
    r2 = find_cid(text, "cis-2-{1-(4-Bromobenzyl)}cyclohexanecarboxylic acid")
    assert r2.cid == "1"


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
        "Example 402B: 4-(3-Bromophenyl)morpholine-3-carboxamide\n"
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


# ── the dropped-opening-paren defect ──────────────────────────────────────
# The patent's OWN grant XML, not our flattening: a ring-substituent "("
# that should open right after "[" is sometimes simply missing, e.g. the
# real US8952177 Example 25 heading reads "...-6-[5-methylpyridin-2-
# yl)methoxy]-..." instead of "...-6-[(5-methylpyridin-2-yl)methoxy]-...".
# See `anchor.py`'s `_DROPPED_OPEN_PAREN` comment for the full measurement
# (7 US8952177 headings, 16 occurrences across 7 of 137 cached patents) and
# why the fix is scoped this narrowly.

def test_dropped_open_paren_is_repaired():
    text = anchor_text(
        "<description><heading>Example 25</heading>"
        "<heading>racemic cis-2,2-Dimethyl-3-{6-[5-methylpyridin-2-yl)methoxy]-"
        "1-[4-(trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}cyclopropanecarboxylic "
        "acid</heading></description>"
    )
    assert "[5-methylpyridin-2-yl)" not in text
    assert "[(5-methylpyridin-2-yl)methoxy]" in text


def test_well_formed_bracket_is_left_alone():
    """The repair must be a no-op on text that was never broken — most of the
    corpus's 424 well-formed occurrences of this exact shape on US8952177
    alone must not gain a second, doubled "((" from the repair.
    """
    text = anchor_text(
        "<description><heading>racemic cis-2-{1-(4-Bromobenzyl)-6-"
        "[(5-methylpyridin-2-yl)methoxy]-1H-benzimidazol-2-yl}cyclohexanecarboxylic "
        "acid</heading></description>"
    )
    assert "[((5-methylpyridin" not in text
    assert "[(5-methylpyridin-2-yl)methoxy]" in text


def test_repair_does_not_touch_the_other_unconfirmed_corruptions():
    """Two DIFFERENT, unconfirmed defects sit near the confirmed one in the
    same US8952177 document — measured while characterising this fix, see
    `_DROPPED_OPEN_PAREN`'s comment. Neither matches the confirmed shape
    (digit-led run with no nested brackets ending `-yl)`), and the repair
    must leave both alone: inventing a bracket for a shape never confirmed
    against the patent text is worse than leaving a name unanchored.
    """
    # "(" mistyped as the digit "1" — not a DROP, a substitution; no "-yl)"
    # tail either, so `_DROPPED_OPEN_PAREN` must not match it at all.
    typo = "2-Ethyl-2-({4-fluoro-6-[(5-methylpyridin-2-yl)methoxy]-1-[4-1-trifluoromethoxy)benzyl]-1H-benzimidazole}"
    text = anchor_text(f"<description><heading>{typo}</heading></description>")
    assert "[4-1-trifluoromethoxy)" in text          # left untouched

    # a dropped "(" that is NOT the first character after "[", and the run
    # ends in "methoxy)"/"benzyl]", not "-yl)" — a different defect shape.
    mid_drop = "racemic cis-2-{5-Fluoro-1-[2-fluoro-4-trifluoromethoxy)benzyl]-6-[(1-methylpyrazol-3-yl)methoxy]-acid"
    text2 = anchor_text(f"<description><heading>{mid_drop}</heading></description>")
    assert "[2-fluoro-4-trifluoromethoxy)benzyl]" in text2   # left untouched


# ── the occurrence cap: truncated evidence must not become confidence ─────
# `_MAX_OCCURRENCES` used to be 200 with a comment saying it had never been
# observed to bind. Measured over all 137 cached XMLs it bound on 55 names in
# 30 patents, and on US9394297 it turned a 31-way clash into a confident
# `cid="7"`. See that constant's comment in `anchor.py` for the full
# measurement; these lock the behaviour it now guarantees.


def test_truncated_scan_never_returns_a_confident_cid(monkeypatch):
    """The invariant, in isolation: a single agreeing id found under a
    truncated scan is NOT an anchor.

    The text below has three "Example 1\\nfoo-acid" pairs and then, past the
    cap, an "Example 2\\nfoo-acid" that disagrees. With the cap in force the
    scan never reaches the disagreement, so every occurrence it DID read
    agreed on "1" — which is exactly the reasoning that produced the wrong
    `cid="7"` on US9394297. The result must withhold the cid rather than
    report agreement it could not have checked.
    """
    monkeypatch.setattr("patentdb3.sources.anchor._MAX_OCCURRENCES", 3)
    text = ("Example 1\nfoo-acid x\n" * 3) + "Example 2\nfoo-acid y\n"
    r = find_cid(text, "foo-acid")
    assert r.truncated is True
    assert r.cid is None, "a truncated scan reported a confident id"
    # the evidence it did gather is still surfaced — degraded, never silent
    assert r.candidates and r.candidates[0].cid == "1"


def test_truncation_is_reported_even_when_nothing_was_found(monkeypatch):
    """`truncated` is orthogonal to the three result states, so the
    "nothing found" answer carries it too. US9656988's
    "pyrazine-2-carboxamide" is why this matters: at the old cap it returned
    `cid=None, clashed=False, candidates=()` — indistinguishable from a name
    with no id anywhere — while the complete scan finds a 222-way clash.
    """
    monkeypatch.setattr("patentdb3.sources.anchor._MAX_OCCURRENCES", 2)
    r = find_cid("qq nothing qq nothing qq nothing qq", "qq")
    assert r.truncated is True
    assert r.cid is None and not r.clashed and r.candidates == ()


def test_exactly_the_cap_is_not_truncation(monkeypatch):
    """Off-by-one guard. The previous implementation used a `while/else` whose
    `else` fired whenever `seen` reached the cap — including when the name
    occurred EXACTLY `_MAX_OCCURRENCES` times and nothing had been missed. It
    reported truncation, and (after this change would have) withheld a
    perfectly good cid, for a whole class of name that was fully scanned.
    """
    monkeypatch.setattr("patentdb3.sources.anchor._MAX_OCCURRENCES", 3)
    exactly = "Example 9\nzz qq zz qq zz"          # "zz" three times
    r = find_cid(exactly, "zz")
    assert r.truncated is False
    assert r.cid == "9", "a fully-scanned name lost its anchor to a false truncation"

    one_more = exactly + " qq zz"                   # "zz" four times
    assert find_cid(one_more, "zz").truncated is True


def test_untruncated_results_do_not_claim_truncation():
    """The default must stay False on the ordinary path — otherwise every
    caller sees a degraded result and the flag means nothing."""
    r = find_cid("Example 1\nracemic foo-acid\n", "foo-acid")
    assert r.cid == "1" and r.truncated is False
    assert find_cid("no ids at all here", "quinazoline").truncated is False


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


def test_real_xml_cap_clears_the_corpus_maximum():
    """The most-repeated OPSIN-resolved name in the whole 137-patent corpus.

    "1,3-dihydro-2H-isoindole-2-carboxamide" occurs 4,597 times in US9302989
    (counted the way `find_cid` steps: `text.find` with `start = pos + 1`,
    i.e. overlapping). `_MAX_OCCURRENCES` is set above it deliberately — this
    is the measurement that chose the value, so it is asserted rather than
    left in a comment to rot the way the previous one did.
    """
    from patentdb3.sources.anchor import _MAX_OCCURRENCES

    text = anchor_text(_xml("US9302989"))
    name = "1,3-dihydro-2H-isoindole-2-carboxamide"
    n, start = 0, 0
    while True:
        pos = text.find(name, start)
        if pos < 0:
            break
        n += 1
        start = pos + 1
    assert n == 4597, f"occurrence count drifted ({n}, was 4597) — re-measure the cap"
    assert _MAX_OCCURRENCES > n, "the cap now binds on the corpus maximum"
    assert find_cid(text, name).truncated is False


def test_real_xml_752_occurrence_fragment_clashes_rather_than_guessing():
    """THE defect the cap caused, on the real document that produced it.

    US9394297's "6,7-dihydro-1H-pyrrolo[3,2-c]pyridin-4(5H)-one" is a generic
    ring fragment occurring 752 times. At `_MAX_OCCURRENCES = 200` `find_cid`
    returned `cid="7"`, `clashed=False`, ONE candidate — a confident anchor on
    a fragment that 31 different compound ids compete for. The scan simply
    stopped before reaching the occurrences that disagreed.

    This is the module's founding rule ("CLASHES SURFACE, THEY DO NOT VANISH")
    being broken by a performance guard, and it is precisely the failure that
    ships a wrong structure under a real compound number.
    """
    text = anchor_text(_xml("US9394297"))
    r = find_cid(text, "6,7-dihydro-1H-pyrrolo[3,2-c]pyridin-4(5H)-one")
    assert r.truncated is False
    assert r.cid is None, "the 200-cap's fabricated confident anchor is back"
    assert r.clashed is True
    assert len(r.candidates) == 31, f"candidate count changed: {len(r.candidates)} (was 31)"


def test_real_xml_many_occurrence_fragment_reports_a_clash_not_nothing_found():
    """The other direction of the same defect, also on its own real document.

    US9656988's "pyrazine-2-carboxamide" occurs 729 times. At the 200 cap it
    returned `cid=None, clashed=False, candidates=()` — the "no occurrence of
    this name had any id within reach" state, which `AnchorResult`'s docstring
    defines as a DIFFERENT answer from a clash. The complete scan finds 222
    competing ids. A caller asking "should I expect this to ever resolve" was
    being given the wrong answer.
    """
    text = anchor_text(_xml("US9656988"))
    r = find_cid(text, "pyrazine-2-carboxamide")
    assert r.truncated is False
    assert r.cid is None
    assert r.clashed is True, "a 222-way clash is still being reported as 'nothing found'"
    assert len(r.candidates) == 222, f"candidate count changed: {len(r.candidates)} (was 222)"


def test_real_xml_conjunction_heading_with_parenthesised_ids_is_already_solved():
    """The conjunction shape the module docstring lists as unsolved — on the
    patent it cites as a real instance — is in fact already handled.

    US10214537's heading reads "Intermediates Q36-A and Q37-A:
    2,6-Dicyclopropylpiperazine (Q36-A) and 2-Cyclopropyl-6-isopropylpiperazine
    (Q37-A)". Each name carries its own id in TRAILING PARENS, which is
    `_CID_RIGHT`'s shape, so the second name reaches its id by the ordinary
    right-hand rule and needs no sibling context at all.

    The first name genuinely clashes — "2,6-Dicyclopropylpiperazine" has
    Q36-A adjacent on the right and Q37-A inside the heading prefix on the
    left — and that clash is the correct answer, not a miss. Asserted here so
    the docstring's corrected prevalence claim (see "WHAT IS STILL
    UNANCHORED") stays checked rather than quoted.
    """
    text = anchor_text(_xml("US10214537"))

    assert find_cid(text, "2-Cyclopropyl-6-isopropylpiperazine").cid == "Q37-A"
    assert find_cid(
        text, "1-(4-(3-Bromophenyl)-2,6-dicyclopropylpiperazin-1-yl)ethanone").cid == "Q36"
    assert find_cid(
        text,
        "1-(4-(3-Bromophenyl)-2-cyclopropyl-6-isopropylpiperazin-1-yl)ethanone").cid == "Q37"

    first = find_cid(text, "2,6-Dicyclopropylpiperazine")
    assert first.cid is None and first.clashed
    assert {c.cid for c in first.candidates} == {"Q36-A", "Q37-A"}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extract_names_reproduces_measured_anchor_rate():
    """Locks in the headline number from this task's report: with the false-
    anchor fix (`_LEFT_GAP_OK`, the `(?<![A-Za-z0-9])` digit-run guard —
    see `anchor.py`'s "THE FALSE ANCHORS" section) AND the dropped-open-paren
    repair (`anchor._repair_dropped_open_paren` — see that section of the
    docstring), US8952177 anchors 170 of 309 distinct structures, 1 clashed.

    309, not 308: the repair fixes 7 malformed headings (Examples 25, 162,
    163, 166, 171, 173, 186 — see `_DROPPED_OPEN_PAREN`'s comment for why this
    corrects an earlier, unverified description naming 174 instead of 166).
    One of those seven, Example 186, had NO correctly-bracketed restatement
    anywhere else in the document and so was previously not extracted as a
    structure at all — repairing the text is what makes OPSIN see a parseable
    name there for the first time, +1 on `len(out)`. This assertion is about
    CANDIDATE GENERATION, which this file does not otherwise own (it owns
    `anchor.py`) — if it drifts again for an unrelated reason, re-measure
    before assuming a regression here. Skips (not fails) if OPSIN cannot run
    in this environment — this is the one test in the file that needs it.
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
    # 328, not 324, since the heading route landed: `extract_names` now also
    # takes the whole post-id text of a compound-asserting heading as one
    # candidate (a heading declares its own name boundaries, so it needs no
    # seeding — see wiki 41 §4). Purely ADDITIVE on this patent: 0 structures
    # lost, and `anchored`/`clashed` below are unchanged at 183/1, which is
    # what makes this a re-measured number rather than a loosened assertion.
    assert len(out) == 328, (
        f"candidate generation drifted ({len(out)} distinct structures, was 328) — "
        f"re-measure before touching this assertion")
    # 170/309 = 55.0%. Was 165/308 (53.6%) before the dropped-open-paren
    # repair. +5, not +7: of the 7 repaired headings, 5 (Examples 25, 162,
    # 171, 173, 186) newly anchor to their own correct id. The other two,
    # 163 and 166, are each a "(1R*,2S*)" stereoisomer restatement that OPSIN
    # resolves to the SAME InChIKey as its own patent-numbered PRECEDING
    # "racemic cis-"/"trans-" sibling (162 and 165 respectively) — a real,
    # separate OPSIN/dedup collision (verified directly: both text spans give
    # `py2opsin` distinct, correctly-parsed SMILES for "(1R*,2S*)" vs the
    # flat "cis-" form when tested in isolation, but the earlier-positioned
    # "cis-" candidate wins `extract_names`'s position-ordered
    # dedup-by-InChIKey, so 163/166 never get their own entry). That
    # collision is orthogonal to the bracket defect this module fixes — it
    # existed before this fix too, just invisible, because 163/166's own
    # malformed headings produced no OPSIN candidate at all beforehand. Left
    # unresolved here: fixing it is a different, unconfirmed problem (how
    # `extract_names` should treat a "cis"/"trans"-only name that OPSIN
    # happens to resolve to one specific relative-stereo SMILES), out of this
    # task's scope of the dropped-paren defect.
    assert anchored == 183, f"anchor rate changed: {anchored}/328 (was 183/328, see anchor.py docstring)"
    # Still ONE clash — the dropped-paren repair does not touch it. The name
    # is the stereo-UNDEFINED parent — `2-{1-(4-Bromobenzyl)-...}cyclohexane-
    # carboxylic acid`, no descriptor — while Examples 1, 3, 5, 6 and 7 are its
    # stereoisomer variants, each heading stating the same skeleton with a
    # different descriptor. The flat name genuinely occurs beside all five, so
    # `cid` is None and all five candidates are surfaced for reconciliation.
    # Picking one would be a coin flip between five real compounds.
    assert clashed == 1, f"clash count changed: {clashed} (was 1)"
