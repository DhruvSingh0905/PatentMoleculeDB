"""Tests for `patentdb3.sources.name_repair`.

Five concerns, in this order:

1. **Pattern registry / candidate generation** — pure, no OPSIN, no I/O.
   Every `CorruptionPattern` fires on a REAL corrupted string taken verbatim
   from this task's cached corpus (`output_v3/uspto_xml/`), at the exact
   position quoted in the module's own docstring — not a constructed
   stand-in. Section 1b covers the three forms added after the corpus
   survey widened the confirmed surface to five, including the shared-site
   design behind Form C' / Form D.
2. **Confirmation gating** — the governing constraint from the task brief:
   an untrusted pattern must NOT confirm without corroboration, even when
   OPSIN alone would accept the candidate; an already-valid name must never
   be "repaired" into a different string; unrelated garbage must not
   falsely trigger a pattern; Form E (bracket-type substitution) must NEVER
   be repaired, asserted directly against real corpus text, not just
   argued from the short-circuit; `digit_for_paren`'s site regex must not
   have been silently widened past its confirmed two-instance scope.
3. **The 8 known-lost cids** — `repair_name`, called directly on each cid's
   own corrupted heading text (read from the raw XML, not from any
   already-repaired flattening), reproduces the correct structure for 7 of
   them; cid 174 is verified separately NOT to be a text-corruption case at
   all, so `repair_name` correctly returns `None` for it.
4. **Determinism** — `generate_candidates` (pure) proven stable across 50
   in-process repeats and two fresh subprocess interpreters, the same
   method `sources/reagents.py` uses; `repair_name` (OPSIN-touching)
   proven stable across two fresh subprocess interpreters on a small
   battery, skipped where OPSIN cannot run.

5. **The batched form** — `repair_names`, added when the module was wired
   into `table_names.py` and `iupac_names.py`, must be exactly `repair_name`
   over a list: same gates, same registry order, same first-confirmed-wins,
   one entry per input. `repair_name` is now literally
   `repair_names([name], document_text)[0]`, so §§1-4 above already exercise
   the batched path; §5 pins the list-shape contract the two callers depend
   on (positional pairing, `None` for a name that confirms nothing) and
   proves batching does not change a verdict.

`name_repair` is called by `table_names.extract_table_names` (on every
name-column cell no dewrap variant resolves) and by
`iupac_names.extract_names` (on every description seed no variant resolves).
Neither caller is tested here — this file tests the module standalone; the
wiring's own regression cases live in `test_table_names.py` and
`test_structures_wiring.py`.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys

import pytest

from patentdb3.core import config
from patentdb3.sources.name_repair import (
    PATTERN_IDS,
    PATTERNS,
    RepairCandidate,
    RepairResult,
    generate_candidates,
    repair_name,
    repair_names,
)

PID = "US8952177"
CSV_PATH = "/Users/dhruvsingh/Downloads/US8952177 Binding IUPAC Final (2).csv"


def _xml(pid: str = PID) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="ignore")


# ---------------------------------------------------------------------------
# 1. Pattern registry / candidate generation — pure, no OPSIN.
# ---------------------------------------------------------------------------

def test_pattern_ids_are_exactly_the_five_confirmed_forms():
    """FIVE confirmed corruption forms, SIX registry entries: Form C'
    (`stray_opening_bracket`) and Form D (`dropped_close_bracket`) share one
    detector and are two entries for one form — see the module docstring's
    "C' AND D SHARE ONE SITE". Form E ("]" -> "}") is confirmed but
    deliberately has NO entry — see `test_bracket_type_substitution_is_
    never_repaired` below.
    """
    assert PATTERN_IDS == (
        "dropped_open_paren", "digit_for_paren", "ch_as_ey",
        "stray_closing_bracket", "stray_opening_bracket", "dropped_close_bracket",
    )
    assert len(PATTERNS) == len(set(PATTERN_IDS))          # no duplicate ids
    for pat in PATTERNS:
        assert callable(pat.site)
        assert callable(pat.fix)
        assert isinstance(pat.trusted, bool)
        assert isinstance(pat.description, str) and pat.description


def test_dropped_open_paren_fires_on_real_corpus_string():
    """US8952177 Example 25's own heading, verbatim (see anchor.py's
    docstring and this module's own PATTERNS entry for the same citation).
    """
    name = ("cis-2,2-Dimethyl-3-{6-[5-methylpyridin-2-yl)methoxy]-1-[4-"
            "(trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}"
            "cyclopropanecarboxylic acid")
    cands = generate_candidates(name)
    hit = [c for c in cands if c.pattern_id == "dropped_open_paren"]
    assert len(hit) == 1
    assert hit[0].repaired == (
        "cis-2,2-Dimethyl-3-{6-[(5-methylpyridin-2-yl)methoxy]-1-[4-"
        "(trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}"
        "cyclopropanecarboxylic acid")


def test_digit_for_paren_fires_on_real_corpus_string():
    """US8952177 Example 92's own heading, verbatim."""
    name = ("2-Ethyl-2-({6-[(1-methyl-1H-pyrazol-3-yl)methoxy]-1-[2-methyl-"
            "4-1-trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}methyl)"
            "butanoic acid")
    cands = generate_candidates(name)
    hit = [c for c in cands if c.pattern_id == "digit_for_paren"]
    assert len(hit) == 1
    assert hit[0].repaired == (
        "2-Ethyl-2-({6-[(1-methyl-1H-pyrazol-3-yl)methoxy]-1-[2-methyl-"
        "4-(trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}methyl)"
        "butanoic acid")


def test_ch_as_ey_fires_on_real_corpus_string():
    """US10376513's compound #69 table-row name, verbatim (space after the
    line-wrap included, exactly as it sits in the patent's own XML cell).
    """
    name = ("(2R)-1-(3-{3-[1-(4-Amino-3-methyl-1H-pyrazolo[3,4-d]pyrimidin-"
            "1-yl)ethyl]-5-eyloro-6- fluoro-2-methoxyphenyl}azetidin-1-yl)"
            "propan-2-ol")
    cands = generate_candidates(name)
    hit = [c for c in cands if c.pattern_id == "ch_as_ey"]
    assert len(hit) == 1
    assert "5-chloro-6- fluoro-2-methoxyphenyl" in hit[0].repaired


def test_generate_candidates_returns_nothing_for_clean_unrelated_text():
    assert generate_candidates("benzene") == []
    assert generate_candidates("4-fluoropiperidine") == []


def test_multiple_occurrences_each_produce_their_own_candidate():
    """A name with TWO dropped-paren sites (constructed, but the same shape
    as US8952177's real multi-occurrence Examples) yields two candidates,
    left-to-right.
    """
    name = "a-{6-[5-methylpyridin-2-yl)methoxy]}-b-{6-[5-methylpyridin-2-yl)methoxy]}"
    cands = [c for c in generate_candidates(name) if c.pattern_id == "dropped_open_paren"]
    assert len(cands) == 2
    assert cands[0].span[0] < cands[1].span[0]


# ---------------------------------------------------------------------------
# 1b. The two newest forms sharing one site (C' / D), and the stray-bracket
#     forms (C, C') — real corpus strings, verified by a direct before/after
#     OPSIN test at investigation time (reproduced in section 2 below).
# ---------------------------------------------------------------------------

def test_stray_closing_bracket_fires_on_real_corpus_string():
    """US10266548's own heading, verbatim: a stray ')' with no opener,
    immediately after the fused-ring linking prefix 'pyrimidino'.
    """
    name = ("tert-butyl 3-[5-methoxy-4-(pyridin-4-ylamino)pyrimidino)"
            "pyrimidin-2-yl]-1H-indazole-1-carboxylate")
    cands = [c for c in generate_candidates(name)
             if c.pattern_id == "stray_closing_bracket"]
    assert len(cands) == 1
    assert "pyrimidino)pyrimidin" not in cands[0].repaired
    assert "pyrimidinopyrimidin" in cands[0].repaired


def test_stray_closing_bracket_fires_on_second_independent_corpus_string():
    """US11427578's own heading fragment, verbatim (including the opening
    '{' the stray ']' sits inside — omitting it, as an earlier version of
    this test did, leaves '}' ALSO unmatched and changes what is being
    tested): a stray ']' with no opener (mismatched TYPE against the
    enclosing '{'), independent of the US10266548 case above — proves the
    bracket-stack scanner generalises rather than being tuned to one string.
    """
    name = ("4-{5-(hydroxymethyl)-5-methyl-5,6-dihydro-4H-1,3-oxazin-2-yl]"
            "amino}phenoxy")
    cands = [c for c in generate_candidates(name)
             if c.pattern_id == "stray_closing_bracket"]
    assert len(cands) == 1
    assert cands[0].repaired == (
        "4-{5-(hydroxymethyl)-5-methyl-5,6-dihydro-4H-1,3-oxazin-2-ylamino}phenoxy")


def test_unmatched_opener_site_feeds_both_stray_opening_and_dropped_close():
    """The shared-site design itself: one unmatched '(' produces candidates
    from BOTH `stray_opening_bracket` (delete it) AND `dropped_close_bracket`
    (insert ')' somewhere after it) — never just one.
    """
    name = "N-[(1-(oxetan-3-yl)ethyl]urea"     # US11427578's own fragment
    cands = generate_candidates(name)
    ids = {c.pattern_id for c in cands}
    assert "stray_opening_bracket" in ids
    assert "dropped_close_bracket" in ids
    # the dropped_close_bracket candidate that actually closes the '(' before
    # 'ethyl' is somewhere in the fan-out (order not asserted — repair_name,
    # not generate_candidates, is what picks a winner)
    fixed = {c.repaired for c in cands if c.pattern_id == "dropped_close_bracket"}
    assert "N-[(1-(oxetan-3-yl)ethyl)]urea" in fixed


def test_dropped_close_bracket_fires_on_real_corpus_string():
    """US9303033's own heading, verbatim: 'triazin-4(3H-one' for
    'triazin-4(3H)-one' — no delimiter at the drop point, the exact shape
    the module docstring's 'C\\' AND D SHARE ONE SITE' section describes.
    """
    name = "2-(methylthio)pyrazolo[1,5-a][1,3,5]triazin-4(3H-one"
    fixed = {c.repaired for c in generate_candidates(name)
             if c.pattern_id == "dropped_close_bracket"}
    assert "2-(methylthio)pyrazolo[1,5-a][1,3,5]triazin-4(3H)-one" in fixed


def test_dropped_close_bracket_fires_on_no_delimiter_corpus_string():
    """US10035778's own heading, verbatim: 'amino' runs directly into
    'benzoic' with no hyphen — the exact 'no delimiter' shape named in the
    module docstring (97 of 99 US10035778 instances were missed by a
    hyphen-anchored search for this reason).
    """
    name = ("3-Hydroxy-5-((5-hydroxy-1,4,5,6-tetrahydropyrimidin-2-yl)"
            "aminobenzoic acid")
    fixed = {c.repaired for c in generate_candidates(name)
             if c.pattern_id == "dropped_close_bracket"}
    assert ("3-Hydroxy-5-((5-hydroxy-1,4,5,6-tetrahydropyrimidin-2-yl)"
            "amino)benzoic acid") in fixed


def test_digit_for_paren_site_regex_does_not_widen_past_confirmed_scope():
    """Corpus-wide re-measurement of the exact claim in `PATTERNS`'s
    `digit_for_paren` entry: the site regex fires exactly twice across all
    137 cached patents, both in US8952177 (cids 92 and 105) — never widened
    by an automated plausibility screen, per the corpus survey's explicit
    warning that such a screen produces false repairs elsewhere by deleting
    real ring-locant digits.
    """
    from patentdb3.sources.name_repair import _PATTERNS_BY_ID
    site = _PATTERNS_BY_ID["digit_for_paren"].site
    hits = []
    for path in sorted(config.XML_INPUT_DIR.glob("*.xml")):
        raw = path.read_text(errors="ignore")
        for span in site(raw):
            hits.append((path.stem, span))
    assert len(hits) == 2, f"digit_for_paren site regex fired {len(hits)} times, expected 2"
    assert {pid for pid, _ in hits} == {"US8952177"}


# ---------------------------------------------------------------------------
# 2. Confirmation gating — the governing constraint.
# ---------------------------------------------------------------------------

_OPSIN_OR_SKIP = pytest.importorskip("py2opsin")

# `py2opsin`'s DEFAULT `tmp_fpath` is a fixed relative filename shared by
# every caller that does not pass its own — the same hazard
# `name_repair._opsin_batch` avoids with a pid-scoped path. A handful of
# tests below call `py2opsin` directly (to resolve a REFERENCE name, not to
# exercise `name_repair` itself) and must not share that fixed name either;
# `_call_counter` makes each such call's tmp file unique within this
# process, same spirit as the shipped module's own pid-scoping.
import itertools
_call_counter = itertools.count()


def _opsin(name: str, fmt: str) -> str:
    tmp = str(config.OUTPUT_DIR / f".test_name_repair_opsin_{next(_call_counter)}.txt")
    return _OPSIN_OR_SKIP.py2opsin(name, output_format=fmt, tmp_fpath=tmp)


def _repair(name: str, document_text: str = "") -> RepairResult | None:
    import logging
    logging.disable(logging.CRITICAL)
    try:
        return repair_name(name, document_text)
    finally:
        logging.disable(logging.NOTSET)


CORRUPTED_92 = ("2-Ethyl-2-({6-[(1-methyl-1H-pyrazol-3-yl)methoxy]-1-[2-methyl-"
                "4-1-trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}methyl)"
                "butanoic acid")
REPAIRED_92 = ("2-Ethyl-2-({6-[(1-methyl-1H-pyrazol-3-yl)methoxy]-1-[2-methyl-"
               "4-(trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}methyl)"
               "butanoic acid")


def test_untrusted_pattern_does_not_confirm_without_document_text():
    """`digit_for_paren` is `trusted=False`: OPSIN alone must not be enough.
    Confirmed by omitting `document_text` entirely on a real corpus string
    whose repaired form DOES parse (proven by the case above / the corpus
    recovery test below) — the only thing missing here is corroboration.
    """
    assert _repair(CORRUPTED_92, "") is None


def test_untrusted_pattern_does_not_confirm_with_unrelated_document_text():
    unrelated = ("This document discusses an entirely different subject "
                 "with no chemical fragment in common with the candidate "
                 "repair at all, padded to a reasonable length. " * 3)
    assert _repair(CORRUPTED_92, unrelated) is None


def test_untrusted_pattern_confirms_with_genuine_corroboration():
    """The corroborating fragment need not be the whole document — a
    document containing just the repaired fragment restated once,
    elsewhere, is enough. This isolates the corroboration MECHANISM from
    needing the full cached XML (that full-XML case is covered by the
    corpus recovery test below). The probe window straddles the fix with
    12 chars of context each side (`_CORR_WINDOW`), so the corroborating
    text needs the SAME immediate flanking characters as the repaired
    name — here, a '-' immediately before '1-[2-methyl...', the same shape
    US8952177's own Example 105 restatement has.
    """
    document = ("Example 105\n2-Ethyl-2-({6-[(5-methylpyridin-2-yl)methoxy]"
                "-1-[2-methyl-4-(trifluoromethoxy)benzyl]-1H-benzimidazol-2-"
                "yl}methyl)butanoic acid was prepared as described.")
    r = _repair(CORRUPTED_92, document)
    assert r is not None
    assert r.pattern_id == "digit_for_paren"
    assert r.confirmation == "corroborated"
    assert r.repaired == REPAIRED_92
    assert r.smiles
    assert r.inchikey


def test_already_valid_name_is_never_touched():
    """A name that already parses is not "repaired" into a different valid
    string, even though its text superficially matches a site pattern
    nowhere in this specific name — this is the general safety net, not a
    per-pattern check: see module docstring, "WHAT THIS DOES NOT DO".
    """
    already_good = ("cis-2-{1-(4-Bromobenzyl)-6-[(5-methylpyridin-2-yl)"
                     "methoxy]-1H-benzimidazol-2-yl}cyclohexanecarboxylic acid")
    assert _repair(already_good, "irrelevant document text") is None


def test_unrelated_garbage_does_not_falsely_trigger_a_pattern():
    assert _repair("this-is-not-a-chemical-name-at-all-xyz123", "") is None


def test_bracket_type_substitution_is_never_repaired():
    """Form E (']' mistyped as '}') is CONFIRMED but deliberately has no
    `CorruptionPattern` — see the module docstring's "FORM E: WHY IT HAS NO
    PATTERN". US10172859's own text, verbatim:
    '...thieno[3,2-d}pyrimidin-4-yl...' for '...thieno[3,2-d]pyrimidin-4-
    yl...' — OPSIN parses the AS-IS '}' form to the identical SMILES as the
    ']' form (verified directly: both resolve to
    'ClC=1C(=NC=CN1)C(O)C1=CC(=C(C=C1)F)C=1C2=C(N=CN1)C=C(S2)N2CCOCC2'), so
    `repair_name`'s own short-circuit — the original already parses — means
    this never reaches pattern-matching at all. Asserted against the real
    corpus text, not the short-circuit argument alone.
    """
    name = ("(3-Chloropyrazin-2-yl)-[4-fluoro-3-(6-morpholin-4-ylthieno"
            "[3,2-d}pyrimidin-4-yl)phenyl]-methanol")
    xml_path = config.XML_INPUT_DIR / "US10172859.xml"
    if not xml_path.exists():
        pytest.skip("US10172859.xml not cached")
    document = xml_path.read_text(errors="ignore")
    assert _repair(name, document) is None
    # and directly: OPSIN accepts the '}' form exactly as it sits in the
    # patent's own text, which is WHY the short-circuit fires.
    smi = _opsin(name, "SMILES")
    assert smi, "expected OPSIN to already accept the '}' form as-is"


def test_sulfaneylidene_is_not_a_false_positive_for_ch_as_ey():
    """`ch_as_ey`'s bare site regex (any 'ey' followed by 2+ lowercase
    letters) fires on real, UNCORRUPTED corpus text too — measured
    corpus-wide at 218 site matches total, the large majority not
    corruptions. '16-sulfaneylidene' (US11420968, a real, if
    OPSIN-unfriendly, sulfoximine locant morpheme repeated throughout that
    patent) is the concrete case: substituting 'ch' for its 'ey' produces
    'sulfanchlidene', which OPSIN rejects outright — verified directly
    against the corpus's own text, not a synthetic stand-in. Zero of the
    218 corpus-wide site matches produce a confirmed repair other than the
    one genuine case (`ch_as_ey`'s own PATTERNS citation).
    """
    name = ("N-(4-((dimethyl(oxo)-16-sulfaneylidene)amino)phenyl)-4-(4-"
            "chlorophenoxy)benzamide")
    xml_path = config.XML_INPUT_DIR / "US11420968.xml"
    if not xml_path.exists():
        pytest.skip("US11420968.xml not cached")
    document = xml_path.read_text(errors="ignore")
    assert _repair(name, document) is None


# ---------------------------------------------------------------------------
# 3. The 8 known-lost cids — recovery against the real cached corpus.
# ---------------------------------------------------------------------------

# Each cid's OWN corrupted heading text, read directly from
# `output_v3/uspto_xml/US8952177.xml` at the position named in this task's
# report (not reconstructed from the CSV, which carries the CORRECT name).
_KNOWN_LOST = {
    "25": ("cis-2,2-Dimethyl-3-{6-[5-methylpyridin-2-yl)methoxy]-1-[4-"
           "(trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}"
           "cyclopropanecarboxylic acid"),
    "92": CORRUPTED_92,
    "162": ("cis-2-{1-[4-(3,3-Difluoropyrrolidin-1-yl)-2-fluorobenzyl]-6-"
            "[5-methylpyridin-2-yl)methoxy]1H-benzimidazol-2-yl}"
            "cyclohexanecarboxylic acid"),
    "163": ("(1R*,2S*)-2-{1-[4-(3,3-Difluoropyrrolidin-1-yl)-2-fluorobenzyl"
            "]-6-[5-methylpyridin-2-yl)methoxy]-1H-benzimidazol-2-yl}"
            "cyclohexanecarboxylic acid"),
    "171": ("trans-2-{1-[4-(3,3-Difluoroazetidin-1-yl)-2-fluorobenzyl]-6-"
            "[5-methylpyridin-2-yl)methoxy]-1H-benzimidazol-2-yl}"
            "cyclohexanecarboxylic acid"),
    "173": ("cis-2-{1-[2-Fluoro-4-(4-fluoropiperidin-1-yl)benzyl]-6-[3-"
            "fluoro-5-methylpyridin-2-yl)methoxy]-1H-benzimidazol-2-yl}"
            "cyclohexanecarboxylic acid"),
    "174": ("(1R*,2S*)-2-{1-[2-Fluoro-4-(4-fluoropiperidin-1-yl)benzyl]-6-"
            "[(3-fluoro-5-methylpyridin-2-yl)methoxy]-1H-benzimidazol-2-yl}"
            "cyclohexanecarboxylic acid"),
    "186": ("3-{1-[2-Fluoro-4-(4-fluoropiperidin-1-yl)benzyl]-6-[3-fluoro-5-"
            "methylpyridin-2-yl)methoxy]-1H-benzimidazol-2-yl}-2,2-"
            "dimethylpropanoic acid"),
}


def _reference_names() -> dict[str, str]:
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            return {row["compound_id"]: row["iupac_name"]
                    for row in csv.DictReader(f)}
    except OSError:
        pytest.skip(f"ground-truth CSV not found: {CSV_PATH}")


def test_seven_of_eight_known_lost_cids_recover_the_correct_structure():
    """The task's own reference list. `document_text` is the PLAIN
    heading-inclusive flattening (`description_text(..., include_headings=
    True)`) — deliberately NOT `anchor.py::anchor_text`, so this proves
    `name_repair` recovers these on its own evidence, independent of
    `anchor.py`'s own (different-module) dropped-paren fix.

    cid 174 is the one EXCLUSION: its own heading text is not corrupted at
    all (asserted separately below) — its loss is a downstream dedup
    collision in `iupac_names.py::extract_names`, out of this module's
    scope. `repair_name` correctly declines to touch it.
    """
    from patentdb3.sources.uspto_xml import description_text
    text = description_text(_xml(), include_headings=True)
    refs = _reference_names()

    expected_recovered = {"25", "92", "162", "163", "171", "173", "186"}
    recovered = set()
    for cid, corrupted in _KNOWN_LOST.items():
        r = _repair(corrupted, text)
        if r is None:
            continue
        recovered.add(cid)
        ref_clean = re.sub(r"^racemic\s+", "", refs[cid], flags=re.I)
        ref_clean = re.sub(r"\s+as the (TFA|HCl|formate) salt$", "",
                            ref_clean, flags=re.I)
        ref_ik = _opsin(ref_clean, "StdInChIKey")
        if not ref_ik:
            pytest.skip(f"reference name for cid {cid} did not parse — "
                        f"cannot check correctness")
        assert r.inchikey == ref_ik, (
            f"cid {cid}: repaired to a DIFFERENT structure than the "
            f"reference ({r.inchikey} vs {ref_ik})")

    assert recovered == expected_recovered, (
        f"recovered {recovered}, expected {expected_recovered}")


def test_cid_174_is_not_a_text_corruption_case():
    """Verifies the exclusion above rather than assuming it: cid 174's own
    heading text parses via OPSIN AS-IS, so `repair_name` returning `None`
    for it is correct behaviour (nothing to repair), not a missed recovery.
    """
    smi = _opsin(_KNOWN_LOST["174"], "SMILES")
    assert smi, "cid 174's heading was expected to already parse cleanly"
    assert _repair(_KNOWN_LOST["174"], "") is None


# ---------------------------------------------------------------------------
# 4. Determinism.
# ---------------------------------------------------------------------------

_CANDIDATE_BATTERY = [
    ("cis-2,2-Dimethyl-3-{6-[5-methylpyridin-2-yl)methoxy]-1-[4-"
     "(trifluoromethoxy)benzyl]-1H-benzimidazol-2-yl}cyclopropanecarboxylic acid"),
    CORRUPTED_92,
    ("(2R)-1-(3-{3-[1-(4-Amino-3-methyl-1H-pyrazolo[3,4-d]pyrimidin-1-yl)"
     "ethyl]-5-eyloro-6- fluoro-2-methoxyphenyl}azetidin-1-yl)propan-2-ol"),
    "benzene",
    "this-is-not-a-chemical-name-at-all-xyz123",
]


def _candidate_battery_result() -> list[list]:
    return [[[c.pattern_id, c.repaired, list(c.span)] for c in generate_candidates(n)]
            for n in _CANDIDATE_BATTERY]


def test_generate_candidates_is_deterministic_across_repeats():
    first = _candidate_battery_result()
    for _ in range(50):
        assert _candidate_battery_result() == first


def test_generate_candidates_is_deterministic_across_fresh_interpreters():
    """A brand-new `python3` process must produce the byte-identical result —
    rules out set/dict iteration-order leakage or module-import-order
    effects, the same proof `sources/reagents.py` uses.
    """
    in_process = _candidate_battery_result()

    script = (
        "import json\n"
        "from patentdb3.sources.name_repair import generate_candidates\n"
        f"battery = {_CANDIDATE_BATTERY!r}\n"
        "out = [[[c.pattern_id, c.repaired, list(c.span)] "
        "for c in generate_candidates(n)] for n in battery]\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(config.REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == in_process


def test_repair_name_is_deterministic_across_fresh_interpreters():
    """Extends the determinism proof through the OPSIN-touching half, on a
    small battery (OPSIN calls are not free time-wise even though they cost
    no money) — two fresh interpreters must agree with each other.
    """
    script = (
        "import json, logging\n"
        "logging.disable(logging.CRITICAL)\n"
        "from patentdb3.sources.name_repair import repair_name\n"
        f"name = {CORRUPTED_92!r}\n"
        "document = ('Example 105\\n2-Ethyl-2-({6-[(5-methylpyridin-2-yl)methoxy]'\n"
        "            '-1-[2-methyl-4-(trifluoromethoxy)benzyl]-1H-benzimidazol-2-'\n"
        "            'yl}methyl)butanoic acid was prepared as described.')\n"
        "r = repair_name(name, document)\n"
        "out = [r.repaired, r.smiles, r.inchikey, r.pattern_id, r.confirmation] if r else None\n"
        "print(json.dumps(out))\n"
    )
    results = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(config.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        results.append(json.loads(proc.stdout))
    assert results[0] == results[1]
    assert results[0] is not None
    assert results[0][3] == "digit_for_paren"


# ── 5. the batched form ──────────────────────────────────────────────────
#
# `repair_names` exists because both wired callers hand this module thousands
# of names per patent and OPSIN is a java subprocess: `repair_name` pays one
# process launch per name (two, on a win). It must be the same function over
# a list — nothing about the gates, their order, or their verdicts may change
# just because a name arrived with company.

def test_repair_names_returns_one_entry_per_input_in_order():
    """Positional pairing is the whole contract the callers rely on — the
    same contract `sources/opsin.batch` refuses a batch rather than break.
    Pure: none of these three names generates a candidate that could reach
    OPSIN with an empty document, so the list shape is provable without it."""
    out = repair_names(["", "not a name at all", "benzene"])
    assert len(out) == 3
    assert all(r is None for r in out)


def test_repair_names_empty_list_is_empty_not_an_error():
    assert repair_names([]) == []


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_repair_names_agrees_with_repair_name_one_by_one():
    """The equivalence the docstring claims, checked rather than asserted:
    a battery mixing a confirmed repair, an already-valid name (which must
    short-circuit to `None`), and a name nothing fires on, run BOTH ways.

    Batching changes which OPSIN process sees which string. If any gate ever
    read state across a batch — a shared `seen` set, a document normalised
    per call, an index reused — this is the test that would catch it.
    """
    pytest.importorskip("py2opsin")
    document = ("Example 105\n2-Ethyl-2-({6-[(5-methylpyridin-2-yl)methoxy]"
                "-1-[2-methyl-4-(trifluoromethoxy)benzyl]-1H-benzimidazol-2-"
                "yl}methyl)butanoic acid was prepared as described.")
    battery = [
        CORRUPTED_92,                       # confirmed `digit_for_paren`
        "benzene",                          # already valid -> None
        "4-(4-chlorophenyl)-1-methylpiperidine",   # already valid -> None
        "wholly unparseable text",          # no site fires -> None
        CORRUPTED_92,                       # the same name twice, same verdict
    ]
    try:
        one_by_one = [repair_name(n, document) for n in battery]
        batched = repair_names(battery, document)
    except Exception as e:                            # java missing, etc.
        pytest.skip(f"OPSIN unavailable in this environment: {e!r}")
    if all(r is None for r in one_by_one):
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert batched == one_by_one
    assert batched[0] is not None and batched[0].pattern_id == "digit_for_paren"
    assert batched[0] == batched[4]
    assert batched[1] is batched[2] is batched[3] is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_batching_does_not_leak_one_names_corroboration_into_another():
    """`document_text` is normalised ONCE for the whole batch. That is a real
    behaviour change from per-name normalisation and it must not widen what
    corroborates: a name whose probe appears nowhere in the document must
    still fail the gate when it travels in a batch beside one whose probe
    does.
    """
    pytest.importorskip("py2opsin")
    document = ("Example 105\n2-Ethyl-2-({6-[(5-methylpyridin-2-yl)methoxy]"
                "-1-[2-methyl-4-(trifluoromethoxy)benzyl]-1H-benzimidazol-2-"
                "yl}methyl)butanoic acid was prepared as described.")
    # A bracket-imbalanced name whose corrected form is NOT in `document`.
    stranger = "[4-(4-chlorophenyl)-1-methylpiperidine"
    try:
        alone = repair_name(stranger, document)
        together = repair_names([CORRUPTED_92, stranger], document)
    except Exception as e:
        pytest.skip(f"OPSIN unavailable in this environment: {e!r}")
    assert alone is None, alone
    assert together[1] is None, together[1]
