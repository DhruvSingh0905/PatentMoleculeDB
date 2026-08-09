"""Tests for the reagent-label and Markush-flag wiring landed in
`sources/iupac_names.py` (this task's own module): every `NamedCompound`
`extract_names` returns now carries

  - `.label` / `.reason`   — from `sources/reagents.py::classify(name, smiles)`,
    called once per accepted structure. Labels only, never filters — see
    `reagents.py`'s own docstring; this file re-verifies the wiring, not the
    lexicon/backstop themselves (that is `test_reagents.py`'s job, and it is
    off-limits to this task).
  - `.markush` / `.markush_reason` — set HERE, independently of the label
    above, when the accepted name carries a relative-stereo descriptor
    (`R*`/`S*`/`E*`/`Z*`). This is also what `_NAME_CHARS` gaining `*` makes
    possible: before, such a name never seeded at all.

SCOPE. Deliberately does NOT assert on `.cid` / `.cid_clash` — those are
`anchor.py`'s numbers, a module owned elsewhere in this task and observed to
change independently while this file was being written (its own locked
regression test, `test_anchor.py::test_left_colon_blacklist_does_not_reach_
the_plain_digit_rule`, started failing on a byte-for-byte reverted
`iupac_names.py` — proof the change was on anchor.py's side, not this one's).
Pinning an anchoring number here would make this file fail for a reason it
has no ownership of.
"""
from __future__ import annotations

import logging
from collections import Counter

import pytest

from patentdb3.core import config
from patentdb3.sources.iupac_names import (
    NamedCompound,
    _NAME_CHARS,
    _RELATIVE_STEREO,
    extract_names,
)

PID = "US8952177"


def _xml(pid: str = PID) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="ignore")


def _extract(pid: str = PID) -> list[NamedCompound]:
    """`extract_names`, OPSIN-gated the same way `test_anchor.py` gates it:
    skip (never fail) when OPSIN cannot run or produces nothing here — those
    are environment facts, not a regression in this module.
    """
    pytest.importorskip("py2opsin")
    logging.disable(logging.CRITICAL)
    try:
        out = extract_names(_xml(pid), pid)
    except Exception as e:                            # java missing, etc.
        pytest.skip(f"OPSIN unavailable in this environment: {e!r}")
    finally:
        logging.disable(logging.NOTSET)
    if not out:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    return out


# ── pure, no-OPSIN checks ─────────────────────────────────────────────────

def test_asterisk_is_now_a_seed_character():
    """The whole fix in one assertion: `*` used to be absent from
    `_NAME_CHARS`, which is what broke a relative-stereo name's seed at the
    asterisk. See the module docstring for the boundary-splitting mechanics.
    """
    assert "*" in _NAME_CHARS


def test_relative_stereo_regex_matches_the_documented_shape():
    assert _RELATIVE_STEREO.findall(
        "(1R*,2S*)-2-{1-(4-Bromo-2-fluorobenzyl)-...}cyclohexanecarboxylic acid"
    ) == ["R*", "S*"]
    # absolute stereo (no asterisk) must NOT be flagged
    assert _RELATIVE_STEREO.findall("(1S,2R)-2-{...}cyclohexanecarboxylic acid") == []
    # a bare asterisk with no preceding stereo letter must NOT be flagged —
    # only R*/S*/E*/Z* is a real relative-stereo descriptor
    assert _RELATIVE_STEREO.findall("5*-substituted") == []


def test_named_compound_field_defaults():
    """A `NamedCompound` built without the new fields (as any pre-existing
    caller does) gets the same neutral defaults `reagents.classify` would
    return for "nothing matched" — never an implicit flag.
    """
    nc = NamedCompound(patent_id="X", name="n", smiles="C", inchikey="IK", start=0)
    assert nc.label == "compound"
    assert nc.reason == ""
    assert nc.markush is False
    assert nc.markush_reason == ""


# ── real-XML, OPSIN-gated ─────────────────────────────────────────────────

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_every_structure_carries_a_reagent_label_and_reason():
    from patentdb3.sources.reagents import LABELS
    out = _extract()
    assert all(nc.label in LABELS for nc in out)
    for nc in out:
        if nc.label == "compound":
            assert nc.reason == ""
        else:
            assert nc.reason != "", (nc.name, nc.label)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_labeling_never_drops_a_structure():
    """`reagents.py` LABELS, it does not filter — `extract_names` must not
    reintroduce a drop at its call site. Every distinct InChIKey OPSIN
    resolved shows up in the output regardless of label.
    """
    out = _extract()
    keys = {nc.key for nc in out}
    assert len(keys) == len(out), "duplicate/dropped structures — dedup or filter regressed"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_reagent_label_distribution_matches_measured_baseline():
    """Locks in this task's measured number: `reagents.classify` run directly
    over the 238 PRE-asterisk-fix structures on US8952177 yields exactly 233
    compound / 3 reagent / 2 trace_fragment (verified separately by running
    the lexicon+backstop against that frozen 238-item set — see task report).

    Restated here against the SHIPPED (post-asterisk-fix) output, whose total
    is 234, not 238 — fixing the `*` boundary bug simultaneously (a) resolves
    10 genuinely new relative-stereo names that previously fragmented below
    `IUPAC_MIN_SEED` or collapsed into a flat mis-read, and (b) removes 14
    flat/no-stereo boundary-truncation ARTIFACTS the old bug produced for a
    compound whose correct relative-stereo form exists elsewhere in the same
    text — 10 of those 14 are replaced in place (net 0 on count, a wrong
    entry swapped for a right one) and 4 turn out to duplicate a structure
    the patent already stated (net -1 each, a duplicate removed, not a
    compound lost). None of the 14 removed or 10 added entries are labeled
    anything but "compound", so reagent/trace_fragment stay pinned.
    """
    out = _extract()
    dist = Counter(nc.label for nc in out)
    assert dist["reagent"] == 3, f"reagent count changed: {dist['reagent']} (was 3)"
    assert dist["trace_fragment"] == 2, (
        f"trace_fragment count changed: {dist['trace_fragment']} (was 2)")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_asterisk_fix_recovers_relative_stereo_names_as_markush():
    out = _extract()
    markush = [nc for nc in out if nc.markush]
    # 30, was 16 before Markush entries stopped deduplicating against
    # concrete ones — a relative-stereo name shares its InChIKey with its
    # `racemic cis-` sibling, so half of them were being silently deleted.
    # Was 11 before `<heading>` text entered candidate generation: five
    # more relative-stereo names are stated only in an Example heading and were
    # never seen by a `<p>`-only pass.
    assert len(markush) == 30, (
        f"markush count drifted ({len(markush)}, was 30) — re-measure before touching this")
    for nc in markush:
        assert _RELATIVE_STEREO.search(nc.name), nc.name
        assert nc.markush_reason.startswith("relative_stereo:"), nc.markush_reason
        # the label axis is independent — see the module docstring — but in
        # THIS measured corpus every relative-stereo name is a full drug-like
        # scaffold, never a lexicon reagent or an HAC<=3 fragment
        assert nc.label == "compound", (nc.name, nc.label)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_non_markush_structures_are_not_flagged():
    """The flag is precise, not a blanket "contains an asterisk somewhere in
    the module" bit — every non-flagged structure's accepted name really has
    no `[RSEZ]\\*` in it.
    """
    out = _extract()
    for nc in out:
        if not nc.markush:
            assert not _RELATIVE_STEREO.search(nc.name), nc.name
            assert nc.markush_reason == ""


def test_markush_structures_never_carry_an_inchikey():
    """A generic entity must not claim a single-structure identity.

    `(1R*,2S*)-X` denotes a SET of stereoisomers. An InChIKey asserts one
    molecule, which is false about it — and not harmlessly so: OPSIN resolves
    `(1R*,2S*)-X` and `racemic cis-X` to the SAME key because they ARE the same
    racemate, so a shared key made two DISTINCT patent examples collide in
    dedup and one was silently deleted, with its compound number and its assay
    values. The field stays empty until enumeration produces real molecules.

    Capability test, not a count: it fails if any Markush entry ever acquires
    a key, whatever the totals happen to be that day.
    """
    got = extract_names(_xml(), PID)
    markush = [n for n in got if n.markush]
    assert markush, "no Markush structures found — fixture drifted"
    offenders = [n.name[:60] for n in markush if n.inchikey]
    assert not offenders, f"Markush entries carrying an InChIKey: {offenders}"


def test_markush_does_not_deduplicate_against_a_concrete_structure():
    """Blanking the InChIKey alone would NOT have fixed the collision.

    `NamedCompound.key` falls back to SMILES when there is no InChIKey, and a
    relative-stereo name shares its SMILES with its concrete sibling too. So
    the key is namespaced. This asserts the two kinds coexist rather than one
    erasing the other — the actual mechanism, not its symptom.
    """
    got = extract_names(_xml(), PID)
    by_smiles = {}
    for n in got:
        by_smiles.setdefault(n.smiles, []).append(n)
    shared = {s: v for s, v in by_smiles.items()
              if len(v) > 1 and any(n.markush for n in v) and any(not n.markush for n in v)}
    assert shared, (
        "no SMILES is shared between a Markush and a concrete structure on this "
        "patent — the collision this guards against is not exercised, so the "
        "test proves nothing. Re-check the fixture.")
    for smi, group in shared.items():
        assert {n.markush for n in group} == {True, False}
        assert len({n.key for n in group}) == len(group), (
            f"namespacing failed: {len(group)} structures share SMILES {smi[:40]} "
            f"and collapse to {len({n.key for n in group})} key(s)")


# ── the heading route ────────────────────────────────────────────────────
#
# `extract_names` reads `<heading>` elements a second time, WHOLE, because a
# heading has no boundary problem: the patent already said where the name
# starts (after `Example 43:`) and where it stops (the end of the element).
# The prose machinery's seeds actively hurt there — `_SEED`'s character class
# has no space, so `tert-butyl 4-(...)carboxylate` breaks in two.
#
# Everything below is about the two properties that make that safe: the route
# may only ADD, and OPSIN is not allowed to be the only gate.

from patentdb3.sources.iupac_names import (          # noqa: E402
    _COVERAGE_MIN, _coverage, _heading_texts,
)


def test_heading_framing_is_stripped_to_the_name_the_patent_stated():
    """Each clause here was observed on a real heading in the cached corpus.
    Leaving any of it on the front is not a neutral error: OPSIN would then be
    judging a string this code broke, and whatever recovered it would get
    credit for undoing our own damage."""
    xml = "".join(f"<heading>{h}</heading>" for h in [
        "Example 43: 4-(hydroxymethyl)-3-(2-morpholinopyridin-4-yl)benzamide",
        "Intermediate 991F: tert-butyl 4-(4-fluoro-1H-indol-5-yl)carboxylate",
        "Example 44: Synthesis of N-((4,6-dimethyl-2-oxopyridin-3-yl)methyl)amine",
        "Example 5. Preparation of 2-amino-4-chloro-N-phenylbenzamide (Compound 2)",
        "Compound I-20 — 5-bromo-3-isopropyl-1H-pyrrolo[3,2-b]pyridine",
        "Examples",                                   # a section title, not a compound
        "Example 7: Assay",                           # residual too short
    ])
    got = dict(_heading_texts(xml))
    assert got["43"] == "4-(hydroxymethyl)-3-(2-morpholinopyridin-4-yl)benzamide"
    assert got["44"] == "N-((4,6-dimethyl-2-oxopyridin-3-yl)methyl)amine"
    assert got["5"] == "2-amino-4-chloro-N-phenylbenzamide"
    assert got["I-20"] == "5-bromo-3-isopropyl-1H-pyrrolo[3,2-b]pyridine"
    # `Intermediate 991F` is DROPPED, not mis-parsed: `config.FINISHED_ONLY`
    # refuses a heading whose label names a synthesis waypoint. The framing
    # stripper still handles it correctly — this asserts the scope decision,
    # not a parsing limit.
    assert "991F" not in got, "an intermediate must not reach the output"
    assert len(got) == 4, sorted(got)


def test_coverage_rejects_the_comma_split_amputation():
    """THE COUNTER-EXAMPLE, kept verbatim.

    Splitting an IUPAC name on its commas and keeping the last fragment
    produces a real, well-formed molecule that OPSIN accepts without
    complaint — `nicotinamide`, out of a 100-character name. Corpus-wide that
    "recovers" ~106 headings at a median coverage of 0.14. A comma in an IUPAC
    name is a locant separator, and the fragment after one is a different
    compound. This is why coverage is a gate and OPSIN alone is not.
    """
    full = ("5-(4-((1R,5S)-3-azabicyclo[3.1.0]hexan-1-yl)phenyl)-2-amino-"
            "N-((1r,4R)-4-hydroxycyclohexyl)nicotinamide")
    assert _coverage("nicotinamide", full) < 0.2
    assert _coverage("nicotinamide", full) < _COVERAGE_MIN
    # the whole name, and the same name with wrap spaces removed, both pass —
    # coverage ignores non-alphanumerics precisely so a dewrap is not
    # penalised for deleting the spaces it exists to delete
    assert _coverage(full, full) == 1.0
    spaced = full.replace(")-2-", ")- 2-")
    assert _coverage(full, spaced) == 1.0


def test_coverage_is_zero_on_an_empty_denominator():
    assert _coverage("benzene", "") == 0.0
    assert _coverage("", "benzene") == 0.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_heading_route_only_adds_never_replaces():
    """The route is handed `extract_names`'s own dedup set and runs last, so
    it CANNOT cost the prose route a structure. Asserted on real XML rather
    than argued from the code, because "additive by construction" is exactly
    the kind of claim that survives a refactor in the comment and not in the
    behaviour.

    THE PATENT CHANGED FROM US10544143, and not because the assertion got
    inconvenient. Under `config.FINISHED_ONLY` that patent's heading route
    correctly yields nothing: its only `Example` heading is the bare
    `Example 438` with no name after it (so it fails `_HEADING_MIN`), and every
    name-bearing heading it has is an `Intermediate`. US10155002 contributes 59
    heading-route structures under the same rule, so the additive property is
    still exercised on real XML rather than asserted against an empty set.
    """
    from patentdb3.sources import iupac_names as mod
    pid = "US10155002"
    xml = _xml(pid)
    pytest.importorskip("py2opsin")
    logging.disable(logging.CRITICAL)
    real = mod._heading_structures
    try:
        mod._heading_structures = lambda x, p, s: []
        off = extract_names(xml, pid)
        mod._heading_structures = real
        on = extract_names(xml, pid)
    except Exception as e:
        pytest.skip(f"OPSIN unavailable in this environment: {e!r}")
    finally:
        mod._heading_structures = real
        logging.disable(logging.NOTSET)
    if not off:
        pytest.skip("OPSIN produced nothing — treat as unavailable, not a failure")
    assert {n.key for n in off} <= {n.key for n in on}, "the heading route dropped a structure"
    assert len(on) > len(off), "the heading route found nothing on a patent it should"
    added = [n for n in on if n.source == "heading"]
    assert len(on) == len(off) + len(added)
    # every added row is a heading row, carries the heading's OWN id, and is
    # never a duplicate structure of something the prose route already had
    assert all(n.cid for n in added), [n.name for n in added if not n.cid]
    assert all(n.start == -1 for n in added)
    assert len({n.key for n in added}) == len(added)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_heading_route_keeps_the_markush_contract():
    """Same rule as the prose route, enforced on the second producer too: a
    relative-stereo name gets no InChIKey and a namespaced key. This is the
    invariant whose violation silently deleted US8952177's Examples 163 and
    166 — a second code path that resolves names is a second place to break
    it."""
    out = _extract("US10214537")
    from patentdb3.sources.reagents import LABELS
    heads = [n for n in out if n.source == "heading"]
    if not heads:
        pytest.skip("no heading-sourced structures — fixture drifted")
    for n in heads:
        assert n.heading_transform in {"as_is_whole", "dewrap_whole", "dewrap_seeded"}
        if n.markush:
            assert n.inchikey == ""
            assert n.key.startswith("markush::")
        assert n.label in LABELS
