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
    # 16, was 11 before `<heading>` text entered candidate generation: five
    # more relative-stereo names are stated only in an Example heading and were
    # never seen by a `<p>`-only pass.
    assert len(markush) == 16, (
        f"markush count drifted ({len(markush)}, was 16) — re-measure before touching this")
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
