"""Tests for `patentdb3.sources.reagents`.

Two concerns:

1. **Behavior** — the label set is exactly `{"compound", "reagent",
   "trace_fragment"}`, the lexicon tier fires before the structural backstop,
   salt-suffix stripping and unicode quote/dash normalization work, the HAC
   boundary is exactly `<= 3`, and `reason` is populated in the documented
   format for every non-default label.
2. **Determinism** — the HARD REQUIREMENT from the module docstring: the same
   `(name, smiles)` must classify to the same `(label, reason)` every time,
   in-process across many repeats AND across a completely fresh interpreter
   (a subprocess, not just a new call). This is proven, not asserted, by
   actually running both and comparing.

Nothing here touches `patentdb3/sources/iupac_names.py` or wires
`reagents.classify` into any live caller — this module is tested standalone,
per the task's own constraint.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from patentdb3.core import config
from patentdb3.sources.reagents import LABELS, ReagentVerdict, classify, label_of

# A fixed, diverse battery of (name, smiles) pairs used by both the behavior
# tests and the determinism tests, so "what was classified" and "was it
# stable" are checked against the exact same inputs.
BATTERY: list[tuple[str, str]] = [
    # plain lexicon hit (solvent), also happens to sit at HAC 3 — lexicon
    # must win over the structural tier, not just agree with it by luck
    ("dichloromethane", "ClCCl"),
    # lexicon hit via salt-suffix stripping
    ("triethylamine hydrochloride", "CCN(CC)CC"),
    # lexicon hit only reachable through unicode prime normalization
    ("1,1’-bis(diphenylphosphino)ferrocene", ""),
    # unmatched name, HAC exactly at the floor -> trace_fragment
    ("reagent-debris-fragment-x", "CCC"),
    # unmatched name, HAC one above the floor -> compound (backstop must not
    # over-fire)
    ("reagent-debris-fragment-y", "CCCC"),
    # unmatched name, no smiles at all -> compound, empty reason
    ("4-fluoropiperidine", ""),
    # unmatched name, smiles given but unparseable -> compound, not a crash
    ("mystery-fragment-z", "not-a-smiles"),
    # a real, longer drug-like structure that should clear both tiers as a
    # compound (celecoxib)
    ("4-[5-(4-methylphenyl)-3-(trifluoromethyl)pyrazol-1-yl]benzenesulfonamide",
     "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F"),
    # bare small-molecule lexicon hit that is ALSO tiny (methanol, HAC 2) —
    # lexicon must still be checked first
    ("methanol", "CO"),
]


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------

def test_label_set_is_closed():
    assert LABELS == ("compound", "reagent", "trace_fragment")
    for name, smiles in BATTERY:
        assert classify(name, smiles).label in LABELS


def test_lexicon_hit_reports_category_and_key():
    v = classify("dichloromethane", "ClCCl")
    assert v.label == "reagent"
    assert v.reason == "lexicon:solvent:dichloromethane"


def test_lexicon_wins_over_structural_tier_at_same_hac():
    # dichloromethane is HAC 3, exactly at the structural floor, so this only
    # proves something if the lexicon check happens FIRST.
    v = classify("dichloromethane", "ClCCl")
    assert v.label == "reagent"
    assert v.reason.startswith("lexicon:")


def test_salt_suffix_is_stripped_and_recorded():
    v = classify("triethylamine hydrochloride", "CCN(CC)CC")
    assert v.label == "reagent"
    assert v.reason == "lexicon:base:triethylamine:salt=hydrochloride"


def test_unicode_prime_normalizes_to_ascii_lexicon_entry():
    v = classify("1,1’-bis(diphenylphosphino)ferrocene", "")
    assert v.label == "reagent"
    assert v.reason == "lexicon:catalyst_ligand:1,1'-bis(diphenylphosphino)ferrocene"


def test_structural_backstop_fires_at_hac_floor():
    v = classify("reagent-debris-fragment-x", "CCC")  # propane, HAC 3
    assert v.label == "trace_fragment"
    assert v.reason == "structural:hac=3:floor=3"


def test_structural_backstop_does_not_overfire_above_floor():
    v = classify("reagent-debris-fragment-y", "CCCC")  # butane, HAC 4
    assert v.label == "compound"
    assert v.reason == ""


def test_no_smiles_only_lexicon_tier_runs():
    v = classify("4-fluoropiperidine", "")
    assert v.label == "compound"
    assert v.reason == ""


def test_unparseable_smiles_does_not_crash_and_keeps_compound():
    v = classify("mystery-fragment-z", "not-a-smiles")
    assert v.label == "compound"


def test_real_drug_like_structure_is_not_flagged():
    v = classify(
        "4-[5-(4-methylphenyl)-3-(trifluoromethyl)pyrazol-1-yl]benzenesulfonamide",
        "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    )
    assert v.label == "compound"


def test_tiny_lexicon_entry_still_matches_lexicon_not_structural():
    v = classify("methanol", "CO")
    assert v.label == "reagent"
    assert v.reason == "lexicon:solvent:methanol"


def test_label_of_matches_classify():
    for name, smiles in BATTERY:
        assert label_of(name, smiles) == classify(name, smiles).label


def test_default_reason_is_empty_string():
    v = classify("some-unremarkable-name-nobody-listed", "")
    assert v == ReagentVerdict("compound", "")


# ---------------------------------------------------------------------------
# Determinism — the hard requirement
# ---------------------------------------------------------------------------

def _classify_battery() -> list[tuple[str, str]]:
    return [(v.label, v.reason) for v in (classify(n, s) for n, s in BATTERY)]


def test_classification_is_deterministic_across_repeats():
    """Same process, same inputs, many repeats: every run must match the
    first, exactly, for every item in the battery.
    """
    first = _classify_battery()
    for _ in range(50):
        assert _classify_battery() == first


def test_classification_is_deterministic_across_fresh_interpreter():
    """A brand new `python3` process — not just a new call in this one —
    must classify the same battery to the exact same labels and reasons.
    This is what actually rules out set/dict iteration-order leakage,
    uninitialized module state, or anything else that only a fresh
    interpreter would expose.
    """
    in_process = _classify_battery()

    script = (
        "import json, sys\n"
        "from patentdb3.sources.reagents import classify\n"
        f"battery = {BATTERY!r}\n"
        "out = [[v.label, v.reason] for v in "
        "(classify(n, s) for n, s in battery)]\n"
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
    subprocess_result = [tuple(pair) for pair in json.loads(proc.stdout)]

    assert subprocess_result == in_process


def test_classification_is_deterministic_twice_in_fresh_interpreters():
    """Two SEPARATE fresh interpreters must also agree with each other, not
    just with the in-process run — guards against any determinism that
    happens to hold only relative to the test process's own state.
    """
    script = (
        "import json\n"
        "from patentdb3.sources.reagents import classify\n"
        f"battery = {BATTERY!r}\n"
        "out = [[v.label, v.reason] for v in "
        "(classify(n, s) for n, s in battery)]\n"
        "print(json.dumps(out))\n"
    )
    results = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(config.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        results.append(json.loads(proc.stdout))

    assert results[0] == results[1]
