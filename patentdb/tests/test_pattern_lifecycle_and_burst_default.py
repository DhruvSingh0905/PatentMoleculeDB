"""Two decorative controls, measured and removed.

Both are the same defect in different modules: a knob whose name promises a
decision, wired to nothing that can make it.

── 1. the pattern library's promotion lifecycle ────────────────────────────

`assay_pattern_library`'s module docstring promised

    pending  -> freshly extracted from one chunk
    auto_loaded -> >=3 distinct patent fingerprints have used it
    promoted -> curator approved (moves to canonical store)

and `add_pattern` implemented the middle arrow. `_load_active_patterns`
admitted all three statuses identically, so the arrow never changed what
fires. Measured on `patentdb/data/assay_patterns.discoveries.json`
(116 entries, 2026-08-07) and on the shipped corpus artifacts:

  * all 116 entries are `pending`; **32,191** shipped assay rows come from
    them -- 38.1% of the 84,517 records in
    `output_v2/text_extraction/*/assay_tables.json`, counted by the `source`
    field. Every one would ship identically at any status.
  * `len(fingerprints_observed)` is exactly 1 for all 116, so the >=3 branch
    has never been taken and cannot be.

Identity is the SHA of the canonicalised regex (`_pattern_key`), so a second
observation corroborates only by producing a byte-identical regex. It never
happens -- not even within one patent. `first_seen` is written only in
`add_pattern`'s new-entry branch, and US9718790 has 6 entries dated
2026-05-18 and 21 more dated 2026-05-19: the same document's tables, 27
distinct keys, because the model does not emit a stable regex twice.

Re-keying on a layout fingerprint was measured before removing the lifecycle
(see the wiki, `10 - Pattern Library.md`). Using the `repair/gap.py`
convention -- column count, per-column value shape, normalised header words --
the 116 entries fall into 76 groups and **0** reach even 2 patents, because the
header words ARE the assay target and two patents rarely assay the same target
with the same shape. Dropping the header words is the only key that yields
non-zero corroboration, and it puts `P2X3 IC50`, `RORgamma Binding IC50`,
`B-Raf IC50`, `Molecular Weight` and `LCMS (ESI) [M+H] Found` in ONE group --
so "corroborated" would mean a molecular weight vouching for a potency, which
is the fabricated-MEANING failure the foreign and wrong-table gates exist to
block. A gate that can only fire by discarding the thing it vets is not a gate.

What is load-bearing, and must not be deleted with it: `fingerprints_observed`
is READ by `harvest/orchestrator.py::_patent_has_own_patterns`, which feeds the
HARVEST_SKIP gate. It is not a lifecycle field; it is the "we have seen this
patent's format" signal.

── 2. HARVEST_BURST's default ─────────────────────────────────────────────

Every other paid tier defaults off (`IUPAC_BURST`, `LLM_NAME_REPAIR`,
`STRATEGY5_LLM`, `LLM_RECOVERY`, `PUBCHEM_NAME_LOOKUP`, `ASSAY_REALIGN`).
`HARVEST_BURST` defaulted on. Measured at HEAD, zero paid calls:

  * `HARVEST_SKIP` short-circuits the paid arm on 8 of 22 corpus patents
    (`chunks = []`), which hold 131,268 of the 133,926 free pre-library rows.
  * of the 14 where it would run, 13 have a pre-library yield of 0.
  * response-cache coverage of the prompts it would send: **0/19** chunks on
    US9718825 and **0/28** on US10919885, for Agent 1 and Agent 2 alike. Every
    chunk is a paid miss.
  * the default path is batched (`LLM_BATCH=1`) and `_run_chunks_batched`
    neither reads nor writes the `harvest:` chunk cache -- the adaptive cache
    holds 755 entries and 0 in that namespace -- so the tier cannot amortise.
  * that same batched path never reaches the `patent_lm_exceeded` check, which
    sits inside the sequential for-loop it short-circuits, so
    `PER_PATENT_LM_CAP` does not bound the default path.

Turning it off keeps the free arm: `9828990` lifted `prelib_tuples` out of
`harvest_burst`, and `assay_fsm/pipeline.py`'s `else` branch runs it when the
paid tier is disabled.
"""
from __future__ import annotations

import importlib
import json

import pytest

from patentdb.core.assay_fsm import assay_pattern_library as lib


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def library(tmp_path, monkeypatch):
    """Point the module at a throwaway library file and hand back its path."""
    path = tmp_path / "assay_patterns.discoveries.json"
    path.write_text(json.dumps({"schema_version": "1.0", "tokens": []}, indent=1))
    monkeypatch.setattr(lib, "_PATTERNS_PATH", path)
    return path


def _tokens(path) -> list[dict]:
    return json.loads(path.read_text())["tokens"]


# A real library shape, verbatim from
# patentdb/data/assay_patterns.discoveries.json.
_REGEX = r"(?P<cid>\d+)\s+(?P<value0>\d+\.\d+)"
_ASSAYS = ["RORγ Binding IC50 μM"]


# ── 1. the lifecycle is gone ──────────────────────────────────────────────

def test_add_pattern_records_no_status(library):
    """`status` is not written, because nothing ever read it to a decision.

    `_load_active_patterns` admitted `pending`, `auto_loaded` and `promoted`
    identically. A field that three values map to one behaviour is not a
    state machine.
    """
    assert lib.add_pattern(_REGEX, _ASSAYS, "1 0.070", "US10273259") is True
    entry = _tokens(library)[0]
    assert "status" not in entry, (
        "add_pattern still writes a lifecycle status; "
        f"got {entry.get('status')!r}"
    )


def test_no_promotion_branch_however_many_patents_observe(library):
    """Three distinct patents, three observations -- and no state changes.

    This is the exact input the old `>=3 fingerprints and >=3 observations`
    branch waited for. It never arrived in production because identity is a
    regex hash; here it is handed to the function directly, and there is still
    no promotion to make.
    """
    for pid in ("US10273259", "US20240010684A1", "US9745328"):
        lib.add_pattern(_REGEX, _ASSAYS, "1 0.070", pid)
    entry = _tokens(library)[0]
    assert "status" not in entry
    assert "n_observations" not in entry, (
        "n_observations fed only the deleted promotion branch and has no "
        "reader; it should not be written"
    )


def test_load_active_admits_every_entry_with_a_regex(library):
    """The load gate is 'has a usable regex', and nothing else.

    Legacy entries on disk carry `status` from before the lifecycle was
    removed. They must keep loading -- all 116 are `pending`, and dropping
    them would delete 38.1% of the corpus's shipped assay rows.
    """
    library.write_text(json.dumps({"schema_version": "1.0", "tokens": [
        {"key": "a", "regex": _REGEX, "column_assays": _ASSAYS,
         "status": "pending"},
        {"key": "b", "regex": _REGEX + r"\s+", "column_assays": _ASSAYS,
         "status": "auto_loaded"},
        {"key": "c", "regex": _REGEX + r"\s*", "column_assays": _ASSAYS,
         "status": "promoted"},
        {"key": "d", "regex": _REGEX + r"$", "column_assays": _ASSAYS},
        {"key": "e", "regex": "", "column_assays": _ASSAYS},
    ]}, indent=1))
    keys = {e["key"] for e in lib._load_active_patterns()}
    assert keys == {"a", "b", "c", "d"}, (
        "every entry with a regex loads, regardless of any legacy status; "
        "the entry with no regex does not"
    )


def test_fingerprints_observed_is_kept_and_appended(library):
    """The one field with a live consumer, guarded against the deletion.

    `harvest/orchestrator.py::_patent_has_own_patterns` scans
    `fingerprints_observed` for the patent id; it is half of the HARVEST_SKIP
    gate, which short-circuits the paid tier on 8 of 22 corpus patents.
    """
    lib.add_pattern(_REGEX, _ASSAYS, "1 0.070", "US10273259")
    lib.add_pattern(_REGEX, _ASSAYS, "1 0.070", "US9745328")
    entry = _tokens(library)[0]
    assert entry["fingerprints_observed"] == ["US10273259", "US9745328"]
    assert entry["first_seen_patent"] == "US10273259", (
        "first_seen_patent drives `_is_foreign`; it must stay pinned to the "
        "discovering patent"
    )


def test_identity_is_a_dedup_key_not_a_corroboration_key(library):
    """Why the lifecycle could not be repaired in place.

    Two patents describing the SAME table with equivalent regexes land in two
    entries, so a per-key counter can never reach 2. `_canonicalize_regex`
    collapses quantifier variants and anchors; anything else -- a different
    character class, a different separator -- is a different pattern.
    """
    lib.add_pattern(r"(?P<cid>\d{1,3})\s+(?P<value0>\d+\.\d+)", _ASSAYS, "x",
                    "US10273259")
    # Same table, different character class for the value. Semantically the
    # same row; a different SHA.
    lib.add_pattern(r"(?P<cid>\d{1,4})\s+(?P<value0>[\d.]+)", _ASSAYS, "x",
                    "US9745328")
    toks = _tokens(library)
    assert len(toks) == 2
    assert all(len(t["fingerprints_observed"]) == 1 for t in toks), (
        "no entry accumulates a second patent -- which is why a >=3-patent "
        "gate keyed on this identity can never fire"
    )


# ── 2. the paid tier defaults off ─────────────────────────────────────────

def test_harvest_burst_defaults_off(monkeypatch):
    """`HARVEST_BURST` joins every other paid tier at 0.

    0% response-cache coverage on the chunks it would send, no `harvest:`
    entries in the adaptive cache for the batched path to reuse, and no
    `patent_lm_exceeded` check on that path. Opt in with `HARVEST_BURST=1`.
    """
    monkeypatch.delenv("HARVEST_BURST", raising=False)
    from patentdb.core import config
    importlib.reload(config)
    try:
        assert config.HARVEST_BURST_ENABLED is False, (
            "HARVEST_BURST still defaults ON"
        )
    finally:
        importlib.reload(config)


def test_harvest_burst_opt_in_still_works(monkeypatch):
    """Off by default is not removed -- the tier is one env var away."""
    monkeypatch.setenv("HARVEST_BURST", "1")
    from patentdb.core import config
    importlib.reload(config)
    try:
        assert config.HARVEST_BURST_ENABLED is True
    finally:
        monkeypatch.delenv("HARVEST_BURST", raising=False)
        importlib.reload(config)


def test_burst_off_still_runs_the_free_prelib_pass():
    """The free arm survives the default flip.

    `9828990` lifted `prelib_tuples` out of `harvest_burst` precisely so
    `HARVEST_BURST=0` would not also switch off a regex pass that costs
    nothing. The `else` branch in `assay_fsm/pipeline.py` must still call it --
    it is worth 133,926 rows across the corpus.
    """
    import inspect

    from patentdb.core.assay_fsm import pipeline

    src = inspect.getsource(pipeline.extract_for_patent)
    head, sep, tail = src.partition("HARVEST_BURST_ENABLED")
    assert sep, "the flag no longer gates anything in extract_for_patent"
    assert "prelib_tuples" in tail, (
        "the burst-off branch must still run the free pattern-library pass"
    )
