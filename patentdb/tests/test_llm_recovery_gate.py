"""`LLM_RECOVERY=0` must actually stop the paid recovery calls.

Commit `4c26f96` ("Gate the four remaining ad-hoc LLM callers") added a
`LLM_RECOVERY_ENABLED` check to `impossible_fragment_retry.py:199` — but
that line sat inside `_attempt1_focused`, and `_attempt1_focused` and
`_attempt2_wide` had ZERO callers. Both were superseded by the batched
`iupac_burst_targeted` path in `retry_impossible_fragments`. So the gate
was applied to dead code while the live paid call went out ungated.

Traced call sites of `iupac_burst_targeted` (all paid, all previously
behind no flag at all):

    routes/process_patent.py         `_targeted_fill_missing_cids`
    core/impossible_fragment_retry.py `retry_impossible_fragments`
    scripts/eval/targeted_iupac_strategy5.py

`IUPAC_BURST` does NOT cover these — it gates `iupac_burst`, a different
function (the broad density-ranked pass).

The gate now lives at the top of `iupac_burst_targeted` itself rather
than at each call site, so a future caller cannot reintroduce an ungated
path by forgetting to copy the check.
"""
from __future__ import annotations

import inspect

import pytest

from patentdb.core.assay_fsm.harvest import iupac_orchestrator as IO
from patentdb.core import impossible_fragment_retry as IFR


def _boom(*a, **k):  # pragma: no cover - must never run
    raise AssertionError("PAID CALL: LLM fired with LLM_RECOVERY disabled")


def test_targeted_burst_is_silent_when_llm_recovery_disabled(monkeypatch):
    """With the flag off, no LLM call and an empty result."""
    monkeypatch.setattr(IO.config, "LLM_RECOVERY_ENABLED", False, raising=False)
    monkeypatch.setattr(IO, "extract_iupac_pairs", _boom, raising=False)

    out = IO.iupac_burst_targeted(
        patent_id="USTEST",
        text="Example 5 is 2-(4-chlorophenyl)acetic acid. " * 200,
        missing_cids=["5", "6", "7"],
        cost_tracker=None,
    )
    assert out == {}


def test_retry_impossible_fragments_fires_nothing_when_disabled(monkeypatch):
    """The live retry pass must spend nothing with the flag off."""
    monkeypatch.setattr(IO.config, "LLM_RECOVERY_ENABLED", False, raising=False)
    monkeypatch.setattr(IO, "extract_iupac_pairs", _boom, raising=False)

    example_index = {"5": {"canonical_smiles": "", "iupac_name": ""}}
    n_attempted, n_recovered, idx = IFR.retry_impossible_fragments(
        "USTEST", "Example 5 blah. " * 100, example_index,
    )
    assert (n_attempted, n_recovered) == (0, 0)
    assert idx["5"]["canonical_smiles"] == ""


def test_dead_retry_attempt_functions_are_gone():
    """`_attempt1_focused` / `_attempt2_wide` had no callers — removed.

    They carried the ONLY `LLM_RECOVERY_ENABLED` check in the module,
    which is how the live path came to be ungated.
    """
    assert not hasattr(IFR, "_attempt1_focused")
    assert not hasattr(IFR, "_attempt2_wide")
    assert not hasattr(IFR, "_FOCUSED_PROMPT")


def test_no_unused_harvest_cache_imports_remain():
    """Those functions were the only users of these imports."""
    src = inspect.getsource(IFR)
    for name in ("_chunk_fingerprint", "_read_cache", "_write_cache",
                 "_find_cid_context_windows", "extract_iupac_pairs"):
        assert name not in src, f"{name} is imported but no longer used"


def test_gate_lives_in_the_function_not_the_call_site():
    """A new caller must not be able to create an ungated paid path."""
    assert "LLM_RECOVERY_ENABLED" in inspect.getsource(IO.iupac_burst_targeted)
