"""`LLM_RECOVERY=0` must block NEW spend, not answers already paid for.

`test_llm_recovery_gate.py` pins the half of this that was a real defect: the
live paid path went out ungated, so the gate now lives inside
`iupac_burst_targeted`. This file pins the other half.

The gate was placed at the TOP of the function — above `_read_cache()`. A
patent whose windows are all in `adaptive_extraction_rules.json` therefore
returned `{}` under `LLM_RECOVERY=0` even though serving them costs nothing
and opens no socket. On US11566007 that is 19 cids of already-bought IUPAC
names refused at $0.

The cost gate one screen below already draws the line in the right place: it
sits AFTER the cache-hit `continue`, so a cached chunk keeps flowing once a
patent is over budget, and only a chunk that would be BOUGHT is stopped.
`test_iupac_burst_cost.py:126` states that placement as the contract for
`iupac_burst`, naming `iupac_burst_targeted` as the model. The recovery gate
belongs at the same line as the cost gate, for the same reason.

`continue`, not `break`: spend only grows, so the cost gate can abandon the
rest of the loop, but the flag is constant for the whole run — stopping at the
first uncached chunk would throw away every cached chunk after it.

No test here makes a paid call: `extract_iupac_pairs` is replaced by a raiser.
"""
from __future__ import annotations

import pytest

from patentdb.core.assay_fsm.harvest import iupac_orchestrator as IO


class FakeTracker:
    """Never over budget — these tests isolate the RECOVERY gate."""

    def patent_lm_exceeded(self, patent_id):
        return False

    def patent_spend(self, patent_id):
        return 0.0


class FakeLibrary:
    def add_discovery(self, *a, **k):
        pass


def _boom(*a, **k):  # pragma: no cover - must never run
    raise AssertionError("PAID CALL: LLM fired with LLM_RECOVERY disabled")


_NAME = "2-[4-(3-chlorophenyl)piperazin-1-yl]-N-(pyridin-3-ylmethyl)acetamide"

# Two cid mentions far enough apart that the window deduper keeps both.
_TEXT = (
    "Example 5. " + _NAME + " was obtained as a white solid. "
    + ("filler text. " * 400)
    + "Example 9. " + _NAME + " was obtained as a pale solid. "
    + ("filler text. " * 400)
)
_WINDOW = 800


def _windows(patent_id="USTEST", cids=("5", "9")):
    """The (cache_key, cid) pairs `iupac_burst_targeted` will derive."""
    out = []
    for cid in cids:
        for _off, win in IO._find_cid_context_windows(
            _TEXT, cid, window_chars=_WINDOW,
        ):
            key = f"iupac_targeted:{IO._content_fingerprint(win, patent_id)}"
            out.append((key, cid))
    return out


def _cache_for(cids, patent_id="USTEST"):
    """A cache holding a bought answer for every window of `cids`."""
    return {
        key: {"pairs": [{"compound_id": cid, "iupac_name": _NAME}],
              "pattern_meta": None}
        for key, cid in _windows(patent_id, cids)
    }


@pytest.fixture(autouse=True)
def _recovery_off(monkeypatch):
    monkeypatch.setattr(IO.config, "LLM_RECOVERY_ENABLED", False, raising=False)
    monkeypatch.setattr(IO, "extract_iupac_pairs", _boom, raising=False)


def test_the_fixture_text_produces_windows_for_both_cids():
    """Guard on the fixture itself — a text that yields no window would make
    every assertion below pass for the wrong reason."""
    assert {cid for _k, cid in _windows()} == {"5", "9"}


def test_a_cached_chunk_is_served_with_llm_recovery_off(monkeypatch):
    """The whole point: an answer already paid for costs nothing to replay."""
    monkeypatch.setattr(IO, "_read_cache", lambda: _cache_for(("5", "9")))
    monkeypatch.setattr(IO, "_write_cache", lambda data: None)

    out = IO.iupac_burst_targeted(
        patent_id="USTEST", text=_TEXT, missing_cids=["5", "9"],
        cost_tracker=FakeTracker(), library=FakeLibrary(),
        window_chars=_WINDOW,
    )
    assert out == {"5": _NAME, "9": _NAME}


def test_an_uncached_chunk_is_not_bought_with_llm_recovery_off(monkeypatch):
    """Nothing in the cache means nothing to serve — and nothing to buy."""
    monkeypatch.setattr(IO, "_read_cache", lambda: {})
    monkeypatch.setattr(IO, "_write_cache", lambda data: None)

    out = IO.iupac_burst_targeted(
        patent_id="USTEST", text=_TEXT, missing_cids=["5", "9"],
        cost_tracker=FakeTracker(), library=FakeLibrary(),
        window_chars=_WINDOW,
    )
    assert out == {}


def test_an_uncached_chunk_does_not_stop_the_cached_ones_after_it(monkeypatch):
    """`continue`, not `break`.

    Chunks are visited in document order. Cid 5's window is uncached and cid
    9's is cached; a gate that abandoned the loop would return nothing.
    """
    monkeypatch.setattr(IO, "_read_cache", lambda: _cache_for(("9",)))
    monkeypatch.setattr(IO, "_write_cache", lambda data: None)

    out = IO.iupac_burst_targeted(
        patent_id="USTEST", text=_TEXT, missing_cids=["5", "9"],
        cost_tracker=FakeTracker(), library=FakeLibrary(),
        window_chars=_WINDOW,
    )
    assert out == {"9": _NAME}


def test_a_gated_run_does_not_rewrite_the_shared_cache(monkeypatch):
    """`adaptive_extraction_rules.json` is shared with the assay burst and is
    3.5 MB of paid output. A run that bought nothing has nothing to add, and
    rewriting it is a chance to lose it for no gain."""
    written = []
    monkeypatch.setattr(IO, "_read_cache", lambda: _cache_for(("5", "9")))
    monkeypatch.setattr(IO, "_write_cache", lambda data: written.append(data))

    IO.iupac_burst_targeted(
        patent_id="USTEST", text=_TEXT, missing_cids=["5", "9"],
        cost_tracker=FakeTracker(), library=FakeLibrary(),
        window_chars=_WINDOW,
    )
    assert written == []


def test_the_recovery_gate_is_checked_before_the_cost_gate(monkeypatch):
    """The disabled path must not need a cost tracker at all.

    `test_llm_recovery_gate.test_targeted_burst_is_silent_when_llm_recovery_
    disabled` calls this function with `cost_tracker=None`; if the recovery
    gate moved BELOW the cost gate, that test would die on an AttributeError
    instead of measuring the flag.
    """
    monkeypatch.setattr(IO, "_read_cache", lambda: {})
    monkeypatch.setattr(IO, "_write_cache", lambda data: None)

    out = IO.iupac_burst_targeted(
        patent_id="USTEST", text=_TEXT, missing_cids=["5", "9"],
        cost_tracker=None, library=FakeLibrary(), window_chars=_WINDOW,
    )
    assert out == {}
