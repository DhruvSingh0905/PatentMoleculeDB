"""The tracer must be inert when off and truthful when on.

`core/calltrace.py` installs `sys.setprofile` process-wide. A tracer that
leaks — one that stays installed, or that records when `CALLTRACE` is unset —
would slow every production run while looking like it was doing nothing, which
is the exact failure shape CLAUDE.md's wiring section is about. So the first
test is that OFF means OFF, verified against `sys.getprofile()` and against the
absence of an output file, not against a flag.
"""
from __future__ import annotations

import json
import sys

import pytest

from patentdb.core import calltrace


# ── A known call, defined here so its qualname and line are known ──

def _leaf(n: int) -> int:
    return n * 2


def _middle(n: int) -> int:
    return _leaf(n) + _leaf(n)


def _known_caller(n: int) -> int:
    return _middle(n)


@pytest.fixture(autouse=True)
def _no_leaked_profiler():
    """Whatever a test does, the profiler must be uninstalled afterwards."""
    before = sys.getprofile()
    yield
    calltrace.stop()
    sys.setprofile(before)
    assert sys.getprofile() is before


# ── inert when CALLTRACE is unset ─────────────────────────────────

def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv(calltrace.ENV_FLAG, raising=False)
    assert calltrace.enabled() is False

    tr = calltrace.start("off_run", out_dir=tmp_path)
    try:
        assert tr.active is False
        # The decisive check: no hook was installed, so production pays nothing.
        assert sys.getprofile() is None
        _known_caller(3)
    finally:
        calltrace.stop()

    assert not tr.events_path.exists()
    assert not tr.summary_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_disabled_when_flag_is_not_exactly_one(monkeypatch, tmp_path):
    # "0", "true", "" must not enable it — only "1".
    for value in ("0", "true", "yes", ""):
        monkeypatch.setenv(calltrace.ENV_FLAG, value)
        assert calltrace.enabled() is False


def test_context_manager_is_inert_when_off(monkeypatch, tmp_path):
    monkeypatch.delenv(calltrace.ENV_FLAG, raising=False)
    with calltrace.trace("off_ctx", out_dir=tmp_path) as tr:
        _known_caller(2)
    assert tr.active is False
    assert tr.summary == {}
    assert list(tmp_path.iterdir()) == []


# ── records a known call when on ──────────────────────────────────

def test_records_known_call_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv(calltrace.ENV_FLAG, "1")
    assert calltrace.enabled() is True

    with calltrace.trace("on_run", out_dir=tmp_path) as tr:
        calltrace.set_context(patent_id="USTEST123", run_id="on_run")
        assert sys.getprofile() is not None
        _known_caller(5)

    assert tr.active is True
    summary = json.loads(tr.summary_path.read_text())
    assert summary["patents"] == ["USTEST123"]

    rows = {r["fn"]: r for r in summary["per_patent"]["USTEST123"]}
    # `_middle` calls `_leaf` twice, so the call counts are known exactly.
    assert rows["_known_caller"]["calls"] == 1
    assert rows["_middle"]["calls"] == 1
    assert rows["_leaf"]["calls"] == 2
    # file:line points at this module's own definitions.
    assert rows["_leaf"]["loc"].endswith(f":{_leaf.__code__.co_firstlineno}")
    assert rows["_leaf"]["module"] == "patentdb.tests.test_calltrace"
    # Inclusive time must contain self time, and a parent must contain a child.
    assert rows["_middle"]["incl_s"] >= rows["_middle"]["self_s"] >= 0.0
    assert rows["_known_caller"]["incl_s"] >= rows["_leaf"]["incl_s"]

    # The caller chain is recoverable from the edges without the event stream.
    edges = {
        (e["caller"].split(":")[-1], e["callee"].split(":")[-1]): e
        for e in summary["edges"]["USTEST123"]
    }
    assert edges[("_known_caller", "_middle")]["calls"] == 1
    assert edges[("_middle", "_leaf")]["calls"] == 2


def test_ndjson_events_carry_context_and_caller(monkeypatch, tmp_path):
    monkeypatch.setenv(calltrace.ENV_FLAG, "1")
    with calltrace.trace("ndjson_run", out_dir=tmp_path) as tr:
        calltrace.set_context(patent_id="USTEST999", run_id="ndjson_run")
        _known_caller(1)

    lines = [json.loads(x) for x in
             tr.events_path.read_text().splitlines() if x.strip()]
    leaves = [r for r in lines if r["fn"] == "_leaf"]
    assert len(leaves) == 2
    for r in leaves:
        assert r["pid"] == "USTEST999"
        assert r["run"] == "ndjson_run"
        assert r["mod"] == "patentdb.tests.test_calltrace"
        assert r["by"] == "_middle"          # caller, for the chain
        assert r["dur"] >= 0.0
        assert r["loc"].endswith(f":{_leaf.__code__.co_firstlineno}")


def test_event_cap_is_reported_not_silent(monkeypatch, tmp_path):
    """A truncated stream must say so — a cap that hides itself turns a
    partial record into an apparently complete one."""
    monkeypatch.setenv(calltrace.ENV_FLAG, "1")
    with calltrace.trace("capped", out_dir=tmp_path, max_events=1) as tr:
        calltrace.set_context(patent_id="USCAP", run_id="capped")
        _known_caller(1)
    assert tr.summary["events_written"] == 1
    assert tr.summary["events_dropped"] >= 3   # _leaf x2, _middle, _known_caller
    # Aggregates are unaffected by the cap — that is the point of having both.
    rows = {r["fn"]: r for r in tr.summary["per_patent"]["USCAP"]}
    assert rows["_leaf"]["calls"] == 2


def test_max_events_zero_writes_no_event_file(monkeypatch, tmp_path):
    monkeypatch.setenv(calltrace.ENV_FLAG, "1")
    with calltrace.trace("agg_only", out_dir=tmp_path, max_events=0) as tr:
        calltrace.set_context(patent_id="USAGG", run_id="agg_only")
        _known_caller(1)
    assert not tr.events_path.exists()
    assert tr.summary["per_patent"]["USAGG"]


# ── the filter ────────────────────────────────────────────────────

def test_filter_excludes_non_patentdb_frames(monkeypatch, tmp_path):
    """stdlib work must not appear. Tracing it is what produced gigabytes."""
    monkeypatch.setenv(calltrace.ENV_FLAG, "1")
    import json as _json
    with calltrace.trace("filt", out_dir=tmp_path, max_events=0) as tr:
        calltrace.set_context(patent_id="USFILT", run_id="filt")
        _json.dumps({"a": [1, 2, 3]})          # several stdlib Python frames
        _known_caller(1)

    mods = {r["module"] for r in tr.summary["per_patent"]["USFILT"]}
    assert mods == {"patentdb.tests.test_calltrace"}
    # And the tracer never traces itself (which would recurse into its own
    # aggregation on every event).
    assert "patentdb.core.calltrace" not in mods


def test_multiple_patents_are_bucketed_separately(monkeypatch, tmp_path):
    monkeypatch.setenv(calltrace.ENV_FLAG, "1")
    with calltrace.trace("multi", out_dir=tmp_path, max_events=0) as tr:
        calltrace.set_context(patent_id="USA", run_id="multi")
        _known_caller(1)
        calltrace.set_context(patent_id="USB", run_id="multi")
        _known_caller(1)
        _known_caller(1)

    per = tr.summary["per_patent"]
    assert {r["fn"]: r["calls"] for r in per["USA"]}["_leaf"] == 2
    assert {r["fn"]: r["calls"] for r in per["USB"]}["_leaf"] == 4


def test_exception_unwinding_does_not_desync_the_stack(monkeypatch, tmp_path):
    """`return` fires while an exception propagates; if that were missed the
    stack would drift and every later timing would be wrong."""
    monkeypatch.setenv(calltrace.ENV_FLAG, "1")

    def _raises():
        raise ValueError("boom")

    def _catches():
        try:
            _raises()
        except ValueError:
            return _leaf(1)

    with calltrace.trace("exc", out_dir=tmp_path, max_events=0) as tr:
        calltrace.set_context(patent_id="USEXC", run_id="exc")
        _catches()
        _known_caller(1)

    rows = {r["fn"]: r for r in tr.summary["per_patent"]["USEXC"]}
    # 1 from _catches + 2 from the _known_caller chain.
    assert rows["_leaf"]["calls"] == 3
    assert tr.summary["unmatched_returns"] == 0


def test_double_start_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv(calltrace.ENV_FLAG, "1")
    calltrace.start("first", out_dir=tmp_path)
    try:
        with pytest.raises(RuntimeError):
            calltrace.start("second", out_dir=tmp_path)
    finally:
        calltrace.stop()


def test_stop_without_start_is_a_noop():
    assert calltrace.stop() == {}
