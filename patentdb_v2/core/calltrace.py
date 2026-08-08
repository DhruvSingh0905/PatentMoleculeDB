"""Whole-pipeline call tracer. OFF unless ``CALLTRACE=1``.

Why a profiler and not log lines
--------------------------------
The question this answers — *which functions run that should not* — needs a
record of every call in the package, with its caller and its wall time. The
obvious implementation, a log line at the top of each function, is thousands of
edits to production code and becomes its own defect source. ``sys.setprofile``
gets the identical record with **zero** production edits: CPython calls the hook
on every Python-level ``call`` and ``return``, so nothing has to opt in.

What is recorded
----------------
One aggregate row per ``(patent_id, function)``:

    calls, inclusive wall seconds, self wall seconds, slowest single call

and one edge row per ``(patent_id, caller, callee)`` so a caller chain is
recoverable without re-reading the event stream. The per-call NDJSON stream is
written alongside (``{run_id}.jsonl``) and is capped — see ``CALLTRACE_EVENTS``.

The filter
----------
A frame is traced iff ``frame.f_code.co_filename`` sits under the ``patentdb/``
package directory, and is not this file. That is the whole rule. It excludes the
stdlib, rdkit, anthropic, requests, py2opsin's subprocess plumbing and every
other dependency — tracing those produced gigabytes and buried the signal.

Two consequences of the filter that the numbers must be read with:

* **Self time absorbs foreign work.** ``c_call``/``c_return`` events are
  ignored, so time spent inside rdkit, a ``subprocess`` for OPSIN, or an HTTP
  round-trip lands in the *self* time of the nearest enclosing ``patentdb``
  function. A function with large self time and few internal calls is doing
  external work; that is a signal, not an artifact.
* **Inclusive time double-counts under recursion.** A function that calls
  itself has each activation's inclusive time summed. ``calls`` is exact.

Generators resume through a ``call`` event and yield through a ``return``
event, so one generator consumed N times counts as N calls.

Context
-------
``set_context(patent_id=..., run_id=...)`` tags every subsequent record. The
fast path reads a module-level global (a ``ContextVar`` lookup per call event is
real overhead at these volumes); the ContextVar mirror exists for
``get_context()`` and is authoritative for nobody. That makes context
process-wide rather than per-thread — correct for this pipeline, which
processes one patent at a time, and stated here rather than assumed.

Usage
-----
    from patentdb.core import calltrace

    with calltrace.trace(run_id="my_run") as tr:
        calltrace.set_context(patent_id="US8952177")
        process_patent("US8952177")
    print(tr.summary_path)

Nothing in the live pipeline imports this module; it is driven from
``patentdb.scripts.eval.calltrace_run``.
"""
from __future__ import annotations

import contextvars
import json
import os
import sys
import threading
import time
from pathlib import Path

__all__ = [
    "enabled", "set_context", "get_context", "start", "stop", "trace",
    "Tracer", "OUT_DIR",
]

# ── Where output lands ────────────────────────────────────────────
# Not via core.config: importing config has the side effect of creating
# output directories, and a tracer must be importable without changing the
# tree it is measuring.
_PKG_ROOT = Path(__file__).resolve().parent.parent          # .../patentdb
_REPO_ROOT = _PKG_ROOT.parent
OUT_DIR = _REPO_ROOT / "output_v2" / "calltrace"

_PKG_PREFIX = str(_PKG_ROOT) + os.sep
_SELF_FILE = str(Path(__file__).resolve())

ENV_FLAG = "CALLTRACE"
ENV_MAX_EVENTS = "CALLTRACE_EVENTS"
DEFAULT_MAX_EVENTS = 200_000

FILTER_DESCRIPTION = (
    f"frame.f_code.co_filename startswith {_PKG_PREFIX!r} and != this module"
)


def enabled() -> bool:
    """True when ``CALLTRACE=1``. Read at call time, never cached."""
    return os.environ.get(ENV_FLAG, "") == "1"


# ── Module-level state. Globals, not instance attributes: the callback
#    runs millions of times and an attribute lookup per event is not free.
_active = False
_run_id = ""
_t_origin = 0.0

_meta: dict = {}            # code object -> fid (>=0) | -2 (excluded)
# fid -> (qualname, module, relpath, firstlineno, json_fragment, qualname_json).
# The last two are pre-rendered because json.dumps in the hot path costs more
# than the trace is worth: five calls per event over millions of events.
_funcs: list = []
_pid_frag = '"run":"","pid":""'
_agg: dict = {}             # (pidx, fid) -> [calls, incl, self, max_incl]
_edges: dict = {}           # (pidx, caller_fid, fid) -> [calls, incl]
_patents: list = []         # pidx -> patent_id
_pidx = 0

_n_events = 0
_max_events = 0
_dropped_events = 0
_unmatched_returns = 0
_buf: list = []
_fh = None

_UNSEEN = -1
_EXCLUDED = -2

_perf = time.perf_counter

_ctx_patent: contextvars.ContextVar = contextvars.ContextVar(
    "calltrace_patent", default="",
)
_ctx_run: contextvars.ContextVar = contextvars.ContextVar(
    "calltrace_run", default="",
)


class _TLS(threading.local):
    def __init__(self):
        self.stack: list = []


_tls = _TLS()


def set_context(patent_id: str | None = None, run_id: str | None = None) -> None:
    """Tag every subsequent record with this patent / run.

    Safe to call when the tracer is off — it is then a two-assignment no-op,
    which is what lets a runner call it unconditionally.
    """
    global _pidx, _run_id, _pid_frag
    if patent_id is not None:
        _ctx_patent.set(patent_id)
        try:
            _pidx = _patents.index(patent_id)
        except ValueError:
            _patents.append(patent_id)
            _pidx = len(_patents) - 1
    if run_id is not None:
        _ctx_run.set(run_id)
        _run_id = run_id
    _pid_frag = '"run":%s,"pid":%s' % (
        json.dumps(_run_id), json.dumps(_ctx_patent.get()),
    )


def get_context() -> dict:
    """The current ``{patent_id, run_id}``."""
    return {"patent_id": _ctx_patent.get(), "run_id": _ctx_run.get()}


def _classify(code) -> int:
    """Assign a function id to ``code``, or mark it excluded. Once per code
    object, never per call."""
    fn = code.co_filename
    ok = fn.startswith(_PKG_PREFIX) and fn != _SELF_FILE
    if not ok:
        # A module imported through a relative path has a relative
        # co_filename; resolve once before giving up on it.
        try:
            rp = os.path.realpath(fn)
        except (OSError, ValueError):
            rp = fn
        ok = rp.startswith(_PKG_PREFIX) and rp != _SELF_FILE
        if ok:
            fn = rp
    if not ok:
        _meta[code] = _EXCLUDED
        return _EXCLUDED
    rel = fn[len(str(_REPO_ROOT)) + 1:] if fn.startswith(str(_REPO_ROOT)) else fn
    mod = rel[:-3].replace(os.sep, ".") if rel.endswith(".py") else rel
    qual = getattr(code, "co_qualname", None) or code.co_name
    fid = len(_funcs)
    qual_json = json.dumps(qual)
    frag = '"fn":%s,"mod":%s,"loc":%s' % (
        qual_json, json.dumps(mod), json.dumps(f"{rel}:{code.co_firstlineno}"),
    )
    _funcs.append((qual, mod, rel, code.co_firstlineno, frag, qual_json))
    _meta[code] = fid
    return fid


def _profile(frame, event, arg):
    """The hook. Hot path — kept to dict lookups and list ops on purpose."""
    if event == "call":
        code = frame.f_code
        fid = _meta.get(code, _UNSEEN)
        if fid < 0:
            if fid == _UNSEEN:
                fid = _classify(code)
            if fid < 0:
                return
        _tls.stack.append([frame, fid, _perf(), 0.0])
        return

    if event != "return":
        # c_call / c_return / c_exception — deliberately unrecorded; see
        # the module docstring on where that time lands.
        return

    st = _tls.stack
    if not st:
        return
    top = st[-1]
    if top[0] is not frame:
        # A frame whose `call` predates the hook's installation. Counted so
        # that a desync is visible in the summary rather than assumed absent.
        global _unmatched_returns
        _unmatched_returns += 1
        return
    st.pop()
    t1 = _perf()
    incl = t1 - top[2]
    self_t = incl - top[3]
    fid = top[1]
    if st:
        parent = st[-1]
        parent[3] += incl
        caller_fid = parent[1]
    else:
        caller_fid = -1

    k = (_pidx, fid)
    a = _agg.get(k)
    if a is None:
        _agg[k] = [1, incl, self_t, incl]
    else:
        a[0] += 1
        a[1] += incl
        a[2] += self_t
        if incl > a[3]:
            a[3] = incl

    ek = (_pidx, caller_fid, fid)
    e = _edges.get(ek)
    if e is None:
        _edges[ek] = [1, incl]
    else:
        e[0] += 1
        e[1] += incl

    global _n_events, _dropped_events
    if _n_events < _max_events:
        _buf.append(
            '{%s,%s,"t":%.6f,"dur":%.6f,"self":%.6f,"depth":%d,"by":%s}'
            % (
                _pid_frag,
                _funcs[fid][4],
                top[2] - _t_origin,
                incl,
                self_t,
                len(st),
                _funcs[caller_fid][5] if caller_fid >= 0 else "null",
            )
        )
        _n_events += 1
        if len(_buf) >= 4096:
            _flush_buf()
    else:
        _dropped_events += 1


def _flush_buf() -> None:
    global _buf
    if _buf and _fh is not None:
        _fh.write("\n".join(_buf))
        _fh.write("\n")
    _buf = []


class Tracer:
    """Handle returned by :func:`start` / :func:`trace`.

    ``active`` is False when ``CALLTRACE`` is unset — every other attribute is
    still present so a caller never has to branch.
    """

    def __init__(self, run_id: str, out_dir: Path, active: bool):
        self.run_id = run_id
        self.out_dir = out_dir
        self.active = active
        self.events_path = out_dir / f"{run_id}.jsonl"
        self.summary_path = out_dir / f"{run_id}.summary.json"
        self.summary: dict = {}
        self.wall_seconds = 0.0

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc) -> bool:
        summary = stop()
        if self.active:
            self.summary = summary
            self.wall_seconds = summary.get("wall_seconds", 0.0)
        return False


_LAST_SUMMARY: dict = {}
_current: Tracer | None = None


def start(
    run_id: str | None = None,
    *,
    out_dir: Path | None = None,
    max_events: int | None = None,
    force: bool = False,
) -> Tracer:
    """Install the profile hook. No-op (returning an inactive handle) unless
    ``CALLTRACE=1`` or ``force=True``.

    ``force`` exists for the tracer's own test, which must be able to prove the
    hook records a known call without the ambient environment deciding.
    """
    global _active, _t_origin, _max_events, _fh, _current, _run_id
    global _meta, _funcs, _agg, _edges, _patents, _pidx, _pid_frag
    global _n_events, _dropped_events, _unmatched_returns, _buf

    run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR

    if _active:
        raise RuntimeError("calltrace.start(): a trace is already running")
    if not (force or enabled()):
        return Tracer(run_id, out_dir, active=False)

    out_dir.mkdir(parents=True, exist_ok=True)

    _meta = {}
    _funcs = []
    _agg = {}
    _edges = {}
    _patents = []
    _pidx = 0
    _n_events = 0
    _dropped_events = 0
    _unmatched_returns = 0
    _buf = []
    _run_id = run_id
    _ctx_run.set(run_id)
    _ctx_patent.set("")
    _pid_frag = '"run":%s,"pid":""' % json.dumps(run_id)
    if max_events is None:
        max_events = int(os.environ.get(ENV_MAX_EVENTS, DEFAULT_MAX_EVENTS))
    _max_events = max_events

    tr = Tracer(run_id, out_dir, active=True)
    _fh = open(tr.events_path, "w", encoding="utf-8") if max_events > 0 else None
    _current = tr
    _tls.stack = []
    _t_origin = _perf()
    _active = True
    threading.setprofile(_profile)
    sys.setprofile(_profile)
    return tr


def stop() -> dict:
    """Uninstall the hook, write the aggregates, return the summary dict.

    Returns ``{}`` when no trace was running, so a caller can stop
    unconditionally.
    """
    global _active, _fh, _current, _LAST_SUMMARY
    if not _active:
        return {}
    sys.setprofile(None)
    threading.setprofile(None)
    _active = False
    wall = _perf() - _t_origin
    _flush_buf()
    if _fh is not None:
        _fh.close()
        _fh = None

    tr = _current
    summary = _build_summary(wall)
    if tr is not None:
        tr.summary_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
    _LAST_SUMMARY = summary
    _current = None
    _tls.stack = []
    return summary


def _build_summary(wall: float) -> dict:
    per_patent: dict = {}
    for (pidx, fid), (n, incl, self_t, mx) in _agg.items():
        pid = _patents[pidx] if pidx < len(_patents) else ""
        qual, mod, rel, line = _funcs[fid][:4]
        per_patent.setdefault(pid, []).append({
            "fn": qual,
            "module": mod,
            "loc": f"{rel}:{line}",
            "calls": n,
            "incl_s": round(incl, 6),
            "self_s": round(self_t, 6),
            "max_call_s": round(mx, 6),
        })
    for rows in per_patent.values():
        rows.sort(key=lambda r: -r["incl_s"])

    edges: dict = {}
    for (pidx, cfid, fid), (n, incl) in _edges.items():
        pid = _patents[pidx] if pidx < len(_patents) else ""
        caller = "<root>" if cfid < 0 else (
            f"{_funcs[cfid][1]}:{_funcs[cfid][0]}"
        )
        edges.setdefault(pid, []).append({
            "caller": caller,
            "callee": f"{_funcs[fid][1]}:{_funcs[fid][0]}",
            "calls": n,
            "incl_s": round(incl, 6),
        })
    for rows in edges.values():
        rows.sort(key=lambda r: -r["incl_s"])

    return {
        "run_id": _run_id,
        "wall_seconds": round(wall, 4),
        "python": sys.version.split()[0],
        "filter": FILTER_DESCRIPTION,
        "patents": list(_patents),
        "n_distinct_functions": len(_funcs),
        "total_calls": sum(v[0] for v in _agg.values()),
        "events_written": _n_events,
        "events_dropped": _dropped_events,
        "max_events": _max_events,
        "unmatched_returns": _unmatched_returns,
        "per_patent": per_patent,
        "edges": edges,
    }


def trace(
    run_id: str | None = None,
    *,
    out_dir: Path | None = None,
    max_events: int | None = None,
    force: bool = False,
) -> Tracer:
    """Context-manager form of :func:`start` / :func:`stop`."""
    return start(run_id, out_dir=out_dir, max_events=max_events, force=force)
