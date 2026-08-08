"""Every place a compound, name or structure can fail to reach the dump —
recorded, never just counted.

WHY THIS EXISTS
----------------
Every real defect found in this pipeline so far shared one shape: a compound
was dropped and NOTHING recorded that it happened. Dedup-on-InChIKey erased
15 structures on one patent because a Markush name and its concrete sibling
share a key. 53 names were invisible because `description_text` read only
`<p>`. 143 structures carried no compound number and nothing said which ones.
`_opsin()`'s except branch quietly returned `[""] * len(names)` and the run
looked like a patent with no names in it. All four were found by accident,
not by anything designed to catch them.

This module is that design: a single structured sink that every drop point in
`iupac_names.py`, `anchor.py` and `verify.py` writes to, with enough context —
what was dropped, at what position, why, and what it would have been — that a
human can act on the record without re-running anything.

WHAT THIS IS NOT
-----------------
Not a fix. Nothing here changes what `extract_names` returns, what `find_cid`
resolves, or what `verify.dump()` writes to `DUMP_PATH`/`STRUCT_PATH`. Every
call site wraps EXISTING control flow — the same condition that already
decided to keep or drop something decides it again, unchanged; a `record()`
call is added beside that decision, never inside it. If a count anywhere
moves after this module is wired in, that is a bug in the wiring, not a
change in judgement.

THE ARTIFACT
-------------
One JSONL file, `LOSS_LOG`, derived from `config.DUMP` (an existing,
already-canonical constant) rather than hardcoded or placed under a new
directory — it lives beside `DUMP`/`STRUCTURES`/`MANIFEST` in
`patentdb3/out/`, the same directory `verify.dump()` already treats as ITS
output area. `core/config.py` is another agent's file and this task cannot
edit it; the constant that belongs there, for whoever next touches it, is:

    LOSS_LOG = PACKAGE_ROOT / "out" / "loss_log.jsonl"   # beside STRUCTURES

placed in the "THE EXTRACTION ARTIFACTS" block next to `STRUCTURES`.

One row per event, one event per drop. Fields always present: `loss_type`
(closed vocabulary, see the emitters below), `patent_id`. Every other field is
specific to the type — see each call site for what it carries. No wall-clock
timestamp is written INTO a record: that would make two runs over the same
input produce byte-different files, which fails the determinism requirement
outright for no reader-facing benefit. The run-level "when" question is
already answered by `verify.py`'s `MANIFEST_PATH` (`written_at`), which this
module's caller points at the same log via a `loss_log` key — see
`verify.dump()`.

TRUNCATE ON FIRST WRITE, PER PROCESS. The file behaves exactly like
`DUMP_PATH`/`STRUCT_PATH`: fresh per run, not appended across runs, so a stale
log from a previous invocation can never be mistaken for the current one.
Concretely, the first `record()` call in a process opens `LOSS_LOG` in `"w"`
mode; every call after that, across however many patents a run processes,
appends to that SAME handle — this is what lets one multi-patent
`verify.dump()` call produce one log for the whole run without an explicit
"start of run" hook, and it is also why `reset()` exists: to let tests (and
any caller that wants a clean slate mid-process, e.g. between two profiling
passes) point the sink at a different path or force a fresh truncate without
restarting the interpreter.

COST. Every emitter is gated on the module-level `ENABLED` flag
(`LOG_LOSSES` env var, default ON — the user wants the logs). Call sites that
do non-trivial extra work FOR the log alone (an extra document scan, a second
regex pass) check `ENABLED` themselves before doing that work, not only
inside `record()`, so turning the flag off recovers the CPU too, not only the
IO. Measured overhead is reported in the task report, not here — this module
has no corpus to measure against on its own.

DETERMINISM. `record()`'s own serialisation sorts JSON keys and recursively
turns any `set` into a sorted list of its `str()`ed members before writing —
sets are the one place Python's iteration order is not guaranteed stable
across runs or interpreter versions, and nothing else in a record's payload
should ever come from one (every list a call site passes is already built in
a deterministic order — position order, insertion order — by the code that
built it).
"""
from __future__ import annotations

import atexit
import dataclasses
import json
import os
import threading
from pathlib import Path
from typing import Any

from ..core import config

# THE ONE SWITCH. Default ON: "rather have extraneous logs than not enough"
# is the explicit instruction this module exists to satisfy. Set
# `LOG_LOSSES=0` to silence it — e.g. for a timing comparison, or once the
# corpus-wide volume of a specific loss type is understood and not needed
# every run.
ENABLED = os.environ.get("LOG_LOSSES", "1") == "1"

# Derived from `config.DUMP`, not hardcoded, and not a new directory — see
# module docstring. `config.DUMP.parent` is `patentdb3/out/`, the directory
# `DUMP_PATH`/`STRUCT_PATH`/`MANIFEST_PATH` already write to.
LOSS_LOG: Path = config.DUMP.parent / "loss_log.jsonl"

_MAX_STR = 200      # chars kept for any single free-text field
_MAX_ITEMS = 50      # entries kept for any single list field

_lock = threading.RLock()
_fh: Any = None        # the open file handle, lazily created (typed Any: a
                        # real static file-object type here fights the
                        # module-global lazy-open pattern below for no
                        # benefit — every write site is guarded by an
                        # `is None` check first)
_path: Path = LOSS_LOG  # where `_fh` currently points
_ever_opened = False    # has `_path` been opened at all yet THIS process —
                        # see `_open_locked` for why this, not `_fh is None`,
                        # is what decides truncate-vs-append on a lazy reopen


def _truncate(s: str) -> str:
    if len(s) <= _MAX_STR:
        return s
    return s[:_MAX_STR] + f"…[+{len(s) - _MAX_STR} chars]"


def _normalize(v: Any) -> Any:
    """Make a field JSON-safe and deterministic. Recurses into containers.

    - `set` -> sorted list of `str()`ed members (the one non-deterministic
      container Python has; everything else here is already ordered by
      construction — see module docstring).
    - `list`/`tuple` -> normalized element-wise, capped at `_MAX_ITEMS` with
      a trailing marker so one record can never grow unbounded.
    - `dict` -> normalized value-wise, keys sorted (also handled by
      `json.dumps(sort_keys=True)` at the top level, but nested dicts need it
      done here since `sort_keys` only sorts each dict `json` itself writes —
      which it does, recursively, in the stdlib; kept explicit anyway so this
      function's output is stable independent of the caller).
    - dataclass instance -> its field dict, normalized.
    - `str` -> truncated.
    - anything else (int, float, bool, None) -> returned as-is.
    """
    if isinstance(v, set):
        return sorted(str(x) for x in v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _normalize(dataclasses.asdict(v))
    if isinstance(v, (list, tuple)):
        items = [_normalize(x) for x in v]
        if len(items) > _MAX_ITEMS:
            items = items[:_MAX_ITEMS] + [f"...+{len(v) - _MAX_ITEMS} more"]
        return items
    if isinstance(v, dict):
        return {str(k): _normalize(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, str):
        return _truncate(v)
    return v


def _open_locked(path: Path, *, truncate: bool) -> None:
    """Open `path`, closing whatever was open before. Caller must already
    hold `_lock`.

    `truncate` is the whole reason this takes a keyword rather than always
    truncating: `reset()` always wants a fresh file (an explicit clean
    slate), but `record()`'s own LAZY reopen — reached when something has
    already called `close()` mid-process and then logs again — must NOT
    truncate, or every record written before that `close()` is destroyed.
    `_ever_opened` is what tells the two cases apart: false only before
    `_path` has been opened for the very first time in this process, which
    is the ONE moment "TRUNCATE ON FIRST WRITE, PER PROCESS" (see module
    docstring) actually means.
    """
    global _fh, _path, _ever_opened
    if _fh is not None:
        try:
            _fh.close()
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    _fh = path.open("w" if truncate else "a")
    _path = path
    _ever_opened = True


def reset(path: "Path | None" = None) -> Path:
    """Start a fresh log at `path` (default `LOSS_LOG`), truncating it —
    ALWAYS, regardless of whether `path` (or some other path) was already
    open in this process. This is the one deliberate way to force a clean
    slate; `record()`'s own lazy reopen after a `close()` never truncates
    (see `_open_locked`).

    `record()` calls this lazily on its own first invocation in a process, so
    a normal run (one `verify.dump()` call, one test session) never needs to
    call it explicitly. It exists for a caller that wants an explicit clean
    slate — chiefly `tests/test_losses.py`, which redirects to a temp path
    per test so tests cannot see each other's records or the real corpus log.
    """
    with _lock:
        _open_locked(path or LOSS_LOG, truncate=True)
        atexit.register(close)
    return _path


def record(loss_type: str, patent_id: str, **fields: Any) -> None:
    """Append one structured loss record. No-op when `ENABLED` is False.

    `loss_type` and `patent_id` are always present; everything else is
    whatever the call site knows about this specific drop — position, the
    candidate/name text involved, why, and (where meaningful) what would have
    been kept instead. See the emitters in `iupac_names.py`, `anchor.py` and
    `verify.py` for the concrete schema each `loss_type` carries.
    """
    if not ENABLED:
        return
    row: dict[str, Any] = {"loss_type": loss_type, "patent_id": patent_id}
    for k, v in fields.items():
        row[k] = _normalize(v)
    line = json.dumps(row, sort_keys=True, default=str, ensure_ascii=True)
    with _lock:
        if _fh is None:
            # `not _ever_opened` is true only the very first time ANY call
            # opens `_path` in this process — see `_open_locked`. A `close()`
            # earlier in the same process must not turn a later `record()`
            # into a silent truncate of what is already on disk.
            _open_locked(_path, truncate=not _ever_opened)
            atexit.register(close)
        _fh.write(line + "\n")


def flush() -> None:
    """Push whatever is buffered out to disk, WITHOUT closing the handle.

    `record()` never flushes per line (that measurably costs corpus-run
    time for no reader-facing benefit — see the module docstring's COST
    section); Python's own file buffering means a reader opening the same
    path in a SEPARATE handle — another process, or a test reading the file
    back mid-run — can see fewer lines than have actually been `record()`ed
    until something flushes. This is that "something".

    `close()` also flushes (see below) and recording again afterward is
    SAFE too — `_open_locked`'s `truncate=not _ever_opened` guard means a
    lazy reopen after a mid-run `close()` appends, it does not truncate —
    but `flush()` is the cheaper choice for a mid-run peek: no file
    descriptor is torn down and reopened, `_fh` never becomes `None`, and a
    concurrent `record()` call from another thread never has to wait on a
    reopen.
    """
    with _lock:
        if _fh is not None:
            _fh.flush()


def close() -> None:
    """Flush and close the current handle. Safe to call more than once
    (registered with `atexit`, and callable directly — e.g. at the end of a
    `verify.dump()` run, so the file is durable before anything reads it back
    for a summary in the same process).

    Recording again after `close()` is safe — see `_open_locked` — but
    `flush()` is the cheaper choice when the only goal is making sure a
    concurrent reader sees everything written so far.
    """
    global _fh
    with _lock:
        if _fh is not None:
            try:
                _fh.flush()
                _fh.close()
            finally:
                _fh = None


def summary(path: "Path | None" = None) -> dict[str, int]:
    """Counts by `loss_type` for the log at `path` (default: wherever this
    process is currently writing, flushed first so a summary taken mid-run
    sees everything written so far). Read-only; does not disturb the sink —
    a caller who takes a summary and keeps recording afterward gets a log
    that still has everything, summary included.
    """
    flush()
    with _lock:
        p = path or _path
    counts: dict[str, int] = {}
    if not p.exists():
        return counts
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            lt = row.get("loss_type", "unknown")
            counts[lt] = counts.get(lt, 0) + 1
    return dict(sorted(counts.items()))
