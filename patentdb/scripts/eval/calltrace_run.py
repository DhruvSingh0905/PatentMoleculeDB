"""Run `process_patent` under the call tracer and rank what ran.

    python3 -m patentdb.scripts.eval.calltrace_run --patents US10214537
    python3 -m patentdb.scripts.eval.calltrace_run --patents US8952177 --overhead

This is the driver for `core/calltrace.py`. It calls **production**
`process_patent()` — per CLAUDE.md's wiring rule, a benchmark that reimplements
the pipeline measures itself, so nothing here re-derives extraction.

ZERO paid calls
---------------
`anthropic.Anthropic` is replaced by a raiser before the first import of the
pipeline, so a leaked API call is a traceback naming its caller rather than a
silent charge. `--allow-paid` removes the raiser; it exists only so the flag is
a decision someone typed.

Recommended invocation — the flags a $0 corpus run uses:

    MINERU_OCR=0 HARVEST_BURST=0 LLM_NAME_REPAIR=0 STRATEGY5_LLM=0 \
    IUPAC_BURST=0 LLM_RECOVERY=0 CALLTRACE=1 \
    python3 -m patentdb.scripts.eval.calltrace_run --patents US10214537

The effective value of every one of those flags is read back off
`core.config` and written into the summary, so the run's configuration is
recoverable from the artifact instead of from the shell history that produced
it.

Overhead
--------
`--overhead` times the SAME patent three times in a row: warm-up (untraced),
baseline (untraced), traced. The warm-up exists because every cache in this
pipeline is a file on disk — comparing a cold first run against a warm second
one measures the cache, not the profiler.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path


class PaidCallBlocked(Exception):
    """Raised in place of every outbound Anthropic request."""


# Where the blocked requests came from: "module:function:line" -> count. This
# is evidence in its own right — it names every LLM path that still fires under
# the `$0 baseline` flags, which is a different question from what the flags
# claim to disable.
BLOCKED_CALLS: dict[str, int] = {}


class _Blocker:
    """Stands in for `anthropic.Anthropic`. Attribute access is free; CALLING
    anything raises.

    Blocking the CONSTRUCTOR instead (the obvious version, and the first one
    tried here) aborts the run: `api_client._get_client()` sits OUTSIDE the
    `try` in `call_claude_text` (core/api_client.py:160), so the error escapes
    a function whose contract is to return None on failure, and the trace stops
    at the first LLM path rather than covering the pipeline.

    Failing at the transport instead reproduces exactly what an unreachable API
    does: `_call_api` raises inside the try, `call_claude_text` logs
    "Permanent API failure" and returns None, and the run completes. Still zero
    paid calls — no client is ever constructed and no socket is opened.
    """

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Blocker()

    def __call__(self, *a, **k):
        import sys as _sys
        f = _sys._getframe(1)
        site = f"{f.f_globals.get('__name__', '?')}:{f.f_code.co_name}:{f.f_lineno}"
        BLOCKED_CALLS[site] = BLOCKED_CALLS.get(site, 0) + 1
        raise PaidCallBlocked(
            f"PAID CALL BLOCKED at {site} — calltrace_run runs at $0. "
            f"Pass --allow-paid if a real call is intended."
        )


def _block_paid_calls() -> None:
    import anthropic
    anthropic.Anthropic = _Blocker            # type: ignore[assignment]


_FLAGS = [
    "MINERU_OCR_ENABLED", "HARVEST_BURST_ENABLED", "LLM_NAME_REPAIR_ENABLED",
    "STRATEGY5_LLM_ENABLED", "IUPAC_BURST_ENABLED", "LLM_RECOVERY_ENABLED",
    "LLM_BATCH_ENABLED", "HARVEST_SKIP_ENABLED", "SUBSTITUENT_LLM_ENABLED",
    "REPAIR_ENABLED", "REPAIR_AUTOHEAL", "PARSER_REPAIR_APPLY",
    "SNAPSHOT_FREEZE_ENABLED", "RULE_GATES_ENFORCE",
]


def _effective_flags() -> dict:
    from patentdb.core import config
    return {name: getattr(config, name, None) for name in _FLAGS}


def _run_one(patent_id: str, *, write: bool) -> tuple[float, dict]:
    """Time one production `process_patent` call. Returns (seconds, counts)."""
    from patentdb.routes.process_patent import process_patent
    t0 = time.perf_counter()
    res = process_patent(patent_id, write=write)
    dt = time.perf_counter() - t0
    return dt, {
        "compounds": len(res.example_index),
        "assay_cids": len(res.assay_tables),
        "route_class": res.route_class,
        "cost_usd": res.cost_usd,
    }


def cmd_trace(args) -> int:
    from patentdb.core import calltrace

    run_id = args.run_id or time.strftime("trace_%Y%m%d_%H%M%S")
    if not calltrace.enabled() and not args.force:
        print("CALLTRACE is not 1 — nothing would be recorded. "
              "Re-run with CALLTRACE=1 (or pass --force).")
        return 2

    results = {}
    tr = calltrace.trace(run_id, max_events=args.max_events, force=args.force)
    with tr:
        for pid in args.patents:
            calltrace.set_context(patent_id=pid, run_id=run_id)
            print(f"[calltrace] {pid} …", flush=True)
            try:
                dt, counts = _run_one(pid, write=not args.no_write)
                results[pid] = {"seconds": round(dt, 3), **counts}
                print(f"[calltrace] {pid}: {dt:.1f}s "
                      f"{counts['compounds']} compounds", flush=True)
            except Exception as e:              # noqa: BLE001 — recorded, not swallowed
                results[pid] = {"error": repr(e)}
                print(f"[calltrace] {pid}: FAILED {e!r}", flush=True)

    summary = tr.summary
    summary["traced_runs"] = results
    summary["config_flags"] = _effective_flags()
    summary["blocked_paid_calls"] = dict(
        sorted(BLOCKED_CALLS.items(), key=lambda kv: -kv[1])
    )
    tr.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"events   : {tr.events_path}  "
          f"({summary['events_written']:,} written, "
          f"{summary['events_dropped']:,} dropped)")
    print(f"summary  : {tr.summary_path}")
    print(f"functions: {summary['n_distinct_functions']:,} distinct, "
          f"{summary['total_calls']:,} calls in "
          f"{summary['wall_seconds']:.1f}s traced wall")
    if BLOCKED_CALLS:
        print(f"blocked  : {sum(BLOCKED_CALLS.values())} outbound LLM request(s) "
              f"from {len(BLOCKED_CALLS)} site(s):")
        for site, n in summary["blocked_paid_calls"].items():
            print(f"             {n:6d}  {site}")
    return 0


def cmd_overhead(args) -> int:
    """warm-up (untraced) → baseline (untraced) → traced, same patent."""
    from patentdb.core import calltrace

    pid = args.patents[0]
    print(f"[overhead] warm-up (untraced) {pid} …", flush=True)
    warm, _ = _run_one(pid, write=False)
    print(f"[overhead] warm-up  : {warm:.2f}s", flush=True)

    print(f"[overhead] baseline (untraced) {pid} …", flush=True)
    base, base_counts = _run_one(pid, write=False)
    print(f"[overhead] baseline : {base:.2f}s", flush=True)

    run_id = (args.run_id or time.strftime("overhead_%Y%m%d_%H%M%S"))
    tr = calltrace.trace(run_id, max_events=args.max_events, force=True)
    with tr:
        calltrace.set_context(patent_id=pid, run_id=run_id)
        traced, traced_counts = _run_one(pid, write=False)
    print(f"[overhead] traced   : {traced:.2f}s", flush=True)

    ratio = traced / base if base else float("nan")
    out = {
        "patent_id": pid,
        "warmup_s": round(warm, 3),
        "baseline_s": round(base, 3),
        "traced_s": round(traced, 3),
        "overhead_x": round(ratio, 3),
        "overhead_pct": round(100.0 * (ratio - 1.0), 1),
        "max_events": args.max_events,
        "events_written": tr.summary.get("events_written"),
        "total_calls": tr.summary.get("total_calls"),
        "baseline_counts": base_counts,
        "traced_counts": traced_counts,
        "config_flags": _effective_flags(),
    }
    path = calltrace.OUT_DIR / f"{run_id}.overhead.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(json.dumps(out, indent=2))
    print(f"\nwritten: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", required=True,
                    help="comma-separated patent ids")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-events", type=int, default=None,
                    help="cap on per-call NDJSON records (0 = aggregates only). "
                         "Default: CALLTRACE_EVENTS or 200000.")
    ap.add_argument("--no-write", action="store_true",
                    help="do not overwrite output_v2/text_extraction/{pid}/")
    ap.add_argument("--force", action="store_true",
                    help="trace even when CALLTRACE is unset")
    ap.add_argument("--overhead", action="store_true",
                    help="measure tracer overhead on the first --patents entry")
    ap.add_argument("--allow-paid", action="store_true",
                    help="do NOT block anthropic.Anthropic")
    ap.add_argument("--log-level", default="INFO",
                    help="root log level. CLAUDE.md: absence of a log line is "
                         "not evidence unless the level could have emitted it.")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.patents = [p.strip() for p in args.patents.split(",") if p.strip()]
    if args.max_events is None:
        args.max_events = int(os.environ.get("CALLTRACE_EVENTS", "200000"))

    if not args.allow_paid:
        _block_paid_calls()

    return cmd_overhead(args) if args.overhead else cmd_trace(args)


if __name__ == "__main__":
    raise SystemExit(main())
