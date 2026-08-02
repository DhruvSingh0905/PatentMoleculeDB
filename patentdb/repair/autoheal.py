"""Close the loop: a patent that yields nothing escalates itself to the code tier.

The three tiers were all built and only one was wired. `process_patent` called
`repair_patent` — the RULE tier — and stopped there. `repair_capabilities`, the
only tier that can fix a layout no rule kind can express, was reachable from two
eval CLIs and nothing else. So the pipeline could diagnose a total failure
precisely and then do nothing about it:

    US10266548   197 reference compounds, 242 records, every one missing `unit`
                 → 1 gap, `already_known`, 0 asked, 1 capability gap
                 → "value_pattern validated on this layout but produced no
                    records from TABLE-US-00048"
                 → discarded

Three separate breaks produced that, and each is fixed here or by the caller:

  the report was never read     `repair_report` appears on four lines of the
                                orchestrator; three are the assignment. It
                                reached no output file.
  the log was inverted          `if repair_report.rows_recovered:` — a patent
                                recovering 500 rows logged a summary, a patent
                                recovering ZERO logged nothing. The louder the
                                failure, the quieter the output.
  the tier was not connected    nothing ever called the capability tier
                                automatically.

WHY THIS IS BOUNDED. A capability patch is bought per FINGERPRINT and cached by
`PATCH_EPOCH`, so the corpus buys each shape once however many patents show it.
What is not free is `baseline_counts()`, which re-extracts every patent in the
corpus on each `repair_capabilities` call — so an auto-fire that ran per failing
patent would cost one full corpus scan each time. Hence `_attempted`, a
process-level set of fingerprints, and `AUTOHEAL_MAX_PER_RUN`. CLAUDE.md
measures three capability gaps in the entire corpus; the cap is a ceiling on
damage, not a target.

Nothing here decides whether a patch is GOOD. That judgement stays exactly where
it was — one condition, in `parser_repair.verify_patch`: does the patched reader
pick up fewer compounds than before. This module only decides WHEN to ask.
"""
from __future__ import annotations

import json
import logging
import threading

from ..core import config

logger = logging.getLogger(__name__)

# Fingerprints this process has already put to the code tier. Keyed by
# capability, not patent, because that is how the patch cache is keyed — twenty
# patents sharing a broken layout must buy one patch between them, not twenty.
_attempted: set[str] = set()
_spent = 0
_lock = threading.Lock()
# Re-entrancy guard. `repair_capabilities` → `collect_gaps` → `repair_patent`,
# and `verify_patch` runs `repair_patent` again over the corpus. None of those
# re-enter the orchestrator today, but a future caller that does would recurse
# into an unbounded spend, and the failure would look like a slow run.
_healing = threading.local()


def journal_escalations(patent_id: str, report) -> int:
    """Append every escalation and capability gap this patent raised.

    The rule tier journals what it ADOPTED. Nothing recorded what it could not
    fix — which is the half a human actually needs, and the half that names the
    missing capability. Written even when the code tier is disabled or capped,
    so the queue survives a run that chose not to spend.
    """
    rows = []
    for e in getattr(report, "escalations", []):
        rows.append({"patent": patent_id, "kind": "escalation",
                     "capability": e.get("capability"), "table": e.get("table"),
                     "fingerprint": e.get("fingerprint"),
                     "rows_at_stake": e.get("rows_at_stake"),
                     "note": str(e.get("note") or "")[:600]})
    for g in getattr(report, "capability_gaps", []):
        rows.append({"patent": patent_id, "kind": "capability_gap",
                     "capability": g.get("rule_kind"), "table": g.get("table"),
                     "fingerprint": g.get("fingerprint"),
                     "rows_at_stake": g.get("rows_at_stake"),
                     "note": str(g.get("why") or "")[:600],
                     "unparsed_examples": g.get("unparsed_examples") or []})
    for c in getattr(report, "crashed", []):
        rows.append({"patent": patent_id, "kind": "crash",
                     "capability": c.get("stage"), "table": c.get("table"),
                     "fingerprint": c.get("fingerprint"),
                     "rows_at_stake": c.get("rows_at_stake"),
                     "note": str(c.get("error") or "")[:600]})
    if not rows:
        return 0
    try:
        config.ESCALATION_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with config.ESCALATION_JOURNAL.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
    except OSError as e:                       # journalling must never break a run
        logger.warning("autoheal: could not journal escalations: %r", e)
    return len(rows)


def _wants_code_tier(report) -> list[dict]:
    """Gaps this patent raised that no RULE can close.

    Two sources, and both matter. `capability_gaps` is the precise one — a rule
    was available, ran, and produced nothing, so the blocker is below anything a
    rule can reach. `PATENT YIELDED NOTHING` is the blunt one, and it exists for
    the case the precise signal cannot see: US9018217 raised zero gaps at all,
    because the defect that emptied the table also emptied the evidence every
    per-table detector needs.
    """
    gaps = list(getattr(report, "capability_gaps", []))
    if gaps:
        return gaps
    blank = [e for e in getattr(report, "escalations", [])
             if e.get("capability") == "PATENT YIELDED NOTHING"]
    return blank


def maybe_escalate(patent_id: str, report) -> dict | None:
    """Journal what failed, and put it to the code tier if a rule cannot fix it.

    Returns the capability tier's report, or None when nothing was spent.
    """
    journal_escalations(patent_id, report)

    if not config.REPAIR_AUTOHEAL:
        return {"status": "disabled"}
    if getattr(_healing, "active", False):
        return {"status": "reentrant"}         # inside a patch verification
    wanted = _wants_code_tier(report)
    if not wanted:
        return {"status": "nothing_a_patch_could_fix"}

    # One key per capability. A `PATENT YIELDED NOTHING` escalation carries no
    # fingerprint (it is patent-level, not table-level), so it keys on the
    # patent — that failure genuinely is per-document until the code tier has
    # looked at it once.
    key = str(next((w["fingerprint"] for w in wanted if w.get("fingerprint")),
                   f"patent::{patent_id}"))
    global _spent
    with _lock:
        if key in _attempted:
            logger.info("autoheal: %s already put to the code tier this run", key)
            return {"status": "capability_already_bought", "key": key}
        if _spent >= config.AUTOHEAL_MAX_PER_RUN:
            logger.warning(
                "autoheal: %s needs the code tier but this run has spent its "
                "budget (%d); it is journaled in %s and will be picked up by "
                "`capability_repair --repair`",
                patent_id, config.AUTOHEAL_MAX_PER_RUN,
                config.ESCALATION_JOURNAL.name)
            return {"status": "budget_spent", "key": key,
                    "rows_at_stake": max((w.get("rows_at_stake") or 0)
                                         for w in wanted)}
        _attempted.add(key)
        _spent += 1

    at_stake = max((w.get("rows_at_stake") or 0) for w in wanted)
    logger.warning("autoheal: %s yielded nothing a rule can fix (%d rows at "
                   "stake) — escalating to the code tier", patent_id, at_stake)
    _healing.active = True
    try:
        from .capability import repair_capabilities
        rep = repair_capabilities(patent_ids=[patent_id], limit=1)
    except Exception as e:
        # Healing is additive. It must never break a run that would otherwise
        # have produced results — the same contract the rule tier has in
        # `process_patent`.
        logger.warning("autoheal: code tier failed for %s (%r)", patent_id, e)
        return {"status": "code_tier_raised", "key": key, "error": repr(e)[:300]}
    finally:
        _healing.active = False

    logger.warning("autoheal: %s — %d gap(s), %d applied, %d declined",
                   patent_id, rep.get("gaps", 0), rep.get("applied", 0),
                   rep.get("declined", 0))
    for r in rep.get("results", []):
        logger.warning("autoheal:   %s %s — %s",
                       "APPLIED" if r.get("ok") else "declined",
                       r.get("target"), str(r.get("why") or "")[:200])
    rep["status"] = "ran"
    rep["key"] = key
    return rep


def reset() -> None:
    """Forget this process's spend. For tests and for long-lived callers."""
    global _spent
    with _lock:
        _attempted.clear()
        _spent = 0
