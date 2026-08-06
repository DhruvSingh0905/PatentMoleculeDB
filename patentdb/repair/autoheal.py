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
Hence `_attempted`, a process-level set of fingerprints, and
`AUTOHEAL_MAX_PER_RUN`. CLAUDE.md measures three capability gaps in the entire
corpus; the cap is a ceiling on damage, not a target.

WHAT IT COSTS, measured. On US20240010684A1 — 15 compounds, a gap worth 3 rows —
this call was 74.68 s of a 154.65 s traced run (48.3%), 39.53 s untraced. Two
things dominated and only one still does:

    baseline_counts   14.04 s  re-extracted all 137 cached XMLs, every call
                               -> now REMEMBERED per patent (repair/ledger.py)
                                  and re-measured only when the extraction code
                                  moves; the probe below keeps it fresh across
                                  a landed patch, so it costs a file read
    verify_patch      22.49 s  the corpus probe (16.59 s) plus the full test
                               suite (5.82 s), in a scratch copy of the tree

`verify_patch` is the irreducible half and is deliberately untouched: it is the
only thing that measures OTHER patents under the patch, and a capability patch
is a global code change. US10660877 went 860 -> 0 on a patch that touched none
of its rows. Scoping that probe to the patent being repaired would make the tier
cheap and blind at the same time.

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
    if blank:
        return blank
    # A patent can extract plenty and still be WRONG, and none of the signals
    # above will say so — they all score rows produced. An implausible result
    # is a code gap of the same kind: a replicate count the reader drops, a
    # header whose unit did not survive the merge. Ranked below the two above
    # because a patent that yielded nothing is the more urgent failure.
    return [e for e in getattr(report, "escalations", [])
            if str(e.get("capability", "")).startswith("IMPLAUSIBLE:")]


def freeze_result(patent_id: str, records) -> None:
    """Pin this patent's answer so a later patch cannot move it.

    Called once the patent has been through the loop. Everything after this
    point — a capability patch bought for some other document, a widened
    predicate, a new rule — is judged against patents that are NOT yet frozen,
    because those are the only ones it can still help or harm.

    Measured, this is the coupling that stopped every patch landing: one
    `_is_namelike` change cost US10660877 all 860 of its compounds without
    touching a row of it, by changing how a block derived its own header.
    """
    from .snapshot import freeze, is_frozen

    if is_frozen(patent_id):
        return
    try:
        freeze(patent_id, records)
    except Exception as e:                       # freezing must never break a run
        logger.warning("autoheal: could not freeze %s (%r)", patent_id, e)


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
        # Hand over the report we were given. `collect_gaps` used to re-run
        # `repair_patent` on this same document to re-derive it — and with
        # `max_calls=0`, so a patent whose pipeline run bought a rule was
        # re-examined as though it had not. The tier could then act on a
        # different set of gaps than the one `_wants_code_tier` escalated for.
        #
        # `apply` IS NOT PASSED, deliberately. This is the pipeline, and the
        # pipeline did not ask for its reader to be rewritten — it asked for an
        # extraction. `repair_capabilities` reads the omission as "nobody said"
        # and defers to `guard`, which refuses the write, logs it loudly, and
        # journals the verified proposal for `parser_health --force <id>`.
        # This call used to resolve to `config.PARSER_REPAIR_APPLY` (default 1)
        # and rewrite `sources/uspto_assays.py` mid-run; two agents lost
        # measurements to it. Diagnosis is unchanged — only the write is gated.
        rep = repair_capabilities(patent_ids=[patent_id], limit=1,
                                  reports={patent_id: report})
    except Exception as e:
        # Healing is additive. It must never break a run that would otherwise
        # have produced results — the same contract the rule tier has in
        # `process_patent`.
        logger.warning("autoheal: code tier failed for %s (%r)", patent_id, e)
        return {"status": "code_tier_raised", "key": key, "error": repr(e)[:300]}
    finally:
        _healing.active = False

    logger.warning("autoheal: %s — %d gap(s), %d applied, %d proposed (not "
                   "written), %d declined",
                   patent_id, rep.get("gaps", 0), rep.get("applied", 0),
                   rep.get("proposed", 0), rep.get("declined", 0))
    for r in rep.get("results", []):
        # APPLIED means the tree moved. A patch that verified and was refused
        # the write is PROPOSED — it used to print APPLIED, which is how a run
        # that changed nothing read like a run that had healed itself.
        state = ("APPLIED" if r.get("written") else
                 "proposed" if r.get("ok") else "declined")
        logger.warning("autoheal:   %s %s — %s", state, r.get("target"),
                       str(r.get("why") or "")[:200])
    # A patch can APPLY on corpus coverage and still leave the patent that
    # asked for it at zero — measured, twice. That is not a reason to block the
    # patch; corpus coverage is the one condition that cannot be argued with.
    # It IS a reason not to record the capability as bought: releasing the key
    # lets a later run ask again, and `collect_gaps` will by then have read the
    # failed attempt out of the journal and put it in the next prompt.
    # `written`, not `ok`: releasing the key is about a patch that LANDED and
    # still left its patent at zero. A patch the guard refused bought nothing
    # and landed nothing, so re-asking inside this same run would only earn a
    # second refusal at the price of another full corpus verification.
    unresolved = [r for r in rep.get("results", [])
                  if r.get("written") and (r.get("gap_rows_recovered") or 0) <= 0]
    if unresolved:
        with _lock:
            _attempted.discard(key)
        logger.warning("autoheal: %s — patch APPLIED but %s is still unfixed; "
                       "releasing the capability so a later run asks again",
                       patent_id, patent_id)
        rep["target_unresolved"] = True
    rep["status"] = "ran"
    rep["key"] = key
    return rep


def reset() -> None:
    """Forget this process's spend. For tests and for long-lived callers."""
    global _spent
    with _lock:
        _attempted.clear()
        _spent = 0
