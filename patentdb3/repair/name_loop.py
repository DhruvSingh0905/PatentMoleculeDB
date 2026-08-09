"""Per-patent orchestration for the name tier. Library first, then buy.

The `repair/loop.py` of the name tier, in the same order for the same reasons:

  1. FIND the compounds this patent asserts and we failed to resolve
     (`name_gap`).
  2. APPLY everything already known, at $0 — `name_repair`'s six hand-written
     forms AND every rule previously bought. This is the step the transfer
     measurement justifies: 4 patterns serve 98 patent-instances, so a
     library-first retry avoids 94 re-purchases. Skipping it would re-buy a
     rule the corpus already owns, on almost every patent.
  3. BUY a rule for what is left, in batches, retry-as-feedback up to
     `MAX_ATTEMPTS`.
  4. ESCALATE what is still unsolved, so the identical failure is never
     re-bought — expiring only when `NAME_SYNTH_EPOCH` is bumped.

WHAT A NEGATIVE OUTCOME IS
---------------------------
Feedback, not a rejection. A rule that fails is handed its OWN measurement —
"parses but covers 0.14 of the printed name", "the repaired fragment appears
nowhere else in this patent" — and asked again. The measurement is ours and is
not in doubt; what to do about it is the model's job, and a hint from us would
be a hypothesis dressed as evidence.

WHY ADOPTION RE-RUNS THE WHOLE GAP SET
----------------------------------------
A rule is bought to explain a batch, but it is stored globally and may fix gaps
that were not in the batch that paid for it. So every adopted rule is re-applied
to every remaining gap before the next purchase — the same "one purchase, many
beneficiaries" property the library-first step exploits, applied within a single
patent's run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config
from ..sources import name_repair as NR
from .name_gap import NameGap, find_name_gaps
from .name_outcome import NameOutcome, measure, summarise
from .name_rules import NameRuleLibrary, NamePattern, ground

logger = logging.getLogger(__name__)

# Paid attempts at one batch before it is abandoned and recorded. Lives here,
# not in config.py, exactly as `loop.MAX_ATTEMPTS` does — it bounds the loop,
# not the corpus.
MAX_ATTEMPTS = 3

NAME_JOURNAL = config.OUTPUT_DIR / "name_rule_journal.jsonl"


@dataclass
class Repaired:
    """One compound recovered, with the evidence that recovered it."""
    patent_id: str
    cid: str
    original: str
    repaired: str
    smiles: str
    inchikey: str
    pattern_id: str
    source: str                    # "library" | "synthesized"
    coverage: float = 0.0


@dataclass
class NameRepairReport:
    """What the loop did to this patent. The audit object, like RepairReport."""
    patent_id: str = ""
    gaps_found: int = 0
    fixed_by_library: int = 0
    fixed_by_new_rule: int = 0
    rules_adopted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    escalated: int = 0
    usd_spent: float = 0.0
    repaired: list[Repaired] = field(default_factory=list)

    @property
    def fixed(self) -> int:
        return self.fixed_by_library + self.fixed_by_new_rule


def _library_pass(gaps: list[NameGap], lib: NameRuleLibrary,
                  report: NameRepairReport) -> list[NameGap]:
    """Everything already known, applied at $0. Returns what is still broken."""
    remaining: list[NameGap] = []
    for g in gaps:
        # `name_repair.repair_name` carries the six hand-written forms and its
        # own OPSIN + corroboration gate. Reused rather than reimplemented so
        # a hand-written form and a bought one face the identical test.
        try:
            res = NR.repair_name(g.name_text, document_text=g.doc_text)
        except Exception as e:
            logger.warning("name_loop: repair_name raised on %s: %r", g.key, e)
            res = None
        if res is not None:
            report.fixed_by_library += 1
            report.repaired.append(Repaired(
                patent_id=g.patent_id, cid=g.cid, original=g.name_text,
                repaired=res.repaired, smiles=res.smiles, inchikey=res.inchikey,
                pattern_id=res.pattern_id, source="library"))
            continue

        hit = False
        for p in lib.patterns:
            oc = measure(p, g)
            if oc.positive:
                p.confirms += 1
                report.fixed_by_library += 1
                report.repaired.append(Repaired(
                    patent_id=g.patent_id, cid=g.cid, original=g.name_text,
                    repaired=oc.repaired, smiles=oc.smiles,
                    inchikey=oc.inchikey, pattern_id=p.id, source="library",
                    coverage=oc.coverage))
                hit = True
                break
        if not hit:
            remaining.append(g)
    return remaining


def _journal(path: Path, entry: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:                                  # never lose the run
        logger.warning("name_loop: could not write journal: %r", e)


def repair_names(xml: str, patent_id: str = "", *,
                 library: NameRuleLibrary | None = None,
                 journal: Path | None = None,
                 max_calls: int = 6,
                 with_opsin_errors: bool = True) -> NameRepairReport:
    """Find, apply, buy, escalate — for one patent.

    `library` and `journal` are injectable so the harness and the test suite
    can run against a scratch library instead of polluting the shipped one.
    """
    from ..core.cost_tracker import cost_tracker
    from . import name_synthesize as SYN

    report = NameRepairReport(patent_id=patent_id)
    lib = library if library is not None else NameRuleLibrary()
    jpath = journal or NAME_JOURNAL

    # `with_opsin_errors` is threaded through rather than left to the default
    # so the harness can actually ablate it. Its first version pre-computed
    # gaps without the annotation and then called this function, which
    # re-detected WITH it — so the "no-opsin-error" arm was not an ablation at
    # all and reported a difference of zero because there was none to report.
    gaps = find_name_gaps(xml, patent_id, with_opsin_errors=with_opsin_errors)
    report.gaps_found = len(gaps)
    if not gaps:
        return report

    spent_before = cost_tracker.total_cost
    remaining = _library_pass(gaps, lib, report)
    if remaining:
        lib.save()                       # persist the confirms counters

    calls = 0
    while remaining and calls < max_calls:
        batch = remaining[: SYN.BATCH]
        attempts: list[dict] = []
        adopted: NamePattern | None = None

        for _ in range(MAX_ATTEMPTS):
            if calls >= max_calls:
                break
            calls += 1
            p = SYN.propose(batch, attempts=attempts, digest=lib.digest())
            if p is None:
                break                    # no key, or nothing usable came back

            gr = ground(p, [g.name_text for g in batch], batch[0].clean_names)
            if not gr.ok:
                report.rejected.append(f"{p.id}: {gr.why}")
                attempts.append({"site": p.site, "replacement": p.replacement,
                                 "why": gr.why})
                continue

            # Measure on the batch the rule was bought for. A rule that fixes
            # ANY of them is real; `outcome` decides per compound.
            wins = [(g, measure(p, g, compiled=gr.compiled)) for g in batch]
            good = [(g, oc) for g, oc in wins if oc.positive]
            if not good:
                why = next((oc.detail for _, oc in wins if oc.conflicts),
                           "the rule produced nothing on any of these names")
                report.rejected.append(f"{p.id}: {why}")
                attempts.append({"site": p.site, "replacement": p.replacement,
                                 "why": why})
                logger.info("name_loop: %s", summarise(p, wins[0][1]))
                continue

            adopted = p
            break

        if adopted is None:
            # Nothing worked for this batch in MAX_ATTEMPTS. Drop it and move
            # on — recorded, never re-bought within this run.
            report.escalated += len(batch)
            for g in batch:
                _journal(jpath, {"event": "escalate", "patent": g.patent_id,
                                 "cid": g.cid, "name": g.name_text,
                                 "attempts": attempts,
                                 "epoch": SYN.PROMPT_VERSION})
            remaining = remaining[len(batch):]
            continue

        # Adopted. Re-apply to EVERY remaining gap, not just the batch that
        # paid — see the module docstring.
        adopted.confirms = 0
        still: list[NameGap] = []
        for g in remaining:
            oc = measure(adopted, g)
            if oc.positive:
                adopted.confirms += 1
                report.fixed_by_new_rule += 1
                report.repaired.append(Repaired(
                    patent_id=g.patent_id, cid=g.cid, original=g.name_text,
                    repaired=oc.repaired, smiles=oc.smiles,
                    inchikey=oc.inchikey, pattern_id=adopted.id,
                    source="synthesized", coverage=oc.coverage))
                _journal(jpath, {"event": "adopt", "patent": g.patent_id,
                                 "cid": g.cid, "pattern": adopted.to_dict(),
                                 "original": g.name_text,
                                 "repaired": oc.repaired,
                                 "coverage": oc.coverage,
                                 "inchikey": oc.inchikey})
            else:
                still.append(g)

        lib.add(adopted)
        report.rules_adopted.append(adopted.id)
        logger.info("name_loop: %s adopted %s — fixed %d of %d remaining",
                    patent_id, adopted.id, adopted.confirms, len(remaining))
        remaining = still

    report.escalated += len(remaining)
    report.usd_spent = round(cost_tracker.total_cost - spent_before, 6)
    logger.info(
        "name_loop: %s — %d gaps, %d fixed (%d library, %d new), "
        "%d escalated, $%.4f",
        patent_id, report.gaps_found, report.fixed, report.fixed_by_library,
        report.fixed_by_new_rule, report.escalated, report.usd_spent)
    return report
