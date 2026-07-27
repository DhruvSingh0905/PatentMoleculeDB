"""The repair loop: detect a gap, buy one rule, validate it, keep it forever.

    parse → gaps → (already known? stop) → propose → VALIDATE → persist → re-run

Two properties matter more than anything else here.

**Nothing enters the library unvalidated.** A model proposal is a hypothesis.
`rules.validate()` re-runs it against the real rows and an adversarial battery
of NMR/MS/MW/RT lines, and a proposal that fails is recorded as an escalation
rather than silently dropped — so the failure is visible and a human can see
what the model tried.

**Every layout is asked about exactly once.** Answers are keyed by layout
fingerprint and persisted, including the negative answers (`not_assay`,
`escalate`). Re-asking a question already paid for is the fastest way to lose
the entire cost advantage over HARVEST.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..sources.uspto_xml import Table, parse_tables
from .gap import Gap, find_gaps
from .rules import (
    COLUMN_MAP, ESCALATE, NOT_ASSAY, ROW_REGEX, Rejected, Rule, RuleLibrary,
    validate,
)

logger = logging.getLogger(__name__)


@dataclass
class RepairReport:
    patent_id: str
    gaps_found: int = 0
    already_known: int = 0
    proposed: int = 0
    adopted: int = 0
    rejected: int = 0
    escalated: int = 0
    rows_recovered: int = 0
    escalations: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.patent_id}: {self.gaps_found} gaps "
                f"({self.already_known} known) → {self.proposed} asked, "
                f"{self.adopted} adopted, {self.rejected} rejected, "
                f"{self.escalated} escalated, +{self.rows_recovered} rows")


def apply_rule(rule: Rule, table: Table, patent_id: str) -> list:
    """Turn a validated rule into assay records."""
    from ..sources.uspto_assays import (
        AssayRecord, _header_rows_of, normalize_cid, parse_value,
    )
    from .rules import _safe_regex, _search

    if rule.kind in (NOT_ASSAY, ESCALATE):
        return []

    _, data = _header_rows_of(table)
    out: list = []

    if rule.kind == COLUMN_MAP:
        cid_i = rule.payload["cid"]
        for row in data:
            if len(row) <= cid_i:
                continue
            raw = row[cid_i].text.strip()
            if not raw:
                continue
            cid = normalize_cid(raw)
            for a in rule.payload["assays"]:
                i = a["index"]
                if len(row) <= i:
                    continue
                parsed = parse_value(row[i].text)
                if not parsed:
                    continue
                out.append(AssayRecord(
                    cid=cid, assay_name=a.get("name") or "assay",
                    value_numeric=parsed.get("value_numeric"),
                    qualifier=parsed.get("qualifier"),
                    unit=parsed.get("unit") or a.get("unit"),
                    n_runs=parsed.get("n_runs"),
                    letter_grade=parsed.get("letter_grade"),
                    value_text=parsed.get("value_text", ""),
                    table_id=table.table_id,
                    column_header=a.get("name") or "",
                    source="repair_rule_column_map",
                ))

    elif rule.kind == ROW_REGEX:
        pat = _safe_regex(rule.payload["pattern"])
        for row in data:
            line = " | ".join(c.text.strip() for c in row)
            m = _search(pat, line)
            if not m:
                continue
            cid = normalize_cid((m.group("cid") or "").strip())
            if not cid:
                continue
            for g, v in sorted(m.groupdict().items()):
                if not g.startswith("value") or v is None:
                    continue
                parsed = parse_value(v)
                if not parsed:
                    continue
                out.append(AssayRecord(
                    cid=cid, assay_name=rule.payload.get("name") or "assay",
                    value_numeric=parsed.get("value_numeric"),
                    qualifier=parsed.get("qualifier"),
                    unit=parsed.get("unit") or rule.payload.get("unit"),
                    letter_grade=parsed.get("letter_grade"),
                    value_text=parsed.get("value_text", ""),
                    table_id=table.table_id,
                    source="repair_rule_row_regex",
                ))
    return out


def repair_patent(patent_id: str, xml: str, *, library: RuleLibrary | None = None,
                  max_calls: int = 4, dry_run: bool = False) -> tuple[list, RepairReport]:
    """Recover what the deterministic parser missed. Returns (records, report).

    `max_calls` bounds spend per patent. Gaps are ranked by how many rows they
    cost us, so a limited budget is spent where it recovers the most.
    """
    from collections import Counter

    from ..sources.uspto_assays import extract_from_patent

    lib = library or RuleLibrary()
    tables = parse_tables(xml)
    by_id = {t.table_id: t for t in tables}
    baseline = extract_from_patent(xml)
    per_table = Counter(r.table_id for r in baseline)

    gaps = find_gaps(patent_id, tables, dict(per_table))
    report = RepairReport(patent_id=patent_id, gaps_found=len(gaps))
    recovered: list = []
    calls = 0

    for gap in gaps:
        rule = lib.get(gap.fingerprint)
        if rule is not None:
            report.already_known += 1
        else:
            if calls >= max_calls:
                logger.info("repair: %s call budget spent; %d gaps unexamined",
                            patent_id, len(gaps) - report.already_known - calls)
                break
            if dry_run:
                report.proposed += 1
                calls += 1
                continue
            from .synthesize import propose
            rule = propose(gap, patent_id=patent_id)
            calls += 1
            report.proposed += 1
            if rule is None:
                continue

            table = by_id.get(gap.table_id)
            if rule.kind in (COLUMN_MAP, ROW_REGEX) and table is not None:
                try:
                    rule.validated_on = validate(
                        rule, table, baseline_rows=per_table.get(gap.table_id, 0))
                except Rejected as e:
                    # A failed proposal becomes a visible escalation, not a
                    # silent drop — the reason is what a human needs to see.
                    report.rejected += 1
                    report.rejections.append({
                        "fingerprint": gap.fingerprint, "patent": patent_id,
                        "table": gap.table_id, "proposed": rule.kind,
                        "why_rejected": str(e), "sample": gap.sample[:400],
                    })
                    rule = Rule(fingerprint=gap.fingerprint, kind=ESCALATE,
                                payload={"capability": "rejected proposal",
                                         "note": str(e)},
                                source="llm", model=rule.model)
            lib.add(rule)

        if rule.kind == ESCALATE:
            report.escalated += 1
            report.escalations.append({
                "fingerprint": gap.fingerprint, "patent": patent_id,
                "table": gap.table_id, "rows_at_stake": gap.severity,
                "capability": rule.payload.get("capability"),
                "note": rule.payload.get("note"),
            })
            continue
        if rule.kind == NOT_ASSAY:
            continue

        table = by_id.get(gap.table_id)
        if table is None:
            continue
        try:
            got = apply_rule(rule, table, patent_id)
        except Rejected as e:
            logger.warning("repair: rule %s failed at apply time: %s", gap.fingerprint, e)
            continue
        if got:
            report.adopted += 1
            report.rows_recovered += len(got)
            recovered.extend(got)
            lib.record_use(gap.fingerprint, len(got))

    if not dry_run:
        lib.save()
    return recovered, report
