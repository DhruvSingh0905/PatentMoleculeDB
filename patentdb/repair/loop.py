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
import re
from dataclasses import dataclass, field

from ..sources.uspto_xml import (
    Table, assemble_blocks, parse_fidelity, parse_tables,
)
from .gap import Gap, find_gaps
from .rules import (
    BIN_KEY, COLUMN_MAP, ESCALATE, NOT_ASSAY, ROW_REGEX, VALUE_PATTERN,
    Rejected, Rule, RuleLibrary, validate,
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


# Regex syntax, removed to leave the literal words a pattern is really looking
# for. Used only to rank two patterns that both matched — never to match.
_META = re.compile(r"[.^$*+?{}\[\]\\()|]")


def _literal_overlap(pattern: str, name: str) -> int:
    """How specifically `pattern` speaks about `name`, in characters.

    The longest literal word the pattern contains that actually appears in the
    name. Deliberately not the length of the regex match: `IC50.*PK` matches
    the whole of "ic50 pdna-pk" through a wildcard, which makes the vaguest
    pattern look like the most precise one.
    """
    best = 0
    for alt in pattern.split("|"):
        for lit in _META.sub(" ", alt).split():
            if len(lit) > best and lit.lower() in name:
                best = len(lit)
    return best


def _bins_for(payload: dict, assay_name: str | None) -> dict:
    """The bin scale that governs one column.

    A grade letter is not globally meaningful. US10172859 defines three A–D
    scales in one legend: DNA-PK enzymatic in nM, pDNA-PK cellular in μM, and
    a Kv11.1 hERG scale running the OPPOSITE way — `A` is `IC50 < 3 nM` for the
    first and `Ki > 25 μM` for the last. So a key may carry `scales`, each
    naming the assays it governs; `bins` alone stays valid for one scale.

    Two things decide which scale claims a column, and the first one was
    missing:

    **Specificity.** `DNA-` is a substring of `pDNA-`, so first-match-wins gave
    the cellular column the enzymatic scale — 0–3 nM where the patent says
    0–0.5 μM, a 166-fold understatement recorded as a measurement. The more
    specific pattern wins, measured by the longest literal word it shares with
    the column name.

    **Refusal.** When specificity cannot separate two scales that disagree, we
    do not know, and a coin flip between opposite directions is the one outcome
    worse than extracting nothing. The grade is left raw; the record stays
    unusable and the gap detector keeps reporting it.
    """
    scales = payload.get("scales")
    if not scales:
        return payload.get("bins") or {}
    name = (assay_name or "").lower()
    hits: list[tuple[int, dict]] = []
    for s in scales:
        pat = (s.get("match") or "").strip()
        if not pat:
            continue
        try:
            if re.search(pat, name, re.I):
                hits.append((_literal_overlap(pat, name), s))
        except re.error:
            continue
    if hits:
        best = max(score for score, _ in hits)
        winners = [s for score, s in hits if score == best]
        chosen = winners[0].get("bins") or {}
        if len(winners) > 1 and any((w.get("bins") or {}) != chosen for w in winners):
            logger.warning(
                "repair: %d scales claim %r with equal specificity and disagree; "
                "leaving the grade raw rather than guessing", len(winners), assay_name)
            return {}
        return chosen
    # No scale claims this column. Returning the first would be a coin flip
    # between opposite directions, so return nothing and leave the grade raw.
    return next((s.get("bins") or {} for s in scales if not s.get("match")), {})


def _why_nothing_applied(rule: Rule, table: Table) -> str:
    """Say what a validated rule failed to find, in the queue's own terms."""
    from ..sources.uspto_assays import extract_from_tables

    if rule.kind != BIN_KEY:
        return (f"{rule.kind} validated on this layout but produced no records "
                f"from {table.table_id}")
    recs = extract_from_tables([table])
    scales = rule.payload.get("scales") or []
    names = {(r.assay_name or "") for r in recs}
    if scales and all("unnamed" in n.lower() or not n for n in names):
        return (f"bin_key defines {len(scales)} column-scoped scales "
                f"({', '.join((s.get('match') or '?')[:24] for s in scales)}) but "
                f"every record on {table.table_id} is unnamed — the column names "
                f"exist in the header and did not survive multi-row merge, so no "
                f"scale can be bound to a column. Needs header alignment, not a "
                f"better key.")
    return (f"bin_key matched none of the {len(recs)} records on "
            f"{table.table_id}; assay names present: {sorted(names)[:4]}")


def apply_rule(rule: Rule, table: Table, patent_id: str) -> list:
    """Turn a validated rule into assay records."""
    from ..sources.uspto_assays import (
        AssayRecord, _header_rows_of, normalize_cid, parse_value,
    )
    from .rules import _safe_regex, _search


    if rule.kind in (NOT_ASSAY, ESCALATE):
        return []


    if rule.kind == BIN_KEY:
        # Re-read the block's grades as ranges. The records already exist and
        # carry the right cid and assay name — they simply have no number. This
        # upgrades them in place rather than re-extracting.
        from ..sources.uspto_assays import extract_from_tables
        out = []
        for rec in extract_from_tables([table]):
            b = _bins_for(rule.payload, rec.assay_name).get(
                (rec.letter_grade or "").upper())
            if not b or rec.value_numeric is not None:
                continue
            rec.range_lo, rec.range_hi = b.get("lo"), b.get("hi")
            rec.unit = rec.unit or b.get("unit")
            rec.source = "repair_rule_bin_key"
            out.append(rec)
        return out

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
    raw = parse_tables(xml)
    # The ASSEMBLED grid, never a fragment — see assemble_block. Both the
    # detector and the applier read through this one view. When they disagreed,
    # the samples shown to the model were drawn from whichever fragment came
    # first, which on US10172859 is an interleaved NMR annotation: the model
    # then answered `not_assay` — correctly, about the wrong table — and eight
    # genuine assay blocks were permanently dismissed by a negative rule.
    tables = assemble_blocks(raw)
    by_id = {t.table_id: t for t in tables}
    baseline = extract_from_patent(xml)
    per_table = Counter(r.table_id for r in baseline)

    # Pass the RECORDS and the source XML, not a tally: the yield metric must
    # ask whether each record is a usable measurement, and the legend hunt
    # needs the document. Passing counts here silently disabled both.
    gaps = find_gaps(patent_id, tables, baseline, _source_xml=xml)
    report = RepairReport(patent_id=patent_id, gaps_found=len(gaps))

    # Before buying a single rule: is what we are looking at what the patent
    # says? Every gap signal above is computed from the PARSED view, so a defect
    # in the parser is invisible to all of them — it presents as a hard layout,
    # and the loop pays a model to describe damage we inflicted ourselves.
    #
    # `_parse_row` matched `<entry/>` with the paired-tag branch, so every empty
    # cell swallowed its neighbour. US11613531 lost 2,359 cells that way and
    # spent two sessions being reported as "fires on 336/687 held-out rows
    # (49%), just under the 50% floor" — an honest-looking rejection of a
    # perfectly ordinary table. No rule could have fixed it, and the loop had no
    # way to say so.
    #
    # A block whose cells do not reconcile with its source is a CODE defect, not
    # a layout gap. It never reaches the model: the escalation names the file to
    # look in instead of asking for a rule that would encode the corruption.
    broken = {d["table_id"]: d for d in parse_fidelity(xml)}
    if broken:
        logger.warning("repair: %s has %d block(s) that do not reconcile with "
                       "their source; not asking for rules on those",
                       patent_id, len(broken))
    recovered: list = []
    calls = 0

    for gap in gaps:
        defect = broken.get(gap.table_id)
        if defect is not None:
            report.escalated += 1
            report.escalations.append({
                "fingerprint": gap.fingerprint, "patent": patent_id,
                "table": gap.table_id, "rows_at_stake": gap.severity,
                "capability": "PARSER DEFECT — not a layout gap",
                "note": (f"{defect['detail']}. Fix `uspto_xml._parse_row` / the "
                         f"table reader, not this layout. No rule was requested: "
                         f"a rule written against a corrupted view would encode "
                         f"the corruption."),
            })
            continue
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
            if rule.kind in (COLUMN_MAP, ROW_REGEX, VALUE_PATTERN, BIN_KEY) and table is not None:
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
        else:
            # A rule that passed validation and then produced nothing used to
            # vanish here: not adopted, not rejected, not escalated, no trace.
            # US10172859 returned a CORRECT three-scale bin_key and the report
            # read "0 adopted, 0 rejected", which is why the failure looked like
            # model refusal for four rounds. It was not — the scales key on
            # assay name, and every record on that table is named "unnamed assay
            # (letter bin)" because the multi-row header never aligned.
            #
            # Applying to nothing is a real outcome and it names the missing
            # capability precisely, so it escalates like any other.
            report.escalated += 1
            report.escalations.append({
                "fingerprint": gap.fingerprint, "patent": patent_id,
                "table": gap.table_id, "rows_at_stake": gap.severity,
                "capability": "rule validated but matched no rows on apply",
                "note": _why_nothing_applied(rule, table),
            })

    if not dry_run:
        lib.save()
    return recovered, report
