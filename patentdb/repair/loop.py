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

from ..core import config
from ..sources.uspto_xml import (
    Table, assemble_blocks, assembly_fidelity, parse_fidelity, parse_tables,
)
from .gap import Gap, find_gaps, yield_contradictions
from .rules import (
    BIN_KEY, COLUMN_MAP, ESCALATE, NOT_ASSAY, ROW_REGEX, VALUE_PATTERN,
    Invalid, Rejected, Rule, RuleLibrary, validate,
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
    # Proposals the suspended gate objected to and we took anyway.
    adopted_over_objection: int = 0
    escalations: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    # Gaps where a rule was available and produced NOTHING. Not a bad rule —
    # a layout no rule kind can express, which is the queue the code-patch
    # tier reads. See `repair.capability`.
    capability_gaps: list[dict] = field(default_factory=list)
    # Gaps that raised. A FIRST-CLASS outcome, not something the caller is
    # trusted to notice: three patents were skipped entirely by a corpus run
    # because `repair_patent` raised, the runner logged a line, and the totals
    # looked healthy. A failure that preserves the appearance of the counts is
    # the shape of every defect found this week.
    crashed: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.patent_id}: {self.gaps_found} gaps "
                f"({self.already_known} known) → {self.proposed} asked, "
                f"{self.adopted} adopted, {self.rejected} rejected, "
                f"{self.escalated} escalated, "
                f"{self.adopted_over_objection} over-objection, "
                f"{len(self.capability_gaps)} capability-gap, "
                f"{len(self.crashed)} CRASHED, "
                f"+{self.rows_recovered} rows")


# Regex syntax, removed to leave the literal words a pattern is really looking
# for. Used only to rank two patterns that both matched — never to match.
_META = re.compile(r"[.^$*+?{}\[\]\\()|]")

# A cell that LOOKS like it carries a measurement, judged on text alone.
# Deliberately not imported from `gap.py`: the patent-level check below must not
# depend on the column classifier, the assembler, or anything else that can be
# the very thing that failed.
#
# "Short and contains a digit", NOT "is a bare number". The first version
# required the whole cell to parse as a number, which US10266548 answered with
# `shaped=0` while holding 1,049 such cells and 242 extracted records — its
# values carry their unit inline. A denominator meant to be unsuppressable was
# being suppressed by cell FORMATTING, which is the same class of defect it
# exists to catch, one level in.
#
# Breadth is safe here precisely because the condition is conjunctive: this only
# matters when the patent produced ZERO usable measurements. Measured over 83
# patents, widening the test adds exactly one patent — the one it was missing.
_SHAPED = re.compile(r"^(?=.{1,30}$)(?=.*\d).*$", re.S)
_GRADE = re.compile(r"^\s*(\++|[A-E])\s*$")


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

    elif rule.kind == VALUE_PATTERN:
        # There was no branch here at all. `value_pattern` is in the tool
        # schema, has the most careful validator of any kind in `rules.py`
        # (rescue + no-regression + adversarial battery), is named in the
        # `rule.kind in (...)` guard below, and two rules of this kind sit
        # adopted in the library — one reading `12.3 ± 1.4`, validated at 63
        # rescued cells. Every one of them fell through to an empty list. The
        # loop has been recording those as wins and producing nothing.
        from ..sources.uspto_assays import ASSAY, CID, _HEADER_CID, build_columns
        pat = _safe_regex(rule.payload["pattern"])
        cols = build_columns(table)
        kinds = [c.kind for c in cols]
        if CID not in kinds:
            return out
        cid_i = kinds.index(CID)
        # The id column must be one the patent NAMES as identifiers, not one
        # inferred from digit shape. US9233167 TABLE-US-00005 is a PK table
        # headed `PEG-length | Cmax | T1/2 | AUC | ...`; its first column runs
        # 0,1,2,3… and classifies as CID on shape alone. Applying a value rule
        # there mints 54 records whose compound id is a polymer chain length.
        # The live parser produces nothing for that table, correctly, and a
        # repair must not be the thing that invents it.
        if not _HEADER_CID.search((cols[cid_i].header or "").lower()):
            return out
        wanted = rule.payload.get("columns")
        for row in data:
            if len(row) <= cid_i:
                continue
            cid = normalize_cid(row[cid_i].text.strip())
            if not cid:
                continue
            for i, col in enumerate(cols):
                # Only where a measurement could live. A pattern aimed at the
                # id column would otherwise mint records whose value IS the
                # compound number.
                if col.kind != ASSAY or (wanted and i not in wanted):
                    continue
                if len(row) <= i:
                    continue
                s = row[i].text.strip()
                if not s or parse_value(s):
                    continue          # the parser already reads this cell
                m = _search(pat, s)
                if not m:
                    continue
                # Same guard as the validator: a named group can match without
                # participating, and a model describing a two-value cell names
                # its groups `num1`/`num2` rather than `num`.
                from .rules import first_number
                raw_num = first_number(m)
                if not raw_num:
                    continue
                try:
                    v = float(raw_num.replace(",", ""))
                except ValueError:
                    continue
                out.append(AssayRecord(
                    cid=cid, assay_name=col.assay_name or col.header or "assay",
                    value_numeric=v, unit=col.unit, value_text=s,
                    table_id=table.table_id, column_header=col.header,
                    source="repair_rule_value_pattern",
                ))
    return out


def repair_patent(patent_id: str, xml: str, *, library: RuleLibrary | None = None,
                  max_calls: int = 4, dry_run: bool = False,
                  model: str | None = None) -> tuple[list, RepairReport]:
    """Recover what the deterministic parser missed. Returns (records, report).

    `max_calls` bounds spend per patent. Gaps are ranked by how many rows they
    cost us, so a limited budget is spent where it recovers the most.

    `model` overrides the synthesis model for this patent. It exists for
    `scripts.eval.model_bakeoff`, which compares what different models propose:
    without it the bake-off called `propose` with its default every time and
    compared one model against itself under three labels. Answers are cached per
    (fingerprint, model), so switching models does not read another's answer.
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
    # ...and the second question the loop could not ask: does our handling agree
    # with ITSELF? A block yielding nothing while an identically-fingerprinted
    # block in the same document yields hundreds is not a hard layout — the
    # fingerprint is our own claim that one rule serves both. Buying a rule here
    # pays to encode an asymmetry whose fix is already running a few tables away.
    contradictions = {c["table_id"]: c
                      for c in yield_contradictions(tables, baseline)}

    # ...and the third question, which no gap could carry: did ASSEMBLY put the
    # rows in the right compartment? Reported HERE rather than as a gap
    # modifier, because the defect it names deletes its own evidence — a block
    # whose data became header has almost no body left, so every gap detector
    # measures a five-row table and skips it as noise. US10189840 raised zero
    # gaps while yielding zero records from 40 compounds. A signal that only
    # fires when some other signal already fired cannot catch that.
    misassembled = assembly_fidelity(xml)
    for d in misassembled:
        report.escalated += 1
        report.escalations.append({
            "fingerprint": None, "patent": patent_id,
            "table": d["table_id"], "rows_at_stake": d["header_rows"],
            "capability": "ASSEMBLY DEFECT — not a layout gap",
            "note": d["detail"] + " Examples of rows filed as header: "
                    + "; ".join(repr(e) for e in d["examples"][:3]),
        })
    if misassembled:
        logger.warning("repair: %s has %d block(s) whose header outgrew their "
                       "body; fix assemble_block, not the layout",
                       patent_id, len(misassembled))
    misassembled_ids = {d["table_id"] for d in misassembled}

    if broken:
        logger.warning("repair: %s has %d block(s) that do not reconcile with "
                       "their source; not asking for rules on those",
                       patent_id, len(broken))
    recovered: list = []
    calls = 0

    for gap in gaps:
        clash = contradictions.get(gap.table_id)
        if clash is not None:
            report.escalated += 1
            report.escalations.append({
                "fingerprint": gap.fingerprint, "patent": patent_id,
                "table": gap.table_id, "rows_at_stake": gap.severity,
                "capability": "INCONSISTENT HANDLING — not a layout gap",
                "note": clash["detail"],
            })
            continue
        # Already escalated above as a code defect. Asking for a rule as well
        # would pay a model to describe a table whose data we filed as header.
        if gap.table_id in misassembled_ids:
            continue
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
        fresh = False
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
            from .synthesize import SYNTH_MODEL, propose
            rule = propose(gap, patent_id=patent_id, model=model or SYNTH_MODEL)
            calls += 1
            report.proposed += 1
            if rule is None:
                continue

            table = by_id.get(gap.table_id)
            if rule.kind in (COLUMN_MAP, ROW_REGEX, VALUE_PATTERN, BIN_KEY) and table is not None:
                # The gate is SUSPENDED by default. `validate()` still runs and
                # its verdict is still recorded — it is evidence, not authority.
                #
                # No fixed validator anticipates the layouts patents actually
                # use, and this one has been wrong at least as often as right:
                # 0/23 on a correct column_map because `_CID_PAT` had never seen
                # a chemical-name id; a 49% floor on a rule whose real problem
                # was a regex in the reader; an anti-deletion baseline measured
                # with a broken parser. Every one of those cost records and
                # named the input as the fault.
                #
                # So adoption is made REVERSIBLE rather than prevented: the
                # proposal is journaled with the coverage it moves, which is
                # observable and revocable. A veto is neither.
                try:
                    rule.validated_on = validate(
                        rule, table, baseline_rows=per_table.get(gap.table_id, 0))
                except Invalid as e:
                    # A CONTRACT violation, not a verdict. `RULE_GATES_ENFORCE`
                    # suspends the judgement gates because they have been wrong
                    # as often as right; it was never meant to admit a rule the
                    # applier cannot execute. Three such rules entered the
                    # library over the objection "value_pattern must capture a
                    # named group `num`" and crashed two patents at apply time.
                    report.rejected += 1
                    report.rejections.append({
                        "fingerprint": gap.fingerprint, "patent": patent_id,
                        "table": gap.table_id, "proposed": rule.kind,
                        "why_rejected": str(e), "contract": True,
                        "sample": gap.sample[:400], "enforced": True,
                    })
                    rule = Rule(fingerprint=gap.fingerprint, kind=ESCALATE,
                                payload={"capability": "unexecutable proposal",
                                         "note": str(e)},
                                source="llm", model=rule.model)
                except Rejected as e:
                    report.rejections.append({
                        "fingerprint": gap.fingerprint, "patent": patent_id,
                        "table": gap.table_id, "proposed": rule.kind,
                        "why_rejected": str(e), "sample": gap.sample[:400],
                        "enforced": config.RULE_GATES_ENFORCE,
                    })
                    if config.RULE_GATES_ENFORCE:
                        report.rejected += 1
                        rule = Rule(fingerprint=gap.fingerprint, kind=ESCALATE,
                                    payload={"capability": "rejected proposal",
                                             "note": str(e)},
                                    source="llm", model=rule.model)
                    else:
                        report.adopted_over_objection += 1
                        rule.validated_on = {"gate": "suspended",
                                             "objection": str(e)}
                        logger.info("repair: %s adopted over objection — %s",
                                    gap.fingerprint, str(e)[:140])
                except Exception as e:
                    # `validate` executes a MODEL-SUPPLIED regex against real
                    # cells, so any exception here is ours to own, not the
                    # caller's to notice. US12281080 died on an AttributeError
                    # and US10654855/US11254686 on `IndexError('no such
                    # group')`; each cost a whole patent, and the corpus totals
                    # looked healthy because the runner logged a line and moved
                    # on. Recorded as a crash and the gap abandoned — one bad
                    # proposal must not cost the other gaps in this patent.
                    logger.warning("repair: %s crashed validating %s: %r",
                                   patent_id, gap.fingerprint, e)
                    report.crashed.append({
                        "fingerprint": gap.fingerprint, "patent": patent_id,
                        "table": gap.table_id, "stage": "validate",
                        "rule_kind": rule.kind, "error": repr(e)[:200],
                        "rows_at_stake": gap.severity,
                    })
                    continue
            fresh = True

        # These two are ANSWERS that legitimately produce no records, so they
        # are persisted here rather than by the yield test below.
        if rule.kind == ESCALATE:
            if fresh:
                lib.add(rule)
            report.escalated += 1
            report.escalations.append({
                "fingerprint": gap.fingerprint, "patent": patent_id,
                "table": gap.table_id, "rows_at_stake": gap.severity,
                "capability": rule.payload.get("capability"),
                "note": rule.payload.get("note"),
            })
            continue
        if rule.kind == NOT_ASSAY:
            if fresh:
                lib.add(rule)
            continue

        table = by_id.get(gap.table_id)
        if table is None:
            continue

        try:
            got = apply_rule(rule, table, patent_id)
        except Rejected as e:
            logger.warning("repair: rule %s failed at apply time: %s", gap.fingerprint, e)
            continue
        except Exception as e:
            logger.warning("repair: %s crashed applying %s: %r",
                           patent_id, gap.fingerprint, e)
            report.crashed.append({
                "fingerprint": gap.fingerprint, "patent": patent_id,
                "table": gap.table_id, "stage": "apply",
                "rule_kind": rule.kind, "error": repr(e)[:200],
                "rows_at_stake": gap.severity,
            })
            continue

        # A rule that produces NOTHING is not an answer, and must never become
        # the remembered one.
        #
        # This is how US9302989 stayed broken. The gap was found (1,561 rows),
        # Haiku proposed a `column_map` whose column indices were in fact
        # correct, `validate()` reported "fired on 0/1557 held-out rows", the
        # suspended gate adopted it anyway, `apply_rule` returned zero — and
        # `lib.add` had already run, so every later pass saw `already_known`
        # and never asked again. Eight layouts in the library are in that
        # state. The real blocker was one layer below anything a rule can
        # reach: the cell reads `0.0125, nd`, two probe measurements in one
        # cell, and `parse_value` returns None for it.
        #
        # Yield is the one signal here that needs no judgement — it is
        # observed, not scored — and it was the only one being ignored.
        if not got:
            # REMEMBERED, and reported as a capability gap. Both halves matter.
            #
            # Remembered, because the rule is usually not wrong — US9302989's
            # `column_map` names exactly the right columns and will start
            # producing the moment `parse_value` can read `0.0125, nd`. Throwing
            # it away would buy the same answer again every run.
            #
            # Reported, because "known" must stop meaning "answered". That
            # conflation is the whole defect: `lib.add` ran before `apply_rule`,
            # so a rule yielding zero was indistinguishable from a layout that
            # needed nothing, and 1,561 rows sat behind an `already_known` for
            # good. The gap now leaves this tier with its rows still counted and
            # goes to `repair.capability`, which may patch the code instead.
            if fresh:
                lib.add(rule)
            report.capability_gaps.append({
                "fingerprint": gap.fingerprint, "patent": patent_id,
                "table": gap.table_id, "rows_at_stake": gap.severity,
                "rule_kind": rule.kind, "rule_payload": rule.payload,
                "why": _why_nothing_applied(rule, table),
                "unparsed_examples": gap.unparsed_examples[:8],
            })
            continue

        if fresh:
            lib.add(rule)
        if got:
            report.adopted += 1
            report.rows_recovered += len(got)
            recovered.extend(got)
            lib.record_use(gap.fingerprint, len(got))
            _journal_rule(patent_id, gap, rule, got)
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

    # THE PATENT-LEVEL INVARIANT. Runs last, is gated by nothing, and consults
    # neither `gaps` nor `seen_blocks`.
    #
    # Every detector in `gap.py` scores a BLOCK, and each of them measures the
    # parsed view of that block. So a defect large enough to destroy a block's
    # parsed view also destroys the evidence each detector needs to report it:
    # US10189840 loses 89 of 94 rows into the header, `usable_yield` then sees a
    # five-row table, and `find_gaps` skips it at `shaped_cells < 10` as noise.
    # The bug that shrank the table pushed it under the threshold that would
    # have reported it. That shape — a failure that preserves the appearance of
    # the counts — is the recurring defect in this loop's history.
    #
    # This check cannot be silenced that way because its denominator is not
    # parsed at all: measurement-shaped CELLS in the raw tgroups, before
    # assembly, before column classification, before any judgement about what a
    # column means. If a document puts hundreds of numbers in tables and we
    # produce no usable measurement from any of them, that is worth a human's
    # attention whatever the per-table checks concluded — and it is exactly the
    # case where those checks are least able to speak.
    usable_now = sum(1 for r in baseline if r.is_usable) + sum(
        1 for r in recovered if r.is_usable)
    if not usable_now:
        shaped = 0
        for t in raw:
            for row in t.body_rows:
                for c in row:
                    s = c.text.strip()
                    if s and (_SHAPED.match(s) or _GRADE.match(s)):
                        shaped += 1
        # 10, and the number is measured rather than chosen. This floor exists
        # so a patent with no assay data at all stays quiet, but the condition
        # is already conjunctive — it only runs when NOTHING usable came out —
        # so the floor was carrying almost no weight. Across 93 patents exactly
        # one silent patent sits below 20 shaped cells (US9695181, at 18), and
        # every threshold from 1 to 18 fires on the identical set. The old 20
        # was a guess of mine, and its entire observable effect was to exclude
        # one genuinely broken patent from the tier built to fix it.
        if shaped >= config.SILENT_PATENT_MIN_CELLS:
            report.escalated += 1
            report.escalations.append({
                "fingerprint": None, "patent": patent_id, "table": None,
                "rows_at_stake": shaped,
                "capability": "PATENT YIELDED NOTHING",
                "note": (f"{patent_id} produced 0 usable measurements while its "
                         f"tables hold {shaped} measurement-shaped cells across "
                         f"{len({t.table_id for t in raw})} block(s). "
                         f"{len(gaps)} gap(s) were raised and "
                         f"{len(misassembled)} assembly defect(s) found, so the "
                         f"per-table detectors "
                         + ("did not explain this."
                            if not gaps and not misassembled
                            else "may not have explained all of it.")
                         + " The denominator here is raw cells, not parsed "
                           "rows, so no parser defect can suppress it."),
            })
            logger.warning("repair: %s yielded 0 usable measurements from %d "
                           "measurement-shaped cells", patent_id, shaped)

    # DOES THE RESULT MAKE SENSE — asked of the data we produced, with no
    # reference database in the loop. Every check above scores rows produced;
    # none can see a row that came out complete and wrong, and that class has
    # cost more here than any coverage gap. `value_check` catches it against
    # BindingDB, which cannot speak for a compound it has never seen — so the
    # live check has to be chemistry and internal consistency instead.
    #
    # Runs last, over baseline + whatever the loop recovered, so it judges the
    # final answer rather than an intermediate one.
    try:
        from .plausibility import audit as _plausibility
        for f in _plausibility(patent_id, xml, list(baseline) + list(recovered)):
            report.escalated += 1
            report.escalations.append({
                "fingerprint": None, "patent": patent_id, "table": f.table_id,
                "rows_at_stake": f.rows_at_stake,
                "capability": f"IMPLAUSIBLE: {f.kind}",
                "note": f.detail + (" e.g. " + "; ".join(f.examples[:3])
                                    if f.examples else ""),
            })
            logger.warning("plausibility: %s %s — %s", patent_id, f.kind,
                           f.detail[:160])
    except Exception as e:                       # reporting must never break a run
        logger.warning("plausibility: skipped for %s (%r)", patent_id, e)

    if not dry_run:
        lib.save()
    return recovered, report


def _journal_rule(patent_id: str, gap, rule: Rule, produced: list) -> None:
    """Record what a rule actually did, so a bad one can be found and revoked.

    This is what replaces the veto. A gate says "no" using assumptions that have
    been wrong repeatedly; a record says "here is exactly what changed", which is
    checkable after the fact and undoable. `revoke_rule()` drops the rule from
    the library and the next run re-asks.

    Usable-vs-total matters more than the count: a rule that adds 500 records of
    which 0 are usable has produced nothing but noise, and that is visible here
    without anyone having to predict it in advance.
    """
    import json
    from collections import Counter

    usable = sum(1 for r in produced if r.is_usable)
    entry = {
        "patent": patent_id, "table": gap.table_id,
        "fingerprint": gap.fingerprint, "kind": rule.kind,
        "records": len(produced), "usable": usable,
        "assay_names": [n for n, _ in Counter(
            r.assay_name for r in produced if r.assay_name).most_common(6)],
        "sample_values": [
            {"cid": r.cid, "assay": r.assay_name, "value": r.value_numeric,
             "lo": r.range_lo, "hi": r.range_hi, "unit": r.unit}
            for r in produced[:3]],
        "gate": (rule.validated_on or {}).get("gate", "passed"),
        "objection": (rule.validated_on or {}).get("objection"),
        "payload": rule.payload,
    }
    try:
        config.RULE_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with config.RULE_JOURNAL.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:                       # journalling must never break a run
        logger.warning("repair: could not journal rule use: %r", e)


def rule_journal() -> list[dict]:
    import json

    path = config.RULE_JOURNAL
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def revoke_rule(fingerprint: str, *, library: RuleLibrary | None = None) -> dict:
    """Drop a rule from the library. The next run re-asks about that layout.

    The counterpart to a suspended gate: nothing is prevented, everything is
    undoable. Kept deliberately blunt — deleting the answer is enough, because
    the loop rebuilds it from the question.
    """
    lib = library or RuleLibrary()
    rule = lib._rules.pop(fingerprint, None)
    if rule is None:
        return {"ok": False, "why": f"no rule for fingerprint {fingerprint!r}"}
    lib.save()
    return {"ok": True, "revoked": fingerprint, "kind": rule.kind,
            "rows_yielded": rule.rows_yielded}
