"""The repair loop: detect a gap, buy a rule, RUN it, keep what works.

    parse → gaps → (already known? apply + measure) → propose → ground →
    apply → MEASURE → keep, or hand the failure back and try again (x3)

Three properties matter more than anything else here.

**A rule is judged by what it does, not by what it looks like.** The proposal
is applied to the patent it was bought for — only that patent, never a corpus
sweep — and `outcome.measure` reports whether it produced a usable measurement,
whether it contradicts anything the reader already reads, and whether it took a
value from a column that reads as a molecular weight or an m/z. That is the
whole gate. The scoring layer that used to run BEFORE the rule (a coverage
floor, an anti-deletion guard, an adversarial probe battery) is gone: measured
over the 88 rules it objected to, every objection would have deleted correct
data, and the anti-deletion guard was comparing a held-out row count against a
whole-table record count and so could never have been right.

**A failure is feedback, not a verdict.** A negative outcome goes back to the
model as the measurement it produced, and it answers again — `MAX_ATTEMPTS`
times, after which nothing it proposed is kept and the layout is recorded as an
escalation. Reverted and skipped, in the caller's words.

**Every layout is asked about at most MAX_ATTEMPTS times, ever.** Answers are
keyed by layout fingerprint and persisted, including the negative ones
(`not_assay`, `escalate`). Re-asking a question already paid for is the fastest
way to lose the entire cost advantage over HARVEST — so an exhausted layout is
remembered too, and expires only when `SYNTH_EPOCH` moves.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config
from ..sources.uspto_xml import (
    Table, assemble_blocks, assembly_fidelity, parse_fidelity, parse_tables,
)
from .gap import Gap, find_gaps, yield_contradictions
from .outcome import Outcome, measure, summarise
from .rules import (
    BIN_KEY, COLUMN_MAP, ESCALATE, NOT_ASSAY, ROW_REGEX, VALUE_PATTERN,
    Invalid, Rejected, Rule, RuleLibrary, ground,
)

logger = logging.getLogger(__name__)

# Paid attempts at ONE layout before it is reverted and skipped.
#
# Three, and the shape of the number matters more than the value: the loop is
# now a conversation with a measurement in it, so a model that gets it wrong is
# told what happened and can correct — but a layout no rule kind can express
# will fail identically every time, and paying for that indefinitely is how the
# cost advantage over HARVEST is lost. Measured on 88 rules, 36 never produced
# a row in any application; those are the ones that will spend all three.
MAX_ATTEMPTS = 3


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
    # Unusable records dropped because a repaired twin of the SAME
    # (cid, assay, table) exists. Not a loss — see the supersession note in
    # `repair_patent`. Reported so the count is visible rather than implied by
    # a row total that quietly shrank.
    superseded: int = 0
    escalations: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    # Gaps where a rule was available and produced NOTHING. Not a bad rule —
    # a layout no rule kind can express, which is the queue the code-patch
    # tier would read. NOTE: `repair/capability.py` — the code-patch tier —
    # did NOT come across to v3, so nothing consumes this list today. It is
    # still populated because it is the record of which layouts no rule can
    # express, which is the input that tier needs whenever it is rebuilt.
    capability_gaps: list[dict] = field(default_factory=list)
    # Gaps that raised. A FIRST-CLASS outcome, not something the caller is
    # trusted to notice: three patents were skipped entirely by a corpus run
    # because `repair_patent` raised, the runner logged a line, and the totals
    # looked healthy. A failure that preserves the appearance of the counts is
    # the shape of every defect found this week.
    crashed: list[dict] = field(default_factory=list)


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
                  max_calls: int = 8, dry_run: bool = False,
                  model: str | None = None,
                  journal: Path | None = None) -> tuple[list, RepairReport]:
    """Recover what the deterministic parser missed. Returns (records, report).

    `max_calls` bounds TOTAL paid calls per patent, across all gaps and all
    their attempts. It was 4 when a gap cost exactly one call; a gap may now
    cost up to `MAX_ATTEMPTS`, so 4 would have quietly cut the number of gaps
    examined from four to one or two. At Haiku rates a synthesis call is
    ~$0.002, so the ceiling is ~$0.016 per patent and $0 on any layout already
    in the library. Gaps are ranked by how many rows they cost us, so a limited
    budget is spent where it recovers the most.

    `journal` overrides where adopted rules are recorded, and exists because
    the test suite was writing into the production one. The retry tests call
    this function for real — that is the point of them — and while they isolate
    the LIBRARY into a tmp path, `_journal_rule` wrote to `config.RULE_JOURNAL`
    regardless, so ten `US10870641` entries from pytest landed in the artifact a
    human reads to find out what a run did. A test that pollutes a production
    record is how a number stops being attributable.

    `model` overrides the synthesis model for this patent. It exists for
    a model bake-off — comparing what different models propose for the same
    layout — which is why answers are cached per (fingerprint, model) rather
    than per fingerprint alone: switching models must not read another's answer.
    The v2 script that used it (`scripts.eval.model_bakeoff`) is not in v3.
    """
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
        # ── THE GATE. One question, asked of the patent in front of us. ──
        #
        # A rule is kept when applying it to THIS patent — only this patent,
        # never a corpus sweep — produces a usable measurement, contradicts
        # nothing the reader already reads, and takes no value from a column
        # the reader identified as a molecular weight, an m/z, an NMR shift or
        # a retention time. That is `outcome.measure`, and it is the whole of
        # the judgement.
        #
        # A negative outcome is not a rejection, it is FEEDBACK: the model is
        # told what its rule actually did and asked again, up to MAX_ATTEMPTS.
        # This replaces a scoring layer that ran before the rule and tried to
        # predict the same thing. Measured on the 88 rules it objected to,
        # every objection would have deleted correct data.
        rule = lib.get(gap.fingerprint)
        table = by_id.get(gap.table_id)
        if table is None:
            continue
        fresh = False
        attempts: list[dict] = []

        if rule is not None:
            report.already_known += 1
        else:
            if dry_run:
                report.proposed += 1
                calls += 1
                continue
            from .synthesize import SYNTH_MODEL, propose

            # What the library already knows about layouts like this one, as a
            # few lines rather than 165 KB. See `RuleLibrary.digest`.
            digest = lib.digest(gap.signature)
            rule = None
            while len(attempts) < MAX_ATTEMPTS:
                if calls >= max_calls:
                    logger.info("repair: %s call budget spent after %d attempt(s) "
                                "on %s", patent_id, len(attempts), gap.fingerprint)
                    break
                candidate = propose(gap, patent_id=patent_id,
                                    model=model or SYNTH_MODEL,
                                    attempts=attempts, digest=digest)
                calls += 1
                report.proposed += 1
                if candidate is None:
                    break
                # `not_assay` and `escalate` are ANSWERS, not failures. Retrying
                # them would be paying a model to change a verdict it already
                # gave, which is the one thing a retry loop must never do.
                if candidate.kind in (ESCALATE, NOT_ASSAY):
                    rule = candidate
                    fresh = True
                    break
                try:
                    candidate.validated_on = ground(candidate, table)
                except Invalid as e:
                    # It cannot RUN. Hand back the contract violation verbatim —
                    # it names exactly what is wrong, and a model that wrote a
                    # regex with no `num` group can fix that when told.
                    report.rejections.append({
                        "fingerprint": gap.fingerprint, "patent": patent_id,
                        "table": gap.table_id, "proposed": candidate.kind,
                        "why_rejected": str(e), "contract": True,
                        "attempt": len(attempts) + 1, "sample": gap.sample[:400],
                    })
                    attempts.append({"kind": candidate.kind,
                                     "payload": candidate.payload,
                                     "why": str(e)})
                    continue
                except Exception as e:
                    logger.warning("repair: %s crashed grounding %s: %r",
                                   patent_id, gap.fingerprint, e)
                    report.crashed.append({
                        "fingerprint": gap.fingerprint, "patent": patent_id,
                        "table": gap.table_id, "stage": "ground",
                        "rule_kind": candidate.kind, "error": repr(e)[:200],
                        "rows_at_stake": gap.severity,
                    })
                    break

                try:
                    produced = apply_rule(candidate, table, patent_id)
                except Exception as e:
                    logger.warning("repair: %s crashed applying %s: %r",
                                   patent_id, gap.fingerprint, e)
                    report.crashed.append({
                        "fingerprint": gap.fingerprint, "patent": patent_id,
                        "table": gap.table_id, "stage": "apply",
                        "rule_kind": candidate.kind, "error": repr(e)[:200],
                        "rows_at_stake": gap.severity,
                    })
                    break

                oc = measure(candidate, produced, table)
                logger.info("repair: %s attempt %d/%d — %s", gap.fingerprint,
                            len(attempts) + 1, MAX_ATTEMPTS,
                            summarise(candidate, oc))
                if oc.positive:
                    candidate.signature = gap.signature
                    candidate.validated_on = {
                        **(candidate.validated_on or {}),
                        "gate": "outcome", "attempt": len(attempts) + 1,
                        "records": oc.records, "usable": oc.usable,
                        "compounds": oc.compounds,
                    }
                    rule, fresh = candidate, True
                    lib.add(rule)
                    report.adopted += 1
                    report.rows_recovered += len(produced)
                    recovered.extend(produced)
                    lib.record_use(gap.fingerprint, len(produced))
                    _journal_rule(patent_id, gap, rule, produced, oc, journal)
                    break
                attempts.append({"kind": candidate.kind,
                                 "payload": candidate.payload, "why": oc.detail})
                rule = None

            if rule is None:
                # REVERTED AND SKIPPED. Nothing the model proposed survives —
                # not in the library, not in the output. What IS remembered is
                # that this layout resisted MAX_ATTEMPTS, recorded as an
                # escalation so the next run does not re-buy the same failures.
                #
                # An escalation expires on `SYNTH_EPOCH`, which is the property
                # that makes this safe to remember: improve the prompt, the
                # sample or a rule kind, and every layout parked here is asked
                # again automatically. A kept rule is evidence and never
                # expires; a failure is a record of what we could do that day.
                report.escalated += 1
                report.escalations.append({
                    "fingerprint": gap.fingerprint, "patent": patent_id,
                    "table": gap.table_id, "rows_at_stake": gap.severity,
                    "capability": f"no rule survived {len(attempts)} attempt(s)",
                    "note": " | ".join(
                        f"attempt {i + 1} ({a['kind']}): {a['why']}"
                        for i, a in enumerate(attempts)) or "no proposal returned",
                })
                if attempts:
                    lib.add(Rule(
                        fingerprint=gap.fingerprint, kind=ESCALATE,
                        signature=gap.signature,
                        payload={"capability": f"failed {len(attempts)} attempts",
                                 "note": attempts[-1]["why"],
                                 "attempts": attempts},
                        source="llm", model=model or ""))
                continue

        # These two are ANSWERS that legitimately produce no records.
        if rule.kind == ESCALATE:
            if fresh:
                rule.signature = rule.signature or gap.signature
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
                rule.signature = rule.signature or gap.signature
                lib.add(rule)
            continue
        if fresh:
            continue                 # already applied, measured and journalled

        # ── a rule the library already holds ──────────────────────────
        #
        # It is here because it passed the outcome gate on the patent it was
        # bought for. That is not a promise about THIS patent: the fingerprint
        # is our own claim that one rule serves both shapes, and it can be
        # wrong. So a known rule faces the same measurement — it costs nothing,
        # there is no API call, and shipping a contradiction we could have
        # observed is the failure this loop exists to prevent.
        try:
            produced = apply_rule(rule, table, patent_id)
        except Rejected as e:
            logger.warning("repair: rule %s failed at apply time: %s",
                           gap.fingerprint, e)
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

        oc = measure(rule, produced, table)
        if oc.positive:
            report.adopted += 1
            report.rows_recovered += len(produced)
            recovered.extend(produced)
            lib.record_use(gap.fingerprint, len(produced))
            _journal_rule(patent_id, gap, rule, produced, oc, journal)
            continue

        # A known rule that does not hold here. Its records are NOT used.
        #
        # Remembered rather than revoked: US9302989's `column_map` names exactly
        # the right columns and will start producing the moment `parse_value`
        # can read `0.0125, nd`. The rule is insufficient, not wrong, and
        # deleting it would buy the same answer again every run. What changes is
        # that "known" no longer means "answered" — the gap leaves this tier
        # with its rows still counted. (The code-patch tier that consumed
        # these is not in v3 — see RepairReport.capability_gaps.)
        report.capability_gaps.append({
            "fingerprint": gap.fingerprint, "patent": patent_id,
            "table": gap.table_id, "rows_at_stake": gap.severity,
            "rule_kind": rule.kind, "rule_payload": rule.payload,
            "why": f"{oc.detail} — {_why_nothing_applied(rule, table)}"
                   if not produced else oc.detail,
            "unparsed_examples": gap.unparsed_examples[:8],
        })
        if oc.conflicts:
            logger.warning("repair: library rule %s does not hold on %s: %s",
                           gap.fingerprint, patent_id, oc.detail[:200])

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
    # UNION, baseline first. `baseline` used to exist only to measure gaps
    # against and was then dropped, so the one route by which a deterministic
    # XML record could reach `assay_tables` was the re-extract in
    # `process_patent`, gated on a capability patch actually landing. Measured
    # across the 22 shipped artifacts (2026-08-04): 35,888 rows, source
    # distribution `{'<none>': 27868, 'letter_bin': 8020}` — not one row from
    # the source CLAUDE.md records at 99.9% exact-match against BindingDB.
    #
    # SUPERSESSION ONLY — the narrowest dedup the evidence supports.
    #
    # This was NO DEDUP AT ALL, on a stated measurement: "a key including
    # assay_name matches 0.3% of rows". That is still true of an ordinary
    # patent — US11053244 measures 0.2% — and both of the risks the old note
    # names are real and are still avoided here: the key keeps `assay_name`,
    # so an IC50 of 5 nM is never merged with a Ki of 5 nM, and rows the two
    # tiers name differently do not match and both ship.
    #
    # It breaks where the repair loop IMPROVES a record. US10172859 publishes
    # every assay as a letter grade; the loop binds the patent's own key
    # (`B: 3 nM <= IC50 < 7 nM`) and produces a ranged copy — but the reader's
    # unranged copy survived beside it. Measured: 1,123 of 1,290 keys
    # duplicated (87.1%), of which 1,121 are one usable copy beside one
    # unusable copy of the SAME fact.
    #
    # Two things followed, and the second is why this is not cosmetic:
    #   - the dump asserted two measurements where the patent states one, and
    #     "consumers are duplicate-safe" was simply not true of anything that
    #     counts rows;
    #   - every affected table scored a yield of EXACTLY 0.500 against
    #     `find_gaps(min_yield=0.5)`, so the repair's own output pinned the
    #     detector at its threshold and hid the half still unresolved. The
    #     loop was blinded by what it had already fixed.
    #
    # An unusable copy carries nothing its usable twin lacks — same compound,
    # same assay, same table, same grade, only no range. So it is dropped, and
    # ONLY then. 17 of the 1,138 duplicate groups have both copies equally
    # usable; those are different facts and both still ship.
    by_key: dict[tuple, list] = {}
    for r in list(baseline) + list(recovered):
        by_key.setdefault((r.cid, r.assay_name, r.table_id), []).append(r)
    merged = []
    superseded = 0
    for group in by_key.values():
        if len(group) > 1 and any(r.is_usable for r in group):
            keep = [r for r in group if r.is_usable]
            superseded += len(group) - len(keep)
            merged.extend(keep)
        else:
            merged.extend(group)
    if superseded:
        logger.info("repair: %s — %d unusable record(s) superseded by a "
                    "repaired twin", patent_id, superseded)
    report.superseded = superseded
    return merged, report


def _journal_rule(patent_id: str, gap, rule: Rule, produced: list,
                  oc: Outcome, path=None) -> None:
    """Record what a rule actually did, so a bad one can be found afterwards.

    Every rule in this journal PASSED the outcome gate on the patent named in
    the entry — that is what "kept" now means, and it is why the entry carries
    the measurement (`usable`, `compounds`, the attempt it succeeded on) rather
    than an objection a suspended gate raised and we overrode. Those objection
    fields are gone with the gates that wrote them.

    Usable-vs-total still matters more than the count: a rule that adds 500
    records of which 0 are usable has produced nothing but noise. That is now
    condition 1 of the gate rather than something a reader has to notice here.

    THERE IS STILL NO UNDO. Removing a rule keyed by layout fingerprint is easy
    to build and does not exist; until it does, the recovery for a bad rule is
    to delete its entry from `data/layout_rules.json` by fingerprint, which the
    `fingerprint` field here gives you.
    """
    import json
    from collections import Counter

    entry = {
        "patent": patent_id, "table": gap.table_id,
        "fingerprint": gap.fingerprint, "kind": rule.kind,
        "records": len(produced), "usable": oc.usable,
        "compounds": oc.compounds,
        "attempt": (rule.validated_on or {}).get("attempt", 1),
        "assay_names": [n for n, _ in Counter(
            r.assay_name for r in produced if r.assay_name).most_common(6)],
        "sample_values": [
            {"cid": r.cid, "assay": r.assay_name, "value": r.value_numeric,
             "lo": r.range_lo, "hi": r.range_hi, "unit": r.unit}
            for r in produced[:3]],
        "gate": "outcome",
        "payload": rule.payload,
    }
    try:
        target = path or config.RULE_JOURNAL
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:                       # journalling must never break a run
        logger.warning("repair: could not journal rule use: %r", e)


