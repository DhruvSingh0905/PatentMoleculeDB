"""Did applying this rule to THIS patent make the result better or worse?

The one gate. Everything a rule is judged on is measured here, on the patent
the rule was bought for, by running it and looking at what came out. Nothing in
this file has an opinion about what a good rule looks like.

WHY THE JUDGEMENT GATES ARE GONE
--------------------------------
`rules.validate()` used to score a proposal before it ever ran: a coverage
floor, an anti-deletion guard, an adversarial probe battery, an id-shape test.
They were suspended by default (`RULE_GATES_ENFORCE=0`) because they had been
wrong as often as right, and on 2026-08-07 the 88 rules adopted over their
objection were measured against the deterministic reader:

    objection        rules   rows scored   corroborated   contradicted
    anti_deletion       32        28,281   26,539 (93.8%)           0
    coverage_floor       6           135       67 (49.6%)           0
    id_style             5            90       16 (17.8%)           0

Zero contradictions in 28,281 rows. Every one of those 88 vetoes would have
deleted correct data, and the anti-deletion guard was not merely wrong but
arithmetically incapable of being right: it compared `hits`, scored over
HELD-OUT rows only, against `baseline_rows`, counted over the WHOLE table, so
`baseline - hits` came out equal to the sample size in 91% of cases. It was
guaranteed to report a deletion no matter what the rule did.

A gate that fires 32 times and is right zero times is not a gate that needs a
better threshold. So the question changes from "does this rule look safe" —
unanswerable before it runs — to "what did it actually do", which is cheap:
the patent is already parsed, `apply_rule` has already run, and the baseline is
already in hand.

WHAT POSITIVE MEANS
-------------------
Three conditions, all observed on this patent, none of them a preference:

  1. it produced RECORDS. Not usable records — records.

     This condition was `usable > 0` for one measurement's worth of time, and
     the measurement is why it is not. Across the corpus it dropped 301 rows
     from 9 blocks, and every single one was missing exactly one field: the
     unit. `% Aβ Reduction = 42.0` for compound 2, whose unit is the `%` in its
     own name. `Hep CLint = 37.66` under a header that states `mL/min/10^6
     cells`, which is simply not in our unit vocabulary. Correct compound,
     correct assay, correct value, dropped.

     It was also inconsistent. The reader does not gate on usability —
     `repair_patent` returns `baseline + recovered`, and baseline carries
     unusable rows freely; US10710980's block ships 88 reader records of which
     0 are usable. Holding a repair to a standard the deterministic path does
     not meet is a preference, and preferences are what this rewrite removed.

     `usable` is still counted, still reported in `detail`, still journalled.
     A rule adding 500 rows of which 0 are usable remains visible to anyone
     reading the journal — it is just not blocked on the strength of a unit
     vocabulary we know to be incomplete.
  2. it CONTRADICTS nothing the reader already reads. The reader is the
     measured-good path (99.9% exact-match against BindingDB); where the two
     disagree about a cell, the rule is rewriting a correct answer.
  3. it does not draw values from a column the reader positively identified as
     something else — molecular weight, m/z, NMR, retention time, structure.
     Recording an MW as an IC50 is the corruption this pipeline exists to
     prevent, and it is invisible to conditions 1 and 2: such a record is
     perfectly usable and contradicts nothing.

Condition 3 replaces the adversarial probe battery, and is strictly stronger.
The battery ran a proposed regex against five synthetic NMR/MS/MW/RT lines and
asked whether it matched; this asks whether the rule ACTUALLY pulled a value
out of a non-assay column of the real table. Five invented rows are a guess
about what over-reach looks like. The patent's own columns are the thing.

WHAT THIS CANNOT SEE, so it is not the only check in the loop
------------------------------------------------------------
A fabricated UNIT produces usable records, contradicts nothing, and comes from
a legitimate assay column. US9221791 is the case: no unit anywhere near its
`CYP2C9 IC50` columns, a model proposed `nM`, the patent says µM/mM/µg/mL and
never once says nM, and 440 records shipped 1000x low with coverage going UP.
Nothing measurable about the output catches that — the check has to be against
the DOCUMENT, which is why `rules.ground()` still drops ungrounded names and
units before a rule is ever applied. That is not a judgement about rule
quality; it is the same class of statement as "the regex compiles".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..sources.uspto_assays import (
    MS, MW, NMR, RT, STRUCTURE, _header_rows_of, build_columns, parse_value,
)
from .rules import COLUMN_MAP, ROW_REGEX, Rule

# Columns the reader positively identified as carrying something that is not a
# measurement. UNKNOWN and SUBSTITUENT are deliberately absent: an unheaded or
# unclassified column is the AMBIGUOUS case, and claiming it is the single most
# common legitimate repair. Only a positive identification counts as a conflict.
NOT_A_MEASUREMENT = {MW, MS, NMR, RT, STRUCTURE}


@dataclass
class Outcome:
    """What running this rule on this patent did. Observed, not scored."""
    positive: bool = False
    records: int = 0
    usable: int = 0
    compounds: int = 0
    # Each string is one specific thing the rule got wrong about THIS patent,
    # phrased so it can be handed straight back to the model as the reason to
    # try again. Empty on a positive outcome.
    conflicts: list[str] = field(default_factory=list)

    @property
    def detail(self) -> str:
        """The retry message. Says what happened, never what to do instead."""
        if self.positive:
            return (f"{self.records} records, {self.usable} usable, "
                    f"{self.compounds} compounds")
        if not self.records:
            return "the rule ran and produced no records at all"
        return "; ".join(self.conflicts)


def _reader_cells(table) -> dict[tuple[int, int], float]:
    """Every cell the deterministic reader already reads a number out of.

    Keyed by (row, column) so a comparison is exact — no matching records by
    name. The reader names columns from the header and a repair names them from
    the model's reading of the header, so those two strings routinely differ for
    the same column; a key built on them would report contradictions that are
    only spelling. The cell is the thing both sides actually looked at.
    """
    _, data = _header_rows_of(table)
    out: dict[tuple[int, int], float] = {}
    for ri, row in enumerate(data):
        for ci, cell in enumerate(row):
            parsed = parse_value(cell.text)
            if parsed and parsed.get("value_numeric") is not None:
                out[(ri, ci)] = parsed["value_numeric"]
    return out


def _column_conflicts(rule: Rule, table) -> list[str]:
    """Assay columns the rule claims that the reader identified as non-assay."""
    if rule.kind != COLUMN_MAP:
        # VALUE_PATTERN is confined to `col.kind == ASSAY` inside `apply_rule`
        # and so cannot reach these columns; BIN_KEY re-reads records the reader
        # already produced and never picks a column at all.
        return []
    cols = build_columns(table)
    bad = []
    for a in rule.payload.get("assays") or []:
        i = a.get("index")
        if not isinstance(i, int) or i >= len(cols):
            continue
        if cols[i].kind in NOT_A_MEASUREMENT:
            bad.append(
                f"column {i} ({cols[i].header.strip()!r}) reads as "
                f"{cols[i].kind}, not a measurement, but the rule maps it as "
                f"the assay {a.get('name')!r}")
    return bad


def _value_conflicts(rule: Rule, produced: list, table) -> list[str]:
    """Cells where the rule reports a different number than the reader does.

    Only ROW_REGEX can do this. COLUMN_MAP hands each cell to `parse_value` —
    the reader's own parser — so it cannot disagree about a number, only about
    which column a number belongs to (`_column_conflicts`). VALUE_PATTERN skips
    every cell `parse_value` already reads, by construction in `apply_rule`.
    So this is where a regex that learned "a number appears here" shows up.
    """
    if rule.kind != ROW_REGEX:
        return []
    known = _reader_cells(table)
    if not known:
        return []
    _, data = _header_rows_of(table)
    cols = build_columns(table)
    non_assay = {i for i, c in enumerate(cols) if c.kind in NOT_A_MEASUREMENT}
    bad: list[str] = []
    # A row_regex flattens the row to `cell | cell | cell`, so the column a
    # value came from is not recorded. It is recoverable: find the cell whose
    # own parse equals the value the rule emitted.
    vals = {r.value_numeric for r in produced if r.value_numeric is not None}
    for ri in range(len(data)):
        for ci in non_assay:
            if (ri, ci) in known and known[(ri, ci)] in vals:
                bad.append(
                    f"a value it emitted ({known[(ri, ci)]:g}) is the content of "
                    f"column {ci} ({cols[ci].header.strip()!r}), which reads as "
                    f"{cols[ci].kind} — the pattern is matching numbers, not "
                    f"measurements")
                break
        if bad:
            break
    return bad


def measure(rule: Rule, produced: list, table) -> Outcome:
    """Run the gate. `produced` is what `apply_rule` just returned."""
    usable = [r for r in produced if r.is_usable]   # reported, never gated
    oc = Outcome(
        records=len(produced),
        usable=len(usable),
        compounds=len({r.cid for r in produced if r.cid}),
    )
    if not produced:
        return oc                      # detail explains; nothing else to check
    oc.conflicts = _column_conflicts(rule, table) + _value_conflicts(
        rule, produced, table)
    oc.positive = not oc.conflicts
    return oc


def summarise(rule: Rule, oc: Outcome) -> str:
    """One line for the log and the journal."""
    return f"{rule.kind} {'KEPT' if oc.positive else 'REJECTED'}: {oc.detail}"
