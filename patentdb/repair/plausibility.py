"""Does the extracted assay data make SENSE, judged without a reference database.

Every signal in `gap.py` asks whether extraction produced enough rows. None of
them can see a row that came out complete and wrong, and that class has cost
more than any coverage gap here: 99 records read `2.24 nM` where the patent says
300 nM, and every count was healthy because a wrong unit produces records
perfectly.

`value_check` catches those — against BindingDB. That is the wrong dependency
for a live check. BindingDB covers a fraction of what we extract, lags new
patents entirely, cites example numbers patents do not contain, and cannot say
anything at all about a compound it has never seen. A patent we get wrong in a
way BDB cannot see is exactly the patent nobody will notice.

So these are INDEPENDENT expectations: chemistry and internal consistency, no
external data. The shape is borrowed from declarative data-validation suites
(Great Expectations' model — a set of assertions, each producing violations
rather than raising, collected into one result): each check answers one
question, returns what violated it, and blocks nothing.

    n_runs_dropped        the cell stated a replicate count and we did not keep it
    potency_out_of_range  a potency that is not a potency at any real scale
    unnamed_assay         a usable measurement nobody can say the meaning of
    same_assay_disagrees  one assay, two blocks, medians orders of magnitude apart

The last one is the important one and the only one that catches a UNIFORMLY
wrong column. `2.24 nM` is perfectly plausible in isolation; it is only wrong
because the same assay reads 300 nM twelve rows earlier. Absolute bounds cannot
see that and a reference database should not have to.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

# nM is the common denominator; anything not here is not a concentration.
_TO_NM = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "μM": 1e3, "mM": 1e6,
          "pM": 1e-3, "M": 1e9}

# An assay whose value IS a concentration, so a magnitude check is meaningful.
# Deliberately narrow: a percentage, a ratio, a clearance and a half-life all
# live on scales of their own and must never be judged against this one.
_POTENCY = re.compile(r"IC\s*50|EC\s*50|\bK[id]\b|GI\s*50|CC\s*50|LD\s*50", re.I)

# Measured over 54,102 potency records in this corpus: 99.8% land between
# 1e-2 and 1e4 nM, and the tails are 91 records at 1e5, 8 at 1e6, 27 at or
# below zero. The window below is two orders of magnitude WIDER than the
# observed data on each side, because the job is catching a 1,000x unit error,
# not policing pharmacology — a real sub-picomolar binder should not be
# reported as a defect.
_MIN_NM = 1e-3            # 1 pM
_MAX_NM = 1e6             # 1 mM

# A cell that STATES a replicate count. Anchored to a number: `\(\d+\)$` alone
# matches `Tris(...)dipalladium(0)`, `C(10)` and crystallographic `9211(1)`.
_STATES_N = re.compile(
    r"^[<>~≈≥≤]?\s*\d[\d.,]*\s*(?:\(\s*(?P<a>\d{1,3})\s*\)"
    r"|.*?\bn\s*=\s*(?P<b>\d{1,3}))\s*$", re.I)

# How far apart two medians of the SAME assay must be before we call it a
# contradiction rather than biology. 100x: assay-to-assay variability spans
# maybe 10x, and the unit errors this exists to catch are 1,000x.
_DISAGREE_FACTOR = 100.0
_MIN_GROUP = 4            # a median of fewer rows than this is noise


@dataclass
class Flag:
    """One expectation, violated. Never raised — reported."""
    kind: str
    patent_id: str
    detail: str
    rows_at_stake: int = 0
    table_id: str | None = None
    examples: list[str] = field(default_factory=list)


def _nm(rec) -> float | None:
    if rec.value_numeric is None or rec.unit not in _TO_NM:
        return None
    return rec.value_numeric * _TO_NM[rec.unit]


def check_n_runs_dropped(patent_id: str, xml: str, records) -> list[Flag]:
    """Cells that printed a replicate count we did not keep.

    Sparse by nature — most patents never print one — so this is scored against
    the cells that STATE one, read off the raw table. Scoring it against all
    records would report a failure we cannot fix, and scoring it against parsed
    records would hide exactly the cells we failed on.
    """
    from ..sources.uspto_assays import _header_rows_of
    from ..sources.uspto_xml import assemble_blocks, parse_tables

    kept = sum(1 for r in records if r.n_runs is not None)
    stated, examples, per_table = 0, [], {}
    for t in assemble_blocks(parse_tables(xml)):
        _, data = _header_rows_of(t)
        for row in data:
            for c in row:
                s = c.text.strip()
                if s and _STATES_N.match(s):
                    stated += 1
                    per_table[t.table_id] = per_table.get(t.table_id, 0) + 1
                    if len(examples) < 6:
                        examples.append(s[:40])
    missed = stated - kept
    if stated < 5 or missed < max(3, stated * 0.1):
        return []
    worst = max(per_table, key=lambda k: per_table[k]) if per_table else None
    return [Flag(
        kind="n_runs_dropped", patent_id=patent_id, table_id=worst,
        rows_at_stake=missed, examples=examples,
        detail=(f"{stated} cells state a replicate count and only {kept} "
                f"records carry one — {missed} dropped. The count is printed in "
                f"the cell beside the measurement, or in its own `n = N` column; "
                f"`parse_value` reads the first form and `extract_from_tables` "
                f"the second."))]


def check_potency_out_of_range(patent_id: str, xml: str, records) -> list[Flag]:
    """Potencies that are not potencies at any real concentration."""
    bad, examples = [], []
    for r in records:
        if not r.is_usable or not _POTENCY.search(r.assay_name or ""):
            continue
        v = _nm(r)
        if v is None:
            continue
        if v <= 0 or v < _MIN_NM or v > _MAX_NM:
            bad.append(r)
            if len(examples) < 6:
                examples.append(f"{r.cid}: {r.value_numeric} {r.unit} "
                                f"({r.assay_name or '?'})")
    if not bad:
        return []
    return [Flag(
        kind="potency_out_of_range", patent_id=patent_id,
        table_id=bad[0].table_id, rows_at_stake=len(bad), examples=examples,
        detail=(f"{len(bad)} potency value(s) fall outside {_MIN_NM:g}-{_MAX_NM:g} "
                f"nM once converted. Measured over this corpus, 99.8% of "
                f"potencies land between 1e-2 and 1e4 nM, so a value this far "
                f"out is far more likely a unit we attached wrongly than a "
                f"measurement the patent reports."))]


def check_unnamed_assay(patent_id: str, xml: str, records) -> list[Flag]:
    """Usable measurements nobody can say the meaning of.

    A number with a unit and no assay name is not a result. It survives
    `is_usable` because that asks whether the fields are populated, and
    "unnamed assay" is a populated field.
    """
    unnamed = [r for r in records if r.is_usable
               and (not r.assay_name or "unnamed" in r.assay_name.lower())]
    if len(unnamed) < 5:
        return []
    by_table: dict[str, int] = {}
    for r in unnamed:
        by_table[r.table_id] = by_table.get(r.table_id, 0) + 1
    worst = max(by_table, key=lambda k: by_table[k])
    return [Flag(
        kind="unnamed_assay", patent_id=patent_id, table_id=worst,
        rows_at_stake=len(unnamed),
        examples=[f"{r.cid}: {r.value_numeric} {r.unit}" for r in unnamed[:6]],
        detail=(f"{len(unnamed)} usable measurement(s) carry no assay name "
                f"({by_table[worst]} on {worst}). The header exists in the "
                f"document; it did not survive our multi-row merge. A value "
                f"whose meaning is unknown cannot be used."))]


def check_same_assay_disagrees(patent_id: str, xml: str, records) -> list[Flag]:
    """One assay name, two blocks, medians orders of magnitude apart.

    THE check that catches a uniformly wrong column, and the reason this module
    does not depend on BindingDB. US11420968 heads four columns as two pairs
    under `(IC50, nM)`; our merge put the unit on the second of each pair and
    the other two inherited `uM` from a caption. 111 values landed 1,000x low
    and every one of them was individually plausible. Nothing that looks at a
    value on its own can see that; the patent contradicting itself can.

    Compared per assay NAME, across blocks, on medians — a median needs no
    outlier handling and a 100x gap between two medians of the same assay in
    one document is not biology.
    """
    groups: dict[tuple[str, str], list[float]] = {}
    for r in records:
        if not r.is_usable or not _POTENCY.search(r.assay_name or ""):
            continue
        v = _nm(r)
        if v is None or v <= 0:
            continue
        groups.setdefault((r.assay_name.strip().lower(), r.table_id), []).append(v)

    by_assay: dict[str, list[tuple[str, float, int]]] = {}
    for (name, table), vals in groups.items():
        if len(vals) < _MIN_GROUP:
            continue
        by_assay.setdefault(name, []).append(
            (table, statistics.median(vals), len(vals)))

    out: list[Flag] = []
    for name, members in by_assay.items():
        if len(members) < 2:
            continue
        lo = min(members, key=lambda m: m[1])
        hi = max(members, key=lambda m: m[1])
        if lo[1] <= 0 or hi[1] / lo[1] < _DISAGREE_FACTOR:
            continue
        out.append(Flag(
            kind="same_assay_disagrees", patent_id=patent_id, table_id=lo[0],
            rows_at_stake=lo[2] + hi[2],
            examples=[f"{t}: median {m:.4g} nM over {n} rows"
                      for t, m, n in sorted(members, key=lambda m: m[1])],
            detail=(f"assay {name!r} reads a median of {lo[1]:.4g} nM on "
                    f"{lo[0]} ({lo[2]} rows) and {hi[1]:.4g} nM on {hi[0]} "
                    f"({hi[2]} rows) — {hi[1] / lo[1]:.0f}x apart in one "
                    f"document. Assay variability does not span that; a unit "
                    f"attached to one block and not the other does. No "
                    f"reference database was consulted: the patent disagrees "
                    f"with itself.")))
    return out


CHECKS = (check_n_runs_dropped, check_potency_out_of_range,
          check_unnamed_assay, check_same_assay_disagrees)


def audit(patent_id: str, xml: str, records) -> list[Flag]:
    """Run every expectation. Never raises: a broken check must not cost a run."""
    import logging

    out: list[Flag] = []
    for check in CHECKS:
        try:
            out.extend(check(patent_id, xml, list(records)))
        except Exception as e:                   # noqa: BLE001 — reporting only
            logging.getLogger(__name__).warning(
                "plausibility: %s raised on %s: %r", check.__name__, patent_id, e)
    return sorted(out, key=lambda f: -f.rows_at_stake)
