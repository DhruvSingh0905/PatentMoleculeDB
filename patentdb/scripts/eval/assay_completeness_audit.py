"""Per-compound assay completeness audit.

For each compound in BindingDB for a given patent, report:
  - n_bdb_assays      — number of (assay, value) tuples BDB has
  - n_v2_assays       — number of (assay, value) tuples v2 has
  - n_matched         — values that match (within tolerance)
  - n_v2_extra        — values v2 has that BDB doesn't (multi-column)
  - n_bdb_extra       — values BDB has that v2 doesn't (gaps to investigate)

Different from `fidelity_check.py`: that report counts assay-coverage
in aggregate; this one shows per-compound diff lists for spot-checking
"are we capturing ALL measurements for THIS compound", including
replicates and multi-experiment columns. Patents commonly report:
  - IC50 against multiple kinases (PI3Kδ, PI3Kα, PI3Kγ, ...)
  - EC50 in multiple cell lines
  - Replicates with mean ± SD

Usage
-----
    python3 -m patentdb.scripts.eval.assay_completeness_audit \\
        --patent US10899738 --patent US10214537 \\
        [--show-misses N]   # show first N compound-level diffs
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass

from ...core.units import _NM_PER_UNIT, _normalize_unit_token, value_to_uM
from . import fidelity_check as fc

logger = logging.getLogger(__name__)


# ── Measurement identity (duplicate-safe counting) ──────────────────


_NAME_NOISE = re.compile(r"[^a-z0-9]+")


def measurement_key(row: dict) -> tuple:
    """Identity of ONE measurement, deliberately blind to `source`.

    `assay_tables` carries rows from TWO extraction tiers (the existing
    one + uspto_xml), each tagged with its own `source` and never
    deduplicated — ~59 % of the XML tier's rows have a value-equal
    counterpart the other tier already shipped. Any metric that counts
    raw rows therefore counts one measurement twice the moment the
    second tier lands.

    Key = (assay name, value, unit, range). The value is canonicalised
    to μM so a tier printing `290 nM` collapses onto one printing
    `0.29 μM`; an unrecognised unit keeps its own token so `10 %` never
    merges with `10 nM`. `source` is excluded — that is the whole point.

    assay_name is IN the key, against the first instinct to key on value
    alone. Measured over the 22-patent corpus (35,888 rows, still ONE
    tier, so every collapse there is a real measurement lost): a
    value-only key collapses 18.57 % of rows — 70.0 % of US11566007 and
    31.6 % of US10899738, whose multi-column panels legitimately report
    the SAME number against several targets. Adding the normalised name
    collapses 0.26 %, and those are byte-identical rows. Discarding a
    real multi-target readout is precisely the corruption this module
    exists to surface, so the narrower key wins.
    """
    unit = _normalize_unit_token(row.get("unit"))
    value = row.get("value_numeric")
    if value is not None and unit in _NM_PER_UNIT:
        # round(): float scaling (0.29 μM -> 290.00000000000006 nM -> back)
        # must not split one measurement into two keys.
        value_key = ("μM", round(value_to_uM(value, unit), 9))
    else:
        value_key = (unit.lower(), value)
    return (
        _NAME_NOISE.sub(" ", (row.get("assay_name") or "").lower()).strip(),
        value_key,
        row.get("qualifier"),
        row.get("value_low"),
        row.get("value_high"),
        row.get("bin"),
    )


def distinct_measurements(rows: list) -> list:
    """The first row of each `measurement_key` group, order preserved.

    Order-preserving so the caller's own reporting order (and any
    first-wins field it still reads) does not change for a patent that
    has no duplicates at all.
    """
    seen: set = set()
    out: list = []
    for r in rows or []:
        k = measurement_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


@dataclass
class CompoundAssayDiff:
    """Per-compound v2-vs-BDB assay completeness diff."""
    compound_id: str
    n_bdb: int
    n_v2: int
    n_matched: int
    n_bdb_only: int          # values in BDB but not in v2
    n_v2_only: int           # values in v2 but not in BDB
    bdb_only_samples: list   # [(assay, value, unit), ...]
    v2_only_samples: list


def diff_one(
    cid: str,
    bdb_assays: list,
    v2_assays: list,
    tol_pct: float = 5.0,
) -> CompoundAssayDiff:
    """Compare two lists of assay measurements via tolerance match.

    v2 rows are collapsed to DISTINCT measurements first. The matcher
    below is greedy and 1:1 — a BDB row can be consumed only once — so a
    value-equal duplicate from a second tier could never be matched and
    fell straight into `n_v2_only`, the bucket `_print_report` prints
    under "compounds where v2 has EXTRA assays (multi-column captures)".
    Verified: bdb=[10.0 nM] vs v2=[10.0 nM] gives n_v2_only=0; appending
    the same measurement tagged source="uspto_xml" gave n_v2_only=1 —
    merge-tier corruption reading as a positive coverage finding.
    """
    v2_assays = distinct_measurements(v2_assays)
    matched_v2_idx: set[int] = set()
    n_matched = 0
    bdb_only = []
    for ba in bdb_assays:
        ba_val = ba["value"]
        ba_unit = ba["unit"]
        m_idx = None
        for i, va in enumerate(v2_assays):
            if i in matched_v2_idx:
                continue
            v2_val = va.get("value_numeric")
            if v2_val is None:
                continue
            if fc._values_match(ba_val, ba_unit, float(v2_val), va.get("unit") or "", tol_pct):
                m_idx = i
                break
        if m_idx is not None:
            matched_v2_idx.add(m_idx)
            n_matched += 1
        else:
            bdb_only.append((ba.get("assay"), ba_val, ba_unit))
    v2_only = []
    for i, va in enumerate(v2_assays):
        if i in matched_v2_idx:
            continue
        v2_only.append((
            (va.get("assay_name") or "")[:40],
            va.get("value_numeric"),
            va.get("unit") or "",
        ))
    return CompoundAssayDiff(
        compound_id=cid,
        n_bdb=len(bdb_assays),
        n_v2=len(v2_assays),
        n_matched=n_matched,
        n_bdb_only=len(bdb_only),
        n_v2_only=len(v2_only),
        bdb_only_samples=bdb_only,
        v2_only_samples=v2_only,
    )


def audit_patent(patent_id: str, show_misses: int = 5) -> dict:
    bdb = fc.load_bdb_compounds(patent_id)
    _v2_ex, v2_ay = fc.load_v2_extraction(patent_id)

    if not bdb:
        return {
            "patent_id": patent_id, "n_compounds_audited": 0,
            "n_compounds_complete": 0, "n_compounds_partial": 0,
            "n_compounds_missing": 0, "diffs": [],
        }

    diffs: list[CompoundAssayDiff] = []
    n_complete = n_partial = n_missing = 0
    for cid_norm, bc in bdb.items():
        if not bc.assays:
            continue
        v2_assays = v2_ay.get(cid_norm, [])
        d = diff_one(cid_norm, bc.assays, v2_assays)
        diffs.append(d)
        if d.n_bdb_only == 0 and d.n_matched > 0:
            n_complete += 1
        elif d.n_matched > 0:
            n_partial += 1
        else:
            n_missing += 1

    return {
        "patent_id": patent_id,
        "n_compounds_audited": len(diffs),
        "n_compounds_complete": n_complete,
        "n_compounds_partial": n_partial,
        "n_compounds_missing": n_missing,
        "show_misses": show_misses,
        "diffs": diffs,
    }


def _print_report(audit_result: dict) -> None:
    pid = audit_result["patent_id"]
    n = audit_result["n_compounds_audited"]
    nc = audit_result["n_compounds_complete"]
    np = audit_result["n_compounds_partial"]
    nm = audit_result["n_compounds_missing"]
    print(f"\n=== {pid} ===")
    print(f"  Compounds with assay data in BDB: {n}")
    print(f"    fully matched (all BDB assays found in v2): {nc} ({100*nc/max(1,n):.0f}%)")
    print(f"    partial (some BDB assays missing in v2):   {np} ({100*np/max(1,n):.0f}%)")
    print(f"    no v2 assays at all:                       {nm} ({100*nm/max(1,n):.0f}%)")

    diffs = audit_result["diffs"]
    show = audit_result["show_misses"]
    # Sample partial / missing — most informative for diagnosing gaps
    sample = sorted([d for d in diffs if d.n_bdb_only > 0],
                    key=lambda d: -d.n_bdb_only)[:show]
    if sample:
        print(f"\n  Top {len(sample)} compounds with missing BDB assays in v2:")
        for d in sample:
            print(f"    cpd {d.compound_id}: BDB has {d.n_bdb}, v2 has {d.n_v2}, "
                  f"matched {d.n_matched}, BDB-only={d.n_bdb_only}")
            for assay, val, unit in d.bdb_only_samples[:3]:
                print(f"      missing: {assay} = {val} {unit}")
            if d.v2_only_samples:
                for nm_, val, unit in d.v2_only_samples[:2]:
                    print(f"      v2-only:  {nm_} = {val} {unit}")
            print()

    # Also show compounds where v2 has MORE assays than BDB (multi-experiment captures)
    extra_sample = sorted([d for d in diffs if d.n_v2_only >= 2],
                          key=lambda d: -d.n_v2_only)[:3]
    if extra_sample:
        print(f"  Top compounds where v2 has EXTRA assays (multi-column captures):")
        for d in extra_sample:
            print(f"    cpd {d.compound_id}: BDB={d.n_bdb}, v2={d.n_v2}, "
                  f"v2-only={d.n_v2_only}")
            for nm_, val, unit in d.v2_only_samples[:3]:
                print(f"      v2 has: {nm_} = {val} {unit}")
            print()


def _cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", action="append", required=True)
    ap.add_argument("--show-misses", type=int, default=5)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")
    for pat in args.patent:
        result = audit_patent(pat, show_misses=args.show_misses)
        _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
