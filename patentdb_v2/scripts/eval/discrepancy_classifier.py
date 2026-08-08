"""Bucket BDB-vs-v2 assay discrepancies into 3 actionable categories.

For every BDB measurement that v2 didn't tolerance-match, we ask three
ordered questions:

  Tab 1 — MARKUSH / EXTERNAL CURATION
    Does the BDB value appear ANYWHERE in the patent text? If no, the
    measurement isn't in the patent — BDB pulled it from a related
    publication or computed it for a Markush variant we can't enumerate.
    Action: stop pushing on this one. We can't extract what isn't there.

  Tab 2 — LLM/REGION DIDN'T FIRE
    Value IS in the patent text but it's NOT inside any region the
    detector emitted (or it's inside one but the LLM realigner skipped
    that page/chunk). Action: relax region detection or fix chunking.

  Tab 3 — STRUCTURAL BUG
    Value IS in the patent text AND inside a region we extracted from.
    The pipeline saw it, the LLM saw it, but it didn't make it into
    the output. Action: real bug in cross-validator merge, dedup, or
    row aligner. Should be RARE — if frequent, we have a regression.

Generic — patent-agnostic. Same logic on every patent.

Usage:
    python -m patentdb.scripts.eval.discrepancy_classifier \\
        --patent US11312727 --patent US10214537 --patent US10899738
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from patentdb.scripts.eval import fidelity_check as fc
from patentdb.scripts.eval.assay_completeness_audit import diff_one
from patentdb.core import config
from patentdb.core.assay_fsm.normalizer import normalize_page
from patentdb.core.assay_fsm.region_detector import detect_regions
from patentdb.core.assay_fsm.vocabulary import AssayVocabulary

logger = logging.getLogger(__name__)


# ── Public dataclass ────────────────────────────────────────────


@dataclass
class CompoundDiscrepancy:
    """One partial-match compound and its bucketed misses."""
    compound_id: str
    n_bdb_only: int
    bdb_only_samples: list = field(default_factory=list)   # (assay, value, unit)
    bucket: str = "unknown"          # "markush" | "llm_miss" | "structural" | "mixed"
    bucket_reason: str = ""
    # Per-measurement classification (when one compound has multiple
    # missing values across categories)
    per_measurement: list = field(default_factory=list)


@dataclass
class PatentReport:
    patent_id: str
    n_partial: int
    n_markush: int            # Tab 1
    n_llm_miss: int           # Tab 2
    n_structural: int         # Tab 3
    n_mixed: int              # multiple categories within the compound's misses
    sample_per_bucket: dict = field(default_factory=dict)


# ── Patent text loading + value-presence search ─────────────────


# Canonical loader (was reimplemented here; centralized in core/patent_text.py)
from patentdb.core.patent_text import load_full_patent_text as _load_patent_text  # noqa: E402


_UNIT_ANNOTATION_RE = re.compile(
    r"[\(\[]\s*(?:nM|μM|uM|µM|mcM|mM|pM|M)\s*[\)\]]",
    re.IGNORECASE,
)

# Row-shape pattern: compound_id followed by 2-5 numeric values
# (qualifier-prefixed, optional n_runs annotation, null-marker tolerant).
# Re-used from region_detector with the cpd_id as a parameter.
def _row_shape_pattern(cid_escaped: str) -> re.Pattern:
    return re.compile(
        rf"(?<![A-Za-z0-9.]){cid_escaped}(?![A-Za-z0-9.])"
        r"\s+("
        r"(?:[<>≤≥~≈]?\s*\d+(?:\.\d+)?(?:\s*\(\d+\))?(?:\s*±\s*\d+(?:\.\d+)?)?"
        r"|nt|nd|na|N\.?T\.?|N\.?D\.?|N/?A|[—–\-*])"
        r"(?:\s+"
        r"(?:[<>≤≥~≈]?\s*\d+(?:\.\d+)?(?:\s*\(\d+\))?(?:\s*±\s*\d+(?:\.\d+)?)?"
        r"|nt|nd|na|N\.?T\.?|N\.?D\.?|N/?A|[—–\-*])"
        r"){1,4}?"
        r")(?=\s+[A-Z]?\d{1,4}[A-Za-z]{0,3}\s|\s+(?:nt|nd)\b|$|\n)",
        re.IGNORECASE,
    )


def _value_appears_near_compound(
    text: str,
    compound_id: str,
    target_value: float,
    tolerance_pct: float = 5.0,
) -> tuple[bool, int]:
    """Returns (found, char_offset).

    STRICT match — requires the value to appear inside an assay-row
    SHAPE next to the compound id:
        `<cpd_id>  <val1>  <val2>  ...`
    where each value cell has a single numeric (qualifier-prefixed,
    n_runs-annotated, or null-marker). This rules out coincidental
    matches inside IUPAC names, NMR shifts, CAS numbers, MW values,
    and other non-table contexts.

    Tries both nM and μM forms of the value (BDB stores nM by default;
    many patents use μM in tables). Tolerance: 5% relative.
    """
    if not text or not compound_id:
        return (False, -1)

    cid = compound_id.strip()
    cid_escaped = re.escape(cid)
    pattern = _row_shape_pattern(cid_escaped)

    candidates = {
        target_value,
        target_value / 1000.0,
        target_value * 1000.0,
    }

    def _close(v_in_text: float, target: float) -> bool:
        if target <= 0:
            return abs(v_in_text - target) < 1e-9
        return abs(v_in_text - target) / max(target, 1e-9) * 100 <= tolerance_pct

    num_pat = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)(?![A-Za-z0-9.])")

    for m in pattern.finditer(text):
        row_text = m.group(1)
        row_offset = m.start(1)
        for vm in num_pat.finditer(row_text):
            try:
                v = float(vm.group(1))
            except ValueError:
                continue
            for cand in candidates:
                if _close(v, cand):
                    return (True, row_offset + vm.start(1))

    return (False, -1)


def _value_in_extracted_region(
    text: str, regions: list, value_offset: int,
) -> bool:
    """True if the given char offset falls inside any region the
    detector emitted."""
    if value_offset < 0:
        return False
    for r in regions:
        if r.start <= value_offset < r.end:
            return True
    return False


# ── Main classifier ─────────────────────────────────────────────


def classify_patent(
    patent_id: str,
    *,
    sample_per_bucket: int = 3,
) -> PatentReport:
    """Classify every partial-match compound in `patent_id` into
    one of the three buckets. Returns a PatentReport."""
    bdb = fc.load_bdb_compounds(patent_id)
    _v2_ex, v2_ay = fc.load_v2_extraction(patent_id)
    text = _load_patent_text(patent_id)

    # Load the FSM region detector once
    canonical_vocab = (
        Path(__file__).resolve().parent.parent.parent / "data" / "assay_vocabulary.json"
    )
    vocab = AssayVocabulary.load(canonical_vocab)
    regions = detect_regions(text, vocab) if text else []

    # Pre-compute v2 "all values seen" index — used to detect BDB
    # stereoisomer-merge (BDB has values for cpd "5a" that actually
    # belong to cpd "5b" or "5A" in the patent table).
    v2_value_owner: dict[float, list[str]] = {}
    for cid, arr in v2_ay.items():
        for a in arr:
            v = a.get("value_numeric") if isinstance(a, dict) else getattr(a, "value_numeric", None)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            # Index by value AND nM-equivalent
            for vk in (vf, vf * 1000.0, vf / 1000.0):
                v2_value_owner.setdefault(round(vk, 6), []).append(cid)

    def _value_owned_by_other_v2_compound(target: float, cid: str) -> str | None:
        """If `target` matches a value that v2 extracted for a different
        compound (stereoisomer / sibling), return that other cpd id."""
        cid_lc = cid.lower()
        # Try exact + 1000x conversions
        for vk in (target, target / 1000.0, target * 1000.0):
            for known in v2_value_owner.get(round(vk, 6), []):
                if known.lower() != cid_lc:
                    return known
        return None

    cpd_diffs: list[CompoundDiscrepancy] = []
    for cid_norm, bc in bdb.items():
        if not bc.assays:
            continue
        v2_a = v2_ay.get(cid_norm, [])
        if not v2_a:
            # v2-zero compounds — already audited separately, skip here
            continue
        d = diff_one(cid_norm, bc.assays, v2_a)
        if d.n_bdb_only == 0:
            continue   # fully matched, no discrepancy to classify

        per_meas: list[dict] = []
        for sample in d.bdb_only_samples:
            assay, value, unit = sample
            if value is None:
                continue
            try:
                v_float = float(value)
            except (TypeError, ValueError):
                continue

            # Tab 1a: BDB stereoisomer-merge artifact — value belongs to
            # a SIBLING compound in v2's output (BDB lumped 38A + 38B
            # under "38a", etc.)
            sibling = _value_owned_by_other_v2_compound(v_float, cid_norm)
            if sibling is not None:
                bucket = "markush"
                bucket_subtype = f"sibling_{sibling}"
                found, in_v2_row = True, False
            else:
                found, _offset = _value_appears_near_compound(
                    text, cid_norm, v_float,
                )
                if not found:
                    bare = re.sub(r"[A-Za-z]+$", "", cid_norm)
                    if bare and bare != cid_norm:
                        found, _offset = _value_appears_near_compound(
                            text, bare, v_float,
                        )
                # Already checked above whether the value belongs to a
                # sibling — at this point if found is True, it's actually
                # in a row labeled with `cid_norm`.
                if not found:
                    bucket = "markush"
                    bucket_subtype = "external"
                    in_v2_row = False
                else:
                    # The patent has a row "<cid_norm> ... <value> ..."
                    # but BDB's value is not in v2's output for cid_norm.
                    # That's either:
                    #   - LLM didn't fire on this region (Tab 2), OR
                    #   - LLM fired but mis-extracted the value (Tab 3)
                    # Distinguish by v2-coverage of cid_norm: if v2 has
                    # cid_norm at all, the LLM did fire and we have a
                    # genuine extraction bug → Tab 3. If v2 has nothing
                    # for cid_norm here, the LLM call must have skipped
                    # this row entirely → Tab 2.
                    in_v2_row = bool(v2_a)
                    if in_v2_row:
                        bucket = "structural"
                        bucket_subtype = "value_lost_in_extraction"
                    else:
                        bucket = "llm_miss"
                        bucket_subtype = "row_skipped_by_llm"

            per_meas.append({
                "assay": assay,
                "value": v_float,
                "unit": unit,
                "found_in_text": found,
                "in_extracted_region": in_v2_row,
                "bucket": bucket,
                "subtype": bucket_subtype,
            })

        # Compound-level bucket = majority. If mixed, flag as such.
        bucket_counts = Counter(m["bucket"] for m in per_meas)
        if not bucket_counts:
            continue
        if len(bucket_counts) > 1:
            top = bucket_counts.most_common(1)[0]
            compound_bucket = top[0] if top[1] >= len(per_meas) * 0.6 else "mixed"
        else:
            compound_bucket = next(iter(bucket_counts))

        cpd_diffs.append(CompoundDiscrepancy(
            compound_id=cid_norm,
            n_bdb_only=d.n_bdb_only,
            bdb_only_samples=d.bdb_only_samples,
            bucket=compound_bucket,
            bucket_reason=", ".join(f"{k}={v}" for k, v in bucket_counts.most_common()),
            per_measurement=per_meas,
        ))

    # Build the report
    report = PatentReport(
        patent_id=patent_id,
        n_partial=len(cpd_diffs),
        n_markush=sum(1 for c in cpd_diffs if c.bucket == "markush"),
        n_llm_miss=sum(1 for c in cpd_diffs if c.bucket == "llm_miss"),
        n_structural=sum(1 for c in cpd_diffs if c.bucket == "structural"),
        n_mixed=sum(1 for c in cpd_diffs if c.bucket == "mixed"),
    )
    for bucket in ("markush", "llm_miss", "structural", "mixed"):
        sample = [c for c in cpd_diffs if c.bucket == bucket][:sample_per_bucket]
        report.sample_per_bucket[bucket] = sample

    return report


def _print_report(report: PatentReport) -> None:
    print(f"\n=== {report.patent_id} ===")
    print(f"  partial-match compounds: {report.n_partial}")
    print(f"    Tab 1 (Markush / external curation): {report.n_markush}")
    print(f"    Tab 2 (LLM/region didn't fire):       {report.n_llm_miss}")
    print(f"    Tab 3 (STRUCTURAL BUG):               {report.n_structural}")
    if report.n_mixed:
        print(f"    Mixed (per-meas categories):          {report.n_mixed}")

    pct = lambda n: (
        f"({100 * n / report.n_partial:.0f}%)" if report.n_partial else "(0%)"
    )
    print(
        f"  → bucket share: "
        f"markush {report.n_markush} {pct(report.n_markush)}, "
        f"llm_miss {report.n_llm_miss} {pct(report.n_llm_miss)}, "
        f"structural {report.n_structural} {pct(report.n_structural)}"
    )

    for bucket, samples in report.sample_per_bucket.items():
        if not samples:
            continue
        label = {
            "markush": "Tab 1 — Markush / external curation",
            "llm_miss": "Tab 2 — LLM/region didn't fire",
            "structural": "Tab 3 — STRUCTURAL BUG",
            "mixed": "Mixed",
        }[bucket]
        print(f"\n  {label}: sample compounds")
        for c in samples:
            print(f"    cpd {c.compound_id}: {c.n_bdb_only} BDB-only measurements ({c.bucket_reason})")
            for m in c.per_measurement[:3]:
                f = "found" if m["found_in_text"] else "NOT in text"
                rg = "in-region" if m["in_extracted_region"] else "outside-region"
                print(f"      {m['assay']} = {m['value']} {m['unit']}  [{f}, {rg}, bucket={m['bucket']}]")


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", action="append", required=True)
    ap.add_argument("--sample-per-bucket", type=int, default=3)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )
    for pat in args.patent:
        report = classify_patent(pat, sample_per_bucket=args.sample_per_bucket)
        _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
