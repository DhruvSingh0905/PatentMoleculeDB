"""Build the 15-patent text-dominant, BDB-covered test set.

Phase 1 — universe scan over BindingDB_All.tsv (~8 GB, one pass via awk).
Phase 2 — per-candidate GP HTML fetch + `audit_patent_format`.
Phase 3 — filter to text-dominant with margin, take top 15.
Phase 4 — print summary table.

Outputs (all gitignored under output_v2/test_set_v2/):
    bdb_candidates.tsv  — every US patent with ≥100 BDB rows (Phase 1).
    test_set.txt        — final 15 patent IDs, one per line.
    test_set_meta.json  — per-patent signals + counts.

Reuses (no new extraction code):
    routes/google_patents.fetch_patent_text
    routes/google_patents.audit_patent_format
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from patentdb.core import config
from patentdb.routes.google_patents import (
    audit_patent_format,
    fetch_patent_text,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
BDB_TSV   = REPO_ROOT / "data" / "BindingDB_All.tsv"
OUT_DIR   = REPO_ROOT / "output_v2" / "test_set_v2"
EXISTING_TEST_PATENTS = {"US8952177", "US10899738", "US10214537"}

# Filter thresholds (see plan for rationale).
MIN_BDB_ROWS              = 100
# `classify_route` fires text-dominant when n_embedded_smiles >= 30.
# Match the production threshold directly — extra margin was throwing
# out legitimate text-dominant patents on the first pass.
MIN_EMBEDDED_SMILES       = 30
MAX_MARKUSH_PHRASES       = 4     # markush-dominant rule fires at ≥5
MIN_DESCRIPTION_CHARS     = 50_000
TARGET_TEST_SET_SIZE      = 15
GP_FETCH_BUDGET           = 200   # broaden the pool now that filter is at production strictness


def phase1_bdb_universe() -> list[tuple[str, int]]:
    """Stream-process BindingDB_All.tsv via awk; return [(patent_id, row_count), ...]
    sorted descending by row count, filtered to US patents with ≥MIN_BDB_ROWS rows
    and excluding the existing test patents.
    """
    if not BDB_TSV.exists():
        sys.exit(f"ERROR: {BDB_TSV} not found; expected at repo root.")

    logger.info("phase 1: scanning %s (this is the slow step, ~5-10 min)…", BDB_TSV)
    # `Patent Number` is column 22 (1-indexed). Use awk to extract that column
    # only, then `sort | uniq -c` to count rows per patent. Avoids loading
    # the whole file in Python. Shell-quote the path because the repo root
    # may contain spaces.
    import shlex
    bdb_quoted = shlex.quote(str(BDB_TSV))
    awk_cmd = (
        f"awk -F'\\t' 'NR>1 && $22 ~ /^US/ {{print $22}}' {bdb_quoted} "
        f"| sort | uniq -c | sort -rn"
    )
    result = subprocess.run(
        ["bash", "-c", awk_cmd],
        capture_output=True, text=True, check=True,
    )
    universe: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        count_str, patent_id = line.split(None, 1)
        n = int(count_str)
        if n < MIN_BDB_ROWS:
            break  # sorted descending — first sub-threshold means we're done
        if patent_id in EXISTING_TEST_PATENTS:
            continue
        universe.append((patent_id, n))

    logger.info(
        "phase 1: %d candidate US patents with ≥%d BDB rows (excluded 3 test patents)",
        len(universe), MIN_BDB_ROWS,
    )

    # Persist the full list to disk for spot-checks / future runs.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand_path = OUT_DIR / "bdb_candidates.tsv"
    with cand_path.open("w") as f:
        f.write("patent_id\tn_bdb_rows\n")
        for pid, n in universe:
            f.write(f"{pid}\t{n}\n")
    logger.info("phase 1: wrote %s", cand_path)
    return universe


def phase2_gp_audit(universe: list[tuple[str, int]]) -> list[dict]:
    """For up to GP_FETCH_BUDGET top candidates, fetch GP HTML and run
    audit_patent_format. Returns a list of dicts with the four signals
    plus n_bdb_rows."""
    audited: list[dict] = []
    for i, (pid, n_bdb) in enumerate(universe[:GP_FETCH_BUDGET]):
        logger.info("phase 2: [%d/%d] %s (%d BDB rows)", i + 1, GP_FETCH_BUDGET, pid, n_bdb)
        try:
            fetch_patent_text(pid)        # populates the cache; idempotent
            audit = audit_patent_format(pid)
        except Exception as e:
            logger.warning("phase 2: %s — fetch/audit failed: %r", pid, e)
            continue
        # description length + figure count come from the cache file
        # (FormatAudit doesn't surface them).
        cache_path = config.OUTPUT_DIR / "gpatents_cache" / f"{pid}.json"
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}
        audited.append({
            "patent_id": pid,
            "n_bdb_rows": n_bdb,
            "n_embedded_smiles": getattr(audit, "n_embedded_smiles", 0),
            "n_phrase_hits_markush": getattr(audit, "n_phrase_hits_markush", 0),
            "has_substituent_table": bool(getattr(audit, "has_substituent_table", False)),
            "description_chars": len(cache.get("description", "") or ""),
            "figure_count": len(cache.get("figure_image_urls", []) or []),
        })
    return audited


def phase3_filter(audited: list[dict]) -> list[dict]:
    """Apply the text-dominant criteria with margin; sort by n_embedded_smiles
    descending; take top TARGET_TEST_SET_SIZE."""
    passing = [
        a for a in audited
        if a["n_embedded_smiles"]    >= MIN_EMBEDDED_SMILES
        and a["n_phrase_hits_markush"] <= MAX_MARKUSH_PHRASES
        and not a["has_substituent_table"]
        and a["description_chars"]   >= MIN_DESCRIPTION_CHARS
    ]
    passing.sort(key=lambda a: a["n_embedded_smiles"], reverse=True)
    selected = passing[:TARGET_TEST_SET_SIZE]
    logger.info(
        "phase 3: %d/%d candidates passed thresholds; took top %d",
        len(passing), len(audited), len(selected),
    )

    test_set_path = OUT_DIR / "test_set.txt"
    meta_path     = OUT_DIR / "test_set_meta.json"
    with test_set_path.open("w") as f:
        for a in selected:
            f.write(f"{a['patent_id']}\n")
    meta_path.write_text(json.dumps(selected, indent=2))
    logger.info("phase 3: wrote %s + %s", test_set_path, meta_path)
    return selected


def phase4_report(selected: list[dict]) -> None:
    """Print a summary table for human review."""
    print()
    print(f"{'patent_id':>18}  {'n_bdb':>5}  {'n_smiles':>8}  {'n_markush':>9}  "
          f"{'has_sub_tbl':>11}  {'desc_kb':>7}  {'figs':>4}")
    print("-" * 78)
    for a in selected:
        print(
            f"{a['patent_id']:>18}  "
            f"{a['n_bdb_rows']:>5}  "
            f"{a['n_embedded_smiles']:>8}  "
            f"{a['n_phrase_hits_markush']:>9}  "
            f"{str(a['has_substituent_table']):>11}  "
            f"{a['description_chars'] // 1024:>7}  "
            f"{a['figure_count']:>4}"
        )
    print()
    print(f"Test set written to {OUT_DIR / 'test_set.txt'}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    universe = phase1_bdb_universe()
    audited  = phase2_gp_audit(universe)
    selected = phase3_filter(audited)
    phase4_report(selected)


if __name__ == "__main__":
    main()
