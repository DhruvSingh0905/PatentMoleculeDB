"""Export FSM-extracted assay data to a long-form CSV for hand-verification.

One row per (patent, compound, assay) tuple. Columns:
    patent_id              — the patent identifier
    compound_id            — sanitized compound id (lowercase, prefix-stripped)
    iupac_name             — from example_index.json if present
    target                 — best-effort guess from assay name (FLAP / PI3K / Menin / ...)
    assay_name             — the assay-name string the LLM/FSM emitted
    value_uM               — value normalized to μM (ChEMBL convention)
    value_raw              — original numeric the extractor produced
    value_unit             — original unit (μM / nM / mM / etc.)
    qualifier              — '<', '>', '~', '≤', '≥' or empty
    n_runs                 — replicate count if the extractor captured one
    is_null                — True when the cell was a "not tested" / "—" placeholder
    source                 — extraction tier that produced the row (e.g. uspto_xml,
                             letter_bin); blank on rows predating the tag. The only
                             thing separating two tiers' copies of one measurement.
    source_patent_class    — heuristic: "table_primary" / "prose_only" / "no_data" / "image_primary"

Patent-agnostic. No per-patent column or value rewriting. Whatever the
pipeline emits is what lands in the CSV — easier for a human reviewer
to flag mis-extractions when nothing has been "cleaned" downstream.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────


# Canonical unit converter (was reimplemented here; centralized in core/units.py)
from patentdb.core.units import value_to_uM as _to_uM


def _guess_target(assay_name: str) -> str:
    """Best-effort target name extraction from the assay label.

    Looks for known target tokens (FLAP, PI3K, Menin, SGK, ...) anywhere
    in the assay-name string. Returns the first match or "" if none.
    Generic — no per-patent dictionary; relies on the LLM having put
    the target name into the assay_name string at extraction time.
    """
    if not assay_name:
        return ""
    # Common target/cell-line tokens. This is descriptive, not
    # prescriptive — adding/removing entries here only changes the
    # `target` column; downstream consumers can do their own grouping.
    candidates = [
        "FLAP", "PI3K", "Menin", "SGK", "BRD", "LRRK", "JAK", "BTK",
        "EGFR", "VEGFR", "CDK", "GPR", "TRK", "BCR", "MAPK", "FAK",
        "A549", "HEK293", "MV4;11", "MV4-11", "MOLM13", "MOLM-13",
        "CD69", "LTB4", "FRET", "HTRF",
    ]
    al = assay_name.lower()
    for t in candidates:
        if t.lower() in al:
            return t
    return ""


def _classify_patent(n_compounds: int, n_measurements: int) -> str:
    """Heuristic patent classification — purely from extraction stats.

    Classified on COMPOUND count alone. `assay_tables` now merges two
    extraction tiers without deduplication, so `n_measurements` is a
    duplicate-contaminated row count: the old
    `n_measurements / n_compounds >= 1.5` gate crossed on a SINGLE
    duplicated row (50 compounds / 74 rows = 1.48 "small_table" ->
    50 / 75 = 1.50 "table_primary"), and no threshold on that ratio can
    be made safe — every one of them has an edge one duplicate can step
    over. Only a quantity duplicates cannot move is safe, and duplicates
    add rows, never compounds.

    Dropping the ratio is also the more honest label: `assay_tables`
    only contains compounds that HAVE assay rows, so ≥50 of them is a
    table by construction. Re-classified over the 22-patent corpus, 3
    patents move small_table -> table_primary — US9718825 (658
    compounds, ratio 1.48), US11286268 (207, 1.48) and US9302989 (126,
    1.09). Two of those sit on the exact 1.48 knife-edge the ratio made
    fragile, and calling a 658-compound assay table "small" was always
    wrong.
    """
    if n_compounds == 0 or n_measurements == 0:
        return "no_data_or_prose_only"
    if n_compounds >= 50:
        return "table_primary"
    if n_compounds >= 5:
        return "small_table"
    return "sparse"


# ── Main ────────────────────────────────────────────────────────


def export_csv(
    output_path: Path,
    patent_ids: list[str],
    extractions_root: Path,
) -> tuple[int, dict]:
    """Write one big long-form CSV combining all patents.

    Returns (n_rows_written, per_patent_counts).
    """
    # `source` is the extraction tier that produced the row. Without it two
    # tiers reporting the same measurement print as byte-identical lines and
    # neither a human reviewer nor any downstream aggregation over this CSV
    # can tell "one measurement, reported twice" from "two independent
    # measurements". Rows are NOT deduplicated here on purpose — this is the
    # long-form hand-verification export, so both tiers' rows stay visible
    # and the column is what makes them distinguishable.
    fieldnames = [
        "patent_id", "compound_id", "iupac_name", "target",
        "assay_name", "value_uM", "value_raw", "value_unit",
        "qualifier", "n_runs", "is_null", "source", "source_patent_class",
    ]
    n_rows = 0
    per_patent: dict[str, dict] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid in patent_ids:
            assay_path = extractions_root / pid / "assay_tables.json"
            if not assay_path.exists():
                logger.warning("no assay_tables for %s — skipping", pid)
                per_patent[pid] = {"n_compounds": 0, "n_rows": 0, "class": "missing_extraction"}
                continue
            assays = json.loads(assay_path.read_text())
            ex_path = extractions_root / pid / "example_index.json"
            iupacs: dict[str, str] = {}
            if ex_path.exists():
                ex = json.loads(ex_path.read_text())
                for cid, rec in (ex.items() if isinstance(ex, dict) else []):
                    iupacs[str(cid)] = (rec.get("iupac_name") or "").strip()

            n_meas = sum(len(arr) for arr in assays.values())
            patent_class = _classify_patent(len(assays), n_meas)
            per_patent[pid] = {
                "n_compounds": len(assays),
                "n_rows": 0,
                "n_measurements": n_meas,
                "class": patent_class,
            }

            for cid in sorted(assays.keys(), key=lambda s: (len(s), s)):
                iupac = iupacs.get(cid, "") or iupacs.get(cid.lower(), "")
                for a in assays[cid]:
                    value = a.get("value_numeric")
                    unit = a.get("unit") or ""
                    qualifier = a.get("qualifier") or ""
                    assay_name = a.get("assay_name") or ""
                    target = _guess_target(assay_name)
                    is_null = value is None and not qualifier
                    writer.writerow({
                        "patent_id": pid,
                        "compound_id": cid,
                        "iupac_name": iupac,
                        "target": target,
                        "assay_name": assay_name,
                        "value_uM": _to_uM(value, unit) if value is not None else "",
                        "value_raw": a.get("value_raw") or (value if value is not None else ""),
                        "value_unit": unit,
                        "qualifier": qualifier,
                        "n_runs": a.get("n_runs") if a.get("n_runs") is not None else "",
                        "is_null": is_null,
                        "source": a.get("source") or "",
                        "source_patent_class": patent_class,
                    })
                    n_rows += 1
                    per_patent[pid]["n_rows"] += 1
    return n_rows, per_patent


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--extractions-root",
        type=Path,
        default=Path("output_v2/text_extraction"),
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    n_rows, per_patent = export_csv(
        args.output, args.patent, args.extractions_root,
    )
    print(f"\nWrote {n_rows} rows to {args.output}")
    print(f"\n{'Patent':>16}  {'class':>22}  {'cpds':>6}  {'rows':>6}  {'meas':>6}")
    print("-" * 70)
    for pid, info in per_patent.items():
        print(
            f"{pid:>16}  {info['class']:>22}  "
            f"{info['n_compounds']:>6}  {info['n_rows']:>6}  "
            f"{info.get('n_measurements', 0):>6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
