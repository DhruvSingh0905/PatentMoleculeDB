"""Build per-patent CSVs in Jie's exact column format — cleaned.

Improvements over previous version (per user review):
  - Drops US9718825 (markush-dominant — not a clean text-based proof point)
  - Drops US20240010684A1 (mostly empty after col_N cleanup)
  - Filters `col_N` placeholder assay-names (LLM header-recognition failures)
  - Filename now includes coverage stats: <PATENT>_<extracted>cpds_vs_BDB<n>_<pct>cov.csv
  - First row of every CSV is a comment line with the same coverage stats
    so Jie sees the coverage at a glance when she opens the file

Coverage = (number of BDB-known compounds for this patent that v2 has
ANY data for) / (BDB-known compound count). Patents not in BDB show
"BDB=N/A" and skip the coverage column.

n_runs is the number of independent experimental replicates a value
was averaged over — patent text "0.0038 (8)" → n_runs=8.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/Users/dhruvsingh/Downloads/Patent (1)")
DOWNLOADS = Path.home() / "Downloads"
EXTRACTION = REPO / "output_v2" / "text_extraction"
OUT_DIR = DOWNLOADS / "jie_format_per_patent"

# Patents to skip — non-text-based or near-empty after cleaning
SKIP_PATENTS = {
    "US9718825",          # markush-dominant; data we extract is mostly low-quality DECIMER fallout
    "US20240010684A1",    # peptide patent, ~3 real assays, sparse data after col_N cleanup
}


# ── Aggressive canonicalization ─────────────────────────────────


_UNIT_PAREN = re.compile(
    r"\s*[\(\[]\s*(?:nM|μM|uM|µM|mcM|mM|pM|M|%)\s*[\)\]]\s*",
    re.IGNORECASE,
)
_ABBREV_MAP = {
    "wildtype": "wt",
    "wild type": "wt",
    "wild-type": "wt",
    "human whole blood": "hwb",
    "homogeneous time resolved fluorescence": "htrf",
    "fluorescence resonance energy transfer": "fret",
}

# Generic suffix tokens that don't change the assay's identity.
# Stripping these lets `IC50 value` and `IC50` collapse into the same
# canonical assay. Patent-agnostic — these words mean "the measurement"
# not "a different measurement".
_GENERIC_SUFFIX_TOKENS = {
    "value", "values", "result", "results", "data", "measurement",
    "measurements", "reading", "readings",
}

# Match generic placeholder names produced when the LLM realigner
# couldn't infer a real column header from a chunk. Examples seen:
#   col_0, col_1, col_2, col_3                    (numeric col fallback)
#   assay_col_3, assay_col_4                      (alt namespace)
#   assay_1, assay_2, assay_3                     (sequential assay numbering)
#   ic50_1, ic50_2                                (assay-type + index)
# These are noise — the LLM emitted them when it saw values it
# couldn't tie to a header. Real assay names always include a target
# or cell-line token (e.g., "MOLM-13", "Menin Binding", "FRET_IC50").
_PLACEHOLDER_PATTERNS = [
    re.compile(r"^(?:assay_)?col[_\s]?\d+$", re.IGNORECASE),
    re.compile(r"^assay[_\s]?\d+$", re.IGNORECASE),
    re.compile(r"^ic[_\s]?50[_\s]?\d+$", re.IGNORECASE),
    re.compile(r"^ec[_\s]?50[_\s]?\d+$", re.IGNORECASE),
    re.compile(r"^value[_\s]?\d+$", re.IGNORECASE),
    re.compile(r"^column[_\s]?\d+$", re.IGNORECASE),
    re.compile(r"^unknown$", re.IGNORECASE),
]


def _is_placeholder_name(name: str) -> bool:
    """True if the assay_name is an LLM-fallback placeholder we should
    drop. Real assay names always carry a target/cell-line/condition
    token; placeholders are pure-numeric or generic-token fallbacks."""
    s = (name or "").strip().lower()
    if not s:
        return True
    return any(pat.match(s) for pat in _PLACEHOLDER_PATTERNS)


def _canonical_slug(name: str) -> str:
    if not name:
        return ""
    s = _UNIT_PAREN.sub(" ", name).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for full, short in _ABBREV_MAP.items():
        s = s.replace(full, short)
    # Drop generic suffix tokens so `IC50 value` ≡ `IC50`
    tokens = [t for t in s.split() if t not in _GENERIC_SUFFIX_TOKENS]
    s = "".join(tokens)
    return s


def _short_label(name: str, slug: str) -> str:
    s = _UNIT_PAREN.sub(" ", name).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "assay_" + slug[:24]
    return s[:50]


def canonicalize(measurements: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    by_slug: dict[str, list[str]] = defaultdict(list)
    for m in measurements:
        nm = m.get("assay_name") or ""
        if not nm or _is_placeholder_name(nm):
            continue
        by_slug[_canonical_slug(nm)].append(nm)

    name_map: dict[str, str] = {}
    short_labels: dict[str, str] = {}
    used_short: set[str] = set()
    for slug, names in by_slug.items():
        if not names:
            continue
        ctr = Counter(names)
        most_common = ctr.most_common()
        max_count = most_common[0][1]
        canonical = sorted(
            (n for n, c in most_common if c == max_count),
            key=lambda n: (len(n), n),
        )[0]
        for n in names:
            name_map[n] = canonical
        base_short = _short_label(canonical, slug)
        short = base_short
        suffix = 2
        while short in used_short:
            short = f"{base_short}_{suffix}"
            suffix += 1
        used_short.add(short)
        short_labels[canonical] = short
    return name_map, short_labels


# ── BDB stats per patent ────────────────────────────────────────


def _bdb_stats(patent_id: str, v2: dict) -> dict:
    """Returns {bdb_compounds, v2_covers_bdb, coverage_pct} or
    {bdb_compounds: 0, ...} if the patent isn't in BDB."""
    try:
        from patentdb.scripts.eval import fidelity_check as fc
        bdb = fc.load_bdb_compounds(patent_id)
    except Exception:
        return {"bdb_compounds": 0, "v2_covers_bdb": 0, "coverage_pct": None}
    n_bdb = sum(1 for bc in bdb.values() if bc.assays)
    if n_bdb == 0:
        return {"bdb_compounds": 0, "v2_covers_bdb": 0, "coverage_pct": None}
    v2_keys = {k.lower() for k in v2.keys()}
    n_covered = 0
    for cid, bc in bdb.items():
        if not bc.assays:
            continue
        if cid.lower() in v2_keys:
            n_covered += 1
    return {
        "bdb_compounds": n_bdb,
        "v2_covers_bdb": n_covered,
        "coverage_pct": round(100 * n_covered / n_bdb, 1),
    }


# ── Helpers ────────────────────────────────────────────────────


_KNOWN_TARGET_TOKENS = (
    "FLAP", "PI3K", "PI3Kdelta", "Menin", "MEN1", "SGK", "BRD", "LRRK", "JAK", "BTK",
    "CDC7", "MASP1", "MASP2", "EGFR", "VEGFR", "CDK", "GPR", "TRK", "BCR", "MAPK",
    "FAK", "A549", "HEK293", "MV4;11", "MV4-11", "MV-4-11", "MOLM13", "MOLM-13",
    "CD69", "LTB4", "FRET", "HTRF", "HEL", "OCI-AML3", "Colo205", "hERG",
)


def _guess_target(assay_name: str) -> str:
    al = assay_name or ""
    found = [t for t in _KNOWN_TARGET_TOKENS if t.lower() in al.lower()]
    return found[0] if found else ""


from patentdb.core.units import value_to_uM as _normalize_to_uM   # canonical


# ── Main writer ────────────────────────────────────────────────


def write_jie_format(patent_id: str, out_dir: Path) -> dict:
    src = EXTRACTION / patent_id / "assay_tables.json"
    if not src.exists():
        return {"skipped": True}
    v2 = json.loads(src.read_text())
    if not v2:
        return {"skipped": True}

    # Filter out col_N noise from each compound's measurement list
    v2_clean: dict[str, list[dict]] = {}
    n_dropped = 0
    for cid, arr in v2.items():
        kept = [m for m in arr if not _is_placeholder_name(m.get("assay_name") or "")]
        n_dropped += len(arr) - len(kept)
        if kept:
            v2_clean[cid] = kept
    if not v2_clean:
        return {"skipped": True, "n_placeholders_dropped": n_dropped}

    # IUPAC + BDB lookup
    iupac_lookup: dict[str, str] = {}
    ex_path = EXTRACTION / patent_id / "example_index.json"
    if ex_path.exists():
        try:
            ex = json.loads(ex_path.read_text())
            for cid, rec in (ex.items() if isinstance(ex, dict) else []):
                iupac_lookup[str(cid).lower()] = (rec.get("iupac_name") or "").strip()
        except Exception:
            pass
    bdb_stats = _bdb_stats(patent_id, v2_clean)

    # Canonicalize assay names (placeholders already filtered)
    all_meas = [m for arr in v2_clean.values() for m in arr]
    name_map, short_labels = canonicalize(all_meas)

    canon_counts: Counter = Counter()
    for m in all_meas:
        canon = name_map.get(m.get("assay_name") or "", "")
        if canon:
            canon_counts[canon] += 1
    canonical_assays = [c for c, _ in canon_counts.most_common()]

    # Header
    fieldnames = ["patent_id", "compound_id", "iupac_name", "target"]
    for ca in canonical_assays:
        prefix = short_labels[ca]
        fieldnames.append(f"{prefix}_uM")
        fieldnames.append(f"{prefix}_qualifier")
        fieldnames.append(f"{prefix}_n_runs")
        fieldnames.append(f"{prefix}_assay")

    # Per-compound rows
    rows = []
    for cid in sorted(v2_clean.keys(), key=lambda s: (len(s), s)):
        by_canon: dict[str, list[dict]] = defaultdict(list)
        for m in v2_clean[cid]:
            canon = name_map.get(m.get("assay_name") or "", m.get("assay_name") or "")
            by_canon[canon].append(m)
        iupac = iupac_lookup.get(cid.lower(), "")
        targets = sorted({_guess_target(c) for c in by_canon if _guess_target(c)})
        target_str = ", ".join(t for t in targets if t)

        row = {
            "patent_id": patent_id,
            "compound_id": cid,
            "iupac_name": iupac[:200],
            "target": target_str,
        }
        for ca in canonical_assays:
            prefix = short_labels[ca]
            ms = by_canon.get(ca, [])
            if not ms:
                row[f"{prefix}_uM"] = ""
                row[f"{prefix}_qualifier"] = ""
                row[f"{prefix}_n_runs"] = ""
                row[f"{prefix}_assay"] = ""
                continue
            vals_uM = []
            quals = []
            runs = []
            for m in ms:
                v = _normalize_to_uM(m.get("value_numeric"), m.get("unit"))
                vals_uM.append(f"{v}" if v is not None else "")
                quals.append(m.get("qualifier") or "")
                runs.append(str(m.get("n_runs")) if m.get("n_runs") is not None else "")
            row[f"{prefix}_uM"] = ";".join(vals_uM) if len(vals_uM) > 1 else (vals_uM[0] if vals_uM else "")
            row[f"{prefix}_qualifier"] = ";".join(quals).strip(";")
            row[f"{prefix}_n_runs"] = ";".join(runs).strip(";")
            row[f"{prefix}_assay"] = ca
        rows.append(row)

    # Filename: include extraction count + BDB coverage
    n_cpds = len(rows)
    n_meas = sum(len(v2_clean[c]) for c in v2_clean)
    if bdb_stats["bdb_compounds"]:
        fname = (
            f"{patent_id}_{n_cpds}cpds_{n_meas}meas"
            f"_vs_BDB{bdb_stats['bdb_compounds']}_"
            f"{int(round(bdb_stats['coverage_pct']))}pct.csv"
        )
    else:
        fname = f"{patent_id}_{n_cpds}cpds_{n_meas}meas_BDBnone.csv"
    out_path = out_dir / fname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Header comment row (read with `pandas.read_csv(..., comment='#')` if needed)
    header_lines = [
        f"# {patent_id} - extracted by v2 (FSM + cached LLM + HARVEST burst)",
        f"# {n_cpds} compounds, {n_meas} measurements, {len(canonical_assays)} unique assays",
    ]
    if bdb_stats["bdb_compounds"]:
        header_lines.append(
            f"# BDB has {bdb_stats['bdb_compounds']} compounds for this patent; "
            f"v2 covers {bdb_stats['v2_covers_bdb']} ({bdb_stats['coverage_pct']}%)"
        )
    else:
        header_lines.append("# Patent NOT in BDB - no external coverage check possible")
    if n_dropped:
        header_lines.append(
            f"# {n_dropped} measurements dropped (LLM-fallback `col_N` placeholders)"
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return {
        "filename": fname,
        "n_compounds": n_cpds,
        "n_measurements": n_meas,
        "n_assays": len(canonical_assays),
        "n_placeholders_dropped": n_dropped,
        **bdb_stats,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe old files
    for old in OUT_DIR.glob("*.csv"):
        old.unlink()

    print(f"Writing Jie-format CSVs to {OUT_DIR}")
    print()
    print(f"{'Patent':>16}  {'kept?':>6}  {'cpds':>5}  {'meas':>5}  "
          f"{'assays':>7}  {'col_N drop':>11}  {'BDB cpds':>9}  {'cov %':>6}")
    print("-" * 90)

    all_patents = [
        "US10214537", "US10899738", "US11312727", "US8952177",
        "US20230365584A1", "US20240010684A1", "US20240335431A1",
        "US20250163061A1", "US9718825",
    ]
    summary_rows = []
    for pid in all_patents:
        if pid in SKIP_PATENTS:
            print(f"{pid:>16}  {'SKIP':>6}  (in SKIP_PATENTS — not text-primary or near-empty)")
            continue
        stats = write_jie_format(pid, OUT_DIR)
        if stats.get("skipped"):
            print(f"{pid:>16}  {'SKIP':>6}  (no data)")
            continue
        cov = (
            f"{stats['coverage_pct']:.0f}%"
            if stats.get("coverage_pct") is not None else "N/A"
        )
        bdb = stats.get("bdb_compounds", 0) or "N/A"
        print(f"{pid:>16}  {'YES':>6}  {stats['n_compounds']:>5}  "
              f"{stats['n_measurements']:>5}  {stats['n_assays']:>7}  "
              f"{stats.get('n_placeholders_dropped', 0):>11}  "
              f"{str(bdb):>9}  {cov:>6}")
        summary_rows.append({"patent_id": pid, **stats})

    # Master summary CSV
    summary_path = OUT_DIR / "_SUMMARY.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# Per-patent extraction summary - text-primary patents only\n")
        f.write("# bdb_compounds = compounds in BindingDB for this patent\n")
        f.write("# v2_covers_bdb = how many of those v2 has data for\n")
        f.write("# coverage_pct = v2_covers_bdb / bdb_compounds * 100\n")
        f.write("# col_N drops = LLM placeholder columns we filtered out\n")
        fields = ["patent_id", "filename", "n_compounds", "n_measurements",
                  "n_assays", "n_placeholders_dropped",
                  "bdb_compounds", "v2_covers_bdb", "coverage_pct"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nMaster summary: {summary_path}")


if __name__ == "__main__":
    main()
