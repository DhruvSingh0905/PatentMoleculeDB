"""Build per-patent wide-format CSVs (one row per compound, one column
per canonical assay) + canonicalize assay-name variants.

Outputs to ~/Downloads/wide_per_patent/<PATENT>.csv plus a master
~/Downloads/all_patents_wide.xlsx-equivalent (one CSV per patent).

Why wide format:
  Each patent has multiple assays per compound (US8952177: Ki+IC50;
  US10899738: Menin + MV4;11 + MOLM13; US11312727: A549+CC50+FRET).
  Long-form CSV captures everything but is hard to scan visually.
  Wide format puts each assay in its own column so a reviewer can
  read "compound 5: <every assay value>" on one row.

Canonicalization (data-driven, not patent-specific):
  Collapse trivially-different name variants into one canonical name:
    - "FLAP_Binding_wt_HTRF_Ki"  ≡  "FLAP Binding wild type HTRF Ki (μM)"
    - "PI3K_delta_IC50"  ≡  "PI3K delta IC50 value (nM)"  ≡  "PI3K delta IC50 value"
    - "A549_A(H1N1)_EC50"  ≡  "A549_A(H1N1) EC50"
  Method: lowercase + strip punctuation + strip unit annotations →
  if the simplified form matches, treat as the same assay.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/dhruvsingh/Downloads/Patent (1)")
DOWNLOADS = Path.home() / "Downloads"
EXTRACTION = REPO / "output_v2" / "text_extraction"
OUT_DIR = DOWNLOADS / "wide_per_patent"


# ── Assay-name canonicalizer ────────────────────────────────────


_UNIT_PAREN = re.compile(r"\s*[\(\[]\s*(?:nM|μM|uM|µM|mcM|mM|pM|M|%)\s*[\)\]]\s*", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _slug(name: str) -> str:
    """Reduce a name to a comparable slug (lowercase, no punctuation,
    no unit annotation, collapsed whitespace)."""
    if not name:
        return ""
    s = _UNIT_PAREN.sub("", name)
    s = _SLUG_RE.sub("", s.lower())
    return s


def canonicalize_assay_names(
    measurements: list[dict],
) -> dict[str, str]:
    """Group all assay names that share the same slug.
    Return {original_name → canonical_name}.
    Canonical = the most-frequent name in the group."""
    by_slug: dict[str, list[str]] = defaultdict(list)
    for m in measurements:
        nm = m.get("assay_name") or ""
        if not nm:
            continue
        by_slug[_slug(nm)].append(nm)

    name_map: dict[str, str] = {}
    for slug, names in by_slug.items():
        if not names:
            continue
        # Most frequent name wins; ties broken by shortest (cleanest)
        from collections import Counter
        ctr = Counter(names)
        most_common = ctr.most_common()
        max_count = most_common[0][1]
        top = sorted(
            (n for n, c in most_common if c == max_count),
            key=lambda n: (len(n), n),
        )
        canonical = top[0]
        for n in names:
            name_map[n] = canonical
    return name_map


# ── Per-patent wide-format CSV ──────────────────────────────────


# Use the canonical unit converter from core.units (was 6 reimplementations)
from patentdb.core.units import value_to_uM as _normalize_unit_to_uM   # noqa: E402


def write_wide_csv_for_patent(patent_id: str, out_path: Path) -> dict:
    """Produce a wide-format CSV for `patent_id`.
    Returns stats: {n_compounds, n_assays, n_canonicalized_pairs}."""
    src = EXTRACTION / patent_id / "assay_tables.json"
    if not src.exists():
        return {"n_compounds": 0, "n_assays": 0, "n_canonicalized_pairs": 0}
    v2 = json.loads(src.read_text())
    if not v2:
        return {"n_compounds": 0, "n_assays": 0, "n_canonicalized_pairs": 0}

    # Flatten + canonicalize
    all_meas = [m for arr in v2.values() for m in arr]
    name_map = canonicalize_assay_names(all_meas)
    n_canonicalized = sum(1 for k, v in name_map.items() if k != v)

    # Index by (canonical_assay) → set of compound dicts
    canonical_assays: list[str] = []
    seen_assays: set[str] = set()
    for m in all_meas:
        canon = name_map.get(m.get("assay_name") or "", m.get("assay_name") or "")
        if canon and canon not in seen_assays:
            canonical_assays.append(canon)
            seen_assays.add(canon)

    # Stable order: by frequency descending so the most populated
    # assay columns come first
    from collections import Counter
    canon_counts = Counter()
    for m in all_meas:
        canon = name_map.get(m.get("assay_name") or "", m.get("assay_name") or "")
        canon_counts[canon] += 1
    canonical_assays.sort(key=lambda c: (-canon_counts[c], c))

    # Build header — for each canonical assay, emit value + qualifier
    # + n_runs columns
    fieldnames = ["compound_id", "n_measurements"]
    for ca in canonical_assays:
        fieldnames.append(f"{ca}__value_uM")
        fieldnames.append(f"{ca}__qualifier")
        fieldnames.append(f"{ca}__n_runs")
        fieldnames.append(f"{ca}__raw_unit")

    # Row per compound
    rows: list[dict] = []
    for cid in sorted(v2.keys(), key=lambda s: (len(s), s)):
        row = {"compound_id": cid, "n_measurements": len(v2[cid])}
        # Group this compound's measurements by canonical assay
        by_canon: dict[str, list[dict]] = defaultdict(list)
        for m in v2[cid]:
            canon = name_map.get(m.get("assay_name") or "", m.get("assay_name") or "")
            by_canon[canon].append(m)
        for ca in canonical_assays:
            ms = by_canon.get(ca, [])
            if not ms:
                row[f"{ca}__value_uM"] = ""
                row[f"{ca}__qualifier"] = ""
                row[f"{ca}__n_runs"] = ""
                row[f"{ca}__raw_unit"] = ""
                continue
            # If multiple measurements per (compound, canonical-assay),
            # join with ';' so the reader sees all of them
            vals = []
            quals = []
            runs = []
            units = []
            for m in ms:
                v_uM = _normalize_unit_to_uM(m.get("value_numeric"), m.get("unit"))
                vals.append(f"{v_uM}" if v_uM is not None else "")
                quals.append(m.get("qualifier") or "")
                runs.append(str(m.get("n_runs")) if m.get("n_runs") is not None else "")
                units.append(m.get("unit") or "")
            row[f"{ca}__value_uM"] = ";".join(vals)
            row[f"{ca}__qualifier"] = ";".join(quals).strip(";")
            row[f"{ca}__n_runs"] = ";".join(runs).strip(";")
            row[f"{ca}__raw_unit"] = ";".join(set(u for u in units if u))
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return {
        "n_compounds": len(rows),
        "n_assays": len(canonical_assays),
        "n_canonicalized_pairs": n_canonicalized,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing wide-format CSVs to {OUT_DIR}")
    print()
    print(f"{'Patent':>16}  {'cpds':>6}  {'unique-assays':>13}  {'canonicalized':>13}")
    print("-" * 60)
    for pid in [
        "US10214537", "US10899738", "US11312727", "US8952177",
        "US20230365584A1", "US20240010684A1", "US20240335431A1",
        "US20250163061A1", "US9718825",
    ]:
        out_path = OUT_DIR / f"{pid}.csv"
        stats = write_wide_csv_for_patent(pid, out_path)
        print(f"{pid:>16}  {stats['n_compounds']:>6}  {stats['n_assays']:>13}  {stats['n_canonicalized_pairs']:>13}")


if __name__ == "__main__":
    main()
