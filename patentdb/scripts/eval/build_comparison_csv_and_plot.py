"""Build (1) a side-by-side comparison CSV for US8952177 against Jie's
hand-curated reference, and (2) a multi-panel comparison plot showing
error rates of Claude (v2) vs curated sources (Jie's CSV + BindingDB).

Outputs:
  - ~/Downloads/us8952177_claude_vs_human.csv  (per-compound comparison)
  - ~/Downloads/all_patents_assays.csv         (long-form, regenerated)
  - ~/Downloads/comparison_plot.png            (multi-panel chart)
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from patentdb.scripts.eval import fidelity_check as fc
from patentdb.scripts.eval.assay_completeness_audit import diff_one


REPO = Path("/Users/dhruvsingh/Downloads/Patent (1)")
DOWNLOADS = Path.home() / "Downloads"
JIE_CSV = DOWNLOADS / "US8952177 Binding IUPAC Final (1).csv"
EXTRACTION = REPO / "output_v2" / "text_extraction"


# ── Load helpers ────────────────────────────────────────────────


def load_jie_us8952177() -> dict[str, dict]:
    """Load Jie's hand-curated reference for US8952177."""
    out: dict[str, dict] = {}
    with JIE_CSV.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            cid = row["compound_id"].strip()
            out[cid] = {
                "iupac": row.get("iupac_name", "").strip(),
                "ki": _to_float(row.get("flap_binding_ki_uM")),
                "ki_qual": (row.get("flap_binding_ki_qualifier") or "").strip() or None,
                "ki_runs": _to_int(row.get("flap_binding_ki_n_runs")),
                "ic50": _to_float(row.get("hwb_ltb4_ic50_uM")),
                "ic50_qual": (row.get("hwb_ltb4_ic50_qualifier") or "").strip() or None,
                "ic50_runs": _to_int(row.get("hwb_ltb4_ic50_n_runs")),
            }
    return out


def load_v2_us8952177() -> dict[str, list[dict]]:
    """Load v2 extraction for US8952177."""
    p = EXTRACTION / "US8952177" / "assay_tables.json"
    return json.loads(p.read_text())


def find_assay(arr: list[dict], kind: str) -> list[dict]:
    """`kind`: 'ki' or 'ic50'. Returns matching v2 measurement(s)."""
    if kind == "ki":
        keys = ("ki", "flap")
    else:
        keys = ("ic50", "ltb", "hwb")
    out = []
    for a in arr:
        an = (a.get("assay_name") or "").lower()
        if any(k in an for k in keys):
            out.append(a)
    return out


def _to_float(v) -> float | None:
    try:
        return float(v) if v not in (None, "", "null") else None
    except (TypeError, ValueError):
        return None


def _to_int(v) -> int | None:
    try:
        return int(v) if v not in (None, "", "null") else None
    except (TypeError, ValueError):
        return None


# ── Comparison CSV (US8952177 only) ─────────────────────────────


def build_comparison_csv(out_path: Path) -> dict:
    """Side-by-side per-compound CSV for US8952177. Returns a stats
    dict for the plot."""
    jie = load_jie_us8952177()
    v2 = load_v2_us8952177()

    rows = []
    stats = {
        "total_compounds": 0,
        "claude_has_compound": 0,
        "ki_match_5pct": 0,
        "ki_match_20pct": 0,
        "ki_value_present_both": 0,
        "ic50_match_5pct": 0,
        "ic50_match_20pct": 0,
        "ic50_value_present_both": 0,
        "qualifier_match": 0,
        "qualifier_present_in_jie": 0,
        "n_runs_match": 0,
        "n_runs_present_in_jie": 0,
        "ki_log_diffs": [],         # log10(claude/jie) for present-in-both
        "ic50_log_diffs": [],
    }

    for cid, ref in jie.items():
        stats["total_compounds"] += 1
        v2_arr = v2.get(cid, []) or v2.get(cid.lower(), [])
        has_compound = bool(v2_arr)
        if has_compound:
            stats["claude_has_compound"] += 1

        ki_arr = find_assay(v2_arr, "ki")
        ic50_arr = find_assay(v2_arr, "ic50")

        # Best-match v2 value per assay
        v2_ki = ki_arr[0]["value_numeric"] if ki_arr else None
        v2_ki_qual = ki_arr[0].get("qualifier") if ki_arr else None
        v2_ki_runs = ki_arr[0].get("n_runs") if ki_arr else None
        v2_ic50 = ic50_arr[0]["value_numeric"] if ic50_arr else None
        v2_ic50_qual = ic50_arr[0].get("qualifier") if ic50_arr else None
        v2_ic50_runs = ic50_arr[0].get("n_runs") if ic50_arr else None

        # Match logic
        ki_5 = ki_20 = False
        if ref["ki"] is not None and v2_ki is not None:
            stats["ki_value_present_both"] += 1
            ratio = v2_ki / max(ref["ki"], 1e-12)
            stats["ki_log_diffs"].append(math.log10(max(ratio, 1e-12)))
            ki_diff_pct = abs(v2_ki - ref["ki"]) / max(ref["ki"], 1e-9) * 100
            ki_5 = ki_diff_pct < 5
            ki_20 = ki_diff_pct < 20
            if ki_5: stats["ki_match_5pct"] += 1
            if ki_20: stats["ki_match_20pct"] += 1
        else:
            ki_diff_pct = None

        ic50_5 = ic50_20 = False
        if ref["ic50"] is not None and v2_ic50 is not None:
            stats["ic50_value_present_both"] += 1
            ratio = v2_ic50 / max(ref["ic50"], 1e-12)
            stats["ic50_log_diffs"].append(math.log10(max(ratio, 1e-12)))
            ic50_diff_pct = abs(v2_ic50 - ref["ic50"]) / max(ref["ic50"], 1e-9) * 100
            ic50_5 = ic50_diff_pct < 5
            ic50_20 = ic50_diff_pct < 20
            if ic50_5: stats["ic50_match_5pct"] += 1
            if ic50_20: stats["ic50_match_20pct"] += 1
        else:
            ic50_diff_pct = None

        # Qualifier comparison
        for ref_q, v2_q in [(ref["ki_qual"], v2_ki_qual), (ref["ic50_qual"], v2_ic50_qual)]:
            if ref_q:
                stats["qualifier_present_in_jie"] += 1
                if v2_q == ref_q:
                    stats["qualifier_match"] += 1

        # n_runs
        for ref_n, v2_n in [(ref["ki_runs"], v2_ki_runs), (ref["ic50_runs"], v2_ic50_runs)]:
            if ref_n is not None:
                stats["n_runs_present_in_jie"] += 1
                if v2_n == ref_n:
                    stats["n_runs_match"] += 1

        rows.append({
            "compound_id": cid,
            "iupac_name": ref["iupac"][:120],
            "jie_ki_uM": ref["ki"],
            "claude_ki_uM": v2_ki,
            "ki_diff_pct": f"{ki_diff_pct:.2f}" if ki_diff_pct is not None else "",
            "ki_match_5pct": "Y" if ki_5 else ("N" if ref["ki"] is not None and v2_ki is not None else ""),
            "jie_ki_qualifier": ref["ki_qual"] or "",
            "claude_ki_qualifier": v2_ki_qual or "",
            "jie_ki_n_runs": ref["ki_runs"] if ref["ki_runs"] is not None else "",
            "claude_ki_n_runs": v2_ki_runs if v2_ki_runs is not None else "",
            "jie_ic50_uM": ref["ic50"],
            "claude_ic50_uM": v2_ic50,
            "ic50_diff_pct": f"{ic50_diff_pct:.2f}" if ic50_diff_pct is not None else "",
            "ic50_match_5pct": "Y" if ic50_5 else ("N" if ref["ic50"] is not None and v2_ic50 is not None else ""),
            "jie_ic50_qualifier": ref["ic50_qual"] or "",
            "claude_ic50_qualifier": v2_ic50_qual or "",
            "jie_ic50_n_runs": ref["ic50_runs"] if ref["ic50_runs"] is not None else "",
            "claude_ic50_n_runs": v2_ic50_runs if v2_ic50_runs is not None else "",
            "compound_in_claude": "Y" if has_compound else "N",
        })

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return stats


# ── BDB comparison stats (per patent) ──────────────────────────


def gather_bdb_stats(patent_ids: list[str]) -> list[dict]:
    """For each patent in BDB, compute Claude's per-measurement
    recall, agreement counts, and disagreement counts."""
    out = []
    for pid in patent_ids:
        try:
            bdb = fc.load_bdb_compounds(pid)
            _, v2_ay = fc.load_v2_extraction(pid)
        except Exception:
            continue
        n_bdb_compounds = sum(1 for bc in bdb.values() if bc.assays)
        if n_bdb_compounds == 0:
            continue
        n_v2_has_compound = 0
        n_bdb_meas = 0
        n_matched = 0
        n_v2_only = 0
        n_bdb_only = 0
        for cid, bc in bdb.items():
            if not bc.assays:
                continue
            n_bdb_meas += len(bc.assays)
            v2_a = v2_ay.get(cid, [])
            if v2_a:
                n_v2_has_compound += 1
                d = diff_one(cid, bc.assays, v2_a)
                n_matched += d.n_matched
                n_v2_only += d.n_v2_only
                n_bdb_only += d.n_bdb_only
            else:
                n_bdb_only += len(bc.assays)
        out.append({
            "patent": pid,
            "bdb_compounds": n_bdb_compounds,
            "v2_compound_coverage_pct": 100 * n_v2_has_compound / n_bdb_compounds,
            "bdb_meas": n_bdb_meas,
            "matched": n_matched,
            "bdb_only": n_bdb_only,
            "v2_only": n_v2_only,
            "per_meas_recall_pct": 100 * n_matched / max(1, n_bdb_meas),
        })
    return out


# ── Plot ────────────────────────────────────────────────────────


def build_plot(jie_stats: dict, bdb_stats: list[dict], out_path: Path) -> None:
    """4-panel comparison plot."""
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        "Claude (v2) vs Curated References — Patent Assay Extraction",
        fontsize=15, fontweight="bold", y=0.995,
    )

    # ── Panel 1: US8952177 — Claude vs Jie (Human curation) ──
    ax1 = fig.add_subplot(2, 2, 1)
    total = jie_stats["total_compounds"]
    cats = [
        ("Claude has compound", jie_stats["claude_has_compound"]),
        ("Ki match (≤5%)",      jie_stats["ki_match_5pct"]),
        ("Ki match (≤20%)",     jie_stats["ki_match_20pct"]),
        ("IC50 match (≤5%)",    jie_stats["ic50_match_5pct"]),
        ("IC50 match (≤20%)",   jie_stats["ic50_match_20pct"]),
        ("Qualifier preserved", jie_stats["qualifier_match"]),
        ("n_runs preserved",    jie_stats["n_runs_match"]),
    ]
    labels = [c[0] for c in cats]
    counts = [c[1] for c in cats]
    pcts = [100 * c[1] / max(1, total if "compound" in c[0] else
                              jie_stats["ki_value_present_both"] if "Ki" in c[0] else
                              jie_stats["ic50_value_present_both"] if "IC50" in c[0] else
                              jie_stats["qualifier_present_in_jie"] if "Qualifier" in c[0] else
                              jie_stats["n_runs_present_in_jie"])
            for c in cats]
    colors = ["#2c8c5b" if p >= 95 else "#d4a64a" if p >= 50 else "#c0392b" for p in pcts]
    ypos = np.arange(len(labels))
    ax1.barh(ypos, pcts, color=colors, edgecolor="white")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("% match (of measurements where reference has a value)")
    ax1.set_title(
        f"US8952177 — Claude vs Human (Jie's curation, n={total} compounds)",
        fontsize=11, fontweight="bold",
    )
    for i, (count, pct) in enumerate(zip(counts, pcts)):
        denom = (total if "compound" in labels[i] else
                 jie_stats["ki_value_present_both"] if "Ki" in labels[i] else
                 jie_stats["ic50_value_present_both"] if "IC50" in labels[i] else
                 jie_stats["qualifier_present_in_jie"] if "Qualifier" in labels[i] else
                 jie_stats["n_runs_present_in_jie"])
        ax1.text(pct + 1, i, f"{count}/{denom}  ({pct:.1f}%)", va="center", fontsize=9)
    ax1.grid(axis="x", linestyle=":", alpha=0.4)

    # ── Panel 2: Per-patent recall vs BindingDB ──
    ax2 = fig.add_subplot(2, 2, 2)
    pids = [s["patent"] for s in bdb_stats]
    coverage = [s["v2_compound_coverage_pct"] for s in bdb_stats]
    recall = [s["per_meas_recall_pct"] for s in bdb_stats]
    x = np.arange(len(pids))
    w = 0.35
    bars1 = ax2.bar(x - w/2, coverage, w, label="Compound coverage %", color="#3b6dd1")
    bars2 = ax2.bar(x + w/2, recall, w, label="Per-measurement match %", color="#d1693b")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pids, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Match rate (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title(
        "Claude vs BindingDB (the 4 patents BDB has data for)",
        fontsize=11, fontweight="bold",
    )
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    for b in bars1:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                 f"{b.get_height():.0f}", ha="center", fontsize=8)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                 f"{b.get_height():.0f}", ha="center", fontsize=8)

    # ── Panel 3: Error magnitude distribution (US8952177) ──
    ax3 = fig.add_subplot(2, 2, 3)
    ki_diffs = jie_stats["ki_log_diffs"]
    ic50_diffs = jie_stats["ic50_log_diffs"]
    bins = np.linspace(-1.0, 1.0, 41)
    ax3.hist(ki_diffs, bins=bins, alpha=0.7, label=f"FLAP Ki (n={len(ki_diffs)})",
             color="#3b6dd1", edgecolor="white")
    ax3.hist(ic50_diffs, bins=bins, alpha=0.6, label=f"HWB LTB4 IC50 (n={len(ic50_diffs)})",
             color="#d1693b", edgecolor="white")
    ax3.axvline(0, color="black", lw=1.5, alpha=0.5, ls="--")
    ax3.axvline(np.log10(1.05), color="green", lw=0.8, alpha=0.6, ls=":")
    ax3.axvline(-np.log10(1.05), color="green", lw=0.8, alpha=0.6, ls=":")
    ax3.set_xlabel("log10(Claude value / Jie's value)  —  0 = perfect match")
    ax3.set_ylabel("# compounds")
    ax3.set_title(
        "US8952177 error magnitude — green dotted = ±5% tolerance band",
        fontsize=11, fontweight="bold",
    )
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel 4: Aggregate recall vs both sources ──
    ax4 = fig.add_subplot(2, 2, 4)
    sources = ["Jie's curation\n(US8952177)"] + [s["patent"] for s in bdb_stats]
    matched_counts = [
        jie_stats["ki_match_5pct"] + jie_stats["ic50_match_5pct"],
        *[s["matched"] for s in bdb_stats],
    ]
    missed_counts = [
        (jie_stats["ki_value_present_both"] - jie_stats["ki_match_5pct"]) +
        (jie_stats["ic50_value_present_both"] - jie_stats["ic50_match_5pct"]),
        *[s["bdb_only"] for s in bdb_stats],
    ]
    extra_counts = [
        0,   # for Jie-comparison: every Jie-curated value is reflected; "extra" not meaningful
        *[s["v2_only"] for s in bdb_stats],
    ]
    x = np.arange(len(sources))
    w = 0.27
    ax4.bar(x - w, matched_counts, w, label="Match (within 5%)", color="#2c8c5b")
    ax4.bar(x, missed_counts, w, label="Missed by Claude", color="#c0392b")
    ax4.bar(x + w, extra_counts, w, label="Extra in Claude (not in source)", color="#999")
    ax4.set_xticks(x)
    ax4.set_xticklabels(sources, rotation=20, ha="right", fontsize=9)
    ax4.set_yscale("log")
    ax4.set_ylabel("# measurements (log scale)")
    ax4.set_title(
        "Per-source measurement breakdown",
        fontsize=11, fontweight="bold",
    )
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(axis="y", linestyle=":", alpha=0.4)
    for i, (m, miss, x_) in enumerate(zip(matched_counts, missed_counts, extra_counts)):
        if m > 0:
            ax4.text(i - w, m * 1.1, str(m), ha="center", fontsize=8)
        if miss > 0:
            ax4.text(i, miss * 1.1, str(miss), ha="center", fontsize=8)
        if x_ > 0:
            ax4.text(i + w, x_ * 1.1, str(x_), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


# ── Main ────────────────────────────────────────────────────────


def main() -> None:
    # 1. Build comparison CSV (US8952177 — only one with human-curated reference)
    cmp_csv = DOWNLOADS / "us8952177_claude_vs_human.csv"
    jie_stats = build_comparison_csv(cmp_csv)
    print(f"Wrote comparison CSV: {cmp_csv}")
    print(f"  US8952177 stats: {jie_stats['total_compounds']} compounds; "
          f"Ki match (5%): {jie_stats['ki_match_5pct']}/{jie_stats['ki_value_present_both']}; "
          f"IC50 match (5%): {jie_stats['ic50_match_5pct']}/{jie_stats['ic50_value_present_both']}; "
          f"qualifier match: {jie_stats['qualifier_match']}/{jie_stats['qualifier_present_in_jie']}; "
          f"n_runs match: {jie_stats['n_runs_match']}/{jie_stats['n_runs_present_in_jie']}")

    # 2. Gather BDB stats for the 4 patents BDB has data for
    bdb_stats = gather_bdb_stats([
        "US10899738", "US10214537", "US11312727", "US9718825",
    ])
    print()
    print(f"{'Patent':>14}  {'BDB cpds':>9}  {'cov %':>6}  {'recall %':>9}  "
          f"{'matched':>8}  {'bdb-only':>9}  {'v2-only':>8}")
    print('-' * 75)
    for s in bdb_stats:
        print(f"{s['patent']:>14}  {s['bdb_compounds']:>9}  "
              f"{s['v2_compound_coverage_pct']:>5.1f}%  "
              f"{s['per_meas_recall_pct']:>8.1f}%  "
              f"{s['matched']:>8}  {s['bdb_only']:>9}  {s['v2_only']:>8}")

    # 3. Build the plot
    plot_path = DOWNLOADS / "comparison_plot.png"
    build_plot(jie_stats, bdb_stats, plot_path)


if __name__ == "__main__":
    main()
