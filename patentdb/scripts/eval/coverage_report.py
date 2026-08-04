"""The four coverage numbers, as a chart and a markdown table.

    python3 -m patentdb.scripts.eval.coverage_report

Reads `docs/reports/corpus/*.csv` (written by `corpus_export`) plus
`repair.value_check`, and writes `docs/reports/coverage.html` +
`coverage.md`.

FOUR PANELS, NOT FOUR BARS ON ONE AXIS. Each number has a different
denominator and putting them on a shared scale would invite exactly the
comparison that is wrong:

    assays       out of every patent in the corpus
    n_runs       out of all records — and reported beside the number of CELLS
                 that print a replicate count at all, which is the real
                 ceiling. Cells and records are different units (one bare
                 `n = N` column is inherited by every assay column beside it),
                 so they are deliberately not divided by one another
    BDB coverage out of the reference ligands for the patents whose structures
                 were resolved AND that BindingDB covers; the rest have no
                 InChIKeys to match and are not misses
    BDB accuracy out of the values where both sides describe the same
                 molecule and both state a number

Each panel prints its own numerator and denominator for that reason.

Self-contained SVG and inline CSS — no CDN, no JS — so the file survives
being emailed, and it renders in light or dark.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# One hue per panel, fixed order, never cycled. Steps chosen for a >=8 CVD
# separation on both surfaces and a contrast pass against the ink colour.
_HUES = ["#3b6fd4", "#2e8b73", "#b3763a", "#8558b8"]


def _corpus_rows(root):
    p = root / "docs" / "reports" / "corpus"
    need = [p / "patents.csv", p / "records.csv"]
    if not all(f.exists() for f in need):
        raise SystemExit(f"run corpus_export first — missing {[str(f) for f in need if not f.exists()]}")
    pats = list(csv.DictReader(open(p / "patents.csv")))
    return pats


def _n_runs_denominator() -> int:
    """How many CELLS in the corpus state a replicate count at all.

    `_n_stating_cells` returns (cells stating n, of which are bare `n = N`
    columns) — the second value is a subset, not a denominator, and reading it
    as one produced "418 captured of 20 stated" = 2090%.

    This is a CELL count and `n_runs` is carried per RECORD, so the two are not
    a ratio: one bare `n = N` column is inherited by every assay column beside
    it, which is why 421 records carry a replicate count while only 418 cells
    print one. It is reported as a CEILING, not as a capture rate.
    """
    from ...core import config
    from .assay_verify import _n_stating_cells

    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    stated = 0
    for f in sorted(xml_dir.glob("*.xml")):
        try:
            total, _bare = _n_stating_cells(f.read_text(errors="ignore"))
        except Exception:
            continue
        stated += total
    return stated


def collect(root) -> dict:
    pats = _corpus_rows(root)
    n_patents = len(pats)
    with_assays = sum(1 for r in pats if int(r["n_compounds"] or 0) > 0)
    records = sum(int(r["n_records"] or 0) for r in pats)
    compounds = sum(int(r["n_compounds"] or 0) for r in pats)
    n_runs_recs = sum(int(r["n_with_n_runs"] or 0) for r in pats)

    structured = [r for r in pats if r["structures"] == "yes" and int(r["bdb_ligands"] or 0)]
    bdb_lig = sum(int(r["bdb_ligands"]) for r in structured)
    bdb_match = sum(int(r["bdb_matched"]) for r in structured)

    try:
        os.environ["REPAIR"] = "0"
        from ...repair.value_check import check_corpus
        t = check_corpus()["total"]
    except Exception as e:
        logger.warning("value_check unavailable (%r)", e)
        t = {}
    agree = int(t.get("agree", 0))
    rng = int(t.get("range_contains", 0))
    wrong = (int(t.get("wrong_scale", 0)) + int(t.get("disagree", 0))
             + int(t.get("range_misses", 0)))
    comparable = agree + rng + wrong

    try:
        stated = _n_runs_denominator()
    except Exception as e:
        logger.warning("n_runs cell count unavailable (%r)", e)
        stated = 0

    return {
        "patents": n_patents, "records": records, "compounds": compounds,
        "with_assays": with_assays,
        "n_runs_cells_stating": stated, "n_runs_records": n_runs_recs,
        "bdb_matched": bdb_match, "bdb_ligands": bdb_lig,
        "bdb_patents": len(structured),
        "value_correct": agree + rng, "value_comparable": comparable,
        "value_agree": agree, "value_range": rng, "value_wrong": wrong,
        "value_refs": int(t.get("refs", 0)),
    }


def _panel(i, title, num, den, note, caveat):
    pct = (100.0 * num / den) if den else 0.0
    # Never round a near-perfect score UP to a perfect one. 16,896 of 16,899 is
    # 99.98%, and printing "100.0%" erases the three values that are wrong —
    # which are the only ones a reader of this panel needs to know about.
    txt = f"{pct:.2f}" if 99.0 < pct < 100.0 else f"{pct:.1f}"
    hue = _HUES[i % len(_HUES)]
    return f"""
  <section class="panel">
    <h2>{title}</h2>
    <div class="hero" style="color:{hue}">{txt}<span class="pc">%</span></div>
    <svg viewBox="0 0 100 6" preserveAspectRatio="none" role="img"
         aria-label="{title}: {num} of {den}">
      <rect x="0" y="0" width="100" height="6" rx="3" class="track"/>
      <rect x="0" y="0" width="{max(pct, 0.6):.2f}" height="6" rx="3" fill="{hue}"/>
    </svg>
    <p class="frac"><strong>{num:,}</strong> of <strong>{den:,}</strong> &mdash; {note}</p>
    <p class="caveat">{caveat}</p>
  </section>"""


def render(d: dict) -> str:
    panels = [
        _panel(0, "Patents yielding assay data",
               d["with_assays"], d["patents"],
               "patents in the corpus produce at least one usable measurement",
               f"{d['patents'] - d['with_assays']} produce nothing: the open cases "
               f"the repair loop exists for."),
        _panel(1, "Records carrying a replicate count",
               d["n_runs_records"], d["records"],
               "assay records that carry n_runs",
               f"The ceiling is low because patents rarely print it: only "
               f"<strong>{d['n_runs_cells_stating']:,} cells</strong> in the whole "
               f"corpus state a replicate count. This measures how often n is "
               f"REPORTED, not how well it is read — and cells and records are "
               f"different units, so the two are not a ratio."),
        _panel(2, "BindingDB structure coverage",
               d["bdb_matched"], d["bdb_ligands"],
               f"reference ligands matched, across the {d['bdb_patents']} patents "
               f"whose structures were resolved",
               f"The other {d['patents'] - d['bdb_patents']} patents either never had "
               f"the structure resolver run or are absent from BindingDB, so they have "
               f"no InChIKeys to match and are excluded rather than counted as misses."),
        _panel(3, "BindingDB value accuracy",
               d["value_correct"], d["value_comparable"],
               "values correct where both sides describe the same molecule",
               f"{d['value_agree']:,} agree within 5% and {d['value_range']:,} are "
               f"ranges containing the true value; <strong>{d['value_wrong']}</strong> "
               f"are wrong. Recall is "
               f"{100.0*d['value_comparable']/max(d['value_refs'],1):.1f}% of "
               f"{d['value_refs']:,} reference values."),
    ]
    return f"""<title>PatentMoleculeDB — coverage</title>
<style>
  :root {{ --bg:#ffffff; --ink:#14171c; --dim:#5b6472; --line:#e3e7ee; --card:#f7f9fc; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1216; --ink:#e8ecf2; --dim:#96a0b0; --line:#232a33; --card:#161b21; }}
  }}
  :root[data-theme="dark"] {{ --bg:#0f1216; --ink:#e8ecf2; --dim:#96a0b0; --line:#232a33; --card:#161b21; }}
  :root[data-theme="light"] {{ --bg:#ffffff; --ink:#14171c; --dim:#5b6472; --line:#e3e7ee; --card:#f7f9fc; }}
  body {{ background:var(--bg); color:var(--ink); margin:0; padding:2.5rem 1.25rem 4rem;
         font:16px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; margin:0 0 .35rem; letter-spacing:-.01em; }}
  .sub {{ color:var(--dim); margin:0 0 2rem; font-size:.95rem; }}
  .grid {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); }}
  .panel {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
            padding:1.15rem 1.25rem 1.25rem; }}
  .panel h2 {{ font-size:.82rem; font-weight:600; text-transform:uppercase;
               letter-spacing:.06em; color:var(--dim); margin:0 0 .5rem; }}
  .hero {{ font-size:2.9rem; font-weight:650; line-height:1; letter-spacing:-.02em; }}
  .pc {{ font-size:1.2rem; font-weight:500; margin-left:.1rem; }}
  svg {{ width:100%; height:6px; display:block; margin:.85rem 0 .7rem; }}
  .track {{ fill:var(--line); }}
  .frac {{ margin:0 0 .45rem; font-size:.9rem; }}
  .caveat {{ margin:0; font-size:.82rem; color:var(--dim); }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; margin-top:.5rem; }}
  th,td {{ text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--line); }}
  th {{ color:var(--dim); font-weight:600; }}
  td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .scroll {{ overflow-x:auto; }}
  footer {{ margin-top:2.5rem; color:var(--dim); font-size:.82rem; }}
</style>
<div class="wrap">
  <h1>PatentMoleculeDB — coverage</h1>
  <p class="sub">{d['patents']} patents &middot; {d['records']:,} assay records &middot;
     {d['compounds']:,} distinct compounds. Each panel states its own denominator:
     the four are not on a common scale and should not be compared to one another.</p>
  <div class="grid">{''.join(panels)}</div>
  <h2 style="font-size:1rem;margin:2.25rem 0 .25rem">The numbers</h2>
  <div class="scroll"><table>
    <tr><th>Metric</th><th class="n">Numerator</th><th class="n">Denominator</th><th class="n">%</th></tr>
    <tr><td>Patents yielding assay data</td><td class="n">{d['with_assays']}</td>
        <td class="n">{d['patents']}</td><td class="n">{100.0*d['with_assays']/d['patents']:.1f}</td></tr>
    <tr><td>Cells in the corpus stating a replicate count</td>
        <td class="n">{d['n_runs_cells_stating']:,}</td><td class="n">&mdash;</td>
        <td class="n">&mdash;</td></tr>
    <tr><td>n_runs as a share of all records</td><td class="n">{d['n_runs_records']:,}</td>
        <td class="n">{d['records']:,}</td>
        <td class="n">{100.0*d['n_runs_records']/max(d['records'],1):.1f}</td></tr>
    <tr><td>BindingDB structure coverage</td><td class="n">{d['bdb_matched']:,}</td>
        <td class="n">{d['bdb_ligands']:,}</td>
        <td class="n">{100.0*d['bdb_matched']/max(d['bdb_ligands'],1):.1f}</td></tr>
    <tr><td>BindingDB value accuracy</td><td class="n">{d['value_correct']:,}</td>
        <td class="n">{d['value_comparable']:,}</td>
        <td class="n">{100.0*d['value_correct']/max(d['value_comparable'],1):.2f}</td></tr>
    <tr><td>BindingDB value recall</td><td class="n">{d['value_comparable']:,}</td>
        <td class="n">{d['value_refs']:,}</td>
        <td class="n">{100.0*d['value_comparable']/max(d['value_refs'],1):.1f}</td></tr>
  </table></div>
  <footer>Value accuracy counts a record correct when it agrees within 5% of the
    reference or reports a range containing it. Structure coverage is an InChIKey
    comparison and needs the structure resolver to have run.</footer>
</div>"""


def markdown(d: dict) -> str:
    return f"""# PatentMoleculeDB — coverage

{d['patents']} patents · {d['records']:,} assay records · {d['compounds']:,} distinct compounds.

**Each row has a different denominator. They are not comparable to each other.**

| Metric | Numerator | Denominator | % |
|---|---:|---:|---:|
| Patents yielding assay data | {d['with_assays']} | {d['patents']} | {100.0*d['with_assays']/d['patents']:.1f} |
| Cells stating a replicate count | {d['n_runs_cells_stating']:,} | — | — |
| n_runs as a share of all records | {d['n_runs_records']:,} | {d['records']:,} | {100.0*d['n_runs_records']/max(d['records'],1):.1f} |
| BindingDB structure coverage | {d['bdb_matched']:,} | {d['bdb_ligands']:,} | {100.0*d['bdb_matched']/max(d['bdb_ligands'],1):.1f} |
| BindingDB value accuracy | {d['value_correct']:,} | {d['value_comparable']:,} | {100.0*d['value_correct']/max(d['value_comparable'],1):.2f} |
| BindingDB value recall | {d['value_comparable']:,} | {d['value_refs']:,} | {100.0*d['value_comparable']/max(d['value_refs'],1):.1f} |

- **{d['patents']-d['with_assays']} patents produce nothing** — the open cases the repair loop exists for.
- **n_runs is sparse because patents rarely print it**, not because we miss it: only {d['n_runs_cells_stating']:,} cells in the entire corpus state a replicate count, against {d['records']:,} records. Cells and records are different units — one bare `n = N` column is inherited by every assay column beside it — so they are deliberately not shown as a ratio.
- **Structure coverage is over {d['bdb_patents']} patents**, the ones whose structures were resolved. The other {d['patents']-d['bdb_patents']} never had the resolver run and have no InChIKeys to match — excluded, not counted as misses.
- **{d['value_wrong']} values are wrong** in the whole corpus ({d['value_agree']:,} agree within 5%, {d['value_range']:,} are ranges containing the true value).
"""


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/reports")
    a = ap.parse_args()
    from ...core import config

    d = collect(config.REPO_ROOT)
    out = config.REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.html").write_text(render(d))
    (out / "coverage.md").write_text(markdown(d))
    (out / "coverage.json").write_text(json.dumps(d, indent=1))
    print(json.dumps(d, indent=1))
    print(f"\nwrote {out/'coverage.html'}, coverage.md, coverage.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
