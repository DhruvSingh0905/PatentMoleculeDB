# Phase 1 Walkthrough — OCR-tool evaluation (FINAL)

## What was done

1. Hand-curated **7 ground-truth assay tables** across the 3 image-based patents (62 expected `(compound_id, assay_name, value, unit, qualifier)` tuples)
2. Built `scripts/eval/assay_table_eval.py` — tool-agnostic harness with a deliberately minimal parser
3. Scored **current PaddleX** (baseline)
4. Installed **MinerU v3.1.0** in isolated venv (`venv_mineru/`), ran on all 8 ground-truth pages
5. Scored **MinerU pipeline backend** (CPU-friendly; ~30s/page)
6. Tried VLM backend; on CPU-only Mac it was downloading multi-GB models indefinitely — killed. VLM backend deferred until we have GPU access.

## Files added this phase

```
patent_extraction_v2/scripts/eval/
├── assay_table_eval.py                  ~300 LOC — head-to-head scoring harness
└── ground_truth_assay_tables.json       7 tables, 62 expected tuples
mineru_test_pdfs/                        Single-page PDFs for the 8 ground-truth pages
mineru_output/                           MinerU's raw output (markdown + crops + JSON metadata)
{patent_id}/all_pages_mineru/            MinerU output in eval-harness path
venv_mineru/                             Isolated venv for MinerU (~3 GB)
```

## Final scores (after parser improvements: title-row promotion + LaTeX stripping)

```
=== current PaddleX ===
table                       expected  extr  exact  val   cpd   notes
US10899738/p181#0              7       71  100%  100%  100%   ← clean single-col
US10899738/p180#0              7      153  100%  100%  100%   ← two-up
US10899738/p179#0             12       73    0%   58%   58%   ← two-up w/ noise
US9718825/p105#0              12      237    0%   50%   50%   ← title row
US9718825/p108#0               6       75    0%  100%  100%
US11312727/p220#0             15        0    0%    0%    0%   ← bleed (canonical)
US11312727/p235#0              3        0    0%    0%    0%   ← bleed
TOTAL                         62      609   23%   53%   53%

=== MinerU (pipeline backend) ===
table                       expected  extr  exact  val   cpd   notes
US10899738/p181#0              7       61    0%   57%   57%   ← LaTeX header munged
US10899738/p180#0              7       65    0%   43%   43%
US10899738/p179#0             12       63    0%   33%   33%
US9718825/p105#0              12       67    0%   50%   50%   ← title row handled cleanly
US9718825/p108#0               6       63    0%   33%   50%
US11312727/p220#0             15        0    0%    0%    0%   ← compound IDs lost (became <img>)
US11312727/p235#0              3        0    0%    0%    0%   ← same
TOTAL                         62      319    0%   31%   32%
```

## Conclusions

### MinerU is structurally cleaner than PaddleX in 4 important ways:
1. **Bleed-through is gone**: structure-image atom labels (OH, O, R) are NOT split into separate `<tr>` rows. Instead they're saved as `<img>` references. This eliminates the entire row-coalescer compensation class.
2. **Multi-page tables are merged**: PaddleX's page 181 gave us 7 compounds; MinerU's gave us 75+ (continuation tables get joined automatically).
3. **Side-by-side layouts are split into separate logical tables**: US9718825's two-up Example layout becomes two clean single-column tables in MinerU output, no need for a side-by-side splitter.
4. **Title rows are properly marked with `colspan` and easy to skip** — the eval harness's title-row-promotion logic worked on all MinerU outputs.

### MinerU's failure modes are different from PaddleX's:
1. **Compound IDs inside structure-image cells are LOST as `<img>` tags** (US11312727 bleed pages). MinerU saved 2 structure crops on page 220, but the bleed-pattern table has 5 compounds — so 3 compound IDs were neither imaged NOR transcribed. **Critical regression for image-heavy patents.**
2. **OCR errors in headers**: `IC50` becomes `IC5o` (digit 0 → letter o), subscripts wrapped in LaTeX `$\mathrm{IC}_{50}$`. Both can be handled by parser preprocessing.
3. **Adjacent values sometimes merged into one cell** (e.g., `0.045 0.022` for compound 356). Requires post-processing to detect + split.

### Honest verdict — neither tool is a clean winner:

| Property | Current PaddleX | MinerU pipeline |
|---|---|---|
| Clean tables (PaddleX format works) | ✓ 100% | partial (LaTeX in headers) |
| Two-up layouts | partial w/ splitter | ✓ split into separate tables |
| Title rows | needs promotion | ✓ properly marked |
| Bleed-through | needs coalescer | ✓ no bleed (BUT compound IDs lost as images) |
| Compound IDs in image cells | preserved as text | ✗ replaced with `<img>` tags |
| Multi-page table merging | no | ✓ joined |

**For US11312727's bleed tables (the WORST case in our set), neither tool gives the minimal parser any traction.** PaddleX produces stranded values across 14 fake rows; MinerU produces a clean table but with `<img>` instead of compound IDs.

### What this means for the rebuild

We have 3 realistic options:

**Option 1: Stick with PaddleX + port compensation strategically**
- Keep using existing PaddleX `.md` files
- Port these specific compensation pieces from old `extract_assays.py`:
  - Title-row promotion (~30 LOC)
  - Side-by-side splitter (~50 LOC)
  - Atom-label row coalescer (~50 LOC)
  - Mojibake repair (~20 LOC)
- ~150 LOC of focused compensation, with clear docstrings explaining "this exists because PaddleX OCR has X bug"
- **Pros**: known working, no new install, fastest path to v2
- **Cons**: still ~150 LOC of compensation; doesn't match the "root cause first" principle

**Option 2: Use MinerU + write a 2-pass extractor that reads compound IDs from the saved `<img>` crops**
- Use MinerU output (cleaner table structure)
- For rows where `<td>` content is just `<img src="...">`, OCR the crop (DECIMER first, then small Sonnet vision call as fallback) to extract the compound ID below the structure
- ~80 LOC of new code, ~$0.001/structure crop in Sonnet costs
- **Pros**: correctly handles the bleed case (which is 51 pages on US11312727); structurally clean output
- **Cons**: introduces an LM call per structure image (cap-gated), more moving parts

**Option 3: Try MinerU VLM backend on a GPU machine**
- VLM backend (`hybrid-auto-engine`) is what MinerU advertises for "image inside cells"
- CPU on this Mac: indefinite (GBs of model weights, slow inference)
- On a GPU: ~10-30s per page; could process all 8 patents in ~30 min
- **Pros**: potentially solves everything at the source
- **Cons**: requires GPU; if it doesn't help, we wasted setup time

## Decision needed from you

1. **Pick an option** (1, 2, or 3)
2. If **Option 3**: do we have GPU access? (Linux box, cloud instance, etc.)
3. **Add more ground-truth tables before phase 2?** Currently 7 tables / 62 tuples. We could expand to 15-20 to be more robust, but it's incremental — could also do it after we pick an OCR direction.

**My recommendation**: **Option 2** (MinerU + image-cell post-processing). Reasoning:
- MinerU's structural cleanups (bleed gone, multi-page merging, title-rows marked) eliminate ~120 LOC of the OLD pipeline's compensation
- The remaining problem (compound IDs in `<img>` cells) is a LOCALIZED issue with a clean fix: when we hit an `<img>`-only cell, run DECIMER on the saved crop. DECIMER is local + free; Sonnet vision is fallback for compounds DECIMER fails on (~$0.001/cell).
- The Markush enumeration code we salvaged ALREADY uses DECIMER cascade for chemistry images — this just extends it to "extract compound ID text from above/below the structure"
- Total new code: ~80 LOC, 0 cents per page average (DECIMER local), audit-monitored

If you prefer minimum-risk, **Option 1** (PaddleX + targeted port) is also reasonable — 150 LOC of well-documented compensation isn't terrible, and we know it works.

## Next phase preview

If **Option 1**: Phase 2 builds Detection + Classification on existing PaddleX output; Phase 3 ports the title-row/splitter/coalescer/mojibake repairs with strict docstrings.

If **Option 2**: Re-OCR all 8 patents with MinerU pipeline backend (~30 min wall-clock total), then Phase 2 builds Detection + Classification on MinerU output; Phase 3 builds the compound-ID-from-crop extraction.

If **Option 3**: First need GPU access; deferred until that's available.
