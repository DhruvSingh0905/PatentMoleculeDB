# OCSR Benchmark Results — US10899738 Structure Images

## Test: 10 cropped structure images from US10899738 (Cpd.No. format patent)

| Tool | Valid SMILES (RDKit) | BDB Match | Drug-like MW | Cost/image |
|---|---|---|---|---|
| DECIMER (open source) | 3/10 (30%) | 1/10 | 1/3 | Free |
| Claude Sonnet Vision | 6/10 (60%) | 0/10 | TBD | ~$0.03 |
| Claude Opus Vision | 8/10 (80%) | 0/10 | TBD | ~$0.15 |

## Key Findings

1. **No tool reliably produces BDB-matching SMILES from patent images**
2. **DECIMER** produces multi-fragment/truncated SMILES — looks valid but RDKit rejects most
3. **Opus Vision** has highest valid SMILES rate (80%) — best for complex drug structures
4. **Sonnet Vision** cheapest but 40% failure rate — not reliable enough as primary
5. Patent structure drawings are low quality (OCR artifacts, overlapping text) — challenging for all tools

## Recommended Approach

- Primary: **Opus Vision** for image-heavy patents (80% valid SMILES)
- Validation: Cross-check with **DECIMER** — if both agree, high confidence
- Fallback: Flag compounds where neither produces valid SMILES as "needs_manual_review"
- Cost: ~$0.15/compound for Opus, negligible for DECIMER

## Remaining Gap

Even with image extraction, BDB matching is low (0-1/10 exact matches).
The images produce structurally valid but slightly different molecules —
likely from OCR rendering artifacts or OCSR model interpretation differences.
MW cross-validation against MS data from the patent table could help filter.
