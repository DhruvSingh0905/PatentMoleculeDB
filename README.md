# PatentMoleculeDB — Patent Compound Extraction Pipeline

## Progress (Updated: April 12, 2026)

### Benchmark: US10214537 vs BindingDB Ground Truth

```
Target: 774 compounds (BindingDB has for this patent)

Recall (compounds found):
v1  ████░░░░░░░░░░░░░░░░  10.3%  (80/774)   — example pages only
v8  █████████░░░░░░░░░░░░  49.0%  (379/774)  — + compound tables
v10 █████████████░░░░░░░░  65.1%  (504/774)  — + claims parsing
v12 ███████████████░░░░░░  76.0%  (588/774)  — fixed claims name extraction
                                    ▲ current

Precision (correct molecules):
v7  ██████████░░░░░░░░░░░  53%
v8  ██████████████░░░░░░░  71%
v10 ████████████░░░░░░░░░  64%   ← dropped, OCR artifacts
v12 ██████████████░░░░░░░  74%   ← recovered after claims fix
                                    ▲ current

Target: 99% precision, 99% recall
```

### Pipeline Architecture

```
Patent PDF → Markdown (pre-extracted)
    ↓
[1] DETECT pages (local, no API)
    ↓
[2] EXTRACT compounds from 3 sources:
    • Example sections (Claude Sonnet)
    • Compound tables (regex parser)
    • Claims section (semicolon-split parser)
    ↓
[3] IUPAC → SMILES (5-stage fault-tolerant):
    Stage 1: OPSIN direct (free, deterministic) — 76%
    Stage 2: Rule-based OCR fix + OPSIN  — 91%
    Stage 2b: Vision OCR for truncated names
    Stage 3a: Sonnet cleans name + OPSIN
    Stage 3b: Opus direct SMILES (last resort)
    ↓
[4] VALIDATE: RDKit + MW check + InChIKey
    ↓
[5] BENCHMARK vs BindingDB
```

### Key Numbers

| Metric | Value |
|---|---|
| Compounds extracted | 946 |
| Validated SMILES | 842 |
| BindingDB matches | 588/774 (76% recall) |
| Budget spent | ~$30 of $200 |
| Patents processed | 1 (US10214537) |
| Tests passing | 181 |

### What Works
- OPSIN deterministic compiler (no LLM-generated SMILES)
- Rule-based OCR cleaning (pyrolo→pyrrolo, space fixes, bracket fixes)
- Multi-source extraction (examples + tables + claims)
- MW cross-validation against MS data
- Synthesis route extraction (for future retrosynthesis project)

### Detailed Accuracy Analysis (v12)
```
Of 639 compounds where we CAN compare (same Example #):
  633 correct molecule (99.1% molecular accuracy)
  6 wrong molecule (grabbed intermediate instead of final product)

Of 215 unmatched validated compounds:
  60 stereo mismatches (right molecule, OPSIN can't encode stereo)
  ~155 genuine BDB gaps (compounds we correctly extracted, BDB doesn't have)

Of 186 missing BDB compounds:
  ~93% we have the Example # but different SMILES (mostly stereo/OCR)
  ~7% we don't have at all
```

### Ceiling for Text-Only Extraction
Stereo accuracy requires structure IMAGE extraction (Vision path).
OPSIN can't assign cis/trans stereochemistry from IUPAC names.
Text-only ceiling: ~76% recall, ~99% molecular accuracy.

### Next Steps
- Fix OCR artifact patterns in claims-extracted names
- Diagnose precision errors individually
- Scale to remaining 7 patents after hitting 99% on US10214537
