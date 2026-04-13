# PatentMoleculeDB — Patent Compound Extraction Pipeline

Extracts drug-like small molecules from pharmaceutical patents, converts IUPAC names to SMILES, validates against BindingDB, and outputs compounds ready for a Geometric GNN docking pipeline.

## Architecture

```
                        ┌─────────────────────────┐
                        │     Patent Input         │
                        │  (ID: e.g. US10214537)   │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │   Route 1    │  │   Route 2    │  │   Route 3    │
         │   Google     │  │  Claude OCR  │  │   DECIMER    │
         │   Patents    │  │  Pipeline    │  │  Image       │
         │   (FREE)     │  │  (Sonnet)    │  │  Pipeline    │
         │              │  │              │  │  (FREE)      │
         │ Clean USPTO  │  │ PDF→MD→LLM  │  │ Structure    │
         │ XML text     │  │ extraction   │  │ images→      │
         │ + OPSIN      │  │ + 5-stage    │  │ SMILES       │
         │              │  │ SMILES       │  │              │
         └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                │                 │                  │
                └────────────────┼──────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Merge + Deduplicate     │
                    │  (prefer longest IUPAC,  │
                    │   prefer final product)  │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Validate                │
                    │  • RDKit canonicalize    │
                    │  • InChIKey generation   │
                    │  • MW check (150-800)    │
                    │  • Lipinski + PAINS      │
                    │  • Salt stripping        │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Output                  │
                    │  • JSON per patent       │
                    │  • BindingDB benchmark   │
                    │  • Cost tracking         │
                    └─────────────────────────┘
```

### Route Details

**Route 1 — Google Patents (Primary, $0)**
- Scrapes clean text from patents.google.com (same data as USPTO XML)
- Zero OCR errors — text comes from official XML, not PDF image recognition
- Regex extraction of IUPAC names from claims lists and example sections
- OPSIN deterministic IUPAC→SMILES conversion (no LLM needed)
- Best results on "Example N:" format patents

**Route 2 — Claude OCR Pipeline (Fallback, ~$4/patent)**
- PDF→Markdown pre-extracted text processed by Claude Sonnet
- 5-stage fault-tolerant SMILES conversion:
  1. PubChem lookup (free, stereo-aware)
  2. OPSIN direct (free, deterministic)
  3. Rule-based OCR cleaning + OPSIN (fixes pyrolo→pyrrolo, etc.)
  4. Claude Sonnet name cleaning + OPSIN
  5. Claude Opus direct SMILES (last resort)
- Handles complex page layouts, truncated names, multi-page compounds

**Route 3 — DECIMER Image Pipeline (For structure-only patents, $0)**
- Layout-to-Local Hybrid: Sonnet reads table layout → crop individual structures → DECIMER converts
- Handles "Cpd.No." format patents where compounds are defined by drawings only
- DECIMER: open-source deep learning chemical image recognition
- Opus Vision fallback for DECIMER failures

### Layout-Aware Router
Automatically detects patent format and routes to optimal pipeline:
- **Text format** (5/8 patents): Route 1 + Route 2
- **Hybrid format** (3/8 patents): All three routes
- **Image format**: Route 3 primary

## Results

### Google Patents Route (All 8 Patents, $0 Cost)

| Patent | Compounds | Validated | BDB Match | Precision | Recall |
|---|---|---|---|---|---|
| US10214537 | 680 | 680 | 547/774 | 82.6% | 70.7% |
| US10899738 | 0 | 0 | 0/380 | — | — |
| US11312727 | 0 | 0 | 0/376 | — | — |
| US20230365584A1 | 8 | 8 | N/A | N/A | N/A |
| US20240010684A1 | 0 | 0 | N/A | N/A | N/A |
| US20240335431A1 | 0 | 0 | N/A | N/A | N/A |
| US20250163061A1 | 0 | 0 | N/A | N/A | N/A |
| US9718825 | 0 | 0 | 0/630 | — | — |

### OCR Pipeline (Fallback)

| Patent | Drug-like | BDB Matched | BDB Total | Recall | Cost |
|---|---|---|---|---|---|
| US10214537 | 839 | 595 | 774 | 76.9% | $4.04 |
| US10899738 | 31 | 11 | 380 | 2.9% | $5.25 |

### Combined Ensemble (Google Patents + OCR Fallback)
- US10214537: **680 compounds at $0** (Google Patents alone) vs 595 at $4 (OCR pipeline alone)
- Google Patents provides cleaner extraction with higher yield for text-format patents
- OCR pipeline catches edge cases Google Patents regex misses
- Image pipeline (DECIMER) needed for Cpd.No. format patents (US10899738, etc.)

### Accuracy Analysis (US10214537, v12 OCR pipeline)
```
Of 639 compounds where we CAN compare (same Example #):
  633 correct molecule (99.1% molecular accuracy)
  6 wrong molecule (grabbed intermediate instead of final product)

Of 215 unmatched validated compounds:
  60 stereo mismatches (right molecule, OPSIN can't encode stereo)
  ~155 genuine BDB gaps (compounds we correctly extracted, BDB doesn't have)
```

## Cost Model

| Route | Per Patent | 100K Patents |
|---|---|---|
| Google Patents (Route 1) | $0 | $0 |
| Claude Sonnet (Route 2) | ~$4 | ~$400K |
| DECIMER (Route 3) | $0 | $0 |
| Batch API (50% off) | ~$2 | ~$200K |
| Google Patents + Fallback | ~$1-2 | ~$100-200K |

## Module Structure

```
patent_extraction/
  config.py              # Central configuration, patent IDs, cost ceilings
  models.py              # Pydantic data models (Compound, PatentResult, etc.)
  pipeline.py            # Main orchestrator — runs all routes
  google_patents.py      # Route 1: Google Patents clean text extraction
  detect_structures.py   # Local page classification (no API)
  layout_router.py       # Auto-detect patent format (text/image/hybrid)
  extract_compounds.py   # Claude Sonnet page-level extraction
  parse_tables.py        # HTML table + claims section parsing
  iupac_to_smiles.py     # 5-stage IUPAC→SMILES conversion
  ocr_autocorrect.py     # Levenshtein-based OCR error correction
  image_pipeline.py      # Route 3: DECIMER + Opus Vision
  smiles_utils.py        # RDKit validation, canonicalization, drug-likeness
  cross_validate.py      # Multi-source cross-validation
  benchmark.py           # BindingDB ground truth comparison
  api_client.py          # Claude API wrapper with retry + caching
  batch_client.py        # Anthropic Batch API (50% cheaper)
  api_cache.py           # SHA256-keyed response cache
  cost_tracker.py        # Real-time cost tracking with ceiling
  progress.py            # Pipeline progress tracking
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run single patent
python3 -m patent_extraction.pipeline US10214537

# Run Google Patents test on all 8
python3 run_ensemble_test.py

# Run benchmark
python3 -m patent_extraction.benchmark
```

## Key Technical Decisions
- **OPSIN over LLMs** for IUPAC→SMILES: Deterministic, no hallucination, 92% success on clean text
- **Google Patents over OCR**: Same USPTO data, zero OCR errors, free
- **DECIMER over Claude Vision** for structure images: Free, local, purpose-built, 100% valid SMILES
- **InChIKey matching**: Full 27-char for stereo-aware, 14-char connectivity-only for stereo-agnostic
- **Salt stripping**: Dual storage (salt form + parent) for flexible matching
- **Cost ceiling**: Hard $200 budget with threshold warnings at $50/$100/$150
