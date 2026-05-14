"""Configuration for patent molecule extraction pipeline."""

import os
from pathlib import Path

# API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Load from .env if not in environment. Search both v2-package-local
# (patent_extraction_v2/.env) and the repo root (where the project's
# canonical .env actually lives — same one v1 uses). Fall through if
# absent so unit tests / CI can run without a key.
if not ANTHROPIC_API_KEY:
    _candidate_env_paths = [
        Path(__file__).parent.parent / ".env",        # patent_extraction_v2/.env
        Path(__file__).parent.parent.parent / ".env", # repo root (canonical location)
    ]
    for _env_path in _candidate_env_paths:
        if not _env_path.exists():
            continue
        for line in _env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                ANTHROPIC_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
        if ANTHROPIC_API_KEY:
            break
MODEL_OPUS = "claude-opus-4-6"
MODEL_SONNET = "claude-sonnet-4-6"
DEFAULT_MODEL = MODEL_SONNET  # Sonnet for $200 proof phase; upgrade to Opus after budget increase

# Paths
# PACKAGE_ROOT = patent_extraction_v2/  (source code lives here)
# REPO_ROOT    = the actual project root where patent PDF folders + the old
#                output/ directory live. We point at it explicitly so v2 can
#                read the same patent data without duplicating PDFs.
# OUTPUT_DIR   = isolated to output_v2/ at REPO_ROOT so old + new pipelines
#                can run side-by-side without clobbering each other's results.
PACKAGE_ROOT = Path(__file__).parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = REPO_ROOT  # Patent folders (US11312727/, etc.) sit at repo root
OUTPUT_DIR = REPO_ROOT / "output_v2"
IMAGES_DIR = OUTPUT_DIR / "images"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"
# Where the old pipeline's output lives (for side-by-side comparison + harness reads)
LEGACY_OUTPUT_DIR = REPO_ROOT / "output"

# ── Assay-extraction pipeline feature flag ───────────────────────
# When True, `routes/google_assays.extract_assays_for_patent` delegates
# to the new generic FSM + always-fire LLM pipeline at
# `core.assay_fsm.pipeline.extract_for_patent`. Default True so the
# new pipeline is the source of truth; set ASSAY_FSM=0 to fall back
# to the legacy code paths during the migration window.
ASSAY_FSM_ENABLED = os.environ.get("ASSAY_FSM", "1") == "1"

# ── HARVEST burst tier (coverage safety net) ──────────────────────
# When True, the pipeline runs the gap detector after the cheap tier
# completes; on signal trip, fires the HARVEST 5-agent burst over the
# full patent text. Default ON. Disable via `HARVEST_BURST=0` for
# regression debugging or cost-controlled re-runs.
HARVEST_BURST_ENABLED = os.environ.get("HARVEST_BURST", "1") == "1"

# Patent IDs
PATENT_IDS = [
    "US10214537",
    "US10899738",
    "US11312727",
    "US20230365584A1",
    "US20240010684A1",
    "US20240335431A1",
    "US20250163061A1",
    "US9718825",
]

DEV_PATENT = "US10214537"  # Primary development/testing patent

# Cost — $200 proof budget for 8 patents
COST_THRESHOLDS = [50, 100, 150, 200]
COST_CEILING = 200

# Opus pricing (per million tokens)
PRICING = {
    MODEL_OPUS: {"input": 15.0, "output": 75.0},
    MODEL_SONNET: {"input": 3.0, "output": 15.0},
}

# Extraction
IMAGE_DPI_DEFAULT = 300
IMAGE_DPI_HIGH = 400
IMAGE_MIN_SIZE = 50  # Minimum pixel dimension for valid structure images
IMAGE_VARIANCE_THRESHOLD = 100  # Minimum pixel variance to not be blank

# Results — canonical output location
RESULTS_DIR = OUTPUT_DIR / "results"

# Step versions — bump when you change a step's logic.
# Downstream steps auto-invalidate via the DAG in step_cache.py.
STEP_VERSIONS = {
    "detect":           "v2",  # manifest now has example_to_page + formula_pages + image_dense_pages
    "context_fold":     "v1",
    "format_route":     "v1",
    "google_patents":   "v5",  # trimmed cascade: skip OCR-only stages on clean text
    "google_tables":    "v1",
    "adaptive":         "v1",
    "markush_context":  "v1",
    "page_extraction":  "v1",
    "inline_extraction": "v1",   # NEW — Tier-2 inline SMILES/InChI from manifest
    "table_extraction": "v1",
    "iupac_to_smiles":  "v3",  # is_clean_text param controls OCR-stage skipping
    "synthesis_block":  "v1",  # NEW — LM-based synthesis block extractor (Route 1d)
    "image_pipeline":   "v3",  # NEW — substituent-table suppression
    "cross_validate":   "v1",
    "assay_extraction": "v2",   # cascade: baseline + repaired + LM fallback (guarded)
    "markush_enumeration": "v2",  # bumped: provenance + per-scaffold caching
    "markush_enumeration_scaffold": "v1",  # NEW — per-scaffold cache key
}

# Memory mode for the upstream-OCR wrappers (DECIMER-Seg, PaddleOCR text-det,
# PP-Structure cell-det). When True, each wrapper's `unload()` actually frees
# its model singletons + runs gc; subsequent calls re-load from scratch.
# When False, models stay hot in RAM after first load and `unload()` is a no-op.
#
# - Local laptop (16-32 GB): leave True. Sequential phases keep peak under
#   one model's footprint (~6 GB for PaddleOCR is the worst case).
# - Cloud / GPU box (64+ GB): set to False to maximize throughput across
#   many pages — first-load cost amortizes over the batch.
#
# Override via env var at runtime: `EXTRACTION_UNLOAD_MODELS=0` for cloud,
# `=1` for local. Default favors safety on the laptop.
UNLOAD_MODELS_AFTER_USE = (
    os.environ.get("EXTRACTION_UNLOAD_MODELS", "1") not in ("0", "false", "False", "")
)

# Per-patent LM spend cap — guards Stages 3a/3b + synthesis-block route
PER_PATENT_LM_CAP = 0.20  # USD

# Per-patent image-pipeline cost cap (Sonnet/Opus Vision + Sonnet layout calls)
# Separate from LM cap so a Tier-B patent can spend on diagrams without burning LM budget
PER_PATENT_IMAGE_CAP = 0.50  # USD

# Per-page crop cap — avoid runaway DECIMER calls on dense schemes
PER_PAGE_CROP_CAP_DEFAULT = 30

# Markush
MARKUSH_ENUMERATION_CAP = 1_000_000
MARKUSH_BRANCH_PRUNE_THRESHOLD = 10_000_000

# Drug-likeness filters
LIPINSKI_MW_MAX = 500
LIPINSKI_LOGP_MAX = 5
LIPINSKI_HBD_MAX = 5
LIPINSKI_HBA_MAX = 10
MW_DRUG_MIN = 200
MW_DRUG_MAX = 800
SA_SCORE_MAX = 6.0

# API retry
MAX_RETRIES = 3
RETRY_MIN_WAIT = 4
RETRY_MAX_WAIT = 60

# Ensure output directories exist
for d in [OUTPUT_DIR, IMAGES_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
