"""Configuration for patent molecule extraction pipeline."""

import os
from pathlib import Path

# API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_OPUS = "claude-opus-4-6"
MODEL_SONNET = "claude-sonnet-4-6"
DEFAULT_MODEL = MODEL_SONNET  # Sonnet for $200 proof phase; upgrade to Opus after budget increase

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT  # Patent folders are at project root
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

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
