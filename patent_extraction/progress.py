"""Per-patent progress tracking with JSON checkpointing."""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config


class ProgressTracker:
    """Tracks extraction progress for a single patent."""

    def __init__(self, patent_id: str):
        self.patent_id = patent_id
        self.data = {
            "patent_id": patent_id,
            "status": "initialized",
            "compounds_found": 0,
            "iupac_extracted": 0,
            "smiles_from_text": 0,
            "smiles_from_image": 0,
            "inline_smiles": 0,
            "tier_1": 0,
            "tier_2": 0,
            "tier_3": 0,
            "failures": 0,
            "api_calls": 0,
            "cost_usd": 0.0,
            "last_updated": "",
        }
        self._checkpoint_path = config.CHECKPOINTS_DIR / f"{patent_id}_progress.json"

        # Load existing checkpoint if available
        if self._checkpoint_path.exists():
            with open(self._checkpoint_path) as f:
                self.data.update(json.load(f))

    def update(self, **kwargs):
        """Update progress fields and save checkpoint."""
        self.data.update(kwargs)
        self.data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save()

    def increment(self, field: str, amount: int = 1):
        """Increment a numeric field."""
        self.data[field] = self.data.get(field, 0) + amount
        self.data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save()

    def _save(self):
        """Save progress to checkpoint file."""
        with open(self._checkpoint_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def summary(self) -> dict:
        return dict(self.data)
