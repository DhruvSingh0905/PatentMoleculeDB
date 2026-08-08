"""Confidence-gated resolution memory for the reconciler (BLUEPRINT L5).

A human-readable, reviewable store of the reconciler's LLM-resolved decisions,
keyed by an EXACT context hash (patent + assay-kind + both sources' candidate
lists). Reuse is exact-match → deterministic and impossible to cross-contaminate
across patents. High-confidence resolutions auto-activate (reused on every future
run, no LLM); low-confidence stay `pending` for human review. A human can edit any
entry; `human` status overrides and is never re-asked. The LLM-call count trends to
zero as the corpus stabilises.
"""
from __future__ import annotations

import hashlib
import json

from .. import config

_PATH = config.OUTPUT_DIR / "text_extraction" / "_cache" / "resolution_memory.json"


def context_key(patent_id: str, kind: str, a_pairs: dict, b_pairs: dict) -> str:
    """Stable exact key for a (patent, assay-kind, A-candidates, B-candidates)."""
    blob = json.dumps(
        [patent_id, kind, sorted(a_pairs.items()), sorted(b_pairs.items())],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


class ResolutionMemory:
    def __init__(self) -> None:
        try:
            self.d: dict = json.loads(_PATH.read_text())
        except (OSError, ValueError):
            self.d = {}

    def get(self, key: str) -> dict | None:
        """Return the resolution only if active (auto-trusted high-conf or human)."""
        e = self.d.get(key)
        return e if e and e.get("status") in ("active", "human") else None

    def put(self, key: str, *, resolved: dict, source_used: str, confidence: str,
            reason: str, meta: dict) -> None:
        if self.d.get(key, {}).get("status") == "human":
            return                                   # never override a human decision
        self.d[key] = {
            "resolved": resolved,
            "source_used": source_used,
            "confidence": confidence,
            # confidence-gated: high → reused forever; low → review queue
            "status": "active" if confidence == "high" else "pending",
            "reason": reason,
            **meta,
        }

    def save(self) -> None:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(self.d, indent=1, sort_keys=True))

    def pending(self) -> list[tuple[str, dict]]:
        return [(k, v) for k, v in self.d.items() if v.get("status") == "pending"]
