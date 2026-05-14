"""Step-level caching for the extraction pipeline.

Each pipeline step has a version string (in config.STEP_VERSIONS).
A step's effective cache key = sha256(own_version + upstream_versions + input_hash).
If a cached result exists for that key, the step is skipped.

Bump a step's version in config.py when you change its logic.
Any downstream steps auto-invalidate via the DAG.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

# Step dependency DAG: {step: [upstream_steps]}
# When an upstream step's version changes, all downstream caches invalidate.
STEP_DAG: dict[str, list[str]] = {
    "detect":           [],
    "context_fold":     ["detect"],
    "format_route":     ["detect"],
    "google_patents":   [],
    "google_tables":    [],
    "adaptive":         ["google_patents", "google_tables"],
    "synthesis_block":  ["google_patents"],
    "markush_context":  ["detect", "google_patents"],
    "page_extraction":  ["detect"],
    "inline_extraction": ["detect"],
    "table_extraction": [],
    "iupac_to_smiles":  ["page_extraction", "table_extraction", "google_patents",
                         "google_tables", "markush_context"],
    "image_pipeline":   ["detect", "format_route"],
    "cross_validate":   ["iupac_to_smiles", "image_pipeline"],
    "assay_extraction": [],
    # Markush enumeration consumes validated text+image compounds, depends on
    # everything that produces canonical_smiles upstream
    "markush_enumeration": [
        "google_patents", "google_tables", "iupac_to_smiles",
        "image_pipeline", "cross_validate",
    ],
    # Per-scaffold cache — same DAG ancestry as markush_enumeration; shares
    # the version since both move together.
    "markush_enumeration_scaffold": [
        "google_patents", "google_tables", "iupac_to_smiles",
        "image_pipeline", "cross_validate",
    ],
}


def _collect_upstream_versions(step_name: str, visited: set | None = None) -> list[str]:
    """Recursively collect all upstream step versions in deterministic order."""
    if visited is None:
        visited = set()
    if step_name in visited:
        return []
    visited.add(step_name)

    versions = []
    for upstream in sorted(STEP_DAG.get(step_name, [])):
        versions.extend(_collect_upstream_versions(upstream, visited))
        versions.append(f"{upstream}={config.STEP_VERSIONS.get(upstream, 'v0')}")
    return versions


def effective_key(step_name: str, input_data: dict | list | str) -> str:
    """Compute cache key: hash(step_version_chain + input_hash)[:16].

    Args:
        step_name: Pipeline step identifier (must be in STEP_DAG).
        input_data: Step input — dict, list, or string. Gets JSON-serialized for hashing.

    Returns:
        16-char hex string.
    """
    # Build version chain: upstream versions + own version
    chain = _collect_upstream_versions(step_name)
    chain.append(f"{step_name}={config.STEP_VERSIONS.get(step_name, 'v0')}")
    version_str = "|".join(chain)

    # Hash input data
    if isinstance(input_data, str):
        input_str = input_data
    else:
        input_str = json.dumps(input_data, sort_keys=True, default=str)

    combined = f"{version_str}::{input_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _step_dir(patent_id: str) -> Path:
    """Return per-patent step cache directory, creating if needed."""
    d = config.RESULTS_DIR / patent_id / "steps"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(patent_id: str) -> Path:
    return config.RESULTS_DIR / patent_id / "manifest.json"


def load_manifest(patent_id: str) -> dict[str, str]:
    """Load {step_name: cache_key} manifest for a patent."""
    path = _manifest_path(patent_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(patent_id: str, manifest: dict[str, str]):
    path = _manifest_path(patent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def get_cached(patent_id: str, step_name: str, input_data: dict | list | str) -> dict | list | None:
    """Return cached step output if cache key matches, else None."""
    key = effective_key(step_name, input_data)
    cache_file = _step_dir(patent_id) / f"{step_name}_{key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def store(patent_id: str, step_name: str, input_data: dict | list | str, result: dict | list) -> Path:
    """Write step result to cache file and update manifest."""
    key = effective_key(step_name, input_data)
    cache_file = _step_dir(patent_id) / f"{step_name}_{key}.json"

    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Update manifest
    manifest = load_manifest(patent_id)
    manifest[step_name] = key
    _save_manifest(patent_id, manifest)

    return cache_file
