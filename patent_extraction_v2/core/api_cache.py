"""API response cache — hash inputs, store responses, never process twice.

Uses SHA256 of (model + prompt + image_hash) as cache key.
Stores responses as JSON files in output/cache/.
Saves $100+ by avoiding re-processing same patents across iterations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

CACHE_DIR = config.OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_stats = {"hits": 0, "misses": 0}


def _hash_key(model: str, prompt: str, image_data: str | None = None) -> str:
    """Generate cache key from model + prompt + optional image data."""
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(prompt.encode())
    if image_data:
        h.update(image_data[:1000].encode())  # First 1KB of image for speed
    return h.hexdigest()[:24]


def get_cached(model: str, prompt: str, image_data: str | None = None) -> str | None:
    """Look up cached response. Returns response text or None."""
    key = _hash_key(model, prompt, image_data)
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            _stats["hits"] += 1
            return data.get("response")
        except (json.JSONDecodeError, KeyError):
            pass

    _stats["misses"] += 1
    return None


def store_cached(
    model: str, prompt: str, response: str,
    image_data: str | None = None,
    input_tokens: int = 0, output_tokens: int = 0,
):
    """Store API response in cache."""
    key = _hash_key(model, prompt, image_data)
    cache_file = CACHE_DIR / f"{key}.json"

    data = {
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    cache_file.write_text(json.dumps(data))


def cache_stats() -> dict:
    """Return cache hit/miss statistics."""
    total = _stats["hits"] + _stats["misses"]
    rate = _stats["hits"] / max(total, 1)
    cached_files = len(list(CACHE_DIR.glob("*.json")))
    return {
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "hit_rate": f"{rate:.0%}",
        "cached_responses": cached_files,
    }
