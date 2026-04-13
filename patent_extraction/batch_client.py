"""Anthropic Message Batches API client — 50% cheaper than real-time.

Uses client.messages.batches.create() to submit up to 10K requests.
Processed within 24 hours (usually much faster). Half the cost.

Usage:
    batch = BatchClient()
    batch.add("request_1", "claude-sonnet-4-6", "Convert this IUPAC...")
    batch.add("request_2", "claude-sonnet-4-6", "Convert this IUPAC...")
    batch_id = batch.submit()
    results = batch.wait_and_collect(batch_id)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anthropic

from . import config
from .api_cache import get_cached, store_cached

logger = logging.getLogger(__name__)


class BatchClient:
    """Manages batch API requests for bulk SMILES conversion."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.requests: list[dict] = []
        self._cached_results: dict[str, str] = {}

    def add(
        self,
        custom_id: str,
        model: str,
        prompt: str,
        max_tokens: int = 200,
    ) -> bool:
        """Add a request to the batch. Returns True if added, False if cached.

        Checks cache first — if we've already processed this exact prompt,
        stores the cached result and skips the API call.
        """
        cached = get_cached(model, prompt)
        if cached is not None:
            self._cached_results[custom_id] = cached
            return False  # Not added to batch (already cached)

        self.requests.append({
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        })
        return True

    def submit(self) -> str | None:
        """Submit the batch. Returns batch_id or None if all cached."""
        if not self.requests:
            logger.info(f"Batch: all {len(self._cached_results)} requests cached, no API call needed")
            return None

        logger.info(f"Batch: submitting {len(self.requests)} requests ({len(self._cached_results)} cached)")

        try:
            batch = self.client.messages.batches.create(requests=self.requests)
            logger.info(f"Batch submitted: {batch.id}")
            return batch.id
        except Exception as e:
            logger.error(f"Batch submission failed: {e}")
            return None

    def wait_and_collect(self, batch_id: str, poll_interval: int = 30) -> dict[str, str]:
        """Wait for batch completion and return results.

        Args:
            batch_id: The batch ID from submit().
            poll_interval: Seconds between status checks.

        Returns:
            Dict of custom_id → response text. Includes cached results.
        """
        if batch_id is None:
            return dict(self._cached_results)

        # Poll for completion
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)

            if batch.processing_status == "ended":
                break

            logger.info(
                f"Batch {batch_id}: {batch.processing_status} "
                f"({batch.request_counts.succeeded} done, "
                f"{batch.request_counts.processing} processing)"
            )
            time.sleep(poll_interval)

        # Collect results
        results = dict(self._cached_results)

        for result in self.client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                text = result.result.message.content[0].text.strip()
                results[custom_id] = text

                # Cache the result for future runs
                # Find the original request to get the prompt
                for req in self.requests:
                    if req["custom_id"] == custom_id:
                        prompt = req["params"]["messages"][0]["content"]
                        model = req["params"]["model"]
                        store_cached(model, prompt, text)
                        break
            else:
                logger.warning(f"Batch request {custom_id} failed: {result.result.type}")

        logger.info(
            f"Batch complete: {len(results)} results "
            f"({len(self._cached_results)} cached, {len(results) - len(self._cached_results)} new)"
        )
        return results

    @property
    def pending_count(self) -> int:
        return len(self.requests)

    @property
    def cached_count(self) -> int:
        return len(self._cached_results)
