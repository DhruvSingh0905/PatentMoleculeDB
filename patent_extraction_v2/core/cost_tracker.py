"""Cost tracking with threshold alerts and ceiling enforcement."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

from . import config

logger = logging.getLogger(__name__)


class CostCeilingExceeded(Exception):
    """Raised when cumulative API cost exceeds the hard ceiling."""
    pass


class CostTracker:
    """Tracks cumulative API costs and alerts at thresholds."""

    def __init__(self):
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0
        self._alerted: set[int] = set()
        # Per-patent LM spend tracker (for $0.20/patent cap)
        self.per_patent: dict[str, float] = {}
        # Per-patent image-pipeline spend (Vision layout calls + Vision SMILES fallbacks)
        # Separate from per_patent so image work doesn't trip the LM cap and vice versa
        self.per_patent_image: dict[str, float] = {}
        # Per-patent per-route LM spend, populated only inside `attribute(route)`
        # context managers. This is the "isolated" cost source the audit
        # uses for Markush ROI — diffing total_cost across Step 2.7 was
        # unreliable because async upstream calls could land in the diff.
        self.per_patent_route: dict[str, dict[str, float]] = {}
        # Thread-local stack of active route names. Each thread has its own
        # stack so concurrent route calls don't poison each other.
        self._attrib_local = threading.local()

    def compute_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Compute cost in USD for a single API call."""
        pricing = config.PRICING.get(model, config.PRICING[config.MODEL_OPUS])
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        return cost

    def record(self, input_tokens: int, output_tokens: int, model: str,
               patent_id: str = "") -> float:
        """Record an API call and check thresholds.

        Args:
            input_tokens, output_tokens, model: API call details.
            patent_id: If provided, cost is also tracked per-patent for the
                       PER_PATENT_LM_CAP guard.

        Returns:
            Cost of this call in USD.

        Raises:
            CostCeilingExceeded: If cumulative cost exceeds the hard ceiling.
        """
        cost = self.compute_cost(input_tokens, output_tokens, model)
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1
        if patent_id:
            self.per_patent[patent_id] = self.per_patent.get(patent_id, 0.0) + cost
            # Attribute to the innermost active route (if any). This makes
            # Markush ROI cost robust to async upstream calls that complete
            # mid-Markush — only calls actually inside the `with attribute(...)`
            # block accumulate to that route's bucket.
            stack = getattr(self._attrib_local, "stack", None)
            if stack:
                route = stack[-1]
                bucket = self.per_patent_route.setdefault(patent_id, {})
                bucket[route] = bucket.get(route, 0.0) + cost

        # Check thresholds
        for threshold in config.COST_THRESHOLDS:
            if self.total_cost >= threshold and threshold not in self._alerted:
                self._alerted.add(threshold)
                logger.warning(
                    f"COST ALERT: ${self.total_cost:.2f} spent "
                    f"(threshold ${threshold} reached). "
                    f"Calls: {self.call_count}"
                )

        # Hard ceiling
        if self.total_cost >= config.COST_CEILING:
            raise CostCeilingExceeded(
                f"Hard ceiling ${config.COST_CEILING} reached. "
                f"Total spent: ${self.total_cost:.2f} over {self.call_count} calls."
            )

        return cost

    def patent_lm_exceeded(self, patent_id: str) -> bool:
        """True if patent's LM spend has hit PER_PATENT_LM_CAP."""
        return self.per_patent.get(patent_id, 0.0) >= config.PER_PATENT_LM_CAP

    def reset_patent(self, patent_id: str):
        """Reset per-patent counters (called at start of each patent run)."""
        self.per_patent[patent_id] = 0.0
        self.per_patent_image[patent_id] = 0.0
        self.per_patent_route[patent_id] = {}

    def patent_spend(self, patent_id: str) -> float:
        """Return current LM spend for a patent."""
        return self.per_patent.get(patent_id, 0.0)

    # ── Per-route attribution ────────────────────────────────────
    @contextmanager
    def attribute(self, route: str):
        """Attribute all `record()` costs inside the block to `route`.

        Use as:
            with cost_tracker.attribute("markush_enumeration"):
                run_markush_enumeration(...)

        Costs from API calls actually issued inside the block are added
        to `per_patent_route[patent_id][route]`. Async upstream calls
        that complete during the block are NOT attributed to this route
        (they were issued under a different attribution scope or none
        at all). This is the isolation property that `patent_spend()`
        diffs lack.

        Nesting is supported (innermost wins); the stack is per-thread
        so parallel route calls don't poison each other.
        """
        stack = getattr(self._attrib_local, "stack", None)
        if stack is None:
            stack = []
            self._attrib_local.stack = stack
        stack.append(route)
        try:
            yield
        finally:
            stack.pop()

    def route_spend(self, patent_id: str, route: str) -> float:
        """Return cost attributed to a specific route for a patent."""
        return self.per_patent_route.get(patent_id, {}).get(route, 0.0)

    def all_route_spend(self, patent_id: str) -> dict[str, float]:
        """Return per-route cost map for a patent."""
        return dict(self.per_patent_route.get(patent_id, {}))

    # ── Image-pipeline budget (separate counter) ──────────────────
    def record_image(self, cost: float, patent_id: str = ""):
        """Record image-pipeline spend (Vision layout / Vision fallback).

        Image cost is ALSO tracked under total_cost via the underlying
        record() call in api_client; this method is only for the per-patent
        image-only counter that gates the image cascade.
        """
        if patent_id:
            self.per_patent_image[patent_id] = self.per_patent_image.get(patent_id, 0.0) + cost

    def patent_image_exceeded(self, patent_id: str) -> bool:
        """True if patent's image-pipeline spend has hit PER_PATENT_IMAGE_CAP."""
        return self.per_patent_image.get(patent_id, 0.0) >= config.PER_PATENT_IMAGE_CAP

    def patent_image_spend(self, patent_id: str) -> float:
        """Return current image-pipeline spend for a patent."""
        return self.per_patent_image.get(patent_id, 0.0)

    def summary(self) -> dict:
        """Return a summary of costs."""
        return {
            "total_cost_usd": round(self.total_cost, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "per_patent": {k: round(v, 4) for k, v in self.per_patent.items()},
            "per_patent_image": {k: round(v, 4) for k, v in self.per_patent_image.items()},
            "per_patent_route": {
                pid: {r: round(c, 4) for r, c in routes.items()}
                for pid, routes in self.per_patent_route.items()
            },
        }


# Global singleton
cost_tracker = CostTracker()
