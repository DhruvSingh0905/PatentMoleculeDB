"""Cost tracking with threshold alerts and ceiling enforcement."""

from __future__ import annotations

import logging

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

    def compute_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Compute cost in USD for a single API call."""
        pricing = config.PRICING.get(model, config.PRICING[config.MODEL_OPUS])
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        return cost

    def record(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Record an API call and check thresholds.

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

    def summary(self) -> dict:
        """Return a summary of costs."""
        return {
            "total_cost_usd": round(self.total_cost, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
        }


# Global singleton
cost_tracker = CostTracker()
