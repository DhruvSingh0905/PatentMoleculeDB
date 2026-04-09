"""Tests for cost tracker."""

import pytest
from patent_extraction.cost_tracker import CostTracker, CostCeilingExceeded
from patent_extraction import config


class TestCostTracker:
    def test_compute_cost_opus(self):
        tracker = CostTracker()
        # 1000 input tokens + 100 output tokens at Opus pricing
        cost = tracker.compute_cost(1000, 100, config.MODEL_OPUS)
        expected = 1000 * 15.0 / 1_000_000 + 100 * 75.0 / 1_000_000
        assert abs(cost - expected) < 1e-6

    def test_compute_cost_sonnet(self):
        tracker = CostTracker()
        cost = tracker.compute_cost(1000, 100, config.MODEL_SONNET)
        expected = 1000 * 3.0 / 1_000_000 + 100 * 15.0 / 1_000_000
        assert abs(cost - expected) < 1e-6

    def test_record_accumulates(self):
        tracker = CostTracker()
        tracker.record(1000, 100, config.MODEL_OPUS)
        tracker.record(1000, 100, config.MODEL_OPUS)
        assert tracker.call_count == 2
        assert tracker.total_input_tokens == 2000
        assert tracker.total_output_tokens == 200
        assert tracker.total_cost > 0

    def test_ceiling_exceeded(self):
        tracker = CostTracker()
        # Simulate hitting the ceiling
        # At Opus: $15/M input + $75/M output
        # To hit $5000: need about 66M output tokens
        # Let's mock it by recording large amounts
        with pytest.raises(CostCeilingExceeded):
            for _ in range(100):
                tracker.record(1_000_000, 1_000_000, config.MODEL_OPUS)

    def test_summary(self):
        tracker = CostTracker()
        tracker.record(500, 50, config.MODEL_OPUS)
        s = tracker.summary()
        assert "total_cost_usd" in s
        assert "call_count" in s
        assert s["call_count"] == 1

    def test_threshold_alerts(self, caplog):
        """Thresholds should trigger warnings."""
        import logging
        tracker = CostTracker()
        # Record enough to pass $1000 threshold
        # At Opus: need ~13.3M output tokens for $1000
        # Use big output to trigger threshold
        with caplog.at_level(logging.WARNING):
            try:
                tracker.record(0, 14_000_000, config.MODEL_OPUS)
            except CostCeilingExceeded:
                pass
        assert any("COST ALERT" in r.message for r in caplog.records)
