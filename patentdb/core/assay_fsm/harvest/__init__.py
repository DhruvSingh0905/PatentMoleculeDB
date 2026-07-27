"""HARVEST burst — multi-agent LLM safety-net extractor.

Reimplementation of HARVEST (bioRxiv 10.64898/2026.03.15.711910)
agents 1, 2, 5 as a coverage-gap-triggered burst over patent text.

Public API:
  - `harvest_burst(text, *, patent_id, cache)` — runs the full burst
  - `HarvestResult` — burst output (targets + tuples + diagnostics)
  - `Target` — Agent 1 output type
  - `ActivityTuple` — Agent 2/5 output type with provenance
"""
from __future__ import annotations

from .agent1_targets import Target, Agent1Result, extract_targets, summarize_targets
from .agent2_activities import ActivityTuple, Agent2Result, extract_activities
from .agent5_validate import Agent5Result, validate
from .orchestrator import HarvestResult, harvest_burst

__all__ = [
    "harvest_burst",
    "HarvestResult",
    "Target",
    "Agent1Result",
    "extract_targets",
    "summarize_targets",
    "ActivityTuple",
    "Agent2Result",
    "extract_activities",
    "Agent5Result",
    "validate",
]
