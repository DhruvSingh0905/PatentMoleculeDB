"""HARVEST Agent 1 — Target Extraction.

Reads a chunk of patent text, identifies every biological target
that has activity measurements in the chunk. Output drives Agent 2's
interpretation — a `Ki` column header means different things across
target classes, so Agent 2 needs the target context to canonicalize.

One LLM call per chunk. Cached by chunk fingerprint via the existing
`AdaptiveExtractionCache`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ...api_client import call_claude_text
from .prompts import AGENT1_TARGET_EXTRACTION

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """One biological target the patent measures activity against."""
    target_name: str
    organism: str = ""
    isoform: str = ""
    assay_type: str = ""        # binding / enzyme_inhibition / cellular_potency / ...
    units: str = ""
    conditions: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        return cls(
            target_name=str(d.get("target_name") or "").strip(),
            organism=str(d.get("organism") or "").strip(),
            isoform=str(d.get("isoform") or "").strip(),
            assay_type=str(d.get("assay_type") or "").strip(),
            units=str(d.get("units") or "").strip(),
            conditions=str(d.get("conditions") or "").strip(),
        )


@dataclass
class Agent1Result:
    targets: list[Target] = field(default_factory=list)
    raw_response: str = ""
    truncated: bool = False


def extract_targets(
    chunk_text: str,
    *,
    max_text_chars: int = 6000,
    max_tokens: int = 2000,
) -> Agent1Result:
    """Run Agent 1 on a single chunk. Returns targets + raw response.

    Robust to truncated JSON (re-uses the truncation-tolerant parser
    from the legacy realigner path). On any failure, returns an empty
    Agent1Result — Agent 2 can still run with no target context, just
    less effectively.
    """
    if not chunk_text:
        return Agent1Result()

    truncated_input = len(chunk_text) > max_text_chars
    chunk = chunk_text[:max_text_chars]
    prompt = AGENT1_TARGET_EXTRACTION.format(chunk_text=chunk)

    try:
        raw = call_claude_text(prompt, max_tokens=max_tokens)
    except Exception as e:
        logger.warning("agent1: LLM call failed: %r", e)
        return Agent1Result(truncated=truncated_input)

    if not raw:
        return Agent1Result(truncated=truncated_input)

    parsed = _parse_array_json(raw)
    if parsed is None:
        logger.warning("agent1: response not parseable JSON: %r", raw[:200])
        return Agent1Result(raw_response=raw, truncated=truncated_input)

    targets: list[Target] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        t = Target.from_dict(item)
        if t.target_name:
            targets.append(t)

    if targets:
        logger.info("agent1: extracted %d targets from chunk", len(targets))
    return Agent1Result(targets=targets, raw_response=raw, truncated=truncated_input)


def summarize_targets(targets: list[Target]) -> str:
    """Render a target list as compact context for Agent 2's prompt."""
    if not targets:
        return "(none found)"
    lines = []
    for t in targets:
        bits = [t.target_name]
        if t.organism and t.organism != "unspecified":
            bits.append(f"({t.organism})")
        if t.isoform:
            bits.append(f"[{t.isoform}]")
        if t.assay_type:
            bits.append(f"— {t.assay_type}")
        if t.units:
            bits.append(f"in {t.units}")
        if t.conditions:
            bits.append(f"({t.conditions})")
        lines.append(" ".join(bits))
    return "\n".join(f"  - {ln}" for ln in lines)


def _parse_array_json(raw: str) -> list | None:
    """Tolerant JSON-array parser. Strips code fences, recovers from
    leading/trailing prose."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if not m:
            return None
        try:
            out = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(out, list):
        return None
    return out
