"""HARVEST Agent 2 — Activity Extraction with span tracking.

Reads a chunk of patent text + Agent 1's targets. Returns a list of
(compound_id, assay_name, value, unit, qualifier, n_runs) tuples,
each tagged with `source_quote` and `source_offset` so the
downstream pattern distiller can reverse-engineer regexes from the
extractions.

Critical addition vs the existing `realign_table_via_llm`: every
tuple is provenance-tracked. When the distiller sees 5 tuples that
all came from `Compound 5 had IC50 of 12 nM` style sentences, it
knows what regex to write.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ...api_client import call_claude_text
from .agent1_targets import Target, summarize_targets
from .prompts import AGENT2_ACTIVITY_EXTRACTION

logger = logging.getLogger(__name__)


@dataclass
class ActivityTuple:
    """One (compound, assay, value) measurement with provenance."""
    compound_id: str
    assay_name: str
    value: float
    unit: str = ""
    qualifier: str | None = None
    n_runs: int | None = None
    source_quote: str = ""
    source_offset: int = -1
    confidence: str = "medium"          # "high" / "medium" / "low"
    validation_reason: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ActivityTuple | None":
        cid = str(d.get("compound_id") or "").strip()
        if not cid:
            return None
        raw_value = d.get("value")
        if raw_value is None:
            return None
        try:
            v = float(raw_value)
        except (TypeError, ValueError):
            return None
        # n_runs is optional + defensive
        n_runs_raw = d.get("n_runs")
        try:
            n_runs = int(n_runs_raw) if n_runs_raw not in (None, "", "null") else None
        except (TypeError, ValueError):
            n_runs = None
        return cls(
            compound_id=cid,
            assay_name=str(d.get("assay_name") or "").strip(),
            value=v,
            unit=str(d.get("unit") or "").strip(),
            qualifier=(str(d["qualifier"]).strip() if d.get("qualifier") else None),
            n_runs=n_runs,
            source_quote=str(d.get("source_quote") or "").strip(),
            source_offset=int(d.get("source_offset") or -1),
            confidence=str(d.get("confidence") or "medium").strip(),
            validation_reason=str(d.get("validation_reason") or "").strip(),
        )

    def to_dict(self) -> dict:
        return {
            "compound_id": self.compound_id,
            "assay_name": self.assay_name,
            "value": self.value,
            "unit": self.unit,
            "qualifier": self.qualifier,
            "n_runs": self.n_runs,
            "source_quote": self.source_quote,
            "source_offset": self.source_offset,
            "confidence": self.confidence,
            "validation_reason": self.validation_reason,
        }


@dataclass
class Agent2Result:
    tuples: list[ActivityTuple] = field(default_factory=list)
    raw_response: str = ""
    n_unverified: int = 0       # tuples whose source_quote didn't match input


def extract_activities(
    chunk_text: str,
    targets: list[Target],
    *,
    max_text_chars: int = 6000,
    max_tokens: int = 8000,
) -> Agent2Result:
    """Run Agent 2 on a single chunk with Agent 1's target list as context.

    Validates every tuple's source_quote presence in the input and
    drops or downgrades non-matching ones. The LLM is asked to be
    truthful about quotes; we still verify.
    """
    if not chunk_text:
        return Agent2Result()

    chunk = chunk_text[:max_text_chars]
    prompt = AGENT2_ACTIVITY_EXTRACTION.format(
        chunk_text=chunk,
        targets_summary=summarize_targets(targets),
    )

    try:
        raw = call_claude_text(prompt, max_tokens=max_tokens)
    except Exception as e:
        logger.warning("agent2: LLM call failed: %r", e)
        return Agent2Result()

    if not raw:
        return Agent2Result()

    parsed = _parse_array_json(raw)
    if parsed is None:
        logger.warning("agent2: response not parseable JSON: %r", raw[:200])
        return Agent2Result(raw_response=raw)

    tuples: list[ActivityTuple] = []
    n_unverified = 0
    for item in parsed:
        if not isinstance(item, dict):
            continue
        t = ActivityTuple.from_dict(item)
        if t is None:
            continue
        # Verify the source_quote is genuinely in the input. If not,
        # keep the tuple but flag it for Agent 5 to downgrade.
        if t.source_quote and t.source_quote not in chunk:
            n_unverified += 1
            t.confidence = "low"
            t.validation_reason = (
                t.validation_reason
                + ("; " if t.validation_reason else "")
                + "source_quote_not_in_input"
            ).strip("; ")
        tuples.append(t)

    if tuples:
        logger.info(
            "agent2: extracted %d tuples (%d unverified) from chunk",
            len(tuples), n_unverified,
        )
    return Agent2Result(
        tuples=tuples,
        raw_response=raw,
        n_unverified=n_unverified,
    )


def _parse_array_json(raw: str) -> list | None:
    """Tolerant JSON-array parser with truncation recovery for
    arrays of objects."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return _coerce_list(json.loads(s))
    except json.JSONDecodeError:
        pass
    # Recover truncated array by closing at last complete object
    # Find first '['
    start = s.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    last_complete = -1
    for i in range(start + 1, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete = i + 1
        elif ch == "]" and depth == 0:
            try:
                return _coerce_list(json.loads(s[start:i + 1]))
            except json.JSONDecodeError:
                break
    if last_complete > 0:
        try:
            return _coerce_list(json.loads(s[start:last_complete] + "]"))
        except json.JSONDecodeError:
            return None
    return None


def _coerce_list(obj) -> list | None:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("tuples"), list):
        return obj["tuples"]
    return None
