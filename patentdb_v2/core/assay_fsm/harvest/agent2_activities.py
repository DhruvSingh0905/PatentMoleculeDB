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
from .prompts import AGENT2_ACTIVITY_EXTRACTION, AGENT2B_ROW_PATTERN

logger = logging.getLogger(__name__)


# "Looks like a table row": compound-id-shape token (letters? + optional
# dash + 1-4 digits + optional letter suffix), followed by whitespace,
# followed by either a qualified number OR a letter-grade A-E. Used as
# a structural signal to decide whether to fire Agent 2b — if a chunk
# has many of these rows, an LLM regex pass is justified even when
# Agent 2's per-tuple extraction returned nothing.
_TABLE_ROW_SIGNAL_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Z]{0,3}-?\d{1,4}[A-Za-z]{0,3}"
    r"\s+"
    r"(?:[<>~≤≥≈]?\s*\d+(?:\.\d+)?|[A-E]|\+{1,5}|\*{1,5})"
    r"(?=\s|$|\n)",
    re.MULTILINE,
)


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
class RowPattern:
    """Structural row-regex the LLM detected — used to deterministically
    extract the rest of the same table from the full patent text
    without paying for more LLM calls. The regex must be a Python re
    pattern with named groups `cid`, `value0`, `value1`, … in order
    matching `column_assays`.
    """
    regex: str
    column_assays: list[str] = field(default_factory=list)
    example_match: str = ""
    header_text: str = ""        # table header verbatim — the firing anchor

    @classmethod
    def from_dict(cls, d: dict) -> "RowPattern | None":
        rx = (d.get("regex") or "").strip()
        if not rx:
            return None
        cols = d.get("column_assays") or []
        if not isinstance(cols, list):
            cols = []
        return cls(
            regex=rx,
            column_assays=[str(c) for c in cols],
            example_match=str(d.get("example_match") or "").strip(),
            header_text=str(d.get("header_text") or "").strip(),
        )


@dataclass
class Agent2Result:
    tuples: list[ActivityTuple] = field(default_factory=list)
    raw_response: str = ""
    n_unverified: int = 0       # tuples whose source_quote didn't match input
    row_patterns: list[RowPattern] = field(default_factory=list)


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
        if t.source_quote and t.source_quote not in chunk:
            n_unverified += 1
            t.confidence = "low"
            t.validation_reason = (
                t.validation_reason
                + ("; " if t.validation_reason else "")
                + "source_quote_not_in_input"
            ).strip("; ")
        tuples.append(t)

    # ─ Second LLM pass: row-pattern extraction ─
    # Fire on either signal:
    #   (a) Agent 2 returned tuples — we have ground-truth examples to
    #       help the regex generalize.
    #   (b) Agent 2 returned NOTHING but the chunk has strong table-row
    #       signal (≥10 cid-shape rows in the text). Agent 2 may have
    #       failed because the table uses unfamiliar layouts (letter
    #       grades, missing units, idiosyncratic cid formats) — exactly
    #       the case where we most need Agent 2b to discover a regex.
    # Gating Agent 2b on Agent 2's success blocks recovery on patents
    # whose tables Agent 2 doesn't immediately understand. The pattern
    # feedback loop is the cheaper, more transferable extraction; it
    # should not be downstream of the harder, per-row task.
    patterns: list[RowPattern] = []
    n_row_signal = len(_TABLE_ROW_SIGNAL_RE.findall(chunk))
    if tuples or n_row_signal >= 10:
        patterns = _extract_row_patterns(
            chunk, tuples, max_tokens=2000,
            row_signal_count=n_row_signal,
        )

    if tuples or patterns:
        logger.info(
            "agent2: extracted %d tuples (%d unverified), %d row_patterns from chunk",
            len(tuples), n_unverified, len(patterns),
        )
    return Agent2Result(
        tuples=tuples,
        raw_response=raw,
        n_unverified=n_unverified,
        row_patterns=patterns,
    )


def _extract_row_patterns(
    chunk: str,
    tuples: list[ActivityTuple],
    *,
    max_tokens: int = 2000,
    row_signal_count: int = 0,
) -> list[RowPattern]:
    """Run AGENT2B prompt for row-pattern extraction. Returns at most a
    handful of patterns (one per distinct table layout in the chunk).
    Each pattern is validated: regex must compile + example_match must
    match the regex. Bad patterns are dropped silently.

    `row_signal_count` is the FSM's heuristic count of cid-shape rows in
    the chunk (computed by the caller). Used in the prompt as a hint:
    "expect at least this many rows to match your regex". When `tuples`
    is empty (Agent 2 failed) but `row_signal_count` is high, this is
    the signal-rich case where we most need a regex.
    """
    # Build the tuples_summary (first 5 tuples) for the prompt
    summary_lines = []
    for t in tuples[:5]:
        v = f"{t.value}"
        if t.qualifier:
            v = f"{t.qualifier}{v}"
        summary_lines.append(
            f"  - {t.compound_id} | {t.assay_name} | {v} {t.unit}".rstrip()
        )
    if summary_lines:
        tuples_summary = "\n".join(summary_lines)
    elif row_signal_count:
        tuples_summary = (
            f"  (Agent 2 found no tuples here, but the chunk has ~{row_signal_count} "
            f"rows that look like `compound-id + value`. Look for those rows.)"
        )
    else:
        tuples_summary = "  (none)"

    prompt = AGENT2B_ROW_PATTERN.format(
        chunk_text=chunk[:6000],
        tuples_summary=tuples_summary,
    )
    try:
        raw = call_claude_text(prompt, max_tokens=max_tokens)
    except Exception as e:
        logger.warning("agent2b: row-pattern LLM call failed: %r", e)
        return []
    if not raw:
        return []

    parsed = _parse_array_json(raw)
    if not parsed:
        return []
    out: list[RowPattern] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        rp = RowPattern.from_dict(item)
        if rp is None:
            continue
        try:
            compiled = re.compile(rp.regex)
        except re.error as e:
            logger.warning("agent2b: discarded invalid regex %r: %s", rp.regex[:80], e)
            continue
        if rp.example_match and not compiled.search(rp.example_match):
            logger.warning(
                "agent2b: discarded pattern — regex %r doesn't match its example_match %r",
                rp.regex[:80], rp.example_match[:80],
            )
            continue
        out.append(rp)
    return out


def _parse_array_json(raw: str) -> list | None:
    """Tolerant JSON-array parser with truncation recovery for
    arrays of objects. Handles three shapes:
      - bare JSON `[…]`
      - whole response wrapped in ``` … ``` fence
      - prose preamble followed by a ```json … ``` block
    """
    s = raw.strip()
    # Pull out fenced JSON block from anywhere (handles `Looking at
    # the data, … ```json [...] ``` …` shaped responses).
    fence_match = re.search(
        r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```",
        s, re.DOTALL,
    )
    if fence_match:
        s = fence_match.group(1)
    elif s.startswith("```"):
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
