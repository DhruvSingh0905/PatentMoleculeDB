"""LLM agent that extracts (compound_id, IUPAC name) pairs from a patent
text chunk — used by the Strategy 5 IUPAC HARVEST fallback when the
deterministic regex strategies undershoot.

Single Claude call per chunk. Reuses `core/api_client.call_claude_text`,
which includes a prompt-hash cache (re-runs are $0) and per-patent
cost attribution.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ...api_client import call_claude_text
from ... import config

logger = logging.getLogger(__name__)


@dataclass
class IupacExtractResult:
    """One LLM call's output."""
    pairs: list[dict] = field(default_factory=list)
    """Each pair: {compound_id: str, iupac_name: str, source_pattern: str}."""

    pattern_meta: dict | None = None
    """Optional: {pattern_name, description, regex_or_heuristic, example_patent}.
    Set when the LLM identifies a reusable extraction pattern."""

    raw_response: str = ""
    """Original LLM text — kept for debugging."""


_PROMPT_TEMPLATE = """\
You are a chemistry-aware IUPAC name extractor. Given a patent text snippet,
identify EVERY (compound number, IUPAC name) pair you can find.

Also classify the FORMAT used in this snippet so future runs can skip the
LLM. Common formats:
  - TABLE_COMMA_ORDERED_LIST: "TABLE 1 <iupac_1>, <iupac_2>, <iupac_3>, ..."
    — position implies compound number, no explicit numbering
  - REVERSE_PAREN: "<iupac> (Compound 5A)"
  - HEADER_PREFIX: "Example 5: <iupac>" or "Compound 5: <iupac>"
  - CLAIMS_NUMBERED: "<iupac> (5);"
  - OTHER: format you see but doesn't match above

Return STRICT JSON. No prose before/after the JSON block.

Schema:
{{
  "pairs": [
    {{"compound_id": "5", "iupac_name": "...", "source_pattern": "TABLE_COMMA_ORDERED_LIST"}},
    ...
  ],
  "pattern_meta": {{
    "pattern_name": "TABLE_COMMA_ORDERED_LIST",
    "description": "TABLE 1 lists IUPAC names separated by top-level commas; \
position implies compound number",
    "regex_or_heuristic": "after r'\\\\bTABLE\\\\s*1\\\\b', depth-aware split on commas"
  }}
}}

Text snippet:
---BEGIN---
{chunk}
---END---

Return ONLY the JSON object."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# Salvage pattern for malformed-JSON responses: when the LLM hits its
# token limit mid-pair list, we still get the prefix of valid pairs.
# This regex extracts each {"compound_id":"...","iupac_name":"..."}
# block individually, allowing other keys/values to be ignored.
_PAIR_FRAGMENT_RE = re.compile(
    r'"compound_id"\s*:\s*"([^"]+)"\s*,\s*"iupac_name"\s*:\s*"([^"]+(?:\\"[^"]*)*)"',
    re.DOTALL,
)


def _salvage_pairs_from_partial_json(raw: str) -> list[dict]:
    """Extract {compound_id, iupac_name} pairs from a malformed/truncated
    JSON string. Used when full json.loads fails (LLM hit max_tokens
    mid-pairs-list, unterminated string, etc.). Returns whatever pairs
    we can read regardless of overall structure validity.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for m in _PAIR_FRAGMENT_RE.finditer(raw):
        cid = m.group(1).strip()
        nm = m.group(2).strip().replace('\\"', '"')
        if not cid or not nm or cid in seen:
            continue
        seen.add(cid)
        out.append({
            "compound_id": cid,
            "iupac_name": nm,
            "source_pattern": "OTHER",
        })
    return out


def extract_iupac_pairs(
    chunk_text: str,
    patent_id: str,
    model: str | None = None,
) -> IupacExtractResult:
    """Run the LLM agent on one chunk. Returns parsed pairs + pattern metadata.

    Empty chunk or LLM failure → empty result.
    """
    if not chunk_text or not chunk_text.strip():
        return IupacExtractResult()

    prompt = _PROMPT_TEMPLATE.format(chunk=chunk_text)
    response = call_claude_text(
        prompt=prompt,
        model=model or config.DEFAULT_MODEL,
        patent_id=patent_id,
        max_tokens=8192,   # bumped from 4096 — large IUPAC chunks can
                            # contain 30+ pairs and blow past 4k tokens
    )
    if response is None:
        return IupacExtractResult()

    # The model occasionally wraps JSON in ```json ... ```; extract the
    # first {...} block.
    m = _JSON_BLOCK_RE.search(response)
    if not m:
        logger.warning(
            "agent_iupac_extract: no JSON block in response (patent=%s, "
            "len=%d)",
            patent_id, len(response),
        )
        return IupacExtractResult(raw_response=response)
    raw_json = m.group(0)
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # Salvage: even if the full JSON failed, try to extract individual
        # {compound_id, iupac_name} pairs via line-by-line regex. This
        # recovers pairs from truncated / unterminated responses where
        # the LLM ran out of tokens mid-pairs-list.
        salvaged_pairs = _salvage_pairs_from_partial_json(raw_json)
        if salvaged_pairs:
            logger.info(
                "agent_iupac_extract: JSON parse failed but salvaged "
                "%d pairs from partial JSON (patent=%s)",
                len(salvaged_pairs), patent_id,
            )
            return IupacExtractResult(
                pairs=salvaged_pairs,
                pattern_meta=None,
                raw_response=response,
            )
        logger.warning(
            "agent_iupac_extract: JSON parse failed and no salvageable "
            "pairs (patent=%s)", patent_id,
        )
        return IupacExtractResult(raw_response=response)

    pairs = data.get("pairs", []) or []
    # Sanitize: every pair needs compound_id + iupac_name as strings
    clean_pairs = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        cid = str(p.get("compound_id", "")).strip()
        nm = str(p.get("iupac_name", "")).strip()
        if not cid or not nm:
            continue
        clean_pairs.append({
            "compound_id": cid,
            "iupac_name": nm,
            "source_pattern": str(p.get("source_pattern", "OTHER")).strip(),
        })

    pm = data.get("pattern_meta")
    if isinstance(pm, dict):
        # Sanitize pattern_meta keys we care about
        pattern_meta = {
            "pattern_name": str(pm.get("pattern_name", "OTHER")).strip(),
            "description": str(pm.get("description", "")).strip(),
            "regex_or_heuristic": str(pm.get("regex_or_heuristic", "")).strip(),
        }
    else:
        pattern_meta = None

    return IupacExtractResult(
        pairs=clean_pairs,
        pattern_meta=pattern_meta,
        raw_response=response,
    )
