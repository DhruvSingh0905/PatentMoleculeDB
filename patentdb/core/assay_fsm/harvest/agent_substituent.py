"""LLM agent that extracts R-group / substituent definitions from a patent
text chunk — the fallback arm of the substituent-table scan when the
deterministic `routes.text_markush` symbolic regex undershoots.

Symbolic-first by design: this LLM call fires ONLY on chunks where the
free regex pass found a substituent-table SHAPE (high R-label density)
but resolved few/no option lists — typically TABLE-format definitions
("R1 | methyl, ethyl | R2 | H, Cl") that the prose-oriented regex
(`R1 is selected from ...`) cannot match. Single Claude call per chunk.

Reuses `core/api_client.call_claude_text`, which carries a prompt-hash
cache (re-runs are $0) and per-patent cost attribution. Cost is gated by
the orchestrator (Markush-likely patents only, capped chunk count).
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
class SubstituentExtractResult:
    """One LLM call's output."""
    definitions: dict[str, list[str]] = field(default_factory=dict)
    """{label -> [option_text, ...]}, e.g. {"R1": ["methyl", "ethyl"], ...}.
    Option text (not SMILES) — the orchestrator resolves to SMILES with the
    same SUBSTITUENT_LIBRARY/OPSIN/expand_substituent chain the symbolic
    path uses, so resolution stays consistent across both arms."""

    raw_response: str = ""
    """Original LLM text — kept for debugging."""


_PROMPT_TEMPLATE = """\
You are a chemistry-aware Markush R-group extractor. The snippet below is
from a pharmaceutical patent that defines a genus (Markush) structure. Each
R-group position (R1, R2, ..., and single-letter positions like G, X, Y, Z,
or superscripted forms like R^1, R1a) is defined as a list of allowed
substituent options.

Extract EVERY R-group position and its list of allowed substituent option
NAMES. Handle BOTH formats:
  - Prose:  "R1 is selected from the group consisting of methyl, ethyl, and
            phenyl; R2 is hydrogen or halogen"
  - Table:  rows/cells pairing a label with its options, e.g.
            "R1 | methyl, ethyl, propyl" or a column of labels beside a
            column of option lists.

Rules:
  - Return the option NAMES verbatim (e.g. "methyl", "(C1-C4)-alkyl",
    "optionally substituted phenyl", "hydrogen", "halogen"). Do NOT convert
    to SMILES — downstream code resolves names to SMILES.
  - Do NOT invent options not present in the text.
  - Skip prose that is not a substituent definition (assay results, synthesis
    steps, claim boilerplate).
  - If the snippet contains NO R-group definitions, return {{"definitions": {{}}}}.

Return STRICT JSON, no prose before/after:
{{
  "definitions": {{
    "R1": ["methyl", "ethyl", "phenyl"],
    "R2": ["hydrogen", "halogen"]
  }}
}}

Text snippet:
---BEGIN---
{chunk}
---END---

Return ONLY the JSON object."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# A label looks like R<digits><optional letter>, or a bare single-letter
# position. Used to sanitize keys the LLM returns.
_LABEL_RE = re.compile(r"^(?:R\d{1,3}[a-z]?|[GXYZ])$", re.IGNORECASE)


def looks_like_substituent_table(chunk_text: str) -> int:
    """Cheap ($0) score for "this chunk defines R-group substituents".

    Counts distinct R-label definition cues. The orchestrator ranks chunks
    by this and only sends the top few to the LLM — so the LLM never reads
    a chunk that has no substituent-table shape. Patent-agnostic: matches
    the SHAPE (R-label + definition verb / list), no patent IDs.
    """
    if not chunk_text:
        return 0
    # Distinct R-labels that are followed by a definition cue ("is", "=",
    # "selected from", a colon, or a pipe/cell separator).
    labels = set(re.findall(
        r"\b(R\d{1,3}[a-z]?)\s*(?:is\b|=|:|\||\bselected from\b|\brepresents\b)",
        chunk_text, re.IGNORECASE,
    ))
    # "selected from the group consisting of" is a very strong cue.
    n_consisting = len(re.findall(r"selected from", chunk_text, re.IGNORECASE))
    return len(labels) + n_consisting


def extract_substituent_defs(
    chunk_text: str,
    patent_id: str,
    model: str | None = None,
    cost_category: str = "lm",
) -> SubstituentExtractResult:
    """Run the LLM agent on one chunk. Returns parsed {label: [options]}.

    Empty/blank chunk or LLM failure → empty result (caller falls back to
    whatever the symbolic pass already produced).
    """
    if not chunk_text or not chunk_text.strip():
        return SubstituentExtractResult()

    prompt = _PROMPT_TEMPLATE.format(chunk=chunk_text)
    response = call_claude_text(
        prompt=prompt,
        model=model or config.DEFAULT_MODEL,   # Sonnet by default (cost)
        patent_id=patent_id,
        max_tokens=4096,
        cost_category=cost_category,
    )
    if response is None:
        return SubstituentExtractResult()

    m = _JSON_BLOCK_RE.search(response)
    if not m:
        logger.warning(
            "agent_substituent: no JSON block in response (patent=%s, len=%d)",
            patent_id, len(response),
        )
        return SubstituentExtractResult(raw_response=response)

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        logger.warning(
            "agent_substituent: JSON parse failed (patent=%s)", patent_id,
        )
        return SubstituentExtractResult(raw_response=response)

    raw_defs = data.get("definitions", {}) or {}
    clean: dict[str, list[str]] = {}
    if isinstance(raw_defs, dict):
        for label, opts in raw_defs.items():
            label = str(label).strip()
            if not _LABEL_RE.match(label):
                continue
            if not isinstance(opts, list):
                continue
            seen: set[str] = set()
            deduped: list[str] = []
            for o in opts:
                o = str(o).strip()
                key = o.lower()
                if len(o) > 1 and key not in seen:
                    seen.add(key)
                    deduped.append(o)
            if deduped:
                # Normalize label casing to R<n> (uppercase R).
                norm = re.sub(r"^r", "R", label, flags=re.IGNORECASE)
                clean[norm] = deduped

    return SubstituentExtractResult(definitions=clean, raw_response=response)
