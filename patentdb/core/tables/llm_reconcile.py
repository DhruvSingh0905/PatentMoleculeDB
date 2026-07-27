"""LLM resolution of reconciliation conflicts (BLUEPRINT L5 oracle).

When the deterministic reconciler can't settle a (patent, assay-kind) — sources
disagree, an off-by-one row shift, notes/images mangling a value column, unit
confusion — the LLM sees BOTH sources' FULL compound→value lists (the agreeing
compounds act as anchors to detect a systematic shift) and returns the correct
values for the disagreeing compounds. This is how OCR quirks get fixed — once,
by the model — instead of per-patent deterministic patches.

Cheap + deterministic: one batched call per conflicted assay-group, cached in the
API response cache AND the reviewable resolution_memory (exact key). Re-runs and a
stable corpus cost ~0 LLM.
"""
from __future__ import annotations

import json
import re

from .. import config
from ..api_client import call_claude_text_batch
from .resolution_memory import ResolutionMemory, context_key

_PROMPT = """Two independent OCR extractions of ONE assay column from patent {patent} \
(assay kind: {kind}) disagree for some compounds. Either may have OCR errors: rows \
shifted by one (off-by-one), merged cells, stereo-notes or structure-images landing in \
a value cell, unit confusion (nM vs µM), or decimal slips.

SOURCE A (MinerU table grid), compound = value:
{a}

SOURCE B (Google Patents text), compound = value:
{b}

Compounds where A and B AGREE are already settled — use them as ANCHORS to detect any \
systematic shift between the sources. For each DISAGREEING compound, give the correct \
value. If one source is clearly mangled (its values are shifted by a row, or contain \
non-numeric notes), trust the other; otherwise judge per compound.

Return ONLY JSON, no prose:
{{"resolved": {{"<compound>": <number or "+"/"++" grade or null-if-untestable>, ...}},
  "source_used": "A"|"B"|"mixed", "confidence": "high"|"low", "reason": "<= 12 words"}}
Include only the disagreeing compounds in "resolved"."""


def _fmt(pairs: dict, cap: int = 150) -> str:
    items = sorted(pairs.items(), key=lambda kv: _cidsort(kv[0]))[:cap]
    return ", ".join(f"{c}={v}" for c, v in items)


def _cidsort(c: str):
    m = re.match(r"([A-Za-z-]*)(\d+)", c or "")
    return (m.group(1), int(m.group(2))) if m else (c, 0)


def _parse(text: str | None) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("resolved"), dict) else None


def resolve_groups(groups: list[dict], *, memory: ResolutionMemory,
                   patent_id: str = "") -> dict[int, dict]:
    """groups[i] = {patent, kind, a_pairs, b_pairs}.
    Returns {i: {"resolved": {cid: value}, "confidence": "high"|"low"}}.

    Checks memory first (no LLM on cached/active); batches the misses; stores
    results confidence-gated. Bulletproof against OCR quirks because the model
    sees both full lists, not just the conflicting cell.
    """
    out: dict[int, dict] = {}
    reqs: list[dict] = []
    miss: list[tuple[int, dict, str]] = []
    for i, g in enumerate(groups):
        key = context_key(g["patent"], g["kind"], g["a_pairs"], g["b_pairs"])
        cached = memory.get(key)
        if cached is not None:
            out[i] = {"resolved": cached["resolved"],
                      "confidence": cached.get("confidence", "low")}
            continue
        reqs.append({
            "prompt": _PROMPT.format(patent=g["patent"], kind=g["kind"],
                                     a=_fmt(g["a_pairs"]), b=_fmt(g["b_pairs"])),
            "max_tokens": 3000,
            "cache_key": f"reconcile:{key}",
            "model": config.DEFAULT_MODEL,
        })
        miss.append((i, g, key))

    if reqs:
        responses = call_claude_text_batch(reqs, patent_id=patent_id)
        for (i, g, key), resp in zip(miss, responses):
            parsed = _parse(resp)
            if parsed is None:
                continue
            conf = "high" if parsed.get("confidence") == "high" else "low"
            res = {str(k): v for k, v in parsed["resolved"].items()}
            memory.put(key, resolved=res,
                       source_used=str(parsed.get("source_used", "mixed")),
                       confidence=conf, reason=str(parsed.get("reason", ""))[:120],
                       meta={"patent": g["patent"], "kind": g["kind"]})
            out[i] = {"resolved": res, "confidence": conf}
        memory.save()
    return out
