"""LLM-driven discovery of patent-specific compound-id prefixes.

Some patents use non-standard compound numbering — e.g. US11254686
labels its compounds `Z1, Z2, … Z10` instead of `Compound 1, Cpd. 2`.
Hardcoding every patent's prefix would be fragile, so we discover them.

How it works:
  1. Pre-extraction scans the patent text for `[A-Z]{1,3}\\d+` patterns
     that aren't already in the canonical vocabulary.
  2. Any candidate prefix seen ≥ MIN_OBSERVATIONS times near assay-table
     keywords (IC50, Ki, %INH, …) is sent to a single LLM call:
       "These tokens recur in the text near assay tables; are they
        compound-id codes or something else (units, kinase isoforms,
        publication codes)?"
  3. The LLM returns yes/no + the surface forms.
  4. Confirmed prefixes are appended to `assay_vocabulary.discoveries.json`
     with status=auto_loaded (so they're used THIS run after a reload).

Designed to fire at most once per patent. Cost: ~$0.005 per discovery call.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .compound_id import _load_prefixes, reload_prefixes

logger = logging.getLogger(__name__)


_DISCOVERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "assay_vocabulary.discoveries.json"

# A "candidate" cid-prefix pattern: 1-3 uppercase letters, optional dash,
# followed by 1-4 digits, with no preceding alphanumeric (so we don't
# grab the trailing chunk of "ABC123" inside a longer token).
#
# The optional dash catches the very common `I-0020` / `IIa-12` / `Cmpd-5`
# patent formats — without it, dashed cids never reach the LLM verifier
# and silently slip through as unknown tokens.
_CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,3})-?(\d{1,4})\b")

# Assay-table keyword anchor — discovery only fires for prefixes that
# CO-OCCUR with assay vocabulary, ruling out IUPAC fragments + journal
# citations.
_ASSAY_ANCHOR_RE = re.compile(
    r"\b(?:IC\s*50|EC\s*50|GI\s*50|CC\s*50|Ki|Kd|%\s*INH|inhibition|"
    r"binding|activity|μM|uM|nM|nanomolar|micromolar)\b",
    re.IGNORECASE,
)

# How many times a candidate must appear NEAR an assay anchor before we
# fire the LLM. Tuned so a one-off "ATP10" or "CYP3A4" doesn't trigger.
MIN_OBSERVATIONS = 8
ANCHOR_WINDOW = 400   # chars on either side of a candidate hit


def _scan_candidates(text: str) -> dict[str, int]:
    """Count candidate prefixes that co-occur with an assay anchor within
    ANCHOR_WINDOW chars. Returns {prefix_letters: hit_count}.
    """
    known = {p.lower() for p in _load_prefixes()}
    counts: Counter[str] = Counter()
    # Pre-compute anchor positions so we can do range checks cheaply.
    anchor_starts = sorted(m.start() for m in _ASSAY_ANCHOR_RE.finditer(text))
    if not anchor_starts:
        return {}
    # For each candidate, binary search for nearby anchor
    import bisect
    for cm in _CANDIDATE_RE.finditer(text):
        prefix = cm.group(1)
        if prefix.lower() in known:
            continue
        pos = cm.start()
        idx = bisect.bisect_left(anchor_starts, pos)
        # Closest anchor on either side
        nearest = None
        for j in (idx - 1, idx):
            if 0 <= j < len(anchor_starts):
                d = abs(anchor_starts[j] - pos)
                if nearest is None or d < nearest:
                    nearest = d
        if nearest is not None and nearest <= ANCHOR_WINDOW:
            counts[prefix] += 1
    # Filter by threshold
    return {p: n for p, n in counts.items() if n >= MIN_OBSERVATIONS}


def _sample_context(text: str, prefix: str, n: int = 5) -> list[str]:
    """Return up to `n` short snippets where `prefix` appears with digits,
    each ~200 chars wide. Used as evidence for the LLM call. Matches
    both `Z1` and `I-0020` forms (optional dash between letters and digits).
    """
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(prefix)}-?\d{{1,4}}\b")
    out: list[str] = []
    for m in pat.finditer(text):
        s = max(0, m.start() - 100)
        e = min(len(text), m.end() + 200)
        snippet = re.sub(r"\s+", " ", text[s:e]).strip()
        if snippet and snippet not in out:
            out.append(snippet)
        if len(out) >= n:
            break
    return out


def _llm_confirm(prefix: str, samples: list[str], patent_id: str) -> dict[str, Any]:
    """Ask the LLM whether `prefix` is a compound-id format.

    Returns: {"is_compound_id": bool, "surface_forms": [str, ...], "reason": str}
    """
    try:
        from .api_client import call_claude_text
    except Exception as e:
        logger.warning("cid_prefix_discovery: api_client unavailable (%r) — skipping LLM call", e)
        return {"is_compound_id": False, "surface_forms": [], "reason": "no api client"}

    sample_block = "\n\n".join(f"  • {s}" for s in samples)
    prompt = (
        f"Patent {patent_id} contains repeated tokens like "
        f"`{prefix}1`, `{prefix}2`, … `{prefix}N` near assay-table keywords "
        f"(IC50, Ki, etc.). Below are 5 verbatim snippets.\n\n"
        f"{sample_block}\n\n"
        f"Is `{prefix}` being used as a compound-id prefix in this patent? "
        f"It could instead be a kinase isoform code (e.g., CYP3A4), a publication "
        f"citation, or a unit notation. Respond with strict JSON:\n"
        f'  {{"is_compound_id": <bool>, "surface_forms": [<strings>], '
        f'"reason": "<one short sentence>"}}\n\n'
        f"If `is_compound_id` is true, list the surface forms a parser should "
        f"strip — typically just [\"{prefix}\"], but include variants if the "
        f"patent uses any (e.g., [\"Z\", \"Z-\", \"Cpd Z\"])."
    )
    try:
        from . import config as _cfg
        if not _cfg.LLM_RECOVERY_ENABLED:
            raw = None
        else:
            raw = call_claude_text(
                prompt,
                patent_id=patent_id,
                max_tokens=300,
            )
        if not raw:
            return {"is_compound_id": False, "surface_forms": [], "reason": "no llm response"}
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"is_compound_id": False, "surface_forms": [], "reason": "no json"}
        parsed = json.loads(m.group(0))
        return {
            "is_compound_id": bool(parsed.get("is_compound_id")),
            "surface_forms": [s for s in parsed.get("surface_forms", []) if isinstance(s, str)],
            "reason": str(parsed.get("reason", ""))[:200],
        }
    except Exception as e:
        logger.warning("cid_prefix_discovery: LLM call failed (%r)", e)
        return {"is_compound_id": False, "surface_forms": [], "reason": str(e)[:200]}


def _append_to_discoveries(
    prefix: str, surface_forms: list[str], patent_id: str, reason: str,
) -> bool:
    """Append a confirmed COMPOUND_ID_PREFIX entry to the discoveries file
    with status=auto_loaded so it's picked up THIS run. Patent-id is
    recorded as first_seen_patent; surface_forms are deduped.
    """
    if not surface_forms:
        surface_forms = [prefix]
    data: dict[str, Any]
    if _DISCOVERIES_PATH.exists():
        try:
            data = json.loads(_DISCOVERIES_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            data = {"schema_version": "1.0", "tokens": []}
    else:
        data = {"schema_version": "1.0", "tokens": []}
    tokens = data.setdefault("tokens", [])
    lemma = f"cid_prefix_{prefix.lower()}"
    existing = next((t for t in tokens if t.get("lemma") == lemma), None)
    if existing is None:
        tokens.append({
            "lemma": lemma,
            "class": "COMPOUND_ID_PREFIX",
            "surface_forms": list(dict.fromkeys(surface_forms)),
            "status": "auto_loaded",
            "fingerprints_observed": [patent_id],
            "first_seen_patent": patent_id,
            "first_seen": str(__import__("datetime").date.today()),
            "n_observations": 1,
            "llm_reason": reason,
        })
        new_entry = True
    else:
        for s in surface_forms:
            if s not in existing["surface_forms"]:
                existing["surface_forms"].append(s)
        if patent_id not in existing.get("fingerprints_observed", []):
            existing["fingerprints_observed"].append(patent_id)
        existing["n_observations"] = existing.get("n_observations", 0) + 1
        existing["status"] = "auto_loaded"
        new_entry = False
    data["last_updated"] = str(__import__("datetime").date.today())
    _DISCOVERIES_PATH.write_text(json.dumps(data, indent=2))
    return new_entry


def discover_cid_prefixes(text: str, patent_id: str) -> int:
    """Top-level entry. Scans `text`, fires the LLM on each strong
    candidate, appends confirmed prefixes to the discoveries file, and
    clears the vocabulary cache so they take effect immediately.

    Returns the number of newly-confirmed prefixes.
    """
    candidates = _scan_candidates(text)
    if not candidates:
        return 0
    logger.info(
        "cid_prefix_discovery: %s — %d candidate prefix(es) above threshold: %s",
        patent_id, len(candidates), dict(sorted(candidates.items(), key=lambda kv: -kv[1])),
    )
    n_confirmed = 0
    for prefix, count in sorted(candidates.items(), key=lambda kv: -kv[1]):
        samples = _sample_context(text, prefix, n=5)
        if not samples:
            continue
        verdict = _llm_confirm(prefix, samples, patent_id)
        logger.info(
            "cid_prefix_discovery: %s prefix %r (%d hits) → "
            "is_compound_id=%s reason=%r",
            patent_id, prefix, count,
            verdict.get("is_compound_id"), verdict.get("reason", "")[:80],
        )
        if verdict.get("is_compound_id"):
            n_confirmed += int(_append_to_discoveries(
                prefix, verdict.get("surface_forms") or [prefix],
                patent_id, verdict.get("reason", ""),
            ))
    if n_confirmed:
        reload_prefixes()
        logger.info(
            "cid_prefix_discovery: %s — appended %d new prefix(es) to discoveries; "
            "vocabulary cache cleared",
            patent_id, n_confirmed,
        )
    return n_confirmed
