"""IUPAC HARVEST fallback — Strategy 5 in the IUPAC extraction stack.

Fires when deterministic regex Strategies 1-4 in `routes/google_patents.py`
fail to recover ≥50% of the compounds the assay HARVEST already identified.
Uses a single-shot LLM agent (`agent_iupac_extract`) on high-density chunks
of patent text, persists discovered patterns to `IupacPatternLibrary`, and
caches results by chunk fingerprint so re-runs are $0.

Mirrors the shape of `harvest/orchestrator.py` (the assay version).

Public entry point:
    iupac_burst(patent_id, text, existing_pairs, n_assay_compounds, cost_tracker)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from .agent_iupac_extract import extract_iupac_pairs
from ..iupac_pattern_library import IupacPattern, IupacPatternLibrary

logger = logging.getLogger(__name__)


# ── Gap signal ─────────────────────────────────────────────────────


GAP_THRESHOLD = 0.5
"""Fire only when n_iupac / n_assay_compounds < this. Anchored on assay
HARVEST's compound count because that's our most reliable count of
'how many compounds this patent actually has'."""


def _iupac_gap_signal(
    n_existing_iupac: int,
    n_assay_compounds: int,
) -> bool:
    """True if there's a meaningful coverage shortfall."""
    if n_assay_compounds <= 0:
        return False
    coverage = n_existing_iupac / n_assay_compounds
    return coverage < GAP_THRESHOLD


# ── Chunking ───────────────────────────────────────────────────────


def _chunk_high_density_regions(
    text: str,
    chunk_size: int = 8_000,
    max_chunks: int = 4,
) -> list[tuple[int, str]]:
    """Pick top-N chunks of `chunk_size` chars ranked by IUPAC-density
    proxy (paren+bracket count). Returns [(absolute_offset, chunk_text)].

    Why density-ranked: the LLM cost is bounded by chunk count, so we
    only spend on regions likely to contain IUPAC names. Background +
    abstract sections won't qualify.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [(0, text)]

    # Score each window of chunk_size chars by paren+bracket density
    # (cheap proxy for IUPAC content)
    stride = chunk_size // 2
    scored: list[tuple[int, int]] = []  # (score, start_offset)
    for start in range(0, len(text) - chunk_size + 1, stride):
        sub = text[start:start + chunk_size]
        score = sub.count("(") + sub.count("[") + sub.count("{")
        scored.append((score, start))

    scored.sort(key=lambda x: -x[0])
    selected: list[tuple[int, str]] = []
    used_starts: list[int] = []
    for score, start in scored:
        # Skip overlapping chunks (within stride of an already-selected one)
        if any(abs(start - u) < stride for u in used_starts):
            continue
        selected.append((start, text[start:start + chunk_size]))
        used_starts.append(start)
        if len(selected) >= max_chunks:
            break
    return selected


# ── Fingerprinting ─────────────────────────────────────────────────


def _chunk_fingerprint(chunk: str) -> str:
    """A stable structural hash for an IUPAC text chunk.

    Captures shape (length bucket, paren/bracket/comma density, TABLE
    presence, salt/acid suffix density) without leaking specific tokens.
    Two patents with similar IUPAC-listing layouts produce the same fp →
    a previously-learned pattern auto-applies via cache + library.

    NOTE: Use this only for *pattern-library* lookups (regex shapes that
    can be re-applied across patents). DO NOT use for caching concrete
    `(compound_id, iupac_name)` pairs — those must use
    `_content_fingerprint` so different patents don't share their
    compound data.
    """
    if not chunk:
        return "empty"
    n = len(chunk)
    parens = chunk.count("(") + chunk.count("[") + chunk.count("{")
    commas = chunk.count(",")
    has_table = bool(re.search(r"\bTABLE\s*\d", chunk, re.IGNORECASE))
    n_acid = len(re.findall(r"\bacid\b", chunk, re.IGNORECASE))
    n_salt = len(re.findall(r"\bsalt\b", chunk, re.IGNORECASE))
    payload = json.dumps({
        "len_bucket": "small" if n < 2000 else "medium" if n < 8000 else "large",
        "paren_per_kb": round(parens / max(1, n / 1000), 1),
        "comma_per_kb": round(commas / max(1, n / 1000), 1),
        "has_table_marker": has_table,
        "acid_density": "high" if n_acid > 5 else "low",
        "salt_density": "high" if n_salt > 3 else "low",
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _content_fingerprint(chunk: str, patent_id: str = "") -> str:
    """Content-based hash for cache keys carrying CONCRETE compound data.

    The shape-based `_chunk_fingerprint` is shared across patents (by
    design — for pattern-library reuse). Using it as a cache key for
    `(compound_id, iupac_name)` pairs caused cross-patent contamination:
    patent A's compounds got served to patent B because their TABLE
    layouts had the same shape signature.

    This fingerprint includes:
      - patent_id  (so different patents always get different keys)
      - sha256 of the actual chunk text (so different chunks of the
        same patent also get different keys)
    """
    h = hashlib.sha256()
    h.update(patent_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(chunk.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


# ── Cache helpers (shares the adaptive_extraction_rules.json file) ──


def _cache_path() -> Path:
    """Same file the assay HARVEST burst uses, with `iupac:` prefix to
    keep namespaces clean."""
    from ... import config
    return Path(config.OUTPUT_DIR) / "text_extraction" / "_cache" / "adaptive_extraction_rules.json"


def _read_cache() -> dict:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_cache(data: dict) -> None:
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Targeted chunking for chunk-and-verify ────────────────────────


def _find_cid_context_windows(
    text: str,
    cid: str,
    *,
    window_chars: int = 4000,
) -> list[tuple[int, str]]:
    """Find every place in `text` where `cid` is mentioned as a compound
    label (e.g., "Example 152", "Compound 152", "152.", "(152)", " 152 :")
    and return surrounding windows.

    Returns list of (offset, window_text). Multiple hits → one window per
    hit, so `iupac_burst_targeted` can fire the LLM on each.
    """
    if not cid or not text:
        return []
    # Patterns to match this cid as a compound label, case-insensitive.
    # Negative lookbehind/ahead avoids matching "152" inside "1525" etc.
    cid_re = re.escape(cid)
    patterns = [
        rf"\bExample\s+{cid_re}(?![0-9])",      # "Example 152"
        rf"\bCompound\s+{cid_re}(?![0-9])",     # "Compound 152"
        rf"\bCpd\.?\s*(?:No\.?\s*)?{cid_re}(?![0-9])",  # "Cpd. No. 152"
        rf"(?<![0-9.]){cid_re}\s*[.:](?![0-9])",        # "152." or "152:"
        rf"\(\s*{cid_re}\s*\)(?![0-9])",        # "(152)"
    ]
    seen_offsets: list[int] = []
    windows: list[tuple[int, str]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            off = m.start()
            # Skip if a previous pattern already produced a window very close
            if any(abs(off - prior) < window_chars // 2 for prior in seen_offsets):
                continue
            seen_offsets.append(off)
            half = window_chars // 2
            start = max(0, off - half)
            end = min(len(text), off + half)
            windows.append((off, text[start:end]))
    return windows


def iupac_burst_targeted(
    patent_id: str,
    text: str,
    missing_cids: list[str],
    cost_tracker,
    library: IupacPatternLibrary | None = None,
    *,
    window_chars: int = 5000,
) -> dict[str, str]:
    """Chunk-and-verify Strategy 5: fire LLM on text windows that
    specifically mention each missing cid. More thorough than
    `iupac_burst` (which density-ranks chunks); guarantees coverage
    of every cid that HARVEST has but example_index lacks.

    Returns: {compound_id → iupac_name} for the cids the LLM resolved.
    Cids the LLM couldn't extract (no match in the chunk) silently
    omitted.

    NO per-patent cap check — caller controls total budget. The user
    explicitly requested compute-unconstrained behavior here.
    """
    if not missing_cids or not text:
        return {}

    library = library or IupacPatternLibrary.default()
    cache = _read_cache()
    new_pairs: dict[str, str] = {}

    logger.info(
        "iupac_burst_targeted: %s — searching for %d missing cids",
        patent_id, len(missing_cids),
    )

    # Group: build a single ordered list of (target_cid, offset, window)
    # so we can fire LLM once per unique window even if multiple cids
    # share a window (often the case in TABLE-1 ordered lists).
    targets: list[tuple[str, int, str]] = []
    for cid in missing_cids:
        wins = _find_cid_context_windows(text, cid, window_chars=window_chars)
        if not wins:
            logger.info(
                "iupac_burst_targeted: no context for cid=%s in %s text",
                cid, patent_id,
            )
            continue
        for off, win in wins:
            targets.append((cid, off, win))

    # Dedup overlapping windows (within window_chars/2 of each other) so
    # we don't fire LLM redundantly. Keep the FIRST window per offset.
    targets.sort(key=lambda t: t[1])
    deduped: list[tuple[list[str], int, str]] = []
    for cid, off, win in targets:
        if deduped and abs(off - deduped[-1][1]) < window_chars // 2:
            # Merge: another cid hits the same/nearby window
            if cid not in deduped[-1][0]:
                deduped[-1][0].append(cid)
            continue
        deduped.append(([cid], off, win))

    logger.info(
        "iupac_burst_targeted: %s — %d unique chunks for %d cids",
        patent_id, len(deduped), len(missing_cids),
    )

    for chunk_idx, (cids, offset, chunk_text) in enumerate(deduped):
        # CONTENT-based key for the pairs cache (prevents cross-patent
        # leakage). Pattern-library still uses the shape fp.
        cache_key = f"iupac_targeted:{_content_fingerprint(chunk_text, patent_id)}"
        fp = _chunk_fingerprint(chunk_text)  # for library only

        cached = cache.get(cache_key)
        if cached is not None:
            for p in cached.get("pairs", []):
                cid = p.get("compound_id")
                nm = p.get("iupac_name")
                if cid and nm:
                    new_pairs.setdefault(cid, nm)
            continue

        result = extract_iupac_pairs(chunk_text, patent_id=patent_id)
        cache[cache_key] = {
            "pairs": result.pairs,
            "pattern_meta": result.pattern_meta,
            "_target_cids": cids,
            "_offset": offset,
        }
        if result.pattern_meta and result.pattern_meta.get("pattern_name"):
            library.add_discovery(
                IupacPattern(
                    pattern_name=result.pattern_meta["pattern_name"],
                    description=result.pattern_meta.get("description", ""),
                    regex_or_heuristic=result.pattern_meta.get("regex_or_heuristic", ""),
                    example_patent=patent_id,
                ),
                fingerprint=fp,
                patent_id=patent_id,
            )
        for p in result.pairs:
            cid = p["compound_id"]
            nm = p["iupac_name"]
            if cid and nm:
                new_pairs.setdefault(cid, nm)
        logger.info(
            "iupac_burst_targeted: chunk %d (offset %d, target=%s) → +%d pairs",
            chunk_idx, offset, cids[:3], len(result.pairs),
        )

    _write_cache(cache)
    logger.info(
        "iupac_burst_targeted: %s → %d new pairs (target was %d cids)",
        patent_id, len(new_pairs), len(missing_cids),
    )
    return new_pairs


# ── Public entry ───────────────────────────────────────────────────


def iupac_burst(
    patent_id: str,
    text: str,
    existing_pairs: dict[str, str],
    n_assay_compounds: int,
    cost_tracker,
    library: IupacPatternLibrary | None = None,
    *,
    force: bool = False,
    chunk_size: int = 15_000,
    max_chunks: int = 40,
) -> dict[str, str]:
    """Extract additional (compound_id → iupac_name) pairs via LLM
    fallback. Default behavior fires only when Strategies 1-4 undershoot;
    pass `force=True` to bypass the gap check and the per-patent budget
    cap (used when the caller has external budget controls or wants to
    overwrite wrong IUPACs, not just fill missing ones).

    Args:
        patent_id: Used for cost attribution + library provenance.
        text: Full patent description text.
        existing_pairs: What Strategies 1-4 already produced (compound_id → name).
        n_assay_compounds: From assay_tables.json — the "ground truth" count.
        cost_tracker: Per-patent LM cap enforced via `patent_lm_exceeded()`
            unless `force=True`.
        library: Optional override (defaults to repo-root paths).
        force: Skip gap-check and per-patent cap. Use when the caller is
            running a deliberate broader pass (e.g. to refresh entries
            from a previous extraction).
        chunk_size: Char window per LLM call.
        max_chunks: Max chunks to fire LLM on.

    Returns:
        New pairs the LLM extracted. Empty dict if no signal AND not force.
    """
    if not force:
        # 1. Gap check — only fire when we're meaningfully short
        if not _iupac_gap_signal(len(existing_pairs), n_assay_compounds):
            logger.info(
                "iupac_burst: no gap signal for %s (n_iupac=%d, n_assay=%d). "
                "Skipping LLM fallback.",
                patent_id, len(existing_pairs), n_assay_compounds,
            )
            return {}

        # 2. Per-patent budget gate
        if cost_tracker.patent_lm_exceeded(patent_id):
            logger.info(
                "iupac_burst: per-patent LM cap already hit for %s. Skipping.",
                patent_id,
            )
            return {}

    library = library or IupacPatternLibrary.default()
    cache = _read_cache()
    new_pairs: dict[str, str] = {}

    chunks = _chunk_high_density_regions(text, chunk_size=chunk_size, max_chunks=max_chunks)
    if not chunks:
        return {}

    logger.info(
        "iupac_burst: firing on %s (n_iupac=%d, n_assay=%d, chunks=%d)",
        patent_id, len(existing_pairs), n_assay_compounds, len(chunks),
    )

    for chunk_idx, (_offset, chunk_text) in enumerate(chunks):
        # CONTENT-based key for pairs cache (prevents cross-patent leakage).
        cache_key = f"iupac:{_content_fingerprint(chunk_text, patent_id)}"
        fp = _chunk_fingerprint(chunk_text)  # for library only

        # 3. Cache check. iupac_burst is FILL-ONLY — never overwrites
        # cids already in existing_pairs. Empirically, broad-density
        # chunks can return the same cid with a different (sometimes
        # worse) IUPAC than what regex strategies produced; overwriting
        # caused regressions on cleanly-formatted patents. The targeted
        # variant (`iupac_burst_targeted`) is the right tool when the
        # caller knows specific cids have wrong IUPACs.
        cached = cache.get(cache_key)
        if cached is not None:
            for p in cached.get("pairs", []):
                cid = p.get("compound_id")
                nm = p.get("iupac_name")
                if not (cid and nm):
                    continue
                if cid in existing_pairs:
                    continue
                if cid in new_pairs:
                    continue
                new_pairs[cid] = nm
            logger.info(
                "iupac_burst: cache hit for chunk %d (fp=%s, +%d pairs)",
                chunk_idx, fp, len(cached.get("pairs", [])),
            )
            continue

        # 4. Fire LLM agent
        result = extract_iupac_pairs(chunk_text, patent_id=patent_id)
        if not result.pairs:
            logger.info(
                "iupac_burst: chunk %d (fp=%s) → 0 pairs from LLM",
                chunk_idx, fp,
            )
            # Persist empty result to avoid re-firing
            cache[cache_key] = {"pairs": [], "pattern_meta": None}
            continue

        # 5. Vocab learning — patterns the LLM identified
        if result.pattern_meta and result.pattern_meta.get("pattern_name"):
            pattern = IupacPattern(
                pattern_name=result.pattern_meta["pattern_name"],
                description=result.pattern_meta.get("description", ""),
                regex_or_heuristic=result.pattern_meta.get("regex_or_heuristic", ""),
                example_patent=patent_id,
            )
            library.add_discovery(pattern, fingerprint=fp, patent_id=patent_id)

        # 6. Persist to cache + collect pairs
        cache[cache_key] = {
            "pairs": result.pairs,
            "pattern_meta": result.pattern_meta,
        }
        for p in result.pairs:
            cid = p["compound_id"]
            nm = p["iupac_name"]
            if not (cid and nm):
                continue
            if cid in existing_pairs:
                continue
            if cid not in new_pairs:
                new_pairs[cid] = nm

        # 7. Per-patent cap re-check (fail-closed mid-loop) — bypassed when force=True
        if not force and cost_tracker.patent_lm_exceeded(patent_id):
            logger.info(
                "iupac_burst: per-patent LM cap hit mid-burst at chunk %d "
                "for %s. Stopping.",
                chunk_idx, patent_id,
            )
            break

    # Persist cache once at end (atomic-ish write)
    _write_cache(cache)

    logger.info(
        "iupac_burst: %s → %d new pairs (existing=%d, total=%d)",
        patent_id, len(new_pairs), len(existing_pairs),
        len(existing_pairs) + len(new_pairs),
    )
    return new_pairs
