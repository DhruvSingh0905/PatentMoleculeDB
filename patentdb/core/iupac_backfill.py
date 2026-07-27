"""Stage 8 — IUPAC backfill from PubChem.

GP-embedded compounds (Strategy 0) ship canonical SMILES + InChIKey but
NO IUPAC name — Google Patents' ``<span itemprop="smiles">`` tag doesn't
have a paired IUPAC text field. This leaves ~50-60% of every patent's
example_index entries with an empty `iupac_name`, which is unhelpful
for human review of the workbook.

PubChem's PUG-REST API serves canonical IUPAC names keyed by InChIKey
for free, with no auth. ~200ms per call. Results are cached locally
so re-runs are zero-cost.

Endpoint:
    https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{IK}
    /property/IUPACName/JSON

When PubChem doesn't have the compound (small private-patent novel
chemistry), the call returns 404 and we leave the IUPAC empty. The
cache records the 404 so we don't retry.

Patent-agnostic. Runs as a post-extraction step on `example_index`.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

from . import config

logger = logging.getLogger(__name__)

CACHE_FILE = config.OUTPUT_DIR / "text_extraction" / "_cache" / "pubchem_iupac.json"

# PubChem PUG-REST is rate-limited to ~5 req/sec. Add a small sleep
# between consecutive misses to stay polite.
_INTER_REQUEST_SLEEP = 0.05   # 50ms → ~20 req/sec headroom
_TIMEOUT = 15


def _load_cache() -> dict[str, str | None]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, str | None]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True))


class _TransientPubChemError(Exception):
    """Network failure or 5xx — caller should NOT cache the result, so a
    later run can retry once PubChem / the network recovers."""


def _fetch_iupac(inchikey: str) -> Optional[str]:
    """One PubChem hit. Returns the IUPAC string, None when PubChem
    confirms no entry exists (404), or raises `_TransientPubChemError`
    on network / 5xx failures so the caller can skip caching.
    """
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
        f"{inchikey}/property/IUPACName/JSON"
    )
    try:
        r = requests.get(url, timeout=_TIMEOUT)
    except requests.RequestException as e:
        raise _TransientPubChemError(f"network error: {e!r}") from e
    if r.status_code == 404:
        return None
    if r.status_code >= 500:
        raise _TransientPubChemError(f"PubChem 5xx: {r.status_code}")
    if r.status_code != 200:
        # 4xx other than 404 — treat as definitive (bad request), cache as None
        logger.debug(
            "PubChem returned %d for %s — caching as no-entry",
            r.status_code, inchikey,
        )
        return None
    try:
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            return props[0].get("IUPACName") or None
    except (ValueError, KeyError):
        pass
    return None


def backfill_iupacs_for_example_index(
    example_index: dict,
    *,
    max_lookups: int | None = None,
    cache: dict[str, str | None] | None = None,
) -> int:
    """Fill in empty `iupac_name` fields by looking up the entry's
    InChIKey in PubChem. Returns the number of NEW iupacs added.

    Mutates `example_index` in place. Uses a JSON cache so reruns
    skip already-resolved (or already-known-absent) keys.

    Args:
        example_index: cid → record dict. Each record needs `inchikey`.
        max_lookups: stop after this many *cache-miss* PubChem hits.
            Default unlimited.
        cache: pre-loaded cache dict. Pass to share across patents
            in one process; defaults to loading from disk.
    """
    if cache is None:
        cache = _load_cache()
    n_filled = 0
    n_new_misses = 0

    targets = [
        (cid, rec) for cid, rec in example_index.items()
        if rec.get("inchikey") and not (rec.get("iupac_name") or "").strip()
    ]
    if not targets:
        return 0

    logger.info(
        "iupac_backfill: %d entries with InChIKey but no IUPAC",
        len(targets),
    )

    save_every = 100
    new_resolutions_since_save = 0
    n_transient_failures = 0

    for _cid, rec in targets:
        ik = rec["inchikey"]
        if ik in cache:
            iupac = cache[ik]
            if iupac:
                rec["iupac_name"] = iupac
                rec["iupac_source"] = "pubchem_backfill"
                n_filled += 1
            continue
        # Cache miss — call PubChem
        if max_lookups is not None and n_new_misses >= max_lookups:
            break
        try:
            iupac = _fetch_iupac(ik)
        except _TransientPubChemError as e:
            # Network blip or PubChem 5xx — DON'T cache; retry next run.
            n_transient_failures += 1
            logger.debug("iupac_backfill: transient failure for %s: %s", ik, e)
            if _INTER_REQUEST_SLEEP:
                time.sleep(_INTER_REQUEST_SLEEP)
            continue
        cache[ik] = iupac
        new_resolutions_since_save += 1
        n_new_misses += 1
        if iupac:
            rec["iupac_name"] = iupac
            rec["iupac_source"] = "pubchem_backfill"
            n_filled += 1
        if _INTER_REQUEST_SLEEP:
            time.sleep(_INTER_REQUEST_SLEEP)
        if new_resolutions_since_save >= save_every:
            _save_cache(cache)
            new_resolutions_since_save = 0

    if new_resolutions_since_save:
        _save_cache(cache)
    logger.info(
        "iupac_backfill: filled %d IUPACs (%d PubChem calls, "
        "%d cache hits, %d transient failures, %d total entries targeted)",
        n_filled,
        n_new_misses,
        len(targets) - n_new_misses - n_transient_failures,
        n_transient_failures,
        len(targets),
    )
    return n_filled
