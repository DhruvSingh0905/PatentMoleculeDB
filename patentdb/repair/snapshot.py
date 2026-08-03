"""A patent's result is FROZEN when it is processed. Later patches do not move it.

Every capability patch until now was a global code change, so improving one
patent silently re-extracted the other 102. That is the coupling behind every
declined patch in this loop's history, and it is not a bug in any of them:

    US10660877  860 -> 0    one patch to `_is_namelike`, and a patent that was
                            never the target lost everything. The patch was
                            locally correct.
    US11420968  664 -> 552
    US10626094  412 -> 382

Measured, the patch that cost US10660877 its 860 compounds did not touch a
single one of its rows — it changed `_header_rows_of`'s promotion, the block
stopped deriving its own header, and 2,542 records became unreadable. Nothing
in a per-patch review could have predicted that, and no amount of context given
to the model would have.

So stop asking a patch not to change other patents, and stop re-deriving
patents that are already done. A snapshot is the extracted result plus the
journal head that produced it. `records_for()` returns the snapshot when one
exists, so:

  * a patent processed today keeps today's answer for good;
  * a patch adopted tomorrow is judged on the patent it targets and on patents
    not yet frozen, which is the only population it can still help or harm;
  * the corpus total stops being a moving floor that every new patch has to
    clear against work that is already finished.

The cost of this is real and worth stating plainly: a later patch that would
genuinely have IMPROVED a frozen patent does not get to. That is the trade —
stability over retroactive gain — and it is reversible per patent with
`thaw()`, or wholesale by deleting the directory. `freeze()` records the
journal head so it is always knowable which code produced a given answer.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass

from ..core import config

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = config.OUTPUT_DIR / "extraction_snapshots"


def _journal_head() -> str:
    """The id of the most recent applied patch — which code made this answer."""
    try:
        from .parser_repair import journal_read
        applied = [e for e in journal_read() if e.get("applied")]
        return str(applied[-1].get("id")) if applied else "base"
    except Exception:
        return "unknown"


def path_for(patent_id: str):
    return SNAPSHOT_DIR / f"{patent_id.upper()}.json"


def is_frozen(patent_id: str) -> bool:
    return path_for(patent_id).exists()


def frozen_ids() -> set[str]:
    if not SNAPSHOT_DIR.exists():
        return set()
    return {p.stem.upper() for p in SNAPSHOT_DIR.glob("*.json")}


def freeze(patent_id: str, records) -> dict | None:
    """Record this patent's result as final. Returns what was written, or None.

    A patent that produced NOTHING is not frozen. Freezing is for finished
    work, and zero is not an answer — it is the open case the whole repair loop
    exists for. Pinning it would mean a patch that finally reads the document
    could never be seen downstream, because the snapshot would go on reporting
    the failure it was taken during.
    """
    recs = list(records)
    if not any(getattr(r, "is_usable", False) for r in recs):
        logger.info("snapshot: %s produced nothing — not freezing a failure",
                    patent_id)
        return None
    payload = {
        "patent": patent_id.upper(),
        "journal_head": _journal_head(),
        "compounds": sorted({r.cid for r in recs if r.is_usable and r.cid}),
        "n_records": len(recs),
        "n_usable": sum(1 for r in recs if r.is_usable),
        "records": [_as_row(r) for r in recs if r.is_usable],
    }
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path_for(patent_id).write_text(json.dumps(payload, default=str))
    except OSError as e:
        logger.warning("snapshot: could not freeze %s: %r", patent_id, e)
    return payload


def _as_row(rec) -> dict:
    if is_dataclass(rec) and not isinstance(rec, type):
        return {k: v for k, v in asdict(rec).items() if k in _RECORD_KEYS}
    return {k: getattr(rec, k, None) for k in _RECORD_KEYS}


_RECORD_KEYS = {"cid", "assay_name", "value_numeric", "unit", "qualifier",
                "n_runs", "range_lo", "range_hi", "letter_grade", "table_id",
                "column_header", "value_text", "source"}


def load(patent_id: str) -> dict | None:
    p = path_for(patent_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def thaw(patent_id: str) -> bool:
    """Let this patent be re-derived by current code. The escape hatch."""
    p = path_for(patent_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def compounds(patent_id: str) -> set[str] | None:
    snap = load(patent_id)
    return set(snap["compounds"]) if snap else None


def records_for(patent_id: str, xml: str, *, refresh: bool = False):
    """This patent's records: the frozen answer when there is one.

    The whole point of the module. Callers that want the CURRENT code's opinion
    — the patch verifier, an audit — pass `refresh=True` and get a live parse.
    """
    from ..sources.uspto_assays import extract_from_patent

    if not refresh:
        snap = load(patent_id)
        if snap is not None:
            return snap, True
    return list(extract_from_patent(xml)), False
