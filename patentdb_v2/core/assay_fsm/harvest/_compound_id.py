"""Local re-export of the canonical compound-id sanitizer.

Delegates to `core.compound_id.normalize_cid_key` so HARVEST agrees with
every other writer in the pipeline on what counts as a valid compound-id
and how to spell it. Kept as a thin shim to avoid the routes→core→routes
import cycle that drove the original copy-paste.
"""
from __future__ import annotations

from ...compound_id import normalize_cid_key


def sanitize_compound_id(raw: str) -> str | None:
    """Normalize a compound-ID string to a canonical dict key, or None
    if invalid. Delegates to `core.compound_id.normalize_cid_key`.
    """
    return normalize_cid_key(raw)
