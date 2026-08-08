"""Harvest compound IDs from MS-data-only references in GP text.

Why this exists
---------------
On compound-table patents (US10899738 e.g.), some compounds are
referenced ONLY via mass-spec data without their IUPAC names being
present in the prose:

    Cpd. No. 55; MS (ESI) m/z: [M+H] + calcd, 468.3; found, 469.5.
    Cpd. No. 81; MS (ESI) m/z 467.3 [M+H] + .

These compounds are real (BindingDB has them, the patent uses them),
but `extract_patent_google` and `extract_table_compounds_from_gp`
both miss them because there's no IUPAC name to attach. This causes
text-route coverage to be artificially low.

This harvester scans the GP text for these patterns and registers
STUB entries in `example_index` with:
  - compound_id: extracted from the regex
  - iupac_name: None
  - canonical_smiles: None
  - is_structure_only: True   ← flag indicating this is a known
                                  text-route gap (compound exists in
                                  the patent's MS section but the
                                  IUPAC name lives elsewhere — likely
                                  a structure drawing in a table or
                                  a Markush template)
  - ms_mh_plus_calcd / ms_mh_plus_found: optional MW values from MS

Downstream consumers (Markush enumerator, image-route reconciler) can
fill in the SMILES later. The stub is enough to:
  (a) include the compound_id in coverage metrics (cov% goes up)
  (b) signal "this is NOT a parser failure — IUPAC just isn't here"
  (c) provide MS m/z as a downstream verification signal once SMILES
      arrives from another route

Patent-agnostic — same regex on every patent. No hardcoded compound
IDs or patent IDs.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..core import config

logger = logging.getLogger(__name__)


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass
class MSDataStub:
    """One MS-data-only compound reference, ready to merge into example_index."""
    compound_id: str
    ms_mh_plus_calcd: float | None = None
    ms_mh_plus_found: float | None = None
    source_offset: int | None = None    # char offset in GP text


# ── Regex patterns ──────────────────────────────────────────────────


# Common MS-data shapes seen across patents. We accept either form
# 'Cpd. No. 55; MS (ESI) m/z: [M+H] + calcd, 468.3; found, 469.5.'
# or
# 'Cpd. No. 81; MS (ESI) m/z 467.3 [M+H] + .'
_MS_REF_RE = re.compile(
    r"""
    \b(?:Cpd\.?\s*No\.?|Compound|Example|Cpd|Ex\.?)\s*
    (\d{1,4}[A-Za-z]{0,3})        # compound id (group 1)
    \s*[;:.]?\s*
    (?:MS|HRMS|LC[/-]?MS|HPLC[/-]?MS)\s*       # MS keyword
    [\s\S]{0,200}?                  # arbitrary noise (m/z, units, etc.)
    \[?\s*M\s*\+\s*H\s*\]?[\s+]?    # [M+H]+ or M+H or M+H+
    [\s\S]{0,80}?                   # text up to value
    (?:                             # one of two value patterns:
        calcd?\.?,?\s*(\d+(?:\.\d+)?)   # group 2: calcd value
        [\s;,]+(?:found|obsd?\.?)\s*[:,]?\s*(\d+(?:\.\d+)?)   # group 3: found value
        |
        (\d+(?:\.\d+)?)                  # group 4: bare value (no calcd/found split)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _parse_float(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def harvest_ms_stubs(patent_id: str) -> list[MSDataStub]:
    """Scan the GP cache for MS-data-only compound references and
    return one MSDataStub per unique compound_id found.

    De-dups by compound_id (keeps first occurrence's MS values).
    """
    cache_path = config.OUTPUT_DIR.parent / "output" / "gpatents_cache" / f"{patent_id}.json"
    if not cache_path.exists():
        return []
    try:
        gp = json.loads(cache_path.read_text())
    except Exception as e:
        logger.warning(f"GP cache hydrate failed for {patent_id}: {e!r}")
        return []
    text = gp.get("description", "") or ""
    if not text:
        return []

    stubs_by_id: dict[str, MSDataStub] = {}
    for m in _MS_REF_RE.finditer(text):
        cid = m.group(1).strip()
        if cid in stubs_by_id:
            continue
        calcd = _parse_float(m.group(2))
        found = _parse_float(m.group(3))
        bare = _parse_float(m.group(4))
        # If only the bare form matched, treat as "found" (observed value)
        if bare is not None and calcd is None and found is None:
            found = bare
        stubs_by_id[cid] = MSDataStub(
            compound_id=cid,
            ms_mh_plus_calcd=calcd,
            ms_mh_plus_found=found,
            source_offset=m.start(),
        )
    return list(stubs_by_id.values())


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", action="append", required=True)
    args = ap.parse_args(argv)

    print(f"{'patent':>14}  {'#stubs':>7}  {'sample stubs (cpd_id, M+H found):':<50}")
    print("-" * 80)
    for pat in args.patent:
        stubs = harvest_ms_stubs(pat)
        sample = ", ".join(
            f"{s.compound_id}={s.ms_mh_plus_found}"
            for s in stubs[:5]
        )
        print(f"{pat:>14}  {len(stubs):>7}  {sample}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
