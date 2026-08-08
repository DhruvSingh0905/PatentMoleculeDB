"""Chunk-and-verify Strategy 5 runner.

After text_index has produced example_index.json + assay_tables.json
for a patent, this script:

  1. Reads both files
  2. Computes the gap: every cid in assay_tables that lacks a USABLE
     IUPAC + SMILES in example_index
  3. Fires `iupac_burst_targeted` on text windows mentioning each
     missing cid
  4. OPSIN-converts the returned IUPACs to SMILES + InChIKeys
  5. Merges into example_index.json (Strategy 5 source wins)

A cid in example_index counts as "usable" only if it has a non-empty
SMILES — entries with IUPAC but failed OPSIN conversion are still
considered missing, since they don't contribute to InChIKey matching.

Usage:
    python -m patentdb.scripts.eval.targeted_iupac_strategy5 \
        --patent US8952177 --patent US10899738 --patent US10214537
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from patentdb.core.assay_fsm.harvest.iupac_orchestrator import (
    iupac_burst_targeted,
)
from patentdb.core.cost_tracker import cost_tracker
from patentdb.core.iupac_to_smiles import _try_opsin
from patentdb.core.patent_text import load_patent_description
from patentdb.core.smiles_utils import get_inchikey, validate_smiles

logger = logging.getLogger(__name__)


def _compute_missing_cids(
    example_index: dict,
    assay_tables: dict,
) -> list[str]:
    """Return cids that have HARVEST assay rows but no usable IUPAC+SMILES
    in example_index. A cid is "usable" only if its example_index entry
    has a non-empty canonical_smiles (i.e., made it through the OPSIN
    cascade), since that's what InChIKey matching depends on."""
    cids_with_assays: set[str] = set()
    for cid, arr in assay_tables.items():
        if any(a.get("value_numeric") is not None for a in arr):
            cids_with_assays.add(cid)

    cids_with_iupac: set[str] = set()
    for cid, rec in example_index.items():
        if (rec.get("canonical_smiles") or "").strip():
            cids_with_iupac.add(cid)

    return sorted(cids_with_assays - cids_with_iupac, key=lambda s: (len(s), s))


def _merge_pairs_into_index(
    patent_id: str,
    example_index_path: Path,
    new_pairs: dict[str, str],
) -> tuple[int, int]:
    """OPSIN-convert each new pair, then merge into example_index.json.
    Returns (n_added, n_overwrote)."""
    ex = json.loads(example_index_path.read_text())
    n_added = 0
    n_overwrote = 0
    for cid, iupac in new_pairs.items():
        cleaned = re.sub(
            r"^(?:racemic|rac\.?|meso)\s+", "", iupac, flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        smiles, _err = _try_opsin(cleaned)
        if not (smiles and validate_smiles(smiles)):
            logger.warning("OPSIN failed for cid=%s: %r", cid, iupac[:80])
            continue
        rec = {
            "compound_id": f"Cpd. No. {cid}",
            "iupac_name": iupac,
            "canonical_smiles": smiles,
            "inchikey": get_inchikey(smiles),
            "source": "iupac_harvest_targeted",
        }
        # Don't overwrite a working v2 entry whose smiles already resolves
        # — only fill in gaps. If the existing entry has different SMILES
        # but same cid, we still overwrite (Strategy-5-targeted is the
        # highest-rank IUPAC source for cid mappings).
        if cid in ex:
            n_overwrote += 1
        else:
            n_added += 1
        ex[cid] = rec

    example_index_path.write_text(json.dumps(ex, indent=2, ensure_ascii=False))
    return n_added, n_overwrote


def run_for_patent(patent_id: str) -> dict:
    repo_extraction = Path("output_v2/text_extraction") / patent_id
    ex_path = repo_extraction / "example_index.json"
    ay_path = repo_extraction / "assay_tables.json"
    if not (ex_path.exists() and ay_path.exists()):
        logger.warning("missing extraction outputs for %s — skipping", patent_id)
        return {"patent": patent_id, "skipped": True}

    ex = json.loads(ex_path.read_text())
    ay = json.loads(ay_path.read_text())
    missing_cids = _compute_missing_cids(ex, ay)
    logger.info(
        "%s: example_index=%d, assay_tables=%d, missing cids=%d",
        patent_id, len(ex), len(ay), len(missing_cids),
    )
    if not missing_cids:
        return {"patent": patent_id, "missing_before": 0, "added": 0,
                "overwrote": 0, "missing_after": 0}

    text, src_format = load_patent_description(patent_id, prefer_format="auto")
    if not text:
        logger.warning("no text for %s", patent_id)
        return {"patent": patent_id, "skipped": True}

    new_pairs = iupac_burst_targeted(
        patent_id=patent_id,
        text=text,
        missing_cids=missing_cids,
        cost_tracker=cost_tracker,
    )
    # Restrict to cids we actually targeted (LLM might extract bonus pairs
    # we didn't ask for; keep them too — they're free now that the chunk
    # was processed)
    n_added, n_overwrote = _merge_pairs_into_index(patent_id, ex_path, new_pairs)
    # Recount missing after merge
    ex_after = json.loads(ex_path.read_text())
    missing_after = _compute_missing_cids(ex_after, ay)
    return {
        "patent": patent_id,
        "missing_before": len(missing_cids),
        "added": n_added,
        "overwrote": n_overwrote,
        "missing_after": len(missing_after),
    }


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", action="append", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    print(f"Pre-run cost: ${cost_tracker.total_cost:.4f}")
    print()
    print(f"{'patent':<14}  {'missing':>8}  {'added':>6}  {'overwrote':>10}  {'still_missing':>14}")
    print("-" * 60)
    for pid in args.patent:
        r = run_for_patent(pid)
        if r.get("skipped"):
            print(f"{pid:<14}  (skipped)")
            continue
        print(
            f"{pid:<14}  {r['missing_before']:>8}  {r['added']:>6}  "
            f"{r['overwrote']:>10}  {r['missing_after']:>14}"
        )
    print()
    print(f"Post-run cost: ${cost_tracker.total_cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
