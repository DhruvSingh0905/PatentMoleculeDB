"""Step 2.2 — IUPAC → SMILES via 3-stage fault-tolerant pipeline.

Stage 1: OPSIN direct (free, deterministic, guaranteed correct if succeeds)
Stage 2: Rule-based cleaning → OPSIN retry (free)
Stage 3: Sonnet cleaning with error context → OPSIN retry (cheap)

No LLM-generated SMILES. All SMILES come from OPSIN (deterministic compiler).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from py2opsin import py2opsin

from . import config
from .api_client import call_claude_text
from .models import Compound, MarkushContext
from .smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey,
    strip_salt, compute_drug_likeness, molecular_weight,
)

logger = logging.getLogger(__name__)

MAX_PARALLEL = 10
MIN_SMILES_MW = 100  # Lower than before — OPSIN output is guaranteed correct

# OPSIN failure log
OPSIN_FAILURES_LOG = config.LOGS_DIR / "opsin_failures.jsonl"


# ============================================================
# Stage 2: Rule-based IUPAC cleaning
# ============================================================

def rule_based_clean(name: str) -> str:
    """Deterministic fixes for known OPSIN failure patterns.

    Based on empirical analysis of US10214537 extraction failures:
    - pyrolo→pyrrolo (Claude drops an 'r' during extraction)
    - pyranyl→pyran (extraction artifact)
    - Salt suffix stripping
    - Stereo prefix normalization
    """
    n = name.strip()

    # Fix known extraction typos
    n = n.replace('pyrolo[', 'pyrrolo[')
    n = n.replace('Pyrolo[', 'Pyrrolo[')
    n = n.replace('pyrran', 'pyran')
    n = n.replace('pyranyl-', 'pyran-')

    # Strip salt suffixes (OPSIN can't parse these)
    n = re.sub(r',\s*\d*\s*HCl$', '', n)
    n = re.sub(r',\s*\d*\s*TFA$', '', n)
    n = re.sub(r'\s+as the \w+ salt$', '', n, flags=re.IGNORECASE)

    # Normalize stereochemistry prefixes
    n = re.sub(r'^\(\((?:C|c)is\)\)-?', '', n)   # ((Cis))- → strip
    n = re.sub(r'^\((cis|trans)\)-', '', n, flags=re.IGNORECASE)
    n = re.sub(r'^\((\d+[RS])\)-', '', n)         # (3R)- → strip

    return n


# ============================================================
# Stage 3: LLM cleaning with error context
# ============================================================

CLEANING_PROMPT = """OPSIN parser failed on this IUPAC chemical name with error:
{error}

Raw name from patent: {raw_name}

Fix the IUPAC name so OPSIN can parse it. Common issues:
- Missing double letters in ring names (pyrrolo not pyrolo)
- Unmatched brackets/parentheses
- OCR artifacts (l vs 1, O vs 0)
- Salt suffixes that need stripping

Output ONLY the corrected IUPAC name. Nothing else."""


def _llm_clean(raw_name: str, error_msg: str, patent_id: str, compound_id: str) -> str | None:
    """Use Sonnet to clean an IUPAC name that OPSIN can't parse."""
    prompt = CLEANING_PROMPT.format(error=error_msg, raw_name=raw_name)

    response = call_claude_text(
        prompt=prompt,
        model=config.MODEL_SONNET,
        patent_id=patent_id,
        compound_id=f"{compound_id}_clean",
        max_tokens=300,
    )

    if response:
        return response.strip().split("\n")[0].strip()
    return None


# ============================================================
# Core conversion pipeline
# ============================================================

def _try_opsin(name: str) -> tuple[str | None, str]:
    """Try OPSIN conversion. Returns (smiles, error_msg)."""
    import warnings
    error_msg = ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = py2opsin(name)
        except Exception as e:
            return None, str(e)

        # Capture OPSIN warnings as error messages
        if caught:
            error_msg = str(caught[-1].message)

    if result and isinstance(result, str) and len(result) > 3:
        # Validate with RDKit as double-check
        if validate_smiles(result):
            return result, ""

    return None, error_msg or "OPSIN returned empty/invalid result"


def _convert_single(compound: Compound) -> Compound:
    """Convert one compound through the 3-stage pipeline."""
    if not compound.iupac_name:
        compound.failure_reason = "no_iupac_name"
        return compound

    raw_name = compound.iupac_name

    # Stage 1: OPSIN on raw name (free, instant)
    smiles, error = _try_opsin(raw_name)
    if smiles:
        return _finalize(compound, smiles, stage="opsin_direct")

    # Stage 2: Rule-based clean → OPSIN (free, instant)
    cleaned = rule_based_clean(raw_name)
    if cleaned != raw_name:
        smiles, error = _try_opsin(cleaned)
        if smiles:
            return _finalize(compound, smiles, stage="rule_cleaned")

    # Stage 3: LLM clean with error context → OPSIN (cheap)
    llm_cleaned = _llm_clean(
        raw_name, error,
        compound.patent_id,
        compound.example_number or "unknown",
    )
    if llm_cleaned:
        smiles, error2 = _try_opsin(llm_cleaned)
        if smiles:
            return _finalize(compound, smiles, stage="llm_cleaned")

    # All stages failed — log and flag
    compound.failure_reason = "opsin_all_stages_failed"
    compound.processing_status = "failed"

    _log_opsin_failure(
        patent_id=compound.patent_id,
        compound_id=compound.example_number or "unknown",
        raw_name=raw_name,
        cleaned_name=cleaned if cleaned != raw_name else None,
        llm_cleaned_name=llm_cleaned,
        error=error,
        is_truncated=raw_name.count('(') != raw_name.count(')'),
    )

    return compound


def _finalize(compound: Compound, smiles: str, stage: str) -> Compound:
    """Finalize a compound with an OPSIN-generated SMILES."""
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        compound.failure_reason = f"rdkit_rejected_opsin_output"
        compound.processing_status = "failed"
        return compound

    mw = molecular_weight(canonical)
    if mw and mw < MIN_SMILES_MW:
        compound.failure_reason = f"mw_too_low_{mw:.0f}"
        compound.processing_status = "failed"
        return compound

    compound.smiles_from_text = smiles
    compound.canonical_smiles = canonical
    compound.inchikey = get_inchikey(canonical)

    salt_result = strip_salt(canonical)
    compound.parent_smiles = salt_result.get("parent_smiles")
    compound.parent_inchikey = salt_result.get("parent_inchikey")

    compound.drug_likeness = compute_drug_likeness(canonical)
    compound.processing_status = "validated"
    return compound


def _log_opsin_failure(
    patent_id: str, compound_id: str, raw_name: str,
    cleaned_name: str | None, llm_cleaned_name: str | None,
    error: str, is_truncated: bool,
):
    """Log OPSIN failure for analysis and iteration."""
    # Categorize
    if is_truncated:
        category = "truncated_name"
    elif 'pyrolo' in raw_name.lower():
        category = "fused_ring_typo"
    elif re.search(r',\s*\d*\s*HCl', raw_name):
        category = "salt_suffix"
    elif re.search(r'^\(?(cis|trans)', raw_name, re.IGNORECASE):
        category = "stereochem"
    else:
        category = "other"

    entry = {
        "patent_id": patent_id,
        "compound_id": compound_id,
        "raw_iupac": raw_name[:200],
        "cleaned_iupac": cleaned_name[:200] if cleaned_name else None,
        "llm_cleaned": llm_cleaned_name[:200] if llm_cleaned_name else None,
        "opsin_error": error[:200],
        "failure_category": category,
        "is_truncated": is_truncated,
    }

    with open(OPSIN_FAILURES_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================
# Public API
# ============================================================

# Backward-compatible single conversion (used by tests)
def convert_iupac_to_smiles(
    compound: Compound,
    markush_context: MarkushContext | None = None,
) -> Compound:
    return _convert_single(compound)

# Keep the name for pipeline.py
finalize_compound = _finalize


def convert_batch(
    compounds: list[Compound],
    markush_context: MarkushContext | None = None,
) -> list[Compound]:
    """Convert all compounds via 3-stage pipeline in parallel."""
    to_convert = [c for c in compounds if c.iupac_name]
    if not to_convert:
        return compounds

    patent_id = compounds[0].patent_id if compounds else "unknown"
    logger.info(f"Patent {patent_id}: Converting {len(to_convert)} IUPAC names (OPSIN + rules + LLM fallback)")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(_convert_single, c): c for c in to_convert}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                c = futures[future]
                logger.error(f"Conversion error for {c.example_number}: {e}")
                c.failure_reason = str(e)
                c.processing_status = "failed"

    # Stats
    validated = sum(1 for c in to_convert if c.processing_status == "validated")
    failed = sum(1 for c in to_convert if c.processing_status == "failed")
    logger.info(f"Patent {patent_id}: IUPAC→SMILES complete. Validated: {validated}, Failed: {failed}")

    return compounds
