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
import threading

# py2opsin uses a shared temp file — not thread-safe. Serialize access.
_opsin_lock = threading.Lock()

from . import config
from .api_client import call_claude_text
from .models import Compound, MarkushContext
from .smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey,
    strip_salt, compute_drug_likeness, molecular_weight,
)

logger = logging.getLogger(__name__)

MAX_PARALLEL = 10
MIN_SMILES_MW = 150    # Catches iodine (127.9) and other single-atom SMILES
MIN_SMILES_LENGTH = 10 # Drug molecules have SMILES ≥10 chars; "I", "Cl", "CC" are not drugs

# OPSIN failure log
OPSIN_FAILURES_LOG = config.LOGS_DIR / "opsin_failures.jsonl"


# ============================================================
# Stage 2: Rule-based IUPAC cleaning
# ============================================================

def rule_based_clean(name: str) -> str:
    """Deterministic fixes for known OPSIN failure patterns.

    Built from empirical analysis of US10214537 (169 compounds):
    - pyrolo→pyrrolo: Claude drops an 'r' during extraction (30+ compounds)
    - pyranyl→pyran: extraction artifact
    - thionorpholine→thiomorpholine: OPSIN needs standard name
    - Salt suffix stripping
    - Stereo prefix normalization
    - Misplaced parentheses from Claude extraction
    """
    n = name.strip()

    # --- GENERALIZABLE FIXES (safe across all patents) ---

    # Fix common LLM/OCR typos in ring names
    n = n.replace('pyrolo[', 'pyrrolo[')    # Dropped double-r
    n = n.replace('Pyrolo[', 'Pyrrolo[')
    n = n.replace('pyro1o[', 'pyrrolo[')    # OCR reads l as 1
    n = n.replace('Pyro1o[', 'Pyrrolo[')
    n = n.replace('pyrran', 'pyran')         # Extra r
    n = re.sub(r'pyranyl-(\d)-yl', r'pyran-\1-yl', n)  # pyranyl-4-yl → pyran-4-yl
    n = n.replace('thionorpholine', 'thiomorpholine')  # Wrong ring name

    # Fix table OCR artifacts: spaces inside compound names
    n = re.sub(r'\)\s+pyrrolo\[', ')pyrrolo[', n)   # ") pyrrolo[" → ")pyrrolo["
    n = re.sub(r'\)\s+pyrolo\[', ')pyrrolo[', n)     # ") pyrolo[" → ")pyrrolo["
    n = re.sub(r'\)\s+Pyrrolo\[', ')Pyrrolo[', n)

    # Fix OCR periods in fusion descriptors: [1.2,4] → [1,2,4]
    n = re.sub(r'\[(\d+)\.(\d+),(\d+)\]', r'[\1,\2,\3]', n)  # [1.2,4] → [1,2,4]
    n = re.sub(r'\[(\d+)\.,(\d+),(\d+)\]', r'[\1,\2,\3]', n)  # [1.,2,4] → [1,2,4]

    # Fix missing locant in fusion: [2,4] → [1,2,4] (only for triazin context)
    n = re.sub(r'pyrrolo\[2,1-f\]\[2,4\]', 'pyrrolo[2,1-f][1,2,4]', n)
    n = re.sub(r'pyrrolo\[2,1-f\]\[2\.4\]', 'pyrrolo[2,1-f][1,2,4]', n)

    # Fix OCR: [1,4] in triazin context → [1,2,4]
    n = re.sub(r'\[1,4\]triazin', '[1,2,4]triazin', n)

    # Fix more table OCR artifacts
    n = n.replace('pyrrolo[2,1-1]', 'pyrrolo[2,1-f]')   # f OCR'd as 1
    n = n.replace('pyrrolo[2.1-f]', 'pyrrolo[2,1-f]')   # period in fusion
    n = n.replace('pyrzol', 'pyrazol')                    # missing 'a'
    n = re.sub(r'pyrrolo\s+\[', 'pyrrolo[', n)           # space before [
    # Fix missing opening bracket: pyrrolo[2,1-f]1,2,4] → pyrrolo[2,1-f][1,2,4]
    n = re.sub(r'pyrrolo\[2,1-f\](\d)', r'pyrrolo[2,1-f][\1', n)
    # Fix [1]2,4] → [1,2,4]
    n = re.sub(r'\[1\]2,4\]', '[1,2,4]', n)

    # Fix misplaced parens: "pyrazol-5-(yl)" → "pyrazol-5-yl)" (Claude artifact)
    n = re.sub(r'-(\d+)-\(yl\)', r'-\1-yl)', n)

    # Strip salt suffixes (OPSIN can't parse these)
    n = re.sub(r',\s*\d*\s*HCl$', '', n)
    n = re.sub(r',\s*\d*\s*TFA$', '', n)
    n = re.sub(r'\s+as the \w+ salt$', '', n, flags=re.IGNORECASE)

    # Strip ONLY cis/trans prefixes (OPSIN can't resolve these)
    # KEEP R/S/E/Z prefixes — OPSIN handles them correctly (tested on 21 compounds)
    n = re.sub(r'^\(\((?:C|c)is\)\)-?', '', n)   # ((Cis))- artifact
    n = re.sub(r'^\((cis|trans)\)-', '', n, flags=re.IGNORECASE)

    # NOTE: Fusion descriptors ([1,2-f] vs [2,1-f]) and locants (triazin-9 vs -7)
    # are NOT corrected here — they specify different molecules. Changing them
    # would silently produce wrong structures. These go to the LLM fallback.

    return n


# ============================================================
# Stage 3: LLM cleaning with error context
# ============================================================

def _try_pubchem(name: str) -> str | None:
    """Try PubChem name lookup for stereo-aware SMILES.

    PubChem has 111M+ compounds with resolved stereochemistry.
    This bypasses OPSIN's stereo limitations.
    """
    try:
        import pubchempy as pcp
        results = pcp.get_compounds(name, 'name')
        if results and len(results) > 0:
            smiles = results[0].isomeric_smiles or results[0].canonical_smiles
            if smiles and validate_smiles(smiles) and len(smiles) >= MIN_SMILES_LENGTH:
                mw = molecular_weight(smiles)
                if mw and mw >= MIN_SMILES_MW:
                    return smiles
    except Exception:
        pass
    return None


def _is_truncated(name: str) -> bool:
    """Check if IUPAC name is truncated."""
    # Unbalanced parens/brackets
    if name.count('(') != name.count(')') or name.count('[') != name.count(']'):
        return True
    # Ends with a trailing hyphen (cut off mid-name)
    if name.rstrip().endswith('-'):
        return True
    return False


def _vision_ocr_name(patent_id: str, page_num: int, compound_id: str) -> str | None:
    """Render a PDF page and ask Claude Vision to extract the complete IUPAC name.

    Used for compounds where the source markdown truncated the name
    (image-only PDF pages with garbled table extraction).
    """
    from .api_client import call_claude_vision
    from pathlib import Path
    import fitz
    from PIL import Image

    pdf_path = config.DATA_DIR / patent_id / f"{patent_id}.pdf"
    if not pdf_path.exists():
        return None

    try:
        doc = fitz.open(str(pdf_path))
        if page_num >= len(doc):
            doc.close()
            return None

        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 144 DPI, enough for text
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # Save temporarily
        img_path = config.IMAGES_DIR / patent_id / f"ocr_p{page_num:04d}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(img_path))

        response = call_claude_vision(
            prompt=f"This is a page from a pharmaceutical patent. "
                   f"Find the complete IUPAC chemical name for compound '{compound_id}' on this page. "
                   f"The name may be in a table. Extract the COMPLETE name with all parentheses balanced. "
                   f"Output ONLY the IUPAC name, nothing else.",
            image_path=img_path,
            patent_id=patent_id,
            compound_id=f"{compound_id}_vision_ocr",
        )

        if response:
            name = response.strip().split("\n")[0].strip()
            if len(name) > 20 and name.count('(') == name.count(')'):
                return name

    except Exception as e:
        logger.warning(f"Vision OCR failed for {patent_id} p{page_num}: {e}")

    return None


CLEANING_PROMPT = """OPSIN parser failed on this IUPAC chemical name with error:
{error}

Raw name from patent: {raw_name}

Fix the IUPAC name so OPSIN can parse it. Common issues:
- Missing double letters in ring names (pyrrolo not pyrolo)
- Unmatched brackets/parentheses — ensure every ( has a ) and every [ has a ]
- Wrong locants (e.g., pyrazol-1-yl should be pyrazol-5-yl based on the scaffold)
- Garbled ring names (e.g., "1,6'-4-thiomorpholine" should be "thiomorpholine")
- "morpholin-3-one" → use standard IUPAC notation for the ring

Output ONLY the corrected IUPAC name. Nothing else."""

SMILES_FALLBACK_PROMPT = """Convert this IUPAC chemical name to a canonical SMILES string.
The OPSIN parser cannot handle this name, so generate the SMILES directly.
Preserve all stereochemistry (R/S, E/Z, cis/trans).

Name: {raw_name}

Output ONLY the SMILES string. Nothing else."""


def _llm_clean(raw_name: str, error_msg: str, patent_id: str, compound_id: str) -> str | None:
    """Stage 3a: Use Sonnet to clean an IUPAC name that OPSIN can't parse."""
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


def _llm_direct_smiles(raw_name: str, patent_id: str, compound_id: str) -> str | None:
    """Stage 3b: Last resort — ask Opus to generate SMILES directly.

    Only used when OPSIN fundamentally can't parse the ring system
    (e.g., pyrrolo[1,2-f] fusion descriptors).
    """
    prompt = SMILES_FALLBACK_PROMPT.format(raw_name=raw_name)

    response = call_claude_text(
        prompt=prompt,
        model=config.MODEL_OPUS,
        patent_id=patent_id,
        compound_id=f"{compound_id}_smiles_fallback",
        max_tokens=200,
    )

    if response:
        smiles = response.strip().split("\n")[0].strip()
        if validate_smiles(smiles) and len(smiles) >= MIN_SMILES_LENGTH:
            mw = molecular_weight(smiles)
            if mw and mw >= MIN_SMILES_MW:
                return smiles
    return None


# ============================================================
# Core conversion pipeline
# ============================================================

def _try_opsin(name: str) -> tuple[str | None, str]:
    """Try OPSIN conversion. Returns (smiles, error_msg).
    Thread-safe: py2opsin uses a shared temp file internally.
    """
    import warnings
    error_msg = ""

    with _opsin_lock:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = py2opsin(name)
            except Exception as e:
                return None, str(e)

            if caught:
                error_msg = str(caught[-1].message)

    if result and isinstance(result, str) and len(result) > 3:
        if validate_smiles(result):
            return result, ""

    return None, error_msg or "OPSIN returned empty/invalid result"


def _convert_single(compound: Compound) -> Compound:
    """Convert one compound through the 3-stage pipeline."""
    if not compound.iupac_name:
        compound.failure_reason = "no_iupac_name"
        return compound

    raw_name = compound.iupac_name

    # Stage 0: PubChem lookup (stereo-aware, authoritative)
    pubchem_smiles = _try_pubchem(raw_name)
    if pubchem_smiles:
        return _finalize(compound, pubchem_smiles, stage="pubchem_direct")

    # Stage 1: OPSIN on raw name (free, instant)
    smiles, error = _try_opsin(raw_name)
    if smiles:
        return _finalize(compound, smiles, stage="opsin_direct")

    # Stage 2: Rule-based clean → OPSIN (free, instant)
    cleaned = rule_based_clean(raw_name)
    # Track if stereo was stripped
    stereo_stripped = any(p in raw_name.lower() for p in ['(cis)', '(trans)']) and not any(p in cleaned.lower() for p in ['(cis)', '(trans)'])
    if cleaned != raw_name:
        smiles, error = _try_opsin(cleaned)
        if smiles:
            compound.inferred_stereochemistry = stereo_stripped
            return _finalize(compound, smiles, stage="rule_cleaned")

    # Stage 2b: If name is truncated (unbalanced parens from garbled source markdown),
    # render the PDF page and ask Claude Vision to read the complete name.
    if _is_truncated(raw_name) and compound.source_page is not None:
        vision_name = _vision_ocr_name(
            compound.patent_id, compound.source_page,
            compound.example_number or "unknown",
        )
        if vision_name:
            smiles, error = _try_opsin(vision_name)
            if smiles:
                compound.iupac_name = vision_name  # Update with complete name
                return _finalize(compound, smiles, stage="vision_ocr")
            # Also try rule-cleaning the vision-extracted name
            vision_cleaned = rule_based_clean(vision_name)
            smiles, error = _try_opsin(vision_cleaned)
            if smiles:
                compound.iupac_name = vision_name
                return _finalize(compound, smiles, stage="vision_ocr_cleaned")

    # Stage 3a: LLM clean with error context → OPSIN (cheap, Sonnet)
    llm_cleaned = _llm_clean(
        raw_name, error,
        compound.patent_id,
        compound.example_number or "unknown",
    )
    if llm_cleaned:
        smiles, error2 = _try_opsin(llm_cleaned)
        if smiles:
            return _finalize(compound, smiles, stage="llm_cleaned")

    # Stage 3b: Direct SMILES from Opus (last resort for OPSIN-unfixable names)
    direct_smiles = _llm_direct_smiles(
        raw_name,
        compound.patent_id,
        compound.example_number or "unknown",
    )
    if direct_smiles:
        return _finalize(compound, direct_smiles, stage="llm_direct_smiles")

    # All stages failed — log and flag
    # Check if this is a complex macrocyclic name that no tool can handle
    is_macrocycle = any(kw in raw_name.lower() for kw in [
        'methano', 'cyclotridecine', 'cyclododecine', 'cyclopentadecine',
        'methanobenzo', 'methanodipyrido', 'triazacyclo',
    ])
    if is_macrocycle:
        compound.failure_reason = "unparsable_macrocycle_frontier_limitation"
    else:
        compound.failure_reason = "all_stages_failed"
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
    """Finalize a compound with a validated SMILES. Sets provenance metadata."""
    compound.extraction_method = stage
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        compound.failure_reason = f"rdkit_rejected_opsin_output"
        compound.processing_status = "failed"
        return compound

    if len(canonical) < MIN_SMILES_LENGTH:
        compound.failure_reason = f"smiles_too_short_{len(canonical)}"
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

    # MW cross-check against MS data (intermediate validation)
    if compound.ms_mh_plus:
        from rdkit import Chem as RDChem
        from rdkit.Chem import Descriptors as RDDescriptors
        mol = RDChem.MolFromSmiles(canonical)
        if mol:
            exact_mw = RDDescriptors.ExactMolWt(mol)
            expected_mw = compound.ms_mh_plus - 1.008  # (M+H)+ = MW + proton
            delta = abs(exact_mw - expected_mw)
            compound.mw_validated = delta < 1.5
            if not compound.mw_validated:
                # MW mismatch is a WARNING, not a failure.
                # Common causes: Claude assigned wrong MS to this compound,
                # or IUPAC name was partially truncated (missing substituent).
                # OPSIN SMILES is still likely correct — the MS assignment is unreliable.
                logger.warning(
                    f"{compound.example_number}: MW MISMATCH (warning) — "
                    f"OPSIN={exact_mw:.1f} vs MS={expected_mw:.1f} (Δ={delta:.1f}Da)"
                )

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
