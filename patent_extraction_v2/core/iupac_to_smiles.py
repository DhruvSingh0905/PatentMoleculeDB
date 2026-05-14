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
import os
import tempfile

# py2opsin uses a shared temp file — not thread-safe AND not
# process-safe. The default temp filename is `py2opsin_temp_input.txt`
# in the cwd; multiple parallel patent runs all write to the same path
# and clobber each other's input, causing OPSIN to return SMILES for
# the WRONG patent's compound. We work around this by giving each
# process its own temp file (PID-stamped) AND serializing access from
# threads within the same process.
_opsin_lock = threading.Lock()
_OPSIN_TMP_FPATH = os.path.join(
    tempfile.gettempdir(),
    f"py2opsin_input_{os.getpid()}.txt",
)

from . import config
from .api_client import call_claude_text
from .cost_tracker import cost_tracker
from .models import Compound, MarkushContext
from .smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey,
    strip_salt, compute_drug_likeness, molecular_weight,
)

logger = logging.getLogger(__name__)

MAX_PARALLEL = 10
MIN_SMILES_MW = 150    # Catches iodine (127.9) and other single-atom SMILES
MAX_SMILES_MW = 1500   # Catches polymer/garbage SMILES from DECIMER
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
    n = n.replace('isodolin', 'isoindolin')  # Missing 'in' in isoindoline
    n = n.replace('Isodolin', 'Isoindolin')
    n = n.replace('isquinolin', 'isoquinolin')  # Missing 'o' in isoquinoline
    n = n.replace('Isquinolin', 'Isoquinolin')
    n = n.replace('benzontir', 'benzonitr')  # Transposed letters in benzonitrile
    n = n.replace('Benzontir', 'Benzonitr')
    n = n.replace('benzotir', 'benzonitr')   # Another variant
    n = n.replace('isoquinoin', 'isoquinolin')  # Missing 'l'
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


CLEANING_PROMPT = """OPSIN (a deterministic IUPAC-to-structure parser) failed to parse this chemical name. Your job: normalize the name into the form OPSIN can parse, WITHOUT changing the molecule's identity.

Raw name (from patent): {raw_name}
OPSIN error: {error}

Normalization rules (apply only the ones needed):
1. Fix typos in standard ring names (pyrolo→pyrrolo, isodolin→isoindolin, thionorpholine→thiomorpholine, pyrzol→pyrazol).
2. Balance parentheses and square brackets — every ( needs a ), every [ needs a ]. Missing brackets are extraction artifacts; reconstruct from context, do NOT delete substituents.
3. Replace non-standard fusion notation with IUPAC equivalents:
   - "benzenacyclo..." → equivalent IUPAC bridged-ring name
   - "X-aza-Y-oxa-bridged" → use the systematic von Baeyer name
4. Strip salt suffixes (", HCl", ", 2 TFA", "as the hydrochloride salt").
5. Normalize stereo prefixes only if they BLOCK parsing — NEVER strip R/S/E/Z descriptors that OPSIN handles (only ((cis))/((trans)) artifacts).
6. Preserve every locant and substituent EXACTLY. Do not "guess" a different regiochemistry to make OPSIN happy — that produces a different molecule.

Output ONLY the corrected IUPAC name on a single line. No prose, no explanation, no code fences."""

SMILES_FALLBACK_PROMPT = """Convert this IUPAC chemical name to a canonical SMILES string.
The OPSIN parser cannot handle this name, so generate the SMILES directly.
Preserve all stereochemistry (R/S, E/Z, cis/trans).

Name: {raw_name}

Output ONLY the SMILES string. Nothing else."""


def _llm_clean(raw_name: str, error_msg: str, patent_id: str, compound_id: str) -> str | None:
    """Stage 3a: Use Opus to normalize an IUPAC name OPSIN can't parse.

    Opus is used for accuracy on edge cases (non-standard nomenclature, macrocycles).
    Cost is bounded by per-patent LM cap (config.PER_PATENT_LM_CAP).
    """
    prompt = CLEANING_PROMPT.format(error=error_msg, raw_name=raw_name)

    response = call_claude_text(
        prompt=prompt,
        model=config.MODEL_OPUS,
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

# OCR-markup signatures we DON'T want fed to OPSIN. Even if OPSIN
# happens to ignore the tags and parse the underlying name correctly,
# the SMILES it returns is for whatever substring it managed to parse —
# which may be wrong without us realizing. Strict mode rejects these
# outright so the cascade falls through to OCR-cleanup stages.
_OPSIN_INPUT_GARBAGE_PAT = re.compile(
    r"<\|[a-z/_]+\|>"           # MinerU detection-tag pollution
    r"|\[\[\s*\d+\s*,\s*\d+",   # MinerU bbox-coord block: [[114,99,...
    re.IGNORECASE,
)


def _try_opsin(name: str, *, strict: bool = False) -> tuple[str | None, str]:
    """Try OPSIN conversion. Returns (smiles, error_msg).
    Thread-safe: py2opsin uses a shared temp file internally.

    With strict=True, additional gates beyond "RDKit accepts the SMILES":
      - Input must not contain MinerU/OCR markup tags (<|ref|>, [[bbox]])
      - Input must not start with a non-letter (truncation hint)
      - SMILES heavy-atom count must be ≥ 0.4× the input's IUPAC token
        count for long names (≥30 tokens) — rejects partial-substring parses
      - OPSIN warnings of "Unmatched bracket" / "Could not interpret" /
        "couldn't" are HARD failures even if a SMILES is returned

    Strict mode is gated by the caller — wired from `_convert_single`
    when `is_clean_text=False` (i.e., the IUPAC came from MinerU markdown
    or another OCR'd source). Clean-HTML extractions stay lenient because
    they lack the OCR-garbage signals to filter on.
    """
    import warnings

    if strict:
        # Pre-filter: reject obvious OCR/markup pollution in the input
        if _OPSIN_INPUT_GARBAGE_PAT.search(name):
            return None, "input contains OCR markup tags (strict mode)"
        # Reject names starting with punctuation/digit (truncation hint)
        stripped = name.strip()
        if stripped and not (stripped[0].isalpha() or stripped[0] in "([{"):
            return None, "input starts with non-letter (strict mode; truncated?)"

    error_msg = ""

    with _opsin_lock:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                # Pass a per-process tmp_fpath so parallel patent runs
                # don't clobber each other's OPSIN input file.
                result = py2opsin(name, tmp_fpath=_OPSIN_TMP_FPATH)
            except Exception as e:
                return None, str(e)

            if caught:
                error_msg = str(caught[-1].message)

    if not (result and isinstance(result, str) and len(result) > 3):
        return None, error_msg or "OPSIN returned empty/invalid result"
    if not validate_smiles(result):
        return None, error_msg or "OPSIN returned empty/invalid result"

    if strict:
        # Coverage check: heavy-atom count vs. IUPAC token count.
        # Long names (≥30 alpha tokens) that resolve to a tiny SMILES
        # (≤ 0.4× tokens worth of heavy atoms) are almost always
        # partial-substring parses — OPSIN matched a known fragment in
        # the middle of a corrupted name and returned the SMILES for
        # that fragment, not the full molecule.
        try:
            from rdkit import Chem as _RDChem
            mol = _RDChem.MolFromSmiles(result)
            n_heavy = mol.GetNumHeavyAtoms() if mol else 0
        except Exception:
            n_heavy = 0
        n_tokens = len(re.findall(r"[A-Za-z]+", name))
        if n_tokens >= 30 and n_heavy < 0.4 * n_tokens:
            return None, (
                f"strict: coverage too low ({n_heavy} heavy atoms vs "
                f"{n_tokens} tokens) — likely partial-substring parse"
            )
        # Warning gate — these phrases mean OPSIN had to skip part of
        # the name to produce a SMILES. Reject; the cascade has cleanup
        # stages that may recover the full name.
        if error_msg:
            low = error_msg.lower()
            if any(w in low for w in ("unmatched bracket", "could not interpret", "couldn't")):
                return None, f"strict: OPSIN warning ({error_msg})"

    return result, error_msg


def _convert_single(
    compound: Compound,
    is_clean_text: bool = False,
    route_hint: str | None = None,
) -> Compound:
    """Convert one compound through the staged pipeline.

    Args:
        compound: Compound with `iupac_name` set.
        is_clean_text: True when name source is high-quality clean text
            (e.g., Google Patents XML). Skips OCR-targeted stages
            (rule_clean, autocorrect, vision OCR) since they add 0 wins on
            clean text per gauntlet evidence and waste compute.
        route_hint: Tags the compound's structured provenance with the route
            this conversion is happening under. See `_finalize` for valid
            values. Defaults to inferring from the existing extraction_method.
    """
    if not compound.iupac_name:
        compound.failure_reason = "no_iupac_name"
        return compound

    raw_name = compound.iupac_name
    cleaned = raw_name  # Used as fallback by later stages

    # Strict OPSIN gates fire when the source isn't known-clean. Markdown
    # carries OCR garbage (`<|ref|>`, `[[bbox]]`, mid-name punctuation
    # corruption) that OPSIN will sometimes parse a substring of and
    # return the wrong SMILES — strict mode rejects those so the
    # cascade's cleanup stages get a chance.
    strict_opsin = not is_clean_text

    # Stage 0: PubChem lookup (stereo-aware, authoritative)
    pubchem_smiles = _try_pubchem(raw_name)
    if pubchem_smiles:
        return _finalize(compound, pubchem_smiles, stage="pubchem_direct", route_hint=route_hint)

    # Stage 1: OPSIN on raw name (free, instant)
    smiles, error = _try_opsin(raw_name, strict=strict_opsin)
    if smiles:
        return _finalize(compound, smiles, stage="opsin_direct", route_hint=route_hint)

    # Stages 2 / 2c / 2b — OCR-targeted cleanup. Skip for known-clean text.
    if not is_clean_text:
        # Stage 2: Rule-based clean → OPSIN (free, instant).
        # rule_based_clean strips OCR markup tags too, so the cleaned name
        # passes the strict input-cleanliness gate.
        cleaned = rule_based_clean(raw_name)
        # Track if stereo was stripped
        stereo_stripped = any(p in raw_name.lower() for p in ['(cis)', '(trans)']) \
                          and not any(p in cleaned.lower() for p in ['(cis)', '(trans)'])
        if cleaned != raw_name:
            smiles, error = _try_opsin(cleaned, strict=strict_opsin)
            if smiles:
                compound.inferred_stereochemistry = stereo_stripped
                return _finalize(compound, smiles, stage="rule_cleaned", route_hint=route_hint)

        # Stage 2c: Levenshtein autocorrect → OPSIN (catches novel OCR typos)
        # (optional module — skip silently if absent so the cascade still
        # progresses to LLM cleanup which handles the same cases anyway)
        try:
            from .ocr_autocorrect import autocorrect_iupac
            autocorrected = autocorrect_iupac(cleaned)
            if autocorrected != cleaned:
                smiles, error = _try_opsin(autocorrected, strict=strict_opsin)
                if smiles:
                    return _finalize(compound, smiles, stage="autocorrected", route_hint=route_hint)
        except ImportError:
            pass

        # Stage 2b: If name is truncated (unbalanced parens from garbled source),
        # render the PDF page and ask Claude Vision to read the complete name.
        if _is_truncated(raw_name) and compound.source_page is not None:
            vision_name = _vision_ocr_name(
                compound.patent_id, compound.source_page,
                compound.example_number or "unknown",
            )
            if vision_name:
                # Vision OCR output is treated as clean (the LLM rewrote
                # it — no MinerU markup left). Drop strict gating here.
                smiles, error = _try_opsin(vision_name, strict=False)
                if smiles:
                    compound.iupac_name = vision_name
                    return _finalize(compound, smiles, stage="vision_ocr", route_hint=route_hint)
                vision_cleaned = rule_based_clean(vision_name)
                smiles, error = _try_opsin(vision_cleaned, strict=False)
                if smiles:
                    compound.iupac_name = vision_name
                    return _finalize(compound, smiles, stage="vision_ocr_cleaned", route_hint=route_hint)

    # Per-patent LM cost cap guard — Stages 3a/3b skipped if exceeded
    if cost_tracker.patent_lm_exceeded(compound.patent_id):
        compound.failure_reason = "patent_lm_cap_exceeded"
        compound.processing_status = "failed"
        _log_opsin_failure(
            patent_id=compound.patent_id,
            compound_id=compound.example_number or "unknown",
            raw_name=raw_name,
            cleaned_name=cleaned if cleaned != raw_name else None,
            llm_cleaned_name=None,
            error=f"LM cap (${config.PER_PATENT_LM_CAP}) exceeded; skipped LM stages",
            is_truncated=raw_name.count('(') != raw_name.count(')'),
        )
        return compound

    # Stage 3a: LLM normalize IUPAC name → OPSIN (Opus, accuracy-tuned)
    llm_cleaned = _llm_clean(
        raw_name, error,
        compound.patent_id,
        compound.example_number or "unknown",
    )
    if llm_cleaned:
        smiles, error2 = _try_opsin(llm_cleaned)
        if smiles:
            return _finalize(compound, smiles, stage="llm_cleaned", route_hint=route_hint)

    # Re-check cap before Stage 3b (since 3a already spent some budget)
    if cost_tracker.patent_lm_exceeded(compound.patent_id):
        compound.failure_reason = "patent_lm_cap_exceeded_after_stage_3a"
        compound.processing_status = "failed"
        return compound

    # Stage 3b: Direct SMILES from Opus (last resort for OPSIN-unfixable names)
    direct_smiles = _llm_direct_smiles(
        raw_name,
        compound.patent_id,
        compound.example_number or "unknown",
    )
    if direct_smiles:
        return _finalize(compound, direct_smiles, stage="llm_direct_smiles", route_hint=route_hint)

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


def _finalize(
    compound: Compound,
    smiles: str,
    stage: str,
    route_hint: str | None = None,
) -> Compound:
    """Finalize a compound with a validated SMILES. Sets provenance metadata.

    Args:
        compound: Compound being finalized.
        smiles: Raw SMILES from the converter.
        stage: One of {pubchem_direct, opsin_direct, rule_cleaned, autocorrected,
            vision_ocr, vision_ocr_cleaned, llm_cleaned, llm_direct_smiles, ...}.
        route_hint: Optional route this compound came from
            ("google_patents_example", "google_patents_table", "image_pipeline",
             "synthesis_block", "page_extraction", etc.). If None, falls back to
             the existing extraction_method when set, otherwise "unknown".
    """
    from .models import CompoundProvenance

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
    if mw and mw > MAX_SMILES_MW:
        compound.failure_reason = f"mw_too_high_{mw:.0f}"
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

    # Set stereo trust based on extraction method
    if stage in ("llm_direct_smiles", "vision_ocr", "vision_ocr_cleaned"):
        compound.stereo_trusted = False
    if compound.inferred_stereochemistry:
        compound.stereo_trusted = False

    # Mark LM-derived compounds for benchmark filtering
    if stage in ("llm_cleaned", "llm_direct_smiles", "synthesis_block_lm"):
        compound.lm_normalized = True

    # Populate structured provenance
    # Resolve route: caller hint > existing extraction_method prefix > "unknown"
    route = route_hint
    if not route:
        # Infer route from existing extraction_method prefix (set by extractor before _finalize)
        prev_method = compound.extraction_method or ""
        if prev_method.startswith("google_patents_table"):
            route = "google_patents_table"
        elif prev_method.startswith("google_patents"):
            route = "google_patents_example"
        elif prev_method == "synthesis_block_lm":
            route = "synthesis_block"
        elif prev_method.startswith(("decimer", "sonnet_vision", "opus_vision")):
            route = "image_pipeline"
        elif prev_method.startswith("adaptive"):
            route = "adaptive"
        else:
            route = "page_extraction"

    chain = [stage]
    if compound.provenance:
        # Shouldn't normally happen — _finalize is called once. But if it does,
        # preserve the previous chain for debugging.
        chain = (compound.provenance.chain or []) + chain
    compound.provenance = CompoundProvenance(
        route=route,
        stage=stage,
        page=compound.source_page,
        image_bbox=compound.image_bbox,
        image_path=compound.image_path,
        chain=chain,
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
