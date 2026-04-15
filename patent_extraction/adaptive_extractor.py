"""Adaptive compound extractor — discovers patent format then extracts.

Instead of hardcoded regex per patent format, this module:
1. Samples the first few thousand chars of patent text
2. Uses a small Claude call to identify the compound naming pattern
3. Generates extraction logic adapted to that specific patent
4. Extracts all compounds using the discovered pattern

Cost: ~$0.01 per patent (one small Sonnet call to discover format).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import config
from .api_client import call_claude_text
from .iupac_to_smiles import _try_opsin, rule_based_clean
from .models import Compound, CompoundSource, IupacSource
from .smiles_utils import validate_smiles, canonicalize_smiles

logger = logging.getLogger(__name__)

DISCOVER_PROMPT = """Analyze this patent text and identify how compound examples are formatted.

Look for patterns like:
- "Example 1: IUPAC_NAME" or "Example 1 IUPAC_NAME"
- "Cpd. No. 1 ... Chemical Name" in tables
- Semicolon-separated compound lists in claims
- Numbered compounds with synthesis descriptions

For the FIRST 3 examples you find, extract:
1. The compound identifier (e.g., "Example 1", "Cpd. No. 148")
2. The IUPAC chemical name

Return ONLY valid JSON:
{
  "format_type": "example_colon" | "example_space" | "cpd_table" | "claims_list" | "other",
  "separator": "the text pattern between compound ID and name",
  "compounds": [
    {"id": "1", "name": "full IUPAC name here"},
    {"id": "2", "name": "full IUPAC name here"},
    {"id": "3", "name": "full IUPAC name here"}
  ]
}

Patent text sample:
"""

BATCH_EXTRACT_PROMPT = """Extract ALL compound examples from this patent text section.

For each compound, extract:
1. The example number (just the integer)
2. The FULL IUPAC chemical name (the final product name, NOT intermediates)

Rules:
- Only extract the FINAL PRODUCT name for each Example, not intermediate steps
- The IUPAC name is usually the first chemical name after "Example N"
- Skip intermediates labeled (i), (ii), (iii) etc.
- Copy names EXACTLY as written
- If an Example only describes synthesis without naming the final product, skip it

Return ONLY a JSON array:
[{"id": "1", "name": "full IUPAC name"}, {"id": "2", "name": "full IUPAC name"}, ...]

Patent text:
"""


def _discover_format(patent_id: str, text: str) -> dict | None:
    """Use a small Claude call to discover this patent's compound format."""
    # Sample: first occurrence of "Example" + surrounding context
    sample_start = text.find('Example 1')
    if sample_start < 0:
        sample_start = text.find('Cpd')
    if sample_start < 0:
        sample_start = max(0, len(text) // 4)  # Middle of text

    sample = text[max(0, sample_start - 200):sample_start + 3000]

    response = call_claude_text(
        prompt=DISCOVER_PROMPT + sample[:3000],
        model=config.MODEL_SONNET,
        patent_id=patent_id,
        compound_id="format_discovery",
        max_tokens=500,
    )

    if not response:
        return None

    try:
        cleaned = response.strip()
        if '```' in cleaned:
            m = re.search(r'```(?:json)?\s*(.*?)```', cleaned, re.DOTALL)
            cleaned = m.group(1) if m else cleaned
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Format discovery JSON parse failed for {patent_id}")
        return None


def _extract_with_discovered_format(
    patent_id: str,
    text: str,
    format_info: dict,
) -> list[dict]:
    """Extract compounds using the discovered format pattern."""
    compounds = []
    seen = set()
    fmt = format_info.get("format_type", "other")

    # Use the discovered examples to build a regex pattern
    sample_compounds = format_info.get("compounds", [])
    if not sample_compounds:
        return []

    # Figure out the pattern from examples
    first_id = sample_compounds[0].get("id", "")
    first_name = sample_compounds[0].get("name", "")

    if fmt == "example_colon":
        # Pattern: "Example N: IUPAC_NAME"
        pattern = re.compile(
            r'Example\s+(\d+)\s*[:.]\s*'
            r'([A-Za-z0-9\-\(\)\[\],\s\'\u2019]{20,300}?)'
            r'(?=\s+Example\s+\d|\s+Intermediate|\s+Step|\s+\(\w+\)\s+[A-Z]|\.\s+[A-Z][a-z])',
            re.DOTALL
        )
    elif fmt == "example_space":
        # Pattern: "Example N IUPAC_NAME" (no colon)
        pattern = re.compile(
            r'Example\s+(\d+)\s+'
            r'([A-Za-z0-9\-\(\)\[\],\s]{20,300}?)'
            r'(?=\s+Example\s+\d|\s+\(\w+\)\s+[A-Z]|\.\s+[A-Z][a-z])',
            re.DOTALL
        )
    elif fmt == "cpd_table":
        # Use google_patents_tables.py logic (already handles this)
        return []
    elif fmt == "claims_list":
        # Semicolon-separated
        segments = text.split(';')
        for seg in segments:
            seg = seg.strip()
            m = re.search(r'\((\d{1,4})\)\s*\.?\s*$', seg)
            if not m:
                continue
            num = m.group(1)
            name = seg[:m.start()].strip()
            if len(name) > 20 and num not in seen:
                name_lower = name.lower()
                if any(kw in name_lower for kw in [
                    'yl', 'one', 'amine', 'amide', 'nitrile', 'phenyl',
                    'pyrrolo', 'triazin', 'pyrazol', 'piperidin', 'morpholin',
                ]):
                    seen.add(num)
                    compounds.append({'num': num, 'name': name})
        return compounds
    else:
        # Generic: try both colon and space patterns
        pattern = re.compile(
            r'Example\s+(\d+)\s*[:\s]+'
            r'([A-Za-z0-9\-\(\)\[\],\s]{20,300}?)'
            r'(?=\s+Example\s+\d|\s+Intermediate|\s+\(\w+\)\s+[A-Z])',
            re.DOTALL
        )

    for m in pattern.finditer(text):
        num = m.group(1)
        name = m.group(2).strip()
        # Clean line-break hyphens
        name = re.sub(r'- (\w)', r'-\1', name)
        name = re.sub(r'\s+', ' ', name)
        if num not in seen and len(name) > 20:
            seen.add(num)
            compounds.append({'num': num, 'name': name})

    return compounds


def _batch_extract_with_claude(patent_id: str, text: str) -> list[dict]:
    """Use Claude to extract ALL compounds from text in chunks.

    For patents where regex fails (complex formats, interleaved synthesis steps),
    Claude reads the text and finds compound names directly.
    Cost: ~$0.50-2.00 per patent (depends on text length).
    """
    compounds = []
    seen = set()

    # Find the examples section
    examples_start = text.find('EXAMPLES')
    if examples_start < 0:
        examples_start = text.find('Examples')
    if examples_start < 0:
        examples_start = text.find('Example 1')
    if examples_start < 0:
        examples_start = 0

    examples_text = text[examples_start:]

    # Process in 8K char chunks (fits in Sonnet context with room)
    chunk_size = 8000
    for i in range(0, min(len(examples_text), 100000), chunk_size):
        chunk = examples_text[i:i + chunk_size]

        # Skip chunks without example markers
        if not re.search(r'Example\s+\d+', chunk, re.IGNORECASE):
            continue

        response = call_claude_text(
            prompt=BATCH_EXTRACT_PROMPT + chunk,
            model=config.MODEL_SONNET,
            patent_id=patent_id,
            compound_id=f"adaptive_batch_{i // chunk_size}",
            max_tokens=2000,
        )

        if not response:
            continue

        try:
            cleaned = response.strip()
            if '```' in cleaned:
                m = re.search(r'```(?:json)?\s*(.*?)```', cleaned, re.DOTALL)
                cleaned = m.group(1) if m else cleaned
            # Find JSON array
            start = cleaned.find('[')
            end = cleaned.rfind(']')
            if start >= 0 and end > start:
                entries = json.loads(cleaned[start:end + 1])
                for entry in entries:
                    num = str(entry.get('id', ''))
                    name = entry.get('name', '')
                    if num and name and len(name) > 15 and num not in seen:
                        seen.add(num)
                        compounds.append({'num': num, 'name': name})
        except json.JSONDecodeError:
            continue

    logger.info(f"Adaptive batch extraction {patent_id}: {len(compounds)} compounds from {min(len(examples_text), 100000) // chunk_size + 1} chunks")
    return compounds


def extract_compounds_adaptive(patent_id: str) -> list[Compound]:
    """Adaptive extraction: discover format, then extract all compounds.

    Strategy:
    1. Try regex-based extraction first (free)
    2. If regex finds <20 compounds, use Claude batch extraction (~$0.50-2.00)

    Works on any patent format without hardcoded regex patterns.
    """
    cache_path = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
    if not cache_path.exists():
        logger.info(f"Adaptive extractor: no GP cache for {patent_id}")
        return []

    with open(cache_path) as f:
        gp_data = json.load(f)

    text = gp_data.get('description', '') + '\n' + gp_data.get('claims', '')
    if len(text) < 1000:
        return []

    # Step 1: Discover format (~$0.01)
    format_info = _discover_format(patent_id, text)
    if not format_info:
        logger.warning(f"Adaptive extractor: format discovery failed for {patent_id}")
        return []

    fmt = format_info.get("format_type", "unknown")
    logger.info(f"Adaptive extractor {patent_id}: discovered format '{fmt}'")

    # Step 2: Try regex-based extraction first (free)
    raw_compounds = _extract_with_discovered_format(patent_id, text, format_info)

    # Step 3: Add Claude-discovered examples
    for c in format_info.get("compounds", []):
        num = str(c.get("id", ""))
        name = c.get("name", "")
        if num and name and len(name) > 15 and num not in {r['num'] for r in raw_compounds}:
            raw_compounds.append({'num': num, 'name': name})

    logger.info(f"Adaptive extractor {patent_id}: regex found {len(raw_compounds)} compound names")

    # Step 4: If regex found <20, use Claude batch extraction (generalizable, not patent-specific)
    if len(raw_compounds) < 20:
        batch = _batch_extract_with_claude(patent_id, text)
        seen_nums = {r['num'] for r in raw_compounds}
        for entry in batch:
            if entry['num'] not in seen_nums:
                raw_compounds.append(entry)
                seen_nums.add(entry['num'])

    # Step 4: Convert through OPSIN (free)
    validated = []
    for entry in raw_compounds:
        name = entry['name']
        smiles, _ = _try_opsin(name)
        if not smiles:
            cleaned = rule_based_clean(name)
            smiles, _ = _try_opsin(cleaned)

        if not smiles or not validate_smiles(smiles) or len(smiles) < 10:
            continue

        compound = Compound(
            patent_id=patent_id,
            example_number=f"Example {entry['num']}",
            iupac_name=name,
            iupac_source=IupacSource.PATENT_VERBATIM,
            source=CompoundSource.EXEMPLIFIED,
            extraction_method="adaptive_gp_opsin",
            processing_status="text_done",
        )
        from .iupac_to_smiles import _finalize
        _finalize(compound, smiles, stage="adaptive_gp_opsin")
        if compound.processing_status == "validated":
            validated.append(compound)

    logger.info(f"Adaptive extractor {patent_id}: {len(validated)} validated via OPSIN")
    return validated
