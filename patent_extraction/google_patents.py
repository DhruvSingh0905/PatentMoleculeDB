"""Google Patents clean text extraction — Route 1 in the pipeline.

Scrapes clean patent text directly from Google Patents. Same underlying
USPTO XML data as BigQuery, but free and requires no auth.

This eliminates OCR errors entirely — the text comes from the official
XML, not from PDF image recognition.

Benchmark: 86.2% recall vs 76.9% from OCR markdown (on US10214537).
Cost: $0 (no API calls needed).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import requests

from . import config
from .models import Compound, CompoundSource, IupacSource
from .iupac_to_smiles import rule_based_clean, _try_opsin, _finalize
from .smiles_utils import validate_smiles

logger = logging.getLogger(__name__)

CACHE_DIR = config.OUTPUT_DIR / "gpatents_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_patent_text(patent_id: str) -> dict[str, str] | None:
    """Fetch clean text from Google Patents.

    Args:
        patent_id: e.g., "US10214537"

    Returns:
        Dict with 'description' and 'claims' text, or None if unavailable.
    """
    # Check cache
    cache_file = CACHE_DIR / f"{patent_id}.json"
    if cache_file.exists():
        import json
        data = json.loads(cache_file.read_text())
        logger.info(f"Google Patents cache hit for {patent_id}")
        return data

    url = f"https://patents.google.com/patent/{patent_id}/en"
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200:
            logger.warning(f"Google Patents returned {r.status_code} for {patent_id}")
            return None

        # Extract description
        desc_match = re.search(
            r'<section itemprop="description"[^>]*>(.*?)</section>',
            r.text, re.DOTALL
        )
        claims_match = re.search(
            r'<section itemprop="claims"[^>]*>(.*?)</section>',
            r.text, re.DOTALL
        )

        result = {}
        for key, match in [('description', desc_match), ('claims', claims_match)]:
            if match:
                html = match.group(1)
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text).strip()
                result[key] = text
            else:
                result[key] = ''

        if not result.get('description'):
            logger.warning(f"No description found for {patent_id}")
            return None

        # Cache
        import json
        cache_file.write_text(json.dumps(result))

        logger.info(
            f"Google Patents {patent_id}: "
            f"desc={len(result['description']):,} chars, "
            f"claims={len(result['claims']):,} chars"
        )

        # Be polite — don't hammer Google
        time.sleep(1)

        return result

    except Exception as e:
        logger.error(f"Google Patents fetch failed for {patent_id}: {e}")
        return None


def extract_compounds_from_clean_text(
    patent_id: str,
    text: str,
) -> list[Compound]:
    """Extract compounds from clean patent text using regex + OPSIN.

    No Claude API calls needed — the text is clean enough for regex extraction.

    Args:
        patent_id: Patent identifier.
        text: Clean patent text (description + claims).

    Returns:
        List of Compound objects with validated SMILES.
    """
    compounds = []
    seen = set()

    # Pattern 1: "IUPAC_NAME (NUMBER);" in claims sections
    segments = text.split(';')
    for seg in segments:
        seg = seg.strip()
        m = re.search(r'\((\d{1,4})\)\s*\.?\s*$', seg)
        if not m:
            continue
        num = int(m.group(1))
        name = seg[:m.start()].strip()

        if len(name) < 20 or num in seen:
            continue

        # Must look like a chemical name
        name_lower = name.lower()
        if not any(kw in name_lower for kw in [
            'pyrrolo', 'pyrrol', 'triazin', 'pyrazol', 'piperidin',
            'morpholin', 'piperazin', 'benzamide', 'benzonitrile',
            'phenyl', 'oxazol', 'thiomorpholin', 'pyridin',
            'oxetan', 'azetidin', 'pyrrolidin', 'imidazol',
        ]):
            continue

        seen.add(num)
        compounds.append({'num': num, 'name': name, 'source': 'claims_list'})

    # Pattern 2: "Example N: IUPAC_NAME" in description
    for m in re.finditer(
        r'Example\s+(\d+)[:\s]+([\w\-\(\)\[\],\s]{30,200}?)(?=Example\s+\d+|Intermediate|Table|\.\s)',
        text
    ):
        num = int(m.group(1))
        name = m.group(2).strip()
        if num not in seen and len(name) > 20:
            seen.add(num)
            compounds.append({'num': num, 'name': name, 'source': 'example_section'})

    logger.info(f"Google Patents {patent_id}: found {len(compounds)} compound names in clean text")

    # Convert through OPSIN sequentially (no threading — avoids temp file race)
    validated = []

    for c in compounds:
        name = c['name']

        # Try OPSIN direct
        smiles, error = _try_opsin(name)
        if not smiles:
            cleaned = rule_based_clean(name)
            smiles, error = _try_opsin(cleaned)

        if not smiles or not validate_smiles(smiles) or len(smiles) < 10:
            continue

        # Verify: re-run OPSIN to catch temp file contamination
        verify_smiles, _ = _try_opsin(name)
        if verify_smiles:
            from .smiles_utils import canonicalize_smiles
            c1 = canonicalize_smiles(smiles)
            c2 = canonicalize_smiles(verify_smiles)
            if c1 != c2:
                logger.warning(f"OPSIN verification mismatch for {c['num']}: {c1[:40]} vs {c2[:40]}")
                smiles = verify_smiles  # Use the verified one

        compound = Compound(
            patent_id=patent_id,
            example_number=f"Example {c['num']}",
            iupac_name=name,
            iupac_source=IupacSource.PATENT_VERBATIM,
            source=CompoundSource.EXEMPLIFIED,
            extraction_method=f"google_patents_{c['source']}",
            processing_status="text_done",
        )
        _finalize(compound, smiles, stage="google_patents_opsin")
        if compound.processing_status == "validated":
            validated.append(compound)

    logger.info(f"Google Patents {patent_id}: {len(validated)} compounds validated via OPSIN")
    return validated


def extract_patent_google(patent_id: str) -> list[Compound]:
    """Full Google Patents extraction — fetch + extract + validate.

    This is the primary Route 1 in the pipeline.
    Cost: $0 (no API calls).
    """
    text_data = fetch_patent_text(patent_id)
    if not text_data:
        return []

    full_text = (text_data.get('description', '') + '\n' +
                 text_data.get('claims', ''))

    return extract_compounds_from_clean_text(patent_id, full_text)
