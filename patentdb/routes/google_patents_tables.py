"""Extract compounds from Google Patents clean text for table-format patents.

For patents like US10899738 where compounds are defined in tables
(Cpd.No. | Structure | Chemical Name), the Google Patents clean text
has the names without OCR errors.

Parses the flat text by finding compound numbers followed by IUPAC names.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..core import config
from ..core.iupac_to_smiles import _try_opsin, rule_based_clean, _finalize
from ..core.models import Compound, CompoundSource, IupacSource
from ..core.smiles_utils import validate_smiles

logger = logging.getLogger(__name__)


def extract_table_compounds_from_gp(patent_id: str) -> list[Compound]:
    """Extract compounds from Google Patents clean text for table-format patents.

    Parses flat number sequences where compound entries appear as:
    Cpd.No. [garbled structure] Chemical Name Cpd.No. ...

    Returns validated Compound objects with SMILES from OPSIN.
    """
    cache_path = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
    if not cache_path.exists():
        return []

    with open(cache_path) as f:
        text = json.load(f).get('description', '')
    if not text:
        return []

    # Find TABLE sections with compound entries
    table_start = text.find('TABLE 1')
    if table_start < 0:
        table_start = text.find('Table 1')
    if table_start < 0:
        return []

    # Skip substituent/fragment tables (Ar/R1 columns — not full compound names)
    table_header = text[table_start:table_start + 300]
    if re.search(r'\bAr\b.*\bR[12]\b|\bR[12]\b.*\bAr\b', table_header):
        logger.info(f"Google Patents tables {patent_id}: skipping substituent table (Ar/R1 format)")
        return []

    # Find end of compound tables (next major section)
    table_text = text[table_start:]

    # Tokenize
    tokens = re.split(r'\s+', table_text)

    # Parse: compound_number token... chemical_name compound_number ...
    # Compound numbers are standalone integers 1-999
    # Names contain chemical fragments (yl, one, amine, etc.)
    entries = {}
    current_cpd = None
    current_parts = []

    for tok in tokens:
        if re.match(r'^\d{1,3}$', tok) and 1 <= int(tok) <= 999:
            if current_cpd and current_parts:
                name = ' '.join(current_parts)
                name = re.sub(r'- (\w)', r'-\1', name)  # Fix line-break hyphens
                name = re.sub(r'\s+', ' ', name).strip()
                if current_cpd not in entries:
                    entries[current_cpd] = name
            current_cpd = tok
            current_parts = []
        else:
            if len(tok) > 2 or tok in ('N-', '1-', '2-', '3-', '4-', '5-', '(3H)-', '(2H)-'):
                current_parts.append(tok)

    # Save last
    if current_cpd and current_parts:
        name = ' '.join(current_parts)
        name = re.sub(r'- (\w)', r'-\1', name)
        entries[current_cpd] = name

    # Filter to actual IUPAC names
    chemical_keywords = [
        'yl)', 'one', 'amine', 'amide', 'nitrile', 'piperidin', 'cyclopent',
        'phenyl', 'benzyl', 'methyl', 'propyl', 'ethyl', 'fluoro', 'chloro',
        'morpholin', 'pyrrolidin', 'azetidin', 'piperazin', 'indolin',
        'isoquinolin', 'isobenzofuran', 'tetrahydro', 'oxo', 'carboxamide',
    ]
    iupac_entries = {
        cpd: name for cpd, name in entries.items()
        if len(name) > 20 and any(k in name.lower() for k in chemical_keywords)
    }

    logger.info(f"Google Patents tables {patent_id}: {len(iupac_entries)} IUPAC names from clean text")

    # Convert through OPSIN
    validated = []
    for cpd_num, name in iupac_entries.items():
        smiles, err = _try_opsin(name)
        if not smiles:
            cleaned = rule_based_clean(name)
            smiles, err = _try_opsin(cleaned)

        if not smiles or not validate_smiles(smiles) or len(smiles) < 10:
            continue

        compound = Compound(
            patent_id=patent_id,
            example_number=f"Cpd. No. {cpd_num}",
            iupac_name=name,
            iupac_source=IupacSource.PATENT_VERBATIM,
            source=CompoundSource.EXEMPLIFIED,
            extraction_method="google_patents_table_opsin",
            processing_status="text_done",
        )
        _finalize(
            compound, smiles, stage="google_patents_table_opsin",
            route_hint="google_patents_table",
        )
        if compound.processing_status == "validated":
            validated.append(compound)

    logger.info(f"Google Patents tables {patent_id}: {len(validated)} validated via OPSIN")
    return validated
