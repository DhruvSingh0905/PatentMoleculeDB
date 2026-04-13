"""Step 2.5 — Parse compound tables to extract Example→IUPAC name→M+H mappings.

Many patents have compound tables (TABLE 49 in US10214537) with format:
  Ex. No. | Structure | Name | M+H

The "Name" column contains the full IUPAC name. We parse these tables
to extract compounds that aren't in the Example sections (iupacs_clean).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from . import config
from .models import Compound, CompoundSource, IupacSource, PageManifest

logger = logging.getLogger(__name__)


def parse_compound_table(html: str) -> list[dict]:
    """Parse a compound table with Ex. No. / Name / M+H columns.

    Args:
        html: Raw HTML table content.

    Returns:
        List of {example_number, iupac_name, mh_plus} dicts.
    """
    compounds = []
    soup = BeautifulSoup(html, 'html.parser')

    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if not rows:
            continue

        # Detect column layout from header row
        header_cells = rows[0].find_all('td')
        header_text = [c.get_text(strip=True).lower() for c in header_cells]

        # Find relevant column indices
        ex_col = None
        name_col = None
        mh_col = None

        for i, h in enumerate(header_text):
            if 'ex' in h and 'no' in h or h.startswith('ex'):
                ex_col = i
            elif 'name' in h:
                name_col = i
            elif 'm + h' in h or 'm+h' in h or 'mh+' in h.replace(' ', ''):
                mh_col = i

        # Fallback: if no header detected, try common layouts
        # Layout 1: Ex. No. | Structure | Name | M+H (4 columns)
        if ex_col is None and len(header_cells) >= 3:
            ex_col = 0
            name_col = 2 if len(header_cells) >= 4 else 1
            mh_col = 3 if len(header_cells) >= 4 else 2

        if ex_col is None:
            continue

        # Parse data rows
        for row in rows[1:]:
            cells = row.find_all('td')
            if len(cells) <= max(filter(None, [ex_col, name_col, mh_col]), default=0):
                continue

            ex_text = cells[ex_col].get_text(strip=True) if ex_col is not None and ex_col < len(cells) else ""
            name_text = cells[name_col].get_text(' ', strip=True) if name_col is not None and name_col < len(cells) else ""
            mh_text = cells[mh_col].get_text(strip=True) if mh_col is not None and mh_col < len(cells) else ""

            # Validate example number
            ex_match = re.match(r'(\d+)', ex_text)
            if not ex_match:
                continue

            example_number = f"Example {ex_match.group(1)}"

            # Clean IUPAC name
            iupac = name_text.strip()
            # Remove line break artifacts
            iupac = re.sub(r'\s+', ' ', iupac)
            # Skip empty or very short names
            if len(iupac) < 15:
                iupac = None

            # Parse M+H value
            mh_plus = None
            if mh_text:
                mh_match = re.search(r'(\d+\.?\d*)', mh_text)
                if mh_match:
                    try:
                        mh_plus = float(mh_match.group(1))
                        if not (100 < mh_plus < 2000):
                            mh_plus = None
                    except ValueError:
                        pass

            if iupac or mh_plus:
                compounds.append({
                    'example_number': example_number,
                    'iupac_name': iupac,
                    'mh_plus': mh_plus,
                })

    return compounds


def extract_table_compounds(
    patent_id: str,
    data_dir: Path | None = None,
) -> list[Compound]:
    """Extract compounds from all table pages in a patent.

    Processes both table/ directory and all_pages/ for table content.

    Args:
        patent_id: Patent identifier.
        data_dir: Root data directory.

    Returns:
        List of Compound objects extracted from tables.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    all_compounds: list[Compound] = []
    seen_ids: set[str] = set()

    # Process all_pages/ looking for compound tables
    # (TABLE 49 etc. with Ex. No. | Structure | Name | M+H)
    all_pages_dir = data_dir / patent_id / "all_pages"
    if all_pages_dir.exists():
        for page_file in sorted(all_pages_dir.glob("page_*.md")):
            text = page_file.read_text(encoding="utf-8")

            # Only process pages that have table content with compound-like data
            if '<table>' not in text.lower():
                continue
            if 'name' not in text.lower() or ('ex' not in text.lower()):
                continue

            # Extract page number
            page_match = re.search(r'page_(\d+)', page_file.stem)
            page_num = int(page_match.group(1)) if page_match else None

            # Parse tables on this page
            table_compounds = parse_compound_table(text)

            for tc in table_compounds:
                ex_id = tc['example_number']
                dedup_key = ex_id.lower().replace(' ', '')

                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                compound = Compound(
                    patent_id=patent_id,
                    example_number=ex_id,
                    iupac_name=tc.get('iupac_name'),
                    iupac_source=IupacSource.PATENT_VERBATIM,
                    ms_mh_plus=tc.get('mh_plus'),
                    source=CompoundSource.EXEMPLIFIED,
                    source_page=page_num,
                    processing_status="text_done",
                )
                all_compounds.append(compound)

    # Also parse claims pages for dense compound lists: "IUPAC_NAME (NUMBER); ..."
    if all_pages_dir.exists():
        for page_file in sorted(all_pages_dir.glob("page_*.md")):
            text = page_file.read_text(encoding="utf-8")

            # Only process pages with semicolon-separated compound lists
            if text.count(';') < 5 or '(' not in text:
                continue

            page_match = re.search(r'page_(\d+)', page_file.stem)
            page_num = int(page_match.group(1)) if page_match else None

            # Strip markdown tags and normalize whitespace
            clean = re.sub(r'<\|ref\|>\w+<\|/ref\|><\|det\|>\[\[\d+.*?\]\]<\|/det\|>', '', text)
            clean = re.sub(r'\s+', ' ', clean)

            # Split by semicolons — each segment ends with "IUPAC_NAME (NUMBER)"
            segments = clean.split(';')
            for seg in segments:
                seg = seg.strip()
                m = re.search(r'\((\d{1,4})\)\s*\.?\s*$', seg)
                if not m:
                    continue
                num = int(m.group(1))
                name = seg[:m.start()].strip()

                if len(name) < 20:
                    continue

                # Filter: must look like a real IUPAC compound name
                # (contains ring/heterocycle indicators, not Markush fragments)
                name_lower = name.lower()
                has_ring = any(kw in name_lower for kw in [
                    'pyrrolo', 'pyrrol', 'triazin', 'pyrazol', 'piperidin',
                    'morpholin', 'piperazin', 'benzamide', 'benzonitrile',
                    'phenyl', 'oxazol', 'thiomorpholin', 'pyridin',
                    'oxetan', 'azetidin', 'pyrrolidin', 'imidazol',
                ])
                if not has_ring:
                    continue

                ex_id = f"Example {num}"
                dedup_key = ex_id.lower().replace(' ', '')
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                compound = Compound(
                    patent_id=patent_id,
                    example_number=ex_id,
                    iupac_name=name,
                    iupac_source=IupacSource.PATENT_VERBATIM,
                    source=CompoundSource.EXEMPLIFIED,
                    source_page=page_num,
                    processing_status="text_done",
                )
                all_compounds.append(compound)

    logger.info(
        f"Patent {patent_id}: Extracted {len(all_compounds)} compounds from tables + claims"
    )
    return all_compounds
