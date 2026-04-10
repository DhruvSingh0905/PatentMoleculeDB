"""Step 2.1 — Page-level compound name + synthesis extraction.

Sends entire iupacs_clean/ pages to Claude. Extracts:
  - compound_id (full verbatim: "Example 1", "Cpd. No. 243")
  - iupac_name (or null if not present)
  - is_intermediate
  - synthesis_steps (for future retrosynthesis project — free data from same page)

Does NOT convert IUPAC→SMILES (that's Step 2.2 — separate call for debuggability).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .api_client import call_claude_text
from .markdown_parser import parse_markdown_file, extract_page_number
from .models import Compound, CompoundSource, IupacSource, SynthesisStep, PageManifest

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are extracting chemical compound information from a pharmaceutical patent page.

Extract ALL compounds mentioned on this page. For each compound return:

[{
  "compound_id": "Example 1" or "Cpd. No. 243" or "Intermediate 1A" (use the FULL designation from the patent),
  "iupac_name": "the complete IUPAC chemical name, or null if not present on this page",
  "is_intermediate": false,
  "synthesis_steps": [
    {
      "step": 1,
      "starting_materials": ["IUPAC or common name of each starting material"],
      "reagents": ["reagent names"],
      "conditions": "temperature, time, solvent (e.g. 'RT, 20h, DCM')",
      "product": "intermediate name or 'final product'"
    }
  ]
}]

Rules:
- Use the FULL compound designation verbatim from the patent (e.g., "Example 1", not just "1")
- Extract the COMPLETE IUPAC name — do not truncate
- If synthesis steps are described, extract them. If not, leave synthesis_steps as []
- Do NOT convert IUPAC names to SMILES
- If no compounds are on this page, return []
- Return ONLY the JSON array, no other text"""


def extract_compounds_from_page(
    page_text: str,
    patent_id: str,
    page_num: int,
) -> list[dict]:
    """Extract compound data from a single page.

    Args:
        page_text: Full text content of the page.
        patent_id: Patent identifier for logging.
        page_num: Page number for logging.

    Returns:
        List of extracted compound dicts (raw Claude output, parsed JSON).
    """
    if not page_text.strip():
        return []

    prompt = f"{EXTRACTION_PROMPT}\n\nPatent page content:\n\n{page_text}"

    response = call_claude_text(
        prompt=prompt,
        patent_id=patent_id,
        compound_id=f"page_{page_num}",
    )

    if response is None:
        logger.error(f"Patent {patent_id} page {page_num}: API call failed")
        return []

    # Parse JSON response
    try:
        cleaned = response.strip()
        # Strip markdown code fences
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        compounds = json.loads(cleaned)
        if not isinstance(compounds, list):
            logger.warning(f"Patent {patent_id} page {page_num}: response not a list")
            return []

        return compounds

    except json.JSONDecodeError:
        logger.warning(
            f"Patent {patent_id} page {page_num}: response not valid JSON. "
            f"Response preview: {response[:200]}"
        )
        return []


def _parse_synthesis_steps(raw_steps: list[dict]) -> list[SynthesisStep]:
    """Convert raw synthesis step dicts to SynthesisStep objects."""
    steps = []
    for raw in raw_steps:
        try:
            steps.append(SynthesisStep(
                step_number=raw.get("step", len(steps) + 1),
                starting_materials=raw.get("starting_materials", []),
                reagents=raw.get("reagents", []),
                conditions=raw.get("conditions"),
                product=raw.get("product", ""),
            ))
        except Exception as e:
            logger.warning(f"Failed to parse synthesis step: {e}")
    return steps


def extract_all_compounds(
    patent_id: str,
    manifest: PageManifest,
    data_dir: Path | None = None,
) -> list[Compound]:
    """Extract all compounds from a patent's iupacs_clean pages.

    Args:
        patent_id: Patent identifier.
        manifest: PageManifest with example_pages identified.
        data_dir: Root data directory.

    Returns:
        List of Compound objects with names and synthesis routes populated.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    iupacs_dir = data_dir / patent_id / "iupacs_clean"
    if not iupacs_dir.exists():
        logger.warning(f"Patent {patent_id}: No iupacs_clean directory")
        return []

    all_compounds: list[Compound] = []
    seen_compound_ids: set[str] = set()
    pages_processed = 0

    # Process all iupacs_clean pages (they're pre-filtered to compound content)
    for page_file in sorted(iupacs_dir.glob("page_*.md")):
        page_num = extract_page_number(page_file)
        if page_num is None:
            continue

        # Read the full page text (raw markdown — Claude can handle the format)
        page_text = page_file.read_text(encoding="utf-8")

        raw_compounds = extract_compounds_from_page(page_text, patent_id, page_num)
        pages_processed += 1

        for raw in raw_compounds:
            compound_id = raw.get("compound_id", "")
            if not compound_id:
                continue

            # Handle duplicate compound_ids (same compound across pages)
            # Keep first occurrence, skip later ones
            dedup_key = compound_id.strip().lower()
            if dedup_key in seen_compound_ids:
                continue
            seen_compound_ids.add(dedup_key)

            # Parse synthesis steps
            synthesis = _parse_synthesis_steps(raw.get("synthesis_steps", []))

            compound = Compound(
                patent_id=patent_id,
                example_number=compound_id,
                iupac_name=raw.get("iupac_name"),
                iupac_source=IupacSource.PATENT_VERBATIM,
                is_intermediate=raw.get("is_intermediate", False),
                source=CompoundSource.EXEMPLIFIED,
                source_page=page_num,
                synthesis_route=synthesis,
                processing_status="text_done",
            )
            all_compounds.append(compound)

    logger.info(
        f"Patent {patent_id}: Extracted {len(all_compounds)} compounds "
        f"from {pages_processed} pages"
    )

    return all_compounds
