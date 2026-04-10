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

CRITICAL rules for IUPAC names:
- Extract the COMPLETE IUPAC name — NEVER truncate. Include every character.
- Verify all parentheses () and brackets [] are balanced. Every open must have a close.
- Copy the name EXACTLY as written in the patent — do not alter locants, substituent positions, or ring numbering.
- Double-check fused ring notation like pyrrolo[2,1-f] — preserve the exact letters (pyrrolo has double r).
- If the name is split across multiple text blocks on the page, reconstruct the full name.
- Do NOT convert IUPAC names to SMILES.

Other rules:
- Use the FULL compound designation verbatim from the patent (e.g., "Example 1", not just "1")
- If synthesis steps are described, extract them. If not, leave synthesis_steps as []
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

    # Parse JSON response — handle Claude adding text before/after JSON
    try:
        cleaned = response.strip()

        # Strip markdown code fences (Claude often wraps JSON in ```json ... ```)
        if "```" in cleaned:
            # Extract content between first ``` and last ```
            parts = cleaned.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    cleaned = part
                    break

        # If response contains a JSON array somewhere, find it
        if not cleaned.startswith("["):
            start = cleaned.find("[")
            if start >= 0:
                # Find the matching closing bracket
                depth = 0
                for i in range(start, len(cleaned)):
                    if cleaned[i] == "[":
                        depth += 1
                    elif cleaned[i] == "]":
                        depth -= 1
                    if depth == 0:
                        cleaned = cleaned[start:i+1]
                        break

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


MAX_PARALLEL_PAGES = 5  # Concurrent page extraction calls


def extract_all_compounds(
    patent_id: str,
    manifest: PageManifest,
    data_dir: Path | None = None,
) -> list[Compound]:
    """Extract all compounds from a patent's iupacs_clean pages.

    Parallelizes page extraction with MAX_PARALLEL_PAGES concurrent calls.

    Args:
        patent_id: Patent identifier.
        manifest: PageManifest with example_pages identified.
        data_dir: Root data directory.

    Returns:
        List of Compound objects with names and synthesis routes populated.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if data_dir is None:
        data_dir = config.DATA_DIR

    iupacs_dir = data_dir / patent_id / "iupacs_clean"
    if not iupacs_dir.exists():
        logger.warning(f"Patent {patent_id}: No iupacs_clean directory")
        return []

    # Only process pages that detection flagged as having example headers
    example_pages = set(manifest.example_pages) if manifest.example_pages else None

    # Collect pages to process
    page_tasks: list[tuple[int, str]] = []
    for page_file in sorted(iupacs_dir.glob("page_*.md")):
        page_num = extract_page_number(page_file)
        if page_num is None:
            continue
        if example_pages is not None and page_num not in example_pages:
            continue
        page_text = page_file.read_text(encoding="utf-8")
        if page_text.strip():
            page_tasks.append((page_num, page_text))

    logger.info(
        f"Patent {patent_id}: Processing {len(page_tasks)} pages "
        f"with {MAX_PARALLEL_PAGES} parallel workers"
    )

    # Process pages in parallel
    all_raw_results: list[tuple[int, list[dict]]] = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PAGES) as executor:
        futures = {
            executor.submit(extract_compounds_from_page, text, patent_id, pnum): pnum
            for pnum, text in page_tasks
        }
        for future in as_completed(futures):
            pnum = futures[future]
            try:
                raw = future.result()
                all_raw_results.append((pnum, raw))
            except Exception as e:
                logger.error(f"Patent {patent_id} page {pnum}: extraction failed: {e}")

    # Sort by page number for deterministic ordering
    all_raw_results.sort(key=lambda x: x[0])

    # Deduplicate and build Compound objects
    all_compounds: list[Compound] = []
    seen_compound_ids: set[str] = set()
    pages_processed = len(all_raw_results)

    for page_num, raw_compounds in all_raw_results:
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
