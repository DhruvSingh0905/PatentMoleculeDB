"""Step 2.0 — Extract Markush scaffold context from patent claims.

Sends the full claims section to Claude once per patent. The scaffold context
is NOT used for enumeration — it's fed as context to the IUPAC→SMILES step
to help Claude disambiguate complex nomenclature.

Quality tiers:
  high_confidence: valid SMILES (passes RDKit) + complete R-group definitions
  partial: natural language description but no valid SMILES
  failed: couldn't extract coherent scaffold → IUPAC conversion runs without context
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .api_client import call_claude_text
from .markdown_parser import parse_markdown_file, get_text_content
from .models import MarkushContext, PageManifest
from .smiles_utils import validate_smiles

logger = logging.getLogger(__name__)

MARKUSH_EXTRACTION_PROMPT = """You are a pharmaceutical patent analyst. Extract the core molecular scaffold (Markush structure) from these patent claims.

Return a JSON object:
{
  "scaffold_smiles": "canonical SMILES of the core scaffold with [1*], [2*] etc for variable positions, or null if you cannot determine it",
  "scaffold_description": "natural language description of the scaffold and its variable positions",
  "r_group_definitions": {
    "1": ["list", "of", "allowed", "substituents", "as", "SMILES"],
    "2": ["F", "Cl", "Br"]
  }
}

Rules:
- Use [1*], [2*], [3*] for variable attachment points in the scaffold SMILES
- List R-group options as SMILES fragments where possible
- If you can't determine a valid SMILES, still provide the natural language description
- Expand nested definitions (e.g., "optionally substituted phenyl" → list concrete options)
- Resolve cross-references between claims (e.g., "Claim 2: compound of Claim 1 where R1 is methyl")

Return ONLY the JSON object."""


def extract_markush_context(
    patent_id: str,
    manifest: PageManifest,
    data_dir: Path | None = None,
) -> MarkushContext:
    """Extract Markush scaffold context from a patent's claims section.

    Args:
        patent_id: Patent identifier.
        manifest: PageManifest with markush_pages identified.
        data_dir: Root data directory.

    Returns:
        MarkushContext with quality tier assigned.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    if not manifest.markush_pages:
        logger.info(f"Patent {patent_id}: No Markush pages detected, skipping context extraction")
        return MarkushContext(patent_id=patent_id, quality="failed")

    # Collect all claims text from Markush-flagged pages
    all_pages_dir = data_dir / patent_id / "all_pages"
    claims_text_parts = []

    for page_num in sorted(manifest.markush_pages):
        page_file = all_pages_dir / f"page_{page_num:04d}.md"
        if not page_file.exists():
            continue
        elements = parse_markdown_file(page_file)
        text = get_text_content(elements)
        if text.strip():
            claims_text_parts.append(f"--- Page {page_num} ---\n{text}")

    if not claims_text_parts:
        logger.warning(f"Patent {patent_id}: Markush pages found but no text content")
        return MarkushContext(patent_id=patent_id, quality="failed")

    claims_text = "\n\n".join(claims_text_parts)

    # Truncate if too long (keep within reasonable token limits)
    max_chars = 30000  # ~7500 tokens
    if len(claims_text) > max_chars:
        claims_text = claims_text[:max_chars] + "\n[... truncated]"

    # Call Claude
    full_prompt = f"{MARKUSH_EXTRACTION_PROMPT}\n\nPatent claims text:\n\n{claims_text}"

    response = call_claude_text(
        prompt=full_prompt,
        patent_id=patent_id,
        compound_id="markush_context",
    )

    if response is None:
        logger.error(f"Patent {patent_id}: Markush context extraction API call failed")
        return MarkushContext(patent_id=patent_id, quality="failed")

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Patent {patent_id}: Markush response not valid JSON, treating as partial")
        return MarkushContext(
            patent_id=patent_id,
            scaffold_description=response[:2000],
            quality="partial",
        )

    # Build MarkushContext with quality assessment
    scaffold_smiles = data.get("scaffold_smiles")
    scaffold_description = data.get("scaffold_description", "")
    r_groups = data.get("r_group_definitions", {})

    # Validate scaffold SMILES with RDKit
    smiles_valid = scaffold_smiles and validate_smiles(scaffold_smiles)

    if smiles_valid and r_groups:
        quality = "high_confidence"
    elif scaffold_description:
        quality = "partial"
        if not smiles_valid:
            scaffold_smiles = None  # Don't pass bad SMILES downstream
    else:
        quality = "failed"

    context = MarkushContext(
        patent_id=patent_id,
        scaffold_smiles=scaffold_smiles if smiles_valid else None,
        scaffold_description=scaffold_description,
        r_group_definitions=r_groups,
        quality=quality,
    )

    logger.info(
        f"Patent {patent_id}: Markush context quality={quality}, "
        f"scaffold_smiles={'valid' if smiles_valid else 'invalid/missing'}, "
        f"r_groups={len(r_groups)} positions"
    )

    return context


def get_context_for_prompt(context: MarkushContext) -> str:
    """Format Markush context for inclusion in IUPAC→SMILES prompt.

    Args:
        context: MarkushContext to format.

    Returns:
        String to insert in the prompt, or empty string if failed quality.
    """
    if context.quality == "failed":
        return ""

    parts = []

    if context.quality == "high_confidence" and context.scaffold_smiles:
        parts.append(f"Core scaffold SMILES: {context.scaffold_smiles}")

    if context.scaffold_description:
        parts.append(f"Scaffold description: {context.scaffold_description}")

    if context.r_group_definitions:
        for pos, options in context.r_group_definitions.items():
            parts.append(f"Position {pos} options: {', '.join(options[:20])}")

    if not parts:
        return ""

    return "This compound belongs to a patent with the following molecular scaffold:\n" + "\n".join(parts)
