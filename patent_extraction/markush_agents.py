"""Multi-agent Markush extraction — PatentFinder-style specialized agents.

Three focused agents instead of one monolithic call:
1. FormulaAgent: extracts scaffold SMILES from claims + images
2. SubstituentAgent: extracts R-group definitions from text
3. ConsistencyAgent: validates Markush against extracted examples

Each agent gets only the relevant pages/context, reducing token usage.
Results are combined into a richer MarkushContext.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import config
from .api_client import call_claude_text, call_claude_vision
from .models import MarkushContext
from .smiles_utils import validate_smiles

logger = logging.getLogger(__name__)


FORMULA_PROMPT = """You are analyzing a pharmaceutical patent's claims section.
Extract the core chemical scaffold (Formula I or equivalent):

1. The scaffold SMILES with attachment points marked as [1*], [2*], etc.
2. A brief description of the core structure.

Return ONLY valid JSON:
{"scaffold_smiles": "...", "description": "..."}

If you cannot determine the scaffold, return:
{"scaffold_smiles": null, "description": "..."}"""


SUBSTITUENT_PROMPT = """Extract all R-group definitions from this patent text.
For each variable position (R1, R2, R3, G, X, Y, etc.), list the allowed substituents.

Return ONLY valid JSON:
{
  "R1": ["methyl", "ethyl", "cyclopropyl"],
  "R2": ["hydrogen", "fluoro", "chloro"],
  ...
}

Only include positions you can clearly identify. Use IUPAC fragment names."""


CONSISTENCY_PROMPT = """Given this Markush scaffold and R-group definitions, check if the following example compounds are consistent with the Markush definition.

Scaffold: {scaffold}
R-groups: {r_groups}

Example compounds (SMILES):
{examples}

For each example, determine:
1. Does it match the scaffold pattern?
2. Which R-group values does it use?

Return JSON:
[{{"example": "Cpd 1", "matches_scaffold": true, "r_values": {{"R1": "methyl", "R2": "F"}}}}]"""


def _extract_formula(patent_id: str, manifest, data_dir: Path) -> dict:
    """FormulaAgent: Extract scaffold from Markush pages (focused, minimal context)."""
    # Only send the first 2-3 Markush pages, not all of them
    markush_pages = sorted(manifest.markush_pages)[:3]
    if not markush_pages:
        return {"scaffold_smiles": None, "description": "No Markush pages found"}

    # Build focused context — just the claims text, stripped of images
    context_parts = []
    total_chars = 0
    for page_num in markush_pages:
        for subdir in ['iupacs_clean', 'all_pages']:
            page_path = data_dir / patent_id / subdir / f"page_{page_num:04d}.md"
            if page_path.exists():
                text = page_path.read_text()
                # Strip image markers to save tokens
                text = re.sub(r'<\|ref\|>.*?<\|/det\|>', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 100:
                    context_parts.append(f"--- Page {page_num} ---\n{text[:5000]}")
                    total_chars += min(len(text), 5000)
                break
        if total_chars > 12000:
            break

    if not context_parts:
        return {"scaffold_smiles": None, "description": "No readable Markush text"}

    prompt = FORMULA_PROMPT + "\n\nPatent text:\n" + "\n".join(context_parts)

    response = call_claude_text(
        prompt=prompt,
        model=config.MODEL_SONNET,
        patent_id=patent_id,
        compound_id="markush_formula",
        max_tokens=500,
    )

    if not response:
        return {"scaffold_smiles": None, "description": "API call failed"}

    try:
        # Parse JSON from response
        cleaned = response.strip()
        if '```' in cleaned:
            cleaned = re.search(r'```(?:json)?\s*(.*?)```', cleaned, re.DOTALL)
            cleaned = cleaned.group(1) if cleaned else response
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"scaffold_smiles": None, "description": response[:200]}


def _extract_substituents(patent_id: str, manifest, data_dir: Path) -> dict[str, list[str]]:
    """SubstituentAgent: Extract R-group definitions from text."""
    # First try symbolic regex (free, no API)
    r_groups = _symbolic_r_group_parse(patent_id, data_dir)

    # If regex found <3 R-groups, supplement with LM call
    if len(r_groups) < 3:
        markush_pages = sorted(manifest.markush_pages)[:5]
        context_parts = []
        total_chars = 0

        for page_num in markush_pages:
            for subdir in ['iupacs_clean', 'all_pages']:
                page_path = data_dir / patent_id / subdir / f"page_{page_num:04d}.md"
                if page_path.exists():
                    text = page_path.read_text()
                    # Only send text containing R-group patterns
                    if re.search(r'R\d|selected from|consisting of', text, re.IGNORECASE):
                        text = re.sub(r'<\|ref\|>.*?<\|/det\|>', '', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        context_parts.append(text[:3000])
                        total_chars += min(len(text), 3000)
                    break
            if total_chars > 10000:
                break

        if context_parts:
            prompt = SUBSTITUENT_PROMPT + "\n\nPatent text:\n" + "\n".join(context_parts)
            response = call_claude_text(
                prompt=prompt,
                model=config.MODEL_SONNET,
                patent_id=patent_id,
                compound_id="markush_substituents",
                max_tokens=1000,
            )

            if response:
                try:
                    cleaned = response.strip()
                    if '```' in cleaned:
                        m = re.search(r'```(?:json)?\s*(.*?)```', cleaned, re.DOTALL)
                        cleaned = m.group(1) if m else cleaned
                    lm_groups = json.loads(cleaned)
                    # Merge with regex results
                    for label, options in lm_groups.items():
                        if label not in r_groups:
                            r_groups[label] = options
                        else:
                            # Add new options
                            existing = set(r_groups[label])
                            for opt in options:
                                if opt not in existing:
                                    r_groups[label].append(opt)
                except json.JSONDecodeError:
                    pass

    return r_groups


def _symbolic_r_group_parse(patent_id: str, data_dir: Path) -> dict[str, list[str]]:
    """Regex-based R-group extraction (free, no API calls)."""
    r_groups = {}
    pages_dir = data_dir / patent_id / 'all_pages'
    if not pages_dir.exists():
        return r_groups

    for pf in sorted(pages_dir.glob('page_*.md')):
        text = pf.read_text()
        text = re.sub(r'<[^>]+>', ' ', text)

        for m in re.finditer(
            r'(R\d+[a-z]?|[GXY])\s+is\s+(?:selected from (?:the group )?consisting of\s+)?(.*?)(?:\.\s+[A-Z]|;\s+[A-Z]|;\s+R\d|\.\s*$)',
            text, re.DOTALL
        ):
            label = m.group(1)
            options_text = m.group(2).strip()
            options_text = re.sub(r'\s+', ' ', options_text)
            options = re.split(r',\s+|\s+and\s+', options_text)
            options = [o.strip() for o in options if len(o.strip()) > 1]
            if options:
                if label not in r_groups:
                    r_groups[label] = []
                r_groups[label].extend(options)

    return r_groups


def _check_consistency(
    patent_id: str,
    scaffold_smiles: str | None,
    r_groups: dict[str, list[str]],
    example_smiles: list[str],
) -> dict:
    """ConsistencyAgent: Check if examples match Markush definition."""
    if not scaffold_smiles or not example_smiles:
        return {"coverage": 0, "details": []}

    # Use RDKit substructure matching (free, no API)
    from rdkit import Chem

    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    if not scaffold_mol:
        # Try as SMARTS
        scaffold_mol = Chem.MolFromSmarts(scaffold_smiles)
    if not scaffold_mol:
        return {"coverage": 0, "details": [{"error": "invalid scaffold"}]}

    matches = 0
    details = []
    for smi in example_smiles[:20]:  # Check first 20
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        has_match = mol.HasSubstructMatch(scaffold_mol)
        if has_match:
            matches += 1
        details.append({"smiles": smi[:40], "matches": has_match})

    coverage = matches / max(len(example_smiles[:20]), 1)
    return {"coverage": coverage, "matches": matches, "total": len(example_smiles[:20]), "details": details}


def extract_markush_multiagent(
    patent_id: str,
    manifest,
    data_dir: Path | None = None,
    example_smiles: list[str] | None = None,
) -> MarkushContext:
    """Run multi-agent Markush extraction.

    Three focused agents with minimal context each:
    1. FormulaAgent → scaffold SMILES (2-3 pages, ~5K tokens)
    2. SubstituentAgent → R-group definitions (regex first, LM fallback, ~3K tokens)
    3. ConsistencyAgent → validate against examples (RDKit only, 0 tokens)

    Total: ~8K tokens vs ~30K for monolithic approach (73% savings).
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    logger.info(f"Markush multi-agent: {patent_id}")

    # Agent 1: Formula
    formula_result = _extract_formula(patent_id, manifest, data_dir)
    scaffold_smiles = formula_result.get("scaffold_smiles")
    scaffold_desc = formula_result.get("description", "")

    if scaffold_smiles and not validate_smiles(scaffold_smiles):
        logger.warning(f"  FormulaAgent: invalid scaffold SMILES, keeping as description")
        scaffold_desc = f"{scaffold_desc} (SMILES: {scaffold_smiles})"
        scaffold_smiles = None

    logger.info(f"  FormulaAgent: scaffold={'valid' if scaffold_smiles else 'text-only'}")

    # Agent 2: Substituents
    r_groups = _extract_substituents(patent_id, manifest, data_dir)
    logger.info(f"  SubstituentAgent: {len(r_groups)} R-groups found")

    # Agent 3: Consistency (free — RDKit only)
    consistency = {"coverage": 0}
    if example_smiles:
        consistency = _check_consistency(patent_id, scaffold_smiles, r_groups, example_smiles)
        logger.info(f"  ConsistencyAgent: {consistency['coverage']:.0%} scaffold coverage")

    # Determine quality
    if scaffold_smiles and len(r_groups) >= 2:
        quality = "high_confidence"
    elif scaffold_desc or len(r_groups) >= 1:
        quality = "partial"
    else:
        quality = "failed"

    return MarkushContext(
        patent_id=patent_id,
        scaffold_smiles=scaffold_smiles,
        scaffold_description=scaffold_desc,
        r_group_definitions=r_groups,
        quality=quality,
    )
