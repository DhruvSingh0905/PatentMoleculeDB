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

STEP 1: Extract the core chemical scaffold (Formula I or equivalent) as SMILES.
Use [*:1], [*:2], etc. at every variable (R-group) position.

STEP 2: For EVERY attachment point [*:n] in your scaffold, determine which R-group
label from the patent it corresponds to. Examine:
- The Formula (I) text or drawing to see where each R-label appears
- The R-group definition sections ("R1 is...", "R2 is...", "G is...", "X is...")
- Any descriptions like "R1 is attached to the nitrogen of the amide"

You MUST account for ALL [*:n] positions. For each one, either:
- Assign it to an R-label with brief evidence, OR
- Mark r_label as null if it is a fixed part of the core (not variable)

Return ONLY valid JSON:
{
  "scaffold_smiles": "...[*:1]...[*:6]...",
  "positions": [
    {"pos": 1, "r_label": "R2", "evidence": "R2 is on the cyclopentyl carbon"},
    {"pos": 2, "r_label": "R1a", "evidence": "R1a connects to the piperidine N"},
    {"pos": 3, "r_label": "G", "evidence": "G is the pyrimidine substituent"},
    {"pos": 4, "r_label": null, "evidence": "Fixed methyl, not variable"}
  ]
}

If you cannot determine the scaffold at all, return:
{"scaffold_smiles": null, "positions": [], "description": "why it failed"}"""


FORMULA_IMAGE_PROMPT = """This is a Markush structure drawing from a pharmaceutical patent (Formula I or equivalent).

STEP 1: Convert the drawn structure to SMILES with [*:1], [*:2], etc. at every
position where an R-group label (R1, R2, G, X, Y, etc.) is drawn.

STEP 2: For EVERY [*:n], identify which R-label is drawn next to that bond.
You MUST map ALL attachment points — the labels are visible in the drawing.

Return ONLY valid JSON:
{
  "scaffold_smiles": "...[*:1]...[*:6]...",
  "positions": [
    {"pos": 1, "r_label": "R1", "evidence": "R1 drawn on the nitrogen bond"},
    {"pos": 2, "r_label": "R2", "evidence": "R2 drawn on the carbon"},
    {"pos": 3, "r_label": "G", "evidence": "G label at the ring position"}
  ]
}"""


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


def _parse_formula_result(raw_result: dict) -> dict:
    """Parse FormulaAgent output into standardized format with position_mapping.

    Validates that mapping position numbers actually exist in the scaffold SMILES.
    If LM used wrong indices, attempts to re-map by sequential order.
    """
    result = dict(raw_result)

    # Ensure scaffold_smiles uses [n*] notation (some LMs output [*:n])
    smi = result.get('scaffold_smiles')
    if smi:
        smi = re.sub(r'\[\*:(\d+)\]', r'[\1*]', smi)

        # Connected-scaffold filter: enforce single connected component
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:
                # Keep only the largest fragment (most heavy atoms)
                core = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                smi_clean = Chem.MolToSmiles(core, canonical=True)
                # Re-add [n*] notation (RDKit may have changed it)
                smi_clean = re.sub(r'\[\*:(\d+)\]', r'[\1*]', smi_clean)
                logger.info(f"  Scaffold cleaned: {len(frags)} fragments → kept largest ({core.GetNumHeavyAtoms()} atoms)")
                smi = smi_clean

        result['scaffold_smiles'] = smi

    # Get actual placeholder positions from scaffold
    scaffold_positions = set()
    if smi:
        scaffold_positions = set(re.findall(r'\[(\d+)\*\]', smi))

    # Convert positions array to position_mapping dict
    if 'positions' in result:
        mapping = {}
        ordered_labels = []  # Track order for fallback
        for p in result['positions']:
            r_label = p.get('r_label')
            pos = p.get('pos')
            if r_label and pos is not None:
                pos_str = str(pos)
                if pos_str in scaffold_positions:
                    # Direct match — pos number exists in scaffold
                    mapping[str(r_label)] = pos_str
                ordered_labels.append(str(r_label) if r_label else None)

        # If few direct matches, LM likely used array indices instead of [n*] numbers
        # Fall back to sequential assignment
        if len(mapping) < len(scaffold_positions) // 2 and ordered_labels:
            sorted_positions = sorted(scaffold_positions, key=int)
            valid_labels = [l for l in ordered_labels if l is not None]
            sequential_mapping = {}
            for label, pos in zip(valid_labels, sorted_positions):
                sequential_mapping[label] = pos

            # Use sequential if it covers more positions
            if len(sequential_mapping) > len(mapping):
                logger.info(f"  Position mapping: using sequential fallback ({len(sequential_mapping)} vs {len(mapping)} direct)")
                mapping = sequential_mapping

        result['position_mapping'] = mapping
    elif 'position_mapping' not in result:
        result['position_mapping'] = {}
    else:
        # Validate existing position_mapping against scaffold
        if scaffold_positions:
            validated = {}
            for label, pos in result['position_mapping'].items():
                if str(pos) in scaffold_positions:
                    validated[label] = str(pos)
            if len(validated) < len(result['position_mapping']):
                logger.warning(f"  Dropped {len(result['position_mapping']) - len(validated)} invalid position mappings")
            result['position_mapping'] = validated

    return result


def _extract_formula_from_image(patent_id: str, manifest, data_dir: Path) -> dict:
    """FormulaAgent Path B: Extract scaffold from Markush structure images via Vision.

    For patents where Formula I is a drawing, not text. Uses Claude Vision
    to read the structure and produce scaffold SMILES + position mapping.
    """
    import fitz
    from PIL import Image

    markush_pages = sorted(manifest.markush_pages)[:3]
    if not markush_pages:
        return {"scaffold_smiles": None, "description": "No Markush pages", "position_mapping": {}}

    pdf_path = data_dir / patent_id / f"{patent_id}.pdf"
    if not pdf_path.exists():
        return {"scaffold_smiles": None, "description": "No PDF", "position_mapping": {}}

    images_dir = config.IMAGES_DIR / patent_id
    images_dir.mkdir(parents=True, exist_ok=True)

    # Render first Markush page and send to Vision
    for page_num in markush_pages[:2]:
        img_path = str(images_dir / f"markush_p{page_num:04d}.png")

        try:
            doc = fitz.open(str(pdf_path))
            if page_num >= len(doc):
                doc.close()
                continue
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(img_path)
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to render page {page_num}: {e}")
            continue

        response = call_claude_vision(
            prompt=FORMULA_IMAGE_PROMPT,
            image_path=img_path,
            model=config.MODEL_SONNET,
            patent_id=patent_id,
            compound_id="markush_formula_image",
        )

        if not response:
            continue

        try:
            cleaned = response.strip()
            if '```' in cleaned:
                m = re.search(r'```(?:json)?\s*(.*?)```', cleaned, re.DOTALL)
                cleaned = m.group(1) if m else cleaned
            raw = json.loads(cleaned)
            result = _parse_formula_result(raw)
            if result.get("scaffold_smiles") and validate_smiles(result["scaffold_smiles"]):
                logger.info(f"  FormulaAgent (image): valid scaffold from page {page_num}, mapping={result.get('position_mapping', {})}")
                return result
        except (json.JSONDecodeError, Exception):
            continue

    return {"scaffold_smiles": None, "description": "Vision extraction failed", "position_mapping": {}}


def _extract_formula(patent_id: str, manifest, data_dir: Path) -> dict:
    """FormulaAgent: Extract scaffold + position mapping from Markush pages.

    Sends both scaffold pages AND R-group definition pages so the LM can
    produce the [n*] ↔ R-label mapping alongside the scaffold SMILES.
    """
    markush_pages = sorted(manifest.markush_pages)[:5]  # More pages for mapping context
    if not markush_pages:
        return {"scaffold_smiles": None, "description": "No Markush pages found", "position_mapping": {}}

    # Build focused context — scaffold pages + R-group definition pages
    context_parts = []
    total_chars = 0

    # Include pages with Formula/scaffold AND pages with R-group definitions
    for page_num in markush_pages:
        for subdir in ['iupacs_clean', 'all_pages']:
            page_path = data_dir / patent_id / subdir / f"page_{page_num:04d}.md"
            if page_path.exists():
                text = page_path.read_text()
                text = re.sub(r'<\|ref\|>.*?<\|/det\|>', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                # Prioritize pages with R-group definitions or Formula references
                has_rgroups = bool(re.search(r'R\d+\s+is|selected from|Formula\s*\(?[IVX]', text, re.IGNORECASE))
                if len(text) > 100 and (has_rgroups or total_chars < 5000):
                    context_parts.append(f"--- Page {page_num} ---\n{text[:4000]}")
                    total_chars += min(len(text), 4000)
                break
        if total_chars > 15000:
            break

    if not context_parts:
        return {"scaffold_smiles": None, "description": "No readable Markush text", "position_mapping": {}}

    prompt = FORMULA_PROMPT + "\n\nPatent text:\n" + "\n".join(context_parts)

    response = call_claude_text(
        prompt=prompt,
        model=config.MODEL_SONNET,
        patent_id=patent_id,
        compound_id="markush_formula",
        max_tokens=1000,
    )

    if not response:
        return {"scaffold_smiles": None, "description": "API call failed", "position_mapping": {}}

    try:
        cleaned = response.strip()
        if '```' in cleaned:
            m = re.search(r'```(?:json)?\s*(.*?)```', cleaned, re.DOTALL)
            cleaned = m.group(1) if m else cleaned
        raw = json.loads(cleaned)
        result = _parse_formula_result(raw)
        logger.info(f"  FormulaAgent (text): mapping={result.get('position_mapping', {})}")
        return result
    except json.JSONDecodeError:
        return {"scaffold_smiles": None, "description": response[:200], "position_mapping": {}}


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


def _clean_rgroup_text(text: str) -> str:
    """Clean R-group definition text before splitting into options.

    Removes pagination artifacts, figure references, LaTeX fragments,
    and normalizes whitespace and formatting.
    """
    t = text
    # Remove page references: [123, 456], page 123, FIG. 1
    t = re.sub(r'\[\[?\d{2,4}[,\s\d]*\]\]?', '', t)
    t = re.sub(r'(?:page|FIG\.?)\s*\d+', '', t, flags=re.IGNORECASE)
    # Remove "text" annotations from OCR
    t = re.sub(r'\btext\b', '', t, flags=re.IGNORECASE)
    # Remove "(continued)" markers
    t = re.sub(r'\(continued\)', '', t, flags=re.IGNORECASE)
    # Remove LaTeX math fragments
    t = re.sub(r'\\[\(\)]', '', t)
    t = re.sub(r'\\mathrm\{[^}]*\}', '', t)
    t = re.sub(r'\\\w+', '', t)
    # Remove bare numbers that are page/figure refs (3+ digits alone)
    t = re.sub(r'\b\d{3,}\b', '', t)
    # Fix "G- 1" → remove if it's a page ref (G-1 followed by G-2 pattern)
    t = re.sub(r'G-\s*\d+\.?\s*', '', t)
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    # Fix broken hyphens from line breaks
    t = re.sub(r'- (\w)', r'-\1', t)
    return t


def _is_valid_rgroup_option(option: str) -> bool:
    """Filter out junk R-group options (page refs, empty, non-chemical)."""
    o = option.strip().lower()
    if len(o) < 2:
        return False
    # Reject pure numbers (page refs)
    if re.match(r'^\d+$', o):
        return False
    # Reject "in another embodiment" etc.
    if any(kw in o for kw in ['embodiment', 'wherein', 'aspect', 'example', 'table', 'figure']):
        return False
    # Must contain at least one chemical term or be a known pattern
    chemical_terms = [
        'alkyl', 'aryl', 'hetero', 'cyclo', 'phenyl', 'methyl', 'ethyl',
        'propyl', 'butyl', 'fluoro', 'chloro', 'bromo', 'amino', 'hydroxy',
        'oxy', 'thio', 'sulfonyl', 'carbonyl', 'hydrogen', 'halogen', 'cyano',
        'morpholin', 'piperidin', 'pyrrolidin', 'pyridin', 'pyrimidin',
        'absent', 'h', 'f', 'cl', 'br', 'oh', 'nh',
    ]
    if any(term in o for term in chemical_terms):
        return True
    # Also accept if it looks like a chemical formula
    if re.search(r'[CNOS]\d|[a-z]-[a-z]', o):
        return True
    return False


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
            r'(R\d+[a-z]?|[GXY])\s+is\s+(?:selected from (?:the (?:group|series) )?consisting of\s+)?(.*?)(?:\.\s+[A-Z]|;\s+[A-Z]|;\s+R\d|\.\s*$)',
            text, re.DOTALL
        ):
            label = m.group(1)
            options_text = m.group(2).strip()
            # Clean R-group text before splitting
            options_text = _clean_rgroup_text(options_text)
            options = re.split(r',\s+|\s+and\s+', options_text)
            options = [o.strip() for o in options if len(o.strip()) > 1]
            # Filter out junk entries
            options = [o for o in options if _is_valid_rgroup_option(o)]
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


def classify_markush_difficulty(r_groups: dict[str, list[str]], scaffold_smiles: str | None) -> str:
    """Classify Markush as LOW/MED/HIGH difficulty for routing."""
    n_rgroups = len(r_groups)
    has_scaffold = scaffold_smiles is not None
    max_options = max((len(v) for v in r_groups.values()), default=0)
    total_options = sum(len(v) for v in r_groups.values())

    if has_scaffold and n_rgroups <= 3 and max_options <= 10:
        return "LOW"
    elif has_scaffold and n_rgroups <= 8 and total_options <= 50:
        return "MED"
    else:
        return "HIGH"


def extract_markush_multiagent(
    patent_id: str,
    manifest,
    data_dir: Path | None = None,
    example_smiles: list[str] | None = None,
) -> MarkushContext:
    """Run multi-agent Markush extraction with difficulty-aware routing.

    LOW difficulty: symbolic-only, skip LM agents
    MED difficulty: two-path (symbolic + LM), run enumerator if coverage < 50%
    HIGH difficulty: full multi-agent + consistency checking

    Total: ~8K tokens vs ~30K for monolithic approach (73% savings).
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    logger.info(f"Markush multi-agent: {patent_id}")

    # Agent 1: Formula + Position Mapping
    # Image-first: most direct R-label signal (labels are drawn next to bonds)
    logger.info(f"  FormulaAgent: trying image path first (most direct R-label signal)")
    formula_result = _extract_formula_from_image(patent_id, manifest, data_dir)
    scaffold_smiles = formula_result.get("scaffold_smiles")
    scaffold_desc = formula_result.get("description", "")
    lm_position_mapping = formula_result.get("position_mapping", {})

    if scaffold_smiles and not validate_smiles(scaffold_smiles):
        scaffold_desc = f"{scaffold_desc} (SMILES: {scaffold_smiles})"
        scaffold_smiles = None

    # If image failed, fall back to text-based extraction
    if not scaffold_smiles:
        logger.info(f"  FormulaAgent: image path failed — trying text")
        text_result = _extract_formula(patent_id, manifest, data_dir)
        scaffold_smiles = text_result.get("scaffold_smiles")
        scaffold_desc = text_result.get("description", scaffold_desc)
        lm_position_mapping = text_result.get("position_mapping", lm_position_mapping)
        if scaffold_smiles and not validate_smiles(scaffold_smiles):
            scaffold_desc = f"{scaffold_desc} (SMILES: {scaffold_smiles})"
            scaffold_smiles = None
    elif not lm_position_mapping:
        # Image gave scaffold but no mapping — try text for mapping only
        text_result = _extract_formula(patent_id, manifest, data_dir)
        text_mapping = text_result.get("position_mapping", {})
        if text_mapping:
            lm_position_mapping = text_mapping
            logger.info(f"  FormulaAgent: using text mapping to supplement image scaffold")

    logger.info(f"  FormulaAgent: scaffold={'valid' if scaffold_smiles else 'text-only'}")

    # Agent 2: Substituents (symbolic first, LM only if needed)
    r_groups = _symbolic_r_group_parse(patent_id, data_dir)
    difficulty = classify_markush_difficulty(r_groups, scaffold_smiles)
    logger.info(f"  Difficulty: {difficulty} (symbolic found {len(r_groups)} R-groups)")

    # Difficulty-aware routing
    if difficulty == "LOW":
        # Symbolic-only, skip LM
        logger.info(f"  SubstituentAgent: LOW difficulty, symbolic-only")
    else:
        # MED/HIGH: supplement with LM
        r_groups = _extract_substituents(patent_id, manifest, data_dir)
        logger.info(f"  SubstituentAgent: {len(r_groups)} R-groups found (symbolic + LM)")

    # Agent 3: Consistency (free — RDKit only)
    consistency = {"coverage": 0}
    if example_smiles:
        consistency = _check_consistency(patent_id, scaffold_smiles, r_groups, example_smiles)
        logger.info(f"  ConsistencyAgent: {consistency['coverage']:.0%} scaffold coverage")

    # Resolve position mapping: [n*] ↔ R-label
    from .markush_mapper import resolve_position_mapping
    from .models import MarkushFormula, RGroupDef

    # Build temporary MarkushFormula for mapping resolution
    temp_rgroups = {
        label: RGroupDef(label=label, options_text=opts)
        for label, opts in r_groups.items()
    }
    temp_formula = MarkushFormula(
        patent_id=patent_id, core_smiles=scaffold_smiles, r_groups=temp_rgroups,
    )
    position_mapping = resolve_position_mapping(
        temp_formula,
        lm_mapping=lm_position_mapping,
        example_compounds=[(s, s) for s in (example_smiles or [])[:5]],
    )
    if position_mapping:
        logger.info(f"  PositionMapping: {position_mapping}")
    else:
        logger.info(f"  PositionMapping: unresolved (using positional heuristic)")

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
        position_mapping=position_mapping,
        quality=quality,
    )
