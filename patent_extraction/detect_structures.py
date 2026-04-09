"""Smart page detection — scan patent files to build processing manifests.

Classifies each page by content type (images, examples, Markush, tables, inline SMILES)
so we only send relevant pages to the API. No API calls — runs entirely locally.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rdkit import Chem

from . import config
from .markdown_parser import parse_markdown_file, extract_page_number
from .models import PageManifest

logger = logging.getLogger(__name__)

# --- Example/Compound header patterns (patent-specific) ---

EXAMPLE_PATTERNS = [
    re.compile(r'Example\s+(\d+[A-Za-z]?)', re.IGNORECASE),
    re.compile(r'Cpd\.?\s*No\.?\s*(\d+)', re.IGNORECASE),
    re.compile(r'Compound\s+(\d+[A-Za-z]?)', re.IGNORECASE),
    re.compile(r'Intermediate\s+(\d+[A-Za-z]?)', re.IGNORECASE),
    re.compile(r'Ex\.?\s*No\.?\s*(\d+)', re.IGNORECASE),
]

# --- Markush patterns ---

MARKUSH_PATTERNS = [
    re.compile(r'R\d+\s+is\b', re.IGNORECASE),
    re.compile(r'wherein\s+R\d?', re.IGNORECASE),
    re.compile(r'selected\s+from\s+the\s+group', re.IGNORECASE),
    re.compile(r'optionally\s+substituted', re.IGNORECASE),
    re.compile(r'formula\s+\(?[IVX]+\)?', re.IGNORECASE),
]

# --- Inline SMILES/InChI detection ---

# InChI is unambiguous
INCHI_PATTERN = re.compile(r'InChI=1S?/[A-Za-z0-9/+\-\(\),\.]+')

# SMILES candidates: ≥10 chars, surrounded by whitespace/line boundaries
# Must contain at least one SMILES-specific character: ()[]=#@/\
SMILES_CANDIDATE_PATTERN = re.compile(
    r'(?:^|\s)([A-Za-z0-9@+\-\[\]\(\)\\/#=.]{10,})(?:\s|$)',
    re.MULTILINE,
)
SMILES_REQUIRED_CHARS = re.compile(r'[\(\)\[\]=@#/\\]')


def _find_inline_smiles(text: str) -> list[dict]:
    """Find inline SMILES and InChI strings in text.

    Uses tight heuristic: ≥10 chars, must contain SMILES-specific characters,
    must pass RDKit validation with ≥5 atoms. Rejects English words, gene names, etc.
    """
    results = []
    seen = set()

    # InChI (unambiguous)
    for match in INCHI_PATTERN.finditer(text):
        inchi_str = match.group(0)
        if inchi_str not in seen:
            mol = Chem.MolFromInchi(inchi_str)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                seen.add(inchi_str)
                results.append({
                    "raw": inchi_str,
                    "smiles": smiles,
                    "type": "inchi",
                    "start": match.start(),
                })

    # SMILES candidates
    for match in SMILES_CANDIDATE_PATTERN.finditer(text):
        candidate = match.group(1)
        if candidate in seen:
            continue

        # Must contain SMILES-specific characters
        if not SMILES_REQUIRED_CHARS.search(candidate):
            continue

        # Reject if it looks like a common non-SMILES pattern
        # Patent refs, gene names, etc.
        if _is_likely_false_positive(candidate):
            continue

        # RDKit validation — the definitive filter
        mol = Chem.MolFromSmiles(candidate)
        if mol is not None and mol.GetNumAtoms() >= 5:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            seen.add(candidate)
            results.append({
                "raw": candidate,
                "smiles": canonical,
                "type": "smiles",
                "start": match.start(),
            })

    return results


def _is_likely_false_positive(candidate: str) -> bool:
    """Reject strings that look like SMILES but aren't.

    Filters out patent reference numbers, gene names, abbreviations, etc.
    """
    # All uppercase with no lowercase — likely an abbreviation or gene name
    # (real SMILES with aromatic rings use lowercase: c, n, o, s)
    if candidate.isupper() and len(candidate) < 20:
        return True

    # Starts with common non-SMILES prefixes
    if re.match(r'^(US|EP|WO|PCT|PDB|BRCA|TP53|EGFR|HER|CDK|JAK)\d', candidate, re.IGNORECASE):
        return True

    # Mostly digits with few chemical characters — likely a patent/reference number
    digit_ratio = sum(c.isdigit() for c in candidate) / len(candidate)
    if digit_ratio > 0.7:
        return True

    return False


def _has_example_header(text: str) -> list[str]:
    """Find example/compound headers in text. Returns list of matched IDs."""
    matches = []
    for pattern in EXAMPLE_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(match.group(1))
    return matches


def _has_markush_patterns(text: str) -> bool:
    """Check if text contains Markush structure definitions."""
    return any(p.search(text) for p in MARKUSH_PATTERNS)


def _has_image_markers(text: str) -> int:
    """Count image markers in raw text."""
    return text.count('<|ref|>image<|/ref|>')


def _has_table_markers(text: str) -> bool:
    """Check if text contains table markers."""
    return '<|ref|>table<|/ref|>' in text or '<table>' in text.lower()


def detect_patent(patent_id: str, data_dir: Path | None = None) -> PageManifest:
    """Scan a patent's files and build a processing manifest.

    Args:
        patent_id: Patent identifier (e.g., "US10214537").
        data_dir: Root data directory. Defaults to config.DATA_DIR.

    Returns:
        PageManifest with classified pages.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    patent_dir = data_dir / patent_id
    manifest = PageManifest(patent_id=patent_id)

    if not patent_dir.exists():
        logger.warning(f"Patent directory not found: {patent_dir}")
        return manifest

    # Scan all_pages/ for all content types
    _seen_image = set()
    all_pages_dir = patent_dir / "all_pages"
    if all_pages_dir.exists():
        for page_file in sorted(all_pages_dir.glob("page_*.md")):
            page_num = extract_page_number(page_file)
            if page_num is None:
                continue

            text = page_file.read_text(encoding="utf-8")

            # Image markers
            if _has_image_markers(text) > 0 and page_num not in _seen_image:
                manifest.image_pages.append(page_num)
                _seen_image.add(page_num)

            # Example headers
            if _has_example_header(text):
                manifest.example_pages.append(page_num)

            # Markush patterns
            if _has_markush_patterns(text):
                manifest.markush_pages.append(page_num)

            # Inline SMILES/InChI
            inline = _find_inline_smiles(text)
            if inline:
                manifest.inline_smiles_pages.append(page_num)
                for item in inline:
                    manifest.inline_structures.append({
                        "page": page_num,
                        "raw": item["raw"],
                        "smiles": item["smiles"],
                        "type": item["type"],
                    })

    # Scan table/ directory
    table_dir = patent_dir / "table"
    if table_dir.exists():
        for page_file in sorted(table_dir.glob("page_*.md")):
            page_num = extract_page_number(page_file)
            if page_num is not None:
                manifest.table_pages.append(page_num)

    # Scan iupacs_clean/ for additional example pages not in all_pages
    iupacs_dir = patent_dir / "iupacs_clean"
    if iupacs_dir.exists():
        existing_example_pages = set(manifest.example_pages)
        for page_file in sorted(iupacs_dir.glob("page_*.md")):
            page_num = extract_page_number(page_file)
            if page_num is None:
                continue
            if page_num not in existing_example_pages:
                text = page_file.read_text(encoding="utf-8")
                if _has_example_header(text):
                    manifest.example_pages.append(page_num)

        # Sort after adding
        manifest.example_pages = sorted(set(manifest.example_pages))

    logger.info(
        f"Patent {patent_id}: "
        f"{len(manifest.image_pages)} image pages, "
        f"{len(manifest.example_pages)} example pages, "
        f"{len(manifest.markush_pages)} markush pages, "
        f"{len(manifest.table_pages)} table pages, "
        f"{len(manifest.inline_structures)} inline structures"
    )

    return manifest


def detect_all_patents(data_dir: Path | None = None) -> dict[str, PageManifest]:
    """Scan all patents and return manifests.

    Args:
        data_dir: Root data directory.

    Returns:
        Dict of patent_id -> PageManifest.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    manifests = {}
    for patent_id in config.PATENT_IDS:
        manifests[patent_id] = detect_patent(patent_id, data_dir)

    return manifests


def save_manifests(manifests: dict[str, PageManifest], output_dir: Path | None = None):
    """Save manifests to JSON files.

    Args:
        manifests: Dict of patent_id -> PageManifest.
        output_dir: Output directory. Defaults to config.OUTPUT_DIR.
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    for patent_id, manifest in manifests.items():
        path = output_dir / f"{patent_id}_manifest.json"
        with open(path, "w") as f:
            json.dump(manifest.model_dump(), f, indent=2)
        logger.info(f"Saved manifest: {path}")
