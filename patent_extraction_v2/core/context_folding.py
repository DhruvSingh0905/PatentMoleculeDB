"""Context folding — two-pass extraction for token-efficient processing.

Pass 1: Fast local scan of ALL pages to classify sections and estimate
compound density. No API calls. ~0.5s per patent.

Pass 2: Allocate extraction budget proportionally — dense example sections
get full Claude extraction, sparse sections get regex-only, irrelevant
sections (background, prior art, sequence listings) are skipped entirely.

Typical savings: 60-80% token reduction on long patents (>100 pages).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


@dataclass
class SectionInfo:
    """Metadata about a contiguous section of pages."""
    section_type: str  # "examples", "markush", "bioactivity", "background", "claims", "other"
    pages: list[int] = field(default_factory=list)
    compound_density: float = 0.0  # Estimated compounds per page
    has_tables: bool = False
    has_structures: bool = False
    char_count: int = 0
    estimated_tokens: int = 0
    priority: str = "normal"  # "high", "normal", "low", "skip"


@dataclass
class PatentSections:
    """Structured representation of a patent's content sections."""
    patent_id: str
    total_pages: int = 0
    sections: list[SectionInfo] = field(default_factory=list)
    example_pages: list[int] = field(default_factory=list)
    markush_pages: list[int] = field(default_factory=list)
    bioactivity_pages: list[int] = field(default_factory=list)
    skip_pages: list[int] = field(default_factory=list)

    @property
    def pages_to_process(self) -> list[int]:
        """Pages that need Claude extraction (high + normal priority)."""
        result = set()
        for s in self.sections:
            if s.priority in ("high", "normal"):
                result.update(s.pages)
        return sorted(result)

    @property
    def token_estimate(self) -> int:
        """Estimated total tokens for pages_to_process."""
        return sum(s.estimated_tokens for s in self.sections if s.priority in ("high", "normal"))

    @property
    def savings_pct(self) -> float:
        """Percentage of tokens saved by skipping low-priority sections."""
        total = sum(s.estimated_tokens for s in self.sections)
        kept = self.token_estimate
        return (1 - kept / max(total, 1)) * 100


def scan_patent_sections(
    patent_id: str,
    data_dir: Path | None = None,
) -> PatentSections:
    """Pass 1: Fast local scan to classify all pages. No API calls.

    Classifies each page into sections and assigns priority:
    - HIGH: Dense example pages, compound tables, bioactivity tables
    - NORMAL: Sparse examples, claims with compounds
    - LOW: Abstract, background, general methods
    - SKIP: Prior art references, sequence listings, patent formalities
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    pages_dir = data_dir / patent_id / "all_pages"
    if not pages_dir.exists():
        return PatentSections(patent_id=patent_id)

    page_files = sorted(pages_dir.glob("page_*.md"))
    result = PatentSections(patent_id=patent_id, total_pages=len(page_files))

    # Classify each page
    page_classifications = {}
    for pf in page_files:
        page_num = int(re.search(r'page_(\d+)', pf.stem).group(1))
        text = pf.read_text(encoding="utf-8")
        char_count = len(text)

        classification = _classify_page(text)
        classification["page_num"] = page_num
        classification["char_count"] = char_count
        classification["estimated_tokens"] = char_count // 4  # Rough estimate
        page_classifications[page_num] = classification

    # Group consecutive pages into sections
    current_type = None
    current_section = None

    for page_num in sorted(page_classifications.keys()):
        info = page_classifications[page_num]
        page_type = info["type"]

        if page_type != current_type:
            if current_section:
                result.sections.append(current_section)
            current_section = SectionInfo(
                section_type=page_type,
                pages=[page_num],
                has_tables=info.get("has_tables", False),
                has_structures=info.get("has_structures", False),
                char_count=info["char_count"],
                estimated_tokens=info["estimated_tokens"],
                compound_density=info.get("compound_count", 0),
            )
            current_type = page_type
        else:
            current_section.pages.append(page_num)
            current_section.char_count += info["char_count"]
            current_section.estimated_tokens += info["estimated_tokens"]
            current_section.compound_density += info.get("compound_count", 0)
            current_section.has_tables = current_section.has_tables or info.get("has_tables", False)
            current_section.has_structures = current_section.has_structures or info.get("has_structures", False)

    if current_section:
        result.sections.append(current_section)

    # Assign priorities
    for section in result.sections:
        if section.section_type == "examples":
            section.priority = "high"
            result.example_pages.extend(section.pages)
        elif section.section_type == "bioactivity":
            section.priority = "high"
            result.bioactivity_pages.extend(section.pages)
        elif section.section_type == "markush":
            section.priority = "normal"
            result.markush_pages.extend(section.pages)
        elif section.section_type == "claims":
            section.priority = "normal"
        elif section.section_type in ("background", "references", "formalities"):
            section.priority = "skip"
            result.skip_pages.extend(section.pages)
        else:
            section.priority = "low"

    logger.info(
        f"Context folding {patent_id}: {result.total_pages} pages → "
        f"{len(result.pages_to_process)} to process "
        f"({result.savings_pct:.0f}% token savings, "
        f"~{result.token_estimate:,} tokens estimated)"
    )

    return result


def _classify_page(text: str) -> dict:
    """Classify a single page by content type."""
    text_lower = text.lower()
    stripped = re.sub(r'<[^>]+>', '', text)

    result = {
        "type": "other",
        "compound_count": 0,
        "has_tables": "<table>" in text_lower,
        "has_structures": "<|ref|>image" in text,
    }

    # Count compound identifiers
    examples = len(re.findall(r'Example\s+\d+', text, re.IGNORECASE))
    cpd_nos = len(re.findall(r'Cpd\.?\s*(?:No\.?)?\s*\d+', text, re.IGNORECASE))
    result["compound_count"] = examples + cpd_nos

    # Classification rules (order matters — first match wins)
    if re.search(r'IC50|Ki\b|EC50|binding.*affinity|inhibit.*activity', text, re.IGNORECASE):
        if result["has_tables"]:
            result["type"] = "bioactivity"
    elif result["compound_count"] >= 2:
        result["type"] = "examples"
    elif re.search(r'Formula\s*\(?[IVX]+\)?|R\d+\s+is\s+(?:selected|chosen)', text, re.IGNORECASE):
        result["type"] = "markush"
    elif re.search(r'(?:claim|claims)\s+\d', text_lower):
        result["type"] = "claims"
    elif re.search(r'prior art|background|field of (?:the )?invention|references cited', text_lower):
        result["type"] = "background"
    elif re.search(r'sequence listing|<400>|SEQ ID', text):
        result["type"] = "formalities"
    elif re.search(r'(?:19|20)\d{2}[;,]\s+\d+|(?:J\.|Chem\.|Biol\.|Med\.)', text):
        # Literature references section
        if result["compound_count"] == 0:
            result["type"] = "references"

    return result
