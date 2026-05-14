"""Parser for the custom markdown annotation format used in patent extractions.

Format: <|ref|>element_type<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
Followed by content text until the next marker or end of file.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import ParsedElement

# Matches: <|ref|>TYPE<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
# Allows optional whitespace, handles truncated trailing markers
MARKER_PATTERN = re.compile(
    r'<\|ref\|>(\w+)<\|/ref\|>'
    r'<\|det\|>'
    r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
    r'<\|/det\|>',
    re.MULTILINE
)


def parse_markdown_page(text: str) -> list[ParsedElement]:
    """Parse a markdown page into structured elements.

    Args:
        text: Raw markdown content of a page file.

    Returns:
        List of ParsedElement with type, bounding box, and associated text content.
    """
    elements: list[ParsedElement] = []
    matches = list(MARKER_PATTERN.finditer(text))

    for i, match in enumerate(matches):
        element_type = match.group(1)
        bbox = [int(match.group(j)) for j in range(2, 6)]

        # Content is everything between this marker's end and the next marker's start
        content_start = match.end()
        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            content_end = len(text)

        content = text[content_start:content_end].strip()

        elements.append(ParsedElement(
            element_type=element_type,
            bbox=bbox,
            content=content,
        ))

    return elements


def parse_markdown_file(filepath: Path) -> list[ParsedElement]:
    """Parse a markdown file into structured elements.

    Args:
        filepath: Path to the .md file.

    Returns:
        List of ParsedElement.
    """
    text = filepath.read_text(encoding="utf-8")
    return parse_markdown_page(text)


def extract_page_number(filepath: Path) -> int | None:
    """Extract page number from filename like page_0042.md.

    Args:
        filepath: Path to the page file.

    Returns:
        Page number as int, or None if not parseable.
    """
    match = re.search(r'page_(\d+)', filepath.stem)
    if match:
        return int(match.group(1))
    return None


def get_text_content(elements: list[ParsedElement]) -> str:
    """Concatenate all text-type element content.

    Args:
        elements: Parsed elements from a page.

    Returns:
        Combined text content.
    """
    return "\n".join(
        el.content for el in elements
        if el.element_type == "text" and el.content
    )


def get_image_bboxes(elements: list[ParsedElement]) -> list[list[int]]:
    """Extract bounding boxes for all image elements.

    Args:
        elements: Parsed elements from a page.

    Returns:
        List of [x1, y1, x2, y2] bounding boxes.
    """
    return [el.bbox for el in elements if el.element_type == "image"]


def get_table_content(elements: list[ParsedElement]) -> list[str]:
    """Extract raw HTML table content.

    Args:
        elements: Parsed elements from a page.

    Returns:
        List of table HTML strings.
    """
    return [el.content for el in elements if el.element_type == "table" and el.content]
