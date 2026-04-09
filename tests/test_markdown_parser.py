"""Tests for markdown annotation parser."""

import pytest
from pathlib import Path

from patent_extraction.markdown_parser import (
    parse_markdown_page,
    extract_page_number,
    get_text_content,
    get_image_bboxes,
    get_table_content,
)


# Sample markdown content from actual patent files
SAMPLE_PAGE = """<|ref|>sub_title<|/ref|><|det|>[[161, 85, 440, 128]]<|/det|>
## 1 MACROCYCLIC FLU ENDONUCLEASE INHIBITORS

<|ref|>text<|/ref|><|det|>[[123, 182, 477, 246]]<|/det|>
This application is a national phase entry of International Application No. PCT/IB2019/058591.

<|ref|>image<|/ref|><|det|>[[121, 789, 400, 930]]<|/det|>

<|ref|>table<|/ref|><|det|>[[123, 123, 608, 930]]<|/det|>
<table><tr><td>Compound</td><td>IC50</td></tr><tr><td>1</td><td>0.043</td></tr></table>
"""

SAMPLE_WITH_TABLE_CAPTION = """<|ref|>table_caption<|/ref|><|det|>[[582, 138, 655, 149]]<|/det|>
TABLE 1A

<|ref|>table<|/ref|><|det|>[[375, 156, 860, 900]]<|/det|>
<table><tr><td>Compound number</td><td>EC50</td></tr></table>
"""

SAMPLE_TRUNCATED = """<|ref|>text<|/ref|><|det|>[[123, 102, 480, 128]]<|/det|>
Some text content here.

<|ref|>image<|/ref|><|det|>[[480, 270, 860, 930]]<"""


class TestParseMarkdownPage:
    def test_parses_multiple_elements(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        assert len(elements) == 4

    def test_element_types(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        types = [el.element_type for el in elements]
        assert types == ["sub_title", "text", "image", "table"]

    def test_bounding_boxes(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        assert elements[0].bbox == [161, 85, 440, 128]
        assert elements[1].bbox == [123, 182, 477, 246]
        assert elements[2].bbox == [121, 789, 400, 930]

    def test_text_content(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        assert "MACROCYCLIC" in elements[0].content
        assert "national phase" in elements[1].content

    def test_image_has_no_text(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        image_el = [el for el in elements if el.element_type == "image"][0]
        # Image content is empty or just whitespace before next marker
        assert not image_el.content or image_el.content.strip() == ""

    def test_table_content_preserved(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        table_el = [el for el in elements if el.element_type == "table"][0]
        assert "<table>" in table_el.content
        assert "IC50" in table_el.content

    def test_table_caption(self):
        elements = parse_markdown_page(SAMPLE_WITH_TABLE_CAPTION)
        assert elements[0].element_type == "table_caption"
        assert "TABLE 1A" in elements[0].content

    def test_truncated_marker_ignored(self):
        """Truncated markers at end of file should not break parsing."""
        elements = parse_markdown_page(SAMPLE_TRUNCATED)
        # Should parse the complete elements, ignore the truncated one
        assert len(elements) >= 1
        assert elements[0].element_type == "text"

    def test_empty_page(self):
        elements = parse_markdown_page("")
        assert elements == []

    def test_no_markers(self):
        elements = parse_markdown_page("Just plain text without any markers.")
        assert elements == []


class TestExtractPageNumber:
    def test_standard_format(self):
        assert extract_page_number(Path("page_0042.md")) == 42

    def test_zero_padded(self):
        assert extract_page_number(Path("page_0002.md")) == 2

    def test_large_number(self):
        assert extract_page_number(Path("page_0358.md")) == 358

    def test_no_match(self):
        assert extract_page_number(Path("readme.md")) is None

    def test_with_directory(self):
        assert extract_page_number(Path("/some/dir/page_0100.md")) == 100


class TestGetTextContent:
    def test_extracts_text_elements(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        text = get_text_content(elements)
        assert "national phase" in text
        assert "<table>" not in text  # Should not include table content

    def test_empty_elements(self):
        assert get_text_content([]) == ""


class TestGetImageBboxes:
    def test_extracts_image_bboxes(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        bboxes = get_image_bboxes(elements)
        assert len(bboxes) == 1
        assert bboxes[0] == [121, 789, 400, 930]

    def test_no_images(self):
        elements = parse_markdown_page(SAMPLE_WITH_TABLE_CAPTION)
        bboxes = get_image_bboxes(elements)
        assert bboxes == []


class TestGetTableContent:
    def test_extracts_table_html(self):
        elements = parse_markdown_page(SAMPLE_PAGE)
        tables = get_table_content(elements)
        assert len(tables) == 1
        assert "<table>" in tables[0]

    def test_no_tables(self):
        page = """<|ref|>text<|/ref|><|det|>[[0, 0, 100, 100]]<|/det|>
Just text."""
        elements = parse_markdown_page(page)
        tables = get_table_content(elements)
        assert tables == []


class TestWithRealFiles:
    """Tests that run against actual patent files (integration tests)."""

    @pytest.fixture
    def patent_dir(self):
        d = Path("/Users/dhruvsingh/Downloads/Patent (1)/US11312727")
        if not d.exists():
            pytest.skip("Patent data not available")
        return d

    def test_parse_real_page(self, patent_dir):
        page = patent_dir / "all_pages" / "page_0002.md"
        from patent_extraction.markdown_parser import parse_markdown_file
        elements = parse_markdown_file(page)
        assert len(elements) > 0
        types = {el.element_type for el in elements}
        assert "text" in types

    def test_parse_real_table_page(self, patent_dir):
        page = patent_dir / "table" / "page_0219.md"
        from patent_extraction.markdown_parser import parse_markdown_file
        elements = parse_markdown_file(page)
        tables = get_table_content(elements)
        assert len(tables) > 0
        assert "table" in tables[0].lower()

    def test_parse_real_iupac_page(self, patent_dir):
        page = patent_dir / "iupacs_clean" / "page_0011.md"
        if not page.exists():
            pytest.skip("iupacs_clean page not available")
        from patent_extraction.markdown_parser import parse_markdown_file
        elements = parse_markdown_file(page)
        # Should have image markers (synthesis examples have structure images)
        bboxes = get_image_bboxes(elements)
        # May or may not have images on this specific page, but parsing should work
        assert isinstance(bboxes, list)
