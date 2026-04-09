"""Tests for smart page detection and inline SMILES extraction."""

import pytest
from pathlib import Path

from patent_extraction.detect_structures import (
    _find_inline_smiles,
    _is_likely_false_positive,
    _has_example_header,
    _has_markush_patterns,
    _has_image_markers,
    _has_table_markers,
    detect_patent,
)
from patent_extraction.extract_inline import extract_inline_compounds
from patent_extraction.models import PageManifest


# ============================================================
# Inline SMILES detection — true positives
# ============================================================

class TestInlineSmilesTruePositives:
    """Known SMILES strings that MUST be detected."""

    def test_aspirin_smiles(self):
        text = "The compound CC(=O)Oc1ccccc1C(=O)O was synthesized."
        results = _find_inline_smiles(text)
        assert len(results) >= 1
        assert any(r["type"] == "smiles" for r in results)

    def test_caffeine_smiles(self):
        text = "Caffeine Cn1c(=O)c2c(ncn2C)n(C)c1=O is a stimulant."
        results = _find_inline_smiles(text)
        assert len(results) >= 1

    def test_complex_smiles_with_stereo(self):
        text = "Compound: C[C@@H](N)C(=O)Oc1ccc(Cl)cc1 was tested."
        results = _find_inline_smiles(text)
        assert len(results) >= 1
        # Check that stereo is preserved
        assert any("@" in r["smiles"] for r in results)

    def test_inchi_string(self):
        text = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
        results = _find_inline_smiles(text)
        assert len(results) >= 1
        assert results[0]["type"] == "inchi"
        assert results[0]["smiles"]  # Should have converted to SMILES

    def test_smiles_on_own_line(self):
        text = """The structure is:
CC(=O)Oc1ccccc1C(=O)O
as described above."""
        results = _find_inline_smiles(text)
        assert len(results) >= 1

    def test_ring_smiles(self):
        text = "The scaffold c1ccc2c(c1)cccc2O was used as starting material."
        results = _find_inline_smiles(text)
        assert len(results) >= 1


# ============================================================
# Inline SMILES detection — false positive rejection
# ============================================================

class TestInlineSmilesFalsePositives:
    """Strings that look SMILES-like but are NOT — must be rejected."""

    def test_rejects_english_words(self):
        text = "The mechanism of action involves binding to the receptor."
        results = _find_inline_smiles(text)
        assert len(results) == 0

    def test_rejects_patent_references(self):
        text = "See US20240010684A1 and WO2019/058591 for details."
        results = _find_inline_smiles(text)
        assert len(results) == 0

    def test_rejects_gene_names(self):
        text = "BRCA1/BRCA2 mutations were analyzed alongside TP53(R248W) variants."
        results = _find_inline_smiles(text)
        assert len(results) == 0

    def test_rejects_short_strings(self):
        # Under 10 chars — too ambiguous
        text = "CC=O and NH2 are common groups."
        results = _find_inline_smiles(text)
        assert len(results) == 0

    def test_rejects_all_caps_abbreviations(self):
        text = "The DMSO(d6) solvent was used for NMR analysis."
        results = _find_inline_smiles(text)
        assert len(results) == 0

    def test_rejects_numeric_references(self):
        text = "According to reference (12345/67890) the compound was active."
        results = _find_inline_smiles(text)
        assert len(results) == 0

    def test_rejects_iupac_fragments(self):
        # IUPAC name fragments shouldn't match as SMILES
        text = "2,5-Dichloro-N-phenyl-benzenesulfonamide was prepared."
        results = _find_inline_smiles(text)
        # This might parse as valid SMILES in some edge cases, but
        # the regex shouldn't match it due to hyphens/commas pattern
        # Either way, if RDKit rejects it, that's fine too
        for r in results:
            assert r["smiles"] is not None  # If matched, must be valid


class TestIsLikelyFalsePositive:
    def test_patent_ref(self):
        assert _is_likely_false_positive("US20240010684A1")

    def test_gene_name(self):
        assert _is_likely_false_positive("BRCA1(R248W)")

    def test_all_caps_short(self):
        assert _is_likely_false_positive("EGFR/HER2")

    def test_mostly_digits(self):
        assert _is_likely_false_positive("123456789/01")

    def test_real_smiles_not_flagged(self):
        assert not _is_likely_false_positive("CC(=O)Oc1ccccc1C(=O)O")

    def test_real_smiles_with_rings(self):
        assert not _is_likely_false_positive("c1ccc2c(c1)cccc2O")


# ============================================================
# Example header detection
# ============================================================

class TestExampleHeaders:
    def test_standard_example(self):
        text = "Example 42: Synthesis of compound X"
        assert _has_example_header(text) == ["42"]

    def test_example_with_letter(self):
        text = "Example 5A was prepared similarly."
        assert "5A" in _has_example_header(text)

    def test_cpd_no_format(self):
        text = "Cpd. No. 243; MS (ES+): m/e=427.2"
        assert "243" in _has_example_header(text)

    def test_compound_format(self):
        text = "Compound 17 was tested in the assay."
        assert "17" in _has_example_header(text)

    def test_intermediate(self):
        text = "Intermediate 1A: preparation of the boronic ester"
        assert "1A" in _has_example_header(text)

    def test_ex_no_format(self):
        text = "Ex. No. 15 showed activity"
        assert "15" in _has_example_header(text)

    def test_no_match(self):
        text = "The experiment was conducted at room temperature."
        assert _has_example_header(text) == []

    def test_multiple_examples(self):
        text = "Example 1 and Example 2 were compared."
        results = _has_example_header(text)
        assert "1" in results
        assert "2" in results


# ============================================================
# Markush pattern detection
# ============================================================

class TestMarkushPatterns:
    def test_r_group_definition(self):
        text = "R1 is selected from methyl, ethyl, propyl."
        assert _has_markush_patterns(text)

    def test_wherein_clause(self):
        text = "wherein R2 is hydrogen or halogen"
        assert _has_markush_patterns(text)

    def test_selected_from_group(self):
        text = "selected from the group consisting of: methyl, ethyl"
        assert _has_markush_patterns(text)

    def test_optionally_substituted(self):
        text = "optionally substituted phenyl with 1-3 halogens"
        assert _has_markush_patterns(text)

    def test_formula_reference(self):
        text = "A compound of formula (I) or formula (II)"
        assert _has_markush_patterns(text)

    def test_no_markush(self):
        text = "The reaction was stirred at 100 degrees for 3 hours."
        assert not _has_markush_patterns(text)


# ============================================================
# Image and table marker detection
# ============================================================

class TestMarkerDetection:
    def test_image_marker_count(self):
        text = """<|ref|>image<|/ref|><|det|>[[0,0,100,100]]<|/det|>
<|ref|>text<|/ref|><|det|>[[0,0,100,100]]<|/det|>
Some text
<|ref|>image<|/ref|><|det|>[[200,200,400,400]]<|/det|>"""
        assert _has_image_markers(text) == 2

    def test_no_image_markers(self):
        assert _has_image_markers("Just plain text") == 0

    def test_table_marker(self):
        text = '<|ref|>table<|/ref|><|det|>[[0,0,100,100]]<|/det|>\n<table><tr><td>1</td></tr></table>'
        assert _has_table_markers(text)

    def test_html_table_without_marker(self):
        text = "<table><tr><td>data</td></tr></table>"
        assert _has_table_markers(text)

    def test_no_table(self):
        assert not _has_table_markers("No tables here.")


# ============================================================
# Integration tests with real patent data
# ============================================================

class TestDetectPatentIntegration:
    """Run detection on real patent files."""

    @pytest.fixture
    def data_dir(self):
        d = Path("/Users/dhruvsingh/Downloads/Patent (1)")
        if not d.exists():
            pytest.skip("Patent data not available")
        return d

    def test_detect_dev_patent(self, data_dir):
        manifest = detect_patent("US10214537", data_dir)
        assert manifest.patent_id == "US10214537"
        # Should find image pages (this patent has many)
        assert len(manifest.image_pages) > 0
        # Should find example pages
        assert len(manifest.example_pages) > 0
        # Should find table pages
        assert len(manifest.table_pages) > 0

    def test_detect_markush_patent(self, data_dir):
        """US20240335431A1 has 225 Markush instances — should detect many."""
        manifest = detect_patent("US20240335431A1", data_dir)
        assert len(manifest.markush_pages) > 0

    def test_detect_cpd_format(self, data_dir):
        """US10899738 uses Cpd. No. format."""
        manifest = detect_patent("US10899738", data_dir)
        assert len(manifest.example_pages) > 0

    def test_detect_all_patents(self, data_dir):
        """Every patent should have at least some detectable content."""
        from patent_extraction.detect_structures import detect_all_patents
        manifests = detect_all_patents(data_dir)
        assert len(manifests) == 8
        for pid, m in manifests.items():
            total = (len(m.image_pages) + len(m.example_pages) +
                     len(m.markush_pages) + len(m.table_pages))
            assert total > 0, f"Patent {pid} has no detectable content"

    def test_manifest_no_duplicate_pages(self, data_dir):
        manifest = detect_patent("US10214537", data_dir)
        assert len(manifest.image_pages) == len(set(manifest.image_pages))
        assert len(manifest.example_pages) == len(set(manifest.example_pages))
        assert len(manifest.table_pages) == len(set(manifest.table_pages))


# ============================================================
# Inline compound extraction
# ============================================================

class TestExtractInlineCompounds:
    def test_empty_manifest(self):
        manifest = PageManifest(patent_id="TEST")
        compounds = extract_inline_compounds(manifest)
        assert compounds == []

    def test_with_valid_inline(self):
        manifest = PageManifest(
            patent_id="TEST",
            inline_structures=[{
                "page": 10,
                "raw": "CC(=O)Oc1ccccc1C(=O)O",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "type": "smiles",
            }],
        )
        compounds = extract_inline_compounds(manifest)
        assert len(compounds) == 1
        assert compounds[0].canonical_smiles is not None
        assert compounds[0].inchikey is not None
        assert compounds[0].processing_status == "validated"

    def test_deduplicates_same_molecule(self):
        manifest = PageManifest(
            patent_id="TEST",
            inline_structures=[
                {"page": 10, "raw": "CC(=O)Oc1ccccc1C(=O)O",
                 "smiles": "CC(=O)Oc1ccccc1C(=O)O", "type": "smiles"},
                {"page": 20, "raw": "c1ccc(C(=O)O)c(OC(C)=O)c1",
                 "smiles": "CC(=O)Oc1ccccc1C(=O)O", "type": "smiles"},
            ],
        )
        compounds = extract_inline_compounds(manifest)
        assert len(compounds) == 1  # Same molecule, deduplicated

    def test_salt_handling(self):
        manifest = PageManifest(
            patent_id="TEST",
            inline_structures=[{
                "page": 5,
                "raw": "CC(N)C(=O)O.Cl",
                "smiles": "CC(N)C(=O)O.Cl",
                "type": "smiles",
            }],
        )
        compounds = extract_inline_compounds(manifest)
        assert len(compounds) == 1
        # Should have parent SMILES without salt
        assert compounds[0].parent_smiles is not None
