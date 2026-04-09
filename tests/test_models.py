"""Tests for data models."""

import pytest
from patent_extraction.models import (
    AssayResult,
    Compound,
    CompoundSource,
    ConfidenceTier,
    DrugLikenessResult,
    MarkushDefinition,
    PageManifest,
    ParsedElement,
    PatentResult,
)


class TestParsedElement:
    def test_basic_creation(self):
        el = ParsedElement(element_type="text", bbox=[100, 200, 300, 400], content="hello")
        assert el.element_type == "text"
        assert el.bbox == [100, 200, 300, 400]
        assert el.content == "hello"

    def test_image_element(self):
        el = ParsedElement(element_type="image", bbox=[0, 0, 500, 500])
        assert el.element_type == "image"
        assert el.content == ""

    def test_table_element(self):
        el = ParsedElement(
            element_type="table",
            bbox=[10, 10, 800, 900],
            content="<table><tr><td>data</td></tr></table>",
        )
        assert "table" in el.content


class TestCompound:
    def test_minimal_creation(self):
        c = Compound(patent_id="US10214537")
        assert c.patent_id == "US10214537"
        assert c.processing_status == "pending"
        assert c.confidence_tier == ConfidenceTier.SINGLE_UNVALIDATED
        assert c.source == CompoundSource.EXEMPLIFIED
        assert c.assay_data == []

    def test_full_compound(self):
        c = Compound(
            patent_id="US10214537",
            example_number="42",
            iupac_name="2,5-Dichloro-N-phenyl-benzenesulfonamide",
            smiles_from_text="O=S(=O)(Nc1ccccc1)c1cc(Cl)ccc1Cl",
            smiles_from_image="O=S(=O)(Nc1ccccc1)c1cc(Cl)ccc1Cl",
            canonical_smiles="O=S(=O)(Nc1ccccc1)c1cc(Cl)ccc1Cl",
            parent_smiles="O=S(=O)(Nc1ccccc1)c1cc(Cl)ccc1Cl",
            inchikey="ABCDEFGHIJKLMN-OPQRSTUVWX",
            confidence_tier=ConfidenceTier.DUAL_CONFIRMED,
            source=CompoundSource.EXEMPLIFIED,
            source_page=30,
            assay_data=[
                AssayResult(assay_name="SGK-1 IC50", value_raw="<0.0012", unit="uM",
                           qualifier="<", value_numeric=0.0012)
            ],
        )
        assert c.confidence_tier == 1
        assert len(c.assay_data) == 1
        assert c.assay_data[0].qualifier == "<"

    def test_markush_compound(self):
        c = Compound(
            patent_id="US10214537",
            source=CompoundSource.MARKUSH_ENUMERATED,
            markush_parent="Formula I",
            r_groups={"1": "methyl", "2": "F"},
            canonical_smiles="Cc1ccc(F)cc1",
        )
        assert c.source == "markush_enumerated"
        assert c.r_groups["1"] == "methyl"


class TestAssayResult:
    def test_numeric_value(self):
        a = AssayResult(assay_name="IC50", value_raw="0.043", value_numeric=0.043, unit="uM")
        assert a.value_numeric == 0.043

    def test_qualified_value(self):
        a = AssayResult(assay_name="CC50", value_raw=">10", qualifier=">",
                       value_numeric=10.0, unit="uM")
        assert a.qualifier == ">"

    def test_categorical_value(self):
        a = AssayResult(assay_name="PI3K IC50", value_raw="A", unit="nM")
        assert a.value_numeric is None


class TestDrugLikenessResult:
    def test_passing_compound(self):
        d = DrugLikenessResult(
            mw=350.0, logp=2.5, hbd=2, hba=5,
            lipinski_passes=True, pains_clean=True, sa_score=3.0,
        )
        assert d.lipinski_passes
        assert d.pains_clean

    def test_failing_compound(self):
        d = DrugLikenessResult(
            mw=800.0, logp=7.0, hbd=8, hba=15,
            lipinski_passes=False, pains_clean=False, sa_score=8.0,
        )
        assert not d.lipinski_passes


class TestMarkushDefinition:
    def test_creation(self):
        m = MarkushDefinition(
            formula_name="Formula I",
            core_scaffold="c1ccc([1*])c([2*])c1",
            variables={"1": ["C", "CC"], "2": ["F", "Cl"]},
            estimated_combinations=4,
            patent_id="US10214537",
        )
        assert m.estimated_combinations == 4
        assert len(m.variables["1"]) == 2


class TestPageManifest:
    def test_creation(self):
        m = PageManifest(
            patent_id="US10214537",
            image_pages=[11, 12, 13],
            example_pages=[17, 18],
            markush_pages=[3, 4],
            table_pages=[330],
        )
        assert len(m.image_pages) == 3


class TestPatentResult:
    def test_empty_result(self):
        r = PatentResult(patent_id="US10214537")
        assert r.exemplified_compounds == []
        assert r.markush_enumerated == []


class TestConfidenceTier:
    def test_tier_values(self):
        assert ConfidenceTier.DUAL_CONFIRMED == 1
        assert ConfidenceTier.SINGLE_VALIDATED == 2
        assert ConfidenceTier.SINGLE_UNVALIDATED == 3

    def test_tier_ordering(self):
        assert ConfidenceTier.DUAL_CONFIRMED < ConfidenceTier.SINGLE_VALIDATED
        assert ConfidenceTier.SINGLE_VALIDATED < ConfidenceTier.SINGLE_UNVALIDATED
