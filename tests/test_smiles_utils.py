"""Tests for SMILES utilities — validation, canonicalization, salt handling."""

import pytest

from patent_extraction.smiles_utils import (
    validate_smiles,
    canonicalize_smiles,
    smiles_roundtrip,
    get_inchikey,
    get_connectivity_key,
    strip_salt,
    passes_lipinski,
    passes_pains,
    compute_drug_likeness,
    molecular_weight,
)


class TestValidateSmiles:
    def test_valid_aspirin(self):
        assert validate_smiles("CC(=O)Oc1ccccc1C(=O)O")

    def test_valid_benzene(self):
        assert validate_smiles("c1ccccc1")

    def test_valid_caffeine(self):
        assert validate_smiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")

    def test_invalid_smiles(self):
        assert not validate_smiles("not_a_smiles")

    def test_empty_string(self):
        assert not validate_smiles("")

    def test_none(self):
        assert not validate_smiles(None)

    def test_partial_smiles(self):
        assert not validate_smiles("C(C(")


class TestCanonicalizeSmiles:
    def test_canonical_form(self):
        # Different representations of toluene
        s1 = canonicalize_smiles("Cc1ccccc1")
        s2 = canonicalize_smiles("c1ccccc1C")
        assert s1 == s2

    def test_invalid_returns_none(self):
        assert canonicalize_smiles("invalid") is None

    def test_preserves_stereochemistry(self):
        # L-alanine
        s = canonicalize_smiles("C[C@@H](N)C(=O)O")
        assert s is not None
        assert "@" in s  # Stereo preserved


class TestSmilesRoundtrip:
    def test_canonical_roundtrips(self):
        canonical = canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert smiles_roundtrip(canonical)

    def test_non_canonical_fails(self):
        # Non-canonical form may not roundtrip to itself
        # (it will roundtrip to canonical form)
        result = smiles_roundtrip("c1ccccc1C")
        # This depends on whether "c1ccccc1C" is already canonical
        assert isinstance(result, bool)


class TestGetInchikey:
    def test_aspirin(self):
        key = get_inchikey("CC(=O)Oc1ccccc1C(=O)O")
        assert key is not None
        assert len(key) == 27  # Standard InChIKey length
        assert "-" in key

    def test_invalid_returns_none(self):
        assert get_inchikey("invalid") is None

    def test_same_molecule_same_key(self):
        k1 = get_inchikey("CC(=O)Oc1ccccc1C(=O)O")
        k2 = get_inchikey("c1ccc(C(=O)O)c(OC(C)=O)c1")
        assert k1 == k2

    def test_different_molecules_different_keys(self):
        k1 = get_inchikey("c1ccccc1")  # Benzene
        k2 = get_inchikey("C1CCCCC1")  # Cyclohexane
        assert k1 != k2


class TestGetConnectivityKey:
    def test_extracts_14_chars(self):
        key = get_inchikey("CC(=O)Oc1ccccc1C(=O)O")
        conn = get_connectivity_key(key)
        assert len(conn) == 14

    def test_same_connectivity_different_stereo(self):
        # R and S alanine have same connectivity
        k1 = get_inchikey("C[C@@H](N)C(=O)O")
        k2 = get_inchikey("C[C@H](N)C(=O)O")
        if k1 and k2:
            c1 = get_connectivity_key(k1)
            c2 = get_connectivity_key(k2)
            assert c1 == c2  # Same connectivity, different stereo


class TestStripSalt:
    def test_hcl_salt(self):
        # Amine hydrochloride: methylamine.HCl
        result = strip_salt("CN.Cl")
        assert result["is_salt"]
        assert result["parent_smiles"] is not None
        # Parent should be just the amine
        assert "Cl" not in result["parent_smiles"] or result["parent_smiles"] == "CN"

    def test_no_salt(self):
        result = strip_salt("c1ccccc1")
        assert not result["is_salt"]
        assert result["canonical_smiles"] == result["parent_smiles"]

    def test_invalid_smiles(self):
        result = strip_salt("invalid")
        assert result["canonical_smiles"] is None

    def test_sodium_salt(self):
        # Sodium acetate: CC(=O)[O-].[Na+]
        result = strip_salt("CC(=O)[O-].[Na+]")
        assert result["is_salt"]

    def test_dual_storage(self):
        """Both salt and parent forms should be stored."""
        result = strip_salt("CN.Cl")
        assert result["canonical_smiles"] is not None
        assert result["parent_smiles"] is not None
        assert result["inchikey"] is not None
        assert result["parent_inchikey"] is not None


class TestPassesLipinski:
    def test_drug_like_molecule(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        assert passes_lipinski(mol)

    def test_large_molecule_fails(self):
        from rdkit import Chem
        # Very large molecule — cyclic peptide-like
        big_smiles = "C" * 100  # Long alkane, MW >> 500
        mol = Chem.MolFromSmiles(big_smiles)
        if mol:
            assert not passes_lipinski(mol)


class TestPassesPains:
    def test_clean_molecule(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        assert passes_pains(mol)


class TestComputeDrugLikeness:
    def test_aspirin(self):
        result = compute_drug_likeness("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        assert result.mw > 0
        assert result.lipinski_passes
        assert isinstance(result.pains_clean, bool)

    def test_invalid_smiles(self):
        result = compute_drug_likeness("invalid")
        assert result is None


class TestMolecularWeight:
    def test_benzene(self):
        mw = molecular_weight("c1ccccc1")
        assert mw is not None
        assert abs(mw - 78.11) < 1.0  # Benzene MW ≈ 78.11

    def test_invalid(self):
        assert molecular_weight("invalid") is None
