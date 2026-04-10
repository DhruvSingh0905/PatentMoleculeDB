"""RDKit utilities for SMILES validation, canonicalization, InChIKey, and salt handling."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, inchi
from rdkit.Chem.SaltRemover import SaltRemover

from .models import DrugLikenessResult
from . import config

# Initialize salt remover and PAINS filter once
_salt_remover = SaltRemover()

_pains_params = FilterCatalog.FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_pains_catalog = FilterCatalog.FilterCatalog(_pains_params)


def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid using RDKit.

    Args:
        smiles: SMILES string to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> str | None:
    """Convert SMILES to canonical form.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_roundtrip(smiles: str) -> bool:
    """Check that SMILES survives a round-trip through RDKit.

    Args:
        smiles: Canonical SMILES to test.

    Returns:
        True if MolToSmiles(MolFromSmiles(s)) == s.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return Chem.MolToSmiles(mol, canonical=True) == smiles


def get_inchikey(smiles: str) -> str | None:
    """Generate InChIKey from SMILES.

    Args:
        smiles: SMILES string.

    Returns:
        InChIKey string, or None if conversion fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi_str = inchi.MolToInchi(mol)
    if inchi_str is None:
        return None
    return inchi.InchiToInchiKey(inchi_str)


def get_connectivity_key(inchikey: str) -> str:
    """Extract the 14-character connectivity layer from an InChIKey.

    This allows stereo-agnostic matching.

    Args:
        inchikey: Full 27-character InChIKey.

    Returns:
        First 14 characters (connectivity layer).
    """
    return inchikey[:14]


def strip_salt(smiles: str) -> dict:
    """Remove salt/counterion from a SMILES and return both forms.

    Args:
        smiles: SMILES string (may contain salt as dot-separated components).

    Returns:
        Dict with canonical_smiles, parent_smiles, inchikey, parent_inchikey, is_salt.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "canonical_smiles": None,
            "parent_smiles": None,
            "inchikey": None,
            "parent_inchikey": None,
            "is_salt": False,
        }

    parent = _salt_remover.StripMol(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    parent_smiles = Chem.MolToSmiles(parent, canonical=True)

    return {
        "canonical_smiles": canonical,
        "parent_smiles": parent_smiles,
        "inchikey": get_inchikey(canonical),
        "parent_inchikey": get_inchikey(parent_smiles),
        "is_salt": mol.GetNumAtoms() != parent.GetNumAtoms(),
    }


def passes_lipinski(mol) -> bool:
    """Check Lipinski Rule of Five.

    Args:
        mol: RDKit Mol object.

    Returns:
        True if the molecule passes Lipinski (allows 1 violation).
    """
    violations = 0
    if Descriptors.MolWt(mol) > config.LIPINSKI_MW_MAX:
        violations += 1
    if Descriptors.MolLogP(mol) > config.LIPINSKI_LOGP_MAX:
        violations += 1
    if Descriptors.NumHDonors(mol) > config.LIPINSKI_HBD_MAX:
        violations += 1
    if Descriptors.NumHAcceptors(mol) > config.LIPINSKI_HBA_MAX:
        violations += 1
    return violations <= 1  # Lipinski allows 1 violation


def passes_pains(mol) -> bool:
    """Check for PAINS (pan-assay interference compounds) alerts.

    Args:
        mol: RDKit Mol object.

    Returns:
        True if the molecule has NO PAINS alerts (is clean).
    """
    return not _pains_catalog.HasMatch(mol)


def compute_drug_likeness(smiles: str) -> DrugLikenessResult | None:
    """Compute all drug-likeness metrics for a SMILES.

    Args:
        smiles: SMILES string.

    Returns:
        DrugLikenessResult, or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # Synthetic accessibility score (Ertl)
    try:
        from rdkit.Chem import RDConfig
        import sys, os
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
        sa_score = sascorer.calculateScore(mol)
    except ImportError:
        logging.getLogger(__name__).warning(
            "SA_Score module not available. SA scores will default to 0.0. "
            "Install from RDKit Contrib or set RDBASE."
        )
        sa_score = 0.0
    except Exception as e:
        logging.getLogger(__name__).warning(f"SA_Score calculation failed: {e}")
        sa_score = 0.0

    return DrugLikenessResult(
        mw=round(mw, 2),
        logp=round(logp, 2),
        hbd=hbd,
        hba=hba,
        lipinski_passes=passes_lipinski(mol),
        pains_clean=passes_pains(mol),
        sa_score=round(sa_score, 2),
    )


def molecular_weight(smiles: str) -> float | None:
    """Get molecular weight from SMILES.

    Args:
        smiles: SMILES string.

    Returns:
        Molecular weight in Da, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)
