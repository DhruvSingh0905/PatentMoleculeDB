"""Markush enumerator — generate concrete molecules from Markush definitions.

For unnamed compounds defined by R-group combination tables,
this module instantiates the Markush core with specific substituents.

Uses the SUBSTITUENT_LIBRARY from markush_mapper.py for name→SMILES conversion.
"""

from __future__ import annotations

import logging
import re
from itertools import product

from rdkit import Chem

from . import config
from .models import MarkushFormula, Compound, CompoundSource, IupacSource
from .markush_mapper import lookup_substituent, SUBSTITUENT_LIBRARY
from .smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey,
    strip_salt, compute_drug_likeness, molecular_weight,
)

logger = logging.getLogger(__name__)

MIN_MW = 150
MAX_MW = 1500


def _instantiate_core(
    core_smiles: str,
    assignments: dict[str, str],
    position_mapping: dict[str, str] | None = None,
) -> str | None:
    """Graph-level substituent attachment using RDKit mol editing.

    Instead of brittle string replacement, this:
    1. Converts core SMILES to mol with dummy atoms [*]
    2. For each R-assignment, finds the dummy atom by map number
    3. Attaches the substituent fragment at the graph level
    4. Removes dummy atoms and sanitizes

    This handles ring closures, aromaticity, and valence correctly.
    """
    # Normalize core SMILES: convert [n*] to [*:n] (atom map notation)
    normalized = core_smiles
    for m in re.finditer(r'\[(\d+)\*\]', core_smiles):
        n = m.group(1)
        normalized = normalized.replace(f'[{n}*]', f'[*:{n}]')

    core_mol = Chem.RWMol(Chem.MolFromSmiles(normalized))
    if core_mol is None:
        return None

    # Build label → atom map number using position_mapping
    label_to_mapnum = {}
    if position_mapping:
        for label, pos in position_mapping.items():
            label_to_mapnum[label] = int(pos)
    else:
        # Fallback: extract numeric part of label
        for label in assignments:
            num = re.search(r'\d+', label)
            if num:
                label_to_mapnum[label] = int(num.group(0))

    # Attach each substituent
    for label, sub_name in assignments.items():
        sub_smi = lookup_substituent(sub_name)
        if sub_smi is None or sub_smi == "":
            sub_smi = "[H]"

        mapnum = label_to_mapnum.get(label)
        if mapnum is None:
            continue

        # Find the dummy atom in core with this map number
        dummy_idx = None
        for atom in core_mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == mapnum:
                dummy_idx = atom.GetIdx()
                break

        if dummy_idx is None:
            continue

        # Get the neighbor of the dummy (the attachment point in the core)
        dummy_atom = core_mol.GetAtomWithIdx(dummy_idx)
        neighbors = list(dummy_atom.GetNeighbors())
        if not neighbors:
            continue
        attach_idx = neighbors[0].GetIdx()

        # Handle hydrogen substituent: just remove the dummy
        if sub_smi == "[H]":
            core_mol.RemoveAtom(dummy_idx)
            continue

        # Parse substituent fragment
        frag_mol = Chem.MolFromSmiles(sub_smi)
        if frag_mol is None:
            continue

        # Combine core + fragment
        combo = Chem.RWMol(Chem.CombineMols(core_mol, frag_mol))

        # The fragment atoms start after the core atoms
        frag_start = core_mol.GetNumAtoms()

        # Find a good attachment atom in the fragment (first non-H atom)
        frag_attach = frag_start  # Default: first atom of fragment
        for i in range(frag_start, combo.GetNumAtoms()):
            if combo.GetAtomWithIdx(i).GetAtomicNum() > 1:
                frag_attach = i
                break

        # Add bond between core attachment point and fragment attachment point
        combo.AddBond(attach_idx, frag_attach, Chem.BondType.SINGLE)

        # Remove the dummy atom (adjust index if needed)
        # Dummy is still at dummy_idx in the combo mol
        combo.RemoveAtom(dummy_idx)

        # Update core_mol for next iteration
        try:
            Chem.SanitizeMol(combo)
            core_mol = combo
        except Exception:
            # Sanitization failed — try to continue with unsanitized mol
            core_mol = combo

    # Remove remaining dummies (unassigned R-groups → treat as H)
    atoms_to_remove = []
    for atom in core_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atoms_to_remove.append(atom.GetIdx())

    # Remove in reverse order to preserve indices
    for idx in sorted(atoms_to_remove, reverse=True):
        core_mol.RemoveAtom(idx)

    # Final sanitization and canonicalization
    try:
        Chem.SanitizeMol(core_mol)
        mol = Chem.RemoveHs(core_mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        try:
            return Chem.MolToSmiles(core_mol)
        except Exception:
            return None


def enumerate_markush(
    markush: MarkushFormula,
    cap: int = 100,
    mode: str = "sample",
    r_value_table: list[dict[str, str]] | None = None,
) -> list[Compound]:
    """Generate concrete molecules from Markush definition.

    Modes:
    - "sample": random sample up to cap from all R-group combinations
    - "exhaustive": all combinations (capped)
    - "specified": use explicit R-value assignments from patent tables

    Returns list of validated Compound objects.
    """
    if not markush.core_smiles:
        logger.warning(f"No core SMILES for {markush.patent_id} — cannot enumerate")
        return []

    compounds = []

    if mode == "specified" and r_value_table:
        # Use explicit assignments from patent tables
        for row in r_value_table[:cap]:
            smiles = _instantiate_core(markush.core_smiles, row)
            if smiles:
                compound = _validate_and_build(markush.patent_id, smiles, row)
                if compound:
                    compounds.append(compound)

    elif mode in ("sample", "exhaustive"):
        # Build option lists per R-group (SMILES only)
        r_labels = []
        r_options = []

        for label, rdef in sorted(markush.r_groups.items()):
            # Convert text options to SMILES
            smiles_options = []
            for opt_text in rdef.options_text[:10]:  # Cap options per R-group
                smi = lookup_substituent(opt_text)
                if smi is not None:
                    smiles_options.append((opt_text, smi))

            # Also use pre-resolved SMILES
            for smi in rdef.options_smiles:
                if smi and (smi, smi) not in [(s[1], s[1]) for s in smiles_options]:
                    smiles_options.append((smi, smi))

            if smiles_options:
                r_labels.append(label)
                r_options.append(smiles_options)

        if not r_options:
            logger.warning(f"No resolvable R-groups for {markush.patent_id}")
            return []

        # Enumerate combinations
        total_combinations = 1
        for opts in r_options:
            total_combinations *= len(opts)

        logger.info(
            f"Markush enumeration {markush.patent_id}: "
            f"{len(r_labels)} R-groups, {total_combinations} total combinations, "
            f"cap={cap}"
        )

        count = 0
        for combo in product(*r_options):
            if count >= cap:
                break

            assignments = {label: name for label, (name, _) in zip(r_labels, combo)}
            smi_assignments = {label: smi for label, (_, smi) in zip(r_labels, combo)}

            # Try to instantiate
            smiles = _instantiate_core(markush.core_smiles, smi_assignments)
            if smiles:
                compound = _validate_and_build(markush.patent_id, smiles, assignments)
                if compound:
                    compounds.append(compound)

            count += 1

    logger.info(f"Markush enumeration {markush.patent_id}: {len(compounds)} valid compounds from {cap} attempts")
    return compounds


def _validate_and_build(
    patent_id: str,
    smiles: str,
    r_assignments: dict[str, str],
) -> Compound | None:
    """Validate SMILES and build Compound object."""
    if not validate_smiles(smiles):
        return None

    canonical = canonicalize_smiles(smiles)
    if not canonical:
        return None

    mw = molecular_weight(canonical)
    if not mw or mw < MIN_MW or mw > MAX_MW:
        return None

    inchikey = get_inchikey(canonical)
    salt_result = strip_salt(canonical)
    drug_likeness = compute_drug_likeness(canonical)

    # Build compound ID from R-assignments
    r_desc = ', '.join(f'{k}={v}' for k, v in sorted(r_assignments.items())[:3])
    cpd_id = f"Markush ({r_desc})"

    return Compound(
        patent_id=patent_id,
        example_number=cpd_id,
        iupac_name=None,
        iupac_source=IupacSource.GENERATED,
        canonical_smiles=canonical,
        inchikey=inchikey,
        parent_smiles=salt_result.get("parent_smiles"),
        parent_inchikey=salt_result.get("parent_inchikey"),
        source=CompoundSource.MARKUSH_ENUMERATED,
        extraction_method="markush_enumeration",
        confidence_score=0.7,
        drug_likeness=drug_likeness,
        r_groups=r_assignments,
        processing_status="validated",
    )
