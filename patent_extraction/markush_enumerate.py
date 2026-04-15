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

    Strategy: combine core + all fragments into one mol, then bond
    matching dummies and remove them all at once. This avoids the
    atom-index-shift problem when attaching sequentially.
    """
    # Normalize core SMILES: convert [n*] to [*:n] (atom map notation)
    normalized = core_smiles
    for m in re.finditer(r'\[(\d+)\*\]', core_smiles):
        n = m.group(1)
        normalized = normalized.replace(f'[{n}*]', f'[*:{n}]')

    core_mol = Chem.MolFromSmiles(normalized)
    if core_mol is None:
        return None

    # Build label → atom map number using position_mapping
    label_to_mapnum = {}
    if position_mapping:
        for label, pos in position_mapping.items():
            label_to_mapnum[label] = int(pos)
    else:
        for label in assignments:
            num = re.search(r'\d+', label)
            if num:
                label_to_mapnum[label] = int(num.group(0))

    # Phase 1: Combine core + ALL fragments into one molecule
    combo = Chem.RWMol(Chem.MolFromSmiles(normalized))
    if combo is None:
        return None

    # Track: (core_dummy_idx, core_attach_idx, frag_dummy_idx, frag_attach_idx)
    bonds_to_add = []
    dummies_to_remove = set()
    hydrogen_dummies = set()  # Core dummies to just remove (R=H)

    for label, sub_name in assignments.items():
        if Chem.MolFromSmiles(sub_name) is not None:
            sub_smi = sub_name
        else:
            sub_smi = lookup_substituent(sub_name)
            if sub_smi is None or sub_smi == "":
                sub_smi = "[H]"

        mapnum = label_to_mapnum.get(label)
        if mapnum is None:
            continue

        # Find core dummy with this map number
        core_dummy_idx = None
        for atom in combo.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == mapnum:
                core_dummy_idx = atom.GetIdx()
                break
        if core_dummy_idx is None:
            continue

        core_dummy = combo.GetAtomWithIdx(core_dummy_idx)
        core_neighbors = list(core_dummy.GetNeighbors())
        if not core_neighbors:
            continue
        core_attach_idx = core_neighbors[0].GetIdx()

        # Handle hydrogen
        if sub_smi in ("[H]", "[H][*]", "[H][*:1]", "[H][*:2]", "[H][*:3]"):
            hydrogen_dummies.add(core_dummy_idx)
            continue

        # Parse fragment
        frag_mol = Chem.MolFromSmiles(sub_smi)
        if frag_mol is None:
            cleaned = re.sub(r'\[\*:\d+\]', '[*]', sub_smi)
            frag_mol = Chem.MolFromSmiles(cleaned)
        if frag_mol is None:
            hydrogen_dummies.add(core_dummy_idx)
            continue

        # Find fragment dummy
        frag_dummy_idx = None
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                frag_dummy_idx = atom.GetIdx()
                break

        # Add fragment to combo
        frag_start = combo.GetNumAtoms()
        combo = Chem.RWMol(Chem.CombineMols(combo, frag_mol))

        if frag_dummy_idx is not None:
            frag_dummy_in_combo = frag_start + frag_dummy_idx
            frag_dummy_atom = combo.GetAtomWithIdx(frag_dummy_in_combo)
            frag_neighbors = list(frag_dummy_atom.GetNeighbors())
            if frag_neighbors:
                frag_attach = frag_neighbors[0].GetIdx()
                bonds_to_add.append((core_attach_idx, frag_attach))
                dummies_to_remove.add(core_dummy_idx)
                dummies_to_remove.add(frag_dummy_in_combo)
            else:
                hydrogen_dummies.add(core_dummy_idx)
        else:
            # No dummy — attach first heavy atom
            frag_attach = frag_start
            for i in range(frag_start, combo.GetNumAtoms()):
                if combo.GetAtomWithIdx(i).GetAtomicNum() > 1:
                    frag_attach = i
                    break
            bonds_to_add.append((core_attach_idx, frag_attach))
            dummies_to_remove.add(core_dummy_idx)

    # Phase 2: Add all bonds at once
    for a, b in bonds_to_add:
        try:
            combo.AddBond(a, b, Chem.BondType.SINGLE)
        except Exception:
            pass

    # Phase 3: Remove all dummies at once (reverse order to preserve indices)
    all_dummies = dummies_to_remove | hydrogen_dummies
    # Also remove any remaining unmapped dummies
    for atom in combo.GetAtoms():
        if atom.GetAtomicNum() == 0:
            all_dummies.add(atom.GetIdx())

    for idx in sorted(all_dummies, reverse=True):
        try:
            combo.RemoveAtom(idx)
        except Exception:
            pass

    # Phase 4: Sanitize and canonicalize
    try:
        Chem.SanitizeMol(combo)
        mol = Chem.RemoveHs(combo)
        return Chem.MolToSmiles(mol)
    except Exception:
        try:
            return Chem.MolToSmiles(combo)
        except Exception:
            return None


def enumerate_markush(
    markush: MarkushFormula,
    cap: int = 100,
    mode: str = "sample",
    r_value_table: list[dict[str, str]] | None = None,
    decomposition_results: list[dict] | None = None,
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

    elif mode == "decomposition" and decomposition_results:
        # Use actual R-group assignments from RGroupDecompose
        # Each result is a dict: {Core: ..., R1: smi, R2: smi, ...}
        seen_smiles = set()
        for result in decomposition_results[:cap]:
            core_smi = result.get('Core', '')
            core_smi = re.sub(r'\[\*:(\d+)\]', r'[\1*]', core_smi)

            assignments = {}
            for key, smi in result.items():
                if key == 'Core':
                    continue
                assignments[key] = smi

            reconstructed = _instantiate_core(core_smi, assignments)
            if reconstructed and reconstructed not in seen_smiles:
                seen_smiles.add(reconstructed)
                compound = _validate_and_build(markush.patent_id, reconstructed,
                                               {k: v[:30] for k, v in assignments.items()})
                if compound:
                    compounds.append(compound)

        logger.info(f"Markush decomposition {markush.patent_id}: {len(compounds)} from {len(decomposition_results)} decomposed")

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

            # Build position_mapping from the SMILES themselves
            # Data-derived SMILES contain [*:n] which tells us the position
            pos_map = {}
            for lbl, smi in smi_assignments.items():
                # Extract position from [*:n] in the SMILES
                m = re.search(r'\[\*:(\d+)\]', smi)
                if m:
                    pos_map[lbl] = m.group(1)
                else:
                    # Fallback: try label number
                    num = re.search(r'\d+', lbl)
                    if num:
                        pos_map[lbl] = num.group(0)

            smiles = _instantiate_core(markush.core_smiles, smi_assignments, pos_map or None)
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
