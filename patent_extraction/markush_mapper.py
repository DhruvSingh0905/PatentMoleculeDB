"""MapIUPACToMarkush — map IUPAC names to R-group assignments.

Three-path ensemble:
  Path A: Symbolic alignment (free) — Murcko scaffold diff → fragment matching
  Path B: LM with Markush context (~$0.02) — Claude assigns R-values
  Path C: Reconciliation — compare A+B, pick the one matching OPSIN structure

Also includes the substituent library and Markush instantiation logic
shared with markush_enumerate.py.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import config
from .api_client import call_claude_text
from .models import MarkushFormula, MarkushMapping, RGroupDef
from .smiles_utils import validate_smiles, canonicalize_smiles, get_inchikey, get_connectivity_key

logger = logging.getLogger(__name__)


# ============================================================
# Substituent SMILES library — curated common drug fragments
# ============================================================

SUBSTITUENT_LIBRARY = {
    # Alkyl — common patent notations
    "hydrogen": "[H]", "h": "[H]",
    "methyl": "C", "ethyl": "CC", "propyl": "CCC", "n-propyl": "CCC",
    "isopropyl": "CC(C)", "i-propyl": "CC(C)",
    "butyl": "CCCC", "n-butyl": "CCCC", "isobutyl": "CC(C)C",
    "tert-butyl": "C(C)(C)C", "t-butyl": "C(C)(C)C",
    "pentyl": "CCCCC", "hexyl": "CCCCCC",
    # Patent-style alkyl ranges (use representative)
    "c1 alkyl": "C", "c2 alkyl": "CC", "c3 alkyl": "CCC",
    "c1-c6 alkyl": "C", "c1-c4 alkyl": "C", "c1-c3 alkyl": "C",
    "(c1-c4)-alkyl": "C", "(c1-c3)-alkyl": "C", "(c1- c4)- alkyl": "C",
    "(c1- c2)- alkyl": "C", "c1-c8 alkyl": "C",
    # Patent-style shorthand
    "- ch3": "C", "- ch2ch2": "CC", "- cf3": "C(F)(F)F",
    "- cd3": "C([2H])([2H])[2H]", "- chf2": "C(F)F",
    "ch3": "C", "cf3": "C(F)(F)F", "chf2": "C(F)F",
    # Cycloalkyl — with patent notations
    "cyclopropyl": "C1CC1", "cyclobutyl": "C1CCC1", "cyclopentyl": "C1CCCC1",
    "cyclohexyl": "C1CCCCC1", "cycloheptyl": "C1CCCCCC1",
    "c3-c6 cycloalkyl": "C1CCCC1", "c3-c7 cycloalkyl": "C1CCCC1",
    "(c3-c7)-cycloalkyl": "C1CCCC1", "(c3- c7)- cycloalkyl": "C1CCCC1",
    "c3-c10 cycloalkyl": "C1CCCC1",
    # Alkyl-cycloalkyl combos
    "- (c1- c2)- alkyl- (c3- c7)- cycloalkyl": "CC1CCCC1",
    "- (c1- c4)- alkyl- (c3- c7)- cycloalkyl": "CC1CCCC1",
    "c1-3 alkyl-cyclopropyl": "CC1CC1", "c1-3 alkyl-cyclobutyl": "CC1CCC1",
    # Linkers
    "- ch2- ch2-": "CC", "-ch2-ch2-": "CC", "- o- ch2-": "OC",
    "-o-ch2-": "OC", "- s- ch2-": "SC", "s-(o)2-ch2": "CS(=O)(=O)",
    "-ch2-": "C", "ch2": "C",
    # Oxetanyl
    "oxetanyl": "C1COC1", "oxetane": "C1COC1",
    # Aryl
    "phenyl": "c1ccccc1", "4-fluorophenyl": "c1cc(F)ccc1",
    "4-chlorophenyl": "c1cc(Cl)ccc1", "4-methoxyphenyl": "c1cc(OC)ccc1",
    "naphthyl": "c1ccc2ccccc2c1", "aryl": "c1ccccc1",
    # Heteroaryl
    "pyridyl": "c1ccncc1", "pyridin-3-yl": "c1ccncc1",
    "pyrimidinyl": "c1ccncn1", "morpholinyl": "C1COCCN1",
    "piperidinyl": "C1CCNCC1", "pyrrolidinyl": "C1CCNC1",
    "pyrazolyl": "c1cc[nH]n1", "imidazolyl": "c1c[nH]cn1",
    "triazolyl": "c1cn[nH]n1", "oxazolyl": "c1cocn1",
    "thiazolyl": "c1cscn1", "thienyl": "c1ccsc1",
    "furyl": "c1ccoc1", "indolyl": "c1ccc2[nH]ccc2c1",
    # Halogens
    "fluoro": "F", "chloro": "Cl", "bromo": "Br", "iodo": "I",
    "halogen": "F",  # Representative
    "f": "F", "cl": "Cl", "br": "Br",
    # Functional groups
    "amino": "N", "hydroxy": "O", "hydroxyl": "O",
    "methoxy": "OC", "ethoxy": "OCC", "alkoxy": "OC",
    "cyano": "C#N", "nitro": "N(=O)=O",
    "acetyl": "C(=O)C", "formyl": "C=O",
    "carboxyl": "C(=O)O", "sulfonyl": "S(=O)(=O)",
    "methylsulfonyl": "CS(=O)(=O)", "trifluoromethyl": "C(F)(F)F",
    "difluoromethyl": "C(F)F",
    # Common linkers
    "oxy": "O", "thio": "S", "carbonyl": "C(=O)",
    "sulfonamido": "NS(=O)(=O)", "carboxamido": "C(=O)N",
    "absent": "",  # No substitution
}


def lookup_substituent(name: str) -> str | None:
    """Look up substituent SMILES from name. Tries library first, then OPSIN."""
    name_clean = name.strip().lower()
    name_clean = re.sub(r'^[-—]', '', name_clean).strip()
    name_clean = re.sub(r'optionally substituted\s*', '', name_clean)

    # Direct lookup
    if name_clean in SUBSTITUENT_LIBRARY:
        return SUBSTITUENT_LIBRARY[name_clean]

    # Partial match
    for key, smi in SUBSTITUENT_LIBRARY.items():
        if key in name_clean:
            return smi

    # Try OPSIN as fallback
    try:
        from py2opsin import py2opsin
        result = py2opsin(name_clean)
        if result and validate_smiles(result):
            return result
    except Exception:
        pass

    return None


# ============================================================
# Position mapping: scaffold [n*] ↔ patent R-label alignment
# ============================================================

def resolve_position_mapping(
    markush: MarkushFormula,
    lm_mapping: dict[str, str] | None = None,
    example_compounds: list[tuple[str, str]] | None = None,  # [(iupac, smiles), ...]
) -> dict[str, str]:
    """Resolve the mapping between scaffold [n*] positions and patent R-labels.

    Hybrid approach:
    1. Start with LM-proposed mapping (from FormulaAgent)
    2. Validate with structural consistency
    3. Score against example compounds
    4. Fall back to positional heuristic if needed

    Returns: {R-label: placeholder_number} e.g. {"R1": "1", "G": "2"}
    """
    if not markush.core_smiles:
        return {}

    # Count placeholders in scaffold
    import re
    placeholders = set(re.findall(r'\[(\d+)\*\]', markush.core_smiles))
    r_labels = set(markush.r_groups.keys())

    if not placeholders or not r_labels:
        return {}

    # Step 1: Use LM mapping if available
    if lm_mapping:
        # Validate: does each mapped placeholder exist?
        valid_mapping = {}
        for pos, label in lm_mapping.items():
            if pos in placeholders and label in r_labels:
                valid_mapping[label] = pos
            elif label in r_labels:
                # LM might have used different notation — try fuzzy match
                for p in placeholders:
                    if p not in valid_mapping.values():
                        valid_mapping[label] = p
                        break
        if valid_mapping:
            return valid_mapping

    # Step 2: Structural consistency — check atom environments
    mapping = _structural_environment_mapping(markush)
    if mapping:
        return mapping

    # Step 3: Positional heuristic — map in order
    # Sort R-labels by numeric value, map to placeholders in order
    sorted_labels = sorted(r_labels, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)
    sorted_positions = sorted(placeholders, key=int)

    return {label: pos for label, pos in zip(sorted_labels, sorted_positions)}


def _structural_environment_mapping(markush: MarkushFormula) -> dict[str, str] | None:
    """Try to map R-labels to positions based on chemical environment."""
    if not markush.core_smiles:
        return None

    mol = Chem.MolFromSmiles(markush.core_smiles)
    if mol is None:
        return None

    import re
    # Find placeholder atoms and their neighbors
    placeholder_envs = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            # Find which [n*] this is by position in SMILES
            idx = atom.GetIdx()
            neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
            neighbor_symbols = [n.GetSymbol() for n in neighbors]
            neighbor_aromatic = [n.GetIsAromatic() for n in neighbors]
            placeholder_envs[str(idx)] = {
                'neighbors': neighbor_symbols,
                'aromatic': any(neighbor_aromatic),
            }

    # Match environments to R-group constraints from text
    mapping = {}
    for label, rdef in markush.r_groups.items():
        # Analyze R-group options to infer expected environment
        opts_lower = ' '.join(rdef.options_text).lower()
        expected_on_n = 'amino' in opts_lower or 'amine' in opts_lower or 'n(' in opts_lower
        expected_on_o = 'oxy' in opts_lower or 'hydroxy' in opts_lower
        expected_halogen = 'halogen' in opts_lower or 'fluoro' in opts_lower or 'chloro' in opts_lower

        # Try to find a matching placeholder
        for pos, env in placeholder_envs.items():
            if pos in mapping.values():
                continue  # Already assigned
            if expected_on_n and 'N' in env['neighbors']:
                mapping[label] = pos
                break
            elif expected_on_o and 'O' in env['neighbors']:
                mapping[label] = pos
                break
            elif expected_halogen and 'C' in env['neighbors'] and not env['aromatic']:
                mapping[label] = pos
                break

    return mapping if mapping else None


def score_mapping_with_examples(
    markush: MarkushFormula,
    mapping: dict[str, str],
    examples: list[tuple[str, str]],  # [(iupac, smiles), ...]
) -> float:
    """Score a position mapping by how many examples it can explain.

    For each example, try to instantiate the Markush core with R-values
    that produce a molecule matching the example's connectivity.
    """
    if not markush.core_smiles or not examples:
        return 0.0

    explained = 0
    for iupac, smiles in examples:
        example_mol = Chem.MolFromSmiles(smiles)
        if not example_mol:
            continue

        # Try the symbolic alignment with this mapping applied
        aligned = _symbolic_alignment(markush, smiles)
        if aligned.confidence in ('high', 'medium'):
            explained += 1

    return explained / max(len(examples), 1)


# ============================================================
# Path A: Symbolic alignment (free)
# ============================================================

def _symbolic_alignment(
    markush: MarkushFormula,
    compound_smiles: str,
) -> MarkushMapping:
    """Align a compound to Markush by comparing scaffold + fragments."""
    mol = Chem.MolFromSmiles(compound_smiles)
    if mol is None:
        return MarkushMapping(confidence="low", alignment_method="symbolic")

    # Get Murcko scaffold
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(scaffold)
    except Exception:
        return MarkushMapping(confidence="low", alignment_method="symbolic")

    # If we have the Markush core, check substructure match
    assignments = {}
    if markush.core_smiles:
        core_mol = Chem.MolFromSmiles(markush.core_smiles.replace('[1*]', '[*]').replace('[2*]', '[*]'))
        if core_mol and mol.HasSubstructMatch(core_mol):
            # Compound matches the Markush core — try to identify R-groups
            # For each R-group, find the matching substituent
            for label, rdef in markush.r_groups.items():
                for opt_text, opt_smi in zip(rdef.options_text, rdef.options_smiles):
                    if opt_smi:
                        sub_mol = Chem.MolFromSmiles(opt_smi)
                        if sub_mol and mol.HasSubstructMatch(sub_mol):
                            assignments[label] = opt_text
                            break

    confidence = "high" if len(assignments) >= 2 else ("medium" if assignments else "low")
    return MarkushMapping(
        r_assignments=assignments,
        instance_smiles=compound_smiles,
        confidence=confidence,
        alignment_method="symbolic",
    )


# ============================================================
# Path B: LM with Markush context
# ============================================================

MARKUSH_MAP_PROMPT = """Given this Markush definition and IUPAC name, identify which R-group values were chosen.

Markush Definition:
{markush_summary}

IUPAC Name: {iupac_name}

For each R-group position, determine the chosen value. Return ONLY valid JSON:
{{"R1": "chosen value", "R2": "chosen value", ...}}

Only include R-groups you can confidently identify from the name."""


def _lm_alignment(
    markush: MarkushFormula,
    iupac_name: str,
    patent_id: str,
) -> MarkushMapping:
    """Use LM to assign R-values based on Markush context + IUPAC name."""
    # Build summary
    summary_parts = [f"Core scaffold: {markush.core_smiles or markush.core_smarts or 'unknown'}"]
    for label, rdef in list(markush.r_groups.items())[:10]:  # Cap at 10 R-groups
        opts = ', '.join(rdef.options_text[:5])
        summary_parts.append(f"{label}: {opts}")

    summary = '\n'.join(summary_parts)

    prompt = MARKUSH_MAP_PROMPT.format(
        markush_summary=summary,
        iupac_name=iupac_name,
    )

    response = call_claude_text(
        prompt=prompt,
        model=config.MODEL_SONNET,
        patent_id=patent_id,
        compound_id="markush_map",
        max_tokens=300,
    )

    if not response:
        return MarkushMapping(confidence="low", alignment_method="lm")

    try:
        import json
        cleaned = response.strip()
        if '```' in cleaned:
            m = re.search(r'```(?:json)?\s*(.*?)```', cleaned, re.DOTALL)
            cleaned = m.group(1) if m else cleaned
        assignments = json.loads(cleaned)
        if isinstance(assignments, dict):
            confidence = "high" if len(assignments) >= 2 else "medium"
            return MarkushMapping(
                r_assignments=assignments,
                confidence=confidence,
                alignment_method="lm",
            )
    except Exception:
        pass

    return MarkushMapping(confidence="low", alignment_method="lm")


# ============================================================
# Path C: Reconciliation
# ============================================================

def _reconcile(
    path_a: MarkushMapping,
    path_b: MarkushMapping,
    compound_smiles: str | None,
) -> MarkushMapping:
    """Reconcile symbolic and LM alignments."""
    # If both agree on ≥1 assignment, high confidence
    agreed = {}
    disagreed = {}

    all_labels = set(path_a.r_assignments.keys()) | set(path_b.r_assignments.keys())
    for label in all_labels:
        a_val = path_a.r_assignments.get(label, '')
        b_val = path_b.r_assignments.get(label, '')
        if a_val and b_val:
            if a_val.lower() == b_val.lower():
                agreed[label] = a_val
            else:
                disagreed[label] = (a_val, b_val)
        elif a_val:
            agreed[label] = a_val
        elif b_val:
            agreed[label] = b_val

    # For disagreements, prefer LM (it has Markush context, symbolic is heuristic)
    for label, (a_val, b_val) in disagreed.items():
        agreed[label] = b_val  # LM wins on disagreement

    confidence = "high" if len(agreed) >= 2 and not disagreed else "medium"

    return MarkushMapping(
        r_assignments=agreed,
        instance_smiles=compound_smiles,
        confidence=confidence,
        alignment_method="reconciled",
    )


# ============================================================
# Public API
# ============================================================

def map_iupac_to_markush(
    markush: MarkushFormula,
    iupac_name: str,
    compound_smiles: str | None = None,
    patent_id: str = "",
    use_lm: bool = True,
) -> MarkushMapping:
    """Map an IUPAC name to R-group assignments against a Markush definition.

    Three-path ensemble:
      Path A: Symbolic (free) — scaffold diff + fragment matching
      Path B: LM (cheap) — Claude assigns R-values with Markush context
      Path C: Reconcile A+B

    Args:
        markush: Canonical Markush definition.
        iupac_name: Raw IUPAC name from patent.
        compound_smiles: OPSIN-derived SMILES (if available).
        patent_id: For API logging.
        use_lm: If False, skip Path B (symbolic only).

    Returns:
        MarkushMapping with r_assignments, confidence, and method.
    """
    # Path A: Symbolic (always runs, free)
    path_a = MarkushMapping(confidence="low", alignment_method="symbolic")
    if compound_smiles:
        path_a = _symbolic_alignment(markush, compound_smiles)

    # Path B: LM (optional, ~$0.02)
    path_b = MarkushMapping(confidence="low", alignment_method="lm")
    if use_lm and markush.r_groups:
        path_b = _lm_alignment(markush, iupac_name, patent_id)

    # Path C: Reconcile
    if path_a.confidence != "low" or path_b.confidence != "low":
        result = _reconcile(path_a, path_b, compound_smiles)
    else:
        # Both failed — return best available
        result = path_b if path_b.r_assignments else path_a

    return result
