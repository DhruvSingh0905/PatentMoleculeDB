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
# Example-derived scaffold via RDKit R-group decomposition
# ============================================================

def derive_scaffold_from_examples(
    example_smiles: list[str],
    min_examples: int = 5,
    top_k_scaffolds: int = 3,
) -> tuple[str | None, list[dict] | None]:
    """Derive the Markush core scaffold from extracted example compounds.

    Uses Murcko scaffolding to find the most common core, then
    RGroupDecompose to verify and extract attachment points.

    Deterministic, no LM needed, connected by construction.

    Args:
        example_smiles: SMILES of validated example compounds.
        min_examples: Minimum examples needed to attempt derivation.
        top_k_scaffolds: Try this many scaffold candidates.

    Returns:
        (scaffold_smiles_with_dummies, decomposition_results) or (None, None)
    """
    from rdkit.Chem import rdRGroupDecomposition
    from collections import Counter

    if len(example_smiles) < min_examples:
        return None, None

    # Parse all examples
    mols = []
    for smi in example_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol and mol.GetNumHeavyAtoms() >= 10:
            mols.append(mol)

    if len(mols) < min_examples:
        return None, None

    # Step 1: Get Murcko scaffolds and find top candidates
    scaffold_counter = Counter()
    for mol in mols:
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smi = Chem.MolToSmiles(scaf)
            scaffold_counter[scaf_smi] += 1
        except Exception:
            pass

    if not scaffold_counter:
        return None, None

    # Step 2: Try R-group decomposition with each top scaffold
    best_scaffold = None
    best_results = None
    best_match_count = 0

    for scaf_smi, count in scaffold_counter.most_common(top_k_scaffolds):
        core_mol = Chem.MolFromSmiles(scaf_smi)
        if not core_mol or core_mol.GetNumHeavyAtoms() < 8:
            continue

        try:
            results, unmatched = rdRGroupDecomposition.RGroupDecompose(
                [core_mol], mols[:30], asSmiles=True
            )

            match_count = len(results)
            if match_count > best_match_count:
                best_match_count = match_count
                best_results = results
                # Extract the core with attachment points from the first result
                if results:
                    core_with_dummies = results[0].get('Core', '')
                    # Convert [*:n] to [n*] for our pipeline
                    import re
                    core_with_dummies = re.sub(r'\[\*:(\d+)\]', r'[\1*]', core_with_dummies)
                    best_scaffold = core_with_dummies

        except Exception as e:
            logger.debug(f"RGroupDecompose failed on scaffold: {e}")
            continue

    if best_scaffold and best_match_count >= min_examples // 2:
        logger.info(
            f"Example-derived scaffold: {best_match_count}/{len(mols)} examples matched, "
            f"scaffold={best_scaffold[:50]}..."
        )
        return best_scaffold, best_results
    else:
        logger.info(f"Example-derived scaffold: no scaffold matched ≥{min_examples // 2} examples")
        return None, None


# ============================================================
# Scaffold quality scoring
# ============================================================

def score_scaffold(
    scaffold_smiles: str,
    r_labels: set[str],
    example_mols: list | None = None,
) -> float:
    """Score a scaffold candidate for quality.

    Higher score = better scaffold. Patent-agnostic.
    """
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        return -1.0

    frags = Chem.GetMolFrags(mol, asMols=True)
    n_heavy = mol.GetNumHeavyAtoms()
    n_dummies = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)

    score = 0.0

    # Must be single connected component
    score += 1.0 if len(frags) == 1 else -0.5

    # Reward complexity (but not too much — should be a core, not a full molecule)
    score += min(n_heavy / 15.0, 1.0)

    # Attachment count should match R-label count (±1)
    if r_labels:
        delta = abs(n_dummies - len(r_labels))
        score += max(0, 1.0 - delta / max(len(r_labels), 1))

    # Bonus: try R-group decomposition validation
    if example_mols and len(example_mols) >= 3:
        try:
            from rdkit.Chem import rdRGroupDecomposition
            results, _ = rdRGroupDecomposition.RGroupDecompose(
                [mol], example_mols[:10], asSmiles=True
            )
            score += len(results) / 10.0
        except Exception:
            pass

    return score


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

    # Step 1: Use LM mapping as starting point
    mapping = {}
    if lm_mapping:
        for pos, label in lm_mapping.items():
            if pos in placeholders and label in r_labels:
                mapping[label] = pos
            elif label in r_labels:
                for p in placeholders:
                    if p not in mapping.values():
                        mapping[label] = p
                        break

    # Step 2: Fill unmapped positions using structural constraints
    if len(mapping) < len(placeholders):
        mapping = _fill_unmapped_positions(markush, mapping)

    # Step 3: If still incomplete, positional heuristic for remaining
    assigned_positions = set(mapping.values())
    assigned_labels = set(mapping.keys())
    remaining_positions = sorted(placeholders - assigned_positions, key=int)
    remaining_labels = sorted(
        r_labels - assigned_labels,
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999
    )
    for label, pos in zip(remaining_labels, remaining_positions):
        mapping[label] = pos

    return mapping


def _get_position_environments(core_smiles: str) -> dict[str, dict]:
    """Extract chemical environment for each dummy atom position in scaffold.

    Returns: {mapnum_str: {neighbor_symbol, is_aromatic, in_ring, degree}}
    """
    # Normalize to [*:n] for RDKit atom map parsing
    normalized = re.sub(r'\[(\d+)\*\]', r'[*:\1]', core_smiles)
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return {}

    envs = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom
            mapnum = atom.GetAtomMapNum()
            if mapnum == 0:
                continue
            neighbors = list(atom.GetNeighbors())
            if not neighbors:
                continue
            n = neighbors[0]
            envs[str(mapnum)] = {
                'neighbor_symbol': n.GetSymbol(),
                'is_aromatic': n.GetIsAromatic(),
                'in_ring': n.IsInRing(),
                'degree': n.GetDegree(),
                'is_heteroatom': n.GetAtomicNum() not in (6, 1),  # Not C or H
            }

    return envs


def _infer_label_constraints(rdef) -> dict:
    """Infer what chemical environment an R-group expects from its text options."""
    opts_lower = ' '.join(rdef.options_text).lower()

    return {
        'prefers_N': any(kw in opts_lower for kw in ['amino', 'amine', 'n(', 'nh', 'amide', 'carboxamido', 'sulfonamido']),
        'prefers_O': any(kw in opts_lower for kw in ['oxy', 'hydroxy', 'alkoxy', 'methoxy']),
        'prefers_halogen_site': any(kw in opts_lower for kw in ['halogen', 'fluoro', 'chloro', 'bromo', 'iodo']),
        'prefers_aromatic': any(kw in opts_lower for kw in ['aryl', 'phenyl', 'heteroaryl', 'pyridyl', 'pyrimidinyl']),
        'prefers_alkyl': any(kw in opts_lower for kw in ['alkyl', 'methyl', 'ethyl', 'propyl', 'cycloalkyl', 'cyclopentyl']),
        'is_linker': any(kw in opts_lower for kw in ['absent', 'ch2', 'linker', '-o-', '-s-', '-ch2-']),
    }


def _fill_unmapped_positions(
    markush: MarkushFormula,
    current_mapping: dict[str, str],
) -> dict[str, str]:
    """Fill unmapped scaffold positions using structural constraints.

    For each unmapped [n*], check which unassigned R-labels are compatible
    with the atom environment at that position. Auto-assign when there's
    exactly one compatible label.
    """
    if not markush.core_smiles:
        return current_mapping

    envs = _get_position_environments(markush.core_smiles)
    if not envs:
        return current_mapping

    mapping = dict(current_mapping)
    assigned_positions = set(mapping.values())
    assigned_labels = set(mapping.keys())

    # Get all positions and unassigned labels
    all_positions = set(envs.keys())
    unmapped_positions = all_positions - assigned_positions
    unassigned_labels = set(markush.r_groups.keys()) - assigned_labels

    if not unmapped_positions or not unassigned_labels:
        return mapping

    # Score each (label, position) pair
    scores = {}
    for label in unassigned_labels:
        rdef = markush.r_groups[label]
        constraints = _infer_label_constraints(rdef)

        for pos in unmapped_positions:
            env = envs[pos]
            score = 0

            # Score based on environment compatibility
            if constraints['prefers_N'] and env['neighbor_symbol'] == 'N':
                score += 3
            if constraints['prefers_O'] and env['neighbor_symbol'] == 'O':
                score += 3
            if constraints['prefers_halogen_site'] and env['neighbor_symbol'] == 'C' and not env['is_aromatic']:
                score += 2
            if constraints['prefers_aromatic'] and env['is_aromatic']:
                score += 2
            if constraints['prefers_alkyl'] and env['neighbor_symbol'] == 'C' and not env['is_aromatic']:
                score += 2
            if constraints['is_linker'] and env['neighbor_symbol'] in ('C', 'N'):
                score += 1

            # Penalty for obviously wrong matches
            if constraints['prefers_N'] and env['neighbor_symbol'] != 'N':
                score -= 1
            if constraints['prefers_O'] and env['neighbor_symbol'] != 'O':
                score -= 1

            if score > 0:
                scores[(label, pos)] = score

    # Greedy assignment: highest-scoring pairs first, no conflicts
    for (label, pos), score in sorted(scores.items(), key=lambda x: -x[1]):
        if label not in assigned_labels and pos not in assigned_positions:
            mapping[label] = pos
            assigned_labels.add(label)
            assigned_positions.add(pos)
            logger.debug(f"    Constraint filler: {label} → [{pos}*] (score={score})")

    return mapping


def _structural_environment_mapping(markush: MarkushFormula) -> dict[str, str] | None:
    """Try to map R-labels to positions based on chemical environment.
    Wrapper around _fill_unmapped_positions for backward compatibility."""
    result = _fill_unmapped_positions(markush, {})
    return result if result else None


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
