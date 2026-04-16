"""Markush enumerator — generate concrete molecules from Markush definitions.

For unnamed compounds defined by R-group combination tables,
this module instantiates the Markush core with specific substituents.

Uses the SUBSTITUENT_LIBRARY from markush_mapper.py for name→SMILES conversion.
Scored enumeration prioritizes chemically likely R-group combinations.
"""

from __future__ import annotations

import heapq
import logging
import math
import re
from collections import Counter, defaultdict
from itertools import product

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

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


# ============================================================
# Global fragment vocabulary — data-driven from all extracted compounds
# ============================================================

_GLOBAL_VOCAB_CACHE: dict[str, int] | None = None


def build_global_fragment_vocab(
    force_rebuild: bool = False,
) -> dict[str, int]:
    """Build a global set of medicinal-chemistry fragments by BRICS-decomposing
    all validated compounds across all patents.

    Returns {canonical_fragment_smiles: frequency_count}, cached to disk.
    ~1,100 unique fragments from ~1,900 compounds. Takes <5 seconds.
    """
    global _GLOBAL_VOCAB_CACHE
    if _GLOBAL_VOCAB_CACHE is not None and not force_rebuild:
        return _GLOBAL_VOCAB_CACHE

    import json
    import glob
    from rdkit.Chem import BRICS

    cache_path = config.OUTPUT_DIR / "fragment_vocab.json"

    # Try disk cache first
    if cache_path.exists() and not force_rebuild:
        with open(cache_path) as f:
            _GLOBAL_VOCAB_CACHE = json.load(f)
        logger.info(f"Global fragment vocab: loaded {len(_GLOBAL_VOCAB_CACHE)} from cache")
        return _GLOBAL_VOCAB_CACHE

    # Build from all patent outputs
    all_smiles = set()
    for pattern in ['output/results/*/combined.json',
                     'output/final_v2/*.json', 'output/final/*.json']:
        for path in glob.glob(str(config.PROJECT_ROOT / pattern)):
            try:
                with open(path) as f:
                    data = json.load(f)
                for c in data.get('exemplified_compounds', []):
                    smi = c.get('canonical_smiles')
                    if smi and c.get('processing_status') == 'validated':
                        all_smiles.add(smi)
            except Exception:
                pass

    # BRICS fragment all compounds
    from collections import Counter
    frag_counts = Counter()
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                frags = BRICS.BRICSDecompose(mol)
                for f in frags:
                    canon = canonicalize_smiles(f)
                    if canon:
                        frag_counts[canon] += 1
            except Exception:
                pass

    # Filter: keep fragments appearing 2+ times or with >= 3 heavy atoms
    vocab = {}
    for frag, count in frag_counts.items():
        mol = Chem.MolFromSmiles(frag)
        if mol and (count >= 2 or mol.GetNumHeavyAtoms() >= 3):
            vocab[frag] = count

    # Save cache
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(vocab, f, indent=1)

    _GLOBAL_VOCAB_CACHE = vocab
    logger.info(f"Global fragment vocab: built {len(vocab)} from {len(all_smiles)} compounds")
    return vocab


def _filter_compatible_fragments(
    local_options: set[str],
    global_vocab: dict[str, int],
    max_additions: int = 20,
) -> list[str]:
    """Filter global vocab to fragments compatible with existing local options.

    Compatibility: similar MW range and Tanimoto > 0.1 to at least one local option.
    Returns top-N most frequent compatible fragments not already in local set.
    """
    if not local_options:
        # No local options — return most frequent global fragments
        return sorted(global_vocab.keys(), key=lambda x: -global_vocab[x])[:max_additions]

    # Compute local MW range and fingerprints
    local_fps = []
    local_mws = []
    for smi in local_options:
        # Strip dummy atoms for MW/FP comparison
        clean = re.sub(r'\[\d+\*\]', '[H]', re.sub(r'\[\*:\d+\]', '[H]', smi))
        mol = Chem.MolFromSmiles(clean)
        if mol:
            try:
                mol = Chem.RemoveHs(mol)
                if mol.GetNumHeavyAtoms() > 0:
                    local_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
                    local_mws.append(Chem.Descriptors.ExactMolWt(mol))
            except Exception:
                pass

    if not local_fps:
        return sorted(global_vocab.keys(), key=lambda x: -global_vocab[x])[:max_additions]

    mw_lo = min(local_mws) - 80 if local_mws else 0
    mw_hi = max(local_mws) + 150 if local_mws else 600

    # Score each global fragment
    candidates = []
    for frag_smi, count in global_vocab.items():
        if frag_smi in local_options:
            continue

        # Strip BRICS dummy notation for comparison
        clean = re.sub(r'\[\d+\*\]', '[H]', frag_smi)
        mol = Chem.MolFromSmiles(clean)
        if not mol or mol.GetNumHeavyAtoms() < 1:
            continue

        try:
            mol = Chem.RemoveHs(mol)
            mw = Chem.Descriptors.ExactMolWt(mol)
        except Exception:
            continue

        if mw < mw_lo or mw > mw_hi:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        max_sim = max(DataStructs.TanimotoSimilarity(fp, lfp) for lfp in local_fps)

        # Accept if somewhat similar (Tc > 0.1) — broader than before
        if max_sim > 0.1:
            # Score by global frequency × similarity
            candidates.append((count * (1 + max_sim), frag_smi))

    candidates.sort(reverse=True)
    return [smi for _, smi in candidates[:max_additions]]


# ============================================================
# R-group frequency statistics
# ============================================================

def build_rgroup_stats(
    decomposition_results: list[dict],
) -> dict:
    """Build frequency + co-occurrence stats from decomposed examples.

    Returns:
        {
            'freq': {R_label: {smiles: count}},
            'pair': {(R_label_i, R_label_j): {(smi_i, smi_j): count}},
            'total': int,
        }
    """
    freq: dict[str, Counter] = defaultdict(Counter)
    pair: dict[tuple, Counter] = defaultdict(Counter)
    total = 0

    for result in decomposition_results:
        r_vals = {}
        for key, smi in result.items():
            if key == 'Core':
                continue
            r_vals[key] = smi
            freq[key][smi] += 1

        # Pairwise co-occurrence
        labels = sorted(r_vals.keys())
        for ii in range(len(labels)):
            for jj in range(ii + 1, len(labels)):
                pair_key = (labels[ii], labels[jj])
                pair_val = (r_vals[labels[ii]], r_vals[labels[jj]])
                pair[pair_key][pair_val] += 1

        total += 1

    return {'freq': dict(freq), 'pair': dict(pair), 'total': total}


def score_combination(
    assignments: dict[str, str],
    stats: dict,
    lambda_pair: float = 0.3,
    lambda_local: float = 0.5,
    local_fragments: set[str] | None = None,
) -> float:
    """Score an R-group combination by frequency likelihood + localness.

    Score = sum(log P(Ri=ri))                     # marginal frequency
          + lambda_pair * sum(log P(Ri, Rj))      # co-occurrence
          + lambda_local * sum(is_local(ri))       # local bonus

    Higher = more likely (less negative).
    """
    freq = stats['freq']
    pair = stats['pair']
    total = max(stats['total'], 1)

    score = 0.0

    # Marginal frequency
    for label, smi in assignments.items():
        if label in freq:
            count = freq[label].get(smi, 0)
            # Laplace smoothing
            p = (count + 0.1) / (total + 0.1 * len(freq.get(label, {})))
            score += math.log(p)
        else:
            score -= 5.0  # Unknown label penalty

    # Pairwise co-occurrence
    labels = sorted(assignments.keys())
    for ii in range(len(labels)):
        for jj in range(ii + 1, len(labels)):
            pair_key = (labels[ii], labels[jj])
            if pair_key in pair:
                pair_val = (assignments[labels[ii]], assignments[labels[jj]])
                count = pair[pair_key].get(pair_val, 0)
                p = (count + 0.01) / (total + 0.01 * len(pair.get(pair_key, {})))
                score += lambda_pair * math.log(p)

    # Localness bonus: prefer fragments seen in this patent's examples
    if local_fragments:
        for label, smi in assignments.items():
            if smi in local_fragments:
                score += lambda_local

    return score


# ============================================================
# Diversity selection (MaxMin in Morgan FP space)
# ============================================================

def maxmin_diverse_subset(
    smiles_list: list[str],
    max_n: int,
) -> list[str]:
    """Select diverse subset using MaxMin algorithm on Morgan FPs."""
    if len(smiles_list) <= max_n:
        return smiles_list

    fps = []
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
            valid.append(smi)

    if len(valid) <= max_n:
        return valid

    # Start with first compound
    selected = [0]
    remaining = set(range(1, len(valid)))

    while len(selected) < max_n and remaining:
        best_idx = -1
        best_min_dist = -1

        for idx in remaining:
            # Min distance to any already selected
            min_dist = min(
                1.0 - DataStructs.TanimotoSimilarity(fps[idx], fps[s])
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)
        else:
            break

    return [valid[i] for i in selected]


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

    # Phase 0: Detect bridging groups (same fragment for multiple R-labels)
    # Group assignments by SMILES to find bridges
    smi_to_labels = {}
    for label, sub_name in assignments.items():
        if sub_name not in smi_to_labels:
            smi_to_labels[sub_name] = []
        smi_to_labels[sub_name].append(label)

    # For bridging groups, only process once but connect all positions
    processed_bridges = set()

    # Phase 1: Combine core + ALL fragments into one molecule
    combo = Chem.RWMol(Chem.MolFromSmiles(normalized))
    if combo is None:
        return None

    bonds_to_add = []
    dummies_to_remove = set()
    hydrogen_dummies = set()

    for label, sub_name in assignments.items():
        # Skip if this is a duplicate of a bridging group already processed
        if sub_name in processed_bridges:
            continue
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

        # Find ALL fragment dummies (bridging groups have multiple)
        frag_dummies = []
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                frag_dummies.append((atom.GetIdx(), atom.GetAtomMapNum()))

        # Add fragment to combo
        frag_start = combo.GetNumAtoms()
        combo = Chem.RWMol(Chem.CombineMols(combo, frag_mol))

        if frag_dummies:
            # For each fragment dummy, find the matching core dummy by map number
            for frag_d_idx, frag_d_mapnum in frag_dummies:
                frag_d_in_combo = frag_start + frag_d_idx
                frag_d_atom = combo.GetAtomWithIdx(frag_d_in_combo)
                frag_neighbors = list(frag_d_atom.GetNeighbors())

                if not frag_neighbors:
                    continue

                frag_attach = frag_neighbors[0].GetIdx()

                # Find the core dummy with matching map number
                matched_core_dummy = None
                for atom in combo.GetAtoms():
                    if (atom.GetAtomicNum() == 0 and
                        atom.GetAtomMapNum() == frag_d_mapnum and
                        atom.GetIdx() < frag_start):  # Must be in core part
                        matched_core_dummy = atom.GetIdx()
                        break

                if matched_core_dummy is not None:
                    core_d_atom = combo.GetAtomWithIdx(matched_core_dummy)
                    core_d_neighbors = list(core_d_atom.GetNeighbors())
                    if core_d_neighbors:
                        core_attach = core_d_neighbors[0].GetIdx()
                        bonds_to_add.append((core_attach, frag_attach))
                        dummies_to_remove.add(matched_core_dummy)
                        dummies_to_remove.add(frag_d_in_combo)

            # Mark this as processed (don't add the same bridge again)
            if len(frag_dummies) > 1:
                processed_bridges.add(sub_name)
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

    elif mode == "scored" and decomposition_results:
        # Scored enumeration: prioritize chemically likely R-group combinations
        # Uses frequency statistics from decomposition to score novel combinations
        stats = build_rgroup_stats(decomposition_results)

        # Build option lists per R-group from library + common fragments
        r_labels = []
        r_options = []

        for label, rdef in sorted(markush.r_groups.items()):
            smiles_set = set()

            # 1. Options from decomposition (data-derived)
            for smi in rdef.options_smiles:
                if smi:
                    smiles_set.add(smi)

            # 2. Options from text definitions
            for opt_text in rdef.options_text:
                smi = lookup_substituent(opt_text)
                if smi is not None:
                    smiles_set.add(smi)

            # 3. Expand with global fragment vocabulary (BRICS-derived)
            # Use two-layer approach: local (patent-specific) + global (cross-patent)
            global_vocab = build_global_fragment_vocab()
            compatible = _filter_compatible_fragments(smiles_set, global_vocab, max_additions=15)
            local_set = set(smiles_set)  # Track what's local for scoring
            smiles_set.update(compatible)

            smiles_list = sorted(smiles_set)
            if smiles_list:
                r_labels.append(label)
                r_options.append(smiles_list)

        if not r_options:
            logger.warning(f"No resolvable R-groups for {markush.patent_id}")
            return []

        total_combinations = 1
        for opts in r_options:
            total_combinations *= len(opts)

        logger.info(
            f"Scored enumeration {markush.patent_id}: "
            f"{len(r_labels)} R-groups, {total_combinations} total combinations, "
            f"scoring top {cap}"
        )

        # ---- Fast scored enumeration ----
        # Strategy: rank each R-group's options by marginal frequency,
        # take top-K per position, then enumerate the pruned product space.
        # This clears most of the search space using the scoring signal.

        # Rank options per position by frequency
        pruned_options = []
        for i, label in enumerate(r_labels):
            opts = r_options[i]
            label_freq = stats['freq'].get(label, {})
            total = max(stats['total'], 1)

            # Score each option by frequency (Laplace-smoothed)
            scored = []
            for smi in opts:
                count = label_freq.get(smi, 0)
                p = (count + 0.1) / (total + 0.1 * len(label_freq))
                scored.append((p, smi))

            scored.sort(reverse=True)

            # Keep top-K per position based on frequency concentration
            # Aggressive pruning: top options that cover 90% of frequency mass
            if len(scored) <= 3:
                top_k = len(scored)
            else:
                cum_p = 0
                top_k = 0
                for p, _ in scored:
                    cum_p += p
                    top_k += 1
                    if cum_p > 0.90 and top_k >= 2:
                        break
                # Hard cap: never more than 10 per position to keep space tractable
                top_k = min(top_k + 1, len(scored), 10)

            pruned = [smi for _, smi in scored[:top_k]]
            pruned_options.append(pruned)

            logger.debug(f"  {label}: {len(opts)} → {len(pruned)} (top covers {sum(p for p,_ in scored[:top_k]):.0%})")

        # Compute pruned combinatorial space
        pruned_total = 1
        for opts in pruned_options:
            pruned_total *= len(opts)

        logger.info(
            f"  Pruned: {total_combinations} → {pruned_total} combinations "
            f"({100*(1-pruned_total/max(total_combinations,1)):.0f}% cleared)"
        )

        # Enumerate pruned space (up to cap)
        scored_combos = []
        combo_counter = 0

        if pruned_total <= 500_000:
            # Can score all in pruned space
            for combo in product(*pruned_options):
                assignments = {label: smi for label, smi in zip(r_labels, combo)}
                score = score_combination(assignments, stats, local_fragments=local_set)
                scored_combos.append((-score, combo_counter, assignments))
                combo_counter += 1
            scored_combos.sort()
        else:
            # Still too large — weighted random sample from pruned space
            import random
            random.seed(42)
            seen_combos = set()

            for _ in range(min(cap * 5, 5000)):
                combo = []
                for i, label in enumerate(r_labels):
                    opts = pruned_options[i]
                    label_freq = stats['freq'].get(label, {})
                    weights = [label_freq.get(s, 0.1) for s in opts]
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    chosen = random.choices(opts, weights=weights, k=1)[0]
                    combo.append(chosen)

                combo_key = tuple(combo)
                if combo_key in seen_combos:
                    continue
                seen_combos.add(combo_key)

                assignments = {label: smi for label, smi in zip(r_labels, combo)}
                score = score_combination(assignments, stats, local_fragments=local_set)
                scored_combos.append((-score, combo_counter, assignments))
                combo_counter += 1

            scored_combos.sort()

        # Instantiate top-scored combinations
        seen_smiles = set()
        for neg_score, _, smi_assignments in scored_combos:
            if len(compounds) >= cap:
                break

            pos_map = {}
            for lbl, smi in smi_assignments.items():
                m = re.search(r'\[\*:(\d+)\]', smi)
                if m:
                    pos_map[lbl] = m.group(1)
                else:
                    num = re.search(r'\d+', lbl)
                    if num:
                        pos_map[lbl] = num.group(0)

            smiles = _instantiate_core(markush.core_smiles, smi_assignments, pos_map or None)
            if smiles and smiles not in seen_smiles:
                seen_smiles.add(smiles)
                compound = _validate_and_build(markush.patent_id, smiles, smi_assignments)
                if compound:
                    compounds.append(compound)

        logger.info(
            f"Scored enumeration {markush.patent_id}: {len(compounds)} valid from "
            f"{len(scored_combos)} scored combinations"
        )

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
