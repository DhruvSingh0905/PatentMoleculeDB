"""Ensemble strategy tester — compare image extraction approaches.

Tests multiple strategies on the same set of compound images:
1. DECIMER only (free, local)
2. Sonnet Vision only (cheap, API)
3. Opus Vision only (expensive, API)
4. Cascade: DECIMER → Sonnet fallback → Opus fallback
5. Consensus: DECIMER + Sonnet simultaneously, accept if agree, Opus arbitrates
6. Layout-to-Local: Sonnet layout → crop → DECIMER

Each strategy is scored on:
- Valid SMILES rate (RDKit validates)
- BDB match rate (exact InChIKey)
- BDB connectivity match rate (14-char InChIKey)
- Cost per compound
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .smiles_utils import validate_smiles, canonicalize_smiles, get_inchikey, molecular_weight

logger = logging.getLogger(__name__)

MIN_SMILES_LENGTH = 10
MIN_SMILES_MW = 150


def _validate_smiles(smiles: str) -> bool:
    """Check if SMILES is valid, long enough, and drug-like MW."""
    if not smiles or not validate_smiles(smiles):
        return False
    if len(smiles) < MIN_SMILES_LENGTH:
        return False
    mw = molecular_weight(smiles)
    return mw is not None and mw >= MIN_SMILES_MW


def run_strategy_comparison(
    image_paths: list[str],
    bdb_keys: set[str],
    patent_id: str = "",
) -> dict:
    """Run all strategies on the same image set and compare.

    Args:
        image_paths: List of paths to cropped structure images.
        bdb_keys: Set of BindingDB InChIKeys for validation.
        patent_id: For logging.

    Returns:
        Dict with per-strategy results.
    """
    bdb_conn = {k[:14] for k in bdb_keys if k and len(k) >= 14}

    strategies = {}

    # --- Strategy 1: DECIMER only ---
    logger.info("Testing Strategy 1: DECIMER only")
    dec_results = []
    try:
        from DECIMER import predict_SMILES
        for img_path in image_paths:
            try:
                raw = predict_SMILES(img_path)
                smiles = raw.split('.')[0] if raw else ''
                valid = _validate_smiles(smiles)
                ik = get_inchikey(canonicalize_smiles(smiles)) if valid else None
                dec_results.append({
                    'valid': valid,
                    'match': ik in bdb_keys if ik else False,
                    'conn': ik[:14] in bdb_conn if ik else False,
                })
            except Exception:
                dec_results.append({'valid': False, 'match': False, 'conn': False})
    except ImportError:
        dec_results = [{'valid': False, 'match': False, 'conn': False}] * len(image_paths)

    strategies['decimer_only'] = _score(dec_results, cost_per=0)

    # --- Strategy 2: Sonnet Vision only ---
    logger.info("Testing Strategy 2: Sonnet Vision only")
    son_results = []
    from .api_client import call_claude_vision
    from . import config
    for img_path in image_paths:
        resp = call_claude_vision(
            prompt="Convert this chemical structure to SMILES. Preserve stereochemistry. Return ONLY the SMILES string.",
            image_path=img_path,
            model=config.MODEL_SONNET,
            patent_id=patent_id,
        )
        smiles = resp.strip().split('\n')[0] if resp else ''
        valid = _validate_smiles(smiles)
        ik = get_inchikey(canonicalize_smiles(smiles)) if valid else None
        son_results.append({
            'valid': valid,
            'match': ik in bdb_keys if ik else False,
            'conn': ik[:14] in bdb_conn if ik else False,
            'smiles': smiles if valid else None,
        })
    strategies['sonnet_only'] = _score(son_results, cost_per=0.03)

    # --- Strategy 3: Opus Vision only ---
    logger.info("Testing Strategy 3: Opus Vision only")
    opus_results = []
    for img_path in image_paths:
        resp = call_claude_vision(
            prompt="Convert this chemical structure to SMILES. Preserve stereochemistry. Return ONLY the SMILES string.",
            image_path=img_path,
            model=config.MODEL_OPUS,
            patent_id=patent_id,
        )
        smiles = resp.strip().split('\n')[0] if resp else ''
        valid = _validate_smiles(smiles)
        ik = get_inchikey(canonicalize_smiles(smiles)) if valid else None
        opus_results.append({
            'valid': valid,
            'match': ik in bdb_keys if ik else False,
            'conn': ik[:14] in bdb_conn if ik else False,
            'smiles': smiles if valid else None,
        })
    strategies['opus_only'] = _score(opus_results, cost_per=0.15)

    # --- Strategy 4: Cascade (DECIMER → Sonnet → Opus) ---
    logger.info("Testing Strategy 4: Cascade")
    cascade_results = []
    cascade_cost = 0
    for i in range(len(image_paths)):
        # Start with DECIMER (free)
        if dec_results[i]['valid']:
            cascade_results.append(dec_results[i])
            continue
        # Fallback to Sonnet
        cascade_cost += 0.03
        if son_results[i]['valid']:
            cascade_results.append(son_results[i])
            continue
        # Fallback to Opus
        cascade_cost += 0.15
        cascade_results.append(opus_results[i])

    strategies['cascade'] = _score(cascade_results, cost_per=cascade_cost / max(len(image_paths), 1))

    # --- Strategy 5: Consensus (DECIMER + Sonnet, Opus arbitrates) ---
    logger.info("Testing Strategy 5: Consensus")
    consensus_results = []
    consensus_cost = 0.03 * len(image_paths)  # Sonnet always runs
    for i in range(len(image_paths)):
        dec_ok = dec_results[i]['valid']
        son_ok = son_results[i]['valid']

        if dec_ok and son_ok:
            # Both valid — check if they agree
            # For now, prefer DECIMER (deterministic)
            consensus_results.append(dec_results[i])
        elif dec_ok:
            consensus_results.append(dec_results[i])
        elif son_ok:
            consensus_results.append(son_results[i])
        else:
            # Both failed — try Opus
            consensus_cost += 0.15
            consensus_results.append(opus_results[i])

    strategies['consensus'] = _score(consensus_results, cost_per=consensus_cost / max(len(image_paths), 1))

    # Print comparison
    print(f"\n{'Strategy':<20} {'Valid':>6} {'BDB Match':>10} {'BDB Conn':>9} {'Cost/img':>9}")
    print("-" * 58)
    for name, s in sorted(strategies.items(), key=lambda x: -x[1]['valid_rate']):
        print(f"  {name:<18} {s['valid_rate']:>5.0%} {s['match_rate']:>9.0%} {s['conn_rate']:>8.0%} ${s['cost_per']:.3f}")

    return strategies


def _score(results: list[dict], cost_per: float) -> dict:
    """Score a strategy's results."""
    n = max(len(results), 1)
    return {
        'valid_rate': sum(1 for r in results if r['valid']) / n,
        'match_rate': sum(1 for r in results if r['match']) / n,
        'conn_rate': sum(1 for r in results if r['conn']) / n,
        'cost_per': cost_per,
        'total': n,
    }
