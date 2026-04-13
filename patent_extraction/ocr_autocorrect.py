"""Levenshtein-based OCR autocorrect for IUPAC name fragments.

When OPSIN fails on a name, find the closest valid IUPAC fragment
by edit distance and substitute. More generalizable than manual
regex rules — catches novel OCR errors automatically.

Uses a dictionary of common IUPAC ring/group fragments.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Dictionary of valid IUPAC fragments commonly found in pharma patents
IUPAC_FRAGMENTS = [
    # Ring systems
    "pyrrolo", "pyrrole", "pyrazol", "pyrazole", "pyridin", "pyridine",
    "pyrimidin", "pyrimidine", "piperazin", "piperazine", "piperidin",
    "piperidine", "morpholin", "morpholine", "thiomorpholin", "thiomorpholine",
    "imidazol", "imidazole", "triazin", "triazine", "triazol", "triazole",
    "oxazol", "oxazole", "thiazol", "thiazole", "oxetan", "azetidin",
    "pyrrolidin", "pyrrolidine", "tetrahydro", "dihydro",
    "benzamide", "benzonitrile", "benzoic", "benzene", "phenyl",
    "cyclopropyl", "cyclobutyl", "cyclopentyl", "cyclohexyl",
    "oxazolidin", "isoxazol",

    # Common substituents
    "methyl", "ethyl", "propyl", "isopropyl", "butyl", "trifluoromethyl",
    "difluoro", "fluoro", "chloro", "bromo", "iodo", "amino", "hydroxy",
    "methoxy", "acetyl", "carboxamide", "sulfonamide", "methanesulfonyl",

    # Fusion descriptors
    "pyrrolo[2,1-f]", "pyrrolo[1,2-f]", "pyrrolo[1,2-a]",
    "pyrazolo[3,4-d]", "pyrazolo[4,3-c]",
]


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr

    return prev[-1]


def autocorrect_iupac(name: str, max_distance: int = 2) -> str:
    """Attempt to fix OCR errors in an IUPAC name using edit distance.

    Scans the name for fragments that are close (edit distance ≤ max_distance)
    to known valid IUPAC fragments, and substitutes the closest match.

    Args:
        name: The IUPAC name with potential OCR errors.
        max_distance: Maximum edit distance for a correction (default 2).

    Returns:
        Corrected name, or original if no corrections found.
    """
    corrected = name
    changes = []

    # Extract potential fragment words from the name
    # Split on common delimiters: -, (, ), [, ], ,
    tokens = re.split(r'([-\(\)\[\],])', name)

    for i, token in enumerate(tokens):
        if len(token) < 4:  # Skip short tokens
            continue
        if token in '- ( ) [ ] ,':  # Skip delimiters
            continue

        token_lower = token.lower()

        # Check if this token is already a valid fragment
        if token_lower in [f.lower() for f in IUPAC_FRAGMENTS]:
            continue

        # Find closest valid fragment
        best_match = None
        best_dist = max_distance + 1

        for fragment in IUPAC_FRAGMENTS:
            if abs(len(token) - len(fragment)) > max_distance:
                continue  # Length difference too large
            dist = _levenshtein(token_lower, fragment.lower())
            if dist <= max_distance and dist < best_dist:
                best_dist = dist
                best_match = fragment

        if best_match and best_dist > 0:
            # Preserve original casing pattern
            if token[0].isupper():
                replacement = best_match[0].upper() + best_match[1:]
            else:
                replacement = best_match

            tokens[i] = replacement
            changes.append(f"'{token}' → '{replacement}' (dist={best_dist})")

    if changes:
        corrected = ''.join(tokens)
        logger.info(f"OCR autocorrect: {', '.join(changes)}")

    return corrected
