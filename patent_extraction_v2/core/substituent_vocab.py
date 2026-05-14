"""Cross-patent substituent name → SMILES expansion (used by markush text-side).

Three layers, fast → slow, free → paid:

  Layer 1: ALKYL_RANGE_EXPANDER
    Programmatic. Handles "(Cn-Cm)-alkyl" / "(Cn-Cm)-alkenyl" /
    "(Cn-Cm)-alkynyl" / "(Cn-Cm)-cycloalkyl" patterns common in
    pharmaceutical patent claim text. Generates concrete SMILES for
    every chain length in the range (linear-only; isomers aren't
    enumerated here — the global fragment vocab in markush.enumerate
    handles diversity expansion). Free, deterministic.

  Layer 2: CHEMBL_FRAGMENT_LIBRARY
    A downloaded curated name→SMILES table for ~10K named substituents
    (see download_chembl_fragments.py for the one-time download). Free
    on lookup, ~10MB resident.

  Layer 3: ADAPTIVE LM CACHE   (NOT YET IMPLEMENTED)
    LM-populated cache for novel patterns the above layers miss.
    Fires once per new pattern, caches forever. Per the rebuild plan:
    "adaptive caches over hand-curated rules". Stub left for later phase.

These layers are accessed via `expand_substituent(text)`, which is
the single entry point the text-Markush extractor calls AFTER the
existing SUBSTITUENT_LIBRARY (~80 entries) misses.

Generalizability rule
---------------------
None of the patterns reference patent IDs. The range syntax variations
(parens vs unsubscripts vs LaTeX subscripts) are normalized to a
single form before extraction so the pattern set stays small.
"""
from __future__ import annotations

import re

# ── Helper — normalize the noisy patent shorthand for ranges ────────

# OCR / LaTeX produces a wide variety of representations of "C1 to C4":
#   (C1-C4), (C_1-C_4), (c_{1}-c_{4}), C1-4, c1-c4, _1_4, etc.
# We collapse all of them into a single canonical (n, m) tuple.
#
# Strategy: strip LaTeX subscript markup (_, {, }), strip stray spaces
# and parens, lowercase the chain-type word, then regex-extract the two
# integers + the chain type.

# `_LATEX_RANGE_PAIR` runs FIRST (before `_LATEX_BRACES`) to preserve the
# range-separator info: PaddleX/MinerU OCR commonly emits "(C_{1}-C_{4})"
# as "(_1_4)" where the dash and the C-prefix are gone. After
# `_LATEX_BRACES` strips the underscore prefix, "_1_4" becomes "14"
# (the two integers run together) and the range is unrecoverable. By
# running this pre-pass, "_1_4" → "1-4" first, then the rest of
# normalization proceeds normally.
_LATEX_RANGE_PAIR = re.compile(r"_\{?(\d+)\}?\s*[-]?\s*_\{?(\d+)\}?")
_LATEX_BRACES = re.compile(r"_\{?(\d+)\}?")          # "_1" / "_{1}" → "1"
_NORM_WHITESPACE = re.compile(r"\s+")
_NORM_DASHES = re.compile(r"[‐‑‒–—―]+")             # all unicode dash-likes → "-"
_RANGE_RE = re.compile(
    r"""
    \(?\s*                       # optional opening paren (sometimes lost in OCR)
    [cC]?\s*(\d+)                # first carbon count, optional 'C' prefix
    \s*[-=]\s*                   # range separator (- or =, OCR confuses them)
    [cC]?\s*(\d+)                # second carbon count
    \s*\)?                       # optional closing paren
    \s*-?\s*                     # optional joining dash
    (alk(?:yl|enyl|ynyl)|cycloalk(?:yl|enyl|ynyl))   # chain type
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize(text: str) -> str:
    """Collapse the dialectal variations of `(Cn-Cm)-alkyl` into
    a single canonical form before the range regex runs.

    Order matters: the LaTeX-range-pair pass MUST run before the
    plain LaTeX-brace stripper, otherwise "_1_4" loses its separator
    and becomes "14" (unrecoverable as a range).
    """
    s = text.strip().lower()
    s = _LATEX_RANGE_PAIR.sub(r"\1-\2", s)  # "_1_4" → "1-4" (preserve range)
    s = _LATEX_BRACES.sub(r"\1", s)         # remaining "_{n}"/"_n" → n
    s = _NORM_DASHES.sub("-", s)            # unicode dashes → "-"
    s = _NORM_WHITESPACE.sub(" ", s)        # multiple spaces → 1
    return s


# ── Layer 1: programmatic range expander ────────────────────────────

def _linear_alkyl(n: int) -> str:
    """SMILES of n-carbon linear alkyl (n ≥ 1). methyl=C, ethyl=CC, ..."""
    return "C" * n


def _linear_alkenyl(n: int) -> str:
    """SMILES of n-carbon linear alk-1-enyl (n ≥ 2). ethenyl=C=C, ..."""
    if n < 2:
        return ""
    # CH2=CH-(CH2)_(n-2)
    return "C=C" + "C" * (n - 2)


def _linear_alkynyl(n: int) -> str:
    """SMILES of n-carbon linear alk-1-ynyl (n ≥ 2). ethynyl=C#C, ..."""
    if n < 2:
        return ""
    return "C#C" + "C" * (n - 2)


def _cycloalkyl(n: int) -> str:
    """SMILES of n-carbon cycloalkyl (n ≥ 3). cyclopropyl=C1CC1, ..."""
    if n < 3:
        return ""
    return "C1" + "C" * (n - 1) + "1"


_GENERATORS = {
    "alkyl": _linear_alkyl,
    "alkenyl": _linear_alkenyl,
    "alkynyl": _linear_alkynyl,
    "cycloalkyl": _cycloalkyl,
    "cycloalkenyl": _linear_alkenyl,    # cyclic-with-double-bond is rare here; default to linear alkenyl shape
    "cycloalkynyl": _linear_alkynyl,    # similarly rare
}


def expand_alkyl_range(text: str) -> list[str]:
    """Expand a `(Cn-Cm)-alkyl/alkenyl/...` pattern into concrete SMILES.

    Returns the list of generated SMILES (one per chain length in
    [n, m]) or an empty list if `text` does not match the pattern.

    Examples
    --------
    >>> expand_alkyl_range("(C1-C4)-alkyl")
    ['C', 'CC', 'CCC', 'CCCC']
    >>> expand_alkyl_range("(c_{1}-c_{4}) -alkyl")
    ['C', 'CC', 'CCC', 'CCCC']
    >>> expand_alkyl_range("(C2-C6)-alkenyl")
    ['C=C', 'C=CC', 'C=CCC', 'C=CCCC', 'C=CCCCC']
    """
    s = _normalize(text)
    m = _RANGE_RE.search(s)
    if not m:
        return []

    n = int(m.group(1))
    upper = int(m.group(2))
    chain_type = m.group(3).lower()

    if upper < n:
        return []
    if upper - n > 24:                  # sanity cap — patents don't claim > C24 ranges
        return []

    gen = _GENERATORS.get(chain_type)
    if gen is None:
        return []

    smiles_list: list[str] = []
    for k in range(n, upper + 1):
        s_smi = gen(k)
        if s_smi and s_smi not in smiles_list:
            smiles_list.append(s_smi)
    return smiles_list


# ── Layer 1b: plain "alkyl" / "aryl" / etc. with no range specified ──

# Conservative defaults for unranged chain types. Patents that say
# just "alkyl" without a Cn-Cm prefix conventionally mean C1-C6 in
# medicinal chemistry context. We expand to a small library of the
# most common chain lengths so the option list isn't empty for
# downstream enumeration.
_DEFAULT_RANGES = {
    "alkyl": (1, 6),
    "alkenyl": (2, 6),
    "alkynyl": (2, 6),
    "cycloalkyl": (3, 7),
    "cycloalkenyl": (3, 7),
    "cycloalkynyl": (5, 8),
    "alkoxy": (1, 4),         # alkoxy = O-alkyl; we'll prepend O- when expanding
    "alkoxyl": (1, 4),
}


def expand_unranged_chain(text: str) -> list[str]:
    """If `text` is a plain chain-type word (no Cn-Cm range), expand
    it to a default range. Returns [] for anything else.
    """
    norm = text.strip().lower()
    norm = re.sub(r"\s+", "", norm)
    norm = re.sub(r"^optionally\s*substituted\s*", "", norm)

    if norm not in _DEFAULT_RANGES:
        return []

    n, m = _DEFAULT_RANGES[norm]
    chain_type = norm
    if chain_type in ("alkoxy", "alkoxyl"):
        # O-CH3, O-CH2CH3, ...
        return ["O" + "C" * k for k in range(n, m + 1)]
    gen = _GENERATORS.get(chain_type)
    if gen is None:
        return []
    out: list[str] = []
    for k in range(n, m + 1):
        s_smi = gen(k)
        if s_smi and s_smi not in out:
            out.append(s_smi)
    return out


# ── Layer 2: ChEMBL fragment library (loaded lazily) ────────────────

_CHEMBL_PATH = None        # set by download script; see chembl_fragments.json
_CHEMBL_DICT: dict[str, str] | None = None


def _load_chembl_dict() -> dict[str, str]:
    """Lazy-load the ChEMBL fragment library (name → SMILES).

    Returns an empty dict if the library isn't downloaded yet — callers
    should fall back through the chain transparently.
    """
    global _CHEMBL_DICT, _CHEMBL_PATH
    if _CHEMBL_DICT is not None:
        return _CHEMBL_DICT
    from . import config
    if _CHEMBL_PATH is None:
        _CHEMBL_PATH = config.OUTPUT_DIR / "vocab" / "chembl_fragments.json"
    loaded: dict[str, str] = {}
    if _CHEMBL_PATH.exists():
        import json
        try:
            data = json.loads(_CHEMBL_PATH.read_text())
            if isinstance(data, dict):
                loaded = {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
    _CHEMBL_DICT = loaded
    return loaded


def lookup_chembl(text: str) -> str | None:
    """Look up a substituent name in the ChEMBL fragment library.

    Returns the canonical SMILES for the named fragment, or None if
    not found (or the library hasn't been downloaded yet). Caller
    should fall back to OPSIN.
    """
    d = _load_chembl_dict()
    if not d:
        return None
    norm = text.strip().lower()
    if norm in d:
        return d[norm]
    # Loose match: strip leading "optionally substituted "
    norm2 = re.sub(r"^optionally\s+substituted\s+", "", norm)
    if norm2 in d:
        return d[norm2]
    return None


# ── Public entry point ──────────────────────────────────────────────

def expand_substituent(text: str) -> list[str]:
    """Resolve a substituent text-name to one OR MORE SMILES.

    Order of resolution (each layer feeds back to the caller as a list
    of candidate SMILES):
      1. Range expander  → list of concrete chain-length SMILES
      2. Unranged chain  → list of default chain-length SMILES
      3. ChEMBL lookup   → one named-fragment SMILES (wrapped in list)
      4. None matched    → empty list (caller falls back to OPSIN/LM)

    Returning a list (not a single SMILES) is by design: range
    expansion is one-name-to-many-SMILES. The caller (`text_markush.py`)
    should append all returned SMILES to the R-group's options_smiles.
    """
    if not text:
        return []
    expanded = expand_alkyl_range(text)
    if expanded:
        return expanded
    unranged = expand_unranged_chain(text)
    if unranged:
        return unranged
    chembl = lookup_chembl(text)
    if chembl:
        return [chembl]
    return []
