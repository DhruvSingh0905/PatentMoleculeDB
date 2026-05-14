"""Stage 0 — Universal text normalization.

Applied to EVERY page before any downstream processing. The order
matters:
  1. Mojibake repair (UTF-8 → Latin-1 → UTF-8 double-encoding artifacts)
  2. HTML entity unescape (`&gt;` → `>`)
  3. Unicode NFKC normalization (collapses compatibility decompositions:
     fullwidth digits, ligatures, subscripts where it makes sense)

Steps 1 and 2 must run BEFORE step 3 because NFKC may alter byte
sequences that the mojibake table is keyed on.

`_MOJIBAKE_REPAIRS` is the canonical authority for the artifacts we've
seen across patents. Adding entries here is a generic improvement
that benefits ALL patents — never add a patent-specific repair.
"""
from __future__ import annotations

import html
import unicodedata


# Common UTF-8 → Latin-1 → UTF-8 double-encoding artifacts. The repair
# is order-independent because each key is unique. Originally extracted
# from `routes/google_assays.py:_MOJIBAKE_REPAIRS` and extended.
_MOJIBAKE_REPAIRS: dict[str, str] = {
    # Greek letters used in unit symbols and chemistry
    "Î¼": "μ",   "Âµ": "µ",
    "Î²": "β",   "Î±": "α",   "Î³": "γ",   "Î´": "δ",
    "Î»": "λ",   "Î¸": "θ",   "Ï€": "π",
    # Math operators used as qualifiers
    "â‰¤": "≤",  "â‰¥": "≥",
    "â‰ˆ": "≈",  "â‰ª": "≪",  "â‰«": "≫",
    # Subscripts (often in IC₅₀, etc.)
    "â‚‚": "₂",  "â‚ƒ": "₃",  "â‚„": "₄",
    "â‚…": "₅",  "â‚†": "₆",  "â‚‡": "₇",
    "â‚ˆ": "₈",  "â‚‰": "₉",  "â‚€": "₀",
    # Superscripts and signs
    "Â²": "²",   "Â³": "³",   "Â°": "°",   "Â±": "±",
    # Punctuation / quotes (UTF-8-as-Latin-1)
    "â€™": "'",  "â€\x9c": "\"", "â€\x9d": "\"",
    "â€\x98": "'",
    # Em-dash / en-dash / minus sign — used as null markers in many
    # patent assay tables. Without this repair the byte-sequence
    # `â\x80\x94` reaches the tokenizer as garbage.
    "â\x80\x94": "—",   # em-dash
    "â\x80\x93": "–",   # en-dash
    "â\x80\x92": "‒",   # figure dash
    "â\x88\x92": "−",   # minus sign
}


def repair_mojibake(text: str) -> str:
    """Apply known UTF-8 → Latin-1 → UTF-8 double-encoding reversals.

    Idempotent: running twice produces the same result. Free, O(n) per
    pattern. Patent-agnostic.
    """
    for bad, good in _MOJIBAKE_REPAIRS.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def normalize_page(text: str) -> str:
    """Apply Stage 0 normalization in the canonical order.

    Args:
        text: Raw page text (may contain mojibake, HTML entities,
              compatibility-form Unicode).

    Returns:
        Normalized text — same length offsets are NOT preserved
        (HTML unescape collapses `&gt;` 4 chars → 1 char, etc.). Use
        the returned text for all downstream processing.
    """
    if not text:
        return text
    text = repair_mojibake(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return text
