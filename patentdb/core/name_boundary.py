"""Where does the chemical name stop and the synthesis paragraph begin?

An example header in a patent reads

    Example 109
    1-(3-(4-Amino-...)phenyl)-N-(3,3-difluorocyclobutyl)-3,3-difluoro-
    cyclobutanecarboxamide To 3,3-difluorocyclobutyl-1-amine (4.99 mg,
    0.047 mmol) was added Intermediate N27 in DMF ...

and the name is only the first line. The extractor took the whole chunk, so
466 to 791 characters of procedure were stored as `iupac_name`. The name at
the FRONT is correct — this is an unterminated capture, not a misread — and it
costs structures, because a name with a paragraph glued to it does not parse:

    name <= 120 chars   1688 names   96.6% resolve to a structure
    name 121-250         906 names   81.0%
    name >  250          165 names   37.0%

The previous terminator was a list of nine literal phrases (`was prepared`,
`MS (ESI)`, `Step A`, ...). Enumerating prose is a losing game — it missed
`To 3,3-difluorocyclobutyl-1-amine`, `A solution of LiHMDS`, `Intermediate
132A:` and `was suspended in` on the first five patents looked at.

So cut on GRAMMAR instead of on phrases. Two signals, both of which a
systematic chemical name cannot contain:

  A PROSE VERB OR CONNECTIVE as a whitespace-delimited word. IUPAC names are
  single tokens joined by hyphens, brackets and locants; a bare ` was `,
  ` added `, ` stirred ` is always English. Multi-word name tails that DO
  occur (`... hydrochloride`, `compound with methanesulfonic acid`) contain
  none of these.

  A SENTENCE START — a capitalised word after a lowercase one, where the
  capitalised word is an English opener (`To`, `A`, `The`, `Intermediate`,
  `Step`). Names capitalise their first letter and their element symbols, not
  mid-name words.

OPSIN is not consulted. It is a 270 ms JVM subprocess per call, so searching
for the longest parsable prefix would cost hours across the corpus; it is used
to MEASURE this module (`scripts/eval/name_boundary_check`), not to run it.
"""
from __future__ import annotations

import re

# Words that are English prose and never a standalone token in a systematic
# name. Deliberately not "of", "acid", "ester" or "with": those appear in real
# name tails such as `compound with methanesulfonic acid`.
_PROSE_WORDS = (
    "was", "were", "is", "are", "has", "have", "been", "being",
    "added", "stirred", "cooled", "warmed", "heated", "obtained", "gave",
    "give", "given", "afford", "afforded", "yielded", "purified", "washed",
    "concentrated", "filtered", "dissolved", "suspended", "treated",
    "prepared", "synthesized", "isolated", "diluted", "extracted", "dried",
    "quenched", "charged", "degassed", "evaporated", "removed", "collected",
    "then", "after", "under", "using", "according", "followed", "following",
    "described", "above", "below", "title", "mixture", "solution",
    "suspension", "residue", "filtrate", "product", "yield", "purification",
    "chromatography", "which", "that", "this", "these", "there", "into",
    "onto", "over", "during", "while", "whereupon", "affording", "giving",
)
_PROSE_RE = re.compile(r"\s+(?:" + "|".join(_PROSE_WORDS) + r")\b", re.IGNORECASE)

# A capitalised sentence opener sitting after a lowercase letter or a digit —
# `...carboxamide To 3,3-...`, `...-2-one Intermediate 132A:`.
_OPENERS = ("To", "A", "An", "The", "After", "Under", "Into", "In", "At", "By",
            "For", "From", "Then", "When", "While", "Intermediate", "Step",
            "Example", "Preparation", "Synthesis", "Method", "Procedure",
            "Compound", "Scheme", "General", "Alternatively", "Similarly",
            "Analogously", "Purification", "Reference", "Note", "Yield")
_OPENER_RE = re.compile(r"(?<=[a-z0-9\)\]])\s+(?:" + "|".join(_OPENERS) + r")\b")

# Characterisation data, which follows the name directly with no verb.
_ANALYTICAL_RE = re.compile(
    r"\s+(?:MS|LC[-\s]?MS|LCMS|HRMS|ESI|APCI|NMR|1H\s*NMR|13C\s*NMR|HPLC|"
    r"TLC|Rf|R\.?t\.?|Retention|m/z|calcd|Calcd|Anal)\b")

# Reagent quantities — `(4.99 mg, 0.047 mmol)`. A name never states an amount.
_AMOUNT_RE = re.compile(r"\s*\(\s*[\d.]+\s*(?:mg|g|mL|ml|L|mmol|mol|µL|uL|equiv)\b")

# A sentence boundary: full stop, then a space and a capital. `2.5` and
# `N,N-dimethyl` are unaffected because both sides must be non-digit.
_SENTENCE_RE = re.compile(r"(?<=[a-z\)\]])\.\s+(?=[A-Z])")

_MOJIBAKE = re.compile(r"Ã.|Â.|â€.|î.|Î¼")


def demojibake(text: str) -> str:
    """Undo UTF-8 that was decoded as Latin-1 — `Î¼l` -> `μl`, `â78Â°` -> `−78°`.

    Applied only when it demonstrably helps: the repaired string must contain
    strictly fewer mojibake markers than the original. A correct string that
    happens to hold a real `Â` (rare, but legal) is therefore left alone, and
    a double-encoded one is not made worse by a second pass.
    """
    if not text or not _MOJIBAKE.search(text):
        return text
    try:
        fixed = text.encode("latin-1", "strict").decode("utf-8", "strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return fixed if len(_MOJIBAKE.findall(fixed)) < len(_MOJIBAKE.findall(text)) else text


def terminate_name(chunk: str, *, min_len: int = 12) -> str:
    """The chemical name at the front of `chunk`, without the prose after it.

    Returns the whole (stripped, de-mojibaked) chunk when no boundary is
    found — a name this cannot cut is one it must not mangle.
    """
    s = demojibake((chunk or "").strip())
    if not s:
        return s
    cut, found = len(s), False
    for rx in (_PROSE_RE, _OPENER_RE, _ANALYTICAL_RE, _AMOUNT_RE, _SENTENCE_RE):
        m = rx.search(s)
        if m and m.start() < cut:
            cut, found = m.start(), True
    # A boundary inside the first few characters means the chunk was never a
    # name: it is prose from character zero, a Scheme description that picked
    # up an example number —
    #
    #   ") can be treated with base to give the acids 37, followed by
    #    reduction to the aldehydes 38. Enamine formation with optimally
    #    substituted amines ..."
    #
    # Returning the whole string here (the first version did) leaves 791
    # characters of narrative in `iupac_name`, where OPSIN fails on it and the
    # LLM cascade is then paid to "clean" prose into a plausible-but-wrong
    # structure. Return EMPTY instead, which is the caller's existing signal to
    # skip the chunk entirely.
    #
    # `found` matters. Without it this tested the CUT POSITION, and for a string
    # containing no boundary at all that is simply its length — so the short
    # REAL names `pyrene`, `chrysene`, `anthracene`, `as-indacene` and
    # `9H-fluorene` were thrown away for being under twelve characters. A short
    # name is not prose; only an early BOUNDARY means prose.
    if found and cut < min_len:
        return ""
    out = s[:cut].strip()
    # A name never ends on an open bracket or a joining hyphen.
    while out and out[-1] in "([{-,;:":
        out = out[:-1].rstrip()
    return out or s
