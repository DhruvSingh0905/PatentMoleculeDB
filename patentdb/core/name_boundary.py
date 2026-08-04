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

# NMR and MS readouts captured where a name should be. A systematic name never
# contains a chemical shift, a coupling constant, a multiplicity or a spectro-
# meter frequency, so any of these is proof the chunk is characterisation data:
#
#   `1H-NMR (DMSO-d6,400 MHz) δ (ppm): 7.66-7.64 (m,1H),7.55-7.49 (m,2H)`
#
# `terminate_name` alone could not reject these. It cuts TRAILING prose, and
# these are spectra from character zero with nothing in front to keep — 130 of
# the 310 compounds that have a name and no structure are this exact shape,
# 125 of them in one patent.
_SPECTRA_RE = re.compile(
    r"\bNMR\b|δ|\(ppm\)|\bMHz\b|\bm/z\b|\bHz\b"
    r"|\((?:m|s|d|t|q|dd|dt|br\s?s),\s*\d+\s*H\)|\bJ\s*=", re.IGNORECASE)

# A systematic name is lowercase-dominant: locants and brackets aside, it is
# built from lowercase morphemes. An all-caps token run is a column header or
# an abbreviation list (`QC- ACN- TFA- XB`), never a name.
_HAS_WORD = re.compile(r"[a-z]{3}")


def looks_like_spectra(text: str) -> bool:
    """Is this characterisation data rather than a chemical name?"""
    return bool(text) and bool(_SPECTRA_RE.search(text))


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


def terminate_name(chunk: str) -> str:
    """The chemical name at the front of `chunk`, without the prose after it.

    Returns "" when nothing name-like survives — the caller's existing signal
    to skip the chunk. Returns the whole (stripped, de-mojibaked) chunk when no
    boundary is found: a name this cannot cut is one it must not mangle.

    There is deliberately NO minimum length. An earlier version rejected a cut
    shorter than twelve characters as "prose from the start", which threw away
    `Benzamide` out of `Benzamide MS (ESI) m/z 435.2` and, before that, the
    bare names `pyrene`, `chrysene` and `9H-fluorene`. Short is not the same as
    wrong; what matters is whether the surviving text READS like a name, which
    is tested at the end.
    """
    s = demojibake((chunk or "").strip())
    if not s:
        return s
    # NOT tested here. Checking for spectra BEFORE the cut throws away every
    # legitimate name that happens to be followed by its own MS or NMR line —
    # `...carboxamide MS (ESI) m/z 435.2` is a name plus data, and the whole
    # point of this function is to keep the first part. The test is applied to
    # what SURVIVES the cut, at the end.
    if not _HAS_WORD.search(s):
        return ""
    cut, found = len(s), False
    for rx in (_PROSE_RE, _OPENER_RE, _ANALYTICAL_RE, _AMOUNT_RE, _SENTENCE_RE):
        m = rx.search(s)
        if m and m.start() < cut:
            cut, found = m.start(), True
    out = s[:cut].strip()
    # A name never ends on an open bracket or a joining hyphen.
    while out and out[-1] in "([{-,;:":
        out = out[:-1].rstrip()
    if not out:
        return ""
    # What survived the cut must still look like a NAME. A chunk that was
    # spectra from the start has no boundary to cut at — `1H-NMR (DMSO-d6,400
    # MHz) δ (ppm): 7.66-7.64 (m,1H)` survives whole — so this is where it is
    # caught. Applied AFTER the cut so `...carboxamide MS (ESI) m/z 435.2`
    # keeps its name instead of being discarded with its data.
    # ...and it must OPEN like one. A name starts with a letter, a locant
    # digit, or an opening bracket — never with a closing bracket or a comma.
    # `") can be treated with base to give the acids 37 ..."` cuts down to
    # `") can be"`, which has a lowercase word and no spectra and would
    # otherwise survive as a name.
    if out[0] not in "([{" and not out[0].isalnum():
        return ""
    if looks_like_spectra(out) or not _HAS_WORD.search(out):
        return ""
    return out
