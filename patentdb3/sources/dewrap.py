"""Typesetting whitespace inside a chemical name, and the candidates that undo it.

WHY THIS IS ITS OWN MODULE
--------------------------
`sources/opsin.py` is the precedent: three modules each grew a private copy of
one wrapper, all three carried the same bug, and fixing the copy you happened
to be reading looked like fixing the problem. This module exists so the same
thing does not happen to the whitespace repair.

The repair was written for TABLE CELLS (`sources/table_names.py`), where USPTO
emits a printed line break as a literal ASCII space inside one `<entry>`:

    ...1-(4-Amino-3-\\x20methyl-1H-pyrazolo[3,4-\\x20d]pyrimidin-1-yl)ethyl...

verified at the byte level against the cached XML, not inferred from rendered
text. **The identical defect is in `<heading>` text**, which the description
route reads and the table route never sees:

    Intermediate 991F: tert-butyl 4-(...)-3,6-dihydropyridine-1 (2H)-carboxylate
    Example 108: 4-(hydroxymethyl)-...-N-(3-(trifluoromethyl) phenyl)benzamide

Same cause, same fix, two callers — so the regex and the candidate generator
live here and both import them. Nothing about the table route's behaviour
changed when this moved; `table_names.dewrap_candidates` is this function,
re-exported under the name its own tests already use.

WHAT `WRAP_ADJACENT` MATCHES, AND WHY THAT SET
----------------------------------------------
Whitespace whose immediate neighbour on EITHER side is one of `-()[]{},`.
Measured over every cell in every identified Name column corpus-wide: of the
7,764 such cells carrying an internal space, the character immediately before
the space is a hyphen in 24,283 of ~35,000 tallied adjacencies, with `,` and
`)` a distant second and third; by-hand reading shows wrap points also land
after `]` and `)` and before `{`. A genuine multi-word tail — `...carboxylic
acid`, `...hydrochloride salt` — is untouched, because neither side of that
space is punctuation.

The one naive alternative already tried and rejected ("collapse whitespace
BETWEEN word characters") recovers nothing on US10376513: every one of its
wrap spaces sits next to punctuation, which is exactly the set that heuristic
excludes by construction.

WHAT THIS MODULE DOES **NOT** DECIDE
------------------------------------
Which candidate is right. It returns them least-invasive-first and OPSIN is
the acceptance gate, same as everywhere else in this package. It also does not
decide WHERE to apply itself: an AGGRESSIVE strip is safe on a table cell,
which is one field with one value in it, and is NOT safe on running
description prose, where a genuine multi-word tail and a sentence boundary
both live. `table_names` asks for all three candidates; `iupac_names` asks
only for `targeted` and says so at its own call site.
"""
from __future__ import annotations

import re

# Whitespace immediately before OR after one of these is a typesetting wrap
# point, not a real space — see the module docstring for the corpus evidence
# behind this specific character set.
WRAP_ADJACENT = re.compile(
    r"(?<=[-‐‑‒–—()\[\]{},])\s+"
    r"|\s+(?=[-‐‑‒–—()\[\]{},])")


def targeted(text: str) -> str:
    """`text` with only the wrap-adjacent whitespace removed."""
    return WRAP_ADJACENT.sub("", text)


def dewrap_candidates(text: str) -> list[tuple[str, str]]:
    """`[(label, candidate), ...]`, least-invasive first, for OPSIN to judge.

    `label` is one of "none" / "targeted" / "aggressive" — recorded on the
    accepted record so a corpus measurement can attribute a recovery to a
    specific cause rather than reporting one undifferentiated total.

    AGGRESSIVE removes every whitespace run: a strict superset of TARGETED,
    offered for a split with no adjacent punctuation at all, at the cost of
    also welding a genuine multi-word tail into one token — which then simply
    fails to parse and costs nothing. Corpus-wide it has never been the
    winning candidate (all 581 cells that needed a dewrap resolved via
    TARGETED); it is kept because 137 patents is not proof about patent 138.
    """
    out = [("none", text)]
    seen = {text}

    tgt = targeted(text)
    if tgt not in seen:
        out.append(("targeted", tgt))
        seen.add(tgt)

    aggressive = re.sub(r"\s+", "", text)
    if aggressive not in seen:
        out.append(("aggressive", aggressive))
        seen.add(aggressive)

    return out
