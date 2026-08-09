"""Did this rule actually repair the compound, or just make something parse?

The `repair/outcome.py` of the name tier, and the conditions are stricter for a
measured reason: **OPSIN acceptance alone is worthless here.**

WHY THIS GATE IS TWO-SIDED WHERE THE TABLE GATE IS NOT
--------------------------------------------------------
The table tier can gate on "it produced records" because a table either yields
rows or it does not. A chemical name is a compositional grammar in which **most
truncations of a valid name are themselves valid names**, so a mutation that
merely parses proves nothing at all. Measured over this corpus before this file
was written:

  - Trimming 1-8 characters off the ends of failing seeds made **11,446 of
    82,398 (13.9%) parse.** The most common text removed to earn it:
    `'yl'` 3,899, `'2-yl'` 1,305, `'5-yl'` 469, `'1-yl'` 391. `-2-yl` is the
    suffix that makes `pyridin-2-yl` a SUBSTITUENT; deleting it yields
    `pyridin`, which OPSIN resolves to pyridine. Every one of those was a
    smaller, different molecule, and OPSIN certified all of them.
  - Splitting names on commas recovered 106 — **median coverage 0.14**, 104 of
    106 below 0.5. It pulls `nicotinamide` out of a 100-character name.
    Commas in an IUPAC name are locant separators.

Shipping either would have attached thousands of wrong structures to real
compound numbers, each individually plausible and none detectable downstream.
So a repair must clear THREE independent conditions, and the last two are what
separates a repair from an amputation:

  1. **OPSIN parses it.** Necessary, nowhere near sufficient.
  2. **Coverage >= 0.9, with containment.** The accepted string must be a
     contiguous alphanumeric sub-piece of the heading's own name text, and
     cover at least 90% of it. `iupac_names._coverage` is reused rather than
     reimplemented — a second definition of coverage is how the first version
     of this project's own before/after measurement came to compare two
     different molecules and score 1.00.
  3. **Corroborated in the document.** The repaired fragment, whitespace
     removed, must appear VERBATIM somewhere else in the same patent. This is
     `name_repair`'s gate and it is mandatory here — `trusted` is never granted
     to a synthesized rule. Corroboration cannot be satisfied by coincidence:
     the search key is the CORRECTED text, which by construction differs from
     what sits at the corrupted position, so a hit can only come from a
     genuinely different, correctly-spelled occurrence.

WHAT THIS CANNOT SEE, so `name_rules.ground()` still exists
-------------------------------------------------------------
A rule that repairs one name correctly while REWRITING fifty already-correct
ones still passes all three conditions — the fifty are not in this gap and are
never measured here. That damage is invisible after the fact and has to be
blocked before, which is `ground()`'s collateral check against the patent's own
already-parsing names. Same division of labour as `rules.ground()` versus
`outcome.measure` on the table side.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..sources import name_repair as NR
from ..sources.iupac_names import _coverage
from ..sources.opsin import batch as opsin_batch
from .name_gap import NameGap
from .name_rules import NamePattern, apply_pattern

logger = logging.getLogger(__name__)

COVERAGE_MIN = 0.9


@dataclass
class NameOutcome:
    """What running this rule on this compound did. Observed, not scored."""
    positive: bool = False
    repaired: str = ""
    smiles: str = ""
    inchikey: str = ""
    coverage: float = 0.0
    corroborated: bool = False
    parsed_any: bool = False        # some candidate parsed, even if rejected
    conflicts: list[str] = field(default_factory=list)

    @property
    def detail(self) -> str:
        """The retry message. Says what happened, never what to do instead."""
        if self.positive:
            return (f"repaired to {self.repaired[:80]!r} — parses, "
                    f"coverage {self.coverage:.2f}, corroborated in the document")
        return "; ".join(self.conflicts) or "the rule produced no candidate at all"


def measure(p: NamePattern, gap: NameGap, *, compiled=None) -> NameOutcome:
    """Run `p` against `gap`'s name and report what came out.

    Every candidate the rule generates is tried; the FIRST that clears all
    three conditions wins, in left-to-right site order, so the result is
    deterministic and does not depend on a score.
    """
    cands = apply_pattern(p, gap.name_text, compiled)
    oc = NameOutcome()
    if not cands:
        oc.conflicts = ["the rule matched nothing in this name"]
        return oc

    smis = opsin_batch([c for c, _ in cands], "SMILES", gap.patent_id)
    doc_norm = NR._normalize_ws(gap.doc_text) if gap.doc_text else ""

    # Reasons collected across candidates so the retry message names the
    # condition that actually stopped it, rather than a generic failure.
    failed_cov: list[str] = []
    failed_corr: list[str] = []
    for (repaired, span), smi in zip(cands, smis):
        if not smi:
            continue
        oc.parsed_any = True
        cov = _coverage(repaired, gap.name_text)
        if cov < COVERAGE_MIN:
            failed_cov.append(
                f"{repaired[:60]!r} parses but covers only {cov:.2f} of the "
                f"name the patent states — that is a fragment of a different, "
                f"smaller molecule, not a repair")
            continue
        probe = NR._probe_window(repaired, span)
        corrob = bool(doc_norm) and NR._normalize_ws(probe) in doc_norm
        if not corrob:
            failed_corr.append(
                f"{repaired[:60]!r} parses and covers {cov:.2f}, but the "
                f"repaired fragment {probe[:40]!r} appears nowhere else in this "
                f"patent — nothing confirms the correction is the right one")
            continue
        keys = opsin_batch([repaired], "StdInChIKey", gap.patent_id)
        oc.positive = True
        oc.repaired = repaired
        oc.smiles = smi
        oc.inchikey = keys[0] if keys else ""
        oc.coverage = cov
        oc.corroborated = True
        return oc

    if not oc.parsed_any:
        oc.conflicts = ["every candidate the rule produced was still rejected "
                        "by OPSIN"]
    else:
        oc.conflicts = (failed_cov + failed_corr)[:3]
    return oc


def summarise(p: NamePattern, oc: NameOutcome) -> str:
    """One line for the log and the journal."""
    return f"{p.id} {'KEPT' if oc.positive else 'REJECTED'}: {oc.detail}"
