"""Did this assembly plan build the right molecules? Measured, on this table.

THE ONE GATE, and the reason this tier could be built at all. A markush row
states no structure, so there is nothing to compare an assembly against — until
you notice that the table states three other things, and that all three can be
wrong.

    1. THE PRINTED MASS. 591 of the 653 rows in the corpus's four real
       substituent tables print their own m/z, in a column. Assemble the row,
       weigh it, compare. This is decisive: a wrong join at a wrong atom
       changes the formula and the number moves by tens of Da.

    2. A HELD-OUT NAME. Some rows print a compound name beside the drawing.
       Assemble those rows from scaffold-plus-substituents ANYWAY, ignoring
       the name, and compare InChIKeys. Where the mass agrees only to within
       1.5 Da, this agrees exactly or not at all.

    3. CROSS-ROW INVARIANCE. Every row of one table is the same core with
       different substituents. So every assembled molecule must contain a
       common scaffold, and they must not all be identical. A slot wired to
       the wrong atom breaks the first; a slot not wired at all makes every
       row come out the same, which is the failure that looks most like
       success.

WHY THERE IS NO SCORE
----------------------
`outcome.py` learned this the expensive way: 88 rules were adopted over the
objection of three judgement gates, and the gates were wrong every single time
— 28,281 rows scored, zero contradictions. A gate that fires and is never right
does not need a better threshold. So nothing here has an opinion about what a
good plan looks like. It runs the plan and reports what came out.

ADOPTION NEEDS A REFEREE, AND SILENCE IS NOT ONE
-------------------------------------------------
`positive` is False when nothing could check the table. US10626094's 31 rows
print no mass and no name; an assembly of them is unfalsifiable, and adopting
an unfalsifiable answer is how a wrong structure ships looking like a right
one. Those rows stay marked, which is what was decided for the OPSIN-limited
compounds and is the same reasoning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..sources import mass_gate

logger = logging.getLogger(__name__)

# How many of the rows that CAN be checked must agree before a plan is kept.
# Not tuned: a plan is one slot map applied to every row of one table, so it is
# either right for the table or wrong for it. A majority that is not near-total
# means the plan is right about some rows by accident, and `contradicted`
# below refuses it regardless of what this says.
MIN_AGREE = 0.6


@dataclass
class MarkushOutcome:
    built: int = 0                    # rows that produced a legal molecule
    failed: int = 0                   # rows the builder refused, by name
    mass_agrees: int = 0
    mass_contradicts: int = 0
    name_agrees: int = 0
    name_disagrees: int = 0
    share_scaffold: bool = True
    all_identical: bool = False
    refused: dict = field(default_factory=dict)     # error -> count
    referee: str = "none"

    @property
    def checked(self) -> int:
        return (self.mass_agrees + self.mass_contradicts
                + self.name_agrees + self.name_disagrees)

    @property
    def agreed(self) -> int:
        return self.mass_agrees + self.name_agrees

    @property
    def contradicted(self) -> bool:
        """Any row the document positively disagrees with kills the plan.

        Not a rate. One row whose printed mass says the molecule is 40 Da out
        is a statement by the patent that this assembly is wrong, and a plan is
        one rule for the whole table — so it is wrong for the whole table.
        """
        return bool(self.mass_contradicts or self.name_disagrees)

    @property
    def positive(self) -> bool:
        if not self.built or self.all_identical or not self.share_scaffold:
            return False
        if self.contradicted:
            return False
        if not self.checked:
            return False              # UNFALSIFIABLE. See the module docstring.
        return self.agreed / self.checked >= MIN_AGREE


def measure(gap, assembled: dict) -> MarkushOutcome:
    """`assembled` is `{cid -> smiles}` from applying one plan to one table."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")

    oc = MarkushOutcome()
    mols: dict[str, object] = {}
    for cid, smi in assembled.items():
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            oc.failed += 1
            continue
        oc.built += 1
        mols[cid] = m

    for cid, m in mols.items():
        want = gap.printed_mass.get(cid)
        if want is None:
            continue
        got = ExactMolWt(m) + mass_gate.PROTON
        if abs(got - want) <= mass_gate.tolerance(want):
            oc.mass_agrees += 1
        else:
            oc.mass_contradicts += 1

    for cid, m in mols.items():
        want = gap.held_out.get(cid)
        if not want:
            continue
        from ..sources import opsin
        ref = opsin.batch([want], patent_id=gap.patent_id)
        r = Chem.MolFromSmiles(ref[0]) if ref and ref[0] else None
        if r is None:
            continue                  # OPSIN could not read it: no verdict
        if Chem.MolToInchiKey(r) == Chem.MolToInchiKey(m):
            oc.name_agrees += 1
        else:
            oc.name_disagrees += 1

    if len(mols) > 1:
        keys = {Chem.MolToInchiKey(m) for m in mols.values()}
        oc.all_identical = len(keys) == 1
        oc.share_scaffold = _share_a_core(list(mols.values()))

    oc.referee = ("printed_mass" if oc.mass_agrees or oc.mass_contradicts
                  else "held_out_name" if oc.name_agrees or oc.name_disagrees
                  else "none")
    return oc


def _share_a_core(mols: list) -> bool:
    """Do all these molecules contain one common substructure worth calling a core?

    Cheap and deliberately weak: the maximum common substructure of a whole
    table is expensive and this only has to catch the case where a slot was
    wired to the wrong atom, which shatters the shared ring system. A core is
    "at least half the smallest molecule".
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    if len(mols) < 2:
        return True
    sample = mols[:8]                 # the shape repeats; eight settles it
    try:
        res = rdFMCS.FindMCS(sample, timeout=10, completeRingsOnly=True,
                             ringMatchesRingOnly=True)
    except Exception as e:
        logger.debug("MCS failed, not treating that as a failure: %r", e)
        return True
    if res.canceled or not res.smartsString:
        return True                   # no verdict is not a negative verdict
    smallest = min(m.GetNumHeavyAtoms() for m in sample)
    return res.numAtoms >= smallest * 0.5


def summarise(oc: MarkushOutcome) -> str:
    bits = [f"built {oc.built}", f"refused {oc.failed}"]
    if oc.mass_agrees or oc.mass_contradicts:
        bits.append(f"mass {oc.mass_agrees} agree / {oc.mass_contradicts} contradict")
    if oc.name_agrees or oc.name_disagrees:
        bits.append(f"name {oc.name_agrees} agree / {oc.name_disagrees} disagree")
    if oc.all_identical:
        bits.append("EVERY ROW IDENTICAL — a slot is not wired")
    if not oc.share_scaffold:
        bits.append("NO SHARED CORE — a slot is wired to the wrong atom")
    if not oc.checked:
        bits.append("NOTHING COULD CHECK THIS TABLE")
    return "; ".join(bits)
