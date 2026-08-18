"""Find a substituent table we could not turn into molecules, and fingerprint it.

THE THIRD TIER. `gap.py` finds a table whose VALUES we could not read;
`name_gap.py` finds a NAME OPSIN would not parse. This finds a table whose rows
name a compound that exists only as a scaffold plus substituents — nothing is
malformed, and there is simply no molecule until the pieces are joined.

WHAT MAKES THIS TIER DIFFERENT, AND HARDER
-------------------------------------------
The other two tiers are judged against something the document already states.
A layout rule is judged on whether records came out. A name repair is judged by
re-running OPSIN. A markush row states no answer at all: `markush=True` rows
carry 0 InChIKeys by design, because a markush NAME denotes a set.

So the first question is not "what is the rule" but "what could possibly say
this assembly is wrong". Measured over the corpus, the answer is better than
the wiki recorded:

    653 data rows in the four real substituent tables
    591 (90.5%) print their own m/z, IN A COLUMN
    653 (100%) give at least one substituent as text

`CLAUDE.md` says only one of eleven markush patents prints an MS. That was
measured with a mass finder that read `MS (ESI...)` and nothing else; the
substituent tables print `MS (m/e)` as a column header over `481.0 (M + H)`.
The referee was there the whole time.

WHAT IS FINGERPRINTED
----------------------
The LAYOUT, never the patent — the same discipline as `gap.layout_signature`.
A plan bought for `Ar | R1 | drawing | MS` is worth reusing on every table of
that shape, and worth nothing if it is keyed to US9718825.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from ..sources import markush as MK
from ..sources import mass_gate, opsin
from ..sources.uspto_assays import CID, SUBSTITUENT, UNKNOWN, build_columns
from ..sources.uspto_xml import assemble_blocks, parse_tables

logger = logging.getLogger(__name__)

# A cell that states a compound NAME rather than a substituent: long, and made
# of the characters nomenclature uses. Held-out rows like these are referee #2.
_NAMEISH = re.compile(r"^[A-Za-z0-9\[\]\(\)\{\},'’\-\+\s]{25,}$")


@dataclass
class MarkushGap:
    """One substituent table whose rows have no molecule yet."""
    patent_id: str
    table_id: str
    fingerprint: str
    signature: str                       # the pre-hash form, for neighbour search
    scaffold_ref: str = ""               # `<chemistry>` id of the shared core
    rows: list = field(default_factory=list)          # MarkushRow
    headings: list = field(default_factory=list)      # slot column headings
    names: dict = field(default_factory=dict)         # slot text -> OPSIN smiles
    printed_mass: dict = field(default_factory=dict)  # cid -> m/z the row prints
    held_out: dict = field(default_factory=dict)      # cid -> the row's own name
    n_rows: int = 0

    @property
    def referees(self) -> list[str]:
        """What could CONFIRM an assembly of this table. May be empty.

        Order is priority order, and an empty list is a real answer: nothing in
        the document can check this table, so the loop must mark rather than
        adopt.

        CROSS-ROW INVARIANCE IS DELIBERATELY NOT HERE. It can only refute — all
        rows sharing a core and differing from each other is consistent with
        every row being the wrong molecule in the same way. Counting it as a
        referee would let a table with no mass and no name enter the loop,
        fail `MarkushOutcome.positive` on the missing evidence it was never
        going to have, and spend `MAX_ATTEMPTS` paid calls proving it. It still
        runs inside `measure`, where refuting is exactly its job.
        """
        out = []
        if self.printed_mass:
            out.append("printed_mass")
        if self.held_out:
            out.append("held_out_name")
        return out


def layout_signature(headings: list[str], n_slots: int, has_image: bool,
                     has_mass: bool, scaffold_where: str) -> str:
    """The shape, normalised. Two tables with this signature want one plan.

    Headings are lower-cased and stripped of their digits so `R1 | R2` and
    `R3 | R4` are the same shape — the NUMBER is data the row supplies, not
    part of the layout. The count is kept, because two slots and three slots
    are different problems.
    """
    heads = sorted({re.sub(r"\d+", "#", h.strip().lower()) for h in headings})
    return (f"slots={n_slots}|heads={','.join(heads)}|"
            f"image={int(has_image)}|mass={int(has_mass)}|scaffold={scaffold_where}")


def _scaffold_of(xml: str, table_id: str) -> tuple[str, str]:
    """`(chemistry id, where it was found)`. Both `""` when there is none."""
    from ..sources.cid_first import _header_scaffolds, _intro_scaffolds

    intro = _intro_scaffolds(xml)
    if table_id in intro:
        return intro[table_id], "intro"
    for tid, scaf in _header_scaffolds(xml).values():
        if tid == table_id:
            return scaf, "banner"
    return "", "none"


def find_gaps(patent_id: str, xml: str) -> list[MarkushGap]:
    """Every substituent table in this patent that yielded no molecule."""
    from ..sources.cid_first import _markush_cids

    markush_tables = set(_markush_cids(xml).values())
    if not markush_tables:
        return []
    masses = mass_gate.reported_masses(xml)
    out: list[MarkushGap] = []
    for t in assemble_blocks(parse_tables(xml)):
        if t.table_id not in markush_tables:
            continue
        vals = MK.slot_values(t.body_rows, t.header_rows)
        names = _resolve_names(vals, patent_id)
        rows = MK.classify(t.body_rows, t.header_rows, names=names)
        if not rows:
            continue
        headings = sorted({h for r in rows for h in r.slots})
        scaf, where = _scaffold_of(xml, t.table_id)
        held = _held_out_names(t)
        printed = {r.cid: masses[r.cid] for r in rows if r.cid in masses}
        sig = layout_signature(
            headings, len(headings),
            any(r.fragment_ref for r in rows), bool(printed), where)
        out.append(MarkushGap(
            patent_id=patent_id, table_id=t.table_id,
            fingerprint=hashlib.sha1(sig.encode()).hexdigest()[:12],
            signature=sig, scaffold_ref=scaf, rows=rows, headings=headings,
            names=names, printed_mass=printed, held_out=held, n_rows=len(rows)))
    return out


def _resolve_names(values: list[str], patent_id: str) -> dict:
    """`{slot text -> one-point SMILES}` for the cells that are NAMES.

    One OPSIN call for the whole table. What it refuses is not a failure — it
    is a condensed formula, and `markush._group_fragment` reads those under a
    grammar whose gate is the written hydrogen count.
    """
    if not values:
        return {}
    smiles = opsin.radicals(values, patent_id=patent_id)
    return {v: s for v, s in zip(values, smiles) if s}


def _held_out_names(table) -> dict:
    """`{cid -> the compound name this row prints}`, for the rows that print one.

    REFEREE #2, AND THE ONLY ONE THAT CHECKS THE WHOLE MOLECULE. A row that
    names its compound can be assembled from scaffold-plus-substituents anyway
    and the two answers compared. Where the mass agrees only to 1.5 Da, this
    agrees or does not.

    A name column is found by `build_columns`, never by cell length: a long
    cell is as likely to be an NMR trace, and US11548900 prints both.
    """
    try:
        cols = build_columns(table)
    except Exception:
        return {}
    id_col = next((c.index for c in cols if c.kind == CID), None)
    if id_col is None:
        return {}
    name_cols = [c.index for c in cols
                 if c.kind in (UNKNOWN, SUBSTITUENT)
                 and re.search(r"\bname\b", c.header or "", re.I)]
    if not name_cols:
        return {}
    out = {}
    for row in table.body_rows:
        if len(row) <= id_col:
            continue
        cid = (row[id_col].text or "").strip().split()
        if not cid:
            continue
        for i in name_cols:
            if i >= len(row):
                continue
            txt = " ".join((row[i].text or "").split())
            if _NAMEISH.match(txt):
                out[cid[0]] = txt
                break
    return out


def sample(gap: MarkushGap, n: int = 3) -> str:
    """The smallest fragment that still explains the table, for a model.

    Same economics as `gap.Gap.sample`: a header and a few rows, a few hundred
    tokens. Asking about a layout must not cost what re-reading the document
    costs, or the tier has no advantage over doing it by hand.
    """
    lines = [f"table {gap.table_id}   slot columns: {', '.join(gap.headings)}",
             f"shared scaffold drawing: {gap.scaffold_ref or 'NONE FOUND'}"]
    for r in gap.rows[:n]:
        bits = "  ".join(f"{k}={v!r}" for k, v in r.slots.items())
        lines.append(f"  cid {r.cid}: {bits}"
                     + (f"  drawing={r.fragment_ref}" if r.fragment_ref else "")
                     + (f"  prints m/z {gap.printed_mass[r.cid]}"
                        if r.cid in gap.printed_mass else ""))
    return "\n".join(lines)
