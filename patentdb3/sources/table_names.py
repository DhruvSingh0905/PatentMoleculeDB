"""Compound names stated directly in TABLE CELLS, resolved by OPSIN.

WHY THIS EXISTS
----------------
`uspto_xml.description_text` drops `<tables>` wholesale before any name gets
extracted. That is the right call for VALUES — a table of IC50s flattened
into prose is an unparseable run of numbers — but it is not true of NAMES.
Some substituent tables carry the fully-enumerated compound name beside its
own R-groups: US10376513 TABLE-US-00001 heads a column `Name` next to
`R2`/`R4`/`R5`/`R3`, and its row `cid=69` reads

    (2R)-1-(3-{3-[1-(4-Amino-3- methyl-1H-pyrazolo[3,4- d]pyrimidin-1-
    yl)ethyl]-5-eyloro-6- fluoro-2-methoxyphenyl}azetidin- 1-yl)propan-2-ol

The patent already did the Markush enumeration; there is a concrete name
sitting there, keyed by the SAME compound number ("69") the assay tables
already use. This module reads it. It does not enumerate anything itself —
see `markush/` for tagging, and note that module does no enumeration either.

WHAT COUNTS AS A "NAME COLUMN" — TWO SIGNALS, NOT ONE
-------------------------------------------------------
  A. EXPLICIT HEADER. A column whose assembled header text contains the word
     "Name" (`Name`, `Compound Name`, `IUPAC Name`, ...). The strongest
     signal available; a cell found this way is kept even when its row's id
     column is blank (a stereoisomer continuation row legitimately carries
     no id of its own, and OPSIN — not the presence of an id — is still the
     acceptance gate for whether the text is a name at all).

  B. UNLABELED SHAPE. Patents also publish tables shaped like `Ex. | <blank>
     | <blank> | LCMS`, where the compound name sits in a column with no
     header text at all — "obvious from context" to a human, invisible to a
     header search. US10214537's TABLE-US-00015 through TABLE-US-00062 are
     almost all this shape, and header-only detection alone recovers well
     under half of that patent's name-bearing cells (measured: 402 vs 805
     once this signal is added — see the corpus table in this module's
     report). A column is accepted this way only when
     `uspto_assays.build_columns` could not classify it as anything at all
     (`kind == UNKNOWN` — so it is provably not a CID/NMR/MS/MW/RT/structure/
     substituent column), a majority of its non-empty cells are long and
     hyphen/bracket-heavy (the same shape `uspto_xml._looks_like_chem_name`
     tests for), AND the row carries a non-blank value in the table's own id
     column. That last condition exists because this signal alone is weaker
     than an explicit header and DOES fire on things that are not names:
     measured on US10376513, dropping the id requirement adds 69 cells, all
     of them in rows with no id at all — wrapped legend/footnote prose that
     happens to be hyphen-heavy, not compound names. Requiring an id removes
     every one of them and costs nothing on the rows that are genuine.

Every candidate cell, from either signal, is rejected first if it matches the
same NMR/MS "this is prose, not a name" shape that `uspto_xml._PROSE_CELL`
polices for header rows — US10172859 TABLE-US-00003 has a genuine `Name`
column, but interleaved characterisation rows (blank id, an NMR shift list in
that same column position, because the row is really a continuation line
with nothing else to put there) share the column with real names. Kept here
as an independent, local copy rather than importing the private name — this
module does not depend on `uspto_xml`'s or `uspto_assays`'s underscore-
prefixed internals anywhere, only their public functions
(`parse_tables`, `assemble_blocks`, `build_columns`, `normalize_cid`), so it
keeps working if those modules' private helpers are renamed or rewritten.

THE LINE-WRAP PROBLEM: A SPACE INJECTED MID-CELL, NOT A SPLIT ACROSS ENTRIES
------------------------------------------------------------------------------
`uspto_xml.py`'s own module docstring describes line-wrap as two SEPARATE
`<entry>` elements ("...1H-benzimidazol-2-" / "yl}cyclohexanecarboxylic
acid,"), and v2 (`patentdb_v2/sources/uspto_xml.py`) built `looks_wrapped` /
`join_wrapped_cells` / `join_candidates` to rejoin exactly that shape.
Measured directly against the raw bytes of this corpus's cached XML, that is
NOT what table-cell wrapping looks like here. `US10376513`'s row for cid 69
is ONE `<entry>` end to end; there is no second `<entry>` to join. What is
actually inside that one `<entry>`, at the byte level, is a single literal
ASCII space where the source's typesetting wrapped a printed line:

    ...4-Amino-3-\x20methyl-1H-pyrazolo[3,4-\x20d]pyrimidin-1-yl)ethyl...

verified with `raw.find(b'Amino-3- methyl')` against the cached bytes, not
inferred from the rendered text. A corpus-wide scan of every `<entry\b`+
`<entry\b` pair inside `<tables>` blocks (829 patents' worth of candidate
pairs where one cell ends in a hyphen and the next starts lower-case) found
477 matches, and by-hand inspection of the first 15 shows every one is two
DIFFERENT header labels stacked in adjacent columns (`DNA-` / `pDNA-`), not a
split name — zero genuine cross-entry name wraps were found. So
`join_wrapped_cells`'s cross-`<entry>` join is not needed here and is
deliberately NOT ported; what v3's cells need is a rejoin WITHIN one cell.

`join_candidates` (v2) is also not ported, for a reason specific to what it
solves: it exists because a hyphen AT a v2 break point is ambiguous — the
break might have landed on a real nomenclature hyphen or might not have, so
v2 asks OPSIN about both readings. That ambiguity does not arise here: v3's
defect is an injected SPACE, and the hyphen beside it is never in question —
it is present in the raw cell, unambiguously real ("azetidin-1-yl" needs its
hyphen to mean anything), and the question is only whether the space beside
it is real or wrap noise. So this module asks OPSIN about SPACE removal, not
hyphen removal.

Measured (see `_looks_wrapped_char` below) over every cell in every
identified Name column corpus-wide: of 7,764 such cells carrying an internal
space, the character immediately before the space is a hyphen in 24,283 of
~35,000 tallied adjacencies — the dominant, specific signal — with `,` and
`)` a distant second and third. A close reading of by-hand examples shows
those wrap points are not confined to "space after hyphen": `azaspiro[2.5]
octane`, `imidazo[1,5-a]pyrazin- 3- yl`, `4- {[4-` wrap after `]`, `)`, `-`
and before `{` too. So the rule below is not "hyphen-adjacent" narrowly, but
"adjacent to any bracket/hyphen/comma" — which is also why the ONE naive fix
already tried and rejected in the task brief ("collapse whitespace BETWEEN
word characters") recovered nothing on US10376513: every one of its wrap
spaces sits next to punctuation, which is exactly the set that heuristic
excludes by construction (it only touches a space with a letter/digit on
BOTH sides).

Two dewrap candidates are generated per cell, in addition to the cell as-is,
and OPSIN — not either heuristic — decides which (if any) is right, exactly
the same "brute force, parser is the gate" pattern `iupac_names.py` already
uses for prose:

  - TARGETED: remove only whitespace runs where the character immediately
    before or after is one of `-()[]{},` (the observed wrap-adjacent set).
    A genuine multi-word tail ("...carboxylic acid", "...hydrochloride
    salt") is untouched, because neither side of that space is punctuation.
  - AGGRESSIVE: remove every whitespace run in the cell. A strict superset
    of TARGETED and of the rejected "between word characters" idea, offered
    as a fallback for the cases neither of those precise rules covers
    (single-word splits with no adjacent punctuation at all) — at the cost
    of also breaking a genuine multi-word tail into one token, which then
    simply fails to parse and costs nothing.

TARGETED is tried before AGGRESSIVE (and the untouched cell before both) —
OPSIN keeps the least-invasive candidate that parses, recorded in
`.dewrap` so the corpus measurement can attribute each recovery to a cause.

THE CORRUPTION FAMILY IS EXPLICITLY OUT OF SCOPE
--------------------------------------------------
`5-eyloro` (a character substituted for `chloro`) and `propan-2-ol2` (a
footnote digit fused onto the name — `<sup>2</sup>` keeps its content when
tags are stripped, per `uspto_xml._text`'s own documented behaviour) are a
DIFFERENT defect family, assigned to another agent. Nothing here attempts to
repair either; the corpus measurement in this module's report counts how
much of the remaining residue they account for, without touching them.

WHAT THIS DOES NOT DO
------------------------
No Markush enumeration (the patent already enumerated these), no filtering
on `reagents.classify`'s verdict (labelled, never dropped — same contract
`iupac_names.py` keeps), and no wiring into `process_patent` or any other
caller — this module is deliberately a clean, standalone function.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from ..core import config
from .opsin import batch as _opsin_batch_shared
from .reagents import classify as _classify_reagent
from .uspto_assays import CID, UNKNOWN, build_columns, normalize_cid
from .uspto_xml import Table, assemble_blocks, parse_tables

logger = logging.getLogger(__name__)

# ── name-column detection ────────────────────────────────────────────────

# Signal A: the word "name" anywhere in a column's assembled header text.
_NAME_HEADER = re.compile(r"\bname\b", re.I)

# A local copy, not an import — see "WHAT COUNTS AS A NAME COLUMN" above for
# why this module does not reach into `uspto_xml`/`uspto_assays` privates.
# Same NMR/MS shape `uspto_xml._PROSE_CELL` rejects a header row for: an
# NMR shift entry, a mass-spec trace, a chromatography method line.
_PROSE_LIKE = re.compile(
    r"\(m,|\(d,|\(dd,|\(t,|\(s,|\bJ\s*=|\bppm\b|\bMS\s*:|\bM\s*\+\s*H|"
    r"\bRt\b|\bHPLC\b|\bChiral(?:cel|pak)\b|δ\s*\d", re.I)

# A long, hyphen/bracket-heavy string: the same shape test
# `uspto_xml._looks_like_chem_name` uses for its WIDE substring list, kept
# here as an independent copy for the same reason as `_PROSE_LIKE` above.
_CHEM_SUBSTR = re.compile(
    r'(?:yl|phenyl|methyl|imidazol|pyrimidin|morpholin|triazol|'
    r'chloro|bromo|fluoro|ethyl|propyl|cyclo|oxo|amino|nitro|'
    r'benz|piperid|piperaz|pyridine|pyrazol|oxazol|thiazol)', re.I)

# Minimum length before a cell is even considered a name candidate ("Me",
# "F", "Cl", bare compound ids all fall under this).
MIN_CELL_LEN = 8
# Minimum length + minimum punctuation density for signal B's shape test.
_MIN_CHEM_LEN = 20
# Signal B fires only when a column's non-empty cells clear this fraction
# (and this floor count) as chemical-name-shaped.
_MIN_CHEM_SAMPLES = 3
_MIN_CHEM_FRACTION = 0.5


def _looks_chem_shaped(text: str) -> bool:
    """Same test as `uspto_xml._looks_like_chem_name`: long, and either a
    chemical-name substring or heavy hyphen/bracket punctuation."""
    if len(text) < _MIN_CHEM_LEN:
        return False
    return bool(_CHEM_SUBSTR.search(text)) or (
        text.count("-") + text.count("[") + text.count("(")) >= 3


def _column_headers(table: Table) -> list[str]:
    """Per-column header text, respecting CALS `namest`/colspan.

    Deliberately NOT `uspto_assays.merge_header`: that function's offset
    search scores candidate alignments with `classify_column`, which
    penalises a column it cannot name (`UNKNOWN` with non-empty text scores
    -1) — exactly the case for a bare "Name" header, since `classify_column`
    has no idea what a name column is. That scoring is correct for its own
    job (aligning assay columns) and actively fights this one. Every table
    seen in this corpus with an explicit "Name" header is a single
    full-width header row with no positional ambiguity (`namest` pinned or
    row width == table width), so a plain left-to-right accumulation,
    honouring `col_start` when the source declares it, reproduces the
    correct column position without needing — or risking — that scoring.
    """
    cols: list[list[str]] = [[] for _ in range(table.n_cols)]
    for row in table.header_rows:
        pos = 0
        for cell in row:
            span = max(1, cell.colspan)
            start = cell.col_start if cell.col_start >= 0 else pos
            text = cell.text.strip()
            if text:
                for i in range(start, min(start + span, table.n_cols)):
                    cols[i].append(text)
            pos = start + span
    out = []
    for parts in cols:
        seen: set[str] = set()
        uniq = []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                uniq.append(p)
        out.append(" ".join(uniq))
    return out


def _name_columns(table: Table) -> tuple[set[int], set[int], int | None]:
    """(signal-A columns, signal-B columns, cid column index) for one block."""
    headers = _column_headers(table)
    sig_a = {i for i, h in enumerate(headers) if _NAME_HEADER.search(h)}

    try:
        cols = build_columns(table)
    except Exception as e:                     # a malformed block scores nothing
        logger.warning("table_names: build_columns failed on %s: %r",
                        table.table_id, e)
        return sig_a, set(), None
    cid_idx = next((c.index for c in cols if c.kind == CID), None)
    unknown_idx = {c.index for c in cols if c.kind == UNKNOWN}

    sig_b: set[int] = set()
    for i in unknown_idx - sig_a:
        vals = [r[i].text.strip() for r in table.body_rows
                if len(r) > i and r[i].text.strip()]
        if len(vals) < _MIN_CHEM_SAMPLES:
            continue
        n_chem = sum(1 for v in vals
                     if _looks_chem_shaped(v) and not _PROSE_LIKE.search(v))
        if n_chem >= max(_MIN_CHEM_SAMPLES, len(vals) * _MIN_CHEM_FRACTION):
            sig_b.add(i)
    return sig_a, sig_b, cid_idx


# ── line-wrap repair ──────────────────────────────────────────────────────

# Whitespace immediately before OR after one of these is a typesetting wrap
# point, not a real space — see the module docstring for the corpus evidence
# behind this specific character set.
_WRAP_ADJACENT = re.compile(
    r"(?<=[-‐‑‒–—()\[\]{},])\s+"
    r"|\s+(?=[-‐‑‒–—()\[\]{},])")


def dewrap_candidates(text: str) -> list[tuple[str, str]]:
    """`[(label, candidate), ...]`, least-invasive first, for OPSIN to judge.

    `label` is one of "none" / "targeted" / "aggressive" — recorded on the
    accepted `TableName` so the corpus measurement can attribute a recovery
    to a specific cause rather than reporting one undifferentiated total.
    """
    out = [("none", text)]
    seen = {text}

    targeted = _WRAP_ADJACENT.sub("", text)
    if targeted not in seen:
        out.append(("targeted", targeted))
        seen.add(targeted)

    aggressive = re.sub(r"\s+", "", text)
    if aggressive not in seen:
        out.append(("aggressive", aggressive))
        seen.add(aggressive)

    return out


# ── OPSIN ───────────────────────────────────────────────────────────────

def _opsin_batch(names: list[str], fmt: str, patent_id: str = "") -> list[str]:
    """DELEGATES to `sources/opsin.batch` — see that module.

    This was one of THREE independent copies of the same wrapper, all ending in
    `list(out) + [""] * (len(names) - len(out))`, which returns an oversized
    list unchanged when OPSIN gives back more results than it was sent. Every
    caller pairs by position via `zip`, so a mid-list extra silently hands each
    name the next name's structure. Kept as a name because the call sites read
    better for it; it carries no logic of its own.
    """
    return _opsin_batch_shared(names, fmt, patent_id)



# ── data model ────────────────────────────────────────────────────────────

_RELATIVE_STEREO = re.compile(r"[RSEZ]\*")


@dataclass
class TableName:
    """One compound name read from a table cell and confirmed by OPSIN."""
    patent_id: str
    name: str                    # the accepted (possibly dewrapped) string
    smiles: str
    inchikey: str
    raw_cell: str                # the untouched cell text, for audit
    dewrap: str                  # "none" | "targeted" | "aggressive"
    table_id: str
    row_index: int
    column_index: int
    column_signal: str           # "header" | "unlabeled"
    cid: str | None = None       # the patent's own compound number, if present
    source: str = "table"
    label: str = "compound"
    reason: str = ""
    markush: bool = False
    markush_reason: str = ""

    @property
    def key(self) -> str:
        """Structure identity for dedup — namespaced exactly like
        `iupac_names.NamedCompound.key`, and for the same reason: a markush
        (relative-stereo) name carries no InChIKey (see `extract_table_names`
        below), and falling back to bare SMILES would let a markush entry
        collide with an unrelated concrete structure that happens to share
        one arbitrary stereo-resolution. The `"markush::"` prefix keeps the
        two kinds of thing from deduplicating against each other while still
        letting two markush entries with the same generic scaffold collapse.
        A caller comparing this module's output against
        `iupac_names.extract_names`'s MUST compare on `.key`, not raw
        `.inchikey` — two different markush names both carry `inchikey ==
        ""`, which would read as a false match if compared directly.
        """
        if self.markush:
            return "markush::" + (self.smiles or self.name)
        return self.inchikey or self.smiles


# ── extraction ──────────────────────────────────────────────────────────

def extract_table_names(xml: str, patent_id: str = "") -> list[TableName]:
    """Every OPSIN-parseable compound name stated in this patent's tables.

    One entry per (table, row, name-column) cell that resolves — NOT deduped
    by structure, because the point of reading these is the row's own
    compound number, and two different cids naming the same structure is a
    fact worth keeping, not a duplicate to collapse. See the module
    docstring for what counts as a name column, how a wrapped cell is
    repaired, and what is deliberately out of scope.
    """
    try:
        blocks = assemble_blocks(parse_tables(xml))
    except Exception as e:
        logger.warning("table_names: %s — parse_tables/assemble_blocks failed: %r",
                        patent_id, e)
        return []

    # 1. Collect every candidate cell (table_id, row, column, cid, raw text)
    #    before touching OPSIN at all.
    Candidate = tuple  # (table_id, row_idx, col_idx, signal, cid, raw_text)
    candidates: list[tuple[str, int, int, str, str | None, str]] = []

    for t in blocks:
        try:
            sig_a, sig_b, cid_idx = _name_columns(t)
        except Exception as e:
            logger.warning("table_names: %s — %s column detection failed: %r",
                            patent_id, t.table_id, e)
            continue
        if not sig_a and not sig_b:
            continue
        for ri, row in enumerate(t.body_rows):
            cid_text = (row[cid_idx].text.strip()
                        if cid_idx is not None and cid_idx < len(row) else "")
            cid = normalize_cid(cid_text) if cid_text else None
            for ci in sig_a | sig_b:
                if ci >= len(row):
                    continue
                raw = row[ci].text.strip()
                if len(raw) < MIN_CELL_LEN:
                    continue
                if _PROSE_LIKE.search(raw):
                    continue
                signal = "header" if ci in sig_a else "unlabeled"
                # Signal B is weaker than an explicit header; it only earns
                # trust when the row also carries its own id (see module
                # docstring — this is what keeps wrapped legend prose out).
                if signal == "unlabeled" and not cid:
                    continue
                candidates.append((t.table_id, ri, ci, signal, cid, raw))

    if not candidates:
        return []

    # 2. Fan every candidate cell out into its dewrap variants, batch SMILES.
    flat: list[tuple[int, str, str]] = []       # (candidate idx, dewrap label, text)
    for idx, c in enumerate(candidates):
        for label, cand_text in dewrap_candidates(c[5]):
            flat.append((idx, label, cand_text))

    smiles = _opsin_batch([f[2] for f in flat], "SMILES")

    # 3. Least-invasive successful dewrap wins, per candidate cell.
    best: dict[int, tuple[str, str, str]] = {}   # idx -> (dewrap, text, smiles)
    for (idx, label, text), smi in zip(flat, smiles):
        if not smi or idx in best:
            continue
        best[idx] = (label, text, smi)
    if not best:
        return []

    # 4. Second batch, InChIKey — same name string, not the SMILES, exactly
    #    the two-stage pattern `iupac_names.extract_names` uses.
    idxs = sorted(best)
    keys = _opsin_batch([best[i][1] for i in idxs], "StdInChIKey")

    out: list[TableName] = []
    for i, ik in zip(idxs, keys):
        table_id, row_idx, col_idx, signal, cid, raw = candidates[i]
        dewrap, name, smi = best[i]
        verdict = _classify_reagent(name, smi)
        stereo = _RELATIVE_STEREO.findall(name)
        is_markush = bool(stereo)
        # A MARKUSH NAME GETS NO InChIKey — same rule and same reason as
        # `iupac_names.extract_names`: `(1R*,2S*)-X` denotes a SET of
        # stereoisomers, so a single-structure identifier is a false claim.
        # OPSIN still resolves it to SOME concrete key (it has to pick one),
        # and keeping that key would let a downstream InChIKey join silently
        # assert an identity this cell never stated. See `TableName.key`
        # for how dedup against the description route stays correct once
        # this is blank.
        if is_markush:
            ik = ""
        out.append(TableName(
            patent_id=patent_id, name=name, smiles=smi, inchikey=ik,
            raw_cell=raw, dewrap=dewrap, table_id=table_id, row_index=row_idx,
            column_index=col_idx, column_signal=signal, cid=cid,
            label=verdict.label, reason=verdict.reason,
            markush=is_markush,
            markush_reason=("relative_stereo:" + ",".join(stereo)) if stereo else "",
        ))

    n_with_cid = sum(1 for tn in out if tn.cid)
    n_dewrapped = sum(1 for tn in out if tn.dewrap != "none")
    logger.info(
        "table_names: %s — %d candidate cells, %d resolved (%d dewrapped, "
        "%d carry a row id)",
        patent_id, len(candidates), len(out), n_dewrapped, n_with_cid)
    return out
