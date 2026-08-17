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

**THAT SCAN LOOKED SIDEWAYS ONLY, AND A SECOND WRAP RUNS DOWNWARD.** Every
pair it tested was `<entry>` beside `<entry>` — two columns of the same row.
It never compared a cell with the cell BELOW it, so it could not have found
the other shape, and the other shape is real and common: the same column
index on consecutive `<row>` elements, one compound spread down six rows
(US9763922 TABLE-US-00002). Read one row at a time, as this module did until
`_records` was added, every fragment is rejected by OPSIN and the compound is
lost — or worse, a fragment parses ALONE and a piece of a name is shipped as
the compound's structure. That is not hypothetical: US9763922 cid 124 carried
SMILES `O1CCCCC1` — tetrahydropyran, one substituent of an 11-row name —
until the rejoin replaced it. The refusal above still stands for the
horizontal case; see `_records` for the vertical one, and note that its
record boundary is NOT "the id cell is non-empty", because the id wraps too.

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

THE CORRUPTION FAMILY: A SECOND STAGE, AND THE ORDER IS MEASURED
------------------------------------------------------------------
`5-eyloro` (a character substituted for `chloro`) is a DIFFERENT defect from
a wrap space, and `sources/name_repair.py` is the module that repairs it —
under two gates (OPSIN acceptance, plus verbatim in-document corroboration
for every pattern but one). This module now calls it, as a RESCUE stage on
the cells no dewrap variant resolved, and never on a cell that already
parsed.

**The composition order was measured, not assumed**, because the two stages
mutate the same string on different axes and either could go first:

    A  repair-then-dewrap : generate_candidates(raw) -> dewrap each repaired
    B  dewrap-then-repair : dewrap(raw)              -> generate_candidates each

Run over the FULL 137-patent cached corpus (`output_v3/uspto_xml/`), on the
16,054 name-column cells no dewrap variant resolves:

    order A   206,846 OPSIN candidates   18 cells recovered
    order B   206,785 OPSIN candidates   18 cells recovered
    identical accepted string and SMILES on all 18; 0 recovered by one only

**The order does not matter here, and the reason is specific rather than
lucky:** 17 of the 18 recover at `dewrap == "none"`, and every one of the 18
comes from a BRACKET-STACK pattern (`stray_opening_bracket` 14,
`stray_closing_bracket` 3, `dropped_close_bracket` 1) — scanners that count
brackets and are blind to whitespace, so removing spaces first cannot change
which sites they find. `name_repair`'s three REGEX-based patterns, the ones
whose site detection a wrap space genuinely could break, recover **zero**
table cells, so the axis on which the orders differ is never exercised.
Order B is what is implemented, on the two grounds that separate them at
all: 61 fewer candidates, and it keeps this module's own stage first so
`.dewrap` still records what the dewrap did independently of `.repair`.

All 18 are `corroborated`; none is accepted on OPSIN alone. The yield is
**18 of 16,054 unresolved cells (0.11%)**, for roughly a 5.7x increase in
this module's OPSIN candidate volume (36,157 -> ~243,000 corpus-wide). It is
kept because the cost is local CPU on a java subprocess and $0, and because
the 18 are confirmed rather than guessed — but "the rescue stage recovers a
tenth of a percent of the residue" is the honest headline, not a recovery
story.

`propan-2-ol2` (a footnote digit fused onto the name — `<sup>2</sup>` keeps
its content when tags are stripped, per `uspto_xml._text`'s own documented
behaviour) remains OUT OF SCOPE and is deliberately untouched by both
modules. It is not a hypothetical: **US10376513's cid 69 — the worked
example at the top of this docstring, and `name_repair.ch_as_ey`'s single
confirmed corpus case — carries all three defects at once.** Measured
directly by running the current tree on that cell: TARGETED dewrap removes
every embedded space correctly, `ch_as_ey` proposes the `eyloro` -> `chloro`
fix, and the result still fails OPSIN purely on the trailing `2`. That cell
is the reason `ch_as_ey` contributes 0 of the 18 above.

A THIRD SHAPE: THE VERTICAL RECORD, WHERE A COMPOUND IS A RUN OF ROWS
------------------------------------------------------------------------
Signals A and B both assume one compound per ROW and one field per COLUMN.
US9265734 TABLE-US-00006 publishes 199 HDAC inhibitors the other way round —
the FIELD NAMES run down column 0 and each compound is a consecutive run of
rows:

    Record 1      | -
    Structure     | <chemistry>
    Comp id       | R119
    HDAC1 IC50    | 7000
    (nM)          | -
    HDAC3 IC50    | 1100
    (nM)          | -
    Chemical_name | N-(2-aminophenyl)-6-(phenylsulfonamido)hexanamide

Read a row at a time, column 1 is in turn a cid, two IC50 values and an IUPAC
name, so no single `Column.kind` describes it and `_name_columns` finds
nothing at all. Measured: `extract_table_names("US9265734")` returned **0**
before this section existed, against 199 `Chemical_name` fields the patent
states in full.

**THE DETECTOR IS IMPORTED, NOT REIMPLEMENTED.**
`uspto_assays.vertical_blocks` already decides whether a block is vertical and
where each record starts, and it is public for exactly this reason — its own
docstring says "Public because BOTH tracks lose this layout". Its thresholds
(`_VERT_MAX_LABEL_UNIQ`, `_VERT_MIN_LABEL_ROWS`, the evenly-spaced-anchor
rule) were scored against all 122 two- and three-column tables in the 137
cached patents and match exactly one block. Copying that here is the specific
mistake `sources/opsin.py` exists to prevent: three private copies of one
wrapper all carried the same bug, and fixing the copy in front of you looked
like fixing it. So this module calls `vertical_blocks` and adds only the
question the assay reader has no reason to ask — *which pair is the name*.

What IS local is the pair of label spellings, `_VERT_ID_LABEL` and
`_VERT_NAME_LABEL`. Those are one-line constants of the same class as
`_PROSE_LIKE` and `_CHEM_SUBSTR` above, kept local under the same rule this
module already states — it depends on `uspto_assays`'s public functions and
never on its underscore-prefixed internals, so `_VERT_CID_LABEL` is not
importable without breaking that. Neither constant WIDENS anything: signal
A's `_NAME_HEADER` (`\bname\b`) does not match `Chemical_name` at all, because
`_` is a word character and there is no word boundary in front of `name` —
and widening it to reach one table would change how every header in 137
patents scores. A separate pattern for a separate layout is the precedent
`uspto_assays._VERT_CID_LABEL` sets, for that same reason, one function away.

The pass is gated to blocks that produced no row-wise candidate at all, which
is the same anti-regression rule `uspto_assays.extract_from_tables`'s PASS 4
applies to the assay side: it can only ADD. `column_signal == "vertical"`
marks what it added, so the yield is readable off the artifact without
re-deriving it from the XML.

**A NAME WRAPS DOWNWARD HERE TOO, AND WITH NO LABEL AT ALL.** 7 of the 199
records spill the name onto a following row whose label cell is EMPTY:

    Chemical_name | N-(2-amino-4-fluorophenyl)-6-(6-fluoro-1-oxo-3,4-dihydroisoquinolin-2(1H)-
                  | yl)hexanamide

That is the `_records` case (see below) in a different layout, so it reuses
the same two constants rather than new ones: a label-less row joins the name
above it only when that name is visibly UNFINISHED (`_UNFINISHED_TAIL`), and
the fragments are glued with `_REJOIN_GLUE` and handed to the same
`dewrap_candidates` every other cell goes through. `rows_joined` counts them,
so TARGETED-first ordering in `_variants` applies unchanged.

WHAT THIS DOES NOT DO
------------------------
No Markush enumeration (the patent already enumerated these) and no filtering
on `reagents.classify`'s verdict (labelled, never dropped — same contract
`iupac_names.py` keeps).

It is called by `verify.dump`, gated on `config.IUPAC_NAMES` alongside
`iupac_names.extract_names`, and its rows land in the SAME
`config.STRUCTURES` artifact distinguished by `NamedCompound.source` /
`TableName.source` (`"description"` vs `"table"`). It still merges nothing
and joins nothing: a structure this module reads and a structure the
description route reads are two rows, not one, even when they share a
`.key` — deduplicating them would throw away the row id (`cid`) that is the
whole reason for reading tables in the first place.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from ..core import config
from .dewrap import WRAP_ADJACENT as _WRAP_ADJACENT
from .dewrap import dewrap_candidates
from .name_repair import repair_names as _repair_names
from .opsin import batch as _opsin_batch_shared
from .reagents import classify as _classify_reagent
from .uspto_assays import (
    CID, UNKNOWN, build_columns, normalize_cid, vertical_blocks)
from .uspto_xml import Table, assemble_blocks, description_text, parse_tables

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


def _accept(signal: str, cid: str | None, raw: str) -> bool:
    """Is this cell worth asking OPSIN about? ONE definition, three callers'
    worth of layouts.

    Factored out when the vertical layout was added, so a candidate found by
    reading a table sideways and one found by reading it downward face exactly
    the same three gates — a second copy of these lines would be the drift this
    module's `cell_resolution` docstring already argues against for the OPSIN
    verdict itself.

    Signal B is weaker than an explicit header, so it earns trust only when the
    row also carries its own id (see module docstring — this is what keeps
    wrapped legend prose out). Signal A and the vertical layout both name their
    field outright and are held to no such condition.
    """
    if len(raw) < MIN_CELL_LEN:
        return False
    if _PROSE_LIKE.search(raw):
        return False
    return not (signal == "unlabeled" and not cid)


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
#
# MOVED TO `sources/dewrap.py`, IMPORTED BACK UNDER THE SAME NAMES. The
# whitespace defect this module was written for is not a table defect — the
# identical injected space appears in `<heading>` text, which the description
# route reads and this module never sees. Two callers means the regex and the
# candidate generator live in one place, for exactly the reason
# `sources/opsin.py` exists: three private copies of one wrapper all carried
# the same bug, and fixing the copy in front of you looked like fixing it.
#
# `_WRAP_ADJACENT` and `dewrap_candidates` keep their names here because they
# are this module's public surface and its tests already import them; neither
# carries logic of its own any more. See `dewrap.py` for the corpus evidence
# behind the character set, and for why AGGRESSIVE is offered here (a table
# cell is one field with one value) but not on description prose.


# ── cross-<row> continuation ─────────────────────────────────────────────
#
# A SECOND, VERTICAL WRAP — NOT the cross-`<entry>` join this module's
# docstring refuses. That refusal is about HORIZONTAL adjacency (`<entry>`
# beside `<entry>`, two different columns), and it stands: 477 candidate
# pairs, all of them stacked header labels. This is the orthogonal case —
# the SAME column index on CONSECUTIVE `<row>` elements, which that scan
# never looked at. US9763922 TABLE-US-00002, read verbatim off the assembled
# block:
#
#     [0] col0='Example 7'  col1='1-[3-[6-(1-ethylpyrazol-'
#     [1] col0=''           col1='4-yl)-3,4-dihydro-2H-'
#     [2] col0=''           col1='quinolin-1-yl]-1-(oxetan-'
#     ...
#     [5] col0=''           col1='5-yl]ethanone'
#
# Six rows, one compound. Offered to OPSIN one row at a time — which is what
# this module did before — every fragment is rejected and the compound is
# lost, even though the name is complete and correct once the column is read
# down instead of across.
#
# THE RECORD BOUNDARY IS NOT "THE ID CELL IS NON-EMPTY". When the id column
# is narrow the source wraps the id ITSELF, putting the label on one row and
# its number on the next (TABLE-US-00014: `Example` / `107`, with the name
# starting on the `Example` row). A boundary test of "id cell non-empty"
# splits that record in two and produces two fragments instead of one name —
# measured, not supposed: that exact test scored 0/7 on TABLE-US-00014 and
# 0/3 on TABLE-US-00016 while scoring 20/20 on TABLE-US-00002, and the
# difference between those tables is nothing but where the id wrapped.
#
# So the boundary is the id becoming COMPLETE: a bare label word with no
# number of its own cannot end a record, and the next non-empty id cell
# joins it rather than starting a new one.
_LABEL_ONLY = re.compile(
    r"^(?:examples?|compounds?|cpds?|ex|entry|entries|nos?\.?|#)[\s.:#]*$", re.I)

# AN ID-LESS ROW IS NOT AUTOMATICALLY A CONTINUATION, and assuming it was
# broke a guard this module documents at length. Signal A's own note says a
# stereoisomer continuation row "legitimately carries no id of its own" — such
# a row holds a COMPLETE name, and gluing it onto the record above both
# destroys it and corrupts that record. Two tests state the case directly
# (`test_extract_resolves_signal_b_only_with_a_row_id`, where a fourth id-less
# row named `...-1-butylpiperidine` must be dropped rather than absorbed, and
# `test_us10172859_rescues_exactly_the_measured_cell`).
#
# So a row continues the record above ONLY when that record is visibly
# UNFINISHED — its text so far ends on one of the wrap characters
# `dewrap.WRAP_ADJACENT` was measured for. `...(1-ethylpyrazol-` is unfinished;
# `...-1-propylpiperidine` is not, and the row beneath it starts fresh.
_UNFINISHED_TAIL = re.compile(r"[-‐‑‒–—(\[{,]$")

# Rejoined fragments are glued with ONE SPACE, then handed to the SAME
# `dewrap_candidates` every other cell already goes through — no new
# machinery, and no second guess about which glue is right. TARGETED removes
# a space whose neighbour is `-()[]{},` (`...-6-(1-` + `methylpyrazol...`),
# and leaves a genuine word break alone (`...carboxylic` + `acid`); OPSIN,
# not this rule, decides which reading survives.
_REJOIN_GLUE = " "


def _records(table: Table, cid_idx: int | None,
             name_cols: set[int]) -> list[tuple[int, str, dict[int, str], int]]:
    """Body rows grouped into records: (row_index, cid_text, {col: text}, n_rows).

    With no id column there is no boundary signal at all, so every row is its
    own record and this is exactly the previous behaviour — the rejoin can
    only ever fire where the table states ids.
    """
    if cid_idx is None:
        return [(ri, "", {ci: row[ci].text.strip()
                          for ci in name_cols if ci < len(row)}, 1)
                for ri, row in enumerate(table.body_rows)]

    out: list[tuple[int, str, dict[int, str], int]] = []
    cur_id = ""
    cur_cols: dict[int, list[str]] = {}
    cur_row = -1
    n_rows = 0

    def flush() -> None:
        if cur_row >= 0:
            out.append((cur_row, cur_id,
                        {ci: _REJOIN_GLUE.join(p).strip()
                         for ci, p in cur_cols.items()}, n_rows))

    def unfinished() -> bool:
        """Does the open record end mid-name in EVERY name column it fills?"""
        if cur_row < 0:
            return False
        parts = [p for p in cur_cols.values() if p]
        return bool(parts) and all(_UNFINISHED_TAIL.search(p[-1]) for p in parts)

    for ri, row in enumerate(table.body_rows):
        id_text = (row[cid_idx].text.strip()
                   if cid_idx < len(row) else "")
        if id_text and cur_row >= 0 and _LABEL_ONLY.match(cur_id):
            # the open record's id is still just a label word — this cell is
            # the rest of THAT id, not the start of a new record
            cur_id = f"{cur_id} {id_text}".strip()
        elif id_text or not unfinished():
            # a new id, or an id-less row the record above has no claim on
            flush()
            cur_id, cur_cols, cur_row, n_rows = id_text, {}, ri, 0
        n_rows += 1
        for ci in name_cols:
            frag = row[ci].text.strip() if ci < len(row) else ""
            if frag:
                cur_cols.setdefault(ci, []).append(frag)
    flush()
    return out


# ── vertical records: one compound per RUN of rows ───────────────────────
#
# See "A THIRD SHAPE" in the module docstring. The DETECTION is
# `uspto_assays.vertical_blocks`, imported; only the two label spellings and
# the name field's own downward wrap live here.

# The vertical layout's spelling of "this pair is the compound's id". A
# verbatim local copy of `uspto_assays._VERT_CID_LABEL` — that name is
# private, and this module's contract (module docstring, "WHAT COUNTS AS A
# NAME COLUMN") is that it imports that module's public functions and none of
# its underscore-prefixed internals. Same rule that already makes `_PROSE_LIKE`
# and `_CHEM_SUBSTR` local copies rather than imports.
_VERT_ID_LABEL = re.compile(
    r"^\s*(?:comp(?:ound)?|cpd|example|entry)\s*"
    r"(?:\.?\s*(?:id|no|number|#))?\s*[:.]?\s*$", re.I)

# `Chemical_name`, `Compound Name`, `IUPAC name`, bare `Name`. NOT `_NAME_HEADER`
# widened: `\bname\b` cannot match `Chemical_name` (no word boundary after `_`),
# and loosening the pattern that scores every column header in 137 patents to
# buy one table is the trade `uspto_assays._VERT_CID_LABEL` refuses by name.
# A FULL-STRING match, unlike signal A's substring search, because a vertical
# label IS the whole field name — `Structure` and `LC/MS Calc'd (M + H)` sit in
# the same column and must not be read as name fields.
_VERT_NAME_LABEL = re.compile(
    r"^\s*(?:(?:chemical|chem|compound|cpd|iupac|systematic|product)"
    r"[\s_.\-]*)*names?\s*[:.]?\s*$", re.I)

# `vertical_blocks` reads `(col 0, col 1)` of a two-column tgroup, so every
# value it returns came from column index 1. Recorded on the row rather than
# left at 0 so `TableName.column_index` still names the cell it was read from.
_VERT_VALUE_COL = 1


def _vertical_candidates(
        tables: list[Table], skip: set[str],
) -> list[tuple[str, int, int, str, str | None, str, int]]:
    """Name candidates from vertical-record blocks, in `cell_resolution` shape.

    `skip` holds the table ids that already produced a row-wise candidate. A
    block in that set is left alone, so this pass can only ADD — the same
    anti-regression gate `uspto_assays.extract_from_tables`'s PASS 4 applies on
    the assay side, and for the same reason: a layout read two ways would emit
    the same compound twice.

    `row_index` is the RECORD's index within the block, not a `<row>` index.
    A vertical record spans a dozen rows; the record ordinal is what identifies
    it, and it is what the assay side of the same block counts by.

    Takes RAW tgroups, not `assemble_blocks`'s grid: `_header_rows_of` promotes
    this block's first data row (`Comp id | R119`) to a header and loses that
    compound with it — measured at 198 records instead of 199, the same defect
    `_vertical_pairs` documents.
    """
    by_block: dict[str, list[Table]] = {}
    for t in tables:
        by_block.setdefault(t.table_id, []).append(t)

    out: list[tuple[str, int, int, str, str | None, str, int]] = []
    for block_id, raw_block in by_block.items():
        if block_id in skip:
            continue
        try:
            blocks = vertical_blocks(raw_block)
        except Exception as e:                 # a malformed block scores nothing
            logger.warning("table_names: %s — vertical_blocks failed: %r",
                            block_id, e)
            continue
        for bi, block in enumerate(blocks):
            cid = ""
            for lab, val in block:
                if val and _VERT_ID_LABEL.match(lab):
                    cid = normalize_cid(val)   # "" for a non-id; its own gate
                    if cid:
                        break
            # Fragments of each name field, in order. `cur` is the open field:
            # a labelled row that is not a name label closes it, and a
            # label-less row extends it only while the text so far ends
            # mid-name (see `_UNFINISHED_TAIL` — 7 of US9265734's 199 records
            # wrap this way, and nothing else in the block has a blank label).
            fields: list[list[str]] = []
            cur: list[str] | None = None
            for lab, val in block:
                if _VERT_NAME_LABEL.match(lab):
                    cur = [val] if val else None
                    if cur is not None:
                        fields.append(cur)
                elif lab:
                    cur = None
                elif cur and val and _UNFINISHED_TAIL.search(cur[-1]):
                    cur.append(val)
                else:
                    cur = None
            for parts in fields:
                out.append((block_id, bi, _VERT_VALUE_COL, "vertical",
                            cid or None, _REJOIN_GLUE.join(parts).strip(),
                            len(parts)))
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
    column_signal: str           # "header" | "unlabeled" | "vertical"
    cid: str | None = None       # the patent's own compound number, if present
    source: str = "table"
    label: str = "compound"
    reason: str = ""
    markush: bool = False
    markush_reason: str = ""
    # `name_repair`'s pattern id when this cell only resolved after a
    # CONFIRMED character/bracket repair applied on top of the dewrap; "" when
    # the dewrapped (or untouched) cell parsed on its own. A non-empty value
    # is the ONLY way to tell a cell that needed the rescue stage from one
    # that did not, which is what the stage's yield is measured on — see
    # "THE CORRUPTION FAMILY" in the module docstring.
    repair: str = ""
    # How many `<row>` elements this name was read from. 1 is the ordinary
    # case; >1 means the cross-`<row>` rejoin fired (see `_records`). Kept as
    # a count rather than a flag so the rejoin's yield can be measured off the
    # artifact alone, without re-deriving it from the XML.
    rows_joined: int = 1
    # Set by `sources/mass_gate.py`. Same contract as the identically-named
    # fields on `iupac_names.NamedCompound` — read that docstring, not a second
    # copy of it here. Blank means UNCHECKED, never "fine".
    mass_check: str = ""
    mass_delta: str = ""

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

def cell_resolution(xml: str, patent_id: str = "") -> tuple[
        list[tuple[str, int, int, str, str | None, str, int]],
        dict[int, tuple[str, str, str, str]]]:
    """`(cells, resolved)` — the ONE definition of "did this table cell resolve".

    `cells` is every candidate a name column offered; `resolved` maps a cell's
    index to the `(dewrap, accepted name, smiles, repair)` that won for it. A
    cell ABSENT from `resolved` produced no structure at all.

    Split out for `repair/name_gap.py`, and `extract_table_names` is built on
    top of it rather than beside it — the same discipline
    `iupac_names.heading_resolution` carries, and for the same reason. A
    `cid | Name` row is the patent asserting in its own voice that a compound
    with that number is named here, exactly as a heading does; the heal loop
    could not see any of them because the gap detector only ever read
    headings. On US10376513 that is the difference between 9 gaps and 155:
    its Name columns are detected correctly and every cell fails OPSIN on a
    `<sup>` footnote digit fused to the tail (`...propan-2-ol2`).

    Asking this question a second way would be the specific bug
    `heading_resolution` documents — a re-implementation drifts from the
    shipping extractor, and the loop then buys rules for failures that do not
    exist.
    """
    try:
        tabs = parse_tables(xml)
        blocks = assemble_blocks(tabs)
    except Exception as e:
        logger.warning("table_names: %s — parse_tables/assemble_blocks failed: %r",
                        patent_id, e)
        return [], {}

    # 1. Collect every candidate cell (table_id, row, column, cid, raw text)
    #    before touching OPSIN at all.
    # (table_id, row_idx, col_idx, signal, cid, raw, n_rows)
    candidates: list[tuple[str, int, int, str, str | None, str, int]] = []

    for t in blocks:
        try:
            sig_a, sig_b, cid_idx = _name_columns(t)
        except Exception as e:
            logger.warning("table_names: %s — %s column detection failed: %r",
                            patent_id, t.table_id, e)
            continue
        if not sig_a and not sig_b:
            continue
        name_cols = sig_a | sig_b
        for ri, cid_text, cells, n_rows in _records(t, cid_idx, name_cols):
            cid = normalize_cid(cid_text) if cid_text else None
            for ci, raw in cells.items():
                signal = "header" if ci in sig_a else "unlabeled"
                if not _accept(signal, cid, raw):
                    continue
                candidates.append((t.table_id, ri, ci, signal, cid, raw, n_rows))

    # 1b. The vertical layout, on blocks the row-wise pass found nothing in.
    #     `skip` is what makes this strictly additive — see
    #     `_vertical_candidates` and "A THIRD SHAPE" in the module docstring.
    seen_blocks = {c[0] for c in candidates}
    candidates.extend(
        c for c in _vertical_candidates(tabs, seen_blocks)
        if _accept(c[3], c[4], c[5]))

    if not candidates:
        return [], {}

    def _variants(c) -> list[tuple[str, str]]:
        """Dewrap variants for one candidate, least-invasive first — EXCEPT on
        a rejoined record, where "least-invasive" would mean keeping a space
        this module injected itself.

        `_REJOIN_GLUE` puts a space at every fragment boundary, and OPSIN
        tolerates most of them, so `none` wins and the stored name carries
        artefacts of our own joining (`pyrazolo[4,3- c]pyridin-5- yl`).
        TARGETED removes exactly the punctuation-adjacent spaces — which is
        what every glue space is — and leaves a genuine word break alone, so
        it is tried first here. Verified before adopting: on all 701 rejoined
        names in the 20-patent sample that parsed as-is with a glue space,
        TARGETED yields the IDENTICAL SMILES (0 different, 0 unparseable), so
        this reorders spellings and cannot reorder answers.
        """
        variants = dewrap_candidates(c[5])
        if c[6] > 1:
            variants = sorted(variants, key=lambda v: v[0] != "targeted")
        return variants

    # 2. Fan every candidate cell out into its dewrap variants, batch SMILES.
    flat: list[tuple[int, str, str]] = []       # (candidate idx, dewrap label, text)
    for idx, c in enumerate(candidates):
        for label, cand_text in _variants(c):
            flat.append((idx, label, cand_text))

    smiles = _opsin_batch([f[2] for f in flat], "SMILES", patent_id)

    # 3. Least-invasive successful dewrap wins, per candidate cell.
    best: dict[int, tuple[str, str, str, str]] = {}   # idx -> (dewrap, text, smiles, repair)
    for (idx, label, text), smi in zip(flat, smiles):
        if not smi or idx in best:
            continue
        best[idx] = (label, text, smi, "")

    # 3b. THE RESCUE STAGE — `name_repair` on what dewrapping alone could not
    #     save. See "THE CORRUPTION FAMILY" in the module docstring for the
    #     measurement that fixed the composition ORDER (dewrap first, repair
    #     second) and for what the stage is worth.
    unresolved = [i for i in range(len(candidates)) if i not in best]
    if unresolved:
        # Corroboration needs the patent's own flat text. `description_text`,
        # NOT `anchor.anchor_text`: `anchor_text` applies its own
        # dropped-open-paren repair to the document, and corroborating a
        # proposed repair against text we already repaired ourselves would be
        # circular — `name_repair`'s whole claim is that the corrected
        # spelling occurs in the patent AS PUBLISHED, somewhere the corruption
        # did not reach. `include_headings=True` because a compound's clean
        # restatement is usually its own `<heading>`, which is exactly the
        # text `description_text`'s default drops.
        doc = description_text(xml, include_headings=True)
        probes: list[tuple[int, str]] = []          # (candidate idx, dewrap label)
        strings: list[str] = []
        for i in unresolved:
            for label, cand_text in _variants(candidates[i]):
                probes.append((i, label))
                strings.append(cand_text)
        for (i, label), res in zip(probes, _repair_names(strings, doc, patent_id)):
            if res is None or i in best:
                continue
            best[i] = (label, res.repaired, res.smiles, res.pattern_id)
        n_rescued = sum(1 for v in best.values() if v[3])
        logger.info("table_names: %s — %d cell(s) unresolved after dewrap, "
                    "%d rescued by name_repair", patent_id, len(unresolved),
                    n_rescued)

    return candidates, best


def extract_table_names(xml: str, patent_id: str = "") -> list[TableName]:
    """Every OPSIN-parseable compound name stated in this patent's tables.

    One entry per (table, row, name-column) cell that resolves — NOT deduped
    by structure, because the point of reading these is the row's own
    compound number, and two different cids naming the same structure is a
    fact worth keeping, not a duplicate to collapse. See the module
    docstring for what counts as a name column, how a wrapped cell is
    repaired, and what is deliberately out of scope.

    The candidate collection and the OPSIN verdict live in `cell_resolution`,
    which `repair/name_gap.py` also calls; this function only turns the cells
    that WON into rows. One definition, two readers.
    """
    candidates, best = cell_resolution(xml, patent_id)
    if not best:
        return []

    # 4. Second batch, InChIKey — same name string, not the SMILES, exactly
    #    the two-stage pattern `iupac_names.extract_names` uses.
    idxs = sorted(best)
    keys = _opsin_batch([best[i][1] for i in idxs], "StdInChIKey", patent_id)

    out: list[TableName] = []
    for i, ik in zip(idxs, keys):
        table_id, row_idx, col_idx, signal, cid, raw, n_rows = candidates[i]
        dewrap, name, smi, repair = best[i]
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
            repair=repair, rows_joined=n_rows,
        ))

    n_with_cid = sum(1 for tn in out if tn.cid)
    n_dewrapped = sum(1 for tn in out if tn.dewrap != "none")
    n_repaired = sum(1 for tn in out if tn.repair)
    n_rejoined = sum(1 for tn in out if tn.rows_joined > 1)
    logger.info(
        "table_names: %s — %d candidate cells, %d resolved (%d dewrapped, "
        "%d repaired, %d rejoined across rows, %d carry a row id)",
        patent_id, len(candidates), len(out), n_dewrapped, n_repaired,
        n_rejoined, n_with_cid)
    return out
