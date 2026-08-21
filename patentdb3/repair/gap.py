"""Locate WHERE extraction failed, precisely enough to ask a cheap question.

The economics of the repair loop live in this file. HARVEST costs ~$8.81 per
patent (25.7M input tokens across the 22-patent corpus) because it re-reads the
whole document in 6,000-character chunks and asks the model to emit data. The
replacement asks the model for a *rule* instead of data, and to do that cheaply
it must send the smallest fragment that still explains the failure — a header
plus two or three rows, a few hundred tokens, not a chunked document.

So the job here is: given a patent we have already parsed, identify the
specific tables that visibly contain data we did not extract, and cut a minimal,
self-contained sample from each.

A gap is described by a LAYOUT FINGERPRINT rather than a patent id. Layouts
repeat across patents — the same law firm and the same drafting software
produce the same table shapes — so a rule learned once should be reused
everywhere that shape appears, and a rule should never be keyed to the patent
it was learned from.
"""
from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from dataclasses import dataclass, field

from ..sources.uspto_assays import (
    _CID_PAT, ASSAY, CID, build_columns, is_placeholder_name, table_legend,
    _header_rows_of, merge_header,
)
from ..sources.uspto_xml import Table

# A cell that carries a measurement-shaped payload but that we did not read.
_NUMERIC = re.compile(r"^\s*[<>~≈≥≤]?\s*\d*\.?\d+\s*$")
_BIN = re.compile(r"^\s*(\++|[A-E])\s*$")
_NULLISH = {"-", "--", "—", "ND", "NT", "N/A", "NA", "n.d.", "n.t.", ""}
_ASSAY_WORDS = re.compile(r"IC\s*50|EC\s*50|\bKi\b|\bKd\b|inhibit|potenc|activit|assay", re.I)


@dataclass
class Gap:
    """One table whose contents we could not fully turn into records."""
    patent_id: str
    table_id: str
    n_cols: int
    n_data_rows: int
    n_extracted: int
    fingerprint: str
    reason: str
    sample: str                      # the minimal fragment to show a model
    # The fingerprint before hashing — see `layout_signature`. Carried so the
    # rule adopted for this gap can be stored with a comparable shape, which is
    # what lets a later gap be shown the handful of rules learned on layouts
    # most like it instead of nothing at all.
    signature: str = ""
    headers: list[str] = field(default_factory=list)
    column_kinds: list[str] = field(default_factory=list)
    # Cells we could see but could not read. The single most informative thing
    # we can show a model: without them it sees a healthy-looking table and
    # proposes a column_map, because nothing on screen says which cells failed.
    unparsed_examples: list[str] = field(default_factory=list)
    # Text found ANYWHERE in the document that appears to define this table's
    # grade symbols. Position-based capture failed three times; searching by
    # shape is what makes an unseen legend grammar learnable.
    legend_candidates: list[str] = field(default_factory=list)
    # A bigger view of the SAME table, built at detection time and free. Served
    # only if the model asks for it (`request_more_context`). The first sample
    # is deliberately small because it is billed on every gap; this one is paid
    # for only when the model says the small one was not enough.
    expanded_sample: str = ""
    # The table's ORIGINAL CALS XML. The one view we have not transformed, so
    # the only one that can contradict the others — served on request, because
    # a model that says "the source disagrees with what you showed me" has found
    # a parser bug, which is worth more than any rule.
    raw_source: str = ""
    # WHAT THIS GAP IS ASKING FOR. A detector knows which question it raised;
    # without saying so, every gap on a block that happens to carry a legend
    # came back as a `bin_key`, because the legend instruction in `synthesize`
    # is unconditional. US11547697's four PI3K columns share one name AND the
    # block has a `+`/`++` scale, so the loop answered the scale question it
    # was not asked and the columns stayed indistinguishable.
    asks: str = ""

    @property
    def severity(self) -> int:
        """Rows we are leaving on the floor — used to spend budget where it pays."""
        return max(0, self.n_data_rows - self.n_extracted)


def layout_signature(table: Table, headers: list[str]) -> str:
    """The fingerprint's PREIMAGE — the shape itself, before it is hashed.

    `n_cols|shape,shape,…|headerwords,…`, about 60-120 characters. The hash is
    what keys the library; this is what a human or a model can actually read,
    and it is the only form in which one layout can be COMPARED to another.

    That comparison is the point. The library is 172 rules and growing, and a
    model asked about a new layout is shown nothing about what has already been
    learned — so it re-derives from scratch every time and cannot notice that a
    near-identical shape was solved three patents ago. Sending the library is
    not an option (165 KB); sending the few most similar signatures with the
    rule that worked for each costs ~80 tokens apiece.

    Split out rather than inlined so the two uses cannot drift: whatever goes
    into the hash is exactly what similarity is measured on.
    """
    _, data = _header_rows_of(table)
    shapes: list[str] = []
    for i in range(table.n_cols):
        vals = [r[i].text.strip() for r in data[:25] if len(r) > i and r[i].text.strip()]
        if not vals:
            shapes.append("empty")
        elif sum(bool(_NUMERIC.match(v)) for v in vals) > len(vals) * 0.6:
            shapes.append("num")
        elif sum(bool(_BIN.match(v)) for v in vals) > len(vals) * 0.6:
            shapes.append("bin")
        elif sum(bool(_CID_PAT.match(v)) for v in vals) > len(vals) * 0.6:
            shapes.append("cid")
        elif sum(v.count(",") >= 2 for v in vals) > len(vals) * 0.6:
            shapes.append("list")
        else:
            shapes.append("text")
    # Header words, normalised: lowercase, digits stripped, sorted per column.
    words = []
    for h in headers[: table.n_cols]:
        toks = sorted({w.lower() for w in re.findall(r"[A-Za-z]{2,}", h or "")})
        words.append("+".join(toks[:6]))
    return f"{table.n_cols}|{','.join(shapes)}|{','.join(words)}"


def layout_fingerprint(table: Table, headers: list[str]) -> str:
    """Stable id for a table SHAPE, deliberately independent of its content.

    Two tables share a fingerprint when a rule written for one should work on
    the other: same column count, same per-column value shape, same normalised
    header words. Compound ids, patent numbers and the actual measurements are
    excluded — otherwise every table is unique and nothing is ever reused.
    """
    return hashlib.sha256(
        layout_signature(table, headers).encode()).hexdigest()[:16]


# Character entities that encode TEXT — Greek letters, degrees, dashes, primes.
# Everything except the three that encode MARKUP.
_TEXT_ENTITY = re.compile(r"&(#\d+|#[xX][0-9a-fA-F]+|(?!lt;|gt;|amp;)[A-Za-z][A-Za-z0-9]*);")


def raw_block(xml: str, table_id: str) -> str:
    """The `<tables>` element for one block, with its text made readable.

    The STRUCTURE is kept verbatim — `colspec` widths, `namest`/`nameend`
    spans, the thead/tbody boundary. That is the whole reason to show the
    source at all: it states how a header tgroup maps onto a body tgroup of a
    different width, which is what US10376513 turned on, and it is the one view
    we have not transformed, so the only one that can contradict the others.

    The TEXT is unescaped, and that is new. Every patent here spells micromolar
    `&#x3bc;M`, so a model reading this block saw a unit written as an entity
    reference and we then measured its answer against the parsed text, where
    the same unit reads `μM`. Traced across 208 proposed units: `μM` appears 25
    times in US10172859 and 6 times in US9302989, and searching the un-unescaped
    source for it finds zero — which is how a correct reading gets recorded as
    an invention. It also explains a defect already noted in `synthesize`: a
    model that returned `(?P&lt;cid&gt;\\d+)` was echoing the escaping it had
    been shown.

    `&lt;`, `&gt;` and `&amp;` are DELIBERATELY left alone. In cell content
    `&lt;0.1` is a qualifier, and unescaping it to `<0.1` inside an XML view
    produces something that reads as a malformed tag — trading one ambiguity
    for a worse one. The prompt says what they mean instead.
    """
    if not (xml and table_id):
        return ""
    m = re.search(r"<tables\b[^>]*id=\"" + re.escape(table_id) + r"\".*?</tables>",
                  xml, re.S)
    if not m:
        return ""
    return _TEXT_ENTITY.sub(lambda e: html.unescape(e.group(0)), m.group(0))


def guessed_units(table: Table) -> list[dict]:
    """Assay columns whose unit was INFERRED, beside siblings that state one.

    A unit is either read off the column's own header or guessed from
    somewhere else — a legend, a caption, a previous tgroup. Guessing is
    usually right and usually the only option, so it is not by itself a defect.

    What IS a defect is disagreement WITHIN one table: some columns naming a
    unit and others not. That does not happen in a well-formed header; it
    happens when a spanning label lands on part of the group it covers.
    US11420968 heads four columns as two pairs under `(IC50, nM)`, our merge
    put it on the second of each pair, and the other two fell through to a
    caption about apoptosis biology that mentions uM. 111 values, 1000x low,
    every count healthy — no yield signal can see this, because a wrong unit
    produces records perfectly.

    So the flag is not "a unit was guessed". It is "this table cannot agree
    with itself about its own units", which is a header we mangled.
    """
    from ..sources.uspto_assays import ASSAY, _unit_from, build_columns

    assay = [c for c in build_columns(table) if c.kind == ASSAY and c.unit]
    if len(assay) < 2:
        return []
    stated = [c for c in assay if _unit_from(c.header or "")]
    guessed = [c for c in assay if not _unit_from(c.header or "")]
    if not stated or not guessed:
        return []
    said = {_unit_from(c.header or "") for c in stated}
    return [{"header": c.header, "index": c.index, "guessed": c.unit,
             "siblings_state": sorted(said),
             "agrees": c.unit in said} for c in guessed]


def unlabelled_assays(table: Table) -> list[dict]:
    """Assay columns the DOCUMENT never named — called assays by their shape.

    A column with no header whose cells are all grades or all `+`/`++` is
    almost certainly an assay, and reading it is usually right. What is never
    knowable from the cells is WHAT IT MEASURED, and that is the half a count
    cannot check: the records come out complete, carry a plausible value, and
    are labelled with a name the patent does not contain.

    The failures are not hypothetical and they do not look like failures.
    US9221791's `HPLC Method` column is single letters and produced 82 potency
    grades. US20240010684A1's peptide column is the twenty amino-acid letters
    and produced 40. Both were caught by the shape of their ALPHABET, not by
    any count — every row read, every record well-formed.

    So this is raised as a question rather than settled as a guess. The loop
    can see the table, its neighbours and the prose around it, and can answer
    what the column measures or say that it measures nothing. Asking costs one
    call; asserting costs a column of invented measurements that nothing
    downstream can distinguish from real ones.
    """
    from ..sources.uspto_assays import ASSAY, build_columns

    return [{"header": c.header, "index": c.index, "assay_name": c.assay_name}
            for c in build_columns(table)
            if c.kind == ASSAY and c.label_source == "shape"]


def duplicate_labels(table: Table) -> list[dict]:
    """Assay columns in ONE table that share an identical name.

    A table names its columns so a reader can tell them apart. Two that arrive
    with the same name are not two readings of one assay — they are two assays
    whose distinguishing header row was dropped, and the output cannot say
    which is which.

    US11547697 heads a four-column PI3K panel `PI3K α` / `PI3K β` / `PI3K γ` /
    `PI3K δ` over `IC50` over `(nM)`, and all four columns come out named
    `PI3K`. The values are right and correctly paired; what is gone is the
    isoform, which is the one thing that panel exists to measure. Nothing else
    detects it: every row is read, every record well-formed, every count clean,
    and a reader of the output cannot even see that a distinction was lost.

    Only where the DOCUMENT would distinguish them. A patent that genuinely
    prints the same assay twice is rare, and the loop can say so — which is why
    this is raised as a question rather than settled here. 27 blocks over 13
    patents corpus-wide, 1.1% of blocks, so it is a precise signal and not a
    dragnet.
    """
    from collections import Counter as _C

    from ..sources.uspto_assays import ASSAY, build_columns

    cols = [c for c in build_columns(table) if c.kind == ASSAY and c.assay_name]
    dup = {n for n, k in _C(c.assay_name for c in cols).items() if k > 1}
    return [{"header": c.header, "index": c.index, "assay_name": c.assay_name}
            for c in cols if c.assay_name in dup]


def dead_assay_columns(table: Table, records) -> list[dict]:
    """Assay columns holding data that yields nothing, beside siblings that do.

    The detector's unit has always been the ROW: how many compounds did this
    table give us. That makes a table's COLUMNS invisible. US11649247 Table 15
    heads two assays, `HT1080 (R132C, 2-hydroxyglutarate) IC50 (uM)` reading
    `0.00275 ± 0.00046, n = 3` and `HT1080 (R132C, aKG) IC50 (uM)` reading
    `>20.0`. `parse_value` could not read the first, so the potency was lost
    and every compound survived only as its own inactive ceiling — 20 uM
    recorded where the patent says 2.75 nM.

    Every gate passed. All 15 compounds were read, so row coverage was 100%;
    the read-fraction gate skipped the table; fidelity was clean; the suite was
    green; BindingDB checks values, and the values that survived were right.

    A sibling column that DOES yield is what makes this a defect rather than an
    observation: the same rows, the same ids, one column read and one not. That
    is a cell-format gap, and `value_pattern` is the rule kind for it — which is
    why this is raised as an ordinary gap rather than escalated.
    """
    from ..sources.uspto_assays import ASSAY, build_columns

    cols = build_columns(table)
    assay = [c for c in cols if c.kind == ASSAY]
    if len(assay) < 2:
        return []
    produced = Counter((r.column_header or r.assay_name or "") for r in records
                       if r.table_id == table.table_id)
    _, data = _header_rows_of(table)
    out = []
    for c in assay:
        if produced.get(c.header or c.assay_name or ""):
            continue
        cells = [r[c.index].text.strip() for r in data
                 if len(r) > c.index and r[c.index].text.strip()
                 and r[c.index].text.strip() not in _NULLISH]
        # An INVERTED table's data column is a list of compound ids, not
        # measurements: US11566007 writes one row per potency grade with
        # `A028, A075, A076, ...` beside it. `extract_inverted` reads those and
        # attributes the records to the block rather than to this column, so
        # they look dead from here — ten tables' worth of false alarm.
        if sum(1 for v in cells if v.count(",") >= 2) > len(cells) * 0.5:
            continue
        # A column of `+`/`++`/`A` is readable; what it lacks is a KEY, and
        # that is a different failure with its own detector. This one exists
        # for cell FORMATS we cannot parse, so grade columns are not its
        # business — US11566007's inverted blocks otherwise raise two more.
        if sum(1 for v in cells if _BIN.match(v)) > len(cells) * 0.5:
            continue
        if len(cells) >= 5:
            out.append({"header": c.header, "index": c.index,
                        "populated": len(cells), "examples": cells[:6]})
    # Only a defect if a SIBLING yields — otherwise the whole table is a gap
    # already, and the row-level checks own it.
    return out if len(out) < len(assay) else []


def _id_shape(text: str) -> str:
    """`48-1` and `52-4` → `99-9`. Digits and letters erased, punctuation kept."""
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", text.strip()))


def coherent_unread_ids(table: Table) -> list[str]:
    """Rows whose identifier we could not read, when they all look the SAME.

    The read-fraction gate below assumes the rows a mostly-read table leaves
    behind are blanks and malformed junk. That holds for scattered leftovers
    and fails completely for a systematic one: US20240335431 numbers some of
    its examples `48-1`, `48-2`, `52-4`, and `_CID_PAT` allows a trailing
    LETTER pair (`100AA`) but no trailing digit group. 24 of 80 rows dropped
    from one table, 32 reference compounds across two patents — and every
    table stayed above the 0.5 line, so the detector never asked about any of
    them.

    A remainder is noise; a category is a grammar we do not know. The
    discriminator is homogeneity, not size: three or more unreadable ids
    collapsing to one or two shapes is a second identifier convention, and it
    is worth a question no matter how much of the table already parsed.

    This is the RCA's outstanding recommendation — a check rejecting ~100% of
    a homogeneous set should suspect itself — applied to the detector.

    Scoped to the column already classified as the identifier, and only in a
    table that also has an assay column. The looser version of this — any short
    digit-bearing cell in the leading columns — fired on 50+ blocks corpus-wide:
    m/z values, `CDCl3`, `C(10)`, `0.05`, `(R132H)`, primer sequences. Every one
    of those would have been a paid question about a characterisation table.
    The signal is not "a cell we cannot read exists"; it is "the id column
    itself holds a second convention", and only the id column can say that.
    """
    from ..sources.uspto_assays import build_columns
    cols = build_columns(table)
    kinds = [c.kind for c in cols]
    if ASSAY not in kinds or CID not in kinds:
        return []
    idx = kinds.index(CID)
    _, data = _header_rows_of(table)
    unread, read = [], 0
    for r in data:
        if len(r) <= idx:
            continue
        txt = r[idx].text.strip()
        if not txt or txt in _NULLISH:
            continue
        if _CID_PAT.fullmatch(txt):
            read += 1
        elif len(txt) <= 14:
            unread.append(txt)
    # Needs a working majority to compare against: a column we read NOTHING of
    # is a different failure, already reported by the reason-derived gaps.
    if len(unread) < 3 or read < len(unread):
        return []
    shapes = Counter(_id_shape(u) for u in unread)
    dominant = sum(n for _, n in shapes.most_common(2))
    return unread if dominant >= len(unread) * 0.8 else []


def _sample_of(table: Table, headers: list[str], max_rows: int = 4,
               *, expand: bool = False) -> str:
    """A compact, readable rendering of the table's shape.

    Kept small on purpose: this is what a paid model reads. Header, a few data
    rows, and the legend if there is one — enough to infer a rule, not enough
    to be expensive.
    """
    _, data = _header_rows_of(table)
    lines = []
    legend = table_legend(table)
    if legend:
        lines.append(f"LEGEND: {legend[:900 if expand else 220]}")
    if table.caption:
        # Keep the clause that says what was measured, not the buffer recipe —
        # captions run for paragraphs and this is billed input.
        cap = table.caption
        best = max((c for c in re.split(r"[.;]", cap)),
                   key=lambda c: len(_ASSAY_WORDS.findall(c)), default=cap)
        cap = best.strip() if _ASSAY_WORDS.search(best) else cap[:160]
        lines.append(f"CAPTION: {cap[:220]}")
    # Columns are labelled by index because the rule the model returns refers
    # to them by index. Without this it cannot say "column 0 is the id" with
    # any confidence, and its first response to an unheadered table was to
    # escalate for exactly that reason.
    # Text just before the table. A bin key that defines the grades is usually
    # here rather than in the caption, and without it a model shown a column of
    # `A`/`B`/`D` has no way to turn them into measurements.
    prev = (getattr(table, "preceding", "") or "").strip()
    if prev:
        lines.append(f"TEXT BEFORE TABLE: ...{prev[-(1200 if expand else 320):]}")
    nxt = (getattr(table, "following", "") or "").strip()
    if nxt:
        lines.append(f"TEXT AFTER TABLE: {nxt[:1200 if expand else 320]}...")
    lines.append(f"COLUMNS: {table.n_cols} (indices 0..{table.n_cols - 1})")
    if any(headers):
        lines.append("HEADER: " + " | ".join(
            f"[{i}] {h[:36]}" if h else f"[{i}] (blank)" for i, h in enumerate(headers)))
    # ...and the header rows UNMERGED, when merging them was lossy.
    #
    # A multi-row header whose cells carry no CALS `namest` has no column
    # position in the source, so `_choose_offsets` infers one — and on
    # US10172859 it left-packs, landing "IC50 DNA-PK / IC50 pDNA-PK /
    # Ki [Kv1.11 hERG]" in columns 0-2 of a table whose grades are in 3-5. The
    # model then cannot bind a bin scale to a column and correctly escalates.
    #
    # The information is not missing from the patent; our merge destroys it.
    # So send the rows as they are and let the model align them against the
    # data rows below, which is a reading task it is good at and our offset
    # search is not. Only when the merge is suspect — fewer labelled columns
    # than the table is wide — because this is billed input.
    hdr_rows, _ = _header_rows_of(table)
    if hdr_rows and (expand or sum(1 for h in headers if h.strip()) < table.n_cols):
        for j, hr in enumerate(hdr_rows[:10 if expand else 4]):
            cells = [c.text.strip() for c in hr]
            if any(cells):
                lines.append(f"HEADER ROW {j} (raw, {len(cells)} cells, column "
                             "position NOT reliable): "
                             + " | ".join(c[:36] for c in cells))
        lines.append("The merged HEADER above may be misaligned: these rows "
                     "carry no column position in the source. Use the data "
                     "rows to decide which label belongs to which index.")
    shown = 0
    for row in data:
        cells = [c.text.strip() for c in row]
        if not any(cells):
            continue
        lines.append("ROW: " + " | ".join(
            f"[{i}] {c[:36]}" for i, c in enumerate(cells)))
        shown += 1
        if shown >= max_rows:
            break
    return "\n".join(lines)


def find_gaps(patent_id: str, tables: list[Table], extracted_by_table,
              *, min_rows: int = 5, max_read_fraction: float = 0.5,
              min_yield: float = 0.5, _source_xml: str = "") -> list[Gap]:
    """Tables that visibly hold data we did not extract.

    `extracted_by_table` maps table_id → records produced. A table with many
    data rows and few records is a gap; the *reason* is derived from what the
    classifier decided, so the eventual question to the model is specific
    ("no column was identified as the compound id") rather than "this failed".
    """
    # Record counts arrive keyed by `<tables>` id, but a block is routinely
    # split into several tgroups (a header tgroup plus continuations). Comparing
    # a block's whole record count against ONE tgroup's row count made every
    # multi-tgroup table look over-read: US10071079's main assay table scored
    # 2005/982 = 2.04 and was skipped as fully parsed, so the largest assay
    # table in the patent was invisible to the detector. Row counts are summed
    # per block so the two sides of the comparison agree.
    rows_per_block: dict[str, int] = {}
    for t in tables:
        _, d = _header_rows_of(t)
        rows_per_block[t.table_id] = rows_per_block.get(t.table_id, 0) + sum(
            1 for r in d if any(c.text.strip() for c in r))

    # The detector MUST classify a table the same way the extractor does, or it
    # judges a different table than the one that actually ran. Continuation
    # tgroups carry no header of their own and inherit one; without replicating
    # that here, every continuation looked header-less, no column classified as
    # an assay, and the value-level check was skipped entirely — which is why
    # US10071079's main assay table stayed invisible even after the alarm existed.
    inherit_by_width: dict[int, list[str]] = {}
    inherit_by_block: dict[str, list[str]] = {}

    # PRIMARY SIGNAL: measurement-shaped cells in versus USABLE records out.
    # Runs first and on its own terms — the per-guard checks below are the older
    # proxy heuristics, kept only for blocks this cannot judge.
    gaps: list[Gap] = []
    seen_blocks: set[str] = set()
    judged_blocks: set[str] = set()
    # Distinct compounds per block, which is what the read-fraction gate below
    # actually wants. `extracted_by_table` counts RECORDS, and a table with
    # three assay columns yields ~3 per row — so a block can score over 1.0
    # while most of its rows were never read. Measured on this corpus:
    # US11613531 TABLE-US-00001 scores 446/691 = 0.65 and is skipped as
    # mostly-parsed, while covering 219 of its 691 rows (0.32).
    cids_per_block: dict[str, set[str]] = {}
    homogeneous_blocks: set[str] = set()
    dead_col_blocks: set[str] = set()
    unit_blocks: set[str] = set()
    unlabelled_blocks: set[str] = set()
    dup_label_blocks: set[str] = set()
    records_list: list = []
    if not isinstance(extracted_by_table, dict):
        records = list(extracted_by_table)
        records_list = records
        for _r in records:
            if _r.cid:
                cids_per_block.setdefault(_r.table_id, set()).add(_r.cid)
        extracted_by_table = Counter(r.table_id for r in records)
        by_id = {t.table_id: t for t in tables}
        for tid, d in usable_yield(tables, records).items():
            if d["shaped_cells"] < 10:
                continue
            judged_blocks.add(tid)          # measured: the proxies must stay quiet
            # A PARTIAL READ IS STILL A MISS, and this is the gate that hid it.
            #
            # `min_yield` alone asks "did we get most of it", so a block read
            # at 0.7 raised nothing — and because `judged_blocks` above also
            # silences every proxy detector, it raised nothing from any other
            # check either. Reading most of a table is the easiest way for a
            # defect to stay invisible: the output looks populated.
            #
            # The floor two lines up already says what "enough cells to judge"
            # means. Use the same number for what it MISSED, rather than add a
            # second tuning knob: ten shaped cells are worth judging, so ten
            # shaped cells unread are worth raising. Measured over 137 patents:
            # 9 blocks and 854 unread cells sat above `min_yield` and silent,
            # 760 of them in four blocks.
            unread = max(0, d["shaped_cells"] - d["usable"])
            if d["yield"] >= min_yield and unread < 10:
                continue
            grades = {r.letter_grade for r in records
                      if r.table_id == tid and r.letter_grade}
            legends = hunt_legends(_source_xml or "", grades) if grades else []
            t = by_id.get(tid)
            if t is None:
                continue
            hr, _ = _header_rows_of(t)
            hdrs = merge_header(t, hr)
            what = d["dominant_missing"]
            fix = _MISSING_TO_REPAIR.get(what, "no records were produced for this block")
            gaps.append(Gap(
                patent_id=patent_id, table_id=tid, n_cols=t.n_cols,
                n_data_rows=rows_per_block.get(tid, 0), n_extracted=d["usable"],
                fingerprint=layout_fingerprint(t, hdrs),
signature=layout_signature(t, hdrs),
                reason=(f"{d['usable']} usable measurements from {d['shaped_cells']} "
                        f"measurement-shaped cells ({d['yield']:.0%}) — {fix}"),
                sample=_sample_of(t, hdrs), headers=hdrs,
                expanded_sample=_sample_of(t, hdrs, 24, expand=True),
                column_kinds=[c.kind for c in build_columns(t)],
                legend_candidates=legends,
            ))
            seen_blocks.add(tid)
    for t in tables:
        hdr_rows, data = _header_rows_of(t)
        rows = [r for r in data if any(c.text.strip() for c in r)]

        # Register this tgroup's header BEFORE any skip. A header-only tgroup
        # has zero data rows, so `min_rows` used to discard it first — and with
        # it the only copy of the block's real header. The continuation that
        # followed then inherited a stale header of the same width from an
        # unrelated block: US10071079's 982-row assay table picked up an LCMS
        # characterisation header, classified as [cid, structure, ms, rt, rt],
        # found no assay column, and the value-level check never ran. The
        # extractor registers first and skips second; the detector must agree
        # with it or it judges a table that never existed.
        own = merge_header(t, hdr_rows)
        if any(own):
            inherit_by_width[t.n_cols] = own
            inherit_by_block[t.table_id] = own

        if len(rows) < min_rows:
            continue
        # One gap per block. Blocks the yield metric already judged are done —
        # the proxy guards below exist only for blocks it cannot measure (no
        # assay column identified, so no denominator).
        if t.table_id in seen_blocks or t.table_id in judged_blocks:
            continue
        block_rows = rows_per_block.get(t.table_id, len(rows))
        headers = own
        if not any(headers):
            headers = (inherit_by_width.get(t.n_cols)
                       or inherit_by_block.get(t.table_id) or headers)
        cols = build_columns(
            t, inherited=inherit_by_width.get(t.n_cols) or inherit_by_block.get(t.table_id),
            data_rows=rows)
        kinds = [c.kind for c in cols]
        got = extracted_by_table.get(t.table_id, 0)

        # Does the table look like it holds measurements at all?
        payload = 0
        for r in rows[:40]:
            for c in r:
                s = c.text.strip()
                if _NUMERIC.match(s) or _BIN.match(s) or s.count(",") >= 2:
                    payload += 1
        if payload < len(rows[:40]):
            continue                       # genuinely not a data table

        # VALUE-LEVEL gap: the id column parses, an assay column is identified,
        # and yet the cells in it produce no value. This is the signal that was
        # missing — every bug found by hand so far was silent at row level:
        # `_VALUE_PAT` rejecting any number ≥1000, and rejecting `24 (*)` where
        # the parenthetical is a footnote rather than a replicate count. In both
        # cases coverage looked fine because the rows were simply never counted,
        # so no row-level gap was ever raised. Reading a cell and discarding it
        # is a different failure from failing to read the table, and it needs
        # its own alarm.
        cid_idx = next((c.index for c in cols if c.kind == CID), None)
        assay_idx = [c.index for c in cols if c.kind == ASSAY]
        if cid_idx is not None and assay_idx:
            from ..sources.uspto_assays import parse_value
            # Counted per CELL, not per row. Requiring every populated cell in a
            # row to fail made the alarm almost unreachable: in
            # `['345','5.1','37.6','1412']` the small values parse and only
            # `1412` is dropped, so the row still yields records and looks
            # healthy. Both silent parser bugs behaved exactly that way — they
            # ate a subset of cells across many rows — and a row-level count
            # missed them entirely when this was tested by reverting the fixes.
            unread_cells = 0
            total_cells = 0
            example: str | None = None
            unparsed: list[str] = []
            for r in rows[:80]:
                if len(r) <= cid_idx or not _CID_PAT.match(r[cid_idx].text.strip()):
                    continue
                for i in assay_idx:
                    if len(r) <= i:
                        continue
                    cell = r[i].text.strip()
                    if not cell or cell in _NULLISH:
                        continue
                    total_cells += 1
                    if not parse_value(cell):
                        unread_cells += 1
                        example = example or cell
                        if len(unparsed) < 8 and cell not in unparsed:
                            unparsed.append(cell)
            if total_cells and unread_cells >= max(3, total_cells * 0.05):
                gaps.append(Gap(
                    patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
                    n_data_rows=block_rows, n_extracted=got,
                    fingerprint=layout_fingerprint(t, headers),
signature=layout_signature(t, headers),
                    reason=(f"{unread_cells} of {total_cells} populated assay cells "
                            f"cannot be parsed as a value (e.g. {example!r})"),
                    sample=_sample_of(t, headers), headers=headers,
                    expanded_sample=_sample_of(t, headers, 24, expand=True),
                    column_kinds=kinds, unparsed_examples=unparsed,
                ))
                seen_blocks.add(t.table_id)
                continue

        # Compare against rows that COULD yield a value, not the raw row count.
        # A table whose assay cells all read `ND` / `—` is fully understood —
        # the compounds were not tested — and counting those as unread would
        # spend a repair call on data that does not exist.
        if cid_idx is not None and assay_idx:
            extractable = 0
            for r in rows:
                cells = [r[i].text.strip() for i in assay_idx if len(r) > i]
                if any(c and c not in _NULLISH for c in cells):
                    extractable += 1
            # scale the sampled tgroup's extractable ratio to the whole block
            if rows:
                extractable = int(extractable / len(rows) * block_rows)
        else:
            extractable = block_rows
        if got >= extractable:
            seen_blocks.add(t.table_id)
            continue   # everything extractable was read
        # Only ask about tables we are largely FAILING on. A table where the
        # parser already reads most rows is a bad question: measured on
        # US20240335431A1, every proposal for such a table came back yielding
        # fewer rows than the parser already managed, and was rejected by the
        # anti-deletion guard. Paying for a proposal that cannot win is waste,
        # and the leftover rows in a mostly-read table are usually blank or
        # malformed rather than a layout we failed to understand.
        # Rows read, not records produced — see `cids_per_block`. Falls back to
        # the record count when a caller passed a bare tally and cannot say.
        rows_read = len(cids_per_block.get(t.table_id, ())) or got
        if got and rows_read / max(block_rows, 1) > max_read_fraction:
            seen_blocks.add(t.table_id)
            continue

        # A block with a compound id but NO assay column is a characterisation
        # table (MW, m/z, retention time) unless its own prose says otherwise.
        # Flagging those made the detector shout about every healthy patent —
        # US9718825 is 100% extracted and was raising 7 gaps. Reuse the caption
        # gate the extractor already trusts rather than inventing a new test.
        if ASSAY not in kinds:
            from ..sources.uspto_assays import caption_assay_hint
            if not caption_assay_hint(f"{t.caption} {table_legend(t)}")[0]:
                continue

        if CID not in kinds and ASSAY not in kinds:
            reason = "no compound-id column and no assay column identified"
        elif CID not in kinds:
            reason = "assay columns found but no compound-id column identified"
        elif ASSAY not in kinds:
            reason = "compound-id column found but no column identified as an assay"
        else:
            reason = f"columns classified but only {got} of {block_rows} rows produced records"

        seen_blocks.add(t.table_id)
        gaps.append(Gap(
            patent_id=patent_id,
            table_id=t.table_id,
            n_cols=t.n_cols,
            n_data_rows=block_rows,
            n_extracted=got,
            fingerprint=layout_fingerprint(t, headers),
signature=layout_signature(t, headers),
            reason=reason,
            sample=_sample_of(t, headers),
            expanded_sample=_sample_of(t, headers, 24, expand=True),
            headers=headers,
            column_kinds=kinds,
        ))
    # Units this table could not agree with itself about. Runs on its own, like
    # the two passes below, because a wrong unit costs no rows and every
    # yield-shaped check therefore scores the table perfectly.
    for t in tables:
        if t.table_id in unit_blocks:
            continue
        # Only where the guess CONTRADICTS what the table states elsewhere.
        # A guessed unit matching its siblings is a guess we have no evidence
        # against, and paying a model to look at it is the noise the
        # read-fraction gate exists to avoid. Corpus-wide that is the
        # difference between 4 gaps and 0 — and it would still have caught
        # US11420968, whose columns said uM and nM at the same time.
        odd = [g for g in guessed_units(t) if not g["agrees"]]
        if not odd:
            continue
        unit_blocks.add(t.table_id)
        heads = merge_header(t)
        worst = odd[0]
        gaps.append(Gap(
            patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
            n_data_rows=rows_per_block.get(t.table_id, len(t.body_rows)),
            n_extracted=len(cids_per_block.get(t.table_id, ())),
            fingerprint=layout_fingerprint(t, heads),
signature=layout_signature(t, heads),
            reason=(f"{len(odd)} assay column(s) have a GUESSED unit while "
                    f"sibling columns in the same table state one: "
                    f"{worst['header'][:40]!r} was given {worst['guessed']!r} "
                    f"from outside its header, and its siblings say "
                    f"{'/'.join(worst['siblings_state'])}"
                    + ("" if worst["agrees"] else " — which disagrees. A header "
                       "label spanning several columns has probably landed on "
                       "only one of them")),
            sample=_sample_of(t, heads), headers=heads,
            expanded_sample=_sample_of(t, heads, 24, expand=True),
            column_kinds=[c.kind for c in build_columns(t)],
        ))

    # Blocks whose RECORDS carry a placeholder name. The pass below asks the
    # same question of COLUMNS, and columns are not the only way a name is
    # minted: the inverted-table path names a whole block at once, so a block
    # of 818 records called `assay (binned)` raised nothing at all while a
    # single unnamed column did. The signal is the name, not the code path.
    for t in tables:
        if t.table_id in unlabelled_blocks:
            continue
        mine = [r for r in records_list if r.table_id == t.table_id]
        if not mine or not all(is_placeholder_name(r.assay_name) for r in mine):
            continue
        unlabelled_blocks.add(t.table_id)
        heads = merge_header(t)
        gaps.append(Gap(
            patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
            n_data_rows=rows_per_block.get(t.table_id, len(t.body_rows)),
            n_extracted=len(cids_per_block.get(t.table_id, ())),
            fingerprint=layout_fingerprint(t, heads),
            signature=layout_signature(t, heads),
            asks="column_names",
            reason=(f"every record from this table carries a placeholder name "
                    f"({mine[0].assay_name!r}) — we read {len(mine)} "
                    f"measurements and cannot say what was measured. Name the "
                    f"assay from the table's own rows or the prose around it, "
                    f"or say that it is not a measurement"),
            sample=_sample_of(t, heads), headers=heads,
            expanded_sample=_sample_of(t, heads, 24, expand=True),
            column_kinds=[c.kind for c in build_columns(t)],
        ))

    # Assay columns the document never named. Its own pass, for the reason the
    # two around it have one: this costs no rows either. The column reads
    # perfectly and every count scores it perfectly; what is missing is the
    # name of what was measured, which only the surrounding document can give.
    for t in tables:
        if t.table_id in unlabelled_blocks:
            continue
        bare = unlabelled_assays(t)
        if not bare:
            continue
        unlabelled_blocks.add(t.table_id)
        heads = merge_header(t)
        gaps.append(Gap(
            patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
            n_data_rows=rows_per_block.get(t.table_id, len(t.body_rows)),
            n_extracted=len(cids_per_block.get(t.table_id, ())),
            fingerprint=layout_fingerprint(t, heads),
            signature=layout_signature(t, heads),
            reason=(f"{len(bare)} column(s) were read as assays because their "
                    f"CELLS look like assay values, and the document never "
                    f"names them: column(s) "
                    f"{', '.join(str(b['index']) for b in bare)} carry no "
                    f"header. Say what each one measures, or say that it is "
                    f"not a measurement — a synthetic method, an HPLC method, "
                    f"a substituent or a sequence all look like this and none "
                    f"of them is an assay"),
            sample=_sample_of(t, heads), headers=heads,
            expanded_sample=_sample_of(t, heads, 24, expand=True),
            column_kinds=[c.kind for c in build_columns(t)],
        ))

    # Columns a reader cannot tell apart. Its own pass, for the reason its
    # neighbours have one: this costs no rows either.
    for t in tables:
        if t.table_id in dup_label_blocks:
            continue
        dups = duplicate_labels(t)
        if not dups:
            continue
        dup_label_blocks.add(t.table_id)
        heads = merge_header(t)
        names = sorted({d["assay_name"] for d in dups})
        gaps.append(Gap(
            patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
            n_data_rows=rows_per_block.get(t.table_id, len(t.body_rows)),
            n_extracted=len(cids_per_block.get(t.table_id, ())),
            fingerprint=layout_fingerprint(t, heads),
            signature=layout_signature(t, heads),
            asks="column_names",
            reason=(f"{len(dups)} assay columns share {len(names)} name(s) — "
                    f"{', '.join(repr(n[:40]) for n in names)} — so the output "
                    f"cannot say which column is which. Give each column the "
                    f"name the table's own header rows give it; a panel usually "
                    f"states the shared metric on one row and what differs "
                    f"(an isoform, a mutant, a cell line) on another"),
            sample=_sample_of(t, heads), headers=heads,
            expanded_sample=_sample_of(t, heads, 24, expand=True),
            column_kinds=[c.kind for c in build_columns(t)],
        ))

    # Assay columns that yield nothing while a sibling in the same table does.
    # Its own pass for the same reason as the one below: every other check
    # scores a table by the rows it produced, and a dead column costs no rows.
    for t in tables:
        if t.table_id in dead_col_blocks:
            continue
        dead = dead_assay_columns(t, records_list)
        if not dead:
            continue
        dead_col_blocks.add(t.table_id)
        heads = merge_header(t)
        worst = max(dead, key=lambda d: d["populated"])
        n_live = len([c for c in build_columns(t) if c.kind == ASSAY]) - len(dead)
        gaps.append(Gap(
            patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
            n_data_rows=rows_per_block.get(t.table_id, len(t.body_rows)),
            n_extracted=len(cids_per_block.get(t.table_id, ())),
            fingerprint=layout_fingerprint(t, heads),
signature=layout_signature(t, heads),
            reason=(f"assay column {worst['header'][:44]!r} holds "
                    f"{worst['populated']} populated cells and produced NO "
                    f"records, while {n_live} sibling assay column(s) in the "
                    f"same table did. Same rows, "
                    f"same ids, one cell format we cannot read — e.g. "
                    f"{', '.join(repr(e) for e in worst['examples'][:3])}"),
            sample=_sample_of(t, heads), headers=heads,
            expanded_sample=_sample_of(t, heads, 24, expand=True),
            unparsed_examples=worst["examples"][:8],
        ))

    # Gated by NOTHING above — deliberately its own pass, and it does not
    # consult `seen_blocks`.
    #
    # Every other check scores a block against what the parser managed to find
    # in it, so a block can pass all of them while dropping a whole class of
    # rows. US20240335431 TABLE-US-00001 scores `yield: 1.0` — 55 shaped cells,
    # 55 records, 55 usable — and 24 further rows are simply absent from the
    # denominator, because the rows whose id we could not read never became
    # cells to count. Perfect marks on the half of the table we can see.
    #
    # A block may therefore raise this AND another gap: "I cannot read these
    # ids" and "these values have no unit" are different failures and both are
    # worth asking about.
    for t in tables:
        if t.table_id in homogeneous_blocks:
            continue
        odd = coherent_unread_ids(t)
        if not odd:
            continue
        homogeneous_blocks.add(t.table_id)
        heads = merge_header(t)
        shapes = sorted({_id_shape(o) for o in odd})
        gaps.append(Gap(
            patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
            n_data_rows=rows_per_block.get(t.table_id, len(t.body_rows)),
            n_extracted=len(cids_per_block.get(t.table_id, ())),
            fingerprint=layout_fingerprint(t, heads),
signature=layout_signature(t, heads),
            reason=(f"{len(odd)} rows carry an identifier we cannot read, and they "
                    f"are all the same shape ({'/'.join(shapes[:2])}) — e.g. "
                    f"{', '.join(odd[:4])}. The rest of the table parsed cleanly, "
                    f"so this is a second identifier convention rather than "
                    f"damaged rows"),
            sample=_sample_of(t, heads), headers=heads,
            expanded_sample=_sample_of(t, heads, 24, expand=True),
            unparsed_examples=odd[:8],
        ))

    # Attached HERE, once, and never in a `Gap(...)` call. It was a keyword on
    # one of three construction sites and absent from the other two, so the
    # model's `request_more_context("raw_source")` returned nothing for two
    # thirds of gaps — a capability that reads as present everywhere except
    # where it runs. A fourth site would have inherited the same bug; a loop
    # over the finished list cannot.
    for g in gaps:
        g.raw_source = raw_block(_source_xml or "", g.table_id)
    gaps.sort(key=lambda g: -g.severity)
    # A SPECIALISED READER REFUSED, AND A LAXER ONE EMITTED IN ITS PLACE.
    #
    # The same category as the two passes above: this costs no ROWS, so every
    # count scores it perfectly, and the records are attached to the wrong
    # compound. A table laid out `Compound | Ki | Compound | Ki | Compound | Ki`
    # is read by `_column_groups`, which refuses whenever it cannot prove the
    # grouping — correctly, since its docstring says a false positive is worse
    # than the under-read it replaces. Control then falls to the ordinary path,
    # which assumes ONE compound column and hangs every assay column off it.
    #
    # US11708332 TABLE-US-00016 shipped `28f`'s Ki under `161e` that way. Only
    # the rows whose borrowed value happened to be IDENTICAL were ever visible,
    # as 13 duplicate records; the rest could not be seen from the output at
    # all. Measured over the corpus: 131 tables carry two or more compound-id
    # columns, the grouped reader handles 16 and refuses 115, and on 10 of
    # those the fallback emitted anyway.
    #
    # RAISED EVEN THOUGH NO REGEX REPAIRS IT. The loop may well answer that it
    # cannot buy a rule for this and escalate — which is the right outcome and
    # is recorded, where returning None is not. A refusal the loop never hears
    # about is how the next one of these gets made.
    from ..sources.uspto_assays import _column_groups, _split_rows
    for t in tables:
        if t.table_id in unlabelled_blocks:
            continue
        mine = [r for r in records_list if r.table_id == t.table_id]
        if not mine:
            continue
        try:
            cols = build_columns(t)
            id_cols = [c for c in cols if c.kind == CID]
            if len(id_cols) < 2 or _column_groups(cols, _split_rows(t)[1]):
                continue
        except Exception:
            continue
        heads = merge_header(t)
        gaps.append(Gap(
            patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
            n_data_rows=rows_per_block.get(t.table_id, len(t.body_rows)),
            n_extracted=len(cids_per_block.get(t.table_id, ())),
            fingerprint=layout_fingerprint(t, heads),
            signature=layout_signature(t, heads),
            reason=(f"this table has {len(id_cols)} compound-id columns and the "
                    f"reader built for that layout refused it, so its "
                    f"{len(mine)} record(s) came from the path that assumes "
                    f"ONE compound column — every assay column was attached to "
                    f"the first compound, which is wrong for all but the first "
                    f"group. Say which assay columns belong to which compound "
                    f"column, or say that this is not a repeating layout"),
            sample=_sample_of(t, heads), headers=heads,
            expanded_sample=_sample_of(t, heads, 24, expand=True),
            column_kinds=[c.kind for c in cols],
        ))


    return gaps


def classify_like_extractor(tables: list[Table]) -> dict[int, list]:
    """Classify every tgroup exactly as `extract_from_tables` does.

    Detector/extractor divergence has now caused three separate bugs: a
    header-only tgroup skipped before its header was registered, a detector
    that never inherited headers at all, and the same omission reintroduced in
    usable_yield. Each time the detector judged a table the extractor never
    saw. There is one implementation of the walk now, and both sides call it.
    """
    by_width: dict[int, list[str]] = {}
    by_block: dict[str, list[str]] = {}
    out: dict[int, list] = {}
    for idx, t in enumerate(tables):
        hr, data = _header_rows_of(t)
        own = merge_header(t, hr)
        if any(own):                       # register BEFORE any skip
            by_width[t.n_cols] = own
            by_block[t.table_id] = own
        rows = [r for r in data if any(c.text.strip() for c in r)]
        out[idx] = build_columns(
            t, inherited=by_width.get(t.n_cols) or by_block.get(t.table_id),
            data_rows=rows)
    return out


def usable_yield(tables: list[Table], records) -> dict[str, dict]:
    """Per `<tables>` block: measurement-shaped cells in, usable records out.

    ONE metric replaces the proxies. Every detector bug this session came from
    measuring something correlated with success rather than success itself —
    records produced, rows read, coverage ratio — and each proxy was blind to a
    failure the others caught:

        rows-vs-cells      a bug eating SOME cells per row left rows healthy
        block-vs-tgroup    counts keyed by block, rows counted per tgroup
        records-vs-values  1,351 grade-only records looked fully extracted

    Counting measurement-shaped CELLS against USABLE records is immune to all
    three, and to failure modes not yet seen: anything that stops a cell
    becoming an actionable measurement shows up as a shortfall, whatever caused
    it. The dominant missing field then names the repair.
    """
    from collections import Counter
    shaped: dict[str, int] = {}
    classified = classify_like_extractor(tables)
    for idx, t in enumerate(tables):
        _, data = _header_rows_of(t)
        rows = [r for r in data if any(c.text.strip() for c in r)]
        cols = classified[idx]
        cid_i = next((c.index for c in cols if c.kind == CID), None)
        assay_i = [c.index for c in cols if c.kind == ASSAY]
        # Only cells in ASSAY columns count. Counting every numeric cell beside a
        # cid made characterisation blocks (MW, m/z, retention time) look like
        # 840 unextracted measurements on US9718825, whose assay data is in fact
        # 100% extracted. The denominator has to be the same population the
        # numerator is drawn from, or the ratio is meaningless.
        if cid_i is None or not assay_i:
            continue
        n = 0
        for r in rows:
            if len(r) <= cid_i or not _CID_PAT.match(r[cid_i].text.strip()):
                continue
            for i in assay_i:
                if len(r) <= i:
                    continue
                v = r[i].text.strip()
                if v and v not in _NULLISH and (_NUMERIC.match(v) or _BIN.match(v)):
                    n += 1
        shaped[t.table_id] = shaped.get(t.table_id, 0) + n

    out: dict[str, dict] = {}
    by_block: dict[str, list] = {}
    for rec in records:
        by_block.setdefault(rec.table_id, []).append(rec)
    for tid, n_shaped in shaped.items():
        recs = by_block.get(tid, [])
        usable = sum(1 for r in recs if r.is_usable)
        missing = Counter(f for r in recs for f in r.missing_fields())
        out[tid] = {
            "shaped_cells": n_shaped,
            "records": len(recs),
            "usable": usable,
            "yield": usable / n_shaped if n_shaped else 0.0,
            "dominant_missing": missing.most_common(1)[0][0] if missing else None,
        }
    return out


# What each missing field means for repair — the diagnosis maps straight to a
# rule kind, so the detector says what to ask for rather than just "this failed".
_MISSING_TO_REPAIR = {
    "value": ("the cells hold grades or a format our parser cannot read; needs a "
              "bin key or a value_pattern"),
    "unit": ("values parse but no unit was found in header, legend or "
             "description; needs unit resolution"),
    "assay_name": "columns produce values but none could be named; needs a column_map",
    "cid": "no column could be read as a compound identifier; needs a column_map",
}


# Symbols a patent grades compounds with, and every comparison glyph seen in the
# wild. U+2266/U+2267 (LESS-THAN OVER EQUAL TO) are CJK-compatibility variants
# that appear as often as the U+2264/U+2265 forms in this corpus, and matching
# only the latter silently loses half of them.
_CMP = r"[<>≤≥≦≧]|&lt;|&gt;"
_LEGEND_SHAPE = re.compile(
    rf"(?:\b[A-Z]\b|\+{{1,5}})\s*(?:{_CMP})\s*\d+(?:\.\d+)?\s*"
    rf"(?:nM|[μuµ]M|mM|pM|nanomolar|micromolar)", re.I)


# Two bin definitions this far apart in the flattened text still belong to one
# legend. Sized from US10172859, whose two scales are separated by their second
# scale's name plus punctuation.
_LEGEND_JOIN = 120
_LEGEND_LOOKBACK = 260
# Where a scale starts: its assay label, or the first symbol of the run.
_LEGEND_ANCHOR = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9 .()/\[\]-]{0,40}?\s*:\s*)?(?:\b[A-E]\b|\+{1,5})\s*(?::|[<>≤≥≦≧])")
# Analytical-method prose that habitually abuts a legend and is never part of it.
_LEGEND_STOP = re.compile(
    r"\b(?:Analysis|NMR|Instruments?|Bruker|Reference\s*:\s*TMS|LC[-/]?MS|HPLC)\b", re.I)


def hunt_legends(xml: str, symbols: set[str], *, max_hits: int = 4) -> list[str]:
    """Search the WHOLE document for text that defines these grade symbols.

    Position-based capture cannot work. The key for US9656988 is a trailing
    footnote ROW INSIDE its own table — not a caption, not a paragraph before
    the block, not one after it. Three separate windowing attempts missed it,
    each for a different structural reason.

    So stop guessing where it lives and search for what it looks like: a symbol
    next to a comparator next to a number with a concentration unit. That is
    format-agnostic — it finds `A ≦ 10 nM; 10 nM < B ≦ 100 nM`, `++++: IC50 ≥ 1
    uM` and `is marked "+++"` alike — and it makes the loop able to LEARN a
    legend grammar we have never implemented, instead of needing a new parser
    per format.

    Only spans mentioning a symbol the table actually uses are returned, so an
    unrelated key elsewhere in the patent is not attached to the wrong table.
    """
    import html as _html
    flat = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", xml)))

    # A legend is a RUN of these shapes, not one of them, and its length is set
    # by the patent — US10172859 defines two scales over ~350 characters. A
    # fixed window around a single match cut that run open: the span began
    # midway through the second bin of the first scale, so neither that scale's
    # name nor its opening bound was ever sent. The model then reported two
    # scales and escalated, which was the only correct answer available to it.
    # Cluster adjacent matches and take the whole run instead.
    hits = [(m.start(), m.end()) for m in _LEGEND_SHAPE.finditer(flat)]
    if not hits:
        return []
    clusters: list[list[int]] = [[hits[0][0], hits[0][1]]]
    for s, e in hits[1:]:
        if s - clusters[-1][1] <= _LEGEND_JOIN:
            clusters[-1][1] = e
        else:
            clusters.append([s, e])

    out: list[str] = []
    for start, end in clusters:
        # Reach back for the scale's own label — "Kv11.1 hERG:", "IC 50 :" —
        # which sits before the first bin and is what tells the model WHICH
        # column a scale governs. Then trim forward to a bin or label boundary
        # so the span does not open mid-number.
        lead = flat[max(0, start - _LEGEND_LOOKBACK):start]
        anchor = _LEGEND_ANCHOR.search(lead)
        span = (lead[anchor.start():] if anchor else lead[-90:]) + flat[start:end + 80]
        # Drop trailing analytical boilerplate: NMR/MS method prose routinely
        # abuts a legend and crowded out ~150 chars of the send window.
        cut = _LEGEND_STOP.search(span)
        if cut and cut.start() > 40:
            span = span[:cut.start()]
        span = span.strip()
        if symbols and not any(
                re.search(rf"(?:\b{re.escape(s)}\b|{re.escape(s)}\s*(?:{_CMP}))", span)
                for s in symbols):
            continue
        if any(span[:60] in o for o in out):
            continue
        out.append(span)
        if len(out) >= max_hits:
            break
    return out


def yield_contradictions(tables: list[Table], records) -> list[dict]:
    """Blocks whose OWN TWIN extracts fine — same fingerprint, opposite outcome.

    The loop has exactly one hypothesis for every failure: this layout is
    unusual, so buy a rule for it. It cannot form the other hypothesis — that
    our handling is inconsistent — because it has no notion of internal
    agreement. So a table yielding nothing always looks like a hard patent, even
    when an identically-shaped table two pages earlier yielded hundreds.

    US11566007 is the case. `TABLE-US-00006` and `TABLE-US-00007` share
    fingerprint `1fed7af816615b20`; the first produces 761 records and the
    second produces zero. The fingerprint is our claim that a rule written for
    one works on the other, and the yields say that claim is false. Nothing in
    the loop could represent that contradiction, so eleven blocks were escalated
    as unreadable layouts while the working counterpart sat in the same document.

    Two things follow, and the second is the useful one:

      1. It is a CODE inconsistency, not a layout gap — no rule can fix it, and
         asking for one spends money to encode the asymmetry.
      2. **The answer already exists in our own output.** A configuration that
         works on this exact shape is running a few tables away. The repair is a
         transfer, not a question, and it costs nothing.

    Measured over the 63-patent cache: 306 distinct fingerprints, 105 seen more
    than once, and exactly ONE contradiction — so this is a precise signal, not
    a noisy heuristic.
    """
    from collections import Counter, defaultdict

    got = Counter(r.table_id for r in records)
    groups: dict[str, list[tuple]] = defaultdict(list)
    for t in tables:
        hr, data = _header_rows_of(t)
        n_rows = sum(1 for r in data if any(c.text.strip() for c in r))
        if n_rows < 5:
            continue
        fp = layout_fingerprint(t, merge_header(t, hr))
        groups[fp].append((t.table_id, n_rows, got.get(t.table_id, 0)))

    out: list[dict] = []
    for fp, members in groups.items():
        if len(members) < 2:
            continue
        live = [m for m in members if m[2] > 0]
        dead = [m for m in members if m[2] == 0]
        if not live or not dead:
            continue
        best = max(live, key=lambda m: m[2])
        for tid, n_rows, _ in dead:
            out.append({
                "table_id": tid, "fingerprint": fp,
                "twin": best[0], "twin_records": best[2], "twin_rows": best[1],
                "rows": n_rows,
                "detail": (f"{tid} yields 0 records from {n_rows} rows, but "
                           f"{best[0]} has the SAME layout fingerprint {fp} and "
                           f"yields {best[2]} from {best[1]} rows. A rule cannot "
                           f"fix this — the layouts are identical by our own "
                           f"measure, so the difference is in our code path. The "
                           f"working configuration is already in this document."),
            })
    return out
