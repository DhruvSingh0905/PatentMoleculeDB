"""Assay extraction from USPTO CALS tables — deterministic, no LLM.

The lesson that shapes this module: **most numbers in a patent table are not
assay values.** A naive "grab every number next to a compound id" sweep over
US10544143 produced 6,063 pairs, and almost all of them were molecular weights,
LC-MS m/z and retention times. US11292791's third column is
`1H NMR (CD3OD, 400 MHz) δ` — a numeric goldmine of pure noise. The existing
pipeline needs an LLM and a corroboration validator largely to undo damage of
exactly this kind.

With real table markup that damage is avoidable at the source, because the
patent already tells us what each column is. So the pipeline here is:

    merge multi-row headers  →  classify each column  →  only read assay columns

Everything downstream depends on step two. A column we cannot confidently
classify is skipped, not guessed: a missing assay is recoverable, a molecular
weight recorded as an IC50 is a lie the database cannot detect later.

Layout facts this handles, all observed in real grants:
  - headers stacked across up to 7 `thead` rows, merged column-wise
    (`Compound`/`No.` → "Compound No.")
  - header cells spanning columns via `namest`/`nameend`
  - continuation tables with no header of their own, inheriting the previous
    tgroup's header (US9718825, US10544143)
  - compound ids as `12`, `I-2300`, `Z1`, `A1`
  - letter-grade potency bins (`A`-`E`) instead of numbers (US11254686)
  - blank spacer rows interleaved between data rows
  - run counts as `(8)` in the cell or in a following column
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

from ..core import config
from .uspto_xml import Table

logger = logging.getLogger(__name__)

# ── column kinds ──────────────────────────────────────────────────
CID = "cid"
ASSAY = "assay"
NRUNS = "nruns"
NMR = "nmr"
MS = "ms"
MW = "mw"
RT = "rt"
STRUCTURE = "structure"
SUBSTITUENT = "substituent"
UNKNOWN = "unknown"

# Columns we must never read a value from. Kept explicit rather than implied
# by "not an assay", so that adding a new kind fails closed.
NON_ASSAY = {NMR, MS, MW, RT, STRUCTURE, SUBSTITUENT, CID, NRUNS, UNKNOWN}

_UNIT_PAT = re.compile(
    r"\(\s*(n[mM]|[μuµ][mM]|m[mM]|p[mM]|nmol|µg/mL|ug/mL|%|percent)\s*\)|"
    r"\b(nM|µM|μM|uM|mM|pM)\b|"
    # Spelled-out units. Patents routinely state the unit in a table legend as
    # a word — "IC50's are micromolar." — rather than as a symbol in the
    # header. Missing these left correctly-read values unitless, which is the
    # difference between a usable measurement and an unusable number.
    r"\b(micromolar|nanomolar|millimolar|picomolar)\b", re.I)

_SPELLED_UNIT = {
    "micromolar": "uM", "nanomolar": "nM",
    "millimolar": "mM", "picomolar": "pM",
}

# A cell that is a measurement: optional qualifier, a number, optional unit,
# optional parenthesised run count.
# NOTE the integer part is `\d+`, not `\d{1,3}`. Written as `\d{1,3}(?:,\d{3})*`
# to allow thousands separators, it silently rejected every un-separated value
# of 1000 or more — `1511.5`, `8618`, `1412` all failed to parse and were
# dropped without a trace. Patents report nM potencies above 1000 constantly,
# so this quietly discarded the entire weak-activity tail of the corpus.
# The trailing parenthetical is captured loosely as `paren` and interpreted in
# `parse_value`, because it is not always a run count. Patents also put a
# footnote marker there — `24 (*)`, `0.83 (A)` — and requiring digits made those
# cells unparseable, so whole rows were dropped with their values sitting in
# plain sight.
_VALUE_PAT = re.compile(
    r"^\s*(?P<qual>[<>~≈≥≤]|>=|<=)?\s*"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
    r"\s*(?P<unit>nM|µM|μM|uM|mM|pM|%)?"
    r"(?:\s*\(\s*(?P<paren>[^)]{1,12}?)\s*\))?\s*$")

_NRUNS_ONLY = re.compile(r"^\s*\(\s*(\d{1,3})\s*\)\s*$")
_LETTER_BIN = re.compile(r"^\s*([A-E])\s*$")

# A compound id may be written bare (`12`, `I-2300`, `Z1`, `A1`, `5a`) or
# spelled out with a label (`Example 1`, `Cpd. No. 5`, `Ex. 7`). Rejecting the
# labelled forms cost an entire 1,108-row assay table on US10245267: both value
# columns classified correctly as assays, but the id column read as `unknown`,
# so the table produced nothing at all.
_CID_LABEL = re.compile(
    r"^\s*(?:examples?|ex\.?|compounds?|cpds?\.?(?:\s*nos?\.?)?|entry|nos?\.?|#)\s*[:.]?\s*",
    re.I)
# The trailing suffix covers separated stereoisomers, which patents label
# `488-A` / `488-B` (or `12a` / `12b`). Without the hyphenated form those rows
# fail the id test and the whole compound is dropped.
#
# TWO letters, not one. US11312727 separates four stereoisomers per example and
# labels them `100AA`, `100AB`, `100BA`, `100BB` — 135 of its 382 compounds —
# alongside 227 single-letter ids in the same document. Allowing one letter
# matched none of the four and cost every one of them: measured against
# BindingDB, which uses the patent's own labels, it was 131 of the 171 compounds
# we were missing across the whole reference corpus.
_CID_CORE = r"(?:[A-Za-z]{1,3}[-–]?)?\d{1,5}(?:[-–]?[a-zA-Z]{1,2})?"
_CID_PAT = re.compile(rf"^\s*(?:{_CID_LABEL.pattern[2:]})?{_CID_CORE}\s*$", re.I)


def normalize_cid(text: str) -> str:
    """`Example 007` / `Cpd. No. 7` / `7` → `7`.

    One canonical form, so a value found in a table headed `Example N` lands on
    the same compound as one found in a table headed `Cpd. No. N`.
    """
    s = (text or "").strip()
    s = _CID_LABEL.sub("", s).strip()
    # Preserve a prefix letter (A1, I-2300); only strip padding zeros.
    m = re.match(r"^([A-Za-z]{1,3}[-–]?)?0*(\d+)([-–]?[a-zA-Z])?$", s)
    if m:
        return f"{m.group(1) or ''}{m.group(2)}{m.group(3) or ''}"
    return s

_HEADER_CID = re.compile(
    r"\b(compound|cpd|example|ex#|entry|structure\s*no|no\.?|number|id)\b", re.I)
_HEADER_NMR = re.compile(r"\bnmr\b|δ\s*\(?ppm|chemical\s+shift", re.I)
_HEADER_MS = re.compile(
    r"\bms\b|\bm/z\b|\[m\s*[+±]|\bm\s*\+\s*h\b|lc[- ]?ms|hrms|mass\s+spec|"
    r"\bfound\b|\bobs(?:erved)?\b|esi", re.I)
_HEADER_MW = re.compile(r"\bmw\b|molecular\s+weight|\bcalc(?:d|ulated)?\b|exact\s+mass", re.I)
_HEADER_RT = re.compile(r"\brt\b|retention\s+time|\bt_?r\b|\bmethod\b|\bpurity\b", re.I)
_HEADER_STRUCT = re.compile(r"\bstructure\b|\bstruct\.?\b", re.I)
_HEADER_SUBST = re.compile(r"^\s*R\s*\d*\s*$|^\s*(Ar|X|Y|Z)\s*\d*\s*$", re.I)
_HEADER_NRUNS = re.compile(r"^\s*\(?\s*n\s*\)?\s*$|\bn\s*=|\bruns?\b|\breps?\b", re.I)

# Names/metrics that mark a real bioassay column.
_HEADER_ASSAY = re.compile(
    r"\b(ic\s*50|ec\s*50|ed\s*50|gi\s*50|cc\s*50|ki\b|kd\b|kb\b|"
    r"pic50|pec50|pki|pkd|"
    r"%\s*inh|percent\s+inh|inhibition|activity|potency|binding|affinity|"
    r"clint|clearance|t1/2|half[- ]life|papp|permeab|solubilit|"
    r"emax|hill|selectivity|ratio|"
    r"cyp|herg|ppb|auc|cmax)\b", re.I)


# Words that mark surrounding prose as describing a bioassay. Used only to
# decide whether an unlabelled table is assay data at all — never to name a
# specific column.
_CAPTION_ASSAY = re.compile(
    r"\b(assay|inhibit(?:ion|ory)?|activit(?:y|ies)|potenc(?:y|ies)|"
    r"binding|affinit(?:y|ies)|ic\s*50|ec\s*50|ki\b|kd\b|"
    r"antagonis|agonis|efficac|selectivit|cytotox|antiprolifer)\b", re.I)

# Semi-quantitative potency notation (`+`, `++`, `+++`). Same idea as letter
# bins: record the grade, never invent a number for it.
_PLUS_BIN = re.compile(r"^\s*(\++)\s*$")


def caption_assay_hint(caption: str) -> tuple[str | None, str | None]:
    """(assay name, unit) inferred from a table's caption, or (None, None).

    Patents frequently publish a table with no header at all, naming the assay
    only in the sentence before it ("Additional in vitro Raf inhibition data is
    provided in the following Table"). Such tables classify as entirely unknown
    and get dropped — the single largest source of missed assay values in the
    corpus. The caption cannot label individual columns, but it can answer the
    question that decides whether the table is read at all.

    Deliberately conservative: a caption with no assay language yields nothing,
    so MS/characterisation tables ("The following table of compounds were
    prepared using the aforementioned methods") stay excluded.
    """
    cap = (caption or "").strip()
    if not cap or not _CAPTION_ASSAY.search(cap):
        return None, None
    unit = _unit_from(cap)
    # Name it from the clause carrying the assay language, not the whole
    # paragraph — captions run to several sentences of buffer composition.
    best = ""
    for clause in re.split(r"[.;:]", cap):
        if _CAPTION_ASSAY.search(clause):
            words = clause.strip()
            if not best or len(words) < len(best):
                best = words
    name = re.sub(r"\s+", " ", best)[:80].strip() or "assay (from caption)"
    return name, unit


_METRIC = re.compile(r"\b(ic\s*50|ec\s*50|ки|ki|kd|gi\s*50|cc\s*50)\b", re.I)


def infer_units_from_description(description: str, assay_names: list[str]) -> dict[str, str]:
    """Resolve units for assays whose table never states one.

    Some tables name the assay but not its unit, deferring to the methods
    section ("Using the assays described above..."). US10245267 reports
    `C-Raf FL IC50` with values like 0.000145 and no unit anywhere in the
    table; the unit lives in prose further up the document.

    A wrong unit is a 1000-fold error — far worse than no unit — so this is
    deliberately strict: a unit is accepted only when it appears in the same
    sentence as both a distinctive token of the assay name and a potency
    metric, and only when every such sentence agrees. Any conflict yields
    nothing and the value stays unitless.
    """
    if not description or not assay_names:
        return {}
    sentences = re.split(r"(?<=[.;])\s+", description)
    out: dict[str, str] = {}
    for name in assay_names:
        # Distinctive tokens: drop generic metric words so "IC50" alone can't match.
        tokens = [w for w in re.findall(r"[A-Za-z][\w-]{2,}", name)
                  if not _METRIC.fullmatch(w) and w.lower() not in
                  {"the", "and", "for", "with", "assay", "gmean", "mean", "value", "values"}]
        if not tokens:
            continue
        found: set[str] = set()
        for s in sentences:
            if not _METRIC.search(s):
                continue
            if not any(re.search(rf"\b{re.escape(tok)}\b", s, re.I) for tok in tokens):
                continue
            u = _unit_from(s)
            if u and u != "%":
                found.add(u)
        if len(found) == 1:
            out[name] = found.pop()
        elif len(found) > 1:
            logger.info("unit for %r ambiguous across methods text (%s); leaving unset",
                        name, sorted(found))
    return out


@lru_cache(maxsize=1)
def _vocab() -> tuple[frozenset[str], dict[str, str], frozenset[str]]:
    """(assay-type lemmas, qualifier surface → canonical, null markers).

    Reuses the vocabulary the project already learned rather than restating it.
    """
    path = config.PACKAGE_ROOT / "data" / "assay_vocabulary.json"
    assays: set[str] = set()
    quals: dict[str, str] = {}
    nulls: set[str] = set()
    try:
        tokens = json.loads(path.read_text()).get("tokens", [])
    except Exception as e:              # vocabulary is an optimization, not a dependency
        logger.warning("assay vocabulary unavailable (%r); using built-ins", e)
        tokens = []
    for t in tokens:
        cls = t.get("class")
        forms = [s.lower() for s in (t.get("surface_forms") or []) if s]
        if cls == "ASSAY_TYPE":
            assays.update(forms + [str(t.get("lemma", "")).lower()])
        elif cls == "QUALIFIER":
            for s in forms:
                quals[s] = t.get("maps_to") or s
        elif cls == "NULL_MARKER":
            nulls.update(forms + [str(t.get("lemma", "")).lower()])
    nulls.update({"-", "--", "—", "nd", "n.d.", "nt", "n.t.", "na", "n/a", ""})
    return frozenset(a for a in assays if a), quals, frozenset(nulls)


# ── data model ────────────────────────────────────────────────────

@dataclass
class Column:
    index: int
    header: str
    kind: str
    unit: str | None = None
    assay_name: str | None = None


@dataclass
class AssayRecord:
    cid: str
    assay_name: str
    value_numeric: float | None = None
    qualifier: str | None = None
    unit: str | None = None
    n_runs: int | None = None
    letter_grade: str | None = None
    # Bounds when the patent published a bin rather than a number. Both may be
    # set with value_numeric None — a compound known to be 0.1-1 uM is a real
    # record, and inventing a point value for it would be a fabrication.
    range_lo: float | None = None
    range_hi: float | None = None
    value_text: str = ""
    table_id: str = ""
    column_header: str = ""
    source: str = "uspto_xml_table"
    unit_source: str = "column"   # "column" | "caption" | "description"

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}

    def missing_fields(self) -> list[str]:
        """What stops this from being a usable measurement.

        The completeness contract. Every detector bug this session came from
        measuring a PROXY for success — records produced, rows read, coverage
        ratio — and each proxy was blind to some failure the others caught.
        A grade-only record with no key satisfies "a record was produced" while
        containing no measurement at all.

        So the single question is asked of the output itself: is this a usable
        measurement, and if not, which field is missing? The missing field IS
        the diagnosis, and it maps directly onto a repair:

            value      -> value_pattern, or a bin key that was never found
            unit       -> unit resolution from header/legend/description
            assay_name -> column_map
            cid        -> column_map
        """
        missing = []
        if not self.cid:
            missing.append("cid")
        if not self.assay_name or "unnamed" in self.assay_name.lower():
            missing.append("assay_name")
        if (self.value_numeric is None
                and self.range_lo is None and self.range_hi is None):
            missing.append("value")
        if not self.unit:
            missing.append("unit")
        return missing

    @property
    def is_usable(self) -> bool:
        """A measurement a chemist could act on: identified, named, quantified."""
        return not self.missing_fields()


# ── header handling ───────────────────────────────────────────────

_TABLE_TITLE = re.compile(r"^\s*TABLE\s*[-–]?\s*[0-9IVXLC]*\s*[-–]?\s*[0-9]*\s*$", re.I)


def _row_width(row) -> int:
    """Columns this row OCCUPIES, counting empty cells.

    An empty `<entry/>` is a position, not an absence — CALS writes a label
    sitting over columns 6-8 of a nine-column table as six empty entries and
    three full ones. Skipping them made that row look three wide and sent it to
    the offset search, which then had to rediscover a placement the source had
    already stated. Rows that are genuinely short (fewer entries than the table
    has columns, no padding) still need the search and still get it.
    """
    return sum(max(1, c.colspan) for c in row)


_SHAPE_NUM = re.compile(r"^\s*[<>~≈≥≤]?\s*\d*\.?\d+\s*$")
_SHAPE_BIN = re.compile(r"^\s*(\++|[A-E])\*?\s*$")


def _column_shapes(table: Table, data) -> list[str]:
    """What each column's VALUES look like — the positional evidence a
    header row without `namest` does not carry.

    A measurement column holds numbers or grade symbols; an id column holds
    short identifiers; a name column holds long text. Which is which is
    knowable from the body even when the header's own position is not.
    """
    shapes: list[str] = []
    for i in range(table.n_cols):
        vals = [r[i].text.strip() for r in data[:40] if len(r) > i and r[i].text.strip()]
        # Column 0 is tested for an identifier FIRST. Everywhere else `num`
        # wins ties, because a measurement column of small integers must not
        # read as ids — but the leftmost column is where patents conventionally
        # put the example number, and without this a bare-integer id column
        # reads as `num`, which then scores an assay label sitting on top of it
        # as a good fit and drags the whole header one column left.
        if not vals:
            shapes.append("empty")
        elif i == 0 and sum(bool(_CID_PAT.match(v)) for v in vals) > len(vals) * 0.6:
            shapes.append("cid")
        elif sum(bool(_SHAPE_NUM.match(v)) for v in vals) > len(vals) * 0.6:
            shapes.append("num")
        elif sum(bool(_SHAPE_BIN.match(v)) for v in vals) > len(vals) * 0.6:
            shapes.append("bin")
        elif sum(bool(_CID_PAT.match(v)) for v in vals) > len(vals) * 0.6:
            shapes.append("cid")
        else:
            shapes.append("text")
    return shapes


def _header_coherence(headers: list[str], shapes: list[str] | None = None) -> int:
    """How much a candidate header assignment 'makes sense'.

    Used to choose between possible alignments of a short header row. A patent
    that omits leading columns in one header row gives no positional hint, and
    a fixed left- or right-alignment rule gets some layouts right and others
    catastrophically wrong (it shifts every assay name one column, mislabelling
    values rather than dropping them). Scoring the *result* instead lets the
    data decide: the alignment that yields a clean id column and unit-bearing
    assay names is the one the typesetter meant.
    """
    score = 0
    n_cid = 0
    for i, h in enumerate(headers):
        col = classify_column(h, [])
        if col.kind == ASSAY:
            score += 3 if col.unit else 2
        elif col.kind == CID:
            n_cid += 1
        elif col.kind in (NMR, MS, MW, RT, STRUCTURE):
            score += 1          # a confidently-excluded column is also a win
        elif col.kind == UNKNOWN and h.strip():
            score -= 1          # text we failed to make sense of

        # Does the label sit over values of the right kind?
        #
        # Without this the score is blind to the body, so an alignment that
        # parks "IC50 / Ki" over the id and structure columns scores the same
        # as one that parks them over the grades. On US10172859 the blind score
        # left-packed three assay names into columns 0-2 of a table whose
        # measurements are in 3-5, and every record came out unnamed — which in
        # turn made a correct three-scale bin_key unbindable.
        #
        # The body is the positional evidence the header row lost.
        if not shapes or i >= len(shapes) or not h.strip():
            continue
        shape = shapes[i]
        if col.kind == ASSAY:
            score += 4 if shape in ("num", "bin") else -4
        elif col.kind == CID:
            # `num` counts as compatible: a column of plain integers is shaped
            # like a measurement and like an example number both, and demanding
            # `cid` here penalised the CORRECT alignment on every table whose
            # ids are bare integers.
            score += 3 if shape in ("cid", "num") else -3
        elif col.kind == STRUCTURE:
            score += 2 if shape == "text" else -1
    score += 3 if n_cid == 1 else (-2 if n_cid > 1 else 0)
    return score


def _choose_offsets(table: Table, rows) -> list[int]:
    """Pick a starting column for each header row.

    Rows carrying CALS `namest` are authoritative and pinned at 0 (the cells
    place themselves). For the rest we try every feasible offset and keep the
    combination that scores best.

    Unpinned rows of the SAME WIDTH are solved together, not one at a time.
    They are lines of one stacked label block — US10172859's Table 6 heads its
    three right-hand columns with::

        IC50   IC50   Ki
        DNA-   pDNA-  [Kv1.11

    which is one set of labels typeset over two lines, so they cover the same
    columns by construction. Scoring each line on its own put the first at
    columns 3-5 and its own continuation at 0-2: both assay columns came out
    called `IC50 PK`, and since the patent measures DNA-PK in nM and pDNA-PK in
    μM, nothing downstream could tell which was which. Sharing the offset makes
    that split unrepresentable rather than merely unlikely.
    """
    _, data = _header_rows_of(table)
    shapes = _column_shapes(table, data)

    offsets = []
    for row in rows:
        if any(c.col_start >= 0 for c in row):
            offsets.append(0)
            continue
        slack = table.n_cols - _row_width(row)
        offsets.append(0 if slack <= 0 else None)   # type: ignore[arg-type]

    groups: dict[int, list[int]] = {}
    for i, off in enumerate(offsets):
        if off is None:
            groups.setdefault(_row_width(rows[i]), []).append(i)

    for width, idxs in groups.items():
        best_off, best_score = 0, None
        for cand in range(table.n_cols - width + 1):
            trial = [0 if o is None else o for o in offsets]
            for i in idxs:
                trial[i] = cand
            s = _header_coherence(_merge_with_offsets(table, rows, trial), shapes)
            if best_score is None or s > best_score:
                best_off, best_score = cand, s
        for i in idxs:
            offsets[i] = best_off
    return [0 if o is None else o for o in offsets]


def _is_legend_row(row) -> bool:
    """A sentence of prose occupying few cells, rather than column labels.

    Column labels are short strings spread across the table's width; a legend
    is one long run of text in one or two cells, often a fragment because the
    typesetter wrapped the sentence across several rows.
    """
    cells = [c.text.strip() for c in row if c.text.strip()]
    if not cells or len(cells) > 2:
        return False
    text = " ".join(cells)
    return len(text) > 55 and " " in text.strip()


def _looks_like_header_row(row) -> bool:
    """A row of labels rather than data.

    Used to promote header rows that patents put in `tbody`. The test is that
    no cell parses as a measurement and at least one carries alphabetic text —
    a data row always has at least one number.
    """
    texts = [c.text.strip() for c in row if c.text.strip()]
    if not texts:
        return False
    if any(_VALUE_PAT.match(t) for t in texts):
        return False
    return any(re.search(r"[A-Za-z]", t) for t in texts)


def _header_rows_of(table: Table) -> tuple[list, list]:
    """Split into (header rows, data rows).

    `thead` is authoritative when present. Otherwise the leading run of
    label-only rows in `tbody` is promoted — US8952177 puts its entire header
    ("Cmp No." / "FLAP Binding wild type HTRF Ki (μM)") in `tbody`, and reading
    only `thead` loses every assay name in the patent.
    """
    if table.header_rows:
        return table.header_rows, table.body_rows
    header, i = [], 0
    for i, row in enumerate(table.body_rows):
        if not any(c.text.strip() for c in row):        # spacer — keep scanning
            continue
        if _is_legend_row(row):
            # Prose legend printed above the column labels. Merging it into the
            # header would smear a whole sentence across the column names; it is
            # picked up separately by `table_legend`, which is where its unit
            # gets read from.
            continue
        if _looks_like_header_row(row) and len(header) < 8:
            header.append(row)
        else:
            break
    return header, table.body_rows[i:] if header else table.body_rows


def table_legend(table: Table) -> str:
    """Prose legend printed inside the table, above its column labels.

    Patents put the table's explanatory sentence in the table itself, wrapped
    across single-cell rows by the typesetter:

        "Selected compound structures and Raf inhibition data: numbering"
        "corresponds to the Examples above ... IC50's are"
        "micromolar."

    That sentence frequently carries the unit — and it is the ONLY place
    US10245267 states it. Read row-by-row the words split apart and the unit is
    invisible, so the rows are rejoined before anything is matched against them.
    Distinguished from column labels by being long prose in few cells, where a
    label row is short strings spread across most columns.
    """
    parts: list[str] = []
    for row in table.body_rows[:12]:
        cells = [c.text.strip() for c in row if c.text.strip()]
        if not cells:
            continue
        text = " ".join(cells)
        # Stop at the first row that looks like data or like column labels.
        if any(_VALUE_PAT.match(c) for c in cells):
            break
        if len(cells) > 2 and len(text) < 60:
            break                       # short strings across columns = labels
        if len(text) < 12 and not parts:
            continue                    # a bare "TABLE 1" caption line
        parts.append(text)
        if len(parts) >= 6:
            break
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def merge_header(table: Table, header_rows=None) -> list[str]:
    """Collapse stacked header rows into one string per column.

    Two things make this non-trivial, both seen in real grants:

    - Patents stack an assay name vertically across rows ("Ave" / "A2B" /
      "cAMP" / "IC50"), so every row must contribute, top to bottom.
    - Header rows often omit leading columns, so a row's first cell is not
      necessarily column 0. We honour CALS `namest` where present and only
      fall back to left-to-right accumulation when it isn't; getting this
      wrong shifts every assay name one column sideways, which silently
      mislabels values rather than dropping them.
    """
    rows = table.header_rows if header_rows is None else header_rows
    return _merge_with_offsets(table, rows, _choose_offsets(table, rows))


def _merge_with_offsets(table: Table, rows, offsets: list[int]) -> list[str]:
    cols: list[list[str]] = [[] for _ in range(table.n_cols)]
    for row, offset in zip(rows, offsets):
        pos = offset
        for cell in row:
            span = max(1, cell.colspan)
            start = cell.col_start if cell.col_start >= 0 else pos
            text = cell.text.strip()
            # Drop the table's own title ("TABLE 569") — it is a caption, not
            # a column label, and folding it in corrupts every assay name.
            if text and not _TABLE_TITLE.match(text):
                for i in range(start, min(start + span, table.n_cols)):
                    cols[i].append(text)
            pos = start + span
            if pos >= table.n_cols:
                break
    # De-duplicate repeated fragments while preserving order (a name spanning
    # several columns otherwise repeats itself).
    out = []
    for parts in cols:
        seen, uniq = set(), []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                uniq.append(p)
        out.append(_join_header_lines(uniq))
    return out


def _join_header_lines(parts: list[str]) -> str:
    """Join the lines of one stacked column label.

    A fragment ending in a hyphen is a word the typesetter broke across lines,
    not two words: US10172859 stacks `DNA-` over `PK`, meaning `DNA-PK`. Joining
    those with a space yields `IC50 DNA- PK`, which reads fine to a human and
    breaks every consumer that matches on the name — the bin-key scale patterns
    for that patent are `DNA-?PK`, and a stray space made them bind nothing, so
    three tables extracted only their hERG column.
    """
    text = ""
    for p in parts:
        if not text:
            text = p
        elif text.endswith("-"):
            text += p
        else:
            text += " " + p
    return text.strip()


def _unit_from(text: str) -> str | None:
    m = _UNIT_PAT.search(text or "")
    if not m:
        return None
    raw = (m.group(1) or m.group(2) or m.group(3) or "").strip()
    low = raw.lower()
    if low in _SPELLED_UNIT:
        return _SPELLED_UNIT[low]
    return {"um": "uM", "µm": "uM", "μm": "uM", "nm": "nM", "mm": "mM",
            "pm": "pM", "percent": "%"}.get(low, raw)


def classify_column(header: str, samples: list[str]) -> Column:
    """Decide what a column holds, from its header first, its values second.

    Order matters and is deliberately conservative: the exclusions (NMR, MS,
    MW, RT, structure) are checked BEFORE the assay test, because headers like
    "LCMS IC50 method" would otherwise read as an assay. Anything unrecognised
    becomes UNKNOWN and is skipped rather than guessed.
    """
    h = (header or "").strip()
    low = h.lower()

    if _HEADER_NMR.search(low):
        return Column(-1, h, NMR)
    if _HEADER_MW.search(low):
        return Column(-1, h, MW)
    if _HEADER_MS.search(low):
        return Column(-1, h, MS)
    if _HEADER_RT.search(low):
        return Column(-1, h, RT)
    if _HEADER_STRUCT.search(low):
        return Column(-1, h, STRUCTURE)
    if _HEADER_NRUNS.match(h):
        return Column(-1, h, NRUNS)
    if _HEADER_SUBST.match(h):
        return Column(-1, h, SUBSTITUENT)
    if _HEADER_CID.search(low) and not _HEADER_ASSAY.search(low):
        return Column(-1, h, CID)

    assay_lemmas, _, _ = _vocab()
    is_assay = bool(_HEADER_ASSAY.search(low)) or any(a in low for a in assay_lemmas if len(a) > 2)
    unit = _unit_from(h)
    if is_assay or (unit and unit != "%"):
        return Column(-1, h, ASSAY, unit=unit, assay_name=h or "unnamed assay")

    # Headerless continuation columns: fall back to the shape of the data.
    if not h:
        vals = [s for s in samples if s]
        if vals:
            if sum(bool(_NRUNS_ONLY.match(v)) for v in vals) > len(vals) * 0.6:
                return Column(-1, h, NRUNS)
            if sum(bool(_LETTER_BIN.match(v)) for v in vals) > len(vals) * 0.6:
                return Column(-1, h, ASSAY, assay_name="unnamed assay (letter bin)")
    return Column(-1, h, UNKNOWN)


def _label_bearing(table: Table, rows) -> list[int]:
    """Data columns that a header would actually name.

    Run-count columns (`(8)`) are structural: patents write them beside the
    value and never label them, so the header of a 3-column table can describe
    a 5-column data table. Excluding them is what makes the two line up.
    """
    keep = []
    for i in range(table.n_cols):
        vals = [r[i].text.strip() for r in rows if len(r) > i and r[i].text.strip()]
        if vals and sum(bool(_NRUNS_ONLY.match(v)) for v in vals) > len(vals) * 0.6:
            continue
        keep.append(i)
    return keep


def _fit_inherited(inherited: list[str], table: Table, rows) -> list[str]:
    """Map a previous tgroup's header onto this (headerless) continuation.

    Widths often differ — US8952177's header table declares 3 columns while its
    data table has 5, because each value carries an unlabelled `(n)` column.
    Assigning positionally would put the LTB4 assay name on a run-count column
    and silently mislabel every value, so labels go onto the label-bearing
    columns in order instead.
    """
    headers = [""] * table.n_cols
    if len(inherited) == table.n_cols:
        return list(inherited)
    targets = _label_bearing(table, rows)
    for label, idx in zip(inherited, targets):
        headers[idx] = label
    return headers


def build_columns(table: Table, inherited: list[str] | None = None,
                  data_rows=None, inherited_unit: str | None = None) -> list[Column]:
    hdr_rows, body = _header_rows_of(table)
    headers = merge_header(table, hdr_rows)
    rows = body if data_rows is None else data_rows
    if inherited and not any(headers):
        headers = _fit_inherited(inherited, table, rows)
    cols: list[Column] = []
    for i in range(table.n_cols):
        samples = [r[i].text for r in rows[:40] if len(r) > i]
        c = classify_column(headers[i] if i < len(headers) else "", samples)
        c.index = i
        cols.append(c)

    # Exactly one id column. If the header didn't name one, take the leftmost
    # column whose values actually look like compound ids.
    if not any(c.kind == CID for c in cols):
        best, best_score = None, 0.0
        for c in cols:
            vals = [r[c.index].text for r in rows
                    if len(r) > c.index and r[c.index].text.strip()]
            if not vals:
                continue
            score = sum(bool(_CID_PAT.match(v)) for v in vals) / len(vals)
            if score > best_score:
                best, best_score = c, score
        if best is not None and best_score >= 0.7:
            best.kind = CID
            best.assay_name = None

    # Last resort: a table with no header anywhere, whose caption says it is
    # assay data. Unlabelled numeric columns become assays named from the
    # caption. Gated on the caption so characterisation tables (MS, purity)
    # stay excluded — without that gate this would re-import exactly the noise
    # column classification exists to remove.
    # A unit stated only in the table's legend or caption applies to every
    # assay column that didn't carry its own. Done before the caption fallback
    # so a table with proper headers but no inline unit still gets one.
    legend = table_legend(table)
    ctx_unit = _unit_from(legend) or _unit_from(table.caption) or inherited_unit
    if ctx_unit and ctx_unit != "%":
        for c in cols:
            if c.kind == ASSAY and not c.unit:
                c.unit = ctx_unit

    if not any(c.kind == ASSAY for c in cols):
        cap_name, cap_unit = caption_assay_hint(table.caption)
        if cap_name:
            for c in cols:
                if c.kind != UNKNOWN or c.header.strip():
                    continue
                vals = [r[c.index].text.strip() for r in rows
                        if len(r) > c.index and r[c.index].text.strip()]
                if not vals:
                    continue
                usable = sum(bool(_VALUE_PAT.match(v) or _LETTER_BIN.match(v)
                                  or _PLUS_BIN.match(v)) for v in vals)
                if usable >= len(vals) * 0.7:
                    c.kind = ASSAY
                    c.assay_name = cap_name
                    c.unit = cap_unit
    return cols


# ── value parsing ─────────────────────────────────────────────────

def parse_value(cell: str) -> dict | None:
    """Parse one measurement cell. None when it holds no usable value."""
    s = (cell or "").strip()
    _, quals, nulls = _vocab()
    if s.lower() in nulls:
        return None
    lb = _LETTER_BIN.match(s)
    if lb:
        return {"letter_grade": lb.group(1).upper(), "value_text": s}
    pb = _PLUS_BIN.match(s)
    if pb:
        # `+`/`++`/`+++` is an ordinal potency band. Recorded as a grade; any
        # number we assigned to it would be invented.
        return {"letter_grade": pb.group(1), "value_text": s}
    m = _VALUE_PAT.match(s)
    if not m:
        return None
    qual = m.group("qual")
    if qual:
        qual = quals.get(qual.lower(), qual)
        qual = {">=": "≥", "<=": "≤", "≈": "~"}.get(qual, qual)
    try:
        num = float(m.group("num").replace(",", ""))
    except ValueError:
        return None
    # A parenthetical of digits is a replicate count; anything else is a
    # footnote marker and carries no measurement meaning.
    paren = (m.group("paren") or "").strip()
    n_runs = int(paren) if paren.isdigit() and len(paren) <= 3 else None
    return {
        "value_numeric": num,
        "qualifier": qual,
        "unit": m.group("unit"),
        "n_runs": n_runs,
        "annotation": paren if paren and not paren.isdigit() else None,
        "value_text": s,
    }


def _is_spacer(row) -> bool:
    return not any(c.text.strip() for c in row)


# ── extraction ────────────────────────────────────────────────────

# A cell holding a list of compound ids rather than a value. Patents invert the
# usual layout for large screens: one row per potency bin, every compound in
# that bin listed inside a single cell.
_BIN_SYMBOL = re.compile(r"^\s*(\++|[A-E]|NT|N\.?T\.?)\s*$", re.I)


def _cid_list(text: str) -> list[str]:
    """Compound ids from a comma/space separated cell, or [] if it isn't one."""
    s = (text or "").strip()
    if s.count(",") < 2:
        return []
    parts = [p.strip() for p in s.split(",")]
    ids = [p for p in parts if p and _CID_PAT.match(p)]
    # Require most of the cell to be ids, so a prose sentence with a couple of
    # numbers in it is not mistaken for a compound list.
    return ids if len(ids) >= max(3, int(len(parts) * 0.7)) else []


def extract_inverted(tables: list[Table], bin_key: dict, *,
                     assay_name: str, unit: str | None,
                     table_id: str = "") -> list[AssayRecord]:
    """Read 'one row per bin, compound ids listed in a cell' tables.

    US11566007 publishes 1,827 compound-bin assignments this way:

        IC50*  |  Examples
          +    |  A104, A107, A108, ... (253 ids in one cell)
         ++    |  A10, A105, ...        (296 ids)

    The normal row-per-compound reader sees a table with almost no numeric
    cells and skips it entirely, which is how these were missed. Every id in
    the cell gets its own record carrying the bin's range.
    """
    out: list[AssayRecord] = []
    for t in tables:
        # The bin symbol appears once, on the first row of its group; the
        # compound list then wraps across many following rows with no symbol of
        # their own. Requiring symbol and ids in the same row caught only the
        # first fragment — 84 of 1,827 assignments on US11566007 — so the
        # current symbol is carried forward until a new one appears.
        current_sym: str | None = None
        for row in t.body_rows:
            cells = [c.text.strip() for c in row]
            sym = next((c for c in cells if _BIN_SYMBOL.match(c)), None)
            if sym:
                current_sym = sym
            if not current_sym:
                continue
            ids: list[str] = []
            for c in cells:
                ids = _cid_list(c)
                if ids:
                    break
            if not ids:
                continue
            if current_sym.upper().replace(".", "") == "NT":
                continue                      # explicitly not tested
            sym = current_sym
            rng = bin_key.get(sym)
            for cid in ids:
                out.append(AssayRecord(
                    cid=normalize_cid(cid),
                    assay_name=assay_name,
                    value_numeric=None,
                    unit=(rng.unit if rng else unit),
                    letter_grade=sym,
                    range_lo=rng.lo if rng else None,
                    range_hi=rng.hi if rng else None,
                    value_text=sym,
                    table_id=table_id or t.table_id,
                    column_header=assay_name,
                    source="uspto_xml_bin_table",
                ))
    return out


def _best_per_block(raw: list[Table]) -> list[Table]:
    """Per `<tables>` block, whichever view yields more usable records.

    The two views are the block's raw tgroups (each parsed on its own, headers
    inherited from the previous one) and the single grid `assemble_block` builds
    from them. Neither dominates: assembly is required wherever a block is
    fragmented per compound, and loses wherever a block's columns are spanned
    without `namest`. Scored on usable records — the completeness contract —
    rather than on row or record count, because a restructuring that doubles the
    record count while stripping the unit off every one of them is not a win.
    """
    from .uspto_xml import assemble_block

    order: list[str] = []
    for t in raw:
        if t.table_id not in order:
            order.append(t.table_id)

    out: list[Table] = []
    for tid in order:
        frag = [t for t in raw if t.table_id == tid]
        merged = assemble_block(raw, tid)
        if merged is None:
            out.extend(frag)
            continue
        n_frag = sum(1 for r in extract_from_tables(frag) if r.is_usable)
        n_merged = sum(1 for r in extract_from_tables([merged]) if r.is_usable)
        if n_merged >= n_frag:
            out.append(merged)
        else:
            out.extend(frag)
    return out


def extract_from_patent(xml: str) -> list[AssayRecord]:
    """Full extraction for one patent.

    Three passes, because patents publish assay data in three shapes:
      1. row per compound, value in a cell        (the common case)
      2. row per potency bin, compound ids listed (inverted; US11566007)
      3. bin symbols in a normal table, defined by a legend (US11292791)

    Bin keys are resolved per `<tables>` block and never shared between them —
    two patents in this corpus assign incompatible ranges to `++++`.
    """
    from . import bin_legend
    from .uspto_xml import description_text, parse_tables

    raw = parse_tables(xml)
    # One assembled grid per <tables> block. A block is fragmented into many
    # tgroups and no single one of them is the table; see `assemble_block`.
    #
    # Assembly is applied per block and only where it WINS. Some blocks encode
    # column spans that survive the per-tgroup path and are unrecoverable once
    # merged: US8952177 TABLE-US-00003 writes "FLAP Binding wild / Human Whole
    # Blood" as two colspan-1 cells that actually span two sub-columns each,
    # and CALS `namest` is absent, so the pairing exists nowhere in the source.
    # Assembling that block costs 169 units; assembling US10172859 gains 498
    # rows. Choosing per block keeps both — and the same anti-deletion rule the
    # repair loop applies to model-proposed rules applies here: a restructuring
    # must add usable records, never remove them.
    tables = _best_per_block(raw)
    records = extract_from_tables(tables)

    assembled = {t.table_id: t for t in tables}

    # Resolve a bin key for each <tables> block from its own legend, caption or
    # trailing footnote. The key often sits AFTER the data it explains.
    #
    # Legend text is harvested from the RAW tgroups, not the assembled grid:
    # assembly keeps only the block's data width, and a legend is routinely
    # published as a narrow fragment alongside it. Records, by contrast, come
    # from the assembled grid.
    by_block: dict[str, list[Table]] = {}
    for t in raw:
        by_block.setdefault(t.table_id, []).append(t)

    for block_id, raw_block in by_block.items():
        block = [assembled[block_id]] if block_id in assembled else raw_block
        # ...INCLUDING the prose immediately before and after the block.
        #
        # Each of US11566007's bin tables states its own key, and eleven of them
        # state it in the paragraph above rather than in a caption or a cell.
        # Harvesting only caption + legend + cells found the key for two blocks
        # and missed it for eleven, which read as an unreadable layout.
        #
        # It is NOT a transfer from the blocks that worked, and that distinction
        # is worth 10x: TABLE-US-00006 defines a FOUR-grade scale where `++++` is
        # `IC50 >= 1 uM`, while TABLE-US-00007 defines a FIVE-grade scale where
        # `++++` is `10 uM > IC50 >= 1 uM`. Carrying one key to the other looked
        # like a free +2,261 records and would have silently rewritten an
        # upper-bounded bin as an unbounded one. Each block's own key is the only
        # safe answer, and each block has one.
        text = " ".join(
            [raw_block[0].caption] + [table_legend(t) for t in raw_block]
            + [c.text for t in raw_block for r in t.body_rows for c in r
               if bin_legend.looks_like_key(c.text)]
            + [(raw_block[0].preceding or "")[-1200:]]
        )
        if not bin_legend.looks_like_key(text):
            continue
        key = bin_legend.parse_bin_key(text)
        if not key:
            continue
        name, unit = caption_assay_hint(raw_block[0].caption)
        name = name or _assay_name_from(text) or "assay (binned)"
        records.extend(extract_inverted(
            block, key, assay_name=name,
            unit=unit or next((b.unit for b in key.values() if b.unit), None),
            table_id=block_id,
        ))
        # Bin symbols sitting in ordinary row-per-compound tables.
        for r in records:
            if r.table_id == block_id and r.letter_grade and r.range_lo is None \
                    and r.range_hi is None and r.letter_grade in key:
                br = key[r.letter_grade]
                r.range_lo, r.range_hi = br.lo, br.hi
                r.unit = r.unit or br.unit

    unitless = sorted({r.assay_name for r in records if not r.unit})
    if unitless:
        resolved = infer_units_from_description(description_text(xml), unitless)
        for r in records:
            if not r.unit and r.assay_name in resolved:
                r.unit = resolved[r.assay_name]
                r.unit_source = "description"
    return records


def _assay_name_from(text: str) -> str | None:
    """Best-effort assay name from a bin table's own heading text."""
    m = re.search(r"([A-Za-z0-9 /()\-]{4,60}?(?:IC\s*50|EC\s*50|Ki|Kd))", text)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def extract_from_tables(tables: list[Table]) -> list[AssayRecord]:
    """Turn a patent's CALS tables into assay records.

    Continuation tables (same column count, no header of their own) inherit the
    most recent header — patents split one logical table across many tgroups
    when it spans pages, and the later pieces carry no header at all.
    """
    out: list[AssayRecord] = []
    last_header: dict[int, list[str]] = {}
    by_table_id: dict[str, list[str]] = {}
    unit_by_table_id: dict[str, str] = {}

    for t in tables:
        hdr_rows, data_rows = _header_rows_of(t)
        headers = merge_header(t, hdr_rows)
        # A legend/caption unit stated once in a <tables> block applies to the
        # data tgroups that follow it under the same id.
        _u = _unit_from(table_legend(t)) or _unit_from(t.caption)
        if _u and _u != "%":
            unit_by_table_id[t.table_id] = _u
        if any(headers):
            last_header[t.n_cols] = headers
            # Scope cross-width inheritance to the SAME `<tables>` element.
            # A patent splits one logical table into a header tgroup and a data
            # tgroup of different widths under a single id (US8952177: 3-column
            # header, 5-column data). Inheriting by "most recent header of any
            # width" instead leaks a header onto an unrelated later table —
            # it stamped `CBP IC50` onto US11292791's `+`/`++` bins, which is
            # mislabelled data, strictly worse than no label at all.
            by_table_id[t.table_id] = headers
        cols = build_columns(t, inherited=last_header.get(t.n_cols) or by_table_id.get(t.table_id),
                             inherited_unit=unit_by_table_id.get(t.table_id),
                             data_rows=data_rows)

        cid_col = next((c for c in cols if c.kind == CID), None)
        assay_cols = [c for c in cols if c.kind == ASSAY]
        if cid_col is None or not assay_cols:
            continue

        for row in data_rows:
            if _is_spacer(row) or len(row) <= cid_col.index:
                continue
            raw_cid = row[cid_col.index].text.strip()
            if not raw_cid or not _CID_PAT.match(raw_cid):
                continue
            cid = normalize_cid(raw_cid)
            if not cid:
                continue

            for c in assay_cols:
                if len(row) <= c.index:
                    continue
                parsed = parse_value(row[c.index].text)
                if not parsed:
                    continue
                n_runs = parsed.get("n_runs")
                # A bare "(8)" in the next column is this value's run count.
                if n_runs is None and len(row) > c.index + 1:
                    nxt = _NRUNS_ONLY.match(row[c.index + 1].text)
                    if nxt and cols[c.index + 1].kind in (NRUNS, UNKNOWN):
                        n_runs = int(nxt.group(1))
                out.append(AssayRecord(
                    cid=cid,
                    assay_name=c.assay_name or c.header or "unnamed assay",
                    value_numeric=parsed.get("value_numeric"),
                    qualifier=parsed.get("qualifier"),
                    unit=parsed.get("unit") or c.unit,
                    n_runs=n_runs,
                    letter_grade=parsed.get("letter_grade"),
                    value_text=parsed.get("value_text", ""),
                    table_id=t.table_id,
                    column_header=c.header,
                ))
    return out


def to_assay_tables(records: list[AssayRecord]) -> dict[str, list[dict]]:
    """Group into the `assay_tables.json` shape the pipeline already writes."""
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r.cid, []).append({
            "assay_name": r.assay_name,
            "value_numeric": r.value_numeric,
            "unit": r.unit,
            "qualifier": r.qualifier,
            "n_runs": r.n_runs,
            "letter_grade": r.letter_grade,
            "source": r.source,
        })
    return out


def column_report(tables: list[Table]) -> list[dict]:
    """What each column was classified as — for auditing a patent's parse."""
    rows: list[dict] = []
    last_header: dict[int, list[str]] = {}
    by_table_id: dict[str, list[str]] = {}
    unit_by_table_id: dict[str, str] = {}
    for t in tables:
        hdr_rows, data_rows = _header_rows_of(t)
        headers = merge_header(t, hdr_rows)
        # A legend/caption unit stated once in a <tables> block applies to the
        # data tgroups that follow it under the same id.
        _u = _unit_from(table_legend(t)) or _unit_from(t.caption)
        if _u and _u != "%":
            unit_by_table_id[t.table_id] = _u
        if any(headers):
            last_header[t.n_cols] = headers
            # Scope cross-width inheritance to the SAME `<tables>` element.
            # A patent splits one logical table into a header tgroup and a data
            # tgroup of different widths under a single id (US8952177: 3-column
            # header, 5-column data). Inheriting by "most recent header of any
            # width" instead leaks a header onto an unrelated later table —
            # it stamped `CBP IC50` onto US11292791's `+`/`++` bins, which is
            # mislabelled data, strictly worse than no label at all.
            by_table_id[t.table_id] = headers
        for c in build_columns(t, inherited=last_header.get(t.n_cols) or by_table_id.get(t.table_id),
                             inherited_unit=unit_by_table_id.get(t.table_id),
                               data_rows=data_rows):
            rows.append({"table": t.table_id, "col": c.index,
                         "header": c.header[:60], "kind": c.kind, "unit": c.unit})
    return rows
