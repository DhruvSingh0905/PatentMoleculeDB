"""USPTO grant full-text XML — the primary text and table source.

Why this exists
---------------
Every other source in this pipeline reconstructs structure that USPTO already
published. Google Patents renders clean text but drops `<table>` structure;
MinerU recovers table structure by OCR-ing a PDF, paying for it with `<|ref|>`
tags, `[[bbox]]` pollution, merged rows and character errors (`2-y1}` for
`2-yl}`). USPTO's own grant XML carries the tables as OASIS/CALS markup —
exact cells, no OCR, qualifiers intact.

Measured on US8952177: a first-pass parser over this XML reproduced 358 of the
371 (cid, value) assay pairs the LLM-backed pipeline produces — 96.5% — with
zero API calls, and introduced no pair the old path didn't already have.

What it does NOT fix
--------------------
Line-wrapped chemical names. USPTO preserves the *typeset* line break as
separate `<entry>` elements:

    <entry>...1H-benzimidazol-2-</entry>
    <entry>yl}cyclohexanecarboxylic acid,</entry>

That is the same `2-yl` split we have been blaming on OCR; it is inherent to
the source document. The saving grace is that here the wrap points are explicit
element boundaries rather than something to infer from OCR soup, so
`join_wrapped_cells` can repair them deterministically and let OPSIN adjudicate.

Coverage
--------
`PTGRXML` begins in **2002** — there is no grant full-text XML before that.
Older grants must fall through to the Google Patents / MinerU tiers.

Access
------
Needs `USPTO_API_KEY` (free, from data.uspto.gov). A freshly issued key returns
403 for several minutes until its AWS usage plan attaches — that is propagation,
not a bad key. Rate limits are 60 req/min (120 off-peak).
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"
_CACHE_DIR = config.OUTPUT_DIR / "uspto_xml"

# Grant full-text XML starts here. Anything earlier has no XML at all.
FIRST_XML_GRANT_YEAR = 2002

_TIMEOUT_S = 90
_MAX_RETRIES = 3


class UsptoUnavailable(Exception):
    """No XML for this patent — caller should fall through to the next tier.

    Deliberately distinct from a transport error. 'This patent predates XML'
    and 'the network flaked' demand different responses, and collapsing them
    into a bare `return None` is how the rest of this codebase lost the ability
    to tell a real failure from an empty document.
    """


# ── data model ────────────────────────────────────────────────────

@dataclass
class Cell:
    text: str
    colspan: int = 1
    # 0-based column this cell starts at, from CALS `namest` when present.
    # Header rows frequently omit leading columns entirely — a row reading
    # "Ave | Ave | 450" may begin at column 1, not 0 — so accumulating spans
    # left-to-right misaligns the merged header. -1 means "not declared";
    # the caller falls back to accumulation.
    col_start: int = -1


@dataclass
class Table:
    """One CALS `tgroup` — a single logical grid."""
    table_id: str
    n_cols: int
    header_rows: list[list[Cell]] = field(default_factory=list)
    body_rows: list[list[Cell]] = field(default_factory=list)
    # Text immediately preceding the table. Assay names usually live here or in
    # the header, so the caller needs it to label columns.
    caption: str = ""
    # Bounded raw text immediately before the block, any element type. Legends
    # live here as often as they live in a caption.
    preceding: str = ""
    # Bounded raw text immediately after the block. Keys are often footnotes.
    following: str = ""

    @property
    def header_text(self) -> str:
        return " | ".join(
            c.text for row in self.header_rows for c in row if c.text
        )


# ── fetch ─────────────────────────────────────────────────────────

def _api_key() -> str:
    key = os.environ.get("USPTO_API_KEY", "")
    if not key:
        env = config.REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("USPTO_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    return key


def _get(url: str, key: str) -> bytes:
    """GET with the API key, retrying only genuinely transient failures."""
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        req = urllib.request.Request(url, headers={"X-API-Key": key})
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # 403 on a brand-new key means the usage plan hasn't propagated.
            # 429 is rate limiting. Both are worth waiting out; 404 is not.
            if e.code in (403, 429, 500, 502, 503) and attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt * 5
                logger.warning(
                    "uspto: HTTP %s on %s — retrying in %ss (attempt %d/%d)",
                    e.code, url.rsplit("/", 1)[-1], wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
                last = e
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    raise last if last else RuntimeError("unreachable")


def _normalize_patent_number(patent_id: str) -> str | None:
    """'US8952177' / 'US8952177B2' / '8952177' → '8952177'. None if not a grant."""
    s = patent_id.upper().strip()
    s = re.sub(r"^US", "", s)
    if re.match(r"^\d{4}0\d{6}", s):      # 20230365584A1 — publication, not grant
        return None
    s = re.sub(r"[A-Z]\d?$", "", s)       # strip kind code
    return s if s.isdigit() else None


def _normalize_publication_number(patent_id: str) -> str | None:
    """'US20240335431A1' → '20240335431'. None if not a publication number."""
    s = re.sub(r"^US", "", patent_id.upper().strip())
    s = re.sub(r"[A-Z]\d?$", "", s)
    return s if re.match(r"^\d{4}0\d{6}$", s) else None


def fetch_grant_xml(patent_id: str, *, refresh: bool = False) -> str:
    """Return the grant full-text XML for `patent_id`, caching it locally.

    Raises UsptoUnavailable when the patent has no grant XML (pre-2002, a
    publication rather than a grant, or simply absent from the index).
    """
    cached = _CACHE_DIR / f"{patent_id.upper()}.xml"
    if cached.exists() and not refresh:
        return cached.read_text(encoding="utf-8", errors="ignore")

    key = _api_key()
    if not key:
        raise UsptoUnavailable("USPTO_API_KEY is not set")

    # A document is either a granted patent (PTGRXML) or a pre-grant
    # publication (APPXML). Both are reachable the same way — the search
    # response carries a direct per-document file URI — they simply live under
    # different metadata keys. Publications are ~15% of BindingDB's
    # patent-linked rows, so refusing them forfeits a large slice of the corpus.
    number = _normalize_patent_number(patent_id)
    pub = _normalize_publication_number(patent_id) if number is None else None
    if number is None and pub is None:
        raise UsptoUnavailable(f"{patent_id}: not a US grant or publication number")

    if number is not None:
        query = f"applicationMetaData.patentNumber:{number}"
    else:
        query = f"applicationMetaData.earliestPublicationNumber:US{pub}A1"

    # 1) Look the document up to find where its XML lives.
    url = f"{_SEARCH_URL}?q={query}&limit=1"
    try:
        payload = json.loads(_get(url, key))
    except urllib.error.HTTPError as e:
        raise UsptoUnavailable(f"{patent_id}: search failed HTTP {e.code}") from e

    bag = payload.get("patentFileWrapperDataBag") or []
    if not bag:
        raise UsptoUnavailable(f"{patent_id}: not found in the USPTO index")

    # Prefer the grant text when it exists (it is the final, examined version);
    # otherwise fall back to the pre-grant publication.
    grant_meta = bag[0].get("grantDocumentMetaData") or {}
    pgpub_meta = bag[0].get("pgpubDocumentMetaData") or {}
    file_uri = grant_meta.get("fileLocationURI") or pgpub_meta.get("fileLocationURI")
    if not file_uri:
        grant_date = (bag[0].get("applicationMetaData") or {}).get("grantDate") or ""
        year = grant_date[:4]
        if year and year.isdigit() and int(year) < FIRST_XML_GRANT_YEAR:
            raise UsptoUnavailable(
                f"{patent_id}: granted {year}, before XML coverage begins "
                f"({FIRST_XML_GRANT_YEAR})"
            )
        raise UsptoUnavailable(f"{patent_id}: no grant or publication XML published")

    # 2) One ~600 KB GET for the patent's own XML.
    xml = _get(file_uri, key).decode("utf-8", errors="ignore")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".xml.tmp")
    tmp.write_text(xml, encoding="utf-8")
    os.replace(tmp, cached)
    logger.info("uspto: cached %s (%d KB)", patent_id, len(xml) // 1024)
    return xml


# ── parse ─────────────────────────────────────────────────────────

_TAG = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    """Strip markup and normalize entities/whitespace within one cell."""
    # Keep sub/superscript content but drop the tags; patents use them for
    # units and charges (e.g. IC<sub>50</sub>, [M+H]<sup>+</sup>).
    s = _TAG.sub("", fragment)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_row(row_xml: str) -> list[Cell]:
    cells: list[Cell] = []
    # SELF-CLOSING FIRST. With the paired form leading, `[^>]*` matched the `/`
    # of `<entry/>`, so an empty cell took the paired branch and `(.*?)</entry>`
    # ran forward to the NEXT closing tag — swallowing the following cell.
    #
    # CALS uses `<entry/>` as a positional placeholder: a label sitting over
    # columns 6-8 of a 9-column table is written as six empty entries and three
    # full ones. Losing them did not just drop blanks, it shifted every
    # subsequent cell left, so US11254686's nine-entry header rows came back as
    # 3/5/7/7 with no column position at all — and the offset search then had to
    # guess an alignment the patent had stated outright.
    for m in re.finditer(r"<entry\b([^>]*?)/>|<entry\b([^>]*)>(.*?)</entry>",
                         row_xml, re.S):
        attrs = m.group(1) if m.group(1) is not None else (m.group(2) or "")
        body = m.group(3) or ""
        span, start = 1, -1
        st = re.search(r'namest="(\w+)"', attrs)
        en = re.search(r'nameend="(\w+)"', attrs)
        if st:
            digits = re.sub(r"\D", "", st.group(1))
            if digits:
                start = int(digits) - 1          # colspec names are 1-based
        if st and en:
            try:
                span = max(1, int(re.sub(r"\D", "", en.group(1)))
                           - int(re.sub(r"\D", "", st.group(1))) + 1)
            except ValueError:
                span = 1
        cells.append(Cell(_text(body), span, start))
    return cells


def parse_tables(xml: str) -> list[Table]:
    """Extract every CALS `tgroup` as a Table.

    One `<tables>` element can hold several `tgroup`s — patents routinely split
    a single logical table across groups when it spans pages. Each is returned
    separately; stitching them is the caller's decision, because merging on the
    wrong boundary is exactly how rows get misattributed.
    """
    out: list[Table] = []
    for tbl in re.finditer(r"<tables\b([^>]*)>(.*?)</tables>", xml, re.S):
        id_match = re.search(r'id="([^"]+)"', tbl.group(1) or "")
        table_id = id_match.group(1) if id_match else ""
        block = tbl.group(2)
        # Many tgroups carry no header of their own — the assay names and units
        # live in the paragraph immediately before the table ("TABLE 4: hERG
        # IC50 (μM)"). Without this, those tables classify as all-unknown and
        # are silently dropped, which accounted for the single largest block of
        # missed assay values. Look back a bounded window for the nearest
        # preceding paragraph or heading.
        window = xml[max(0, tbl.start() - 3000):tbl.start()]
        prev = re.findall(r"<(?:p|heading)\b[^>]*>(.*?)</(?:p|heading)>", window, re.S)
        caption = _text(prev[-1]) if prev else ""
        # Raw preceding text, tags stripped, regardless of element type. Bin-key
        # legends are frequently NOT in a <p>: US9656988 states
        # "IC 50 : A <= 10 nM; 10 nM < B <= 100 nM; ..." in the trailing content
        # of the PREVIOUS <tables> element, so a <p>-only look-back cannot see
        # it and the grades are unreadable without it.
        preceding = _text(window)[-700:]
        # ...and AFTER it. A bin key is as often a footnote as a caption:
        # US9656988 prints "IC 50 : A <= 10 nM; 10 nM < B <= 100 nM; ..."
        # immediately following TABLE-US-00001, serving both that table and the
        # next. Looking only backwards finds it for one table and misses it for
        # the other.
        following = _text(xml[tbl.end():tbl.end() + 1200])[:700]
        for tg in re.finditer(r"<tgroup\b([^>]*)>(.*?)</tgroup>", block, re.S):
            attrs, body = tg.group(1), tg.group(2)
            m = re.search(r'cols="(\d+)"', attrs)
            n_cols = int(m.group(1)) if m else 1
            head = re.search(r"<thead\b[^>]*>(.*?)</thead>", body, re.S)
            tbody = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", body, re.S)
            header_rows = [
                _parse_row(r) for r in
                re.findall(r"<row>(.*?)</row>", head.group(1), re.S)
            ] if head else []
            body_rows = [
                _parse_row(r) for r in
                re.findall(r"<row>(.*?)</row>", tbody.group(1) if tbody else body, re.S)
            ]
            out.append(Table(
                table_id=table_id, n_cols=n_cols,
                header_rows=header_rows, body_rows=body_rows,
                caption=caption, preceding=preceding, following=following,
            ))
    return out


# A cell that is really running prose — an NMR shift list, an MS trace, a
# chromatography method — rather than a column name. USPTO interleaves these as
# their own narrow tgroups between data rows, and they land in `thead` often
# enough that a naive "first header we see" picks one up.
_PROSE_CELL = re.compile(
    r"\(m,|\(d,|\(dd,|\(s,|\(t,|\bJ\s*=|\bppm\b|\bMS\s*:|\bM\s*\+\s*H|"
    r"\bRt\b|\bHPLC\b|\bChiral(?:cel|pak)\b|δ", re.I)


# A compound-identifier cell: "1", "12a", "A-7", "Ex. 203". Deliberately tight —
# it is used to tell a data row from an interleaved annotation row, so admitting
# prose would defeat the purpose.
_ID_CELL = re.compile(r"(?:(?:Ex(?:ample)?\.?|Cpd|Compound)\s*)?[A-Za-z]{0,3}[-–]?\d{1,5}[-–]?[a-z]?", re.I)


def _opens_with_id(cells: list["Cell"]) -> bool:
    texts = [c.text.strip() for c in cells if c.text.strip()]
    return bool(texts) and bool(_ID_CELL.fullmatch(texts[0]))


def _row_key(cells: list["Cell"]) -> tuple[str, ...]:
    """Identity of a row by its visible text, for de-duplicating repeats."""
    return tuple(c.text.strip() for c in cells)


def _is_namelike(cells: list["Cell"], *, declared: bool = False) -> bool:
    """Does this row look like column names rather than data or prose?

    `declared` means the row came out of a `<thead>`: the patent has already
    said it is a header, so the guesswork below is not ours to redo.
    """
    texts = [c.text.strip() for c in cells if c.text.strip()]
    if len(texts) < 2:
        return False
    if any(_PROSE_CELL.search(t) or len(t) > 60 for t in texts):
        return False
    # A fragment torn out of running prose — "diethylamine)", "4H)." — is short
    # and free of NMR keywords, so the checks above pass it. Harvesting headers
    # across every fragment width let these into the column names, and a header
    # reading "hERG] 4H)." is one the model cannot map a bin scale onto.
    # Unbalanced brackets are the tell: a real column label closes what it opens.
    #
    # ...unless the typesetter wrapped the label across rows, which is exactly
    # what a multi-row header IS. US11254686's last header row reads
    # `Compound | IC50 | ... | protein) | 10^6 cells)` — the `(` sits two rows
    # above — and rejecting it cost the names of columns 0 and 3 outright. So
    # this only polices rows we are PROMOTING out of a body; a row the source
    # marked as header is taken at its word.
    if not declared:
        for t in texts:
            for op, cl in (("(", ")"), ("[", "]")):
                if t.count(cl) > t.count(op):
                    return False
    # All-numeric rows are data, not headers.
    return not all(re.fullmatch(r"[\d.,;:<>=~\s-]+", t) for t in texts)


def assemble_block(tables: list["Table"], table_id: str) -> "Table | None":
    """Every tgroup of one `<tables>` block, stitched into the grid it encodes.

    USPTO does not put a logical table in one tgroup. It fragments it, and the
    fragmentation is not a fixed shape:

      * a 2-row title stub, then a 1-row header stub, then the data
        (US9656988 TABLE-US-00001 — the header says `BTK IC50`, the data tgroup
        has no header at all)
      * 141 alternating tgroups, a 6-col data row followed by a 3-col NMR
        annotation, per compound (US10172859 TABLE-US-00006)

    Picking one tgroup loses the rest. `max(..., len(body_rows))` — the previous
    behaviour — kept 72 of 570 rows on US10172859 and discarded the tgroup
    holding the column names on US9656988, which is why 830 correctly-binned
    records came back as `unnamed assay`.

    So: take the block's *modal width* (weighted by rows, not by tgroup count —
    a block can hold more annotation tgroups than data ones), concatenate the
    body of every tgroup at that width, and take the header from the first
    width-matching sibling that has one. Fragments at other widths are
    interleaved annotation and are dropped.

    This is the single view of a block. The detector and the repair path must
    both read through it — having two ways to count a block's rows is what
    produced five successive detector/applier mismatches.
    """
    same = [t for t in tables if t.table_id == table_id]
    if not same:
        return None

    # Which width carries the DATA is not the width with the most rows. On
    # US10172859 the interleaved NMR/MS annotations (3 cols, 314 rows) outnumber
    # the assay rows (6 cols, 123 rows) — weighting by row count picks the prose.
    # The discriminator that actually separates them is the compound id: an
    # assay row opens with one, an annotation row opens with "MS: 453.2 (M + H".
    # Nor is it a question of width alone. US8952177 TABLE-US-00003 interleaves
    # 4-col and 5-col tgroups and BOTH are data — the short rows are compounds
    # with a trailing value omitted. Selecting one width discards 190 of 359.
    # So classify each fragment by whether its rows open with a compound id, and
    # keep every fragment that does, at whatever width.
    kin, filled = [], 0
    for t in same:
        n = sum(1 for r in t.body_rows if any(c.text.strip() for c in r))
        if not n:
            continue
        filled += n
        if sum(1 for r in t.body_rows if _opens_with_id(r)) * 2 >= n:
            kin.append(t)
    if not filled:                               # header-only block
        return max(same, key=lambda t: len(t.body_rows))
    if kin:
        width = max(t.n_cols for t in kin)
    else:
        # No identifiers anywhere — fall back to the widest well-populated
        # fragment, which is the old behaviour and right for legend tables.
        rows_by_width: dict[int, int] = {}
        for t in same:
            rows_by_width[t.n_cols] = rows_by_width.get(t.n_cols, 0) + sum(
                1 for r in t.body_rows if any(c.text.strip() for c in r))
        width = max(rows_by_width, key=lambda w: (rows_by_width[w], w))
        kin = [t for t in same if t.n_cols == width]

    # A header split over several rows ("IC50 / DNA-PK") arrives as several
    # tgroups' theads, and a header repeated at each page break arrives again in
    # the body. Collect the distinct name-like rows across the block: that both
    # assembles the multi-row header and gives us the set to suppress below.
    #
    # Harvest from every sibling at the chosen width, not just the data ones. A
    # header fragment carries no compound ids, so it is never in `kin`; on
    # US9656988 the second header row `['#','IC50']` lives there, and without it
    # the column reads as "BTK" rather than "BTK IC50" and classifies as
    # non-assay — the whole table then yields nothing.
    header_rows: list[list[Cell]] = []
    seen_hdr: set[tuple[str, ...]] = set()
    data_ids = {id(t) for t in kin}
    # At ANY width, not just the chosen one: US8952177 TABLE-US-00003 publishes
    # its header as narrow fragments — one 2-cell row carrying both assay names,
    # continuation rows carrying the units — inside a 5-column block. Row width
    # is not column position, and `_choose_offsets` already aligns partial
    # header rows against the data; feeding it fewer rows is what loses columns.
    for t in same:
        rows = [(r, True) for r in t.header_rows]
        if id(t) not in data_ids:
            # a header fragment's "body" is header — but only by inference, so
            # those rows still face the full heuristic
            rows += [(r, False) for r in t.body_rows]
        for r, declared in rows:
            if not _is_namelike(r, declared=declared):
                continue
            k = _row_key(r)
            if k not in seen_hdr:
                seen_hdr.add(k)
                header_rows.append(r)
    if not header_rows:
        # No thead anywhere at this width. The header may have been demoted into
        # the body of an early fragment (US9656988: the `['Example #','BTK IC50']`
        # tgroup carries it as its only body row).
        for t in same:
            for r in t.header_rows + t.body_rows:
                # Same id guard as the promotion below: a graded data row is
                # name-like by shape, and claiming it as the header both loses
                # the row and mislabels every column after it.
                if _opens_with_id(r):
                    break
                if _is_namelike(r) and len({c.text.strip() for c in r if c.text.strip()}) > 1:
                    header_rows = [r]
                    seen_hdr.add(_row_key(r))
                    break
            if header_rows:
                break

    body: list[list[Cell]] = []
    for t in kin:
        for r in t.body_rows:
            if not any(c.text.strip() for c in r):
                continue
            if _row_key(r) in seen_hdr:          # header repeated at a page break
                continue
            body.append(r)

    # A multi-row header ("Compd / ID", "Btk / (IC50)") only ever has its first
    # row in `thead`; the continuation rows are demoted into the body. Promote
    # any name-like rows still leading the assembled body — they are header, and
    # left in place they parse as a compound called "ID" with a value of "(IC50)".
    # Stop at the first compound id. Without that guard a graded row like
    # ["1", "D"] reads as name-like — two short non-numeric cells — and the
    # promotion eats real data off the top of the table.
    while (len(header_rows) < 5 and body and _is_namelike(body[0])
           and not _opens_with_id(body[0])):
        header_rows.append(body.pop(0))

    first = kin[0]
    return Table(
        table_id=table_id, n_cols=width,
        header_rows=header_rows, body_rows=body,
        caption=first.caption, preceding=first.preceding,
        following=same[-1].following,
    )


def parse_fidelity(xml: str) -> list[dict]:
    """Blocks where our Cells do not account for the source's own elements.

    The one question the repair loop could never ask: **is what I am looking at
    what the patent says?** Every gap signal in this codebase is computed from
    the PARSED view, so a defect in the parser is invisible to all of them — it
    presents as a hard layout, and the loop dutifully buys rules to paper over
    damage we inflicted ourselves.

    That is not hypothetical. `_parse_row` matched `<entry/>` with the paired-tag
    branch, so every empty cell swallowed its neighbour: US11254686's nine-entry
    header rows arrived as five, US11613531 lost 440 measurements, and the loop
    spent three sessions proposing renames and bin keys for a regex bug. This
    check compares the two views directly and needs no model, no reference data
    and no judgement — `<entry>` elements in, Cells out, and they must match.

    It is the machine-checkable form of the rule this repo already states for
    humans: never diagnose from the parsed view.
    """
    parsed: dict[str, list[int]] = {}
    for t in parse_tables(xml):
        acc = parsed.setdefault(t.table_id, [0, 0])
        for r in t.header_rows + t.body_rows:
            acc[0] += len(r)
            acc[1] += 1

    out: list[dict] = []
    for tbl in re.finditer(r"<tables\b([^>]*)>(.*?)</tables>", xml, re.S):
        m = re.search(r'id="([^"]+)"', tbl.group(1) or "")
        table_id = m.group(1) if m else ""
        block = tbl.group(2)
        want_entries = len(re.findall(r"<entry\b", block))
        want_rows = len(re.findall(r"<row\b", block))
        got_entries, got_rows = parsed.get(table_id, [0, 0])
        if got_entries == want_entries and got_rows == want_rows:
            continue
        out.append({
            "table_id": table_id,
            "source_entries": want_entries, "parsed_cells": got_entries,
            "source_rows": want_rows, "parsed_rows": got_rows,
            "detail": (f"{table_id}: source declares {want_entries} <entry> in "
                       f"{want_rows} <row>, parser produced {got_entries} cells "
                       f"in {got_rows} rows — {want_entries - got_entries} cells "
                       f"lost before any extraction logic ran"),
        })
    return out


def assemble_blocks(tables: list["Table"]) -> list["Table"]:
    """`assemble_block` over every block, in document order."""
    seen: list[str] = []
    for t in tables:
        if t.table_id not in seen:
            seen.append(t.table_id)
    out = []
    for tid in seen:
        a = assemble_block(tables, tid)
        if a is not None:
            out.append(a)
    return out


# Retained name: several call sites ask for "the tgroup with the data" and now
# get the assembled block, which is what they always meant.
block_data_tgroup = assemble_block


def description_text(xml: str) -> str:
    """Plain description text, paragraph per line.

    Paragraphs keep their `<p id=...>` order, which is what the Example-N
    slicing logic downstream depends on.
    """
    m = re.search(r"<description\b[^>]*>(.*?)</description>", xml, re.S)
    if not m:
        return ""
    body = m.group(1)
    # Drop tables — they're handled structurally by parse_tables and would
    # otherwise flatten into unparseable runs of numbers in the prose.
    body = re.sub(r"<tables\b.*?</tables>", "\n", body, flags=re.S)
    paras = re.findall(r"<p\b[^>]*>(.*?)</p>", body, re.S)
    return "\n".join(t for t in (_text(p) for p in paras) if t)


# ── line-wrap repair ──────────────────────────────────────────────

# A cell that ends mid-name: trailing hyphen, or an open bracket/brace that the
# cell never closes. Patent typesetting breaks names at exactly these points.
_OPENERS = "([{"
_CLOSERS = ")]}"


def _unbalanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
    return depth != 0


def looks_wrapped(text: str) -> bool:
    """True if this cell appears to be the first half of a broken name."""
    t = text.rstrip()
    if not t:
        return False
    return t.endswith("-") or _unbalanced(t)


def join_wrapped_cells(cells: list[str]) -> list[str]:
    """Rejoin chemical names that USPTO split across consecutive cells.

    USPTO emits the typeset line break as a separate `<entry>`:

        "...1H-benzimidazol-2-"  +  "yl}cyclohexanecarboxylic acid,"
          →  "...1H-benzimidazol-2-yl}cyclohexanecarboxylic acid,"

    The join is purely structural — a trailing hyphen or an unclosed bracket
    means the name continues — so no model is needed to decide where to join.

    The hyphen is KEPT. In IUPAC nomenclature a hyphen before a locant or
    suffix is semantic (`benzimidazol-2-yl`), and the typesetter breaks *at* an
    existing hyphen rather than inserting one; dropping it yields
    `benzimidazol-2yl`, which OPSIN rejects. Where that assumption is wrong the
    ambiguity is genuine, so `join_candidates` hands both readings to OPSIN
    rather than guessing here.
    """
    out: list[str] = []
    buf = ""
    for cell in cells:
        piece = cell.strip()
        if not piece:
            if buf:
                out.append(buf)
                buf = ""
            continue
        buf = buf + piece if buf else piece
        if not looks_wrapped(buf):
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def join_candidates(name: str) -> list[str]:
    """Readings of a joined name, best-guess first, for OPSIN to adjudicate.

    Joining is structural but not always unambiguous: a hyphen at the break
    point is usually part of the name, occasionally a typesetter's artifact.
    Rather than pick, emit both and let the caller accept whichever OPSIN
    parses — the parser is a far better oracle than a heuristic, and this is
    the pattern the IUPAC cascade already relies on.
    """
    cands = [name]
    # The alternative reading: the hyphen was a break artifact, not nomenclature.
    for m in re.finditer(r"-", name):
        i = m.start()
        if 0 < i < len(name) - 1 and name[i + 1].isalpha():
            alt = name[:i] + name[i + 1:]
            if alt not in cands:
                cands.append(alt)
            break
    return cands


def is_available(patent_id: str) -> bool:
    """Cheap check used by the source cascade to decide whether to try tier 1."""
    if (_CACHE_DIR / f"{patent_id.upper()}.xml").exists():
        return True
    return _normalize_patent_number(patent_id) is not None and bool(_api_key())
