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
    for m in re.finditer(r"<entry\b([^>]*)>(.*?)</entry>|<entry\b([^>]*)/>",
                         row_xml, re.S):
        attrs = m.group(1) or m.group(3) or ""
        body = m.group(2) or ""
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
                caption=caption,
            ))
    return out


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
