"""MinerU markdown → Table grid (BLUEPRINT L1, deterministic, no LLM).

MinerU/PaddleX emits real `<table><tr><td>…</td></tr></table>` grids (no `<th>`;
the header is the first non-title `<tr>`), preceded by `<|ref|>table_caption…`
layout annotations. Some tables are "n-up" — the column block repeats side by
side (`Example no. | IC50 | <junk> | Example no. | IC50`) with margin line-numbers
in the gap column. We split those into one Table per block; deciding which column
is the id vs a value vs junk is L3's job, not ours.
"""
from __future__ import annotations

import html
import re

from .model import Table

_TABLE_RE = re.compile(r"<table>(.*?)</table>", re.S | re.I)
_TR_RE = re.compile(r"<tr>(.*?)</tr>", re.S | re.I)
_TD_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
# MinerU layout annotation carrying a table's caption text on the next line(s).
_CAPTION_RE = re.compile(
    r"<\|ref\|>table_caption<\|/ref\|><\|det\|>.*?<\|/det\|>\s*\n([^\n]+)", re.S
)
# A "TABLE 8" / "Table 12A" / "TABLE IV" label (no trailing "-continued").
_LABEL_RE = re.compile(r"\bTABLE\s+(?:\d+[A-Z]?|[IVXLCM]+)\b", re.I)


def _delatex(s: str) -> str:
    """Render MinerU's inline LaTeX/MathML to plain text.
    `$\\mathrm{LTB}_4 \\mathrm{IC}_{50}(\\mu\\mathrm{M})$` → `LTB4 IC50 (µM)`."""
    if "\\" not in s and "$" not in s and "_" not in s:
        return s
    s = s.replace("$", "")
    s = re.sub(r"\\(?:mathrm|mathit|text|mathbf)\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mu\b", "µ", s)
    s = re.sub(r"_\{?([A-Za-z0-9]+)\}?", r"\1", s)      # subscripts: _{50} → 50
    s = re.sub(r"\^\{?([A-Za-z0-9]+)\}?", r"\1", s)      # superscripts
    s = re.sub(r"\\[a-zA-Z]+", " ", s)                   # drop other commands
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


# A cell that is a single numeric value with a European decimal comma
# (e.g. "0,0079", "<0,0012", "1,36 (2)") — NOT a comma-separated list.
_VALCOMMA = re.compile(r"^[<>≤≥~]?\s*\d+,\d+\s*(?:\([^)]*\))?$")


def _fix_comma(s: str) -> str:
    return s.replace(",", ".") if _VALCOMMA.match(s) else s


def _unspace_kinds(s: str) -> str:
    """Collapse OCR char-spacing in assay kinds: "I C _ 5 0" → "IC50", "K i" → "Ki".
    Otherwise the kind regex (and classification) silently miss the column."""
    s = re.sub(r"\b([IEAGCX])\s*[_·]?\s*C\s*[_·]?\s*(\d)\s*[_·]?\s*(\d)\b",
               r"\1C\2\3", s, flags=re.I)
    s = re.sub(r"\bK\s*[_·]?\s*([id])\b", r"K\1", s, flags=re.I)
    return s


def _clean(s: str) -> str:
    """Strip tags, decode HTML entities, de-LaTeX, collapse OCR-spaced kinds, fix commas."""
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(html.unescape(s))           # "&amp;gt;2" → "&gt;2" → ">2"
    s = re.sub(r"\s+", " ", s).strip()
    return _fix_comma(_unspace_kinds(_delatex(s)))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# A first-cell that is a compound id: optional short prefix, then digits.
# "131", "A28", "Cpd. 5" → cid;  "Example no.", "IC50 [μM]", "" → NOT a cid.
_CID_CELL = re.compile(
    r"^(?:cpd\.?|compound|example|ex\.?|no\.?)?\s*[A-Za-z]{0,3}-?\d{1,5}[A-Za-z]{0,3}$",
    re.I,
)


def _is_cid_cell(s: str) -> bool:
    s = s.strip()
    return bool(s) and bool(_CID_CELL.match(s))


# A cell carrying a real value: a decimal, or scientific notation.
_DEC = re.compile(r"\d\.\d|\d\s*[eE][+-]?\d")


def _has_value(row: list[str]) -> bool:
    return any(_DEC.search(c) for c in row)


# A value bled onto the end of a header cell after the unit/paren:
# "P2X3 IC50 (μM) 0.025" → "P2X3 IC50 (μM)".  Leaves names like "Nav1.6" alone.
_BLED = re.compile(r"(\)|\]|n[mM]|[µμu][mM])\s+[<>≤≥~]?\s*[\d.]+\s*$")


def _strip_bled(h: str) -> str:
    return _BLED.sub(r"\1", h).strip()


def _nup_blocks(header: list[str]) -> list[tuple[int, int]]:
    """Column ranges for n-up layouts. If the first header label recurs, each
    recurrence starts a new block; otherwise the whole row is one block."""
    if not header:
        return [(0, 0)]
    first = _norm(header[0])
    if not first:
        return [(0, len(header))]
    starts = [i for i, h in enumerate(header) if _norm(h) == first]
    if len(starts) < 2:
        return [(0, len(header))]
    starts.append(len(header))
    return [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]


def _split(rows: list[list[str]], *, caption, label, page, span, source) -> list[Table]:
    # 1. Title rows: leading rows where only the first cell has text (spanning title).
    title_parts: list[str] = []
    body_start = 0
    for r in rows:
        nonempty = [c for c in r if c]
        if len(nonempty) == 1 and r and r[0]:
            title_parts.append(r[0])
            body_start += 1
        else:
            break
    body = rows[body_start:]
    if not body:
        return []

    # 2. Header rows = leading rows whose FIRST cell is NOT a compound id; data
    #    begins at the first cid-led row. Handles multi-row headers (MinerU often
    #    splits "FLAP Binding wild" / "type HTRF Ki (µM)" across two <tr>s).
    h_end = 0
    for r in body:
        # Data begins at the first row that is cid-led OR carries a real value.
        # The value check stops a value row (whose cid OCR'd into col0 as a blank
        # or a structure fragment) from being swept into the header.
        if r and (_is_cid_cell(r[0]) or _has_value(r)):
            break
        h_end += 1
    if h_end >= len(body):                 # no cid-led row (abbreviation / R-group /
        h_end = 1                          # property table): keep it, row0=header,
                                           # rest=data — L2 classifies, L1 never drops
    header_rows, data = body[:h_end], body[h_end:]
    if not data:
        return []
    width = max(len(r) for r in body)
    if header_rows:                        # merge multi-row header column-wise
        header = [
            _strip_bled(" ".join(hr[c].strip() for hr in header_rows
                                 if c < len(hr) and hr[c].strip()))
            for c in range(width)
        ]
    else:                                  # headerless table (row0 was already data)
        header = [""] * width

    full_caption = " ".join(p for p in [caption, *title_parts] if p).strip()
    if not label:
        m = _LABEL_RE.search(full_caption)
        label = m.group(0) if m else ""

    out: list[Table] = []
    for lo, hi in _nup_blocks(header):
        h = tuple(header[lo:hi])
        if not any(c.strip() for c in h):
            continue
        block_rows = []
        for r in data:
            cells = tuple(r[lo:hi]) if len(r) >= hi else tuple(r[lo:]) + ("",) * (hi - len(r))
            if any(c.strip() for c in cells):
                block_rows.append(cells)
        if block_rows:
            out.append(Table(
                header=h, rows=tuple(block_rows), source=source,
                caption=full_caption, table_label=label, page=page, char_span=span,
            ))
    return out


def parse_mineru_page(md_text: str, *, page: int | None = None,
                      source: str = "mineru") -> list[Table]:
    """Parse every `<table>` on a MinerU page into Table objects."""
    out: list[Table] = []
    for m in _TABLE_RE.finditer(md_text):
        rows = [
            [_clean(td) for td in _TD_RE.findall(tr.group(1))]
            for tr in _TR_RE.finditer(m.group(1))
        ]
        rows = [r for r in rows if r]
        if not rows:
            continue
        # nearest caption annotation before this table
        caption = ""
        cm = list(_CAPTION_RE.finditer(md_text[:m.start()]))
        if cm:
            caption = _clean(cm[-1].group(1))
        out.extend(_split(
            rows, caption=caption, label="", page=page,
            span=(m.start(), m.end()), source=source,
        ))
    return out
