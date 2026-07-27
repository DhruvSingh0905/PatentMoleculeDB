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
import re
from dataclasses import dataclass, field

from ..sources.uspto_assays import (
    _CID_PAT, ASSAY, CID, build_columns, table_legend,
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
    headers: list[str] = field(default_factory=list)
    column_kinds: list[str] = field(default_factory=list)

    @property
    def severity(self) -> int:
        """Rows we are leaving on the floor — used to spend budget where it pays."""
        return max(0, self.n_data_rows - self.n_extracted)


def layout_fingerprint(table: Table, headers: list[str]) -> str:
    """Stable id for a table SHAPE, deliberately independent of its content.

    Two tables share a fingerprint when a rule written for one should work on
    the other: same column count, same per-column value shape, same normalised
    header words. Compound ids, patent numbers and the actual measurements are
    excluded — otherwise every table is unique and nothing is ever reused.
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
    raw = f"{table.n_cols}|{','.join(shapes)}|{','.join(words)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _sample_of(table: Table, headers: list[str], max_rows: int = 4) -> str:
    """A compact, readable rendering of the table's shape.

    Kept small on purpose: this is what a paid model reads. Header, a few data
    rows, and the legend if there is one — enough to infer a rule, not enough
    to be expensive.
    """
    _, data = _header_rows_of(table)
    lines = []
    legend = table_legend(table)
    if legend:
        lines.append(f"LEGEND: {legend[:220]}")
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
    lines.append(f"COLUMNS: {table.n_cols} (indices 0..{table.n_cols - 1})")
    if any(headers):
        lines.append("HEADER: " + " | ".join(
            f"[{i}] {h[:36]}" if h else f"[{i}] (blank)" for i, h in enumerate(headers)))
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


def find_gaps(patent_id: str, tables: list[Table], extracted_by_table: dict[str, int],
              *, min_rows: int = 5, max_read_fraction: float = 0.5) -> list[Gap]:
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

    gaps: list[Gap] = []
    seen_blocks: set[str] = set()
    for t in tables:
        hdr_rows, data = _header_rows_of(t)
        rows = [r for r in data if any(c.text.strip() for c in r)]
        if len(rows) < min_rows:
            continue
        # One gap per block, raised from its largest tgroup.
        if t.table_id in seen_blocks:
            continue
        block_rows = rows_per_block.get(t.table_id, len(rows))
        headers = merge_header(t, hdr_rows)
        if any(headers):
            inherit_by_width[t.n_cols] = headers
            inherit_by_block[t.table_id] = headers
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
            if total_cells and unread_cells >= max(3, total_cells * 0.05):
                gaps.append(Gap(
                    patent_id=patent_id, table_id=t.table_id, n_cols=t.n_cols,
                    n_data_rows=block_rows, n_extracted=got,
                    fingerprint=layout_fingerprint(t, headers),
                    reason=(f"{unread_cells} of {total_cells} populated assay cells "
                            f"cannot be parsed as a value (e.g. {example!r})"),
                    sample=_sample_of(t, headers), headers=headers,
                    column_kinds=kinds,
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
        if got and got / max(block_rows, 1) > max_read_fraction:
            seen_blocks.add(t.table_id)
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
            reason=reason,
            sample=_sample_of(t, headers),
            headers=headers,
            column_kinds=kinds,
        ))
    gaps.sort(key=lambda g: -g.severity)
    return gaps


def gaps_for_patent(patent_id: str, xml: str) -> list[Gap]:
    """Convenience wrapper: parse, extract, and report what was left behind."""
    from collections import Counter

    from ..sources.uspto_assays import extract_from_patent
    from ..sources.uspto_xml import parse_tables
    tables = parse_tables(xml)
    per_table = Counter(r.table_id for r in extract_from_patent(xml))
    return find_gaps(patent_id, tables, dict(per_table))
