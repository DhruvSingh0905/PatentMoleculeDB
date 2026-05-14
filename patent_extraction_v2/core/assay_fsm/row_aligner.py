"""Stage 4 — Token-stream → typed AssayResult rows.

Walks the Token stream from the tokenizer and emits one `AlignedRow`
per detected compound row. A row consists of:
  - a `compound_id` (the first integer-valued NUMBER on the row)
  - a list of value cells, each with (value, qualifier, unit, n_runs)

The aligner does NOT try to map cells to canonical assay names —
that's Stage 3 (header understanding via LLM). The aligner just
groups tokens into row buckets.

Algorithm:
  1. Drop UNKNOWN tokens (prose noise) and lone whitespace.
  2. Walk tokens left-to-right tracking a "row buffer".
  3. A new row starts when we see a NUMBER with integer value AND
     either:
       - the previous token was a ROW_BREAK or COMPOUND_ID_PREFIX,
       - OR the row buffer is empty,
       - OR the row buffer has already accumulated ≥2 value cells
         (at which point we expect the next bare integer to be the
         next row's compound id).
  4. NULL_MARKER tokens become "no-value" cells inside the current row.
  5. ROW_BREAK flushes the current row.

Patent-agnostic — operates purely on token classes, no header-name
or compound-id knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tokenizer import Token, TokenStream
from .vocabulary import TokenClass


@dataclass
class ValueCell:
    """One cell in an aligned row — a value or a null marker."""
    value: float | None = None
    qualifier: str | None = None
    n_runs: int | None = None
    raw: str = ""           # the surface text the token had
    is_null: bool = False   # True if this came from a NULL_MARKER


@dataclass
class AlignedRow:
    """One compound's row, with value cells in column order."""
    compound_id: str
    cells: list[ValueCell] = field(default_factory=list)
    span: tuple[int, int] = (0, 0)


def align_rows(stream: TokenStream, *, expected_n_columns: int | None = None) -> list[AlignedRow]:
    """Walk a token stream and emit one AlignedRow per compound row.

    Args:
        stream: The TokenStream from `tokenizer.tokenize(...)`.
        expected_n_columns: Optional hint — when set, the aligner will
            try to flush a row after collecting this many value cells
            (helps disambiguate where one row ends and the next begins
            when ROW_BREAK signals are noisy or absent in plain-text
            tables). When None, relies on row-break heuristics only.

    Returns:
        List of AlignedRow. Empty list is a valid result.
    """
    rows: list[AlignedRow] = []
    cur_id: str | None = None
    cur_cells: list[ValueCell] = []
    cur_span_start: int | None = None
    cur_span_end: int | None = None

    def _flush() -> None:
        nonlocal cur_id, cur_cells, cur_span_start, cur_span_end
        if cur_id is not None and cur_cells:
            rows.append(AlignedRow(
                compound_id=cur_id,
                cells=cur_cells,
                span=(cur_span_start or 0, cur_span_end or 0),
            ))
        cur_id = None
        cur_cells = []
        cur_span_start = None
        cur_span_end = None

    tokens = [t for t in stream.tokens if t.klass != TokenClass.UNKNOWN]
    i = 0
    while i < len(tokens):
        t = tokens[i]

        # ROW_BREAK is a strong signal — flush whatever we have
        if t.klass == TokenClass.ROW_BREAK:
            _flush()
            i += 1
            continue

        # COMPOUND_ID_PREFIX: skip; the next NUMBER is the cpd_id
        if t.klass == TokenClass.COMPOUND_ID_PREFIX:
            i += 1
            continue

        # If we don't have a current id yet, look for one
        if cur_id is None:
            cid = _maybe_compound_id(t)
            if cid is not None:
                cur_id = cid
                cur_span_start = t.span[0]
                cur_span_end = t.span[1]
                i += 1
                continue
            # Skip orphan tokens until we find a row-starter
            i += 1
            continue

        # We have a cur_id — collect value cells
        if t.klass == TokenClass.NUMBER:
            cur_cells.append(ValueCell(
                value=t.value,
                qualifier=t.qualifier,
                n_runs=t.n_runs,
                raw=t.surface,
            ))
            cur_span_end = t.span[1]
            i += 1
            # If we hit the expected column count AND the next token
            # looks like a new row's compound id, flush.
            if expected_n_columns is not None and len(cur_cells) >= expected_n_columns:
                if i < len(tokens):
                    nxt = tokens[i]
                    if nxt.klass == TokenClass.ROW_BREAK or _maybe_compound_id(nxt) is not None:
                        _flush()
            continue

        if t.klass == TokenClass.NULL_MARKER:
            cur_cells.append(ValueCell(is_null=True, raw=t.surface))
            cur_span_end = t.span[1]
            i += 1
            if expected_n_columns is not None and len(cur_cells) >= expected_n_columns:
                if i < len(tokens):
                    nxt = tokens[i]
                    if nxt.klass == TokenClass.ROW_BREAK or _maybe_compound_id(nxt) is not None:
                        _flush()
            continue

        # ASSAY_TYPE / UNIT in the middle of a row stream — usually means
        # we crossed into a header section; flush the row.
        if t.klass in (TokenClass.ASSAY_TYPE, TokenClass.UNIT):
            _flush()
            i += 1
            continue

        # Fallback: skip unrecognized tokens
        i += 1

    _flush()
    return rows


def _maybe_compound_id(t: Token) -> str | None:
    """Return a compound-id string if this token looks like one.

    Compound IDs come in two flavors:
      - explicit COMPOUND_ID class (from the generic regex pattern)
      - integer-valued NUMBER tokens with no qualifier and no n_runs
        (these are bare digits that COULD be values OR cpd_ids — the
         row aligner uses position to decide)
    """
    if t.klass == TokenClass.COMPOUND_ID:
        return t.surface.strip()
    if t.klass == TokenClass.NUMBER:
        # Bare digits with no qualifier, no n_runs, integer-valued, ≤4 digits
        if (
            t.qualifier is None
            and t.n_runs is None
            and t.value is not None
            and t.value.is_integer()
            and 0 < t.value < 10000
        ):
            # Reject decimals masquerading as integers (e.g. "1.0" → 1.0)
            if "." not in t.surface:
                return str(int(t.value))
    return None


def detect_n_columns(rows: list[AlignedRow]) -> int | None:
    """Estimate the number of value columns from a sample of rows.

    Use the median row length over the first 20 rows. The row aligner
    can be re-run with this as `expected_n_columns` for sharper
    boundaries.
    """
    if not rows:
        return None
    lengths = sorted(len(r.cells) for r in rows[:20])
    if not lengths:
        return None
    return lengths[len(lengths) // 2]
