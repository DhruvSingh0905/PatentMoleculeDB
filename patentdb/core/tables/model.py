"""Canonical table + measurement data model (BLUEPRINT L1).

The whole point: capture the table grid the source ALREADY has — header row +
cells, verbatim — plus provenance, so no downstream stage has to re-derive
structure. Frozen dataclasses → deterministic, hashable, diff-friendly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cell:
    """One grid cell, verbatim text + its position."""
    text: str
    row: int
    col: int


@dataclass(frozen=True)
class Table:
    """A parsed table grid, structure preserved.

    `header` is the verbatim column-label row (NOT yet resolved to assays — that
    is L3's job). `rows` are the data rows as verbatim cell strings. One n-up
    block = one Table (the adapter splits repeated-header layouts).
    """
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    source: str                       # "mineru" | "gp_text"
    caption: str = ""                 # title/caption text (assay context)
    table_label: str = ""             # e.g. "TABLE 8"
    page: int | None = None
    char_span: tuple[int, int] | None = None

    @property
    def n_cols(self) -> int:
        return len(self.header)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def column(self, idx: int) -> list[str]:
        return [r[idx] if idx < len(r) else "" for r in self.rows]


@dataclass(frozen=True)
class AssayMeasurement:
    """A single resolved measurement (BLUEPRINT L4 output).

    `encoding`: "numeric" (a point value), "range" (letter-grade / bin → low/high),
    or "grade" (qualitative only). `provenance` records exactly how this was
    derived so any record is auditable back to its source cell + resolution path.
    """
    cid: str
    assay: str                        # canonical name OR verbatim header (if unresolved)
    value_raw: str
    unit: str = ""
    value_numeric: float | None = None
    value_low: float | None = None
    value_high: float | None = None
    qualifier: str | None = None
    encoding: str = "numeric"
    provenance: dict = field(default_factory=dict)
    # provenance keys: source, table_label, page, char_span,
    #                  header_verbatim, resolution ("vocab"|"legend"|"llm_resolved"|"verbatim")
