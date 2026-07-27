"""Deterministic table-extraction layer (BLUEPRINT L1-L6).

Replaces the heuristic assay FSM: parse the table grid the source already
provides (never re-derive it), bind columns to assays via a frozen vocab +
a confidence-gated resolution memory, and emit fully-provenanced records.
"""
from .model import AssayMeasurement, Cell, Table

__all__ = ["Table", "Cell", "AssayMeasurement"]
