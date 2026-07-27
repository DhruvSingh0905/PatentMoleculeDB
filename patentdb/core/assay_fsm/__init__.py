"""Generic assay-table extraction pipeline.

A patent-agnostic pipeline that combines a vocabulary-driven FSM with
an always-fire (fingerprint-cached) LLM realigner. Replaces the
legacy hardcoded-header + regex extraction in `routes/google_assays.py`.

Architecture (see plan: ~/.claude/plans/this-was-the-entire-synthetic-iverson.md):

  Stage 0 — Universal text normalization (NFKC + HTML unescape + mojibake)
  Stage 1 — Region detection (assay_keyword + numeric density + unit annotation)
  Stage 2 — Vocabulary-driven FSM tokenizer
  Stage 3 — Header understanding (folded into Stage 5 LLM call)
  Stage 4 — Token-stream row aligner
  Stage 5 — ALWAYS-FIRE LLM realigner (fingerprint-cached)
  Stage 6 — Vocabulary auto-extension (≥3-fingerprint gate)

Hard rules (enforced by `tests/test_assay_fsm_us8952177.py`):
  - NO patent-specific identifiers in code or vocabulary JSON
  - LLM ALWAYS fires on cache miss (no structural-failure pre-flight)
  - LLM-discovered tokens promote to runtime only at ≥3 distinct fingerprints
"""
from __future__ import annotations

# Public API surface — keep this minimal and stable.
from .pipeline import extract_page, extract_for_patent
from .vocabulary import AssayVocabulary, TokenClass
from .tokenizer import Token, TokenStream

__all__ = [
    "extract_page",
    "extract_for_patent",
    "AssayVocabulary",
    "Token",
    "TokenStream",
    "TokenClass",
]
