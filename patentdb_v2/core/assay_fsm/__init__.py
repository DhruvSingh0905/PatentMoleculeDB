"""Generic assay-table extraction pipeline.

A patent-agnostic pipeline built around a vocabulary-driven FSM, with a
fingerprint-cached LLM realigner behind `config.ASSAY_REALIGN_ENABLED`
(default OFF). Replaces the legacy hardcoded-header + regex extraction in
`routes/google_assays.py`.

Architecture (see plan: ~/.claude/plans/this-was-the-entire-synthetic-iverson.md):

  Stage 0 — Universal text normalization (NFKC + HTML unescape + mojibake)
  Stage 1 — Region detection (assay_keyword + numeric density + unit annotation)
  Stage 2 — Vocabulary-driven FSM tokenizer
  Stage 3 — Header understanding (folded into Stage 5 LLM call)
  Stage 4 — Token-stream row aligner
  Stage 5 — LLM realigner (fingerprint-cached), OFF unless ASSAY_REALIGN=1
  Stage 6 — Vocabulary auto-extension (≥3-fingerprint gate)

Hard rules (enforced by `tests/test_assay_fsm_us8952177.py`):
  - NO patent-specific identifiers in code or vocabulary JSON
  - LLM-discovered tokens promote to runtime only at ≥3 distinct fingerprints

Stage 5 used to be listed here as ALWAYS-FIRE, with "LLM ALWAYS fires on
cache miss (no structural-failure pre-flight)" as a hard rule. It was the
right rule for OCR input and stopped being one when OCR left the tree in
`43d037e`; `tests/test_realign_gate.py` carries the measurement that
replaced it.
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
