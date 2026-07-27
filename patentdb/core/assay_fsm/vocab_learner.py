"""Stage 6 — Vocabulary auto-extension.

After the cross_validator merges FSM and LLM outputs, this stage
mines the LLM result for tokens the FSM didn't know about.

The discovery rule: any non-numeric, non-whitespace substring the
LLM put in `qualifiers` or used as a column name must be representable
by the FSM's token vocabulary. If a substring isn't, it's a candidate
new vocabulary entry.

Categorization is heuristic:
  - Single non-alpha character next to a number → QUALIFIER candidate
  - Lowercase 1-3 char string in null-marker position → NULL_MARKER candidate
  - Title-case string in column header → ASSAY_TYPE candidate
  - Unit-like string (`<unit>`-shaped) → UNIT candidate

Discoveries are appended to `assay_vocabulary.discoveries.json` with
provenance. The ≥3-fingerprint gate (in `vocabulary.py`) prevents
single-shot LLM hallucinations from polluting runtime tokenization.

Patent-agnostic — no patent ids, target names, or compound ids ever
flow into the discoveries file.
"""
from __future__ import annotations

import logging
import re

from ..adaptive_extraction_cache import RealignedTable
from .vocabulary import AssayVocabulary, TokenClass

logger = logging.getLogger(__name__)


# Surface forms we KNOW are legitimate non-vocab tokens — we don't
# want to "discover" the digit 9 as a new qualifier. Numbers and
# punctuation that have no semantic meaning in the assay grammar.
_BORING = re.compile(r"^[\d\s.,;:!?\(\)\[\]\{\}\"'\\/]+$")


def learn_from_llm_output(
    realigned: RealignedTable,
    *,
    vocab: AssayVocabulary,
    fingerprint: str,
    patent_id: str,
) -> int:
    """Scan the LLM realigner output for tokens the FSM doesn't know.

    Args:
        realigned: The LLM result for a region.
        vocab: Current loaded vocab (knows what's already canonical).
        fingerprint: Cache fingerprint of the region.
        patent_id: Source patent (for provenance).

    Returns:
        Number of NEW or UPDATED discoveries appended.
    """
    if realigned is None or not realigned.rows:
        return 0

    n_added = 0

    # 1. Qualifier discovery — any non-empty qualifier value should be
    #    a known QUALIFIER surface form.
    seen_qualifiers: set[str] = set()
    for row in realigned.rows:
        for q in (row.qualifiers or {}).values():
            if not q:
                continue
            seen_qualifiers.add(str(q))
    for q in seen_qualifiers:
        if _is_known_qualifier(q, vocab):
            continue
        if _BORING.match(q):
            continue
        # Generate a safe lemma slug from the surface
        lemma = _safe_lemma("q_" + q)
        vocab.add_discovery(
            lemma=lemma,
            klass=TokenClass.QUALIFIER,
            surface=q,
            fingerprint=fingerprint,
            patent_id=patent_id,
        )
        n_added += 1

    # 2. Assay type discovery — column names the FSM tokenizer would
    #    not classify as ASSAY_TYPE. We only flag the canonical-form
    #    tokens (no spaces, alphanumeric only) — phrasal column headers
    #    are handled by the LLM at extraction time and don't need to
    #    enter the FSM vocab.
    seen_assay_names: set[str] = set()
    for row in realigned.rows:
        for name in (row.values or {}).keys():
            seen_assay_names.add(str(name))
    for name in seen_assay_names:
        candidate = _maybe_assay_type_candidate(name)
        if not candidate:
            continue
        if _is_known_assay_type(candidate, vocab):
            continue
        vocab.add_discovery(
            lemma=_safe_lemma("at_" + candidate),
            klass=TokenClass.ASSAY_TYPE,
            surface=candidate,
            fingerprint=fingerprint,
            patent_id=patent_id,
        )
        n_added += 1

    # 3. Unit discovery — strings in `realigned.units.values()` that
    #    don't match a known UNIT surface form.
    for u in (realigned.units or {}).values():
        if not u:
            continue
        if _is_known_unit(u, vocab):
            continue
        vocab.add_discovery(
            lemma=_safe_lemma("u_" + u),
            klass=TokenClass.UNIT,
            surface=u,
            fingerprint=fingerprint,
            patent_id=patent_id,
        )
        n_added += 1

    if n_added:
        logger.info(
            "vocab_learner: %d discovery candidates appended for %s (fp=%s)",
            n_added, patent_id, fingerprint[:8],
        )
    return n_added


# ── Helpers ──────────────────────────────────────────────────────


def _is_known_qualifier(s: str, vocab: AssayVocabulary) -> bool:
    s_norm = s.strip()
    for e in vocab.entries_of_class(TokenClass.QUALIFIER):
        if s_norm in e.surface_forms or s_norm == e.maps_to:
            return True
    return False


def _is_known_assay_type(s: str, vocab: AssayVocabulary) -> bool:
    s_norm = s.strip().lower()
    for e in vocab.entries_of_class(TokenClass.ASSAY_TYPE):
        if any(s_norm == sf.lower() for sf in e.surface_forms):
            return True
        if e.canonical and s_norm == e.canonical.lower():
            return True
    return False


def _is_known_unit(s: str, vocab: AssayVocabulary) -> bool:
    s_norm = s.strip()
    for e in vocab.entries_of_class(TokenClass.UNIT):
        if s_norm in e.surface_forms or s_norm == e.canonical:
            return True
    return False


def _maybe_assay_type_candidate(name: str) -> str | None:
    """Extract an ASSAY_TYPE-shaped substring from a (possibly phrasal)
    column header. Returns None if the name doesn't contain one.

    Heuristic: look for `(I|E|A|X|G|C)C\d{2,3}` or `K[id]` patterns.
    """
    m = re.search(r"\b((?:[IEAXGC]C\d{2,3}|K[id])(?:_?app)?)\b", name)
    return m.group(1) if m else None


def _safe_lemma(s: str) -> str:
    """Convert an arbitrary surface to a safe identifier.

    Lemmas should be simple ascii identifiers so they're easy to
    grep / reference in the discoveries file.
    """
    out = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return out.strip("_").lower() or "unknown"
