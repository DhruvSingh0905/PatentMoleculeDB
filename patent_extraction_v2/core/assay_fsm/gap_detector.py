"""Tier 2 — Coverage-gap detector.

After the cheap pipeline (FSM + cached LLM realigner) finishes, we
score whether HARVEST burst should fire. The signals are structural
(counts, ratios) — patent-agnostic.

Signals (any one is enough):
  1. zero_output_with_signal — pipeline emitted nothing for the page,
     but the page contains assay-keyword evidence + numeric/grade
     evidence. Strongest signal that we missed real data.
  2. all_llm_chunks_empty — the LLM realigner fired one or more times
     but every chunk returned 0 rows. Suggests the format isn't one
     the structured-JSON prompt handles.
  3. low_compound_density — extracted compound count is much lower
     than the count of compound-id-shaped tokens in the patent text
     adjacent to assay keywords. Suggests under-extraction.

All thresholds are tunable. Initial values calibrated so known
table-primary patents (where the cheap pipeline already works) DO
NOT trigger the burst, while patents whose data is in unrecognized
formats (letter-grade tables, prose-only, Markush enumeration only)
DO trigger it. The burst is the safety net, not the default.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


_ASSAY_KEYWORD_RE = re.compile(
    r"\b(?:IC[\s_-]?50|EC[\s_-]?50|CC[\s_-]?50|AC[\s_-]?50|XC[\s_-]?50"
    r"|GI[\s_-]?50|MIC[\s_-]?50?|IC[\s_-]?90|EC[\s_-]?90"
    r"|K_?[id]|pIC[\s_-]?50|pK_?[id])\b",
    re.IGNORECASE,
)

_VALUE_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?\s*(?:nM|μM|uM|µM|mcM|mM|pM)(?![A-Za-z])",
    re.IGNORECASE,
)

_LETTER_GRADE_ROW_RE = re.compile(
    r"(?:^|[\s\n])(\d{1,4})\.?\s+([A-E])(?=\s|$|\n)",
    re.MULTILINE,
)

# Compound-id-shape near assay keyword. Used to estimate "expected
# compound count" — every patent that lists 100 compound IDs near
# assay keywords ought to have 100 extracted compounds.
_COMPOUND_NEAR_ASSAY_RE = re.compile(
    r"(?:Cpd\.?\s*(?:No\.?)?|Compound|Example|Cmp\.?\s*(?:No\.?)?|Ex\.?)\s*"
    r"(\d{1,4}[A-Za-z]{0,3})",
    re.IGNORECASE,
)


@dataclass
class GapScore:
    fire_burst: bool = False
    signals: list[str] = field(default_factory=list)
    n_assay_keywords: int = 0
    n_value_unit_hits: int = 0
    n_letter_grade_hits: int = 0
    n_compound_id_mentions: int = 0
    n_compounds_extracted: int = 0
    empty_llm_chunk_rate: float = 0.0


def score(
    text: str,
    fsm_diag: dict,
    fsm_results: dict,
    *,
    min_compound_id_mentions_for_density_signal: int = 30,
    density_threshold: float = 0.4,
    empty_chunk_rate_threshold: float = 0.95,
) -> GapScore:
    """Return a GapScore. `fire_burst=True` means the HARVEST burst
    should fire for this patent."""
    out = GapScore()
    if not text:
        return out

    out.n_assay_keywords = len(list(_ASSAY_KEYWORD_RE.finditer(text)))
    out.n_value_unit_hits = len(list(_VALUE_UNIT_RE.finditer(text)))
    out.n_letter_grade_hits = len(list(_LETTER_GRADE_ROW_RE.finditer(text)))
    out.n_compound_id_mentions = len(set(
        m.group(1).lower() for m in _COMPOUND_NEAR_ASSAY_RE.finditer(text)
    ))
    out.n_compounds_extracted = len(fsm_results)

    # Signal 1: zero output + signal present
    has_signal = (
        out.n_assay_keywords > 0
        and (out.n_value_unit_hits >= 3 or out.n_letter_grade_hits >= 3)
    )
    if not fsm_results and has_signal:
        out.signals.append("zero_output_with_signal")

    # Signal 2: every LLM chunk returned empty
    n_chunks = fsm_diag.get("n_llm_chunks", 0) or 0
    n_compounds = fsm_diag.get("n_compounds", out.n_compounds_extracted) or 0
    if n_chunks > 0:
        # Rough heuristic: an LLM chunk should produce at least 1
        # compound on average if the data is there. We treat a chunk
        # as "empty" if the average compound-yield is <0.5/chunk —
        # that's the threshold below which we conclude the LLM
        # systematically failed.
        avg_per_chunk = n_compounds / max(1, n_chunks)
        if avg_per_chunk < 0.5:
            out.empty_llm_chunk_rate = 1.0 - avg_per_chunk * 2
            if out.empty_llm_chunk_rate >= empty_chunk_rate_threshold:
                out.signals.append("all_llm_chunks_empty")

    # Signal 3: low compound density vs compound-id mention density
    if (
        out.n_compound_id_mentions >= min_compound_id_mentions_for_density_signal
        and out.n_compounds_extracted < out.n_compound_id_mentions * density_threshold
    ):
        out.signals.append("low_compound_density")

    out.fire_burst = bool(out.signals)

    if out.fire_burst:
        logger.info(
            "gap_detector: signals %s "
            "(extracted=%d, expected≥%d, keywords=%d, val_hits=%d, grade_hits=%d, "
            "empty_chunk_rate=%.2f)",
            out.signals,
            out.n_compounds_extracted,
            int(out.n_compound_id_mentions * density_threshold),
            out.n_assay_keywords,
            out.n_value_unit_hits,
            out.n_letter_grade_hits,
            out.empty_llm_chunk_rate,
        )
    return out
