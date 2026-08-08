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
#
# Built dynamically from `core.compound_id._load_prefixes()` so that
# LLM-discovered patent-specific prefixes (e.g. `Z` for US11254686's
# Z1…Z700 codes) are picked up automatically. Hardcoding the prefixes
# here was a silent bug: any patent that didn't use the standard
# Cpd/Compound/Example labels matched zero, the density signal never
# fired, and HARVEST never got dispatched — exactly the case where
# the LLM safety net was most needed.


def _build_compound_near_assay_re() -> "re.Pattern[str]":
    """Construct the cid-near-assay matcher from the live vocabulary.
    Reads canonical + auto_loaded `COMPOUND_ID_PREFIX` surface forms
    via the same loader the tokenizer uses, so discovery additions take
    effect immediately on the next call.
    """
    try:
        from ..compound_id import _load_prefixes
        prefixes = _load_prefixes()
    except Exception:
        prefixes = ["Cpd", "Cpd.", "Compound", "Example", "Ex", "Ex.",
                    "Cmp", "Cmp.", "Cmp No.", "Cmp. No.", "Cpd. No.",
                    "Compound No.", "Example No."]
    alts = []
    for p in prefixes:
        esc = re.escape(p).replace(r"\.", r"\.?").replace(r"\ ", r"\s*")
        alts.append(esc)
    return re.compile(
        rf"(?:{'|'.join(alts)})\s*(\d{{1,4}}[A-Za-z]{{0,3}})",
        re.IGNORECASE,
    )


def _compound_near_assay_re() -> "re.Pattern[str]":
    """Lazy accessor — rebuilt each call so vocabulary edits (LLM
    discoveries appended mid-pipeline) take effect immediately. Cheap:
    ~50 regex alternations + compile.
    """
    return _build_compound_near_assay_re()


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
    # Dynamic vocab-aware matcher — picks up LLM-discovered prefixes.
    pat = _compound_near_assay_re()
    # Count BOTH prefixed forms (e.g. "Compound 5") AND bare-cid-shape
    # mentions of any known prefix's stem (e.g. "Z5" when "Z" is in vocab).
    mentions = set(m.group(1).lower() for m in pat.finditer(text))
    # Augment: for every single-letter prefix in vocabulary, also count
    # bare `<letter><digits>` forms (Z1, A12, etc.) — these are how
    # most patent-specific cids actually appear in body text.
    try:
        from ..compound_id import _load_prefixes
        for p in _load_prefixes():
            if len(p) <= 3 and p.isalpha():
                bare = re.compile(
                    rf"(?<![A-Za-z0-9]){re.escape(p)}-?(\d{{1,4}}[A-Za-z]{{0,3}})\b",
                    re.IGNORECASE,
                )
                mentions.update(m.group(1).lower() for m in bare.finditer(text))
    except Exception:
        pass
    out.n_compound_id_mentions = len(mentions)
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
