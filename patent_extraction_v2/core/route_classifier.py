"""Decide whether a patent is text-dominant or markush-dominant.

Routes any patent into one of:
    - `text_dominant`     : compounds described as numbered examples in
                             text (regex extraction works); HARVEST runs
    - `markush_dominant`  : compounds defined via Markush + substituent
                             tables (text alone undershoots); Markush
                             enumeration takes over
    - `mixed`             : both signals strong; run text path first,
                             fall back to Markush if completeness < 95%

Heuristic-first: most patents resolve from `audit_patent_format` signals
+ regex compound count without any LLM call. Genuinely ambiguous cases
(few compounds AND substituent table present) fire one Claude call
(~$0.005) to disambiguate.

Patent-agnostic — no per-patent flags.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .api_client import call_claude_text
from . import config

logger = logging.getLogger(__name__)


# Threshold for "regex got most of the compounds" — under this, we treat
# the regex as having undershot.
MIN_TEXT_DOMINANT_COMPOUNDS = 30


@dataclass
class RouteDecision:
    """Output of the classifier — what pipeline to run + why."""

    route_class: str         # "text_dominant" | "markush_dominant" | "mixed"
    reason: str              # human-readable signal trace
    used_llm: bool = False   # True if the ambiguous-case LLM fired
    cost_usd: float = 0.0    # incremental LM cost for this decision


def classify_route(
    *,
    has_substituent_table: bool,
    n_pre_compounds: int,
    patent_text_excerpt: str = "",
    patent_id: str = "",
    n_embedded_smiles: int = 0,
    n_phrase_hits_markush: int = 0,
) -> RouteDecision:
    """Decide the route for one patent.

    Args:
        has_substituent_table: From `audit_patent_format(patent_id).has_substituent_table`.
            Detects the "Ar / R1 / R2" column-header signature in the text.
            Brittle — fails on most Markush patents because HTML table
            structure is lost during tag-stripping. Kept as a last-resort
            signal alongside the new GP-HTML signals.
        n_pre_compounds: How many compounds the regex strategies (1-4)
            extracted on this patent. Anchored against
            MIN_TEXT_DOMINANT_COMPOUNDS to decide "did regex get most".
        patent_text_excerpt: Optional 1-3kB sample of the patent text.
            Used ONLY when the heuristic is ambiguous and we fall back
            to a single LLM call. If empty, ambiguous → text_dominant.
        patent_id: For cost attribution / logging.
        n_embedded_smiles: From `audit_patent_format(patent_id).n_embedded_smiles`
            — count of `<span itemprop="smiles">` entries with paired
            InChIKey in the GP HTML. Strong text-dominant signal: GP
            embeds full SMILES per drawn compound, which means the
            patent describes specific exemplified molecules rather than
            a Markush enumeration.
        n_phrase_hits_markush: From `audit_patent_format(patent_id)
            .n_phrase_hits_markush` — count of "compound of formula", "wherein
            each occurrence", "independently selected from" hits in
            description+claims. Five or more is strongly Markush.

    Heuristic ladder (most patents resolve from GP signals alone):
      1. n_embedded_smiles >= 30 → text_dominant (GP gave us a clean
         compound list; ignore phrase signals — even "wherein" Markush
         claims often appear alongside exemplified compounds).
      2. n_embedded_smiles == 0 AND n_phrase_hits_markush >= 5
         → markush_dominant (no GP-drawn compounds; pure Markush).
      3. has_substituent_table=True → markush_dominant (legacy regex).
      4. n_pre_compounds >= 30 → text_dominant (regex found enough).
      5. n_pre_compounds >= 5 AND has_substituent_table → AMBIGUOUS → LLM.
      6. fallback: text_dominant (sparse patent, no Markush signal).

    Returns:
        RouteDecision with route_class set.
    """
    # 1. GP-embedded SMILES is the most reliable text-dominant signal.
    if n_embedded_smiles >= 30:
        return RouteDecision(
            route_class="text_dominant",
            reason=f"GP HTML has {n_embedded_smiles} embedded SMILES — "
                   f"text-dominant",
        )

    # 2. No GP-embedded compounds AND multiple Markush phrases → likely Markush.
    if n_embedded_smiles == 0 and n_phrase_hits_markush >= 5:
        return RouteDecision(
            route_class="markush_dominant",
            reason=f"GP HTML has 0 embedded SMILES + {n_phrase_hits_markush} "
                   f"Markush phrase hits → markush",
        )

    # 3. Legacy Ar/R1/R2 substituent-table detector — only fires on a
    #    minority of Markush patents but still useful when GP HTML is absent.
    if has_substituent_table and n_embedded_smiles < 10:
        return RouteDecision(
            route_class="markush_dominant",
            reason=f"substituent table detected and only {n_embedded_smiles} "
                   f"GP-embedded SMILES → markush",
        )

    # 4. Regex extracted enough compounds → text-dominant
    if n_pre_compounds >= MIN_TEXT_DOMINANT_COMPOUNDS:
        return RouteDecision(
            route_class="text_dominant",
            reason=f"regex found {n_pre_compounds} cpds (GP-embedded={n_embedded_smiles}, "
                   f"phrases={n_phrase_hits_markush}, sub_table={has_substituent_table}) "
                   f"→ text",
        )

    # 5. Ambiguous: sparse but has_substituent_table → LLM if we have excerpt
    if has_substituent_table and patent_text_excerpt:
        return _llm_classify_ambiguous(
            patent_text_excerpt, n_pre_compounds, patent_id,
        )

    # 6. Fallback — sparse patent, no Markush signal
    return RouteDecision(
        route_class="text_dominant",
        reason=f"fallback: regex found {n_pre_compounds} cpds, GP-embedded="
               f"{n_embedded_smiles}, phrases={n_phrase_hits_markush}; "
               f"no Markush signal → text",
    )


def _llm_classify_ambiguous(
    text_excerpt: str,
    n_pre_compounds: int,
    patent_id: str,
) -> RouteDecision:
    """Single-shot classifier. Returns RouteDecision with used_llm=True."""
    prompt = f"""\
You are reading a patent excerpt. Decide whether the patent describes
its compounds primarily through:

  (A) TEXT-based examples — numbered compounds with full IUPAC names in
      paragraphs, like "Example 5: <iupac>" or "Compound 12 was prepared
      by..." Regex IUPAC extraction works on this format.

  (B) MARKUSH structures — a generic core scaffold with R-group / Ar
      substituent tables that define which combinations are claimed.
      Text mentions of compounds usually reference an "X" or "Compound
      number" via the table, not full IUPAC names. The compounds need
      to be enumerated from the scaffold + substituent combinations.

Regex pre-extracted only {n_pre_compounds} compounds from this patent;
this is below the typical text-dominant threshold (30+).

Return EXACTLY one of these tokens, on a single line, no other text:
    TEXT_DOMINANT
    MARKUSH_DOMINANT
    MIXED

Patent excerpt (truncated to 2000 chars):
---BEGIN---
{text_excerpt[:2000]}
---END---

Your one-token answer:"""
    response = call_claude_text(
        prompt=prompt,
        model=config.DEFAULT_MODEL,
        patent_id=patent_id,
        max_tokens=20,
    )
    if response is None:
        # API failure — default to text (safe; HARVEST still runs)
        return RouteDecision(
            route_class="text_dominant",
            reason="ambiguous; LLM call failed; defaulting to text",
            used_llm=True,
        )
    answer = response.strip().upper()
    if "MARKUSH" in answer:
        return RouteDecision(
            route_class="markush_dominant",
            reason=f"ambiguous ({n_pre_compounds} cpds + substituent table); "
                   f"LLM classified as MARKUSH",
            used_llm=True,
        )
    if "MIXED" in answer:
        return RouteDecision(
            route_class="mixed",
            reason=f"ambiguous ({n_pre_compounds} cpds + substituent table); "
                   f"LLM classified as MIXED",
            used_llm=True,
        )
    return RouteDecision(
        route_class="text_dominant",
        reason=f"ambiguous ({n_pre_compounds} cpds + substituent table); "
               f"LLM classified as TEXT (or unclear: {answer[:30]!r})",
        used_llm=True,
    )
