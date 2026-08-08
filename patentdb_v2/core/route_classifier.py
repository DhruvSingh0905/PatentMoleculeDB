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
    from . import config as _cfg
    if not _cfg.LLM_RECOVERY_ENABLED:
        response = None
    else:
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


# ── Markush region TAGGING (replaces the text/markush GATE) ─────────
# Philosophy shift (2026-05): we no longer ROUTE a patent into a single
# "text_dominant" or "markush_dominant" bucket and run one pipeline. The
# binary gate caused real data loss — a Markush-classified patent
# (US11566007) skipped HARVEST + Strategy 5 and returned 0 despite having
# 155 numbered examples with assay data. Instead, EVERY patent runs the
# full text-extraction path (specific exemplified compounds exist in
# nearly every patent, Markush or not), and we merely TAG the regions
# that carry Markush-genus signal so a downstream enumeration step knows
# where to look. The tag is additive metadata, never a gate.

import re as _re

_MARKUSH_PHRASE_RES = [
    _re.compile(r"compound\s+of\s+formula\s+\(?[IVX]+\)?", _re.IGNORECASE),
    _re.compile(r"wherein\s+each\s+occurrence", _re.IGNORECASE),
    _re.compile(r"independently\s+selected\s+from", _re.IGNORECASE),
    _re.compile(r"\bwherein\b[^.]{0,40}\bR\s*\d", _re.IGNORECASE),
    _re.compile(r"\bis\s+selected\s+from\s+the\s+group\s+consisting\s+of", _re.IGNORECASE),
    _re.compile(r"represents?\s+(?:hydrogen|halogen|optionally\s+substituted)", _re.IGNORECASE),
]


@dataclass
class MarkushRegion:
    """A text region carrying Markush-genus signal — a hint for the
    downstream enumeration step, NOT a gate on extraction."""
    offset: int             # char offset of the region start in the patent text
    n_phrase_hits: int      # Markush phrase density in this region
    snippet: str            # first ~200 chars for human inspection


_C_FIGURE_RE = _re.compile(r"-C\d{4,6}\.(?:png|tif|gif|jpe?g)", _re.IGNORECASE)
_D_FIGURE_RE = _re.compile(r"-D\d{4,6}\.(?:png|tif|gif|jpe?g)", _re.IGNORECASE)


@dataclass
class StructureImageTag:
    """Conservative, POSITIVE-evidence tag that a patent's compound
    structures are drawn as figure images that GOOGLE PATENTS ITSELF
    did not resolve to embedded SMILES — i.e. structures only an image-
    recognition step could recover.

    Why the GP-embedded-SMILES gap, not raw figure count: EVERY chemistry
    patent draws its compounds (US9718790 has 2,930 C-figures yet 94 %
    text coverage). Raw figure count is therefore useless as a "needs
    image recognition" signal. The discriminating signal is the GAP
    between structures DRAWN (C-figures) and structures GP already
    machine-read (its `<meta itemprop="smiles">` embeds). A large gap
    means Google's own OCR didn't cover the drawings — an EXTERNAL,
    positive signal, independent of whether OUR text pipeline succeeded.

    This is the guard against "our discovery failed, therefore assume
    images": we never tag from a text-extraction shortfall. We tag from
    (figures present) AND (GP's structured data didn't cover them).
    """
    n_structure_figures_c: int   # `C#####` — GP inline compound-structure drawings
    n_drawing_pages_d: int       # `D#####` — full drawing pages (may hold structures)
    n_gp_embedded_smiles: int    # structures GP already machine-read
    unresolved_gap: int          # max(0, C-figures − GP-embedded) — image-recog target size
    needs_image_recognition: bool  # GP MISSED the structures (sparse coverage) → image-recog to GET them
    gp_structures_unverified: bool  # GP read the drawn structures but its chem-OCR is the SOLE,
    #                                 unvalidated source (no independent text corroboration). It may
    #                                 be WRONG — US10273259: GP machine-read 93% of figures yet they
    #                                 are only 12% BDB-correct (sulfonyl→methyl, 9bR→9bS). → image-
    #                                 recog to VALIDATE/CORRECT. Distinct from `needs_*` (GP missed).
    structure_figure_urls: list[str]   # C-figure URLs (capped) for the downstream cropper


def tag_structure_image_figures(
    figure_image_urls: list[str],
    *,
    n_gp_embedded_smiles: int = 0,
    n_independent_text_structures: int = 0,
    gap_threshold: int = 50,
) -> StructureImageTag:
    """Tag a patent for image-based structure recognition.

    DETECTION SIGNAL: GP's `C#####` figure URLs — GP explicitly labels every
    compound-structure drawing, so this is the cleanest, most complete enumera-
    tion of "compounds that are drawn, not text" (empirically beats MinerU PNG
    embeds, which mix in schemes/headers and are absent on pure-GP patents). Use
    MinerU PNG embeds only as a fallback when GP carries no figures.

    Two distinct flags, because GP having read a structure ≠ having read it
    CORRECTLY:
      • `needs_image_recognition` — GP MISSED the drawings (sparse coverage):
        gap ≥ threshold AND GP read < half. Run image-recog to GET structures.
        (US11566007: embedded=0; US9718825: GP read 22%.)
      • `gp_structures_unverified` — GP READ the drawings but its chem-OCR is the
        SOLE source and nothing independent corroborates it. GP's count looking
        "sufficient" is NOT proof of correctness (US10273259: GP read 93%, only
        12% BDB-correct). Run image-recog to VALIDATE/CORRECT. Suppressed only
        when an independent text route corroborates a strong majority of the
        structures (then GP is cross-checked and trusted).

    Args:
      figure_image_urls   — GP figure URLs (C#### = inline structures,
                            D#### = drawing pages).
      n_gp_embedded_smiles — structures GP machine-read into `<meta smiles>`.
      n_independent_text_structures — structures our own NON-GP text routes
                            resolved (example-header / IUPAC harvest / tables).
                            Used to corroborate GP; does not, by itself, tag.
    """
    urls = figure_image_urls or []
    c_figs = [u for u in urls if _C_FIGURE_RE.search(u)]
    d_figs = [u for u in urls if _D_FIGURE_RE.search(u)]
    n_drawn = len(c_figs) + len(d_figs)
    gap = max(0, n_drawn - n_gp_embedded_smiles)
    needs = gap >= gap_threshold and n_gp_embedded_smiles < 0.5 * max(1, n_drawn)
    # GP read the figures, but is it cross-checked? Independent text corroboration
    # of a strong majority means GP was validated; otherwise GP is the lone,
    # unverified source for drawn compounds — flag it for image-recog validation.
    # Empirically, text-extractable patents independently corroborate 30-86% of
    # GP's structures; figure-based patents (GP is the lone reader) sit at 0-9%.
    # A 0.20 cut cleanly separates them.
    corroborated = n_independent_text_structures >= 0.20 * max(1, n_gp_embedded_smiles)
    gp_unverified = (
        n_drawn >= gap_threshold
        and n_gp_embedded_smiles > 0
        and not needs
        and not corroborated
    )
    return StructureImageTag(
        n_structure_figures_c=len(c_figs),
        n_drawing_pages_d=len(d_figs),
        n_gp_embedded_smiles=n_gp_embedded_smiles,
        unresolved_gap=gap,
        needs_image_recognition=needs,
        gp_structures_unverified=gp_unverified,
        structure_figure_urls=(c_figs or d_figs)[:5000],
    )


def tag_markush_regions(
    text: str,
    *,
    window: int = 4000,
    min_hits: int = 4,
) -> list[MarkushRegion]:
    """Scan `text` in non-overlapping windows and tag those whose
    Markush-phrase density meets `min_hits`. Returns the tagged regions
    sorted by density (densest first).

    This does NOT decide a route. The caller always runs the full text
    extraction; these tags simply mark where the genus is defined so a
    future Markush enumerator can expand R-groups from the right place.
    """
    if not text:
        return []
    tagged: list[MarkushRegion] = []
    n = len(text)
    pos = 0
    while pos < n:
        chunk = text[pos:pos + window]
        hits = sum(len(rx.findall(chunk)) for rx in _MARKUSH_PHRASE_RES)
        if hits >= min_hits:
            snippet = " ".join(chunk[:200].split())
            tagged.append(MarkushRegion(offset=pos, n_phrase_hits=hits, snippet=snippet))
        pos += window
    tagged.sort(key=lambda r: -r.n_phrase_hits)
    return tagged
