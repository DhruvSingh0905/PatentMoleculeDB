"""Top-level patent processing orchestrator — text vs Markush routing
+ impossible-fragment retry. The canonical entry point per Plan v3.

Pipeline:
    1. Fetch text via load_patent_description(prefer="auto")
    2. Run regex strategies 1-4 (skip Strategy 5; orchestrator decides)
    3. Classify route (text_dominant | markush_dominant | mixed)
    4a. text_dominant: HARVEST burst (already gap-gated) + Strategy 5 fallback
    4b. markush_dominant: keep text matches; Markush stubbed for now
    4c. mixed: text path; if completeness < 95%, also stub-Markush
    5. Retry impossible fragments (recover suspicious cids)
    6. Write outputs

Replaces `routes/text_index.py:extract_text_index`. The legacy entry
re-exports a `to_legacy_dict()` shape so existing scripts keep working.

Patent-agnostic; no per-patent flags. Cost-bounded at $0.50/patent total.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config
from ..core.cost_tracker import cost_tracker
from ..core.patent_text import load_patent_description
from ..core.route_classifier import classify_route, RouteDecision
from ..core.impossible_fragment_retry import retry_impossible_fragments
from ..core.assay_fsm.harvest.iupac_orchestrator import iupac_burst
from .google_patents import (
    audit_patent_format,
    extract_compounds_from_clean_text,
    fetch_patent_text,
)
from .google_patents_tables import extract_table_compounds_from_gp
from .google_tables import extract_table_compounds
from .google_ms_stubs import harvest_ms_stubs

logger = logging.getLogger(__name__)


@dataclass
class PatentResult:
    """Structured output of the orchestrator."""

    patent_id: str
    route_class: str             # text_dominant | markush_dominant | mixed
    route_reason: str
    example_index: dict           # cid → record dict
    assay_tables: dict            # cid → list[AssayResult-as-dict]
    markush_enumerated: list      # list of compound dicts; [] when stubbed
    retry_attempts: int = 0
    retry_recovered: int = 0
    cost_usd: float = 0.0
    audit: dict = field(default_factory=dict)
    text_source_format: str = ""  # mineru_markdown | google_html | google_html_fetched

    def to_legacy_dict(self) -> dict:
        """Shape that `extract_text_index` used to return — for back-compat
        with `write_text_index` and any caller that expected the old format.
        """
        return {
            "example_index": self.example_index,
            "assay_tables": self.assay_tables,
            "audit": {
                **self.audit,
                "patent_id": self.patent_id,
                "route_class": self.route_class,
                "route_reason": self.route_reason,
                "retry_attempts": self.retry_attempts,
                "retry_recovered": self.retry_recovered,
                "text_source_format": self.text_source_format,
            },
        }


# ── Internals ──────────────────────────────────────────────────────


# Pattern for `## Example N` (or `## Example N:`) markdown headers
# followed by an IUPAC name on the next non-empty line. Patent-agnostic
# — most US patents that use markdown-extracted text from MinerU OCR
# have this structure. The cid mapping is AUTHORITATIVE: the patent
# itself labels each compound, no position-counting needed.
_EXAMPLE_HEADER_PAT = re.compile(
    r"^##\s+Example\s+(\d+(?:[A-Z]{1,4})?)\s*:?\s*\n+([^\n#]{20,500})",
    re.MULTILINE,
)

# Pattern for GP description text — same idea as the MinerU header
# above, but operates on Google Patents' flat description where
# whitespace is normalized to single spaces. Captures `Example N`
# followed by IUPAC text up to the next `Example M` marker (or 500
# chars, whichever first). GP's text is OCR-clean, so this is the
# preferred source when available.


def _merge_example_iupacs_from_gp_description(
    patent_id: str, example_index: dict,
) -> int:
    """Lift `Example N → IUPAC` pairs from Google Patents' description text.

    Patent-agnostic. GP renders the patent body as flat text with all
    structural separators collapsed to single spaces, so the same regex
    library that walks MinerU markdown headers also works here — just
    without the markdown `##` anchor. Crucially, GP's text is OCR-clean
    (no backslash-escaped asterisks, no `1H-benzimidazol--yl` double
    dashes, no `2-y1}` digit-for-L typos), which is why OPSIN succeeds
    on headers where the MinerU pass silently failed.

    For each `Example N` marker:
      - Slice text from the marker up to the next `Example M` marker
        (capped at 500 chars to avoid eating synthesis paragraphs).
      - The first sentence-like chunk is the IUPAC name; everything
        after is "Step A", MS data, NMR, etc. — discard.
      - OPSIN it (with rule_clean + LLM cascade fallback like the
        MinerU pass).
      - If OPSIN succeeds AND the resulting InChIKey differs from
        what example_index has under that cid, REPLACE. If the cid
        isn't in example_index yet, ADD.

    Returns the number of cids overridden or added.

    This runs BEFORE the MinerU-based pass because GP text is cleaner.
    The MinerU pass remains as a fallback for patents GP doesn't have
    (older / non-US / withheld content).
    """
    from ..core.iupac_to_smiles import _convert_single, _try_opsin, rule_based_clean
    from ..core.smiles_utils import (
        canonicalize_smiles, get_inchikey, validate_smiles, molecular_weight,
    )
    from ..core.models import Compound, CompoundSource, IupacSource

    gp_cache = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
    if not gp_cache.exists():
        return 0
    try:
        desc = json.loads(gp_cache.read_text()).get("description", "") or ""
    except (OSError, json.JSONDecodeError):
        return 0
    if not desc:
        return 0

    # Slice text by Example N markers — each compound's IUPAC sits
    # between its own marker and the next.
    markers = [(m.start(), m.group(1)) for m in re.finditer(
        r"\bExample\s+(\d+(?:[A-Z]{1,4})?)\b", desc,
    )]
    if not markers:
        return 0

    # For each cid, take the FIRST occurrence's slice (later occurrences
    # are cross-references in synthesis prose, not the example header).
    seen_cids: set[str] = set()
    n_overrides = n_added = n_llm_rescued = 0

    for i, (pos, cid) in enumerate(markers):
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        next_pos = markers[i + 1][0] if i + 1 < len(markers) else len(desc)
        chunk = desc[pos:min(next_pos, pos + 800)]
        # Strip the `Example N` prefix and any leading punctuation.
        m_strip = re.match(r"^Example\s+\d+(?:[A-Z]{1,4})?\s*[\.:]*\s*", chunk)
        if not m_strip:
            continue
        iupac_chunk = chunk[m_strip.end():].strip()
        # Some patents render example headers as
        # "Example N\nSynthesis of <IUPAC>" / "Preparation of <IUPAC>" /
        # "Synthesis Example: <IUPAC>". Strip these section labels —
        # otherwise OPSIN sees "Synthesis of ..." (a prose fragment, not
        # a name) and either fails or, worse, the LLM cascade "cleans"
        # the prose into a plausible-but-wrong SMILES.
        iupac_chunk = re.sub(
            r"^(?:Synthesis|Preparation|Procedure|Method)\s+(?:of|for)?\s*[\.:]*\s*",
            "", iupac_chunk, flags=re.IGNORECASE,
        ).strip()
        if not iupac_chunk or len(iupac_chunk) < 20:
            continue
        # The IUPAC ends at the first phrase boundary that's clearly
        # post-name material: "was purified", "MS (ESI)", "Method",
        # "Step", "The title compound", "racemic" mid-sentence (these
        # are synthesis-description boilerplate).
        end_re = re.compile(
            r"\b(?:was\s+(?:prepared|purified|obtained|synthesized)"
            r"|MS\s*\(ESI\)|1H\s*NMR|Method\s+[0-9A-Z]|Step\s+[A-Z]\b"
            r"|The\s+title\s+compound|To\s+a\s+(?:stirred|solution)"
            r"|prepared\s+using|using\s+(?:similar|analogous))",
            re.IGNORECASE,
        )
        em = end_re.search(iupac_chunk)
        if em:
            iupac_chunk = iupac_chunk[:em.start()].strip()
        # Trim trailing "racemic ..." continuations that are actually
        # the next sentence in synthesis prose.
        iupac_chunk = re.sub(r"\s+racemic\s+.*$", "", iupac_chunk, flags=re.IGNORECASE)
        # Strip "as the TFA salt" / "as the HCl salt" tails.
        iupac_chunk = re.sub(
            r"\s+as\s+the\s+\w+\s+salt\s*$", "", iupac_chunk, flags=re.IGNORECASE,
        ).strip()
        if not iupac_chunk or len(iupac_chunk) < 20:
            continue

        # OPSIN it
        cleaned = re.sub(r"^(?:racemic|rac\.?|meso)\s+", "", iupac_chunk, flags=re.IGNORECASE).strip()
        smiles, _err = _try_opsin(cleaned, strict=False)
        stage_label = "gp_description_example_header"
        if not smiles:
            rc = rule_based_clean(cleaned)
            if rc != cleaned:
                smiles, _err = _try_opsin(rc, strict=False)
                if smiles:
                    stage_label = "gp_description_example_header_rule_cleaned"
        if not (smiles and validate_smiles(smiles)):
            compound = Compound(
                patent_id=patent_id,
                example_number=cid,
                iupac_name=iupac_chunk,
                iupac_source=IupacSource.PATENT_VERBATIM,
                source=CompoundSource.EXEMPLIFIED,
            )
            # is_clean_text=True — GP description is OCR-clean, so skip
            # the rule/autocorrect/vision stages and go straight to LLM.
            _convert_single(compound, is_clean_text=True, route_hint="page_extraction")
            if compound.canonical_smiles and validate_smiles(compound.canonical_smiles):
                smiles = compound.canonical_smiles
                iupac_chunk = compound.iupac_name or iupac_chunk
                stage_label = f"gp_description_example_header_{compound.extraction_method}"
                n_llm_rescued += 1
            else:
                continue

        canon = canonicalize_smiles(smiles)
        if not canon:
            continue
        mw = molecular_weight(canon)
        if mw is None or mw < 150:
            continue
        new_ik = get_inchikey(canon) or ""

        existing = example_index.get(cid)
        if existing and existing.get("inchikey") == new_ik:
            continue   # same molecule, leave alone

        # GP structured-data SMILES (Strategy 0) is the gold standard for
        # BDB-aligned patents — BDB curators often share the same source.
        # When the existing entry was bridged from a `gp_embedded_meta`
        # GP-canonical SMILES, don't override unless the new IUPAC OPSINs
        # to a SMILES whose CONNECTIVITY (stereo-flat InChIKey prefix)
        # agrees with the existing one. Otherwise the LLM cleanup is
        # producing a plausible-but-different molecule and we'd regress
        # BDB matches.
        if existing and existing.get("canonical_smiles"):
            ex_method = (existing.get("extraction_method") or "")
            ex_source = (existing.get("source") or "")
            ex_ik = existing.get("inchikey", "") or ""
            looks_gold = (
                ex_method == "gp_embedded_meta"
                or ex_source == "examples"
                and ex_method in ("gp_embedded_meta", "google_patents_table_opsin",
                                   "google_patents_example", "opsin_direct")
            )
            if looks_gold and ex_ik[:14] != new_ik[:14]:
                # Different connectivity from a trusted source — skip.
                continue

        rec = {
            "compound_id": f"Example {cid}",
            "iupac_name": iupac_chunk,
            "canonical_smiles": canon,
            "inchikey": new_ik,
            "source": "examples",
            "extraction_method": stage_label,
        }
        if existing:
            for k in ("ms_mh_plus_calcd", "ms_mh_plus_found"):
                if existing.get(k) is not None:
                    rec[k] = existing[k]
            n_overrides += 1
        else:
            n_added += 1
        example_index[cid] = rec

    if n_overrides or n_added:
        logger.info(
            "%s: GP-description Example pass — %d cids overridden, %d added "
            "(%d via LLM rescue)",
            patent_id, n_overrides, n_added, n_llm_rescued,
        )
    return n_overrides + n_added


def _merge_explicit_example_iupacs(
    patent_id: str, example_index: dict, *, data_dir=None,
) -> int:
    """Override example_index entries with IUPACs lifted from the patent's
    own `## Example N` markdown headers. AUTHORITATIVE — the patent
    explicitly labels each compound under each header, so this trumps
    Strategy 4's brittle position-based numbering.

    For each match:
      - Run OPSIN on the IUPAC text (with rule_clean fallback)
      - If OPSIN succeeds AND the resulting InChIKey differs from
        what example_index has under that cid, REPLACE — the patent's
        explicit header is more trustworthy than density-extracted text.
      - If example_index doesn't have the cid yet, ADD.

    Returns: number of cids overridden or added.
    """
    from ..core import config
    from ..core.iupac_to_smiles import _try_opsin, rule_based_clean
    from ..core.smiles_utils import (
        canonicalize_smiles, get_inchikey, validate_smiles, molecular_weight,
    )

    if data_dir is None:
        data_dir = config.DATA_DIR
    pages_dir = Path(data_dir) / patent_id / "all_pages"
    if not pages_dir.exists():
        return 0

    from ..core.iupac_to_smiles import _convert_single
    from ..core.models import Compound, CompoundSource, IupacSource

    n_overrides = n_added = n_llm_rescued = 0
    for page_file in sorted(pages_dir.glob("page_*.md")):
        try:
            text = page_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in _EXAMPLE_HEADER_PAT.finditer(text):
            cid = m.group(1).strip()
            iupac_raw = m.group(2).strip()
            # Strip markdown-escape sequences from MinerU output —
            # `(1R\*,2R\*)` becomes `(1R*,2R*)`, `2-y1}` stays as-is.
            # Without this, OPSIN fails immediately on backslash chars
            # and we silently skip the authoritative header override.
            iupac_raw = re.sub(r"\\([*{}\[\]()_])", r"\1", iupac_raw)
            # Strip trailing "as the TFA salt" / "as the HCl salt" — those
            # confuse OPSIN. Also collapse mid-name line breaks.
            iupac = re.sub(r"\s+as\s+the\s+\w+\s+salt\s*$", "", iupac_raw, flags=re.IGNORECASE)
            iupac = re.sub(r"\s+", " ", iupac).strip()

            # OPSIN it (fast path)
            cleaned = re.sub(r"^(?:racemic|rac\.?|meso)\s+", "", iupac, flags=re.IGNORECASE).strip()
            smiles, _err = _try_opsin(cleaned, strict=False)
            stage_label = "explicit_example_header"
            if not smiles:
                rc = rule_based_clean(cleaned)
                if rc != cleaned:
                    smiles, _err = _try_opsin(rc, strict=False)
                    if smiles:
                        stage_label = "explicit_example_header_rule_cleaned"

            # FULL LLM cascade fallback: when raw + rule_clean both fail,
            # OPSIN can't parse the OCR-corrupted header text (`2-y1}` for
            # `2-yl}`, missing parens, etc.). Route through _convert_single
            # which fires the LLM normalize stage. Without this, the
            # authoritative-header override silently SKIPS bad headers,
            # leaving wrong IUPACs from earlier strategies in place.
            if not (smiles and validate_smiles(smiles)):
                compound = Compound(
                    patent_id=patent_id,
                    example_number=cid,
                    iupac_name=iupac,
                    iupac_source=IupacSource.PATENT_VERBATIM,
                    source=CompoundSource.EXEMPLIFIED,
                )
                _convert_single(
                    compound, is_clean_text=False,
                    route_hint="page_extraction",
                )
                if compound.canonical_smiles and validate_smiles(compound.canonical_smiles):
                    smiles = compound.canonical_smiles
                    iupac = compound.iupac_name or iupac
                    stage_label = f"explicit_example_header_{compound.extraction_method}"
                    n_llm_rescued += 1
                else:
                    continue
            canon = canonicalize_smiles(smiles)
            if not canon:
                continue
            mw = molecular_weight(canon)
            if mw is None or mw < 150:
                continue
            new_ik = get_inchikey(canon) or ""

            existing = example_index.get(cid)
            if existing and existing.get("inchikey") == new_ik:
                continue   # same molecule, leave alone

            # Override (or add). The patent's explicit header trumps everything.
            rec = {
                "compound_id": f"Example {cid}",
                "iupac_name": iupac,
                "canonical_smiles": canon,
                "inchikey": new_ik,
                "source": "examples",
                "extraction_method": stage_label,
            }
            # Preserve MS data if it was already attached
            if existing:
                for k in ("ms_mh_plus_calcd", "ms_mh_plus_found"):
                    if existing.get(k) is not None:
                        rec[k] = existing[k]
                n_overrides += 1
            else:
                n_added += 1
            example_index[cid] = rec

    if n_overrides or n_added:
        logger.info(
            "%s: explicit Example header pass — %d cids overridden, %d added",
            patent_id, n_overrides, n_added,
        )
    return n_overrides + n_added


def _merge_ms_stubs(patent_id: str, example_index: dict) -> None:
    """Add MS-data-only stubs for compound_ids that only appear in the
    patent's MS section (no IUPAC name in prose). Mutates index in place.
    """
    try:
        ms_stubs = harvest_ms_stubs(patent_id)
    except Exception as e:
        logger.warning("ms_stubs harvest failed for %s: %s", patent_id, e)
        return
    n_added = 0
    for stub in ms_stubs:
        cid = (stub.compound_id or "").strip()
        # Strip prefix
        for prefix in ("Cpd. No. ", "Cpd.No. ", "Compound ", "Example ", "Cpd ", "Ex. "):
            if cid.startswith(prefix):
                cid = cid[len(prefix):].strip()
                break
        if not cid:
            continue
        if cid in example_index:
            # Augment with MS values; don't overwrite IUPAC/SMILES
            existing = example_index[cid]
            if not existing.get("ms_mh_plus_calcd"):
                existing["ms_mh_plus_calcd"] = stub.ms_mh_plus_calcd
            if not existing.get("ms_mh_plus_found"):
                existing["ms_mh_plus_found"] = stub.ms_mh_plus_found
            continue
        example_index[cid] = {
            "compound_id": stub.compound_id,
            "iupac_name": "",
            "canonical_smiles": "",
            "inchikey": "",
            "source": "ms_stub",
            "is_structure_only": True,
            "ms_mh_plus_calcd": stub.ms_mh_plus_calcd,
            "ms_mh_plus_found": stub.ms_mh_plus_found,
        }
        n_added += 1
    if n_added:
        logger.info("%s: added %d MS-stub entries", patent_id, n_added)


# Source-trust ranking for duplicate-cid resolution. Lower number =
# more trusted. When two records collide on cid AND share an InChIKey
# (i.e., same molecule, just from different extraction paths), keep
# the one with the lower rank so the displayed IUPAC is the cleanest.
_SOURCE_RANK = {
    "gp_embedded_meta":                  0,   # GP canonical SMILES (no IUPAC, but trusted)
    "pubchem_direct":                    1,   # canonical lookup
    "opsin_direct":                      2,   # OPSIN on raw IUPAC text — clean
    "google_patents_table_opsin":        3,   # GP HTML table parser
    "rule_cleaned":                      4,   # rule-based OCR fix → OPSIN
    "autocorrected":                     5,
    "vision_ocr":                        6,
    "vision_ocr_cleaned":                7,
    "llm_cleaned":                       8,   # LLM produced IUPAC may keep tail artifacts
    "llm_direct_smiles":                 9,
    "iupac_harvest_targeted":           10,
    "strategy5_iupac_harvest_broad":    11,
    "impossible_fragment_retry_attempt_1": 12,
}


def _looks_clean(iupac: str) -> bool:
    """Return True if the iupac string looks free of synthesis-prose
    artifacts (no `Step N`, em-dash, mmol/g quantity, stray reagent
    names, etc.). Used to prefer the cleaner of two records that
    represent the same molecule under the same cid.
    """
    if not iupac:
        return True
    if re.search(r"\b(?:mmol|mg|mL|equiv)\b", iupac):
        return False
    if re.search(r"^Step\s+\d|—", iupac):
        return False
    if re.search(r"\b(?:diphenylphosphoryl|triethylamine|HBTU|DIPEA)\b", iupac, re.IGNORECASE):
        return False
    if re.search(r"\d+\.?\d*\s*g[,\s]", iupac):
        return False
    return True


def _compounds_to_example_index(compounds: list) -> dict:
    """Convert a list[Compound] into the dict form example_index.json uses.

    Dedup rule (smarter than simple first-write-wins):
      - If new record is for the SAME cid AND SAME InChIKey AS existing,
        keep whichever record has the cleaner IUPAC text. Prefer trusted
        source (`_SOURCE_RANK`); break ties by `_looks_clean`.
      - If new record has a SMILES and existing doesn't, replace fully.
      - Otherwise first-write-wins.
    """
    from ..core.compound_id import normalize_cid_key
    out: dict = {}
    for c in compounds:
        raw = c.example_number or ""
        # Use the canonical normalizer when possible; fall back to the
        # raw string for non-numeric labels like "Intermediate A" or
        # "Example 2 Step C" which legitimately need to stay distinct.
        cid = normalize_cid_key(raw) or raw.strip()
        if not cid:
            continue
        rec = {
            "compound_id": c.example_number,
            "iupac_name": c.iupac_name or "",
            "canonical_smiles": c.canonical_smiles or "",
            "inchikey": c.inchikey or "",
            "source": "examples",
            "extraction_method": c.extraction_method or "",
        }
        if getattr(c, "failure_reason", None):
            rec["failure_reason"] = c.failure_reason

        if cid not in out:
            out[cid] = rec
            continue

        existing = out[cid]

        # Existing has no SMILES; new one does → upgrade.
        if not existing.get("canonical_smiles") and rec["canonical_smiles"]:
            out[cid] = rec
            continue

        # Both have SMILES. If they're for the same molecule (same
        # InChIKey), keep whichever has the cleaner IUPAC text.
        if (
            existing.get("inchikey")
            and existing["inchikey"] == rec["inchikey"]
            and rec["canonical_smiles"]
        ):
            ex_clean = _looks_clean(existing.get("iupac_name", ""))
            new_clean = _looks_clean(rec.get("iupac_name", ""))
            ex_rank = _SOURCE_RANK.get(existing.get("extraction_method", ""), 99)
            new_rank = _SOURCE_RANK.get(rec.get("extraction_method", ""), 99)
            # Keep new only if:
            # (a) new is clean and existing is not, OR
            # (b) both are clean (or both dirty) and new is from a
            #     more-trusted source
            if (new_clean and not ex_clean) or (
                new_clean == ex_clean and new_rank < ex_rank
            ):
                out[cid] = rec
    return out


def _strategy5_after_classification(
    patent_id: str,
    text: str,
    example_index: dict,
    n_assay_compounds: int,
    *,
    overwrite_on_collision: bool = False,
) -> dict:
    """Run Strategy 5 IUPAC HARVEST fallback. Adds new cids by default.

    Args:
        overwrite_on_collision: When True, replace existing example_index
            entries with LLM output when the LLM has a pair for that cid
            and the LLM's IUPAC OPSINs to a valid SMILES. Used for
            noisy-source patents (MinerU markdown) where regex cid
            mappings drift due to OCR artifacts.

    Mutates example_index in place. Returns it.
    """
    # ── Pre-library pass — apply auto-promoted IUPAC patterns first ──
    # Mirrors the assay-pattern library re-application in harvest_burst:
    # any patterns that the LLM has discovered on PRIOR patents and
    # auto-promoted (≥3 fingerprints) fire here against the full text
    # for free (regex-only). Reduces the LLM gap for the burst below.
    from ..core.assay_fsm.iupac_pattern_library import IupacPatternLibrary
    lib = IupacPatternLibrary.default()
    prelib_pairs: dict[str, str] = {}
    try:
        for row in lib.apply_patterns_to_text(text, patent_id=patent_id):
            cid = row.get("compound_id")
            nm = row.get("iupac_name")
            if cid and nm and cid not in prelib_pairs:
                prelib_pairs[cid] = nm
    except Exception as e:
        logger.warning("%s: IUPAC pre-library pass failed: %r", patent_id, e)
    if prelib_pairs:
        logger.info(
            "%s: IUPAC pre-library pass extracted %d pairs (zero LLM cost) "
            "BEFORE iupac_burst",
            patent_id, len(prelib_pairs),
        )

    # Run broad-mode iupac_burst in `force=True` so it returns ALL pairs
    # the LLM extracted, not just ones for cids we don't already have.
    # We then decide per-cid whether to overwrite based on the policy flag.
    existing_pairs_for_burst: dict[str, str] = {} if overwrite_on_collision else {
        cid: rec.get("iupac_name", "")
        for cid, rec in example_index.items()
        if rec.get("iupac_name")
    }
    # Pre-library pairs ALSO count as "already known" so the LLM uses
    # its budget on cids the library didn't catch.
    if not overwrite_on_collision:
        for cid, nm in prelib_pairs.items():
            existing_pairs_for_burst.setdefault(cid, nm)

    new_pairs = iupac_burst(
        patent_id=patent_id,
        text=text,
        existing_pairs=existing_pairs_for_burst,
        n_assay_compounds=n_assay_compounds,
        cost_tracker=cost_tracker,
        force=True,  # bypass gap check; we want to extract from this patent
        library=lib,
    )

    # ── Post-LLM library re-application ──
    # iupac_burst registers any LLM-discovered patterns into the library
    # (in "pending" status until 3 patents corroborate). Within THIS
    # patent run we want the immediate amplification: re-apply ALL
    # active patterns (canonical + freshly-pending from this run) across
    # the full text and pick up pairs the LLM didn't see in its chunks.
    try:
        post_lib_pairs = lib.apply_patterns_to_text(
            text,
            fresh_patterns=lib.list_pending(),
            patent_id=patent_id,
        )
    except Exception as e:
        logger.warning("%s: IUPAC post-LLM re-application failed: %r", patent_id, e)
        post_lib_pairs = []
    n_post_added = 0
    for row in post_lib_pairs:
        cid = row.get("compound_id")
        nm = row.get("iupac_name")
        if not (cid and nm):
            continue
        if cid not in new_pairs and cid not in existing_pairs_for_burst:
            new_pairs[cid] = nm
            n_post_added += 1
    if n_post_added:
        logger.info(
            "%s: IUPAC post-LLM re-application added %d pairs across "
            "%d active patterns (zero LLM cost)",
            patent_id, n_post_added, len(lib.list_pending()),
        )

    # Merge pre-library pairs into new_pairs so the downstream OPSIN
    # loop processes them with the same dedup logic.
    for cid, nm in prelib_pairs.items():
        if cid not in new_pairs:
            new_pairs[cid] = nm

    if not new_pairs:
        return example_index

    import re
    from ..core.iupac_to_smiles import _try_opsin
    from ..core.smiles_utils import get_inchikey, validate_smiles

    # ── Pass 1: OPSIN each (cid, iupac). Collect the names OPSIN can't
    # parse so we can batch a single LLM-direct-SMILES fallback for them.
    # Complex fused-ring / macrocyclic nomenclature (benzo[e]indole,
    # cycloundecaphane, …) defeats OPSIN — verified on US10273259 where
    # 5/5 sampled BDB compounds had their IUPAC verbatim in the patent
    # text but OPSIN returned nothing, so the compound was silently
    # skipped and never matched BDB. The LLM-direct fallback converts
    # those names; batching keeps it cheap (50% discount + concurrent).
    cid_to_smiles: dict[str, str] = {}
    opsin_failures: list[tuple[str, str]] = []  # (cid, cleaned_iupac)
    for cid, iupac in new_pairs.items():
        cleaned = re.sub(
            r"^(?:racemic|rac\.?|meso)\s+", "", iupac, flags=re.IGNORECASE,
        ).strip()
        smiles, _err = _try_opsin(cleaned)
        if smiles and validate_smiles(smiles):
            cid_to_smiles[cid] = smiles
        elif cleaned:
            opsin_failures.append((cid, cleaned))

    n_llm_recovered = 0
    if opsin_failures:
        from ..core.api_client import call_claude_text_batch
        from ..core.iupac_to_smiles import SMILES_FALLBACK_PROMPT
        reqs = [
            {
                # Sonnet, not Opus — IUPAC→SMILES recovery doesn't need the
                # premium model and Opus was burning 5× the credits for
                # basic name recovery. Verified 9/15 recovery on US10273259.
                "prompt": SMILES_FALLBACK_PROMPT.format(raw_name=name),
                "model": config.DEFAULT_MODEL,
                "max_tokens": 200,
                "cache_key": f"smiles_fallback:{name}",
            }
            for _cid, name in opsin_failures
        ]
        results = call_claude_text_batch(reqs, patent_id=patent_id)
        for (cid, _name), resp in zip(opsin_failures, results):
            if not resp:
                continue
            cand = resp.strip().split("\n")[0].strip()
            if validate_smiles(cand):
                cid_to_smiles[cid] = cand
                n_llm_recovered += 1
        logger.info(
            "%s: Strategy 5 broad — OPSIN failed on %d names; LLM-direct "
            "recovered %d (batched)",
            patent_id, len(opsin_failures), n_llm_recovered,
        )

    n_added = 0
    n_overwrote = 0
    n_alias_added = 0
    for cid, smiles in cid_to_smiles.items():
        iupac = new_pairs[cid]
        opsin_ik = get_inchikey(smiles) or ""
        rec = {
            "compound_id": f"Cpd. No. {cid}",
            "iupac_name": iupac,
            "canonical_smiles": smiles,
            "inchikey": opsin_ik,
            "source": "iupac_harvest_broad" if overwrite_on_collision else "iupac_harvest",
            "extraction_method": "strategy5_iupac_harvest_broad",
        }
        if cid in example_index:
            existing = example_index[cid]
            if not overwrite_on_collision and existing.get("canonical_smiles"):
                # Conservative mode: keep regex-extracted (typically
                # GP-embedded) SMILES as primary, BUT preserve the
                # OPSIN-derived IK as an alias so the BDB cross-ref
                # can still match the molecule via the IK that BDB
                # happens to track. Fixes US10273259-class patents
                # where GP <meta itemprop="smiles"> disagrees with
                # the patent's own IUPAC text — verified: 5/5 sampled
                # BDB IKs in US10273259 were unmatched-by-primary but
                # OPSIN of the verbatim patent text reproduces them.
                existing_ik = (existing.get("inchikey") or "").strip()
                if opsin_ik and opsin_ik != existing_ik:
                    aliases = existing.get("inchikey_aliases") or []
                    if opsin_ik not in aliases:
                        aliases.append(opsin_ik)
                        existing["inchikey_aliases"] = aliases
                        n_alias_added += 1
                continue
            n_overwrote += 1
        else:
            n_added += 1
        example_index[cid] = rec
    logger.info(
        "%s: Strategy 5 broad → +%d added, %d overwrote, +%d OPSIN-IK aliases "
        "preserved (LLM produced %d pairs, overwrite=%s)",
        patent_id, n_added, n_overwrote, n_alias_added,
        len(new_pairs), overwrite_on_collision,
    )
    return example_index


def ms_tolerance_da(ms_value: float) -> float:
    """Mass-agreement tolerance in Da, chosen from how the value was reported.

    The bridge used a flat ±5 Da. At drug-like weights (300-600 Da) that is
    ~8,000-17,000 ppm — wide enough that many different analogs and isomers
    of the same series pass, which is exactly how "right chemistry, wrong
    compound ID" gets produced. Measured against BindingDB, 19.9% of
    GP-derived records land in that bucket.

    Real instruments do not have one tolerance, so neither should we. We
    can't see the original string by this point (the value is already a
    float), but the decimal places that survived parsing tell us the
    measurement regime:

      523.0      → unit-resolution LC-MS, as printed in most synthesis
                   examples. Tolerance ±0.5 Da (half a mass unit).
      523.2      → low-precision report. ±0.3 Da.
      523.2451   → HRMS. 20 ppm, the conventional accuracy band for
                   Orbitrap/TOF-class instruments.

    Returned as absolute Da so the caller compares uniformly.
    """
    if ms_value <= 0:
        return 0.5
    # Count surviving decimal places without float-repr noise.
    as_text = f"{ms_value:.6f}".rstrip("0")
    decimals = len(as_text.split(".")[1]) if "." in as_text else 0
    if decimals == 0:
        return 0.5           # integer m/z — unit resolution
    if decimals <= 2:
        return 0.3           # one or two places — still nominal mass
    return ms_value * 20e-6  # HRMS — 20 ppm


def _detect_patent_cid_prefixes(patent_id: str) -> list[str]:
    """Return the patent's discovered compound-id prefixes (e.g. ``['I-']``
    for US9718790, learned by `cid_prefix_discovery`). Sorted longest-first
    so multi-char prefixes match before single-char ones (``II-`` before
    ``I-``).

    Empty list = no patent-specific prefix discovered, fall back to bare
    digit candidates only.
    """
    from pathlib import Path
    import json as _json
    disc_path = Path(__file__).resolve().parent.parent / "data" / "assay_vocabulary.discoveries.json"
    if not disc_path.exists():
        return []
    try:
        d = _json.loads(disc_path.read_text())
    except (OSError, _json.JSONDecodeError):
        return []
    out: list[str] = []
    for tok in d.get("tokens", []):
        if tok.get("class") != "COMPOUND_ID_PREFIX":
            continue
        if tok.get("status") not in ("auto_loaded", "promoted"):
            continue
        if patent_id in tok.get("fingerprints_observed", []) \
                or tok.get("first_seen_patent") == patent_id:
            out.extend(tok.get("surface_forms", []))
    return sorted(set(out), key=lambda s: (-len(s), s))


def _detect_cid_pad_width(prefixes: list[str], assay_tables: dict) -> int:
    """Inspect assay_tables keys to learn the zero-pad width the patent
    uses for its cid digits. E.g. US9718790 uses ``I-0020`` (width 4),
    while US11254686 uses ``Z1`` (width 1). Returns 0 if no prefixed
    cids are found in assay_tables.
    """
    from collections import Counter
    widths: Counter[int] = Counter()
    for k in assay_tables:
        for p in prefixes:
            if k.startswith(p):
                tail = k[len(p):]
                if tail.isdigit():
                    widths[len(tail)] += 1
                break
    return widths.most_common(1)[0][0] if widths else 0


def _bridge_gp_to_harvest_cids(
    patent_id: str,
    example_index: dict,
    assay_tables: dict,
) -> dict:
    """Bridge Strategy 0 GP-embedded entries to HARVEST's patent-numbered cids.

    Strategy 0 emits cids like `GP107` (positional in GP HTML) with valid
    SMILES + InChIKey but NO assay data. HARVEST extracts assay values
    keyed by the patent's actual compound number (`107`). When the
    InChIKey of `GP107` matches what would be at patent cid `107` (or
    any other cid), we want the assay data to flow through.

    Two-stage merge:
      Stage A: GP{idx} + patent_cid in example_index sharing the SAME
               InChIKey → merge (keep patent cid, drop GP{idx})
      Stage B: GP{idx} has InChIKey IK; assay_tables has cid C with
               IUPAC that OPSINs to IK → rename GP{idx} → C (so
               HARVEST's assay rows flow through to this molecule)

    Returns the (potentially modified) example_index.
    """
    # Build InChIKey → list of (cid, rec) pairs in example_index
    ik_to_cids: dict[str, list] = {}
    for cid, rec in example_index.items():
        ik = (rec.get("inchikey") or "").strip()
        if not ik:
            continue
        ik_to_cids.setdefault(ik, []).append(cid)

    # ── Stage A: merge GP{idx} + patent_cid sharing the SAME InChIKey ──
    n_merged_a = 0
    cids_to_delete = []
    for ik, cids in ik_to_cids.items():
        if len(cids) < 2:
            continue
        gp_cids = [c for c in cids if c.startswith("GP")]
        patent_cids = [c for c in cids if not c.startswith("GP")]
        if not gp_cids or not patent_cids:
            continue
        # Prefer patent_cid that has HARVEST assay data
        patent_cid = next(
            (c for c in patent_cids if c in assay_tables),
            patent_cids[0],
        )
        for gp_cid in gp_cids:
            gp_rec = example_index.get(gp_cid, {})
            patent_rec = example_index.get(patent_cid, {})
            if not patent_rec.get("canonical_smiles") and gp_rec.get("canonical_smiles"):
                patent_rec["canonical_smiles"] = gp_rec["canonical_smiles"]
                patent_rec["inchikey"] = gp_rec.get("inchikey", patent_rec.get("inchikey", ""))
                example_index[patent_cid] = patent_rec
            cids_to_delete.append(gp_cid)
            n_merged_a += 1
    for cid in cids_to_delete:
        example_index.pop(cid, None)

    # ── Stage B: rename orphan GP{idx} → HARVEST cid via IUPAC→InChIKey ──
    # For HARVEST cids that have assay data but no matching example_index
    # entry, try to find their InChIKey by OPSIN'ing their associated
    # IUPAC name (if any). Then check if a GP{idx} has that same InChIKey
    # — if so, RENAME the GP{idx} → HARVEST cid so the assay flows.
    #
    # The IUPAC for HARVEST cids comes from Strategy 5 broad's discoveries
    # which are stored in example_index under the HARVEST cid (added by
    # _strategy5_after_classification). So this stage really only fires
    # for compounds that Strategy 5 found AND Strategy 0 found, with a
    # cid mismatch — should be small but we'll catch them if possible.
    n_renamed_b = 0
    rebuild_ik_to_cids: dict[str, list] = {}
    for cid, rec in example_index.items():
        ik = (rec.get("inchikey") or "").strip()
        if ik:
            rebuild_ik_to_cids.setdefault(ik, []).append(cid)

    # For each orphan GP{idx} (no patent-cid sibling with same InChIKey)
    orphan_gp = []
    for ik, cids in rebuild_ik_to_cids.items():
        if all(c.startswith("GP") for c in cids):
            for gp_cid in cids:
                orphan_gp.append((gp_cid, ik))

    # For each assay_tables cid not in example_index, try to match its
    # InChIKey (compute from any IUPAC we have).
    # The cleanest signal: if example_index already has the patent_cid
    # (with IUPAC but maybe no SMILES), Stage A would have caught it.
    # So orphan GP cids don't have a matching patent cid.
    # Fallback: positional match — GP{N} ≈ HARVEST cid {N} in many patents.
    # Risky, so only apply when InChIKey can be cross-checked via
    # Strategy 5 broad's IUPAC pairs.
    # Pre-compute the GP→patent positional offset from Stage A's successes.
    # If Stage A established that GP1↔patent_cid 1, GP2↔patent_cid 2, ...,
    # then a Stage B rename for GP107→107 is corroborated. If Stage A
    # produced very few aligned pairs, we can't trust positional renaming.
    aligned_pairs = sum(
        1 for c in cids_to_delete
        if c.startswith("GP") and c[2:].isdigit()
    )
    positional_alignment_trusted = aligned_pairs >= 5

    # Stage B+ — patent-prefixed candidates.
    #
    # Many patents number their compounds with a non-canonical prefix
    # (US9718790: `I-0020`, US11254686: `Z1`). Strategy 0 names them
    # `GP107` (positional) and Strategy 5/HARVEST names them `I-0107` —
    # so even when Stage A finds zero shared InChIKeys (e.g. when
    # Strategy-5 OPSIN is unreliable for the patent's IUPAC syntax),
    # we can still bridge via positional alignment to the patent's
    # natural cid form. Trust signal: ≥25 % of GP{N} indices have a
    # corresponding `<prefix><N_padded>` key in assay_tables.
    patent_prefixes = _detect_patent_cid_prefixes(patent_id)
    pad_width = _detect_cid_pad_width(patent_prefixes, assay_tables) if patent_prefixes else 0

    prefix_aligned_pairs = 0
    digit_aligned_pairs = 0
    n_gp_with_digits = 0
    gp_ints: list[int] = []
    for c in example_index:
        if c.startswith("GP") and c[2:].isdigit():
            gp_ints.append(int(c[2:]))
            n_gp_with_digits += 1

    if patent_prefixes and pad_width:
        for n in gp_ints:
            n_padded = str(n).zfill(pad_width)
            for p in patent_prefixes:
                if f"{p}{n_padded}" in assay_tables:
                    prefix_aligned_pairs += 1
                    break

    # Bare-digit positional signal: even patents without a discovered
    # cid prefix (e.g. US11292791 uses just "1", "31", "250" as Example
    # numbers) can be bridged when GP{N} indexes line up 1:1 with the
    # patent's digit-keyed assay rows. Count distinct alignments —
    # include Stage A merges (already removed from example_index)
    # because they're confirmed positional pairs.
    n_ay_digits = sum(1 for k in assay_tables if k.isdigit())
    if n_ay_digits:
        stage_a_gp_ints = {
            int(c[2:]) for c in cids_to_delete
            if c.startswith("GP") and c[2:].isdigit()
        }
        gp_int_set = set(gp_ints) | stage_a_gp_ints
        for k in assay_tables:
            if k.isdigit() and int(k) in gp_int_set:
                digit_aligned_pairs += 1

    prefix_alignment_trusted = (
        n_gp_with_digits >= 20
        and prefix_aligned_pairs >= max(20, n_gp_with_digits // 4)
    )
    # For digit-only patents, fire when most digit ay keys have a
    # corresponding GP{int}. Threshold is generous because the signal
    # is fragile when the patent is small.
    digit_alignment_trusted = (
        n_ay_digits >= 3
        and digit_aligned_pairs >= max(3, n_ay_digits * 3 // 4)
    )
    if patent_prefixes or n_ay_digits:
        logger.info(
            "%s: bridge — patent prefixes %s, pad_width=%d, prefix-aligned "
            "%d/%d GP, digit-aligned %d/%d ay-digit (trusted: prefix=%s digit=%s)",
            patent_id, patent_prefixes, pad_width,
            prefix_aligned_pairs, n_gp_with_digits,
            digit_aligned_pairs, n_ay_digits,
            prefix_alignment_trusted, digit_alignment_trusted,
        )

    # Trust signal for the IK-OVERWRITE path. We re-enable overwriting,
    # but the rename loop preserves the displaced IK as
    # ``inchikey_aliases`` so downstream cross-refs can match against
    # either form. That removes the only reason overwrite was unsafe
    # before (the 28 BDB matches lost on US9718790's 2nd-pass test were
    # all cases where BDB's curation tracked the Strategy-5 / OPSIN-
    # derived IK; the alias list now lets BDB still find them).
    #
    # Empirical net on US10246453: +41 v2_has_ay (111 → 152), 0 BDB
    # matches lost (smart-overwrite simulation).
    overwrite_trusted = (
        positional_alignment_trusted
        or prefix_alignment_trusted
        or digit_alignment_trusted
    )

    def _candidates_for(suffix: str) -> list[tuple[str, str]]:
        """Enumerate assay_tables candidate keys for a GP{suffix}.
        Patent-prefixed candidates first (preferred — they're the
        patent's natural identifier), then bare-digit fallback.

        Returns ``(candidate_cid, mode)`` pairs where mode is one of:
          - ``"fresh"``    — target is not in example_index yet
          - ``"fill"``     — target is in example_index but lacks an IK;
                             we'll inject GP's IK (always safe)
          - ``"overwrite"``— target is in example_index with a different
                             IK (probably Strategy-5 OPSIN-derived);
                             we'll replace it with GP's authoritative
                             IK so BDB can find the molecule. Only
                             emitted when `overwrite_trusted` is True
                             (positional alignment empirically
                             validated), otherwise we'd risk mis-
                             association on patents where GP{N} doesn't
                             line up with the patent's Example N.

        The IK-missing branch lets us recover patents like US11292791
        where Strategy 5 added bare-digit entries (``"1"``, ``"31"``)
        for assay rows but had no IUPAC to compute an IK from.

        The IK-overwrite branch recovers patents like US10246453 where
        Strategy 5's OPSIN parses gave wrong IKs to the digit entries
        (66 such cases empirically); without overwriting them, BDB
        matches the GP* (which has authoritative IK but no assays)
        and we lose the rows the digit entry actually has.
        """
        cands: list[tuple[str, str]] = []

        def _classify(c: str) -> str | None:
            if c not in assay_tables:
                return None
            if c not in example_index:
                return "fresh"
            existing_ik = (example_index.get(c, {}).get("inchikey") or "").strip()
            if not existing_ik:
                return "fill"
            if overwrite_trusted:
                return "overwrite"
            return None

        if patent_prefixes and pad_width and suffix.isdigit():
            n_padded = str(int(suffix)).zfill(pad_width)
            for p in patent_prefixes:
                cand = f"{p}{n_padded}"
                mode = _classify(cand)
                if mode:
                    cands.append((cand, mode))
        if suffix.isdigit():
            mode = _classify(suffix)
            if mode:
                cands.append((suffix, mode))
        return cands

    cids_to_rename: dict[str, tuple[str, str]] = {}  # gp_cid → (new_cid, mode)
    for gp_cid, ik in orphan_gp:
        suffix = gp_cid[2:]
        cands = _candidates_for(suffix)
        if not cands:
            continue
        new_cid, mode = cands[0]
        # We can't verify InChIKey match here (HARVEST cid has no SMILES,
        # else Stage A would have caught it). Use the strongest available
        # signal: MS [M+H]+ — gated by `ms_tolerance_da`, which picks the
        # tolerance from the reported precision instead of the old flat
        # ±5 Da (see that function for why the flat value was unsafe).
        # Without MS, only rename when *some* positional-alignment signal
        # is trusted (Stage A or the new prefix-based signal).
        from ..core.smiles_utils import molecular_weight
        stored_mw = molecular_weight(example_index[gp_cid].get("canonical_smiles", "")) or 0
        ms_mh = None
        for a in assay_tables[new_cid]:
            name = (a.get("assay_name") or "").lower()
            if "ms" in name or "m+h" in name:
                ms_mh = a.get("value_numeric")
                break
        ms_verified = (
            ms_mh is not None
            and stored_mw
            and abs((ms_mh - 1.008) - stored_mw) <= ms_tolerance_da(ms_mh)
        )
        if (
            ms_verified
            or positional_alignment_trusted
            or prefix_alignment_trusted
            or digit_alignment_trusted
        ):
            # Take the rename: either MS confirms or the positional /
            # prefix alignment signal is strong enough on its own.
            #
            # Note: a *failing* MS check is NOT treated as evidence
            # against the rename. Patent MS columns are unreliable:
            # they may be [M+Na]+ adducts (+22 Da), [M-H]- in
            # negative mode (-2 Da), fragment ions, integer-rounded
            # parent masses, or — most often on US9718790 — column-
            # misaligned LLM extractions where the "MS" cell holds
            # a different column's value. Verified empirically:
            # at every positional offset -5..+14 only 7-9 % of MS
            # values agree within 5 Da of the GP-stored MW, with no
            # peak — so MS is statistical noise on this patent and
            # mustn't gate the bridge. Prefix-alignment at 94 % is
            # the trustworthy signal.
            cids_to_rename[gp_cid] = (new_cid, mode)
        # else: no MS, no alignment signal → skip (avoid silent mis-rename)

    n_overwritten = 0
    for gp_cid, (new_cid, mode) in cids_to_rename.items():
        rec = example_index.pop(gp_cid)
        rec["compound_id"] = f"Cpd. No. {new_cid}"
        if new_cid not in example_index:
            # mode == "fresh"
            example_index[new_cid] = rec
        else:
            # mode in {"fill", "overwrite"}. Target already exists in
            # example_index. We inject GP's authoritative SMILES/IK so
            # the BDB cross-reference can match this entry. For
            # "fill" (target has no IK) this is purely additive. For
            # "overwrite" (target has a Strategy-5 IK that conflicts
            # with GP's) we trust GP for the primary IK because Google
            # Patents' embedded structured data is the same source BDB
            # curates from. The displaced Strategy-5 IK is preserved
            # in ``inchikey_aliases`` so BDB rows whose curation
            # happened to follow the OPSIN derivation are still
            # findable.
            existing = example_index[new_cid]
            if rec.get("canonical_smiles"):
                if mode == "overwrite" or not (existing.get("canonical_smiles") or "").strip():
                    existing["canonical_smiles"] = rec["canonical_smiles"]
            if rec.get("inchikey"):
                if mode == "overwrite" or not (existing.get("inchikey") or "").strip():
                    if mode == "overwrite":
                        # Preserve the displaced IK as an alias so the
                        # BDB cross-ref can match against either form.
                        old_ik = (existing.get("inchikey") or "").strip()
                        if old_ik and old_ik != rec["inchikey"]:
                            aliases = existing.get("inchikey_aliases") or []
                            if old_ik not in aliases:
                                aliases.append(old_ik)
                            existing["inchikey_aliases"] = aliases
                        n_overwritten += 1
                    existing["inchikey"] = rec["inchikey"]
                elif (
                    rec["inchikey"] != (existing.get("inchikey") or "").strip()
                    and rec["inchikey"]
                ):
                    # Even when we don't overwrite the primary IK,
                    # collect GP's IK as an alias so BDB still finds
                    # this entry via either form. Happens when
                    # overwrite_trusted is False or _classify returned
                    # "fresh" (path that doesn't go through here).
                    aliases = existing.get("inchikey_aliases") or []
                    if rec["inchikey"] not in aliases:
                        aliases.append(rec["inchikey"])
                    existing["inchikey_aliases"] = aliases
        n_renamed_b += 1

    if n_merged_a or n_renamed_b:
        logger.info(
            "%s: bridge — %d GP+patent merged (Stage A), %d orphan GP renamed "
            "(Stage B; %d Strategy-5 IK overwrites)",
            patent_id, n_merged_a, n_renamed_b, n_overwritten,
        )
    return example_index


def _resolve_text_sources(patent_id: str) -> tuple[str, str, str, str]:
    """STAGE 1 — text sources.

    Returns (primary_text, primary_format, secondary_text, secondary_format).

    Primary is HTML (clean, no OCR artifacts) when available, falling back
    to MinerU markdown or live HTTP fetch. Secondary is the *other* source
    when both exist — density extraction runs on both because they're
    complementary (HTML has GP-embedded SMILES + clean Examples section;
    markdown has per-page tables and figure-attached IUPAC blocks the
    HTML's section-stripping discards).

    Raises RuntimeError if no text source is available.
    """
    # Stage 0: ensure MinerU OCR has been run. The FSM / HARVEST table
    # parser needs `<table>` structure to produce real assay-column names
    # (without it, every assay name collapses to a `col_N` placeholder).
    # `run_mineru_for_patent` is idempotent: if `{pid}/all_pages/` already
    # has pages it returns instantly; otherwise it downloads the PDF and
    # runs MinerU (~5-10 min CPU). Failures are logged but non-fatal —
    # extraction continues on GP description alone.
    pages_dir = config.DATA_DIR / patent_id / "all_pages"
    has_pages = pages_dir.exists() and any(pages_dir.glob("page_*.md"))
    if not has_pages:
        try:
            from ..core.mineru_runner import run_mineru_for_patent
            logger.info(
                "process_patent %s: no MinerU pages on disk — running OCR now "
                "(one-time, ~5-10 min)",
                patent_id,
            )
            run_mineru_for_patent(patent_id)
        except Exception as e:
            logger.warning(
                "process_patent %s: MinerU OCR failed (%r) — continuing with "
                "GP description only; assay-column names will degrade to col_N",
                patent_id, e,
            )

    text, src_format = load_patent_description(patent_id, prefer_format="auto")
    if not text:
        td = fetch_patent_text(patent_id)
        if td is None:
            raise RuntimeError(f"no text source available for {patent_id}")
        text = td.get("description", "") + "\n" + td.get("claims", "")
        src_format = "google_html_fetched"
    logger.info(
        "process_patent %s: primary text source = %s (%d chars)",
        patent_id, src_format, len(text),
    )
    text_md, md_format = "", ""
    if src_format in ("google_html", "google_html_fetched"):
        try:
            text_md, md_format = load_patent_description(
                patent_id, prefer_format="markdown",
            )
            if text_md:
                logger.info(
                    "process_patent %s: secondary text source = %s "
                    "(%d chars) — running density extraction on both",
                    patent_id, md_format, len(text_md),
                )
        except FileNotFoundError:
            pass
    return text, src_format, text_md, md_format


def _collect_pre_extraction(
    patent_id: str,
    text: str,
    src_format: str,
    text_md: str,
    md_format: str,
) -> dict:
    """STAGE 2 — pre-extraction merge (no LLM cost).

    Runs every deterministic source and collapses results into a
    cleanly-deduplicated example_index. Sources fired in this stage:
        - Strategy 0 (GP-embedded SMILES from HTML, free)
        - Strategies 1-4 density on HTML (full cascade, may use LLM)
        - Strategies 1-4 density on MinerU markdown (fast cascade, no LLM)
        - tables_html: HTML <table> blocks via MinerU
        - tables_gp: GP HTML "Cpd. No. ... Chemical Name" sequences
        - explicit_example_header: ## Example N markdown headers (overrides)
        - ms_stubs: MS-only entries for compounds not extracted elsewhere

    The merge uses _SOURCE_RANK + _looks_clean to pick the cleanest IUPAC
    when multiple sources produce the same molecule.
    """
    # Stage 0 — discover any patent-specific compound-id prefixes (e.g.
    # `Z1, Z2, …` in US11254686). Adds confirmed prefixes to the
    # auto-loaded vocabulary so the rest of pre-extraction + HARVEST
    # can parse them. Fires at most once per patent; ~1 LLM call per
    # unknown prefix that survives the frequency filter.
    try:
        from ..core.cid_prefix_discovery import discover_cid_prefixes
        discover_cid_prefixes(text, patent_id)
    except Exception as e:
        logger.warning(
            "%s: cid_prefix_discovery failed (%r) — continuing with "
            "canonical vocabulary only",
            patent_id, e,
        )

    examples = extract_compounds_from_clean_text(
        patent_id, text,
        skip_strategy_5=True,
        text_source_format=src_format,
    )
    examples_md: list = []
    if text_md:
        examples_md = extract_compounds_from_clean_text(
            patent_id, text_md,
            skip_strategy_5=True,
            text_source_format=md_format,
            cascade_mode="fast",
        )
        logger.info(
            "process_patent %s: secondary (markdown) density extraction → %d compounds",
            patent_id, len(examples_md),
        )
    tables_html = extract_table_compounds(patent_id)
    tables_gp = extract_table_compounds_from_gp(patent_id)

    all_pre_compounds = list(examples) + list(examples_md) + list(tables_html) + list(tables_gp)
    pre_example_index = _compounds_to_example_index(all_pre_compounds)
    # GP description is OCR-clean and patent-agnostic; run it FIRST so
    # its (cid → IUPAC → InChIKey) mappings populate example_index
    # before Stage A bridge tries to merge GP-embedded SMILES against
    # patent cids. The MinerU `## Example N` pass below is the fallback
    # for patents GP doesn't carry.
    _merge_example_iupacs_from_gp_description(patent_id, pre_example_index)
    _merge_explicit_example_iupacs(patent_id, pre_example_index)
    _merge_ms_stubs(patent_id, pre_example_index)

    logger.info(
        "process_patent %s: pre-extraction → %d cids "
        "(examples=%d, examples_md=%d, tables_html=%d, tables_gp=%d, %d with SMILES)",
        patent_id, len(pre_example_index),
        len(examples), len(examples_md), len(tables_html), len(tables_gp),
        sum(1 for r in pre_example_index.values() if r.get("canonical_smiles")),
    )
    return pre_example_index


def _post_process(
    patent_id: str,
    text: str,
    example_index: dict,
    assay_tables: dict,
    *,
    retry_budget: float,
) -> tuple[dict, int, int]:
    """STAGE 5 — post-processing: retry → bridge → backfill.

    These run together because they're all "trim and polish" — they don't
    create new molecules, just sanitize / link / enrich what's already there.

        retry_impossible_fragments: re-extract IUPAC for entries with
            broken SMILES via LLM (capped at 100 cids / 5 min runtime).
        _bridge_gp_to_harvest_cids: merge GP{idx} into patent cids by
            InChIKey so HARVEST assay rows flow through. Single pass —
            runs LAST so it sees retry-discovered patent cids.
        backfill_iupacs_for_example_index: PubChem InChIKey → IUPAC for
            entries with empty iupac_name (mostly GP-embedded). Display only.

    Returns (example_index, n_retry_attempted, n_retry_recovered).
    """
    n_attempted, n_recovered, example_index = retry_impossible_fragments(
        patent_id, text, example_index,
        budget_remaining_usd=retry_budget,
    )
    example_index = _bridge_gp_to_harvest_cids(
        patent_id, example_index, assay_tables,
    )
    # Generalized InChIKey-keyed cid collapse. The GP↔patent_cid bridge
    # above only handles one class of namespace collision (positional
    # GP{idx} merging with the patent's own numbering). The pattern
    # feedback loop creates another class: pattern_library rows under
    # `I-0020` while example_index already has the molecule under bare
    # `0020`. Same InChIKey, different cid surface form → join fails.
    # This pass walks both maps, groups cids by InChIKey, picks a
    # canonical surface form per cluster, and migrates assay rows.
    _collapse_cids_by_inchikey(patent_id, example_index, assay_tables)
    try:
        from ..core.iupac_backfill import backfill_iupacs_for_example_index
        backfill_iupacs_for_example_index(example_index)
    except Exception as e:
        logger.warning("%s: iupac_backfill failed: %r", patent_id, e)
    return example_index, n_attempted, n_recovered


def _collapse_cids_by_inchikey(
    patent_id: str,
    example_index: dict,
    assay_tables: dict,
) -> None:
    """Generalized cid-namespace collapse. Mutates both dicts in place.

    Two passes:

    A. For each assay_tables cid that has no matching example_index
       entry, search for a near-match cid in example_index (surface-form
       variants: case-fold, strip leading zeros, strip dash, etc.) and
       migrate the assay rows under that example_index cid. Otherwise
       the rows are orphan — the workbook can't render them.

    B. For each InChIKey present in example_index under multiple cid
       surface forms, pick a canonical cid (preference order: non-GP,
       most assay rows, shortest cid string, alphabetic). Migrate assay
       rows from every other cid → canonical cid; drop duplicate
       example_index entries.

    Patent-agnostic. Eliminates the bug class where the same molecule
    appears under different cid keys depending on extraction path.
    """
    from collections import defaultdict
    # ── Pass A: orphan assay_tables cids → nearest example_index cid ──
    n_orphan_migrated = 0
    ex_keys = set(example_index.keys())
    ex_keys_lower = {k.lower(): k for k in ex_keys}
    for ay_cid in list(assay_tables.keys()):
        if ay_cid in ex_keys:
            continue
        variants = [
            ay_cid.lower(),
            ay_cid.upper(),
            ay_cid.lstrip("0"),                          # "0020" → "20"
            ay_cid.replace("-", ""),                     # "I-0020" → "I0020"
            ay_cid.replace("-", "").lstrip("0"),         # "I-0020" → "I0020" → "I0020"
        ]
        # Also: if the cid has a letter prefix, try stripping it
        if ay_cid[:1].isalpha():
            tail = ay_cid.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-")
            if tail:
                variants.append(tail)
                variants.append(tail.lstrip("0"))
        target = None
        for v in variants:
            if v and v in ex_keys:
                target = v
                break
            if v and v.lower() in ex_keys_lower:
                target = ex_keys_lower[v.lower()]
                break
        if target and target != ay_cid:
            assay_tables.setdefault(target, []).extend(assay_tables[ay_cid])
            del assay_tables[ay_cid]
            n_orphan_migrated += 1

    # ── Pass B: collapse duplicate cids sharing one InChIKey ─────────
    ik_to_cids: dict[str, list[str]] = defaultdict(list)
    for cid, rec in example_index.items():
        ik = (rec.get("inchikey") or "").strip()
        if ik:
            ik_to_cids[ik].append(cid)

    def _canon_rank(cid: str) -> tuple:
        # Prefer: non-GP (patent native), then most assay rows, then
        # shortest cid (cleaner namespace), then alpha tiebreak.
        return (
            cid.startswith("GP"),
            -len(assay_tables.get(cid, [])),
            len(cid),
            cid,
        )

    # Option A: keep example_index entries for ALL surface forms so any
    # downstream join (workbook builder, BDB cross-ref) can still resolve
    # any cid surface form to its InChIKey. Only the assay_tables rows
    # get migrated to the canonical cid (so the workbook reads them
    # consistently).
    #
    # To make the BDB cross-ref deterministic, every non-canonical
    # surface form also gets a ``canonical_cid`` pointer back to the
    # row-holding entry. Without this, a BDB-IK → cid lookup over the
    # dict could land on the wrong sibling (the one Pass B left empty),
    # and the cross-ref would falsely report v2_has_ay=0 for that
    # molecule even though the rows exist under the canonical cid. On
    # US10246453, 168 InChIKeys have ≥2 surface forms in example_index
    # and 64 of them returned a row-less sibling under non-deterministic
    # dict iteration — exactly the −16 ``v2_has_ay`` regression we
    # observed earlier.
    n_collapsed = 0
    for ik, cids in ik_to_cids.items():
        if len(cids) < 2:
            continue
        canon = min(cids, key=_canon_rank)
        for c in cids:
            if c == canon:
                continue
            if c in assay_tables:
                assay_tables.setdefault(canon, []).extend(assay_tables.pop(c))
                n_collapsed += 1
            # DO NOT pop example_index[c] — keep the surface form so
            # joins from any extraction path still resolve. But mark
            # the alias so cross-refs can hop to the canonical cid.
            if c in example_index and c != canon:
                example_index[c]["canonical_cid"] = canon

    if n_orphan_migrated or n_collapsed:
        logger.info(
            "%s: cid-collapse — migrated %d orphan assay cids + collapsed "
            "%d duplicate-InChIKey cids",
            patent_id, n_orphan_migrated, n_collapsed,
        )


def _run_text_dominant(
    patent_id: str,
    text: str,
    example_index: dict,
    *,
    text_source_format: str = "",
) -> tuple[dict, dict]:
    """text_dominant + mixed share this body: HARVEST + Strategy 5 +
    targeted fill on cids HARVEST has but example_index lacks.

    Source-aware overwrite policy: when text_source_format is
    `mineru_markdown`, the regex cid mappings are likely off (OCR-induced
    comma-split drift), so we run broad Strategy 5 with `overwrite=True`.
    For HTML sources (clean), we keep regex IUPACs and let Strategy 5
    only fill new cids.
    """
    # 1. HARVEST burst (gap-gated inside extract_for_patent)
    from ..core.assay_fsm.pipeline import extract_for_patent
    raw_assays = extract_for_patent(patent_id)
    if isinstance(raw_assays, tuple):
        raw_assays = raw_assays[0]
    assay_tables = {
        cid: [
            {
                "assay_name": a.assay_name,
                "value_numeric": a.value_numeric,
                "unit": a.unit,
                "qualifier": a.qualifier,
                "n_runs": getattr(a, "n_runs", None),
            }
            for a in arr
        ]
        for cid, arr in raw_assays.items()
    }
    # 1b. Output validator — drop spurious assay rows whose (cid, value)
    #     doesn't appear inside any <table> block in the source markdown.
    #     Catches NMR coupling constants (J=X.X Hz), MS m/z fragments,
    #     and other non-table numeric values that HARVEST misattributed.
    from ..core.assay_fsm.output_validator import filter_assays_to_table_blocks
    assay_tables, _ = filter_assays_to_table_blocks(
        assay_tables, patent_id, config.DATA_DIR,
    )
    n_assay_compounds = len(assay_tables)

    # 2. Strategy 5 broad — overwrite on noisy markdown sources.
    #    ALWAYS runs (even when Strategy 0 dominates) — it's a single LLM
    #    call covering the whole patent, and empirically picks up 5-30
    #    additional compounds that GP-embedded missed (compounds with
    #    Markush wildcards, compounds drawn but where GP didn't extract
    #    a SMILES, etc.). Cheap (~$0.05) and high-value.
    overwrite = (text_source_format == "mineru_markdown")
    example_index = _strategy5_after_classification(
        patent_id, text, example_index, n_assay_compounds,
        overwrite_on_collision=overwrite,
    )

    # 3. Targeted fill — chunk-and-verify per missing cid.
    #    ALWAYS RUNS (un-gated) — even when Strategy 0 has ≥ HARVEST count,
    #    the cid namespaces don't overlap (GP positional vs HARVEST patent
    #    cids), so HARVEST cids without a co-existing patent cid in
    #    example_index can't bridge to GP{idx} entries via InChIKey.
    #    Targeted fill extracts IUPACs for those HARVEST cids → OPSIN →
    #    InChIKey → Stage A bridge merges with matching GP{idx}.
    #    Empirically (US10214537): un-gating recovers ~25 percentage
    #    points of value-match coverage. Cost: ~$0.50 + 5-10 min.
    example_index = _targeted_fill_missing_cids(
        patent_id, text, example_index, assay_tables,
    )

    # 4. Letter-grade potency bins (deterministic, no LLM). Some patents
    #    (US11566007, US11292791) report potency as +/++/+++ bins with a
    #    *Key legend instead of numeric values. We capture these as HONEST
    #    ranges (value_numeric=None; value_low/high carry the literal patent
    #    thresholds). Merged LAST — AFTER _targeted_fill_missing_cids — so the
    #    hundreds of binned example-IDs that lack a structure do NOT each
    #    trigger an LLM IUPAC-recovery call (cost). Bins carry potency, not
    #    structures; structure recovery for them is the separate image arm.
    from .letter_bin_assays import (
        dehallucinate_geomean_grades,
        extract_letter_bin_assays,
    )
    bin_assays = extract_letter_bin_assays(patent_id)
    if bin_assays:
        n_new = sum(1 for c in bin_assays if c not in assay_tables)
        n_rec = 0
        for cid, recs in bin_assays.items():
            assay_tables.setdefault(cid, []).extend(recs)
            n_rec += len(recs)
        logger.info(
            "%s: merged %d letter-bin records over %d cids (%d new cids)",
            patent_id, n_rec, len(bin_assays), n_new,
        )

    # 4b. De-fabricate geomean letter-grade midpoints → honest ranges. Some
    #     patents (US11292791) state a prose legend ("a value greater than 1 µM
    #     and less than 1000 µM is marked '+'") and the FSM collapsed each grade
    #     to sqrt(low·high) as a precise number. Replace those fabricated points
    #     with the honest range the legend states. No-op when no such legend.
    dehallucinate_geomean_grades(patent_id, assay_tables)

    # NOTE: bridge does NOT run here. It runs ONCE after retry in
    # process_patent's main flow, where it has the most-complete view
    # (retry adds patent cids that this function's bridge would miss).
    return example_index, assay_tables


def _targeted_fill_missing_cids(
    patent_id: str,
    text: str,
    example_index: dict,
    assay_tables: dict,
) -> dict:
    """Fire `iupac_burst_targeted` on every cid that HARVEST has assays
    for but example_index lacks a SMILES-resolved entry.

    This is the "chunk and verify" mechanic — we know HARVEST found
    these compounds, so they're definitely IN the patent text; we just
    need the LLM to read the right window to recover their IUPACs.

    Mutates example_index in place. Returns it.
    """
    from ..core.assay_fsm.harvest.iupac_orchestrator import iupac_burst_targeted
    from ..core.iupac_to_smiles import _try_opsin
    from ..core.smiles_utils import get_inchikey, validate_smiles

    missing_cids = [
        cid for cid in assay_tables.keys()
        if not (example_index.get(cid, {}).get("canonical_smiles") or "").strip()
    ]
    if not missing_cids:
        return example_index
    logger.info(
        "%s: targeted fill — %d cids HARVEST has but example_index lacks SMILES",
        patent_id, len(missing_cids),
    )
    new_pairs = iupac_burst_targeted(
        patent_id=patent_id,
        text=text,
        missing_cids=missing_cids,
        cost_tracker=cost_tracker,
        window_chars=8000,
    )
    import re as _re
    n_added = 0
    for cid, iupac in new_pairs.items():
        cleaned = _re.sub(
            r"^(?:racemic|rac\.?|meso)\s+", "", iupac, flags=_re.IGNORECASE,
        ).strip()
        smiles, _err = _try_opsin(cleaned)
        if not (smiles and validate_smiles(smiles)):
            continue
        example_index[cid] = {
            "compound_id": f"Cpd. No. {cid}",
            "iupac_name": iupac,
            "canonical_smiles": smiles,
            "inchikey": get_inchikey(smiles) or "",
            "source": "iupac_harvest_targeted",
            "extraction_method": "iupac_harvest_targeted",
        }
        n_added += 1
    logger.info(
        "%s: targeted fill produced %d pairs → %d successfully merged",
        patent_id, len(new_pairs), n_added,
    )
    return example_index




def _write_outputs(
    patent_id: str,
    example_index: dict,
    assay_tables: dict,
    audit: dict,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Persist example_index.json + assay_tables.json + route_audit.json."""
    if out_dir is None:
        out_dir = config.OUTPUT_DIR / "text_extraction" / patent_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "example_index": out_dir / "example_index.json",
        "assay_tables": out_dir / "assay_tables.json",
        "route_audit": out_dir / "route_audit.json",
    }
    payloads = {
        "example_index": json.dumps(example_index, indent=2, ensure_ascii=False),
        "assay_tables": json.dumps(assay_tables, indent=2, ensure_ascii=False),
        "route_audit": json.dumps(audit, indent=2, ensure_ascii=False, default=str),
    }
    # Serialize everything BEFORE touching disk, then write each file via
    # temp-file + atomic rename. Previously these were three bare write_text
    # calls: a crash or a serialization error partway through left
    # example_index.json and assay_tables.json describing different runs, and
    # a reader had no way to tell. Same-directory temp keeps the rename atomic.
    for name, text in payloads.items():
        target = paths[name]
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, target)
    return paths


# ── Public entry ───────────────────────────────────────────────────


def _uspto_xml_for(patent_id: str) -> str:
    """Grant/publication XML for the repair loop, or "" when unavailable."""
    from ..sources.uspto_xml import UsptoUnavailable, fetch_grant_xml
    try:
        return fetch_grant_xml(patent_id)
    except (UsptoUnavailable, Exception):
        return ""


def process_patent(
    patent_id: str,
    *,
    max_total_cost_usd: float = 5.00,
    out_dir: Path | None = None,
    write: bool = True,
) -> PatentResult:
    """Run the full extraction pipeline for one patent.

    Args:
        patent_id: e.g. "US8952177"
        max_total_cost_usd: Hard cap on per-patent LM spend (across
            Strategy 5 broad + retry pass).
        out_dir: Where to write outputs. Default: output/text_extraction/{pid}/
        write: If False, skip writing JSON files; just return the result.

    Returns:
        PatentResult with route_class, example_index, assay_tables,
        retry stats, cost. Use `.to_legacy_dict()` to get the shape
        that `extract_text_index` used to return.
    """
    initial_spend = cost_tracker.patent_spend(patent_id)

    # ── STAGE 1 — text sources ────────────────────────────────────
    text, src_format, text_md, md_format = _resolve_text_sources(patent_id)

    # ── STAGE 2 — pre-extraction (no-LLM-cost sources collapsed) ──
    audit = audit_patent_format(patent_id)
    pre_example_index = _collect_pre_extraction(
        patent_id, text, src_format, text_md, md_format,
    )

    # ── STAGE 3 — route classification ────────────────────────────
    decision: RouteDecision = classify_route(
        has_substituent_table=getattr(audit, "has_substituent_table", False),
        n_pre_compounds=len(pre_example_index),
        patent_text_excerpt=text[:3000],
        patent_id=patent_id,
        n_embedded_smiles=getattr(audit, "n_embedded_smiles", 0),
        n_phrase_hits_markush=getattr(audit, "n_phrase_hits_markush", 0),
    )
    logger.info(
        "process_patent %s: route=%s | reason=%s | used_llm=%s",
        patent_id, decision.route_class, decision.reason, decision.used_llm,
    )

    # ── STAGE 4 — branch (LLM-augmented HARVEST or Markush stub) ──
    # No route GATE. EVERY patent runs the full text-extraction path —
    # specific exemplified compounds (numbered examples + assay tables)
    # exist in nearly every patent, Markush-heavy or not. The old
    # `markush_dominant` branch stubbed those patents to 0 and lost real
    # data (US11566007: 155 examples + A-prefix assay table → 0). The
    # classifier's verdict is retained only as an informational tag.
    example_index, assay_tables = _run_text_dominant(
        patent_id, text, pre_example_index,
        text_source_format=src_format,
    )

    # ── Gap-triggered repair ──────────────────────────────────────
    # Only fires on tables the deterministic parser largely failed to read, and
    # asks for a RULE rather than for data: one small call per distinct table
    # LAYOUT, reused free on every later patent sharing that shape. A patent
    # whose tables all parse costs nothing here.
    #
    # Every proposal is validated before use (see repair/rules.py) — it must
    # beat the coverage the parser already achieves, generalise to rows the
    # model never saw, and survive an adversarial NMR/MS/MW battery. Proposals
    # that fail become escalations, so a bad answer is visible rather than
    # silently applied.
    repair_report = None
    if config.REPAIR_ENABLED:
        try:
            from ..repair.loop import repair_patent
            repaired, repair_report = repair_patent(
                patent_id, _uspto_xml_for(patent_id),
                max_calls=config.REPAIR_MAX_CALLS_PER_PATENT,
            )
            for rec in repaired:
                assay_tables.setdefault(rec.cid, []).append({
                    "assay_name": rec.assay_name,
                    "value_numeric": rec.value_numeric,
                    "unit": rec.unit,
                    "qualifier": rec.qualifier,
                    "n_runs": rec.n_runs,
                    "source": rec.source,
                })
            # UNCONDITIONAL. This used to be `if repair_report.rows_recovered:`,
            # so a patent recovering 500 rows logged a summary and a patent
            # recovering ZERO logged nothing at all — the louder the failure,
            # the quieter the output. Every silent patent in this corpus was
            # silent here first.
            logger.info("repair: %s", repair_report.summary())
            # ...and the report is no longer dropped on the floor. It carries
            # the escalations and capability gaps naming what no rule could
            # fix; `maybe_escalate` journals them and, when the blocker is
            # below what a rule can reach, puts the patent to the code tier.
            # Bounded per capability and per run — see repair/autoheal.
            from ..repair.autoheal import maybe_escalate
            healed = maybe_escalate(patent_id, repair_report)
            if healed and healed.get("applied"):
                # The reader changed underneath us. Re-extract so this patent
                # benefits from the patch it just paid for, rather than the
                # next run being the first to see it.
                from ..sources.uspto_assays import extract_from_patent
                for rec in extract_from_patent(_uspto_xml_for(patent_id)):
                    if not rec.is_usable:
                        continue
                    assay_tables.setdefault(rec.cid, []).append({
                        "assay_name": rec.assay_name,
                        "value_numeric": rec.value_numeric,
                        "unit": rec.unit,
                        "qualifier": rec.qualifier,
                        "n_runs": rec.n_runs,
                        "source": "autoheal_" + (rec.source or "uspto_xml"),
                    })
        except Exception as e:
            # Repair is additive. It must never be able to break a run that
            # would otherwise have produced results.
            logger.warning("repair: skipped for %s (%r)", patent_id, e)

    # Markush-genus region TAGS — additive metadata for a downstream
    # enumeration step. Tagging does NOT change what we extract here; it
    # marks WHERE the claimed genus is defined so a future Markush
    # enumerator can expand R-groups from the right windows.
    from ..core.route_classifier import tag_markush_regions, tag_structure_image_figures
    markush_regions = tag_markush_regions(text)
    markush_result: list = []   # enumerated virtual compounds — stubbed

    # Conservative structure-image tag (positive evidence only): does the
    # patent draw its compounds as figure images? Read GP figure URLs and
    # count C-figures (inline structures) / D-figures (drawing pages).
    # NEVER inferred from extraction shortfall — a fully-text-extracted
    # patent still gets its figures tagged. Downstream image recognition
    # uses this to know which patents/figures hold drawn structures.
    structure_image_tag = None
    _n_indep_text = 0
    try:
        _gp_cache = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
        if _gp_cache.exists():
            _gp = json.loads(_gp_cache.read_text())
            _fig_urls = _gp.get("figure_image_urls", []) or []
            # GP stores machine-read structures under `embedded_compounds`
            # (each {smiles, inchi_key}); count those with a real SMILES.
            _n_embed = sum(
                1 for e in (_gp.get("embedded_compounds", []) or [])
                if isinstance(e, dict) and (e.get("smiles") or "").strip()
            )
            # Independent-text corroboration: structures whose SMILES came from
            # a NON-GP route (OPSIN/LLM/PubChem off a text IUPAC), i.e. NOT
            # GP's own machine chem-OCR. When GP is the lone reader (few
            # independent structures relative to many GP-embedded SMILES), GP's
            # SMILES are unverified — the old gate trusted GP's OCR *count*, not
            # its *correctness* (US10273259: 1,118 GP structures, ~6% text-
            # corroborated, only 12% BDB-correct → GP chem-OCR was wrong).
            _n_indep_text = sum(
                1 for r in example_index.values()
                if r.get("canonical_smiles")
                and r.get("extraction_method") != "gp_embedded_meta"
            )
            structure_image_tag = tag_structure_image_figures(
                _fig_urls, n_gp_embedded_smiles=_n_embed,
                n_independent_text_structures=_n_indep_text,
            )
            if structure_image_tag.needs_image_recognition:
                logger.info(
                    "%s: structure-image tag — NEEDS IMAGE RECOGNITION: %d C-figs "
                    "+ %d D-pages drawn, only %d machine-read by GP → %d unresolved "
                    "drawn structures (image recognition target)",
                    patent_id,
                    structure_image_tag.n_structure_figures_c,
                    structure_image_tag.n_drawing_pages_d,
                    structure_image_tag.n_gp_embedded_smiles,
                    structure_image_tag.unresolved_gap,
                )
            elif structure_image_tag.gp_structures_unverified:
                logger.info(
                    "%s: structure-image tag — GP-UNVERIFIED: %d C-figs drawn, "
                    "%d machine-read by GP, but only %d independent-text "
                    "structures (%.0f%% corroboration < threshold) → GP is the "
                    "lone reader; its SMILES are flagged for image validation "
                    "(DECIMER-Seg will confirm-or-correct)",
                    patent_id,
                    structure_image_tag.n_structure_figures_c,
                    structure_image_tag.n_gp_embedded_smiles,
                    _n_indep_text,
                    100.0 * _n_indep_text / max(1, structure_image_tag.n_gp_embedded_smiles),
                )
            elif structure_image_tag.n_structure_figures_c:
                logger.info(
                    "%s: structure-image tag — %d C-figs drawn, %d machine-read "
                    "by GP (gap %d < threshold; structures already resolved, no "
                    "image recognition needed)",
                    patent_id,
                    structure_image_tag.n_structure_figures_c,
                    structure_image_tag.n_gp_embedded_smiles,
                    structure_image_tag.unresolved_gap,
                )
    except Exception as e:
        logger.warning("%s: structure-image tagging failed: %r", patent_id, e)
    if markush_regions:
        logger.info(
            "%s: tagged %d Markush-genus region(s) (densest %d phrase hits "
            "@offset %d) for downstream enumeration; text path already ran "
            "(%d compounds, %d assay cids)",
            patent_id, len(markush_regions),
            markush_regions[0].n_phrase_hits, markush_regions[0].offset,
            len(example_index), len(assay_tables),
        )

    # ── STAGE 5 — post-processing (retry, bridge, backfill) ───────
    spent_so_far = cost_tracker.patent_spend(patent_id) - initial_spend
    retry_budget = max(0.0, max_total_cost_usd - spent_so_far)
    example_index, n_attempted, n_recovered = _post_process(
        patent_id, text, example_index, assay_tables,
        retry_budget=retry_budget,
    )

    # ── Substituent-table scan (harvest-integrated, $0, LM-free) ──
    # Ride the SAME description text the assay/IUPAC harvest already
    # scanned — no second OCR/page pass (the "scan for substituent
    # tables as we scan IUPACs/assays, avoid redundancy" requirement).
    # A rich table (many R-labels whose options resolve to SMILES) flags
    # a true Markush genus whose species an enumeration step can expand
    # from a DECIMER-read scaffold. Distinct from gp_structures_unverified
    # (which flags GP machine-misread *specific* structures, not a genus).
    _markush_dict = None
    _n_md_labels = _n_md_resolved = 0
    _markush_enumerable = False
    try:
        from .text_markush import extract_markush_dict_hybrid, write_markush_dict
        # Symbolic-first ($0); LLM arm fires ONLY when opt-in
        # (SUBSTITUENT_LLM=1) AND this patent looks Markush (genus regions
        # or drawn structure figures present) AND symbolic undershot. The
        # triple gate keeps pure assay patents at $0.
        _markush_likely = bool(markush_regions) or (
            structure_image_tag is not None
            and structure_image_tag.n_structure_figures_c > 0
        )
        _markush_dict = extract_markush_dict_hybrid(
            patent_id, text,
            allow_llm=getattr(config, "SUBSTITUENT_LLM_ENABLED", False),
            markush_likely=_markush_likely,
        )
        if write:
            write_markush_dict(_markush_dict)
        _n_md_resolved = _markush_dict.audit.get("n_options_smiles_resolved", 0)
        _n_md_labels = len(_markush_dict.r_groups)
        _n_md_positions_with_smiles = sum(
            1 for e in _markush_dict.r_groups.values() if e.options_smiles
        )
        # "Enumerable" = enough distinct R-positions actually resolving to
        # SMILES that combinatorial expansion is meaningful. Shape-based,
        # patent-agnostic; NOT a per-patent threshold.
        _markush_enumerable = (
            _n_md_labels >= 3
            and _n_md_positions_with_smiles >= 3
            and _n_md_resolved >= 10
        )
        if _markush_enumerable:
            logger.info(
                "%s: substituent-table scan — ENUMERABLE genus: %d R-labels, "
                "%d positions w/ SMILES, %d options resolved → enumeration would "
                "expand species once a DECIMER-read scaffold is supplied",
                patent_id, _n_md_labels, _n_md_positions_with_smiles, _n_md_resolved,
            )
    except Exception as e:
        logger.warning("%s: substituent-table scan failed (%r)", patent_id, e)

    # ── STAGE 6 — persist + return ────────────────────────────────
    final_spend = cost_tracker.patent_spend(patent_id) - initial_spend
    # Route-tag breakdown so we can see at a glance how many compounds
    # came from each extraction path (Strategy 0 GP-embedded vs density
    # OPSIN vs LLM cleanup vs retry, etc.).
    from collections import Counter
    route_breakdown = Counter(
        (r.get("extraction_method") or "untagged")
        for r in example_index.values() if r.get("canonical_smiles")
    )
    logger.info(
        "%s: route breakdown — %s",
        patent_id,
        ", ".join(f"{k}={v}" for k, v in route_breakdown.most_common()),
    )
    audit_dict = {
        "merged_unique_compounds": len(example_index),
        "n_with_iupac": sum(1 for r in example_index.values() if r.get("iupac_name")),
        "n_with_smiles": sum(1 for r in example_index.values() if r.get("canonical_smiles")),
        "n_with_inchikey": sum(1 for r in example_index.values() if r.get("inchikey")),
        "asy_compounds": len(assay_tables),
        "route_breakdown": dict(route_breakdown),
        # Substituent-table scan (harvest-integrated, $0). `enumerable`
        # flags a true Markush genus whose species the enumeration path
        # (markush.step.enumerate_from_scaffold_table) can expand once a
        # DECIMER-read scaffold is supplied. Distinct from the image-tag
        # signals below (which flag GP-misread specific structures).
        "markush_table": {
            "n_labels": _n_md_labels,
            "n_options_resolved_to_smiles": _n_md_resolved,
            "enumerable": _markush_enumerable,
        },
        # Markush-genus region tags — additive metadata for a downstream
        # enumeration step. Always present (text path always ran); a
        # non-empty list just means "this patent ALSO defines a claimed
        # genus at these offsets", not "we skipped specific extraction".
        "markush_regions": [
            {"offset": r.offset, "n_phrase_hits": r.n_phrase_hits, "snippet": r.snippet}
            for r in markush_regions
        ],
        # Conservative structure-image tag (positive evidence only):
        # drawn structures GP did NOT machine-read → image-recognition
        # target. Never inferred from our own extraction shortfall.
        "structure_image_figures": (
            {
                "n_structure_figures_c": structure_image_tag.n_structure_figures_c,
                "n_drawing_pages_d": structure_image_tag.n_drawing_pages_d,
                "n_gp_embedded_smiles": structure_image_tag.n_gp_embedded_smiles,
                "unresolved_gap": structure_image_tag.unresolved_gap,
                "needs_image_recognition": structure_image_tag.needs_image_recognition,
                "gp_structures_unverified": structure_image_tag.gp_structures_unverified,
                "n_independent_text_structures": _n_indep_text,
                "structure_figure_urls": structure_image_tag.structure_figure_urls,
            }
            if structure_image_tag is not None else None
        ),
    }
    result = PatentResult(
        patent_id=patent_id,
        route_class=decision.route_class,
        route_reason=decision.reason,
        example_index=example_index,
        assay_tables=assay_tables,
        markush_enumerated=markush_result,
        retry_attempts=n_attempted,
        retry_recovered=n_recovered,
        cost_usd=final_spend,
        audit=audit_dict,
        text_source_format=src_format,
    )
    if write:
        _write_outputs(patent_id, example_index, assay_tables,
                       result.to_legacy_dict()["audit"], out_dir=out_dir)
    return result
