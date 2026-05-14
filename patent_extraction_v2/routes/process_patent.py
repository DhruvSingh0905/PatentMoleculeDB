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
_GP_EXAMPLE_PAT = re.compile(
    r"\bExample\s+(\d+(?:[A-Z]{1,4})?)\b[\s\.:]*([A-Z\(\[\{][^\n]{20,500})",
)


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
    # Run broad-mode iupac_burst in `force=True` so it returns ALL pairs
    # the LLM extracted, not just ones for cids we don't already have.
    # We then decide per-cid whether to overwrite based on the policy flag.
    new_pairs = iupac_burst(
        patent_id=patent_id,
        text=text,
        existing_pairs={} if overwrite_on_collision else {
            cid: rec.get("iupac_name", "")
            for cid, rec in example_index.items()
            if rec.get("iupac_name")
        },
        n_assay_compounds=n_assay_compounds,
        cost_tracker=cost_tracker,
        force=True,  # bypass gap check; we want to extract from this patent
    )
    if not new_pairs:
        return example_index

    import re
    from ..core.iupac_to_smiles import _try_opsin
    from ..core.smiles_utils import get_inchikey, validate_smiles

    n_added = 0
    n_overwrote = 0
    for cid, iupac in new_pairs.items():
        # Strip non-IUPAC stereo descriptors that OPSIN doesn't accept
        # ("racemic", "rac", "meso") — same cleanup the manual extraction
        # script and `_clean_name` apply.
        cleaned = re.sub(
            r"^(?:racemic|rac\.?|meso)\s+", "", iupac, flags=re.IGNORECASE,
        ).strip()
        smiles, _err = _try_opsin(cleaned)
        if not (smiles and validate_smiles(smiles)):
            continue
        rec = {
            "compound_id": f"Cpd. No. {cid}",
            "iupac_name": iupac,
            "canonical_smiles": smiles,
            "inchikey": get_inchikey(smiles) or "",
            "source": "iupac_harvest_broad" if overwrite_on_collision else "iupac_harvest",
            "extraction_method": "strategy5_iupac_harvest_broad",
        }
        if cid in example_index:
            existing = example_index[cid]
            if not overwrite_on_collision and existing.get("canonical_smiles"):
                # Conservative mode: keep regex-extracted SMILES
                continue
            n_overwrote += 1
        else:
            n_added += 1
        example_index[cid] = rec
    logger.info(
        "%s: Strategy 5 broad → +%d added, %d overwrote (LLM produced %d pairs, overwrite=%s)",
        patent_id, n_added, n_overwrote, len(new_pairs), overwrite_on_collision,
    )
    return example_index


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

    cids_to_rename = {}
    for gp_cid, ik in orphan_gp:
        # Try positional match: GP107 → HARVEST cid 107
        suffix = gp_cid[2:]
        if not (suffix.isdigit() and suffix in assay_tables and suffix not in example_index):
            continue
        # We can't verify InChIKey match here (HARVEST cid has no SMILES,
        # else Stage A would have caught it). Use the strongest available
        # signal: MS [M+H]+ — gate by ±5 Da against stored MW. Without
        # MS, only rename when Stage A established positional alignment
        # for the patent (i.e., GP indexing matches patent numbering).
        from ..core.smiles_utils import molecular_weight
        stored_mw = molecular_weight(example_index[gp_cid].get("canonical_smiles", "")) or 0
        ms_mh = None
        for a in assay_tables[suffix]:
            name = (a.get("assay_name") or "").lower()
            if "ms" in name or "m+h" in name:
                ms_mh = a.get("value_numeric")
                break
        if ms_mh is not None:
            if abs((ms_mh - 1.008) - stored_mw) <= 5.0:
                cids_to_rename[gp_cid] = suffix
        elif positional_alignment_trusted:
            cids_to_rename[gp_cid] = suffix
        # else: no MS, no Stage A alignment → skip (avoid silent mis-rename)

    for gp_cid, new_cid in cids_to_rename.items():
        rec = example_index.pop(gp_cid)
        rec["compound_id"] = f"Cpd. No. {new_cid}"
        # Don't replace existing — only move when target cid is empty
        if new_cid not in example_index:
            example_index[new_cid] = rec
            n_renamed_b += 1

    if n_merged_a or n_renamed_b:
        logger.info(
            "%s: bridge — %d GP+patent merged (Stage A), %d orphan GP renamed (Stage B)",
            patent_id, n_merged_a, n_renamed_b,
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
    try:
        from ..core.iupac_backfill import backfill_iupacs_for_example_index
        backfill_iupacs_for_example_index(example_index)
    except Exception as e:
        logger.warning("%s: iupac_backfill failed: %r", patent_id, e)
    return example_index, n_attempted, n_recovered


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


def _run_markush_dominant(
    patent_id: str,
    text: str,
    example_index: dict,
) -> tuple[dict, dict, list]:
    """Markush-dominant branch — STUBBED until Markush pipeline is ready.

    Currently:
      - Keeps regex-extracted text matches as-is (already in example_index)
      - Skips HARVEST burst (assay_tables empty)
      - Returns markush_enumerated = [] (stub)

    When Markush pipeline goes live, replace with:
        from ..markush.step import should_enumerate_markush, run_markush_enumeration
        validated = [c for c in pre_compounds if c.canonical_smiles]
        if should_enumerate_markush(audit, validated, completeness=0.0)[0]:
            markush_enumerated = run_markush_enumeration(patent_id, validated)
    """
    logger.info(
        "%s: route=markush_dominant — Markush enumeration stubbed "
        "(not yet production-ready); returning text matches only",
        patent_id,
    )
    return example_index, {}, []


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
    paths["example_index"].write_text(
        json.dumps(example_index, indent=2, ensure_ascii=False)
    )
    paths["assay_tables"].write_text(
        json.dumps(assay_tables, indent=2, ensure_ascii=False)
    )
    paths["route_audit"].write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str)
    )
    return paths


# ── Public entry ───────────────────────────────────────────────────


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
    markush_result: list = []
    if decision.route_class == "markush_dominant":
        example_index, assay_tables, markush_result = _run_markush_dominant(
            patent_id, text, pre_example_index,
        )
    else:
        # text_dominant or mixed — both run text path (HARVEST + Strategy 5)
        example_index, assay_tables = _run_text_dominant(
            patent_id, text, pre_example_index,
            text_source_format=src_format,
        )
        if decision.route_class == "mixed":
            # If mixed and completeness < 95%, the Markush stub is still []
            # for now (per user). When live, fire it here.
            n_smi = sum(1 for r in example_index.values() if r.get("canonical_smiles"))
            n_assay = len(assay_tables) or 1
            completeness = n_smi / n_assay if n_assay else 0.0
            if completeness < 0.95:
                logger.info(
                    "%s: mixed route — completeness %.0f%% < 95%%, would run "
                    "Markush enumeration here (currently stubbed)",
                    patent_id, completeness * 100,
                )

    # ── STAGE 5 — post-processing (retry, bridge, backfill) ───────
    spent_so_far = cost_tracker.patent_spend(patent_id) - initial_spend
    retry_budget = max(0.0, max_total_cost_usd - spent_so_far)
    example_index, n_attempted, n_recovered = _post_process(
        patent_id, text, example_index, assay_tables,
        retry_budget=retry_budget,
    )

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
