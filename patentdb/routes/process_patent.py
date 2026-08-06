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
from ..core.assay_name_guard import is_retention_time_name
from ..core.name_boundary import looks_like_spectra, terminate_name
from ..core.cost_tracker import cost_tracker
from ..core.patent_text import load_patent_description, load_uspto_description
from ..core.route_classifier import classify_route, RouteDecision
from ..core.impossible_fragment_retry import retry_impossible_fragments
from ..core.assay_fsm.harvest.iupac_orchestrator import iupac_burst
from .google_patents import (
    audit_patent_format,
    extract_compounds_from_clean_text,
    fetch_patent_text,
)
from .google_patents_tables import extract_table_compounds_from_gp
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


# Pattern for an `Example N` header alone on a line, followed by an IUPAC
# name on the next non-empty line. Patent-agnostic. The cid mapping is
# AUTHORITATIVE: the patent itself labels each compound, no
# position-counting needed.
#
# The `##` prefix is OPTIONAL. It was mandatory while this read MinerU
# markdown, where `##` is the OCR renderer's heading marker — nothing the
# patent itself contains. Against the grant XML, whose block tags become
# bare newlines, a required `##` matches nothing at all. Making it
# optional is what lets one pattern serve both, and it MATCHES MORE:
# US8952177 goes 168 -> 190 headers, US10544143 37 -> 51, US10273259
# 31 -> 48, with no id found in the markdown that the XML misses.
_EXAMPLE_HEADER_PAT = re.compile(
    r"^\s*(?:##\s+)?Example\s+(\d+(?:[A-Z]{1,4})?)\s*:?\s*$\n+\s*([^\n#]{20,500})",
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
        # The IUPAC ends where the synthesis paragraph begins. This was a list
        # of nine literal phrases and enumerating prose is a losing game — it
        # missed `To 3,3-difluorocyclobutyl-1-amine`, `A solution of LiHMDS`,
        # `Intermediate 132A:` and `was suspended in` on the first five
        # patents that were looked at, storing 466-791 characters of procedure
        # as `iupac_name`. `terminate_name` cuts on grammar instead: a prose
        # verb, a capitalised sentence opener, characterisation data, or a
        # reagent quantity — none of which a systematic name can contain. It
        # also repairs UTF-8-read-as-Latin-1 (`Î¼l` -> `μl`).
        #
        # Measured over the 14,921 stored names, batched through OPSIN:
        # 41 names rescued, ZERO broken. Small, and the zero is the point —
        # a terminator that clips a good name costs more than the prose does.
        iupac_chunk = terminate_name(iupac_chunk)
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
            # the OCR-cleanup stages and go straight to LLM.
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
    own `Example N` headers. AUTHORITATIVE — the patent explicitly labels
    each compound under each header, so this trumps Strategy 4's brittle
    position-based numbering.

    Reads the patent's own GRANT XML (then the GP description), not MinerU
    markdown. It used to glob `{data_dir}/{pid}/all_pages/page_*.md` and
    was the single largest markdown-only contributor to shipped output:
    92 records across the corpus, 85 of them InChIKeys no other route
    produced (74 on US8952177 alone — 26% of that patent).

    Repointing loses none of them. The XML carries a SUPERSET of the same
    headers, measured by counting matches of each pattern on both sources:

        US8952177    markdown 168  ->  XML 190   (0 md-only ids)
        US10544143   markdown  37  ->  XML  51   (0 md-only ids)
        US10273259   markdown  31  ->  XML  48   (0 md-only ids)

    The markdown pattern required a literal `##` heading, which is a
    MinerU rendering artifact rather than anything the patent contains;
    the line-anchored form below matches the same headers in the XML's
    block-tag-derived line breaks AND the ones MinerU's page splitting
    had broken.

    For each match:
      - Run OPSIN on the IUPAC text (with rule_clean fallback)
      - If OPSIN succeeds AND the resulting InChIKey differs from
        what example_index has under that cid, REPLACE — the patent's
        explicit header is more trustworthy than density-extracted text.
      - If example_index doesn't have the cid yet, ADD.

    Returns: number of cids overridden or added.
    """
    from ..core.iupac_to_smiles import _try_opsin, rule_based_clean
    from ..core.patent_text import load_gp_description
    from ..core.smiles_utils import (
        canonicalize_smiles, get_inchikey, validate_smiles, molecular_weight,
    )

    # `data_dir` is vestigial — the source is the patent's own text now.
    _ = data_dir

    from ..core.iupac_to_smiles import _convert_single
    from ..core.models import Compound, CompoundSource, IupacSource

    # XML first (the patent's own words), GP description second. Both are
    # scanned: they render the same headers differently and neither is a
    # strict superset of the other on every patent. Duplicate cids are
    # harmless — the InChIKey comparison below no-ops on a repeat.
    sources = [
        load_uspto_description(patent_id),
        load_gp_description(patent_id),
    ]

    n_overrides = n_added = n_llm_rescued = 0
    for text in sources:
        if not text:
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
    # `autocorrected` / `vision_ocr` / `vision_ocr_cleaned` sat at 5-7.
    # Both cascade stages were unreachable and are gone (see
    # `core/iupac_to_smiles` module docstring), so no record can carry
    # those methods and a rank for them would never be consulted.
    "opsin_relaxed_stereo_dropped":      5,   # connectivity ok, stereo dropped
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


# A mass reading, not a name — `496.1 (M + H) +`, `[M+H]+ 528.3`, `(M + 1)`.
#
# Anchored on the ADDUCT, never on the number, and case-sensitive on the `M`.
# `(m-tolyl)` opens a bracket with an m and a hyphen, and an axial-chirality
# descriptor legally opens one with an M — `(M)-1,1'-binaphthyl` — so both a
# case-blind test and a "bracket, M, sign" test condemn real names. Requiring
# the ion token AND the closing bracket leaves both alone: in `(M)-` the
# bracket closes before the sign, and `m-` is lowercase.
_MASS_ADDUCT_RE = re.compile(r"[\[(]\s*M\s*[+-]\s*(?:H|Na|K|Li|NH4|1|2)\s*[\])]")


def _is_chemical_name(raw: str) -> bool:
    """Is `raw` a chemical name, or is it characterisation data wearing one?

    Composed from the detectors that already exist rather than a fourth one:
    `terminate_name` (grammar-based prose/analytical cut, returns "" for a
    chunk that was spectra from character zero or carries no lowercase word),
    `looks_like_spectra` and `is_retention_time_name`, plus the adduct test
    above — `496.1 (M + H) +` contains no NMR marker and no chromatography, so
    the three existing detectors alone let it through.

    The test is applied to what SURVIVES `terminate_name`, not to the raw
    chunk, so a real name that merely trails its own MS line keeps its name:
    `…-2-carboxamide LCMS (m/z) (M+H)=528.3` cuts to the carboxamide and
    passes. Measured on US10273259: the description holds 1,036 mass readings
    in `NNN.N (M + H) +` form, 643 distinct, and every one that a pattern
    captures as an `iupac` reaches the paid batch below.
    """
    kept = terminate_name((raw or "").strip())
    if not kept:
        return False
    if looks_like_spectra(kept) or is_retention_time_name(kept):
        return False
    return not _MASS_ADDUCT_RE.search(kept)


# Extraction methods whose SMILES was derived FROM the stored `iupac_name`.
# For these the name is the evidence: if the name is a mass reading, the
# structure is a model's guess at what `496.1 (M + H) +` might be, and the
# record goes. Everything else (GP-embedded meta, table OPSIN off a different
# cell) has an independent structure, so only the false name is cleared.
_NAME_DERIVED_METHODS = (
    "strategy5_iupac_harvest_broad",
    "iupac_harvest_targeted",
    "iupac_harvest",
    "llm_direct_smiles",
    "llm_cleaned",
    "opsin_direct",
    "pubchem_direct",
)


def _reject_non_name_records(example_index: dict, patent_id: str = "") -> tuple[dict, int, int]:
    """Strip records whose `iupac_name` is not a name. Returns (index, dropped, cleared).

    Three outcomes, decided by what the record would still be worth without
    the false name:

      * structure derived FROM the name (`_NAME_DERIVED_METHODS`) → drop it;
        the molecule is a guess at what a mass reading might be.
      * no structure either → drop it; a record with neither a name nor a
        SMILES identifies nothing and only inflates `merged_unique_compounds`.
        Measured: 139 of US11292791's 150 such records are exactly this.
      * structure from an independent route → keep the molecule, clear the
        name into `iupac_rejected` so the discard is auditable rather than
        silent.

    THE write point for compound identity. Measured over the 22 shipped
    artifacts (2026-08-04): 214 of 17,786 records carry an `iupac_name` that is
    not a name, 47 of them with a `canonical_smiles`; 56 of those names are
    literal mass readings across six patents, 20 with a structure — including
    US10273259's `654.1 2.39 (M + H)+`. They arrived through five different
    upstream routes (`examples`, `llm_cleaned`, `gp_description_example_header
    _llm_cleaned`, `tables`, `strategy5_iupac_harvest_broad`), which is why
    guarding the routes one at a time is how the next capture pattern ships the
    next 214. This is the one choke point every record passes on its way to
    disk. Applying it costs 170 records dropped and 44 names cleared — 1.2% of
    the corpus, none of it a molecule with a name.

    A missing compound is recoverable; a mass reading recorded as a molecule is
    not — the same trade `_DIMENSIONLESS` makes for units.
    """
    dropped = cleared = 0
    out: dict = {}
    for cid, rec in example_index.items():
        name = (rec.get("iupac_name") or "").strip() if isinstance(rec, dict) else ""
        if not name or _is_chemical_name(name):
            out[cid] = rec
            continue
        method = rec.get("extraction_method") or rec.get("source") or ""
        if any(method.startswith(m) for m in _NAME_DERIVED_METHODS) or not (
            rec.get("canonical_smiles") or ""
        ).strip():
            dropped += 1
            continue
        rec = dict(rec)
        rec["iupac_name"] = ""
        rec["iupac_rejected"] = name
        cleared += 1
        out[cid] = rec
    if dropped or cleared:
        logger.warning(
            "%s: non-name iupac_name guard — dropped %d name-derived record(s), "
            "cleared the name on %d independently-sourced one(s)",
            patent_id, dropped, cleared,
        )
    return out, dropped, cleared


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

    if not config.IUPAC_BURST_ENABLED:
        logger.info(
            "%s: iupac_burst disabled (IUPAC_BURST=1 to enable) — Strategy 5 "
            "contributes pattern-library pairs only, at zero LLM cost",
            patent_id,
        )
        new_pairs = {}
    else:
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
    from ..core.iupac_to_smiles import (
        MIN_SMILES_LENGTH, MIN_SMILES_MW, _try_opsin,
    )
    from ..core.smiles_utils import get_inchikey, molecular_weight, validate_smiles

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
    n_not_a_name = 0
    for cid, iupac in new_pairs.items():
        cleaned = re.sub(
            r"^(?:racemic|rac\.?|meso)\s+", "", iupac, flags=re.IGNORECASE,
        ).strip()
        # A mass reading is not a name and asking a model to convert one is
        # not a recoverable failure — it is a paid request for a molecule
        # that does not exist, and the answer is un-checkable. On US10273259
        # a single pending pattern captured the `(M + H) +` column and 720 of
        # 721 requests in the batch below were readings like `496.1 (M + H) +`.
        # The pattern's OWN discovery description said "No IUPAC names are
        # present in this snippet — only spectral data is provided per
        # compound number"; nothing downstream read it. This is where that is
        # read, in the shape of the string itself.
        if not _is_chemical_name(cleaned):
            n_not_a_name += 1
            continue
        smiles, _err = _try_opsin(cleaned)
        if smiles and validate_smiles(smiles):
            cid_to_smiles[cid] = smiles
        elif cleaned:
            opsin_failures.append((cid, cleaned))
    if n_not_a_name:
        logger.info(
            "%s: Strategy 5 broad — %d of %d captured pairs are not chemical "
            "names (mass readings / NMR / retention times); not sent to OPSIN "
            "or to the LLM batch",
            patent_id, n_not_a_name, len(new_pairs),
        )

    n_llm_recovered = 0
    if opsin_failures and not config.STRATEGY5_LLM_ENABLED:
        logger.info(
            "%s: Strategy 5 LLM recovery disabled — %d OPSIN failures left "
            "unresolved (STRATEGY5_LLM=1 to enable)",
            patent_id, len(opsin_failures),
        )
        opsin_failures = []
    if opsin_failures:
        from ..core.api_client import call_claude_text_batch
        from ..core.iupac_to_smiles import SMILES_FALLBACK_PROMPT
        # DEDUP. The request is keyed by the NAME — same prompt, same
        # cache_key — so N cids sharing one name were N identical paid
        # questions: 447 distinct names across 721 requests on US10273259,
        # 274 of them bought nothing. The answer is fanned back out to every
        # cid below, so collapsing the QUESTION does not collapse the ANSWER.
        cids_by_name: dict[str, list[str]] = {}
        for cid, name in opsin_failures:
            cids_by_name.setdefault(name, []).append(cid)
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
            for name in cids_by_name
        ]
        results = call_claude_text_batch(reqs, patent_id=patent_id)
        for (name, cids), resp in zip(cids_by_name.items(), results):
            if not resp:
                continue
            cand = resp.strip().split("\n")[0].strip()
            # The floors `core/iupac_to_smiles._llm_direct_smiles` has always
            # had, and this sibling path never did. `validate_smiles` alone
            # says only "rdkit parsed it": `I` (127.9 Da) and `CCO` (46 Da)
            # both pass it, and a model answering an impossible question
            # answers small. A drug is not 3 characters of SMILES.
            if not (validate_smiles(cand) and len(cand) >= MIN_SMILES_LENGTH):
                continue
            mw = molecular_weight(cand)
            if not mw or mw < MIN_SMILES_MW:
                continue
            for cid in cids:
                cid_to_smiles[cid] = cand
            # Counted in CIDS, as before the dedup — one answer fanned out to
            # eight compound ids recovered eight compounds, not one.
            n_llm_recovered += len(cids)
        logger.info(
            "%s: Strategy 5 broad — OPSIN failed on %d names (%d distinct, "
            "%d duplicate requests saved); LLM-direct recovered %d (batched)",
            patent_id, len(opsin_failures), len(cids_by_name),
            len(opsin_failures) - len(cids_by_name), n_llm_recovered,
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
               HARVEST's assay rows flow through to this molecule).
               Stage B may also OVERWRITE C's InChIKey when C already
               holds a different one — see `_candidates_for`.
      Stage C: restore any molecule A and B destroyed between them.

    Stage C exists because A and B are individually defensible and jointly
    lossy. A drops GP{M} on the grounds that patent cid C already holds
    that InChIKey; B then re-points C at a different molecule, retracting
    the only evidence that made A's drop safe. Measured on the 2026-08-05
    corpus: 735 GP-embedded molecules across 9 patents ended the run with
    no entry holding their InChIKey as primary — US9694016 lost 245 —
    surviving only as strings in some other entry's `inchikey_aliases`,
    which no downstream join reads as a molecule.

    The invariant Stage C enforces: every InChIKey that is some entry's
    primary on the way IN is some entry's primary on the way OUT. It is
    insert-only into unclaimed cids, so it cannot cost an existing
    binding.

    Returns the (potentially modified) example_index.
    """
    # Snapshot every molecule handed to us, so Stage C can tell what was
    # destroyed. GP-keyed records win the tie because their positional
    # index is the only thing that says which patent cid the molecule
    # belongs to. `dict(rec)` is required, not optional: Stage B mutates
    # the target record in place, so a reference would show the
    # overwritten InChIKey and the victim would look like a survivor.
    entry_primary: dict[str, tuple[str, dict]] = {}
    for cid, rec in example_index.items():
        ik = (rec.get("inchikey") or "").strip()
        if not ik:
            continue
        prev = entry_primary.get(ik)
        if prev is None or (cid.startswith("GP") and not prev[0].startswith("GP")):
            entry_primary[ik] = (cid, dict(rec))

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

    def _positional_cids(suffix: str) -> list[str]:
        """The patent cids a `GP{suffix}` could positionally BE, best
        first: the patent's own prefixed form (`I-0107`) before the bare
        digit (`107`). Says nothing about whether they are free or
        occupied — `_candidates_for` decides that for renaming, Stage C
        for restoring. Shared so a restore can never land somewhere the
        rename would not have gone.
        """
        if not suffix.isdigit():
            return []
        out: list[str] = []
        if patent_prefixes and pad_width:
            n_padded = str(int(suffix)).zfill(pad_width)
            out += [f"{p}{n_padded}" for p in patent_prefixes]
        out.append(suffix)
        return out

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

        for cand in _positional_cids(suffix):
            mode = _classify(cand)
            if mode:
                cands.append((cand, mode))
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
        # EVERY candidate MS reading, not the first one found. This took the
        # first row whose name mentioned MS and `break`, which was fine while
        # one tier wrote the dict and became order-dependent the moment a
        # second tier started appending to the same list: two tiers reporting
        # different `[M+H]+` values for one compound made the rename decision
        # a function of which happened to be appended first. Measured — the
        # order `[good_ms, bad_ms]` renamed `GP107 -> 107` and `[bad_ms,
        # good_ms]` left it orphaned, on identical data.
        #
        # ANY corroborating reading verifies. MS is one of four alternative
        # trust signals here (the others are positional, prefix and digit
        # alignment) — it is evidence FOR a rename, never a veto, so a second
        # tier misreading a mass cannot revoke a match the first one earned.
        ms_candidates = [
            a.get("value_numeric") for a in assay_tables[new_cid]
            if "ms" in (a.get("assay_name") or "").lower()
            or "m+h" in (a.get("assay_name") or "").lower()
        ]
        ms_verified = bool(stored_mw) and any(
            v is not None and abs((v - 1.008) - stored_mw) <= ms_tolerance_da(v)
            for v in ms_candidates
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

    # ── Stage C: restore molecules Stages A and B destroyed between them ──
    #
    # Insert-only, and only into cids nothing else claimed — so this can
    # add a molecule but can never take one, nor move an assay row off the
    # entry that earned it. That is what makes it safe to run
    # unconditionally rather than behind another trust flag.
    #
    # Target order, and why:
    #   1. the molecule's own positional cid, via the SAME enumeration
    #      Stage B renames through. This is the rename Stage B would have
    #      made had Stage A not eaten the record first — it introduces no
    #      trust assumption the bridge was not already making, which is
    #      why it is gated on the same `*_alignment_trusted` flags. 284 of
    #      the 735 land here and re-attach 1,329 assay rows that were
    #      otherwise orphaned (their cid existed in assay_tables with no
    #      molecule to hang on); another 114 land on a free positional cid
    #      that has no rows.
    #   2. the cid it arrived under. For a Stage-A victim that is its GP
    #      key, always free — nothing re-creates GP keys after Strategy 0.
    #      337 land here: molecule kept, no cid claimed.
    #
    #   Counts are from an A/B replay of this function, old vs new, over
    #   one reconstructed input per patent — NOT from a pipeline run, and
    #   not from the shipped artifacts. Independently predicting the same
    #   split from example_index + the GP cache gave 286/112/337, agreeing
    #   to within 1%.
    #   3. a `~displaced` suffix, for a molecule overwritten in place that
    #      has no GP twin and therefore no free natural home. Ugly on
    #      purpose — a visibly odd cid is recoverable, a molecule silently
    #      deleted is not, and the alternative is to discard a structure we
    #      hold. Expected to be rare in production: the obvious candidates
    #      are Strategy 5's OPSIN keys, and those are written as ALIASES
    #      only (`_strategy5_after_classification`, the `:926` branch), so
    #      they are never in `entry_primary` and never restored. It fires
    #      974-735=239 times in the offline A/B replay purely because that
    #      replay's reconstruction has to promote such an alias back to a
    #      primary to rebuild the pre-bridge state — treat that count as an
    #      artefact of the harness, not a production estimate.
    positional_restore_ok = (
        positional_alignment_trusted
        or prefix_alignment_trusted
        or digit_alignment_trusted
    )
    post_primary = {
        (r.get("inchikey") or "").strip() for r in example_index.values()
    }
    post_primary.discard("")
    n_restored = 0
    n_restored_onto_assay_cid = 0
    for ik, (orig_cid, rec) in entry_primary.items():
        if ik in post_primary:
            continue
        target = None
        if positional_restore_ok and orig_cid.startswith("GP"):
            target = next(
                (c for c in _positional_cids(orig_cid[2:])
                 if c not in example_index),
                None,
            )
        if target is None and orig_cid not in example_index:
            target = orig_cid
        if target is None:
            target = f"{orig_cid}~displaced"
            if target in example_index:      # pathological; don't clobber
                continue
        rec = dict(rec)
        if not target.startswith("GP") and "~" not in target:
            rec["compound_id"] = f"Cpd. No. {target}"
        rec["restored_from"] = orig_cid
        example_index[target] = rec
        post_primary.add(ik)
        n_restored += 1
        if target in assay_tables:
            n_restored_onto_assay_cid += 1

    if n_merged_a or n_renamed_b or n_restored:
        logger.info(
            "%s: bridge — %d GP+patent merged (Stage A), %d orphan GP renamed "
            "(Stage B; %d Strategy-5 IK overwrites), %d molecules restored "
            "(Stage C; %d onto a cid that has assay rows)",
            patent_id, n_merged_a, n_renamed_b, n_overwritten,
            n_restored, n_restored_onto_assay_cid,
        )
    return example_index


def _resolve_text_sources(patent_id: str) -> tuple[str, str, str, str]:
    """STAGE 1 — text sources.

    Returns (primary_text, primary_format, secondary_text, secondary_format).

    Primary is Google Patents HTML (clean, no OCR artifacts) when
    available. When nothing is on disk we PULL — first the GP scrape,
    then the patent's own USPTO grant/publication XML. Secondary is now
    always ("", ""); the tuple shape is kept because `_collect_pre_
    extraction` and the tests take four values.

    **No OCR.** This function used to shell out to MinerU whenever a
    patent had no `page_*.md`, and a stack sample caught the cost: 3,317
    of 3,335 samples blocked in `subprocess.communicate`, ~5-10 min per
    patent, for a tier-3 source ranked below the GP HTML it already had.
    Both the OCR call and the markdown read are gone. The replacement is
    the fetch chain below — every source it can reach is a download of
    the document itself, not a re-reading of a picture of it.

    Raises RuntimeError if no text source is available.
    """
    text, src_format = load_patent_description(patent_id, prefer_format="auto")

    if not text:
        # Tier 2 — pull the Google Patents scrape. Populates
        # `output_v2/gpatents_cache/{pid}.json` for every later reader.
        logger.info(
            "process_patent %s: no text on disk — fetching Google Patents",
            patent_id,
        )
        td = fetch_patent_text(patent_id)
        if td is not None:
            text = td.get("description", "") + "\n" + td.get("claims", "")
            src_format = "google_html_fetched"

    if not text:
        # Tier 1 by rank, last by availability — the patent's own XML.
        # `fetch_grant_xml` writes `output_v2/uspto_xml/{pid}.xml`, which
        # is what `load_uspto_description` reads, so one fetch serves the
        # assay CALS path too. Pre-2002 grants legitimately have none
        # (`UsptoUnavailable`); that is not an error worth raising over.
        logger.info(
            "process_patent %s: no GP text — fetching USPTO XML", patent_id,
        )
        try:
            from ..sources import uspto_xml as _uspto
            _uspto.fetch_grant_xml(patent_id)
            text = load_uspto_description(patent_id)
            if text:
                src_format = "uspto_xml"
        except Exception as e:
            logger.warning(
                "process_patent %s: USPTO XML fetch failed (%r)", patent_id, e,
            )

    if not text:
        raise RuntimeError(f"no text source available for {patent_id}")

    logger.info(
        "process_patent %s: primary text source = %s (%d chars)",
        patent_id, src_format, len(text),
    )
    return text, src_format, "", ""


def _collect_pre_extraction(
    patent_id: str,
    text: str,
    src_format: str,
    text_md: str = "",
    md_format: str = "",
) -> dict:
    """STAGE 2 — pre-extraction merge (no LLM cost).

    Runs every deterministic source and collapses results into a
    cleanly-deduplicated example_index. Sources fired in this stage:
        - Strategy 0 (GP-embedded SMILES from HTML, free)
        - Strategies 1-4 density on HTML (full cascade, may use LLM)
        - tables_gp: GP HTML "Cpd. No. ... Chemical Name" sequences
        - explicit_example_header: `Example N` headers in the grant XML
          / GP description (overrides)
        - ms_stubs: MS-only entries for compounds not extracted elsewhere

    `text_md` / `md_format` are always "" — they carried the MinerU
    markdown tier, which no longer exists. The parameters remain so the
    caller's tuple unpack and the existing tests keep their shape.

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
    # The secondary density pass over MinerU markdown, and `tables_html`
    # (`google_tables.extract_table_compounds`, which globbed
    # `all_pages/page_*.md`), are both gone with the OCR tier.
    #
    # `tables_html` cost nothing to lose in structures: measured over the
    # 15 corpus patents that have markdown on disk it returned 1,157
    # compounds and 1,134 names but **0 SMILES** — the route never calls
    # the cascade, so it only ever seeded cid->name pairs. The GP table
    # route below is the one that resolves molecules
    # (`google_patents_table_opsin`, 855 shipped records).
    tables_gp = extract_table_compounds_from_gp(patent_id)

    all_pre_compounds = list(examples) + list(tables_gp)
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
        "(examples=%d, tables_gp=%d, %d with SMILES)",
        patent_id, len(pre_example_index),
        len(examples), len(tables_gp),
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
    import re as _re
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
    def _numeric_core(cid: str) -> str | None:
        """The identifier's digits, ignoring prefix and zero-padding.

        `GP107` -> `107`, `I-0020` -> `20`, `0020` -> `20`, `14` -> `14`.
        None when the cid carries no digits at all (a name-as-id row).
        """
        m = _re.search(r"\d+", cid or "")
        return (m.group(0).lstrip("0") or "0") if m else None

    n_collapsed = 0
    n_refused = 0
    for ik, cids in ik_to_cids.items():
        if len(cids) < 2:
            continue
        canon = min(cids, key=_canon_rank)
        canon_core = _numeric_core(canon)
        for c in cids:
            if c == canon:
                continue
            # SURFACE FORMS ONLY. This pass exists to join `GP107`/`107` and
            # `I-0020`/`0020` — the SAME identifier written two ways, which is
            # what the docstring above describes and every example in it.
            # Sharing an InChIKey was used as the whole test, so it also merged
            # DISTINCT example numbers whenever two of them resolved to the
            # same structure.
            #
            # US8952177 lost 11 compounds that way: 190 -> 179, with the same
            # 359 records redistributed rather than dropped. `14` was absorbed
            # into `10`, `158` into `156`, and so on across 9 groups. Those are
            # different compounds in the patent's own TABLE-US-00003 with
            # different measurements — cid 10 Ki=0.0045 against cid 14
            # Ki=0.056, cid 156 Ki=0.0068 against cid 158 Ki=0.27, a 40x
            # difference merged onto one identifier.
            #
            # A shared InChIKey between two distinct example numbers is a real
            # signal — a duplicate structure, or an upstream mis-assignment —
            # but it is never licence to discard one of them. Refusals are
            # counted so the signal stays visible instead of becoming silence.
            if canon_core is None or _numeric_core(c) != canon_core:
                n_refused += 1
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
                # Which arm produced this row. `harvest/orchestrator.py:68`
                # sets it on the AssayResult and its comment says dropping it
                # "is what left 27,868 shipped rows unattributed" — but the fix
                # landed on the producer only, and this dict rebuilt the row
                # without it. Measured today: 27,708 of 38,000 shipped records
                # (73%) carry no provenance, so "did the pattern library or the
                # burst produce this row" could only be answered by re-running
                # a simulation, never from the artifact.
                "source": getattr(a, "source", "") or "",
            }
            for a in arr
        ]
        for cid, arr in raw_assays.items()
    }
    # 1b. There is no output validator here any more.
    #
    #     `core/assay_fsm/output_validator.py` used to run at this point to
    #     drop rows whose (cid, value) it could not corroborate. Its
    #     corroboration machinery was dead code (no callers) and its two
    #     keep/drop branches read `validation_reason` / `source_quote`,
    #     which the row dict built above does not carry — the burst writes
    #     that field as `source`. The one branch still able to fire was a
    #     property-table heuristic, and replaying the corpus from cache
    #     measured it deleting 707 rows on US10273259 (46% of the patent,
    #     110 whole compounds), 681 of them the patent's PRIMARY assay
    #     `RORγ Binding IC50 μM`, 77% of a sampled 60 verifiable against
    #     the flattened raw grant XML.
    #
    #     See `tests/test_output_validator_removed.py` for the full trace.
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
    # Last line of defence, applied to whatever any caller hands us: a mass
    # reading must not reach disk as an `iupac_name`. `process_patent` already
    # ran this before computing the audit counts, so on that path it is a
    # no-op; it is here so a future caller — or a future capture pattern —
    # cannot ship the 214 records this guard was written for.
    example_index, _, _ = _reject_non_name_records(example_index, patent_id)
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
            # Fetched ONCE. `_uspto_xml_for` ran three times per patent here
            # and `extract_from_patent` three times behind it, for one document
            # that cannot change mid-run.
            patent_xml = _uspto_xml_for(patent_id)
            # `repaired` is the deterministic XML baseline UNIONED with what the
            # rules recovered — every run, not only the runs where a capability
            # patch lands. It used to be the rule output alone, which is why all
            # 35,888 shipped rows came from `{'<none>', 'letter_bin'}` and none
            # from the source that scores 99.9% exact against BindingDB.
            repaired, repair_report = repair_patent(
                patent_id, patent_xml,
                max_calls=config.REPAIR_MAX_CALLS_PER_PATENT,
            )
            for rec in repaired:
                # All 14 AssayRecord fields, not 6. The old literal wrote
                # value_numeric/unit/qualifier/n_runs/assay_name/source and
                # dropped the rest, which silently emptied every BINNED record:
                # those carry value_numeric None by design and keep the whole
                # measurement in the bounds. 45 of the 166 rules in
                # data/layout_rules.json are bin_key, so this was not a corner.
                # Names are the JSON ones (see core.models.AssayRow).
                assay_tables.setdefault(rec.cid, []).append({
                    "assay_name": rec.assay_name,
                    "value_numeric": rec.value_numeric,
                    "unit": rec.unit,
                    "qualifier": rec.qualifier,
                    "n_runs": rec.n_runs,
                    "source": rec.source,
                    "value_low": rec.range_lo,
                    "value_high": rec.range_hi,
                    "value_raw": rec.value_text,
                    "bin": rec.letter_grade,
                    "table_id": rec.table_id,
                    "column_header": rec.column_header,
                    "unit_source": rec.unit_source,
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
            from ..repair.autoheal import freeze_result, maybe_escalate
            healed = maybe_escalate(patent_id, repair_report)
            # Pin the answer. From here a patch bought for another document
            # cannot re-derive this one — see repair/snapshot. `repaired`
            # already carries the baseline, so re-extracting to prepend it
            # would freeze every baseline record twice.
            freeze_result(patent_id, list(repaired))
            if healed and healed.get("applied"):
                # The reader changed underneath us. Re-extract so this patent
                # benefits from the patch it just paid for, rather than the
                # next run being the first to see it.
                from ..sources.uspto_assays import extract_from_patent
                for rec in extract_from_patent(patent_xml):
                    if not rec.is_usable:
                        continue
                    assay_tables.setdefault(rec.cid, []).append({
                        "assay_name": rec.assay_name,
                        "value_numeric": rec.value_numeric,
                        "unit": rec.unit,
                        "qualifier": rec.qualifier,
                        "n_runs": rec.n_runs,
                        "source": "autoheal_" + (rec.source or "uspto_xml"),
                        "value_low": rec.range_lo,
                        "value_high": rec.range_hi,
                        "value_raw": rec.value_text,
                        "bin": rec.letter_grade,
                        "table_id": rec.table_id,
                        "column_header": rec.column_header,
                        "unit_source": rec.unit_source,
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
    # Run the non-name guard BEFORE the audit counts, not inside the writer:
    # `n_with_iupac` / `route_breakdown` are read as the record of what
    # shipped, and an audit that counts records the writer then removes is
    # exactly the "totals look healthy while patents are missing" failure this
    # file has produced before.
    example_index, _n_nonname_dropped, _n_nonname_cleared = _reject_non_name_records(
        example_index, patent_id,
    )
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
        # What the non-name guard removed on this run. Reported, not silent —
        # a guard whose work is invisible is indistinguishable from a route
        # that found nothing.
        "non_name_iupac": {
            "records_dropped": _n_nonname_dropped,
            "names_cleared": _n_nonname_cleared,
        },
        # What this run actually cost, broken down by route.
        #
        # `cost_tracker` is an in-memory singleton and NOTHING persisted it, so
        # no artifact on disk recorded what any run spent — which made "where
        # is the cost coming from" unanswerable from the evidence and left
        # every figure quoted about it an estimate. Per the backtrace rule in
        # CLAUDE.md, a number nothing produces is not a result.
        #
        # `patent_spend` is language-model spend only; image spend has its own
        # bucket and its own cap, so both are reported rather than summed into
        # one figure that hides which ceiling a patent is near.
        "cost": {
            "lm_usd": round(final_spend, 4),
            "image_usd": round(cost_tracker.patent_image_spend(patent_id), 4),
            "by_route": cost_tracker.all_route_spend(patent_id),
            "lm_cap_usd": config.PER_PATENT_LM_CAP,
            "lm_cap_hit": cost_tracker.patent_lm_exceeded(patent_id),
        },
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
