"""Unified text-side extraction orchestrator (Phase B + Phase C combined).

Reads a patent's Google Patents text response (cached on disk via
google_patents.fetch_patent_text) and emits two structured indices that
downstream routes consume:

  example_index.json   compound_id → {iupac_name, canonical_smiles,
                       inchikey, source ("examples"|"tables"|"merged"), ...}
  assay_tables.json    compound_id → list of {assay_name, value_numeric,
                       unit, qualifier, n_runs}

Plus a per-patent route_audit.json with provenance + counts so we can
spot regressions or false-positive bursts.

Why this exists
---------------
The image-side pipeline has no access to text. The text route extracts
the AUTHORITATIVE compound roster (compound_id, IUPAC name, SMILES) and
assay data via the Google Patents Route 1a (examples) + Route 1b (Cpd.
No. tables) + assay extractor. That output is what:
  - Phase D's Markush gate uses to decide stereo-vs-Markush
  - Image-side label_association uses to validate compound IDs
  - Markush enumeration uses as the validated_compounds input

Patent-agnostic — same pipeline runs on every patent. Per-patent
caching is provided by google_patents.fetch_patent_text (lives in
output/gpatents_cache/) so iteration is cheap.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core import config
from ..core.models import AssayResult, Compound
from .google_assays import extract_assays_for_patent
from .google_ms_stubs import harvest_ms_stubs
from .google_patents import extract_patent_google
from .google_patents_tables import extract_table_compounds_from_gp
from .google_tables import extract_table_compounds

logger = logging.getLogger(__name__)


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass
class ExampleRecord:
    """One compound's text-extracted metadata."""
    compound_id: str                 # e.g., "Example 5", "Cpd. No. 41"
    iupac_name: str | None = None
    canonical_smiles: str | None = None
    inchikey: str | None = None
    parent_smiles: str | None = None
    confidence_tier: int | None = None
    source: str = "examples"         # "examples" | "tables" | "merged" | "ms_stub"
    source_page: int | None = None
    is_intermediate: bool = False
    failure_reason: str | None = None
    # MS-data-only stub flags. When `is_structure_only=True`, this entry
    # was registered ONLY because the patent's MS section references the
    # compound by ID; the actual IUPAC + SMILES live in a structure
    # drawing or Markush template that the text route can't reach. The
    # ms_mh_plus values are the patent's reported [M+H]+ for downstream
    # cross-check once SMILES arrives from another route (image, Markush
    # enumeration, BindingDB).
    is_structure_only: bool = False
    ms_mh_plus_calcd: float | None = None
    ms_mh_plus_found: float | None = None


# ── Compound key normalization ──────────────────────────────────────


_COMPOUND_PREFIX_RE = re.compile(
    r"^(?:cpd\.?\s*no\.?|compound|example|cpd|ex\.?)\s*",
    re.IGNORECASE,
)


def _norm_compound_id(cid: str | None) -> str:
    """Lowercased + prefix-stripped compound ID for cross-route dedup.

    Patents reference the same compound with several surface forms:
      "Cpd. No. 5" / "Cpd. No 5" / "Compound 5" / "Example 5" / "Ex. 5"
    All MUST collapse to the same key so the merge in
    `_build_example_index` doesn't keep the same compound twice as
    `cpdno5` and `example5` (which then dict-overwrites by file
    iteration order, sometimes losing the entry with the SMILES).

    Strip the leading-prefix word, then drop all whitespace +
    punctuation, then lowercase. Result for all forms above: "5".

    Returns "" for empty/falsy input.
    """
    if not cid:
        return ""
    s = cid.strip()
    # Iteratively strip the prefix in case of doubled prefixes
    # (e.g., OCR'd "Cpd. No. Example 5")
    for _ in range(3):
        new = _COMPOUND_PREFIX_RE.sub("", s).strip()
        if new == s:
            break
        s = new
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ── Compound merge (Route 1a + Route 1b) ────────────────────────────


def _build_example_index(
    examples: list[Compound],
    tables: list[Compound],
) -> dict[str, ExampleRecord]:
    """Merge the example-route and table-route outputs into one
    compound_id → ExampleRecord dict.

    Collision resolution priority:
      1. Strategy-source rank: TABLE-1 ordered-list (`table1_ordered_list`)
         > claims_list > reverse_paren > density_example > density_cpd.
         Position-anchored sources (TABLE 1) are unambiguous; prose
         sources (Example markers) can have Example-vs-compound numbering
         offsets.
      2. Tiebreaker: more complete data (IUPAC + SMILES present).
      3. Final tiebreaker: longer IUPAC name (more characters → more
         specific structure).

    This prevents an Example-marker strategy from overwriting a TABLE-1
    position match for the SAME compound number when they actually point
    to different molecules in patents like US8952177.
    """
    # Strategy rank — lower number wins on collision. Anchored on the
    # `extraction_method` suffix produced by routes/google_patents.py.
    _STRATEGY_RANK = {
        "table1_ordered_list": 0,
        "claims_list": 1,
        "reverse_paren": 2,
        "iupac_harvest": 3,
        "density_example": 4,
        "density_cpd": 5,
        "density_compound": 5,
        "density_table": 6,
    }

    def _rank(c: Compound) -> int:
        em = (c.extraction_method or "").replace("google_patents_", "").strip()
        return _STRATEGY_RANK.get(em, 99)

    out: dict[str, ExampleRecord] = {}
    for src_name, src in [("examples", examples), ("tables", tables)]:
        for c in src:
            cid = c.example_number
            if not cid:
                continue
            key = _norm_compound_id(cid)
            rec = ExampleRecord(
                compound_id=cid,
                iupac_name=c.iupac_name,
                canonical_smiles=c.canonical_smiles,
                inchikey=c.inchikey,
                parent_smiles=c.parent_smiles,
                confidence_tier=int(c.confidence_tier) if c.confidence_tier else None,
                source=src_name,
                source_page=c.source_page,
                is_intermediate=bool(c.is_intermediate),
                failure_reason=c.failure_reason,
            )
            existing = out.get(key)
            if existing is None:
                out[key] = rec
                # Stash strategy rank for future collision decisions
                out[key]._strategy_rank = _rank(c)   # type: ignore[attr-defined]
                continue
            # Compare strategies. Lower rank wins.
            cur_rank = getattr(existing, "_strategy_rank", 99)
            new_rank = _rank(c)
            if new_rank < cur_rank:
                rec.source = "merged"
                rec._strategy_rank = new_rank   # type: ignore[attr-defined]
                out[key] = rec
                continue
            if new_rank > cur_rank:
                existing.source = "merged"
                continue
            # Tiebreaker: more complete data
            cur_score = (
                (1 if existing.iupac_name else 0)
                + (1 if existing.canonical_smiles else 0)
                + (len(existing.iupac_name or "") / 1000.0)
            )
            new_score = (
                (1 if rec.iupac_name else 0)
                + (1 if rec.canonical_smiles else 0)
                + (len(rec.iupac_name or "") / 1000.0)
            )
            if new_score > cur_score:
                rec.source = "merged"
                rec._strategy_rank = new_rank   # type: ignore[attr-defined]
                out[key] = rec
            else:
                existing.source = "merged"
    # Strip the temporary attribute before returning
    for r in out.values():
        if hasattr(r, "_strategy_rank"):
            try:
                delattr(r, "_strategy_rank")
            except AttributeError:
                pass
    return out


# ── Assay false-positive filter ─────────────────────────────────────


def _is_likely_compound_id_aliased_as_assay(
    compound_id: str, ar: AssayResult,
) -> bool:
    """Detect the "compound ID column read as assay value" false positive.

    The v1 assay extractor sometimes treats the compound_id column of
    a table as another assay (e.g., produces an assay row
    `assay_name="SGK-1 enzymatic activity", value=5` for compound 5).
    Signal:
      - assay_name lacks any unit/concentration token
      - unit is empty
      - value_numeric exactly equals the integer in compound_id

    This is generic across patents — no hardcoded compound IDs.
    """
    if ar.unit:
        return False
    if ar.value_numeric is None:
        return False
    if not float(ar.value_numeric).is_integer():
        return False
    # Extract integer prefix of compound_id (e.g., "Cpd. No. 5A" → 5)
    import re as _re
    m = _re.search(r"\d+", compound_id or "")
    if not m:
        return False
    if int(m.group(0)) != int(ar.value_numeric):
        return False
    # Concentration / activity unit-words in the assay name → keep
    name_l = (ar.assay_name or "").lower()
    unit_signals = ("μm", "um", "nm", "mm", "mol/l", "ic50", "ec50",
                    "ki ", "kd ", "%", "fold")
    if any(sig in name_l for sig in unit_signals):
        return False
    return True


def _filter_assays(
    raw: dict[str, list[AssayResult]],
) -> tuple[dict[str, list[AssayResult]], dict]:
    """Drop the obvious "compound ID as assay value" false positives.

    Returns the filtered dict plus an audit dict with the count of
    dropped entries (so we can track this metric across patents).
    """
    out: dict[str, list[AssayResult]] = {}
    dropped = 0
    for cid, results in raw.items():
        kept: list[AssayResult] = []
        for ar in results:
            if _is_likely_compound_id_aliased_as_assay(cid, ar):
                dropped += 1
                continue
            kept.append(ar)
        if kept:
            out[cid] = kept
    return out, {
        "n_compound_ids_in": len(raw),
        "n_compound_ids_out": len(out),
        "n_assay_values_in": sum(len(v) for v in raw.values()),
        "n_assay_values_out": sum(len(v) for v in out.values()),
        "n_assay_values_dropped_as_compound_id_aliasing": dropped,
    }


# ── Main entry ──────────────────────────────────────────────────────


def extract_text_index(patent_id: str) -> dict:
    """Run the full text-side extraction for one patent.

    Returns a dict with keys:
      example_index  dict[normalized_id, ExampleRecord]
      assay_tables   dict[compound_id, list[AssayResult]]
      audit          per-route counts + filter stats
    """
    logger.info(f"text_index: {patent_id} — starting")

    examples = extract_patent_google(patent_id)
    # Two table sources merge here:
    #   Route 1b-html — extract_table_compounds — parses HTML <table> blocks
    #     in PaddleX-OCR'd MD (for patents with literal table HTML).
    #   Route 1b-gp   — extract_table_compounds_from_gp — parses GP clean
    #     text for "Cpd.No. … Chemical Name" sequences. Class-B patents
    #     like US10899738 produce hundreds of compounds via this path.
    tables_html = extract_table_compounds(patent_id)
    tables_gp = extract_table_compounds_from_gp(patent_id)
    tables = list(tables_html) + list(tables_gp)
    example_index = _build_example_index(examples, tables)

    # Route 1c — MS-data-only stub harvester. Adds entries for
    # compound_ids that appear ONLY in the patent's MS section (no IUPAC
    # name extractable from prose). Each is flagged is_structure_only=True
    # so the fidelity report can distinguish "missing because parser failed"
    # from "missing because no text exists for this compound on this page".
    ms_stubs = harvest_ms_stubs(patent_id)
    n_stub_added = 0
    for stub in ms_stubs:
        key = _norm_compound_id(stub.compound_id)
        if not key:
            continue
        if key in example_index:
            # Already have richer entry from examples/tables route — just
            # attach the MS values for downstream verification, don't
            # overwrite the IUPAC/SMILES we already have.
            existing = example_index[key]
            if existing.ms_mh_plus_calcd is None:
                existing.ms_mh_plus_calcd = stub.ms_mh_plus_calcd
            if existing.ms_mh_plus_found is None:
                existing.ms_mh_plus_found = stub.ms_mh_plus_found
            continue
        # New stub: register as structure-only entry
        example_index[key] = ExampleRecord(
            compound_id=stub.compound_id,
            source="ms_stub",
            is_structure_only=True,
            ms_mh_plus_calcd=stub.ms_mh_plus_calcd,
            ms_mh_plus_found=stub.ms_mh_plus_found,
        )
        n_stub_added += 1
    if n_stub_added:
        logger.info(
            "ms-stub harvester: added %d structure-only stubs for %s "
            "(of %d total MS references; %d already covered by examples/tables)",
            n_stub_added, patent_id, len(ms_stubs), len(ms_stubs) - n_stub_added,
        )

    raw_assays = extract_assays_for_patent(patent_id)
    assay_tables, assay_audit = _filter_assays(raw_assays)

    audit = {
        "patent_id": patent_id,
        "examples_route_count": len(examples),
        "tables_route_count": len(tables),
        "ms_stubs_added": n_stub_added,
        "merged_unique_compounds": len(example_index),
        "n_with_iupac": sum(1 for r in example_index.values() if r.iupac_name),
        "n_with_smiles": sum(1 for r in example_index.values() if r.canonical_smiles),
        "n_with_inchikey": sum(1 for r in example_index.values() if r.inchikey),
        "n_structure_only": sum(1 for r in example_index.values() if r.is_structure_only),
        **assay_audit,
    }
    return {
        "example_index": example_index,
        "assay_tables": assay_tables,
        "audit": audit,
    }


def write_text_index(patent_id: str, out_dir: Path | None = None) -> dict[str, Path]:
    """Run the full extraction pipeline + persist outputs.

    Migrated to call the new `process_patent` orchestrator (Plan v3).
    The legacy `extract_text_index` body is retained as
    `_extract_text_index_legacy` below in case anyone needs the old
    text-only path; new code should call `process_patent` directly.
    """
    if out_dir is None:
        out_dir = config.OUTPUT_DIR / "text_extraction" / patent_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the new orchestrator (text vs Markush routing + retry pass)
    from .process_patent import process_patent
    process_patent(patent_id, out_dir=out_dir, write=True)
    return {
        "example_index": out_dir / "example_index.json",
        "assay_tables": out_dir / "assay_tables.json",
        "audit": out_dir / "route_audit.json",
    }


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run v2 text-side extraction (GP route).")
    ap.add_argument("--patent", action="append", required=True,
                    help="Patent ID (repeatable). Example: --patent US9718825")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    print(f"{'patent':>14}  {'route':>16}  {'merged':>7}  {'iupac':>6}  "
          f"{'smiles':>7}  {'assays':>7}  {'retried':>8}  {'recovered':>10}")
    print("-" * 90)
    for pat in args.patent:
        paths = write_text_index(pat)
        audit = json.loads(paths["audit"].read_text())
        print(f"{pat:>14}  "
              f"{audit.get('route_class','?'):>16}  "
              f"{audit.get('merged_unique_compounds', 0):>7}  "
              f"{audit.get('n_with_iupac', 0):>6}  "
              f"{audit.get('n_with_smiles', 0):>7}  "
              f"{audit.get('asy_compounds', 0):>7}  "
              f"{audit.get('retry_attempts', 0):>8}  "
              f"{audit.get('retry_recovered', 0):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
