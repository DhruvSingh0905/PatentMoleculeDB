"""Per-Example joiner — explainability layer over the structured-provenance Compound list.

Given the post-pipeline state (final compounds + enumerated + manifest), produces
a per-Example summary so a human can answer:
- Which Example IDs got a structure?
- From which routes (text/image/Markush)?
- Which Examples are still unrecovered?
- For Markush-enumerated structures: which scaffold + R-assignments produced them?

Persisted to `output/results/{patent_id}/per_example.json`.

Patent-agnostic: works on any patent whose manifest has `example_to_page`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Compound, PageManifest

logger = logging.getLogger(__name__)


def _normalize_id(s: str | None) -> str:
    """Normalize an Example/Cpd.No ID for fuzzy matching across routes."""
    if not s:
        return ""
    return s.strip().lower().replace(".", "").replace(" ", "")


def build_per_example_view(
    compounds: list,
    enumerated: list,
    manifest,
) -> dict[str, dict]:
    """Produce a per-Example summary from structured provenance.

    Returns:
        {
            "Example 5": {
                "page": 29,
                "structures": [
                    {"smiles": "...", "route": "google_patents_example",
                     "stage": "opsin_direct", "scaffold_inchikey": null, ...},
                    ...
                ],
                "sources_present": ["google_patents_example", "image_pipeline"],
                "n_structures": 2,
                "has_structure": true,
            },
            ...
        }
    """
    # Initialize from manifest's example→page index — every known Example
    # gets a row even if no extractor recovered it.
    view: dict[str, dict] = {}
    example_to_page = getattr(manifest, 'example_to_page', None) or {}
    for ex_id, page in example_to_page.items():
        view[ex_id] = {
            "page": page,
            "structures": [],
            "sources_present": [],
            "n_structures": 0,
            "has_structure": False,
        }

    # Index: normalized example_id → canonical example_id
    norm_to_canonical = {_normalize_id(k): k for k in view}

    def _attribute(c, ex_id_canonical: str | None):
        """Append a structure entry to view[ex_id_canonical]."""
        if not ex_id_canonical or ex_id_canonical not in view:
            return
        prov = getattr(c, 'provenance', None)
        chain = getattr(c, 'provenance_chain', None) or []
        # Primary provenance entry
        entries = [prov] if prov else []
        entries.extend(chain)
        # Build flat structure record
        struct = {
            "smiles": c.canonical_smiles,
            "route": prov.route if prov else (c.extraction_method or "unknown"),
            "stage": prov.stage if prov else None,
            "page": prov.page if prov else c.source_page,
            "scaffold_inchikey": prov.scaffold_inchikey if prov else None,
            "scaffold_index": prov.scaffold_index if prov else None,
            "r_assignments": prov.r_assignments if prov else None,
            "enumeration_mode": prov.enumeration_mode if prov else None,
            "scoring_rank": prov.scoring_rank if prov else None,
            "image_path": prov.image_path if prov else c.image_path,
            "extra_routes_in_chain": [
                {"route": e.route, "stage": e.stage} for e in chain
            ],
        }
        view[ex_id_canonical]["structures"].append(struct)
        # Track unique routes that touched this Example
        all_routes = {struct["route"]}
        all_routes.update(e.route for e in chain)
        existing_sources = set(view[ex_id_canonical]["sources_present"])
        view[ex_id_canonical]["sources_present"] = sorted(existing_sources | all_routes)

    # Attribute exemplified compounds
    for c in compounds + enumerated:
        if not c.canonical_smiles:
            continue
        ex_norm = _normalize_id(c.example_number)
        canonical = norm_to_canonical.get(ex_norm)
        if canonical is None:
            # Compound has an example_number that isn't in the manifest
            # (e.g., "Cpd. No. 148" or "Markush(R1=...)" — Markush IDs won't match
            # an Example header). Add a synthetic row for it so the joiner
            # output is comprehensive.
            synthetic_id = c.example_number or "unknown"
            if synthetic_id not in view:
                view[synthetic_id] = {
                    "page": c.source_page,
                    "structures": [],
                    "sources_present": [],
                    "n_structures": 0,
                    "has_structure": False,
                    "synthetic": True,   # mark as not from manifest example_to_page
                }
                norm_to_canonical[_normalize_id(synthetic_id)] = synthetic_id
            _attribute(c, synthetic_id)
        else:
            _attribute(c, canonical)

    # Finalize derived fields
    for ex_id, row in view.items():
        row["n_structures"] = len(row["structures"])
        row["has_structure"] = row["n_structures"] > 0

    return view


def per_example_stats(view: dict[str, dict]) -> dict:
    """High-level counts used by audit summary + bench output."""
    n_total = len(view)
    n_with_struct = sum(1 for v in view.values() if v["has_structure"])
    by_route = {}
    for v in view.values():
        for r in v["sources_present"]:
            by_route[r] = by_route.get(r, 0) + 1
    examples_filled_by_markush_only = sum(
        1 for v in view.values()
        if v["has_structure"]
        and v["sources_present"] == ["markush_enumeration"]
    )
    return {
        "total_examples_in_manifest": n_total,
        "examples_with_structure": n_with_struct,
        "examples_without_structure": n_total - n_with_struct,
        "examples_per_route": by_route,
        "examples_filled_by_markush_only": examples_filled_by_markush_only,
    }
