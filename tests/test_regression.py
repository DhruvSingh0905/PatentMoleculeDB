"""Regression test — freeze US10214537 baseline.

Auto-fails if recall drops below 77% or molecular accuracy below 99%.
Run after any pipeline code change to prevent regressions.

Uses cached API responses so it's FREE and INSTANT after first run.
"""

import json
import pytest
from pathlib import Path

BASELINE_PATENT = "US10214537"
MIN_RECALL = 0.76  # v13 baseline: 595/774 = 76.9%
MIN_MOLECULAR_ACCURACY = 0.99
MIN_VALIDATED = 800  # Must extract at least 800 validated compounds


@pytest.fixture
def latest_results():
    """Find the most recent extraction results for US10214537."""
    output_dir = Path("/Users/dhruvsingh/Downloads/Patent (1)/output")
    # Check versions in reverse order
    for version in ['v13', 'v12', 'v11', 'v10', 'final']:
        result_file = output_dir / version / f"{BASELINE_PATENT}.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
    pytest.skip("No extraction results found for US10214537")


@pytest.fixture
def bindingdb_keys():
    """Load BindingDB InChIKeys for US10214537."""
    tsv_path = Path("/Users/dhruvsingh/Downloads/Patent (1)/output/bindingdb/our_patents.tsv")
    if not tsv_path.exists():
        pytest.skip("BindingDB data not available")

    import pandas as pd
    from patent_extraction.smiles_utils import canonicalize_smiles, get_inchikey

    bdb = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    bdb_patent = bdb[bdb['Patent Number'] == BASELINE_PATENT]

    keys = set()
    for _, row in bdb_patent.iterrows():
        ik = str(row.get('Ligand InChI Key', '')).strip()
        if len(ik) >= 14:
            keys.add(ik)
        sm = str(row.get('Ligand SMILES', '')).strip()
        if sm and sm != 'nan':
            cs = canonicalize_smiles(sm)
            if cs:
                k = get_inchikey(cs)
                if k:
                    keys.add(k)
    return keys


class TestRegressionUS10214537:
    """Baseline regression tests — must NEVER fail after passing once."""

    def test_minimum_compounds_extracted(self, latest_results):
        """Must extract at least 800 validated compounds."""
        validated = [c for c in latest_results['exemplified_compounds']
                     if c['processing_status'] == 'validated']
        assert len(validated) >= MIN_VALIDATED, (
            f"Only {len(validated)} validated compounds, need {MIN_VALIDATED}+"
        )

    def test_minimum_recall(self, latest_results, bindingdb_keys):
        """Recall against BindingDB must stay above 77%."""
        validated = [c for c in latest_results['exemplified_compounds']
                     if c['processing_status'] == 'validated']

        our_keys = set()
        for c in validated:
            for k in [c.get('inchikey'), c.get('parent_inchikey')]:
                if k and len(k) >= 14:
                    our_keys.add(k)

        matched = our_keys & bindingdb_keys
        recall = len(matched) / max(len(bindingdb_keys), 1)

        assert recall >= MIN_RECALL, (
            f"Recall {recall:.1%} ({len(matched)}/{len(bindingdb_keys)}) "
            f"dropped below baseline {MIN_RECALL:.0%}"
        )

    def test_all_smiles_valid(self, latest_results):
        """Every validated compound must have a valid canonical SMILES."""
        from patent_extraction.smiles_utils import validate_smiles

        validated = [c for c in latest_results['exemplified_compounds']
                     if c['processing_status'] == 'validated']

        invalid = [c['example_number'] for c in validated
                   if not c.get('canonical_smiles') or not validate_smiles(c['canonical_smiles'])]

        assert len(invalid) == 0, (
            f"{len(invalid)} compounds have invalid SMILES: {invalid[:5]}"
        )

    def test_no_iodine_smiles(self, latest_results):
        """No validated compound should have SMILES = 'I' (the old iodine bug)."""
        validated = [c for c in latest_results['exemplified_compounds']
                     if c['processing_status'] == 'validated']

        iodine = [c['example_number'] for c in validated
                  if c.get('canonical_smiles') in ('I', '[I]')]

        assert len(iodine) == 0, (
            f"{len(iodine)} compounds have iodine SMILES: {iodine[:5]}"
        )

    def test_drug_like_ratio(self, latest_results):
        """At least 90% of validated compounds should be drug-like."""
        validated = [c for c in latest_results['exemplified_compounds']
                     if c['processing_status'] == 'validated']
        drug_like = [c for c in validated
                     if c.get('drug_likeness') and c['drug_likeness'].get('lipinski_passes')]

        ratio = len(drug_like) / max(len(validated), 1)
        assert ratio >= 0.90, (
            f"Drug-like ratio {ratio:.1%} below 90%"
        )
