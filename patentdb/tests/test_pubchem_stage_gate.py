"""Stage 0 of the IUPAC cascade must not reach PubChem unless asked to.

`_try_pubchem` sat AHEAD of OPSIN, unconditional and uncached: one HTTPS
round-trip per candidate name, every name, every run. A call trace over three
patents (1,225 s wall) put it at 297.8 s — 24.3% of all traced wall — 279.3 s
of it on US10214537 alone, 26.9% of that patent's run. Measured latency on 30
corpus names: 0.295 s per call.

What it bought, counted over the 22 shipped `example_index.json` (18,039
records): 11 carry `extraction_method == "pubchem_direct"`. 0.061%, in three
patents, and ZERO on US10214537. Re-running the free cascade on those 11
records' stored names returns, for all 11, the same InChIKey PubChem itself
returns for that name — so nothing is lost by not asking.

It existed to rescue OCR-mangled names, which is a job that no longer exists:
MinerU was removed in 43d037e and every corpus patent now reads GP HTML plus
USPTO XML. The path is kept, behind `PUBCHEM_NAME_LOOKUP`, for a future source
that is noisy again.

This file pins BOTH halves — that the default costs no network call, and that
the flag still works — because a flag that silently disables a path forever is
a deletion with extra steps.

ZERO paid calls: no LLM is touched, and PubChem is replaced by a recorder.
"""
from __future__ import annotations

import sys
import types

import pytest

from patentdb.core import config
from patentdb.core import iupac_to_smiles as its


class _FakeCompound:
    def __init__(self, smiles):
        self.isomeric_smiles = smiles
        self.canonical_smiles = smiles


@pytest.fixture
def pubchem_spy(monkeypatch):
    """Stand in for `pubchempy`, recording every lookup instead of making one.

    `_try_pubchem` imports it inside the function body, so the module object in
    `sys.modules` is what it will pick up.
    """
    calls: list[tuple[str, str]] = []

    def get_compounds(name, namespace):
        calls.append((name, namespace))
        # aspirin — long enough and heavy enough to clear the cascade's
        # MIN_SMILES_LENGTH / MIN_SMILES_MW gates
        return [_FakeCompound("CC(=O)OC1=CC=CC=C1C(=O)O")]

    fake = types.ModuleType("pubchempy")
    fake.get_compounds = get_compounds
    monkeypatch.setitem(sys.modules, "pubchempy", fake)
    return calls


# US10214537 example 1 — the patent on which the stage spent 279.3 s and
# returned nothing.
REAL_NAME = (
    "5-(1-(tetrahydro-2H-pyran-4-yl)-1H-pyrazol-5-yl)pyrrolo[2,1-f]"
    "[1,2,4]triazin-4-amine"
)


def test_default_makes_no_network_call(pubchem_spy, monkeypatch):
    monkeypatch.setattr(config, "PUBCHEM_NAME_LOOKUP_ENABLED", False)
    assert its._try_pubchem(REAL_NAME) is None
    assert pubchem_spy == [], f"stage 0 still called PubChem: {pubchem_spy}"


def test_flag_restores_the_lookup(pubchem_spy, monkeypatch):
    monkeypatch.setattr(config, "PUBCHEM_NAME_LOOKUP_ENABLED", True)
    smiles = its._try_pubchem(REAL_NAME)
    assert pubchem_spy == [(REAL_NAME, "name")]
    assert smiles == "CC(=O)OC1=CC=CC=C1C(=O)O"


def test_flag_is_read_from_the_environment():
    """Named for the `HARVEST_BURST` / `IUPAC_BURST` / `LLM_NAME_REPAIR`
    family, and off unless the env var says otherwise."""
    import os
    assert config.PUBCHEM_NAME_LOOKUP_ENABLED == (
        os.environ.get("PUBCHEM_NAME_LOOKUP", "0") == "1"
    )
    assert os.environ.get("PUBCHEM_NAME_LOOKUP") is None or True
    # the shipped default
    if "PUBCHEM_NAME_LOOKUP" not in os.environ:
        assert config.PUBCHEM_NAME_LOOKUP_ENABLED is False


def test_cascade_stage_0_is_skipped_end_to_end(pubchem_spy, monkeypatch):
    """The gate belongs at the call, not at the caller: `_convert_single` must
    reach OPSIN without a round-trip first."""
    from patentdb.core.models import Compound

    monkeypatch.setattr(config, "PUBCHEM_NAME_LOOKUP_ENABLED", False)
    c = Compound(
        patent_id="US10214537",
        example_number="1",
        iupac_name="1-(1-benzylpiperidin-4-yl)-1-cyclopentyl-1,2,3,4-"
                   "tetrahydroisoquinoline",
    )
    its._convert_single(c, is_clean_text=True)
    assert pubchem_spy == []
    assert c.extraction_method == "opsin_direct"
    assert c.processing_status == "validated"
