"""The Strategy-5 OPSIN-fallback batch must never pay to convert a mass reading.

Measured on US10273259 (2026-08-04): the patent's own description carries 1,036
occurrences of a bare mass reading in `NNN.N (M + H) +` form, 643 of them
distinct. When a discovered IUPAC pattern captures that column — and one in
`data/iupac_patterns.discoveries.json` did, its own description conceding "No
IUPAC names are present in this snippet — only spectral data is provided" —
every one of those readings reaches `_strategy5_after_classification`, fails
OPSIN (it is not a name), and becomes a paid LLM request asking for the SMILES
of a number. The shipped artifacts show what comes back: of 17,786 records,
214 carry an `iupac_name` that is not a name and 47 of those carry a
`canonical_smiles` invented for it. 56 of the 214, across six patents, are
literal mass readings — US10273259's `654.1 2.39 (M + H)+` among them.

Four separate holes, one per test below:

  * nothing checks that the string is a NAME before the batch is built,
  * nothing collapses duplicates (447 distinct across 721 requests in the run
    that motivated this file),
  * the acceptance gate is `validate_smiles(cand)` alone, where the sibling
    path `core/iupac_to_smiles._llm_direct_smiles` also enforces
    MIN_SMILES_MW=150 and MIN_SMILES_LENGTH=10, and
  * nothing at the write point stops a mass reading being stored as a name,
    so a future pattern repeating the mistake ships the same records again.

ZERO paid calls: `anthropic.Anthropic` is replaced by a raiser, so a leak is an
ERROR rather than a silent degradation, and the batch entry point is a recorder.
"""
from __future__ import annotations

import json

import pytest

from patentdb.routes import process_patent as pp


# ── Real strings, with provenance ─────────────────────────────────

# Verbatim from US10273259's description (the `(M + H) +` column), plus the one
# that shipped into its example_index.json as an `iupac_name` with a SMILES.
MASS_READINGS = [
    "496.1 (M + H) +",
    "500.2 (M + H) +",
    "165.9 (M + Na) +",
    "654.1 2.39 (M + H)+",
    "654.2 (M + H) +",
]

# Verbatim from US11292791 / US10246453 example_index.json — characterisation
# data that shipped in the `iupac_name` field.
SPECTRA_READINGS = [
    "1H-NMR (CDCl,400 MHz) δ (ppm): 7.40-7.26 (m,7H),4.81-4.77 (m,1H)",
    "1.89-1.78 (m,1H).",
    "7.40 (s,1H), 7.00 (d,J= 2.2",
    "465.2 (400 MHz,CDOD)8 8.72 (d,",
]

# Verbatim from shipped `strategy5_iupac_harvest_broad` records (US9302989,
# US11254686, US9745328). These are what the batch exists to recover.
GENUINE_NAMES = [
    "4-chloro-N-{4-[(3-phenylpropyl)carbamoyl]phenyl}-1,3-dihydro-2H-isoindole-2-carboxamide",
    "tert-Butyl (S)-3-(2-((1-(pyridin-2-yl)ethyl)amino)thieno[3,2-d]pyrimidine-4-carboxamido)azetidine-1-carboxylate",
    "5-{3-Fluoro-2'-[(4-pyrazin-2-ylpiperazin-1-yl)sulfonyl]biphenyl-4-yl}pyrazin-2-amine",
]

# A drug-like SMILES (MW 285, 21 chars) and two that the sibling path's floors
# exist to reject: ethanol (MW 46) and iodine (MW 127.9, 1 char).
DRUGLIKE_SMILES = "CC(=O)Nc1ccc(cc1)S(=O)(=O)Nc1ccccn1"
TOO_LIGHT_SMILES = "CCO"
TOO_SHORT_SMILES = "I"


@pytest.fixture(autouse=True)
def _no_paid_calls(monkeypatch):
    """A real API call is an ERROR here, not a degraded result."""
    import anthropic

    def _blocked(*a, **k):
        raise AssertionError("PAID CALL: anthropic.Anthropic() was constructed")

    monkeypatch.setattr(anthropic, "Anthropic", _blocked)
    import patentdb.core.api_client as api
    monkeypatch.setattr(api, "call_claude_text", _blocked, raising=False)


def _drive_batch(monkeypatch, new_pairs, *, opsin_parses=(), batch_reply=None):
    """Run `_strategy5_after_classification` over `new_pairs` and return
    `(requested_names, example_index)`.

    `iupac_burst` is the LLM stage upstream of the batch; stubbing it lets the
    test hand the function the exact pairs a pattern capture would produce.
    OPSIN is stubbed rather than run because 700 JVM subprocesses is a
    four-minute test — `opsin_parses` names the strings it should succeed on.
    """
    requested: list[str] = []

    monkeypatch.setattr(pp, "iupac_burst", lambda **kw: dict(new_pairs))

    from patentdb.core.assay_fsm import iupac_pattern_library as ipl
    monkeypatch.setattr(
        ipl.IupacPatternLibrary, "apply_patterns_to_text",
        lambda self, text, **kw: [], raising=True,
    )

    import patentdb.core.iupac_to_smiles as its
    monkeypatch.setattr(
        its, "_try_opsin",
        lambda name, **kw: ((DRUGLIKE_SMILES, "") if name in opsin_parses
                            else (None, "opsin: no parse")),
    )

    import patentdb.core.api_client as api

    def _record(reqs, patent_id="", **kw):
        for r in reqs:
            requested.append(r["cache_key"].split("smiles_fallback:", 1)[1])
        return [batch_reply] * len(reqs)

    monkeypatch.setattr(api, "call_claude_text_batch", _record)

    example_index: dict = {}
    pp._strategy5_after_classification(
        "US10273259", "irrelevant — iupac_burst is stubbed", example_index, 0,
    )
    return requested, example_index


# ── 1. a mass reading must never become a paid request ────────────


def test_mass_readings_are_never_paid_for(monkeypatch):
    pairs = {str(100 + i): s for i, s in enumerate(MASS_READINGS + SPECTRA_READINGS)}
    requested, index = _drive_batch(monkeypatch, pairs)
    assert requested == [], (
        f"{len(requested)} paid requests for non-names, e.g. {requested[:3]}"
    )
    assert index == {}


def test_genuine_names_are_still_paid_for(monkeypatch):
    """The gate must not buy its silence by dropping the real work. These three
    are shipped `strategy5_iupac_harvest_broad` names; OPSIN is stubbed to fail
    on them, exactly the case the batch exists for."""
    pairs = {str(200 + i): s for i, s in enumerate(GENUINE_NAMES)}
    requested, _ = _drive_batch(monkeypatch, pairs)
    assert sorted(requested) == sorted(GENUINE_NAMES)


def test_mixed_batch_keeps_only_the_names(monkeypatch):
    pairs = {}
    for i, s in enumerate(MASS_READINGS + SPECTRA_READINGS + GENUINE_NAMES):
        pairs[str(300 + i)] = s
    requested, _ = _drive_batch(monkeypatch, pairs)
    assert sorted(requested) == sorted(GENUINE_NAMES)


# ── 2. duplicates are one question, not N ─────────────────────────


def test_identical_names_are_requested_once(monkeypatch):
    """447 distinct names across 721 requests in the run that motivated this
    file. The cache_key is per-name, so the duplicates are not even different
    questions — they are the same question asked 274 more times."""
    name = GENUINE_NAMES[0]
    pairs = {str(400 + i): name for i in range(8)}
    requested, index = _drive_batch(
        monkeypatch, pairs, batch_reply=DRUGLIKE_SMILES,
    )
    assert requested == [name], f"{len(requested)} requests for 1 distinct name"
    # Deduping the QUESTION must not dedupe the ANSWER: all 8 cids still
    # resolve, or the fix would be a coverage regression.
    assert len(index) == 8
    assert all(r["canonical_smiles"] == DRUGLIKE_SMILES for r in index.values())


# ── 3. the floors the sibling path already enforces ───────────────


@pytest.mark.parametrize("bad", [TOO_LIGHT_SMILES, TOO_SHORT_SMILES])
def test_acceptance_gate_enforces_mw_and_length_floors(monkeypatch, bad):
    """`core/iupac_to_smiles._llm_direct_smiles` rejects a returned SMILES
    below MIN_SMILES_MW=150 / MIN_SMILES_LENGTH=10. This path accepted anything
    `validate_smiles` liked, so a model that answered a bad question with
    `CCO` had its answer stored as the compound."""
    pairs = {"500": GENUINE_NAMES[0]}
    _, index = _drive_batch(monkeypatch, pairs, batch_reply=bad)
    assert index == {}, f"stored {bad!r} as a compound"


def test_acceptance_gate_keeps_a_druglike_answer(monkeypatch):
    pairs = {"501": GENUINE_NAMES[0]}
    _, index = _drive_batch(monkeypatch, pairs, batch_reply=DRUGLIKE_SMILES)
    assert index["501"]["canonical_smiles"] == DRUGLIKE_SMILES


# ── 4. the write point: no mass reading may ship as an iupac_name ──


def test_write_point_drops_a_record_whose_name_is_a_mass_reading(tmp_path):
    """US10273259 shipped `654.1 2.39 (M + H)+` as an `iupac_name` WITH a
    SMILES. The structure was derived from that string, so the structure is
    unfounded and the record goes."""
    index = {
        "1": {
            "compound_id": "Cpd. No. 1",
            "iupac_name": "654.1 2.39 (M + H)+",
            "canonical_smiles": DRUGLIKE_SMILES,
            "extraction_method": "strategy5_iupac_harvest_broad",
        },
    }
    pp._write_outputs("US10273259", index, {}, {}, out_dir=tmp_path)
    written = json.loads((tmp_path / "example_index.json").read_text())
    assert written == {}


def test_write_point_keeps_an_independently_sourced_structure(tmp_path):
    """A GP-embedded SMILES is not derived from the name, so the molecule
    stands and only the false name is cleared — dropping it would throw away a
    structure the mass reading had nothing to do with."""
    index = {
        "2": {
            "compound_id": "Cpd. No. 2",
            "iupac_name": "496.1 (M + H) +",
            "canonical_smiles": DRUGLIKE_SMILES,
            "extraction_method": "gp_embedded_meta",
        },
    }
    pp._write_outputs("US10273259", index, {}, {}, out_dir=tmp_path)
    written = json.loads((tmp_path / "example_index.json").read_text())
    assert list(written) == ["2"]
    assert not written["2"]["iupac_name"]
    assert written["2"]["canonical_smiles"] == DRUGLIKE_SMILES


def test_write_point_leaves_real_names_alone(tmp_path):
    index = {
        str(i): {
            "compound_id": f"Cpd. No. {i}",
            "iupac_name": nm,
            "canonical_smiles": DRUGLIKE_SMILES,
            "extraction_method": "strategy5_iupac_harvest_broad",
        }
        for i, nm in enumerate(GENUINE_NAMES)
    }
    pp._write_outputs("US9302989", index, {}, {}, out_dir=tmp_path)
    written = json.loads((tmp_path / "example_index.json").read_text())
    assert [r["iupac_name"] for r in written.values()] == GENUINE_NAMES


def test_write_point_keeps_a_name_that_merely_trails_its_own_ms_data(tmp_path):
    """`terminate_name` cuts the data off a real name. Only a chunk that was
    characterisation data from character zero is rejected — a name followed by
    its MS line is a name."""
    nm = GENUINE_NAMES[0] + " LCMS (m/z) (M+H)=528.3, Rt 1.21 min"
    index = {
        "9": {
            "compound_id": "Cpd. No. 9",
            "iupac_name": nm,
            "canonical_smiles": DRUGLIKE_SMILES,
            "extraction_method": "strategy5_iupac_harvest_broad",
        },
    }
    pp._write_outputs("US9302989", index, {}, {}, out_dir=tmp_path)
    written = json.loads((tmp_path / "example_index.json").read_text())
    assert written["9"]["iupac_name"] == nm


# ── 5. the library must not store a capture the model disclaimed ───


def _library(tmp_path):
    from patentdb.core.assay_fsm.iupac_pattern_library import IupacPatternLibrary
    return IupacPatternLibrary(
        canonical_path=tmp_path / "iupac_patterns.json",
        discoveries_path=tmp_path / "iupac_patterns.discoveries.json",
    )


DISCLAIMER = (
    "Compound numbers appear as standalone integers on their own line, "
    "followed by NMR and LCMS characterization data. No IUPAC names are "
    "present in this snippet — only spectral data is provided per compound "
    "number."
)
CAPTURE = r"^(?P<cid>\d{3,4})\s*\n\s*(?P<iupac>(?!\s*1\s*H)[^\n]{10,})"


def test_a_self_disclaimed_capture_is_never_runtime_active(tmp_path):
    """The verbatim discovery that shipped 720 of 721 mass readings into the
    paid batch. Its description is the model saying the region holds no names;
    the regex is the model filling in the output format anyway. `list_pending`
    is applied at runtime, so `pending` here means live."""
    from patentdb.core.assay_fsm.iupac_pattern_library import IupacPattern
    lib = _library(tmp_path)
    lib.add_discovery(
        IupacPattern(
            pattern_name="OTHER",
            description=DISCLAIMER,
            regex_or_heuristic=CAPTURE,
        ),
        fingerprint="f303acee5bee87f3",
        patent_id="US11292791",
    )
    assert lib.list_pending() == []
    assert lib.runtime_active() == []
    # Recorded, not dropped: the observation stays for curation.
    stored = json.loads((tmp_path / "iupac_patterns.discoveries.json").read_text())
    assert [p["status"] for p in stored["patterns"]] == ["rejected"]


def test_repeated_self_disclaimed_proposals_never_auto_promote(tmp_path):
    """The promotion gate is ≥3 fingerprints and ≥3 observations. A pattern
    that is wrong on one patent is wrong on three."""
    from patentdb.core.assay_fsm.iupac_pattern_library import IupacPattern
    lib = _library(tmp_path)
    for i in range(4):
        lib.add_discovery(
            IupacPattern(
                pattern_name="OTHER",
                description=DISCLAIMER,
                regex_or_heuristic=CAPTURE,
            ),
            fingerprint=f"fp{i}",
            patent_id=f"US{9000000 + i}",
        )
    assert lib.runtime_active() == []
    assert lib.list_pending() == []


def test_an_ordinary_discovery_still_goes_pending(tmp_path):
    """The refusal is narrow. A discovery that claims to capture names is
    stored and applied exactly as before — this is not a ban on discovery."""
    from patentdb.core.assay_fsm.iupac_pattern_library import IupacPattern
    lib = _library(tmp_path)
    lib.add_discovery(
        IupacPattern(
            pattern_name="HEADER_PREFIX",
            description="'Example <n>' followed by the IUPAC name on the next line.",
            regex_or_heuristic=r"Example\s+(?P<cid>\d+)\s+(?P<iupac>[A-Z0-9(][^\n]{20,})",
        ),
        fingerprint="fp0",
        patent_id="US9302989",
    )
    assert [p.pattern_name for p in lib.list_pending()] == ["HEADER_PREFIX"]


def test_the_shipped_library_has_no_self_disclaimed_capture_live():
    """Guards the revoke in `data/iupac_patterns.discoveries.json` itself."""
    from patentdb.core.assay_fsm.iupac_pattern_library import (
        IupacPatternLibrary, _disclaims_names, _is_capture_regex,
    )
    lib = IupacPatternLibrary.default()
    live = lib.runtime_active() + lib.list_pending()
    offenders = [
        p.pattern_name for p in live
        if _disclaims_names(p.description) and _is_capture_regex(p.regex_or_heuristic)
    ]
    assert offenders == []
