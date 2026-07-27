"""Vocabulary loader unit tests.

Verifies the contract that the FSM tokenizer + vocab learner depend on:
  - Canonical vocab loads cleanly with all expected token classes.
  - Patent-leak guard rejects entries with US-style patent IDs.
  - Discovery lifecycle: pending → auto_loaded at ≥3 fingerprints.
  - Curator promote/reject moves entries between files.
  - mtime-tracked reload picks up file changes mid-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from patentdb.core.assay_fsm.vocabulary import (
    AssayVocabulary,
    TokenClass,
    VocabEntry,
    _contains_patent_leak,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def temp_vocab(tmp_path: Path) -> AssayVocabulary:
    """Minimal canonical vocab + empty discoveries file."""
    canonical = tmp_path / "vocab.json"
    discoveries = tmp_path / "vocab.discoveries.json"
    canonical.write_text(json.dumps({
        "schema_version": "1.0",
        "tokens": [
            {
                "lemma": "IC50", "class": "ASSAY_TYPE",
                "surface_forms": ["IC50", "IC 50", "IC₅₀"],
            },
            {
                "lemma": "approx", "class": "QUALIFIER",
                "surface_forms": ["~", "≈"], "maps_to": "~",
            },
            {
                "lemma": "uM", "class": "UNIT",
                "surface_forms": ["μM", "uM"],
                "canonical": "μM", "nM_factor": 1000.0,
            },
            {
                "lemma": "n_runs_paren", "class": "RUN_COUNT",
                "regex": r"\((\d{1,3})\)",
                "capture_group": 1,
            },
        ],
    }))
    discoveries.write_text(json.dumps({"schema_version": "1.0", "tokens": []}))
    return AssayVocabulary.load(canonical, discoveries)


# ── Canonical loading ────────────────────────────────────────────


def test_load_canonical_entries(temp_vocab: AssayVocabulary) -> None:
    assert len(temp_vocab.entries) == 4
    classes = set(temp_vocab.all_classes())
    assert {"ASSAY_TYPE", "QUALIFIER", "UNIT", "RUN_COUNT"} <= classes


def test_lookup_by_surface(temp_vocab: AssayVocabulary) -> None:
    e = temp_vocab.lookup_surface("IC50")
    assert e is not None
    assert e.lemma == "IC50"
    assert e.klass == TokenClass.ASSAY_TYPE


def test_regex_entry_compiled(temp_vocab: AssayVocabulary) -> None:
    """RUN_COUNT regex entry must end up in regex_entries (compiled)."""
    n_runs_entries = [
        (pat, ent) for pat, ent in temp_vocab.regex_entries
        if ent.lemma == "n_runs_paren"
    ]
    assert len(n_runs_entries) == 1
    pat, ent = n_runs_entries[0]
    m = pat.search("0.0038 (8)")
    assert m is not None
    assert m.group(1) == "8"


def test_real_canonical_vocab_loads(tmp_path: Path) -> None:
    """The shipped canonical vocab must load without errors and have
    coverage for the four token classes the FSM depends on.
    """
    real = Path(__file__).resolve().parent.parent / "data" / "assay_vocabulary.json"
    assert real.exists(), f"shipped canonical vocab missing: {real}"
    v = AssayVocabulary.load(real)
    classes = set(v.all_classes())
    assert TokenClass.ASSAY_TYPE in classes
    assert TokenClass.QUALIFIER in classes
    assert TokenClass.UNIT in classes
    assert TokenClass.NULL_MARKER in classes
    assert TokenClass.RUN_COUNT in classes


# ── Patent-leak guard ────────────────────────────────────────────


def test_patent_leak_detector_us_id() -> None:
    assert _contains_patent_leak("US10899738")
    assert _contains_patent_leak("us10899738")
    assert _contains_patent_leak("Some US10899738 prefix")


def test_patent_leak_detector_wo_id() -> None:
    assert _contains_patent_leak("WO1997/039355")


def test_patent_leak_detector_clean() -> None:
    assert not _contains_patent_leak("IC50")
    assert not _contains_patent_leak("0.0038")
    assert not _contains_patent_leak("μM")


def test_load_rejects_patent_leak_entry(tmp_path: Path) -> None:
    canonical = tmp_path / "v.json"
    canonical.write_text(json.dumps({
        "tokens": [
            {"lemma": "compound_US10899738", "class": "ASSAY_TYPE",
             "surface_forms": ["IC50"]},
        ],
    }))
    v = AssayVocabulary.load(canonical, tmp_path / "v.discoveries.json")
    # Bad entry was skipped; vocab is empty
    assert len(v.entries) == 0


# ── Discovery lifecycle ─────────────────────────────────────────


def test_discovery_pending_then_auto_loaded(temp_vocab: AssayVocabulary) -> None:
    # First observation — status pending
    temp_vocab.add_discovery(
        lemma="weird_qualifier_1",
        klass=TokenClass.QUALIFIER,
        surface="≪≪",
        fingerprint="fp1",
        patent_id="USXXXXXXXX",
    )
    pending = temp_vocab.list_pending()
    assert any(t["lemma"] == "weird_qualifier_1" and t["status"] == "pending" for t in pending)

    # Second observation, different fingerprint
    temp_vocab.add_discovery(
        lemma="weird_qualifier_1",
        klass=TokenClass.QUALIFIER,
        surface="≪≪",
        fingerprint="fp2",
        patent_id="USYYYYYYYY",
    )
    pending = temp_vocab.list_pending()
    entry = next(t for t in pending if t["lemma"] == "weird_qualifier_1")
    assert entry["status"] == "pending"  # still <3 fingerprints
    assert len(entry["fingerprints_observed"]) == 2

    # Third observation — flips to auto_loaded
    temp_vocab.add_discovery(
        lemma="weird_qualifier_1",
        klass=TokenClass.QUALIFIER,
        surface="≪≪",
        fingerprint="fp3",
        patent_id="USZZZZZZZZ",
    )
    pending = temp_vocab.list_pending()
    entry = next(t for t in pending if t["lemma"] == "weird_qualifier_1")
    assert entry["status"] == "auto_loaded"
    assert entry["n_observations"] >= 3


def test_discovery_rejects_patent_leak(temp_vocab: AssayVocabulary) -> None:
    temp_vocab.add_discovery(
        lemma="bad_US10899738",   # patent id in lemma
        klass=TokenClass.QUALIFIER,
        surface="???",
        fingerprint="fp1",
        patent_id="USXXXXXXXX",
    )
    pending = temp_vocab.list_pending()
    assert not any(t["lemma"] == "bad_US10899738" for t in pending)


def test_curator_promote(temp_vocab: AssayVocabulary) -> None:
    # Add and auto-load
    for fp in ("fp1", "fp2", "fp3"):
        temp_vocab.add_discovery(
            lemma="new_qualifier",
            klass=TokenClass.QUALIFIER,
            surface="≪≪",
            fingerprint=fp,
            patent_id="USXXXXXXXX",
        )
    assert temp_vocab.promote_to_canonical("new_qualifier") is True

    # Reload and verify it's in canonical now
    temp_vocab.reload(force=True)
    found = [e for e in temp_vocab.entries if e.lemma == "new_qualifier"]
    assert len(found) == 1
    assert found[0].klass == TokenClass.QUALIFIER


def test_curator_reject(temp_vocab: AssayVocabulary) -> None:
    temp_vocab.add_discovery(
        lemma="bogus",
        klass=TokenClass.QUALIFIER,
        surface="zzz",
        fingerprint="fp1",
        patent_id="USXXXXXXXX",
    )
    assert temp_vocab.reject("bogus") is True
    pending = temp_vocab.list_pending()
    # rejected entries are NOT in pending list
    assert not any(t["lemma"] == "bogus" for t in pending)


# ── Reload semantics ────────────────────────────────────────────


def test_reload_no_change_returns_false(temp_vocab: AssayVocabulary) -> None:
    assert temp_vocab.reload() is False


def test_reload_after_canonical_change(temp_vocab: AssayVocabulary) -> None:
    import time

    # Pause to ensure mtime delta on filesystems with 1s resolution
    time.sleep(1.05)

    canonical_path = temp_vocab.canonical_path
    data = json.loads(canonical_path.read_text())
    data["tokens"].append({
        "lemma": "NewToken",
        "class": "ASSAY_TYPE",
        "surface_forms": ["NewToken"],
    })
    canonical_path.write_text(json.dumps(data))

    assert temp_vocab.reload() is True
    assert any(e.lemma == "NewToken" for e in temp_vocab.entries)
