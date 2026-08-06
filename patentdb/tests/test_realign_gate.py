"""The assay realigner and the adaptive rule firer must not spend without a flag.

Two paid paths in `core/assay_fsm/llm_realigner.py` + `core/adaptive_extraction_cache.py`
fired on every patent with no feature flag at all. `43d037e` traced the first
and deliberately left it alone pending this measurement; the second was never
traced.

WHAT `realign_region` COSTS. It is `pipeline._process_region`'s Stage 5, once
per detected region, bounded only by a fingerprint cache and
`PER_PATENT_REALIGN_CAP` ($1.50). Counting chunks over the 22 corpus patents'
GP descriptions with `chunk_region` (no LLM touched): 270 chunks, mean 12.3 per
patent. Priced from the 644 realign-shaped responses in `output_v2/cache`,
whose token counts `api_cache.store_cached` recorded — median $0.0391, mean
$0.0449 per call — that is **$0.55 per new patent, $12.12 for the corpus**, and
US9718790 alone (49 chunks, $2.20) would run into the cap and truncate its own
table.

WHAT IT BUYS. Replaying `extract_for_patent` on all 22 patents with the
realigner on and off (HARVEST_BURST=0, so the free pattern-library gap-fill is
present; zero paid calls, zero blocked calls — every fingerprint miss was
served by the response cache):

  - rows      39,063 -> 41,529  (+2,466). Turning it OFF produces MORE rows:
    its rows occupy (compound, assay) slots that `_merge_into(gap_fill_only=
    True)` then refuses to the free pattern library.
  - compounds with >=1 row  12,461 -> 10,846 (-1,615), and 0 gained.
  - distinct (cid, assay, value) triples: -9,358 / +11,829.

WHAT SURVIVES THE REST OF THE PIPELINE. `extract_for_patent` is one of three
contributors to a shipped `assay_tables.json`; `process_patent.py:2213` appends
the deterministic USPTO CALS baseline unconditionally. Of the 1,615 compounds
that lose every row, **1,610 are already carried by
`sources.uspto_assays.extract_from_patent`** — net 5 compounds, and two of the
five are `D` and `10`. Of the 9,358 lost triples, 8,010 are reproduced by that
same extractor; of the 1,348 that are not, 642 are US11254686's dimensionless
`A2A/A1 ratio` and `A2A IC50 / A2B IC50` rows plus 58 clearances stamped "nM or
assay units" — the exact fabrication `_DIMENSIONLESS` exists to block, and the
patent whose 99 records read 2.24 nM against BindingDB's 300 nM.

So the default is OFF: a spend of up to the $1.50 cap per patent, ungated, for
5 compounds the deterministic reader misses and one large block of ratios the
project has already ruled worse than a missing assay. `ASSAY_REALIGN=1`
restores it for a source the CALS reader cannot handle.

`derive_rule_via_llm` — the second `call_claude_text` — had ZERO callers, along
with `_RULE_PROMPT`, `detect_column_miss`, `ExtractionRule`, `AssayColumn` and
`AdaptiveExtractionCache.get`/`.put`. Nothing to gate; removed, and pinned here
so a future edit cannot quietly restore an ungated firer.

ZERO paid calls in this file: every LLM entry point is replaced by a raiser.
"""
from __future__ import annotations

import inspect

import pytest

from patentdb.core import adaptive_extraction_cache as AEC
from patentdb.core import config
from patentdb.core.assay_fsm import llm_realigner as LR
from patentdb.core.assay_fsm.region_detector import Region


# A region shaped like the tables the realigner reads: compound id + two
# numeric cells per row, with a unit-bearing header line.
REGION_TEXT = (
    "TABLE 1\n"
    "Compound  PI3K delta IC50 value (nM)  CD69 IC50 value (nM)\n"
    "1   0.52   3.1\n"
    "2   1.40   9.8\n"
    "3   0.07   0.4\n"
)


def _region() -> Region:
    return Region(start=0, end=len(REGION_TEXT), text=REGION_TEXT, n_row_hits=3)


@pytest.fixture
def llm_spy(monkeypatch):
    """Record every realigner LLM call instead of making one."""
    calls: list[str] = []

    def _fake(page_text, *a, **k):
        calls.append(page_text)
        return AEC.RealignedTable(
            rows=[AEC.RealignedRow(compound_id="1",
                                   values={"PI3K delta IC50": 0.52})],
            units={"PI3K delta IC50": "nM"},
            notes="fake",
        )

    monkeypatch.setattr(AEC, "realign_table_via_llm", _fake)
    monkeypatch.setattr(LR, "realign_table_via_llm", _fake)
    return calls


@pytest.fixture
def empty_cache(monkeypatch, tmp_path):
    """A cache with nothing in it, that never writes to the shared file."""
    cache = AEC.AdaptiveExtractionCache(cache_path=tmp_path / "rules.json")
    monkeypatch.setattr(AEC.AdaptiveExtractionCache, "_save", lambda self: None)
    return cache


def test_default_fires_no_llm_call(llm_spy, empty_cache, monkeypatch):
    monkeypatch.setattr(config, "ASSAY_REALIGN_ENABLED", False)
    res = LR.realign_region(_region(), patent_id="US10214537", cache=empty_cache)
    assert llm_spy == [], f"realigner still called the LLM: {len(llm_spy)} call(s)"
    assert res.rows == []
    assert res.n_chunks == 0


def test_flag_restores_the_realigner(llm_spy, empty_cache, monkeypatch):
    monkeypatch.setattr(config, "ASSAY_REALIGN_ENABLED", True)
    res = LR.realign_region(_region(), patent_id="US10214537", cache=empty_cache)
    assert len(llm_spy) == 1
    assert [r.compound_id for r in res.rows] == ["1"]


def test_gate_is_inside_the_firer_not_only_its_caller(monkeypatch):
    """`43d037e` moved the `LLM_RECOVERY` check INSIDE `iupac_burst_targeted`
    for exactly this reason: a gate on the caller is undone by the next caller.
    `realign_table_via_llm` must refuse on its own."""
    monkeypatch.setattr(config, "ASSAY_REALIGN_ENABLED", False)

    def _blocked(*a, **k):
        raise AssertionError("paid call reached the API client")

    monkeypatch.setattr(AEC, "call_claude_text", _blocked)
    assert AEC.realign_table_via_llm("Compound 1 0.52 3.1", patent_id="X") is None
    assert "ASSAY_REALIGN_ENABLED" in inspect.getsource(AEC.realign_table_via_llm)


def test_cache_hits_are_gated_too(llm_spy, empty_cache, monkeypatch):
    """A cached fingerprint is free, but it is still this tier's output — and
    the whole point of the measurement is that the free-because-cached rows
    displace the free pattern library's. OFF must mean off for both."""
    monkeypatch.setattr(config, "ASSAY_REALIGN_ENABLED", True)
    hot = LR.realign_region(_region(), patent_id="US10214537", cache=empty_cache)
    assert hot.rows and not hot.cache_hit

    monkeypatch.setattr(config, "ASSAY_REALIGN_ENABLED", False)
    cold = LR.realign_region(_region(), patent_id="US10214537", cache=empty_cache)
    assert cold.rows == [], "gated run still replayed the fingerprint cache"


def test_flag_is_read_from_the_environment():
    """Named for the `HARVEST_BURST` / `IUPAC_BURST` / `PUBCHEM_NAME_LOOKUP`
    family, and off unless the env var says otherwise."""
    import os
    assert config.ASSAY_REALIGN_ENABLED == (
        os.environ.get("ASSAY_REALIGN", "0") == "1"
    )
    if "ASSAY_REALIGN" not in os.environ:
        assert config.ASSAY_REALIGN_ENABLED is False


def test_the_dead_rule_firer_is_gone():
    """`derive_rule_via_llm` and its whole surface had no callers anywhere in
    the tree — the second ungated `call_claude_text` bought a rule nothing
    read. Removed rather than gated; this pins it."""
    for name in ("derive_rule_via_llm", "_RULE_PROMPT", "detect_column_miss",
                 "ExtractionRule", "AssayColumn"):
        assert not hasattr(AEC, name), f"{name} came back into the module"
    for method in ("get", "put"):
        assert not hasattr(AEC.AdaptiveExtractionCache, method), (
            f"AdaptiveExtractionCache.{method} came back — it hydrated an "
            f"ExtractionRule nothing produced"
        )


def test_only_one_paid_entry_point_remains():
    """One `call_claude_text` in the module, and it is behind the flag."""
    src = inspect.getsource(AEC)
    assert src.count("call_claude_text(") == 1, (
        "a second ungated LLM call site reappeared in adaptive_extraction_cache"
    )
