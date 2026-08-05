"""Cost-safety tests for the broad IUPAC burst.

Re-extracting US11292791 cost $0.535, of which $0.45 was 11 `iupac_burst`
broad calls that produced ZERO usable compounds. Three defects combined to
produce that bill, and each is pinned below:

  D1  the cap check sat at the BOTTOM of the chunk loop, and the zero-result
      path `continue`d straight past it — so a patent where the LLM finds
      nothing paid the MOST. Measured exposure: 40 chunks x $0.041 = $1.64
      against a $0.20 PER_PATENT_LM_CAP.
  D2  chunks were ranked by paren+bracket count, which on this patent ranked
      the `δ (ppm) … (m, 4H) … LCMS (ES, m/z)` NMR block (~790 parens per
      15 KB, zero compound names) above every example paragraph. All 11
      chunks that fired held ZERO IUPAC-shaped names.
  D3  `_find_cid_context_windows` had no per-cid ceiling and applied its
      loose table-row fallback to 1-character cids, so the single cid `"1"`
      fanned out to 55 paid windows.

These are cost regressions, so they are pinned with a fake LLM and a fake
cost tracker — no test here makes a paid call.
"""
import pytest

from patentdb.core.assay_fsm.harvest import iupac_orchestrator as IO


# ── fakes ──────────────────────────────────────────────────────────


class FakeTracker:
    """Spends a fixed amount per LLM call and reports against a $0.20 cap."""

    def __init__(self, cost_per_call=0.041, cap=0.20):
        self.spend = 0.0
        self.cost_per_call = cost_per_call
        self.cap = cap

    def patent_lm_exceeded(self, patent_id):
        return self.spend >= self.cap

    def patent_spend(self, patent_id):
        return self.spend


class _Result:
    def __init__(self, pairs=None, pattern_meta=None):
        self.pairs = pairs or []
        self.pattern_meta = pattern_meta


class FakeLibrary:
    def add_discovery(self, *a, **k):
        pass


@pytest.fixture
def no_cache(monkeypatch):
    """Isolate from the on-disk rule cache — these tests must not read or
    write `adaptive_extraction_rules.json`."""
    monkeypatch.setattr(IO, "_read_cache", lambda: {})
    monkeypatch.setattr(IO, "_write_cache", lambda data: None)


def _iupac_text(n_names, prefix=""):
    """Text carrying `n_names` IUPAC-shaped compound names."""
    name = ("2-[4-(3-chlorophenyl)piperazin-1-yl]-N-"
            "(pyridin-3-ylmethyl)acetamide")
    return prefix + " ".join(
        f"Example {i}. {name}\n" for i in range(1, n_names + 1)
    )


# ── D1: the cap is checked BEFORE the spend ────────────────────────


def test_cap_stops_the_burst_when_every_chunk_returns_nothing(no_cache, monkeypatch):
    """The exact US11292791 shape: the LLM finds nothing, chunk after chunk.

    Before the fix the zero-pair branch `continue`d past a bottom-of-loop cap
    check, so all 40 chunks fired for $1.64 against a $0.20 cap. The cap now
    sits above the call, so spend stops within one call of the cap.
    """
    tracker = FakeTracker()
    calls = {"n": 0}

    def fake_extract(chunk, patent_id=None):
        calls["n"] += 1
        tracker.spend += tracker.cost_per_call
        return _Result(pairs=[])          # the zero-result path

    monkeypatch.setattr(IO, "extract_iupac_pairs", fake_extract)

    IO.iupac_burst(
        "USTEST", _iupac_text(4000), {}, 100, tracker,
        library=FakeLibrary(), force=True, chunk_size=15_000, max_chunks=40,
    )

    # 5 calls x $0.041 = $0.205 — the first call that crosses the cap is
    # paid for, the 35 after it are not. Pre-fix this was 40 calls / $1.64.
    assert calls["n"] <= 6, (
        f"zero-result chunks fired {calls['n']} LLM calls; the cap must be "
        "consulted BEFORE each call, not after the pairs are inspected"
    )
    assert tracker.spend <= tracker.cap + tracker.cost_per_call


def test_cap_is_checked_before_the_first_call_not_after(no_cache, monkeypatch):
    """A patent already at its cap when the burst starts must not pay at all —
    including under force=True, which bypasses the GAP check and nothing else."""
    tracker = FakeTracker()
    tracker.spend = 0.25          # already over cap on arrival
    calls = {"n": 0}

    def fake_extract(chunk, patent_id=None):
        calls["n"] += 1
        return _Result(pairs=[])

    monkeypatch.setattr(IO, "extract_iupac_pairs", fake_extract)
    IO.iupac_burst(
        "USTEST", _iupac_text(4000), {}, 100, tracker,
        library=FakeLibrary(), force=True, chunk_size=15_000, max_chunks=40,
    )
    assert calls["n"] == 0


def test_cache_hits_still_flow_past_the_in_loop_cap(monkeypatch):
    """The in-loop cap gates NEW spend only.

    It is placed AFTER the cache-hit branch, matching `iupac_burst_targeted`,
    so a chunk already in the cache is free and keeps flowing once the patent
    goes over budget mid-burst. Moving the gate above the cache check would
    silently drop recovery on every re-run of a capped patent.
    """
    text = _iupac_text(4000)
    chunks = IO._chunk_high_density_regions(text, chunk_size=15_000, max_chunks=40)
    assert chunks, "fixture text must produce chunks"
    cache = {
        f"iupac:{IO._content_fingerprint(ch, 'USTEST')}": {
            "pairs": [{"compound_id": f"c{i}", "iupac_name": "propan-2-ol"}],
            "pattern_meta": None,
        }
        for i, (_off, ch) in enumerate(chunks)
    }
    monkeypatch.setattr(IO, "_read_cache", lambda: cache)
    monkeypatch.setattr(IO, "_write_cache", lambda data: None)

    def boom(chunk, patent_id=None):
        raise AssertionError("a cached chunk must never fire the LLM")

    monkeypatch.setattr(IO, "extract_iupac_pairs", boom)

    class OverBudgetAfterEntry(FakeTracker):
        """Under cap at the pre-loop entry gate, over cap inside the loop —
        isolates the in-loop gate from the pre-loop one."""

        def __init__(self):
            super().__init__()
            self.n_checks = 0

        def patent_lm_exceeded(self, patent_id):
            self.n_checks += 1
            return self.n_checks > 1

    tracker = OverBudgetAfterEntry()
    out = IO.iupac_burst(
        "USTEST", text, {}, 100, tracker,
        library=FakeLibrary(), force=True, chunk_size=15_000, max_chunks=40,
    )
    assert out, "cache hits must survive the in-loop cost gate"
    assert tracker.n_checks == 1, (
        "a cache hit must `continue` before the cap check — the gate exists "
        "to stop NEW spend, and a cached chunk spends nothing"
    )


# ── D2: a chunk with no IUPAC names is never bought ────────────────


NMR_BLOCK = (
    "1 H-NMR (CD 3 OD, 400 MHz) δ (ppm): 7.46-7.32 (m, 4H), 4.84-4.71 (m, 1H), "
    "4.70-4.55 (m, 2H), 3.79 (s, 3H), 2.34-2.04 (m, 3H), 1.47 (d, J = 7.2 Hz, "
    "3H), 1.14 (d, J = 6.8 Hz, 3H). LCMS (ES, m/z): 485 [M + H] + . "
)


def test_nmr_block_scores_zero_iupac_shapes():
    """~120 parens per repetition and not one compound name. This is the text
    the paren-density proxy ranked FIRST on US11292791."""
    assert IO._count_iupac_shapes(NMR_BLOCK * 200) == 0


def test_real_iupac_names_score_above_zero():
    """The probe must fire on the names it exists to find — otherwise
    skip-on-zero would silently disable the burst entirely."""
    assert IO._count_iupac_shapes(_iupac_text(10)) >= 10


def test_zero_iupac_chunk_is_not_selected():
    """The core of D2. An NMR-only document has nothing for this burst to
    extract, so it must select NO chunks — at max_chunks=40 the old ranker
    would have handed back 40 guaranteed-empty paid calls."""
    text = NMR_BLOCK * 2000
    assert len(text) > 15_000
    assert IO._chunk_high_density_regions(text, chunk_size=15_000, max_chunks=40) == []


def test_iupac_dense_region_outranks_paren_dense_region():
    """US11292791 in miniature: an NMR block with far more parens than the
    example paragraphs. The names must win."""
    names = _iupac_text(300)
    text = (NMR_BLOCK * 400) + names + (NMR_BLOCK * 400)
    sel = IO._chunk_high_density_regions(text, chunk_size=15_000, max_chunks=3)
    assert sel, "a document containing IUPAC names must select at least one chunk"
    for _off, ch in sel:
        assert IO._count_iupac_shapes(ch) > 0, (
            "no selected chunk may be free of IUPAC names — every such chunk "
            "is a guaranteed-empty paid LLM call"
        )


def test_burst_makes_no_calls_when_no_chunk_has_names(no_cache, monkeypatch):
    """End-to-end on the burst: zero name shapes → zero spend."""
    def boom(chunk, patent_id=None):
        raise AssertionError("fired the LLM on text with no IUPAC names")

    monkeypatch.setattr(IO, "extract_iupac_pairs", boom)
    out = IO.iupac_burst(
        "USTEST", NMR_BLOCK * 2000, {}, 100, FakeTracker(),
        library=FakeLibrary(), force=True, chunk_size=15_000, max_chunks=40,
    )
    assert out == {}


def test_short_text_without_names_is_not_selected():
    """The `len(text) <= chunk_size` shortcut obeys the same rule."""
    assert IO._chunk_high_density_regions(NMR_BLOCK, chunk_size=15_000) == []
    assert IO._chunk_high_density_regions(_iupac_text(3), chunk_size=15_000)


# ── D3: cid fan-out is bounded ─────────────────────────────────────


def test_windows_per_cid_are_bounded():
    """`"1"` matched `1.` / `1:` 74 times in US11292791's XML and fanned out
    to 55 paid 5 KB windows for ONE compound."""
    text = " ".join(f"1. some text here number {i} " * 20 for i in range(60))
    wins = IO._find_cid_context_windows(text, "1", window_chars=5000)
    assert len(wins) <= IO.MAX_WINDOWS_PER_CID


def test_single_char_cid_does_not_use_the_loose_table_row_fallback():
    """`1` followed by two numbers is a table row, not a compound label — it
    matches every row of every numeric table in the document."""
    rows = " ".join(f"{i} {i}.5 {i}00 QC-ACN " for i in range(1, 300))
    assert IO._find_cid_context_windows(rows, "1", window_chars=4000) == []


def test_multi_char_cid_keeps_the_loose_table_row_fallback():
    """The fallback earns its place on dense-table patents like US10544143,
    whose 3-4 digit cids appear nowhere but table rows. Gating it on cid
    length must not disable it there."""
    rows = "prelude text " + " ".join(f"{i} {i}.5 {i}00 QC-ACN " for i in range(100, 400))
    assert IO._find_cid_context_windows(rows, "325", window_chars=4000)


def test_specific_label_matches_are_preferred_over_table_rows():
    """The ceiling truncates, so what survives must be the best evidence.
    Patterns run most-specific-first: an `Example NNN` hit must be kept even
    when hundreds of table rows also match."""
    label = "Example 118 is 2-[4-(3-chlorophenyl)piperazin-1-yl]acetamide. "
    rows = " ".join(f"118 {i}.59 {i}.9 1.42 QC-ACN " for i in range(400))
    wins = IO._find_cid_context_windows(label + rows, "118", window_chars=4000)
    assert len(wins) <= IO.MAX_WINDOWS_PER_CID
    assert "Example 118" in wins[0][1]
