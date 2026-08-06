"""The per-patent baseline ledger, and the protection it is meant to replace.

`baseline_counts()` re-extracted all 137 cached XMLs on every capability call.
Measured on US20240010684A1 — 15 compounds, a gap worth 3 rows — that was
14.04 s of the tier's 39.53 s untraced (47.69 s of 74.68 s under the tracer).

The saving is TEMPORAL, not populational: the corpus is still the population of
the blocking condition, it is just not re-derived while the code that produced
it has not moved. These tests pin both halves — that a warm ledger measures
nothing, and that it still hands back every patent — because scoping the
population is the tempting version and it silently breaks the one condition
`verify_patch` blocks on.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture()
def led(tmp_path, monkeypatch):
    from patentdb.repair import ledger

    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "baselines.json")
    monkeypatch.setattr(ledger, "code_key", lambda: "KEY-A")
    return ledger


def _corpus(tmp_path, pids):
    d = tmp_path / "xml"
    d.mkdir(exist_ok=True)
    for p in pids:
        (d / f"{p}.xml").write_text("<x/>")
    return d


def test_a_warm_ledger_measures_nothing_and_still_returns_everyone(led, tmp_path):
    """The whole point: same answer, no re-extraction.

    A baseline that returned only the patents it re-measured would be cheaper
    and useless — `verify_patch` compares this sum against a corpus-wide probe
    total, so a short population makes the blocking condition compare two
    numbers drawn from different ones.
    """
    xml_dir = _corpus(tmp_path, ["USA", "USB", "USC"])
    seen = []

    def measure(path):
        seen.append(path.stem)
        return {"USA": 10, "USB": 20, "USC": 30}[path.stem], path.stem != "USC"

    first = led.counts(xml_dir, measure)
    assert sorted(seen) == ["USA", "USB", "USC"]
    assert {k: v for k, v in first.items() if k != "_clean"} == {
        "USA": 10, "USB": 20, "USC": 30}
    assert first["_clean"] == {"USA", "USB"}, "fidelity verdict must survive"

    seen.clear()
    second = led.counts(xml_dir, measure)
    assert seen == [], "nothing may be re-measured while the code has not moved"
    assert second == first, "and the answer must be identical, not merely close"


def test_a_code_change_invalidates_every_entry(led, tmp_path, monkeypatch):
    """A content hash, not an mtime. When it moves, the numbers are re-derived."""
    xml_dir = _corpus(tmp_path, ["USA", "USB"])
    calls = []

    def measure(path):
        calls.append(path.stem)
        return 5, True

    led.counts(xml_dir, measure)
    assert len(calls) == 2
    calls.clear()

    led.counts(xml_dir, measure)
    assert calls == []

    monkeypatch.setattr(led, "code_key", lambda: "KEY-B")
    led.counts(xml_dir, measure)
    assert sorted(calls) == ["USA", "USB"], "a moved key must re-measure"


def test_a_patent_that_cannot_be_measured_is_omitted_not_zeroed(led, tmp_path):
    """A raise is not a measurement of zero.

    Recording it as 0 would hand the next patch a floor of nothing to clear on
    that patent, which is the anti-deletion condition inverted.
    """
    xml_dir = _corpus(tmp_path, ["USA", "USBAD"])

    def measure(path):
        if path.stem == "USBAD":
            raise RuntimeError("boom")
        return 7, True

    out = led.counts(xml_dir, measure)
    assert out["USA"] == 7
    assert "USBAD" not in out
    assert "USBAD" not in led.load()["patents"]


def test_the_probe_result_is_adopted_instead_of_recomputed(led):
    """`verify_patch` already measured the corpus under the patched tree.

    Re-deriving those counts after the patch is written would be recomputing a
    number we are holding — and it is the reason a landed patch used to cost the
    NEXT gap a full 137-file rescan.
    """
    n = led.record({"USA": 100, "USB": 200, "USCRASH": -1, "_clean": set()},
                   per_clean={"USA": True, "USB": False},
                   journal_id="0007-abc")
    assert n == 2, "-1 is a crash, not a count, and `_clean` is not a patent"

    per = led.load()["patents"]
    assert per["USA"]["compounds"] == 100
    assert per["USA"]["clean"] is True
    assert per["USB"]["clean"] is False
    assert per["USA"]["best"]["journal_id"] == "0007-abc"
    assert "USCRASH" not in per


def test_best_remembers_the_high_water_mark_across_several_patches(led):
    """The protection a one-step baseline structurally cannot give.

    `verify_patch` compares against the state immediately before this patch, so
    a sequence that walks a patent 860 -> 800 -> 700 clears the corpus condition
    three times whenever other patents rise, and nothing remembers the 860.
    US10660877 went 860 -> 0 on an `_is_namelike` patch that touched none of its
    own rows; this is the record that would have named it.
    """
    led.record({"US10660877": 860}, journal_id="0001-aaa")
    led.record({"US10660877": 800}, journal_id="0002-bbb")
    led.record({"US10660877": 700}, journal_id="0003-ccc")

    entry = led.load()["patents"]["US10660877"]
    assert entry["compounds"] == 700, "current follows the tree"
    assert entry["best"]["compounds"] == 860, "best does not follow it down"
    assert entry["best"]["journal_id"] == "0001-aaa"

    assert led.regressions_vs_best({"US10660877": 700}) == {"US10660877": [860, 700]}
    assert led.regressions_vs_best({"US10660877": 900}) == {}, "a gain is not a loss"
    assert led.regressions_vs_best({"US10660877": -1}) == {}, "a crash is not a loss"
    assert led.best("US10660877")["compounds"] == 860


def test_a_corrupt_or_missing_ledger_reads_as_empty(led, tmp_path):
    """Bookkeeping never breaks a run. A bad file re-measures; it does not raise."""
    assert led.load()["patents"] == {}
    led.LEDGER_PATH.write_text("{not json")
    assert led.load()["patents"] == {}
    led.LEDGER_PATH.write_text(json.dumps({"version": 999, "patents": {"X": {}}}))
    assert led.load()["patents"] == {}, "a future schema is not readable as this one"


def test_invalidate_forgets_best_too(led):
    led.record({"USA": 10})
    assert led.stats()["patents"] == 1
    assert led.invalidate() is True
    assert led.load()["patents"] == {}
    assert led.best("USA") is None


# ── the wiring ────────────────────────────────────────────────────

def test_baseline_counts_reports_fidelity_per_patent_from_the_probe():
    """`_PROBE` was summing away a verdict it computes anyway.

    `parse_fidelity` runs per file in the probe and only its LENGTH was kept.
    That per-file verdict is exactly `baseline_counts`'s `_clean` set — the one
    that decides whose count is a trustworthy floor — so without it an adopted
    baseline would carry a stale `clean` flag across a patch that changed
    fidelity.
    """
    from patentdb.repair import parser_repair

    assert "per_clean" in parser_repair._PROBE
    assert '"per_clean": per_clean' in parser_repair._PROBE


def test_verify_patch_reports_below_best_and_never_blocks_on_it():
    """Recorded, not enforced — the same call this tier makes everywhere else.

    A patch that supersedes a corrupted reading must be allowed to lower it; a
    gate here would be the fourth judgement-shaped gate this file has removed.
    """
    import inspect

    from patentdb.repair import parser_repair

    src = inspect.getsource(parser_repair.verify_patch)
    assert "below_best" in src
    assert "regressions_vs_best" in src
    # The ONE blocking condition is still the corpus compound count, and the
    # best-known check must sit AFTER the only `return` that declines a patch.
    decline = src.index("picks up FEWER compounds")
    assert src.index("below_best") > decline


def test_adopt_baseline_stamps_the_tree_as_it_is_now(led, monkeypatch):
    """The probe measured the PATCHED tree, so the counts belong to the new key.

    Called before the modules are written, this would file the patched corpus
    under the unpatched tree's key and then serve, for ever, a baseline nobody
    ever measured. The ordering is the whole correctness argument.
    """
    from patentdb.repair import parser_repair

    monkeypatch.setattr(led, "code_key", lambda: "KEY-AFTER-PATCH")
    n = parser_repair.adopt_baseline(
        {"per_patent": {"USA": 12, "USB": 34}, "per_clean": {"USA": True}},
        journal_id="0009-zzz")
    assert n == 2
    per = led.load()["patents"]
    assert per["USA"]["code_key"] == "KEY-AFTER-PATCH"
    assert per["USB"]["compounds"] == 34

    # ...and that key is what a following `counts()` reuses, so the patch that
    # just landed does NOT cost the next gap a 137-file rescan.
    xml_dir = led.LEDGER_PATH.parent / "xml"
    xml_dir.mkdir(exist_ok=True)
    for p in ("USA", "USB"):
        (xml_dir / f"{p}.xml").write_text("<x/>")

    def _never(path):
        raise AssertionError("a landed patch must not trigger a re-measurement")

    out = led.counts(xml_dir, _never)
    assert {k: v for k, v in out.items() if k != "_clean"} == {"USA": 12, "USB": 34}


def test_a_landed_patch_adopts_the_probe_only_after_it_is_written():
    """Structural, because the alternative is a test that rewrites `sources/`.

    Both apply sites must call `adopt_baseline` AFTER `write_text`, and the
    capability site must call it only when the patch was actually applied —
    `verify_patch` returning ok is not the same as `PARSER_REPAIR_APPLY`
    allowing the write.
    """
    import inspect

    from patentdb.repair import capability, parser_repair

    reader = inspect.getsource(parser_repair.repair_reader)
    assert reader.index("module.write_text(patched)") < reader.index("adopt_baseline")

    cap = inspect.getsource(capability._try_one)
    assert cap.index("mod.write_text(text)") < cap.index("adopt_baseline")
    assert 'if entry["applied"]:' in cap, (
        "adopting on a verdict rather than on the write would record a corpus "
        "the tree never had")


def test_collect_gaps_reuses_a_report_the_caller_already_has(monkeypatch):
    """`autoheal` is handed the report `process_patent` just produced.

    Re-running `repair_patent` to re-derive it was not only a second full pass
    over the document — it used `max_calls=0`, so a patent whose pipeline run
    BOUGHT a rule was re-examined as though it had not, and the tier could act
    on a different set of gaps than the one that triggered it.
    """
    from patentdb.repair import capability

    def _never(*a, **k):
        raise AssertionError("repair_patent must not run when a report is given")

    import patentdb.repair.loop as loop
    monkeypatch.setattr(loop, "repair_patent", _never)

    class Rep:
        capability_gaps = [{"fingerprint": "fp-1", "patent": "USX", "table": "T1",
                            "rows_at_stake": 12, "rule_kind": "column_map",
                            "why": "validated but produced nothing"}]
        escalations: list = []
        crashed: list = []

    gaps = capability.collect_gaps(["USX"], reports={"USX": Rep()})
    assert [g["fingerprint"] for g in gaps] == ["fp-1"]
    assert gaps[0]["rows_at_stake"] == 12
