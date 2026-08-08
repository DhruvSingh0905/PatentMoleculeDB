"""Tests for `sources/losses.py` — the structured loss-observability sink
every drop point in `iupac_names.py`, `anchor.py` and `verify.py` writes to.

Every test redirects the sink to a file under pytest's own `tmp_path` via
`losses.reset(...)` before writing anything. None of them touch the real
corpus artifact (`losses.LOSS_LOG`, under `patentdb3/out/`) or the
`ENABLED` flag's real environment-derived value beyond the one test that
exercises it directly (which restores it afterward) — a test suite that
wrote into the shared corpus log, or left the flag flipped for tests after
it, would make this file interfere with whatever else is reading that log.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from patentdb3.sources import losses


def _read(path):
    """Flush first — `record()` never flushes per line (see
    `losses.flush()`'s docstring), so a raw `open()` on the same path from
    this SEPARATE file handle can otherwise see fewer lines than have
    actually been written.
    """
    losses.flush()
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path):
    """Every test gets its own fresh log file, never the real corpus one.

    `reset()` is exactly the API a real orchestrator (`verify.dump()`) is
    expected to call once per run; using it here — rather than poking at
    `losses._fh`/`losses._path` directly — means these tests exercise the
    same entry point production code does.
    """
    path = tmp_path / "loss_log.jsonl"
    losses.reset(path)
    yield path
    losses.close()


# ── record() basics ─────────────────────────────────────────────────────

def test_record_writes_one_json_line_per_call(_isolated_log):
    losses.record("seed_too_short", "US0000001", position=5, candidate="foo")
    losses.record("seed_too_short", "US0000001", position=9, candidate="bar")
    rows = _read(_isolated_log)
    assert len(rows) == 2
    assert rows[0] == {"loss_type": "seed_too_short", "patent_id": "US0000001",
                        "position": 5, "candidate": "foo"}
    assert rows[1]["candidate"] == "bar"


def test_loss_type_and_patent_id_always_present(_isolated_log):
    losses.record("x", "US1")
    row = _read(_isolated_log)[0]
    assert row["loss_type"] == "x"
    assert row["patent_id"] == "US1"


def test_extra_fields_pass_through_unmodified_when_small(_isolated_log):
    losses.record("x", "US1", floor=12, length=7, reason="no_digit_bracket_dash")
    row = _read(_isolated_log)[0]
    assert row["floor"] == 12
    assert row["length"] == 7
    assert row["reason"] == "no_digit_bracket_dash"


# ── reset() ──────────────────────────────────────────────────────────────

def test_reset_truncates_a_previous_log_at_the_same_path(tmp_path):
    path = tmp_path / "t.jsonl"
    losses.reset(path)
    losses.record("x", "P1", a=1)
    losses.record("x", "P1", a=2)
    assert len(_read(path)) == 2

    losses.reset(path)          # fresh start at the SAME path
    assert path.read_text() == ""
    losses.record("x", "P1", a=3)
    rows = _read(path)
    assert len(rows) == 1
    assert rows[0]["a"] == 3


def test_reset_returns_the_path_now_in_use(tmp_path):
    path = tmp_path / "elsewhere.jsonl"
    assert losses.reset(path) == path


def test_record_without_an_explicit_reset_still_works(tmp_path):
    """`record()` lazily opens the sink on its own first call — a caller
    that never resets (a standalone `extract_names()` call, an interactive
    session) must not be required to plumb one through. Redirected to a
    tmp path first so this test does not fall through to the real corpus
    log's default location.
    """
    path = tmp_path / "lazy.jsonl"
    losses.reset(path)          # still isolate from the real corpus log
    losses.close()               # simulate "never opened yet" for THIS path
    losses.record("x", "P1", a=1)
    assert len(_read(path)) == 1


# ── ENABLED short-circuits everything, including file creation ──────────

def test_disabled_is_a_true_no_op(tmp_path, monkeypatch):
    path = tmp_path / "off.jsonl"
    losses.reset(path)
    monkeypatch.setattr(losses, "ENABLED", False)
    losses.record("x", "P1", a=1)
    losses.close()
    assert path.read_text() == ""


def test_disabled_then_reenabled_resumes_writing(tmp_path, monkeypatch):
    path = tmp_path / "toggle.jsonl"
    losses.reset(path)
    monkeypatch.setattr(losses, "ENABLED", False)
    losses.record("x", "P1", a=1)
    monkeypatch.setattr(losses, "ENABLED", True)
    losses.record("x", "P1", a=2)
    rows = _read(path)
    assert len(rows) == 1
    assert rows[0]["a"] == 2


# ── summary() ────────────────────────────────────────────────────────────

def test_summary_counts_by_loss_type(_isolated_log):
    losses.record("a", "P1")
    losses.record("a", "P1")
    losses.record("b", "P1")
    assert losses.summary(_isolated_log) == {"a": 2, "b": 1}


def test_summary_on_a_path_with_no_records_is_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    losses.reset(path)
    assert losses.summary(path) == {}


def test_summary_reads_back_without_disturbing_the_sink(_isolated_log):
    losses.record("a", "P1")
    counts1 = losses.summary(_isolated_log)
    losses.record("a", "P1")
    counts2 = losses.summary(_isolated_log)
    assert counts1 == {"a": 1}
    assert counts2 == {"a": 2}


def test_summary_defaults_to_the_currently_active_sink(_isolated_log):
    losses.record("a", "P1")
    losses.record("a", "P1")
    assert losses.summary() == {"a": 2}     # no path arg — reads `_isolated_log`


# ── normalization: truncation, sets, lists, dataclasses ──────────────────

def test_long_string_field_is_truncated(_isolated_log):
    long = "x" * 5000
    losses.record("t", "P1", candidate=long)
    row = _read(_isolated_log)[0]
    assert len(row["candidate"]) < len(long)
    assert row["candidate"].startswith("x" * losses._MAX_STR)
    assert "chars]" in row["candidate"]


def test_short_string_field_is_untouched(_isolated_log):
    losses.record("t", "P1", candidate="short")
    row = _read(_isolated_log)[0]
    assert row["candidate"] == "short"


def test_set_field_normalized_to_a_sorted_list(_isolated_log):
    losses.record("t", "P1", ids={"c", "a", "b"})
    row = _read(_isolated_log)[0]
    assert row["ids"] == ["a", "b", "c"]


def test_set_normalization_is_order_independent(tmp_path):
    """The whole point of sorting a set before writing: two sets built by
    inserting the same members in a DIFFERENT order must serialize
    IDENTICALLY, because Python does not guarantee set iteration order is
    stable across builds — this is the one place the module docstring calls
    out explicitly.
    """
    p1, p2 = tmp_path / "s1.jsonl", tmp_path / "s2.jsonl"
    losses.reset(p1)
    losses.record("t", "P1", ids={"z", "a", "m"})
    losses.reset(p2)          # `reset()` closes p1's handle as a side effect
    losses.record("t", "P1", ids={"m", "z", "a"})
    losses.flush()             # p2 has not been closed yet — flush before reading
    assert p1.read_bytes() == p2.read_bytes()


def test_long_list_field_is_capped(_isolated_log):
    losses.record("t", "P1", items=list(range(200)))
    row = _read(_isolated_log)[0]
    assert len(row["items"]) == losses._MAX_ITEMS + 1     # +1 marker entry
    assert row["items"][-1] == f"...+{200 - losses._MAX_ITEMS} more"
    assert row["items"][:losses._MAX_ITEMS] == list(range(losses._MAX_ITEMS))


def test_short_list_field_is_untouched(_isolated_log):
    losses.record("t", "P1", items=[3, 1, 2])
    row = _read(_isolated_log)[0]
    assert row["items"] == [3, 1, 2]        # order preserved, NOT sorted —
                                             # only sets are; a list is
                                             # already deterministic by
                                             # construction (see docstring)


def test_dataclass_field_is_normalized_to_its_dict(_isolated_log):
    @dataclasses.dataclass
    class Thing:
        cid: str
        distance: int

    losses.record("t", "P1", thing=Thing(cid="12", distance=3))
    row = _read(_isolated_log)[0]
    assert row["thing"] == {"cid": "12", "distance": 3}


def test_nested_dict_keys_are_sorted(_isolated_log):
    losses.record("t", "P1", info={"zeta": 1, "alpha": 2})
    losses.flush()
    line = _isolated_log.read_text()
    # the nested dict's own keys, not just the top-level ones
    assert line.index('"alpha"') < line.index('"zeta"')


# ── determinism ────────────────────────────────────────────────────────

def test_record_never_writes_a_wall_clock_timestamp(_isolated_log):
    """Same-input-same-log requires no field whose value changes between two
    runs made seconds apart — see the module docstring's DETERMINISM
    section.
    """
    losses.record("t", "P1", a=1)
    row = _read(_isolated_log)[0]
    assert "ts" not in row
    assert "timestamp" not in row
    assert "time" not in row


def test_two_identical_call_sequences_produce_byte_identical_files(tmp_path):
    def run(path):
        losses.reset(path)
        losses.record("a", "P1", position=1, name="foo")
        losses.record("b", "P2", position=2, name="bar", ids={"x", "y"})
        losses.close()
        return path.read_bytes()

    first = run(tmp_path / "run1.jsonl")
    second = run(tmp_path / "run2.jsonl")
    assert first == second


def test_top_level_keys_are_sorted_in_the_raw_line(_isolated_log):
    losses.record("t", "P1", zeta=1, alpha=2)
    losses.flush()
    line = _isolated_log.read_text().strip()
    assert line.index('"alpha"') < line.index('"zeta"')


# ── close() ────────────────────────────────────────────────────────────

def test_close_is_idempotent(_isolated_log):
    losses.record("a", "P1")
    losses.close()
    losses.close()               # must not raise
    losses.record("a", "P1")     # lazily reopens the SAME path afterward
    assert len(_read(_isolated_log)) == 2


# ── real call sites: anchor.find_cid, no OPSIN needed ────────────────────

def test_find_cid_logs_anchor_not_found(_isolated_log):
    from patentdb3.sources.anchor import find_cid

    r = find_cid("no ids anywhere near this text at all", "quinazoline",
                  patent_id="US_TEST")
    assert r.cid is None and not r.clashed
    rows = [row for row in _read(_isolated_log) if row["loss_type"] == "anchor_not_found"]
    assert len(rows) == 1
    assert rows[0]["patent_id"] == "US_TEST"
    assert rows[0]["name"] == "quinazoline"


def test_find_cid_logs_anchor_clash(_isolated_log):
    from patentdb3.sources.anchor import find_cid

    text = "Example 1\nfoo-acid text Example 2\nfoo-acid text"
    r = find_cid(text, "foo-acid", patent_id="US_TEST")
    assert r.clashed
    rows = [row for row in _read(_isolated_log) if row["loss_type"] == "anchor_clash"]
    assert len(rows) == 1
    assert rows[0]["patent_id"] == "US_TEST"
    assert {c["cid"] for c in rows[0]["candidates"]} == {"1", "2"}


def test_find_cid_resolving_cleanly_logs_neither(_isolated_log):
    """The common case — an id found and uncontested — must not appear in
    the loss log at all; it is not a loss.
    """
    from patentdb3.sources.anchor import find_cid

    text = "Example 1\nracemic cis-2-{1-(4-Bromobenzyl)}cyclohexanecarboxylic acid\n"
    r = find_cid(text, "cis-2-{1-(4-Bromobenzyl)}cyclohexanecarboxylic acid",
                 patent_id="US_TEST")
    assert r.cid == "1" and not r.clashed
    types = {row["loss_type"] for row in _read(_isolated_log)}
    assert "anchor_not_found" not in types
    assert "anchor_clash" not in types


def test_anchor_text_logs_tables_dropped_when_a_table_is_present(_isolated_log):
    from patentdb3.sources.anchor import anchor_text

    xml = (
        "<description><p>intro</p>"
        "<tables><table><row><entry>4-bromo-2-fluorobenzylamine derivative "
        "is a real chemical looking string</entry></row></table></tables>"
        "<p>more text</p></description>"
    )
    anchor_text(xml, patent_id="US_TABLES")
    rows = [row for row in _read(_isolated_log) if row["loss_type"] == "tables_dropped"]
    assert len(rows) == 1
    assert rows[0]["patent_id"] == "US_TABLES"
    assert rows[0]["chars"] > 0


def test_anchor_text_logs_nothing_when_there_is_no_table(_isolated_log):
    from patentdb3.sources.anchor import anchor_text

    anchor_text("<description><p>just prose, no tables here</p></description>",
                patent_id="US_NO_TABLES")
    types = {row["loss_type"] for row in _read(_isolated_log)}
    assert "tables_dropped" not in types
