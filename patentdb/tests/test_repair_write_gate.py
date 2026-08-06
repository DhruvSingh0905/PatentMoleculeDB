"""An extraction run must not rewrite the code it is being measured against.

THE DEFECT THIS PINS. `patentdb/sources/uspto_assays.py` was rewritten by
ordinary `process_patent` runs. Reproduced on US20240010684A1 with every paid
call blocked, so nothing here is a story about a model misbehaving:

    REPAIR unset  -> uspto_assays.py sha e1982d48 -> 88d774ec, journal 37 -> 38
    REPAIR=0      -> byte-identical, journal unchanged

`REPAIR=0` was never the hole; the DEFAULT was. Four hops, every one ON by
default: `REPAIR_ENABLED` -> `maybe_escalate` -> `REPAIR_AUTOHEAL` ->
`PARSER_REPAIR_APPLY` -> `capability.py:1280 mod.write_text(text)`. It was
silent because the model answer is CACHED and replayed at zero cost, and
because the `retry` counter that is supposed to break that cache
(`capability.py:744`) is fed by `_annotate_prior_attempts`, which skips any
entry that RECOVERED rows — so a patch that worked replays for ever. The
journal proves it: `0030`/`0034`/`0036`/`0038` share the id suffix `09a09002`,
and an id is `sha256(after_source + signature)[:8]`.

WHY THE TESTS ARE SHAPED LIKE THIS. The obvious test — run `process_patent` and
diff the tree — needs the network, a corpus and minutes, so it would be skipped
in CI and would not have caught this. These drive the REAL write path in
`_try_one` with a stubbed proposal and a stubbed verdict, which is the only
thing that ever wrote, and assert on the bytes of the actual tracked files.

The last test is the durable one: it reads `repair/` with the AST and fails on
any NEW direct write to a tracked source file. Every guard in this project's
history was eventually routed around by a caller that did not know it existed.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SOURCES = REPO / "patentdb" / "sources"


def _fingerprint() -> dict[str, str]:
    """Byte-exact state of every tracked source file."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCES.glob("*.py"))}


@pytest.fixture(autouse=True)
def _no_operator_permission(monkeypatch):
    """Every test starts with nobody having asked, whatever the environment."""
    from patentdb.repair import guard

    monkeypatch.delenv(guard.SELF_PATCH_ENV, raising=False)
    monkeypatch.setattr(guard._asked, "on", False, raising=False)


# ── the contract ──────────────────────────────────────────────────

def test_pipeline_run_leaves_tracked_source_byte_identical(monkeypatch, caplog):
    """The headline. A verified patch, nobody asking, and the tree must not move.

    `do_apply=None` is exactly what `autoheal` produces: it calls
    `repair_capabilities` without naming an intent, because `process_patent`
    asked for an extraction and not for its reader to be rewritten.
    """
    caplog.set_level(logging.WARNING, logger="patentdb.repair.guard")

    before = _fingerprint()
    entry = _run_try_one(monkeypatch, do_apply=None)

    assert _fingerprint() == before, "a run with nobody asking rewrote the tree"
    assert entry["applied"] is False
    # Loud, not silent: a refusal nobody can see is the same defect one layer up.
    assert "REFUSED to modify tracked source" in caplog.text


def test_no_write_reaches_sources_when_nobody_asked(monkeypatch):
    """Runtime sentinel, in place of trusting any one gate.

    Watches `Path.write_text` for the whole call and asserts nothing under
    `patentdb/sources/` was written. This is the invariant stated directly, and
    it holds however many layers a future caller adds between the orchestrator
    and the write.
    """
    seen: list[str] = []
    real = Path.write_text

    def spy(self, data, *a, **k):
        try:
            p = self.resolve()
        except OSError:                              # pragma: no cover
            p = self
        if p.parent == SOURCES:
            seen.append(str(p))
        return real(self, data, *a, **k)

    monkeypatch.setattr(Path, "write_text", spy)
    _run_try_one(monkeypatch, do_apply=None)
    assert seen == [], f"tracked source was written with nobody asking: {seen}"


def test_the_proposal_survives_the_refusal(monkeypatch):
    """Refusing the write must not throw the diagnosis away.

    The tier is gated, not removed. The journal entry still carries the full
    before/after source, which is what `parser_health --force` applies later.
    """
    entry = _run_try_one(monkeypatch, do_apply=None)

    assert entry["patches"], "the patch bodies were dropped"
    assert entry["patches"][0]["after_source"], "no after_source to --force"
    assert entry["patches"][0]["before_source"], "no before_source to revert to"
    assert any("not applied" in o for o in entry["objections"])


def test_an_operator_who_asks_still_patches_the_tree(monkeypatch, tree_restored):
    """The self-healing tier is preserved. It just has to be asked."""
    from patentdb.repair import guard

    before = _fingerprint()
    with guard.operator_request():
        entry = _run_try_one(monkeypatch, do_apply=None)
    assert entry["applied"] is True
    assert _fingerprint() != before, "an operator asked and nothing happened"


def test_self_patch_env_var_is_the_other_way_to_ask(monkeypatch, tree_restored):
    from patentdb.repair import guard

    monkeypatch.setenv(guard.SELF_PATCH_ENV, "1")
    entry = _run_try_one(monkeypatch, do_apply=None)
    assert entry["applied"] is True


def test_parser_repair_apply_zero_still_vetoes_everything(monkeypatch):
    """The old kill switch keeps its meaning, over the top of the new gate."""
    from patentdb.core import config
    from patentdb.repair import guard

    monkeypatch.setattr(config, "PARSER_REPAIR_APPLY", False)
    with guard.operator_request():
        assert guard.may_write_tracked_source() is False
        assert guard.may_write_tracked_source(explicit=True) is False


# ── the guard itself ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "explicit,asked,env,expected",
    [(None, False, "0", False),      # the pipeline — the case that broke
     (None, True, "0", True),        # a repair CLI
     (None, False, "1", True),       # SELF_PATCH=1
     (True, False, "0", True),       # a caller naming its intent
     (False, True, "1", False)],     # ...and naming it as "no"
)
def test_permission_truth_table(monkeypatch, explicit, asked, env, expected):
    from patentdb.repair import guard

    monkeypatch.setenv(guard.SELF_PATCH_ENV, env)
    if asked:
        with guard.operator_request():
            assert guard.may_write_tracked_source(explicit=explicit) is expected
    else:
        assert guard.may_write_tracked_source(explicit=explicit) is expected


def test_unchanged_is_not_an_application(tmp_path):
    """A cached proposal replayed onto a tree that already holds it did nothing.

    Journalling that as an application is how one proposal came to claim four.
    """
    from patentdb.repair import guard

    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    with guard.operator_request():
        assert guard.write_tracked_source(f, "x = 2\n") == guard.Outcome.WRITTEN
        assert guard.write_tracked_source(f, "x = 2\n") == guard.Outcome.UNCHANGED
    assert f.read_text() == "x = 2\n"


def test_refusal_does_not_touch_the_file(tmp_path):
    from patentdb.repair import guard

    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    mtime = f.stat().st_mtime_ns
    assert guard.write_tracked_source(f, "x = 2\n") == guard.Outcome.REFUSED
    assert f.read_text() == "x = 1\n"
    assert f.stat().st_mtime_ns == mtime


def test_permission_is_restored_and_does_not_leak_across_threads():
    """`verify_patch` probes in a subprocess and the greedy loops use threads.

    A global flag would hand the permission to work the operator never named.
    """
    from patentdb.repair import guard

    seen = []
    with guard.operator_request():
        with guard.operator_request():
            assert guard.operator_requested() is True
        assert guard.operator_requested() is True, "nesting clobbered the outer state"
        t = threading.Thread(target=lambda: seen.append(guard.operator_requested()))
        t.start()
        t.join()
    assert guard.operator_requested() is False
    assert seen == [False], "permission leaked into another thread"


def test_a_refused_patch_is_not_counted_or_logged_as_applied(monkeypatch):
    """`applied` must count WRITES, not verdicts.

    It counted `verify_patch`'s opinion, so a patch that verified and was never
    written reported `applied: 1`. `process_patent` reads that field and
    re-extracts on it, believing the reader underneath it changed — which is a
    second, quieter version of the same defect: a run reporting that it healed
    itself when the tree never moved.
    """
    entry = _run_try_one(monkeypatch, do_apply=None)
    assert entry["applied"] is False

    from patentdb.repair import capability

    monkeypatch.setattr(capability, "collect_gaps",
                        lambda pids, reports=None: [])
    rep = capability.repair_capabilities(patent_ids=["USTEST"])
    assert rep["applied"] == 0 and rep["proposed"] == 0


def test_try_one_reports_written_separately_from_ok():
    """The two are different questions and were the same field."""
    import inspect

    from patentdb.repair import capability

    src = inspect.getsource(capability._try_one)
    assert '"written": bool(entry["applied"])' in src, (
        "_try_one must report whether the tree moved, not only whether the "
        "verifier approved")


# ── the journal ───────────────────────────────────────────────────

@pytest.fixture()
def journal(tmp_path, monkeypatch):
    """A scratch journal. The real one is the revert mechanism, never a fixture."""
    from patentdb.core import config
    from patentdb.repair import parser_repair

    path = tmp_path / "journal.jsonl"
    monkeypatch.setattr(config, "PARSER_REPAIR_JOURNAL", path)
    return parser_repair


def test_an_empty_after_source_cannot_prepend_itself_into_a_file(journal, tmp_path):
    """`str.replace("", x, 1)` PREPENDS. It does not no-op.

    Entry `0002-58499a23` in the live journal carries an empty before AND after
    source, so every `X not in text` check passes vacuously. Both empty happens
    to be inert; one empty would inject a whole function at line 1 and report
    `ok`. A model supplies these strings, so "it will not be empty" is not a
    property anything here can rely on.
    """
    mod = tmp_path / "victim.py"
    mod.write_text("REAL = 1\n")
    journal.journal_append({"action": "capability_patch", "applied": True,
                            "module": str(mod), "signature": "s",
                            "before_source": "", "after_source": ""})
    entry_id = journal.journal_read()[0]["id"]

    r = journal.revert(entry_id)
    assert r["ok"] is False and "empty" in r["why"]
    assert mod.read_text() == "REAL = 1\n", "revert corrupted the file"

    r = journal.apply_journaled(entry_id)
    assert r["ok"] is False and "empty" in r["why"]
    assert mod.read_text() == "REAL = 1\n", "--force corrupted the file"


def test_journal_state_separates_applied_from_still_in_the_tree(journal, tmp_path):
    """`applied: True` is a claim about the past. `live` is about the present."""
    mod = tmp_path / "m.py"
    mod.write_text("def f():\n    return 2\n")
    journal.journal_append({"action": "capability_patch", "applied": True,
                            "module": str(mod), "signature": "s1",
                            "before_source": "def f():\n    return 1\n",
                            "after_source": "def f():\n    return 2\n"})
    journal.journal_append({"action": "capability_patch", "applied": True,
                            "module": str(mod), "signature": "s2",
                            "before_source": "def g():\n    return 1\n",
                            "after_source": "def g():\n    return 9\n"})
    states = {s["id"][:4]: s["state"] for s in journal.journal_state()}
    assert states == {"0001": "live", "0002": "stale"}


def test_a_replayed_proposal_is_marked_as_a_re_application(journal, tmp_path):
    """One proposal applied twice must not read as two independent repairs.

    This is `0030`/`0034`/`0036`/`0038` — same `after_source`, same signature,
    therefore the same id suffix, four entries.
    """
    mod = tmp_path / "m.py"
    mod.write_text("x = 2\n")
    same = {"action": "capability_patch", "applied": True, "module": str(mod),
            "signature": "sig", "before_source": "x = 1\n", "after_source": "x = 2\n"}
    journal.journal_append(dict(same))
    journal.journal_append(dict(same))
    st = journal.journal_state()
    assert st[0]["duplicate_of"] is None
    assert st[1]["duplicate_of"] == st[0]["id"]


def test_reconcile_marks_without_deleting_and_is_idempotent(journal, tmp_path):
    """History is never rewritten — the journal IS the revert mechanism."""
    mod = tmp_path / "m.py"
    mod.write_text("nothing here\n")
    journal.journal_append({"action": "capability_patch", "applied": True,
                            "module": str(mod), "signature": "s",
                            "before_source": "a\n", "after_source": "b\n"})
    before = len(journal.journal_read())

    first = journal.reconcile_journal()
    assert first["appended"] is True and first["stale"] == 1
    assert len(journal.journal_read()) == before + 1, "an entry was removed"

    second = journal.reconcile_journal()
    assert second["appended"] is False, "reconciling twice appended twice"


# ── the durable one ───────────────────────────────────────────────

def test_no_module_writes_a_tracked_source_except_through_the_guard():
    """AST scan: a NEW direct write to `sources/` fails here.

    `guard.py` is the sanctioned writer. `greedy.py` and `iterate.py` are
    exempt and listed by name: they write a candidate, reload, measure and
    restore in a `finally`, which is the only way to evaluate a patch to
    `_is_namelike` at all, and neither is reachable from `process_patent`
    (`import_audit` puts them in test-only and eval-only respectively).
    """
    exempt_modules = {"guard.py", "greedy.py", "iterate.py"}
    # `verify_patch` copies every tracked file into a temp sandbox and writes
    # the candidate THERE, then runs a probe against it in a subprocess. That
    # write is the whole acceptance test and never touches the real tree.
    exempt_functions = {"verify_patch"}
    offenders = []
    for mod in sorted((REPO / "patentdb" / "repair").glob("*.py")):
        if mod.name in exempt_modules:
            continue
        text = mod.read_text()
        tree = ast.parse(text)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name in exempt_functions:
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "write_text"):
                    src = ast.get_source_segment(text, node) or ""
                    # The three JSON artifacts (ledger, rule library, snapshot)
                    # are not source and are named by their own variables.
                    if any(w in src for w in ("mod.", "module.", "m.write_text")):
                        offenders.append(f"{mod.name}:{fn.name}:{node.lineno}: {src[:60]}")
    assert not offenders, (
        "these write a source file directly instead of through "
        "guard.write_tracked_source:\n  " + "\n  ".join(offenders))


def test_repair_zero_keeps_the_whole_tier_out_of_the_run():
    """`REPAIR=0` must still short-circuit everything, including the code tier.

    Measured: `REPAIR=0` never did reproduce the defect — the block in
    `process_patent` does enclose `maybe_escalate`, and the tree came back
    byte-identical. It was the DEFAULT that wrote. This pins the half that was
    already right, because the obvious "fix" is to move the autoheal call out
    from under the flag so escalations are always journaled, and that would
    restore a paid, tree-writing path to a run that asked for neither.
    """
    src = (REPO / "patentdb" / "routes" / "process_patent.py").read_text()
    tree = ast.parse(src)
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test)
        if "REPAIR_ENABLED" not in test:
            continue
        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        if "maybe_escalate" in body:
            guarded = True
    assert guarded, (
        "maybe_escalate is no longer inside `if config.REPAIR_ENABLED:` — "
        "REPAIR=0 would reach the code-patch tier")


def test_autoheal_does_not_name_an_intent_to_patch():
    """The pipeline's call must stay `apply`-less, so the guard decides.

    Passing `apply=True` here would restore the defect exactly, and it would
    look like a harmless explicit default.
    """
    src = (REPO / "patentdb" / "repair" / "autoheal.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None))
             == "repair_capabilities"]
    assert calls, "autoheal no longer calls the capability tier"
    for c in calls:
        assert not any(k.arg == "apply" for k in c.keywords), (
            "autoheal names an apply intent; the pipeline must defer to guard")


# ── harness ───────────────────────────────────────────────────────

@pytest.fixture()
def tree_restored():
    """Put every tracked source file back byte-for-byte after the test.

    Byte-exact from memory, NOT `git checkout` — a test must never discard a
    developer's uncommitted work in `sources/`, and two of these tests
    deliberately let the patcher write.
    """
    saved = {p: p.read_bytes() for p in sorted(SOURCES.glob("*.py"))}
    try:
        yield
    finally:
        for path, data in saved.items():
            if path.read_bytes() != data:
                path.write_bytes(data)


def _run_try_one(monkeypatch, *, do_apply):
    """Drive the REAL write path in `capability._try_one`, with no model call.

    Everything paid or slow is stubbed; the splice, the guard and the journal
    are the production code. Returns the journal entry it would have written.
    """
    from patentdb.repair import capability

    target = "classify_column"
    mod, _ = capability.PATCHABLE[target]
    current = capability._function_source(mod, target)
    assert current, f"{target} not found in {mod.name}"
    # A real, syntactically valid rewrite: the same function with one comment
    # line added. Big enough to change the bytes, small enough to be obviously
    # reversible if a test ever leaves it behind.
    body = current.rstrip() + "\n    # patched by test_repair_write_gate\n"

    monkeypatch.setattr(capability, "propose_capability_patch",
                        lambda g, t, model=None, history=None: {
                            "diagnosis": "stub", "patches": [
                                {"target": target, "function_source": body}]})
    monkeypatch.setattr(capability, "verify_patch",
                        lambda *a, **k: {"ok": True, "why": "stub",
                                         "per_patent": {"USTEST": 9},
                                         "repaired_usable": 9,
                                         "total_usable": 9})
    monkeypatch.setattr(capability, "_bad_values_now", lambda pid: 0)

    captured = {}

    def _fake_journal(entry):
        captured.update(entry)
        return "0000-test"

    # `_try_one` imports both of these from `parser_repair` INSIDE the function,
    # so the binding that matters is on `parser_repair`, not on `capability`.
    # Patching the wrong one is not inert: it let the real `adopt_baseline`
    # write a `USTEST` entry into `output_v2/repair_baselines.json`.
    monkeypatch.setattr("patentdb.repair.parser_repair.journal_append",
                        _fake_journal)
    monkeypatch.setattr("patentdb.repair.parser_repair.adopt_baseline",
                        lambda *a, **k: None)

    gap = {"patent": "USTEST", "table": "T1", "fingerprint": "fp-test",
           "rows_at_stake": 5, "rule_kind": "column_map", "rule_payload": {},
           "why": "stub"}
    capability._try_one(gap, object(), "stub-model", {"USTEST": 0}, do_apply,
                        last=True)
    assert captured, "no journal entry was produced"
    return captured
