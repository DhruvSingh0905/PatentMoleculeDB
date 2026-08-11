"""Make the production artifacts unreachable from the test suite.

WHY THIS EXISTS
---------------
`sources/losses.py` opens `LOSS_LOG` — `patentdb3/out/loss_log.jsonl`, the real
one — on the FIRST `record()` call in a process, in `"w"` mode. That is correct
for a pipeline run (fresh log per run, never appended across runs) and wrong
for a test process, where it means any test that so much as calls
`extract_names` silently truncates a corpus artifact somebody may be reading.

`test_structures_wiring.py` redirects it properly for its own duration and then
tears down with `losses.reset(losses.LOSS_LOG)` — which re-points at the
production path AND truncates it, because truncating is what `reset` does.
Every other test file that touches the extractor never redirects at all.

This was found during an independent audit of the corpus artifacts: the auditor
was reading `out/loss_log.jsonl` while a test run emptied it. The artifact and
`latest.json` then disagreed, with nothing raising and nothing logged — the
manifest still described a file that no longer had those contents.

THE FIX IS NOT "REMEMBER TO REDIRECT". Remembering is what failed. This
fixture is `autouse` and session-scoped, so the production path is never the
active one inside a test process, whatever any individual test forgets to do.
A test that WANTS a specific path still calls `losses.reset(tmp)` exactly as
before — this only changes what the default is.

THE SAME THING WAS TRUE OF `structures.tsv`, AND I GOT IT WRONG ONCE.
An earlier version of this file said the dump artifacts needed no guard,
because every `verify.dump()` call in the suite takes the `redirected` fixture
in `test_structures_wiring.py`. That was true of that file and false of the
suite: `test_repair_gate.py::test_self_heal_flag_selects_the_pipeline`
redirects `DUMP_PATH` and `MANIFEST_PATH` and NOT `STRUCT_PATH`, because it
predates the structures artifact and was never updated when `dump()` grew a
third output. Every run of the suite therefore overwrote the real
`out/structures.tsv` with one patent's rows — 64,465 -> 328 — while leaving
`reader_dump.tsv` and `latest.json` correct, so the manifest described a file
whose contents no longer matched it and nothing raised.

Measured, not argued: with a sentinel written into `out/structures.tsv`, each
test file run alone leaves it untouched; the full suite replaces it. The
per-file bisect points at `test_repair_gate.py`.

So all three dump paths are redirected here too. The lesson is the one the
loss-log case already taught and I did not generalise: a path that a test CAN
write is a path a test eventually WILL write, and "I checked and no test writes
it" is a claim about today's tests.

AND IT WAS A CLAIM ABOUT TODAY'S TESTS, AGAIN. Two paths were still missing:
the RULE LIBRARIES and `config.RULE_JOURNAL`. `verify.dump(heal=True)` builds
`RuleLibrary()` with no override and `repair_patent()` calls `lib.save()`
unconditionally, so a test that heals a patent with real gaps rewrites the
TRACKED `patentdb3/data/layout_rules.json` — 172 checked-in rules that ship
with the repo. This stayed invisible only by luck: the one existing test using
`dump(heal=True)` runs a zero-gap patent, so the save round-trips
byte-identical.

It was found the way the other two were — by something outside the suite
doing it for real. An agent surveying candidate patents called `repair_patent()`
unguarded, mutated the tracked library, and wrote 17 entries to the production
rule journal. Both were restored, and the same fix applies: make the
production path unreachable rather than asking every future test to remember.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from patentdb3.sources import losses


@pytest.fixture(autouse=True, scope="session")
def _never_write_production_artifacts(tmp_path_factory):
    """Point every artifact path at scratch files for the whole session.

    A test that redirects explicitly still works — this only changes what the
    default is, so forgetting costs nothing.
    """
    from patentdb3 import verify

    from patentdb3.core import config
    from patentdb3.repair import name_rules as NRULES
    from patentdb3.repair import rules as LRULES

    out = tmp_path_factory.mktemp("artifacts")
    verify.DUMP_PATH = out / "reader_dump.tsv"
    verify.STRUCT_PATH = out / "structures.tsv"
    verify.MANIFEST_PATH = out / "latest.json"

    # THE RULE LIBRARIES ARE TRACKED DATA, NOT OUTPUT. `layout_rules.json`'s
    # 172 rules ship with the repo (CLAUDE.md calls this a deliberate
    # asymmetry against v2), so a test rewriting it is a source change nobody
    # asked for. The libraries are COPIED rather than emptied: a test that
    # exercises the library-first path must still see the real rules, or it
    # would silently measure a $0 pass that finds nothing.
    for mod, attr in ((LRULES, "_LIBRARY"), (NRULES, "LIBRARY_PATH")):
        real = getattr(mod, attr, None)
        if real is None:
            continue
        scratch_lib = out / f"{Path(real).name}"
        if Path(real).exists():
            scratch_lib.write_text(Path(real).read_text())
        else:
            scratch_lib.write_text(json.dumps({"schema": 1, "rules": [],
                                               "patterns": []}))
        setattr(mod, attr, scratch_lib)

    # The rule journals are append-only audit trails of what really ran. A
    # test appending to them puts fixture rows into a corpus record.
    config.RULE_JOURNAL = out / "rule_adoption_journal.jsonl"
    try:
        from patentdb3.repair import name_loop as NLOOP
        NLOOP.NAME_JOURNAL = out / "name_rule_journal.jsonl"
    except Exception:                       # module optional; never block a run
        pass

    scratch = out / "loss_log.jsonl"
    losses.reset(scratch)
    yield scratch
    # Deliberately NOT `losses.reset(losses.LOSS_LOG)`: that is the teardown
    # that caused the problem, because `reset` truncates whatever it is given.
    # `close()` releases the handle without re-pointing, so the production file
    # is left exactly as the last real run wrote it.
    losses.close()


@pytest.fixture(autouse=True)
def _fail_if_a_test_targets_a_production_artifact():
    """Catch a future test that hardcodes a real artifact path.

    Cheap, and it fails loudly at the moment the mistake is made rather than
    quietly deleting a corpus run somebody is mid-way through reading.
    """
    before = losses._path
    yield
    if Path(losses._path) == Path(losses.LOSS_LOG):
        raise AssertionError(
            f"a test re-pointed `losses` at the PRODUCTION log "
            f"({losses.LOSS_LOG}); it was {before}. Use `losses.reset(tmp_path "
            f"/ 'loss_log.jsonl')` and do not reset back to `LOSS_LOG` in "
            f"teardown — `reset` truncates what it is given.")
