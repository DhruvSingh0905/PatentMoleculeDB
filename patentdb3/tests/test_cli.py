"""The front door: does it open, does it stop politely, does it leak a key.

Three things are worth a test here and the rest is not. `cli.py` runs no
extraction of its own — `run` calls `verify.one`/`verify.dump` and reads the
manifest back — so testing what it REPORTS would be testing the reader twice,
and slowly. What is only true of this module is:

  - `python3 -m patentdb3` resolves at all, and every help screen renders.
    `__main__.py` is one line and one line is enough to get wrong.
  - a patent nobody has fetched, with no key to fetch it, ends in a SENTENCE.
    Without the guard it ends as `UsptoUnavailable` three frames down inside a
    fetch, which reads to a new user as a broken install rather than as the one
    missing setting it is.
  - `setup` writes only what it was handed, and prints no value back. A key
    echoed once is a key in a scrollback, a screen recording and a CI log, and
    a key written into a file git tracks is a key in a history that deleting
    the line does not clean.

NO REAL CREDENTIAL APPEARS BELOW. The one key-shaped string is an obvious
placeholder, and it exists so a test can assert it is ABSENT from stdout.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from patentdb3 import cli
from patentdb3.core import config

# Key-shaped, and deliberately not a key. Asserted absent from output.
FAKE = "placeholder-not-a-real-key-0000"

# No grant has this number, so it can never be cached and the miss is stable.
UNCACHED = "US0000001"


def test_the_module_and_every_help_screen_run():
    """`python3 -m patentdb3` and both subcommands render help and exit 0."""
    for args in ([], ["setup"], ["run"]):
        r = subprocess.run([sys.executable, "-m", "patentdb3", *args, "--help"],
                           cwd=config.REPO_ROOT, capture_output=True, text=True)
        assert r.returncode == 0, f"{args} --help exited {r.returncode}: {r.stderr}"
        assert "usage: patentdb3" in r.stdout
    # The two commands must be reachable from the top-level screen, or the
    # front door names nothing a user can type.
    top = subprocess.run([sys.executable, "-m", "patentdb3", "--help"],
                         cwd=config.REPO_ROOT, capture_output=True, text=True)
    assert "setup" in top.stdout and "run" in top.stdout


def test_an_uncached_patent_with_no_key_stops_with_a_sentence(monkeypatch, capsys):
    """Exit 1 and say which setting is missing. Never a traceback."""
    monkeypatch.delenv("USPTO_API_KEY", raising=False)
    assert not (config.XML_INPUT_DIR / f"{UNCACHED}.xml").exists()

    rc = cli.run(UNCACHED, write=False, heal=False)          # must not raise

    assert rc == 1
    out = capsys.readouterr().out
    assert "USPTO_API_KEY" in out
    assert UNCACHED in out
    # The give-away that an exception escaped and something printed it.
    assert "Traceback" not in out


def test_setup_writes_no_key_it_was_not_given(monkeypatch, tmp_path, capsys):
    """A blank answer keeps the current value, and writes nothing at all.

    Both halves matter. The file must not GAIN a key name, and a session that
    changed nothing must not create the file — running `setup` to LOOK at the
    configuration is the common case and it must be side-effect free.
    """
    for name, _needed in cli.KEYS:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"

    rc = cli.setup(env_path=env, ask=lambda q: "", ask_secret=lambda q: "")

    assert rc == 0
    assert not env.exists()
    out = capsys.readouterr().out
    assert "not set" in out


def test_setup_writes_a_switch_without_inventing_a_key(monkeypatch, tmp_path):
    """A switch change is persisted; the untouched keys stay out of the file."""
    for name, _needed in cli.KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SELF_HEAL", "1")
    env = tmp_path / ".env"
    answers = iter(["1", "off", ""])          # switch 1 -> off, then done

    rc = cli.setup(env_path=env, ask=lambda q: next(answers),
                   ask_secret=lambda q: "")

    assert rc == 0
    written = env.read_text()
    assert "SELF_HEAL=0" in written
    for name, _needed in cli.KEYS:
        assert name not in written


def test_setup_never_prints_the_key_it_was_given(monkeypatch, tmp_path, capsys):
    """The value reaches the file and nothing else. Not stdout, not the summary."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("USPTO_API_KEY", raising=False)
    env = tmp_path / ".env"
    secrets = iter([FAKE, ""])                # anthropic given, uspto blank

    cli.setup(env_path=env, ask=lambda q: "", ask_secret=lambda q: next(secrets))

    assert f"ANTHROPIC_API_KEY={FAKE}" in env.read_text()
    assert "USPTO_API_KEY" not in env.read_text()
    assert FAKE not in capsys.readouterr().out


def test_setup_refuses_an_env_file_git_does_not_ignore(monkeypatch, tmp_path, capsys):
    """A tracked `.env` is a key in a git history, so this must not write.

    `git_ignores` is stubbed rather than a real repo being built, because the
    branch under test is what `setup` DOES with the answer, and a tri-state is
    easier to get wrong than to obtain.
    """
    monkeypatch.setattr(cli, "git_ignores", lambda p: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("USPTO_API_KEY", raising=False)
    env = tmp_path / ".env"
    secrets = iter([FAKE, ""])

    rc = cli.setup(env_path=env, ask=lambda q: "", ask_secret=lambda q: next(secrets))

    assert rc == 1
    assert not env.exists()
    out = capsys.readouterr().out
    assert "REFUSING" in out
    assert FAKE not in out


def test_load_env_never_clobbers_an_exported_name(monkeypatch, tmp_path):
    """`os.environ` wins over both files — config's own precedence.

    An exported variable is what someone typed on the command line for THIS
    run. A file quietly overriding it is how a `SELF_HEAL=0 python3 -m ...`
    invocation silently bills.
    """
    monkeypatch.setenv("SELF_HEAL", "0")
    monkeypatch.delenv("GP_ENABLED", raising=False)
    f = tmp_path / ".env"
    f.write_text("# a comment\nSELF_HEAL=1\nGP_ENABLED=1\n\nbroken line\n")

    cli.load_env((f,))

    import os
    assert os.environ["SELF_HEAL"] == "0"     # the export held
    assert os.environ["GP_ENABLED"] == "1"    # the file filled the gap


@pytest.mark.parametrize("returncode,expected", [(0, True), (1, False), (128, None)])
def test_git_ignores_is_a_tri_state(monkeypatch, returncode, expected):
    """"Cannot answer" is not "not ignored".

    Collapsing 128 into False would make `setup` refuse to write anywhere
    outside a git checkout, which is every scratch path in this file.
    """
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode, b"", b""))
    assert cli.git_ignores(config.REPO_ROOT / ".env") is expected
