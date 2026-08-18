"""The front door: set the credentials up once, then run one patent.

    python3 -m patentdb3 setup                     # credentials + switches
    python3 -m patentdb3 run US9718825             # read it, print what came out
    python3 -m patentdb3 run US9718825 --dump      # + write the artifacts
    python3 -m patentdb3 run US9718825 --dump --no-heal   # reader only, $0

WHY A THIRD ENTRY POINT, AND WHAT IT IS NOT
--------------------------------------------
`verify.py` and `recognise.py` are operator tools. They assume you already know
that `ANTHROPIC_API_KEY` buys only the repair loops, that `USPTO_API_KEY` is
needed only for a patent nobody has fetched yet, and that an empty structures
dump can mean either "the route is switched off" or "the route ran and found
none". A new user knows none of that and there is nowhere to read it.

So this adds NO extraction and NO measurement of its own. `run` calls
`verify.one` and `verify.dump`, then reads the manifest and the two TSVs those
write and says what is in them in words. Every number it prints is read back
off an artifact, never accumulated in memory alongside one — so if a number
here disagrees with `out/latest.json`, the manifest is right and this is the
bug. That is the discipline in CLAUDE.md ("name the producer, check the
artifact") expressed as a rule about this file.

WHY IT LOADS `.env` AND THE OTHER ENTRY POINTS DO NOT
------------------------------------------------------
`core/config.py` reads `.env` for exactly ONE name, `ANTHROPIC_API_KEY`, and
`sources/uspto_xml._api_key` reads it for a second, `USPTO_API_KEY`. Every
switch below — `SELF_HEAL`, `IUPAC_NAMES`, `GP_ENABLED`, `MARKUSH_ASSEMBLY`,
`RECOGNISER_BACKEND`, `RECOGNISER` — is an `os.environ.get` and nothing else.

A `setup` that wrote `SELF_HEAL=0` into `.env` and stopped there would
therefore be a SILENT NO-OP: the file would say one thing, the next run would
do another, and the disagreement would surface as an unattributable number
weeks later. That is the failure class this repo's CLAUDE.md is mostly about,
so `main()` loads both `.env` files into `os.environ` BEFORE it imports
`config`, in config's own precedence — an already-exported name wins, then
`patentdb3/.env`, then the repo root's.

That load lives in `main()` and NOT at import time on purpose. `import
patentdb3.cli` from the test suite would otherwise push a developer's real
switches into every other test in the same process, and a suite whose result
depends on an untracked file is a suite that cannot be trusted to fail.

SECRETS ARE NEVER PRINTED, AND NEVER READ BACK
-----------------------------------------------
`setup` reports a key as `set` or `not set` and reads a new one through
`getpass`, so no value reaches stdout, a terminal scrollback, a screen share or
a CI log. It also refuses to write to an env file that git does not ignore —
checked with `git check-ignore` rather than by parsing `.gitignore`, because
the patterns there are root-anchored and negated in places and a hand-rolled
matcher would get exactly the case that matters wrong.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

# The same two files `config._load_key` reads, in the same order: nearest
# first, and the first file to name a variable wins.
ENV_FILES = (PACKAGE_ROOT / ".env", REPO_ROOT / ".env")

# WHERE A NEW VALUE IS WRITTEN — the repo root, and not `patentdb3/.env`.
# `config._load_key` reads both; `uspto_xml._api_key` reads only this one. A
# USPTO key written to the package copy would satisfy nothing and look set.
ENV_PATH = REPO_ROOT / ".env"

# The two credentials, and the ONE thing each buys. Both are optional: the
# reader is deterministic, offline and free, and OPSIN is a java subprocess.
KEYS = (
    ("ANTHROPIC_API_KEY",
     "the repair loops only (rule synthesis, name repair, markush assembly). "
     "Without it the reader still runs and `propose` returns None."),
    ("USPTO_API_KEY",
     "fetching a patent that is not already in output_v3/uspto_xml/. "
     "A cached patent needs no key and no network."),
)

# THE SWITCHES A USER IS LIKELY TO WANT, their legal values, and what each
# actually does. Read `core/config.py` for the measurements behind the
# defaults — this is a menu, not a second copy of that argument.
SWITCHES = (
    ("SELF_HEAL", ("1", "0"),
     "Run the paid repair loop after the reader. Off is the $0 baseline that "
     "any claim about what the loop recovers has to be stated against."),
    ("IUPAC_NAMES", ("1", "0"),
     "Run the identity track. Off writes an EMPTY structures dump, which is "
     "not the same as one that ran and found nothing."),
    ("GP_ENABLED", ("1", "0"),
     "Fetch Google Patents image URLs for compounds the patent only drew. "
     "GP is an image source here and never a structure source."),
    ("MARKUSH_ASSEMBLY", ("1", "0"),
     "Assemble substituent-table compounds from a scaffold plus its table. "
     "Costs $0 and yields nothing until the drawings are recognised."),
    ("RECOGNISER_BACKEND", ("off", "file", "colab"),
     "Where recognised structures are read from. `off` invents no molecule "
     "and waits on no GPU."),
    ("RECOGNISER", ("molscribe", "decimer"),
     "Which recogniser the results file is keyed on. MolScribe is the only "
     "one of the two that can mark an attachment point."),
)

_BOOL = ("1", "0")


# ── .env ──────────────────────────────────────────────────────────────────
def load_env(files=ENV_FILES) -> None:
    """Merge the env files into `os.environ`, nearest first, without clobbering.

    `setdefault` is the whole precedence rule: a name already exported beats
    every file, and the first file to carry it beats the rest. That is what
    `config._load_key` does for one variable; this does it for all of them, so
    a switch written by `setup` is a switch the next `run` in this process
    actually honours.
    """
    for path in files:
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            name, _, value = line.partition("=")
            os.environ.setdefault(name.strip(),
                                  value.strip().strip('"').strip("'"))


def git_ignores(path: Path) -> bool | None:
    """Does git ignore `path`? `None` when git cannot answer.

    Three outcomes, and the third is why this returns a tri-state rather than a
    bool: 0 ignored, 1 NOT ignored, anything else (no git binary, not a work
    tree, a path outside any checkout) means the question does not apply. Only
    a definite 1 is a reason to refuse — treating "cannot answer" as "not
    ignored" would make `setup` unusable outside a checkout, which is where a
    test writes its scratch env file.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(path.parent), "check-ignore", "-q", str(path)],
            capture_output=True)
    except OSError:
        return None
    return {0: True, 1: False}.get(r.returncode)


def write_env(path: Path, updates: dict[str, str]) -> None:
    """Set each name in `updates`, keeping every other line of the file as-is.

    Rewrites a name in place where it already appears and appends where it does
    not, so a hand-written comment or an unrelated variable survives. The
    process's own `os.environ` is updated too — otherwise `setup` would report a
    switch it just wrote and `run` in the same shell would use the old one.
    """
    lines = path.read_text().splitlines() if path.exists() else []
    for name, value in updates.items():
        row = f"{name}={value}"
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{name}="):
                lines[i] = row
                break
        else:
            lines.append(row)
        os.environ[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ── setup ─────────────────────────────────────────────────────────────────
def _switch_value(config, name: str, choices: tuple[str, ...]) -> str:
    """What the pipeline will actually use, in words rather than in `.env`'s `1`.

    Read off `config` and not off `os.environ`, so a variable that was never
    given a value is shown as the default it really takes rather than as blank.
    `on`/`off` because this column is read by a human deciding whether to change
    it; `write_env` is what has to speak `1`/`0`.
    """
    v = getattr(config, name, None)
    if choices == _BOOL:
        return "on" if v else "off"
    return str(v)


def setup(env_path: Path | None = None, *, ask=input,
          ask_secret=getpass.getpass) -> int:
    """Walk through the credentials and the switches, and write what changed.

    `ask` and `ask_secret` are arguments so the flow is testable without a tty
    and without a real key ever existing. `ask_secret` is `getpass` for the
    same reason the report says `set` rather than the value: a key echoed once
    is a key in a scrollback, a screen recording and a CI log.

    Writes NOTHING when nothing was entered. A blank answer keeps the current
    value, and the file is only opened if `updates` is non-empty — so running
    this to LOOK at the configuration cannot change it.
    """
    from .core import config

    path = Path(env_path) if env_path else ENV_PATH
    updates: dict[str, str] = {}

    print(f"\npatentdb3 setup\n\nCredentials -> {path}")
    print("  Both are optional. Nothing here is ever printed back.\n")
    for name, needed_for in KEYS:
        state = "set" if os.environ.get(name) else "not set"
        print(f"  {name:<20}{state}")
        print(f"  {'':<20}needed for: {needed_for}\n")

    for name, _needed in KEYS:
        got = ask_secret(f"  new {name} (blank = keep current): ").strip()
        if got:
            updates[name] = got

    print("\nSwitches (patentdb3/core/config.py)\n")
    for i, (name, choices, what) in enumerate(SWITCHES, 1):
        print(f"  {i}  {name:<20}{_switch_value(config, name, choices)}")
        print(f"     {what}")

    print("\n  Enter a number to change one. Blank when you are done.")
    while True:
        pick = ask("  switch number: ").strip()
        if not pick:
            break
        if not pick.isdigit() or not 1 <= int(pick) <= len(SWITCHES):
            print(f"  not a switch number (1-{len(SWITCHES)})")
            continue
        name, choices, _what = SWITCHES[int(pick) - 1]
        shown = "on/off" if choices == _BOOL else "/".join(choices)
        value = ask(f"  {name} [{shown}]: ").strip().lower()
        if choices == _BOOL:
            value = {"on": "1", "yes": "1", "true": "1",
                     "off": "0", "no": "0", "false": "0"}.get(value, value)
        if value not in choices:
            print(f"  {name} takes {shown}")
            continue
        updates[name] = value
        print(f"  {name} -> {value}")

    if not updates:
        print("\nnothing changed.")
        return 0

    ignored = git_ignores(path)
    if ignored is False:
        # REFUSED, NOT WARNED. The next `git add -A` would commit a live key,
        # and a secret in a git history is not removed by deleting the line.
        print(f"\nREFUSING to write: git does not ignore {path}.")
        print("Add `.env` to .gitignore, then run setup again.")
        return 1

    write_env(path, updates)
    print(f"\nwrote {', '.join(sorted(updates))} -> {path}")
    print("Values are not shown. Re-run `setup` to see which names are set.")
    return 0


# ── run ───────────────────────────────────────────────────────────────────
def _rows(path: Path, patent_id: str):
    """This patent's rows out of a TSV artifact. Empty when it is not there."""
    if not path.exists():
        return
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("patent_id") == patent_id:
                yield row


def _artifact(recorded: str, fallback: Path) -> Path:
    """The manifest's own path, falling back to the configured one.

    Same precedence as `to_excel._load`: the recorded path is the record of
    what a run wrote and wins while it exists, and the configured location is
    reached for only once the checkout has moved and the record has gone stale.
    """
    p = Path(recorded)
    return p if p.exists() else fallback


# Every structures row falls in exactly one of these, and the four are printed
# whether or not they are zero. THE FIRST VERSION PRINTED ONLY THE FIRST TWO
# and it was wrong in the way that matters: US9718825 writes 636 structure rows
# of which 35 resolve and 601 are `substituent_table` markush rows — a cid, no
# SMILES, no drawing of its own. Those 601 vanished between `dump`'s own "wrote
# 636 structures" and this report, with nothing saying where. An unexplained
# gap between two numbers on one screen is how a reader learns to trust
# neither, so the buckets are exhaustive by construction and `_bucket` has no
# way to return anything else.
BUCKETS = (
    ("resolved to a structure", "resolved",
     "joined to the patent's own compound id"),
    ("drawn, not named", "drawn",
     "the structure is a picture — see `recognise`"),
    ("markush, not assembled", "markush",
     "a scaffold plus a substituent table — see MARKUSH_ASSEMBLY"),
    ("no structure of any kind", "none",
     "the patent gives this id nothing to resolve"),
)
# Best-first, and it has to be applied PER COMPOUND and not per row. The two
# identity routes both write a row for a cid they both reach, so counting
# distinct cids bucket by bucket double-counts: US10730863 read 70 resolved and
# 364 drawn against 428 assay compounds — 101.4%, printed with a straight face.
# `_by_cid` keeps the best bucket each cid reached, so the four are disjoint and
# their percentages cannot exceed 100.
_RANK = {b: i for i, (_t, b, _w) in enumerate(BUCKETS)}


def _bucket(row: dict) -> str:
    """Which of `BUCKETS` one structures row is. Precedence, not a guess.

    A SMILES beats everything because it is an answer; a drawing beats a
    markush flag because a whole-compound picture is one recognition call away
    while an assembly needs the table too.
    """
    if row.get("smiles"):
        return "resolved"
    if row.get("drawn_ref"):
        return "drawn"
    if row.get("markush_kind"):
        return "markush"
    return "none"


def report(patent_id: str, manifest_path: Path) -> None:
    """Say what landed in the artifacts, in words, reading them back off disk.

    Deliberately NOT accumulated while `dump` runs. A count carried in memory
    beside the file it describes is a count that can disagree with it, and this
    project has already paid for one artifact that its own manifest no longer
    matched.
    """
    from .core import config

    man = json.loads(manifest_path.read_text())
    dump_path = _artifact(man["dump"], config.DUMP)
    struct_path = _artifact(man["structures"], config.STRUCTURES)

    assay_cids = {r["cid"] for r in _rows(dump_path, patent_id) if r.get("cid")}
    by_cid: dict[str, str] = {}
    rows: Counter = Counter()
    mass: Counter = Counter()
    for r in _rows(struct_path, patent_id):
        b = _bucket(r)
        rows[b] += 1
        cid = r.get("cid")
        if cid and _RANK[b] < _RANK.get(by_cid.get(cid, "none"), 99):
            by_cid[cid] = b
        if b == "resolved":
            mass[r.get("mass_check") or "unchecked"] += 1

    # THE POPULATION IS THE ASSAY COMPOUNDS, AND ONLY THOSE. An assay compound
    # with no structures row at all is `none` — it is a miss, and leaving it
    # out of the tally would hide exactly the compounds this pipeline exists to
    # find. Conversely `table_names` resolves ids the assay tables never
    # measure (6 of them on US10730863), and counting those against this
    # denominator printed 101.4%. They are reported on their own line instead.
    placed = Counter(by_cid.get(cid, "none") for cid in assay_cids)
    off_population = sorted(set(by_cid) - assay_cids)

    n = len(assay_cids) or 1        # denominator, named on its own line
    print(f"\n{patent_id} — what came out")
    print(f"  assay records          {man['rows']:>8,}   "
          f"one row per (compound, assay) measurement")
    print(f"  assay compounds        {len(assay_cids):>8,}   "
          f"the distinct compounds those rows measure — the denominator below")
    print(f"\n  of those {len(assay_cids):,} compounds, by what the patent gives:")
    for title, b, what in BUCKETS:
        print(f"    {title:<24}{placed[b]:>7,}  "
              f"{100 * placed[b] / n:>5.1f}%   {what}")
    print(f"  {sum(rows.values()):,} structures row(s) written in total "
          f"(manifest: {man.get('structures_rows', 0):,}). A compound both "
          f"routes reach is one compound and two rows.")
    if off_population:
        print(f"  {len(off_population):,} further compound(s) were resolved that "
              f"no assay table measures — outside the denominator above.")
    if not man.get("iupac_names"):
        print("  IDENTITY TRACK OFF (IUPAC_NAMES=0) — every line above is 0 by "
              "configuration, not by measurement.")

    agrees, contra = mass["agrees"], mass["contradicts"]
    resolved_rows = rows["resolved"]
    print(f"\n  the patent's own printed mass confirms {agrees:,} of the "
          f"{agrees + contra:,} structures it can weigh")
    if contra:
        # NOT A FAILURE COUNT AND NOT DROPPED ROWS. `mass_gate` stamps the
        # verdict and ships the row; the loss log names each one.
        print(f"  {contra:,} contradict the mass their own row prints — each is "
              f"in {man.get('loss_log', 'the loss log')}")
    if mass.get("gate_unavailable"):
        print(f"  {mass['gate_unavailable']:,} more could have been weighed, but "
              f"rdkit is not installed — 'not checked', never 'fine'")
    print(f"  the other {resolved_rows - agrees - contra:,} of {resolved_rows:,} "
          f"resolved row(s) print no mass, so they are NOT CHECKABLE.")
    print("  Unchecked never means checked and fine.")

    if man.get("self_heal"):
        print(f"\n  repair loop: {man.get('gaps_found', 0)} gap(s), "
              f"{man.get('rules_adopted', 0)} rule(s) adopted, "
              f"+{man.get('rows_recovered', 0):,} rows, "
              f"${man.get('usd_spent', 0):.4f} spent")
    else:
        print("\n  repair loop: OFF — this is the deterministic reader, $0")

    print("\n  where it landed:")
    print(f"    assay records   {dump_path}")
    print(f"    structures      {struct_path}")
    print(f"    manifest        {manifest_path}")


def run(patent_id: str, *, write: bool, heal: bool | None) -> int:
    """Read one patent and report. Returns a process exit code.

    Calls `verify.one` and `verify.dump` and reimplements neither. `write` maps
    to verify's `--dump`, which is also the only switch that runs the identity
    routes and the mass gate at all — they live inside `dump()`, not inside the
    reader — so a run without it can report assay records and nothing about
    compounds. That is stated on the way out rather than shown as a zero.
    """
    from . import verify
    from .core import config
    from .sources.uspto_xml import UsptoUnavailable

    pid = patent_id.strip().upper()
    cached = config.XML_INPUT_DIR / f"{pid}.xml"
    if not cached.exists() and not os.environ.get("USPTO_API_KEY", ""):
        # A PLAIN SENTENCE, NOT A TRACEBACK. Without the key this ends as
        # `UsptoUnavailable("USPTO_API_KEY is not set")` raised three frames
        # down inside a fetch, which reads as a broken install rather than as
        # the one missing setting it is.
        print(f"{pid} is not cached in {config.XML_INPUT_DIR}, and "
              f"USPTO_API_KEY is not set.")
        print("A cached patent needs no key. To fetch this one, get a free key "
              "from data.uspto.gov and run:")
        print("    python3 -m patentdb3 setup")
        return 1

    if write and heal is not False and config.SELF_HEAL:
        print("self-heal is ON: this calls the API and spends up to "
              f"${config.PER_PATENT_LM_CAP:.2f} on this patent. "
              "--no-heal is the free reader.")

    try:
        verify.one(pid, show=0)
        if write:
            verify.dump([pid], heal=heal)
    except (FileNotFoundError, UsptoUnavailable) as e:
        print(f"{pid}: no grant XML — {e}")
        print("Grant full-text XML begins in 2002; there is none before that.")
        return 1

    if not write:
        print(f"\n{pid}: nothing was written, and no compound was resolved.")
        print("The identity track and the mass check both run inside "
              "verify.dump(), so add --dump to get them and the artifacts.")
        return 0

    report(pid, verify.MANIFEST_PATH)
    return 0


# ── entry point ───────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="patentdb3",
        description="Patent ID in, structured chemistry out.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "setup",
        description="Show what the pipeline needs, and write it to .env. "
                    "No key is ever printed back.",
        help="credentials and switches, written to .env")

    p = sub.add_parser(
        "run",
        description="Read one patent and report what came out.",
        help="run one patent and report what came out")
    p.add_argument("patent_id", help="a grant number, e.g. US9718825")
    p.add_argument("--dump", action="store_true",
                   help="write the artifacts. Also the only way the identity "
                        "routes and the mass gate run at all — they live "
                        "inside verify.dump().")
    p.add_argument("--no-heal", action="store_true",
                   help="skip the paid repair loops. The $0 reader baseline.")
    return ap


def main(argv: list[str] | None = None) -> int:
    # BEFORE `config` IS IMPORTED ANYWHERE. `config` snapshots every switch at
    # import time, so loading the env files after it would leave `setup`
    # reporting one value and the pipeline using another.
    load_env()
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "setup":
            return setup()
        return run(a.patent_id, write=a.dump,
                   heal=False if a.no_heal else None)
    except (EOFError, KeyboardInterrupt):
        # `setup` on a pipe hits EOF on the first prompt. That is a cancelled
        # session, not a crash, and it must not print a stack.
        print("\ncancelled — nothing was written.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
