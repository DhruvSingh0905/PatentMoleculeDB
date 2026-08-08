"""What each patent scored, remembered per patent instead of re-derived.

The problem this replaces
------------------------
`parser_repair.baseline_counts()` is the floor a capability patch is judged
against: distinct usable COMPOUNDS per patent, summed over the corpus. It was
computed by re-extracting every cached XML on every call — 137 files, 119 MB,
`extract_from_patent` and `parse_fidelity` each once per file.

Measured on US20240010684A1, a **15-compound patent whose gap was worth 3 rows**
(`autoheal: US20240010684A1 yielded nothing a rule can fix (3 rows at stake)`):

    repair_capabilities              39.53 s   untraced, PARSER_REPAIR_APPLY=0
      verify_patch                   22.49 s     probe 16.59 + suite 5.82
      baseline_counts                14.04 s   <- this module
      _bad_values_now                 2.51 s

Under the call tracer the same call reads 74.68 s of a 154.65 s run (48.3%),
with `baseline_counts` at 47.69 s; the tracer inflates pure-Python work ~3.4x
and leaves the subprocess in `verify_patch` alone, which is why both numbers
are quoted.

`_BASELINE_CACHE` already memoised it per process, keyed on the patchable
modules' `st_mtime_ns`. That cache dies with the process — and a landed patch,
which is precisely when the next gap is about to ask, invalidates it.

The observation
---------------
The before-side of the comparison is a pure function of (extraction code, XML).
It cannot change while neither does, and when it DOES change — a patch lands —
the number under the new code was *already computed*, in the sandbox, by the
probe that verified the patch. `verify_patch` returns `per_patent` for the whole
corpus under exactly the tree that is about to be written. So the ledger is
maintained by the runs that would otherwise invalidate it, and the steady-state
cost of a baseline is a file read.

What this does NOT do
---------------------
It does not narrow the population. `verify_patch`'s blocking condition compares
a corpus-wide before-total to a corpus-wide after-total, and feeding it a
baseline covering fewer patents than the probe measured would compare two
numbers drawn from different populations — the single most common error this
repo has recorded. `counts()` therefore always returns an entry for every XML in
the directory; it just declines to re-measure the ones nothing has touched.

The per-patent record
---------------------
Each patent carries what it scores NOW and the best it has ever scored, with the
code state that achieved it:

    "US10660877": {
      "compounds": 860, "clean": true, "code_key": "9f3c…", "at": "…",
      "best": {"compounds": 860, "code_key": "9f3c…", "journal_id": "0007-ab12cd34"}
    }

`best` exists because "did the corpus total fall" cannot see a patent walked
down in steps. US10660877 went 860 -> 0 on one `_is_namelike` patch that touched
none of its rows; a sequence that takes it 860 -> 800 -> 700 while the corpus
total rises each time passes the blocking condition three times, and nothing
today remembers the 860. `regressions_vs_best()` answers that from a file read,
which is what lets a later patch be judged against the best combination seen for
THIS patent without re-deriving the corpus.

Recorded, never enforced — the same call the rest of this tier makes. Coverage
is the one condition that blocks; everything else is journalled and applied
anyway, because every judgement-shaped gate here has eventually blocked
something correct.

The code key
------------
A CONTENT hash, not mtimes. `_baseline_key` used `st_mtime_ns`, which a `touch`
moves without changing behaviour and which a `--revert` moves while restoring
the exact bytes that were measured before. Content-addressing makes a revert
land back on the entries it already has.

It covers every `*.py` under `patentdb/sources/` plus the tracked assay
vocabulary — that is where all `capability.PATCHABLE` targets live, where
`AssayRecord.is_usable` is defined (`uspto_assays.py:416`), and what
`uspto_assays` reads at `config.PACKAGE_ROOT / "data" / "assay_vocabulary.json"`.
It is a superset of what the old mtime key covered.

It is NOT complete, and the gap is worth stating: a change under `core/` that
alters extraction — a new `AssayRow` field, a units table — does not move the
key, so the ledger would serve a stale number. `core/` was left out because it
changes constantly for reasons that do not touch extraction and would thrash
every entry. `invalidate()` is the escape hatch, and `stats()["code_key"]` makes
the current key readable so a suspect ledger can be checked rather than guessed
at.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from ..core import config

logger = logging.getLogger(__name__)

VERSION = 1

# Beside `parser_repair_journal.jsonl` and `escalation_journal.jsonl`, which is
# where this tier's other durable state already lives.
LEDGER_PATH = config.OUTPUT_DIR / "repair_baselines.json"


def _code_files() -> list[Path]:
    """Every file whose CONTENT decides what `extract_from_patent` produces."""
    src = Path(__file__).resolve().parent.parent / "sources"
    out = sorted(p for p in src.glob("*.py") if p.is_file())
    vocab = config.PACKAGE_ROOT / "data" / "assay_vocabulary.json"
    if vocab.exists():
        out.append(vocab)
    return out


def code_key() -> str:
    """A content hash of the extraction sources. Cheap: ~10 files, read once."""
    h = hashlib.sha256()
    for p in _code_files():
        try:
            body = p.read_bytes()
        except OSError:                          # unreadable is a state too
            body = b""
        h.update(p.name.encode())
        h.update(hashlib.sha256(body).digest())
    return h.hexdigest()[:16]


def load() -> dict:
    """The ledger, or an empty one. A corrupt file is never an error here."""
    try:
        data = json.loads(LEDGER_PATH.read_text())
    except (OSError, ValueError):
        return {"version": VERSION, "patents": {}}
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return {"version": VERSION, "patents": {}}
    data.setdefault("patents", {})
    return data


def _save(data: dict) -> None:
    """Write atomically. A half-written ledger reads as empty and re-measures,
    which is slow but correct; a torn one that parses would be neither."""
    try:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LEDGER_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(LEDGER_PATH)
    except OSError as e:                         # never let bookkeeping break a run
        logger.warning("ledger: could not write %s: %r", LEDGER_PATH.name, e)


def _touch_best(entry: dict, n: int, ck: str, journal_id: str | None) -> None:
    """Promote `best` when this measurement beats it. Ties keep the older one —
    the first code state to reach a number is the one worth naming."""
    best = entry.get("best") or {}
    if n > int(best.get("compounds", -1)):
        entry["best"] = {"compounds": n, "code_key": ck,
                         "journal_id": journal_id, "at": _now()}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def counts(xml_dir: Path, measure) -> dict:
    """`{pid: compounds, "_clean": set}` for every XML in `xml_dir`.

    `measure(path) -> (compounds, clean)` is called ONLY for patents with no
    entry or an entry from a different code state. Everything else is read.

    THERE IS NO per-patent restriction, and the omission is the design.

    Scoping the measurement to the patent being repaired is the obvious way to
    make a 3-row gap cheap, and it cannot be done here safely. `verify_patch`
    blocks on `sum(before) < sum(after)` where the after-side is the probe's
    corpus-wide total; serving a remembered number for the patents the caller
    declined to re-measure puts a stale addend into that sum, and serving
    nothing for them puts a hole in it. Understated, a patch passes a condition
    it should have failed; overstated, a correct patch is declined. Either way
    the totals still LOOK healthy, which is the shape of nearly every defect
    this loop has recorded.

    So the saving here is temporal, not populational: a number is not re-derived
    while the code that produced it has not moved. That is worth the whole cost
    (14.04 s -> a file read on US20240010684A1) without weakening the condition
    by one patent, because the case where the ledger IS stale — a patch just
    landed — is handled by `record()` rather than by a rescan.
    """
    ck = code_key()
    data = load()
    per = data["patents"]

    out: dict = {}
    clean: set[str] = set()
    measured = reused = 0
    for path in sorted(xml_dir.glob("*.xml")):
        pid = path.stem
        entry = per.get(pid)
        if entry and entry.get("code_key") == ck:
            out[pid] = int(entry["compounds"])
            if entry.get("clean"):
                clean.add(pid)
            reused += 1
            continue
        try:
            n, is_clean = measure(path)
        except Exception as e:                   # noqa: BLE001 — probe only
            logger.warning("ledger: %s could not be measured (%r)", pid, e)
            continue
        out[pid] = n
        if is_clean:
            clean.add(pid)
        entry = per.setdefault(pid, {})
        entry.update(compounds=n, clean=bool(is_clean), code_key=ck, at=_now())
        _touch_best(entry, n, ck, entry.get("best", {}).get("journal_id"))
        measured += 1

    if measured:
        _save(data)
    logger.info("ledger: baseline for %d patent(s) — %d measured, %d reused",
                len(out), measured, reused)
    out["_clean"] = clean
    return out


def record(per_patent: dict, *, ck: str | None = None,
           per_clean: dict | None = None, journal_id: str | None = None) -> int:
    """Adopt a corpus measurement someone else already paid for.

    Called with `verify_patch`'s probe output at the moment a patch is written
    to the tree. The probe ran the patched modules over every cached XML in a
    sandbox copy of the tracked tree, so its `per_patent` IS the new baseline —
    computing it again would be re-deriving a number we are holding.

    A patent the probe scored -1 (extraction RAISED) is skipped rather than
    recorded as zero: a crash is not a measurement, and writing it would hand
    the next patch a floor of nothing to clear.
    """
    ck = ck or code_key()
    data = load()
    per = data["patents"]
    n_written = 0
    for pid, n in (per_patent or {}).items():
        if pid == "_clean" or not isinstance(n, int) or n < 0:
            continue
        entry = per.setdefault(pid, {})
        entry.update(compounds=n, code_key=ck, at=_now())
        if per_clean is not None and pid in per_clean:
            entry["clean"] = bool(per_clean[pid])
        _touch_best(entry, n, ck, journal_id)
        n_written += 1
    if n_written:
        _save(data)
        logger.info("ledger: adopted %d per-patent count(s) at code_key %s",
                    n_written, ck)
    return n_written


def best(patent_id: str) -> dict | None:
    """The best this patent has ever scored, and the code state that did it."""
    return (load()["patents"].get(patent_id) or {}).get("best")


def regressions_vs_best(per_patent: dict) -> dict:
    """`{pid: [best_ever, now]}` for every patent now below its own record.

    The protection the corpus total cannot give. `verify_patch` asks whether the
    SUM fell; this asks whether any individual document was walked down, however
    many patches ago its high-water mark was set.

    Reported, never enforced — a patch that genuinely supersedes an old reading
    should be allowed to lower it, and the journal is what makes that reviewable.
    """
    per = load()["patents"]
    out: dict[str, list[int]] = {}
    for pid, now in (per_patent or {}).items():
        if pid == "_clean" or not isinstance(now, int) or now < 0:
            continue
        top = int(((per.get(pid) or {}).get("best") or {}).get("compounds", -1))
        if top > now:
            out[pid] = [top, now]
    return out


def invalidate() -> bool:
    """Forget every measurement. The escape hatch for a key that missed a change.

    Deletes the file rather than rewriting it, so the next `counts()` re-measures
    from the current tree with no stale `best` surviving.
    """
    try:
        LEDGER_PATH.unlink()
        return True
    except OSError:
        return False


def stats() -> dict:
    """What the ledger holds and whether it matches the tree it was measured on."""
    ck = code_key()
    per = load()["patents"]
    fresh = sum(1 for e in per.values() if e.get("code_key") == ck)
    return {"path": str(LEDGER_PATH), "code_key": ck, "patents": len(per),
            "fresh": fresh, "stale": len(per) - fresh,
            "compounds": sum(int(e.get("compounds", 0)) for e in per.values())}
