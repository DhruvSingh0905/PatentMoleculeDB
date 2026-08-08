"""Minimal functional check for v3 — is the reader alive and what does it emit?

Deliberately NOT an output pipeline. It merges nothing and resolves no names.
Its jobs are to prove the move works end to end and to show the reader's RAW
shape, because the assembly is going to be designed around what this dumps
rather than around what anyone assumes it produces.

    python3 -m patentdb3.verify                 # summary over a default sample
    python3 -m patentdb3.verify US8952177       # one patent, with rows
    python3 -m patentdb3.verify --all           # every cached XML
    python3 -m patentdb3.verify US8952177 --dump          # heals (SELF_HEAL=1)
    python3 -m patentdb3.verify US8952177 --dump --no-heal  # reader only, $0

SELF-HEAL. `--dump` runs the repair loop per patent by default, governed by
`config.SELF_HEAL`. `--no-heal` (or `SELF_HEAL=0`) gives the deterministic
reader alone, which is the baseline any claim about what the loop recovers has
to be stated against. The manifest records which mode ran, so the two dumps are
never mistaken for each other.

THE DUMP GOES TO ONE PLACE, AND THIS IS IT:

    patentdb3/out/reader_dump.tsv        (DUMP_PATH, below)

**Overwrite that file. Do not add a second one.** Every later stage — the pivot
to Jie's wide sheet, any scoring, any hand-check — reads DUMP_PATH. The failure
this rule exists to prevent is the one that shaped v2: a new output file gets
added beside the old one, both are half-maintained, and the variables pointing
at them drift until nobody can say which artifact a number came from. If this
script needs to emit something different, change what it writes here rather
than writing somewhere else.

The dump is the reader's records as-is — one row per (compound, assay)
measurement, every field the reader carries, nothing derived. It is long
format; Jie's sheet is wide, and that pivot is a later step that reads this.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from datetime import datetime

from .core import config
from .core.cost_tracker import cost_tracker
from .repair.loop import repair_patent
from .repair.rules import RuleLibrary
from .sources.uspto_assays import extract_from_patent
from .sources.uspto_xml import UsptoUnavailable, fetch_grant_xml, parse_fidelity, parse_tables

# v2's cached grant XML — an INPUT. v3 writes nothing here; see config.
XML_DIR = config.XML_INPUT_DIR

# The one canonical dump location. See the module docstring before changing it.
# Both defined in core.config — the single place every path lives, so a new
# consumer imports from there rather than from this script.
DUMP_PATH = config.DUMP
MANIFEST_PATH = config.MANIFEST
FIELDS = ("cid", "assay_name", "value_numeric", "qualifier", "unit", "n_runs",
          "letter_grade", "range_lo", "range_hi", "table_id", "column_header")


def _xml(pid: str) -> str:
    """The patent's grant XML, fetched on first use and cached thereafter.

    Fetching only became possible once `fetch_grant_xml` was ported into v3 —
    before that this read the directory and raised if the document was not
    already there, so the corpus was frozen at whatever someone else had
    downloaded. One search call plus one ~600 KB GET, and free.
    """
    cached = XML_DIR / f"{pid}.xml"
    if cached.exists():
        return cached.read_text(errors="ignore")
    return fetch_grant_xml(pid)


def one(pid: str, *, show: int = 12) -> tuple[int, int]:
    """Print what the reader emits for one patent. Returns (records, usable)."""
    xml = _xml(pid)
    tables = parse_tables(xml)
    fid = parse_fidelity(xml)
    recs = extract_from_patent(xml)
    usable = [r for r in recs if getattr(r, "is_usable", True)]
    cids = {r.cid for r in recs}

    print(f"\n{pid}")
    print(f"  tables {len(tables):>5}   records {len(recs):>6,}   "
          f"usable {len(usable):>6,}   compounds {len(cids):>5,}")
    # parse_fidelity's shape varies; print it rather than assume a field.
    print(f"  fidelity: {fid}")

    if show and recs:
        print(f"  {'cid':<10}{'assay_name':<34}{'value':>12} {'unit':<6}"
              f"{'n':>3}  {'table':<18}{'column_header'}")
        for r in recs[:show]:
            v = r.value_numeric if r.value_numeric is not None else (
                r.letter_grade or f"{r.range_lo}..{r.range_hi}")
            print(f"  {str(r.cid)[:9]:<10}{str(r.assay_name)[:33]:<34}"
                  f"{str(v)[:11]:>12} {str(r.unit or '')[:5]:<6}"
                  f"{str(r.n_runs or ''):>3}  {str(r.table_id or '')[:17]:<18}"
                  f"{str(r.column_header or '')[:40]}")

    # what a row actually carries — the question the assembly turns on
    missing = Counter()
    for r in recs:
        for f in ("table_id", "column_header", "unit"):
            if not getattr(r, f, None):
                missing[f] += 1
    if recs:
        print("  rows missing:  " + "  ".join(
            f"{f}={missing[f]:,} ({missing[f]/len(recs):.0%})" for f in
            ("table_id", "column_header", "unit")))
    return len(recs), len(usable)


def dump(pids: list[str], *, heal: bool | None = None) -> int:
    """Write every record produced for `pids` to DUMP_PATH, as TSV.

    Overwrites. One row per measurement, every field the reader carries and
    nothing derived — no unit conversion, no pivot, no name resolution. What a
    later stage needs that is not here is a signal the READER should carry it,
    not that this script should compute it.

    HEALING RUNS PER PATENT, not as a batch afterwards. `repair_patent` takes
    one document, finds ITS gaps against ITS own baseline, and returns
    `baseline + recovered` — so the loop is scoped to the patent in front of it
    and never sweeps the corpus. `heal=None` follows `config.SELF_HEAL`.

    This function previously called `extract_from_patent` directly and so
    exercised none of the repair tier. A dump produced that way is a reader
    measurement, and reading it as a pipeline measurement is the exact mistake
    this project keeps paying for — which is why the manifest now records the
    mode.
    """
    heal = config.SELF_HEAL if heal is None else heal
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    lib = RuleLibrary() if heal else None
    n = 0
    done: list[str] = []
    recovered_total = adopted_total = gaps_total = 0
    spend_before = cost_tracker.total_cost
    with DUMP_PATH.open("w") as fh:
        fh.write("patent_id\t" + "\t".join(FIELDS) + "\n")
        for pid in pids:
            try:
                if heal:
                    recs, rep = repair_patent(pid, _xml(pid), library=lib)
                    gaps_total += rep.gaps_found
                    adopted_total += rep.adopted
                    recovered_total += rep.rows_recovered
                    if rep.gaps_found or rep.rows_recovered:
                        print(f"  {pid}: {rep.gaps_found} gap(s), "
                              f"{rep.already_known} known, {rep.adopted} adopted, "
                              f"+{rep.rows_recovered:,} rows, "
                              f"{rep.escalated} escalated")
                else:
                    recs = extract_from_patent(_xml(pid))
            except (FileNotFoundError, UsptoUnavailable) as e:
                print(f"  {pid}: no XML ({e})")
                continue
            for r in recs:
                vals = [str(getattr(r, f, "") if getattr(r, f, None) is not None else "")
                        .replace("\t", " ").replace("\n", " ") for f in FIELDS]
                fh.write(pid + "\t" + "\t".join(vals) + "\n")
                n += 1
            done.append(pid)

    # The manifest. Written EVERY run, overwritten in place, so it always names
    # the current dump. Anything downstream reads the path from here rather than
    # hardcoding one — that is what stops a consumer quietly reading a stale file
    # after the dump moves or is regenerated, which is exactly how v2 ended up
    # with numbers nobody could attribute to a run.
    spent = cost_tracker.total_cost - spend_before
    MANIFEST_PATH.write_text(json.dumps({
        "dump": str(DUMP_PATH),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "patents": done,
        "rows": n,
        "fields": ["patent_id", *FIELDS],
        "source_xml_dir": str(XML_DIR),
        # WHICH PIPELINE PRODUCED THIS. Without it, a reader-only dump and a
        # healed dump are indistinguishable files, and every comparison between
        # them silently crosses a code boundary.
        "self_heal": heal,
        "gaps_found": gaps_total if heal else None,
        "rules_adopted": adopted_total if heal else None,
        "rows_recovered": recovered_total if heal else None,
        "usd_spent": round(spent, 4) if heal else 0.0,
    }, indent=2) + "\n")

    print(f"\nwrote {n:,} rows for {len(done)} patent(s) -> {DUMP_PATH}")
    if heal:
        print(f"self-heal ON: {gaps_total} gap(s), {adopted_total} rule(s) adopted, "
              f"+{recovered_total:,} rows recovered, ${spent:.4f} spent")
    else:
        print("self-heal OFF (SELF_HEAL=0) — reader only, $0")
    print(f"manifest -> {MANIFEST_PATH}")
    return n


def main(argv: list[str]) -> int:
    if "--dump" in argv:
        pids = [a for a in argv[1:] if not a.startswith("-")] or ["US8952177"]
        # CLI beats the env var, which beats the default. `--no-heal` is how you
        # get the reader-only baseline any recovery claim must be stated against.
        heal = False if "--no-heal" in argv else (True if "--heal" in argv else None)
        for pid in pids:
            one(pid, show=6)
        dump(pids, heal=heal)
        return 0
    if "--all" in argv:
        pids = sorted(p.split("/")[-1][:-4] for p in glob.glob(str(XML_DIR / "*.xml")))
        show = 0
    elif len(argv) > 1:
        pids, show = [a for a in argv[1:] if not a.startswith("-")], 12
    else:
        pids, show = ["US8952177", "US9718790", "US10544143"], 6

    tot = use = 0
    for pid in pids:
        try:
            r, u = one(pid, show=show)
        except FileNotFoundError:
            print(f"\n{pid}\n  no cached XML")
            continue
        tot += r
        use += u
    print(f"\n{len(pids)} patents   {tot:,} records   {use:,} usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
