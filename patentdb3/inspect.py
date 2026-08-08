"""Trace one shipped assay record back to the CALS cell it came from.

v3 has a dump, a manifest and an xlsx, and no way to answer "where did THIS
number come from" — see CLAUDE.md, "Backtrace every result to the code that
produced it". This is that tool. It answers one question: for a
(patent_id, cid) pair, put the shipped row and the patent's own source table
next to each other so a human can judge whether the extraction is right. No
automated check can make that judgement; this only assembles the two views.

    python3 -m patentdb3.inspect                        # patents in the dump
    python3 -m patentdb3.inspect US8952177               # cids for one patent
    python3 -m patentdb3.inspect US8952177 3              # record(s) + source row(s)
    python3 -m patentdb3.inspect US8952177 3 --full        # + the whole raw table block

Every mode prints the dump's provenance first (path, timestamp, self-heal mode,
row count) straight from `config.MANIFEST` — the artifact naming which run
produced `config.DUMP` — so a stale dump is never mistaken for a fresh one.
That was the single most expensive mistake in this project's history (see
CLAUDE.md: "37,593 compounds", the restored-backup incident).

Two views of a record, shown separately and never merged into one verdict:

  SHIPPED       — the row as it sits in config.DUMP right now.
  CURRENT CODE  — `uspto_assays.extract_from_patent` re-run on the same cached
                  XML, filtered to the same cid, this instant.

They can legitimately differ: the dump may have been written with the repair
tier on (`manifest["self_heal"]`), which can add rows a reader-only rerun will
never reproduce, and the code may simply have changed since the dump was
written. Both are reported as facts, not flagged as errors — this tool states
what each side is, not which one is right.

RAW SOURCE. For every table_id a record points at, `repair.gap.raw_block`
returns that patent's own `<tables>` element, structure intact, entities
unescaped. This prints only the `<row>` elements whose cells contain the cid
verbatim — usually one or two lines, always readable in a terminal — and falls
back to the first part of the whole block with a note if no row matches
(uncommon shapes: the id spread across a `namest`/`nameend` span, a row that
opens with a compound name instead of a bare id). `--full` prints the entire
block instead, for exactly that case.

No API calls, no network, no spend. XML is read only from the on-disk cache at
`config.XML_INPUT_DIR`; a patent not already cached there is reported as
missing rather than fetched. `extract_from_patent` is the deterministic reader
only — this never calls `repair_patent`, the one path in this package that can
spend money.
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys
from collections import defaultdict

from .core import config
from .repair.gap import raw_block
from .sources.uspto_assays import extract_from_patent

VALUE_FIELDS = ("value_numeric", "qualifier", "unit", "n_runs", "letter_grade",
                 "range_lo", "range_hi")


# ── dump + manifest ─────────────────────────────────────────────────────

def _load_dump() -> list[dict]:
    if not config.DUMP.exists():
        print(f"no dump at {config.DUMP}\n"
              f"run `python3 -m patentdb3.verify PID --dump` first")
        sys.exit(1)
    with config.DUMP.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _print_provenance() -> None:
    print(f"dump      {config.DUMP}")
    if not config.MANIFEST.exists():
        print(f"manifest  MISSING at {config.MANIFEST} — provenance of the dump "
              f"above is unknown; do not trust its numbers")
        print()
        return
    m = json.loads(config.MANIFEST.read_text())
    print(f"manifest  {config.MANIFEST}")
    print(f"written   {m.get('written_at')}   self_heal={m.get('self_heal')}   "
          f"rows={m.get('rows', 0):,}   patents={len(m.get('patents', []))}")
    if m.get("self_heal"):
        print(f"          repair tier: {m.get('gaps_found', 0)} gap(s), "
              f"{m.get('rules_adopted', 0)} rule(s) adopted, "
              f"+{m.get('rows_recovered', 0):,} recovered rows, "
              f"${m.get('usd_spent', 0):.4f} spent")
    print()


def _fmt_row(get) -> str:
    """Render a record's value the same way regardless of whether `get` reads
    a DUMP dict (all strings, '' for absent) or an AssayRecord (typed, None
    for absent)."""
    def val(f):
        v = get(f)
        return None if v in (None, "") else v

    v = val("value_numeric")
    if v is not None:
        shown = f"{v} {val('unit') or ''}".strip()
    else:
        lo, hi = val("range_lo"), val("range_hi")
        if lo is not None or hi is not None:
            shown = f"{lo or '?'}..{hi or '?'} {val('unit') or ''}".strip()
        elif val("letter_grade") is not None:
            shown = f"grade {val('letter_grade')}"
        else:
            shown = "(no value)"
    q = val("qualifier")
    if q:
        shown = f"{q}{shown}"
    n = val("n_runs")
    if n:
        shown += f"  (n={n})"
    return (f"  {val('assay_name') or '(unnamed)':<45} {shown:<22} "
            f"table={val('table_id') or '-':<16} col={val('column_header') or '-'!r}")


# ── raw source lookup ────────────────────────────────────────────────────

_ROW = re.compile(r"<row>(.*?)</row>", re.S)
_ENTRY = re.compile(r"<entry\b[^>]*?/>|<entry\b[^>]*>(.*?)</entry>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _cell_text(fragment: str) -> str:
    if "<" in fragment:
        fragment = _TAG.sub("", fragment)
    return " ".join(html.unescape(fragment).split())


def _rows_matching_cid(block: str, cid: str) -> list[str]:
    """Raw `<row>...</row>` text for every row with a cell reading `cid`
    verbatim — the same string comparison a human does by eye, nothing fancier.
    """
    hits = []
    for m in _ROW.finditer(block):
        cells = [_cell_text(e.group(1) or "") for e in _ENTRY.finditer(m.group(1))]
        if cid in cells:
            hits.append(m.group(0).strip())
    return hits


def _xml_for(pid: str) -> str | None:
    cached = config.XML_INPUT_DIR / f"{pid}.xml"
    if not cached.exists():
        return None
    return cached.read_text(errors="ignore")


def _show_source(xml: str, table_ids: list[str], cid: str, full: bool) -> None:
    for table_id in table_ids:
        block = raw_block(xml, table_id)
        print(f"\n  --- source: {table_id} ---")
        if not block:
            print(f"  (table_id {table_id!r} not found in the cached XML)")
            continue
        if full:
            print(block)
            continue
        hits = _rows_matching_cid(block, cid)
        if not hits:
            print(f"  (no <row> found with a cell reading exactly {cid!r} — "
                  f"showing the block's first 1200 chars; use --full for all of it)")
            print(block[:1200])
            continue
        for h in hits:
            print(f"  {h}")
        if len(hits) > 1:
            print(f"  ({len(hits)} rows matched {cid!r} in this table)")


# ── modes ─────────────────────────────────────────────────────────────────

def _list_patents(rows: list[dict]) -> None:
    counts = defaultdict(int)
    for r in rows:
        counts[r["patent_id"]] += 1
    print(f"{len(counts)} patent(s) in the dump:\n")
    for pid in sorted(counts):
        print(f"  {pid:<16} {counts[pid]:>6,} rows")
    print("\npython3 -m patentdb3.inspect PID           # list its cids")


def _list_cids(rows: list[dict], pid: str) -> None:
    mine = [r for r in rows if r["patent_id"] == pid]
    if not mine:
        print(f"{pid} is not in the dump (see the patent list above)")
        sys.exit(1)
    by_cid: dict[str, list[str]] = defaultdict(list)
    for r in mine:
        by_cid[r["cid"]].append(r["assay_name"])
    print(f"{pid}: {len(mine)} rows, {len(by_cid)} compound(s)\n")
    for cid in sorted(by_cid, key=lambda c: (len(c), c)):
        names = ", ".join(sorted(set(by_cid[cid]))[:3])
        print(f"  {cid:<8} {len(by_cid[cid]):>3} row(s)   {names}")
    print(f"\npython3 -m patentdb3.inspect {pid} CID       # inspect one compound")


def _inspect(rows: list[dict], pid: str, cid: str, full: bool) -> None:
    shipped = [r for r in rows if r["patent_id"] == pid and r["cid"] == cid]
    print(f"SHIPPED  ({pid}, cid={cid})")
    if not shipped:
        print("  (no rows for this patent+cid in the dump)")
    for r in shipped:
        print(_fmt_row(r.get))

    xml = _xml_for(pid)
    if xml is None:
        print(f"\nno cached XML for {pid} at {config.XML_INPUT_DIR / (pid + '.xml')} "
              f"— cannot show current-code output or source rows")
        return

    current = [r for r in extract_from_patent(xml) if str(r.cid) == cid]
    print(f"\nCURRENT CODE  (extract_from_patent, reader only, re-run just now, $0)")
    if not current:
        print("  (reader produces no rows for this cid right now)")
    for r in current:
        print(_fmt_row(lambda f, r=r: getattr(r, f, None)))

    table_ids = list(dict.fromkeys(
        [r["table_id"] for r in shipped if r["table_id"]]
        + [r.table_id for r in current if r.table_id]))
    if table_ids:
        _show_source(xml, table_ids, cid, full)
    else:
        print("\n(no table_id on any row — nothing to look up in the source XML)")


def main(argv: list[str]) -> int:
    full = "--full" in argv
    args = [a for a in argv[1:] if not a.startswith("-")]

    rows = _load_dump()
    _print_provenance()

    if len(args) == 0:
        _list_patents(rows)
    elif len(args) == 1:
        _list_cids(rows, args[0])
    else:
        _inspect(rows, args[0], args[1], full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
