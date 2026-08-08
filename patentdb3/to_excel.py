"""Turn the current reader dump into a workbook for eyeballing.

    python3 -m patentdb3.to_excel            # whatever the last dump run produced
    python3 -m patentdb3.to_excel --no-bdb   # skip the BindingDB sheet

READS THE DUMP PATH FROM THE MANIFEST, NEVER FROM A HARDCODED STRING.
`verify.py --dump` rewrites `out/latest.json` on every run; this reads that and
follows it. So the workbook is always built from the newest extraction, and if
the dump ever moves there is exactly one place that changes. Hardcoding a second
path here is how v2 ended up with consumers quietly reading stale files.

Output goes to ONE place, overwritten: `out/reader_dump.xlsx`.

TOLERANCE. Assay agreement is judged at 1%, not 5%. The reader does not round —
it reports the cell as printed — so the only legitimate gap between our value
and a reference is the reference's own rounding to two or three significant
figures. 5% is wide enough to call a genuinely different measurement a match,
which is the failure mode that lets a scorer improve while the data gets worse.

BindingDB is a REFERENCE. Nothing here writes a BDB value into an extracted
record, and a disagreement is reported, never corrected.
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

from .core import config


MANIFEST_PATH = config.MANIFEST
XLSX_PATH = config.XLSX
BDB_TSV = config.REPO_ROOT / "output" / "bindingdb" / "our_patents.tsv"

# See the module docstring. We do not round, so only the reference's rounding
# is forgivable.
TOLERANCE = 0.01


def _load() -> tuple[list[dict], dict]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"no manifest at {MANIFEST_PATH} — run `python3 -m patentdb3.verify <PID> --dump` first")
    man = json.loads(MANIFEST_PATH.read_text())
    dump = Path(man["dump"])
    if not dump.exists():
        sys.exit(f"manifest names {dump}, which does not exist")
    with dump.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return rows, man


def _bdb(pids: set[str]) -> dict[tuple[str, str], set[float]]:
    """Reference points as {(patent, cid) -> {value_nM}}.

    The ligand name holds `PATENT, CID` pairs joined by `::`, and one row is
    routinely cited against several patents at once — 97% of US8952177's rows
    are that shape. So the cid must be read from THIS patent's own pair; taking
    the first number in the string picks up a sibling patent's compound, whose
    numbering is unrelated.
    """
    out: dict[tuple[str, str], set[float]] = collections.defaultdict(set)
    if not BDB_TSV.exists():
        return out
    csv.field_size_limit(10 ** 9)
    with BDB_TSV.open(newline="") as fh:
        rd = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        hdr = next(rd)
        iP, iN = hdr.index("Patent Number"), hdr.index("BindingDB Ligand Name")
        vcols = [i for i, h in enumerate(hdr) if h in ("IC50 (nM)", "Ki (nM)")]
        for row in rd:
            if len(row) <= max(iP, iN):
                continue
            pid = next((p for p in pids if p in (row[iP] or "")), None)
            if not pid:
                continue
            m = re.search(rf"{pid},\s*(\d{{1,4}})\b", row[iN] or "")
            if not m:
                continue
            for i in vcols:
                v = (row[i] or "").strip().lstrip("<>=~ ")
                try:
                    out[(pid, m.group(1))].add(float(v))
                except ValueError:
                    pass
    return out


def main(argv: list[str]) -> int:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    rows, man = _load()
    pids = {r["patent_id"] for r in rows}
    wb = Workbook()

    ws = wb.active
    ws.title = "records"
    cols = list(rows[0].keys()) if rows else []
    ws.append(cols)
    for r in rows:
        ws.append([r[c] for c in cols])
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for i, c in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(i)].width = max(11, min(38, len(c) + 22))

    # one row per assay: how many measurements, what units, which tables
    sm = wb.create_sheet("assays")
    sm.append(["patent_id", "assay_name", "records", "units", "tables"])
    agg = collections.defaultdict(lambda: [0, set(), set()])
    for r in rows:
        a = agg[(r["patent_id"], r["assay_name"])]
        a[0] += 1
        a[1].add(r["unit"])
        a[2].add(r["table_id"])
    for (p, a), (n, u, t) in sorted(agg.items()):
        sm.append([p, a, n, ", ".join(sorted(x for x in u if x)), ", ".join(sorted(x for x in t if x))])
    for c in sm[1]:
        c.font = Font(bold=True)
    sm.freeze_panes = "A2"
    sm.column_dimensions["B"].width = 42

    if "--no-bdb" not in argv:
        ref = _bdb(pids)
        ours = collections.defaultdict(set)
        for r in rows:
            if r["value_numeric"] and r["unit"] == "uM":
                ours[(r["patent_id"], r["cid"])].add(float(r["value_numeric"]) * 1000)
        sb = wb.create_sheet("bindingdb")
        sb.append(["patent_id", "cid", "bdb_nM", "ours_nM", "agrees_1pct", "note"])
        hit = tot = 0
        for (p, c), vals in sorted(ref.items()):
            mine = ours.get((p, c), set())
            for v in sorted(vals):
                ok = any(abs(v - o) <= TOLERANCE * max(v, 1e-12) for o in mine)
                tot += 1
                hit += ok
                sb.append([p, c, v, ", ".join(f"{o:g}" for o in sorted(mine)) or "—",
                           "yes" if ok else "no",
                           "" if mine else "cid not extracted"])
        for c in sb[1]:
            c.font = Font(bold=True)
        sb.freeze_panes = "A2"
        print(f"bindingdb: {hit}/{tot} agree within {TOLERANCE:.0%}"
              + (f"  ({hit/tot:.1%})" if tot else ""))

    XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(XLSX_PATH)
    print(f"{len(rows):,} records from {man['written_at']} ({', '.join(sorted(pids))})")
    print(f"-> {XLSX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
