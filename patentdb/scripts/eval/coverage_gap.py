"""Coverage-gap diagnostic — run on RAW source, never on the parsed view.

Why this exists
---------------
Three patents were wrongly written off as structurally unextractable
(US11566007, US11292791, US20240335431). Every one of those verdicts came from
testing "is the data in this document?" with the extractor's own filtered view
of the document:

  - the prose search used a helper that strips `<tables>`, so a table's contents
    were invisible to a search for "is it in the text?"
  - the table scan required numeric density, so a table whose payload is 253
    compound *identifiers* in one cell scored zero and was skipped
  - the fetch checked one metadata key and never read its sibling

Negative evidence only means something if the instrument could have returned a
positive. This tool deliberately uses no parser: it flattens the raw XML (strip
tags, unescape entities, collapse whitespace) and counts what is visibly there,
then compares against what extraction produced. A large shortfall is a prompt
to investigate, never a conclusion about the patent.

Usage:
    python3 -m patentdb.scripts.eval.coverage_gap US11566007 US8952177
    python3 -m patentdb.scripts.eval.coverage_gap --all --json gaps.json
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

from patentdb.core import config
from patentdb.sources import uspto_assays as A
from patentdb.sources import uspto_xml as U

# Deliberately loose: this is a census of what a human would see, not a parse.
_CID_TOKEN = re.compile(r"\b([A-Z]{0,3}[-–]?\d{1,5}[a-z]?)\b")
_ASSAY_CTX = re.compile(r"IC\s*50|EC\s*50|\bKi\b|\bKd\b|inhibition|potency|activity", re.I)
# Signals that data is present in a shape the row-per-compound reader misses.
_SIGNALS = {
    "bin_key_legend": re.compile(r"is\s+marked\s*[\"“']?[+A-E]|\*\s*key\s*:", re.I),
    "inverted_cid_list": re.compile(r"(?:[A-Z]{1,3}\d{1,4}\s*,\s*){6,}"),
    "spelled_unit": re.compile(r"\b(?:micromolar|nanomolar|millimolar|picomolar)\b", re.I),
    "plus_bins": re.compile(r"\s\+{1,5}\s"),
    "letter_bins": re.compile(r"\bIC\s*50\s*[^.]{0,40}\b[A-E]\b"),
}


def flatten(xml: str) -> str:
    """Raw text as a human would read it. No parser, no filters."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", xml)))


def census(patent_id: str) -> dict:
    try:
        xml = U.fetch_grant_xml(patent_id)
    except U.UsptoUnavailable as e:
        return {"patent": patent_id, "error": str(e)}

    flat = flatten(xml)

    # Compound-id census: tokens that repeat, so stray numbers don't inflate it.
    counts = Counter(m.group(1) for m in _CID_TOKEN.finditer(flat))
    visible = {c for c, n in counts.items() if n >= 2 and not c.replace("-", "").isdigit() or n >= 3}

    records = A.extract_from_patent(xml)
    extracted = {r.cid for r in records}

    signals = {name: len(pat.findall(flat)) for name, pat in _SIGNALS.items()}
    return {
        "patent": patent_id,
        "chars_raw": len(flat),
        "assay_context_mentions": len(_ASSAY_CTX.findall(flat)),
        "cids_visible_raw": len(visible),
        "cids_extracted": len(extracted),
        "records": len(records),
        "ranged_records": sum(1 for r in records if r.range_lo is not None or r.range_hi is not None),
        "coverage": round(len(extracted) / max(len(visible), 1), 3),
        "signals": {k: v for k, v in signals.items() if v},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("patents", nargs="*")
    ap.add_argument("--all", action="store_true", help="every patent already extracted")
    ap.add_argument("--json", help="write full results here")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="flag patents whose extracted/visible ratio is below this")
    args = ap.parse_args()

    pids = args.patents
    if args.all:
        root = config.OUTPUT_DIR / "text_extraction"
        pids = sorted(d.name for d in root.iterdir()
                      if d.is_dir() and not d.name.startswith("_"))
    if not pids:
        ap.error("give patent ids or --all")

    rows = []
    print(f"{'patent':<16}{'raw cids':>9}{'got':>7}{'cov':>7}{'recs':>7}{'ranged':>8}  signals / error")
    for pid in pids:
        r = census(pid)
        rows.append(r)
        if "error" in r:
            print(f"{pid:<16}{'-':>9}{'-':>7}{'-':>7}{'-':>7}{'-':>8}  {r['error'][:44]}")
            continue
        flag = "  <-- INVESTIGATE" if r["coverage"] < args.threshold else ""
        sig = ",".join(r["signals"]) or "-"
        print(f"{pid:<16}{r['cids_visible_raw']:>9}{r['cids_extracted']:>7}"
              f"{r['coverage']:>7.2f}{r['records']:>7}{r['ranged_records']:>8}  {sig[:38]}{flag}")

    low = [r for r in rows if "error" not in r and r["coverage"] < args.threshold]
    if low:
        print(f"\n{len(low)} patent(s) below {args.threshold:.0%} coverage. These are "
              f"'cause unknown', NOT 'unextractable' — check the raw text before concluding:")
        for r in low:
            print(f"   {r['patent']}: {r['cids_extracted']}/{r['cids_visible_raw']} cids"
                  + (f"; raw text shows {', '.join(r['signals'])}" if r["signals"] else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
