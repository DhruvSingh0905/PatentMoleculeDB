"""Minimal functional check for v3 — is the reader alive and what does it emit?

Deliberately NOT an output pipeline. It merges nothing itself. Any name
resolution is `sources/iupac_names.py::extract_names`'s own OPSIN pass,
dumped as-is — this script does not parse a name or join a structure to an
assay row. Its jobs are to prove the move works end to end and to show the
reader's RAW shape, because the assembly is going to be designed around what
this dumps rather than around what anyone assumes it produces.

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

`--dump` WRITES TWO ARTIFACTS, EACH TO EXACTLY ONE PLACE:

    patentdb3/out/reader_dump.tsv    assay records     (DUMP_PATH, below)
    patentdb3/out/structures.tsv     names/structures  (STRUCT_PATH, below)

**Overwrite those files. Do not add a third artifact, and do not add a second
one beside either.** Every later stage — the pivot to Jie's wide sheet, any
scoring, any hand-check — reads DUMP_PATH and STRUCT_PATH from the manifest.
The failure this rule exists to prevent is the one that shaped v2: a new
output file gets added beside the old one, both are half-maintained, and the
variables pointing at them drift until nobody can say which artifact a number
came from. If this script needs to emit something different, change what it
writes here rather than writing somewhere else.

The assay dump is the reader's records as-is — one row per (compound, assay)
measurement, every field the reader carries, nothing derived. It is long
format; Jie's sheet is wide, and that pivot is a later step that reads this.

The structures dump is TWO identity routes' output as-is, in one file, with
three producers between them:

    sources/iupac_names.extract_names        -> source="description"
                                             -> source="heading"
    sources/table_names.extract_table_names  -> source="table"

`extract_names` seeds the patent's prose (`description`) and separately reads
each `<heading>` whole (`heading` — a heading states where its own name starts
and stops, so seeding it is solving a problem it does not have);
`extract_table_names` reads the table cells `uspto_xml.description_text` drops
before prose extraction begins. All three resolve through the same OPSIN gate,
all three attempt `sources/name_repair` on what OPSIN rejects, and neither
ROUTE merges into the other — a structure both routes see is TWO rows carrying
two different provenances (a text offset or a heading's own id; a table id,
row and column). The `source` column is what tells them apart and the
manifest's `structures_sources` counts them. Nothing derives, joins, or
deduplicates across the two routes here; that is the assembly's job, and this
script exists to show it the raw shape.

Both routes are gated on the single existing `config.IUPAC_NAMES` flag
(default OFF; no second switch — the gate lives here and NOT inside either
extractor, so both stay directly callable for testing). The file is written
even when the routes find nothing, so an empty structures dump still means
"the routes ran and found zero," never "the routes did not run" — the
manifest's `iupac_names` / `structures_rows` / `structures_sources` /
`structures_repaired` fields are what actually distinguish "disabled" from
"ran, found none." See `dump()`.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import sys
from collections import Counter
from datetime import datetime

from .core import config
from .core.cost_tracker import cost_tracker
from .repair.loop import repair_patent
from .repair.rules import RuleLibrary
from .sources import image_ocr, losses, mass_gate
from .sources.cid_first import extract_by_cid
from .sources.iupac_names import NamedCompound, extract_names
from .sources.table_names import TableName, extract_table_names
from .sources.uspto_assays import extract_from_patent
from .sources.uspto_xml import UsptoUnavailable, fetch_grant_xml, parse_fidelity, parse_tables

# v2's cached grant XML — an INPUT. v3 writes nothing here; see config.
XML_DIR = config.XML_INPUT_DIR

# The one canonical dump location for each artifact. See the module docstring
# before changing either. Both defined in core.config — the single place
# every path lives, so a new consumer imports from there rather than from
# this script.
DUMP_PATH = config.DUMP
STRUCT_PATH = config.STRUCTURES
MANIFEST_PATH = config.MANIFEST
FIELDS = ("cid", "assay_name", "value_numeric", "qualifier", "unit", "n_runs",
          "letter_grade", "range_lo", "range_hi", "table_id", "column_header")

# Read off the dataclasses themselves rather than hardcoded, so an attribute
# added to either later shows up in the dump on its own — this script must not
# depend on a field that does not exist yet, and must not need editing when one
# is added.
#
# TWO ROUTES, ONE FILE, ONE HEADER. `extract_names` returns `NamedCompound`
# (prose/headings) and `extract_table_names` returns `TableName` (table cells).
# They are different shapes: `TableName` carries `raw_cell` / `dewrap` /
# `table_id` / `row_index` / `column_index` / `column_signal` and no `start` or
# `cid_clash`; `NamedCompound` is the reverse. The header is their UNION,
# `NamedCompound`'s field order first so an existing consumer's column
# positions do not move, and the row writer below already fills a field a
# record does not carry with "" (`getattr(rec, f, "")`).
#
# NOT a second artifact and NOT a second file. `source` ("description" /
# "table") is what tells the two apart, and the manifest tallies it — see
# `dump()`. Splitting them into two TSVs is exactly the shape the module
# docstring above forbids.
STRUCT_FIELDS = tuple(dict.fromkeys(
    [f.name for f in dataclasses.fields(NamedCompound)]
    + [f.name for f in dataclasses.fields(TableName)]))


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
    """Write every record produced for `pids` to DUMP_PATH, plus every
    structure `extract_names` resolves to STRUCT_PATH, both as TSV.

    Overwrites both. DUMP_PATH is one row per measurement, every field the
    reader carries and nothing derived — no unit conversion, no pivot, no name
    resolution. STRUCT_PATH is one row per structure the two identity routes
    resolved out of the patent — `extract_names` over prose and headings,
    `extract_table_names` over table cells — gated on `config.IUPAC_NAMES` and
    otherwise untouched: no merge with DUMP_PATH, no cid join, and no dedup
    BETWEEN the two routes. What a later stage needs that is not here is a
    signal the READER (or one of the two extractors) should carry it, not that
    this script should compute it.

    HEALING RUNS PER PATENT, not as a batch afterwards. `repair_patent` takes
    one document, finds ITS gaps against ITS own baseline, and returns
    `baseline + recovered` — so the loop is scoped to the patent in front of it
    and never sweeps the corpus. `heal=None` follows `config.SELF_HEAL`.

    This function previously called `extract_from_patent` directly and so
    exercised none of the repair tier. A dump produced that way is a reader
    measurement, and reading it as a pipeline measurement is the exact mistake
    this project keeps paying for — which is why the manifest records the
    mode, and now which OTHER routes ran at all.
    """
    heal = config.SELF_HEAL if heal is None else heal
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRUCT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lib = RuleLibrary() if heal else None
    n = n_struct = n_repaired = n_ocr = 0
    done: list[str] = []
    recovered_total = adopted_total = gaps_total = superseded_total = 0
    flags: list = []
    struct_sources: Counter = Counter()
    spend_before = cost_tracker.total_cost
    with DUMP_PATH.open("w") as fh, STRUCT_PATH.open("w") as sfh:
        fh.write("patent_id\t" + "\t".join(FIELDS) + "\n")
        sfh.write("\t".join(STRUCT_FIELDS) + "\n")
        for pid in pids:
            try:
                xml = _xml(pid)
            except (FileNotFoundError, UsptoUnavailable) as e:
                print(f"  {pid}: no XML ({e})")
                # A patent that never gets its XML loses BOTH artifacts this
                # run would otherwise write for it — the assay dump above
                # AND the structures dump below, since neither ever runs for
                # this pid. One record here covers both losses; nothing
                # downstream can distinguish "we never tried" from "we tried
                # and got nothing" without it.
                losses.record("xml_fetch_failed", pid, stage="initial_fetch", error=str(e))
                continue
            try:
                if heal:
                    recs, rep = repair_patent(pid, xml, library=lib)
                    gaps_total += rep.gaps_found
                    adopted_total += rep.adopted
                    recovered_total += rep.rows_recovered
                    superseded_total += rep.superseded
                    if rep.gaps_found or rep.rows_recovered:
                        print(f"  {pid}: {rep.gaps_found} gap(s), "
                              f"{rep.already_known} known, {rep.adopted} adopted, "
                              f"+{rep.rows_recovered:,} rows, "
                              f"{rep.escalated} escalated")
                else:
                    recs = extract_from_patent(xml)
            except (FileNotFoundError, UsptoUnavailable) as e:
                print(f"  {pid}: no XML ({e})")
                losses.record("xml_fetch_failed", pid, stage="extract_or_heal", error=str(e))
                continue
            # EVERY DUMP IS AUDITED, healed or not. These checks existed and
            # ran in exactly one place — inside `repair_patent` — so the
            # artifact everyone actually reads was never checked, and 28 real
            # flags across 20 patents sat undelivered while the same defects
            # were found later by opening the workbook by hand.
            #
            # Printed, not logged. `logger.warning` is invisible unless someone
            # has configured logging, and nobody reads a channel they have to
            # opt into.
            try:
                from .repair.plausibility import audit as _audit
                for f in _audit(pid, xml, recs):
                    flags.append(f)
            except Exception as e:                      # a broken check must
                print(f"  {pid}: audit skipped ({e!r})")  # never cost a run
            for r in recs:
                vals = [str(getattr(r, f, "") if getattr(r, f, None) is not None else "")
                        .replace("\t", " ").replace("\n", " ") for f in FIELDS]
                fh.write(pid + "\t" + "\t".join(vals) + "\n")
                n += 1

            # STRUCTURES — a second, independent route over the same XML, run
            # per patent right here rather than as a separate pass. Its own
            # try/except so a resolution failure costs nothing already written
            # above: the assay dump must not be able to fail because of it.
            if config.IUPAC_NAMES:
                # WHICH ROUTE — see `config.IDENTITY_ROUTE` for the
                # measurement that retired the prose one.
                _route = (extract_by_cid if config.IDENTITY_ROUTE == "cid_first"
                          else extract_names)
                try:
                    names = _route(xml, pid)
                except Exception as e:
                    print(f"  {pid}: {config.IDENTITY_ROUTE} route failed ({e})")
                    # A crash here loses EVERY structure `extract_names` would
                    # have found for this patent, silently — `names = []`
                    # below makes the rest of the loop behave exactly as
                    # before (empty structures dump for this pid), but now
                    # the reason is on record instead of only on stdout,
                    # which a corpus-scale run does not keep.
                    losses.record("extract_names_exception", pid, error=repr(e))
                    names = []
                # THE SECOND IDENTITY ROUTE, over the same XML — the table
                # cells `uspto_xml.description_text` drops wholesale before
                # `extract_names` ever sees them. Its own try/except for the
                # same reason as above: neither identity route may be able to
                # cost the other one its rows, and neither may cost the assay
                # dump anything.
                #
                # NOT DEDUPED AGAINST `names`, deliberately. A table row
                # carries the patent's own row id (`cid`) and its table
                # coordinates; a description row carries a text offset and an
                # anchored id. Where both routes read the same structure, that
                # is two independent pieces of evidence for it, and collapsing
                # them would throw away whichever provenance lost the coin
                # toss. `source` distinguishes them and `structures_sources`
                # in the manifest counts them.
                try:
                    tnames = extract_table_names(xml, pid)
                except Exception as e:
                    print(f"  {pid}: table_names failed ({e})")
                    losses.record("extract_table_names_exception", pid, error=repr(e))
                    tnames = []
                # TABLE FIRST, THEN FILL THE GAPS. A table row states the
                # id and the name in the same row of the same table — the
                # patent's own pairing, with nothing inferred — so where both
                # routes resolve one cid, the table's answer is the one kept.
                #
                # This REPLACES an earlier deliberate non-dedup ("two
                # independent pieces of evidence"). That was right when the
                # second route was prose-proximity and its provenance was
                # worth keeping visible; it is wrong now that the second route
                # is id-anchored and the two would simply duplicate the same
                # compound under the same cid. The union is what raises
                # coverage 46.8% -> 60.2%; the precedence is what stops it
                # double-counting.
                _table_cids = {t.cid for t in tnames if getattr(t, "cid", None)}
                _filled = [nc for nc in names
                           if not (getattr(nc, "cid", None) in _table_cids)]
                _merged = [*tnames, *_filled]
                # THE THIRD IDENTITY ROUTE, and the only one that reads the
                # PICTURE. Where a patent prints the compound's IUPAC name
                # inside its own drawing, OCR plus OPSIN answers a cid that no
                # text route could — and it REPLACES that cid's drawn marker
                # rather than joining it, so one compound never holds two rows.
                # Off unless `config.IMAGE_OCR`; free and silent when the
                # patent's images are not on disk. See `image_ocr.supersede`.
                try:
                    _merged, _n_ocr = image_ocr.supersede(_merged, pid)
                    n_ocr += _n_ocr
                except Exception as e:
                    print(f"  {pid}: image_ocr failed ({e})")
                    losses.record("image_ocr_exception", pid, error=repr(e))
                # THE ONE CHECK THAT DOUBTS THE ANCHOR, not the name. Run here
                # because this is the first point both routes have been
                # resolved AND merged, so every shipped row is weighed exactly
                # once under the precedence that decided it. It stamps two
                # fields and records a loss; it drops nothing — see
                # `sources/mass_gate.py` for why that is deliberate.
                try:
                    mass_gate.check(_merged, xml, pid)
                except Exception as e:
                    print(f"  {pid}: mass gate failed ({e})")
                    losses.record("mass_gate_exception", pid, error=repr(e))
                for nc in _merged:
                    vals = [str(getattr(nc, f, "") if getattr(nc, f, None) is not None else "")
                            .replace("\t", " ").replace("\n", " ") for f in STRUCT_FIELDS]
                    sfh.write("\t".join(vals) + "\n")
                    n_struct += 1
                    struct_sources[getattr(nc, "source", None) or "unknown"] += 1
                    # `name_repair` is the only thing that sets `.repair`, on
                    # either dataclass. Counting it here is what makes "the
                    # repair tier ran and confirmed N structures" readable off
                    # the manifest instead of only off a log line — the same
                    # reason `self_heal` is recorded for the assay tier.
                    if getattr(nc, "repair", ""):
                        n_repaired += 1
            else:
                # THE BIGGEST LOSS POINT IN THIS FILE: `IUPAC_NAMES` defaults
                # OFF (`core/config.py`), so a plain `verify.dump()` call
                # never even ATTEMPTS structures for `pid` — not "found
                # zero," never tried. One record per pid rather than one for
                # the whole run so this log stays queryable the same way
                # every other `patent_id`-keyed record here is.
                losses.record("structures_route_disabled", pid,
                               reason="config.IUPAC_NAMES is False")

            done.append(pid)

    # THE THIRD ARTIFACT. Flushed/closed before the manifest is written (and
    # before its own summary is read back) so both are accurate about a run
    # that has actually finished, not one still buffered in memory. See
    # `sources/losses.py` for the schema and why it lives beside DUMP/
    # STRUCTURES rather than under a new path.
    losses.close()
    loss_counts = losses.summary()

    # The manifest. Written EVERY run, overwritten in place, so it always names
    # the current dumps. Anything downstream reads the paths from here rather
    # than hardcoding one — that is what stops a consumer quietly reading a
    # stale file after a dump moves or is regenerated, which is exactly how
    # v2 ended up with numbers nobody could attribute to a run.
    spent = cost_tracker.total_cost - spend_before
    MANIFEST_PATH.write_text(json.dumps({
        "dump": str(DUMP_PATH),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "patents": done,
        "rows": n,
        "fields": ["patent_id", *FIELDS],
        "source_xml_dir": str(XML_DIR),
        # WHICH ROUTES RAN. Without this, a reader-only dump and a healed one
        # — or a dump with structures and one without — are indistinguishable
        # files, and every comparison between them silently crosses a code
        # boundary.
        "self_heal": heal,
        "gaps_found": gaps_total if heal else None,
        "rules_adopted": adopted_total if heal else None,
        "rows_recovered": recovered_total if heal else None,
        # Unusable records dropped because a repaired twin of the same
        # (cid, assay, table) exists. Recorded so `rows` shrinking against a
        # naive baseline+recovered sum is explained by the artifact itself.
        "superseded": superseded_total if heal else None,
        "usd_spent": round(spent, 4) if heal else 0.0,
        # STRUCTURES — the second artifact. `iupac_names` says whether the
        # route was even ENABLED this run; `structures_rows` says how many it
        # found (an enabled route finding zero and a disabled route both read
        # 0 here — `iupac_names` is the only field that tells them apart);
        # `structures_sources` says FROM WHAT each row came
        # (`NamedCompound.source`, tallied), so a consumer can see what backs
        # the count without opening the file.
        "structures": str(STRUCT_PATH),
        "iupac_names": config.IUPAC_NAMES,
        "structures_rows": n_struct,
        "structures_fields": list(STRUCT_FIELDS),
        "structures_sources": dict(struct_sources),
        # WHICH OPTIONAL ROUTES ACTUALLY RAN. A count of zero is ambiguous by
        # itself — "off", "on but nothing to read", and "on and found none"
        # are three different states and only one of them is a result. These
        # two flags are what tell them apart, and they belong in the manifest
        # rather than a log line because the manifest is the thing a reader
        # still has a week later.
        "image_ocr": config.IMAGE_OCR,
        "mass_gate": mass_gate.available(),
        # Drawn markers replaced by a name read off the picture. Read this
        # against `image_ocr` above before quoting it.
        "image_ocr_superseded": n_ocr,
        # HOW MANY OF THOSE ROWS EXIST ONLY BECAUSE OF A CONFIRMED TEXT
        # REPAIR (`sources/name_repair.py`, either route). 0 here with a
        # non-zero `structures_rows` means the repair tier ran and confirmed
        # nothing — which is its normal answer on a cleanly typeset patent —
        # and is not the same as it never running. Sits beside
        # `rules_adopted` for the same reason that one does.
        "structures_repaired": n_repaired,
        # THE LOSS LOG — see `sources/losses.py`. `loss_log` is the path (so
        # a consumer never has to guess it, same rule as `dump`/`structures`
        # above); `loss_counts` is `loss_type -> row count` for this exact
        # run, read back from the file itself rather than kept as a running
        # total, so it can never drift from what is actually on disk.
        "loss_log": str(losses.LOSS_LOG),
        "loss_counts": loss_counts,
    }, indent=2) + "\n")

    print(f"\nwrote {n:,} rows for {len(done)} patent(s) -> {DUMP_PATH}")
    if heal:
        print(f"self-heal ON: {gaps_total} gap(s), {adopted_total} rule(s) adopted, "
              f"+{recovered_total:,} rows recovered, ${spent:.4f} spent")
    else:
        print("self-heal OFF (SELF_HEAL=0) — reader only, $0")
    if config.IUPAC_NAMES:
        print(f"wrote {n_struct:,} structures for {len(done)} patent(s) -> {STRUCT_PATH}")
        print("  by source: " + ", ".join(
            f"{s}={c:,}" for s, c in sorted(struct_sources.items()))
            + f"   repaired={n_repaired:,}")
    else:
        print("iupac_names OFF (IUPAC_NAMES=0) — no structures dump")
    if loss_counts:
        total_losses = sum(loss_counts.values())
        top = sorted(loss_counts.items(), key=lambda kv: -kv[1])[:5]
        print(f"loss log: {total_losses:,} record(s), {len(loss_counts)} type(s) "
              f"-> {losses.LOSS_LOG}")
        print("  top types: " + ", ".join(f"{t}={c:,}" for t, c in top))
    elif losses.ENABLED:
        print(f"loss log: 0 records -> {losses.LOSS_LOG}")
    else:
        print("loss log OFF (LOG_LOSSES=0)")
    # THE AUDIT, PRINTED WHERE IT CANNOT BE MISSED. Grouped by kind, worst
    # first, so the line that matters is the first one read. Every flag names a
    # state that cannot be true of real data and needs no reference to detect.
    if flags:
        by_kind: dict[str, list] = {}
        for f in flags:
            by_kind.setdefault(f.kind, []).append(f)
        rows = sorted(by_kind.items(),
                      key=lambda kv: -sum(x.rows_at_stake for x in kv[1]))
        print(f"\nAUDIT — {len(flags)} flag(s) over "
              f"{len({f.patent_id for f in flags})} patent(s):")
        for kind, fs in rows:
            at_stake = sum(f.rows_at_stake for f in fs)
            worst = max(fs, key=lambda f: f.rows_at_stake)
            print(f"  {kind:22} {len(fs):3} patent(s), {at_stake:6,} row(s)")
            print(f"    {worst.patent_id}: {worst.detail[:132]}")
        print("  (these are expectations about the OUTPUT — none needs a "
              "reference database)")
    else:
        print("\nAUDIT — no flags.")
    print(f"manifest -> {MANIFEST_PATH}")
    return n


def main(argv: list[str]) -> int:
    if "--dump" in argv:
        named = [a for a in argv[1:] if not a.startswith("-")]
        # `--all --dump` USED TO DUMP ONE PATENT. This branch is checked first
        # and did not look at `--all`, so with no positional id it fell through
        # to the US8952177 default, printed "wrote 328 structures for 1
        # patent(s)", exited 0, and left a corpus-shaped artifact holding one
        # document. Grading anything off that reports a number that is true
        # about something other than what it was quoted as — the exact failure
        # this repo's CLAUDE.md is mostly about.
        if "--all" in argv:
            pids = sorted(p.split("/")[-1][:-4]
                          for p in glob.glob(str(XML_DIR / "*.xml")))
        else:
            pids = named or ["US8952177"]
        # CLI beats the env var, which beats the default. `--no-heal` is how you
        # get the reader-only baseline any recovery claim must be stated against.
        heal = False if "--no-heal" in argv else (True if "--heal" in argv else None)
        for pid in pids:
            one(pid, show=0 if len(pids) > 3 else 6)
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
