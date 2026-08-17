# PatentMoleculeDB

Patent ID in, structured chemistry data out — that is the goal. **What
`patentdb3/` ships today is narrower than that.** This file describes what
runs. See `ARCH.md` for stage-by-stage detail and the design reasons behind
each module.

## What v3 is, right now

The live package is `patentdb3/`. It is a deterministic table reader, a
gap-triggered rule-repair loop, and a set of compound-identity readers. There
is **no orchestrator** — nothing corresponds to v2's `process_patent()` — and
no single merged per-compound record. Concretely:

- **Assay extraction** (`sources/uspto_assays.py` + `repair/`) reads the
  patent's own USPTO grant XML table markup and turns it into assay records
  keyed by the patent's own compound id (`cid`). Real, running, on by default.
- **Compound identity** (`sources/cid_first.py` + `sources/table_names.py`)
  resolves compound names to SMILES and InChIKey and keys each one by the same
  `cid` the assay records use. Real, running, on by default
  (`config.IUPAC_NAMES = 1`).
- **Assay records and structures still live in two separate files.** Both
  carry `cid` on almost every row, so a consumer CAN join them today on
  `(patent_id, cid)`. Nothing in this tree performs that join. See ARCH.md's
  "What was deliberately left out."
- **Image recognition** (`sources/images.py`, `decimer.py`) is a separate,
  manual, patent-by-patent workflow that runs on a Colab GPU. It is not part
  of `verify.py --dump`.

## Input → Output

**Input:** a patent ID, e.g. `US8952177` (needs USPTO grant/publication XML;
coverage starts at grants from 2002 — see ARCH.md). `sources/uspto_xml.py`
fetches and caches it on first use if it is not already under
`output_v3/uspto_xml/`.

**Output**, written by `python3 -m patentdb3.verify <PID> --dump`, all under
`patentdb3/out/`, each one overwritten on every run:

- `reader_dump.tsv` — assay records, long format: one row per
  (compound, assay) measurement. Fields: `cid, assay_name, value_numeric,
  qualifier, unit, n_runs, letter_grade, range_lo, range_hi, table_id,
  column_header`. No chemistry here — no name, no SMILES, no InChIKey.
- `structures.tsv` — one row per resolved (or explicitly unresolved) compound,
  from two identity routes merged with table-cell precedence — see ARCH.md.
  Carries `name`, `smiles`, `inchikey`, `cid`, `source` (`cid_first` or
  `table` by default), a reagent/trace-fragment label, a Markush flag and
  kind, a pointer to the compound's own drawing when one exists but no name
  does, and — when the compound's own row prints a mass — a verdict on
  whether the resolved structure's mass agrees with it.
- `latest.json` — the manifest. Names which patents are in the dump, when it
  was written, whether the repair loop ran (`self_heal`), what it spent, how
  many structures each identity route produced, whether the image-caption
  route ran, and a count of loss-log records by type. Every consumer reads
  paths from here, never a hardcoded string.
- `loss_log.jsonl` — one structured record per place a candidate name,
  candidate structure or table cell was tried and dropped, with enough
  context (position, candidate text, reason) to act on without re-running
  anything.
- `reader_dump.xlsx` (via `to_excel.py`) — the same assay rows as a workbook,
  plus a BindingDB agreement sheet at 1% tolerance when the reference TSV is
  present locally.

The separate image-recognition workflow (`decimer.py`) writes
`output_v3/decimer/<PID>/` (staging + results) and merges into
`output_v3/image_results.tsv`, scored against `output_v3/image_worklist.tsv`.
See ARCH.md's image-recognition section.

## At a glance

- **One text source.** `sources/uspto_xml.py` fetches and parses the patent's
  own USPTO grant/publication XML — OASIS/CALS `<table>` markup, exact cells,
  no OCR. `fetch_grant_xml` obtains a patent's XML with one search call and
  one ~600 KB GET when it is not already cached; needs `USPTO_API_KEY`. v2's
  two fallback tiers (Google Patents HTML, MinerU OCR) were dropped — see
  `sources/__init__.py` for what a pre-2002 or unindexed patent costs (it is
  skipped outright, loudly, no fallback).
- **A rule-repair loop for assay tables, judged by measurement, not
  prediction.** Where the deterministic reader misses a table layout,
  `repair/` asks Haiku for a *rule* — never for data — applies it to the one
  patent it was bought for, and keeps it only if running it actually produced
  correct records. See ARCH.md for the three-condition outcome gate.
- **Two identity routes run by default**, both keyed by the patent's own
  compound id: `sources/cid_first.py` (`config.IDENTITY_ROUTE = "cid_first"`)
  searches outward from every assay-bearing compound id to the name next to
  it; `sources/table_names.py` reads names a patent already spelled out in a
  table cell. A third route, `sources/iupac_names.py` (brute-force span
  search over the description text, plus a heading pass), is fully built and
  tested but is not called unless `IDENTITY_ROUTE` is set away from
  `cid_first` — see ARCH.md.
- **A second, paid repair tier for names exists in the tree
  (`repair/name_*.py`) and is not wired into `verify.py`.** It is tested
  (`tests/test_name_heal.py`) and runnable on its own
  (`repair/name_harness.py`), but `verify.dump()` never imports it. Only the
  free, hand-written repairs in `sources/name_repair.py` (7 confirmed
  corruption patterns, corroboration-gated, $0) run in production today.
- **Two correctness cross-checks, narrow for two different reasons.**
  `sources/mass_gate.py` weighs a resolved structure against the `MS (ESI)
  (M+H)` the patent prints in that compound's own row — it stamps a verdict
  and never drops a row, and it needs `rdkit`. It is not gated off; it is
  just rarely applicable. Measured over 137 patents (38,671 structures),
  only 74 rows (0.2%) have a printed mass to check against, and almost all
  of those sit on one patent — see ARCH.md, do not overstate its reach.
  `sources/image_ocr.py` reads an IUPAC name printed inside a structure's
  own drawing and replaces a drawn-but-unnamed marker when OPSIN accepts
  it; this one IS gated off by default (`config.IMAGE_OCR = 0`), for CPU
  cost, not correctness.
- **Self-heal is on by default** (`config.SELF_HEAL`). Every dump run records
  in its manifest whether healing ran, whether the caption route ran, and
  whether the mass gate was even available, so a reader-only baseline is
  never mistaken for a healed one.
- **No LLM assay burst, no Markush enumeration, no PubChem backfill, no
  Google-Patents structure bridge.** All v2 machinery; none came across. GP
  is used only as an image-pixel source (`config.GP_ENABLED`, off by
  default) — see ARCH.md for why GP's own annotated structures are refused.

## Repo layout

```
patentdb3/          the live package
  sources/          extraction. No LLM calls except where noted.
    uspto_xml.py      fetch + parse USPTO grant/publication XML
    uspto_assays.py   CALS tables -> assay records (deterministic)
    bin_legend.py     potency-bin legends (+/A-E) -> explicit numeric ranges
    cid_first.py      the default identity route — search from assay cids
    table_names.py    names already spelled out in table cells
    iupac_names.py    prose + heading route; built, tested, off by default
    name_repair.py    7 hand-written corruption repairs, $0, corroboration-gated
    anchor.py         proximity anchoring for the prose route
    dewrap.py         typesetting wrap-space repair, shared by two callers
    opsin.py          the ONE OPSIN batch wrapper
    reagents.py       label a structure reagent / trace_fragment / compound
    mass_gate.py      weigh a structure against the patent's own printed mass
    image_ocr.py      read an IUPAC name off a structure's own drawing
    images.py         image worklist, fetch, scoring — feeds decimer.py
    gp_images.py      Google Patents rendered-image URLs (pixels only)
    losses.py         structured loss log
  repair/
    gap.py loop.py rules.py synthesize.py outcome.py     assay-layout tier
    plausibility.py                                      post-hoc sanity checks
    name_gap.py name_loop.py name_rules.py                name tier — NOT
    name_synthesize.py name_outcome.py                    called by verify.py
    name_harness.py name_capability.py                    (see "At a glance")
  core/              config, models, cost tracking, API client + cache
  data/              assay_vocabulary.json + layout_rules.json (both TRACKED)
  tests/             269 tests, 13 files (see CLAUDE.md)
  out/               dump, structures, manifest, loss log — gitignored
  verify.py          the functional check + the --dump entry point
  to_excel.py         dump -> workbook, with a BindingDB agreement sheet
  inspect.py          backtrace one (patent, cid) to its shipped row and its
                       source table
  decimer.py           drive DECIMER on a Colab GPU, one patent at a time
  decimer_vm.py        runs ON the Colab VM; self-contained, no patentdb3 imports
output_v3/           this package's cache, fetched XML, image cache — gitignored
patentdb_v2/         retired and untracked. Kept on disk for reference only.
                      Never run it; never add to it.
ARCH.md              stage-by-stage architecture and design invariants
CLAUDE.md            repo conventions and the current measured state
```

## Setup

```bash
source venv/bin/activate
pip install py2opsin openpyxl     # requirements.txt is stale. Install both by hand.
pip install -r requirements.txt   # the rest: anthropic, regex, rdkit, pytest
export ANTHROPIC_API_KEY="..."    # only needed to buy NEW assay-layout rules
```

`py2opsin` shells out to a bundled OPSIN jar, so a JDK/JRE must be on `PATH`
(`java -version`). `rdkit` (not `rdkit-pypi`) backs `sources/reagents.py` and
`sources/mass_gate.py`; both degrade to a documented safe fallback if it is
missing. `USPTO_API_KEY` (free, data.uspto.gov) is needed only to *fetch* a
patent's XML that is not already cached under `output_v3/uspto_xml/`; a
freshly issued key 403s for several minutes until its usage plan propagates.

`sources/image_ocr.py`'s caption route additionally needs `easyocr`, `numpy`
and `Pillow` — not in `requirements.txt`, because the route is off by default
(`config.IMAGE_OCR = 0`). Install them only if you turn it on.

## Run

```bash
# what the deterministic reader alone produces for one patent, printed
python3 -m patentdb3.verify US8952177

# reader + repair loop + both identity routes, written to patentdb3/out/
python3 -m patentdb3.verify US8952177 --dump

# reader only, $0, the baseline any repair-loop claim must be stated against
python3 -m patentdb3.verify US8952177 --dump --no-heal

# every cached patent
python3 -m patentdb3.verify --all --dump

# dump -> workbook, with a BindingDB agreement sheet
python3 -m patentdb3.to_excel

# backtrace one shipped row to its source table
python3 -m patentdb3.inspect US8952177 43

# image recognition, one patent, on a Colab GPU you already started
python3 -m patentdb3.decimer plan US10730863
python3 -m patentdb3.decimer run  US10730863 -s <session>
python3 -m patentdb3.decimer ingest US10730863

# tests (269, per CLAUDE.md; no API key needed — the assay-layout
# repair tier degrades to "no rule" without one)
python3 -m pytest patentdb3/tests/ -q
```

## License

Private; not for redistribution.
