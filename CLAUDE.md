# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PatentMoleculeDB: patent ID in → structured chemistry out. Every named compound's IUPAC / canonical SMILES / InChIKey, paired with every assay value the patent reports for it. `README.md` has the elevator pitch; `ARCH.md` is the stage-by-stage architecture — read it before touching `patentdb/routes/process_patent.py`, which is the 2,000-line orchestrator everything funnels through.

## Layout

```
patentdb/          the package — this is the whole codebase
  sources/         RANKED input tiers. uspto_xml (1) → google_html (2) → mineru_ocr (3)
    uspto_xml.py   USPTO grant/publication XML: real CALS tables, no OCR
    uspto_assays.py  CALS tables → assay records (deterministic, no LLM)
    bin_legend.py  potency-bin legends (`+`/`A-E`) → explicit numeric ranges
  repair/          gap-triggered LLM repair loop (see below)
    gap.py         locate + fingerprint failing table layouts
    synthesize.py  ask a model for a RULE, never for data
    rules.py       the learned rule library and its validation gate
  core/            text loaders, models, IUPAC cascade, assay FSM/HARVEST, caching, cost
  core/assay_fsm/  the legacy assay pipeline; harvest/ holds the LLM agents
  core/tables/     table adapters + reconciliation (eval-only, see below)
  routes/          orchestrator + per-source extractors
  markush/         context.py + mapper.py — region TAGGING only, no enumeration
  scripts/eval/    benchmarks + audits (import_audit, coverage_gap, reference_bench)
  tests/           107 tests
  data/            learned vocabularies
data/patents/{pid}/   per-patent sources: {pid}.pdf + all_pages/page_*.md
data/BindingDB_All.tsv
output_v2/         all pipeline output and caches (incl. uspto_xml/ per-patent XML)
output/            frozen v1 results, read-only
docs/              reports + local notes (untracked)
_attic/            retired code, untracked — see _attic/MANIFEST.md
```

**USPTO XML is the primary source.** `sources/uspto_xml.py` fetches a patent's own
grant XML (or APPXML for pre-grant publications) via one ~600 KB request — the
search response carries a direct per-document URI, so no bulk downloads. It
carries OASIS/CALS `<table>` markup, which means exact cells with no OCR:
assay extraction from it scores **99.9% exact-match against BindingDB** on
patents never seen before, where the LLM-backed path scored 42%. Needs a free
`USPTO_API_KEY`; a new key 403s for ~6 minutes until its usage plan propagates.
Coverage starts at grants from **2002** — older documents fall through to tiers 2/3.

There is **one** codebase now. v1 (`patent_extraction/`) was retired to `_attic/v1_codebase/` along with its benchmark harnesses. If you find a doc or comment referring to `patent_extraction_v2`, `pipeline.py`, `pipeline_audit.py`, `--strict-audit`, `combined.json`, or `per_example.json`, it is stale — those are all v1.

## Key commands

```bash
source venv/bin/activate        # Python 3.11 + rdkit-pypi

# Run one patent end-to-end (writes output_v2/text_extraction/{pid}/)
python3 -c "from patentdb.routes.process_patent import process_patent; \
            process_patent('US8952177')"

# Tests (20; no API key needed)
python3 -m pytest patentdb/tests/
python3 -m pytest patentdb/tests/test_assay_fsm_vocabulary.py::test_curator_promote

# Evals / audits — all are `python3 -m patentdb.scripts.eval.<name>`
python3 -m patentdb.scripts.eval.assay_table_eval --tools current,mineru
python3 -m patentdb.scripts.eval.assay_completeness_audit
python3 -m patentdb.scripts.eval.fidelity_check

# MinerU OCR (separate env — GBs of model weights)
source venv_mineru/bin/activate
mineru -p {patent.pdf} -o mineru_output -m auto -b pipeline
```

`ANTHROPIC_API_KEY` is read from the environment, then `patentdb/.env`, then the repo-root `.env`. An absent key is tolerated so tests run.

## Architecture facts that aren't obvious from any single file

- **The route classifier is informational only.** `classify_route()` still tags patents `text-dominant` / `markush-dominant` / `mixed`, but `process_patent` runs the text branch for *every* patent regardless. The old gate zeroed out real data (US11566007: 155 examples → 0). Don't "fix" the classifier by re-gating on it.
- **Markush enumeration is not in the pipeline, and the engine is no longer in the tree.** `markush/enumerate.py` + `step.py` were retired to `_attic/held_out_markush/` — held out pending R-group coverage and a precision check, not deleted as junk. What remains in `markush/` is `context.py` + `mapper.py`, used by `routes/text_markush.py` for region *tagging* only.
- **`route_audit.json` is the audit artifact.** Written by `_write_outputs`. The old `core/audit.py` (RouteRecord, wiring violations, `patent_class` gates) was a v1 port that nothing ever imported; it is gone.
- **HTML beats OCR, always.** `load_patent_description(prefer_format="auto")` returns Google Patents HTML first (`output_v2/gpatents_cache/{pid}.json`) and falls back to MinerU markdown (`data/patents/{pid}/all_pages/page_*.md`) only when GP has nothing. MinerU carries `<|ref|>` tags, `[[bbox]]` pollution, and mid-IUPAC line wraps — but it is still needed for the `<table>` structure GP doesn't render.
- **A clean OPSIN parse from noisy input is treated as a failure.** Strict mode on OCR-noisy sources deliberately rejects plausible parses so the six-stage cascade (OPSIN raw → rule clean → Levenshtein → Vision OCR → LLM normalize → LLM direct SMILES) gets its chance.
- **The assay output validator drops rows that can't be corroborated.** HARVEST misreads NMR coupling constants and MS m/z as assay values; `core/assay_fsm/output_validator.py` keeps a `(cid, value)` only if it appears in a MinerU `<table>` block *or* the GP flat description within 200 chars of a standalone cid token. Either source rescues it — dropping one breaks real rows.
- **The GP↔patent cid bridge is guarded for a reason.** Stage A merges `GP107 → 107` on exact InChIKey match; Stage B's positional fallback fires only when MS `[M+H]⁺` agrees within ±5 Da or Stage A already established ≥5 aligned pairs. Loosening that guard silently mis-renames compounds.
- **`core/tables/` is reachable only from `scripts/eval/`.** Seven modules of table adapters and reconciliation that the live orchestrator does not call — it uses `routes/google_tables.py` and `routes/google_patents_tables.py` instead. Know which one you're editing. The same applies to `core/units.py`, `core/bindingdb.py`, `core/assay_reconciler.py`, and the `routes/` image/geometry modules (`structure_crop`, `page_geometry`, `page_regime`, `label_association`, `table_cell_detection`, `text_detection`, `decimer_segmentation_crop`).

## Feature flags and cost caps (`core/config.py`)

Behavior is env-var switchable; check these before concluding a route "didn't fire".

| Env var | Default | Effect |
|---|---|---|
| `REPAIR` | 1 | Gap-triggered rule synthesis (see below). ~$0.002 per NEW layout, $0 after |
| `REPAIR_MAX_CALLS` | 4 | Per-patent ceiling on repair calls |
| `HARVEST_BURST` | 1 | 5-agent burst over full text when the gap detector trips |
| `LLM_BATCH` | 1 | Message Batches API for the burst (~50% cheaper, async) |
| `HARVEST_SKIP` | 1 | Skip burst when the pattern library already covers the patent (≥500 pre-lib rows) |
| `SUBSTITUENT_LLM` | 0 | Opt-in LLM arm for substituent tables; symbolic scan stays $0 by default |
| `EXTRACTION_UNLOAD_MODELS` | 1 | Free OCR model singletons between phases (set 0 on a 64GB+ box) |

Caps are per-patent and separately bucketed so one runaway can't starve the others: `PER_PATENT_LM_CAP` $0.20 (hard alarm $1.00), `PER_PATENT_IMAGE_CAP` $0.50, `PER_PATENT_REALIGN_CAP` $1.50, `process_patent(max_total_cost_usd=)` $5.00, global `COST_CEILING` $200. Default model is `MODEL_SONNET` (`DEFAULT_MODEL`), not Opus — keep it that way for routine work.

**A model missing from `config.PRICING` is billed at Opus rates.** That silently
overstated the repair loop's cost by 15x until Haiku was added. Add every new
model to `PRICING`; `compute_cost` now warns once when it can't price one.

## The repair loop — ask for a rule, never for data

HARVEST costs **~$8.81/patent** (measured: 25.7M input tokens across a 22-patent
corpus) because it re-reads whole documents in 6,000-char chunks and returns
data tuples nothing can verify. `repair/` replaces that with a different
question: given a ~60-120 token sample of ONE failing table, return a *rule*
that then runs deterministically over every patent sharing that layout.

Rules are keyed by **layout fingerprint** — column count, per-column value
shape, normalised header words, with ids and values deliberately excluded — so
one paid question is reused free everywhere that shape appears. Cost is per
distinct layout, not per patent: **~$0.002 for a new layout, $0 thereafter, $0
for a patent whose tables all parse.** Haiku is the synthesis model; Sonnet was
measured no better once the veto below is in place.

Four outcomes: `column_map`, `row_regex`, `not_assay` (a negative rule, so we
never re-examine a mass-spec table), and `escalate` (carrying *which capability
is missing*, for a human queue).

**Nothing a model proposes is trusted.** `rules.validate()` rejects a proposal
unless it beats the coverage the parser already achieves, generalises to rows
the model never saw, and survives an adversarial NMR/MS/MW/RT battery. Two
guards exist because they caught live failures:

- **anti-deletion** — on its first real run the loop proposed three rules that
  each yielded *fewer* rows than the existing parser. Automated-repair
  literature finds the overwhelming majority of patches that pass naive gates
  "work" by deleting behaviour; a repair must add coverage, not stop erroring.
- **`not_assay` veto** — a wrong `not_assay` is permanent and silent. Haiku
  wrongly dismissed 10 of 12 real assay tables; the veto downgrades any
  `not_assay` whose sample contains potency language to an escalation.

### Three tiers, and a gap belongs to exactly one

A failing table has three possible causes, and asking the wrong tier wastes a
paid call on damage we inflicted ourselves:

| Cause | Detector | Cost | Action |
|---|---|---|---|
| the DOCUMENT is unusual | yield gap | 1 call/layout | buy a rule (`loop` + `rules`) |
| the READER lost cells | `parse_fidelity` | **0** | patch `uspto_xml` (`parser_repair`) |
| our VOCABULARY is too narrow | a rule that yields **0** | **0** | patch the code (`capability`) |

**A rule that produces no records is not an answer.** `lib.add(rule)` used to
run *before* `apply_rule`, so a rule yielding zero was indistinguishable from a
layout that needed nothing — US9302989 sat behind an `already_known` for 1,561
rows while its `column_map`, whose column indices were correct, could never
fire because the cells read `0.0125, nd`. The rule is still remembered (it is
insufficient, not wrong); the gap now leaves the tier as a **capability gap**.

`repair/capability.py` buys one patch per capability — three in the whole
corpus — rewriting up to `MAX_TARGETS=3` functions from a fixed candidate list.
Multi-target because single-function patches for these shapes are inert:
US9302989 needed `classify_column` *and* `extract_from_tables` together.

**A patch is declined for exactly ONE reason: it reads fewer compounds.**
Fidelity discrepancies, a failing suite, and a patch that recovers nothing are
recorded as `objections` and applied anyway. This is the same call already made
for model-proposed rules, and it was earned twice more here: an inert-patch
gate declined a patch recovering 1,238 rows because it measured
`extract_from_patent` while the fix was completed by a `bin_key` rule in the
loop. Coverage is the one signal that cannot be argued with; the journal, not a
gate, is what makes applying safe.

Two things no check can see, so a human must:

- **wrong values that raise the count.** The 10x bin-scale class. A coverage
  check will never catch it — spot-check values before quoting a jump.
- **comments.** A whole-function rewrite silently dropped the block explaining
  why cross-width header inheritance is scoped to one `<tables>` id. Behaviour
  intact, reasoning gone, and no test can catch that. The prompt now demands
  verbatim preservation and it held on its first test (193 → 199 lines) —
  **read every applied patch anyway.**

Model choice differs by tier *because the economics invert*. A rule is bought
per layout (hundreds) → Haiku. A capability patch is bought per capability
(three) and every attempt costs a full corpus re-extraction plus the suite, so
tokens are noise beside compute → `MODEL_LADDER` is Sonnet, then Opus on
decline. Measured: Haiku diagnosed the shape correctly and then wrote code
calling a helper that does not exist.

`PATCH_EPOCH` versions the capability cache the way `SYNTH_EPOCH` versions the
rule cache. Widening the tool without bumping it replayed a stale single-target
answer that parsed as an empty patch list and read as the model declining.

    python3 -m patentdb.scripts.eval.capability_repair            # scan, free
    python3 -m patentdb.scripts.eval.capability_repair --repair   # patch + APPLY
    python3 -m patentdb.scripts.eval.parser_health --history      # shared journal
    python3 -m patentdb.scripts.eval.parser_health --revert 4     # undoes the whole group

`config.py` currently holds **zero unused constants**. It accumulated 18 of them before the cleanup, several describing machinery that no longer ran. If you delete a code path, delete its config with it.

## Clear stale caches when extraction strategy changes

Cached output from an old strategy looks correct and reflects wrong behavior. This has caused repeated debugging dead ends: stale crops that DECIMER "successfully" reads into wrong SMILES; stale LM responses that mask prompt changes.

**Rule: when you change an OCR tool, a bbox heuristic, a parser, or a step's contract, clear the relevant cache BEFORE running.** Never trust a signal that came from a previous strategy's cached state.

```bash
PID=US10899738

# LLM/vision response cache — clear when a prompt changes.
# EXPENSIVE to refill: this is paid API output. Don't clear it casually.
rm -rf output_v2/cache

# Per-patent results
rm -rf output_v2/text_extraction/$PID

# Learned caches shared across patents (all live)
rm -f output_v2/text_extraction/_cache/adaptive_extraction_rules.json
rm -f output_v2/text_extraction/_cache/pubchem_iupac.json

# Google Patents HTML scrape (only when the scraper changes)
rm -f output_v2/gpatents_cache/$PID.json

# Image crops — clear when cropping/bbox/OCR changes
rm -rf output_v2/images/$PID
```

There is **no automatic cache invalidation.** The versioned step-cache DAG that `config.STEP_VERSIONS` used to describe was only ever wired to the held-out Markush step — bumping a step version did nothing for any live route, so both it and `core/step_cache.py` are gone. Every cache above is cleared by hand.

## Audit wiring before reporting any number

Components that look like they're firing but aren't have burned hours here repeatedly — the image pipeline absent from benchmarks, the table parser bypassed, a Markush step producing 0 because US9718825's drawn structures are R-group fragments rather than molecules.

Before quoting any recall/precision/cost figure:

0. **Never diagnose from the parsed view.** Four separate times, "the data
   isn't in this patent" turned out to mean "my parser can't see it" — once
   because the prose search used a helper that strips `<tables>`, once because a
   density scan scored a cell of 253 compound ids as zero. Flatten the RAW XML
   (strip tags, unescape, collapse whitespace) and search that first.
   `scripts/eval/coverage_gap.py` does exactly this and needs no parser.
   Reference data lies too: BindingDB cites Example numbers that patents do not
   contain, and gives different values for the same example under two numbers of
   the same disclosure — `reference_bench.patent_owns_cid` filters those.

1. **Read `output_v2/text_extraction/{pid}/route_audit.json`** — counts per extraction method, route class, retry stats, cost spent. If a route you just wired shows 0 compounds, that's a wiring bug, not a strategy result.
2. **Check `example_index.json` provenance** — every record carries `extraction_method` and `iupac_source`. A compound the patent text plainly names should not be attributed to a fallback route.
3. **Confirm the benchmark ran production code.** Scripts under `scripts/eval/` have historically bypassed `process_patent()`; verify the script actually calls it, or read `route_audit.json` to prove the route fired in production.
4. **Report wiring caveats explicitly.** Never frame a wiring bug as "the strategy didn't help."

A fast structural check first — if the module you're editing isn't in the live set, it cannot be affecting production output:

```bash
python3 -m patentdb.scripts.eval.import_audit --config
```

It builds the AST import graph and reports what's reachable from `process_patent` vs. only from tests or eval scripts, plus unused config constants. Exit code is non-zero when orphans exist, so it works as a CI gate. It should stay at **0 orphans, 0 unused constants** — that's how the codebase got 4,487 LOC of dead modules last time.

## Keeping the tracked set honest

The repo previously tracked two scripts that imported a gitignored package, while leaving live code (`core/tables/`) and the assay vocabulary untracked — so a fresh clone was broken in both directions. `.gitignore` is now root-anchored specifically to prevent that (`/data/` must not shadow `patentdb/data/`).

The invariant: **everything the pipeline needs to run is tracked; everything it produces is not.** After changing `.gitignore`, verify with a clone simulation rather than by eye:

```bash
D=$(mktemp -d); git ls-files | while read f; do mkdir -p "$D/$(dirname "$f")"; cp "$f" "$D/$f"; done
cd "$D" && python3 -c "from patentdb.routes.process_patent import process_patent; print('clone OK')"
```

Only `CLAUDE.md`, `README.md`, and `ARCH.md` are tracked markdown — `*.md` is blocked with three exceptions. Walkthroughs, phase notes, and analysis live in `docs/notes/` and stay local by design.

## Data locations

- `data/patents/{pid}/` — PDF + `all_pages/page_*.md` from MinerU. Untracked, pulled per run. Relocating the corpus is a one-line change to `config.DATA_DIR`; every consumer builds `DATA_DIR / patent_id / ...`.
- `output_v2/text_extraction/{pid}/` — `example_index.json`, `assay_tables.json`, `route_audit.json`.
- `patentdb/data/` — `assay_vocabulary.json` + `assay_target_index.json` are **tracked** (the pipeline can't run without them, and rebuilding the index needs the 8 GB BindingDB dump). The `.discoveries.json` files and `ref_ik_cache.json` are runtime accumulation and are not.
- `data/BindingDB_All.tsv` — full BDB dump (~8 GB); `output/bindingdb/our_patents.tsv` is the filtered subset.

## graphify

This project has a knowledge graph at `graphify-out/`.

- ALWAYS read `graphify-out/GRAPH_REPORT.md` before reading source files, running grep/glob, or answering codebase questions.
- IF `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files.
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep.
- After modifying code, run `graphify update .` (AST-only, no API cost). **The graph is stale as of this reorganization** — it was built from commit `f692dc89`, before the package rename, and every path in it says `patent_extraction_v2`. Rebuild before trusting it.

## Obsidian vault sync

After every meaningful `git commit`, update `~/Main/Projects/Patent Compound Extraction/`:

1. **Decision Log.md** — append a row if the commit represents an architectural decision
2. **Benchmarks.md** — update tables if recall/precision/cost numbers changed
3. **Open Problems.md** — update if blockers shifted or were resolved
4. **Concept pages** — update gotchas if a new bug was found in that area

Do NOT create new files per commit. Keep pages concise and searchable.
