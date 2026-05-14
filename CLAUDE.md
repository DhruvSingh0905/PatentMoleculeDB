# Patent Compound Extraction — Claude Context

## CRITICAL: Two codebases live here right now

- `patent_extraction/` — **v1, the legacy codebase** (~12,000 LOC, 42% compensation logic). Still present for reference + side-by-side benchmarking. **Do not add features here.**
- `patent_extraction_v2/` — **v2, the rebuild in progress**. Nested layout (`core/`, `markush/`, `routes/`). Salvaged ~5,800 LOC of proven-ROI work; new code goes here.
- Outputs: v1 writes to `output/`, v2 writes to `output_v2/` — they run side-by-side without clobbering. See `patent_extraction_v2/PHASE_*_WALKTHROUGH.md` for current rebuild state.

## CRITICAL: Clear stale caches/crops when extraction strategy changes

Whenever you change an upstream layer (OCR tool, bbox-detection logic, table parser, etc.), the cached outputs from the old strategy become misleading — they look correct but reflect the old wrong behavior. Stale caches have caused multiple debugging dead ends:

- Stale image crops (`output/images/*.png`) that DECIMER then "successfully" reads, producing wrong SMILES paired with wrong compound IDs
- Stale `output/results/{pid}/steps/*.json` that re-hydrate to Compounds with old provenance, hiding bugs in the new wiring
- Stale `output/cache/*.json` LM responses that mask new prompt changes

**Rule**: when you change an OCR tool, a bbox heuristic, a parser, or a step's contract, clear the relevant caches BEFORE running. Don't trust an audit signal that came from cached state of a previous strategy.

```bash
# When OCR / bbox / cropping changes:
rm -rf output/images/{patent_id}                    # stale crops
rm -rf output_v2/images/{patent_id}                 # v2 crops too

# When a step's contract or version changes:
rm -rf output/results/{patent_id}/steps             # step cache for that patent
rm -f  output/results/{patent_id}/combined.json     # final output
rm -f  output/results/{patent_id}/audit.json        # audit (will be re-derived)

# When LM prompt changes (forces re-call):
rm -rf output/cache                                  # API response cache

# Nuclear: full reset for a patent
rm -rf output/results/{patent_id} output/images/{patent_id}
rm -rf output_v2/results/{patent_id} output_v2/images/{patent_id}
```

The `step_cache.py` versioned-DAG system handles version bumps automatically (bump `STEP_VERSIONS["foo"]` and downstream caches invalidate). But that doesn't help with image crops or API response caches — those need manual invalidation. **If in doubt, clear and rerun.**

## CRITICAL: Audit pipeline wiring before reporting any result

We have repeatedly built components that look like they're firing but actually aren't (the image pipeline never ran in benchmarks, Route 1b table parser was bypassed by `final_v2/`, the Markush enumeration step generated 0 compounds for US9718825 because its drawn structures are R-group fragments, not full molecules). Misleading numbers waste hours of follow-up.

### Automated check (run on every patent automatically)

`pipeline.py` now writes `output/results/{patent_id}/audit.json` after every patent extraction. Inspect it FIRST when reporting any number:

```bash
python3 -c "import json; a=json.load(open('output/results/US10899738/audit.json')); print('Class:', a['patent_class']); print('Violations:', a['wiring_violations']); print('Markush ROI:', a['markush_roi'])"
```

The audit module (`pipeline_audit.py`) flags:
- Routes that **fired but produced 0 compounds** (almost always a wiring bug)
- Routes that **fired with N compounds but 0 novel** (pure duplication — wasted cost)
- **Class-violation routing** (e.g., Class C ran the image pipeline despite suppression)
- **Markush ROI <0** (spent $$$ for no novel connectivity keys)

`--strict-audit` mode raises on any violation (use in CI):
```bash
python3 -m patent_extraction --strict-audit US10899738
```

### Manual checks still required

The automated audit catches symptoms it knows to look for. For new strategies:

1. **Trace which extraction methods produced compounds**: read `combined.json` and count compounds per `provenance.route` (preferred over the legacy `extraction_method` string).

2. **Confirm benchmark inputs match production**: `run_4patent_bench.py` historically bypasses `pipeline.py`. If you wired a new route into pipeline.py, either change the bench to call `run_patent()` or read the per-patent `audit.json` to confirm it fired in production.

3. **Spot-check `per_example.json`** for a few patents — every Example should have `sources_present` listing the routes that produced its structure. If `sources_present == ["markush_enumeration"]` for an Example that the patent text clearly names, that's a text-route bug.

4. **Watch for "0 from new route" surprises**: the audit catches this automatically and emits a `WIRING:` violation.

5. **Report wiring caveats explicitly**: never frame a wiring bug as "the strategy didn't help." The audit module exists to make this less likely.

## Obsidian Vault Sync

After every meaningful `git commit`, update the Obsidian vault at `~/Main/Projects/Patent Compound Extraction/`:

1. **Decision Log.md** — append a row if the commit represents an architectural decision
2. **Benchmarks.md** — update tables if recall/precision/cost numbers changed
3. **Open Problems.md** — update if blockers shifted or were resolved
4. **Concept pages** — update gotchas if a new bug was found in that area

Do NOT create new files for every commit. Keep pages concise and searchable.

## Project Structure
- `patent_extraction/` — **legacy v1** (do not extend)
- `patent_extraction_v2/` — **rebuild target**; nested layout (`core/`, `markush/`, `routes/`)
- `output/` — v1 extraction results, caches, benchmarks
- `output_v2/` — v2 outputs (separate dir for side-by-side comparison)
- `test_set/` — mini benchmark + Markush test sets
- `venv/` — Python 3.11 with rdkit-pypi (main env)
- `venv_mineru/` — isolated env for MinerU OCR (GBs of model weights)
- `BindingDB_All.tsv` — full BDB dump (~8 GB; `output/bindingdb/our_patents.tsv` is the filtered subset)

## Key Commands
```bash
source venv/bin/activate

# v1 (legacy) — for side-by-side comparison only
python3 run_mini_bench.py        # Fast v1 benchmark (<2 min)
python3 export_csv.py            # Generate Jie's CSV format from v1

# v2 — current rebuild
python3 -m patent_extraction_v2.scripts.eval.assay_table_eval --tools current,mineru
pytest patent_extraction_v2/tests/

# OCR tooling (MinerU)
source venv_mineru/bin/activate
mineru -p {patent.pdf} -o mineru_output -m auto -b pipeline
```

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
