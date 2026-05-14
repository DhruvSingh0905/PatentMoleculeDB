# PatentMoleculeDB

Pipeline that turns a US patent number into structured chemistry data — every named compound's IUPAC, canonical SMILES, and InChIKey, paired with every assay value the patent reports for it.

## Input → Output

**Input:** a patent ID, e.g. `US8952177`.

**Output (under `output_v2/text_extraction/{patent_id}/`):**

- `example_index.json` — one record per compound. `cid → {iupac_name, canonical_smiles, inchikey, extraction_method, iupac_source}`.
- `assay_tables.json` — `cid → list of {assay_name, value_numeric, unit, qualifier, n_runs}`.
- `route_audit.json` — per-source breakdown so you can see which extractor produced each compound.

## At a glance

- Reads Google Patents' clean HTML first; falls back to MinerU PDF→Markdown only when GP doesn't carry the patent.
- Six deterministic compound-discovery sources run in parallel and merge by trust rank, then an LLM assay agent (HARVEST) pulls assay tuples in chunks.
- Every IUPAC parse goes through a six-stage cascade: OPSIN raw → rule clean → autocorrect → vision OCR → LLM normalize → LLM direct SMILES. Each stage only fires when the previous fails.
- Patent-agnostic — no per-patent flags, no hand-rules. Caching at every LLM boundary makes reruns near-zero cost.

See [ARCH.md](ARCH.md) for the full diagram and per-stage detail.

## Repo layout

```
patent_extraction_v2/
  core/         # text loaders, models, OPSIN/LLM IUPAC cascade,
                # cost tracking, assay-FSM (HARVEST), validators
  routes/       # process_patent orchestrator, Google Patents
                # extractor, table parsers, bridge logic
  markush/      # Markush enumeration engine (WIP — not in the
                # live orchestrator)
  data/         # vocabulary JSON, prompt templates
ARCH.md         # detailed architecture
CLAUDE.md       # repo conventions for Claude Code
```

The previous `patent_extraction/` codebase (v1) is kept locally for side-by-side benchmarking but is no longer tracked.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."
```

## Run

```bash
# extract a single patent
python3 -c "from patent_extraction_v2.routes.process_patent import process_patent; \
            process_patent('US8952177')"
```

Outputs land under `output_v2/text_extraction/US8952177/`.

## License

Private; not for redistribution.
