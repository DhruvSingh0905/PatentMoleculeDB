# Architecture

Single-pass pipeline: patent ID in, structured chemistry data out. The orchestrator is `patentdb/routes/process_patent.py:process_patent`. Six stages run in order; within each stage, deterministic work runs first and LLM work only on residual gaps.

```
┌─ 1. Text source (RANKED — patentdb/sources/) ───────────────────┐
│ 1. USPTO grant/publication XML — real CALS tables, no OCR       │
│ 2. Google Patents HTML — clean prose, no table structure        │
│ 3. MinerU PDF→Markdown OCR — last resort                        │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─ 2. Compound discovery (parallel sources → trust-rank merge) ───┐
│ GP structured data, density scans, table parsers, Example-N     │
│ pass on GP, Example-N pass on MinerU, MS stubs                  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─ 3. Route classification (INFORMATIONAL TAG ONLY) ──────────────┐
│ text-dominant / markush-dominant / mixed — does not gate 4      │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─ 4. Text branch — runs for EVERY patent ────────────────────────┐
│ HARVEST → validator → Strategy 5 → targeted fill                │
│ (+ additive Markush region tagging; no enumeration)             │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─ 5. Post-processing ────────────────────────────────────────────┐
│ retry impossible fragments → GP↔patent cid bridge → PubChem     │
│ IUPAC backfill                                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─ 5b. Gap-triggered repair (patentdb/repair/) ───────────────────┐
│ tables we largely failed to read → ask for a RULE (not data) →  │
│ validate → persist by layout fingerprint → re-run. ~$0.002 per  │
│ new layout, $0 on repeat, $0 when everything parsed.            │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─ 6. Outputs ────────────────────────────────────────────────────┐
│ example_index.json · assay_tables.json · route_audit.json       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Text source

Three tiers, tried in order by `patentdb/sources/resolve()`. The ranking is not a
preference — it states how much reconstruction each source forces on us.

| # | Source | Properties |
|---|---|---|
| 1 | **USPTO grant/publication XML** | The patent's own OASIS/CALS `<table>` markup. Exact cells, qualifiers and run counts intact, no OCR. Grants from 2002; publications via APPXML |
| 2 | **Google Patents HTML** | OCR-clean flat text, but renders no table structure. Its machine-read chemistry is unreliable — measured 0.2% exact-match against BindingDB |
| 3 | **MinerU PDF→Markdown** | Recovers table structure by inference and pays for it: `<\|ref\|>` tags, `[[bbox]]` pollution, merged rows, `2-y1}` for `2-yl}` |

Tier 1 changed the economics of everything downstream. Assay extraction from
CALS markup is deterministic and needs no model: **99.9% exact-match against
BindingDB on ten patents never seen before**, versus 42% for the LLM-backed
path, at zero API cost.

One thing XML does *not* fix: chemical names wrapped across typeset lines.
USPTO emits the break as separate `<entry>` elements — the same `2-yl` split we
blamed on OCR is in the source document. `uspto_xml.join_wrapped_cells` repairs
it structurally and lets OPSIN adjudicate the ambiguous hyphen.

---

## Stage 2 — Compound discovery

Six deterministic sources run in parallel, then merge into a single `example_index` keyed by patent cid.

| # | Source | What it produces |
|---|---|---|
| 1 | **GP structured data** | Per-compound SMILES + InChIKey from `<meta itemprop="smiles">` tags. RDKit-canonical. Indexed as `GP1, GP2, …` |
| 2 | **IUPAC density scan (HTML)** | Regex sweep + OPSIN on the clean GP description |
| 3 | **IUPAC density scan (markdown)** | Same regex on the MinerU view as a safety net |
| 4 | **HTML table parser** | `Cpd. No. \| Chemical Name` tables from MinerU `<table>` blocks |
| 5 | **GP table parser** | The same data from GP's rendered tables |
| 6 | **GP description `Example N` pass** | Slices the clean description by every `Example N` marker — the IUPAC sits between this marker and the next |
| 7 | **MinerU `## Example N` pass** | Fallback for the same data when GP is missing |
| 8 | **MS-stub merger** | Backfills compounds that only appear in mass-spec sections |

A trust-rank table picks the cleanest representative per `(cid, InChIKey)` collision.

### The per-IUPAC cascade

Every IUPAC candidate runs through this ladder; each stage fires only when the prior one fails.

1. **OPSIN raw** — free, instant
2. **Rule-based cleaner → OPSIN** — strips known OCR markup
3. **Levenshtein autocorrect → OPSIN** — catches novel typos
4. **Claude Vision OCR → OPSIN** — re-OCRs the PDF page for truncated names
5. **LLM IUPAC normalize → OPSIN** — Opus rewrites the name cleanly
6. **LLM direct SMILES** — last-resort structure generation

Strict mode is engaged when the source is OCR-noisy. A clean OPSIN parse from corrupted input is treated as a fail, not a pass, so the cascade gets a chance.

### The GP `Example N` pass — why it matters

GP renders the description as one long flat string with all `Example N` headers preserved. Same regex library we'd use on MinerU markdown — minus the OCR corruption.

```
"… Example 38 (1S*,2R*)-2-{1-(4-Cyanobenzyl)-6-[(5-methylpyridin-2-yl)methoxy]-
   1H-benzimidazol-2-yl}cyclohexanecarboxylic acid was purified by chiral SFC …"
```

Per cid: find the marker, slice to the next marker (cap 800 chars), strip `Synthesis of` / `Preparation of` prose labels, trim at end-markers (`was prepared`, `MS (ESI)`, `1H NMR`, `Step A`), run the cascade. A guard rail prevents overriding an existing GP structured-data SMILES with a different connectivity.

---

## Stage 3 — Route classification

Cheap signals produce a route tag. **The tag is informational — it does not select a branch.** An earlier version gated on it and zeroed out real data (US11566007: 155 examples + an A-prefix assay table → 0), because specific exemplified compounds exist in nearly every patent, Markush-heavy or not. Stage 4 runs for everything.

| Signal | Meaning |
|---|---|
| `n_embedded_smiles` | How many compounds GP publishes structured data for |
| `n_phrase_hits_markush` | Counts of "wherein each occurrence", "independently selected from" |
| `has_substituent_table` | Whether the body has R-group definition tables |
| `n_pre_compounds` | Total compounds Stage 2 produced |

Outputs `text-dominant`, `markush-dominant`, or `mixed`.

---

## Stage 4 — Text branch (all patents)

### HARVEST (assay extraction)

LLM-driven agent reads the patent in chunks and emits `(compound_id, assay_name, value, unit, qualifier, n_runs)` tuples.

```
patent text (GP description + MinerU pages)
        │
        ▼   chunk into 6 000-char windows
        ▼   rank chunks by assay-signal density
        ▼   top-5 chunks → LLM extraction agent
        ▼   raw rows → dedup by (cid, value, unit, qualifier)
```

Cached per chunk fingerprint. Output keyed by sanitized compound-id — one canonical normalizer collapses `"5A"`, `"5a"`, `"Compound 5A"` to `5A`.

### Output validator (the silent fix)

HARVEST can misattribute synthesis-prose numbers (NMR coupling constants, MS m/z) as assay values. The validator drops any `(cid, value)` triple unless it appears in **either** a MinerU `<table>` block **or** the GP description's flat text, with the cid as a standalone token and the value within 200 characters.

Both sources matter: MinerU sometimes corrupts assay tables (`<tr><td></td><td>0.061 …`), but GP description has the same data cleanly as `25 0.061 (2) 3.12 (1) 26 …`. Validating against either rescues correct rows while still catching prose-leak.

### Strategy 5 broad

One LLM call on the whole patent extracts every `Example→IUPAC` pair it can find. Overwrites existing entries only when the source was OCR-noisy markdown (clean GP entries are kept).

### Targeted fill

For every cid HARVEST has an assay for but where `example_index` has no SMILES, chunk the patent around that cid and ask the LLM for that one compound specifically. Catches edge cases the bulk passes miss.

---

## Markush enumeration — held out of the tree

Patents that describe their chemistry as Markush formulas (`R₁ = …; R₂ = …; combine`) need a different model: enumerate the combinatorial product of a scaffold + R-group libraries rather than extract N named examples.

**The engine is not in the package.** `markush/enumerate.py` and `markush/step.py` now live in `_attic/held_out_markush/` (and in git history). What remains under `markush/` is `context.py` + `mapper.py`, which `routes/text_markush.py` uses to *tag* where the claimed genus is defined — additive metadata, no enumeration. Restoring the engine also means restoring five `config.py` constants removed as unused; see `_attic/MANIFEST.md`.

The design, for whenever it comes back:

| Stage | What it does |
|---|---|
| **M1 — Context** | Find the Markush formula image, send to Claude Vision, get back `{scaffold_smiles, R-group positions}`. Parse R-group definition tables (`R₁ is selected from H, methyl, ethyl, …`) symbolically first, LLM fallback. Cross-check formula consistency. Classify difficulty. |
| **M2 — Multi-level cores** | Run `RGroupDecompose` on the validated text-extracted examples. Derive a coarse scaffold (whole framework) and finer scaffolds (sub-rings + attachment points). Score by example coverage. |
| **M3 — Fragment library** | Build a global fragment vocabulary from every example's substituents. Per position, compute a statistical R-group library. Filter fragments by chemical compatibility. |
| **M4 — Combinatorial assembly** | Instantiate each scaffold with all compatible R-group combinations. MaxMin-diverse subset selection when libraries are large. Validate each result (SMILES parses, MW in drug-like range, no clashing groups). Per-scaffold caching. |
| **M5 — Mapping** | For each enumerated structure, find a matching text-extracted example by InChIKey or stereo-flat skeleton. Symbolic alignment first; LLM-assisted on ambiguous cases. |

**Why it's held out.** The engine produces output on a controlled bench, but it's pending better R-group library coverage on real patents, production-grade fragment-compatibility filters, and a precision check — we generate plausible compounds, but are they the set the claim language actually claims? Until that's answered, enumerated structures would pollute a corpus whose whole value is being trustworthy.

---

## Stage 5 — Post-processing

Three "trim & polish" steps. They never create new molecules — they sanitize, link, enrich.

### 1. Retry impossible fragments

For every entry whose SMILES looks suspicious (broken fragments, MW out of range), re-extract the IUPAC via LLM. Budget-capped.

### 2. GP ↔ patent cid bridge

Two stages:

- **Stage A** — merge a GP-positional cid (`GP107`) with a patent cid (`107`) when their InChIKeys match exactly. Keep the patent cid, drop the GP one. HARVEST's assays now flow to the right molecule.
- **Stage B** — positional fallback. Rename `GP107 → 107` only when MS `[M+H]⁺` agrees within ±5 Da, **or** Stage A has already established ≥5 InChIKey-aligned pairs (so we trust the patent's GP ordering). Without this guard the bridge would silently mis-rename.

### 3. PubChem IUPAC backfill

For compounds that came from GP's structured-data tags (SMILES + InChIKey but no IUPAC text), look the name up on PubChem. Records get stamped `iupac_source: pubchem_backfill` so reviewers can see the provenance. Transient network failures are not cached — retried next run.

---

## Stage 6 — Outputs

Per-patent JSON files under `output_v2/text_extraction/{patent_id}/`:

- `example_index.json` — `cid → {iupac_name, canonical_smiles, inchikey, extraction_method, iupac_source, source}`
- `assay_tables.json` — `cid → [{assay_name, value_numeric, unit, qualifier, n_runs}]`
- `route_audit.json` — counts per extraction method, route class chosen, retry stats, cost spent

---

## Cross-cutting properties

- **Patent-agnostic.** No per-patent flags or hand-rules anywhere in the pipeline. Every signal is computed from text.
- **Caching at every LLM boundary.** Re-running the same patent costs near-zero. Chunk fingerprints, IUPAC pattern fingerprints, PubChem InChIKey lookups all cached locally.
- **Provenance.** Every record carries `extraction_method` and `iupac_source` so reviewers can trust or distrust the row.
- **Cost-bounded.** Per-patent LLM cap (default $5) enforced across all stages. The cascade and budget guards stop calling the LLM once the cap is reached.
- **No automatic cache invalidation.** Every cache is cleared by hand — see the cache table in CLAUDE.md. The versioned step-cache DAG this file used to imply was only ever wired to the held-out Markush step, and has been removed.

## What it doesn't yet do

- Live Markush enumeration (engine held out; see above).
- BDB-side reference-data error detection — Jie sheet flags `ref_iupac_wrong` rows, BDB sheets don't yet.
- Image-only compounds where GP doesn't publish a SMILES (figure URLs captured; downstream image-to-SMILES not yet wired in).
