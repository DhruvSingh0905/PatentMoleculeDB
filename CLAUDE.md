# CLAUDE.md

Set Claude Code Output style to Simplified Technical English ASD-STE100

Write short sentences. Use the active voice. Give one instruction per sentence.
Use simple words. Do not use a noun cluster of more than three words.

## What this is

The goal is: patent ID in, structured chemistry out.

`patentdb3/` is the live package. It does not do that yet. It has two tracks
that do not join:

- **assay values** — a deterministic table reader, plus a rule-repair loop
- **compound identity** — name extraction, gated by OPSIN

There is no orchestrator. No function turns a patent ID into one merged record.
`ARCH.md` has the stage-by-stage detail. Read it before you change `repair/` or
`sources/`.

`patentdb_v2/` is the previous codebase. It is kept for reference only. Its
tests were deleted. Do not add to it. If a document tells you to run
`process_patent`, `scripts/eval/*`, or `route_audit.json`, that document
describes v2.

## Current state

Command: `verify --dump --no-heal` over `sample30`. Graded from
`patentdb3/out/latest.json`. Date: **2026-08-10**.

| population: 8,264 distinct assay compounds | n | % |
|---|---|---|
| resolved to a structure | 4,877 | **59.0** |
| marked drawn (`<chemistry>` in the compound's own row) | 2,897 | **35.1** |
| neither | 490 | **5.9** |

The text track can reach 64.9% at most. Text is near its limit.

**The image track can reach 93.7%.** 2,866 of the 2,897 drawn compounds
(98.9%) hold a whole-compound image in their own row. This is a full census,
2026-08-12. See wiki 45. Use `cid_first._markush_cids` to test a row. Do not
invent a second rule — an ad-hoc one scored 79.2% against the truth of 98.9%.

Quote no coverage number without its date and its population. Re-run the dump
if the tree changed.

## Layout

```
patentdb3/
  sources/           extraction. No LLM calls.
    uspto_xml.py       fetch and parse USPTO XML (CALS tables)
    uspto_assays.py    tables -> assay records
    bin_legend.py      potency bins (+/A-E) -> numeric ranges
    cid_first.py       THE identity route. Search from assay cids to names.
    table_names.py     names in table cells. Cross-row rejoin lives here.
    iupac_names.py     heading names. Holds NamedCompound.
    name_repair.py     7 hand-written corruption repairs. $0.
    anchor.py          document text, and name -> cid anchoring
    dewrap.py          typesetting wrap spaces
    opsin.py           the ONE OPSIN wrapper. Do not write a second.
    reagents.py        label a structure as reagent or compound
    gp_images.py       Google Patents image URLs. Off by default.
    losses.py          structured loss log
  repair/
    gap.py loop.py rules.py synthesize.py outcome.py     layout tier
    name_gap.py name_loop.py name_rules.py               name tier
    name_synthesize.py name_outcome.py name_harness.py
    name_capability.py                                   model-authored code
    plausibility.py                                      post-hoc checks
  core/              config, cost tracking, API client, response cache
  data/              assay_vocabulary.json + layout_rules.json (both TRACKED)
  tests/             269 tests, 13 files
  out/               dump, structures, manifest, loss log — gitignored
  verify.py          functional check and the --dump entry point
output_v3/           v3 cache, fetched XML, journals. Shares nothing with v2.
```

## Commands

```bash
source venv/bin/activate
pip install py2opsin openpyxl        # requirements.txt is stale; both needed

python3 -m patentdb3.verify US8952177            # one patent, printed
python3 -m patentdb3.verify US8952177 --dump     # + repair loop, writes out/
python3 -m patentdb3.verify US8952177 --dump --no-heal   # reader only, $0
python3 -m patentdb3.verify --all --dump         # every cached patent
python3 -m patentdb3.to_excel                    # dump -> workbook + BDB sheet
python3 -m pytest patentdb3/tests/ -q            # 269 tests, ~2.5 min
```

Tests need no API key. Rule synthesis returns `None` without one.

`ANTHROPIC_API_KEY` comes from the environment, then `.env`. `USPTO_API_KEY`
is needed only to fetch a patent that is not in `output_v3/uspto_xml/`.

## Flags and caps (`core/config.py`)

| Variable | Default | Effect |
|---|---|---|
| `SELF_HEAL` | 1 | Run the repair loop after the reader. |
| `IUPAC_NAMES` | 1 | Gate the identity track. Off gives `[]`. |
| `IDENTITY_ROUTE` | `cid_first` | Which identity route runs. |
| `FINISHED_ONLY` | 1 | Refuse intermediates and steps. |
| `GP_ENABLED` | 0 | Fetch Google Patents image URLs. |

| Constant | Value | Meaning |
|---|---|---|
| `SYNTH_MODEL` | Haiku | Regex rule synthesis. |
| `CAPABILITY_MODEL` | Sonnet | Code authoring. Measured; see wiki 40. |
| `PER_PATENT_LM_CAP` | $0.20 | Soft cap per patent. |
| `COST_CEILING` | $200 | Global ceiling. |
| `loop.MAX_ATTEMPTS` | 3 | Paid attempts per layout. |
| `name_rules.ESCALATE_AFTER` | 3 | Failures per class before write-off. |

**Add every new model to `config.PRICING`.** A missing model bills at the Opus
row. This made Sonnet look 5x its true cost and nearly reversed a decision.

## Gotchas

Each item below cost real time. Read this list before you measure anything.

- **Pool evidence across every patent in a class.** One pattern scored
  `damages 0` against one patent's 40 correct names. The same pattern scored
  `damages 2308` against the real 2,265 names.
- **`Cell.text` is tag-stripped.** `"<chemistry" in cell.text` is always
  False. Use a raw-XML regex. See `cid_first._drawing_refs`.
- **The name library's JSON key is `patterns`, not `rules`.** Read the wrong
  key and you invent a persistence bug that does not exist.
- **Grade against the manifest's own patent list.** A manifest from a
  different sample gives a confident, fabricated zero.
- **Make static gates AST-based.** A substring scan for `open` rejects
  `open_count`. That is the vocabulary of bracket balancing.
- **A drawn marker is not always the compound.** In a substituent table the
  row picture is one R-group. It is shared across rows.
- **`conftest.py` must redirect the rule libraries and the journals.**
  `dump(heal=True)` saves `RuleLibrary()` and will rewrite the tracked
  172-rule `layout_rules.json`.
- **Measure at the stage where the code runs.** A repair in `name_repair`
  acts inside the extractor. It never reaches the heal loop.

## Design invariants

Change these only with a measurement.

- **Assertion triggers the heal loop, not rejection.** 82,398 OPSIN
  rejections are 67% hyphenated English. An assertion is a `<heading>` or a
  `cid | Name` table row.
- **Give the gap the residue, not the raw cell.** Otherwise the model repairs
  defects that dewrap already fixed.
- **Fingerprint the defect, not the name.** Chemistry dominates a name's
  shape. The preimage is OPSIN's own error: `kind|token class|position`.
- **A rule never expires. An escalation expires on `NAME_SYNTH_EPOCH`.** A
  rule is evidence. An escalation is a record of one day's capability.
- **Use one hard gate (safety). Make the rest tools.** Adopt the best
  measured attempt, not the last submitted one.
- **A markush name carries no InChIKey.** It denotes a set of stereoisomers.
- **Corroboration fails for table-cell defects.** The cell is the only place
  the compound is named. Use the collateral measurement instead.

## CORE TENET: find the root cause, then decide who owns it

Do not fix a symptom. Dig until you can name the line that is wrong. Then
decide which of two things you are looking at. There are only two.

**It is STRUCTURAL if, and only if:**

- **(a) we feed the loop bad information.** The heal loop can only reason
  about what the reader hands it. A truncated column name, a dropped header
  row, a duplicated record — the loop cannot see past any of these.
- **(b) our parsing logic has a logical error.** The code does not do what its
  own docstring says, or it asserts something the data contradicts.

**Everything else belongs to the heal loop.** A patent with an odd legend, a
new corruption shape, a layout nobody has seen — that is what the loop is for.
Do not hand-write a rule for it.

**Bolting on a fix is not our job.** A new verdict, a widened threshold, a
special case for one patent — each hides the defect and makes the next one
harder to find.

Worked examples from one session, all four found by digging rather than
patching:

| symptom | wrong fix | root cause | owner |
|---|---|---|---|
| 219 compounds "disagree" with BindingDB | add a GREY verdict | the patent reports letter grades; the loop was never run | neither — run the loop |
| half of every graded table unresolved | raise `min_yield` above 0.5 | `repair_patent` returned baseline + repaired with no dedup, so every improved record shipped twice and pinned the yield at exactly 0.500 | (a) bad information |
| `IC50 DNA-` will not bind a bin scale | special-case the name | `_is_namelike` rejected a header row because `hERG]` closes a bracket opened one row above | (b) logical error |
| 43 data rows read as header | restore the fixed cap of 5 | `_opens_with_id` used `fullmatch`, so `1.` was not an id; and `_is_namelike` never asked whether the cells were VALUES | (b) logical error |

Two lessons the table does not show:

**A threshold that hides a defect is the worst fix.** `min_yield = 0.5` and the
5-row header cap both looked like tuning knobs. Both were load-bearing only
because they limited the damage from a misclassification underneath.

**Fix the cause, then re-measure everything.** Removing the header cap exposed
two latent misclassifications that the cap had been masking for the whole life
of the code. That is the fix working, not the fix failing.

## Discipline

**Backtrace every result.** A number is not a result until you name the
function that emitted it and the artifact you read it from.

1. Name the producer. An eval script is not the pipeline.
2. Check the artifact. Read `out/latest.json`. It names the dump, the
   patents, the time, and whether healing ran.
3. State the population and the denominator.
4. Ask if the metric can fail. A score that cannot get worse measures
   nothing.

**Never form an independent hypothesis.** Read the data. Trace the code.
Report what both say. If the evidence does not reach a conclusion, the
conclusion is "not determined". That is a complete answer.

A subagent's claim is not evidence. Verify it against the artifact first.

## Data locations

- `output_v3/uspto_xml/{pid}.xml` — the one text source. Cached on first use.
- `patentdb3/out/` — dump, structures, manifest, loss log. Overwritten each run.
- `patentdb3/data/` — both files are TRACKED. `layout_rules.json` holds 172
  bought rules. Do not delete it to start fresh.
- `output_v3/rule_adoption_journal.jsonl` — audit trail. There is no undo.
  To revert a rule, delete its entry by fingerprint.
- `data/BindingDB_All.tsv` — BDB reference. Never a source of corrections.

Clear `output_v3/cache/` when you change a prompt. A stale response hides the
change. Never clear `output_v2/`.

## Documents

- `graphify-out/GRAPH_REPORT.md` — knowledge graph. **It is stale.** It was
  built before two renames. Run `graphify update .` first.
- `~/Main/Projects/Patent Compound Extraction/` — the wiki. Start at
  `40 - Where v3 Stands`. It holds the current numbers and the full gotcha
  list. Update it after a meaningful commit. Do not add a file per commit.
- Tracked markdown is `CLAUDE.md`, `README.md`, `ARCH.md` only.
