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

## Next fix

**US11547697's four-row header. 2,071 records, diagnosed to the line.**
Dated 2026-08-19. Its TABLE-US-00002 heads four PI3K isoform columns over two
rows, and all four records come out labelled bare `PI3K`:

```
['MTOR', 'PI3K', 'PI3K', 'PI3K', 'PI3K', 'PC3',     'T47D']
['C',    'α',    'β',    'δ',    'γ',    'prolif-', 'prolif-']
['IC50', 'IC50', 'IC50', 'IC50', 'IC50', 'eration', 'eration']
['Structure', '(nM)', '(nM)', '(nM)', '(nM)', '(nM)', '(nM)*']
```

The 10-column tgroup declares **`header_rows = 0`** — all four rows sit at the
top of the BODY and only the first is used, so `α β δ γ` never join. The
values are right and the label is not. US9682141's identical table writes
`PI3K α` in one cell and reads correctly, which is why one patent is right and
the other is not. `check_duplicate_facts` already flags it (862 rows); nothing
acts on the flag.

Then: **`table_names` on US12011444 and US9745328.** 678 compounds — 89% of
the whole `name_in_a_table_cell` miss bucket — sit in these two documents. The
route is sound (75% take rate corpus-wide), so this is a document defect, not
a redesign.

## Deferred, with their sizes

Recorded so nobody re-derives them. Each is real and each is small; none is
worth the build cost today. Date: **2026-08-17**.

| item | size | why deferred |
|---|---|---|
| IUPAC locant to atom | **8 rows corpus-wide** | Procedure is known and verified safe for a monosubstituted ring: position 1 is the atom bonded to the scaffold, and 2/6 and 3/5 are symmetry-equivalent so direction cannot produce a wrong molecule. Breaks on a second substituent, a ring heteroatom or a fused ring. Worth 8 rows. |
| three table layouts | 385 rows | Split headers, repeated column groups, vertical records. |
| dropped IUPAC names | **~1,800**, sized 2026-08-18 | 1,752 (patent, cid) pairs carry a name-like string that never became a structure; range 1,450-2,100. Four patents hold 55% — US12011444 (393), US9745328 (285), US8957068 (183), US10172859 (135), two of which are already the named next fix. **83% of drop events write nothing to the loss log**: `cid_first`'s OPSIN rejections and `table_names`' losing candidates are both silent, so the log undercounts by design. Read `loss_counts` from the manifest, never from `loss_log.jsonl` — the file truncates per process. |
| one structure under several cids | **806 compounds**, 387 structures, sized 2026-08-19 | **98.8% is the DOCUMENT reusing a name, not our defect** — US10245267 states one name under Examples 415, 418 and 419 while printing 484.2, 477.2 and 473.2, and the paragraph disproves the name it sits under. Only 10 are ours, in `cid_first._name_texts`, where a forward span reads past its own marker into the next list item's name. See wiki 52. |
| `cid_clash` is a detector nobody runs | **0 of 38,222 rows** | The column exists in `structures.tsv` and is empty, and not for want of collisions. It is written in ONE place, `iupac_names.py:1300`, inside `extract_names` — and `IDENTITY_ROUTE` defaults to `cid_first`, so `extract_by_cid` runs instead. The artifact's `source` column holds only `cid_first` and `table`, never `description`. A guard on a dead branch is worse than none: the column is present, so the artifact looks checked. |
| one cid, several structures | **122 cids** | The inverse error, all from the table route. US9763922 prints `Example 19` in TABLE-US-00002 and again in TABLE-US-00052 for a different molecule. Both are read correctly; the collision appears only downstream, where something keys on `(patent, cid)` and drops the `table_id` that disambiguates. US9763922 60, US20240360157 23, US9708336 16. |
| structures contradicting their own printed mass | **860** | Newly visible — `mass_gate` now weighs 12,092 structures, up from 3,735. 405 of the deltas exceed 100 Da, which is a different molecule rather than a rounding error. Not yet triaged. See wiki 49. |
| wavy cut bond | 968 rows in 6 patents | **Deliberately last. Do not work on it directly.** No recogniser reads a squiggle across a bond. Its FORMAL meaning in molfile is unspecified stereochemistry (flag 4), not attachment — the patent use is a drawing convention with no format backing. MolScribe does not drop the mark, it INVENTS a group there, returning a terminal isopropyl or tert-butyl, which is why a fragment read is never a substructure of the truth. See the note below for why hand-writing a splicer is the wrong move. |

**The wavy bond is a TEST OF THE LOOP, not a task.** US9718825 states the
convention in its own prose:

> *"the line crossed with the symbol represents the free bond via which the
> group —Z—R3 is bonded to the carbon atom in the 4-position of the
> pyrazolo[3,4-d]pyrimidine ring system."*

The document says where the attachment is. Most patents that use a private
drawing convention are required to explain it, for the same reason: a claim
has to be readable. So the right target is a heal loop that fires **on an
unrecognised image**, reads the paragraph that introduces the table, and writes
the cropping or splicing code itself — the `name_capability` pattern, applied
to pixels instead of text.

Hand-writing a splicer for the squiggle solves 968 rows and teaches the system
nothing. The loop route solves the squiggle AND the next convention nobody has
seen. Do not spend the wavy bond as a one-off.

**The markush MS claim below was WRONG. Corrected 2026-08-17.**

It read: *"only one of those 11 prints an MS, so 97% of it can never be
verified against the document."* That was measured with a mass finder that
required the literal `MS (ESI` — see the gotcha list. The substituent tables
print their m/z in a COLUMN (`MS (m/e)` over `481.0 (M + H)`), which that
finder could not see. Measured over the four real substituent tables:

| | |
|---|---|
| data rows | 653 |
| print their own m/z | **591 (90.5%)** |
| give a substituent as text | 653 (100%) |

So an assembly can be refereed on 90.5% of the population, and the R-groups
need no image recognition. The assemblable population is **644 rows in 2
patents and 4 tables** (US9718825 593, US10626094 51), not 1,152 in 11 — the
other 317 markush rows are `stereo_stripped` / `relative_stereo`, which are
named compounds with loose stereo and a different problem.

Prioritised because Jie says the important structures are here.

Measured over all 137 patents, 44,509 measured compounds, 51% resolved:

| evidence the document gives | total | resolved | missed | take |
|---|---|---|---|---|
| drawn in its own row | 22,897 | 9,285 | **13,612** | 41% |
| appears in prose only | 7,069 | 5,175 | 1,894 | 73% |
| named in a heading | 6,600 | 5,663 | 937 | 86% |
| name in a table cell | 2,981 | 2,222 | 759 | 75% |
| cid IS a name | 420 | 360 | 60 | 86% |
| nothing outside assay tables | 4,542 | 67 | 4,475 | 1% |

Read it this way:

- **Images are 13,612 missed — larger than everything else combined.** A
  drawing does not imply a missing name: 9,285 drawn compounds already
  resolve from text.
- **The reader-side gap is 3,650 (8.2%)** and it is concentrated in about
  eight documents. Document defects, not mechanism failures.
- **"4,475 unfixable" WAS WRONG. Corrected 2026-08-19.** Two of the four named
  patents print the compound number in the SAME TABLE ROW as its drawing, and
  `cid_first._drawing_refs` never sees it: line 973 reads `cells[0]` as the id,
  and US9718790 lays its tables out `Structure | Compound No. | RT | [M+H]`,
  so the first cell tag-strips to `""` and the row is dropped. That function
  returns **0 refs for a 2.6 MB document holding 565 drawn tables**, while
  `build_columns` correctly returns `kind == CID` at index 1 on all 565.
  Two more defects sit beside it: the dict is keyed on the RAW cell while
  every lookup uses `normalize_cid` (`CAP01564` vs `CAP1564`, zero overlap on
  US12065407), and `_resolve` returns at line 1155 before the drawn-marker
  block, so a patent with no name assertion emits no marker at all.
  **3,399 compounds corpus-wide, class (b).** US9303033's 1,270 are a
  different shape — drawings and values in separate `<tables>` paired by
  caption, a positional cross-block join no function performs. US10266548
  gains 0.

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
    markush_gap.py markush_loop.py markush_outcome.py    assembly tier
    plausibility.py                                      post-hoc checks
  core/              config, cost tracking, API client, response cache
  data/              assay_vocabulary.json + layout_rules.json (both TRACKED)
  tests/             269 tests, 13 files
  out/               dump, structures, manifest, loss log — gitignored
  cli.py             front door: `setup` and `run`. Wraps verify, adds nothing.
  verify.py          functional check and the --dump entry point
output_v3/           v3 cache, fetched XML, journals. Shares nothing with v2.
```

## Commands

```bash
source venv/bin/activate
pip install py2opsin openpyxl        # requirements.txt is stale; both needed

python3 -m patentdb3 setup                       # keys + switches -> .env
python3 -m patentdb3 run US8952177 --dump        # front door: read + resolve
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
| `MARKUSH_ASSEMBLY` | 0 | Assemble substituent tables. Needs recognised drawings; blocks by name without them and spends $0. |

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
- **A COLUMN IS NAMED AFTER WHAT IT HOLDS, so the pattern that spots prose
  matches the header.** A table reporting masses has a column headed
  `[M + H]`; one reporting retention times has `Rt` and `(min)`; one naming
  its method has `HPLC`. `_PROSE_CELL` threw all of those header rows away as
  MS traces — 895 corpus-wide, 565 of them US9718790's. What separates them is
  a NUMBER: prose that mentions a mass states the mass, a column name does not.
- **The colspec totals do NOT always match.** This file used to say they did.
  Re-measured 2026-08-19: 592 of 2,384 blocks disagree on column count and
  **17 fail** the width arithmetic, leaving those header cells on the wrong
  grid. Two of the four "same assay disagrees" patents traced back here.
- **Four of five `same_assay_disagrees` flags are real biology.** A reference
  set against an invention set, a detection-limit floor against a ceiling.
  Only US9718825 (666 of 882) is a genuine collapse — four distinct protocols
  share the column header `IC50 [μM]` and the protocol identity lives only in
  each table's caption. Do not treat the flag as a defect count.
- **A document states its shared scaffold in three places.** `<thead>`, the
  `<tbody>` of a leading `cols="1"` `<tgroup>` under an EMPTY thead, and the
  prose paragraph that introduces the table. Reading only the first found 29 of
  673 assemblable rows.
- **A regex for a word must carry `\b`.** `esi` in `_HEADER_MS` matched inside
  "Synth**esi**s", so US10207999's `Chemical Synthesis Example No.` typed as a
  mass column and four tables lost their compound id. Same family as `open`
  matching `open_count`.
- **A search anchored on one vendor's phrasing measures that vendor.**
  `mass_gate.REPORTED` required `MS (ESI`. US8722692 prints `ESI-MS` 500 times
  and `MS (ESI` zero times, so the gate never saw a whole patent. 5 of 137
  patents matched; 31 of 137 print a mass.
- **OPSIN separates a substituent name from a condensed formula, exactly.**
  With `wildcard_radicals` it resolves every name to a `*`-marked fragment and
  refuses `CH3`, `NH2`, `3-CH3`, `2-F`. That is what tells `5-chloro-2,4-
  difluoro-phenyl` from a locant `5` — do not write a heuristic for it.
- **Grep for the module, not the word.** Four files mention "markush" and none
  of them imported `sources/markush.py`; it was dead code with a passing test
  suite, and `CLAUDE.md` quoted a coverage number it had never produced.
- **A filter in front of a parser must accept everything the parser accepts.**
  `bin_legend.looks_like_key` recognised `:`, `=` and five verbs as separators.
  It therefore rejected `+ (greater than 10 microMolar)`, `A ≦ 10 nM`,
  `*** is less than 100 nM`, `"A" represents ... less than 10 nM`, and
  `<1.00 nM=A` — every one of which the parser behind it could read. 9,372
  records, on 15 patents, lost at the filter and never at the parser.
- **A bin key is applied to thousands of rows at once, so a wrong one is
  silent.** Four things go wrong and none show up in the output: the same
  letters mean different ranges in different COLUMNS (US10172859 `B` is 3-7 nM,
  0.5-5 uM or 15-25 uM); the two scales measure different QUANTITIES
  (US10030020 grades nM with `*` and percent with `#`); the legend is a
  `value | symbol` TABLE, so flattening it gives each grade the next grade's
  number (US9221791); or a key defined for the NEXT block sits in the rows of
  this one (US11566007). `test_bin_legend.py` pins all four.
- **`build_columns` is the blast radius.** It reaches 11 files: the assay
  reader, `table_names`, `cid_first`'s markush marker, `mass_gate`, both heal
  tiers and their gates, and the assembly tier. A wrong column decision there
  corrupts every downstream route at once, which is what one 0.0016 score gap
  did to US10253019. Re-measure all seven before changing it. The reach is in
  `graphify-out/graph.json` — run `graphify explain "build_columns"`.
- **A near-100% failure rate is an ATTRIBUTION defect, not a broken
  document.** No real corpus is 94% wrong. When `mass_gate` was widened to
  read prose, US10280164 read 16 of 16 contradicting and US10722495 48 of 51;
  both were the gate reading a molecular formula's subscript (`LCMS calculated
  for C 12 H 18 ClIN 3 OSi (M+H) + m/z=410.0` → **12**) as the mass. It is the
  cheapest tell there is — check the rate per patent before believing any
  corpus total.
- **A heading section is a whole synthesis, so its FIRST mass belongs to Step
  1.** The heading names what the section PRODUCES. Taking the first reported
  every multi-step example as contradicting by exactly what the last step
  still had to add — a constant 243 Da across US9694016 cids 1, 3 and 6. A
  CONSTANT delta across unrelated compounds always means the wrong reference,
  never a wrong structure.
- **A referee must use the route's own cue, not a second one.** `mass_gate`
  reads the compound number with `iupac_names._HEADING_ID` and refuses the
  same headings via `_NOT_A_FINISHED_COMPOUND`. Before that, `Preparation 16`
  donated its mass to `Example 16` — different series, same normalised cid —
  and reported 89 of US20250163061A1's 167 correct structures as wrong.

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

**Do not keep an LLM response cache. Delete `output_v3/cache/` freely.** The
repair loop fixes one small error per gap, and re-running it is quick and
deterministic, so a cached answer buys very little and costs the one thing that
matters: a stale response hides the change you just made. That is not
hypothetical — a gap detector and a prompt fix were both measured as "the loop
cannot do this" across three runs, at $0.00 each, because the cache was
replaying an answer to the previous question. The correct answer appeared on
the first attempt once the cache was cold.

**The only results worth caching are the ones that are expensive to produce
again:** the drawings. `output_v3/recognise/`, `images/`, `gp_images/`,
`decimer/` and `image_results.tsv` hold GPU output and fetched assets — keep
those. `output_v3/cache/` holds nothing but model responses; the three modules
that read it are `repair/synthesize.py`, `repair/name_synthesize.py` and
`repair/name_capability.py`. Never clear `output_v2/`.

## Documents

- `graphify-out/GRAPH_REPORT.md` — knowledge graph. **It is stale.** It was
  built before two renames. Run `graphify update .` first.
- `~/Main/Projects/Patent Compound Extraction/` — the wiki. Start at
  `40 - Where v3 Stands`. It holds the current numbers and the full gotcha
  list. Update it after a meaningful commit. Do not add a file per commit.
- Tracked markdown is `CLAUDE.md`, `README.md`, `ARCH.md` only.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
