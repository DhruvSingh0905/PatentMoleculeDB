# Architecture

This describes `patentdb3/` as it runs today. Read `CLAUDE.md` first — it
holds the current coverage numbers and the next fix. This file holds the
stage-by-stage mechanism: what each module does, and why it does it that way.

**There is still no orchestrator.** `verify.py --dump` writes two files: an
assay dump (`reader_dump.tsv`) and a structures dump (`structures.tsv`). It
does not merge them into one row per compound. Both files now carry a `cid`
column for most rows, so a join is possible — but nothing in this package
performs it. That join is the single biggest gap between what v3 ships and
the product goal.

```
grant XML
   │
   ├─▶ TRACK A — assay values (uspto_assays.py + repair/) ──▶ reader_dump.tsv
   │
   ├─▶ TRACK B — compound identity, three text routes
   │      (cid_first.py, table_names.py, iupac_names.py) ──┐
   │                                                        │
   └─▶ TRACK C — compound identity from the drawing         ├─▶ structures.tsv
          (image_ocr.py, mass_gate.py; DECIMER offline) ────┘

No step joins the two dumps. Both carry `cid`; nothing reads both.
```

---

## Track A — assay extraction

### A1. Text source — one tier, not three

`sources/uspto_xml.py` reads USPTO's own grant/publication XML — OASIS/CALS
`<table>` markup. Cells are exact. Qualifiers survive. There is no OCR
anywhere in this tier.

v2 ranked three tiers: this XML, then Google Patents HTML, then MinerU PDF
OCR. v3 keeps only the first. Google Patents HTML has clean prose but no
table structure, and its machine-read chemistry measured 0.2% exact-match
against BindingDB in v2 — for a package whose whole output is table-derived,
that source contributed nothing. MinerU recovered table structure by OCR
inference and paid for it in `<|ref|>` tags, bounding-box pollution, merged
rows, and character errors.

**The cost of the narrowing, stated plainly.** Grant XML begins at
`FIRST_XML_GRANT_YEAR = 2002`. A patent granted earlier, or simply absent
from the USPTO index, has no fallback: `fetch_grant_xml` raises
`UsptoUnavailable`, `verify.py` prints the skip, and the patent contributes
nothing. Nothing degrades silently — but nothing is produced either. Watch
for this when drawing a corpus from BindingDB, which cites plenty of
pre-2002 patents.

`fetch_grant_xml` can **obtain** a patent's XML, not only parse one already
on disk. One USPTO search call locates the document; one further GET (about
600 KB) retrieves it. This needs `USPTO_API_KEY` (free, from
data.uspto.gov). A freshly issued key returns 403 for several minutes until
its usage plan propagates — that is normal, not a bad key. The API's own
rate limit is 60 requests/minute (120 off-peak).

USPTO's typesetting still wraps a chemical name across two `<entry>`
elements at the byte level (`...benzimidazol-2-</entry><entry>yl}...`). The
parser rejoins this deterministically, because the boundary is an explicit
element edge here, not something to infer from OCR noise the way v2 had to.

### A2. Table parsing and the deterministic reader

`uspto_xml.py` turns raw CALS markup into `Table` objects: multi-row headers
merged column-wise, `namest`/`nameend` spans resolved, and continuation
tables (no header of their own) matched to their parent by column-width
alignment.

`uspto_assays.py`'s founding observation: **most numbers in a patent table
are not assay values.** A naive "grab every number next to a compound id"
sweep over one patent produced 6,063 pairs, almost all of them molecular
weights, LC-MS m/z, and retention times. So the reader classifies every
column before reading any of it:

```
merge multi-row headers → classify each column → read only assay columns
```

Column kinds: `cid`, `assay`, `nruns`, `nmr`, `ms`, `mw`, `rt`, `structure`,
`substituent`, `unknown`. A column the classifier cannot confidently place
is skipped, not guessed. A missing assay value is recoverable later; a
molecular weight recorded as an IC50 is a lie nothing downstream can catch.

`bin_legend.py` turns a potency-bin legend (`+`/`++`/`+++`, or `A`-`E`) into
an explicit numeric range. This is resolved per table, never shared
globally — the same symbol means a different range in different patents.
US11566007 uses `++++` for "IC50 ≥ 1 μM"; US11292791 uses the same symbol for
"0.001-0.01 μM". A global mapping would silently corrupt one or the other.

### A3. The repair loop — ask for a rule, apply it, measure what happened

`repair/` is the deterministic reader's gap-triggered companion. It never
re-reads a whole document for data; it asks a model for a *rule*, once per
distinct table shape, and applies that rule everywhere the shape recurs.

```
parse → find gaps → already have a rule for this layout? apply + measure
                  → no rule? propose (Haiku) → ground (can it even run?) →
                    apply → measure what came out → keep, or hand the
                    measurement back to the model and ask again (up to 3x)
```

**Gap detection (`repair/gap.py`).** A gap is a table with measurement-
shaped cells the reader could not turn into records. It carries a **layout
fingerprint** — column count, per-column value shape (`num`/`bin`/`cid`/
`list`/`text`), normalised header words, hashed to 16 hex characters —
deliberately excluding ids and values, so one paid question is reused free
on every future patent whose tables share that shape. The un-hashed
preimage (`layout_signature`, e.g. `"8|cid,num,num|compound+no,ic+nm,herg"`)
is kept alongside it, because it is the one form two layouts can be
*compared* on — see `RuleLibrary.digest` below.

`find_gaps` runs several independent detectors, because a table can fail in
more than one way at once and a table that passes one check can still fail
another:

| Detector | Catches |
|---|---|
| yield (cells in vs. usable records out) | the ordinary under-read table |
| value-level unread cells | columns identified correctly, cells that will not parse |
| guessed-unit disagreement | a unit inferred for one column that contradicts a sibling column's stated unit |
| dead assay columns | a populated assay column that yields nothing while a sibling column does |
| coherent unread ids | ≥3 unreadable ids collapsing into ≤2 shapes — a second id convention, not noise |

Two further checks run in `repair/loop.py`, before any rule is bought, and
they never reach the model — because a rule written against a corrupted
view would just encode the corruption:

| Check | Meaning | Cost |
|---|---|---|
| `parse_fidelity` | the reader's own parse does not reconcile with the source XML | $0, escalated as `PARSER DEFECT` |
| `assembly_fidelity` | real data rows were filed as header | $0, escalated as `ASSEMBLY DEFECT` |
| `yield_contradictions` | an identically-fingerprinted table elsewhere in the same document yields records; this one yields none | $0, escalated as `INCONSISTENT HANDLING` |

Measured over a 63-patent cache: 306 distinct fingerprints, 105 seen more
than once, and exactly one contradiction — this is a precise signal, not a
noisy one.

A gap the library already holds a rule for, that the rule produces nothing
on, is recorded in `RepairReport.capability_gaps` — not auto-patched. v2's
code-patch tier (`repair/capability.py`, which rewrote parser functions) did
not come across to v3; this is a reporting-only queue with no consumer.

**Rule kinds (`repair/rules.py`):** `column_map` (which column is the id,
which are assays, and their units — covers most failures, nothing to
over-match), `row_regex` (a pattern per row when cell structure is lost),
`value_pattern` (how to read one cell, matched only against cells the
reader already failed on — this is what makes the loop self-healing for a
parser bug too narrow, not only a layout gap), `bin_key` (a legend's grade
symbols as ranges, transcribed from the patent's own text; may carry
several column-scoped `scales` when one legend defines more than one
scale), `not_assay` (a negative rule — a mass-spec table is marked never-
assay so it is never re-examined), `escalate` (nothing worked; carries what
is missing for a human queue).

**`ground()` — the one hard block, not a judgement call.** Can the regex
even compile; is a column index in range; can a `value_pattern` ever
capture a number (a named group `num`); does every unit and name the rule
invents actually appear in *this document's* text (case-sensitive on the
molar `M`, so `254 nm` wavelengths cannot certify a proposed `nM`). This is
the one check that catches a fabricated unit — a fabricated unit produces
records that are otherwise perfectly usable, non-contradicting, and
correctly columned.

**`outcome.py` — the one judgement gate, and it is a measurement, not a
prediction.** v2's `rules.validate()` scored a proposal *before* it ran: a
coverage floor, an anti-deletion guard, an adversarial NMR/MS/MW/RT probe
battery, an id-shape test. Measured on 2026-08-07 by replaying the 88 rules
v2 had adopted *over those gates' objections* against the deterministic
reader:

| objection | rules | rows scored | corroborated | contradicted |
|---|---|---|---|---|
| anti_deletion | 32 | 28,281 | 26,539 (93.8%) | 0 |
| coverage_floor | 6 | 135 | 67 (49.6%) | 0 |
| id_style | 5 | 90 | 16 (17.8%) | 0 |

Zero contradictions across 28,281 rows: every one of those 88 vetoes would
have deleted correct data. The anti-deletion guard specifically compared a
held-out row count against a whole-table baseline count, so the two were
almost never equal regardless of what the rule did.

So a rule is applied to the *one* patent it was bought for — never a corpus
sweep — and kept only if all three hold:

1. **It produced records.** Not usable records — records. This was
   `usable > 0` for a while; that dropped 301 correct rows across 9 blocks
   that were missing only their unit (`% Aβ Reduction` whose unit is the
   `%` in its own name). Usability is still counted and journalled, just
   not gated on.
2. **It contradicts nothing the reader already reads.** Where the two
   disagree about the same cell, the rule is overwriting a correct answer.
3. **It does not draw a value from a column the reader positively
   identified as MW, MS, NMR, RT, or structure.** An unclassified
   (`unknown`/`substituent`) column is deliberately exempt — that is the
   ambiguous, legitimately-recoverable case.

**What this still cannot see: a fabricated but self-consistent unit.** That
class of defect produces usable, non-contradicting, correctly-columned
records — invisible to all three conditions above, which is why `ground()`'s
document-grounding check is a separate, non-negotiable gate rather than
folded into the measured one.

**Retry is feedback, not a verdict.** A negative outcome is handed back to
the model exactly as measured, and the model answers again, up to
`repair.loop.MAX_ATTEMPTS = 3` per layout. Past that, nothing survives: the
layout is recorded as an `escalate` rule so the next run does not re-buy
the identical failure. An escalation is not evidence the way an adopted
rule is — `RuleLibrary.get()` expires an escalation whenever `SYNTH_EPOCH`
(currently `"v20-measured-outcome"`, in `repair/rules.py`) moves, so
improving the synthesis prompt automatically re-asks every layout parked
there. A kept rule never expires. `not_assay` also never expires — it is a
positive claim about the table's content, not a capability gap.

**The library teaches itself.** `RuleLibrary.digest(signature)` shows a new
gap the nearest 2-3 layouts by structural similarity — column count is a
hard filter, then 0.6 × per-column shape overlap + 0.4 × header-word
overlap — filtered to rules with `rows_yielded > 0` in real use. This is
safe because it decides nothing: a model that copies a neighbour that does
not fit produces a rule that fails `outcome.measure`, which costs one retry
and ships nothing.

**A patent-level invariant per-table detectors cannot see.** Every detector
above scores a *parsed* table, so a defect severe enough to shrink a
table's parsed view also shrinks the evidence needed to notice it.
`repair_patent`'s last check ignores parsing entirely: it counts
measurement-shaped cells in the raw CALS tgroups, before assembly, before
column classification. If a patent holds at least `SILENT_PATENT_MIN_CELLS`
(10) such cells and zero usable measurements came out of the whole
document, that is escalated as `PATENT YIELDED NOTHING` regardless of what
every other check concluded.

**`repair/plausibility.py` — does the result make sense, with no reference
database.** Runs last, over baseline plus everything the loop recovered,
and reports (never gates) five independent checks: `n_runs_dropped` (a cell
stated a replicate count the reader did not keep), `potency_out_of_range`
(a concentration outside roughly 1 pM-1 mM — two orders of magnitude wider
than the corpus's own observed 99.8% band, so this catches a 1000x unit
error, not real pharmacology), `unconvertible_unit` (a potency in a unit
nothing downstream can place on a scale, so it silently passes the range
check by never being examined), `unnamed_assay` (a usable measurement with
no assay name), and `same_assay_disagrees` (one assay name, two blocks,
medians two orders of magnitude apart — the only one of the five that can
catch a *uniformly* wrong column, since a single mis-scaled value is
plausible in isolation and only wrong next to its own assay's other values
elsewhere in the patent).

---

## Track B — compound identity from text

Three routes read the patent's own text for a compound's name. All three
resolve through OPSIN — the java IUPAC-name parser wrapped once, in
`sources/opsin.py` — and all three write `patentdb3.out.structures.tsv`
via `verify.dump()`. None of the three merges into another; where two
routes resolve the same compound id, `verify.dump()` applies a fixed
precedence (below), not a union.

**`opsin.py` pairs OPSIN's output to its input by list position, and
refuses the whole batch rather than guess when the lengths disagree.**
Three modules once carried their own copy of this pairing logic, all with
the identical bug: when OPSIN returns one extra result (observed once, a
4,973-output response to a 4,972-name batch on US10730863), an
inserted-not-appended extra silently shifts every later name onto the next
name's structure — a valid name, a valid SMILES and a valid InChIKey that
are not each other's, and nothing downstream can detect it. `opsin.batch()`
now refuses the entire batch on any length mismatch, trading a visible
coverage loss for an invisible correctness bug.

### B1. `cid_first.py` — the default route, searching from the id outward

`config.IDENTITY_ROUTE` defaults to `"cid_first"`. This route inverts the
older approach: instead of finding every OPSIN-parseable name in the
document and then guessing which compound number it belongs to, it starts
from the compound ids `uspto_assays.extract_from_patent` already extracted
— the ones carrying an assay value — and searches *outward*, from each id
to the text next to it.

The id is used exactly as the assay reader produced it; it is not
re-canonicalised, because canonicalising mangles some real ids (`EM09912`
→ `EM9912`) into tokens that occur nowhere in the document.

**What counts as an occurrence, and why the obvious rule fails.** A single
regex alternation over every id, whole-token, was built first and measured:
on one patent it found 32,169 occurrences of 187 ids in 769,303 characters
— one every 24 characters — and 25 sampled at random contained zero
compound-id assertions. They were NMR shifts, reagent quantities,
chromatography settings, and — most often — plain locants inside other
compounds' names. So an occurrence is only counted where the document
*declares* the id, in one of three shapes: a label word within a short
window to the left (`Example 24:`, `Cpd. No. 7`), a colon-separated,
line-initial id (`1D: tert-butyl ...`), or a parenthesised trailing
reference (`...triazin-4-amine (544)`).

The name text is searched forward from a declared id (up to `_LINES_AHEAD =
3` lines, capped at `_SPAN_CAP = 500` characters — the longest observed
name in the shipped structures dump was 343 characters) and, for the
parenthesised shape, backward to the nearest line or semicolon boundary.
Region bounds are set by *neighbouring* id occurrences of any id, not just
this one, so a name separated from id X by an assertion of id Y belongs to
Y structurally, not by a proximity score.

Candidate name text then goes through the same transformation ladder the
heading route (B4) uses — `as_is_whole`, `dewrap_whole`, `dewrap_seeded`,
`stereo_stripped` — and the same coverage gate at `_COVERAGE_MIN = 0.9`.

`config.FINISHED_ONLY` (default on) applies at the id level here: an
occurrence introduced by `Intermediate`/`Step`/`Preparation` does not count
as a declaring assertion, so an id whose only assertions are of that kind
resolves to nothing.

A narrower guard, `_INTERMEDIATE_FOR`, refuses one specific shape on its
own: `Preparation of Intermediate for Example 534.` reads as if it
declares compound 534, but it names the *intermediate*, not the example
compound. Before this guard existed, that shape anchored a synthesis
intermediate's name to the real compound's id — found by `mass_gate.py`
(C5): compound 534's own row printed `MS (ESI) 561 (M+H)`, but the
resolved structure's own computed mass was 185.09, the intermediate's
mass, not the compound's.

Measured on a fixed 20-patent sample, of 7,322 distinct assay compound ids:

| route | resolved | rate |
|---|---|---|
| prose + heading + table (the older combination) | 3,424 | 46.8% |
| `cid_first` alone | 3,668 | 50.1% |
| table + `cid_first` (today's default combination) | 4,409 | 60.2% |

Where the two combinations disagree about the same id (98 of 2,683 ids both
resolve, 3.7%), 88 of the 98 are the older prose route shipping a strict
substring of the right answer — an amputated name truncated at a tail-word
boundary, or a bare heterocycle fragment. That is what the retired
proximity guess cost.

**A compound the patent drew instead of naming.** When `cid_first` cannot
find a name for an id whose own table row carries a `<chemistry>`/`<img>`
element, it emits a row with `name`/`smiles`/`inchikey` all empty and
`drawn_ref`/`drawn_file` set — a marker for "the structure is here, in a
picture," not a failure. Measured over 8 patents, 2,783 assay compounds
with no structure split 75%/25% between "own row carries a drawing" and
"no drawing anywhere, genuinely unexplained." Confusing the two would turn
this package's own defects into a false ceiling.

That marker is refused in two further cases, each recorded with its own
reason rather than folded into "drawn": a table header holding a shared
scaffold with the row supplying only a substituent fragment
(`markush_kind="header_scaffold"`), and a substituent table where the row
picture is one R-group shared across rows, not the compound
(`markush_kind="substituent_table"` — confirmed on 644 compounds across two
patents, 2.8% of everything marked drawn corpus-wide). Both still emit a
row, carrying pointers to the scaffold and fragment chemistry ids in
`markush_parts`, so enumeration — not built here — has somewhere to start.

**A cid can itself be a name.** `uspto_assays`'s own id-column detector
occasionally scores a `Name` column as the id column, landing a 70-344
character IUPAC name in `AssayRecord.cid`. Rather than discard these as
unsearchable, `cid_first` hands them to OPSIN directly, unsearched. One
patent in the sample (US9018217) has no example numbers anywhere — 122
assay records keyed entirely by name — and this branch is the only reason
that patent yields any structures from this route at all.

Wall clock: about 1.5 seconds per patent (max 3.2s), $0, no network.

### B2. `table_names.py` — names already enumerated in a table cell

`uspto_xml.description_text` drops every `<tables>` block before prose
extraction begins — correct for assay values, wrong for names. Some
substituent tables carry the fully-enumerated compound name beside its own
R-groups, keyed by the same compound number the assay tables use. This
route reads it. It does not enumerate anything itself.

A name column is detected two ways: an explicit header containing the word
"Name" (the strong signal, kept even when the row's id column is blank —
OPSIN, not the presence of an id, is still the acceptance gate), or an
unlabelled column that the assay reader could not classify at all
(`kind == UNKNOWN`) where a majority of its non-empty cells are long and
hyphen/bracket-heavy, and the row carries its own id. On one patent, adding
the second signal raised name-bearing cells found from 402 to 805.

**Line-wrap inside one cell, not across two.** USPTO's line-wrap in a table
cell is not the split-`<entry>` shape A1 describes; it is a single literal
ASCII space injected where a printed line wrapped, inside one `<entry>`
(verified at the byte level). `sources/dewrap.py` — shared with the heading
route below — offers OPSIN two candidates besides the untouched cell:
`targeted` (remove only whitespace adjacent to `-()[]{},`, measured over
7,764 name-cells carrying an internal space: the character before the
space is a hyphen in 24,283 of about 35,000 tallied adjacencies) and
`aggressive` (remove every whitespace run — never yet the winning
candidate on this corpus, kept as a fallback for patent 138).

**A second wrap runs downward, not sideways.** The same compound's name can
be spread across the same column index on consecutive `<row>` elements —
one compound over six rows on one patent. Reading a row at a time loses the
compound, or worse, lets one fragment (`O1CCCCC1`, tetrahydropyran — one
substituent of an 11-row name) parse alone and ship as the whole structure.
`table_names._records` rejoins these; the record boundary is "the id
becomes complete," not "the id cell is non-empty," because the id itself
sometimes wraps.

**The rescue stage.** `name_repair.py` (shared with B4, described in B5)
runs on cells no dewrap variant resolved. Measured on the full 137-patent
cache: 18 of 16,054 unresolved name-column cells recover (0.11%), all from
bracket-stack patterns; the three regex character-substitution patterns
recover none from table cells. Kept because the cost is local CPU on a
java subprocess ($0) and every recovery is confirmed, not guessed.

### B3. `iupac_names.py` — shared machinery, and the prose/heading fallback route

This module holds the OPSIN-candidate machinery both `cid_first` and
`table_names` import (`_SEED`, `_variants`, `_coverage`, framing strips, the
`NamedCompound` dataclass). It also implements a second, complete route,
selected by setting `IDENTITY_ROUTE=prose` instead of the default
`cid_first`: find every OPSIN-parseable name in the description text, then
guess which compound number it belongs to by proximity (`anchor.py`).

**Two passes, one corpus.** `extract_names` seeds the full document text
including `<heading>` elements, not just `<p>` prose — both a compound's
name and its `Example N` label are `<heading>` elements, and the paragraph
beneath is only the synthesis procedure. Reading `<p>` alone was measured
to lose the identity twice over.

Brute-force candidate generation, because 63.8% of a 182-name OPSIN
failure sample were boundary errors (48.4% unbalanced brackets, 15.4%
truncation), not names OPSIN genuinely could not parse:

```
seed (name-legal characters, >= IUPAC_MIN_SEED=12, a letter run and a
      digit/bracket/hyphen)
  → fan out: raw / tail-word extended / bracket-balanced left and right /
    leading prose stripped repeatedly / re-anchored at internal
    boundaries — up to IUPAC_MAX_VARIANTS=12 per seed
  → OPSIN, batched over the whole patent at once
  → longest accepted span per seed position wins; dedup on InChIKey
```

A `<heading>` has no boundary problem — the patent has already said where a
name starts and stops — so headings get their own pass, whole, in three
transformations least-invasive-first (`as_is_whole`, `dewrap_whole`,
`dewrap_seeded`), gated by the same `_COVERAGE_MIN = 0.9` containment ratio
that stops a comma-split amputation (`nicotinamide` pulled from the tail of
a much longer name, scoring 1.00 on a bare length ratio and 0.14 on
containment) from being counted as a repair. The heading pass runs last and
only adds to `extract_names`'s own dedup set, so turning it on cannot cost
the prose pass a structure.

**A relative-stereo or unmappable-stereo name is a set, not a molecule.**
`(1R*,2S*)-...` states a relationship between two stereocentres with
absolute configuration left open; OPSIN 2.9.0 also cannot place a
lowercase pseudo-asymmetric locant (`(1r,4R)-...`) at all — measured as the
single largest identity blocker found in this corpus, 155 of 217
asserted-compound losses on a 3-patent sample. Both get `markush=True` and
**no InChIKey**, because a single-structure identifier would be a false
claim about a set. `markush_kind` distinguishes the two from two further,
unrelated markush populations described in B1 (`substituent_table`,
`header_scaffold`) — over 137 patents, 990 flagged rows split roughly
65%/22%/10%/3% across the four kinds, and the two stereo kinds already
*have* a structure; nothing needs enumerating for them.

**Why this route is not the default.** Measured on the same 20-patent
sample as B1's table (this route alone gives 3,424 of 7,322, cid_first
alone gives 3,668), and on the 98 ids where the two disagree, 88 are this
route amputating a name at a tail-word boundary. The proximity guess this
route depends on (`anchor.py`) is the retired part, not the OPSIN
candidate machinery, which both routes still share.

### B4. `anchor.py` — the retired proximity guess, kept as a pure module

Used only by the `prose` route above (`find_cid`), not by `cid_first`,
which anchors structurally instead. Given a name and the document text, it
looks both directions for a compound id sitting within `_ANCHOR_BOUND = 25`
characters, in one of three shapes (`<id> <name>`, `<id>: <name>`, `<name>
(<id>)`), and returns the closest match — or, when two different ids each
have close evidence, a `clashed=True` result carrying every candidate
rather than a silent guess.

A widened window was measured and rejected: past 25 characters, more
matches are clashes, not correct anchors. A cross-reference citation
("...in a manner analogous to Example 1...") reuses the identical
`<id>\s<name>` shape a real heading uses at the identical short distance,
so distance alone cannot separate them; what does is what fills the gap —
a real heading has only whitespace and, at most, a stripped stereo
descriptor between id and name, while a citation always has a verb or
punctuation there.

### B5. `dewrap.py` and `name_repair.py` — shared text-corruption repair

`dewrap.py` holds the one definition of "which whitespace is a typesetting
wrap point" (`WRAP_ADJACENT`, adjacent to any of `-()[]{},` and their
Unicode dash variants), used by both `table_names.py` and `iupac_names.py`.
An aggressive (strip-everything) candidate is offered by the table route,
because a cell is one field with one value; the prose route asks only for
the targeted candidate, because running prose legitimately contains word
breaks an aggressive strip would destroy.

`name_repair.py` holds seven confirmed corruption patterns as a data-driven
registry (`PATTERNS`), each a `(site detector, fix generator)` pair:
`footnote_digit_tail`, `dropped_open_paren`, `digit_for_paren`, `ch_as_ey`,
`stray_closing_bracket`, `stray_opening_bracket`, `dropped_close_bracket`.
Two are `trusted` (accepted on OPSIN parsing alone); the other five also
require the corrected fragment to occur verbatim elsewhere in the same
document as published — corroboration, not just a plausible fix. Wired as
a rescue stage in both `iupac_names.extract_names` and
`table_names.extract_table_names`, run only on spans no dewrap variant
resolved. Over the full 137-patent cache: the prose route's 79 confirmed
repairs net +67 distinct structures (12 collapse into structures another
span already found); the table route's 18 confirmed repairs are all
bracket-stack patterns. Every recovery is purely additive — zero
structures lost on any patent by adding this stage.

### B6. `reagents.py` — label a reagent, never delete it

`classify(name, smiles)` returns one of three labels: `"compound"` (the
default — no evidence either way), `"reagent"` (exact match, after salt-
suffix stripping, against a ~230-entry curated lexicon of solvents, bases,
acids, coupling reagents, and catalysts), or `"trace_fragment"` (no lexicon
match, but an RDKit heavy-atom count of 3 or fewer — OPSIN boundary debris
and bare metal/ion names). A length proxy (`>=45 characters`) was tried
first and rejected by measurement: on a 3-patent, 2,392-structure sample it
discarded 246 real short building blocks while still missing 11 long
spelled-out reagents.

This module never deletes a structure. A caller that used to drop a
reagent-labelled row is exactly the shape this refuses to reintroduce — an
intermediate or a purchased starting material next to the real target is
still useful with no assay value, and this module has no evidence to say
which role a labelled compound actually plays.

### B7. The name-tier repair loop

`repair/name_*.py` parallels the layout tier's shape — gap → synthesize →
outcome → loop — but the axis is different, and it is a separate library
from `repair/rules.py`.

**What counts as a gap.** Not an OPSIN rejection in general; a compound the
patent *asserted* — named, in its own voice, at a heading or in a table
cell — that still resolved to nothing after B1-B5 ran. Two earlier
triggers were tried and measured to fail: every raw OPSIN rejection (a
first version reported 2,746 gaps across 53 patents, of which the first 60
inspected all parsed perfectly well — an anchoring question mistaken for a
parsing one) and shape clustering of failing seeds (five hypothesis-free
skeletons over 82,398 failing seeds, the best putting 91% of failures in
clusters of 5 or more, still too coarse to buy one rule per cluster).

Two populations are excluded from the loop entirely, each recorded with
its own reason rather than treated as an ordinary failure: phane/
macrocyclic names (`[2.2]paracyclophane`-class nomenclature OPSIN 2.9.0
does not implement at all, IUPAC P-26), and headings naming two compounds
at once (a split problem for a future anchor change, not a character fix).
The stereo-unmappable population (B3) is pulled out by a *measurement*
rather than a regex — re-asking OPSIN with `--allowUninterpretableStereo`
and checking whether the verdict flips — at 53 of 630 gaps (8.4%) on the
first 30 patents sampled.

**What a rule looks like.** Not five kinds; one — a `(site regex,
replacement template)` pair, fingerprinted on OPSIN's *own diagnosed error
class* (`kind|token_class|position`), deliberately not on the name's own
shape, since shape clustering already failed as a trigger. `ground()`
checks the site compiles and matches, the edit actually changes the
string, the edit distance at the worst matched site is at most 4
characters (a length-of-replacement bound was tried and rejected — a
7-character correct replacement for a 2-character corruption failed it),
and the site regex touches at most 2% of *already-parsing* document names
(collateral damage).

**Outcome is stricter than the layout tier's, and says why: OPSIN
acceptance alone is worthless here.** Most truncations of a valid name are
themselves valid, smaller names — parsing proves nothing about
correctness. Three conditions, and unlike the layout tier's outcome gate
all three are necessary: (1) OPSIN parses the repaired string; (2)
coverage — a normalised edit distance against the original, not the
containment ratio B1/B3 use, since a repair *substitutes* rather than only
deletes — clears `COVERAGE_MIN = 0.9`; (3) the repaired fragment is
corroborated verbatim elsewhere in the document, exactly as
`name_repair.py`'s untrusted patterns already require.

**Escalation counts failures across patents, not within one.**
`ESCALATE_AFTER = 3` failed attempts at one *failure class* — not one
gap — before that class is written off for the current
`NAME_SYNTH_EPOCH` (currently `"n1"`, in `repair/name_rules.py`, tracked
separately from the layout tier's `SYNTH_EPOCH`). The first version gave
up after a single failed batch and was a measured net loss: 604 gaps
produced 71 repairs, and 537 gaps were skipped without ever being
attempted because one early batch failure marked their whole class dead.

**A rule outranks an escalation for the same class.** Adopting a new
pattern retroactively de-escalates any failure class whose recorded
signature the new pattern's `signatures` list covers — so a capability
gained later is not permanently blocked by a verdict from before it
existed.

**`repair/name_capability.py` exists and is not wired in.** It is a third
tier, above the regex `NamePattern`: model-authored Python, two functions
(`site(name) -> list[(start, end)]`, `fix(span) -> list[str]`) for failure
classes a regex cannot express — concretely, walking bracket depth for
`unmatched_opening_bracket`/`unmatched_closing_bracket`, which together
were 178 of 604 gaps (30%) on the sampled corpus, unsolved after three
genuine regex attempts each. It is gated by five checks before adoption: an
AST-based static check (no imports, no `eval`/`exec`/`open`/`compile`/
dunder-attribute access, exactly the two functions `site`/`fix` and
nothing else at module scope), execution in a **subprocess** under a
restricted builtins set with a 20-second wall-clock cap, a contract check
on the returned shape, a zero-tolerance collateral check (stricter than
the regex tier's 2%), and the same OPSIN-parse-plus-coverage-plus-
corroboration outcome gate every regex rule faces. Its own epoch constant
(`CAPABILITY_EPOCH = "c1"`) exists, but `name_loop.py` never imports or
calls it — it is authored, fully gated, and dormant, in the same state
`loop.py`'s own docstring records for the layout tier's `capability.py`.

---

## Track C — compound identity from the drawing

USPTO's XML states which image file belongs to which compound row; it does
not hold the pixels. This track exists because a full census found 2,866 of
2,897 drawn compounds (98.9%) hold a whole-compound image in their own row
— a drawing does not imply a missing name (many drawn compounds already
resolve from text), but where it is missing, the picture is usually the
whole answer, not a fragment.

### C1. `sources/images.py` — the worklist and the scoring harness

Builds a work queue from the shipped structures/dump artifacts named in
`config.MANIFEST`: a `VALIDATE` job for a compound that already has a
text-derived structure (so a recogniser's answer can be scored against a
known InChIKey) and a `RECOVER` job for one that does not (drawn markers
from B1, and compounds with no text answer at all). Markush rows are
excluded. `fetch()` writes one image to disk and verifies the served
filename matches the name the XML states before accepting it — a mismatch
here would silently attach one compound's picture to another's number.

**Why `VALIDATE` has to be a real population, not a formality.**
`predict_SMILES` returns a bare SMILES with no confidence score, so its
answer can only be trusted as far as it has been checked against something
— and the only check available is a structure this package already
derived from the patent's own words. Measured over the shipped dump: 3,074
compounds carry both a text-derived structure and a picture in their own
row (`VALIDATE`); the other 2,871 have no text answer at all (`RECOVER`)
and cannot be checked this way, which is why `decimer.py` (C4) always runs
the `VALIDATE` half first.

**The image filename, not the `<chemistry>` id, is the join key — and the
file's own number is not the compound number.** A row names a
`<chemistry>` id (`CHEM-US-00315`); that element names a file
(`...-C00315.TIF`). The `C00315` counter runs over every `<chemistry>`
element in the whole document, in document order — scaffolds, schemes and
intermediates included — so it drifts ahead of the example numbering.
Measured corpus-wide, `cid == C-number` in only 545 of 2,834 cases
(19.2%), and those are coincidence: never compute an offset, read the link
the XML states.

`config.IMAGE_SOURCE` picks where the pixels come from. The default,
`gpatents`, is one small GET per image (~3 KB); measured against the files
the XML actually names, it covers 99.7% of them (5,834 of 5,849, across 15
patents). The alternative, `uspto`, is ground truth but impractical on a
laptop — the Red Book product that ships the TIFs beside the XML comes
only as ~3.7 GB weekly TARs with no per-patent split, about 130
grant-weeks for this corpus.

`RESULT_FIELDS` — the contract every recogniser writes to — is
`(patent_id, cid, image_file, n_segments, smiles, inchikey, confidence,
recogniser, error)`. `recogniser` and `confidence` exist so two models'
answers for one compound are two rows, never one overwriting the other
(the row key is `(patent_id, cid, recogniser)`), and so a model's own
per-structure confidence — never comparable across models, never itself a
correctness signal — travels with the answer rather than being discarded.

`score()` and `recovered()` read a recogniser's results file back:
`recovered()` turns `RECOVER`-job rows into `NamedCompound`s
(`source="image"`), carrying `drawn_ref`/`drawn_file` so they can always be
traced to the picture they came from. **Neither is called from
`verify.dump()`.** A DECIMER run's results reach `structures.tsv` only if
a human separately calls `images.recovered()` and merges the output — this
track is a working side-channel, not yet part of the automatic pipeline.

### C2. `sources/gp_images.py` — Google Patents, as an image source only

`config.GP_ENABLED` (default off) gates `image_urls(patent_id)`, which
parses Google Patents' page for `{image filename stem -> rendered PNG URL}`
and caches the result. This is deliberately the only thing GP is used for.

Google Patents' compound microdata was checked directly rather than
assumed: it is entity annotation over the running text (the same kind of
model SureChEMBL uses), not per-drawing structure recognition, and its `id`
field is either an InChIKey or an internal numeric entity id — never this
corpus's own compound number. It cannot join to an assay row, so it is
never read as a structure source here, on either identity route.

What GP *does* serve reliably is the rendered image for a filename this
package already knows to ask for: measured at 1,065/1,065, 480/480, and
843/843 on three tested patents, cid → chemistry-id → image filename → GP
URL, joined end to end.

### C3. `sources/image_ocr.py` — read the name printed inside the picture

Some patents render the compound's IUPAC name inside the drawing, beside
the structure. Where they do, OCR plus OPSIN gives an exact answer with no
structure recognition involved, and — unlike a bare recognised SMILES — the
answer is self-checking: it passes the identical OPSIN gate every
text-derived structure in this package passes.

```
image → OCR (easyocr) → OPSIN parses?  yes → resolved, source="image_ocr"
                                        no  → hand to DECIMER, source="image"
```

Gated by `config.IMAGE_OCR` (default off — not for risk, but because OCR
costs 2-4 seconds per image on a laptop CPU and nothing here caches the
result yet). `supersede()` replaces a drawn marker (B1) whose picture
carries a resolved caption; it never adds a second row for a cid a marker
already occupies, and it never overwrites a row the text track already
resolved — where a caption and the text track would disagree, that
disagreement is left for `mass_gate.py` to flag, not adjudicated here.

Measured over 114 drawn images sampled from two captioning patents: OCR
text length runs 0-185 characters, median 15 — most images return only atom
labels (`CH3`, `OH`), not names. Of 33 images returning 40+ characters (the
`MIN_NAME_CHARS` floor), repairing the render's own character confusions
(a bilevel scan turns `]` and `)` both into `J`; `1` into `l`; a lost
hyphen at a line wrap) took confirmed parses from 3 of 33 to 28 of 33
(85%). The remaining 5 are one compound family whose locants OPSIN cannot
place even after a correct read — recorded, not shipped, because the
skeleton OPSIN would return under a stereo-relaxation flag denotes a set of
stereoisomers.

### C4. `decimer.py` / `decimer_vm.py` — DECIMER on a Colab GPU, one patent at a time

Structure recognition proper, run manually, patent by patent, because a
recogniser with no confidence score can only be trusted as far as it has
been measured — and the measurement is the `VALIDATE` half of C1's
worklist, compounds whose structure the patent's own text already gives. A
corpus-wide run before that measurement exists produces one number nobody
can attribute to anything.

```
python3 -m patentdb3.decimer plan   <PID>
python3 -m patentdb3.decimer run    <PID> -s <colab-session>
python3 -m patentdb3.decimer ingest <PID>
```

`plan` fetches (if `GP_ENABLED`) and runs C3's OCR pass first — cheapest
first, and not an optimisation: where a caption resolves, DECIMER's answer
would be strictly weaker on the identical row. `run` stages the remaining
images, tars them, uploads to a named Colab session, and executes
`decimer_vm.py` there. `decimer_vm.py` is the only file in this tree that
runs somewhere else, and it imports nothing from `patentdb3` for exactly
that reason. `ingest` merges `results.tsv` back into the canonical results
file keyed on `(patent_id, cid)`.

`config.DECIMER_ENABLED` (default on) exists but is not currently read by
either `decimer.py` or `decimer_vm.py` — this track runs whenever a human
invokes the CLI, regardless of the flag. `config.DECIMER_SEGMENTATION`
(default off) documents a decision rather than gating code directly:
`decimer-segmentation`'s pinned dependency (`tensorflow==2.10.1`) would
silently cap `decimer` at version 2.2.2, which has no confidence output at
all — the trade this constant names is refused, and segmentation is
offered as a separate opt-in environment on the VM side instead. The
measured default (`decimer >= 2.8`, no segmentation): 93.4% on 61 known
answers from one patent, about 2 seconds per image on a T4 GPU.

### C5. `sources/mass_gate.py` — does the resolved structure weigh what the patent says

Every other check in this package tests the *name*. This one tests whether
a resolved structure belongs to the compound number it was filed under — a
question a valid name, a valid SMILES, and a valid InChIKey can never
answer, because a wrong anchor is shaped exactly like a success. It was
found by accident: `image_ocr.py` read a caption that disagreed with the
text track for one compound, and the patent's own table row settled it —
the text track had anchored a synthesis intermediate's name to the wrong
compound number (see `cid_first`'s `_INTERMEDIATE_FOR` guard, added because
of this).

`check()` compares a resolved structure's computed mass against the `MS
(ESI) (M+H)` value printed in that compound's own table row, when one is
printed. **Its reach is small, and narrower than the patent count
suggests.** Measured over the full 137-patent corpus (`verify --all --dump
--no-heal`, 38,671 structures): 74 rows carry both a resolved structure and
a printed mass (0.2%), and only 5 of the 137 patents print an MS inside an
assay row at all. On three of those five the two sets do not overlap at
all — US10125101 (7 resolved, 30 with a mass), US10329273 (1 resolved, 23
with a mass), US20240166635 (0 resolved, 195 with a mass). Nearly all 74
weighable rows sit on one patent, US10730863; no second patent has yet been
found where the gate does real work. Do not read "5 patents print a mass"
as "5 patents' worth of checking".

Both the monoisotopic and average mass are computed and the nearer is
kept, because a patent may print either without saying which. The
tolerance is flat, not scaled to molecule size — scaling was tried first
and measured out: on US10730863 a scaled window of 1.31 Da at 596 Da let
three of DECIMER's four wrong answers on that patent sit just inside it,
each off by the characteristic +2.016 Da of a two-hydrogen misread.
Comparing against both masses and keeping the nearer absorbed the reason
the window used to need to scale (a patent may quote monoisotopic on one
row and average on another, ~1.4 Da apart at 600 Da), so a flat, tight
window now only has to cover rounding.

**Nothing is dropped.** A contradicting row keeps its structure and ships
with `mass_check="contradicts"` and a signed `mass_delta` (in Da) attached.
Making the defect visible is the change here; deciding what to do about a
flagged row is left to whoever reads the artifact next. A row with nothing
to check against reads `mass_check=""`, and — because rdkit is an optional
dependency — a row that *had* something to check and could not be, because
rdkit is not installed, reads `mass_check="gate_unavailable"` rather than
the same blank, so the gate's absence cannot be mistaken for "checked and
fine."

---

## Outputs

`verify.py` is explicitly not an output pipeline — merging the two dumps
into one compound record is future work this file is deliberately not
doing. Its shape (long format, every field the reader carries, nothing
derived) is what any later assembly should be designed around.

- `one(pid)` — prints what the reader (and, under `--dump`, the repair
  loop) produced for one patent: table/record/compound counts,
  `parse_fidelity`, a sample of rows, and what fraction of rows are
  missing `table_id`/`column_header`/`unit`.
- `dump(pids)` — writes `reader_dump.tsv` (one row per (compound, assay)
  measurement) and `structures.tsv` (one row per resolved or marked
  compound, from the table route (B2) plus whichever `IDENTITY_ROUTE`
  selects — `cid_first` (B1) by default, or `prose` (B3) — plus C3's
  caption supersession. Precedence: a table-cell row (B2) wins over a
  `cid_first`/prose row (B1/B3) for the same id, and a resolved OCR
  caption (C3) replaces a drawn marker but never a text-resolved row).
  B1 and B3 are alternatives, not additive — they share one config flag
  and only one of them runs in a given dump. Both files are overwritten
  on every run — one dump, one manifest, by design.
- The manifest (`patentdb3/out/latest.json`) records which pipeline
  produced the dump: `self_heal` (repair loop on/off), `gaps_found`/
  `rules_adopted`/`rows_recovered`/`superseded`/`usd_spent` for the layout
  tier; `iupac_names`/`structures_rows`/`structures_sources`/
  `structures_repaired` for the identity dump; `image_ocr`/`mass_gate`/
  `image_ocr_superseded` for what optional routes actually ran (a count of
  zero is ambiguous by itself — off, on-but-found-nothing, and on-and-
  found-none read identically without these flags); and `loss_log`/
  `loss_counts` pointing at the structured drop log below. A reader-only
  dump and a healed dump are otherwise indistinguishable files.

  As one concrete example of the manifest's shape — not a corpus
  measurement — the file on disk on 2026-08-14T13:48:47 covers 1 patent
  (US10730863), self-heal off, 428 assay rows, 444 structures (80 from
  `table`, 364 from `cid_first`), image OCR off, and the mass gate
  available. Its loss log counts 24 `cid_first_not_finished`, 10
  `tables_dropped`, and 3 `mass_contradicts_patent` — a live instance of
  C5's mass gate flagging a real disagreement, on the one patent where the
  gate currently does real work.

- `sources/losses.py` — a single structured JSONL sink
  (`patentdb3/out/loss_log.jsonl`) every drop point in the identity routes
  writes to: what was dropped, at what position, and why, with enough
  context to act on without re-running anything. One JSON object per line,
  always carrying `loss_type` and `patent_id`; every other field is
  specific to the type. It wraps existing control flow — the decision to
  keep or drop something is unchanged; a `record()` call sits beside it.
  Truncated fresh on the first write of a process, then appended to for
  the rest of that run, so a stale log from an earlier invocation can
  never be mistaken for the current one.
- `to_excel.py` reads the dump path from the manifest, never a hardcoded
  string, and writes `records`/`assays` sheets plus — unless `--no-bdb` —
  a `bindingdb` sheet comparing against `output/bindingdb/our_patents.tsv`
  at **1% tolerance**, not v2's 5%: the reader does not round, so the only
  legitimate gap between an extracted value and a reference is the
  reference's own rounding to 2-3 significant figures.

  Directly measured against a single-patent dump (US8952177, self-heal on,
  written 2026-08-07T18:20:45): **175/190 BindingDB reference points agree
  within 1% (92.1%)**. This is one patent from whatever the last `--dump`
  run happened to cover, not a corpus benchmark — re-derive it, and state
  which patents were in the dump, before quoting it for anything that
  matters.
- `inspect.py` — a backtrace tool, not part of the dump pipeline. For one
  `(patent_id, cid)` pair it prints the shipped row (as it sits in
  `reader_dump.tsv` right now) next to the reader re-run on the same
  cached XML this instant, plus the patent's own raw `<row>` elements for
  that compound — three views, shown separately, never merged into one
  verdict. No API calls, no network; a patent not already cached is
  reported missing rather than fetched.

---

## What is still deliberately left out of v3

- **No orchestrator, still.** No `process_patent()`, no route classifier,
  no single call that produces one merged record per compound. Both dumps
  now carry `cid`; the join itself is unbuilt.
- **No Markush enumeration.** The scaffolding to find *where* enumeration
  is needed now exists and is precise — `markush_kind` distinguishes a
  substituent-table row, a header-scaffold row, a relative-stereo name,
  and a stereo-stripped name, and the first two carry the scaffold and
  fragment chemistry ids in `markush_parts` — but nothing composes a
  scaffold and a substituent table into a structure. `rdkit.Chem.molzip`
  is the identified mechanism; it is not called anywhere in this tree.
- **No LLM assay extraction (HARVEST).** The repair loop's synthesis calls
  ask for a *rule*, never for data, at a different cost profile entirely.
- **No output validator**, same as before — removed prior to the v2→v3
  split.
- **GP is never a structure source, on either identity route or the image
  track.** `gp_images.py` is new since the last revision of this document,
  but it only resolves an image URL for a filename this package already
  knows to ask for; it never reads GP's own name/SMILES/InChIKey
  microdata, and nothing in the identity routes joins to GP's own
  compound numbering.
- **DECIMER's results do not reach `structures.tsv` automatically.** They
  land in a separate results file, merged by `decimer.py ingest`; getting
  them into the structures dump needs a human call to
  `sources/images.recovered()` today.
- **The model-authored code tier for name repair exists and is not
  wired in.** `repair/name_capability.py` is fully gated (AST check,
  sandboxed subprocess, zero-collateral requirement, the same outcome
  gate every regex rule faces) and targets exactly the population
  (`unmatched_opening_bracket`/`unmatched_closing_bracket`, 30% of the
  sampled name-tier gaps) the regex tier could not reach — but
  `name_loop.py` never calls it.
- **No code-patch tier for the layout tier either.** `RepairReport.
  capability_gaps` — a gap where a library rule ran and produced nothing —
  is still populated and still has no consumer.
- **No eval/audit scripts.** `scripts/eval/*` from v2 does not exist in
  v3. There is no automated "is this module reachable from production"
  check.

## Cross-cutting properties

- **v2 and v3 share nothing, deliberately.** `core/config.py` derives
  every path from its own `__file__`; all of v3's reads and writes land
  under `output_v3/`. The only thing that ever crossed the boundary is the
  cached grant XML, copied into `output_v3/uspto_xml/` and removed from
  `output_v2/` — an input v3 reads and never writes to.
- **The layout-tier rule library is tracked in git.** `patentdb3/data/
  layout_rules.json` currently holds 172 rules (62 `column_map`, 49
  `bin_key`, 35 `value_pattern`, 16 `escalate`, 9 `not_assay`, 1
  `row_regex`), so a fresh clone gets the free layouts for nothing. Its
  journal (`output_v3/rule_adoption_journal.jsonl`) is gitignored, as is
  everything else under `output_v3/`.
- **The name-tier rule library is tracked at the same path convention**
  (`patentdb3/data/name_rules.json`, per `repair/name_rules.py`'s
  `LIBRARY_PATH`) **but the file does not currently exist on disk** — no
  name-tier pattern has been adopted and persisted in this tree yet.
- **API response caching** (`core/api_cache.py`): SHA256(model + prompt +
  full image data) keys a JSON file under `output_v3/cache/`. The image
  hash is taken in full, not truncated — a truncated hash previously
  collided two different image crops in v2 and served one's answer for
  the other's request.
- **Prompt-cache tokens are priced, not treated as free**
  (`core/cost_tracker.py`): writes at 1.25x the input rate, reads at 0.1x,
  Anthropic's standard 5-minute-TTL multipliers. `usage.input_tokens`
  itself excludes anything served from or written to the prompt cache, so
  omitting these two buckets under-reports spend rather than over-reports
  it — the worse direction for a spend gate to be wrong in.
- **Cost attribution is read off the call stack**, not an opt-in call a
  site has to remember to make. `cost_tracker.derive_route()` walks
  outward past a fixed set of transport-only frames
  (`core.cost_tracker`, `core.api_client`) until it reaches the code that
  actually wanted the answer, and labels the spend
  `<module>:<function>` — a path added tomorrow is attributed the first
  time it spends, with nothing to register.
- **Add every new model to `config.PRICING`.** A model missing from that
  table is billed at the Opus row by `compute_cost` — this has already
  made one model look about 5x its true cost in a real comparison.
- **No automatic cache invalidation**, same as v2: every cache under
  `output_v3/` is cleared by hand.
