"""v3 configuration — only what the reader and the rule tier actually read.

Deliberately NOT a copy of `patentdb/core/config.py` (459 lines). That file
carries every tier's flags — HARVEST_BURST, IUPAC_BURST, STRATEGY5_LLM,
ASSAY_REALIGN, PUBCHEM_NAME_LOOKUP, LLM_RECOVERY, MINERU_OCR and the rest — and
copying it would import the assembly's assumptions into a package that has no
assembly. Every constant below is read by something; `import_audit`-style
checks should keep it that way.

PATH ANCHORING. `OUTPUT_DIR` is derived from this file's own `__file__`. THE
TWO PACKAGES NO LONGER SHARE ANYTHING — an earlier version of this note said
they shared one cache, which was true for about a day and is now false: the
cached grant XML was copied into `output_v3/uspto_xml/` and removed from v2, so
nothing here resolves to an `output_v2` path. What survives from that note is
the hazard it described: relocating this package changes where everything is
read from and written to, with no error, only an empty cache and a fresh log.

Two import-time side effects are inherited on purpose, because the modules that
depend on them assume they have already happened: `LOGS_DIR` is created here,
and `core/api_cache.py:20` creates the response-cache directory when imported.

No feature flags live here, and the rule tier no longer has one: adoption is
decided by measuring what a rule does to the patent in front of it, which is
not a thing anyone needs to switch off. Anything else a caller wants to gate
belongs to the assembly, which does not exist yet.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
#
# READS AND WRITES GO TO DIFFERENT PLACES, DELIBERATELY.
#
# v3 must not be able to poison v2's artifacts, and v2 must not be able to
# explain away a v3 result. Everything v3 WRITES — its response cache, its API
# log, its rule journal, its dumps — lands under `output_v3/`, which v2 has no
# knowledge of. The only thing shared is the cached grant XML, and that is an
# INPUT: v3 reads it and never writes there.
#
# The cost of the split is honest and small: v3 starts with a cold response
# cache, so the repair tier re-pays for any rule it needs to buy fresh. That is
# bounded by the measured lifetime cost of the whole tier ($2.18 / 312 Haiku
# calls), and the 172 rules already bought came across as data in
# `data/layout_rules.json`, so the common shapes cost nothing.
#
# The alternative — one shared directory — was the arrangement for most of a
# day, and it is how "which code version produced this artifact" became
# unanswerable: a corpus that spanned two code versions with no way to tell
# them apart from the files.
PACKAGE_ROOT = Path(__file__).parent.parent          # patentdb3/
REPO_ROOT = PACKAGE_ROOT.parent                      # the repo root

# Everything this package reads or writes lives under here. NOTHING in v3
# resolves to a v2 path — the cached grant XML was copied to `output_v3/` and
# removed from `output_v2/`, so there is no shared directory left at all.
OUTPUT_DIR = REPO_ROOT / "output_v3"
XML_INPUT_DIR = OUTPUT_DIR / "uspto_xml"   # cached grant XML, v3's own copy
LOGS_DIR = OUTPUT_DIR / "logs"
RULE_JOURNAL = OUTPUT_DIR / "rule_adoption_journal.jsonl"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# THE EXTRACTION ARTIFACTS. Every producer writes these; every consumer reads
# them FROM HERE and never hardcodes a path of its own.
#
# `MANIFEST` is rewritten on every dump run and names the current `DUMP`, its
# timestamp, the patents in it and the XML directory it came from. An analytic
# reads the manifest, follows it to the dump, and reports the timestamp
# alongside its numbers. That is the whole mechanism, and it exists because
# analysis pointed at stale artifacts for days in v2 — numbers that were
# correct about a file nobody realised was old.
DUMP = PACKAGE_ROOT / "out" / "reader_dump.tsv"
MANIFEST = PACKAGE_ROOT / "out" / "latest.json"
XLSX = PACKAGE_ROOT / "out" / "reader_dump.xlsx"

# STRUCTURES is the second, and last, artifact `verify --dump` writes — the
# names/structures `sources/iupac_names.py::extract_names` resolves, separate
# from DUMP's assay records because the two are stated at different grains:
# once per compound here, once per (compound, assay) there. Same rule as DUMP
# applies: ONE canonical location, overwritten every run, named in MANIFEST so
# a consumer never has to guess which file a number came from. Do not add a
# second structures file beside this one for the same reason DUMP stays one.
STRUCTURES = PACKAGE_ROOT / "out" / "structures.tsv"


# ── Credentials ──────────────────────────────────────────────────────────
# env first, then patentdb3/.env, then the repo-root .env — the same order v2
# uses. An absent key is TOLERATED so the reader runs and the tests pass with no
# credentials; the rule tier's `propose` returns None instead, which makes
# `repair_patent` deterministic rather than broken.
def _load_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    for env_path in (PACKAGE_ROOT / ".env", REPO_ROOT / ".env"):
        try:
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


ANTHROPIC_API_KEY = _load_key()


# ── Models and pricing ───────────────────────────────────────────────────
MODEL_OPUS = "claude-opus-4-6"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# THE MODEL THAT SPENDS MONEY. Rule synthesis is the only paid path in v3, so
# this is the whole of the model choice and it belongs here rather than in
# `repair/synthesize.py` — a model constant scoped to the module that uses it
# cannot be found by anyone asking what a corpus run costs.
#
# Haiku: the task is small, narrow and highly constrained, and every proposal is
# run and measured before it is kept, so a wrong answer costs one retry. Sonnet
# was measured no better once the outcome gate is in place. There is no
# DEFAULT_MODEL — nothing in v3 calls a model without naming one, and a default
# nobody reads is the constant this file keeps accumulating.
SYNTH_MODEL = MODEL_HAIKU

# THE CODE-AUTHORING TIER RUNS ON SONNET, and the choice is measured. Run
# head-to-head on the two escalated bracket classes, identical tools, turns,
# prompt and gap population:
#
#     class                        Haiku          Sonnet
#     unmatched_closing_bracket     3 of 71       25 of 71
#     unmatched_opening_bracket     1 of 107       7 of 107
#
# Haiku plateaued at 3 across all ten turns with no upward trajectory; Sonnet
# found its 25 on turn 2. Per compound recovered Sonnet is also CHEAPER
# ($0.011 vs $0.031 on the closing class) — the raw bill is larger, the unit
# cost is not, and a rule bought once here is applied corpus-wide.
#
# This tier fires only on classes the $0 library and the Haiku regex tier have
# both already failed on, so it is rare by construction. `SYNTH_MODEL` stays
# Haiku, which is demonstrably sufficient there: 17 rules and 71 repairs
# across 30 patents for $0.46.
#
# MUST be a key of `PRICING`. The A/B that produced the numbers above was run
# with a hardcoded `claude-sonnet-4-5-20250929`, which is not in the table, so
# every Sonnet call was billed at the Opus row and the tier looked ~5x more
# expensive than it is — exactly the failure the PRICING comment below warns
# about, reproduced within a day of reading it.
CAPABILITY_MODEL = MODEL_SONNET

# USD per million tokens. A model MISSING from this table is billed at the Opus
# row by `cost_tracker.compute_cost`, which silently overstated the repair
# loop's cost by 15x until Haiku was added. Add every new model here.
PRICING = {
    MODEL_OPUS: {"input": 15.0, "output": 75.0},
    MODEL_SONNET: {"input": 3.0, "output": 15.0},
    MODEL_HAIKU: {"input": 1.0, "output": 5.0},
}


# ── HTTP / API retry ─────────────────────────────────────────────────────
# Read by `core/api_client.resilient_client`. These live HERE, not beside the
# client, because every knob that changes what a run costs or how long it takes
# has to be findable in one file — a timeout scoped to the module that uses it
# is invisible to anyone measuring the tier, which is how the previous three
# retry constants sat unread until they were deleted.
#
# A capability patch is a long generation over a large prompt, and a dropped
# connection cost two whole patents on the first real run: US10227341 and
# US10266548 both died on APITimeoutError with no retry, losing the gap rather
# than the request.
MAX_RETRIES = 3
API_TIMEOUT_S = 600.0


# ── Cost controls ────────────────────────────────────────────────────────
PER_PATENT_LM_CAP = 0.20          # soft cap, checked by patent_lm_exceeded()
PER_PATENT_LM_HARD_CAP = 1.00     # alarm
PER_PATENT_REALIGN_CAP = 1.50     # realigner bucket; no v3 caller today
COST_CEILING = 200                # USD, global
COST_THRESHOLDS = [50, 100, 150, 200]


# ── Identity extraction ──────────────────────────────────────────────────
# Compound names read out of the patent's OWN description and resolved by
# OPSIN. Free and offline — OPSIN is a java subprocess, no API, no network.
#
# ON by default. It is the only identity route that is ours: the name sits in
# the patent's own prose, so it carries the patent's own compound numbering and
# needs no bridge. The alternative — Google Patents' embedded SMILES — supplied
# 97.2% of v2's 17,459 compounds, of which 27.6% were still keyed by GP's
# positional id and could never be joined to an assay row.
#
# THIS DEFAULTED TO OFF, and the comment here argued both sides at once — it
# opened "On by default because..." and then said "DEFAULT OFF". The two
# blockers it named as the reason are both now resolved, which is why the
# default moved rather than the prose being tidied:
#
#   - "reagent filtering is unbuilt, so a 12-character floor still admits
#     triethylamine" — `sources/reagents.py` exists and labels every structure
#     `compound` / `reagent` / `trace_fragment`. It labels and never filters,
#     deliberately, so a caller can decide; the floor is no longer the only
#     signal available.
#   - "names are not yet anchored to the patent's compound ids, so the
#     structures artifact cannot join the assay rows" — `sources/anchor.py`
#     anchors them. Re-measured against `out/structures.tsv` at manifest
#     2026-08-08T23:08:18 (137 patents, 64,465 rows): US8952177 183 of 328
#     structures carry a cid; the heading route contributes 1,866 corpus-wide
#     and ALL 1,866 carry one. (This read 1,176 until the numeric-entity,
#     framing-stack and trailing-code fixes landed; a number in a comment is a
#     claim about a version of the code and does not update itself.)
#
# So the artifact is no longer "unfiltered, unanchored structures in something
# that looks finished," which was the honest reason to keep it off. What is
# still true, and is NOT a reason to leave the route dark: the two tracks are
# not joined — `DUMP` holds assay rows and `STRUCTURES` holds compounds, and
# nothing merges them. That is a missing consumer, not a defect in this route.
#
#     SELF_HEAL=0 IUPAC_NAMES=0 python3 -m patentdb3.verify US8952177 --dump
#
# is the $0, structure-free baseline any claim about this route must be stated
# against.
IUPAC_NAMES = os.environ.get("IUPAC_NAMES", "1") == "1"

# Shortest span worth asking OPSIN about. Below this the parser accepts
# fragments (`ethyl`, `2-chloro`) that are substituents, not compounds.
IUPAC_MIN_SEED = 12

# Candidate spans generated per seed. The extractor is deliberately brute
# force — trim left, extend right, balance brackets, let OPSIN reject — and
# this bounds the fan-out. 63.8% of OPSIN's real-world failures on this corpus
# were boundary errors (48.4% unbalanced brackets, 15.4% truncation), which is
# what the variants exist to search.
IUPAC_MAX_VARIANTS = 12


# ── Which identity route runs ────────────────────────────────────────────
# SEPARATE FROM `IUPAC_NAMES`, which says WHETHER identity runs at all. Two
# orthogonal questions need two switches: collapsing them means a route cannot
# be turned off without turning off the feature.
#
#   "cid_first"  the compound ids the assay tables define are the search KEY.
#                For each, find where the patent ASSERTS that id and read the
#                name beside it. The anchor is structural.
#   "prose"      the retired route: find every name in the description, then
#                guess which compound number each belongs to by proximity.
#
# MEASURED over the fixed 20-patent sample, distinct assay cids resolved:
#
#     prose + heading + table   3,424 / 7,322   46.8%
#     cid_first alone           3,668 / 7,322   50.1%
#     table + cid_first         4,409 / 7,322   60.2%
#
# and on the 98 cids where the two disagree, 88 are the prose route shipping a
# STRICT SUBSTRING of the right answer — `1,3,4-oxadiazole` filed as compound
# 458 of US10478424, a bare heterocycle with a valid InChIKey and no way for
# anything downstream to tell it from a real result. That is what the proximity
# guess costs, and it is why "prose" is retired rather than kept as a fallback.
IDENTITY_ROUTE = os.environ.get("IDENTITY_ROUTE", "cid_first")


# ── What counts as a deliverable compound ────────────────────────────────
# ON. A patent numbers three different kinds of thing and only one of them is
# a product: `Example 43` is a compound the patent claims, while
# `Intermediate T-1` and `Step 2` are waypoints in somebody's synthesis. Both
# of the latter state an id and a name in a heading, so both mint a cid, and
# neither is a molecule anyone wants shipped.
#
# The cost is real and was measured before this was turned on: dropping them
# removes ~1,100 joined compounds corpus-wide, and it leaves behind the
# corrupted prose duplicate of the same compound (US10071079 T-1 keeps its
# Boc-stripped description row while the correct heading row goes). Neither is
# a reason to keep intermediates — the deliverable wants finished molecules and
# a smaller set of them is worth more than a larger set that mixes in
# synthesis waypoints — but the leftover duplicate is a SEPARATE defect this
# flag exposes rather than causes, and it still needs fixing.
#
# Read by `sources/iupac_names._heading_texts` (which drops the heading, so the
# name is never minted with that id) and by `sources/anchor.find_cid` (which
# refuses to anchor a prose name to a nearby `Intermediate N`). Both are needed:
# the two routes obtain a cid independently.
FINISHED_ONLY = os.environ.get("FINISHED_ONLY", "1") == "1"


# ── Google Patents ───────────────────────────────────────────────────────
# WHAT GP ACTUALLY PUBLISHES, measured off the live HTML rather than assumed.
# The previous version of this comment said GP "embeds SMILES/InChIKey pairs
# it derived by running structure recognition over the patent's drawings" and
# "numbers compounds positionally". Both are wrong, and acting on either would
# have built the wrong bridge:
#
#   - Each compound block carries ELEVEN microdata fields — name, id, count,
#     smiles, inchi_key, domain, sections, match, similarity, svg_small,
#     svg_large — and `domain` reads "Chemical class". The names include bare
#     scaffolds (`1h-pyrrolo[3,2-b]pyridine`) and the literal string
#     `compounds`. This is ENTITY ANNOTATION over the text, the same model
#     SureChEMBL uses, not per-drawing recognition: US10214537 carries 2,078
#     annotated compounds against 1,342 images.
#   - `id` is the InChIKey itself, or a numeric entity id (`150000001875`).
#     Not positional. `sections` is document-section granularity
#     (description / claims / abstract), which cannot locate a compound row.
#
# So GP is NOT a structure source for this pipeline — it cannot join to a
# patent's own compound number, which is the only key our assay rows have.
#
# It IS an exact IMAGE source. `figure_image_urls` matches our XML's
# `<img file="...">` list one-for-one (.TIF -> .png), and the chain
# cid -> <chemistry> -> image filename -> GP png URL joined 100% on every
# patent tested (US10730877 1,065/1,065, US10870641 480/480, US10214537
# 843/843). Our XML only REFERENCES image files we do not hold; GP hosts them
# rendered, at a URL whose directory component is a content hash we cannot
# construct — which is the whole reason a fetch and a cache are needed at all.
GP_ENABLED = os.environ.get("GP_ENABLED", "0") == "1"

# WHERE THE IMAGE PIXELS COME FROM. The XML is ground truth for WHICH image
# belongs to a compound — it states the file name — but it does not hold the
# image. Two sources serve that file:
#
#   gpatents  one HTTP GET per image, ~3 KB each. Measured coverage of the
#             files the XML names: 99.7% (5,834 of 5,849 across 15 patents,
#             including patents never cached before).
#   uspto     ground truth. The Red Book product PTGRDT ships the TIFs with
#             the XML, but ONLY as weekly TAR files of ~3.7 GB — 1,642 of
#             them, and there is no per-patent split (`PTGRDT-SPLT` 404s).
#             Our 137 patents span ~130 grant weeks, so this is ~480 GB.
#
# DEFAULT IS GOOGLE PATENTS, so a laptop fetches kilobytes instead of
# gigabytes. Set IMAGE_SOURCE=uspto on a machine that can hold the TARs.
# Neither source is trusted blindly: the file name the XML states is the key,
# so a fetched image is checked against the name before it is used.
IMAGE_SOURCE = os.environ.get("IMAGE_SOURCE", "gpatents")

# Where fetched images land. `DECIMER.predict_SMILES` takes a FILE PATH, not
# an array, so images must exist on disk before recognition runs.
IMAGE_DIR = OUTPUT_DIR / "images"

# Per-patent {image filename stem -> https URL}. Small JSON, one per patent —
# NOT the 3.7MB HTML it was extracted from, and NOT v2's `gpatents_cache`,
# which v3 never reads (see CLAUDE.md: the only artifact that legitimately
# crosses the v2/v3 boundary is the grant XML).
GP_IMAGE_DIR = OUTPUT_DIR / "gp_images"


# ── Rule tier ────────────────────────────────────────────────────────────
# THE ONE SWITCH THAT SPENDS MONEY. With it on, every patent runs the repair
# loop after the deterministic reader: find this patent's gaps, apply any rule
# the library already holds ($0), and buy one for the layouts it does not.
#
# Default ON, because a self-heal that has to be remembered is a self-heal that
# does not run — and because the cost is bounded and small. Measured over the
# 137-patent corpus: 157 gaps, 101 already covered by a rule at $0, 56 that
# would buy one at ~$0.002 each on Haiku. `PER_PATENT_LM_CAP` and
# `repair_patent(max_calls=)` bound any single document; a layout is bought once
# and is free on every patent that shares it thereafter.
#
# Turn it OFF to measure the reader alone (`SELF_HEAL=0`), which is the honest
# baseline any claim about what the loop recovers has to be stated against.
# Whichever way it is set, the run RECORDS which mode produced the artifact —
# see `verify.dump`'s manifest. A dump that cannot say whether healing ran is a
# number nobody can attribute later, and that has cost this project days.
SELF_HEAL = os.environ.get("SELF_HEAL", "1") == "1"

# `RULE_GATES_ENFORCE` was here and is gone with the gates it switched. There
# is nothing left to suspend: a rule is now kept or dropped by what it does to
# the patent it was bought for (`repair/outcome.py`), and the only checks that
# still block are contract ones — does the regex compile, can it capture a
# number, does every name and unit appear in the document. Those are not
# opinions, so there was never a reason to switch them off.

# A document with measurement-shaped cells in its tables and nothing usable out
# is a defect, not a quiet zero. Counted on RAW tgroup cells before assembly or
# column classification, because every block-level detector measures a view the
# defect can destroy.
SILENT_PATENT_MIN_CELLS = 10
