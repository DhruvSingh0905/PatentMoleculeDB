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
# On by default because it is the only identity route that is ours: the name
# sits in the patent's own prose, so it carries the patent's own compound
# numbering and needs no bridge. The alternative — Google Patents' embedded
# SMILES — supplied 97.2% of v2's 17,459 compounds, of which 27.6% were still
# keyed by GP's positional id and could never be joined to an assay row.
# DEFAULT OFF, and that is a deliberate staging decision rather than doubt
# about the route. `verify --dump` DOES call `extract_names` and writes
# `config.STRUCTURES` when this is on — the wiring exists and is tested. It is
# off because the pieces underneath it are still being validated one at a time:
# reagent filtering is unbuilt, so a 12-character floor still admits
# `triethylamine`, and names are not yet anchored to the patent's compound ids,
# so the structures artifact cannot join the assay rows.
#
# Shipping it on by default would put unfiltered, unanchored structures into an
# artifact that looks finished. Turn it on to exercise the route:
#
#     IUPAC_NAMES=1 python3 -m patentdb3.verify US8952177 --dump
IUPAC_NAMES = os.environ.get("IUPAC_NAMES", "0") == "1"

# Shortest span worth asking OPSIN about. Below this the parser accepts
# fragments (`ethyl`, `2-chloro`) that are substituents, not compounds.
IUPAC_MIN_SEED = 12

# Candidate spans generated per seed. The extractor is deliberately brute
# force — trim left, extend right, balance brackets, let OPSIN reject — and
# this bounds the fan-out. 63.8% of OPSIN's real-world failures on this corpus
# were boundary errors (48.4% unbalanced brackets, 15.4% truncation), which is
# what the variants exist to search.
IUPAC_MAX_VARIANTS = 12


# ── Google Patents ───────────────────────────────────────────────────────
# OFF, and not yet ported. GP embeds SMILES/InChIKey pairs it derived by
# running structure recognition over the patent's drawings — genuinely useful
# where a compound is drawn but never named, and a liability as a foundation:
# it is a third party's OCR of an image we also hold, and it numbers compounds
# positionally so its output cannot be joined to our assay rows without the
# 250-line bridge that left 27.6% of v2's compounds orphaned.
#
# The flag exists now so the decision is explicit rather than implied by an
# absent file. Nothing reads it yet; when the module lands it must check this.
GP_ENABLED = os.environ.get("GP_ENABLED", "0") == "1"


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
