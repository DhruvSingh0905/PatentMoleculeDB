"""Configuration for patent molecule extraction pipeline."""

import os
from pathlib import Path

# API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Load from .env if not in environment. Search both v2-package-local
# (patentdb/.env) and the repo root (where the project's
# canonical .env actually lives — same one v1 uses). Fall through if
# absent so unit tests / CI can run without a key.
if not ANTHROPIC_API_KEY:
    _candidate_env_paths = [
        Path(__file__).parent.parent / ".env",        # patentdb/.env
        Path(__file__).parent.parent.parent / ".env", # repo root (canonical location)
    ]
    for _env_path in _candidate_env_paths:
        if not _env_path.exists():
            continue
        for line in _env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                ANTHROPIC_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
        if ANTHROPIC_API_KEY:
            break
MODEL_OPUS = "claude-opus-4-6"
MODEL_SONNET = "claude-sonnet-4-6"
DEFAULT_MODEL = MODEL_SONNET  # Sonnet for $200 proof phase; upgrade to Opus after budget increase

# Paths
# PACKAGE_ROOT = patentdb/          (source code lives here)
# REPO_ROOT    = project root.
# DATA_DIR     = per-patent source dirs, one per patent ID:
#                  data/patents/{pid}/{pid}.pdf
#                  data/patents/{pid}/all_pages/page_*.md   (MinerU output)
#                Every consumer builds its path as `DATA_DIR / patent_id / ...`,
#                so this one constant relocates the whole corpus.
# OUTPUT_DIR   = output_v2/. Kept at this name deliberately — renaming it would
#                orphan every cached LLM response and GP scrape under it.
#                Reference dumps (BindingDB_All.tsv) sit beside it in data/.
PACKAGE_ROOT = Path(__file__).parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = REPO_ROOT / "data" / "patents"
OUTPUT_DIR = REPO_ROOT / "output_v2"
IMAGES_DIR = OUTPUT_DIR / "images"
LOGS_DIR = OUTPUT_DIR / "logs"

# The BindingDB subset every benchmark scores against.
#
# This was hardcoded to `output/bindingdb/our_patents.tsv` in two modules —
# and `output/` is FROZEN v1, read-only. So the reference could never grow
# with the corpus: adding ten patents left them with zero reference rows, and
# `reference_bench` skips a patent whose reference set is empty (`if not
# ref_all: continue`). A new batch therefore scored not "0% coverage" but
# nothing at all, silently, while the corpus totals looked unchanged.
#
# Prefers a writable copy under OUTPUT_DIR and falls back to the frozen one,
# so an untouched checkout behaves exactly as before. Extend it with an awk
# pass over `data/BindingDB_All.tsv` filtering column 22 (`Patent Number`).
BDB_REFERENCE_TSV = Path(os.environ.get("BDB_REFERENCE_TSV", "")) if os.environ.get(
    "BDB_REFERENCE_TSV") else (
    OUTPUT_DIR / "bindingdb" / "our_patents.tsv"
    if (OUTPUT_DIR / "bindingdb" / "our_patents.tsv").exists()
    else REPO_ROOT / "output" / "bindingdb" / "our_patents.tsv")

# ── HARVEST burst tier (coverage safety net) ──────────────────────
# When True, the pipeline runs the gap detector after the cheap tier
# completes; on signal trip, fires the HARVEST 5-agent burst over the
# full patent text. Default ON. Disable via `HARVEST_BURST=0` for
# regression debugging or cost-controlled re-runs.
HARVEST_BURST_ENABLED = os.environ.get("HARVEST_BURST", "1") == "1"

# MINERU_OCR_ENABLED lived here and is gone with the code it gated.
#
# It defaulted off already, but "off by default" still left an OCR engine one
# env var away from a production run. `process_patent._resolve_text_sources`
# shelled out to MinerU whenever a patent had no `page_*.md`; a stack sample
# caught 3,317 of 3,335 samples blocked in `subprocess.communicate`, ~5-10 min
# per patent, GBs of model weights, to produce a tier-3 source ranked BELOW
# the Google Patents HTML already on disk. The runner also wrote to
# `REPO_ROOT/{pid}/all_pages` while every reader looked in
# `DATA_DIR/{pid}/all_pages`, so enabling it produced pages nothing read.
#
# The pipeline now neither generates nor reads OCR. Where a patent has no
# text, `_resolve_text_sources` PULLS: Google Patents first, then the
# patent's own USPTO grant XML. Measured over the corpus: 22/22 patents have
# both, and 0 had markdown as their only source.

# Strategy 5's paid IUPAC->SMILES recovery batch (process_patent, the
# `opsin_failures` arm). DEFAULT OFF.
#
# Measured corpus-wide yield: at most 57 distinct OPSIN-failing names ever
# produced a structure through it. Against that it cost ~$0.23/patent and,
# worse, minutes of wall-clock per patent: it calls `call_claude_text_batch`
# unconditionally, so a run submits TWO requests to the async Message Batches
# API and polls them against a 1800s cap. Batching is ~50% cheaper per token
# and pays for itself over hundreds of requests; over two it is strictly worse
# than a synchronous call that would return in seconds.
#
# It is also the only live caller of the batch API once HARVEST is off, and it
# never consulted `LLM_BATCH_ENABLED` — the flag is checked in exactly one
# place, `harvest/orchestrator.py`. So with the burst off, LLM_BATCH controlled
# nothing and this path batched regardless.
STRATEGY5_LLM_ENABLED = os.environ.get("STRATEGY5_LLM", "0") == "1"

# The broad IUPAC burst (`harvest/iupac_orchestrator.iupac_burst`, called from
# Strategy 5 with force=True). DEFAULT OFF.
#
# Distinct from the HARVEST assay burst and from Strategy 5's OPSIN-failure
# batch despite the shared naming. It uses the synchronous cached
# `call_claude_text`, so it issues no batches — which is why a run with every
# other tier disabled still showed $0.203/patent and zero batch heartbeats.
#
# It is the one LLM path that DOES honour `patent_lm_exceeded`, checking it in
# four places, which is why it stops dead on the $0.20 cap instead of running
# to the $2.13 the assay burst reached uncapped.
#
# Off by default so the pipeline has a deterministic $0 baseline. Every tier is
# then measured as a marginal contribution against that baseline rather than
# against another tier's output. IUPAC_BURST=1 restores it.
IUPAC_BURST_ENABLED = os.environ.get("IUPAC_BURST", "0") == "1"

# The remaining ad-hoc LLM callers, none of which had a flag. DEFAULT OFF.
#
#   route_classifier.py       classify_route()
#   cid_prefix_discovery.py   compound-id prefix inference
#   impossible_fragment_retry.py
#   adaptive_extraction_cache.py
#
# Found while establishing a $0 baseline: with HARVEST, the cascade, Strategy 5
# and the IUPAC burst all disabled, a run still cost $0.21/patent and hit the
# cap. These four are why.
#
# `classify_route` is the clearest case. CLAUDE.md states the classifier is
# INFORMATIONAL ONLY — `process_patent` runs the text branch for every patent
# regardless of the label, because the old gate zeroed out real data. So the
# call buys a string that changes no behaviour.
LLM_RECOVERY_ENABLED = os.environ.get("LLM_RECOVERY", "0") == "1"

# ── Message Batches API (50% off, async) ──────────────────────────
# When True, HARVEST burst collects all Agent 1 / Agent 2 / Agent 2b
# prompts for a patent up front, submits them as one Message Batch,
# polls until completion, then processes results. Costs roughly half
# the per-token rate (Anthropic batches discount) and runs requests
# concurrently server-side. Disable with `LLM_BATCH=0` for the
# sequential path (faster debug feedback, full price).
LLM_BATCH_ENABLED = os.environ.get("LLM_BATCH", "1") == "1"

# ── HARVEST skip-when-library-covers gate ─────────────────────────
# When True, the pipeline skips the HARVEST burst tier if the pre-applied
# pattern library already extracted enough rows AND prior runs' HARVEST
# hit-rate is below the threshold. This prevents wasted LLM spend on
# patents the library already covers — US11292791's last run had a 3 %
# HARVEST hit-rate against 8,693 pre-library rows; the 75 batched LLM
# calls contributed essentially nothing. With the gate active those
# calls would be skipped (~$0.50 per patent saved).
#
# Defaults are conservative:
#   - need ≥ 500 pre-library rows AND
#   - patent has its own recently-seen pattern (avoid skipping novel
#     patents we know nothing about)
# Tune with env vars HARVEST_SKIP_MIN_PRELIB_ROWS / HARVEST_SKIP_ENABLED.
HARVEST_SKIP_ENABLED = os.environ.get("HARVEST_SKIP", "1") == "1"
HARVEST_SKIP_MIN_PRELIB_ROWS = int(os.environ.get("HARVEST_SKIP_MIN_PRELIB_ROWS", "500"))

# ── Substituent-table LLM fallback ────────────────────────────────
# The substituent-table scan is symbolic-first ($0). When the regex
# undershoots on a Markush-likely patent (TABLE-format definitions the
# prose regex can't read), an LLM agent fires on the top-scoring
# substituent-shaped chunks ONLY. Tightly capped — the LLM never reads a
# chunk without R-label shape, and never fires when symbolic already
# yields an enumerable table. Tune with env SUBSTITUENT_LLM / *_MAX_CHUNKS.
# Default OFF — production substituent scan stays $0 (symbolic-only); the
# LLM arm is opt-in (SUBSTITUENT_LLM=1), so no surprise per-patent cost.
SUBSTITUENT_LLM_ENABLED = os.environ.get("SUBSTITUENT_LLM", "0") == "1"
SUBSTITUENT_LLM_MAX_CHUNKS = int(os.environ.get("SUBSTITUENT_LLM_MAX_CHUNKS", "2"))

# ── Gap-triggered repair loop ─────────────────────────────────────
# Asks a model for an extraction RULE (not data) when a table largely fails
# to parse, validates it, and persists it by layout fingerprint so the cost
# amortises across every patent sharing that shape. Measured at ~$0.001-0.005
# per NEW layout and $0 thereafter, against ~$8.81/patent for the HARVEST
# burst it is intended to replace. Disable with REPAIR=0.
REPAIR_ENABLED = os.environ.get("REPAIR", "1") == "1"
REPAIR_MAX_CALLS_PER_PATENT = int(os.environ.get("REPAIR_MAX_CALLS", "4"))
# A verified parser patch writes itself into the tree. ON by default.
#
# It used to be off, so a human had to flip a switch before anything healed —
# which is not self-healing, it is a queue with extra steps. The safety story is
# reversibility, not permission: every patch, applied or declined, is journaled
# with its full before/after source and its per-patent coverage delta, so any
# state is recoverable with `parser_health --revert`. See repair/parser_repair.
#
# Set 0 for a dry run that journals the proposal without touching the tree.
PARSER_REPAIR_APPLY = os.environ.get("PARSER_REPAIR_APPLY", "1") == "1"

# Where the audit trail lives. Not in the package — it is produced, not needed
# to run — but it IS the revert mechanism, so it is never cleared as a cache.
PARSER_REPAIR_JOURNAL = OUTPUT_DIR / "parser_repair_journal.jsonl"

# Does `rules.validate()` VETO a proposal, or merely describe it?
#
# Suspended by default. No fixed validator can anticipate the layouts patents
# actually use, and this one has been wrong at least as often as it has been
# right: it scored a correct column_map 0/23 because `_CID_PAT` did not know
# chemical-name ids, rejected a parser patch identical to the correct fix on a
# baseline measured with the broken reader, and floored a working rule at 49%.
# Each of those cost real records and reported the input as the problem.
#
# The verdict still RUNS and is still recorded — it is useful evidence, just not
# authority. Adoption is instead made reversible: every rule is journaled with
# the coverage it moved, so a bad one is visible and revocable rather than
# prevented. Same trade the parser-repair tier already makes.
#
# Set 1 to restore vetoes.
RULE_GATES_ENFORCE = os.environ.get("RULE_GATES_ENFORCE", "0") == "1"
RULE_JOURNAL = OUTPUT_DIR / "rule_adoption_journal.jsonl"

# Does a patent that yields NOTHING escalate to the code-patch tier by itself?
#
# It did not, and that made "self-healing" a description of one tier out of
# three. `process_patent` called the rule tier and stopped; `repair_capabilities`
# — the only tier that can fix a layout no rule kind can express — was reachable
# solely from two eval CLIs. So US10266548 (197 reference compounds) produced a
# correct diagnosis naming its own failing table, and that diagnosis was
# garbage-collected: `repair_report` is read for ONE field in the orchestrator
# and dropped, and it reached no output file at all.
#
# The same trade as PARSER_REPAIR_APPLY, one tier up: a fix that waits on a
# human is a queue, and the gap costs records while it waits. Safety is the
# journal and the revert, not permission.
# Snapshot freezing — PAUSED. `repair/snapshot.freeze` pins a patent's answer
# when it is processed so later patches cannot retroactively change it. That is
# the right behaviour for a settled corpus and the wrong one while the corpus is
# being repaired: 97 snapshots were found pinning answers at a single journal
# head, 77 of them for patents with no shipped artifact at all, and every fix
# made afterwards was invisible on the patents it was meant to help. Re-enable
# when the extraction is stable and the numbers are trusted.
# The IUPAC name-REPAIR stages, OFF by default. These take a name our own
# extractor produced, notice OPSIN cannot read it, and pay a model to rewrite
# it. Measured over 22 patents / 17,359 resolved structures:
#
#   86.1% of the records these stages are credited with reproduce for FREE
#   when their stored name is re-run through OPSIN. Only 87 structures — 0.50%
#   — could not have come from a free path, and 37 of those 87 are our own
#   damage: prose contamination, truncation, mojibake.
#
#   Controlled on US10544143, the patent that wants 213 paid calls: blocking
#   the cascade gave 49 structures, serving it from cache gave 50. The entire
#   cascade bought ONE molecule, and four of its five records were relabels of
#   molecules the free path had already produced.
#
#   Classified by why OPSIN failed on those 182 calls: 48.4% unbalanced
#   brackets, 25.3% synthesis prose, 15.4% truncated mid-name, 11% "well
#   formed" (header contamination and mojibake). Essentially none is hard
#   nomenclature — we were paying a model to repair our own parser's output.
#
# NOT gated by this: the broad IUPAC burst, which earns its cost. 3,101
# records (17.9%) have no other route to a cid->name pairing and US11566007
# goes 80 -> 0 without it. Nor the repair loop, which is the self-healing tier
# and runs Haiku at ~$0.002 per new layout.
LLM_NAME_REPAIR_ENABLED = os.environ.get("LLM_NAME_REPAIR", "0") == "1"

SNAPSHOT_FREEZE_ENABLED = os.environ.get("SNAPSHOT_FREEZE", "0") == "1"

REPAIR_AUTOHEAL = os.environ.get("REPAIR_AUTOHEAL", "1") == "1"

# Distinct CAPABILITIES bought per process, not per patent. A capability patch
# is keyed by layout fingerprint and cached by PATCH_EPOCH, so a corpus run buys
# each shape once — but `baseline_counts()` rescans every patent on each call,
# so an unbounded auto-fire would cost one full corpus scan per failing patent.
# CLAUDE.md measures three capability gaps in the whole corpus; two per run is a
# ceiling on damage, not a target.
AUTOHEAL_MAX_PER_RUN = int(os.environ.get("AUTOHEAL_MAX_PER_RUN", "40"))

# Every escalation the loop raises, appended. The rule tier already journals what
# it ADOPTED; nothing recorded what it could not fix, which is the half a human
# needs. Not a cache — it is the queue, and clearing it loses the record of
# every unsolved layout.
ESCALATION_JOURNAL = OUTPUT_DIR / "escalation_journal.jsonl"

# How many measurement-shaped cells a patent must hold before "it yielded
# nothing" is worth reporting. ONE constant, because it was two: `loop.py` and
# `capability._gaps_from_a_silent_patent` each carried their own 20, and moving
# only the first left US9695181 (18 cells) firing the invariant and then
# collecting no gap — it raised and could not be acted on.
#
# 10 is measured, not chosen: across 103 patents exactly one silent patent sits
# below 20, and every threshold from 1 to 18 fires on the identical set.
SILENT_PATENT_MIN_CELLS = int(os.environ.get("SILENT_PATENT_MIN_CELLS", "10"))

# Cost — global budget across all patents
COST_THRESHOLDS = [50, 100, 150, 200]
COST_CEILING = 200

MODEL_HAIKU = "claude-haiku-4-5-20251001"

# Pricing (per million tokens). A model missing from this table gets billed at
# Opus rates by `cost_tracker.compute_cost`, which silently overstated the
# repair loop's cost by ~15x until Haiku was added — keep new models in here.
PRICING = {
    MODEL_OPUS: {"input": 15.0, "output": 75.0},
    MODEL_SONNET: {"input": 3.0, "output": 15.0},
    MODEL_HAIKU: {"input": 1.0, "output": 5.0},
}

# Memory mode for the upstream-OCR wrappers (DECIMER-Seg, PaddleOCR text-det,
# PP-Structure cell-det). When True, each wrapper's `unload()` actually frees
# its model singletons + runs gc; subsequent calls re-load from scratch.
# When False, models stay hot in RAM after first load and `unload()` is a no-op.
#
# - Local laptop (16-32 GB): leave True. Sequential phases keep peak under
#   one model's footprint (~6 GB for PaddleOCR is the worst case).
# - Cloud / GPU box (64+ GB): set to False to maximize throughput across
#   many pages — first-load cost amortizes over the batch.
#
# Override via env var at runtime: `EXTRACTION_UNLOAD_MODELS=0` for cloud,
# `=1` for local. Default favors safety on the laptop.
UNLOAD_MODELS_AFTER_USE = (
    os.environ.get("EXTRACTION_UNLOAD_MODELS", "1") not in ("0", "false", "False", "")
)

# Per-patent LM spend cap — guards Stages 3a/3b + synthesis-block route
PER_PATENT_LM_CAP = 0.20  # USD
# Hard alarm threshold (NOT a stopper — record() logs ONE loud WARNING per
# patent when crossed). The soft cap above is enforced by per-path guards
# (patent_lm_exceeded checks); this alarm is the backstop that makes any
# *unguarded* path visible instead of silently burning budget. Set well
# above the soft cap so normal multi-strategy spend doesn't trip it.
PER_PATENT_LM_HARD_CAP = 1.00  # USD

# Per-patent image-pipeline cost cap (Sonnet/Opus Vision + Sonnet layout calls)
# Separate from LM cap so a Tier-B patent can spend on diagrams without burning LM budget
PER_PATENT_IMAGE_CAP = 0.50  # USD

# Per-patent LLM-realigner cap — the always-fire assay-table extractor. Bigger
# than the LM cap because one large multi-column table can legitimately need
# 15-20 chunked LLM calls (US11254686's 645-row table ≈ $1). Separate bucket so
# it doesn't starve IUPAC name recovery, but still bounded so a runaway can't
# eat the global budget. The global COST_CEILING is the final backstop.
PER_PATENT_REALIGN_CAP = 1.50  # USD

# Drug-likeness filter — read by `smiles_utils.passes_lipinski`.
LIPINSKI_MW_MAX = 500
LIPINSKI_LOGP_MAX = 5
LIPINSKI_HBD_MAX = 5
LIPINSKI_HBA_MAX = 10

# API retry
MAX_RETRIES = 3
RETRY_MIN_WAIT = 4
RETRY_MAX_WAIT = 60

# Ensure output directories exist
for d in [OUTPUT_DIR, IMAGES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
