"""HARVEST agent prompts.

Reimplementation of the 5-agent HARVEST architecture described in
bioRxiv 10.64898/2026.03.15.711910 (Mar 2026). HARVEST has no public
code release as of plan-write time, so these prompts are derived from
the paper's published methodology. Agents 1, 2, 5 are in scope for
this pipeline; Agents 3 (compound resolution) and 4 (protein
resolution) are deferred — we already cover them via `iupac_to_smiles`
+ `text_index`.

Each prompt is JSON-schema-structured and patent-agnostic. No target
names, compound IDs, or patent-specific tokens hardcoded.
"""
from __future__ import annotations


# ── Agent 1 — Target Extraction ─────────────────────────────────


AGENT1_TARGET_EXTRACTION = """\
You are reading a chunk of a pharmaceutical patent. Identify EVERY
biological target whose activity is measured in this chunk.

For each target, return a record describing:
  - target_name: the protein / enzyme / receptor / cell line being assayed.
    Use the patent's exact phrasing where possible.
  - organism: human / mouse / rat / etc., or "unspecified".
  - isoform: subtype info if mentioned (e.g., "alpha", "δ", "wild type"),
    else "".
  - assay_type: one of "binding" / "enzyme_inhibition" / "cellular_potency"
    / "cytotoxicity" / "antiviral" / "other". Pick the most specific.
  - units: the unit the patent reports activity in for this target
    (nM / μM / mM / pM / % / unitless). "" if you genuinely can't tell.
  - conditions: any short condition descriptor mentioned in the same
    sentence (e.g., "ATP-competitive", "FRET-based", "whole-blood
    stimulated with A23187"), else "".

Rules:
  - Only emit a target if the patent describes a measurement against
    it (not just a mention in claims/background).
  - Same target with different organism/isoform = SEPARATE records.
  - If the patent uses a letter-grade scale (A/B/C bins) for a target,
    still emit the target record.
  - Don't invent. If you're unsure, omit the target.

Patent text chunk:
<<<
{chunk_text}
>>>

Return a JSON array (no prose, no markdown):
[
  {{
    "target_name": "...",
    "organism": "...",
    "isoform": "...",
    "assay_type": "...",
    "units": "...",
    "conditions": "..."
  }}
]
"""


# ── Agent 2 — Activity Extraction ───────────────────────────────


AGENT2_ACTIVITY_EXTRACTION = """\
You are extracting compound-level activity measurements from a
pharmaceutical patent chunk. Agent 1 has already identified the
biological targets present in this chunk:

{targets_summary}

Extract every compound activity tuple. Each tuple is one
(compound, assay, value) triple.

Format every tuple as:
  - compound_id: the patent's verbatim compound identifier (e.g., "5",
    "5A", "Compound 12", "Cpd. No. 3", "P17"). Strip prefixes like
    "Compound" / "Cpd." in your output — keep just the bare id.
  - assay_name: pick the matching target_name from Agent 1's list.
    If unsure, use a short canonical assay token (IC50/EC50/Ki/Kd/CC50
    + the target shortname).
  - value: the numeric value as a float. For letter-grade entries
    (A/B/C), CONVERT using the legend present in the chunk:
      * "A < 100 nM" → emit value 50 (geometric midpoint), with
        qualifier="<", confidence="medium" (Agent 5 will downgrade)
      * "B 100-1000 nM" → emit value 316 (geomean), no qualifier
      * "C ≥ 1000 nM" → emit value 1000, qualifier=">"
    Always convert grades to numeric values; don't return letter strings.
  - unit: "μM" / "nM" / "mM" / "pM" / "%" / "" — match the patent.
  - qualifier: ">", "<", "~", "≈", "≤", "≥", or null. Use null if
    the value is exact.
  - n_runs: integer if the patent reports replicate counts (e.g.,
    "0.0038 (8)" → n_runs=8), else null.
  - source_quote: the EXACT contiguous substring from the input
    (≤80 chars) that justified your extraction. Critical: this string
    must appear character-for-character in the input.
  - source_offset: integer character offset where source_quote begins
    in the input.
  - confidence: "high" / "medium" / "low" — your own confidence in
    the extraction. Letter-grade conversions, ambiguous compound ids,
    or interpolated values default to "medium" or "low".

Rules:
  - Don't extract from claims or background prose UNLESS that prose
    contains a specific compound + value (e.g., "Compound 5 had an
    IC50 of 12 nM"). Generic claim language ("compounds may exhibit
    IC50 < 1000 nM") is NOT a measurement.
  - Skip rows where the value is "nt"/"n.t."/"nd"/"—" (not tested).
    Don't emit null-value tuples.
  - If a compound has multiple assays in this chunk, emit one tuple
    per assay.
  - Extract EVERY tuple you can substantiate, including from prose,
    flat-text tables, letter-grade tables, and structured tables.

Patent text chunk:
<<<
{chunk_text}
>>>

Return a JSON array only:
[
  {{
    "compound_id": "...",
    "assay_name": "...",
    "value": <number>,
    "unit": "...",
    "qualifier": null or "...",
    "n_runs": null or <int>,
    "source_quote": "...",
    "source_offset": <int>,
    "confidence": "high|medium|low"
  }}
]
"""


# ── Agent 5 — Validation ────────────────────────────────────────


AGENT5_VALIDATION = """\
You are validating a list of activity tuples extracted from a
pharmaceutical patent. For each tuple, check:

  1. Unit-conversion sanity: an IC50/EC50/Ki/Kd of 100 mM is
     suspicious (cytotoxic territory); 0.0001 pM is suspicious
     (impossible binding affinity). Flag values outside the normal
     pharma window (1 pM to 100 μM for IC50/Ki/Kd; 1 pM to 1 mM for
     CC50/EC50).
  2. Source-quote match: does source_quote appear verbatim in the
     input chunk? If not, the extraction is unreliable.
  3. Target match: assay_name should be plausibly related to one of
     the targets Agent 1 identified.
  4. Internal consistency: if the same compound has two values from
     the same assay AND they differ by more than 2x, flag the pair.

Output: same list of tuples, but with each one tagged:
  - confidence: "high" / "medium" / "low" (downgrade as needed)
  - validation_reason: short phrase if you downgraded, else "".

Don't drop tuples — downstream consumers filter by confidence.

Targets identified by Agent 1:
{targets_summary}

Tuples to validate:
{tuples_json}

Patent text chunk (for source_quote verification):
<<<
{chunk_text}
>>>

Return a JSON array of the SAME LENGTH as the input list, in the
SAME ORDER:
[
  {{
    "compound_id": "...",
    "assay_name": "...",
    "value": <number>,
    "unit": "...",
    "qualifier": null or "...",
    "n_runs": null or <int>,
    "source_quote": "...",
    "source_offset": <int>,
    "confidence": "high|medium|low",
    "validation_reason": "..."
  }}
]
"""
