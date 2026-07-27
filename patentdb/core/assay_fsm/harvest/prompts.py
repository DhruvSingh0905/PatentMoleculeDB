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
  - value: the numeric value as a float. For grade-binned entries,
    CONVERT to a geometric-midpoint numeric value using the LEGEND
    that appears in the same chunk. Three grade conventions are
    common; treat them identically once you know the legend:

    A/B/C letter grades, e.g.
      "A < 100 nM"        → value=50,   qualifier="<"
      "B 100-1000 nM"     → value=316,  qualifier=null  (geomean of 100 & 1000)
      "C ≥ 1000 nM"       → value=1000, qualifier=">"

    +/++/+++/++++ plus grades (Constellation-style; e.g.
    US11292791 encodes CBP/BRD4 IC50 this way). Legend example:
      "++++  ≤ 0.01 μM"       → value=0.005, qualifier="<"
      "+++   0.01–0.1 μM"     → value=0.032, qualifier=null  (geomean of 0.01 & 0.1)
      "++    0.1–1 μM"        → value=0.316, qualifier=null  (geomean of 0.1 & 1)
      "+     1–1000 μM"       → value=31.6,  qualifier=null  (geomean of 1 & 1000)
      "NT"                    → SKIP this tuple entirely (not tested)

    */**/***/**** star grades follow the same rule — read the legend.

    UNKNOWN grade systems: if this chunk uses a different convention
    (color codes, "Active/Inactive", "Low/Med/High", Greek-letter bins,
    etc.) but the chunk contains an explicit numeric legend, apply the
    same geomean-midpoint rule. If there is NO legend visible in this
    chunk for an unknown grade system, emit the tuple with the
    categorical string in ``source_quote`` and value=null —
    downstream code preserves the categorical so the workbook still
    shows it; setting confidence="low" tells Agent 5 to gate it.

    Always convert grades to numeric values WHEN you have a legend.
    Never return "+"/"++"/"A" strings as numeric value; if you can't
    interpret a token, emit value=null with the token in source_quote.
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


# ── Agent 2b — Row-pattern extraction (runs only after Agent 2 found tuples) ──


AGENT2B_ROW_PATTERN = """\
A previous pass extracted compound-activity tuples from this patent text
chunk. We now want the STRUCTURAL PATTERN of the table(s) those tuples
came from, so the rest of the patent's identically-formatted tables can
be extracted with a regex (zero additional LLM cost).

For each distinct row layout in the chunk, return a regex that matches
a single data row. Use Python named groups: `(?P<cid>...)` for the
compound id, `(?P<value0>...)`, `(?P<value1>...)` for each captured
column. Match data rows ONLY — never match the header row.

ALSO return `header_text`: the table's header row VERBATIM — the column
titles the data rows sit beneath (e.g. "Ex. No. CBP IC50 (μM) BRD4 IC50
(μM)"). This is the pattern's anchor: it lets the pipeline re-fire the
regex ONLY on tables that carry this exact header, so the labels are never
applied to a different patent's lookalike table. Include the assay-bearing
column titles in left-to-right order; you may omit decorative columns.

If the table uses GRADE TOKENS instead of numbers, capture the token
as-is — the pipeline handles the conversion. Recognise these grade
conventions and write the value group accordingly:
  - A/B/C/D/E letter bins         → `(?P<value0>[A-E])`
  - +/++/+++/++++ plus bins       → `(?P<value0>\\\\+{{1,5}})`
  - */**/***/**** star bins       → `(?P<value0>\\\\*{{1,5}})`
  - Active/Inactive               → `(?P<value0>Active|Inactive)`
  - colour codes (red/green/...)  → `(?P<value0>Red|Yellow|Green|...)`
  - NT/ND/N\\.A\\.                → optional skip-marker; include in alt
For unfamiliar grade syntaxes, write a character-class regex that
matches the observed tokens verbatim. The pipeline learns from your
regex — if 3+ patents share a grade convention the library auto-
promotes the pattern so it fires before the LLM does on future runs.

Patent text chunk:
<<<
{chunk_text}
>>>

Sample tuples we extracted (for context — match a row that produced
these tuples):
{tuples_summary}

Return strictly a JSON array (one entry per distinct row layout):
[
  {{
    "regex": "(?P<cid>Z\\\\d+)\\\\s+(?P<value0>[A-E])\\\\s+(?P<value1>[A-E])\\\\s+(?P<value2>[\\\\d.]+)",
    "column_assays": ["cAMP IC50 A2B", "cAMP IC50 A2A", "Ratio A2A/A2B"],
    "header_text": "Cpd. cAMP IC50 A2B cAMP IC50 A2A Ratio A2A/A2B",
    "example_match": "Z2 A D 31.36"
  }}
]

If the chunk had no structured tabular data (e.g., it's just prose
mentioning one compound + value), return [] — no patterns to register.
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
