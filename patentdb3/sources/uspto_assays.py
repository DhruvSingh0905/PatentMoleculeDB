"""Assay extraction from USPTO CALS tables — deterministic, no LLM.

The lesson that shapes this module: **most numbers in a patent table are not
assay values.** A naive "grab every number next to a compound id" sweep over
US10544143 produced 6,063 pairs, and almost all of them were molecular weights,
LC-MS m/z and retention times. US11292791's third column is
`1H NMR (CD3OD, 400 MHz) δ` — a numeric goldmine of pure noise. The existing
pipeline needs an LLM and a corroboration validator largely to undo damage of
exactly this kind.

With real table markup that damage is avoidable at the source, because the
patent already tells us what each column is. So the pipeline here is:

    merge multi-row headers  →  classify each column  →  only read assay columns

Everything downstream depends on step two. A column we cannot confidently
classify is skipped, not guessed: a missing assay is recoverable, a molecular
weight recorded as an IC50 is a lie the database cannot detect later.

Layout facts this handles, all observed in real grants:
  - headers stacked across up to 7 `thead` rows, merged column-wise
    (`Compound`/`No.` → "Compound No.")
  - header cells spanning columns via `namest`/`nameend`
  - continuation tables with no header of their own, inheriting the previous
    tgroup's header (US9718825, US10544143)
  - compound ids as `12`, `I-2300`, `Z1`, `A1`
  - letter-grade potency bins (`A`-`E`) instead of numbers (US11254686)
  - blank spacer rows interleaved between data rows
  - run counts as `(8)` in the cell or in a following column
"""
from __future__ import annotations

import functools
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, replace as _dc_replace
from functools import lru_cache

from ..core import config
from ..core.models import AssayRow
from .uspto_xml import Table

logger = logging.getLogger(__name__)

# ── column kinds ──────────────────────────────────────────────────
CID = "cid"
ASSAY = "assay"
NRUNS = "nruns"
NMR = "nmr"
MS = "ms"
MW = "mw"
RT = "rt"
STRUCTURE = "structure"
SUBSTITUENT = "substituent"
UNKNOWN = "unknown"

# Columns we must never read a value from. Kept explicit rather than implied
# by "not an assay", so that adding a new kind fails closed.
NON_ASSAY = {NMR, MS, MW, RT, STRUCTURE, SUBSTITUENT, CID, NRUNS, UNKNOWN}

_UNIT_PAT = re.compile(
    r"\(\s*(n[mM]|[μuµ][mM]|m[mM]|p[mM]|nmol|µg/mL|ug/mL|%|percent|mol/[lL]|[μuµ]mol/[lL]|nmol/[lL])\s*\)|"
    # BARE `M` IS MOLAR, and only ever inside brackets. US10570116 heads its
    # columns `IC50 [M] TNKS1 ELISA` and the pattern could not see it, so the
    # column yielded no unit at all — and `infer_units_from_description` then
    # filled one from the document's PRIOR-ART prose, which cites someone
    # else's `IC50 (TNKS1) = 2 nM`. `5.7e-10 M` is 0.57 nM and shipped as
    # `5.7e-10 nM`: wrong by a factor of 1e9, on 106 records, with every count
    # healthy. Bracketed only — a bare `M` anywhere would match a name.
    r"[\(\[]\s*(M)\s*[\)\]]|"  
    r"\b(nM|µM|μM|uM|mM|pM|mol/L|mol/l)\b|"  
    # Spelled-out units. Patents routinely state the unit in a table legend as
    # a word — "IC50's are micromolar." — rather than as a symbol in the
    # header. Missing these left correctly-read values unitless, which is the
    # difference between a usable measurement and an unusable number.
    r"\b(micromolar|nanomolar|millimolar|picomolar)\b", re.I)

_SPELLED_UNIT = {
    "micromolar": "uM", "nanomolar": "nM",
    "millimolar": "mM", "picomolar": "pM",
}

# A cell that is a measurement: optional qualifier, a number, optional unit,
# optional parenthesised run count.
# NOTE the integer part is `\d+`, not `\d{1,3}`. Written as `\d{1,3}(?:,\d{3})*`
# to allow thousands separators, it silently rejected every un-separated value
# of 1000 or more — `1511.5`, `8618`, `1412` all failed to parse and were
# dropped without a trace. Patents report nM potencies above 1000 constantly,
# so this quietly discarded the entire weak-activity tail of the corpus.
# The trailing parenthetical is captured loosely as `paren` and interpreted in
# `parse_value`, because it is not always a run count. Patents also put a
# footnote marker there — `24 (*)`, `0.83 (A)` — and requiring digits made those
# cells unparseable, so whole rows were dropped with their values sitting in
# plain sight.
# The exponent group is not decoration: US9765018 reports its most potent
# compounds as `6.49E−03`, and without it those nine cells — the sub-nanomolar
# tail, the compounds anyone would care about first — parsed as nothing at all.
# The minus may be ASCII `-`, U+2212 MINUS or an en dash; typesetting picks.
_EXP = r"(?:\s*[Ee]\s*[-−–+]?\s*\d{1,3})?"
_VALUE_PAT = re.compile(
    r"^\s*(?P<qual>[<>~\u2248\u2265\u2264\u2266\u2267\u2a7e\u2a7d]|>=|<=)?\s*"
    r"(?P<num>(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)" + _EXP + r")"
    r"\s*(?P<unit>nM|\u00b5M|\u03bcM|uM|mM|pM|%)?"
    r"(?:\s*\(\s*(?P<paren>[^)]{1,12}?)\s*\))?\s*"
    # Trailing footnote markers (* ** *** # ## ###) appear in patent tables to
    # encode replicate count or significance (e.g. '572**', '1194**').
    # They must not cause the cell to be rejected — the measurement is valid.
    # US9695181 writes '572**' and '1194**' for single-experiment IC50 values.
    r"[*#]*\s*$")
# `1680 ± 150 (n = 4)` / `0.00275 ± 0.00046, n = 3`. The value is the mean and
# the second number is its spread, which we do not store — but rejecting the
# whole cell loses the measurement too. 147 assay cells across four patents,
# and on US11649247 it lost the potency while keeping the `>20.0` ceiling from
# the neighbouring column, so the compound read as inactive when the patent
# reports it at 2.75 nM.
#
# RESTORED BY HAND. A capability patch that correctly taught this pattern to
# accept trailing footnote markers rewrote the whole assignment and dropped
# this block with it. Coverage rose, the reasoning vanished, and no test could
# see the difference — which is why the standing rule is to read every applied
# patch even when every automated check is green. Note also that a net comment
# COUNT is not evidence of preservation: this file gained eight comment lines
# in the same patch that deleted these six.
_MEAN_SD = re.compile(
    r"^\s*(?P<qual>[<>~\u2248\u2265\u2264])?\s*(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?:\u00b1|\+/-|\+-)\s*\d+(?:,\d{3})*(?:\.\d+)?\s*"
    r"(?P<unit>nM|\u00b5M|\u03bcM|uM|mM|pM|%)?"
    r"[\s,;]*(?:\(?\s*n\s*=\s*(?P<n>\d{1,3})\s*\)?)?[\s,;]*"
    # Trailing footnote markers (* ** *** # ## ###) are common in patent tables
    # to encode replicate count or significance. They must not cause the cell to
    # be rejected — the measurement is still valid. US9695181 writes
    # '3.0 ± 1.0*' and '2.2 ± 0.9*'; without this suffix the whole cell fails.
    r"[*#]*\s*$", re.I)

_NRUNS_ONLY = re.compile(r"^\s*\(\s*(\d{1,3})\s*\)\s*$")
# EVERY NAME THIS MODULE USES WHEN IT DOES NOT KNOW WHAT WAS MEASURED.
#
# One set, because the detector that flags an unnamed column has to agree with
# the code that mints the name — and keying the flag on WHICH CODE PATH set it
# is how a flag silently covers only some of them. `label_source` marked the
# column classifier's placeholders and missed `assay (binned)`, minted by the
# inverted-table path, so 818 records went unflagged. The name is the signal;
# where it came from is not.
PLACEHOLDER_ASSAY_NAMES = frozenset({
    "assay (binned)", "unnamed assay", "unnamed assay (letter bin)",
})


def is_placeholder_name(name: str | None) -> bool:
    """Did we emit this name because we could not read one?"""
    return (name or "").strip().lower() in {
        n.lower() for n in PLACEHOLDER_ASSAY_NAMES}


_LETTER_BIN = re.compile(r"^\s*([A-E])\s*$")
# Any single letter, whether or not it is a grade. A GRADE ALPHABET IS CLOSED:
# the legend defines A-E and the column uses nothing else. A column that also
# holds `G`, `N` and `W` is enumerating something — US9221791's `HPLC Method`
# column runs A through W, and 80% of it lands inside A-E purely because `A` is
# the most-used method. On a 60% threshold that is an assay column, and it
# minted 82 records of method letters presented as potency grades.
_ANY_LETTER = re.compile(r"^\s*([A-Za-z])\s*$")
# How much of a grade column may sit outside the grade alphabet. Not zero: a
# stray `N` for "not tested" should not disqualify a real scale.
_STRAY_LETTER_MAX = 0.05

# A compound id may be written bare (`12`, `I-2300`, `Z1`, `A1`, `5a`) or
# spelled out with a label (`Example 1`, `Cpd. No. 5`, `Ex. 7`). Rejecting the
# labelled forms cost an entire 1,108-row assay table on US10245267: both value
# columns classified correctly as assays, but the id column read as `unknown`,
# so the table produced nothing at all.
_CID_LABEL = re.compile(
    r"^\s*(?:examples?|ex\.?|compounds?|cpds?\.?(?:\s*nos?\.?)?|entry|nos?\.?|#)\s*[:.]?\s*",
    re.I)
# The trailing suffix covers separated stereoisomers, which patents label
# `488-A` / `488-B` (or `12a` / `12b`). Without the hyphenated form those rows
# fail the id test and the whole compound is dropped.
#
# TWO letters, not one. US11312727 separates four stereoisomers per example and
# labels them `100AA`, `100AB`, `100BA`, `100BB` — 135 of its 382 compounds —
# alongside 227 single-letter ids in the same document. Allowing one letter
# matched none of the four and cost every one of them: measured against
# BindingDB, which uses the patent's own labels, it was 131 of the 171 compounds
# we were missing across the whole reference corpus.
# The trailing `-N` is a SUB-INDEX, not a range: US20240335431 numbers the two
# separated atropisomers of example 48 as `48-1` and `48-2`, US10376513 writes
# `323-2`. Capped at two digits and only after a bare number, so a value column
# of ranges (`0.5-1.0`, `100-200 nM`) cannot be mistaken for identifiers — the
# decimal point and the unit both fall outside the pattern.
_CID_CORE = (
    r"(?:"
    # Standard form: optional short letter prefix, required digits, optional letter suffix
    r"(?:[A-Za-z]{1,3}[-\u2013]?)?\d{1,5}(?:[-\u2013]?[a-zA-Z]{1,2})?(?:[-\u2013]\d{1,2})?"
    r"|"
    # Roman-numeral / all-letter ids with no digits: IIa, IIb, IIIc, IVd, etc.
    # US9695181 labels compounds IIa/IIb/IIc/IId — all uppercase letters
    # optionally followed by a single lowercase suffix letter. At least 2 chars
    # to avoid matching single-letter column headers like 'R' or 'X'.
    # Must be anchored by the surrounding _CID_PAT so it does not swallow prose.
    r"[A-Z]{2,4}[a-z]?"
    r")"
)
_CID_PAT = re.compile(rf"^\s*(?:{_CID_LABEL.pattern[2:]})?{_CID_CORE}\s*$", re.I)

# How much of a column must read as a compound id before the header-less
# fallback in `build_columns` will call it one. Not a tuning knob: a real id
# column is spoiled only by footnote rows, and US10253019's `Ex.` column sits
# at 0.9984 because of exactly one (`*nd: no data`). Anything that needs this
# lowered is a column the values do not support, and lowering it would hide
# that rather than fix it.
_CID_FALLBACK_MIN = 0.7


# The shape `normalize_cid` strips padding zeros out of, compiled once. It ran
# 204k times over a 137-patent sweep as an inline literal, paying a pattern-cache
# lookup each time.
_CID_SHAPE = re.compile(r"^([A-Za-z]{1,3}[-–]?)?0*(\d+)([-–]?[a-zA-Z])?$")


# Longest string that can be a compound id. THE CORPUS PICKS THIS, not taste:
# the distinct cids extracted over 137 patents are bimodal in length with an
# EMPTY GAP between 29 and 70 characters —
#
#     10-19    8     `α-6-mPEG1-O-Codeine`  (19) — a real US9233167 label
#     20-29   16
#     30-69    0     <- nothing lives here
#     70-349 404     full IUPAC names, e.g. `N-(3-((1R,5S,6R)-3- amino-5-...`
#
# so any threshold in that gap separates the two populations exactly. 40 sits
# in the middle of it.
#
# WHY THIS IS A LENGTH BOUND AND NOT A SHAPE TEST. `_CID_SHAPE` below permits
# one trailing letter, so 505 perfectly good ids — `100AA`, `101BA`, `10-1` —
# already fail it. Rejecting on shape would delete them. Length is the only
# property that separates a compound number from a compound NAME without
# needing to enumerate every id convention a patent might use.
_CID_MAX_LEN = 40


def normalize_cid(text: str) -> str:
    """`Example 007` / `Cpd. No. 7` / `7` → `7`. `""` when it cannot be an id.

    One canonical form, so a value found in a table headed `Example N` lands on
    the same compound as one found in a table headed `Cpd. No. N`.

    RETURNS EMPTY FOR A NON-ID, which is new and is the point. This function
    used to return whatever it was handed when nothing matched, so when
    `build_columns` mistook a `Name` column for the id column, the compound's
    own IUPAC name became its cid: 410 assay rows across US9611261 and
    US9018217, in two tables, each keyed by a 70-344 character "id" that can
    never join to anything. Nothing downstream could tell that from a real id —
    it is a non-empty string in the cid field, so every consumer treated it as
    one.

    The upstream defect (a name column scoring as the id column) is real and
    lives in `build_columns`; this is the contract check that stops it becoming
    silent data. A function that promises a canonical id has to be able to say
    "this is not one".
    """
    s = (text or "").strip()
    s = _CID_LABEL.sub("", s).strip()
    # Preserve a prefix letter (A1, I-2300); only strip padding zeros.
    m = _CID_SHAPE.match(s)
    if m:
        return f"{m.group(1) or ''}{m.group(2)}{m.group(3) or ''}"
    return s if len(s) <= _CID_MAX_LEN else ""

_HEADER_CID = re.compile(
    r"\b(compound|cpd|example|ex#|entry|structure\s*no|no\.?|number|id)\b", re.I)
_HEADER_NMR = re.compile(r"\bnmr\b|δ\s*\(?ppm|chemical\s+shift", re.I)
# `esi` CARRIED NO WORD BOUNDARY AND MATCHED INSIDE "SYNTHESIS". US10207999
# heads four tables `Chemical Synthesis Example No.` and every one classified
# as a mass-spectrum column, so the compound number was typed as an exclusion
# and no value in those tables could be attributed. It stayed invisible while
# the id-column fallback was free to overwrite a named column — the fallback
# quietly picked column 0 back up. Once the fallback stopped overwriting named
# columns, the misclassification underneath surfaced as 26 tables losing their
# id column. That is the same shape as `open` matching `open_count`.
_HEADER_MS = re.compile(
    r"\bms\b|\bm/z\b|\[m\s*[+±]|\bm\s*\+\s*h\b|lc[- ]?ms|hrms|mass\s+spec|"
    r"\bfound\b|\bobs(?:erved)?\b|\besi\b", re.I)
_HEADER_MW = re.compile(r"\bmw\b|molecular\s+weight|\bcalc(?:d|ulated)?\b|exact\s+mass", re.I)
_HEADER_RT = re.compile(r"\brt\b|retention\s+time|\bt_?r\b|\bmethod\b|\bpurity\b", re.I)
_HEADER_STRUCT = re.compile(r"\bstructure\b|\bstruct\.?\b", re.I)
_HEADER_SUBST = re.compile(r"^\s*R\s*\d*\s*$|^\s*(Ar|X|Y|Z)\s*\d*\s*$", re.I)
_HEADER_NRUNS = re.compile(r"^\s*\(?\s*n\s*\)?\s*$|\bn\s*=|\bruns?\b|\breps?\b", re.I)

# A column that cannot carry a molar concentration, either because the quantity
# is dimensionless or because the header states units of its own. Used only to
# STOP a block-level unit being stamped onto it — never to reject the column.
# A composite unit (`uL/min/mg`, `mL/min/10^6 cells`) is matched by the slash:
# whatever it is, it is not the nM the caption was talking about.
# A `%` that is the unit of the column's VALUE, not part of a threshold.
#
# `_unit_from` takes the first unit token in a header, and a percent-inhibition
# column routinely names a concentration too — the one the assay was RUN at.
# `MAGL % Inh 1 μM (mouse)` reports a percentage AT 1 μM, and read left to
# right the first unit is `μM`, so 1,032 records across 11 patents carried a
# concentration unit on a percentage. `at 10 μM (%)` holding `104.0` came out
# as `104 uM` — a dead compound rather than complete inhibition, from the same
# digits.
#
# The lookbehind is what separates the two uses of `%`. Bound to a number it
# states a threshold and the value is something else: US9987276's
# `>50% occupancy` column holds the CONCENTRATION at which 50% is reached, and
# is correctly nM. Standing alone — `% inh`, `(%)`, `[% Qh]` — it is the unit.
_PCT_VALUE = re.compile(r"(?<![\d.])%")
# `FKBP12 Ki (μM) or %` says the column holds either. Neither answer is right
# for every row, so the header's own hedge is honoured and nothing changes.
_PCT_ALT = re.compile(r"\bor\s*\(?\s*%", re.I)


def _percent_header(h: str) -> bool:
    """Does this header say its values are percentages?"""
    return bool(h) and bool(_PCT_VALUE.search(h)) and not _PCT_ALT.search(h)


_DIMENSIONLESS = re.compile(
    r"\bratio\b|\bselectivit|\bfold\b|\bindex\b|\bsel\.|\bshift\b|"
    r"\bclint\b|\bcl\b|\bt1/2\b|\bauc\b|\bpapp\b|"
    r"[\u03bcu]?[LlgG]\s*/\s*(?:min|h|hr|kg|mL)|/\s*10\s*\^?\s*\d|"
    r"\bp(?:IC|EC|K[id])\s*50?\b|\blog\s*[DP]?\b", re.I)

# Names/metrics that mark a real bioassay column.
_HEADER_ASSAY = re.compile(
    r"\b(ic\s*50|ec\s*50|ed\s*50|gi\s*50|cc\s*50|ki\b|kd\b|kb\b|"
    r"pic50|pec50|pki|pkd|"
    r"%\s*inh|percent\s+inh|inhibition|activity|potency|binding|affinity|"
    r"clint|clearance|t1/2|half[- ]life|papp|permeab|solubilit|"
    r"emax|hill|selectivity|ratio|"
    r"cyp|herg|ppb|auc|cmax)\b", re.I)

# The named potency METRICS only — no loose words like "activity" or "binding".
# A header carrying one of these AND a concentration unit is a measurement, and
# outranks the MS/RT exclusions below; see `classify_column`.
_HEADER_POTENCY = re.compile(
    r"\b(ic\s*50|ec\s*50|ed\s*50|gi\s*50|cc\s*50|lc\s*50|ki\b|kd\b|kb\b|"
    r"pic\s*50|pec\s*50|pki|pkd|mic\b|mec\b)", re.I)
_CONC_UNIT = {"nM", "uM", "pM", "mM", "M", "ng/mL", "ug/mL", "mg/mL", "nm", "um"}


# Words that mark surrounding prose as describing a bioassay. Used only to
# decide whether an unlabelled table is assay data at all — never to name a
# specific column.
_CAPTION_ASSAY = re.compile(
    r"\b(assay|inhibit(?:ion|ory)?|activit(?:y|ies)|potenc(?:y|ies)|"
    r"binding|affinit(?:y|ies)|ic\s*50|ec\s*50|ki\b|kd\b|"
    r"antagonis|agonis|efficac|selectivit|cytotox|antiprolifer)\b", re.I)

# Semi-quantitative potency notation (`+`, `++`, `+++`). Same idea as letter
# bins: record the grade, never invent a number for it.
_PLUS_BIN = re.compile(r"^\s*(\++)\s*$")


@lru_cache(maxsize=8192)
def caption_assay_hint(caption: str) -> tuple[str | None, str | None]:
    """(assay name, unit) inferred from a table's caption, or (None, None).

    Patents frequently publish a table with no header at all, naming the assay
    only in the sentence before it ("Additional in vitro Raf inhibition data is
    provided in the following Table"). Such tables classify as entirely unknown
    and get dropped — the single largest source of missed assay values in the
    corpus. The caption cannot label individual columns, but it can answer the
    question that decides whether the table is read at all.

    Deliberately conservative: a caption with no assay language yields nothing,
    so MS/characterisation tables ("The following table of compounds were
    prepared using the aforementioned methods") stay excluded.
    """
    cap = (caption or "").strip()
    if not cap or not _CAPTION_ASSAY.search(cap):
        return None, None
    unit = _unit_from(cap)
    # Name it from the clause carrying the assay language, not the whole
    # paragraph — captions run to several sentences of buffer composition.
    best = ""
    for clause in re.split(r"[.;:]", cap):
        if _CAPTION_ASSAY.search(clause):
            words = clause.strip()
            if not best or len(words) < len(best):
                best = words
    name = re.sub(r"\s+", " ", best)[:80].strip() or "assay (from caption)"
    return name, unit


_METRIC = re.compile(r"\b(ic\s*50|ec\s*50|ки|ki|kd|gi\s*50|cc\s*50)\b", re.I)


def infer_units_from_description(description: str, assay_names: list[str]) -> dict[str, str]:
    """Resolve units for assays whose table never states one.

    Some tables name the assay but not its unit, deferring to the methods
    section ("Using the assays described above..."). US10245267 reports
    `C-Raf FL IC50` with values like 0.000145 and no unit anywhere in the
    table; the unit lives in prose further up the document.

    A wrong unit is a 1000-fold error — far worse than no unit — so this is
    deliberately strict: a unit is accepted only when it appears in the same
    sentence as both a distinctive token of the assay name and a potency
    metric, and only when every such sentence agrees. Any conflict yields
    nothing and the value stays unitless.
    """
    if not description or not assay_names:
        return {}
    sentences = re.split(r"(?<=[.;])\s+", description)
    # Which sentences mention a potency metric does not depend on WHICH assay
    # name we are resolving, and it was being re-decided for every one of them:
    # 231k `_METRIC.search` calls over a 137-patent sweep for a filter that is
    # a property of the document. Same sentences, same order, decided once.
    sentences = [s for s in sentences if _METRIC.search(s)]
    out: dict[str, str] = {}
    for name in assay_names:
        # Distinctive tokens: drop generic metric words so "IC50" alone can't match.
        tokens = [w for w in re.findall(r"[A-Za-z][\w-]{2,}", name)
                  if not _METRIC.fullmatch(w) and w.lower() not in
                  {"the", "and", "for", "with", "assay", "gmean", "mean", "value", "values"}]
        if not tokens:
            continue
        # One alternation for the name's tokens, built once instead of a fresh
        # `\btok\b` per token PER SENTENCE — 230k regex searches over a
        # 137-patent sweep for a pattern that only depends on the name.
        # `\b(?:a|b)\b` succeeds on exactly the strings one of `\ba\b`/`\bb\b`
        # succeeds on: the boundaries sit outside the group, so each
        # alternative is tried at each position under the same anchors.
        tok_re = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in tokens)
                            + r")\b", re.I)
        found: set[str] = set()
        for s in sentences:
            if not tok_re.search(s):
                continue
            u = _unit_from(s)
            if u and u != "%":
                found.add(u)
        if len(found) == 1:
            out[name] = found.pop()
        elif len(found) > 1:
            logger.info("unit for %r ambiguous across methods text (%s); leaving unset",
                        name, sorted(found))
    return out


@lru_cache(maxsize=1)
def _vocab() -> tuple[frozenset[str], dict[str, str], frozenset[str]]:
    """(assay-type lemmas, qualifier surface → canonical, null markers).

    Reuses the vocabulary the project already learned rather than restating it.
    """
    path = config.PACKAGE_ROOT / "data" / "assay_vocabulary.json"
    assays: set[str] = set()
    quals: dict[str, str] = {}
    nulls: set[str] = set()
    try:
        tokens = json.loads(path.read_text()).get("tokens", [])
    except Exception as e:              # vocabulary is an optimization, not a dependency
        logger.warning("assay vocabulary unavailable (%r); using built-ins", e)
        tokens = []
    for t in tokens:
        cls = t.get("class")
        forms = [s.lower() for s in (t.get("surface_forms") or []) if s]
        if cls == "ASSAY_TYPE":
            assays.update(forms + [str(t.get("lemma", "")).lower()])
        elif cls == "QUALIFIER":
            for s in forms:
                quals[s] = t.get("maps_to") or s
        elif cls == "NULL_MARKER":
            nulls.update(forms + [str(t.get("lemma", "")).lower()])
    nulls.update({"-", "--", "—", "nd", "n.d.", "nt", "n.t.", "na", "n/a", ""})
    return frozenset(a for a in assays if a), quals, frozenset(nulls)


@lru_cache(maxsize=1)
def _assay_lemma_re() -> re.Pattern:
    """`any(a in low for a in assay_lemmas if len(a) > 2)`, as one pattern.

    Identical predicate: every lemma is escaped, so an alternation of literals
    searched against an already-lowercased header succeeds on exactly the
    headers the substring scan succeeded on. What it removes is the scan
    itself — 69 lemmas re-filtered by `len(a) > 2` and tested one at a time,
    per column, per table, on every re-extraction. That comprehension is the
    single hottest line in the trace at 7.2M frame entries over three patents.

    Longest first so the alternation's leftmost-match preference cannot report
    a shorter lemma where a longer one also matches — irrelevant to the boolean
    the caller wants, but it keeps the pattern's behaviour describable.

    An empty vocabulary compiles to `(?!)`, which never matches. `re.compile("")`
    matches EVERYTHING, and the vocabulary is explicitly "an optimization, not a
    dependency" — a missing file must not classify every column as an assay.
    """
    lemmas = sorted((a for a in _vocab()[0] if len(a) > 2), key=len, reverse=True)
    if not lemmas:
        return re.compile(r"(?!)")
    return re.compile("|".join(re.escape(a) for a in lemmas))


# ── data model ────────────────────────────────────────────────────

@dataclass
class Column:
    index: int
    header: str
    kind: str
    unit: str | None = None
    assay_name: str | None = None
    # Where the assay LABEL came from. `header` means the document names this
    # column; `shape` means nothing does, and it was called an assay because
    # its cells look like assay values.
    #
    # A `shape` label is a guess about WHAT WAS MEASURED, and no count can
    # check it: the records come out well-formed and carry a name the patent
    # never wrote. US9221791's `HPLC Method` column and US20240010684A1's
    # peptide-sequence column both produced clean records this way. So the
    # marker is not decoration — `repair.gap.unlabelled_assays` turns it into
    # a question for the heal loop, which is the difference between asking and
    # asserting.
    label_source: str = "header"


@dataclass
class AssayRecord:
    cid: str
    assay_name: str
    value_numeric: float | None = None
    qualifier: str | None = None
    unit: str | None = None
    n_runs: int | None = None
    letter_grade: str | None = None
    # Bounds when the patent published a bin rather than a number. Both may be
    # set with value_numeric None — a compound known to be 0.1-1 uM is a real
    # record, and inventing a point value for it would be a fabrication.
    range_lo: float | None = None
    range_hi: float | None = None
    value_text: str = ""
    table_id: str = ""
    column_header: str = ""
    source: str = "uspto_xml_table"
    unit_source: str = "column"   # "column" | "caption" | "description"

    def as_dict(self) -> AssayRow:
        """Serialise under the ONE schema both binned paths now share.

        The dataclass keeps `range_lo`/`range_hi`/`letter_grade`/`value_text`
        as attribute names — six eval scripts getattr them. The dict spells
        them the way `routes/letter_bin_assays.py` already ships 8,020 rows.
        See `core/models.AssayRow` for why the old spelling still reads.
        """
        return AssayRow(
            (AssayRow._ALIASES.get(k, k), v)
            for k, v in self.__dict__.items() if v not in (None, "")
        )

    def missing_fields(self) -> list[str]:
        """What stops this from being a usable measurement.

        The completeness contract. Every detector bug this session came from
        measuring a PROXY for success — records produced, rows read, coverage
        ratio — and each proxy was blind to some failure the others caught.
        A grade-only record with no key satisfies "a record was produced" while
        containing no measurement at all.

        So the single question is asked of the output itself: is this a usable
        measurement, and if not, which field is missing? The missing field IS
        the diagnosis, and it maps directly onto a repair:

            value      -> value_pattern, or a bin key that was never found
            unit       -> unit resolution from header/legend/description
            assay_name -> column_map
            cid        -> column_map
        """
        missing = []
        if not self.cid:
            missing.append("cid")
        if not self.assay_name or "unnamed" in self.assay_name.lower():
            missing.append("assay_name")
        if (self.value_numeric is None
                and self.range_lo is None and self.range_hi is None):
            missing.append("value")
        if not self.unit:
            missing.append("unit")
        return missing

    @property
    def is_usable(self) -> bool:
        """A measurement a chemist could act on: identified, named, quantified."""
        return not self.missing_fields()


# ── header handling ───────────────────────────────────────────────

_TABLE_TITLE = re.compile(r"^\s*TABLE\s*[-–]?\s*[0-9IVXLC]*\s*[-–]?\s*[0-9]*\s*$", re.I)


def _row_width(row) -> int:
    """Columns this row OCCUPIES, counting empty cells.

    An empty `<entry/>` is a position, not an absence — CALS writes a label
    sitting over columns 6-8 of a nine-column table as six empty entries and
    three full ones. Skipping them made that row look three wide and sent it to
    the offset search, which then had to rediscover a placement the source had
    already stated. Rows that are genuinely short (fewer entries than the table
    has columns, no padding) still need the search and still get it.
    """
    return sum(max(1, c.colspan) for c in row)


_SHAPE_NUM = re.compile(r"^\s*[<>~≈≥≤]?\s*\d*\.?\d+\s*$")
_SHAPE_BIN = re.compile(r"^\s*(\++|[A-E])\*?\s*$")


def _count_matching(pattern, values) -> int:
    """How many of `values` the compiled `pattern` matches at position 0.

    Exactly `sum(bool(pattern.match(v)) for v in values)`, written as a loop
    because that expression is the single hottest shape in this module: a
    generator resumes its own frame once per item, and these run over every
    cell of every column of every table. The three-patent call trace attributes
    7.2M frame entries to genexprs of this form. Same predicate, same order,
    same short-circuit structure at the call sites — only the frame is gone.
    """
    n = 0
    for v in values:
        if pattern.match(v):
            n += 1
    return n


def _column_shapes(table: Table, data) -> list[str]:
    """What each column's VALUES look like — the positional evidence a
    header row without `namest` does not carry.

    A measurement column holds numbers or grade symbols; an id column holds
    short identifiers; a name column holds long text. Which is which is
    knowable from the body even when the header's own position is not.
    """
    shapes: list[str] = []
    for i in range(table.n_cols):
        vals = [r[i].text.strip() for r in data[:40] if len(r) > i and r[i].text.strip()]
        # Column 0 is tested for an identifier FIRST. Everywhere else `num`
        # wins ties, because a measurement column of small integers must not
        # read as ids — but the leftmost column is where patents conventionally
        # put the example number, and without this a bare-integer id column
        # reads as `num`, which then scores an assay label sitting on top of it
        # as a good fit and drags the whole header one column left.
        if not vals:
            shapes.append("empty")
        elif i == 0 and _count_matching(_CID_PAT, vals) > len(vals) * 0.6:
            shapes.append("cid")
        elif _count_matching(_SHAPE_NUM, vals) > len(vals) * 0.6:
            shapes.append("num")
        elif _count_matching(_SHAPE_BIN, vals) > len(vals) * 0.6:
            shapes.append("bin")
        elif _count_matching(_CID_PAT, vals) > len(vals) * 0.6:
            shapes.append("cid")
        else:
            shapes.append("text")
    return shapes


def _shapes_of(table: Table, data) -> list[str]:
    """`_column_shapes`, remembered on the table it describes.

    `merge_header` -> `_choose_offsets` -> `_column_shapes` ran 49,230 times
    over three patents for roughly 3,200 distinct tgroups, because every caller
    that wants a header re-derives the body evidence behind it. The shapes are a
    pure function of the table and the rows handed in, and a `Table` is written
    once by `parse_tables`/`assemble_block` and never mutated afterwards — no
    assignment to `.header_rows`, `.body_rows`, `.n_cols` or `Cell.text` exists
    anywhere in the package outside those constructors.

    The reuse test is `data is cached_data`, not equality and not a hash: the
    entry holds its own reference to the row list, so the object cannot be
    collected and its identity cannot be recycled underneath us while the entry
    lives. Handed a different list, this recomputes.
    """
    got = getattr(table, "_shapes_cache", None)
    if got is not None and got[0] is data:
        return got[1]
    shapes = _column_shapes(table, data)
    try:
        table._shapes_cache = (data, shapes)
    except AttributeError:              # a Table variant with __slots__
        pass
    return shapes


def _header_coherence(headers: list[str], shapes: list[str] | None = None) -> int:
    """How much a candidate header assignment 'makes sense'.

    Used to choose between possible alignments of a short header row. A patent
    that omits leading columns in one header row gives no positional hint, and
    a fixed left- or right-alignment rule gets some layouts right and others
    catastrophically wrong (it shifts every assay name one column, mislabelling
    values rather than dropping them). Scoring the *result* instead lets the
    data decide: the alignment that yields a clean id column and unit-bearing
    assay names is the one the typesetter meant.
    """
    score = 0
    n_cid = 0
    for i, h in enumerate(headers):
        col = classify_column(h, [])
        if col.kind == ASSAY:
            score += 3 if col.unit else 2
        elif col.kind == CID:
            n_cid += 1
        elif col.kind in (NMR, MS, MW, RT, STRUCTURE):
            score += 1          # a confidently-excluded column is also a win
        elif col.kind == UNKNOWN and h.strip():
            score -= 1          # text we failed to make sense of

        # Does the label sit over values of the right kind?
        #
        # Without this the score is blind to the body, so an alignment that
        # parks "IC50 / Ki" over the id and structure columns scores the same
        # as one that parks them over the grades. On US10172859 the blind score
        # left-packed three assay names into columns 0-2 of a table whose
        # measurements are in 3-5, and every record came out unnamed — which in
        # turn made a correct three-scale bin_key unbindable.
        #
        # The body is the positional evidence the header row lost.
        if not shapes or i >= len(shapes) or not h.strip():
            continue
        shape = shapes[i]
        if col.kind == ASSAY:
            score += 4 if shape in ("num", "bin") else -4
        elif col.kind == CID:
            # `num` counts as compatible: a column of plain integers is shaped
            # like a measurement and like an example number both, and demanding
            # `cid` here penalised the CORRECT alignment on every table whose
            # ids are bare integers.
            score += 3 if shape in ("cid", "num") else -3
        elif col.kind == STRUCTURE:
            score += 2 if shape == "text" else -1
    score += 3 if n_cid == 1 else (-2 if n_cid > 1 else 0)
    return score


def _choose_offsets(table: Table, rows) -> list[int]:
    """Pick a starting column for each header row.

    Rows carrying CALS `namest` are authoritative and pinned at 0 (the cells
    place themselves). For the rest we try every feasible offset and keep the
    combination that scores best.

    Unpinned rows of the SAME WIDTH are solved together, not one at a time.
    They are lines of one stacked label block — US10172859's Table 6 heads its
    three right-hand columns with::

        IC50   IC50   Ki
        DNA-   pDNA-  [Kv1.11

    which is one set of labels typeset over two lines, so they cover the same
    columns by construction. Scoring each line on its own put the first at
    columns 3-5 and its own continuation at 0-2: both assay columns came out
    called `IC50 PK`, and since the patent measures DNA-PK in nM and pDNA-PK in
    μM, nothing downstream could tell which was which. Sharing the offset makes
    that split unrepresentable rather than merely unlikely.
    """
    _, data = _split_rows(table)
    shapes = _shapes_of(table, data)

    offsets = []
    for row in rows:
        if any(c.col_start >= 0 for c in row):
            offsets.append(0)
            continue
        slack = table.n_cols - _row_width(row)
        offsets.append(0 if slack <= 0 else None)   # type: ignore[arg-type]

    groups: dict[int, list[int]] = {}
    for i, off in enumerate(offsets):
        if off is None:
            groups.setdefault(_row_width(rows[i]), []).append(i)

    for width, idxs in groups.items():
        best_off, best_score = 0, None
        for cand in range(table.n_cols - width + 1):
            trial = [0 if o is None else o for o in offsets]
            for i in idxs:
                trial[i] = cand
            s = _header_coherence(_merge_with_offsets(table, rows, trial), shapes)
            if best_score is None or s > best_score:
                best_off, best_score = cand, s
        for i in idxs:
            offsets[i] = best_off
    return [0 if o is None else o for o in offsets]


def _is_legend_row(row) -> bool:
    """A sentence of prose occupying few cells, rather than column labels.

    Column labels are short strings spread across the table's width; a legend
    is one long run of text in one or two cells, often a fragment because the
    typesetter wrapped the sentence across several rows.
    """
    cells = [c.text.strip() for c in row if c.text.strip()]
    if not cells or len(cells) > 2:
        return False
    text = " ".join(cells)
    return len(text) > 55 and " " in text.strip()


_HAS_ALPHA = re.compile(r"[A-Za-z]")


def _looks_like_header_row(row) -> bool:
    """A row of labels rather than data.

    Used to promote header rows that patents put in `tbody`. The test is that
    no cell parses as a measurement and at least one carries alphabetic text —
    a data row always has at least one number.
    """
    texts = [c.text.strip() for c in row if c.text.strip()]
    if not texts:
        return False
    if any(_VALUE_PAT.match(t) for t in texts):
        return False
    for t in texts:
        if _HAS_ALPHA.search(t):
            return True
    return False


def _header_rows_of(table: Table) -> tuple[list, list]:
    """Split into (header rows, data rows).

    `thead` is authoritative when present. Otherwise the leading run of
    label-only rows in `tbody` is promoted — US8952177 puts its entire header
    ("Cmp No." / "FLAP Binding wild type HTRF Ki (μM)") in `tbody`, and reading
    only `thead` loses every assay name in the patent.
    """
    if table.header_rows:
        return table.header_rows, table.body_rows
    header, i = [], 0
    for i, row in enumerate(table.body_rows):
        if not any(c.text.strip() for c in row):        # spacer — keep scanning
            continue
        if _is_legend_row(row):
            # Prose legend printed above the column labels. Merging it into the
            # header would smear a whole sentence across the column names; it is
            # picked up separately by `table_legend`, which is where its unit
            # gets read from.
            continue
        if _looks_like_header_row(row) and len(header) < 8:
            header.append(row)
        else:
            break
    return header, table.body_rows[i:] if header else table.body_rows


def _split_rows(table: Table) -> tuple[list, list]:
    """`_header_rows_of`, remembered on the table.

    Same reasoning as `_shapes_of`, and the same immutability argument: the
    split is decided by `table.header_rows` and `table.body_rows`, neither of
    which is ever reassigned or mutated in place after the Table is built.
    Callers only read what comes back — no `append`/`pop`/slice assignment on a
    `hdr_rows`/`data_rows`/`data`/`body` binding exists in the package — and the
    `thead` branch already returned `table.header_rows` itself, so nothing here
    shares more than it did before.

    `_header_rows_of` stays the uncached primitive: it is on the capability
    tier's patchable list, so a model rewriting it must see the decision it
    makes and nothing else. This wrapper is what the hot internal callers use.
    """
    got = getattr(table, "_split_cache", None)
    if got is None:
        got = _header_rows_of(table)
        try:
            table._split_cache = got
        except AttributeError:          # a Table variant with __slots__
            return got
    return got


def table_legend(table: Table) -> str:
    """Prose legend printed inside the table, above its column labels.

    Patents put the table's explanatory sentence in the table itself, wrapped
    across single-cell rows by the typesetter:

        "Selected compound structures and Raf inhibition data: numbering"
        "corresponds to the Examples above ... IC50's are"
        "micromolar."

    That sentence frequently carries the unit — and it is the ONLY place
    US10245267 states it. Read row-by-row the words split apart and the unit is
    invisible, so the rows are rejoined before anything is matched against them.
    Distinguished from column labels by being long prose in few cells, where a
    label row is short strings spread across most columns.
    """
    parts: list[str] = []
    for row in table.body_rows[:12]:
        cells = [c.text.strip() for c in row if c.text.strip()]
        if not cells:
            continue
        text = " ".join(cells)
        # Stop at the first row that looks like data or like column labels.
        if any(_VALUE_PAT.match(c) for c in cells):
            break
        if len(cells) > 2 and len(text) < 60:
            break                       # short strings across columns = labels
        if len(text) < 12 and not parts:
            continue                    # a bare "TABLE 1" caption line
        parts.append(text)
        if len(parts) >= 6:
            break
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def merge_header(table: Table, header_rows=None) -> list[str]:
    """Collapse stacked header rows into one string per column.

    Two things make this non-trivial, both seen in real grants:

    - Patents stack an assay name vertically across rows ("Ave" / "A2B" /
      "cAMP" / "IC50"), so every row must contribute, top to bottom.
    - Header rows often omit leading columns, so a row's first cell is not
      necessarily column 0. We honour CALS `namest` where present and only
      fall back to left-to-right accumulation when it isn't; getting this
      wrong shifts every assay name one column sideways, which silently
      mislabels values rather than dropping them.
    """
    rows = table.header_rows if header_rows is None else header_rows
    return _merge_with_offsets(table, rows, _choose_offsets(table, rows))


def _merge_with_offsets(table: Table, rows, offsets: list[int]) -> list[str]:
    cols: list[list[str]] = [[] for _ in range(table.n_cols)]
    for row, offset in zip(rows, offsets):
        pos = offset
        for cell in row:
            span = max(1, cell.colspan)
            start = cell.col_start if cell.col_start >= 0 else pos
            text = cell.text.strip()
            # Drop the table's own title ("TABLE 569") — it is a caption, not
            # a column label, and folding it in corrupts every assay name.
            if text and not _TABLE_TITLE.match(text):
                for i in range(start, min(start + span, table.n_cols)):
                    cols[i].append(text)
            pos = start + span
            if pos >= table.n_cols:
                break
    # De-duplicate repeated fragments while preserving order (a name spanning
    # several columns otherwise repeats itself).
    out = []
    for parts in cols:
        seen, uniq = set(), []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                uniq.append(p)
        out.append(_join_header_lines(uniq))
    return out


def _join_header_lines(parts: list[str]) -> str:
    """Join the lines of one stacked column label.

    A fragment ending in a hyphen is a word the typesetter broke across lines,
    not two words: US10172859 stacks `DNA-` over `PK`, meaning `DNA-PK`. Joining
    those with a space yields `IC50 DNA- PK`, which reads fine to a human and
    breaks every consumer that matches on the name — the bin-key scale patterns
    for that patent are `DNA-?PK`, and a stray space made them bind nothing, so
    three tables extracted only their hERG column.

    THE HYPHEN ITSELF SURVIVES ONLY WHEN THE NEXT FRAGMENT STARTS A NEW WORD.
    `DNA-` over `PK` is a hyphenated name and keeps its hyphen; `Meth-` over
    `od` is one word the typesetter broke, and keeping the hyphen leaves
    `Meth-od`, which no pattern matching `\\bmethod\\b` can see. US9611261 heads
    a synthesis column that way and it was read as an assay for 288 records —
    the reader already knew a `Method` column is not a measurement, and simply
    could not tell that this was one. Case is the signal a typesetter leaves:
    a continuation is lower-case, a second word is not.
    """
    text = ""
    for p in parts:
        if not text:
            text = p
        elif text.endswith("-"):
            text = text[:-1] + p if p[:1].islower() else text + p
        else:
            text += " " + p
    return text.strip()


# mol/l and its prefixed variants: normalise to the canonical symbol used
# throughout the corpus. Patents like US10266548 state the unit as
# "mol/l" (or "mol/L") in the column header; without this mapping the
# unit comes back as the raw string and the assay column gets no unit,
# causing every record to be dropped as unusable.
#
# Both of these tables were dict LITERALS inside `_unit_from`, rebuilt on every
# call that reached them. Lifting them out is why they are up here: a constant
# is also something the capability tier can offer as a target, which a literal
# buried in a function body never was.
# THE ONE TABLE THAT PUTS A CONCENTRATION ON A COMMON SCALE.
#
# There were two, and they disagreed in both directions: `plausibility` knew
# `µM`/`μM` and not `mol/L`, `to_excel` knew `mol/L` and not the Greek spellings.
# So the audit reported 550 records as carrying a unit "nothing downstream can
# convert" while the converter had handled them all along — two lists that must
# agree, kept apart, which is the same shape as the filter that was narrower
# than its parser and the cache that dropped a field.
#
# Every spelling of micro is here because `_UNIT_CANON` does not reach every
# path: a unit read from a rule payload or a legend arrives as the document
# spelled it.
TO_NM = {
    "nM": 1.0, "uM": 1e3, "µM": 1e3, "μM": 1e3, "mM": 1e6, "pM": 1e-3,
    "M": 1e9, "mol/L": 1e9, "mol/l": 1e9,
    "nmol/L": 1.0, "umol/L": 1e3, "µmol/L": 1e3, "μmol/L": 1e3,
}


def to_nM(value, unit):
    """`value` in `unit` as nanomolar, or None when it is not a concentration."""
    f = TO_NM.get((unit or "").strip())
    if f is None:
        return None
    try:
        return float(value) * f
    except (TypeError, ValueError):
        return None


_MOL_UNIT = {
    "mol/l": "mol/L",
    "umol/l": "uM",
    "µmol/l": "uM",
    "μmol/l": "uM",
    "nmol/l": "nM",
    "mmol/l": "mM",
    "pmol/l": "pM",
}
_CASED_UNIT = {"um": "uM", "µm": "uM", "μm": "uM", "nm": "nM", "mm": "mM",
               "pm": "pM", "percent": "%"}


# Memoised because it is pure and its answer is immutable — it reads `text`,
# `_UNIT_PAT`, `_SPELLED_UNIT`, `_MOL_UNIT`, `_CASED_UNIT` and nothing else, and
# returns a string or None, so there is no shared object for a caller to
# corrupt. `build_columns` asks it three times per table (header, legend,
# caption) and `extract_from_tables` twice more, over the same few strings on
# every re-extraction: 127k searches of the corpus's most expensive pattern.
@lru_cache(maxsize=16384)
def _unit_from(text: str) -> str | None:
    m = _UNIT_PAT.search(text or "")
    if not m:
        return None
    raw = (m.group(1) or m.group(2) or m.group(3) or "").strip()
    low = raw.lower()
    if low in _SPELLED_UNIT:
        return _SPELLED_UNIT[low]
    if low in _MOL_UNIT:
        return _MOL_UNIT[low]
    return _CASED_UNIT.get(low, raw)


# The three patterns `classify_column` matched with inline `re.search` literals.
# Same patterns, same flags; `(?i)` becomes `re.I` because an inline global flag
# must be the first thing in a pattern and `re.compile` is where it belongs.
_CID_NOT_P_METRIC = re.compile(r"\bp?(?:ic|ec)\s*50|p(?:ki|kd|k_?i|k_?d)\b", re.I)
_PCT_INH = re.compile(r"%\s*inh")
# A percentage column is an assay when the header names what was MEASURED.
#
# The gate below reads `is_assay or (unit and unit != "%")`, so a column whose
# only claim to being an assay was a unit now correctly read as `%` falls
# through it. That cost 113 real records — `% Effect at 30 μM relative to
# 2'3'-cGAMP` and `% amount of pSer376-SLP-76 @ 20 μM` — which are assay
# results by any reading.
#
# Stated as a REQUIREMENT rather than an exclusion, because the set of things a
# patent reports as a bare percentage and does not mean as an assay is open:
# yield, purity, enantiomeric excess, recovery. None of them names a measured
# activity, so none of them matches, and a new one does not have to be foreseen.
_PCT_MEASURED = re.compile(
    r"\b(?:inhibit\w*|inh\.?|effect|activit\w*|occupanc\w*|amount|response|"
    r"reduction|stabilit\w*|remaining|conversion|displacement|control)\b", re.I)
_P_METRIC = re.compile(r"\bp\s*(?:ic|ec)\s*50\b|\bp\s*(?:ki|kd|k_?i|k_?d)\b", re.I)

# How many distinct (header, samples) pairs to remember. A patent's whole
# corpus of tables is a few thousand columns; the ceiling exists so a corpus
# run cannot grow this without bound, not because anything approaches it.
_COLUMN_CACHE_MAX = 20_000


def _memoise_column(fn):
    """Cache `classify_column` on its arguments, and NEVER share the result.

    Purity, checked by reading the body: it reads `header`, `samples`, the
    module-level `_HEADER_*`/`_LETTER_BIN`/`_PLUS_BIN`/`_NRUNS_ONLY`/
    `_STAR_HASH_BIN`/`_CONC_UNIT` patterns, `_unit_from` (pure over the same
    module constants), `split_top_level` (pure), and `_assay_lemma_re()` /
    `_vocab()` (both `lru_cache`d over one file read, with no `cache_clear`
    anywhere in the package). No global is written, no clock or filesystem is
    read, nothing is appended to. Same arguments, same answer.

    The copy is the part that is not optional. `Column` is a mutable dataclass
    and `build_columns` mutates what this returns — `c.index = i`, `best.kind =
    CID`, `best.assay_name = None`, `c.unit = ctx_unit`, `c.kind = ASSAY` — so
    handing back the cached instance would let one table's classification
    rewrite the answer every later table gets. That is exactly the shape of
    silent extraction damage this module cannot afford, so the cache stores the
    values and every caller receives its own object.

    It is worth the care because the recomputation is real: `_header_coherence`
    calls `classify_column(h, [])` once per candidate offset per header row and
    it is pure in the header string, and `build_columns` re-classifies the same
    columns on every re-extraction of the same table.
    """
    cache: dict[tuple, Column] = {}

    @functools.wraps(fn)
    def wrapper(header: str, samples: list[str]) -> Column:
        key = (header, tuple(samples))
        try:
            got = cache.get(key)
        except TypeError:               # unhashable sample — just compute it
            return fn(header, samples)
        if got is None:
            got = fn(header, samples)
            if len(cache) < _COLUMN_CACHE_MAX:
                cache[key] = got
        # `replace`, not a field-by-field rebuild. Listing the fields here
        # means a field added to `Column` is silently dropped on every cached
        # call and defaults instead — which is what happened to `label_source`:
        # the classifier set it correctly and every caller read the default,
        # so no column ever looked uncertain.
        return _dc_replace(got)

    wrapper.cache_clear = cache.clear       # type: ignore[attr-defined]
    wrapper.cache_size = lambda: len(cache)  # type: ignore[attr-defined]
    return wrapper


@_memoise_column
def classify_column(header: str, samples: list[str]) -> Column:
    """Decide what a column holds, from its header first, its values second.

    Order matters and is deliberately conservative: the exclusions (NMR, MS,
    MW, RT, structure) are checked BEFORE the assay test, because headers like
    "LCMS IC50 method" would otherwise read as an assay. Anything unrecognised
    becomes UNKNOWN and is skipped rather than guessed.

    One case outranks the exclusions, because it is not ambiguous: a header
    that names a potency METRIC *and* carries a concentration unit is a
    measurement whatever platform word sits beside it. US10329273 heads its
    only assay column "h-MGAT LCMS IC50 (nM)" over values 27, 5, 340 — the
    `lc-?ms` exclusion swallowed the entire patent, all 22 reference compounds.
    Both conditions are required, so "LCMS IC50 method" — a metric name with no
    concentration unit — still falls through to the exclusion as before.

    Multi-assay headers: a header of the form "probe 1, probe 2" names two
    assays in one column. When the header contains a comma and each part looks
    like an assay name (or the data cells are comma-separated value lists), the
    column is classified as ASSAY with `assay_name` set to the full header so
    that `extract_from_tables` can split it later.

    Plus/letter-bin promotion: when a column has a non-empty header that is not
    otherwise recognised (not an exclusion, not a known assay lemma) but the
    data cells are predominantly plus-bins (+/++/+++) or letter-grade bins
    (A/B/C/D), the column is promoted to ASSAY using the header as the assay
    name. This handles short assay abbreviations like "FP" that don't appear
    in any vocabulary list. US11286268 has header "FP" over 1258 rows of
    +/++/+++ values that were lost without this.

    Star/hash-bin promotion: cells containing only repeated `*` or `#`
    characters are ordinal potency bands (like +/++/+++). Columns whose data
    is predominantly these symbols are promoted to ASSAY the same way
    plus-bins and letter-grades are.

    p-prefixed metrics: headers like "pIC50", "pEC50", "pKi", "pKd" are
    negative-log10 potency values. The "p" prefix is directly attached to the
    metric name with no word boundary, so standard assay-name regexes that
    rely on \\b fail to match. These are recognised explicitly as assay
    columns (dimensionless — no concentration unit). The header text itself
    is used as the unit (e.g. "pIC50") so that emitted records carry a
    meaningful unit rather than None. US9801872 TABLE-US-00001 heads its only
    assay column "pIC50" over 71 rows; without the unit assignment every
    record failed the usability contract and the patent produced nothing.
    """
    h = (header or "").strip()
    low = h.lower()

    if _HEADER_POTENCY.search(low):
        u = _unit_from(h)
        if u and u in _CONC_UNIT:
            return Column(-1, h, ASSAY, unit=u, assay_name=h)

    if _HEADER_NMR.search(low):
        return Column(-1, h, NMR)
    if _HEADER_MW.search(low):
        return Column(-1, h, MW)
    if _HEADER_MS.search(low):
        return Column(-1, h, MS)
    if _HEADER_RT.search(low):
        return Column(-1, h, RT)
    if _HEADER_STRUCT.search(low):
        return Column(-1, h, STRUCTURE)
    if _HEADER_NRUNS.match(h):
        return Column(-1, h, NRUNS)
    if _HEADER_SUBST.match(h):
        return Column(-1, h, SUBSTITUENT)
    if _HEADER_CID.search(low) and not _HEADER_ASSAY.search(low):
        # Also guard against p-prefixed metrics being swallowed by CID:
        # "pIC50" should not match a CID pattern.
        if not _CID_NOT_P_METRIC.search(low):
            return Column(-1, h, CID)

    lemma_re = _assay_lemma_re()
    is_assay = bool(_HEADER_ASSAY.search(low)) or bool(lemma_re.search(low))
    # Also recognise "% Inh" / "% inh" / "% inhibition" as assay indicators.
    # These appear in headers like "MAGL % Inh 1 uM (mouse)" and are not
    # caught by the standard assay regex or lemma list.
    if not is_assay and _PCT_INH.search(low):
        is_assay = True
    # ...and any other percentage whose header names what was measured.
    if not is_assay and _percent_header(h) and _PCT_MEASURED.search(low):
        is_assay = True
    # p-prefixed potency metrics: "pIC50", "pEC50", "pKi", "pKd" etc.
    # The 'p' is directly attached to the metric name so word-boundary-based
    # patterns miss them. These are dimensionless (-log10 of a concentration).
    # The header text itself is used as the unit so that emitted records carry
    # a meaningful unit (e.g. "pIC50") rather than None — a unit of None fails
    # the usability contract and causes the record to be dropped.
    _p_metric_match = _P_METRIC.search(low)
    if not is_assay and _p_metric_match:
        is_assay = True
    # A percentage column takes `%`, not the concentration it was run at.
    unit = "%" if _percent_header(h) else _unit_from(h)
    # For p-prefixed metrics the header IS the unit (e.g. "pIC50"). When
    # _unit_from returns nothing (because pIC50 is not a concentration unit),
    # use the header text itself as the unit so downstream records are not
    # emitted with unit=None.
    if _p_metric_match and not unit:
        unit = h
    if is_assay or (unit and unit != "%"):
        return Column(-1, h, ASSAY, unit=unit, assay_name=h or "unnamed assay")

    # Multi-assay header: comma-separated sub-names, e.g. "probe 1, probe 2".
    # Detected when the header contains a comma AND the data cells are also
    # comma-separated (same count of parts), or when each comma-separated part
    # of the header looks like an assay name on its own.
    if "," in h:
        parts = split_top_level(h)
        if len(parts) >= 2:
            # Check whether data cells are also comma-separated with the same
            # number of parts — that is the strongest signal.
            n_parts = len(parts)
            multi_value_count = 0
            for s in samples:
                if not s:
                    continue
                cell_parts = split_top_level(s)
                if len(cell_parts) == n_parts:
                    multi_value_count += 1
            if multi_value_count > 0 and multi_value_count >= len([s for s in samples if s]) * 0.4:
                return Column(-1, h, ASSAY, unit=unit, assay_name=h)
            # Fallback: each part of the header looks like an assay name.
            parts_are_assay = 0
            for p in parts:
                pl = p.lower()
                if (_HEADER_ASSAY.search(pl) or lemma_re.search(pl)
                        or _HEADER_POTENCY.search(pl)):
                    parts_are_assay += 1
            if parts_are_assay >= 1:
                return Column(-1, h, ASSAY, unit=unit, assay_name=h)

    # Headerless continuation columns: fall back to the shape of the data.
    # Also applies to columns with a short non-empty header that was not
    # recognised by any rule above — a header like "FP" is an assay
    # abbreviation the vocabulary doesn't know, but the data shape (all
    # plus-bins or letter-grades) is unambiguous.
    if True:  # was: `if not h:` — now also fires for unrecognised short headers
        vals = [s for s in samples if s]
        if vals:
            if _count_matching(_NRUNS_ONLY, vals) > len(vals) * 0.6:
                # Only promote to NRUNS when the header is empty; a named
                # column of parenthesised numbers is unlikely.
                if not h:
                    return Column(-1, h, NRUNS)
            # Star/hash bins (`*`, `**`, `###`) are ordinal grades like +/++/+++.
            plus_or_letter = 0
            for v in vals:
                if (_LETTER_BIN.match(v) or _PLUS_BIN.match(v)
                        or _STAR_HASH_BIN.match(v.strip())):
                    plus_or_letter += 1
            # A grade alphabet is closed. Letters outside it mean the column
            # enumerates rather than grades — see `_ANY_LETTER`.
            stray = sum(1 for v in vals
                        if _ANY_LETTER.match(v) and not _LETTER_BIN.match(v))
            if (plus_or_letter > len(vals) * 0.6
                    and stray <= len(vals) * _STRAY_LETTER_MAX):
                # The data shape decided this, not the document. Marked so the
                # heal loop is asked rather than the guess being asserted.
                return Column(-1, h, ASSAY,
                              assay_name=h if h else "unnamed assay (letter bin)",
                              label_source="header" if h else "shape")
    return Column(-1, h, UNKNOWN)


# The bracket characters `split_top_level` tracks depth with. One search is
# cheaper than the character loop it lets us skip.
_BRACKET = re.compile(r"[\[\](){}]")


def split_top_level(text: str) -> list[str]:
    """Split on commas that are NOT inside brackets.

    `probe 1, probe 2` is two assays. `HT1080 (R132C, 2-hydroxyglutarate)
    IC50 (uM)` is ONE — the comma names a mutant and a metabolite inside a
    parenthesis. Splitting it produced records under the assay name
    `HT1080 (R132C` and paired the other half with `n = 7` from the cell, so
    the run count vanished and the column was named after a fragment.

    Worse, it was intermittent: the sibling column `HT1080 (R132C, aKG) IC50
    (uM)` kept its full name purely because its cells read `>20.0` with no
    comma to split on. The same header parsed two different ways depending on
    what sat under it.
    """
    # With no bracket anywhere, `depth` can never leave 0 and every comma is a
    # split point — which is precisely `str.split(",")`. The loop below is a
    # per-character Python loop run 389k times over a 137-patent sweep, and the
    # overwhelming majority of the strings handed to it are bracket-free cells
    # like `1,234.5`. Same parts, same strip, same drop of empties.
    s = text or ""
    if "," not in s:
        one = s.strip()
        return [one] if one else []
    if not _BRACKET.search(s):
        return [p for p in (x.strip() for x in s.split(",")) if p]

    parts, depth, cur = [], 0, []
    for ch in text or "":
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _redistribute_shared_prefix(parts: list[str]) -> list[str]:
    """Give a comma-split header's shared prefix back to every part.

    `TR-FRET Binding IC50 (uM) probe 1, probe 2` is ONE physical column
    naming TWO assays, and the merged header carries the shared text only
    once, ahead of the first sub-name: `split_top_level` on the comma gives
    `TR-FRET Binding IC50 (uM) probe 1` and `probe 2`. The second part has
    lost the measurement it belongs to.

    The signal is structural: the first part's tail must match every other
    part word-for-word, except for the tokens that carry the digit which
    tells the sub-assays apart (`probe 1` vs `probe 2`). Anything that does
    not fit that shape is left alone — a name is worth more skipped than
    guessed.
    """
    if len(parts) < 2:
        return parts
    first_words = parts[0].split()
    tails = [p.split() for p in parts[1:]]
    tail_len = len(tails[0])
    if tail_len == 0 or len(first_words) <= tail_len:
        return parts
    if any(len(t) != tail_len for t in tails):
        return parts
    first_tail = first_words[-tail_len:]
    for tail in tails:
        for a, b in zip(first_tail, tail):
            if a == b:
                continue
            if not (any(ch.isdigit() for ch in a) and any(ch.isdigit() for ch in b)):
                return parts
    prefix = " ".join(first_words[:-tail_len]).strip()
    if not prefix:
        return parts
    return [parts[0]] + [f"{prefix} {p}" for p in parts[1:]]


def _label_bearing(table: Table, rows) -> list[int]:
    """Data columns that a header would actually name.

    Run-count columns (`(8)`) are structural: patents write them beside the
    value and never label them, so the header of a 3-column table can describe
    a 5-column data table. Excluding them is what makes the two line up.
    """
    keep = []
    for i in range(table.n_cols):
        vals = [r[i].text.strip() for r in rows if len(r) > i and r[i].text.strip()]
        if vals and _count_matching(_NRUNS_ONLY, vals) > len(vals) * 0.6:
            continue
        keep.append(i)
    return keep


def _fit_inherited(inherited: list[str], table: Table, rows) -> list[str]:
    """Map a previous tgroup's header onto this (headerless) continuation.

    Widths often differ — US8952177's header table declares 3 columns while its
    data table has 5, because each value carries an unlabelled `(n)` column.
    Assigning positionally would put the LTB4 assay name on a run-count column
    and silently mislabel every value, so labels go onto the label-bearing
    columns in order instead.
    """
    headers = [""] * table.n_cols
    if len(inherited) == table.n_cols:
        return list(inherited)
    targets = _label_bearing(table, rows)
    for label, idx in zip(inherited, targets):
        headers[idx] = label
    return headers


def build_columns(table: Table, inherited: list[str] | None = None,
                  data_rows=None, inherited_unit: str | None = None) -> list[Column]:
    hdr_rows, body = _split_rows(table)
    headers = merge_header(table, hdr_rows)
    rows = body if data_rows is None else data_rows
    if inherited and not any(headers):
        headers = _fit_inherited(inherited, table, rows)
    cols: list[Column] = []
    for i in range(table.n_cols):
        samples = [r[i].text for r in rows[:40] if len(r) > i]
        c = classify_column(headers[i] if i < len(headers) else "", samples)
        c.index = i
        # A column of `(8)`, `(10)`, `(3)` is a replicate count whatever its
        # header says. Patents write the count in its own column beside the
        # value and let the assay's header SPAN both, so US8952177's run-count
        # columns inherited "FLAP Binding wild type HTRF Ki (μM)" and
        # classified as a second assay — which both invented a duplicate assay
        # column and cost every value its `n` (the attach below only fires on a
        # neighbour typed NRUNS or UNKNOWN). The values are unambiguous where
        # the header is not, so they win here.
        vals = [s.strip() for s in samples if s.strip()]
        if (c.kind == ASSAY and vals
                and _count_matching(_NRUNS_ONLY, vals) > len(vals) * 0.6):
            c = Column(i, c.header, NRUNS)
        cols.append(c)

    # Exactly one id column. If the header didn't name one, take the leftmost
    # column whose values actually look like compound ids.
    #
    # THE CODE USED TO SAY `if score > best_score` AND TAKE THE MAXIMUM, WHICH
    # IS NOT WHAT THE LINE ABOVE PROMISES. US10253019 TABLE-US-00003 heads
    # `Ex. | Structure | TBK1 IC50 | IKKe IC50 | JAK2 IC50 | Name`. `_HEADER_CID`
    # wants the spelled-out `example` and this patent abbreviates, so no header
    # named an id and the fallback ran. `Ex.` scored 0.9984 — one footnote row
    # (`*nd: no data`) breaks its integer run — while `TBK1 IC50`, an assay
    # column of whole-number nM values, scored a perfect 1.0000 and won the id
    # role by 0.0016. Every name in a 620-row table was then joined to an IC50
    # value instead of an example number, and 104 rows shipped stamped `cid=2`.
    #
    # Two things are wrong and both are fixed here.
    #
    # A column whose HEADER already named it is not a candidate. The fallback
    # exists for "the header didn't name an id"; letting it overwrite a column
    # the header DID name is the fallback answering a question that was already
    # answered. A header saying `IC50 (nM)` states the column is a measurement,
    # and no arrangement of its digits makes it a compound number.
    #
    # And among what is left, LEFTMOST wins, as promised. A compound number is
    # the first column of a patent table by near-universal convention, and a
    # comparison by hundredths of a percent between two columns that both look
    # like integers is not a judgement — it is a coin toss with a tie-break
    # nobody chose.
    if not any(c.kind == CID for c in cols):
        named = {c.index for c in cols
                 if c.kind not in (UNKNOWN, SUBSTITUENT) and (c.header or "").strip()}
        best = None
        for c in cols:
            if c.index in named:
                continue
            # `r[c.index].text` was indexed and re-read three times per row,
            # and `.strip()` allocated a copy of every cell only to test it for
            # emptiness. Same rows, same order, same predicate.
            ci = c.index
            vals = []
            for r in rows:
                if len(r) > ci:
                    t = r[ci].text
                    if t and not t.isspace():
                        vals.append(t)
            if not vals:
                continue
            if _count_matching(_CID_PAT, vals) / len(vals) >= _CID_FALLBACK_MIN:
                best = c
                break                      # LEFTMOST, as the comment promises
        if best is not None:
            best.kind = CID
            best.assay_name = None

    # Last resort: a table with no header anywhere, whose caption says it is
    # assay data. Unlabelled numeric columns become assays named from the
    # caption. Gated on the caption so characterisation tables (MS, purity)
    # stay excluded — without that gate this would re-import exactly the noise
    # column classification exists to remove.
    # A unit stated only in the table's legend or caption applies to every
    # assay column that didn't carry its own. Done before the caption fallback
    # so a table with proper headers but no inline unit still gets one.
    #
    # ...but only onto columns that could HOLD that unit. A block-level unit is
    # a statement about the columns the legend describes, not about every
    # numeric column beside them. US11254686 TABLE-US-00003 heads nine columns
    # `Compound | Ave A2B cAMP IC50 | Ave A2A cAMP IC50 | Ratio | A1 cAMP IC50
    # | A3 cAMP IC50 | CYP 450 % INH @ 10 uM | LM CLint (uL/min/mg/protein) |
    # Hep CLint (mL/min/10^6 cells)`, with `A=<10 nM; B=10-50 nM; ...` in the
    # caption. Every one of them was stamped `nM` — including a dimensionless
    # selectivity ratio and two clearances whose own units are printed in their
    # headers. Checked against BindingDB that is 99 records reading `2.24 nM`
    # where the reference says 300 nM: not a near miss, a different quantity.
    #
    # So a header that names a dimensionless quantity, or that carries units of
    # its own we could not parse, keeps no unit at all. The record then fails
    # the usability contract and is dropped, which is the intended trade — a
    # missing assay is recoverable, a ratio recorded as a potency is a lie the
    # database cannot detect later.
    # HEADER FIRST, and it was not a source at all.
    #
    # US11420968 heads four assay columns as two pairs under a spanning
    # `(IC50, nM)`, which our merge lands on only the second column of each
    # pair. The two that missed out had no unit of their own, so they fell
    # through to the caption — a paragraph opening "The Bcl-2 family proteins
    # are central regulators of apoptosis" that mentions uM somewhere — and 111
    # values were recorded 1000x low. BindingDB caught it; nothing else could.
    #
    # The table's own header rows say `nM` twice, in plain sight. A caption is
    # prose about the biology and may name any unit for any reason; a header is
    # a statement about these columns. So the precedence is header, then
    # legend, then caption, then whatever a previous tgroup carried.
    header_text = " ".join(c.text for r in hdr_rows for c in r)
    legend = table_legend(table)
    ctx_unit = (_unit_from(header_text) or _unit_from(legend)
                or _unit_from(table.caption) or inherited_unit)
    if ctx_unit and ctx_unit != "%":
        for c in cols:
            if c.kind == ASSAY and not c.unit and not _DIMENSIONLESS.search(c.header or ""):
                c.unit = ctx_unit

    if not any(c.kind == ASSAY for c in cols):
        cap_name, cap_unit = caption_assay_hint(table.caption)
        if cap_name:
            for c in cols:
                if c.kind != UNKNOWN or c.header.strip():
                    continue
                vals = [r[c.index].text.strip() for r in rows
                        if len(r) > c.index and r[c.index].text.strip()]
                if not vals:
                    continue
                usable = 0
                for v in vals:
                    if (_VALUE_PAT.match(v) or _LETTER_BIN.match(v)
                            or _PLUS_BIN.match(v)):
                        usable += 1
                if usable >= len(vals) * 0.7:
                    c.kind = ASSAY
                    c.assay_name = cap_name
                    c.unit = cap_unit
    return cols


# ── value parsing ─────────────────────────────────────────────────

# `*`/`**` and `#`/`##` are ordinal potency bands, the same idea as `+`/`++`.
# Hoisted to module level: the patch that introduced this compiled it inside
# `parse_value`, which runs for every cell of every table in the corpus.
_STAR_HASH_BIN = re.compile(r"([*]+|[#]+)")

# The ASCII spellings of the three comparison qualifiers, folded to the symbol
# the records carry. Was a dict literal rebuilt inside `parse_value`.
_QUAL_CANON = {">=": "≥", "<=": "≤", "≈": "~"}


def parse_value(cell: str) -> dict | None:
    """Parse one measurement cell. None when it holds no usable value.

    Star/hash bin grades: cells containing only repeated `*` or `#` characters
    (e.g. `**`, `###`) are ordinal potency bands used in patent tables where a
    bin_key legend maps them to numeric ranges. They are recorded as letter
    grades, analogous to `+`/`++`/`+++` and `A`/`B`/`C`.

    Unicode normalization: scientific notation values often use typographic
    characters — unicode minus (U+2212), en-dash (U+2013), non-breaking space
    (U+00A0), or unicode comparison operators (≥ U+2265, ≤ U+2264, ≈ U+2248)
    — that the main value regex cannot match. The cell text is normalized to
    ASCII equivalents before regex matching so that e.g. '≥1.0e−005' parses
    correctly.
    """
    s = (cell or "").strip()
    _, quals, nulls = _vocab()
    if s.lower() in nulls:
        return None
    m_sd = _MEAN_SD.match(s)
    if m_sd:
        try:
            return {"value_numeric": float(m_sd.group("num").replace(",", "")),
                    "qualifier": m_sd.group("qual"),
                    "unit": _SPELLED_UNIT.get((m_sd.group("unit") or "").lower(),
                                              m_sd.group("unit")),
                    "n_runs": int(m_sd.group("n")) if m_sd.group("n") else None,
                    "annotation": None, "value_text": s}
        except ValueError:
            pass
    lb = _LETTER_BIN.match(s)
    if lb:
        return {"letter_grade": lb.group(1).upper(), "value_text": s}
    pb = _PLUS_BIN.match(s)
    if pb:
        # `+`/`++`/`+++` is an ordinal potency band. Recorded as a grade; any
        # number we assigned to it would be invented.
        return {"letter_grade": pb.group(1), "value_text": s}
    # Star-bin (`*`/`**`/`***`) and hash-bin (`#`/`##`/`###`) grades: ordinal
    # potency bands used in tables whose legend maps symbols to numeric ranges.
    # Treated identically to plus-bins.
    sh = _STAR_HASH_BIN.fullmatch(s)
    if sh:
        return {"letter_grade": sh.group(1), "value_text": s}
    # Normalize unicode characters that are typographic variants of ASCII
    # operators and signs before attempting the main value regex. This lets
    # cells like `≥1.0e−005` match without complicating
    # the regex itself. The original `s` is preserved for `value_text`.
    #
    # Every character replaced below is non-ASCII, so an all-ASCII cell (which
    # is nearly every cell in the corpus) cannot contain one and the eleven
    # `str.replace` passes over it cannot change it. `str.isascii()` is a flag
    # check on the string object, not a scan.
    if s.isascii():
        s_norm = s
    else:
        s_norm = s.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')
        s_norm = s_norm.replace('\u00a0', ' ').replace('\u2009', ' ')
        s_norm = s_norm.replace('\u2265', '>=').replace('\u2264', '<=').replace('\u2248', '~')
        s_norm = s_norm.replace('\u2267', '>=').replace('\u2266', '<=')
        s_norm = s_norm.replace('\u2a7e', '>=').replace('\u2a7d', '<=')
    m = _VALUE_PAT.match(s_norm)
    if not m:
        # Also try the original in case normalization broke something.
        m = _VALUE_PAT.match(s)
    if not m:
        return None
    qual = m.group("qual")
    if qual:
        qual = quals.get(qual.lower(), qual)
        qual = _QUAL_CANON.get(qual, qual)
    try:
        # `float` knows `6.49E-03` but not `6.49E−03`; the minus a typesetter
        # chose must not decide whether a measurement survives.
        num = float(m.group("num").replace(",", "")
                    .replace("−", "-").replace("–", "-").replace(" ", ""))
    except ValueError:
        return None
    # A parenthetical of digits is a replicate count; anything else is a
    # footnote marker and carries no measurement meaning.
    paren = (m.group("paren") or "").strip()
    n_runs = int(paren) if paren.isdigit() and len(paren) <= 3 else None
    return {
        "value_numeric": num,
        "qualifier": qual,
        "unit": m.group("unit"),
        "n_runs": n_runs,
        "annotation": paren if paren and not paren.isdigit() else None,
        "value_text": s,
    }


def _is_spacer(row) -> bool:
    # `not any(c.text.strip() for c in row)`, without the generator frame or
    # the throwaway stripped copy of every cell. `t and not t.isspace()` is
    # true on exactly the strings `t.strip()` is truthy on: `""` is falsy and
    # `"".isspace()` is False, so the empty cell is handled by the first term.
    for c in row:
        t = c.text
        if t and not t.isspace():
            return False
    return True


# ── extraction ────────────────────────────────────────────────────

# A cell holding a list of compound ids rather than a value. Patents invert the
# usual layout for large screens: one row per potency bin, every compound in
# that bin listed inside a single cell.
_BIN_SYMBOL = re.compile(r"^\s*(\++|[A-E]|NT|N\.?T\.?)\s*$", re.I)


def _cid_list(text: str) -> list[str]:
    """Compound ids from a comma/space separated cell, or [] if it isn't one."""
    s = (text or "").strip()
    if s.count(",") < 2:
        return []
    parts = [p.strip() for p in s.split(",")]
    ids = [p for p in parts if p and _CID_PAT.match(p)]
    # Require most of the cell to be ids, so a prose sentence with a couple of
    # numbers in it is not mistaken for a compound list.
    return ids if len(ids) >= max(3, int(len(parts) * 0.7)) else []


def extract_inverted(tables: list[Table], bin_key: dict, *,
                     assay_name: str, unit: str | None,
                     table_id: str = "") -> list[AssayRecord]:
    """Read 'one row per bin, compound ids listed in a cell' tables.

    US11566007 publishes 1,827 compound-bin assignments this way:

        IC50*  |  Examples
          +    |  A104, A107, A108, ... (253 ids in one cell)
         ++    |  A10, A105, ...        (296 ids)

    The normal row-per-compound reader sees a table with almost no numeric
    cells and skips it entirely, which is how these were missed. Every id in
    the cell gets its own record carrying the bin's range.
    """
    out: list[AssayRecord] = []
    for t in tables:
        # The bin symbol appears once, on the first row of its group; the
        # compound list then wraps across many following rows with no symbol of
        # their own. Requiring symbol and ids in the same row caught only the
        # first fragment — 84 of 1,827 assignments on US11566007 — so the
        # current symbol is carried forward until a new one appears.
        current_sym: str | None = None
        # HEADER ROWS FIRST, because in an inverted table a header row can BE
        # data. US11566007 TABLE-US-00005 puts `['+', 'A104, A107, A108, ...']`
        # in the thead: the symbol and the first eight compounds of its bucket.
        # Reading only the body lost those eight AND left `current_sym` unset,
        # so every continuation row beneath them — rows that carry compounds
        # and no symbol of their own — was dropped too. 247 grade assignments
        # on one block, silently, while every count looked healthy.
        #
        # Safe to prepend rather than gate: a row with no compound ids is
        # skipped below regardless, so a genuine header like
        # `['IC50*', 'Examples']` still contributes nothing.
        for row in list(t.header_rows) + list(t.body_rows):
            cells = [c.text.strip() for c in row]
            sym = next((c for c in cells if _BIN_SYMBOL.match(c)), None)
            if sym:
                current_sym = sym
            if not current_sym:
                continue
            ids: list[str] = []
            for c in cells:
                ids = _cid_list(c)
                if ids:
                    break
            if not ids:
                continue
            if current_sym.upper().replace(".", "") == "NT":
                continue                      # explicitly not tested
            sym = current_sym
            rng = bin_key.get(sym)
            for cid in ids:
                out.append(AssayRecord(
                    cid=normalize_cid(cid),
                    assay_name=assay_name,
                    value_numeric=None,
                    unit=(rng.unit if rng else unit),
                    letter_grade=sym,
                    range_lo=rng.lo if rng else None,
                    range_hi=rng.hi if rng else None,
                    value_text=sym,
                    table_id=table_id or t.table_id,
                    column_header=assay_name,
                    source="uspto_xml_bin_table",
                ))
    return out


# ── vertical records: one compound per BLOCK of rows ──────────────

# The field NAMES run down column 0; each compound is a consecutive run of rows,
# not a row. `build_columns` cannot represent this at all: its loop (l.1229)
# emits one `Column.kind` per column INDEX for the whole table, and `Column`
# (l.421) holds one `kind`. Column 1 of such a table is in turn a cid, two IC50
# values and an IUPAC name, so whatever single verdict it gets is wrong for most
# of its cells. Measured on US9265734 TABLE-US-00006 the verdict was
# `substituent`; `assay_cols` came back empty and `extract_from_tables` skipped
# the block at its `cid_col is None or (not assay_cols and not prose_cols)`
# guard (l.2118). 199 compounds, 371 measurements and 199 IUPAC names, 0
# records, and — because `find_gaps` drops the block on its payload ratio
# (gap.py:530) and `usable_yield` never sees it (gap.py:826) — no loss logged
# either. This is the silent-block failure, not the unread-ids one.
#
# It is not the heal loop's to fix. The loop writes regexes: value_pattern,
# column_map, bin key. None of those can say "a compound is twelve rows".
#
# DETECTION. Column 0 of a real row-per-compound table holds compound IDS, which
# are distinct. Column 0 of a vertical table holds FIELD NAMES, which repeat once
# per record. `distinct(col0)/count(col0)` therefore separates the two, and on
# this corpus it separates them by a factor of five with an empty gap:
#
#     0.087  US9265734   TABLE-US-00006  <- vertical
#     0.441  US8957068   TABLE-US-00025  <- next lowest, an ordinary table
#     0.505  US10995073  TABLE-US-00011
#     ...    118 more, all higher
#
# All 122 two- and three-column tables in the 137 cached patents with >=30
# populated col-0 cells were scored (2026-08-17). One matched. 0.35 sits INSIDE
# the gap, not at the edge of either population — same argument as _CID_MAX_LEN.
_VERT_MAX_LABEL_UNIQ = 0.35
# Below this the ratio is noise: a 6-row table of three repeated labels scores
# 0.5 by accident. Also what keeps US20250170122 TABLE-US-00004 out — a 366-row
# STRUCTURE GALLERY whose col 0 is 366 <chemistry> refs under one 'Structure'
# label and whose block contains zero numeric tokens. It has no measurements to
# take; its 365 labelled drawings belong to the image route, not here.
_VERT_MIN_LABEL_ROWS = 30
_VERT_MIN_ANCHOR_HITS = 5

# A parenthesised label on its own line is the typesetter wrapping the previous
# label, not a field: US9265734 prints `HDAC1 IC50` and `(nM)` as two rows, the
# second with an empty value cell. Joining them is what lets `_unit_from` read
# the unit off the label — without it every one of the 371 values is unitless
# and `is_usable` is False for all of them.
_VERT_WRAPPED_LABEL = re.compile(r"^\(.*\)$")

# `_CID_LABEL` (l.135) does not match `Comp id` — it spells the word out as
# `compounds?`. This is the vertical layout's own spelling of the same idea, and
# it is deliberately a SEPARATE pattern: widening `_CID_LABEL` would change how
# 137 patents' id columns parse, to buy one table.
_VERT_CID_LABEL = re.compile(
    r"^\s*(?:comp(?:ound)?|cpd|example|entry)\s*"
    r"(?:\.?\s*(?:id|no|number|#))?\s*[:.]?\s*$", re.I)

# THE IUPAC NAMES IN THIS LAYOUT ARE NOT READ, AND THAT IS A REAL GAP.
# Each of the 199 blocks on US9265734 TABLE-US-00006 carries a
# `Chemical_name` field holding a full IUPAC name — 199 compounds that OPSIN
# would resolve with no image recognition at all, which is the highest-yield
# route this package has. `extract_table_names` returns 0 for this patent,
# because the name track reads a row per compound too.
#
# No pattern is defined here for it on purpose. A constant nothing calls is
# the leftover this codebase refuses to keep, and the hook belongs in
# `sources/table_names.py`, not in the assay reader. Written down so the gap
# is a recorded number rather than something rediscovered later.


def _vertical_pairs(tables: list[Table]) -> list[list[str]]:
    """Every `(label, value)` row of a block, header rows INCLUDED.

    Header rows are included on purpose. `_header_rows_of` (l.734) promotes the
    leading run of label-looking rows, and on a vertical table the first such row
    is record 1's DATA: US9265734's assembled grid reports its header as
    `['Comp id', 'R119']`, which is compound R119's id. Reading the raw tgroups
    and taking every row back recovers 199 cids where the assembled view gives
    198 — and the two measurements attached to the lost one.
    """
    pairs: list[list[str]] = []
    for t in tables:
        for row in list(t.header_rows) + list(t.body_rows):
            lab = row[0].text.strip() if len(row) > 0 else ""
            val = row[1].text.strip() if len(row) > 1 else ""
            if not lab and not val:
                continue
            # A wrapped unit line carries no value of its own; fold it back onto
            # the label it belongs to rather than emitting a field named `(nM)`.
            if pairs and not val and _VERT_WRAPPED_LABEL.match(lab):
                pairs[-1][0] = f"{pairs[-1][0]} {lab}"
                continue
            pairs.append([lab, val])
    return pairs


def _vertical_anchor(pairs: list[list[str]]) -> str | None:
    """The label that starts each record, or None if none does.

    Chosen by EVENNESS, not by frequency. `(nM)` and `(M + H)` appear twice per
    record in US9265734, so cutting on the most frequent label would halve every
    block. The record boundary is the label whose occurrences are evenly spaced:
    `Structure` recurs every 12 rows, 199 times, with no exception.
    """
    counts = Counter(l for l, _ in pairs)
    for lab, n in counts.most_common():
        if n < _VERT_MIN_ANCHOR_HITS:
            break                        # most_common is descending; done
        pos = [i for i, (l, _) in enumerate(pairs) if l == lab]
        strides = [b - a for a, b in zip(pos, pos[1:])]
        if not strides:
            continue
        modal, hits = Counter(strides).most_common(1)[0]
        if modal >= 3 and hits >= 0.8 * len(strides):
            return lab
    return None


def vertical_blocks(tables: list[Table]) -> list[list[list[str]]]:
    """One compound per returned block, as `[label, value]` pairs. [] if not.

    Public because BOTH tracks lose this layout, not just the assay one:
    `table_names.extract_table_names` returns 0 for US9265734, so its 199
    `Chemical_name` cells are invisible to the identity route as well. One
    detector, called from both sides — a second implementation would be a second
    thing to keep true.
    """
    # Per TGROUP, not per block. A block's tgroups routinely disagree on column
    # count (see `Table.col_widths`), and this block is the case in point:
    # US9265734 TABLE-US-00006 opens with a 1-column caption tgroup and carries
    # all 2,595 label/value rows in the second. Gating on `tables[0].n_cols`
    # rejected the whole block on the strength of its title bar and returned 0
    # records — the fix that made this function fire, with no threshold moved.
    two = [t for t in tables if t.n_cols == 2]
    if not two:
        return []
    pairs = _vertical_pairs(two)
    labels = [l for l, _ in pairs if l]
    if len(labels) < _VERT_MIN_LABEL_ROWS:
        return []
    if len(set(labels)) > _VERT_MAX_LABEL_UNIQ * len(labels):
        return []                        # col 0 holds ids, not field names
    anchor = _vertical_anchor(pairs)
    if anchor is None:
        return []
    cut = [i for i, (l, _) in enumerate(pairs) if l == anchor]
    blocks = [pairs[s:e] for s, e in zip(cut, cut[1:] + [len(pairs)])]
    # A repeating label column with no measurements in it is a characterisation
    # listing or a structure gallery, and importing it would re-add exactly the
    # noise column classification exists to remove. Same gate, same reason, as
    # the caption fallback in `build_columns` (l.1318).
    lem = _assay_lemma_re()
    scored = sum(1 for b in blocks
                 if any(v and lem.search(l.lower()) and parse_value(v) for l, v in b))
    return blocks if scored >= 0.5 * len(blocks) else []


def extract_vertical(tables: list[Table], *, table_id: str) -> list[AssayRecord]:
    """Assay records from a vertical-record block.

    US9265734 TABLE-US-00006 publishes 199 HDAC inhibitors like this:

        Record 1      | -
        Structure     | <chemistry>
        Comp id       | R119
        HDAC1 IC50    | 7000
        (nM)          | -
        HDAC3 IC50    | 1100
        (nM)          | -
        Chemical_name | N-(2-aminophenyl)-6-(phenylsulfonamido)hexanamide

    Measured yield of this function on that block: 199 blocks, 199 cids, 371
    usable records (HDAC3 199, HDAC1 172 — 27 blocks leave HDAC1 blank), unit
    `nM` on all 371, read from the folded label. The patent today produces 47
    records over 10 distinct cids; 2 of the 199 cids are already among those 10,
    so this is +197 distinct compounds.
    """
    out: list[AssayRecord] = []
    lem = _assay_lemma_re()
    for block in vertical_blocks(tables):
        cid = ""
        for lab, val in block:
            if val and _VERT_CID_LABEL.match(lab) and _CID_PAT.match(val):
                cid = normalize_cid(val)
                break
        if not cid:
            # No id means no record. Inventing one from position would file real
            # measurements under the wrong compound, which is the failure mode
            # `_column_groups` exists to avoid (l.2119 comment) and is strictly
            # worse than losing them.
            continue
        for lab, val in block:
            if not val or not lem.search(lab.lower()):
                continue
            parsed = parse_value(val)
            if not parsed:
                continue
            # The unit is in the label's own parenthetical, which `_vertical_pairs`
            # folded back on. HEADER FIRST is the same precedence rule
            # `build_columns` applies (l.1311): a label is a statement about this
            # field, a caption is prose about the biology.
            unit = parsed.get("unit") or _unit_from(lab)
            name = re.sub(r"\s*\([^)]*\)\s*$", "", lab).strip()
            out.append(AssayRecord(
                cid=cid,
                assay_name=name or "unnamed assay",
                value_numeric=parsed.get("value_numeric"),
                qualifier=parsed.get("qualifier"),
                unit=unit,
                letter_grade=parsed.get("letter_grade"),
                value_text=parsed.get("value_text", ""),
                table_id=table_id,
                column_header=lab,
                source="uspto_xml_vertical",
            ))
    return out


def _best_per_block(raw: list[Table]) -> list[Table]:
    """Per `<tables>` block, whichever view yields more usable records.

    The two views are the block's raw tgroups (each parsed on its own, headers
    inherited from the previous one) and the single grid `assemble_block` builds
    from them. Neither dominates: assembly is required wherever a block is
    fragmented per compound, and loses wherever a block's columns are spanned
    without `namest`. Scored on usable records — the completeness contract —
    rather than on row or record count, because a restructuring that doubles the
    record count while stripping the unit off every one of them is not a win.
    """
    from .uspto_xml import assemble_block

    order: list[str] = []
    for t in raw:
        if t.table_id not in order:
            order.append(t.table_id)

    out: list[Table] = []
    for tid in order:
        frag = [t for t in raw if t.table_id == tid]
        merged = assemble_block(raw, tid)
        if merged is None:
            out.extend(frag)
            continue
        n_frag = sum(1 for r in extract_from_tables(frag) if r.is_usable)
        n_merged = sum(1 for r in extract_from_tables([merged]) if r.is_usable)
        if n_merged >= n_frag:
            out.append(merged)
        else:
            out.extend(frag)
    return out


# How many populated cells a row may have and still be read as legend text.
_LEGEND_MAX_CELLS = 3


def _is_inverted_block(tables) -> bool:
    """Does this block list a GRADE and then the compounds that scored it?

    The shape is unmistakable and does not depend on finding a scale: a row
    holding a bin symbol, and rows holding runs of compound ids. Two such rows
    are required so that a stray `+` beside a single id cannot qualify a table
    that is not inverted at all.
    """
    # THE SYMBOL AND THE LIST MUST SHARE A ROW. Looking for a symbol anywhere
    # in the block and an id list anywhere else matches an ordinary
    # row-per-compound table too: US10172859 TABLE-US-00009 grades compounds
    # `B`/`A`/`A` and its caption names six examples, which satisfied both
    # halves separately and minted 54 records under `assay (binned)` for a
    # table that is not inverted at all.
    #
    # In an inverted table the grade and the compounds it applies to are
    # printed side by side — `['++', 'A028, A075, A076, ...']` — and that
    # pairing is the shape, not the ingredients.
    for t in tables:
        for row in list(t.header_rows) + list(t.body_rows):
            cells = [c.text.strip() for c in row]
            if not any(_BIN_SYMBOL.match(c) for c in cells):
                continue
            if any(_is_id_list_cell(c) for c in cells):
                return True
    return False


def _is_id_list_cell(cell: str) -> bool:
    """Is this cell A LIST OF COMPOUND IDS, rather than text containing some?

    `_cid_list` pulls ids out of any text, which is what it is for. Here that
    is too generous: a chemical name carries a locant run, and
    `[4-Fluoro-3-[7-(2,2,3,3,5,5,6,6-octadeuterio-morpholin-4-...` yields
    `['2','3','3','5','5','6']` — six "compounds" that are ring positions.
    Beside a graded cell in the same row that satisfied the inverted-table
    shape and minted 54 records for an ordinary row-per-compound table.

    An inverted table's cell IS the list: `A028, A075, A076, A087, ...` and
    almost nothing else. So the ids have to account for the cell, not merely
    appear in it.
    """
    ids = _cid_list(cell)
    if len(ids) < 3:
        return False
    return len("".join(ids)) >= 0.5 * len(re.sub(r"[\s,;]", "", cell or ""))


def _legend_lines(tables) -> list[str]:
    """A block's rows in document order, each joined across its cells.

    A legend is laid out as a table as often as it is written as a sentence,
    and when it is, the symbol and its range are usually different CELLS of the
    same row. Testing cells one at a time cannot see such a key: neither `A:`
    nor `IC50 < 3 nM` is a key, and their concatenation is.

    Order is preserved because `parse_sectioned_key` reads it — a heading
    governs the rows BELOW it, so shuffling the rows re-assigns the scales.

    Only NARROW rows are joined, and that limit is what keeps the join safe.
    Joining a data row fabricates keys: US20250163063 grades four kinase
    columns and prints a literal `>10 uM` beside each grade, so the joined row
    reads `... A >10 uM ...` — which is exactly the shape of `A ≦ 10 nM`, a
    symbol followed by a comparison. Every data row in the block then defined
    every grade, all four came out as `>10 uM`, and 2,977 records took the
    first one. A legend row carries a symbol and its meaning and nothing else;
    a data row carries a compound and its results. Three populated cells is the
    line between them, and every legend shape in this corpus sits under it —
    `['', 'A:', 'IC50 < 3 nM']` is two, and a full-span footnote row is one.
    """
    return [" ".join(cells) for cells in _legend_rows(tables)]


def _legend_rows(tables) -> list[list[str]]:
    """The same rows, kept as CELLS.

    A legend table's two orders — symbol first or range first — can only be
    told apart cell by cell, so `parse_bin_table` needs the row unflattened.
    See `_legend_lines` for why only narrow rows qualify.
    """
    out: list[list[str]] = []
    for t in tables:
        for row in t.body_rows:
            cells = [c.text.strip() for c in row if c.text and c.text.strip()]
            if cells and len(cells) <= _LEGEND_MAX_CELLS:
                out.append(cells)
    return out


def _prose_sections_for(prose, records, block_id) -> dict[str, dict]:
    """{column header: key} when a block's prose states one scale per column.

    Returns {} unless the prose introduces two or more, so the ordinary
    single-key path is untouched by this.
    """
    from . import bin_legend

    sections: dict[str, dict] = {}
    for part in prose:
        for heading, body in bin_legend.split_prose_sections(part or ""):
            bins = bin_legend.parse_bin_key(body)
            if bins:
                sections.setdefault(heading, bins)
    if len(sections) < 2:
        return {}
    columns = sorted({r.assay_name for r in records
                      if r.table_id == block_id and r.letter_grade and r.assay_name})
    pairing = bin_legend.assign_sections(columns, list(sections))
    return {col: sections[h] for col, h in pairing.items()}


def _apply_per_column(records, block_id, per_column) -> None:
    """Attach each column's own scale, and never another column's."""
    from . import bin_legend

    if not per_column:
        return
    for r in records:
        if r.table_id != block_id or not r.letter_grade:
            continue
        if r.range_lo is not None or r.range_hi is not None:
            continue
        br = (per_column.get(r.assay_name) or {}).get(r.letter_grade)
        if br is None or not bin_legend.compatible(r.unit, br.unit, r.column_header):
            continue
        r.range_lo, r.range_hi = br.lo, br.hi
        if br.unit:
            r.unit, r.unit_source = br.unit, "bin_key"


def _document_legend(by_block, records) -> dict[str, dict]:
    """Per-column bin scales stated ONCE for the whole document.

    A block-local key is always preferred and is resolved first; this only
    fills what that left empty. The distinction matters because a document may
    redefine a symbol per table, and the nearest definition is the right one —
    so this reads only blocks that publish a scale and no data of their own.

    US10172859 prints `An overview of the working examples is given by Tables
    1-7. The following ranges apply to the biological data reproduced therein:`
    and then one legend block, 31,600 characters before the first data row. The
    key was never in scope: keys resolve per block, and the look-back is 6,000
    characters. The legend was parsed, produced nothing, and cost 1,293
    records that the patent defines in full on its own first page.
    """
    from . import bin_legend

    graded_blocks = {r.table_id for r in records if r.letter_grade}
    sections: dict[str, dict] = {}
    for block_id, raw_block in by_block.items():
        # A block that carries data of its own is not a document-scope legend;
        # whatever key it states belongs to its own rows and was already used.
        if block_id in graded_blocks:
            continue
        found = bin_legend.parse_sectioned_key(_legend_lines(raw_block))
        for heading, key in found.items():
            # LAST definition wins, matching `nearest_key_before`: a scale
            # restated later in the document supersedes the earlier one.
            sections[heading] = key
        # A WHOLE BLOCK that is one scale, named by its own caption. US9221791
        # publishes two, as `value | Rating` tables — `In each case of Table 3
        # the Septoria rating scale is as follows:` and the same for Puccinia —
        # and Table 3 then has a `Septoria Rating` and a `Puccinia Rating`
        # column. Nothing inside either legend says which is which; the caption
        # is the only place the document says it.
        caption = (raw_block[0].caption or "").strip()
        if caption:
            # The heading rows name the quantity AND its unit; the body rows
            # carry bare numbers under them.
            head = " ".join(c.text for t in raw_block for r in t.header_rows
                            for c in r if c.text)
            table_key = bin_legend.parse_bin_table(_legend_rows(raw_block),
                                                   unit_hint=head)
            if table_key:
                sections.setdefault(caption, {}).update(table_key)
    return sections


def extract_from_patent(xml: str) -> list[AssayRecord]:
    """Full extraction for one patent.

    Three passes, because patents publish assay data in three shapes:
    (see `_legend_lines` for how a legend's rows are read)
      1. row per compound, value in a cell        (the common case)
      2. row per potency bin, compound ids listed (inverted; US11566007)
      3. bin symbols in a normal table, defined by a legend (US11292791)

    Bin keys are resolved per `<tables>` block and never shared between them —
    two patents in this corpus assign incompatible ranges to `++++`.
    """
    from . import bin_legend
    from .uspto_xml import description_text, parse_tables

    raw = parse_tables(xml)
    # One assembled grid per <tables> block. A block is fragmented into many
    # tgroups and no single one of them is the table; see `assemble_block`.
    #
    # Assembly is applied per block and only where it WINS. Some blocks encode
    # column spans that survive the per-tgroup path and are unrecoverable once
    # merged: US8952177 TABLE-US-00003 writes "FLAP Binding wild / Human Whole
    # Blood" as two colspan-1 cells that actually span two sub-columns each,
    # and CALS `namest` is absent, so the pairing exists nowhere in the source.
    # Assembling that block costs 169 units; assembling US10172859 gains 498
    # rows. Choosing per block keeps both — and the same anti-deletion rule the
    # repair loop applies to model-proposed rules applies here: a restructuring
    # must add usable records, never remove them.
    tables = _best_per_block(raw)
    records = extract_from_tables(tables)

    assembled = {t.table_id: t for t in tables}

    # Resolve a bin key for each <tables> block from its own legend, caption or
    # trailing footnote. The key often sits AFTER the data it explains.
    #
    # Legend text is harvested from the RAW tgroups, not the assembled grid:
    # assembly keeps only the block's data width, and a legend is routinely
    # published as a narrow fragment alongside it. Records, by contrast, come
    # from the assembled grid.
    by_block: dict[str, list[Table]] = {}
    for t in raw:
        by_block.setdefault(t.table_id, []).append(t)

    for block_id, raw_block in by_block.items():
        block = [assembled[block_id]] if block_id in assembled else raw_block
        # ...INCLUDING the prose immediately before and after the block.
        #
        # Each of US11566007's bin tables states its own key, and eleven of them
        # state it in the paragraph above rather than in a caption or a cell.
        # Harvesting only caption + legend + cells found the key for two blocks
        # and missed it for eleven, which read as an unreadable layout.
        #
        # It is NOT a transfer from the blocks that worked, and that distinction
        # is worth 10x: TABLE-US-00006 defines a FOUR-grade scale where `++++` is
        # `IC50 >= 1 uM`, while TABLE-US-00007 defines a FIVE-grade scale where
        # `++++` is `10 uM > IC50 >= 1 uM`. Carrying one key to the other looked
        # like a free +2,261 records and would have silently rewritten an
        # upper-bounded bin as an unbounded one. Each block's own key is the only
        # safe answer, and each block has one.
        #
        # Rows are joined ACROSS THEIR CELLS before the key test, and that join
        # is the whole reason US10172859 read no scale at all. Its legend is a
        # three-column table: the symbol is in one cell and its range in the
        # next, so `A:` and `IC50 < 3 nM` were tested separately and neither
        # half is a key. Testing the joined row — `A: IC50 < 3 nM` — reads it.
        # The cells themselves are still offered, because a legend written as
        # one cell per grade is equally common.
        #
        # ORDER IS PRECEDENCE, nearest first, and the block's OWN ROWS are
        # nearest of all. Two blocks settle the order between them, and the
        # evidence is the data rather than the layout: US11566007
        # TABLE-US-00006 holds 221 records graded `+++++`, which the FIVE-grade
        # key in its own footer explains and the four-grade key printed above
        # it cannot — that key belongs to the table before. US20230365584A1
        # TABLE-US-00014 is the same shape: its own footer says `C ≥ 1000 nM`
        # while the key above it, for the previous table, says `C ≥ 10 nM`.
        # The caption comes last because it is not a caption so much as a blob
        # of the text around the table, and routinely spans its neighbours.
        key_lines = [ln for ln in _legend_lines(raw_block)
                     if bin_legend.looks_like_key(ln)]
        prose = [
            " ".join(table_legend(t) for t in raw_block),
            bin_legend.nearest_key_before(raw_block[0].preceding or ""),
            raw_block[0].caption,
        ]
        text = " ".join([*key_lines, *(p for p in prose if p)])
        # The shape of the block counts as much as the presence of a key.
        # `looks_like_key` asks whether a SCALE is in scope, and for an
        # inverted table that is the wrong question first: the grades and the
        # compounds they apply to are in the table itself.
        if not bin_legend.looks_like_key(text) and not _is_inverted_block(block):
            continue
        # The block's own rows first, each read alone; then the prose sources
        # in order of distance. Rows are a list and prose is a paragraph, and
        # they have to be read that way round — see `parse_bin_key_lines`.
        key = bin_legend.parse_bin_table(_legend_rows(raw_block))
        for sym, br in bin_legend.parse_bin_key_lines(key_lines).items():
            key.setdefault(sym, br)

        # PROSE THAT STATES SEVERAL SCALES, one per column. Read as one key
        # they contest each other and none survives, which is right but
        # needlessly poor: the prose names the column each governs.
        # See `split_prose_sections` and `assign_sections`.
        per_column = _prose_sections_for(prose, records, block_id)
        for sym, br in bin_legend.parse_bin_key_layered(prose).items():
            key.setdefault(sym, br)
        # A GRADE ASSIGNMENT IS DATA, EVEN WITHOUT ITS SCALE. This gate threw
        # the whole block away when no key resolved — so a table that plainly
        # states `++ → A028, A075, A076, …` produced nothing at all, rather
        # than the grades it prints.
        #
        # US11566007 is the case. Its inverted tables carry hundreds of
        # compound ids inline, so for the later blocks the key — printed before
        # the PREVIOUS table — sits beyond the 6,000-character look-back and is
        # never seen. TABLE-US-00008 finds it and yields 825 records;
        # TABLE-US-00009 has the identical layout, does not find it, and
        # yielded 0. The repair loop diagnosed that itself, refused to buy a
        # rule for it, and escalated as "INCONSISTENT HANDLING — not a layout
        # gap": the same fingerprint cannot be both readable and unreadable.
        #
        # The range is a SECOND fact about a grade, not a precondition for it.
        # Grade-only records are already emitted in thousands elsewhere; only
        # this path required the scale before it would admit the assignment.
        if not key and not per_column and not _is_inverted_block(block):
            continue
        _apply_per_column(records, block_id, per_column)
        name, unit = caption_assay_hint(raw_block[0].caption)
        name = name or _assay_name_from(text) or "assay (binned)"
        records.extend(extract_inverted(
            block, key, assay_name=name,
            unit=unit or next((b.unit for b in key.values() if b.unit), None),
            table_id=block_id,
        ))
        # Bin symbols sitting in ordinary row-per-compound tables.
        for r in records:
            if r.table_id == block_id and r.letter_grade and r.range_lo is None \
                    and r.range_hi is None and r.letter_grade in key:
                br = key[r.letter_grade]
                # The column and the key must measure the same KIND of thing.
                # A block may state two scales, and the one that parsed is not
                # necessarily the one that governs this column.
                if not bin_legend.compatible(r.unit, br.unit, r.column_header):
                    continue
                r.range_lo, r.range_hi = br.lo, br.hi
                # The bounds come from the key, so the unit must come from the
                # key too. Letting a column-derived unit stand over a key-
                # derived range pairs a number with a scale from a different
                # statement: US11485738 defines "A=<100 nM" and had 233 records
                # reading 0-100 mM — the right interval on a scale 10^6 out.
                if br.unit:
                    r.unit, r.unit_source = br.unit, "bin_key"
                elif not r.unit:
                    r.unit = br.unit

    # PASS 3b: a legend block that governs the whole document.
    #
    # Runs LAST of the bin passes and only fills a range that is still empty,
    # so a block that stated its own key keeps it. Nearest definition wins;
    # this is the fallback for a document that states the scale once.
    #
    # A record is matched to a section BY THE NAME of the column it came from,
    # and a column matching no section keeps no range. That refusal is the
    # point. US10172859 defines `A`-`D` three times, once per assay, and `B` is
    # 3-7 nM, 0.5-5 uM or 15-25 uM depending on which column it sits in — so
    # the single-key path that works everywhere else would be wrong here by
    # three orders of magnitude, uniformly, with nothing in the output to show
    # for it. A single-section legend is applied without a name match, because
    # one scale is not ambiguous; two or more must be told apart or dropped.
    doc_sections = _document_legend(by_block, records)
    if doc_sections:
        only = next(iter(doc_sections.values())) if len(doc_sections) == 1 else None
        for r in records:
            if not r.letter_grade or r.range_lo is not None or r.range_hi is not None:
                continue
            key = only
            if key is None:
                heading = bin_legend.section_for_column(
                    r.column_header or r.assay_name or "", doc_sections)
                if heading is None:
                    continue
                key = doc_sections[heading]
            br = key.get(r.letter_grade)
            if br is None or not bin_legend.compatible(r.unit, br.unit, r.column_header):
                continue
            r.range_lo, r.range_hi = br.lo, br.hi
            if br.unit:
                r.unit, r.unit_source = br.unit, "bin_key"

    # PASS 4: vertical-record blocks — one compound per RUN of rows.
    #
    # Gated on the block having produced nothing usable, so this can only ADD.
    # That gate is the whole regression argument: on the 137 cached patents the
    # detector matches exactly one block (US9265734 TABLE-US-00006), and that
    # block's current usable count is 0, so no existing record can be displaced,
    # duplicated or relabelled. It is also the same anti-deletion rule
    # `_best_per_block` applies (l.1561) and the repair loop applies to
    # model-proposed rules: a restructuring must add usable records, never
    # remove them.
    #
    # RAW tgroups, not the assembled grid. `_best_per_block` -> `_header_rows_of`
    # promotes this block's first data row (`Comp id | R119`) to a header and the
    # compound is lost with it; measured, that is 198 cids instead of 199.
    usable_by_block = Counter(r.table_id for r in records if r.is_usable)
    for block_id, raw_block in by_block.items():
        if usable_by_block.get(block_id):
            continue
        records.extend(extract_vertical(raw_block, table_id=block_id))

    unitless = sorted({r.assay_name for r in records if not r.unit})
    if unitless:
        resolved = infer_units_from_description(description_text(xml), unitless)
        for r in records:
            if not r.unit and r.assay_name in resolved:
                r.unit = resolved[r.assay_name]
                r.unit_source = "description"
    return records


def _assay_name_from(text: str) -> str | None:
    """Best-effort assay name from a bin table's own heading text."""
    m = re.search(r"([A-Za-z0-9 /()\-]{4,60}?(?:IC\s*50|EC\s*50|Ki|Kd))", text)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None



# How many candidate headers to remember per column width. A block usually
# inherits from the one just above it; four covers a header that is separated
# from its data by a few characterisation tables without holding the whole
# document.
_INHERIT_DEPTH = 4


def _header_fit(table: "Table", headers: list[str], data_rows) -> int:
    """How well would THIS header serve THIS block — scored, not assumed.

    Returns the number of cells that would become readable measurements under
    the given header. Zero when the header yields no compound-id column or no
    assay column, because such a header cannot produce a record here whatever
    else is true of it.

    WHY THIS EXISTS, and what it does NOT fix.

    Header inheritance was `last_header[n_cols] = headers`: document-global,
    keyed on COLUMN COUNT ALONE, last-writer-wins, with no check that the header
    had anything to do with the block receiving it. An LCMS characterisation
    header and an assay header sharing a width were interchangeable. CLAUDE.md
    records the resulting signature — US10071079's 982-row assay table "picked
    up an LCMS characterisation header, classified as [cid, structure, ms, rt,
    rt], found no assay column". That was treated as a detector bug and the
    coupling was left in place. Scoring the candidates removes it.

    It is a ROBUSTNESS guard, not a validated fix, and the distinction is worth
    recording because I first justified it with a root cause that turned out to
    be false. US10660877 loses all 860 compounds under one capability patch, and
    the reason is NOT inheritance: TABLE-US-00036 declares no header at all, so
    its `['', 'Ex. No.', 'TLR7 IC50 (nM)', ...]` comes from `_header_rows_of`
    promoting its own leading body rows. Patching `_is_namelike` changed that
    promotion and the block destroyed its own header — measured, no width-5
    block produces an IC50 header afterwards, so there was nothing left for any
    chooser to choose. The blast radius of `_is_namelike` is the block itself.

    Measured on this corpus the change moves nothing (32,237 compounds before
    and after), which is what a guard against a latent coupling should do.
    """
    if not any(headers):
        return 0
    try:
        cols = build_columns(table, inherited=headers, data_rows=data_rows)
    except Exception:                            # a bad candidate scores zero
        return 0
    kinds = [c.kind for c in cols]
    if CID not in kinds or ASSAY not in kinds:
        return 0
    cid_i = kinds.index(CID)
    assay_i = [c.index for c in cols if c.kind == ASSAY]
    hits = 0
    for row in list(data_rows)[:20]:
        if len(row) <= cid_i or not _CID_PAT.match(row[cid_i].text.strip()):
            continue
        for i in assay_i:
            if len(row) > i and parse_value(row[i].text):
                hits += 1
    return hits


def _choose_inherited(candidates: list[list[str]], own_block: list[str] | None,
                      table: "Table", data_rows) -> list[str] | None:
    """The header a block should inherit: whichever actually reads its cells.

    Same-block (`by_table_id`) candidates are tried alongside same-width ones
    rather than after them, and the winner is the one that turns the most cells
    into measurements. Ties keep the most recent, which is the old behaviour —
    so a block with only one plausible header behaves exactly as before.
    """
    pool = [h for h in ([own_block] if own_block else []) + list(reversed(candidates)) if h]
    if not pool:
        return None
    if len(pool) == 1:
        return pool[0]
    best, best_score = None, -1
    for h in pool:
        sc = _header_fit(table, h, data_rows)
        if sc > best_score:
            best, best_score = h, sc
    # Every candidate scored zero: none of them can produce a record here, so
    # this is not a choice between headers. Keep the most recent same-width one
    # and let the gap detector report the block, exactly as before.
    return best if best_score > 0 else pool[-1 if own_block is None else 0]


# ── repeating column groups ───────────────────────────────────────

# Relaxed CID pattern: accepts bare integers with optional trailing period,
# e.g. "4." or "12" — these appear when compound numbers are listed without
# a prefix letter. The standard _CID_PAT handles alphanumeric ids like "Cpd-4".
_BARE_INT_CID = re.compile(r'^\d+\.?$')

# Unicode typographic quotation marks that some patents wrap around compound
# ids, e.g. “C1” (U+201C LEFT DOUBLE QUOTATION MARK / U+201D RIGHT
# DOUBLE QUOTATION MARK). Stripping these before id matching lets the
# standard _CID_PAT recognise the underlying id.
_TYPO_QUOTES = '“”‘’«»„‟'

# Both constants above were local to `extract_from_tables`; they are module
# level now because `_column_groups` asks the same question of the same cells
# and a second copy of an id pattern is a second place to get it wrong.

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")

# Evidence thresholds for calling a column an id column. Two fractions, not
# one, because they answer different questions and a single number cannot do
# both jobs. US11136320 TABLE-US-00011 is why: its second id column ends with
# "Reference compound (roblitinib)" spelled down three XML rows, so only 11 of
# its 14 non-empty cells (0.79) parse as ids — while 11 of the 11 that DO parse
# are the same family. A single 0.8 floor over all cells declined that table;
# the split fraction accepts it and still declines a column of prose.
_GROUP_ID_FRACTION = 0.6    # of the non-empty cells, how many parse as an id
_GROUP_FAMILY_PURITY = 0.9  # of the ids that parse, how many share one family
_GROUP_MIN_IDS = 3          # absolute floor — two rows is a coincidence


def _id_family(text: str) -> str:
    """The non-numeric skeleton of an id: `I-0268` and `I-1607` are both `I-#`.

    Zero-padding and digit count vary freely inside one column (US9718790 runs
    `I-0020` through `I-2284`), so the digits carry no family information and
    the prefix and separators carry all of it. The label is stripped first
    because `normalize_cid` strips it too — `compound 64` and `64` are the same
    compound written twice, and must be the same family.
    """
    s = _CID_LABEL.sub("", (text or "").strip().strip(_TYPO_QUOTES)).strip()
    return _DIGITS.sub("#", s)


def _id_column_family(rows, i: int, val_idx: list[int] | None = None) -> str | None:
    """The family this column's ids belong to, or None if it holds no ids.

    `val_idx` scopes the denominator to the rows that are DATA rows for this
    group — those whose own value cells are not all blank. Without it the
    fraction is measured over every non-empty cell in the column, and a
    multi-row label spelled down the column vetoes a table it has no bearing
    on: US11136320 TABLE-US-00011 trails `Reference compound (roblitinib)`
    across three XML rows, which scored 3 ids of 6 cells = 0.50 against a 0.60
    floor and declined a split whose header repeats verbatim. Two of those
    three cells sit beside an empty value, so they are not rows this group
    measures anything in. The third — `Reference` with a value — still counts
    against the fraction, which is right: it is a real cell in a real data row
    that is not an id, and it is what the 0.60 floor is for.
    """
    if val_idx:
        rows = [r for r in rows
                if any(len(r) > v and r[v].text.strip() for v in val_idx)]
    vals = [r[i].text.strip() for r in rows if len(r) > i and r[i].text.strip()]
    ids = [v for v in vals
           if _CID_PAT.match(v.strip(_TYPO_QUOTES)) or _BARE_INT_CID.match(v)]
    if len(ids) < _GROUP_MIN_IDS or len(ids) < len(vals) * _GROUP_ID_FRACTION:
        return None
    fams: dict[str, int] = {}
    for v in ids:
        f = _id_family(v)
        fams[f] = fams.get(f, 0) + 1
    fam = max(fams, key=lambda k: fams[k])
    return fam if fams[fam] >= len(ids) * _GROUP_FAMILY_PURITY else None


def _column_groups(cols: list[Column], rows) -> list[tuple[Column, list[Column]]] | None:
    """Repeating `(id, value…)` column groups, or None for an ordinary table.

    A patent with a long list of one-number results does not print a thousand
    two-column rows; it pours the list into N side-by-side copies of the same
    pair. US9718790 TABLE-US-00569..580 head six columns
    `Compound No. | P2X3 IC50 (μM)` three times over, and every row is three
    compounds:

        I-0268  0.861   I-0943  0.061   I-1607  0.035

    Read as one compound with three values that is wrong twice — I-268 scores
    the median 0.061 against BindingDB's 861 nM, and I-943 and I-1607 get no
    record at all, swallowed as values of their neighbour.

    The whole difficulty is that a FALSE positive is worse than the under-read
    it replaces: splitting an ordinary row invents compounds and files real
    measurements under them, and nothing downstream can detect that. So the
    evidence required is deliberately lopsided.

      - the header must repeat EXACTLY, and every header in the group must be
        stated. Not "similar" — identical after whitespace folding. This is
        what keeps US10125101's `Example in this invention | IC50 | Example in
        WO 2013/178575 | IC50` whole: it is a repeating group by shape, but the
        second column numbers a DIFFERENT DOCUMENT's examples, and splitting it
        would file WO 2013/178575's compound 17 as this patent's. The patent
        says so in the header, and only an exact-match test hears it.
      - the classified KINDS must repeat too, with exactly one id column and at
        least one assay column per group. `No. | Structure | No. | Structure`
        has nothing to split; `Cpd | IC50 | IC50` has no second id.
      - and the body must agree: every repeat's id column must actually hold
        ids, of the same family as the first group's. A merged or inherited
        header can make two columns read alike when the second holds prose.

    Body disagreement returns None rather than trying a coarser tiling. If the
    strongest header evidence available is contradicted by the cells, a weaker
    reading of the same header is not a better answer.
    """
    n = len(cols)
    if n < 4:
        return None
    hdr = [_WS.sub(" ", (c.header or "")).strip().casefold() for c in cols]
    kinds = [c.kind for c in cols]
    body = [r for r in rows if not _is_spacer(r)]
    if len(body) < _GROUP_MIN_IDS:
        return None

    for g in range(2, n // 2 + 1):
        if n % g:
            continue
        reps = n // g
        # A blank header tiles trivially, and trivial is not evidence.
        if not all(hdr[j] for j in range(g)):
            continue
        if any(hdr[k * g + j] != hdr[j] for k in range(1, reps) for j in range(g)):
            continue
        if any(kinds[k * g + j] != kinds[j] for k in range(1, reps) for j in range(g)):
            continue
        offs = [j for j in range(g) if kinds[j] == CID]
        if len(offs) != 1 or not any(kinds[j] == ASSAY for j in range(g)):
            continue

        off = offs[0]
        assay_offs = [j for j in range(g) if kinds[j] == ASSAY]
        fams = []
        for k in range(reps):
            fam = _id_column_family(body, k * g + off,
                                    [k * g + j for j in assay_offs])
            if fam is None:
                return None
            fams.append(fam)
        if len(set(fams)) != 1:
            return None
        return [(cols[k * g + off],
                 [cols[k * g + j] for j in range(g) if kinds[j] == ASSAY])
                for k in range(reps)]
    return None


def _extract_column_groups(table: Table, groups, rows) -> list[AssayRecord]:
    """Read a repeating-group table: one row in, one record per (id, value).

    Deliberately none of the recovery machinery `extract_from_tables` carries —
    no CID fill-down, no name-as-id buffering, no embedded-id scan. Every one of
    those infers an id for a row that does not state one, and in a table whose
    id columns interleave with value columns a wrong inference does not lose a
    record, it files a real measurement under the wrong compound. A group whose
    id cell is blank or unreadable is skipped, which is the trade this module
    makes everywhere: a missing assay is recoverable, a misattributed one is not.
    """
    out: list[AssayRecord] = []
    for row in rows:
        if _is_spacer(row):
            continue
        for cid_col, assay_cols in groups:
            if len(row) <= cid_col.index:
                continue
            raw = row[cid_col.index].text.strip().strip(_TYPO_QUOTES)
            if not raw or not (_CID_PAT.match(raw) or _BARE_INT_CID.match(raw)):
                continue
            cid = normalize_cid(raw)
            if not cid:
                continue
            for c in assay_cols:
                if len(row) <= c.index:
                    continue
                parsed = parse_value(row[c.index].text)
                if not parsed:
                    continue
                out.append(AssayRecord(
                    cid=cid,
                    assay_name=c.assay_name or c.header or "unnamed assay",
                    value_numeric=parsed.get("value_numeric"),
                    qualifier=parsed.get("qualifier"),
                    unit=parsed.get("unit") or c.unit,
                    n_runs=parsed.get("n_runs"),
                    letter_grade=parsed.get("letter_grade"),
                    value_text=parsed.get("value_text", ""),
                    table_id=table.table_id,
                    column_header=c.header,
                ))
    return out


# Pattern to extract a compound identifier embedded in a longer text cell,
# e.g. "Compound 1.001: 1H NMR ..." → "Compound 1.001".  Also matches
# common variants like "Cpd 1.001", "Cmpd-1.001", "Example 1.001".
#
# Hoisted out of `extract_from_tables` (was compiled fresh on every call to
# that function, including the two recursive `_best_per_block` scoring calls
# per block, over every patent in the corpus) — same pattern, same flags.
_EMBEDDED_CID = re.compile(
    r'(?:Compound|Cpd|Cmpd|Example|Ex\.?)'
    r'[\s.#\-]*'
    r'(\d+(?:[\._ ]\d+)?(?:[a-zA-Z])?)',
    re.IGNORECASE,
)

# Pattern to extract embedded assay values from free-text NMR/MS cells.
# Matches substrings like:
#   "FXR EC50 (nM) = 1497"
#   "EC50 (nM) = 167"
#   "IC50 (uM) = 0.045"
# The assay name is captured as group 1, the unit as group 2, and the
# numeric value as group 3. This is a general convention in patent tables
# that bundle characterisation data into a single prose cell.
_EMBEDDED_ASSAY = re.compile(
    r'([A-Za-z][A-Za-z0-9 _/-]{0,30}?'
    r'(?:IC|EC|ED|GI|CC|LC)\s*50'
    r'[A-Za-z0-9 _/-]{0,10}?)'
    r'\s*\(\s*([a-zA-Z%/]+)\s*\)'
    r'\s*=\s*'
    r'(\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)',
    re.IGNORECASE,
)

# Unit symbols as they appear inside an `_EMBEDDED_ASSAY` match, normalised.
# A distinct, smaller table from `_MOL_UNIT`/`_CASED_UNIT` above (those cover
# header/legend/caption unit text; this covers only what shows up inside the
# embedded "NAME (unit) = value" prose match) — kept separate rather than
# merged into either. Was a dict literal rebuilt inside `extract_from_tables`
# on every `_EMBEDDED_ASSAY` match, i.e. per embedded reading found, not per
# call; hoisted here for the same reason `_MOL_UNIT`/`_CASED_UNIT` were.
_EMBEDDED_UNIT = {
    'nm': 'nM', 'um': 'uM', 'mm': 'mM', 'pm': 'pM',
    'uM': 'uM', 'nM': 'nM', 'mM': 'mM', 'pM': 'pM',
}

# Chemical-name pattern: used to detect name-as-id tables where the full
# IUPAC name is used as the compound identifier.
_CHEM_NAME = re.compile(
    r'(?:yl|phenyl|methyl|imidazol|pyrimidin|morpholin|triazol|'
    r'chloro|bromo|fluoro|ethyl|propyl|cyclo|oxo|amino|nitro|'
    r'benz|piperid|piperaz|pyridine|pyrazol|oxazol|thiazol)',
    re.IGNORECASE)


def _is_chem_name(s: str) -> bool:
    """Does this string look like a chemical/IUPAC compound name?"""
    return (len(s) > 15
            and (bool(_CHEM_NAME.search(s))
                 or (s.count('-') + s.count('[') + s.count('(')) >= 3))


def extract_from_tables(tables: list[Table]) -> list[AssayRecord]:
    """Turn a patent's CALS tables into assay records.

    Continuation tables (same column count, no header of their own) inherit the
    most recent header — patents split one logical table across many tgroups
    when it spans pages, and the later pieces carry no header at all.

    Multi-assay columns: when a column header is a comma-separated list of
    assay names (e.g. "probe 1, probe 2") and the data cells are also
    comma-separated, each sub-value is emitted as a separate AssayRecord paired
    with its corresponding sub-name.

    Relaxed CID matching: some tables use bare integers with a trailing period
    ("4.", "5.") as compound identifiers. These are accepted as CIDs when the
    CID column was classified as such and the value matches a relaxed pattern
    (digits optionally followed by a period, or the standard _CID_PAT).

    Typographic-quote-wrapped CIDs: some tables (e.g. US10570116) wrap compound
    identifiers in Unicode left/right double quotation marks (U+201C/U+201D),
    e.g. \u201cC1\u201d. These quotes are stripped from the raw CID cell text
    before pattern matching and normalisation so that the id is recognised and
    the record is emitted.

    CID fill-down: when a compound has multiple sub-rows (e.g. enantiomers,
    racemates, salt forms) the ID is printed only on the first sub-row and
    left blank on the rest. The last-seen CID is carried forward so that
    sub-rows with an empty CID cell still produce records. The carry resets
    at each new table to avoid leaking across unrelated tables.

    Embedded CID extraction: when the CID column cell is blank but another
    column (typically NMR/MS text) contains an embedded compound identifier
    like "Compound 1.001: ...", the CID is extracted from that cell. This
    handles tables where the structure/image column is empty and the compound
    name only appears inside the characterisation data cell.

    Wrapped-name rows: some tables (e.g. US9018217 TABLE-US-00001) use the
    full IUPAC compound name as the identifier, split across two consecutive
    XML rows. The first row has the name fragment in col 0 and the value in
    col 1; the second has the name continuation in col 0 and col 1 blank.
    When the CID column cell does not match any short-id pattern but is a long
    chemical name string, the cell text is used as the CID directly (name-as-id).
    Consecutive rows where the assay columns are all blank are treated as
    name-continuation rows and their col-0 text is appended to the preceding
    name to form the full compound name.

    Embedded-assay extraction: some tables (e.g. US10730863 TABLE-US-00009)
    bundle NMR, EC50 and MS data into a single free-text cell in a column
    whose header contains "NMR". The column classifies as NMR and is skipped
    by the normal assay-column loop. After that loop, any column typed NMR
    (or MS) is scanned for substrings of the form
    "[assay_name] (unit) = value" (e.g. "FXR EC50 (nM) = 1497") and records
    are emitted for each match. This is a general convention — the same
    pattern appears in every table of this patent and recurs across the corpus.
    """
    out: list[AssayRecord] = []
    last_header: dict[int, list[list[str]]] = {}
    by_table_id: dict[str, list[str]] = {}
    unit_by_table_id: dict[str, str] = {}

    for t in tables:
        hdr_rows, data_rows = _split_rows(t)
        headers = merge_header(t, hdr_rows)
        # A legend/caption unit stated once in a <tables> block applies to the
        # data tgroups that follow it under the same id.
        _u = _unit_from(table_legend(t)) or _unit_from(t.caption)
        if _u and _u != "%":
            unit_by_table_id[t.table_id] = _u
        if any(headers):
            # A LIST, not a slot. The single slot meant the most recent header
            # of a given width silently claimed every later block of that
            # width; see `_header_fit`.
            last_header.setdefault(t.n_cols, []).append(headers)
            del last_header[t.n_cols][:-_INHERIT_DEPTH]
            # Scope cross-width inheritance to the SAME `<tables>` element.
            # A patent splits one logical table into a header tgroup and a data
            # tgroup of different widths under a single id (US8952177: 3-column
            # header, 5-column data). Inheriting by "most recent header of any
            # width" instead leaks a header onto an unrelated later table —
            # it stamped `CBP IC50` onto US11292791's `+`/`++` bins, which is
            # mislabelled data, strictly worse than no label at all.
            by_table_id[t.table_id] = headers
        cols = build_columns(
            t,
            inherited=_choose_inherited(last_header.get(t.n_cols) or [],
                                        by_table_id.get(t.table_id), t, data_rows),
            inherited_unit=unit_by_table_id.get(t.table_id),
            data_rows=data_rows)

        cid_col = next((c for c in cols if c.kind == CID), None)
        assay_cols = [c for c in cols if c.kind == ASSAY]
        # Columns typed NMR or MS may contain embedded assay values in
        # free-text prose cells (e.g. "FXR EC50 (nM) = 1497"). Collect them
        # for the embedded-assay scan below.
        prose_cols = [c for c in cols if c.kind in (NMR, MS)]
        if cid_col is None or (not assay_cols and not prose_cols):
            continue

        # Repeating `(id, value)` column groups get their own reader, BEFORE
        # any of the recovery machinery below runs. `_column_groups` returns
        # None for an ordinary table, so this is inert on every layout that
        # is not a side-by-side list.
        #
        # It has to come first because the machinery below is built on the
        # assumption that a row is one compound: CID fill-down, name-as-id
        # buffering and the embedded-id scan all infer an id for a row that
        # does not state one. On a table whose id columns interleave with
        # value columns that assumption is false, and a wrong inference there
        # does not lose a record — it files a real measurement under the
        # wrong compound. US9718790 read `I-0268 0.861 I-0943 0.061 I-1607
        # 0.035` as I-268 = {0.861, 0.061, 0.035}, taking the median 0.061
        # against BindingDB's 861 nM, while I-943 and I-1607 got no record at
        # all — one wrong value and two compounds swallowed, from one row.
        groups = _column_groups(cols, data_rows)
        if groups:
            out.extend(_extract_column_groups(t, groups, data_rows))
            continue

        # CID fill-down: carry the last-seen CID forward within one table so
        # that sub-rows (enantiomers, racemates, salt forms) whose CID cell is
        # blank still produce records.  Reset per table.
        prev_cid: str | None = None

        # Wrapped-name state: buffer for name-as-id tables where the compound
        # name spans two XML rows. prev_name_parts accumulates name fragments;
        # prev_name_value holds the parsed value from the first fragment row;
        # prev_name_assay_col holds the assay column for that value.
        prev_name_parts: list[str] = []
        prev_name_value: dict | None = None
        prev_name_assay_col = None

        def _flush_wrapped(table_id=t.table_id):
            """Emit any buffered wrapped-name record and reset the buffer."""
            nonlocal prev_name_parts, prev_name_value, prev_name_assay_col
            if prev_name_parts and prev_name_value is not None and prev_name_assay_col is not None:
                full_name = " ".join(prev_name_parts).strip()
                c = prev_name_assay_col
                parsed = prev_name_value
                out.append(AssayRecord(
                    cid=full_name,
                    assay_name=c.assay_name or c.header or "unnamed assay",
                    value_numeric=parsed.get("value_numeric"),
                    qualifier=parsed.get("qualifier"),
                    unit=parsed.get("unit") or c.unit,
                    n_runs=parsed.get("n_runs"),
                    letter_grade=parsed.get("letter_grade"),
                    value_text=parsed.get("value_text", ""),
                    table_id=table_id,
                    column_header=c.header,
                ))
            prev_name_parts = []
            prev_name_value = None
            prev_name_assay_col = None

        # Detect name-as-id table: CID column holds long chemical names rather
        # than short alphanumeric ids. Check the first populated CID cells.
        _name_as_id = False
        _sample_cid_vals = [
            r[cid_col.index].text.strip()
            for r in data_rows
            if not _is_spacer(r) and len(r) > cid_col.index
            and r[cid_col.index].text.strip()
        ][:20]
        if _sample_cid_vals:
            _long = sum(1 for v in _sample_cid_vals if _is_chem_name(v))
            _short = sum(1 for v in _sample_cid_vals
                         if _CID_PAT.match(v) or _BARE_INT_CID.match(v)
                         or _CID_PAT.match(v.strip(_TYPO_QUOTES)))
            if _long > 0 and _short == 0:
                _name_as_id = True

        for row in data_rows:
            if _is_spacer(row) or len(row) <= cid_col.index:
                continue
            raw_cid = row[cid_col.index].text.strip()

            # --- Name-as-id (wrapped compound name) handling ---
            # When the table uses full IUPAC names as identifiers, the name may
            # wrap across two consecutive XML rows. The first row has the name
            # fragment and the value; the second has the name continuation and
            # a blank value cell. We buffer fragments and emit when we see a new
            # name fragment with a value (or at end of table).
            if _name_as_id:
                # Check if any assay column has a parseable value in this row
                row_value_parsed = None
                row_value_col = None
                for c in assay_cols:
                    if len(row) > c.index and row[c.index].text.strip():
                        pv = parse_value(row[c.index].text)
                        if pv:
                            row_value_parsed = pv
                            row_value_col = c
                            break

                if raw_cid and row_value_parsed is not None:
                    # New compound: flush any buffered name, start new buffer
                    _flush_wrapped()
                    prev_name_parts = [raw_cid]
                    prev_name_value = row_value_parsed
                    prev_name_assay_col = row_value_col
                elif raw_cid and row_value_parsed is None:
                    # Continuation row: append name fragment to buffer
                    if prev_name_parts:
                        prev_name_parts.append(raw_cid)
                    # else: orphan continuation row with no preceding name, ignore
                # blank raw_cid rows in name-as-id tables are spacers, skip
                continue
            # --- End name-as-id handling ---

            # Strip typographic/Unicode quotation marks from the raw CID before
            # pattern matching. Patents like US10570116 wrap compound ids in
            # U+201C/U+201D (e.g. \u201cC1\u201d); without stripping these the
            # id pattern never matches and every data row is skipped.
            raw_cid_stripped = raw_cid.strip(_TYPO_QUOTES)

            if raw_cid_stripped:
                # Accept standard CID patterns AND bare integers with optional
                # trailing period ("4.", "12") that appear in tables where compound
                # numbers are listed without a prefix letter.
                if not (_CID_PAT.match(raw_cid_stripped) or _BARE_INT_CID.match(raw_cid_stripped)):
                    continue
                cid = normalize_cid(raw_cid_stripped)
                if not cid:
                    continue
                prev_cid = cid
            else:
                # Empty CID cell — try to extract an embedded CID from other
                # columns in this row (e.g. "Compound 1.001: 1H NMR ...").
                embedded_cid: str | None = None
                for ci in range(len(row)):
                    if ci == cid_col.index:
                        continue
                    cell_t = row[ci].text.strip()
                    if not cell_t:
                        continue
                    em = _EMBEDDED_CID.search(cell_t)
                    if em:
                        # Reconstruct a CID string like "Compound 1.001"
                        raw_embedded = em.group(0).strip()
                        embedded_cid = normalize_cid(raw_embedded)
                        if embedded_cid:
                            break
                if embedded_cid:
                    cid = embedded_cid
                    prev_cid = cid
                else:
                    # Fall back to carry-forward from the last-seen CID.
                    # Only do this when the row has at least one parseable assay
                    # value, to avoid adopting spacer/annotation rows.
                    if prev_cid is None:
                        continue
                    has_value = False
                    for c in assay_cols:
                        if len(row) > c.index and row[c.index].text.strip():
                            pv = parse_value(row[c.index].text)
                            if pv:
                                has_value = True
                                break
                    # Also check prose columns for embedded assay values when
                    # there are no dedicated assay columns.
                    if not has_value and not assay_cols:
                        for c in prose_cols:
                            if len(row) > c.index and row[c.index].text.strip():
                                if _EMBEDDED_ASSAY.search(row[c.index].text):
                                    has_value = True
                                    break
                    if not has_value:
                        continue
                    cid = prev_cid

            for c in assay_cols:
                if len(row) <= c.index:
                    continue
                cell_text = row[c.index].text

                # Detect multi-assay column: header is comma-separated names
                # and cell is a comma-separated list of values.
                col_header = c.header or ""
                header_parts = split_top_level(col_header)
                if len(header_parts) < 2:
                    header_parts = []
                else:
                    # The comma split keeps the shared prefix on only the
                    # first sub-name ("... probe 1", "probe 2"). Give it back
                    # to the rest wherever the shape says it is safe to.
                    header_parts = _redistribute_shared_prefix(header_parts)
                cell_parts = (split_top_level(cell_text)
                              if header_parts and "," in cell_text else [])

                # One column, several assays: "probe 1, probe 2" over cells
                # reading "0.00309, 0.00252". `zip` stops at the shorter side,
                # so a cell with fewer parts than the header pairs what it can
                # and drops the rest — which is why the equal-length and
                # unequal-length cases are one branch and not two.
                if len(header_parts) >= 2 and len(cell_parts) >= 2:
                    for sub_name, sub_val in zip(header_parts, cell_parts):
                        parsed = parse_value(sub_val)
                        if not parsed:
                            continue
                        out.append(AssayRecord(
                            cid=cid,
                            assay_name=sub_name,
                            value_numeric=parsed.get("value_numeric"),
                            qualifier=parsed.get("qualifier"),
                            unit=parsed.get("unit") or c.unit,
                            n_runs=parsed.get("n_runs"),
                            letter_grade=parsed.get("letter_grade"),
                            value_text=parsed.get("value_text", ""),
                            table_id=t.table_id,
                            column_header=c.header,
                        ))
                else:
                    # Single-value cell — original behaviour.
                    parsed = parse_value(cell_text)
                    if not parsed:
                        continue
                    n_runs = parsed.get("n_runs")
                    # A bare "(8)" in the next column is this value's run count.
                    if n_runs is None and len(row) > c.index + 1:
                        nxt = _NRUNS_ONLY.match(row[c.index + 1].text)
                        if nxt and cols[c.index + 1].kind in (NRUNS, UNKNOWN):
                            n_runs = int(nxt.group(1))
                    out.append(AssayRecord(
                        cid=cid,
                        assay_name=c.assay_name or c.header or "unnamed assay",
                        value_numeric=parsed.get("value_numeric"),
                        qualifier=parsed.get("qualifier"),
                        unit=parsed.get("unit") or c.unit,
                        n_runs=n_runs,
                        letter_grade=parsed.get("letter_grade"),
                        value_text=parsed.get("value_text", ""),
                        table_id=t.table_id,
                        column_header=c.header,
                    ))

            # --- Embedded-assay scan for NMR/MS prose columns ---
            # When a column is typed NMR or MS (because its header contains
            # "NMR", "MS", "ESI", etc.) but its cells contain embedded assay
            # values of the form "FXR EC50 (nM) = 1497", extract those values
            # and emit records. This is a general convention in patent tables
            # that bundle characterisation data into a single prose cell.
            # The assay name is taken from the matched substring (e.g.
            # "FXR EC50"), stripped of leading/trailing whitespace.
            for c in prose_cols:
                if len(row) <= c.index:
                    continue
                cell_text = row[c.index].text
                if not cell_text:
                    continue
                for m in _EMBEDDED_ASSAY.finditer(cell_text):
                    raw_name = m.group(1).strip()
                    raw_unit = m.group(2).strip()
                    raw_val = m.group(3).replace(',', '')
                    try:
                        num = float(raw_val)
                    except ValueError:
                        continue
                    # Normalise the unit symbol.
                    unit_norm = _EMBEDDED_UNIT.get(raw_unit, raw_unit)
                    out.append(AssayRecord(
                        cid=cid,
                        assay_name=raw_name,
                        value_numeric=num,
                        qualifier=None,
                        unit=unit_norm,
                        n_runs=None,
                        letter_grade=None,
                        value_text=m.group(0),
                        table_id=t.table_id,
                        column_header=c.header,
                    ))
            # --- End embedded-assay scan ---

        # Flush any buffered wrapped-name record at end of table
        _flush_wrapped()

    return out
