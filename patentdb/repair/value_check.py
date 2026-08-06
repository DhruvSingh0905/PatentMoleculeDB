"""Are our NUMBERS right? Checked against BindingDB, not against a gate.

Coverage says we found the compound. It cannot say we read its value
correctly, and the two fail in opposite directions: a coverage gate blocks a
patch that reads less, and waves through a patch that reads MORE of the wrong
thing. Both happened here in one session — a capability patch was declined for
looking inert when it recovered 1,238 rows, and 99 records read a dimensionless
selectivity ratio as a nanomolar potency while every count went up.

BindingDB carries a numeric affinity on 100% of its rows, so this is a lookup,
not a judgement. That matters: every judgement-shaped gate in this codebase has
been wrong at least as often as right, and this one cannot be, because it is
comparing our number to a published number for the same compound.

TOLERANCE is 5%, for rounding. BindingDB stores three significant figures, so
our `0.023456 uM` and its `23.5 nM` are the same measurement — counting that as
a disagreement scores us against a lossier copy of the same patent. Beyond 5%
the buckets separate assay variability from a different quantity entirely:

    agree        within 5%, or our range contains the reference
    variance     5% to 2x    — plausible between assay runs, worth a look
    disagree     2x to 10x   — probably not the same measurement
    wrong_scale  over 10x    — a unit or a quantity error, the Ratio class

Only a range can be checked by containment, and containment is a weaker test
than equality; it is reported separately rather than folded in.

WHAT A BUCKET COUNTS. `buckets` counts COMPOUNDS, one bucket each, as it
always has — every reader of this dict (`scripts/eval/baseline.py`,
`value_check_cli`, `check_corpus`) reads it that way. What changed in 2026-08
is how a compound's many records collapse into that one bucket: the best of
the whole cross-product, which no wrong record could ever lower, became the
median WITHIN an assay name and the best ACROSS assay names (`check_patent`).
`record_buckets` is the other half — a per-RECORD count, so a row that agrees
with nothing is visible even when its compound reads fine. It has to be
visible somewhere: the old scorer gave a wrong row no bucket, no counter and
no example line, which is how 99 dimensionless ratios read as nanomolar
potencies while the score held.
"""
from __future__ import annotations

import collections
import csv
import functools
import logging
import re

from ..core import config
from ..sources.uspto_assays import normalize_cid

logger = logging.getLogger(__name__)

csv.field_size_limit(10 ** 9)

# 5%, for BindingDB's three-significant-figure rounding. Not a fudge factor for
# disagreement — anything outside this is bucketed and reported, not forgiven.
TOLERANCE = 0.05

# BDB publishes affinities in nM; ours carry whatever the patent printed.
_TO_NM = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "μM": 1e3, "mM": 1e6, "pM": 1e-3,
          "M": 1e9}

_BDB = config.BDB_REFERENCE_TSV


# ── how BindingDB names a patent's compound ───────────────────────
#
# This pattern and `_norm_cid` used to live in `scripts/eval/reference_bench`
# and were imported from here at call time. That made an EVAL module part of
# the live set — `import_audit --config` listed exactly one, this one — and
# `_bad_values_now` therefore reached into `scripts/eval/` in the middle of an
# extraction run. The arrow now points the other way: the reader of the
# reference lives with the reference, and the benchmark imports it.
#
# The word "Example" is OPTIONAL, because it is often simply absent:
#
#   US8952177, Example 1
#   US8722692, 1                              <- no keyword at all
#   US9303033, N47, Table 58A, Compound 11    <- the patent's code, then BDB's
#
# Requiring it scored three whole patents at zero that BindingDB holds in full:
# US9303033 has 2,239 rows there, US8722692 732, US9708336 837. Across the 53
# patents with no structures of our own, reading only the strict form found
# 10,147 of 16,222 compounds; reading this form finds 13,453.
#
# The id is ALWAYS the token straight after the patent number, and never the
# trailing "Compound N" — that is BindingDB's own within-table numbering.
# Measured on US9303033: the first token hits our extracted ids 2,491 times and
# misses 12; the trailing one hits ZERO times and misses 2,482. Taking the
# wrong one would have attached 1,237 structures to the wrong compounds.
# The label is optional AND positional. `US11286268, Compound 1` numbers its
# compounds 1..1837 and "Compound 1" is the id; `US9303033, N47, Table 58A,
# Compound 11` uses `N47` and the trailing "Compound 11" is BindingDB's own
# within-table counter. A flat stop-list on the word `compound` gets the second
# right and the first wrong — it cost US11286268 all 1,828 of its reference
# values, which is why its patch could not be checked at all.
#
# What separates them is POSITION, not vocabulary: whatever follows the patent
# number is the id, and a label immediately there is part of the id rather than
# a reason to skip. The stop-list still applies to a BARE token, so
# `US…, Table 5` is not read as compound "Table".
_REF_LABEL = (r"(?:(?:Examples?|Compounds?|(?:C(?:o?m)?pd)\.?\s*(?:No)?\.?|Ex)"
              r"\.?\s*)?")
_REF_STOP = (r"(?!(?:table|scheme|fig(?:ure)?|claim|page|para|"
             r"col(?:umn)?|entry|item|no)\b)")
_EXAMPLE_REF = re.compile(
    r"\b(US\d{7,11}[A-Z]?\d?)\s*,\s*" + _REF_LABEL + _REF_STOP
    + r"([0-9A-Za-z][0-9A-Za-z\-]{0,9})\b", re.I)


# Memoised, and the cache is what makes the key affordable rather than a
# convenience. `load_reference` calls this once per `(patent, compound)` match
# in a 271 MB TSV — 227,292 times in one traced extraction run — over a
# reference whose ids repeat constantly (`1`, `2`, `I-117`). `normalize_cid` is
# a pure function of its argument: it reads the string, two module-level
# regexes and nothing else.
@functools.lru_cache(maxsize=100_000)
def _norm_cid(cid: str) -> str:
    """'007' / 'Example 7' / 'Cpd. No. 7' → '7'. One canonical form.

    Delegates to the extractor's own `normalize_cid`, because "one canonical
    form" has to mean ONE. This function used to re-implement it with
    `s.lstrip("0")`, which only strips zeros at position 0 — so BindingDB's
    `I-0117` stayed `I-0117` while the patent's `I-117` normalised to `I-117`
    and the two never met. On US9718790 that was 1,119 compounds scored as
    missing that we had extracted correctly all along: a benchmark measuring
    its own normaliser rather than the extraction.
    """
    return normalize_cid(str(cid)).upper()


def _to_nm(value, unit):
    return None if value is None or unit not in _TO_NM else value * _TO_NM[unit]


def _norm_target(name: str) -> str:
    """One protein, one spelling. Whitespace only — the names are BDB's.

    Measured on the live file: 201 distinct Target Names among the rows we
    key, and the collapse changes that count by zero. It is here so a stray
    tab or wrapped newline inside a curator's protein name cannot split one
    target into two keys, which would hand back the resolving power the finer
    key exists to keep.
    """
    return re.sub(r"\s+", " ", (name or "").strip())


def load_reference(patent_ids: set[str] | None = None,
                   single_patent_only: bool = True) -> dict:
    """(patent, cid, measure, target) -> set of values in nM. Empty w/o BDB.

    WHICH measurement and WHICH protein are part of the reference point, not
    decoration on it. Keyed on `(patent, cid)` alone — as this returned until
    2026-08 — a 10 nM IC50 against one target and a 500 nM Ki against another
    pour into one set and become alternative right answers, so a value we read
    against the WRONG protein scores `agree` against the right one's number.

    Measured over `config.BDB_REFERENCE_TSV` (271 MB, 95,245 rows): 19,550
    coarse keys where the same rows support 37,573 `(patent, cid, measure,
    target)` keys — 48.0% of the reference's resolving power discarded before
    a single comparison — and 92.8% of the finer keys hold exactly ONE value,
    i.e. the finer key almost always names a unique number where the coarse
    one names a menu (only 52.7% of coarse keys are single-valued).

    US9745328 is the live case: its rows carry BOTH `Arachidonate
    5-lipoxygenase-activating protein` (1,619 rows) and `Flap endonuclease 1`
    (898 rows), two unrelated proteins under one compound id.

    Callers that only want "the numbers BDB publishes for this compound" fold
    the key with `_ref_by_compound` rather than losing it at load time.

    Only rows attributing the ligand to ONE patent count, and that is the
    difference between a benchmark and a rumour. BindingDB routinely names a
    whole family in a single row —

        US9079866, 114::US9884878, Compound 114::US9745328, Compound 114

    — and 46,000 of its 84,000 patent-bearing rows cite two or more, up to 17.
    The affinity was measured once, on one ligand; which family member's table
    it came from is not recorded. Attributing it to all of them scored 21 of
    our 24 corpus-wide value disagreements, every one of them against a patent
    whose own table says something different and is right: US9745328 prints
    `0.0004 uM` for compound 102 where the shared row says `0.1 nM`.

    A family-level value is real evidence about a MOLECULE and no evidence at
    all about this patent's table, which is what we are checking.
    """
    out: dict[tuple[str, str, str, str], set[float]] = collections.defaultdict(set)
    if not _BDB.exists():
        logger.info("value_check: no BindingDB subset at %s", _BDB)
        return out
    with open(_BDB, newline="", encoding="utf-8", errors="ignore") as fh:
        rdr = csv.reader(fh, delimiter="\t")
        hdr = next(rdr)
        try:
            nm_i = hdr.index("BindingDB Ligand Name")
        except ValueError:
            return out
        # The column is located AND named: `Ki (nM)` -> "Ki". Same scan as
        # before, one field wider, so the stream stays a single pass over
        # 271 MB with no extra work on the rows we discard.
        vcols = [(i, h.split("(")[0].strip()) for i, h in enumerate(hdr)
                 if h.split("(")[0].strip() in ("Ki", "IC50", "Kd", "EC50")]
        tgt_i = hdr.index("Target Name") if "Target Name" in hdr else -1
        for row in rdr:
            if len(row) <= nm_i:
                continue
            vals = []
            for i, measure in vcols:
                if i < len(row) and row[i].strip():
                    try:
                        vals.append((measure,
                                     float(re.sub(r"[<>~=]", "", row[i]).strip())))
                    except ValueError:
                        pass
            if not vals:
                continue
            hits = [(m.group(1).upper(), _norm_cid(m.group(2)))
                    for m in _EXAMPLE_REF.finditer(row[nm_i] or "")]
            if single_patent_only and len({p for p, _ in hits}) > 1:
                continue
            target = _norm_target(row[tgt_i]) if 0 <= tgt_i < len(row) else ""
            for pid, cid in hits:
                if patent_ids and pid not in patent_ids:
                    continue
                for measure, val in vals:
                    out[(pid, cid, measure, target)].add(val)
    return out


def _bucket(mine_nm: float, ref_nm: float) -> str:
    if ref_nm <= 0 or mine_nm <= 0:
        return "incomparable"
    if abs(mine_nm - ref_nm) / ref_nm <= TOLERANCE:
        return "agree"
    fold = max(mine_nm / ref_nm, ref_nm / mine_nm)
    return ("variance" if fold <= 2 else
            "disagree" if fold <= 10 else "wrong_scale")


# Best to worst. The index is the severity rank the per-compound median is
# taken over, so the order is load-bearing, not presentation.
_ORDER = ["agree", "range_contains", "variance", "disagree",
          "wrong_scale", "range_misses", "incomparable"]
_BAD = ("wrong_scale", "disagree", "range_misses")


def _ref_by_compound(ref: dict, patent_id: str) -> dict[str, dict[float, str]]:
    """Fold the reference down to `cid -> {value_nM: 'MEASURE TARGET'}`.

    `load_reference` keys on `(patent, cid, measure, target)` because those
    are four different facts. Pairing them with OUR records is a separate
    question with a worse answer: an `AssayRecord` carries a free-text column
    header (`'Ave A2B cAMP IC50'`), not a curator's protein name, so demanding
    a per-target match would score every record whose assay BDB does not hold
    as a disagreement — manufacturing exactly the false failures this module
    exists to tell apart from real ones.

    So the provenance rides along as a LABEL and surfaces in `examples`, where
    a human can see which published number a record was scored against.
    Accepts the pre-2026-08 two-field key unchanged.
    """
    out: dict[str, dict[float, str]] = collections.defaultdict(dict)
    for key, values in ref.items():
        if key[0] != patent_id:
            continue
        label = " ".join(str(x) for x in key[2:4] if x) if len(key) >= 4 else ""
        per = out[key[1]]
        for v in sorted(values):
            per.setdefault(v, label)
    return out


def _score_record(r, refvals: dict[float, str]) -> tuple[str, str] | None:
    """This ONE record's best bucket over the compound's reference values.

    Optimism belongs here and only here. A patent reports several assays, BDB
    may hold several, and we do not know which pairs — so a record is scored
    against its own best match. What it may no longer do is speak for the
    other ten records of the same compound.

    None when the record carries neither a usable value nor a usable range.
    """
    # Hoisted out of the loop: whether this record carries a usable number at
    # all does not depend on WHICH reference value it is held against, and
    # deciding it once turns "no usable value" into one exit instead of one
    # per reference value.
    mv = lo = hi = None
    if r.value_numeric is not None:
        mv = _to_nm(r.value_numeric, r.unit)
        if mv is None:
            return None
    elif r.range_lo is not None or r.range_hi is not None:
        lo, hi = _to_nm(r.range_lo, r.unit), _to_nm(r.range_hi, r.unit)
        if lo is None and hi is None:
            return None
    else:
        return None

    best: tuple[str, str] | None = None
    for rv, label in refvals.items():
        seen = f" [{label}]" if label else ""
        if mv is not None:
            tag = _bucket(mv, rv)
            d = (f"{r.assay_name[:34]!r} ours={r.value_numeric}{r.unit} "
                 f"= {mv:g} nM vs ref {rv:g} nM{seen}")
        else:
            inside = ((lo is None or rv >= lo * (1 - TOLERANCE))
                      and (hi is None or rv <= hi * (1 + TOLERANCE)))
            tag = "range_contains" if inside else "range_misses"
            d = (f"{r.assay_name[:34]!r} ours={lo}-{hi} nM "
                 f"vs ref {rv:g} nM{seen}")
        if best is None or _ORDER.index(tag) < _ORDER.index(best[0]):
            best = (tag, d)
    return best


def check_patent(patent_id: str, records, reference: dict | None = None) -> dict:
    """Score one patent's records against BindingDB. Never raises.

    `buckets` counts COMPOUNDS, one bucket each, as it always has.
    `record_buckets` counts RECORDS and is the new half: it is where a row
    that agrees with nothing is visible even when its compound reads fine.
    """
    ref = reference if reference is not None else load_reference({patent_id})
    mine: dict[str, list] = collections.defaultdict(list)
    for r in records:
        cid = normalize_cid(r.cid or "").upper()
        if cid:
            mine[cid].append(r)

    st = collections.Counter()          # per COMPOUND — every reader's shape
    rst = collections.Counter()         # per RECORD
    cand: list[dict] = []
    for cid, refvals in _ref_by_compound(ref, patent_id).items():
        got = mine.get(cid)
        if not got:
            st["no_record"] += 1
            continue
        scored = [(r, s) for r, s in ((r, _score_record(r, refvals))
                                      for r in got) if s]
        if not scored:
            st["no_value"] += 1
            continue
        for _, (tag, _d) in scored:
            rst[tag] += 1
        # WITHIN one assay name, the MEDIAN. ACROSS assay names, the best.
        #
        # Taking the best of the whole cross-product — what this did until
        # 2026-08 — made the metric monotone in record count: 20 rows against
        # 5 references is 100 chances to land within 5% and no wrong row could
        # ever cost a compound its `agree`. Measured on the old code, one
        # correct record scored {'agree': 1}; that same record plus ten 50x-out
        # ones scored {'agree': 1} — byte-identical with 91% of the evidence
        # wrong, and the wrong rows carried no bucket, no counter, no example
        # row. That is exactly the patch this module exists to catch: one that
        # reads MORE of the wrong thing.
        #
        # Plain median over ALL of a compound's records was measured and
        # rejected. Over the 84 scored patents it moved bad from 3 compounds
        # to 682, and the new flags were mostly OUR OWN correct readings: on
        # US11613531 compound 228 we read `ROCK2 IC50 = 3.0 nM` and `ROCK1
        # IC50 = 4.0 nM`, both BDB's numbers exactly, and the compound scored
        # `wrong_scale` because five further columns (pMLC, NIH3T3, two
        # %-inhibition rows) are assays BDB does not hold and had no right
        # pairing to be scored against. Punishing a patent for reporting more
        # than BindingDB does is the false-failure half of the same defect.
        #
        # Records sharing an assay name are competing claims about ONE
        # quantity, and the reference can corroborate one of them at most, so
        # a strict majority there decides (even splits still resolve
        # optimistically). Records under DIFFERENT names are different
        # quantities, so the compound keeps the best of them. Measured:
        # agree 14,275 -> 13,968, bad 3 -> 187 compounds over 4 patents, and
        # 184 of those 187 are US9718790 (161) + US9718825 (23) where a single
        # assay name covers several distinct columns — 2,262 of US9718790's
        # 2,270 records are all called `P2X3 IC50 (μM)`. That is a real
        # finding about our headers, not a manufactured one.
        groups: dict[str, list] = collections.defaultdict(list)
        for r, s in scored:
            groups[re.sub(r"\s+", " ", (r.assay_name or "").strip().lower())].append(s)
        per_group = []
        for g in groups.values():
            g.sort(key=lambda s: _ORDER.index(s[0]))
            per_group.append(g[(len(g) - 1) // 2])
        tag, detail = min(per_group, key=lambda s: _ORDER.index(s[0]))
        st[tag] += 1
        dist = collections.Counter(s[0] for _, s in scored)
        if tag in _BAD or any(t in _BAD for t in dist):
            cand.append({"cid": cid, "bucket": tag, "detail": detail,
                         "records": dict(dist)})

    # Disagreeing COMPOUNDS first, then compounds that merely hide disagreeing
    # rows, so the 12-row cap cannot be spent on the milder half.
    cand.sort(key=lambda e: (e["bucket"] not in _BAD,
                             -sum(n for t, n in e["records"].items() if t in _BAD)))
    st["refs"] = sum(v for k, v in st.items() if k != "refs")
    bad = st["wrong_scale"] + st["disagree"] + st["range_misses"]
    return {"patent": patent_id, "buckets": dict(st), "bad": bad,
            "checked": st["refs"] - st["no_record"] - st["no_value"],
            "record_buckets": dict(rst),
            "bad_records": rst["wrong_scale"] + rst["disagree"] + rst["range_misses"],
            "examples": cand[:12], "tolerance": TOLERANCE}


def check_corpus(patent_ids: list[str] | None = None) -> dict:
    """Every cached patent BDB covers. Free apart from reading the TSV."""
    from ..sources.uspto_assays import extract_from_patent

    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    pids = patent_ids or sorted(p.stem for p in xml_dir.glob("*.xml"))
    ref = load_reference(set(pids))
    per, tot = {}, collections.Counter()
    for pid in pids:
        try:
            recs = extract_from_patent((xml_dir / f"{pid}.xml").read_text(errors="ignore"))
        except Exception as e:
            logger.warning("value_check: %s raised %r", pid, e)
            continue
        r = check_patent(pid, recs, reference=ref)
        if r["buckets"].get("refs"):
            per[pid] = r
            tot.update(r["buckets"])
    return {"per_patent": per, "total": dict(tot),
            "bad_patents": {p: r["bad"] for p, r in per.items() if r["bad"]}}
