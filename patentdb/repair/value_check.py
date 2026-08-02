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
"""
from __future__ import annotations

import collections
import csv
import logging
import re

from ..core import config

logger = logging.getLogger(__name__)

csv.field_size_limit(10 ** 9)

# 5%, for BindingDB's three-significant-figure rounding. Not a fudge factor for
# disagreement — anything outside this is bucketed and reported, not forgiven.
TOLERANCE = 0.05

# BDB publishes affinities in nM; ours carry whatever the patent printed.
_TO_NM = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "μM": 1e3, "mM": 1e6, "pM": 1e-3,
          "M": 1e9}

_BDB = config.BDB_REFERENCE_TSV


def _to_nm(value, unit):
    return None if value is None or unit not in _TO_NM else value * _TO_NM[unit]


def load_reference(patent_ids: set[str] | None = None,
                   single_patent_only: bool = True) -> dict:
    """(patent, cid) -> set of reference values in nM. Empty when BDB is absent.

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
    from ..scripts.eval.reference_bench import _EXAMPLE_REF, _norm_cid

    out: dict[tuple[str, str], set[float]] = collections.defaultdict(set)
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
        vcols = [i for i, h in enumerate(hdr)
                 if h.split("(")[0].strip() in ("Ki", "IC50", "Kd", "EC50")]
        for row in rdr:
            if len(row) <= nm_i:
                continue
            vals = []
            for i in vcols:
                if i < len(row) and row[i].strip():
                    try:
                        vals.append(float(re.sub(r"[<>~=]", "", row[i]).strip()))
                    except ValueError:
                        pass
            if not vals:
                continue
            hits = [(m.group(1).upper(), _norm_cid(m.group(2)))
                    for m in _EXAMPLE_REF.finditer(row[nm_i] or "")]
            if single_patent_only and len({p for p, _ in hits}) > 1:
                continue
            for pid, cid in hits:
                if patent_ids and pid not in patent_ids:
                    continue
                out[(pid, cid)].update(vals)
    return out


def _bucket(mine_nm: float, ref_nm: float) -> str:
    if ref_nm <= 0 or mine_nm <= 0:
        return "incomparable"
    if abs(mine_nm - ref_nm) / ref_nm <= TOLERANCE:
        return "agree"
    fold = max(mine_nm / ref_nm, ref_nm / mine_nm)
    return ("variance" if fold <= 2 else
            "disagree" if fold <= 10 else "wrong_scale")


def check_patent(patent_id: str, records, reference: dict | None = None) -> dict:
    """Score one patent's records against BindingDB. Never raises."""
    from ..sources.uspto_assays import normalize_cid

    ref = reference if reference is not None else load_reference({patent_id})
    mine: dict[str, list] = collections.defaultdict(list)
    for r in records:
        cid = normalize_cid(r.cid or "").upper()
        if cid:
            mine[cid].append(r)

    st = collections.Counter()
    worst: list[dict] = []
    for (pid, cid), refvals in ref.items():
        if pid != patent_id:
            continue
        got = mine.get(cid)
        if not got:
            st["no_record"] += 1
            continue
        # Best bucket across our records for this compound and the reference
        # values it lists. Deliberately optimistic per compound: a patent
        # reports several assays and BDB may hold several, and we do not know
        # which pairs. The point is to catch a WRONG value, not to grade
        # pairing, so a compound counts as agreeing if any pairing agrees.
        order = ["agree", "range_contains", "variance", "disagree",
                 "wrong_scale", "range_misses", "incomparable"]
        best, detail = None, None
        for r in got:
            for rv in refvals:
                if r.value_numeric is not None:
                    mv = _to_nm(r.value_numeric, r.unit)
                    if mv is None:
                        continue
                    tag = _bucket(mv, rv)
                    d = (f"{r.assay_name[:34]!r} ours={r.value_numeric}{r.unit} "
                         f"= {mv:g} nM vs ref {rv:g} nM")
                elif r.range_lo is not None or r.range_hi is not None:
                    lo = _to_nm(r.range_lo, r.unit)
                    hi = _to_nm(r.range_hi, r.unit)
                    if lo is None and hi is None:
                        continue
                    inside = ((lo is None or rv >= lo * (1 - TOLERANCE))
                              and (hi is None or rv <= hi * (1 + TOLERANCE)))
                    tag = "range_contains" if inside else "range_misses"
                    d = (f"{r.assay_name[:34]!r} ours={lo}-{hi} nM "
                         f"vs ref {rv:g} nM")
                else:
                    continue
                if best is None or order.index(tag) < order.index(best):
                    best, detail = tag, d
        if best is None:
            st["no_value"] += 1
            continue
        st[best] += 1
        if best in ("wrong_scale", "disagree", "range_misses") and len(worst) < 12:
            worst.append({"cid": cid, "bucket": best, "detail": detail})

    st["refs"] = sum(v for k, v in st.items() if k != "refs")
    bad = st["wrong_scale"] + st["disagree"] + st["range_misses"]
    return {"patent": patent_id, "buckets": dict(st), "bad": bad,
            "checked": st["refs"] - st["no_record"] - st["no_value"],
            "examples": worst, "tolerance": TOLERANCE}


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
