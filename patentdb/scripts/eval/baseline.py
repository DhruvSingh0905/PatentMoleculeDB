"""ONE coverage baseline, and an explicit account of why the others disagree.

    python3 -m patentdb.scripts.eval.baseline
    python3 -m patentdb.scripts.eval.baseline --patents US9302989,US8952177
    python3 -m patentdb.scripts.eval.baseline --no-bdb        # skip the 8 GB read

This module exists because three tools reported three different corpus sizes
and none of them named which artifact it was describing:

    pipeline_bench      123 patents / 37,593 compounds   live recompute, UNION
    corpus_export       same union, computed independently in `_pipeline_records`
    assay_tables.json    22 patents /  11,467 compounds   what we actually SHIP

The two eval scripts do not read the pipeline's output at all. They recompute
`uspto_assays.extract_from_patent(xml) | repair.loop.repair_patent(pid,xml)[0]`
at measurement time, from the cached XML. That union is a real capability of
the code. It is NOT the shipped artifact, and the gap between them is not a
rounding error — see `_SHIPPED_MISSES_THE_XML_TIER` below.

Every number this tool prints is labelled with the artifact it came from.
Three, never blended:

  SHIPPED   `output_v2/text_extraction/{pid}/assay_tables.json` — the file a
            consumer reads. Written by `routes/process_patent.py::_write_outputs`.
  UNION     the live recompute the eval scripts perform. What the code CAN read.
  SNAPSHOT  `output_v2/extraction_snapshots/{pid}.json` — the frozen answer
            `repair/snapshot.py` pins, which is the XML tier plus rule-recovered
            records and therefore agrees with UNION, not with SHIPPED.

...and a fourth that no existing tool reports, because assuming it away is the
mistake underneath all three:

  COMBINED  SHIPPED ∪ UNION. **Neither of the other two is a superset.**
            Measured over 20 patents: UNION holds 5,799 compounds SHIPPED does
            not (US11286268 +1,630, US9302989 +1,397 — the CALS tier the
            shipped file never receives), while SHIPPED holds 2,563 that UNION
            does not (US9718790 +1,239 — a patent the TEXT tier reads far
            better than the XML tier). Quoting either alone understates the
            corpus, and "which is real" has no single answer per patent.

            17,143 compounds, and the reference agrees this is the honest
            ceiling: BindingDB `no_record` falls to **9 of 10,540** references,
            against 4,568 for SHIPPED and 1,223 for UNION. Two tiers that each
            look mediocre alone are between them reading almost everything the
            reference knows about — which is the argument for merging them in
            `process_patent`, not for picking a winner.

`REPAIR=0` and the synthesis budget are both forced off: a baseline that buys
new rules while measuring is measuring itself, and this must be re-runnable for
$0. Known rules still apply — those are already paid for.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── The finding this module was built to make un-loseable ────────────────────
#
# `process_patent` builds `assay_tables` from the HARVEST/FSM TEXT tier
# (process_patent.py:1536, `core.assay_fsm.pipeline.extract_for_patent`) and
# then merges exactly two things into it:
#
#   process_patent.py:1810-1818   `repaired` — repair_patent's return value,
#                                 which is ONLY the rule-recovered supplement
#                                 (loop.py:426 `recovered: list = []`,
#                                 extended at loop.py:634 and returned at the
#                                 end; the deterministic baseline computed at
#                                 loop.py:366 is used for gap ranking and
#                                 discarded).
#
#   process_patent.py:1841-1852   `extract_from_patent` — the USPTO CALS tier,
#                                 the one that scores 99.9% against BindingDB —
#                                 merged ONLY under `if healed and
#                                 healed.get("applied")` at line 1837, i.e.
#                                 only on a run where the capability tier
#                                 applied a code patch.
#
# So on every ordinary run the XML tier reaches `assay_tables.json` not at all.
# It IS computed on that run — at process_patent.py:1834-1836 — but only to be
# handed to `freeze_result`, which writes the snapshot. The pipeline reads the
# right data, writes it to the audit artifact, and drops it from the shipped one.
#
# Measured over the 20 patents holding both a cached XML and a shipped file:
# SHIPPED 11,344 compounds / 35,324 records against UNION 14,580 / 31,685.
# US9302989 ships 126 compounds where the XML tier alone reads 1,523, and
# US11286268 ships 207 against 1,837.
#
# The record counts invert (SHIPPED has 3,639 MORE records over 3,236 FEWER
# compounds) and that is not a contradiction: the text tier emits many rows per
# compound — US11566007 alone ships 7,971 letter-bin records over 744 compounds
# — while the XML tier reads far more distinct compounds. Records are the wrong
# unit for this question; compounds are the unit a molecule database is sold in.
#
# Against BindingDB the direction is unambiguous. `no_record` — BDB names a
# compound the artifact does not contain — falls from 4,568 to 1,223 of 10,540
# references, and `agree` rises 3,984 -> 6,975.
_SHIPPED_MISSES_THE_XML_TIER = True


def _shipped_path(pid: str):
    from ...core import config
    return config.OUTPUT_DIR / "text_extraction" / pid / "assay_tables.json"


class _Row:
    """A shipped JSON row given the attribute names the scorers expect.

    `value_check.check_patent` and `snapshot` both read `range_lo`/`range_hi`,
    but `_write_outputs` serialises the same bounds as `value_low`/`value_high`
    (see the letter-bin merge at process_patent.py:1604). Scoring the shipped
    file without this rename silently drops every binned record to `no_value`
    — 7,971 of US11566007's records are bins, so the omission would have read
    as that patent having no comparable data at all.
    """

    __slots__ = ("cid", "assay_name", "value_numeric", "unit", "qualifier",
                 "n_runs", "range_lo", "range_hi", "table_id", "source")

    def __init__(self, cid: str, d: dict):
        self.cid = cid
        self.assay_name = d.get("assay_name") or ""
        self.value_numeric = d.get("value_numeric")
        self.unit = d.get("unit")
        self.qualifier = d.get("qualifier")
        self.n_runs = d.get("n_runs")
        self.range_lo = d.get("value_low", d.get("range_lo"))
        self.range_hi = d.get("value_high", d.get("range_hi"))
        self.table_id = d.get("table_id") or ""
        self.source = d.get("source") or ""


def load_shipped(pid: str) -> list[_Row]:
    """The artifact a consumer actually reads, or [] when the patent never ran."""
    p = _shipped_path(pid)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        logger.warning("baseline: %s has an unreadable assay_tables.json (%r)",
                       pid, e)
        return []
    return [_Row(cid, d) for cid, rows in data.items() for d in rows
            if isinstance(d, dict)]


def load_union(pid: str, xml: str) -> tuple[list, list[str]]:
    """The eval scripts' number: both reading tiers, de-duplicated.

    Deliberately the SAME key as `corpus_export._pipeline_records` so the two
    cannot drift apart again. A tier that raises is recorded and skipped rather
    than aborting the patent — `pipeline_bench` learned that the hard way, and
    a crash that preserves the appearance of the totals is the shape of nearly
    every defect in this area's history.
    """
    from ...repair.loop import repair_patent
    from ...sources.uspto_assays import extract_from_patent

    out, seen, crashed = [], set(), []
    for tier, fn in (("raw", lambda: extract_from_patent(xml)),
                     ("rules", lambda: repair_patent(pid, xml)[0])):
        try:
            recs = list(fn())
        except Exception as e:
            crashed.append(tier)
            logger.warning("baseline: %s %s raised %r", pid, tier, e)
            continue
        for r in recs:
            if not getattr(r, "is_usable", False):
                continue
            key = (r.cid, r.assay_name, r.value_numeric, r.unit, r.table_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out, crashed


def _tally(records) -> dict:
    return {"records": len(records),
            "compounds": len({r.cid for r in records if r.cid})}


def measure(patents: list[str] | None = None, *, with_bdb: bool = True) -> dict:
    """Per-patent SHIPPED vs UNION, plus BDB value agreement for each."""
    # A baseline must not buy anything. `REPAIR=0` is set before `config` is
    # imported because `REPAIR_ENABLED` is read at import time (config.py:117).
    os.environ["REPAIR"] = "0"
    from ...core import config
    from ...repair import snapshot

    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    have_xml = {p.stem: p for p in sorted(xml_dir.glob("*.xml"))}
    have_shipped = {p for p in have_xml if _shipped_path(p).exists()}

    if patents:
        wanted = [p for p in patents if p in have_xml]
    else:
        # The reconciliation set is the INTERSECTION. Comparing a shipped file
        # against a patent that has no XML, or an XML against a patent that was
        # never processed, produces a difference that is not a divergence.
        wanted = sorted(have_shipped)

    reference = {}
    if with_bdb:
        from ...repair.value_check import load_reference
        reference = load_reference(set(wanted))

    from ...repair.value_check import check_patent

    per: dict[str, dict] = {}
    for pid in wanted:
        xml = have_xml[pid].read_text(errors="ignore")
        shipped = load_shipped(pid)
        union, crashed = load_union(pid, xml)

        # COMBINED is de-duplicated on the COMPOUND, not on the record. The two
        # artifacts describe the same measurements in different vocabularies —
        # the text tier's `assay_name` is free text where the XML tier's is a
        # column header — so a record-level key would count most rows twice and
        # invent a corpus larger than either source. Compound identity is the
        # one thing both agree on.
        s_cids = {r.cid for r in shipped if r.cid}
        u_cids = {r.cid for r in union if r.cid}
        row = {"shipped": _tally(shipped), "union": _tally(union),
               "combined": {"records": None, "compounds": len(s_cids | u_cids)},
               "only_shipped": len(s_cids - u_cids),
               "only_union": len(u_cids - s_cids),
               "crashed": crashed}
        row["delta"] = {
            "records": row["union"]["records"] - row["shipped"]["records"],
            "compounds": row["union"]["compounds"] - row["shipped"]["compounds"],
        }

        # A frozen snapshot pins this patent's answer against every later patch
        # (repair/snapshot.py). It is only MASKING something when the live
        # recompute now disagrees with what was pinned — that is the case the
        # `thaw()` escape hatch exists for.
        snap = snapshot.load(pid)
        if snap is not None:
            pinned = set(snap.get("compounds") or [])
            live = {r.cid for r in union if r.cid}
            row["snapshot"] = {
                "frozen": True,
                "journal_head": snap.get("journal_head"),
                "compounds": len(pinned),
                "live_compounds": len(live),
                "drift": len(live) - len(pinned),
                "masking": bool(live - pinned) or bool(pinned - live),
            }
        else:
            row["snapshot"] = {"frozen": False}

        if with_bdb:
            row["bdb"] = {
                "shipped": check_patent(pid, shipped, reference),
                "union": check_patent(pid, union, reference),
                # Scored on the concatenated records so a reference value can
                # be satisfied by whichever tier actually read that compound.
                # `check_patent` already takes the best bucket per compound.
                "combined": check_patent(pid, list(shipped) + list(union),
                                         reference),
            }
        per[pid] = row

    # The snapshot corpus is far larger than the shipped corpus, and it holds
    # the tier the shipped files lack — `freeze_result` is fed
    # `extract_from_patent(...) + repaired` at process_patent.py:1834-1836, the
    # very union `_write_outputs` never receives. So the audit artifact is a
    # BETTER record of what we extracted than the product artifact is.
    frozen_all = snapshot.frozen_ids()
    frozen_compounds = 0
    for pid in frozen_all:
        snap = snapshot.load(pid)
        if snap:
            frozen_compounds += len(snap.get("compounds") or [])

    return {"per_patent": per,
            "n_patents": len(per),
            "totals": _totals(per, with_bdb=with_bdb),
            "coverage": {
                "xml_cached": len(have_xml),
                "shipped_artifacts": len(have_shipped),
                "snapshots": len(frozen_all),
                "snapshot_compounds": frozen_compounds,
                "frozen_never_shipped": len(frozen_all - have_shipped),
                # Patents the eval scripts count and no consumer can read: they
                # have a cached XML but were never run through process_patent.
                "xml_without_shipped": sorted(set(have_xml) - have_shipped),
            }}


def _totals(per: dict, *, with_bdb: bool) -> dict:
    t: dict = {}
    for art in ("shipped", "union", "combined"):
        t[art] = {"records": sum(r[art]["records"] or 0 for r in per.values()),
                  "compounds": sum(r[art]["compounds"] for r in per.values()),
                  "zeros": sorted(p for p, r in per.items()
                                  if r[art]["compounds"] == 0)}
    t["combined"]["records"] = None      # see the note at the COMBINED tally
    t["delta"] = {
        "records": t["union"]["records"] - t["shipped"]["records"],
        "compounds": t["union"]["compounds"] - t["shipped"]["compounds"],
    }
    t["only_shipped"] = sum(r["only_shipped"] for r in per.values())
    t["only_union"] = sum(r["only_union"] for r in per.values())
    if with_bdb:
        for art in ("shipped", "union", "combined"):
            agg: dict = defaultdict(int)
            for r in per.values():
                for k, v in r["bdb"][art]["buckets"].items():
                    agg[k] += v
            t.setdefault("bdb", {})[art] = dict(agg)
    t["frozen"] = sorted(p for p, r in per.items() if r["snapshot"]["frozen"])
    t["snapshot_masking"] = sorted(p for p, r in per.items()
                                   if r["snapshot"].get("masking"))
    return t


# ── reporting ────────────────────────────────────────────────────────────────

def _markdown(out: dict) -> str:
    t, per = out["totals"], out["per_patent"]
    cov = out["coverage"]
    L: list[str] = []
    L.append("# Coverage baseline\n")
    L.append("Generated by `python3 -m patentdb.scripts.eval.baseline`. "
             "Every number names the artifact it came from.\n")

    L.append("## The three artifacts\n")
    L.append("| | what it is | written by |")
    L.append("|---|---|---|")
    L.append("| **SHIPPED** | `output_v2/text_extraction/{pid}/assay_tables.json` "
             "— what a consumer reads | `routes/process_patent.py::_write_outputs` |")
    L.append("| **UNION** | live recompute, `extract_from_patent` ∪ "
             "`repair_patent[0]` | `pipeline_bench`, `corpus_export` |")
    L.append("| **SNAPSHOT** | `output_v2/extraction_snapshots/{pid}.json` "
             "| `repair/snapshot.py::freeze` |")
    L.append("| **COMBINED** | SHIPPED ∪ UNION on compound identity — the "
             "ceiling neither tool reports | this module |\n")

    L.append("## Totals over the reconciliation set\n")
    L.append(f"{out['n_patents']} patents that have BOTH a cached XML and a "
             f"shipped `assay_tables.json`. The corpus holds "
             f"{cov['xml_cached']} cached XMLs and only "
             f"{cov['shipped_artifacts']} shipped artifacts, so the eval "
             f"scripts' patent COUNT is itself not the shipped count.\n")
    L.append("| artifact | records | compounds | patents at zero |")
    L.append("|---|---:|---:|---:|")
    for art in ("shipped", "union"):
        L.append(f"| {art.upper()} | {t[art]['records']:,} | "
                 f"{t[art]['compounds']:,} | {len(t[art]['zeros'])} |")
    L.append(f"| **delta (union − shipped)** | **{t['delta']['records']:+,}** | "
             f"**{t['delta']['compounds']:+,}** | |")
    L.append(f"| **COMBINED** | n/a — see note | "
             f"**{t['combined']['compounds']:,}** | "
             f"{len(t['combined']['zeros'])} |\n")
    L.append(f"Neither artifact contains the other: UNION holds "
             f"**{t['only_union']:,}** compounds SHIPPED lacks, and SHIPPED "
             f"holds **{t['only_shipped']:,}** UNION lacks. COMBINED is "
             f"de-duplicated on compound identity only — the two tiers write "
             f"assay names in different vocabularies, so a record-level union "
             f"would double-count most rows and is deliberately not reported.\n")

    if "bdb" in t:
        L.append("## BindingDB value agreement\n")
        L.append("`repair/value_check.py` buckets at 5% tolerance, "
                 "single-patent citations only. `no_record` = BDB names a "
                 "compound the artifact does not contain.\n")
        keys = ["agree", "range_contains", "variance", "disagree",
                "wrong_scale", "range_misses", "incomparable", "no_value",
                "no_record", "refs"]
        L.append("| artifact | " + " | ".join(keys) + " |")
        L.append("|---" * (len(keys) + 1) + "|")
        for art in ("shipped", "union", "combined"):
            b = t["bdb"][art]
            L.append(f"| {art.upper()} | "
                     + " | ".join(str(b.get(k, 0)) for k in keys) + " |")
        L.append("")

    L.append("## Per-patent divergence\n")
    L.append("Sorted by compounds lost between UNION and SHIPPED.\n")
    L.append("| patent | shipped cmpd | union cmpd | Δ cmpd | combined cmpd | "
             "only ship | only union | shipped rec | union rec | Δ rec | "
             "BDB agree ship→union | BDB no_record ship→union | frozen |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for pid, r in sorted(per.items(), key=lambda kv: kv[1]["delta"]["compounds"],
                         reverse=True):
        bdb = nrec = ""
        if "bdb" in r:
            bdb = (f"{r['bdb']['shipped']['buckets'].get('agree', 0)}"
                   f" → {r['bdb']['union']['buckets'].get('agree', 0)}")
            nrec = (f"{r['bdb']['shipped']['buckets'].get('no_record', 0)}"
                    f" → {r['bdb']['union']['buckets'].get('no_record', 0)}")
        frz = "yes" if r["snapshot"]["frozen"] else "no"
        if r["snapshot"].get("masking"):
            frz = "**MASKING**"
        L.append(f"| {pid} | {r['shipped']['compounds']:,} | "
                 f"{r['union']['compounds']:,} | "
                 f"{r['delta']['compounds']:+,} | "
                 f"{r['combined']['compounds']:,} | {r['only_shipped']:,} | "
                 f"{r['only_union']:,} | {r['shipped']['records']:,} | "
                 f"{r['union']['records']:,} | {r['delta']['records']:+,} | "
                 f"{bdb} | {nrec} | {frz} |")
    L.append("")

    L.append("## Frozen snapshots\n")
    masking = t["snapshot_masking"]
    L.append(f"**All {len(t['frozen'])} of {out['n_patents']}** patents in the "
             "reconciliation set are frozen. A snapshot pins the answer so a "
             "later patch cannot move it, so no capability patch bought from "
             "here on can improve any patent in this table until it is "
             "`thaw()`ed — the divergence above is pinned in place, not "
             "self-correcting.\n")
    L.append(f"Corpus-wide there are **{cov['snapshots']} snapshots holding "
             f"{cov['snapshot_compounds']:,} compounds**, of which "
             f"{cov['frozen_never_shipped']} patents were frozen but never "
             "shipped an `assay_tables.json`. The snapshot is fed "
             "`extract_from_patent(...) + repaired` "
             "(`process_patent.py:1834-1836`) — the exact union the shipped "
             "file never receives — so the audit artifact is currently a more "
             "complete record of what we extracted than the product artifact.\n")
    L.append("A snapshot masks a CHANGE only where the live recompute now "
             "disagrees with what was pinned:\n")
    if masking:
        L.append("| patent | pinned | live | drift | journal head |")
        L.append("|---|---:|---:|---:|---|")
        for pid in masking:
            s = per[pid]["snapshot"]
            L.append(f"| {pid} | {s['compounds']:,} | {s['live_compounds']:,} | "
                     f"{s['drift']:+,} | `{s['journal_head']}` |")
    else:
        L.append("No frozen snapshot currently disagrees with a live recompute.")
    L.append("")

    if cov["xml_without_shipped"]:
        L.append("## Cached XML with no shipped artifact\n")
        L.append(f"{len(cov['xml_without_shipped'])} patents the eval scripts "
                 "count but that `process_patent` never wrote an artifact for. "
                 "They are the bulk of the difference between "
                 "`pipeline_bench`'s patent count and the shipped corpus.\n")
        L.append("```")
        L.append(" ".join(cov["xml_without_shipped"]))
        L.append("```")
    return "\n".join(L) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", help="comma-separated subset")
    ap.add_argument("--no-bdb", action="store_true",
                    help="skip the BindingDB comparison (avoids the TSV read)")
    ap.add_argument("--out-dir", default="docs/reports",
                    help="where the markdown + JSON go")
    a = ap.parse_args()

    pats = [p.strip().upper() for p in a.patents.split(",")] if a.patents else None
    out = measure(pats, with_bdb=not a.no_bdb)
    t = out["totals"]

    print(f"patents (xml AND shipped) : {out['n_patents']}")
    print(f"SHIPPED   : {t['shipped']['records']:,} records, "
          f"{t['shipped']['compounds']:,} compounds, "
          f"{len(t['shipped']['zeros'])} zeros")
    print(f"UNION     : {t['union']['records']:,} records, "
          f"{t['union']['compounds']:,} compounds, "
          f"{len(t['union']['zeros'])} zeros")
    print(f"DELTA     : {t['delta']['records']:+,} records, "
          f"{t['delta']['compounds']:+,} compounds")
    print(f"COMBINED  : {t['combined']['compounds']:,} compounds "
          f"(only-shipped {t['only_shipped']:,}, only-union {t['only_union']:,})")
    if "bdb" in t:
        for art in ("shipped", "union", "combined"):
            b = t["bdb"][art]
            print(f"BDB {art:8s}: agree={b.get('agree', 0)} "
                  f"range_contains={b.get('range_contains', 0)} "
                  f"no_record={b.get('no_record', 0)} refs={b.get('refs', 0)}")
    print(f"frozen    : {len(t['frozen'])}  masking: {t['snapshot_masking']}")

    from ...core import config
    dest = config.REPO_ROOT / a.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "baseline.json").write_text(json.dumps(out, indent=1, default=str))
    (dest / "baseline.md").write_text(_markdown(out))
    print(f"wrote {dest / 'baseline.md'} and {dest / 'baseline.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
