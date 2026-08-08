"""Reconcile MinerU + GP measurement candidates → one set (BLUEPRINT reconciler).

Neither source is trusted blindly. Alignment is by (normalized cid, assay-kind);
values decide the rest. Rules (chosen by the user):
  • AGREE  — same cid+kind, values match → one record, the RICHEST label across
             the pair, provenance both_agree, confidence high.
  • CONFLICT — same cid+kind, values differ → prefer the value that appears
             VERBATIM near the cid in the OTHER source's raw text (corroboration).
             If neither corroborates, keep BOTH, flagged conflict (low) — never
             silently pick.
  • ONE SOURCE only → take it (mineru_only / gp_only), confidence medium.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace

from .llm_reconcile import resolve_groups
from .measure import _KIND
from .model import AssayMeasurement
from .resolution_memory import ResolutionMemory

_STOP = {"ic", "ic50", "ec50", "ki", "kd", "cc50", "gi50", "ed50", "binding",
         "assay", "value", "the", "of", "wild", "type"}


def _norm_cid(c: str) -> str:
    m = re.match(r"^([A-Za-z]*)-?0*(\d+)([A-Za-z]*)$", (c or "").strip())
    return f"{m.group(1)}{m.group(2)}{m.group(3)}".upper() if m else (c or "").strip().upper()


def _kind(assay: str) -> str:
    m = _KIND.search(assay or "")
    return re.sub(r"\s", "", m.group(1).lower()) if m else re.sub(r"[^a-z]", "", (assay or "").lower())[:10]


def _richness(assay: str) -> tuple[int, int]:
    # More alphabetic content = more target context (MinerU grid headers usually
    # carry the full target; GP often loses it). Length-first, distinct-tokens 2nd.
    alpha = len(re.sub(r"[^A-Za-z]", "", assay or ""))
    toks = {t for t in re.findall(r"[a-z0-9]+", (assay or "").lower()) if t not in _STOP}
    return (alpha, len(toks))


def _agree(a: AssayMeasurement, b: AssayMeasurement) -> bool:
    if a.value_numeric is not None and b.value_numeric is not None:
        x, y = a.value_numeric, b.value_numeric
        return round(x, 4) == round(y, 4) or abs(x - y) <= 0.05 * max(abs(x), abs(y), 1e-12)
    if a.encoding == "grade" and b.encoding == "grade":
        return a.value_raw.strip() == b.value_raw.strip()
    return False


def _vstr(m: AssayMeasurement) -> str:
    return re.sub(r"\(\d+\)\s*$", "", m.value_raw).strip()


def _corroborated(m: AssayMeasurement, raw: str, cid: str) -> bool:
    """The value appears verbatim within 120 chars after the cid in `raw`."""
    v = _vstr(m)
    if not raw or not v:
        return False
    for cm in re.finditer(re.escape(cid), raw):
        if v in raw[cm.end(): cm.end() + 120]:
            return True
    return False


def _richest(a: AssayMeasurement, b: AssayMeasurement) -> str:
    return a.assay if _richness(a.assay) >= _richness(b.assay) else b.assay


def _tag(m: AssayMeasurement, resolution: str, confidence: str, **extra) -> AssayMeasurement:
    return replace(m, provenance={**m.provenance, "resolution": resolution,
                                  "confidence": confidence, **extra})


def reconcile(mineru: list[AssayMeasurement], gp: list[AssayMeasurement], *,
              mineru_raw: str = "", gp_raw: str = "") -> list[AssayMeasurement]:
    by: dict = defaultdict(lambda: {"m": [], "g": []})
    for x in mineru:
        by[_norm_cid(x.cid)]["m"].append(x)
    for x in gp:
        by[_norm_cid(x.cid)]["g"].append(x)

    out: list[AssayMeasurement] = []
    for d in by.values():
        gused: set[int] = set()
        for m in d["m"]:
            mk = _kind(m.assay)
            agreed = conflict = None
            for j, g in enumerate(d["g"]):
                if j in gused or _kind(g.assay) != mk:
                    continue
                if _agree(m, g):
                    agreed = j
                    break
                if conflict is None:
                    conflict = j
            if agreed is not None:
                g = d["g"][agreed]
                gused.add(agreed)
                out.append(_tag(replace(m, assay=_richest(m, g), unit=m.unit or g.unit),
                                "both_agree", "high"))
            elif conflict is not None:
                g = d["g"][conflict]
                gused.add(conflict)
                mc = _corroborated(m, gp_raw, m.cid)
                gc = _corroborated(g, mineru_raw, g.cid)
                if mc and not gc:
                    out.append(_tag(m, "conflict_resolved", "med", chose="mineru", lost=_vstr(g)))
                elif gc and not mc:
                    out.append(_tag(replace(g, assay=_richest(m, g)), "conflict_resolved",
                                    "med", chose="gp", lost=_vstr(m)))
                else:                                   # neither corroborated → keep BOTH, flag
                    out.append(_tag(m, "conflict", "low", other=_vstr(g)))
                    out.append(_tag(g, "conflict", "low", other=_vstr(m)))
            else:
                out.append(_tag(m, "mineru_only", "med"))
        for j, g in enumerate(d["g"]):
            if j not in gused:
                out.append(_tag(g, "gp_only", "med"))
    return out


# ── LLM reconciliation layer (BLUEPRINT L5) ─────────────────────────────────

def _pair_value(m: AssayMeasurement):
    return m.value_numeric if m.value_numeric is not None else m.value_raw


def _as_value(rv) -> tuple[float | None, str, str]:
    if isinstance(rv, (int, float)):
        return float(rv), f"{rv:g}", "numeric"
    s = str(rv).strip()
    if re.fullmatch(r"\++|[A-E]", s):
        return None, s, "grade"
    try:
        return float(s), s, "numeric"
    except ValueError:
        return None, s, "numeric"


def _canon(v) -> object:
    """Canonical form for comparing a resolved value against source candidates."""
    try:
        return round(float(str(v).replace("−", "-")), 6)
    except (ValueError, TypeError):
        return str(v).strip()


def reconcile_with_llm(mineru: list[AssayMeasurement], gp: list[AssayMeasurement], *,
                       patent_id: str, mineru_raw: str = "", gp_raw: str = "",
                       memory: ResolutionMemory | None = None) -> list[AssayMeasurement]:
    """Deterministic reconcile + LLM resolution of the flagged conflicts.

    The deterministic pass settles every agreeing / single-source measurement
    (no LLM). Only the genuine conflicts go to the LLM, which sees BOTH sources'
    full cid→value lists per assay-kind (agreeing compounds as anchors) and
    returns the correct values — fixing OCR quirks (off-by-one, mangled cells)
    without any per-patent code. Resolutions are cached in resolution_memory, so
    re-runs and a stable corpus cost ~0 LLM. Degrades safely: if the LLM is
    unavailable, unresolved conflicts stay flagged (never silently picked).
    """
    det = reconcile(mineru, gp, mineru_raw=mineru_raw, gp_raw=gp_raw)
    conflicts = [m for m in det if m.provenance.get("resolution") == "conflict"]
    if not conflicts:
        return det
    memory = memory or ResolutionMemory()

    a_by_kind: dict = defaultdict(dict)
    b_by_kind: dict = defaultdict(dict)
    for m in mineru:
        a_by_kind[_kind(m.assay)][_norm_cid(m.cid)] = _pair_value(m)
    for m in gp:
        b_by_kind[_kind(m.assay)][_norm_cid(m.cid)] = _pair_value(m)

    ckinds = sorted({_kind(m.assay) for m in conflicts})
    groups = [{"patent": patent_id, "kind": k,
               "a_pairs": a_by_kind.get(k, {}), "b_pairs": b_by_kind.get(k, {})}
              for k in ckinds]
    resolved = resolve_groups(groups, memory=memory, patent_id=patent_id)
    kind_res = {ckinds[i]: r for i, r in resolved.items()}

    # richest conflict record per (cid, kind) — carries the best label
    best: dict = {}
    for m in conflicts:
        key = (_norm_cid(m.cid), _kind(m.assay))
        if key not in best or _richness(m.assay) > _richness(best[key].assay):
            best[key] = m

    # Bulletproof: the LLM must CHOOSE a real source value (possibly from a
    # shifted position), never invent one. Reject resolutions not present in
    # either source's candidates for that kind → keep flagged for review.
    valid: dict = {}
    for k in ckinds:
        vs = list(a_by_kind.get(k, {}).values()) + list(b_by_kind.get(k, {}).values())
        valid[k] = {_canon(v) for v in vs}

    out = [m for m in det if m.provenance.get("resolution") != "conflict"]
    for (ncid, k), m in best.items():
        entry = kind_res.get(k) or {}
        rv = entry.get("resolved", {}).get(ncid)
        conf = entry.get("confidence", "low")
        if (rv is None or str(rv).strip().lower() in ("", "null", "none")
                or _canon(rv) not in valid.get(k, set())):
            out.append(_tag(m, "conflict", "low"))           # unresolved / invented → flag
            continue
        vn, vraw, enc = _as_value(rv)
        # propagate the LLM's confidence: low-confidence resolutions are applied
        # for this run but tagged low (review queue), never reported as high.
        out.append(_tag(replace(m, value_numeric=vn, value_raw=vraw, encoding=enc),
                        "llm_reconciled", conf))
    return out
