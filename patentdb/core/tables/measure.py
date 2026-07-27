"""Table → measurements (BLUEPRINT L2 classify · L3 bind · L4 emit).

Deterministic, header-driven. Given a parsed `Table` (from either adapter):
  L2  classify by header  → assay | property | abbreviation | r_group | other
  L3  for an assay table  → find the cid column; parse each value-column header
                            into (assay_name, unit); decode letter-grade legends
  L4  emit one AssayMeasurement per (row, value-column) with full provenance.

No LLM here. A header a deterministic parse can't resolve is kept VERBATIM as the
assay name (provenance="verbatim") for L5 to upgrade later — never dropped, never
stamped with a positional placeholder.
"""
from __future__ import annotations

import re

from .model import AssayMeasurement, Table

_KIND = re.compile(r"\b([IEAGC]C\s?50|K\s?[id]|CC\s?50|GI\s?50|pIC50|pEC50|ED\s?50)\b", re.I)
_UNIT = re.compile(r"\b(nM|µM|μM|uM|mM|pM|M|mol/L|nmol/L|µmol/L|%)\b", re.I)
_PROP = re.compile(
    r"\b(retention|\(min\)|\[M\s*\+\s*H\]|m/z|mol\.?\s*wt|MW\b|molecular weight|"
    r"logP|logD|melting|\bm\.?p\.?\b|clogp|tpsa|HPLC|LCMS|purity|yield|"
    r"calcd|calc'?d|found|exact mass)\b", re.I)
_ABBR = re.compile(r"\b(abbreviation|meaning|definition|term|glossary)\b", re.I)
_RGROUP = re.compile(r"^R\s?\d|^R[a-z]?$|substituent|scaffold", re.I)
_CIDLAB = re.compile(r"\b(ex(?:ample)?|cmp|cpd|compound|ref)\.?\s*(?:no\.?|#|number)?\b", re.I)
_CID_CELL = re.compile(r"^(?:cpd\.?|compound|example|ex\.?)?\s*[A-Za-z]{0,3}-?\d{1,5}[A-Za-z]{0,2}$", re.I)

# value parsing
_RUNS = re.compile(r"\((\d+)\)\s*$")
_GRADE = re.compile(r"^\++$|^[A-E]$")
_NOTVAL = {"", "—", "-", "nt", "n.t.", "nd", "n.d.", "na", "n/a", "*", "ND", "NT"}


def classify_table(t: Table) -> str:
    """L2: classify by header text."""
    hdr = " ".join(t.header)
    cells = " ".join(c for r in t.rows[:6] for c in r)
    if _ABBR.search(hdr):
        return "abbreviation"
    has_assay = bool(_KIND.search(hdr)) or bool(_KIND.search(cells))
    has_prop = bool(_PROP.search(hdr))
    if has_assay and not (has_prop and not _KIND.search(hdr)):
        return "assay"
    if has_prop:
        return "property"
    if _RGROUP.search(hdr):
        return "r_group"
    return "other"


def _cid_column(t: Table) -> int:
    """Index of the compound-id column: a header that looks like a cid label,
    else the column whose cells are most cid-shaped."""
    for i, h in enumerate(t.header):
        if _CIDLAB.search(h):
            return i
    best, best_frac = 0, -1.0
    for i in range(t.n_cols):
        col = t.column(i)
        frac = sum(1 for c in col if _CID_CELL.match(c.strip())) / max(1, len(col))
        if frac > best_frac:
            best, best_frac = i, frac
    return best


def _parse_value_header(h: str) -> tuple[str, str] | None:
    """Value-column header → (assay_name, unit). None if it isn't an assay column
    (e.g. a stray margin-number column or an empty header)."""
    h = h.strip().replace("[", "(").replace("]", ")")     # "[mol/L]" → "(mol/L)"
    if not h or h.isdigit() or h in {"(", ")", "()"}:
        return None
    um = _UNIT.search(h)
    unit = um.group(1) if um else ""
    # assay name = header minus the unit parenthetical/token, cleaned
    name = re.sub(r"\(?\s*" + re.escape(unit) + r"[^)]*\)?", " ", h, count=1) if unit else h
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)            # drop leftover parens
    name = re.sub(r"\s+", " ", name).strip(" ,;.")
    if not name:
        name = h
    return name, unit.replace("uM", "µM").replace("μM", "µM")


def _parse_value(raw: str) -> dict | None:
    """Raw cell → {value_numeric|value_low/high, qualifier, encoding, value_raw}.
    None for not-tested / empty cells."""
    s = raw.strip()
    if s in _NOTVAL or not s:
        return None
    runs = _RUNS.search(s)
    core = _RUNS.sub("", s).strip() if runs else s
    n_runs = int(runs.group(1)) if runs else None
    if _GRADE.match(core):                                 # letter grade
        return {"value_raw": core, "encoding": "grade", "n_runs": n_runs}
    qual = None
    m = re.match(r"^([<>≤≥~])\s*", core)
    if m:
        qual = m.group(1)
        core = core[m.end():]
    core = core.replace("−", "-").replace("–", "-")          # unicode minus → ASCII
    core = core.replace(",", ".") if re.fullmatch(r"\d+,\d+", core) else core
    try:
        val = float(core.replace("×10", "e").replace("·10", "e").replace(" ", ""))
    except ValueError:
        return None
    return {"value_numeric": val, "qualifier": qual, "encoding": "numeric",
            "value_raw": s, "n_runs": n_runs}


def table_to_measurements(t: Table) -> list[AssayMeasurement]:
    """L3+L4: an assay Table → AssayMeasurements (one per row × value-column)."""
    if classify_table(t) != "assay":
        return []
    cid_i = _cid_column(t)
    # resolve each value column's header once
    col_assay: dict[int, tuple[str, str]] = {}
    for i in range(t.n_cols):
        if i == cid_i:
            continue
        parsed = _parse_value_header(t.header[i])
        if parsed:
            col_assay[i] = parsed
    if not col_assay:
        return []
    out: list[AssayMeasurement] = []
    for r in t.rows:
        if cid_i >= len(r):
            continue
        cid = r[cid_i].strip().rstrip(".")               # "61." → "61"
        if not cid or not _CID_CELL.match(cid):
            continue
        for i, (assay, unit) in col_assay.items():
            if i >= len(r):
                continue
            pv = _parse_value(r[i])
            if pv is None:
                continue
            out.append(AssayMeasurement(
                cid=cid, assay=assay, unit=unit,
                value_raw=pv.get("value_raw", r[i]),
                value_numeric=pv.get("value_numeric"),
                qualifier=pv.get("qualifier"),
                encoding=pv["encoding"],
                provenance={
                    "source": t.source, "table_label": t.table_label,
                    "page": t.page, "header_verbatim": t.header[i],
                    "resolution": "verbatim", "n_runs": pv.get("n_runs"),
                },
            ))
    return out
