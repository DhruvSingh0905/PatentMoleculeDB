"""Google-Patents flat-text → Table candidates (BLUEPRINT L1, second source).

GP serves the description as flat text (no grid). Assay tables appear
ROW-INTERLEAVED: a header phrase (cid label + one or more "<target> IC50 (unit)"
columns) immediately followed by "<cid> <val> <val> …" groups repeated. We:
  1. anchor on a cid-label that has an assay-kind token right after it (rejects
     prose like "Ex 320nm" excitation / "for example"),
  2. find where the data run starts (first cid+value that repeats),
  3. split the header span into columns at each assay-kind's unit boundary,
  4. walk the data grouping cid + N values (attaching "(n)" run-counts).

Deliberately NOT parsed here: column-flattened bin lists ("A190 A191 …" + a
+/++ legend) → the legend module; pure prose → L5 resolution. GP is ONE of TWO
candidate sources; a deterministic reconciler chooses against MinerU.
"""
from __future__ import annotations

import re

from ..assay_fsm.normalizer import repair_mojibake
from .model import Table

_KIND = re.compile(r"\b([IEAGC]C\s?50|K\s?[id]|CC\s?50|GI\s?50|pIC50|pEC50|ED\s?50)\b", re.I)
_CIDLAB = re.compile(
    r"\b((?:Ex(?:ample)?|Cmp|Cpd|Compound)\.?\s*(?:No\.?|#|Number)?)\b", re.I
)
_CID = re.compile(r"^[A-Za-z]{0,3}-?\d{1,5}[A-Za-z]{0,2}\.?$")   # trailing "." ok ("61.")


def _is_cid(s: str) -> bool:
    # a cid-shaped token that is NOT itself an assay-kind word ("IC50", "Ki" …)
    return bool(_CID.match(s)) and not _KIND.match(s)
_VAL = re.compile(
    r"^(?:[<>≤≥~]?\s*\d[\d.,]*(?:\s*[eE][+-]?\d+)?|\++|[A-E]|NT|ND|n\.?d\.?|—)$",
    re.I,
)
_RUNS = re.compile(r"^\(\d+\)$")                       # "(8)" replicate count
_UNIT = re.compile(r"\s*,?\s*(?:\([^)]*\)|[µμu]?M|nM|gmean|mol/L|%|nm)+", re.I)
_TOKEN = re.compile(r"\S+")
_STOP = re.compile(r"(?i)^(table|key|\*key|example[s]?|note|wherein|the)$")


def _split_header(header_text: str) -> tuple[str, list[str]]:
    """Header span → (cid_label, [column label per assay-kind, unit-bounded])."""
    m = _CIDLAB.search(header_text)
    cid_label = m.group(1).strip() if m else "id"
    rest = header_text[m.end():] if m else header_text
    kinds = list(_KIND.finditer(rest))
    if not kinds:
        return cid_label, []
    cols, prev = [], 0
    for k in kinds:
        um = _UNIT.match(rest, k.end())
        end = um.end() if um else k.end()
        cols.append(re.sub(r"\s+", " ", rest[prev:end]).strip(" ,;.#"))
        prev = end
    return cid_label, cols


def _find_data_start(tokens: list[str]) -> int:
    """Index of the first token beginning the repeating <cid> <value> data run."""
    for i in range(len(tokens) - 1):
        if _is_cid(tokens[i]) and _VAL.match(tokens[i + 1]):
            hits = sum(
                1 for j in range(i, min(len(tokens) - 1, i + 18))
                if _is_cid(tokens[j]) and _VAL.match(tokens[j + 1])
            )
            if hits >= 3:
                return i
    return -1


def _parse_region(tokens: list[str], n_cols: int) -> list[tuple[str, ...]]:
    """Walk tokens grouping <cid> + n_cols values into rows; "(n)" run-counts
    attach to the preceding value. Stops when the data pattern breaks."""
    rows: list[tuple[str, ...]] = []
    i, misses = 0, 0
    while i < len(tokens):
        if _STOP.match(tokens[i]):
            break
        if _is_cid(tokens[i]):
            cells: list[str] = []
            j = i + 1
            while j < len(tokens) and len(cells) < n_cols:
                tj = tokens[j]
                if _RUNS.match(tj) and cells:          # "(8)" → attach to last value
                    cells[-1] += " " + tj
                    j += 1
                elif _VAL.match(tj):
                    cells.append(tj)
                    j += 1
                else:
                    break
            if cells:
                rows.append((tokens[i], *cells, *([""] * (n_cols - len(cells)))))
                i, misses = j, 0
                continue
        misses += 1
        if misses > 40 and rows:
            break
        i += 1
    return rows


def parse_gp_text(patent_id: str, text: str | None = None) -> list[Table]:
    """Find row-interleaved assay tables in GP text → Table candidates."""
    if text is None:
        import json
        from .. import config
        p = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
        text = json.loads(p.read_text()).get("description", "") if p.exists() else ""
    # Normalise unicode minus / en-dash → ASCII so sci-notation ("7.90E−08")
    # parses. (Leave em-dash "—" alone — it's a not-tested marker.)
    text = repair_mojibake(text or "").replace("−", "-").replace("–", "-")

    out: list[Table] = []
    seen: list[int] = []
    for lab in _CIDLAB.finditer(text):
        if not _KIND.search(text[lab.end(): lab.end() + 40]):
            continue                                   # not a real assay header
        if any(abs(lab.start() - s) < 300 for s in seen):
            continue
        toks = _TOKEN.findall(text[lab.start(): lab.start() + 60000])
        ds = _find_data_start(toks)
        if not (2 <= ds <= 50):             # >50 header tokens ⇒ prose, not a header
            continue
        cid_label, cols = _split_header(" ".join(toks[:ds]))
        if not cols:
            continue
        rows = _parse_region(toks[ds:], len(cols))
        if len(rows) < 3:
            continue
        # Reject degenerate "cid == first value" parses (example-number tables
        # like US9540377 where the only "value" IS the example number repeated).
        same = sum(1 for r in rows if len(r) > 1 and r[0] == r[1])
        if same >= 0.5 * len(rows):
            continue
        seen.append(lab.start())
        header = (cid_label, *cols)
        w = len(header)
        body = tuple((r + ("",) * w)[:w] for r in rows)
        out.append(Table(header=header, rows=body, source="gp_text",
                         char_span=(lab.start(), lab.start())))

    # Dedupe candidates that captured the SAME data run (a prose anchor and the
    # real adjacent-header anchor both reach "A208 0.013 …"): keep the shortest
    # header — the clean one.
    best: dict = {}
    for t in out:
        key = t.rows[0] if t.rows else t.header
        cur = best.get(key)
        if cur is None or len(" ".join(t.header)) < len(" ".join(cur.header)):
            best[key] = t
    return list(best.values())
