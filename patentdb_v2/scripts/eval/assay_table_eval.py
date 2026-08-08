"""Head-to-head OCR-tool quality evaluation for assay table extraction.

Loads ground-truth tables from `ground_truth_assay_tables.json`. For each
candidate OCR source, reads the OCR'd markdown for the same patent+page,
runs a MINIMAL no-heuristic parser, and scores how close the extracted
(compound_id, assay_name, value, unit, qualifier) tuples are to ground truth.

The minimal parser does NOT have:
- mojibake repair
- side-by-side splitting
- bleed-row coalescing
- title-row promotion
- smart unit detection

That's intentional. The whole point is to measure how much of our current
compensation logic each OCR tool actually NEEDS. A perfect tool would let
the minimal parser score 100% on every ground-truth table.

Usage:
    python3 -m patentdb.scripts.eval.assay_table_eval --tools current
    python3 -m patentdb.scripts.eval.assay_table_eval --tools current,mineru
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the patentdb package importable when run as a script
THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from patentdb.core import config


GROUND_TRUTH_PATH = THIS_DIR / "ground_truth_assay_tables.json"


# ── OCR-source resolution ────────────────────────────────────────────────

# Each tool puts its markdown output in a different sibling directory
# next to the patent's PDF. The harness tries the directories in order
# until it finds the page file.
TOOL_TO_DIRS = {
    "current":     ["all_pages"],                # the existing pre-rendered PaddleX output
    "iupacs":      ["iupacs_clean"],             # the upstream's "second pass" filtered output
    "paddlex_v3":  ["all_pages_paddlex_v3"],     # not yet generated
    "mineru":      ["all_pages_mineru"],         # not yet generated
}


def resolve_ocr_path(patent_id: str, page: int, tool: str) -> Path | None:
    """Return path to the markdown file for (patent_id, page) under the chosen tool's dir."""
    candidates = TOOL_TO_DIRS.get(tool, [tool])  # tool name itself is also tried
    for sub in candidates:
        path = config.DATA_DIR / patent_id / sub / f"page_{page:04d}.md"
        if path.exists():
            return path
    return None


# ── Minimal parser (intentionally dumb) ──────────────────────────────────

# Patterns we recognize for "this column has assay values"
ASSAY_HEADER_RE = re.compile(
    r'IC\s*5[0o]|EC\s*5[0o]|Ki\b|Kd\b|FRET|Affinity|Binding|Antiviral',
    re.IGNORECASE
)
UNIT_RE = re.compile(
    r'[\(\[]\s*(nM|μM|µM|uM|mM|Î¼M)\s*[\)\]]',
    re.IGNORECASE
)


def strip_latex(text: str) -> str:
    """Remove LaTeX wrappers so plain regexes can match.
    Handles `$\\mathrm{IC}_{50}$` → `IC50`, `\\mathrm { I C }` → `IC`, etc.
    """
    t = text
    # Remove $...$ wrappers but keep inner content
    t = re.sub(r'\$([^$]*)\$', r'\1', t)
    # Remove \mathrm, \text, \rm wrappers
    t = re.sub(r'\\(?:mathrm|text|rm)\s*\{([^}]*)\}', r'\1', t)
    # Remove subscript braces: _{50} → 50
    t = re.sub(r'_\s*\{\s*([^}]*)\s*\}', r'\1', t)
    # Collapse spaces inside what's left of `{...}` (LaTeX adds spaces around chars)
    t = re.sub(r'\s+', '', t) if any(c in text for c in ('$', '\\', '{')) else t
    return t
QUALIFIER_RE = re.compile(r'^([<>~≤≥]?)\s*(.+)$')
NUMBER_RE = re.compile(r'^[<>~≤≥]?\s*(\d+(?:\.\d+)?)$')


def normalize_unit(raw: str) -> str:
    raw = raw.strip()
    if raw.lower() in ('um', 'μm', 'µm', 'î¼m'):
        return 'μM'
    if raw.lower() == 'nm':
        return 'nM'
    if raw.lower() == 'mm':
        return 'mM'
    return raw


def parse_value(cell: str) -> tuple[float | None, str | None]:
    """Parse a cell into (numeric_value, qualifier). Returns (None, None) if not a number."""
    raw = cell.strip()
    raw = re.sub(r'&amp;gt;', '>', raw)
    raw = re.sub(r'&amp;lt;', '<', raw)
    raw = re.sub(r'&gt;', '>', raw)
    raw = re.sub(r'&lt;', '<', raw)
    if not raw:
        return None, None
    qualifier = None
    if raw.startswith(('<', '>', '~', '≤', '≥')):
        qualifier = raw[0]
        raw = raw[1:].strip()
    try:
        return float(raw), qualifier
    except ValueError:
        return None, qualifier


def parse_table_minimal(text: str, table_index: int = 0) -> list[dict]:
    """Extract (compound_id, assay_name, value, unit, qualifier) tuples from the
    `table_index`-th `<table>` block in `text`.

    NO HEURISTICS. No mojibake repair, no splitting, no coalescing. Just:
      - Take row 0 as headers
      - First column is compound_id
      - Other columns are assays if header matches ASSAY_HEADER_RE
      - For each subsequent row, emit (compound_id, header, value, unit, qualifier) tuples
    """
    tables = re.findall(r'<table>(.*?)</table>', text, re.DOTALL | re.IGNORECASE)
    if table_index >= len(tables):
        return []
    rows = re.findall(r'<tr>(.*?)</tr>', tables[table_index], re.DOTALL | re.IGNORECASE)
    if not rows:
        return []
    cell_re = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)
    def cells_of(row):
        return [re.sub(r'<[^>]+>', '', c).strip() for c in cell_re.findall(row)]

    # Find the "real" header row: skip title rows (single cell, or all cells
    # equal/empty). Header row is the first row with ≥2 distinct non-empty cells.
    # Also accumulate any title-row text as `table_context` — used as the assay
    # name when subsequent header rows lack an assay-keyword cell.
    table_context = ""
    header_idx = 0
    for i, r in enumerate(rows):
        cs = cells_of(r)
        non_empty = [c for c in cs if c.strip()]
        if len(non_empty) <= 1:
            # Title-row (colspan) or empty row — accumulate text as context
            if non_empty:
                table_context += " " + non_empty[0]
            continue
        # Real header row found
        header_idx = i
        break
    headers = cells_of(rows[header_idx])
    if not headers:
        return []

    # Identify which column is the compound id and which are assays
    cpd_col_idx = 0
    for i, h in enumerate(headers):
        if re.search(r'cpd|compound|example|ex\.?\s*no', h, re.I):
            cpd_col_idx = i
            break

    assay_cols = []
    for i, h in enumerate(headers):
        if i == cpd_col_idx:
            continue
        if not h:
            continue
        h_clean = strip_latex(h)
        ctx_clean = strip_latex(table_context)
        if ASSAY_HEADER_RE.search(h_clean) or ASSAY_HEADER_RE.search(ctx_clean):
            unit_m = UNIT_RE.search(h_clean) or UNIT_RE.search(ctx_clean)
            unit = normalize_unit(unit_m.group(1)) if unit_m else ''
            # Strip the unit/parens from the assay name for cleanliness
            assay_name = re.sub(r'\s*[\(\[][^)\]]*[\)\]]\s*$', '', h_clean).strip()
            if not assay_name:
                # If header cell is empty/junk, fall back to context-derived name
                assay_name = re.sub(r'\s*[\(\[][^)\]]*[\)\]]\s*', '', ctx_clean).strip()[:60]
            assay_cols.append((i, assay_name, unit))

    out = []
    for row in rows[header_idx + 1:]:
        cs = cells_of(row)
        if len(cs) <= cpd_col_idx:
            continue
        cpd_id = cs[cpd_col_idx].strip()
        if not cpd_id:
            continue
        # Filter pure noise rows (no digit in compound id)
        if not re.search(r'\d', cpd_id):
            continue
        for col_idx, name, unit in assay_cols:
            if col_idx >= len(cs):
                continue
            value, qualifier = parse_value(cs[col_idx])
            if value is None:
                continue
            out.append({
                "compound_id": cpd_id,
                "assay_name": name,
                "value": value,
                "unit": unit,
                "qualifier": qualifier,
            })
    return out


# ── Scoring ──────────────────────────────────────────────────────────────

def normalize_compound_id(s: str) -> str:
    return re.sub(r'^(?:cpd\.?\s*no\.?\s*|compound\s+|example\s+|ex\.?\s*no\.?\s*)', '',
                  s.strip(), flags=re.IGNORECASE).lower()


@dataclass
class Score:
    table_id: str
    n_expected: int
    n_extracted: int
    exact_matches: int   # all 5 fields match
    value_matches: int   # compound_id + value match (ignore name/unit drift)
    cpd_matches: int     # compound_id present in extraction

    @property
    def precision(self) -> float:
        return self.exact_matches / max(self.n_extracted, 1)

    @property
    def recall(self) -> float:
        return self.exact_matches / max(self.n_expected, 1)

    @property
    def value_recall(self) -> float:
        return self.value_matches / max(self.n_expected, 1)


def score_table(expected: list[dict], extracted: list[dict], table_id: str) -> Score:
    # Build lookup keys
    def key_full(row):
        return (
            normalize_compound_id(row['compound_id']),
            row.get('assay_name', '').strip().lower(),
            float(row['value']),
            normalize_unit(row.get('unit', '') or ''),
            (row.get('qualifier') or '').strip(),
        )
    def key_value(row):
        return (normalize_compound_id(row['compound_id']), float(row['value']))
    def key_cpd(row):
        return normalize_compound_id(row['compound_id'])

    extracted_full = {key_full(r) for r in extracted}
    extracted_value = {key_value(r) for r in extracted}
    extracted_cpd = {key_cpd(r) for r in extracted}

    exact = sum(1 for e in expected if key_full(e) in extracted_full)
    value = sum(1 for e in expected if key_value(e) in extracted_value)
    cpd = sum(1 for e in expected if key_cpd(e) in extracted_cpd)

    return Score(
        table_id=table_id,
        n_expected=len(expected),
        n_extracted=len(extracted),
        exact_matches=exact,
        value_matches=value,
        cpd_matches=cpd,
    )


# ── Main ─────────────────────────────────────────────────────────────────

def run_for_tool(gt_data: dict, tool: str) -> list[Score]:
    scores = []
    for table in gt_data['tables']:
        pid = table['patent_id']
        page = table['page']
        tidx = table.get('table_index', 0)
        table_id = f"{pid}/p{page}#{tidx}"

        ocr_path = resolve_ocr_path(pid, page, tool)
        if ocr_path is None:
            print(f"  {table_id}: OCR file missing for tool={tool} — skipping")
            continue
        text = ocr_path.read_text(encoding='utf-8')
        extracted = parse_table_minimal(text, tidx)
        s = score_table(table['expected'], extracted, table_id)
        scores.append(s)
    return scores


def print_table_report(tool: str, scores: list[Score]):
    if not scores:
        print(f"\n=== {tool}: no scores (all OCR files missing) ===")
        return
    print(f"\n=== {tool} ===")
    print(f"  {'table':<28} {'expected':>8} {'extr':>5} {'exact':>5} {'val':>5} {'cpd':>5}  {'fmt':>6}")
    print(f"  {'-'*28} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5}  {'-'*6}")
    total_e = total_x = total_em = total_vm = total_cm = 0
    for s in scores:
        em_pct = f"{100*s.exact_matches/max(s.n_expected,1):.0f}%"
        vm_pct = f"{100*s.value_matches/max(s.n_expected,1):.0f}%"
        cm_pct = f"{100*s.cpd_matches/max(s.n_expected,1):.0f}%"
        print(f"  {s.table_id:<28} {s.n_expected:>8} {s.n_extracted:>5} "
              f"{s.exact_matches:>3} ({em_pct:>4}) {s.value_matches:>3}({vm_pct:>4}) "
              f"{s.cpd_matches:>3}({cm_pct:>4})")
        total_e += s.n_expected
        total_x += s.n_extracted
        total_em += s.exact_matches
        total_vm += s.value_matches
        total_cm += s.cpd_matches
    print(f"  {'-'*28}")
    em_total = f"{100*total_em/max(total_e,1):.0f}%"
    vm_total = f"{100*total_vm/max(total_e,1):.0f}%"
    cm_total = f"{100*total_cm/max(total_e,1):.0f}%"
    print(f"  {'TOTAL':<28} {total_e:>8} {total_x:>5} "
          f"{total_em:>3}({em_total:>4}) {total_vm:>3}({vm_total:>4}) "
          f"{total_cm:>3}({cm_total:>4})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default="current",
                    help="Comma-separated tool names from TOOL_TO_DIRS keys (e.g. 'current,iupacs,mineru')")
    ap.add_argument("--gt", default=str(GROUND_TRUTH_PATH))
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    print(f"Ground-truth: {len(gt['tables'])} tables, "
          f"{sum(len(t['expected']) for t in gt['tables'])} expected (cpd, assay, value, unit, qual) tuples")

    for tool in args.tools.split(","):
        scores = run_for_tool(gt, tool.strip())
        print_table_report(tool.strip(), scores)


if __name__ == "__main__":
    main()
