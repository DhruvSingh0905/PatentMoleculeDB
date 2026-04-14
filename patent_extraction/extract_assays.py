"""Extract biological assay data (IC50, Ki, EC50) from patent pages.

Parses HTML tables in markdown files to find assay results per compound.
No API calls needed — pure regex/HTML parsing.

Supported formats:
  - <table> with "IC50", "Ki", "EC50" in column headers
  - Values: plain numbers, "<0.001", ">10", "0.090 ± 0.051", "No inhibition"
  - Units in column headers (nM, μM, uM)
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from . import config
from .models import AssayResult

logger = logging.getLogger(__name__)

# Patterns to find assay tables
ASSAY_KEYWORDS = re.compile(
    r'IC50|Ki\b|EC50|Kd\b|binding|inhibit|potency|activity',
    re.IGNORECASE,
)

UNIT_PATTERN = re.compile(r'\((nM|μM|uM|µM|mM)\)', re.IGNORECASE)

# Value parsing
VALUE_PATTERN = re.compile(
    r'^([<>≤≥~]?)\s*(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?$'
)


def _parse_value(raw: str) -> tuple[float | None, str | None]:
    """Parse an assay value string into (numeric, qualifier)."""
    raw = raw.strip()
    raw = html.unescape(raw)  # &lt; → <

    if not raw or raw in ('—', '-', 'N/A', 'n/a', 'NT', 'nd', 'ND'):
        return None, None

    if 'no inhib' in raw.lower() or 'inactive' in raw.lower():
        return None, '>'  # Map to ">" (above detection limit)

    m = VALUE_PATTERN.match(raw)
    if m:
        qualifier = m.group(1) or None
        value = float(m.group(2))
        return value, qualifier

    # Try extracting just the number
    nums = re.findall(r'(\d+\.?\d*)', raw)
    if nums:
        qualifier = None
        if '<' in raw or '&lt;' in raw:
            qualifier = '<'
        elif '>' in raw or '&gt;' in raw:
            qualifier = '>'
        return float(nums[0]), qualifier

    return None, None


def _parse_html_table(table_html: str) -> list[dict]:
    """Parse an HTML table into list of row dicts with header keys."""
    # Extract rows
    rows = re.findall(r'<tr>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
    if len(rows) < 2:
        return []

    def _cells(row_html):
        cells = []
        for m in re.finditer(r'<td([^>]*)>(.*?)</td>', row_html, re.DOTALL | re.IGNORECASE):
            attrs, content = m.group(1), m.group(2)
            text = re.sub(r'<[^>]+>', '', content).strip()
            # Handle colspan: repeat the cell
            colspan_m = re.search(r'colspan=["\']?(\d+)', attrs, re.IGNORECASE)
            colspan = int(colspan_m.group(1)) if colspan_m else 1
            cells.append(text)
            for _ in range(colspan - 1):
                cells.append(text)
        return cells

    headers = _cells(rows[0])
    if not headers:
        return []

    # Sometimes headers span two rows (subheaders)
    # Check if second row looks like subheaders (no numeric values)
    second_cells = _cells(rows[1]) if len(rows) > 1 else []
    data_start = 1
    if second_cells and not any(re.search(r'\d', c) for c in second_cells):
        # Use subheaders as separate columns if they look like assay names
        # Otherwise merge into parent headers
        if any(ASSAY_KEYWORDS.search(s) for s in second_cells):
            headers = second_cells
        else:
            merged = []
            for i, h in enumerate(headers):
                sub = second_cells[i] if i < len(second_cells) else ''
                if sub and sub != h:
                    merged.append(sub)
                else:
                    merged.append(h)
            headers = merged
        data_start = 2

    result = []
    for row_html in rows[data_start:]:
        cells = _cells(row_html)
        if len(cells) < 2:
            continue
        row_dict = {}
        for i, cell in enumerate(cells):
            if i < len(headers):
                row_dict[headers[i]] = cell
            else:
                row_dict[f'col_{i}'] = cell
        result.append(row_dict)

    return result


def _find_compound_col(headers: list[str]) -> str | None:
    """Find the column name that contains compound/example identifiers."""
    for h in headers:
        hl = h.lower()
        if any(k in hl for k in ['example', 'ex.', 'cpd', 'compound', 'no.']):
            return h
    return headers[0] if headers else None


def _clean_assay_name(raw_name: str) -> str:
    """Clean up assay name from table header."""
    name = raw_name

    # Remove unit parenthetical
    name = re.sub(r'\([nμµum]M\)', '', name, flags=re.IGNORECASE)
    # Remove "by example compounds" / "by exampl" suffixes
    name = re.sub(r'\s*by\s+exampl.*$', '', name, flags=re.IGNORECASE)
    # Remove "values for" prefix noise
    name = re.sub(r'^\\?\s*values?\s+for\s+', '', name, flags=re.IGNORECASE)
    # Remove leading backslashes/junk
    name = re.sub(r'^[\\/ ]+', '', name)
    # Remove trailing "Example no." or compound column text
    name = re.sub(r'\s*Example\s+no\.?\s*$', '', name, flags=re.IGNORECASE)
    # Fix common OCR typos in assay names
    name = re.sub(r'\bMeni\b', 'Menin', name)
    name = re.sub(r'InhibitionIC50', 'Inhibition IC50', name)
    # Shorten verbose assay names
    name = re.sub(r'(?:IC50|IC₅₀)\s*values?\s*for\s*(?:the\s*)?(?:inhibition\s*of\s*)?', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(IC_\{50\}\\?\)\s*', '', name)
    name = re.sub(r'enzymatic\s*activity', 'enzymatic activity', name)
    name = re.sub(r'\s*in\s+[\w\s;,]+ cells?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*by\s+example\s+compounds?\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^values?\s+for\s+', '', name, flags=re.IGNORECASE)
    # Normalize SGK-1 variants
    name = re.sub(r'^inhibition of\s+', '', name, flags=re.IGNORECASE)
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def _find_assay_cols(headers: list[str]) -> list[tuple[str, str, str]]:
    """Find assay columns. Returns [(header, assay_name, unit), ...]."""
    assay_cols = []
    for h in headers:
        if ASSAY_KEYWORDS.search(h):
            # Extract unit from header or square brackets
            unit_m = UNIT_PATTERN.search(h)
            if not unit_m:
                unit_m = re.search(r'\[(nM|μM|uM|µM|mM)\]', h, re.IGNORECASE)
            unit = unit_m.group(1) if unit_m else ''
            # Normalize unit
            if unit.lower() in ('μm', 'um', 'µm'):
                unit = 'μM'

            name = _clean_assay_name(h)
            if not name:
                name = h[:50]

            assay_cols.append((h, name, unit))
    return assay_cols


def extract_assays_from_page(text: str) -> dict[str, list[AssayResult]]:
    """Extract assay data from a single page's markdown text.

    Returns:
        Dict of example_number -> list of AssayResult.
    """
    results: dict[str, list[AssayResult]] = {}

    # Find all tables
    tables = re.findall(r'<table>(.*?)</table>', text, re.DOTALL | re.IGNORECASE)

    for table_html in tables:
        # Check if table contains assay data
        if not ASSAY_KEYWORDS.search(table_html):
            continue

        rows = _parse_html_table(f'<table>{table_html}</table>')
        if not rows:
            continue

        headers = list(rows[0].keys())
        cpd_col = _find_compound_col(headers)
        assay_cols = _find_assay_cols(headers)

        if not cpd_col or not assay_cols:
            continue

        logger.debug(f"Found assay table: cpd_col={cpd_col}, assays={[a[1] for a in assay_cols]}")

        for row in rows:
            cpd_id = row.get(cpd_col, '').strip()
            if not cpd_id or not re.search(r'\d', cpd_id):
                continue

            # Normalize compound ID
            cpd_key = cpd_id.lower().replace(' ', '')

            for col_header, assay_name, unit in assay_cols:
                raw_value = row.get(col_header, '').strip()
                if not raw_value:
                    continue

                numeric, qualifier = _parse_value(raw_value)

                assay = AssayResult(
                    assay_name=assay_name,
                    value_raw=raw_value,
                    value_numeric=numeric,
                    qualifier=qualifier,
                    unit=unit,
                )

                if cpd_key not in results:
                    results[cpd_key] = []
                results[cpd_key].append(assay)

    return results


def _extract_assays_from_google_patents(patent_id: str) -> dict[str, list[AssayResult]]:
    """Extract assay data from Google Patents clean text.

    Parses flat number sequences like "Ex PI3K CD69 Ex PI3K CD69..."
    that OCR markdown tables often mangle into single-column format.
    """
    import json

    cache_path = config.OUTPUT_DIR / "gpatents_cache" / f"{patent_id}.json"
    if not cache_path.exists():
        return {}

    text = json.loads(cache_path.read_text()).get('description', '')
    if not text:
        return {}

    results: dict[str, list[AssayResult]] = {}

    # Find assay table sections by looking for IC50 header patterns
    # Pattern: sequences of "ExNum Value1 Value2 ExNum Value1 Value2..."
    # Detect by finding known assay headers
    for header_pattern in [
        r'(?:TABLE\s*\d+\s*)?PI3K\s*delta\s*CD69\s*Ex',  # "TABLE 11 PI3K delta CD69 Ex."
        r'PI3K\s*delta\s*(?:IC\s*50|IC50).*?CD69',       # "PI3K delta IC50 ... CD69"
        r'Menin\s*Binding.*?IC\s*50',
        r'IC\s*50.*?inhibition.*?SGK',
    ]:
        header_matches = list(re.finditer(header_pattern, text, re.IGNORECASE))
        if not header_matches:
            continue

        for header_match in header_matches:

            # Determine number of value columns from header
            header_text = header_match.group(0).lower()
            if 'cd69' in header_text:
                n_cols = 2
                assay_names = ['PI3K delta IC50', 'CD69 IC50']
                units = ['nM', 'nM']
            else:
                n_cols = 1
                assay_names = [re.sub(r'\s+', ' ', header_match.group(0)).strip()]
                units = ['nM']

            # Extract the numeric table after the header
            # Skip past all header tokens — find last (nM) before data
            search_start = header_match.end()
            header_zone = text[search_start:search_start + 100]
            # Find all (nM) in first 100 chars
            nm_positions = [m.end() for m in re.finditer(r'\(nM\)', header_zone)]
            if nm_positions:
                table_start = search_start + nm_positions[-1]
            else:
                table_start = search_start
            table_text = text[table_start:table_start + 10000]

            tokens = re.split(r'\s+', table_text)

            i = 0
            while i < len(tokens) - n_cols:
                tok = tokens[i].strip()
                ex_match = re.match(r'^(\d{1,4})$', tok)
                if not ex_match:
                    i += 1
                    continue

                ex_num = ex_match.group(1)
                vals = []
                valid = True

                for j in range(n_cols):
                    if i + 1 + j >= len(tokens):
                        valid = False
                        break
                    v = tokens[i + 1 + j].strip()
                    if re.match(r'^[<>]?\d+\.?\d*$', v):
                        vals.append(v)
                    elif v in ('—', 'â', '-', '–') or 'â' in v or v.startswith('\u2014'):
                        vals.append(None)
                    else:
                        valid = False
                        break

                if not valid or len(vals) != n_cols:
                    i += 1
                    continue

                cpd_key = ex_num
                # Don't overwrite if we already have data for this compound
                if cpd_key in results:
                    i += 1 + n_cols
                    continue
                results[cpd_key] = []

                for k, val in enumerate(vals):
                    if val is None:
                        continue
                    numeric, qualifier = _parse_value(val)
                    results[cpd_key].append(AssayResult(
                        assay_name=assay_names[k],
                        value_raw=val,
                        value_numeric=numeric,
                        qualifier=qualifier,
                        unit=units[k],
                    ))

                i += 1 + n_cols

    if results:
        logger.info(
            f"Patent {patent_id}: Google Patents assay extraction found "
            f"{len(results)} compounds"
        )
    return results


def extract_assays_for_patent(
    patent_id: str,
    data_dir: Path | None = None,
) -> dict[str, list[AssayResult]]:
    """Extract all assay data for a patent from its markdown pages.

    Scans all_pages/ for assay tables, then supplements with Google Patents
    clean text for values the OCR tables missed.

    Returns dict of normalized compound key -> list of AssayResult.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    all_assays: dict[str, list[AssayResult]] = {}

    for subdir in ['all_pages', 'iupacs_clean']:
        pages_dir = data_dir / patent_id / subdir
        if not pages_dir.exists():
            continue

        for page_file in sorted(pages_dir.glob('page_*.md')):
            text = page_file.read_text(encoding='utf-8')

            # Quick check: skip pages without assay keywords
            if not ASSAY_KEYWORDS.search(text):
                continue

            page_assays = extract_assays_from_page(text)
            for cpd_key, assays in page_assays.items():
                if cpd_key not in all_assays:
                    all_assays[cpd_key] = []
                all_assays[cpd_key].extend(assays)

    # Supplement with Google Patents clean text (catches values OCR tables missed)
    gp_assays = _extract_assays_from_google_patents(patent_id)
    gp_new = 0
    for cpd_key, assays in gp_assays.items():
        if cpd_key not in all_assays:
            all_assays[cpd_key] = assays
            gp_new += 1
        else:
            # Add assay types we don't have yet
            existing_names = {a.assay_name for a in all_assays[cpd_key]}
            for a in assays:
                if a.assay_name not in existing_names:
                    all_assays[cpd_key].append(a)

    logger.info(
        f"Patent {patent_id}: extracted assay data for "
        f"{len(all_assays)} compounds ({sum(len(v) for v in all_assays.values())} total measurements)"
        f"{f' (+{gp_new} from Google Patents)' if gp_new else ''}"
    )
    return all_assays


def attach_assays_to_compounds(compounds: list, assay_data: dict[str, list[AssayResult]]) -> int:
    """Attach assay results to compound objects by matching example numbers.

    Returns number of compounds that got assay data attached.
    """
    attached = 0
    for compound in compounds:
        ex = compound.example_number or ''
        # Try multiple key formats
        keys_to_try = [
            ex.lower().replace(' ', ''),
            re.sub(r'[^0-9]', '', ex),  # Just the number
            f"example{re.sub(r'[^0-9]', '', ex)}",
        ]

        for key in keys_to_try:
            if key in assay_data:
                compound.assay_data = assay_data[key]
                attached += 1
                break

    return attached
