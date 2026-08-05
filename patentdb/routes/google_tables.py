"""Step 2.5 — Parse compound tables to extract Example→IUPAC name→M+H mappings.

Many patents have compound tables (TABLE 49 in US10214537) with format:
  Ex. No. | Structure | Name | M+H

The "Name" column contains the full IUPAC name. We parse these tables
to extract compounds that aren't in the Example sections (iupacs_clean).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from ..core import config
from ..core.models import Compound, CompoundSource, IupacSource, PageManifest
from ..core.name_boundary import looks_like_spectra, terminate_name

logger = logging.getLogger(__name__)


# Header words that mark a column as CHARACTERISATION DATA. Such a column is
# never the name column, and — more importantly — must never be reached by the
# last-column fallback below. US11292791's example tables are
# `Example Number | Structure and Compound Name | 1H NMR, LCMS`, and the
# fallback landed on the NMR column for every one of them.
# Deliberately narrow: bare `ms`, `rt` and `mp` are omitted because they are
# two-letter substrings that collide with real words in merged OCR headers.
_SPECTRAL_HEADER_RE = re.compile(
    r"\bnmr\b|\blcms\b|\blc[-\s/]?ms\b|\bgcms\b|\bhrms\b|\blrms\b|\bmass\b"
    r"|\bm\s*/\s*z\b|\bhplc\b|\buplc\b|\bretention\b|\bmelting\b|\bpurity\b",
    re.IGNORECASE,
)


# MinerU's math escape. A systematic name has no TeX in it.
_LATEX_RE = re.compile(r"\\(?:mathrm|boldsymbol|mathbf|begin|frac|left|right)\b|\$")

# A bracket, or a locant handing off to a letter/bracket (`2-yl`, `4,5-b]`).
# The locant is anchored TIGHT — no space around the hyphen — because
# US9694016's example tables read `57 -structure and name are in the Example-`
# and a space-tolerant pattern reads that `7 -s` as a locant.
_CHEM_SIGNAL_RE = re.compile(r"[\[\]()]|\d-[A-Za-z\[(]")


def _name_header_rank(header: str) -> int:
    """How strongly does this header declare a NAME column? 0 = not one.

    Replaces the rule `'name' in h and 'structure' not in h`, which
    DISQUALIFIED the single most common name header in the corpus:
    MinerU reads US11292791's tables as
    `['example number', 'structure and compound name', '1h nmr, lcms']`
    — the literal words "Structure and Compound Name" — and the word
    "structure" vetoed it. `name_col` then fell through to the last-column
    fallback, i.e. the NMR column: 168 compounds extracted, 153 of them with
    an NMR readout stored as `iupac_name`, and only 3 names OPSIN could parse.

    "structure" in a header means the cell ALSO holds a drawn structure (which
    OCR renders as noise around the name), not that the cell holds no name.
    The only reason to reject a `name` header is that the same header names
    characterisation data — `Structure and Compound Name 1H NMR,LCMS` is a
    merged header whose column cannot be attributed to either.
    """
    h = (header or "").strip()
    if "name" not in h.lower():
        return 0
    if _SPECTRAL_HEADER_RE.search(h):
        return 0
    lower = h.lower()
    if "compound name" in lower or "chemical name" in lower or "iupac" in lower:
        return 3
    if lower == "name":
        return 2
    return 1


def has_compound_table_candidates(patent_id: str, data_dir: Path | None = None) -> bool:
    """Quick scan: does this patent have any HTML table whose header looks
    like a compound-name table (i.e., has both an Ex/Cpd column AND a
    'name' column)?

    Used by pipeline.py to gate Step 2.5 — if no candidates exist, we
    skip the parser entirely and record it as a clean skip rather than
    let the audit trip a false-positive WIRING violation on the empty
    output that legitimate patents (LC-MS instrument tables, assay
    tables) would produce.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    all_pages_dir = data_dir / patent_id / "all_pages"
    if not all_pages_dir.exists():
        return False

    # Cheap header-only scan: look for <td>...name...</td> alongside
    # <td>...ex...no...</td> on the same page. Doesn't fully parse.
    name_pattern = re.compile(r'<td[^>]*>[^<]*\b(?:chemical\s+)?name\b[^<]*</td>', re.I)
    ex_pattern = re.compile(r'<td[^>]*>[^<]*\b(?:ex|cpd|compound)[\s.\-#]*(?:no|number)?\b', re.I)
    for page_file in all_pages_dir.glob("page_*.md"):
        text = page_file.read_text(encoding="utf-8")
        if '<table>' not in text.lower():
            continue
        if name_pattern.search(text) and ex_pattern.search(text):
            return True
    return False


def _clean_name(text: str, strip_example_number: bool = False) -> str | None:
    """A table cell as an `iupac_name`, or None if it isn't one.

    The rejector here is the one the example-header route already runs
    (`terminate_name`, commit b024fc8); this path never had it, and MinerU
    spills NMR text into the name column of the very tables it fails to align
    — so 147 of US11292791's stored names were `1H-NMR (CDCl3, 300 MHz) δ
    (ppm): ...`. Column choice alone does not fix that.
    """
    s = re.sub(r'\s+', ' ', (text or '').strip())
    if strip_example_number:
        # A merged header ("Example Number Structure and Compound Name") puts
        # the example number in front of its own name. WHITESPACE after the
        # number is required: `79 (R)-4-(1-benzyl...` is a number then a name,
        # but `3-chloro-N-(thiazol-2-yl)benzenesulfonamide` opens on a locant
        # and must keep its 3.
        s = re.sub(r'^\s*\d+[.)]?\s+', '', s)
    # Cuts the trailing prose/analytical data off a real name, and returns ""
    # for a cell that was characterisation data from the start.
    s = terminate_name(s)
    if len(s) < 15 or looks_like_spectra(s):
        return None
    # MinerU renders a superscripted mass as LaTeX —
    # `576.2 $( \boldsymbol{\mathrm{M}} + \mathrm{H} )^{+}$`. It reads as a
    # name to every check above (`\boldsymbol` supplies the lowercase run
    # `looks_like_spectra` has no opinion on), and it is never one.
    if _LATEX_RE.search(s):
        return None
    # A cell can be plain English and pass everything above —
    # `Homochiral from peak 2`, `-structure and name are in the Example-`,
    # `2nd eluting isomer`. None of them is a name, and all of them arrive in
    # the name column of a table MinerU mis-segmented. A systematic name of
    # this length carries a locant or a bracket: measured over the 11 MinerU
    # patents, 74 extracted names lack both and exactly ONE of those parses in
    # OPSIN — `trifluoroacetate`, a counter-ion, not a compound.
    if not _CHEM_SIGNAL_RE.search(s):
        return None
    return s


def parse_compound_table(html: str) -> list[dict]:
    """Parse a compound table with Ex. No. / Name / M+H columns.

    Args:
        html: Raw HTML table content.

    Returns:
        List of {example_number, iupac_name, mh_plus} dicts.
    """
    compounds = []
    soup = BeautifulSoup(html, 'html.parser')

    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if not rows:
            continue

        # Detect column layout from header row
        header_cells = rows[0].find_all('td')
        header_text = [c.get_text(strip=True).lower() for c in header_cells]

        # Find relevant column indices
        ex_col = None
        name_col = None
        mh_col = None

        for i, h in enumerate(header_text):
            if any(k in h for k in ['ex', 'cpd', 'compound']) and any(k in h for k in ['no', '#', '.']):
                ex_col = i
            elif h.startswith('ex') or h.startswith('cpd'):
                ex_col = i
            elif 'm + h' in h or 'm+h' in h or 'mh+' in h.replace(' ', ''):
                mh_col = i

        # Fallback: if no header detected, try common layouts
        if ex_col is None and len(header_cells) >= 3:
            ex_col = 0

        # An EXPLICIT name header wins outright — it is tried before, not
        # inside, the elif chain above, because a header can legitimately be
        # both ("Example Number Structure and Compound Name" is one merged
        # cell in 27 of US11292791's tables) and the ex-branch used to consume
        # it, leaving name_col to the positional fallback.
        ranks = [_name_header_rank(h) for h in header_text]
        if any(ranks):
            best = max(ranks)
            cands = [i for i, r in enumerate(ranks) if r == best]
            # Prefer a name column of its own; fall back to the merged cell
            # only when the name header shares a column with ex/M+H.
            own = [i for i in cands if i != ex_col and i != mh_col]
            name_col = own[0] if own else cands[0]

        # Last resort ONLY when no header names a name column: the last column
        # that is neither the ex/M+H column nor characterisation data. The old
        # rule ("Name is always the LAST text column") took the last column
        # unconditionally, which is how an NMR column became `iupac_name`.
        if name_col is None and ex_col is not None:
            for i in range(len(header_cells) - 1, -1, -1):
                if i == ex_col or i == mh_col:
                    continue
                if _SPECTRAL_HEADER_RE.search(header_text[i]):
                    continue
                # A column headed "Structure" with no "name" in it holds a
                # DRAWN structure; whatever OCR put in the cell is noise.
                # US10273259's tables are `Ex.# | Structure | LCMS m/z |
                # HPLC tr | HPLC method` and carry no name at all — reading
                # its structure column yielded 67 rows of `Homochiral from
                # peak 2` and LaTeX-wrapped `(M + H)+`.
                if 'structure' in header_text[i] and 'name' not in header_text[i]:
                    continue
                name_col = i
                break
            # No column of its own left. MinerU collapses some tables into ONE
            # <td> per row (US10246453 has 29 such rows, header
            # `example no.name    ms (es+) m/z`), where the name shares the ex
            # column; refusing to look there dropped every one of those rows.
            if name_col is None:
                name_col = ex_col

        if ex_col is None:
            continue

        # Parse data rows
        for row in rows[1:]:
            cells = row.find_all('td')
            if len(cells) <= max(filter(None, [ex_col, name_col, mh_col]), default=0):
                continue

            ex_text = cells[ex_col].get_text(strip=True) if ex_col is not None and ex_col < len(cells) else ""
            name_text = cells[name_col].get_text(' ', strip=True) if name_col is not None and name_col < len(cells) else ""
            mh_text = cells[mh_col].get_text(strip=True) if mh_col is not None and mh_col < len(cells) else ""

            # Validate example number
            ex_match = re.match(r'(\d+)', ex_text)
            if not ex_match:
                continue

            example_number = f"Example {ex_match.group(1)}"

            iupac = _clean_name(name_text, strip_example_number=(name_col == ex_col))

            # The name header shared a cell with the ex header, and that cell
            # turned out to hold nothing but the number: MinerU segmented the
            # BODY row more finely than the header row it wrote above it —
            # US10246453's `Example No.Name | MS (ES+) m/z 1H NMR` is two
            # header cells over four body cells, and reading only the merged
            # column dropped Examples 325-327 with their names intact one cell
            # to the right. Scan rightwards for the first cell that survives
            # the rejector; the M+H column is skipped because it is known.
            if iupac is None and name_col == ex_col:
                for j in range(ex_col + 1, len(cells)):
                    if j == mh_col:
                        continue
                    cand = _clean_name(cells[j].get_text(' ', strip=True))
                    if cand:
                        iupac = cand
                        break

            # Parse M+H value
            mh_plus = None
            if mh_text:
                mh_match = re.search(r'(\d+\.?\d*)', mh_text)
                if mh_match:
                    try:
                        mh_plus = float(mh_match.group(1))
                        if not (100 < mh_plus < 2000):
                            mh_plus = None
                    except ValueError:
                        pass

            if iupac or mh_plus:
                compounds.append({
                    'example_number': example_number,
                    'iupac_name': iupac,
                    'mh_plus': mh_plus,
                })

    return compounds


def extract_table_compounds(
    patent_id: str,
    data_dir: Path | None = None,
) -> list[Compound]:
    """Extract compounds from all table pages in a patent.

    Processes both table/ directory and all_pages/ for table content.

    Args:
        patent_id: Patent identifier.
        data_dir: Root data directory.

    Returns:
        List of Compound objects extracted from tables.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    all_compounds: list[Compound] = []
    seen_ids: set[str] = set()

    # Process all_pages/ looking for compound tables
    # (TABLE 49 etc. with Ex. No. | Structure | Name | M+H)
    all_pages_dir = data_dir / patent_id / "all_pages"
    if all_pages_dir.exists():
        for page_file in sorted(all_pages_dir.glob("page_*.md")):
            text = page_file.read_text(encoding="utf-8")

            # Only process pages that have table content with compound-like data
            if '<table>' not in text.lower():
                continue
            if 'name' not in text.lower() or ('ex' not in text.lower()):
                continue

            # Extract page number
            page_match = re.search(r'page_(\d+)', page_file.stem)
            page_num = int(page_match.group(1)) if page_match else None

            # Parse tables on this page
            table_compounds = parse_compound_table(text)

            for tc in table_compounds:
                ex_id = tc['example_number']
                dedup_key = ex_id.lower().replace(' ', '')

                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                compound = Compound(
                    patent_id=patent_id,
                    example_number=ex_id,
                    iupac_name=tc.get('iupac_name'),
                    iupac_source=IupacSource.PATENT_VERBATIM,
                    ms_mh_plus=tc.get('mh_plus'),
                    source=CompoundSource.EXEMPLIFIED,
                    source_page=page_num,
                    processing_status="text_done",
                )
                all_compounds.append(compound)

    # Also parse claims pages for dense compound lists: "IUPAC_NAME (NUMBER); ..."
    if all_pages_dir.exists():
        for page_file in sorted(all_pages_dir.glob("page_*.md")):
            text = page_file.read_text(encoding="utf-8")

            # Only process pages with semicolon-separated compound lists
            if text.count(';') < 5 or '(' not in text:
                continue

            page_match = re.search(r'page_(\d+)', page_file.stem)
            page_num = int(page_match.group(1)) if page_match else None

            # Strip markdown tags and normalize whitespace
            clean = re.sub(r'<\|ref\|>\w+<\|/ref\|><\|det\|>\[\[\d+.*?\]\]<\|/det\|>', '', text)
            # Fix hyphenated line breaks: "thiomorpho-\nline" → "thiomorpholine"
            clean = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', clean)
            # Fix line breaks inside compound names: "pyrrolo\n[2,1-f]" → "pyrrolo[2,1-f]"
            clean = re.sub(r'(\w)\s*\n\s*(\[)', r'\1\2', clean)
            clean = re.sub(r'\s+', ' ', clean)

            # Split by semicolons — each segment ends with "IUPAC_NAME (NUMBER)"
            segments = clean.split(';')
            for seg in segments:
                seg = seg.strip()
                m = re.search(r'\((\d{1,4})\)\s*\.?\s*$', seg)
                if not m:
                    continue
                num = int(m.group(1))
                name = seg[:m.start()].strip()

                if len(name) < 20:
                    continue

                # Filter: must look like a real IUPAC compound name
                # (contains ring/heterocycle indicators, not Markush fragments)
                name_lower = name.lower()
                has_ring = any(kw in name_lower for kw in [
                    'pyrrolo', 'pyrrol', 'triazin', 'pyrazol', 'piperidin',
                    'morpholin', 'piperazin', 'benzamide', 'benzonitrile',
                    'phenyl', 'oxazol', 'thiomorpholin', 'pyridin',
                    'oxetan', 'azetidin', 'pyrrolidin', 'imidazol',
                ])
                if not has_ring:
                    continue

                ex_id = f"Example {num}"
                dedup_key = ex_id.lower().replace(' ', '')
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                compound = Compound(
                    patent_id=patent_id,
                    example_number=ex_id,
                    iupac_name=name,
                    iupac_source=IupacSource.PATENT_VERBATIM,
                    source=CompoundSource.EXEMPLIFIED,
                    source_page=page_num,
                    processing_status="text_done",
                )
                all_compounds.append(compound)

    logger.info(
        f"Patent {patent_id}: Extracted {len(all_compounds)} compounds from tables + claims"
    )
    return all_compounds
