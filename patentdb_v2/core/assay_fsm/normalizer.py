"""Stage 0 — Universal text normalization.

Applied to EVERY page before any downstream processing. The order
matters:
  1. Mojibake repair (UTF-8 → Latin-1 → UTF-8 double-encoding artifacts)
  2. HTML entity unescape (`&gt;` → `>`)
  3. Unicode NFKC normalization (collapses compatibility decompositions:
     fullwidth digits, ligatures, subscripts where it makes sense)

Steps 1 and 2 must run BEFORE step 3 because NFKC may alter byte
sequences that the mojibake table is keyed on.

The repair is now GENERIC — `_MOJIBAKE_RUN_RE` finds every UTF-8
lead+continuation run and inverts it — because enumerating the damage was
losing to it. Two ways the hand-written table below was wrong:

  * **19 of its 35 keys could never fire.** They are spelled in cp1252
    (`â€™`, `â‰¤`, `â‚‚`, `Ï€` — where byte 0x80 renders as `€`), while the
    corruption this codebase actually produces is Latin-1 (`â\\x80\\x99`,
    byte 0x80 renders as the C1 control). `"â€™".encode("latin-1")` raises;
    the key is unreachable by construction.
  * **What it did cover was a sample, not a set.** `═` (`\\xe2\\x95\\x90`,
    a drawn double bond), `≡`, `“”` and the em/en spaces are absent, and
    reached `route_audit.json` snippets on 13 of the 22 corpus patents.

The generic inverse costs one regex pass and needs no maintenance: a run is
rewritten only when it decodes as valid UTF-8, so text that is already
correct — the whole `uspto_xml` tier — is returned untouched.

**Per-RUN, not per-string.** `patent_text.load_full_patent_text` concatenates
the clean XML with the corrupt GP scrape and normalises the join. A whole-string
`encode("latin-1")` raises on the XML's real `—` and a guarded whole-string
repair therefore returns the concatenation unchanged, silently repairing
nothing. Only the substring form survives mixed input.
"""
from __future__ import annotations

import html
import re
import unicodedata


# A UTF-8 sequence read as Latin-1: a lead byte (0xC2-0xF4) followed by one or
# more continuation bytes (0x80-0xBF), each decoded to the Latin-1 codepoint of
# the same value. Below 0xC2 there is no valid two-byte UTF-8 lead, so the
# range starts there and a stray `Á`/`À` in correct text is never touched.
_MOJIBAKE_RUN_RE = re.compile("[\u00c2-\u00f4][\u0080-\u00bf]+")


def _invert_latin1_run(m: re.Match[str]) -> str:
    """Re-encode one run as Latin-1 and read it back as the UTF-8 it was.

    Returns the run unchanged when it is not valid UTF-8 — which is what
    happens where the scrape's own `re.sub(r"\\s+", " ")` ate a continuation
    byte that decoded to U+0085 or U+00A0 (both `\\s` under Python's Unicode
    `re`). That damage is not recoverable here; see the note in
    `routes/google_patents.fetch_patent_text`.
    """
    run = m.group(0)
    try:
        return run.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return run


# Residual single-character artifacts the run inverter cannot reach, plus the
# historical table. Kept because a LEAD BYTE ON ITS OWN — `â` with its
# continuation bytes lost, as when a model transcribes only the visible glyph —
# is not a run and has no inverse. Never add a patent-specific repair here.
_MOJIBAKE_REPAIRS: dict[str, str] = {
    # Greek letters used in unit symbols and chemistry
    "Î¼": "μ",   "Âµ": "µ",
    "Î²": "β",   "Î±": "α",   "Î³": "γ",   "Î´": "δ",
    "Î»": "λ",   "Î¸": "θ",   "Ï€": "π",
    # Math operators used as qualifiers
    "â‰¤": "≤",  "â‰¥": "≥",
    "â‰ˆ": "≈",  "â‰ª": "≪",  "â‰«": "≫",
    # Subscripts (often in IC₅₀, etc.)
    "â‚‚": "₂",  "â‚ƒ": "₃",  "â‚„": "₄",
    "â‚…": "₅",  "â‚†": "₆",  "â‚‡": "₇",
    "â‚ˆ": "₈",  "â‚‰": "₉",  "â‚€": "₀",
    # Superscripts and signs
    "Â²": "²",   "Â³": "³",   "Â°": "°",   "Â±": "±",
    # Punctuation / quotes (UTF-8-as-Latin-1)
    "â€™": "'",  "â€\x9c": "\"", "â€\x9d": "\"",
    "â€\x98": "'",
    # Em-dash / en-dash / minus sign — used as null markers in many
    # patent assay tables. Without this repair the byte-sequence
    # `â\x80\x94` reaches the tokenizer as garbage.
    "â\x80\x94": "—",   # em-dash
    "â\x80\x93": "–",   # en-dash
    "â\x80\x92": "‒",   # figure dash
    "â\x88\x92": "−",   # minus sign
}


def repair_mojibake(text: str) -> str:
    """Undo UTF-8 that was decoded as Latin-1.

    Generic run inversion first, then the residual table for anything the
    inverter cannot reach. Idempotent: a repaired string contains no
    lead+continuation run, so the second pass is a no-op. Free, O(n).
    Patent-agnostic.
    """
    if not text:
        return text
    text = _MOJIBAKE_RUN_RE.sub(_invert_latin1_run, text)
    for bad, good in _MOJIBAKE_REPAIRS.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


_HTML_CELL_BOUNDARY_RE = re.compile(
    r"</td>\s*<td[^>]*>",
    re.IGNORECASE,
)
_HTML_ROW_BOUNDARY_RE = re.compile(
    r"</tr>\s*<tr[^>]*>",
    re.IGNORECASE,
)
_HTML_TABLE_TAGS_RE = re.compile(
    r"</?(?:table|thead|tbody|tr|td|th)(?:\s[^>]*)?>",
    re.IGNORECASE,
)


def _flatten_html_tables(text: str) -> str:
    """MinerU emits tables as literal HTML — `<tr><td>I-0020</td><td>0.384</td>…</tr>`.
    The downstream region detector & tokenizer expect cell content
    separated by whitespace, NOT by `</td><td>` tags, so without this
    flatten step rows with multi-column compound tables (3 `cid|value`
    pairs per row) yield zero hits. Strategy:
        `</td><td>`  → `  `  (two-space cell separator)
        `</tr><tr>`  → `\n`  (newline between rows)
        leftover `<table>`/`<thead>`/etc. tags → stripped silently
    """
    text = _HTML_ROW_BOUNDARY_RE.sub("\n", text)
    text = _HTML_CELL_BOUNDARY_RE.sub("  ", text)
    text = _HTML_TABLE_TAGS_RE.sub(" ", text)
    return text


# Matches a MinerU OCR row-merge: a <tr> whose FIRST <td> cell contains
# 2+ whitespace-separated integers (two adjacent rows fused into one).
# Captures the merged-cids cell and every following <td>...</td> cell
# in the row so we can redistribute values positionally.
_MINERU_MERGED_CID_ROW_RE = re.compile(
    r"<tr[^>]*>"
    r"<td[^>]*>\s*(?P<cids>\d{1,4}(?:[A-Za-z]{0,3})?(?:\s+\d{1,4}(?:[A-Za-z]{0,3})?)+)\s*</td>"
    r"(?P<cells>(?:<td[^>]*>[^<]*</td>)+)"
    # MinerU sometimes drops </tr> at end of table — accept </tr>,
    # </table>, or the next <tr> as the row terminator.
    r"(?:</tr>|(?=</table>|<tr[^>]*>))",
    re.IGNORECASE | re.DOTALL,
)
_CELL_RE = re.compile(r"<td[^>]*>([^<]*)</td>", re.IGNORECASE)


def _repair_mineru_row_merges(text: str) -> str:
    """Repair MinerU OCR row-merge errors in HTML tables.

    Failure mode: when MinerU misses the row boundary between two adjacent
    rows, it fuses them into one <tr>. The cid cell becomes ``<td>N1 N2</td>``
    (two integer ids) and value cells may be either ``<td>VALa VALb</td>``
    (when both rows' values OCR'd) or ``<td>VAL</td>`` (when one row's
    value was dropped). Downstream regex extractors require ``<td>(?P<cid>\\d+)</td>``
    and see "N1 N2" — no match.

    Repair: for each affected ``<tr>``, split into N rows (one per cid).
    For each value cell:
      - if it contains exactly N whitespace-separated tokens, distribute
        positionally (cell.split()[i] → row i)
      - otherwise copy the cell to all N rows; the pattern_library
        re-application across the GP-description flat-text version of
        the same table will overwrite with the correct per-cid values
        if available (US10246453's flat-text table at offset ~1.84M
        carries the canonical values; this repair just gets the cids
        into example_index so the BDB cross-ref can resolve them)

    Verified on US10246453 (`<td>156 157</td>`, `<td>72 73</td>`,
    `<td>291 292</td>` — 3 row-merges, 6 cids previously dropped
    silently from assay_tables).

    Patent-agnostic: scans for the structural pattern (no hard-coded
    cids or assay names).

    Idempotent: a row that was already split won't match the regex
    (first <td> contains exactly one integer).
    """
    def split_row(m: re.Match[str]) -> str:
        cids = m.group("cids").split()
        cells = _CELL_RE.findall(m.group("cells"))
        n_cids = len(cids)
        out_rows: list[str] = []
        for i, cid in enumerate(cids):
            row_cells = []
            for cell in cells:
                vals = cell.split()
                if len(vals) == n_cids:
                    row_cells.append(f"<td>{vals[i]}</td>")
                elif len(vals) > 1:
                    # Cell has more tokens than cids — can't split cleanly;
                    # give the i-th to row i if available, else empty.
                    row_cells.append(f"<td>{vals[i] if i < len(vals) else ''}</td>")
                else:
                    # Single value or empty — copy verbatim to every row.
                    # Downstream pattern_library re-application (against
                    # the parallel flat-text in GP description) is the
                    # authoritative cross-check for these per-cid values.
                    row_cells.append(f"<td>{cell}</td>")
            out_rows.append(f"<tr><td>{cid}</td>" + "".join(row_cells) + "</tr>")
        return "".join(out_rows)

    return _MINERU_MERGED_CID_ROW_RE.sub(split_row, text)


def normalize_page(text: str) -> str:
    """Apply Stage 0 normalization in the canonical order.

    Args:
        text: Raw page text (may contain mojibake, HTML entities,
              compatibility-form Unicode).

    Returns:
        Normalized text — same length offsets are NOT preserved
        (HTML unescape collapses `&gt;` 4 chars → 1 char, etc.). Use
        the returned text for all downstream processing.
    """
    if not text:
        return text
    text = repair_mojibake(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    # Repair MinerU OCR row-merges BEFORE flattening tags — the repair
    # operates on <tr>/<td> structure, which _flatten_html_tables
    # erases. After this step, every <tr> has a single-integer cid
    # in its first cell (or wasn't an affected row to begin with).
    text = _repair_mineru_row_merges(text)
    text = _flatten_html_tables(text)
    return text
