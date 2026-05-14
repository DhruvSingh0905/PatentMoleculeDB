"""Google Patents clean text extraction — Route 1 in the pipeline.

Format-agnostic extraction using IUPAC fragment density scoring.
Instead of per-format regex patterns, finds compound names by scoring
text spans near Example/Compound markers for IUPAC-like content.

Cost: $0 (no API calls).
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ..core import config
from ..core.models import Compound, CompoundSource, IupacSource
from ..core.iupac_to_smiles import _try_opsin, _finalize, _convert_single
from ..core.cost_tracker import cost_tracker
from ..core.smiles_utils import validate_smiles

logger = logging.getLogger(__name__)

CACHE_DIR = config.OUTPUT_DIR / "gpatents_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# IUPAC fragments — chemistry-aware signals for name detection.
# A span with high density of these is likely an IUPAC name.
IUPAC_FRAGMENTS = [
    # Ring systems
    'pyrrolo', 'pyrrol', 'piperidin', 'piperazin', 'morpholin', 'azetidin',
    'pyrrolidin', 'indol', 'quinolin', 'isoquinolin', 'benzimidazol',
    'imidazol', 'pyrazol', 'triazol', 'tetrazol', 'oxazol', 'thiazol',
    'triazin', 'pyrimidin', 'pyridin', 'pyridazin', 'oxetan', 'thiomorpholin',
    'isobenzofuran', 'benzofuran', 'isoxazol', 'thiophen',
    # Aryl
    'phenyl', 'benzyl', 'naphth', 'benzo',
    # Functional groups
    'sulfonamide', 'benzamide', 'benzonitrile', 'carboxamide', 'carbonitrile',
    'amino', 'hydroxy', 'methoxy', 'ethoxy', 'cyano', 'nitro', 'acetyl',
    'fluoro', 'chloro', 'bromo', 'iodo',
    # Alkyl / structural
    'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl', 'hexyl',
    'cyclopent', 'cyclohex', 'cycloprop', 'cyclobut',
    # Stereo / bridging
    'dihydro', 'tetrahydro', 'hexahydro', 'octahydro',
    'spiro', 'bicyclo', 'methan',
    # Name suffixes
    'dione', 'one)', 'amine', 'amide', 'nitrile', 'yl)',
    'oyl', 'oxy', 'thio',
]

# Stop words — if a candidate span contains these, it's probably not an IUPAC name
STOP_WORDS = [
    'wherein', 'comprising', 'embodiment', 'method', 'administering',
    'pharmaceutically', 'acceptable', 'salt', 'thereof', 'disclosed',
    'following', 'according', 'invention', 'present', 'described',
    'formula', 'selected from', 'group consisting',
]


# ============================================================
# GP-HTML embedded structured data — itemprop spans
# ============================================================
# Google Patents pages embed canonical SMILES, InChIKey, and SVG
# drawings for every compound figure as <span itemprop="..."> elements
# in the page HTML. Empirically (verified on US9718825/US10899738):
# 158 / 639 compounds get full smiles+inchi_key coverage as inner-text of
# <span> tags. These are GP-canonical (RDKit-ready) so we can bypass
# OPSIN entirely.
#
# Treated as just another regex strategy in this module's library
# alongside the IUPAC-density patterns. Output flows through the
# normal Compound construction path with `extraction_method="gp_embedded_meta"`.
#
# IMPORTANT: SMILES and InChIKey must be captured as PAIRS, not as two
# parallel lists. Some compounds (Markush scaffolds) emit a SMILES with `*`
# wildcards but NO InChIKey, which causes positional misalignment when the
# two lists are read separately. Capturing the adjacent pair in one regex
# guarantees correct alignment.
_GP_EMBED_COMPOUND_PAT = re.compile(
    r'<span\s+itemprop="smiles">([^<]*)</span>'
    r'\s*<span\s+itemprop="inchi_key">([^<]*)</span>',
    re.DOTALL,
)
_GP_FIGURE_IMG_PAT = re.compile(r'<img[^>]+src="(https://patentimages\.[^"]+)"')

# Markush linguistic-signal phrases for audit_patent_format
_MARKUSH_PHRASE_PATS = [
    re.compile(r"compound\s+of\s+formula\s+[IVXivx]+", re.IGNORECASE),
    re.compile(r"wherein\s+each\s+occurrence", re.IGNORECASE),
    re.compile(r"independently\s+selected\s+from", re.IGNORECASE),
]


# ============================================================
# Format Audit — scan patent text to understand its structure
# ============================================================

@dataclass
class FormatAudit:
    """Result of scanning a patent's text structure. Guides extraction strategy."""
    total_example_markers: int = 0
    total_cpd_markers: int = 0
    has_compound_table: bool = False
    has_substituent_table: bool = False
    has_claims_list: bool = False
    estimated_compounds: int = 0
    format_hints: list[str] = field(default_factory=list)
    # GP-HTML structured-data signals — derived from gpatents_cache fields
    # populated by fetch_patent_text. Drive the route classifier's
    # Markush-vs-text decision without needing to reverse-engineer the
    # patent's text structure.
    n_embedded_smiles: int = 0           # count with non-empty SMILES
    n_embedded_inchi_key: int = 0        # count with non-empty InChIKey (paired)
    n_phrase_hits_markush: int = 0       # linguistic Markush phrase hits


def audit_patent_format(patent_id: str) -> FormatAudit:
    """Scan patent text and produce a structural audit. Free, ~1 second.

    Uses both regex signals (legacy Ar/R1/R2 detection) AND structured
    data captured from GP HTML — `embedded_compounds` count + Markush
    linguistic phrase hits. The structured signals are far more reliable
    than the regex on patents that come from MinerU markdown (where
    HTML table structure is lost during tag-stripping).
    """
    text_data = fetch_patent_text(patent_id)
    if not text_data:
        return FormatAudit()

    text = text_data.get('description', '') + '\n' + text_data.get('claims', '')
    audit = FormatAudit()

    # GP-HTML structured-data signals (free — already in cache)
    embedded = text_data.get("embedded_compounds", []) or []
    audit.n_embedded_smiles = sum(
        1 for e in embedded if isinstance(e, dict) and e.get("smiles")
    )
    audit.n_embedded_inchi_key = sum(
        1 for e in embedded
        if isinstance(e, dict) and e.get("smiles") and e.get("inchi_key")
    )
    audit.n_phrase_hits_markush = sum(
        len(pat.findall(text)) for pat in _MARKUSH_PHRASE_PATS
    )

    # Count markers
    example_nums = set(int(m.group(1)) for m in re.finditer(r'Example\s+(\d+)', text))
    cpd_nums = set(int(m.group(1)) for m in re.finditer(r'Cpd\.?\s*(?:No\.?\s*)?(\d+)', text))
    compound_nums = set(int(m.group(1)) for m in re.finditer(r'Compound\s+(\d+)', text))

    audit.total_example_markers = len(example_nums)
    audit.total_cpd_markers = len(cpd_nums | compound_nums)

    # Detect table types — capture wider header window than `[^.]` (the old
    # regex stopped at "Cpd." or "Synthe- sis", missing the column names).
    # Use a fixed character span instead. This is the audit signal that drives
    # patent_class — getting it wrong means routing decisions are wrong.
    table_headers = re.findall(r'TABLE\s+\d+.{0,400}', text, re.IGNORECASE | re.DOTALL)
    for header in table_headers:
        # Compound table: header mentions "Chemical Name" (US10899738 format)
        if re.search(r'Chemical\s+Name', header, re.IGNORECASE):
            audit.has_compound_table = True
            audit.format_hints.append('compound_table')
        # Substituent table: header has Ar+R1, R1+R2, R+Ar columns (US9718825 format)
        # Allow either order; tolerate hyphen-broken words like "Synthe- sis".
        if (
            re.search(r'\bAr\b.*?\bR\s*[12]\b', header, re.IGNORECASE | re.DOTALL)
            or re.search(r'\bR\s*[12]\b.*?\bAr\b', header, re.IGNORECASE | re.DOTALL)
            or re.search(r'\bR\s*1\b.*?\bR\s*2\b', header, re.IGNORECASE | re.DOTALL)
        ):
            audit.has_substituent_table = True
            audit.format_hints.append('substituent_table')

    # Detect claims list (semicolon-delimited with (NUMBER) pattern)
    claims = text_data.get('claims', '')
    if claims and re.search(r'\(\d{1,4}\)\s*;', claims):
        audit.has_claims_list = True
        audit.format_hints.append('claims_list')

    # Check for Example-based format
    if audit.total_example_markers >= 3:
        audit.format_hints.append('example_based')

    # Check for synthesis-prose format: (i)/(ii)/(iii) step markers indicate
    # compound names smeared across synthesis steps (e.g. US9718825)
    if re.search(r'\(i\)\s.{0,2000}?\(ii\)', text, re.DOTALL):
        audit.format_hints.append('synthesis_steps')

    # Estimate expected compounds
    audit.estimated_compounds = max(
        audit.total_example_markers,
        audit.total_cpd_markers,
        len(cpd_nums),
    )

    return audit


# ============================================================
# Name cleaning — consolidated, handles all observed formats
# ============================================================

# Tokens that NEVER appear inside a real IUPAC name. If the cleaned
# name contains one of these, everything from that token onward is
# reagent / stoichiometry / prose leakage and should be cut. Match
# leading char as `[\s(]+` so quantities embedded in synthesis
# protocol parens — `(0.7 g, 1.74 mmol)` — get caught from the
# opening paren rather than from the second quantity inside.
_NON_IUPAC_TAIL_TOKENS = re.compile(
    r"[\s(]+(?:"
    r"\d+\.?\d*\s*(?:g|mg|mL|mmol|mol|μL|nM|μM|equiv|eq)\b"   # (0.7 g), 1.74 mmol, 0.45 mL
    r"|diphenylphosphoryl|triethylamine|triethyla\b"           # reagents
    r"|HBTU|EDC|DIPEA|DMF\b|DCM\b|THF\b|DMSO|MeOH|EtOAc"        # solvents/reagents
    r"|reflux|stirred|added\b|cooled|filtered"                  # protocol verbs
    r"|MS\s*\(ESI\)|m/z\s|1H\s*NMR|13C\s*NMR"                  # spectra headers
    r"|Cpd\.?\s+No\.?|Compound\s+\d|Example\s+\d|Step\s+\d"    # cross-refs
    r"|Intermediate\s+\d"
    r")",
    re.IGNORECASE,
)


def _looks_like_iupac_start(name: str) -> bool:
    """Validate the START of a cleaned IUPAC name. Real IUPAC names begin
    with one of:
      - A locant or stereo descriptor in parens / brackets:
        "(1R,2S)-", "(S)-", "(2-...)", "[1,1'-biphenyl]"
      - A digit-locant: "1-", "2,4-", "1H-"
      - A lowercase chemical word (the most common case): "methyl",
        "ethyl", "phenyl", "rac-", "cis-", "trans-", "racemic", "meso"
      - A capital letter that's part of a chemical descriptor:
        "N-methyl", "S-(...)" (chirality), "L-alanine", "D-glucose"
    Anything else (e.g. "Step", "Synthesis", "The", "When", a
    stray punctuation char) means we have prose contamination.
    """
    if not name:
        return False
    s = name.lstrip()
    if not s:
        return False
    c0 = s[0]
    if c0 in "([":
        return True
    if c0.isdigit():
        return True
    if c0.islower():
        return True
    # Capital-letter starts allowed only for: N-, S-, R-, L-, D-, O-, C-,
    # tert-, Boc-, etc. plus the 1-letter element-style descriptors.
    if c0.isupper() and len(s) >= 2:
        # "N-methyl..." → next char is "-"
        if s[1] in "-,":
            return True
        # Common stereo/locant uppercase prefixes
        for prefix in (
            "Boc-", "Cbz-", "Fmoc-", "Tf-",
            "L-", "D-", "DL-", "rac-",
        ):
            if s.startswith(prefix):
                return True
    return False


def _trim_to_iupac_tail(name: str) -> str:
    """Truncate the name at the first non-IUPAC token (stoichiometry,
    reagent name, NMR/MS header, cross-reference). Patent-agnostic — no
    enumeration of every possible tail, just a list of tokens that
    cannot appear *inside* a real IUPAC name.

    Iterates: after a cut, scan again — sometimes the residual end
    still has trailing intermediate-id codes like `(S55A) S44A` that
    aren't quantity tokens but ARE clearly non-IUPAC tail material.
    """
    if not name:
        return name
    # First pass: cut at non-IUPAC tokens
    while True:
        m = _NON_IUPAC_TAIL_TOKENS.search(name)
        if not m or m.start() < 20:
            break
        name = name[:m.start()].rstrip(' ,;.')
    # Trailing intermediate-code refs: `... (S55A) S44A` — chemists
    # use these as cross-references at end of synthesis prose. Strip
    # any trailing `(S\d+[A-Z]?) S\d+[A-Z]?` repeat.
    name = re.sub(
        r'\s*\(S\d+[A-Z]?\)\s+S\d+[A-Z]?\s*$',
        '', name,
    ).rstrip(' ,;.')
    return name


def _clean_name(name: str) -> str:
    """Clean a candidate IUPAC name extracted from patent text.

    Handles all observed prefix/suffix patterns across 8 patents AND
    Markdown artifacts that arise when text comes from a MinerU OCR pass
    (embedded image links, escaped asterisks, mid-name newlines).

    Final structural validation:
      - The cleaned name must start with a recognized IUPAC pattern
        (lowercase letter, digit, paren, bracket, or known capital
        prefix like N-/S-/L-/Boc-). If it doesn't, return empty so
        the caller skips it.
      - Tail tokens that never appear inside a real IUPAC (mmol, g,
        reagent names, MS / NMR headers, cross-refs) trigger truncation.
    """
    # ── Markdown artifact stripping (MinerU source) ──
    # MinerU detection-tag pollution: "<|ref|>text<|/ref|><|det|>[[114, 99, 480, 950]]<|/det|>"
    # appears mid-IUPAC when MinerU layout detection bleeds into text output.
    name = re.sub(r"<\|[a-z/_]+\|>", " ", name, flags=re.IGNORECASE)
    name = re.sub(r"\[\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\]", " ", name)
    # Embedded image links: "...benzimidazol-2-yl}cyclohexanecarboxylic acid\n\n![](images/abc.jpg)..."
    name = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", name)
    # Backslash-escaped markdown emphasis chars: "(1S\*,2R\*)" → "(1S*,2R*)"
    name = re.sub(r"\\([*_~`])", r"\1", name)
    # Bare backslashes that aren't part of an escape we recognize
    name = name.replace("\\\\", " ").replace("\\n", " ")
    # Newlines, tabs → single space (mid-name line wraps from OCR)
    name = name.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # OCR confuses commas as periods inside heterocycle numbering brackets:
    # "[1.2,4]triazin" (wrong) → "[1,2,4]triazin" (correct).
    # Pattern: `[<digit>.<digit>,...]` where the dot should be a comma.
    name = re.sub(
        r"\[(\d+(?:,\d+)?)\.(\d+(?:,\d+)?)\]",
        lambda m: f"[{m.group(1)},{m.group(2)}]",
        name,
    )

    # Strip "Step 1—Synthesis of" / "Step 2 - Preparation of" header.
    # Em-dash, en-dash, and ASCII hyphen all observed.
    name = re.sub(
        r'^Step\s+\d+\s*[—–\-:]?\s*(?:Synthesis|Preparation)\s+of\s+',
        '', name, flags=re.IGNORECASE,
    )

    # Strip "Synthesis of" / "Preparation of" prefix (no Step header)
    name = re.sub(r'^(?:Synthesis|Preparation)\s+of\s+', '', name, flags=re.IGNORECASE)

    # Strip non-IUPAC stereo descriptors that OPSIN rejects: "racemic",
    # "rac", "meso", and "rac-" before the actual IUPAC name. Also strip
    # leading orphan-segment markers like "S,2R)" or "R,2S)" which arise
    # when the comma splitter cut at "(1S,2R)-" wrongly. Recognized as a
    # leading "[A-Z],\d[A-Z]?\)" with no opening paren.
    name = re.sub(
        r'^(?:racemic|rac\.?|meso)\s+', '', name, flags=re.IGNORECASE,
    )
    name = re.sub(r'^[A-Z]\*?,\s*\d+[A-Z]?\*?\)\s*-?', '', name)

    # Structural tail trim. Walk back from the end to find the rightmost
    # IUPAC-suffix boundary (a letter or recognized closing bracket).
    # Anything past that — quantity tokens, reagent prose, compound IDs —
    # is dropped. Patent-agnostic.
    name = _trim_to_iupac_tail(name)

    # Strip "(Compound N)" / "(Compound 5A)" suffix
    name = re.sub(r'\s*\(Compound\s*[\w]+\)\s*$', '', name)

    # Strip "(S9)" / "(S55A)" / "(Cpd. No. 148)" style IDs (with optional
    # letter suffix like "(S55A)" — earlier the regex required digits-paren).
    name = re.sub(r'\s*\((?:S\d+[A-Z]?|Cpd\.?\s*(?:No\.?\s*)?\d+)\)\s*$', '', name)

    # Strip "(i)" synthesis step markers and everything after
    name = re.sub(r'\s*\(i\)\s*.*$', '', name)

    # Strip "and (stereo variant)" for dual-compound examples
    name = re.sub(r'\s+and\s+\(?\d+\w*[,\s].*$', '', name)

    # Remove line-break artifacts
    name = re.sub(r'\s*-\s+', '-', name)  # "methyl- ethyl" → "methyl-ethyl"
    name = re.sub(r'\s+', ' ', name)

    cleaned = name.strip()

    # Final structural validation: if the cleaned name doesn't start
    # like an IUPAC name, return empty. The caller (_extract_names_by_density
    # and the OPSIN cascade) will then drop this candidate.
    if not _looks_like_iupac_start(cleaned):
        return ""

    return cleaned


# ============================================================
# IUPAC density scoring — format-agnostic name detection
# ============================================================

def _iupac_density(text: str) -> tuple[float, int]:
    """Score a text span for IUPAC fragment density.

    Returns (density, hit_count) where density = hits / len(text).
    Higher density = more likely to be an IUPAC name.
    """
    if not text or len(text) < 10:
        return 0.0, 0

    text_lower = text.lower()

    # Check for stop words (likely prose, not a name)
    if any(sw in text_lower for sw in STOP_WORDS):
        return 0.0, 0

    hits = sum(1 for frag in IUPAC_FRAGMENTS if frag in text_lower)
    density = hits / len(text) if text else 0
    return density, hits


def _extract_names_by_density(text: str) -> list[dict]:
    """Extract compound names by finding IUPAC-dense spans near markers.

    Two strategies:
    1. Marker-adjacent: find IUPAC-dense spans after Example/Cpd markers
    2. Claims list: find semicolon-delimited IUPAC names with (NUMBER) IDs

    Format-agnostic — no hardcoded patterns per patent format.
    Returns list of {num: int, name: str, source: str}.
    """
    candidates = []
    seen_nums = set()

    # ── Strategy 1: Claims list (semicolon-delimited) ──────────
    # Format: "IUPAC_NAME (NUMBER); IUPAC_NAME (NUMBER); ..."
    segments = text.split(';')
    for seg in segments:
        seg = seg.strip()
        m = re.search(r'\((\d{1,4})\)\s*\.?\s*$', seg)
        if not m:
            continue
        num = int(m.group(1))
        name = seg[:m.start()].strip()

        if len(name) < 20 or num in seen_nums:
            continue

        density, hits = _iupac_density(name)
        if hits >= 2 and density > 0.03:
            seen_nums.add(num)
            candidates.append({
                'num': num,
                'name': name,
                'source': 'claims_list',
                'score': density,
            })

    # ── Strategy 2: Marker-adjacent density scan ───────────────
    # Find all Example/Cpd/Compound markers, score text after each
    markers = []
    for m in re.finditer(r'Example\s+(\d+)', text):
        markers.append((int(m.group(1)), m.end(), 'example'))
    for m in re.finditer(r'Cpd\.?\s*(?:No\.?\s*)?(\d+)', text):
        num = int(m.group(1))
        if num not in seen_nums and num not in {mk[0] for mk in markers}:
            markers.append((num, m.end(), 'cpd'))
    for m in re.finditer(r'Compound\s+(\d+)', text):
        num = int(m.group(1))
        if num not in seen_nums and num not in {mk[0] for mk in markers}:
            markers.append((num, m.end(), 'compound'))

    for num, pos, source in markers:
        if num in seen_nums:
            continue

        # Grab next 800 chars after the marker
        window = text[pos:pos + 800]

        # Quick check: does this window have ANY IUPAC content?
        _, window_hits = _iupac_density(window[:200])
        if window_hits < 1:
            continue  # Skip — this Example marker leads to prose, not a compound

        # Skip leading whitespace, colons, numbers (handles "Example 1: NAME",
        # "Example 1 620.26 NAME", "Synthesis of NAME", etc.)
        name_start = 0
        for i, ch in enumerate(window[:150]):
            if ch.isalpha() and window[i:i+3].lower() not in ('the', 'and', 'for', 'was', 'are'):
                name_start = i
                break

        # Find the natural END delimiter for the compound name.
        # IUPAC names in patents are reliably terminated by:
        #   "(Compound N)" / "(Cpd. No. N)" / "(S9)"  — explicit ID
        #   "(i)" / "Step 1" / "(*step*)"             — synthesis section start
        #   ". <Capital>"                              — sentence boundary
        #   "Example N+1"                              — next example
        #   "Intermediate"                             — synthesis intermediate header
        #   "Synthesis of"                             — repeat header
        rest = window[name_start:]

        end_pos = len(rest)
        for delim_pattern in [
            r'\s*\(Compound\s*[\w]+\)',
            r'\s*\(Cpd\.?\s*(?:No\.?\s*)?\d+\)',
            r'\s*\(S\d+\)',
            r'\s*\(i\)\s',
            r'\s+Synthesis\s+of\s',
            r'\s+Intermediate\s+\d',
            r'\s+Step\s+[A-Z]\s*:',         # NEW: "Step A:", "Step B:"
            r'\s+Step\s+\d',
            r'\s+Example\s+\d',
            r'\.\s+[A-Z]',
        ]:
            m = re.search(delim_pattern, rest)
            if m and m.start() < end_pos and m.start() >= 20:
                end_pos = m.start()

        candidate = rest[:end_pos].strip()

        # Validate: must be IUPAC-dense and reasonable length
        if len(candidate) < 20 or len(candidate) > 500:
            continue

        density, hits = _iupac_density(candidate)
        if hits >= 2 and density > 0.02:
            seen_nums.add(num)
            candidates.append({
                'num': num,
                'name': candidate,
                'source': f'density_{source}',
                'score': density,
            })

    # ── Strategy 3: Reverse parenthetical "<IUPAC> (Compound XX)" ──
    # Many patents use this format in synthesis prose:
    #   "Example 5: Synthesis of <IUPAC name 1> (Compound 5A) and
    #    <IUPAC name 2> (Compound 5B) Synthesis of Intermediate 5a: ..."
    # One `Example N:` header can introduce multiple stereoisomer
    # compounds (5A, 5B, 39AA, 39AB, etc.). This strategy walks every
    # `(Compound XX)` match in the patent text and grabs the
    # IUPAC-dense substring immediately before it. Patent-agnostic —
    # just looks for the parenthetical pattern.
    seen_ids: set[str] = set()   # str-keyed, preserves letter suffix
    # Use len-then-string sort so '5' < '5A' < '5B' < '6' in the
    # final candidate list
    for paren_m in re.finditer(
        r"\(Compound\s+(\d{1,4}[A-Za-z]{0,4})\)", text,
    ):
        cid_str = paren_m.group(1)
        if cid_str in seen_ids:
            continue
        # Take up to 400 chars BEFORE the parenthetical
        win_start = max(0, paren_m.start() - 400)
        before = text[win_start:paren_m.start()]
        # Trim to start at the last delimiter that ends a previous
        # IUPAC name segment — we don't want to pick up the "Synthesis
        # of" prefix or material from a prior compound.
        for delim in (
            ") and ",            # end of previous compound's parenthetical
            "Synthesis of ",     # most common prefix on Example N: headers
            ": Synthesis of ",
            "Example ",
        ):
            i = before.rfind(delim)
            if i >= 0:
                before = before[i + len(delim):]
        candidate = before.strip().rstrip(",")
        if len(candidate) < 20 or len(candidate) > 500:
            continue
        density, hits = _iupac_density(candidate)
        if hits < 2 or density < 0.02:
            continue
        # Use the integer prefix as `num` for compatibility with
        # existing _build_example_index keying
        num_match = re.match(r"(\d{1,4})", cid_str)
        if not num_match:
            continue
        num = int(num_match.group(1))
        seen_ids.add(cid_str)
        # Treat letter-suffix variants as distinct entries — emit
        # one candidate per distinct compound id, but keep `num` as
        # the integer part. Downstream merger uses `num` as the key
        # so we emit a synthetic key that preserves the suffix.
        candidates.append({
            'num': num,
            'compound_id': cid_str,    # full id with letter suffix
            'name': candidate,
            'source': f'reverse_paren',
            'score': density,
        })

    # ── Strategy 4: TABLE-1 ordered-list (comma-separated, position-indexed) ──
    # Some patents (US8952177 e.g.) list every compound's IUPAC name in
    # TABLE 1 with NO explicit compound number — position implies cpd N.
    # Format: "TABLE 1 <iupac_1>, <iupac_2>, <iupac_3>, ..., and solvates,
    #          hydrates, and pharmaceutically acceptable salts thereof."
    # IUPAC names contain commas inside parens/brackets/braces (1S,2R), [2,3-d]
    # so naive `split(",")` shatters them. Use depth-tracking splitter.
    #
    # Anchor strategy: match "TABLE 1" followed by whitespace and an
    # IUPAC-dense lookahead — skip "Table 1." back-references in prose.
    # End strategy: stop at the first end-of-list marker we recognize
    # ("thereof.", "The present invention", next TABLE-N), capped at 50k.
    # End signals (in order of strength): only a real end-of-region marker.
    # Patents often have multiple "thereof." within the table region between
    # sub-embodiments; ignore those. The strong end markers are:
    #   - "pharmaceutically acceptable carriers/excipients" (composition prose)
    #   - "method of treating" / "method of inhibiting" (use-claim prose)
    #   - next TABLE-N
    _T1_END_PATTERNS = (
        r"\bpharmaceutically\s+acceptable\s+(?:carriers?|excipients)",
        r"\bmethod\s+of\s+(?:treating|inhibiting|modulating)",
        r"\bTABLE\s*[-–—]?\s*[2-9]\b",
    )
    for table1_match in re.finditer(
        r"\bTABLE\s*[-–—]?\s*1\b\s+", text, re.IGNORECASE,
    ):
        win_start = table1_match.end()
        # Anchor sanity check: next 200 chars must look IUPAC-dense, else
        # this is a back-reference like "compounds listed in Table 1."
        anchor_density, anchor_hits = _iupac_density(text[win_start:win_start + 200])
        if anchor_hits < 2 or anchor_density < 0.05:
            continue

        # Find tightest end-marker
        rest = text[win_start:win_start + 50_000]
        win_end = win_start + len(rest)
        for pat in _T1_END_PATTERNS:
            m = re.search(pat, rest, re.IGNORECASE)
            if m:
                win_end = min(win_end, win_start + m.start())
        window = text[win_start:win_end]
        if len(window) < 200:   # nothing to parse
            continue

        # depth-tracking comma splitter (paren/brace/bracket aware)
        # Recovery: if we encounter unbalanced brackets (real patents have
        # typos like "[4-(trifluoromethoxy)benzyl-1H-" with a missing ']'),
        # the depth would otherwise stay >0 forever and no further commas
        # would split. We reset depth back to 0 whenever we see an
        # IUPAC-end-marker ("acid, " or "salt, ") followed by a likely
        # next-name token (capital letter, opening paren, digit-prefix).
        _RESET_RE = re.compile(
            r"(?:acid|salt|salts?|amine|amide|ester|ate|ide)\s*,\s+"
            r"(?=[\(\[]?[A-Z(]|\d|racemic\b|trans-|cis-)",
            re.IGNORECASE,
        )
        # Also: hard depth cap. Real IUPAC names rarely nest > 6 deep —
        # if depth climbs to 8+, force-reset.
        depth = 0
        buf: list[str] = []
        segments_t1: list[str] = []
        i_ch = 0
        while i_ch < len(window):
            ch = window[i_ch]
            if ch in "([{":
                depth += 1
                if depth > 8:
                    depth = 0
            elif ch in ")]}":
                depth = max(0, depth - 1)
            # Split only on a real list-separator: comma followed by
            # whitespace. Skips chemistry-numbering cases like "2,2-dimethyl"
            # and "1,3-bis-" which have ',N-' (no space) at depth 0.
            if (
                ch == ","
                and depth == 0
                and i_ch + 1 < len(window)
                and window[i_ch + 1].isspace()
            ):
                segments_t1.append("".join(buf).strip())
                buf = []
                i_ch += 1
                continue
            # Recovery: depth>0 but an IUPAC-end pattern is right here?
            if depth > 0:
                m_reset = _RESET_RE.match(window, i_ch)
                if m_reset:
                    # The current `buf` is malformed (an unbalanced bracket
                    # left it stuck at depth>0). Discard it — the previous
                    # segment was already emitted correctly. Just resync at
                    # the next segment start.
                    buf = []
                    depth = 0
                    i_ch = m_reset.end()
                    continue
            buf.append(ch)
            i_ch += 1
        if buf:
            segments_t1.append("".join(buf).strip())

        # Strategy 4 has its OWN numbering (TABLE-1 position). Don't
        # consult `seen_nums` — Strategy 2 (Example markers) and
        # Strategy 4 (TABLE-1 position) can disagree about what
        # "compound 11" is, and we want BOTH to land in `candidates`
        # so the merge in `_build_example_index` can rank by source.
        # (TABLE-1 position-anchored extraction is more reliable than
        # prose Example markers, so the merge prefers it.)
        pos = 0
        seen_pos: set[int] = set()   # local to Strategy 4
        for seg in segments_t1:
            # Skip header preamble that landed in segment 0
            if pos == 0 and seg.lower().startswith("compounds"):
                continue
            candidate = seg.strip().rstrip(",")
            if not (20 <= len(candidate) <= 500):
                continue
            density_t1, hits_t1 = _iupac_density(candidate)
            if hits_t1 < 2 or density_t1 < 0.02:
                continue
            pos += 1
            if pos in seen_pos:
                continue
            seen_pos.add(pos)
            candidates.append({
                'num': pos,
                'compound_id': str(pos),
                'name': candidate,
                'source': 'table1_ordered_list',
                'score': density_t1,
            })
        # Only process the first IUPAC-dense TABLE 1 anchor — successive
        # ones are duplicate matches at neighboring offsets.
        break

    # Sort by compound number, then suffix (so "5", "5A", "5B" stay together)
    candidates.sort(key=lambda c: (c['num'], c.get('compound_id', str(c['num']))))
    return candidates


# ============================================================
# Google Patents text fetching
# ============================================================

def fetch_patent_text(patent_id: str) -> dict | None:
    """Fetch clean text from Google Patents.

    Returns dict with:
      - 'description', 'claims'           : extracted text bodies
      - 'embedded_compounds'              : list of {smiles, inchi_key}
                                            dicts captured from
                                            <span itemprop="smiles"> +
                                            <span itemprop="inchi_key">
                                            adjacent pairs
      - 'figure_image_urls'               : list of figure image URLs
    or None if the page can't be fetched.

    The embedded_compounds entries are GP-canonical (RDKit-ready); they
    bypass OPSIN downstream. Captured by the same regex library used by
    the IUPAC density strategies — these are just additional patterns.
    """
    cache_file = CACHE_DIR / f"{patent_id}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        # Back-compat: fill in missing fields as empty so downstream code
        # can treat them uniformly with `cache.get(key, [])`.
        data.setdefault("embedded_compounds", [])
        data.setdefault("figure_image_urls", [])
        # Defensive HTML-unescape: older caches stored SMILES with raw
        # `&#43;` etc. New caches already unescape at fetch time but this
        # idempotent pass keeps stale cache files usable.
        for entry in data["embedded_compounds"]:
            if isinstance(entry, dict):
                if "&" in entry.get("smiles", ""):
                    entry["smiles"] = html.unescape(entry["smiles"])
                if "&" in entry.get("inchi_key", ""):
                    entry["inchi_key"] = html.unescape(entry["inchi_key"])
        logger.info(f"Google Patents cache hit for {patent_id}")
        return data

    url = f"https://patents.google.com/patent/{patent_id}/en"
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200:
            logger.warning(f"Google Patents returned {r.status_code} for {patent_id}")
            return None

        desc_match = re.search(
            r'<section itemprop="description"[^>]*>(.*?)</section>',
            r.text, re.DOTALL
        )
        claims_match = re.search(
            r'<section itemprop="claims"[^>]*>(.*?)</section>',
            r.text, re.DOTALL
        )

        result: dict = {}
        for key, match in [('description', desc_match), ('claims', claims_match)]:
            if match:
                # NOTE: this local intentionally avoids shadowing the
                # `html` module imported at the top of the file.
                section_html = match.group(1)
                text = re.sub(r'<[^>]+>', ' ', section_html)
                text = re.sub(r'\s+', ' ', text).strip()
                result[key] = text
            else:
                result[key] = ''

        # NEW — capture the structured chemical data we used to discard.
        # Pair-capture: (smiles, inchi_key) in one regex so positional
        # alignment is guaranteed (Markush scaffolds emit SMILES with no
        # InChIKey; reading two parallel lists shifts everything below).
        # Decode HTML entities (e.g. `&#43;` → `+`, `&amp;` → `&`) — GP
        # HTML-encodes SMILES special chars and RDKit can't parse them as-is.
        embedded = [
            {"smiles": html.unescape(s).strip(),
             "inchi_key": html.unescape(ik).strip()}
            for s, ik in _GP_EMBED_COMPOUND_PAT.findall(r.text)
        ]
        result["embedded_compounds"] = embedded
        result["figure_image_urls"]  = _GP_FIGURE_IMG_PAT.findall(r.text)

        if not result.get('description'):
            logger.warning(f"No description found for {patent_id}")
            return None

        cache_file.write_text(json.dumps(result))
        with_ik = sum(1 for c in embedded if c["inchi_key"])
        logger.info(
            f"Google Patents {patent_id}: "
            f"desc={len(result['description']):,} chars, "
            f"claims={len(result['claims']):,} chars, "
            f"embedded_compounds={len(embedded)} ({with_ik} with InChIKey), "
            f"figures={len(result['figure_image_urls'])}"
        )
        time.sleep(1)
        return result

    except Exception as e:
        logger.error(f"Google Patents fetch failed for {patent_id}: {e}")
        return None


# ============================================================
# Main extraction: audit → density extract → OPSIN validate
# ============================================================

def _count_assay_compounds(patent_id: str) -> int:
    """Read assay HARVEST output for `patent_id` and return the compound
    count. Used as the ground-truth count for the Strategy 5 gap check.

    Returns 0 if assay_tables.json is missing — gap check then silent.
    """
    p = config.OUTPUT_DIR / "text_extraction" / patent_id / "assay_tables.json"
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text())
        return len(data) if isinstance(data, dict) else 0
    except (json.JSONDecodeError, OSError):
        return 0


def _strategy0_gp_embedded_compounds(patent_id: str) -> list[Compound]:
    """Strategy 0: GP-embedded SMILES/InChIKey pairs from <span itemprop> tags.

    Reads from the gpatents_cache populated by `fetch_patent_text`. SMILES
    here are GP-canonical (RDKit-ready), so we bypass the OPSIN cascade
    entirely and emit pre-validated Compounds with `processing_status="validated"`.

    Filtering:
      - skip pairs missing either smiles or inchi_key (Markush scaffold stubs)
      - skip SMILES containing wildcard `*` (R-group templates)
      - skip if RDKit can't canonicalize (defensive — shouldn't happen)
      - skip if MW < MIN_SMILES_MW (150) — fragments
      - skip if MW > MAX_SMILES_MW (1500) — polymers / OCR garbage

    Compound IDs are positional (1..N) since GP doesn't carry a stable cid
    mapping in the structured-data tags. They'll be matched against BDB/Jie
    references by InChIKey, so positional cid is fine.
    """
    from ..core.smiles_utils import (
        canonicalize_smiles, get_inchikey, molecular_weight,
    )
    from ..core.iupac_to_smiles import (
        MIN_SMILES_MW, MAX_SMILES_MW, MIN_SMILES_LENGTH,
    )
    from ..core.models import CompoundProvenance

    cache_file = CACHE_DIR / f"{patent_id}.json"
    if not cache_file.exists():
        return []
    try:
        cache = json.loads(cache_file.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    embedded = cache.get("embedded_compounds", [])
    if not embedded:
        return []

    out: list[Compound] = []
    skipped_no_ik = skipped_wildcard = skipped_invalid = 0
    skipped_mw_low = skipped_mw_high = 0

    for idx, entry in enumerate(embedded, start=1):
        smi = (entry.get("smiles") or "").strip()
        ik  = (entry.get("inchi_key") or "").strip()
        if not smi or not ik:
            skipped_no_ik += 1
            continue
        # Markush template SMILES contain `*` wildcards — skip; the
        # paired InChIKey may still be valid for the drawn compound but
        # we can't trust the SMILES, and downstream MW gates will fail
        # on the wildcard atoms anyway.
        if "*" in smi:
            skipped_wildcard += 1
            continue
        canon = canonicalize_smiles(smi)
        if not canon or len(canon) < MIN_SMILES_LENGTH:
            skipped_invalid += 1
            continue
        mw = molecular_weight(canon)
        if mw is None or mw < MIN_SMILES_MW:
            skipped_mw_low += 1
            continue
        if mw > MAX_SMILES_MW:
            skipped_mw_high += 1
            continue

        # Cross-check: GP InChIKey should agree with RDKit canonicalization.
        # If they disagree, GP's pair is corrupted — fall through to use
        # the RDKit-derived InChIKey (more trustworthy than a malformed pair).
        rd_ik = get_inchikey(canon) or ik

        # Use a distinct cid namespace ("GP1", "GP2", ...) so positional
        # GP indexes don't collide with the regex-extracted "Compound 1",
        # "Cpd. No. 5" cids — those refer to the patent's own compound
        # numbering, which is unrelated to the order the SMILES appear
        # in the GP HTML. The cids here function purely as unique keys
        # in example_index; matching against BDB/Jie reference data is
        # done by InChIKey, not cid.
        compound = Compound(
            patent_id=patent_id,
            example_number=f"GP{idx}",
            iupac_name="",                          # GP doesn't embed IUPAC alongside
            iupac_source=IupacSource.PATENT_VERBATIM,
            source=CompoundSource.EXEMPLIFIED,
            smiles_from_text=smi,
            canonical_smiles=canon,
            inchikey=rd_ik,
            extraction_method="gp_embedded_meta",
            processing_status="validated",
            provenance=CompoundProvenance(
                route="google_patents_example",
                stage="gp_embedded_meta",
                chain=["gp_embedded_meta"],
            ),
        )
        out.append(compound)

    logger.info(
        f"Google Patents {patent_id}: Strategy 0 (GP-embedded) — "
        f"{len(out)} validated, "
        f"{skipped_no_ik} no-ik, {skipped_wildcard} wildcard, "
        f"{skipped_invalid} invalid, "
        f"{skipped_mw_low} mw<150, {skipped_mw_high} mw>1500 "
        f"(of {len(embedded)} embedded pairs)"
    )
    return out


def extract_compounds_from_clean_text(
    patent_id: str,
    text: str,
    *,
    skip_strategy_5: bool = False,
    text_source_format: str = "google_html",
    cascade_mode: str = "full",
) -> list[Compound]:
    """Extract compounds using IUPAC density scoring + GP-embedded SMILES.

    Format-agnostic — no hardcoded patterns per patent format.

    Args:
        skip_strategy_5: When True, skip the inline IUPAC HARVEST LLM
            fallback. The new orchestrator (`routes/process_patent`)
            calls Strategy 5 explicitly *after* route classification, so
            it asks for `skip_strategy_5=True` to avoid double-firing.
            Legacy callers (`extract_text_index`) leave it False.
        text_source_format: One of "google_html", "mineru_markdown",
            "google_html_fetched". Controls whether the iupac_to_smiles
            cascade engages OCR-cleanup stages (rule_clean, autocorrect,
            vision_ocr, LLM_clean). "google_html" is treated as clean
            (skips those stages); anything else gets the full cascade.
            Also gates Strategy 0 — GP-embedded compounds only fire when
            the source is HTML (the cache is HTML-derived).
        cascade_mode: "full" (default) runs the 5-stage cascade
            (PubChem → OPSIN → rule_clean → autocorrect → vision → LLM).
            "fast" runs only Stages 0-2 (PubChem, OPSIN, rule_clean) —
            no LLM, no vision, no autocorrect. Used by the markdown
            secondary pass in `process_patent` so it stays under the
            15-min budget when GP-embedded already supplies most of the
            compounds; the markdown pass is only filling gaps anyway,
            so candidates that need LLM cleanup will be picked up by
            Strategy 5 broad later in the orchestrator.
    """
    # ── Strategy 0: GP-embedded SMILES/InChIKey (zero-cost, OPSIN bypass) ──
    # Only fires when the text source is GP HTML, since the cache fields
    # are derived from the HTML scrape.
    gp_embedded = (
        _strategy0_gp_embedded_compounds(patent_id)
        if text_source_format in ("google_html", "google_html_fetched")
        else []
    )

    # Density-based extraction (Strategies 1-4)
    candidates = _extract_names_by_density(text)
    logger.info(f"Google Patents {patent_id}: {len(candidates)} name candidates from density extraction")

    # ── Strategy 5: IUPAC HARVEST fallback ──
    # Fires only when the regex strategies undershoot — uses an LLM agent
    # to extract additional (compound_id, IUPAC name) pairs and persists
    # discovered patterns to `IupacPatternLibrary` for cross-patent reuse.
    # Env-flag default ON; set IUPAC_HARVEST_FALLBACK=0 or skip_strategy_5=True
    # to disable.
    if not skip_strategy_5 and os.environ.get("IUPAC_HARVEST_FALLBACK", "1") == "1":
        n_assay = _count_assay_compounds(patent_id)
        existing_pairs = {c.get('compound_id') or str(c['num']): c['name']
                          for c in candidates if c.get('name')}
        if n_assay > 0 and len(existing_pairs) < 0.5 * n_assay:
            try:
                from ..core.assay_fsm.harvest.iupac_orchestrator import iupac_burst
                new_pairs = iupac_burst(
                    patent_id=patent_id,
                    text=text,
                    existing_pairs=existing_pairs,
                    n_assay_compounds=n_assay,
                    cost_tracker=cost_tracker,
                )
                # Append new pairs as Strategy-5 candidates
                existing_ids = {c.get('compound_id') or str(c['num']) for c in candidates}
                for cid, name in new_pairs.items():
                    if cid in existing_ids:
                        continue
                    # Try to derive int num from cid for Compound.num field
                    m = re.match(r"(\d+)", cid)
                    num = int(m.group(1)) if m else len(candidates) + 1
                    candidates.append({
                        'num': num,
                        'compound_id': cid,
                        'name': name,
                        'source': 'iupac_harvest',
                        'score': 0.0,
                    })
                logger.info(
                    f"Google Patents {patent_id}: Strategy 5 added "
                    f"{len(new_pairs)} candidates (existing {len(existing_pairs)}, "
                    f"assay {n_assay})"
                )
            except Exception as e:
                logger.warning(
                    f"Google Patents {patent_id}: Strategy 5 fallback failed: {e}"
                )

    # Build Compounds and route through 5-stage cascade (handles OPSIN failures via LM)
    # Strategy 0 compounds (GP-embedded) are already validated; seed them in.
    validated: list[Compound] = list(gp_embedded)
    failures = 0
    lm_used = 0

    # Track InChIKeys we already have from Strategy 0 — skip redundant
    # density-extracted candidates that resolve to the same molecule. This
    # is the dedup that makes Strategy 0 the highest-rank source: if GP
    # already gave us the canonical SMILES, we don't need to re-resolve via
    # OPSIN from a possibly-malformed IUPAC name.
    gp_inchikeys_seen: set[str] = {
        c.inchikey for c in gp_embedded if c.inchikey
    }

    for c in candidates:
        name = _clean_name(c['name'])
        # Use suffix-aware compound_id when the candidate has one
        # (reverse_paren strategy emits stereoisomer-distinct IDs like
        # "5A", "5B"). Fall back to the integer `num` for the legacy
        # density-marker strategies.
        cid_str = c.get('compound_id') or str(c['num'])
        compound = Compound(
            patent_id=patent_id,
            example_number=f"Example {cid_str}" if c['source'].endswith('example')
                           else (f"Compound {cid_str}" if c['source'] == 'reverse_paren'
                                 else f"Cpd. No. {cid_str}"),
            iupac_name=name,
            iupac_source=IupacSource.PATENT_VERBATIM,
            source=CompoundSource.EXEMPLIFIED,
            extraction_method=f"google_patents_{c['source']}",
            processing_status="text_done",
        )

        # Tag provenance route based on which sub-route this candidate came from
        # (density extractor recorded c['source'] = "density_example" / "density_table")
        route_hint = (
            "google_patents_table" if "table" in c.get('source', '')
            else "google_patents_example"
        )

        # cascade_mode="fast": only Stage 1 OPSIN (strict on noisy source) +
        # Stage 2 rule_clean. No LLM, no vision, no autocorrect — keeps
        # the markdown secondary pass under budget.
        is_clean = text_source_format == "google_html"
        if cascade_mode == "fast":
            from ..core.iupac_to_smiles import rule_based_clean
            smiles, _err = _try_opsin(name, strict=not is_clean)
            if not (smiles and validate_smiles(smiles) and len(smiles) >= 10):
                cleaned = rule_based_clean(name)
                if cleaned != name:
                    smiles, _err = _try_opsin(cleaned, strict=not is_clean)
            if smiles and validate_smiles(smiles) and len(smiles) >= 10:
                _finalize(compound, smiles, stage="opsin_direct", route_hint=route_hint)
        elif cost_tracker.patent_lm_exceeded(patent_id):
            # If patent's LM cost cap is hit, fall back to OPSIN-only (no LM stages)
            smiles, _ = _try_opsin(name)
            if smiles and validate_smiles(smiles) and len(smiles) >= 10:
                _finalize(compound, smiles, stage="opsin_direct", route_hint=route_hint)
        else:
            # cascade_mode="full": is_clean_text=True only for actual clean HTML.
            # MinerU markdown carries OCR errors (pyrolo, [1.2,4], <|ref|> tags)
            # that need the full cascade (rule_clean → autocorrect → vision_ocr → LLM).
            _convert_single(compound, is_clean_text=is_clean, route_hint=route_hint)

        if compound.processing_status == "validated":
            # Skip if Strategy 0 already produced this exact molecule by
            # InChIKey — GP-embedded SMILES is more trustworthy than our
            # OPSIN parse of a possibly-OCR'd IUPAC name.
            if compound.inchikey and compound.inchikey in gp_inchikeys_seen:
                continue
            validated.append(compound)
            if compound.inchikey:
                gp_inchikeys_seen.add(compound.inchikey)
            if compound.lm_normalized:
                lm_used += 1
        else:
            failures += 1
            logger.debug(f"  All stages failed for #{c['num']}: {name[:60]}")

    logger.info(
        f"Google Patents {patent_id}: {len(validated)} validated "
        f"({len(gp_embedded)} from Strategy 0 GP-embedded, "
        f"{lm_used} via LM), {failures} failures, "
        f"LM spend ${cost_tracker.patent_spend(patent_id):.3f}"
    )
    return validated


def extract_patent_google(patent_id: str) -> list[Compound]:
    """Full text-index extraction — MinerU markdown preferred, HTML fallback.

    Resolution order:
      1. {patent_id}/all_pages/page_*.md      (MinerU-derived markdown)
      2. output_v2/gpatents_cache/{pid}.json  (cached HTML scrape)
      3. live HTTP fetch from Google Patents  (last resort)

    The IUPAC-density strategies operate on a single concatenated text
    blob, so the markdown-vs-HTML choice is just a string source — no
    per-strategy retuning needed. Cost: $0 unless Strategy 5 fires.
    """
    from ..core.patent_text import load_patent_description, load_gp_claims

    text, src_format = load_patent_description(patent_id, prefer_format="auto")
    if not text:
        # Last resort: live fetch from Google Patents
        text_data = fetch_patent_text(patent_id)
        if not text_data:
            return []
        text = (text_data.get('description', '') + '\n'
                + text_data.get('claims', ''))
        src_format = "google_html_fetched"
    else:
        # If we got the description from MinerU/HTML cache, also append
        # claims (HTML cache only — markdown already includes claims as
        # part of the page text).
        if src_format == "google_html":
            claims = load_gp_claims(patent_id)
            if claims:
                text = text + "\n" + claims

    logger.info(
        f"Google Patents {patent_id}: source_format={src_format}, "
        f"text_len={len(text):,}"
    )
    return extract_compounds_from_clean_text(
        patent_id, text, text_source_format=src_format,
    )
