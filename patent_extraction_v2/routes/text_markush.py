"""Text-side Markush dictionary extractor (Phase A of text scaffolding).

Why this exists
---------------
The image-side cropping/labeling pipeline is isolated from text. When
DECIMER reads a structure crop and emits SMILES with `[R...]`, we have
no way (image-side alone) to tell whether that's a real Markush
placeholder or a misread CIP stereo descriptor like literal "(R)" /
"(S)". The image-side detector must be GATED on a text-derived
authoritative R-group dictionary: only labels that the patent's claims
or Markush text ACTUALLY define as substituent positions count as
Markush.

This module produces that dictionary per patent. It is the input to:
  - The image-side Markush gate (Phase D)
  - The downstream Markush enumerator (markush.enumerate.enumerate_markush)

Data flow
---------
PaddleX `.md` pages   (data/{patent_id}/all_pages/)
        │
        ▼
[ symbolic regex (LM-free) ]   ← markush.context._symbolic_r_group_parse
        │   text option names: "methyl", "ethyl", "phenyl", ...
        ▼
[ resolve to SMILES ]           ← markush.mapper.lookup_substituent
        │   ("methyl" → "C")
        ▼
{ "R1": RGroupDef(options_text=[...], options_smiles=[...]),
  "R2": ...,
  "R19a": ... }
        │
        ▼
output_v2/text_extraction/{pid}/markush_dict.json

Generalizability rules (audit per call)
---------------------------------------
- No patent-specific code (no `if patent_id == "USXXXX"`).
- Regex patterns match generic claim phrasing only (no patent IDs):
  R-label + "is" + comma/and-separated option list, with termination on
  the next sentence or the next R-label. See `_RGROUP_DEF_RE` below.
- Substituent name-to-SMILES lookup uses the cross-patent
  `SUBSTITUENT_LIBRARY` (~80 common drug fragments) shared across all
  patents.

Usage
-----
    python3 -m patent_extraction_v2.routes.text_markush --patent US9718825

    # Or as a library:
    from patent_extraction_v2.routes.text_markush import extract_markush_dict
    md = extract_markush_dict("US9718825")
    print(md.r_groups.keys())
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..core import config
from ..core.substituent_vocab import expand_substituent
from ..markush.context import (
    _is_valid_rgroup_option,
    _clean_rgroup_text,
)
from ..markush.mapper import lookup_substituent

logger = logging.getLogger(__name__)


# ── Public dataclasses ──────────────────────────────────────────────

@dataclass
class RGroupEntry:
    """One R-position's text-claimed substituent options + resolved SMILES.

    `options_text` is the raw list of option names parsed from the
    patent's claim/Markush text (e.g., "methyl", "C1-C6 alkyl",
    "optionally substituted phenyl"). `options_smiles` is the subset
    that resolved to a valid SMILES via SUBSTITUENT_LIBRARY (and OPSIN
    as fallback inside lookup_substituent).
    """
    label: str                          # e.g. "R1", "R19a", "G", "X"
    options_text: list[str] = field(default_factory=list)
    options_smiles: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)


@dataclass
class MarkushDict:
    """Per-patent authoritative Markush R-group dictionary.

    This is consumed by:
      - Phase D image-side Markush gate (label-in-dict check)
      - markush.enumerate.enumerate_markush (R-group library input)
    """
    patent_id: str
    method: str = "symbolic"            # "symbolic" | "multi_agent_lm"
    n_pages_scanned: int = 0
    r_groups: dict[str, RGroupEntry] = field(default_factory=dict)
    audit: dict = field(default_factory=dict)

    @property
    def labels(self) -> set[str]:
        """Authoritative set of Markush label tokens for the gate.

        The image-side detector compares DECIMER's `[R...]` placeholders
        against this set (case-insensitive). Anything outside this set
        is treated as non-Markush (stereo descriptor, OCR noise, or an
        R-group the text-extractor missed).
        """
        return {label.lower() for label in self.r_groups}

    def to_json(self) -> dict:
        return {
            "patent_id": self.patent_id,
            "method": self.method,
            "n_pages_scanned": self.n_pages_scanned,
            "r_groups": {label: asdict(entry) for label, entry in self.r_groups.items()},
            "audit": self.audit,
        }


# ── Page enumeration ────────────────────────────────────────────────

def _list_text_pages(patent_id: str) -> list[Path]:
    """Return all `page_NNNN.md` files for the patent in document order."""
    pages_dir = config.DATA_DIR / patent_id / "all_pages"
    if not pages_dir.exists():
        logger.warning(f"No all_pages/ directory for {patent_id}: {pages_dir}")
        return []
    return sorted(pages_dir.glob("page_*.md"))


# ── Per-page R-group parsing (with provenance) ──────────────────────

# Generic claim-style R-group definition pattern.
# Matches: "R1 is selected from the group consisting of methyl, ethyl, ..."
# Also: "R is methyl or ethyl ...", "R^a is ..."
#
# Termination uses a positive LOOKAHEAD so the boundary text isn't consumed —
# the next finditer iteration can then match the next R-label definition.
# This is critical because patents commonly chain definitions with COMMAS:
#
#   "...R19a is selected from ... hydrogen and alkyl, R19b is selected from
#    ... amino, alkoxy, R20a is selected from ..., R20b is selected from
#    ... aryl. The term ..."
#
# Observed boundary frequency across the 3 image-heavy patents:
#   comma + R-is:  35  ← dominant pattern
#   semicolon + R-is:  16
#   period + capital:  1
# A regex that only terminates on `.` or `;` therefore swallows ~70% of the
# cross-R-label boundaries and contaminates the option list of the prior
# R-group with the option lists of every following R-group up to the next
# sentence end. This is generic to claim phrasing — no patent-specific code.
_RGROUP_DEF_RE = re.compile(
    r"(R\d+[a-z]?|[GXY])\s+is\s+"
    r"(?:selected from (?:the (?:group|series) )?consisting of\s+)?"
    r"(.*?)"
    r"(?="
        r"\s*\.\s+[A-Z]"                                            # sentence end (period+capital)
        # Next R-label definition: [, or ;] + optional "and" + R-label + "is"
        # The optional "and" handles claim phrasing like
        # "...hydrogen and alkyl, and R20b is selected from..."
        # which is a common end-of-list connector before the final item.
        r"|\s*[,;]\s+(?:and\s+)?(?:R\d+[a-z]?|[GXY])\s+is\b"
        # Semicolon + capital — general clause boundary (subordinate-clause
        # break in claim text). NOT comma+capital, which would be too
        # aggressive and fire mid-list (e.g. "methyl, Ethyl" if OCR
        # capitalizes mid-list options). Semicolons are reserved for
        # clause-level breaks in formal patent claim phrasing.
        r"|\s*;\s+[A-Z]"
        r"|\s*\.\s*$"                                               # paragraph end
    r")",
    re.DOTALL,
)

# Match a page-NNNN.md filename to its integer page number for source_pages.
_PAGE_NUM_RE = re.compile(r"page_(\d+)\.md$")

# Filter for non-chemistry tokens that the R-group regex catches by
# accident. These are CLAIM-LEVEL artifacts (formula numbers like
# "VIII-A", section-cross-refs like "as applies to alkyl groups in
# general"), not real substituent names. Patent-agnostic: matches
# the structural shape, not specific patent text.
_ROMAN_OR_FORMULA_REF = re.compile(
    r"^(?:[IVXLCDM]+)(?:[-‐‑‒–—][A-Za-z]+)?$",   # "VIII", "VIII-A", "VII-C"
    re.IGNORECASE,
)


def _is_chemistry_option(opt: str) -> bool:
    """Reject tokens that the R-group regex captured by accident.

    Returns False when `opt` is clearly NOT a chemistry substituent
    name (Roman numeral formula refs, multi-word claim cross-refs).
    Returns True otherwise — caller should still validate via
    SUBSTITUENT_LIBRARY/OPSIN/expand_substituent for actual SMILES.
    """
    s = opt.strip()
    if not s:
        return False
    # Roman numeral formula refs: "VIII-A", "VII-C", etc.
    if _ROMAN_OR_FORMULA_REF.match(s):
        return False
    # Long sentence fragments — patent claim cross-references that
    # the regex picked up. Real substituent names are rarely > 6 words
    # and even shorter ones don't include sentence connectors.
    words = s.split()
    if len(words) > 6 and any(w.lower() in {"as", "applies", "such", "thus", "respectively"} for w in words):
        return False
    return True


def _strip_paddlex_layout_artifacts(text: str) -> str:
    """Remove PaddleX layout markers that survive the HTML strip.

    PaddleX MD pages emit `<|ref|>text<|/ref|><|det|>[[x1,y1,x2,y2]]
    <|/det|>` per text block. Stripping `<...>` removes the brackets
    but leaves both the layout-tag name (`text`/`title`/`image`/...)
    and the bbox coords `[[137, 595, 480, 620]]` between adjacent
    blocks. These break the R-group regex terminator because they
    sit between "...alkyl;" and "R1b is..." — preventing the
    semicolon-then-R-label boundary from matching.

    Generic to PaddleX output; no patent-specific tokens.
    """
    # Strip "text [[bbox]]" / "title [[bbox]]" / etc. — the layout-tag
    # word followed by 4-int bracket coords.
    text = re.sub(
        r"\b(?:text|title|image|table|list|caption|formula|figure)\s*\[\[[\d,\s]+\]\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    # Strip any orphan 4-int bracket bbox that survived (no preceding
    # tag word). 4-int because chemistry uses bracket atoms like [H],
    # [C@H], [N+] — those don't contain digits-only contents.
    text = re.sub(r"\[\[[\d,\s]+\]\]", " ", text)
    return text


def _parse_page(page_path: Path) -> dict[str, list[str]]:
    """Run the same symbolic regex as `_symbolic_r_group_parse` but
    scoped to ONE page so we can record per-page provenance.

    Returns dict[label, list[option_text]] for this page only.
    """
    text = page_path.read_text(errors="replace")
    text = re.sub(r"<[^>]+>", " ", text)            # strip HTML / PaddleX angle-tags
    text = _strip_paddlex_layout_artifacts(text)    # strip "text [[bbox]]" residues

    page_groups: dict[str, list[str]] = {}
    for m in _RGROUP_DEF_RE.finditer(text):
        label = m.group(1)
        options_text_raw = _clean_rgroup_text(m.group(2).strip())
        options = [o.strip() for o in re.split(r",\s+|\s+and\s+", options_text_raw)
                   if len(o.strip()) > 1]
        # Strip leading "and " — leftover from "..., and X" patterns
        # where the comma split fires before the "and" connector.
        options = [re.sub(r"^and\s+", "", o, flags=re.IGNORECASE) for o in options]
        # Also strip a trailing period+digit chunk like ". 14" — claim/
        # section numbering that can leak when the regex terminator
        # (\.\s+[A-Z]) doesn't fire because the next char is a digit.
        options = [re.sub(r"\.\s*\d+\s*$", "", o).strip() for o in options]
        options = [o for o in options
                   if o and _is_valid_rgroup_option(o) and _is_chemistry_option(o)]
        if options:
            page_groups.setdefault(label, []).extend(options)
    return page_groups


# ── Main entry point ────────────────────────────────────────────────

def extract_markush_dict(patent_id: str) -> MarkushDict:
    """Extract the authoritative R-group dictionary for `patent_id`.

    Symbolic-only (no LM calls) — for the LOW-difficulty fast path
    that covers most patents. Returns a MarkushDict whose `.labels`
    set is the gate the image-side detector will use to distinguish
    real Markush placeholders from CIP stereo descriptors.

    For HIGH-difficulty patents where the symbolic pass returns
    nothing, the caller can fall back to `markush.context.
    extract_markush_multiagent` (LM-driven). That fallback is NOT
    invoked here by default — it is opt-in to keep cost predictable.
    """
    pages = _list_text_pages(patent_id)
    n_pages = len(pages)

    # Per-page parse so we can attach source_pages provenance
    # (same regex as _symbolic_r_group_parse, but page-scoped)
    label_to_options: dict[str, list[str]] = {}
    label_to_pages: dict[str, set[int]] = {}
    for page_path in pages:
        m = _PAGE_NUM_RE.search(page_path.name)
        page_num = int(m.group(1)) if m else -1
        page_groups = _parse_page(page_path)
        for label, opts in page_groups.items():
            label_to_options.setdefault(label, []).extend(opts)
            if page_num >= 0:
                label_to_pages.setdefault(label, set()).add(page_num)

    # Resolve text options to SMILES via the vocab fallback chain:
    #   1. SUBSTITUENT_LIBRARY (~80 entries, exact + partial match)
    #   2. OPSIN (handles many IUPAC fragment names)
    #   3. expand_substituent → range expander (Cn-Cm patterns) +
    #      unranged-chain defaults + ChEMBL lookup
    # Each layer is patent-agnostic; layers fall through silently to
    # the next on miss.
    r_groups: dict[str, RGroupEntry] = {}
    n_resolved = 0
    n_unresolved = 0
    n_resolved_via_vocab = 0
    for label, options_text in label_to_options.items():
        # Dedup while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for o in options_text:
            key = o.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(o.strip())

        smiles_set: list[str] = []
        for opt in deduped:
            smi = lookup_substituent(opt)
            if smi:
                if smi not in smiles_set:
                    smiles_set.append(smi)
                n_resolved += 1
                continue
            # Layer 3: vocab expansion (range expander, unranged
            # defaults, ChEMBL). Returns a LIST because a "(C1-C4)-
            # alkyl" expands to 4 concrete SMILES.
            expansions = expand_substituent(opt)
            if expansions:
                added = 0
                for s in expansions:
                    if s not in smiles_set:
                        smiles_set.append(s)
                        added += 1
                if added:
                    n_resolved += 1
                    n_resolved_via_vocab += 1
                else:
                    n_unresolved += 1
            else:
                n_unresolved += 1

        r_groups[label] = RGroupEntry(
            label=label,
            options_text=deduped,
            options_smiles=smiles_set,
            source_pages=sorted(label_to_pages.get(label, set())),
        )

    audit = {
        "n_labels": len(r_groups),
        "n_options_text_total": sum(len(e.options_text) for e in r_groups.values()),
        "n_options_smiles_resolved": n_resolved,
        "n_options_resolved_via_vocab_expansion": n_resolved_via_vocab,
        "n_options_unresolved": n_unresolved,
        "lm_calls": 0,
        "cost_usd": 0.0,
        "warnings": [] if r_groups else ["no R-group definitions found by symbolic regex"],
    }

    return MarkushDict(
        patent_id=patent_id,
        method="symbolic",
        n_pages_scanned=n_pages,
        r_groups=r_groups,
        audit=audit,
    )


def write_markush_dict(md: MarkushDict, out_dir: Path | None = None) -> Path:
    """Persist a MarkushDict as JSON for downstream consumption."""
    if out_dir is None:
        out_dir = config.OUTPUT_DIR / "text_extraction" / md.patent_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "markush_dict.json"
    out_path.write_text(json.dumps(md.to_json(), indent=2))
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────

def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract per-patent text-side Markush R-group dictionary.")
    ap.add_argument("--patent", action="append", required=True,
                    help="Patent ID (repeatable). Example: --patent US9718825")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    print(f"{'patent':>14}  {'pages':>5}  {'labels':>6}  {'opts_txt':>8}  {'opts_smi':>8}  {'unresolved':>10}")
    print("-" * 80)
    for pat in args.patent:
        md = extract_markush_dict(pat)
        out_path = write_markush_dict(md)
        print(f"{pat:>14}  {md.n_pages_scanned:>5}  {len(md.r_groups):>6}  "
              f"{md.audit['n_options_text_total']:>8}  "
              f"{md.audit['n_options_smiles_resolved']:>8}  "
              f"{md.audit['n_options_unresolved']:>10}  "
              f"→ {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
