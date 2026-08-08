"""Compound names out of the patent's own text, with OPSIN as the acceptance gate.

WHY THIS EXISTS
---------------
v3 shipped assay values keyed by the patent's compound id and no chemistry at
all — no name, no SMILES, no InChIKey. That was a documented consequence of the
move's boundary, not an accident, but it left the identity half of the product
missing.

The obvious way to fill it is Google Patents, which embeds
`<span itemprop="smiles">` pairs it derived by running structure recognition
over the patent's drawings. Measured on v2's 22 shipped patents, that is where
identity actually came from: 97.2% of 17,459 compounds arrived through a GP
path, 78.5% of names were looked up from PubChem rather than read from the
patent, and 27.6% were still keyed `GP1089` — GP's positional numbering, never
joined to the patent's own. Depending on it means depending on a third party's
OCR of an image we also have, and getting back a compound id that does not
match our assay rows.

The patent names its compounds in text. US8952177's description yields 150
OPSIN-parseable names at >=45 characters; US10214537 yields 850, US10376513
328. Those names carry the patent's own numbering by construction, because they
sit in the patent's own prose — so nothing has to be bridged afterwards.

THE HARD PART IS NOT NOMENCLATURE, IT IS BOUNDARIES
---------------------------------------------------
OPSIN already is a morpheme parser with a full chemical grammar, so there is no
vocabulary to build. What it cannot do is tell us where a name STARTS and STOPS
inside running prose. v2 classified why OPSIN failed on 182 real names:

    48.4%  unbalanced brackets
    15.4%  truncated mid-name
    25.3%  synthesis prose bleeding in
    11.0%  "well-formed" (header contamination, mojibake)

63.8% of failures are boundary errors, not names OPSIN could not read. So the
strategy is brute force: from each seed position generate many candidate spans —
trimmed left, extended right, bracket-balanced — and let OPSIN accept or reject
each one. A wrong candidate simply fails to parse, which costs nothing; the
parser is the detector, and nothing here needs to be clever about chemistry.

That is cheap because OPSIN batches: one subprocess call resolves thousands of
candidates at once.

WHAT THIS IS NOT
----------------
It does not enumerate Markush series. Where a patent gives a drawn scaffold and
a table of R-groups, the R-groups are often text (`Me`, `Et`, `3-CH3`) but the
scaffold is an image, and no name can be composed without it. Those compounds
are simply absent from this module's output, and that absence is visible rather
than papered over.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from ..core import config
from .uspto_xml import description_text

logger = logging.getLogger(__name__)

# Characters an IUPAC name is built from. Deliberately permissive — this only
# has to bound a SEED; OPSIN decides whether the span is a name.
_NAME_CHARS = r"A-Za-z0-9\[\]\(\)\{\}',\-\+\.′’"
_SEED = re.compile(rf"[{_NAME_CHARS}]{{{config.IUPAC_MIN_SEED},400}}")

# Words that legitimately continue a name after a space. A chemical name is
# mostly one token, but not always: `…carboxylic acid`, `…hydrochloride salt`,
# `…, trifluoroacetic acid salt`. Extending across arbitrary words would sweep
# in prose, so the tail is a closed list.
_TAIL_WORDS = ("acid", "salt", "hydrochloride", "hydrobromide", "ester",
               "amide", "anhydride", "hydrate", "oxide", "ketone", "ether")

# Prose that attaches to a name and stops it parsing. Stripped from the LEFT
# only, one token at a time, because the right-hand end is where the parent
# hydride lives and trimming it changes which molecule is meant.
_LEAD_JUNK = re.compile(
    r"^(?:title|the|a|an|of|to|and|from|with|using|gave|afforded|yielded|"
    r"compound|example|step|intermediate|product|crude|desired|racemic|"
    r"pure|above|obtained|prepared|synthesis|preparation)[\s\-]+", re.I)

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def _balance_right(s: str) -> str | None:
    """Trim from the right until brackets balance. The 48.4% failure mode.

    `…benzimidazol-2-yl}cyclohexanecarboxylic acid)` and
    `triazin-7-yl)benzyl)morpholin-3-one` are both real OPSIN rejections from
    this corpus — one closing bracket too many, swept in from the surrounding
    prose. Cutting from the right is safe here in a way it is not in general:
    an unmatched CLOSER means the span started inside a bracket group, so the
    text beyond it belongs to a different name.
    """
    depth = 0
    for i, ch in enumerate(s):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
            if depth < 0:
                return s[:i] or None
    return s if depth == 0 else None


def _balance_left(s: str) -> str | None:
    """Drop a leading fragment when the span opened mid-bracket-group."""
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        ch = s[i]
        if ch in _CLOSE:
            depth += 1
        elif ch in _OPEN:
            depth -= 1
            if depth < 0:
                return s[i + 1:] or None
    return s if depth == 0 else None


def _variants(text: str, start: int, end: int) -> list[str]:
    """Every span worth asking OPSIN about, for one seed. Brute force, bounded.

    Bounded by `IUPAC_MAX_VARIANTS` because a seed near a bracket-heavy stretch
    can otherwise generate dozens, and the batch is what makes this cheap. The
    order matters only in that the caller keeps the LONGEST accepted one.
    """
    raw = text[start:end]
    out: list[str] = []

    def add(s: str | None) -> None:
        if not s:
            return
        s = s.strip(" \t .,;:")
        if len(s) >= config.IUPAC_MIN_SEED and s not in out:
            out.append(s)

    add(raw)
    # ...with a closed-list tail word pulled in across the space
    tail = re.match(rf"\s+({'|'.join(_TAIL_WORDS)})\b", text[end:end + 24], re.I)
    if tail:
        add(raw + text[end:end + tail.end()])
    # ...bracket-balanced both ways
    add(_balance_right(raw))
    add(_balance_left(raw))
    rb = _balance_right(raw)
    if rb:
        add(_balance_left(rb))
    # ...with leading prose peeled off, repeatedly
    cur = raw
    for _ in range(3):
        stripped = _LEAD_JUNK.sub("", cur)
        if stripped == cur:
            break
        cur = stripped
        add(cur)
        add(_balance_right(cur))
    # ...and from each internal boundary, for a name welded onto a preceding
    # word with no space (the `## Example N` / header-contamination shape)
    for m in list(re.finditer(r"[-\)\]\}]", raw))[:4]:
        add(_balance_right(raw[m.end():]))
    return out[: config.IUPAC_MAX_VARIANTS]


@dataclass
class NamedCompound:
    """A name the patent states, resolved to a structure by OPSIN."""
    patent_id: str
    name: str
    smiles: str
    inchikey: str
    start: int                  # offset into the description
    source: str = "description"

    @property
    def key(self) -> str:
        return self.inchikey or self.smiles


def _opsin(names: list[str], fmt: str) -> list[str]:
    """Batch OPSIN. One subprocess for the whole list.

    `tmp_fpath` is pid-scoped: py2opsin writes its input to a shared temp file
    whose default name is a constant, so two processes running concurrently
    overwrite each other's input and silently get each other's answers.
    """
    if not names:
        return []
    from py2opsin import py2opsin

    tmp = str(config.OUTPUT_DIR / f".opsin_in_{os.getpid()}.txt")
    try:
        out = py2opsin(names, output_format=fmt, tmp_fpath=tmp)
    except Exception as e:                       # OPSIN is a java subprocess
        logger.warning("opsin: batch of %d failed: %r", len(names), e)
        return [""] * len(names)
    if isinstance(out, str):
        out = [out]
    return list(out) + [""] * (len(names) - len(out))


def extract_names(xml: str, patent_id: str = "") -> list[NamedCompound]:
    """Every compound name in this patent's description that OPSIN can read.

    Returns one entry per DISTINCT structure (deduped on InChIKey), carrying
    the longest name that produced it. Overlapping spans are resolved
    longest-first, so `…cyclohexanecarboxylic acid` wins over the
    `…cyclohexanecarboxylic` prefix that also parses.
    """
    if not config.IUPAC_NAMES:
        return []
    text = description_text(xml)
    if not text:
        return []

    # 1. seeds — every run of name-legal characters that could start a name
    seeds = [(m.start(), m.end()) for m in _SEED.finditer(text)
             if re.search(r"[a-z]{3}", m.group(0))
             and re.search(r"[\d\[\(\-]", m.group(0))]

    # 2. brute force: fan every seed out into candidate spans
    cands: list[tuple[int, str]] = []
    for s, e in seeds:
        for v in _variants(text, s, e):
            cands.append((s, v))
    if not cands:
        return []
    logger.info("iupac: %s — %d seeds -> %d candidate spans",
                patent_id, len(seeds), len(cands))

    # 3. OPSIN is the acceptance gate. One batch for SMILES, one for InChIKey
    #    over only what survived — the second batch is small.
    strings = [c for _, c in cands]
    smiles = _opsin(strings, "SMILES")
    kept = [(pos, s, smi) for (pos, s), smi in zip(cands, smiles) if smi]
    if not kept:
        return []
    keys = _opsin([s for _, s, _ in kept], "StdInChIKey")

    # 4. longest accepted span per seed position, then dedup by structure
    best: dict[int, tuple[str, str, str]] = {}
    for (pos, name, smi), ik in zip(kept, keys):
        cur = best.get(pos)
        if cur is None or len(name) > len(cur[0]):
            best[pos] = (name, smi, ik)

    out: list[NamedCompound] = []
    seen: set[str] = set()
    for pos in sorted(best):
        name, smi, ik = best[pos]
        k = ik or smi
        if k in seen:
            continue
        seen.add(k)
        out.append(NamedCompound(patent_id=patent_id, name=name, smiles=smi,
                                 inchikey=ik, start=pos))
    logger.info("iupac: %s — %d candidates parsed, %d distinct structures",
                patent_id, len(kept), len(out))
    return out
