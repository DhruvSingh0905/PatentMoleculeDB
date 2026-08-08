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

The patent names its compounds in text, and those names carry the patent's own
numbering by construction — they sit in the patent's own prose, so nothing has
to be bridged afterwards. Measured by running THIS module, not by a probe:

    patent        distinct structures    names >=45 chars
    US8952177             238                  180
    US10214537          1,018                  887
    US10544143            222                  165

For scale: US10544143's Google Patents path produced 237 compounds. Reading the
patent's own text produces 222 of them, offline and at $0.

(An earlier version of this paragraph said "150 / 850 / 328". Those were counts
from a crude throwaway regex used to test whether names were present AT ALL,
written before this module existed, and they undercounted it. A docs pass
caught the contradiction by re-running the code rather than trusting the
comment — which is the only reason it was found.)

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

Naming compounds is only half the join, though: `extract_names` used to return
`.start`, a description-text offset, and nothing else — no link to the assay
rows, which are keyed `(patent_id, cid)` in the patent's OWN numbering ("1",
"43", "I-0020"). `NamedCompound.cid` closes that: see the block comment above
`_ANCHOR_BOUND` for the measurement (heading structure, not prose distance)
that made it possible, and wiki page 29 for the full writeup.

WHAT THIS IS NOT
----------------
It does not enumerate Markush series. Where a patent gives a drawn scaffold and
a table of R-groups, the R-groups are often text (`Me`, `Et`, `3-CH3`) but the
scaffold is an image, and no name can be composed without it. Those compounds
are simply absent from this module's output, and that absence is visible rather
than papered over.

REAGENT LABEL AND MARKUSH FLAG — TWO FIELDS, NOT ONE, AND WHY
---------------------------------------------------------------
Every `NamedCompound` this function returns now carries two independent
verdicts, wired in here rather than left for a caller to compute:

- `.label` / `.reason` — `sources/reagents.py::classify(name, smiles)`, called
  once per accepted structure. `.label` is one of that module's closed set
  (`"compound"`, `"reagent"`, `"trace_fragment"`); `.reason` is its
  machine-readable justification. This module never filters on the result —
  `reagents.py` labels, it does not delete, and this function keeps that
  contract: every structure OPSIN accepted still flows out, reagent or not.
- `.markush` / `.markush_reason` — set HERE, not by `reagents.py`, when the
  accepted name carries a relative-stereo descriptor (`R*`/`S*`/`E*`/`Z*`,
  e.g. `(1R*,2S*)-...`). Such a name does not identify one molecule; it
  states a relationship between two stereocentres ("opposite" or "same") with
  absolute configuration left open, so the resolved SMILES/InChIKey stands in
  for a SET of stereoisomers, not a single compound.

**Why a fourth field instead of a fourth value of `label`:** "is this a
reagent" and "is this structure generic" are orthogonal questions with
independent evidence. `reagents.classify` answers the first from a name
lexicon and a heavy-atom floor; the relative-stereo check here answers the
second from a character pattern in the accepted name — neither test can see
the other's answer, and a real name can be genuinely `("compound", markush)`
(a generic scaffold that is not a lab reagent) just as easily as
`("reagent", not markush)` (a single defined solvent molecule). Folding both
into one `label` string would force a choice between them for any compound
that is both, or invent a fifth combinatorial label for no reason grounded in
either module's evidence. It would also reach into `reagents.LABELS`, a
closed set owned by a module this task does not modify. Two independent
booleans-plus-reason pairs, mirroring the `(label, reason)` shape `reagents`
already uses, keep the two axes auditable separately and let a downstream
consumer filter on either without the other's noise.

`.markush` is `False` by default and only ever set `True` by the relative-
stereo check above — it is not a general "this name is generic" detector
(that would need Markush R-group table detection, which is out of scope; see
"WHAT THIS IS NOT" above) and it is not conflated with `.cid_clash` (a
proximity-anchoring ambiguity about WHICH compound number a name belongs to,
orthogonal to whether the name itself denotes one structure or a set).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from ..core import config
from .anchor import anchor_text as _anchor_text
from .anchor import find_cid as _find_cid
from .reagents import classify as _classify_reagent
from .uspto_xml import description_text

logger = logging.getLogger(__name__)

# Characters an IUPAC name is built from. Deliberately permissive — this only
# has to bound a SEED; OPSIN decides whether the span is a name.
#
# `*` is included so a relative-stereo descriptor — `(1R*,2S*)-2-{...}...` —
# seeds at all. Without it the seed regex breaks the run AT the asterisk
# (it is simply not a class member), splitting one name into fragments too
# short to clear `IUPAC_MIN_SEED` on either side, or into a fragment missing
# the stereo-descriptor prefix entirely — which still parses, but as the
# same flat structure some other seed already found, so the relative-stereo
# name itself was never extracted. OPSIN parses the correctly-bounded string
# fine (verified independently); this was purely a character-class gap. See
# "REAGENT LABEL AND MARKUSH FLAG" above for what happens to a name that
# contains one once it IS extracted.
_NAME_CHARS = r"A-Za-z0-9\[\]\(\)\{\}',\-\+\.′’*"
_SEED = re.compile(rf"[{_NAME_CHARS}]{{{config.IUPAC_MIN_SEED},400}}")

# A relative-stereo descriptor: an R/S/E/Z letter immediately followed by the
# asterisk that means "this centre's configuration is defined only RELATIVE
# to the others named, not absolutely" — `rel-(1R*,2S*)-...` reads "opposite
# configuration at 1 and 2, which enantiomer is unstated." The only legitimate
# role `*` plays in a string OPSIN accepts as a name is this descriptor, so
# its presence on an ACCEPTED name is what flags `.markush` below — never
# checked on rejected candidates or raw text, only on what OPSIN already
# confirmed parses.
_RELATIVE_STEREO = re.compile(r"[RSEZ]\*")

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
    cid: str | None = None      # the patent's OWN compound number, if anchored
    # Set only when `cid` is None because anchoring found a CLASH — occurrences
    # disagreeing about the id — rather than finding nothing. `"402|402B"`,
    # closest-evidence first; see `anchor.AnchorResult` for the full detail
    # this is flattened from. Never a guess: a human or a later pass reads it.
    cid_clash: str | None = None
    # `sources/reagents.py::classify(name, smiles)`, called once per structure
    # in `extract_names` (see "REAGENT LABEL AND MARKUSH FLAG" in the module
    # docstring). LABELS ONLY, NEVER FILTERS: every structure keeps flowing
    # through regardless of what these two fields say — an intermediate or
    # starting material next to the patent's real target is still useful with
    # no assay value, and this module has no evidence to decide which of those
    # a "reagent"-labeled row actually is. `label` defaults to `"compound"`,
    # `classify`'s own default when neither of its tiers fires.
    label: str = "compound"
    reason: str = ""
    # A DIFFERENT, ORTHOGONAL axis from `label`/`reason` — see the module
    # docstring for why this is a separate field rather than a fourth `label`
    # value. `True` when `name` carries a relative-stereo descriptor
    # (`R*`/`S*`/`E*`/`Z*`), meaning the resolved structure stands in for a
    # SET of stereoisomers rather than one molecule. Set in `extract_names`,
    # never by `reagents.classify`.
    markush: bool = False
    markush_reason: str = ""

    @property
    def key(self) -> str:
        return self.inchikey or self.smiles


# ---------------------------------------------------------------------------
# Compound-id anchoring — the logic itself now lives in `sources/anchor.py`,
# a pure module (two strings in, an `AnchorResult` out) kept separate so it
# is unit-testable without running OPSIN or this file's candidate generation
# at all. `_anchor_text` / `_find_cid` below are just the names this module
# calls them by; see `anchor.py`'s docstring for what was measured — the
# proximity rule, the alphanumeric-id extension and its ablation, and why a
# clash surfaces instead of silently refusing to anchor.


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
    `…cyclohexanecarboxylic` prefix that also parses. Each entry also carries
    `.cid`, the patent's own compound number, when one was found close enough
    to trust — `None` otherwise (never a guess).
    """
    # NO FEATURE FLAG HERE, DELIBERATELY. This function used to open with
    # `if not config.IUPAC_NAMES: return []`, duplicating the gate that already
    # sits at the only call site (`verify.dump`). Two gates for one switch is
    # not merely redundant — it made the function untestable in isolation:
    # calling it directly to check its output returned an empty list and looked
    # like a broken extractor rather than a disabled route. A pure function
    # should not consult a global, and whether to RUN this route is the
    # caller's decision, not this module's.
    # ONE CORPUS, headings included. Both the compound's name and its
    # `Example N` number are `<heading>` elements — the `<p>` beneath is the
    # synthesis procedure — so `<p>`-only text loses the identity twice over.
    # Measured over 137 patents against a `<p>`-only baseline: structures
    # 44,959 -> 54,833 (+22.0%) with ZERO patents regressing, anchor rate
    # 37.5% -> 40.8%, precision against the hand-checked reference
    # 89.1% -> 92.2%.
    #
    # A two-corpus alternative was built and benchmarked on the same harness
    # (extract from `<p>` and `<heading>` separately, merge on InChIKey). It
    # tied on coverage — 6 structures apart corpus-wide — and lost on
    # everything else: 532 fewer anchors and 41% slower. The reason is
    # positional: a heading-sourced name sits beside its own number, while the
    # same name restated mid-prose usually does not.
    text = description_text(xml, include_headings=True)
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
        # Reagent/trace-fragment LABEL — never a filter, see the module
        # docstring. Computed here, once per distinct structure, so every
        # caller of this function gets it for free rather than re-deriving it.
        verdict = _classify_reagent(name, smi)
        # Relative-stereo MARKUSH flag — orthogonal to the label above.
        stereo = _RELATIVE_STEREO.findall(name)
        out.append(NamedCompound(
            patent_id=patent_id, name=name, smiles=smi, inchikey=ik, start=pos,
            label=verdict.label, reason=verdict.reason,
            markush=bool(stereo),
            markush_reason=("relative_stereo:" + ",".join(stereo)) if stereo else ""))

    # 5. anchor each structure to the patent's OWN compound number — see
    #    `anchor.py`'s module docstring for what was measured and why.
    # THE SAME STRING `text` — not a second flattening. `anchor_text(xml)` is
    # byte-identical to `description_text(xml, include_headings=True)` (verified
    # on the corpus), so reusing `text` removes one full pass over every
    # document AND makes the offset spaces identical by construction rather
    # than by convention. The "offsets are not interchangeable" hazard that
    # `anchor_text`'s docstring warns about cannot arise when there is one
    # string.
    anchor_src = text
    if anchor_src:
        for nc in out:
            result = _find_cid(anchor_src, nc.name)
            nc.cid = result.cid
            if result.clashed:
                nc.cid_clash = "|".join(c.cid for c in result.candidates)
    anchored = sum(1 for nc in out if nc.cid)
    clashed = sum(1 for nc in out if nc.cid_clash)
    reagents_n = sum(1 for nc in out if nc.label == "reagent")
    trace_n = sum(1 for nc in out if nc.label == "trace_fragment")
    markush_n = sum(1 for nc in out if nc.markush)
    logger.info("iupac: %s — %d candidates parsed, %d distinct structures "
                "(%d reagent, %d trace_fragment, %d markush), "
                "%d anchored to a compound id, %d clashed",
                patent_id, len(kept), len(out), reagents_n, trace_n, markush_n,
                anchored, clashed)
    return out
