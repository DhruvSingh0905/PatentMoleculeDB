"""Read the compound NAME off the picture, when the picture carries one.

WHY THIS RUNS BEFORE ANY STRUCTURE RECOGNITION
------------------------------------------------
Some patents render the compound's IUPAC name inside the image, beside the
structure. When they do, OCR plus OPSIN gives an EXACT answer with no
structure recognition involved — and, unlike `DECIMER.predict_SMILES`, the
answer is self-checking: it passes the same OPSIN gate every text-derived
structure in this package passes. `predict_SMILES` returns a bare SMILES with
no confidence, so a recognised structure can only be checked against an
InChIKey we already have, which by definition does not exist for the compounds
that most need it.

So the image route is two stages, cheapest first:

    image -> OCR -> OPSIN parses?  yes -> resolved, source="image_ocr"
                                    no -> DECIMER, source="image"

The two sources are kept distinct on purpose. If a cid join turns out wrong,
the artifact says which compounds came through which reader.

WHAT THE OCR TEXT LOOKS LIKE, MEASURED
----------------------------------------
Over 114 drawn images sampled from the shipped dump: text length runs 0 to 185
characters, median 15. That median is the giveaway — most images return only
ATOM LABELS (`CH3`, `NH2`, `OH`, `N`), which are not names. 33 images (29%)
returned 40+ characters, and every one of those came from two patents. A
caption is a per-document convention, like every other pattern in this corpus.

Of those 33, THREE parsed as they came. Reading the other 30 showed defects
that are all mechanical — a render artifact, never a chemistry error:

  LEADING ATOM LABELS. OCR reads a label off the structure and prepends it to
  the caption — `F (E)-3-(2-...`, `OH OH 3-((7S)-...`, `HO- (E)-7-(2-...`,
  `OH C Cl 4-(5-(4-...`. The name starts after them.

  A TIGHT CONFUSION SET, from a bilevel render:
      `]` -> `J`      bicyclo[2.2.2Joctan
      `)` -> `J`/`_`  methoxy Jbicyclo, methoxy _ bicyclo
      `1` -> `l`      octan-l-yl
      `5` -> `S`      isoxazol-S-ylmethyl

  BRACKETS AND HYPHENS LOST AT A LINE WRAP. `E)2-` for `(E)-2-`, `4(2-` for
  `4-(2-`, and one `)` printed on both sides of the break (`yloxy } Jethyl`).

  A SPACE WHERE A CHARACTER SHOULD BE. `pyridin-2- y loxy` split one word;
  `pyridin-2-yl methyl` lost a `)`. Neither is reachable by looking at the
  characters alone, because `carboxylic acid` is the same shape — so each such
  space becomes a site with three readings and the parser picks.

Repairing those took 3 -> 28 of 33 (85%). Every substitution above was read
off a failure, not guessed, and OPSIN still decides: a variant is generated
and kept only if it parses. Nothing is accepted on a transformation alone.

THE OTHER 5, AND WHY THEY STAY UNRESOLVED
------------------------------------------
All five are one compound family in one patent, and all five fail the same
way: `(1S,4R)-4-(...)cyclohexane-1-carboxylic acid`, whose locants OPSIN
cannot place. The repair is not what is missing — for one of them this module
produces the exactly-correct string, checked against a hand-corrected name at
similarity 1.000, and OPSIN refuses it anyway.

OPSIN will return a skeleton for them under `--allowUninterpretableStereo`.
That skeleton stands for a SET of stereoisomers, so shipping it as a resolved
compound would claim more than the picture supports. They are recorded instead
— see `_mark_unread` — under the same reason code `repair/name_gap.py` writes
for the identical failure, so one query over the loss log finds both
populations.

WHAT THIS IS NOT
-----------------
Not a measurement of the corpus. 28 of 114 images is 25% of one sample drawn
from two captioning patents. The stage is worth running because it is local,
free, and exact where it fires — not because that number generalises.
"""
from __future__ import annotations

import itertools
import logging
import re
from pathlib import Path

from . import losses as _losses
from .dewrap import targeted as _dewrap
from .opsin import REASON_STEREO as _REASON_STEREO
from .opsin import REASON_TEXT as _REASON_TEXT
from .opsin import batch as _opsin
from .opsin import errors as _errors
from .opsin import stereo_blocked as _stereo_blocked

logger = logging.getLogger(__name__)

SOURCE = "image_ocr"

# Shorter than this and the text is an atom label, not a name. Median OCR
# length over the sample was 15 characters; every parseable name was 100+.
MIN_NAME_CHARS = 40

# ONE atom label, as a WHOLE TOKEN. Token-wise, not a character run: a
# character class containing `(`, `)` and `-` eats `(E)-` off the front of
# `F (E)-3-(2-...`, and `(E)` is a stereo descriptor that belongs to the name.
# Splitting on whitespace and dropping leading label TOKENS cannot do that,
# because `(E)-3-(2-...` is one token and does not match.
_LABEL_TOKEN = re.compile(
    r"^(?:HO-?|OH|NH\d?|CH\d?|H\d?C|Cl|Br|[A-Z][a-z]?\d?|[-–—])$")

# A name opens with a locant, a bracket, or a lowercase word.
_NAME_START = re.compile(r"[(\[\d]|[a-z]")


def _strip_labels(text: str) -> str:
    """Drop the atom-label tokens that precede the name.

    Returns `text` unchanged when nothing is left, or when what remains does
    not open like a name — a wrong strip is worse than no strip, because the
    result still parses often enough to be adopted.
    """
    toks = text.split()
    i = 0
    while i < len(toks) and _LABEL_TOKEN.match(toks[i]):
        i += 1
    out = " ".join(toks[i:]).strip()
    return out if out and _NAME_START.match(out) else text


# EACH AMBIGUOUS CHARACTER IS ITS OWN SITE, and one name routinely needs two
# DIFFERENT answers. Measured on a real caption:
#
#     ...bicyclo[2.2.2Joctan-1-yl)vinylJimidazo[1,2-alpyridine-3-carboxylic acid
#                     ^ J is ']'          ^ J is ')'
#
# A global `replace("J", "]")` produces one reading and a global
# `replace("J", ")")` produces another; NEITHER is the name. The first version
# of this generator did exactly that, so a caption with two J's could not be
# reached however many candidates it emitted.
#
# So the sites are enumerated and resolved independently. `]` closes a `[`,
# `)` closes a `(` — which of the two is right depends on what is open at that
# point, and rather than track bracket depth (the OCR is too noisy to trust a
# count) both are tried and OPSIN decides.
_AMBIGUOUS = {
    "J": ("]", ")"),        # bilevel render: ] and ) both degrade to J
    "_": (")", "-"),        # `...quinolin-3-yl)` and `3,7,8,9_ tetrahydro`
}
# 2^n candidates for n sites, so this is the ceiling on n. Four covers every
# caption observed; past that the OCR is too damaged to be worth the calls.
_MAX_SITES = 4


def _resolve_sites(text: str):
    """Every combination of per-site resolutions, plainest first."""
    sites = [(i, _AMBIGUOUS[ch]) for i, ch in enumerate(text)
             if ch in _AMBIGUOUS]
    if not sites:
        yield text
        return
    if len(sites) > _MAX_SITES:
        # too damaged to enumerate; offer the two global readings and move on
        yield text
        for ch, opts in _AMBIGUOUS.items():
            for o in opts:
                yield text.replace(ch, o)
        return
    yield text
    for combo in itertools.product(*(opts for _, opts in sites)):
        out = list(text)
        for (i, _), rep in zip(sites, combo):
            out[i] = rep
        yield "".join(out)


# Letter-for-digit confusions, each gated on where a LOCANT can appear so they
# fire only there. Read off real captions: `IH-pyrazol` for `1H-`, `(IR,3R)`
# for `(1R,3R)`, `tetrahydro-ZH-pyran` for `-2H-`, `piperidin-l-yl` for
# `-1-yl`, `isoxazol-S-yl` for `-5-yl`.
_LOCANT_FIXES = (
    (re.compile(r"\bI(?=[HR]\b|[HR][-,)])"), "1"),      # IH-, (IR,
    (re.compile(r"(?<=[-(\[,])l(?=H\b|H[-,)])"), "1"),   # -lH-
    (re.compile(r"\bZ(?=H\b|H[-,)])"), "2"),            # ZH-pyran
    (re.compile(r"-l-"), "-1-"),
    (re.compile(r"(?<=[\d.])l(?=[-)\]])"), "1"),
    (re.compile(r"(?<=-)l(?=-?[a-z])"), "1"),            # -l-yl
    (re.compile(r"(?<=-)S(?=-)"), "5"),
)

# A HYPHEN THAT DID NOT SURVIVE THE RENDER. Both directions observed:
#
#     `(1S,4R)4-`  for `(1S,4R)-4-`     a descriptor, then its locant
#     `((S)2-`     for `((S)-2-`
#     `-(4(2-`     for `-(4-(2-`        a locant, then the group it opens
#
# The second shape is the mirror of the first and cost a whole caption on its
# own: `E)2-(4(2-(4-((5-cyclopropyl...` carried BOTH, so restoring only the
# one after `)` still left a name OPSIN would not read. A locant digit is
# never written flush against an opening bracket in a name OPSIN accepts, and
# either way this is a candidate — the parser still decides.
_MISSING_HYPHEN = (
    re.compile(r"(\)[\s]?)(?=\d)"),      # `)2-`  -> `)-2-`
    re.compile(r"(?<=\d)(?=\()"),        # `4(2-` -> `4-(2-`
)


def _hyphens(text: str) -> str:
    for pat in _MISSING_HYPHEN:
        text = pat.sub(lambda m: (m.group(0) or "") + "-", text)
    return text


def _confusions(text: str):
    """The shape confusions a bilevel render produces. Each is a CANDIDATE.

    The fixes COMPOSE — one caption routinely carries several at once
    (`OH 3-((7S)-...imidazo[4,5- fJquinolin-3-yl)...acid 1st eluting isomer`
    has a leading label, a `J`, and a trailing annotation). So each base from
    `_resolve_sites` is offered raw, with locant fixes applied, and with the
    missing hyphen restored on top of those.
    """
    for base in _resolve_sites(text):
        yield base
        fixed = base
        for pat, rep in _LOCANT_FIXES:
            fixed = pat.sub(rep, fixed)
        if fixed != base:
            yield fixed
        for v in (base, fixed):
            hy = _hyphens(v)
            if hy != v:
                yield hy
        # a fused-ring locant: `imidazo[1,2-a]pyridine` -> `[1,2-alpyridine`
        for v in (base, fixed):
            fu = re.sub(r"\[(\d[^\]\[]{0,6}-[a-z])l(?=[a-z])", r"[\1]", v)
            if fu != v:
                yield fu


# A NAME THAT OPENS WITH AN ORPHAN `)` LOST ITS OPENING. Observed:
#
#     )-7-(2-(4-((5-cyclopropyl-...)-1-morpholinoisoquinoline-3-carboxylic acid
#
# The patent wrote `(E)-7-...` and OCR dropped `(E`. The stereo descriptor is
# not recoverable from the picture text, so each common opening is offered and
# OPSIN decides; dropping the orphan entirely is offered too, since a name
# without its descriptor still names the right skeleton.
_ORPHAN_OPEN = ("(E", "(Z", "(R", "(S")


# A CHEMICAL NAME ENDS AT A SUFFIX. What follows is the patent annotating the
# row, and OCR reads it as part of the caption:
#
#     ...cyclohexane-1-carboxylic acid 1st eluting isomer
#     ...quinoline-6-carboxylate 3rd eluting isomer
#     ...cyclohexane-1-carboxylic acid Nums          <- a mangled structure label
#
# Cutting at the LAST name-ending word keeps the whole name and drops the note.
# It cannot truncate a real name, because the cut is placed AFTER the suffix.
_NAME_END = re.compile(
    r"\b(?:acids?|carboxylates?|carboxamides?|carbonitriles?|amides?|amines?|"
    r"esters?|nitriles?|oxides?|ketones?|ethers?|ols?|ones?|ates?|ides?|"
    r"ines?|yls?|dione|trione)\b", re.I)


def _trim_annotation(text: str):
    """`text`, and `text` cut at its last name-ending word."""
    yield text
    last = None
    for m in _NAME_END.finditer(text):
        last = m
    if last and last.end() < len(text):
        cut = text[:last.end()].strip()
        if len(cut) >= MIN_NAME_CHARS:
            yield cut


# THE SAME LOSS, ONE CHARACTER LATER. `_ORPHAN_OPEN` above covers the case
# where OCR dropped `(E` and left a bare `)`. It also drops just the `(` and
# leaves the descriptor standing:
#
#     E)2-(4(2-(4-((5-cyclopropyl-...)vinyl)phenyl)-N,N-dimethylacetamide
#     ^ the patent wrote `(E)-2-...`
#
# The test is structural, not a list of descriptors: a `)` that appears before
# any `(` closes nothing, so its opening was lost. Bounded to the first few
# characters because that is where a stereo descriptor lives — an unmatched
# `)` deep inside a name is a different defect with a different repair, and
# putting a `(` on the front would not fix it.
_LOST_OPEN = re.compile(r"^[^()]{1,12}\)")


def _repair_opening(text: str):
    yield text
    if text.lstrip().startswith(")"):
        rest = text.lstrip()[1:].lstrip("-").lstrip()
        for op in _ORPHAN_OPEN:
            yield f"{op})-{rest}"
        yield rest
    elif _LOST_OPEN.match(text.lstrip()):
        yield "(" + text.lstrip()


# A CLOSING BRACKET THE RENDER PRINTED TWICE. Observed once, and it cost a
# whole caption:
#
#     ...(1-(pyridin-2- y loxy } Jethyl)-6,7,8,9-tetrahydro-...
#                              ^^^ the patent wrote one `)`
#
# Two closers separated by a space, where the name needs one. The space is the
# tell: it is where the printed line wrapped, and the bracket was read on both
# sides of the break. Dropping EITHER member is offered — which one is a real
# character and which is the echo cannot be told from the text — and the
# untouched reading is offered first, because `) )` across a wrap is also what
# a genuine `))` looks like.
_DOUBLED_CLOSE = re.compile(r"[)\]}J_]\s+[)\]}J_]")


def _doubled_close(text: str):
    yield text
    for m in _DOUBLED_CLOSE.finditer(text):
        a, b = m.start(), m.end()
        # The whitespace between the pair goes with whichever member is
        # dropped. It is the wrap point itself and means nothing either way,
        # and leaving it behind produces a double space no other rule expects.
        yield text[:a] + text[b - 1:]          # drop the first member
        yield text[:a + 1] + text[b:]          # drop the second


# A SPACE BETWEEN TWO LETTERS IS THE ONE DEFECT `dewrap` REFUSES TO TOUCH,
# and it has to refuse: `...cyclohexane-1-carboxylic acid` ends in a genuine
# word break with a letter on each side, and `dewrap.targeted` exists so that
# space survives. But OCR produces the same shape from two different accidents:
#
#     pyridin-2- y loxy        the render split one word
#     pyridin-2-yl methyl      a `)` came back as whitespace
#
# Neither is reachable by a rule that only looks at the characters, because
# `carboxylic acid` looks identical to both. So each such space becomes a SITE
# with three readings — leave it, close it, or restore the lost `)` — and they
# are enumerated together, exactly as `_resolve_sites` handles `J`. The
# untouched reading is emitted first, so a caption whose only word space is
# `carboxylic acid` still yields the plain name as its first candidate.
_WORD_SPACE = re.compile(r"(?<=[A-Za-z])\s+(?=[A-Za-z])")
_SPACE_READINGS = (None, "", ")")     # keep / close / a lost closing bracket
# 3^n candidates, and every one is a name OPSIN has to be asked about. Three
# sites is 27, which is the most a caption has ever needed; past that the text
# is damaged beyond what a space rule can reach and the base is offered alone.
_MAX_SPACE_SITES = 3


def _resolve_spaces(text: str):
    """`text`, then every per-site reading of its letter-to-letter spaces."""
    sites = [m.span() for m in _WORD_SPACE.finditer(text)]
    yield text
    if not sites or len(sites) > _MAX_SPACE_SITES:
        return
    for combo in itertools.product(_SPACE_READINGS, repeat=len(sites)):
        if all(c is None for c in combo):
            continue                                   # already yielded
        out, prev = [], 0
        for (a, b), rep in zip(sites, combo):
            out.append(text[prev:a])
            out.append(text[a:b] if rep is None else rep)
            prev = b
        out.append(text[prev:])
        yield "".join(out)


def name_candidates(text: str) -> list[str]:
    """Every reading of one OCR result worth asking OPSIN about.

    Ordered least-invasive first, so the plainest reading that parses wins.
    Deduplicated, because the transformations overlap and OPSIN is a java
    subprocess whose cost is per candidate.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) < MIN_NAME_CHARS:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for base in (text, _strip_labels(text)):
        for trimmed in _trim_annotation(base):
            for opened in _repair_opening(trimmed):
                for once in _doubled_close(opened):
                    for spaced in _resolve_spaces(once):
                        for c in _confusions(spaced):
                            # `_dewrap` ONLY, never a blanket whitespace strip.
                            # A caption ends `...-3-carboxylic acid`, and that
                            # space is part of the name;
                            # `re.sub(r"\s+", "", ...)` welds it into
                            # `carboxylicacid` and OPSIN refuses it.
                            # `dewrap.targeted` closes `octan- 1-yl` and
                            # `methoxy )bicyclo` because the space sits
                            # against punctuation, and leaves a genuine word
                            # break alone — which is what `_resolve_spaces`
                            # above is for.
                            for form in (c, _dewrap(c)):
                                if form and form not in seen:
                                    seen.add(form)
                                    out.append(form)
    return out


def read_text(paths: list[Path]) -> dict[Path, str]:
    """OCR each image. `{path -> text}`; a failure yields "" and is logged.

    THE IMAGES ARE 1-BIT BILEVEL. easyocr's own loader hands OpenCV a bool
    array and `cvtColor` refuses it, so the image is converted here and the
    ARRAY is passed rather than the path. Getting this wrong reports every
    image as carrying no text, which reads exactly like a finding — it cost a
    measurement that said "0 of 24 images have a caption" before the error was
    visible.
    """
    try:
        import easyocr
        import numpy as np
        from PIL import Image
    except ImportError as e:
        # THE SAME SILENCE `mass_gate` REFUSES. Without easyocr every image
        # returns "" and the run reads as "no picture carried a caption",
        # which is a finding rather than a missing dependency. It already cost
        # one measurement that way (see `read_text`'s note on bilevel images),
        # so the absence is recorded, not only logged.
        logger.warning("image_ocr: OCR unavailable (%r); %d image(s) NOT read",
                       e, len(paths))
        _losses.record("image_ocr_unavailable", "",
                       reason=repr(e)[:120], images=len(paths))
        return {p: "" for p in paths}

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    out: dict[Path, str] = {}
    for p in paths:
        try:
            arr = np.array(Image.open(p).convert("RGB"))
            parts = reader.readtext(arr, detail=0, paragraph=True)
            out[p] = re.sub(r"\s+", " ", " ".join(parts)).strip()
        except Exception as e:
            logger.warning("image_ocr: %s — %r", p.name, e)
            out[p] = ""
    return out


def supersede(rows: list, patent_id: str) -> tuple[list, int]:
    """Replace each drawn marker whose picture carries a readable name.

    `rows` is the merged structure list for one patent, exactly as
    `verify.dump()` assembles it. Returns `(rows, n_superseded)`.

    SUPERSESSION, NOT UNION — AND THIS IS THE WHOLE POINT OF THE FUNCTION.
    `cid_first` already emits a row for a drawn compound: the cid, the
    `<chemistry>` id and the image filename, with `name`/`smiles`/`inchikey`
    deliberately EMPTY, because that row asserts where the structure is rather
    than what it is. A resolved caption is an answer for THAT SAME cid. Adding
    it would leave two rows under one compound number, one of them blank, and
    every downstream count would be reading a compound twice — the identical
    shape that once pinned the repair loop's yield at exactly 0.500 (see
    `repair/loop.py`). `verify.dump()` already answers this question the same
    way where the table and cid_first routes overlap: precedence, not union.

    WHAT IS NEVER OVERWRITTEN. Only a row with no InChIKey is replaced. A
    compound the text track resolved keeps its text answer, because that
    answer was reached through the assertion machinery this package trusts and
    a caption has not earned the right to overrule it. Where the two disagree
    the artifact must SAY so rather than pick; that disagreement is what the
    mass gate is for, and it runs after this.

    THE PICTURE POINTER SURVIVES. `resolve()` carries `drawn_ref` and
    `drawn_file` onto the row it returns, so a compound resolved this way is
    still traceable to the image it came from, and `source="image_ocr"` tells
    it apart from a DECIMER read if a cid join ever turns out wrong.

    Costs nothing when there is nothing to read: no local image means no work
    item, and easyocr is never imported.
    """
    from ..core import config
    from . import images as _images

    if not config.IMAGE_OCR:
        return rows, 0
    drawn = [r for r in rows
             if getattr(r, "cid", None)
             and getattr(r, "drawn_file", "")
             and not getattr(r, "inchikey", "")]
    if not drawn:
        return rows, 0
    items = [_images.WorkItem(patent_id=patent_id, cid=r.cid, job="RECOVER",
                              chemistry_id=r.drawn_ref, image_file=r.drawn_file)
             for r in drawn]
    items = [it for it in items if it.local_path.exists()]
    if not items:
        logger.info("image_ocr: %s — %d drawn compound(s), no local image; "
                    "nothing read", patent_id, len(drawn))
        return rows, 0

    got = resolve(items, patent_id)
    if not got:
        return rows, 0
    by_cid = {g.cid: g for g in got if g.cid}

    out, n = [], 0
    for r in rows:
        hit = by_cid.get(getattr(r, "cid", None))
        # The `not r.inchikey` test is repeated here on purpose. `drawn` was
        # built before `resolve()` ran, and one cid can hold more than one row
        # in `rows` — the marker AND a resolved sibling. Only the blank one
        # gives way.
        if hit is not None and getattr(r, "drawn_file", "") and not getattr(
                r, "inchikey", ""):
            out.append(hit)
            n += 1
        else:
            out.append(r)
    logger.info("image_ocr: %s — %d drawn marker(s) superseded by a caption",
                patent_id, n)
    return out, n


def _mark_unread(have: list, plan: list, best: dict, patent_id: str) -> None:
    """Record every image that CARRIED a name and still produced nothing.

    WHY THIS IS NOT A LOG LINE. An image with no caption at all is not a loss
    here — it is DECIMER's input, and the queue already knows about it. An
    image whose caption we read, repaired every way we know, and still could
    not parse is a different thing entirely, and the two were previously the
    same silence.

    The reason code is the same one `repair/name_gap.py` writes, from the same
    `opsin.stereo_blocked` measurement, because it is the same question: is
    OPSIN refusing a grammar it does not have, or is the text broken? Measured
    over 33 captioned images from two patents, after repair:

        parsed                                   28
        `stereo_unmappable`                       5   every one `(1S,4R)-4-…`
                                                      on a cyclohexane
        `text_defect`                             0

    All five are one compound family in one patent. They are left UNRESOLVED
    on purpose: OPSIN will return a skeleton for them under
    `--allowUninterpretableStereo`, but that skeleton stands for a set of
    stereoisomers, and shipping it as a resolved compound would claim more
    than the picture supports.
    """
    unread = [i for i in range(len(have)) if i not in best]
    if not unread:
        return
    by_img: dict[int, list[str]] = {}
    for i, cand in plan:
        by_img.setdefault(i, []).append(cand)

    # ONE subprocess for the stereo test and ONE for the diagnosis, over the
    # whole patent — not one pair per image. A caption generates a few hundred
    # candidates and there can be dozens of captions.
    cands = sorted({c for i in unread for c in by_img.get(i, [])})
    blocked = _stereo_blocked(cands, {}, patent_id) if cands else set()
    plainest = {i: by_img[i][0] for i in unread if by_img.get(i)}
    errs = _errors(sorted(set(plainest.values())), patent_id)

    for i in unread:
        name = plainest.get(i)
        if name is None:
            continue                  # no caption on this image; not a loss
        it = have[i]
        hit = [c for c in by_img[i] if c in blocked]
        _losses.record(
            "image_caption_unparsed", it.patent_id, cid=it.cid,
            drawn_ref=it.chemistry_id, drawn_file=it.image_file,
            name=name, candidates=len(by_img[i]),
            reason=(_REASON_STEREO if hit else _REASON_TEXT),
            opsin_error=errs.get(name, ""))
    logger.info("image_ocr: %s — %d captioned image(s) unresolved and recorded",
                patent_id, len(plainest))


def resolve(items: list, patent_id: str = "") -> list:
    """`NamedCompound` rows for every image whose caption OPSIN accepts.

    `items` are `images.WorkItem`s whose `local_path` exists. Returns rows with
    `source="image_ocr"` and the image reference carried, so a compound
    resolved this way can always be traced back to the picture it came from —
    and told apart from one DECIMER read.
    """
    from .iupac_names import NamedCompound

    have = [it for it in items if it.local_path.exists()]
    if not have:
        return []
    texts = read_text([it.local_path for it in have])

    plan: list[tuple[int, str]] = []
    for i, it in enumerate(have):
        for cand in name_candidates(texts.get(it.local_path, "")):
            plan.append((i, cand))
    if not plan:
        return []

    uniq = sorted({c for _, c in plan})
    smiles = dict(zip(uniq, _opsin(uniq, "SMILES", patent_id)))

    best: dict[int, tuple[str, str]] = {}
    for i, cand in plan:                       # plan is least-invasive first
        smi = smiles.get(cand, "")
        if smi and i not in best:
            best[i] = (cand, smi)

    _mark_unread(have, plan, best, patent_id)
    if not best:
        return []

    order = sorted(best)
    keys = _opsin([best[i][0] for i in order], "StdInChIKey", patent_id)

    out = []
    for i, ik in zip(order, keys):
        it = have[i]
        name, smi = best[i]
        out.append(NamedCompound(
            patent_id=it.patent_id, name=name, smiles=smi, inchikey=ik,
            start=-1, source=SOURCE, cid=it.cid,
            drawn_ref=it.chemistry_id, drawn_file=it.image_file))
    logger.info("image_ocr: %s — %d of %d image(s) carried a parseable name",
                patent_id, len(out), len(have))
    return out
