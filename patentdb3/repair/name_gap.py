"""Where the patent named a compound and we produced nothing.

The `repair/gap.py` of the name tier. Same job — locate the failures worth
paying to fix, and carry enough evidence with each one that a model can answer
— against names rather than table layouts.

WHY "ASSERTED" IS THE TRIGGER, AND NOT ANY OTHER SIGNAL
---------------------------------------------------------
Three candidate triggers were measured over the 137-patent corpus before this
module was written. Two failed:

  - EVERY OPSIN REJECTION. 82,398 failing seeds, of which **67% are hyphenated
    English** (`receptor-mediated`, `above-mentioned`, `post-translational`).
    `_SEED` is permissive by design and OPSIN is correctly refusing them. A
    threshold over this population fires hardest on the patents with the most
    discursive background sections.

  - SHAPE CLUSTERING. Five hypothesis-free skeletons (full class projection,
    run-length collapsed, edges-only, delimiter-only). The best puts 91% of
    failures in clusters of >=5, and the clusters are useless — the largest are
    `1-methylpiperidin-4-yl` (1,048), `1,1,1,3,3,3-Hexafluoropropan-2-yl` (735),
    `heterocycloalkyl)-C1-4` (566): substituent names in prose and Markush
    placeholders out of the claims. **A name's shape is dominated by what kind
    of chemistry it is, while a defect is a small local perturbation**, so any
    projection lossy enough to cluster erases the defect before the chemistry.

What works is not a property of the string at all. A heading reading
`Example 43: 2-methyl-4-(trifluoromethoxy)benzyl...` is the patent stating in
its own voice that a compound numbered 43 exists and is named here. If nothing
resolves, a compound was lost — and saying so **requires no hypothesis about
what went wrong**, which is the property that keeps this trigger correct for
corruption forms nobody has seen yet. A `pyridin-2-yl` mid-sentence asserts
nothing and is not a miss.

Measured on the corpus: 10,725 asserting headings, and after the heading and
de-wrap routes landed, ~584 that still resolve to nothing. That is the
population this module returns.

WHAT IS DELIBERATELY EXCLUDED
------------------------------
  - Headings with no chemical morpheme in the residual text: 379 of the 1,344
    raw misses are section headers like `Alternative Preparation of Intermediate
    T-1`. Not compounds, so not losses.
  - Phane / macrocyclic nomenclature. OPSIN 2.9.0 does not implement IUPAC
    P-26 — verified directly against textbook examples, including
    `[2.2]paracyclophane`, the most-cited phane compound there is. 20 compounds
    corpus-wide. No rule can fix a missing grammar, so escalating them forever
    is the only thing a loop could do, and `_PHANE` keeps them out of it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..sources import iupac_names as IN
from ..sources import table_names as TN
from ..sources import losses as _losses
from ..sources import opsin as OP
from ..sources.anchor import anchor_text
from ..sources.dewrap import targeted as _dewrap_targeted

logger = logging.getLogger(__name__)

# Residual text has to look like chemistry at all. Same morpheme list
# `sources/table_names.py` uses to decide a cell is name-shaped — two different
# definitions of "looks like chemistry" in one codebase is how a denominator
# drifts.
_CHEM = re.compile(
    r'(?:yl|phenyl|methyl|imidazol|pyrimidin|morpholin|triazol|chloro|bromo|'
    r'fluoro|ethyl|propyl|cyclo|oxo|amino|nitro|benz|piperid|piperaz|'
    r'pyridine|pyrazol|oxazol|thiazol)', re.I)

# See the module docstring. Not a defect and not repairable — a grammar OPSIN
# does not implement.
_PHANE = re.compile(r"phane|phan-\d|cyclo(?:non|dec|undec|dodec|tridec)a", re.I)

# Two compounds stated in one heading (`... (Compound 5A) and (18S,Z)-...`).
# Excluded because the repair is a SPLIT, not a character fix, and this tier
# buys character fixes — a split rule would have to decide which half owns the
# id, which is `anchor.py`'s question and was closed as won't-fix (11 real
# recoverable anchors corpus-wide, measured).
#
# `^and \d+` is here because `_HEADING_ID` reads only the FIRST id token, by
# design — that token is the compound number. `Examples 105 and 106. (S)-...`
# therefore arrives as cid 105 with `and 106. (S)-...` still on the front, and
# it is not a corrupted name at all: it is two compounds sharing one heading.
_CONJUNCTION = re.compile(
    r"\)\s+and\s+\(|\band its enantiomer|\band\s+\(\d"
    r"|^and\s+(?:\d|intermediate|example|compound|preparation)", re.I)

# NOT AN OPSIN LIMIT, so it does not come from `sources/opsin.py`. The parser
# was never asked: two compounds share one assertion and the repair is a
# SPLIT, which is `anchor.py`'s question. Recorded under its own reason so a
# later pass can tell it apart from a name OPSIN actually refused.
_REASON_TWO_COMPOUNDS = "two_compounds_one_assertion"

# THE STEREO POPULATION IS DETECTED BY `sources/opsin.stereo_blocked`, NOT
# HERE. A regex over OPSIN's prose used to do it
# (`could not find atom that|<stereoChemistry`); it caught 46 of the 53 real
# cases in a 630-name census and over-reached on one. The replacement asks
# OPSIN the same name again with `--allowUninterpretableStereo` and watches
# whether it flips, which is a measurement rather than a guess about wording.
# See that function for the numbers and for why the element echo is kept as a
# second arm.
#
# The dominant shape is the pseudo-asymmetric CIP descriptor — the lowercase
# `r`/`s` in `(1r,4R)-4-hydroxycyclohexyl`. Measured over
# US10155002 / US10710980 / US11420968: **155 of 217 asserted-compound gaps
# (71%)**; over the first 30 patents of the corpus it is 53 of 630 (8.4%). Both
# are real — it is a per-document convention, like everything else here.
#
# No character rule can add a grammar to the parser, so a loop pointed at these
# buys nothing and escalates forever — and worse, a model handed one will try
# to make it parse by rewriting the chemistry (`acetamido` -> `amino`,
# observed). They therefore leave the loop.
#
# THEY ARE NOT UNRECOVERABLE, they are just not repairable HERE: stripping the
# descriptor makes 154 of the 155 parse. That loses the stereochemistry, so the
# result would stand for a SET of stereoisomers and would have to carry the
# same treatment a Markush name does — flagged, and no InChIKey. That trade was
# put to the user and declined: these stay UNRESOLVED and are recorded, so the
# decision can be revisited against a list rather than re-derived from scratch.


@dataclass
class NameGap:
    """One compound the patent named that we could not resolve."""
    patent_id: str
    cid: str
    name_text: str                 # the assertion's own name text, framing peeled
    opsin_error: str = ""          # OPSIN's own diagnosis, verbatim
    # WHERE THE PATENT MADE THE ASSERTION: `heading` or `table`. Not consumed
    # by `name_outcome.measure`, which reads only `name_text` and `doc_text` and
    # is source-agnostic by construction — this exists so the journal can say
    # which population a bought rule came from, and so a zero on either side is
    # visible rather than assumed.
    source: str = "heading"
    # Names from THIS patent that already parse. Two uses, both load-bearing:
    # `name_rules.ground()` measures collateral damage against them, and the
    # prompt shows a couple so the model can see what an intact name here looks
    # like.
    clean_names: list[str] = field(default_factory=list)
    doc_text: str = ""             # flattened patent text, for corroboration
    # Set by `name_loop` from OPSIN's diagnosis — the FAILURE class, not the
    # name. See `name_rules.name_signature`.
    fingerprint: str = ""
    signature: str = ""

    @property
    def key(self) -> str:
        return f"{self.patent_id}::{self.cid}"


def _opsin_errors(names: list[str]) -> dict[str, str]:
    """OPSIN's own error line per name. Lives in `sources/opsin.py`.

    Kept as a one-line alias rather than inlined at the call sites: this name
    is what the tests and the surrounding commentary already refer to, and the
    move was about removing a SECOND copy from the tree, not about renaming
    the thing. See `opsin.errors` for why it must run the jar directly.
    """
    return OP.errors(names)


def find_name_gaps(xml: str, patent_id: str = "",
                   *, with_opsin_errors: bool = True) -> list[NameGap]:
    """Every compound this patent asserts and `extract_names` did not resolve.

    Runs the real extractor, not a re-implementation of it — "we produced
    nothing" has to mean the shipping code produced nothing, or the loop buys
    rules for failures that do not exist.
    """
    # `heading_resolution` IS the parsing test, and asking it any other way is
    # the bug this replaced. The first version of this function asked "does any
    # structure carry this compound id", which is an ANCHORING test: it
    # reported 2,746 gaps over 53 patents and the first 60 inspected ALL PARSED
    # PERFECTLY WELL — the names were fine, they simply had not been anchored.
    # A heal loop aimed at that population buys character-repair rules for
    # names that were never broken, and every one of those rules would be
    # gate-passing garbage in a permanent library.
    try:
        heads, resolved = IN.heading_resolution(xml, patent_id)
    except Exception as e:
        logger.warning("name_gap: %s — heading_resolution failed: %r", patent_id, e)
        return []
    if not heads:
        return []

    # THE SECOND ASSERTION SITE, and it was missing. A table row reading
    # `cid | Name` states, in the patent's own voice, that a compound with that
    # number is named here — the identical claim a heading makes, and the
    # trigger this module is built on. Only headings were ever read, so an
    # entire population was invisible: on US10376513 the loop saw 9 gaps while
    # 155 measured compounds sat in a correctly-detected `Name` column, every
    # one failing OPSIN on a `<sup>` footnote digit fused to the tail
    # (`...propan-2-ol2`). That defect is precisely what this tier exists to
    # buy a rule for, and it could not see it.
    #
    # `cell_resolution` is the table route's OWN verdict, not a second
    # implementation of it — same contract as `heading_resolution` above and
    # for the same reason.
    try:
        cells, cell_ok = TN.cell_resolution(xml, patent_id)
    except Exception as e:
        logger.warning("name_gap: %s — cell_resolution failed: %r", patent_id, e)
        cells, cell_ok = [], {}

    # The collateral-damage population for `name_rules.ground()`: this
    # patent's own names that DID resolve, from BOTH assertion sites. A rule
    # bought for a table cell must not be allowed to break a heading name.
    clean = [resolved[i][1] for i in sorted(resolved)]
    clean += [cell_ok[i][1] for i in sorted(cell_ok)]
    clean = clean[:400]

    asserted: list[tuple[str, str, str]] = []          # (cid, name_text, source)
    for i, (cid, name_text) in enumerate(heads):
        if i not in resolved:
            asserted.append((cid, name_text, "heading"))
    for i, c in enumerate(cells):
        if i in cell_ok:
            continue
        cid, raw = c[4], c[5]
        # An unnamed row asserts nothing about WHICH compound this is, so it
        # cannot become a gap — the loop would have no id to report a recovery
        # against. Signal B already refuses these; signal A admits them.
        if not cid:
            continue
        # THE GAP IS THE RESIDUE AFTER TRANSFORMATIONS WE ALREADY OWN, not the
        # raw cell. A table cell carries typesetting wrap spaces
        # (`4-Amino-3- methyl`) that `dewrap.targeted` removes and the
        # extractor has ALREADY applied and still failed on. Handing the model
        # the raw string makes those spaces the most conspicuous thing in it,
        # and it spends its attempt there: the first run of this against
        # US10376513 proposed `space_in_methyl` three times, each rejected with
        # "applying the rule leaves every name unchanged", and escalated 164
        # gaps for $0.027 — buying nothing, because the defect it described was
        # one the pipeline fixes before OPSIN ever sees the string.
        #
        # Dewrapped, the only thing left unexplained is the real defect, and
        # OPSIN's own error then names it (`uninterpretable: 2` for the
        # `<sup>` footnote digit fused to `...propan-2-ol2`).
        asserted.append((cid, _dewrap_targeted(raw), "table"))

    # EVERY EXCLUSION BELOW IS RECORDED, not just skipped. Each of these was a
    # bare `continue`, so three different populations left the pipeline through
    # the same silence and none of them was countable afterwards. The reason
    # codes are the vocabulary a later pass reads; `_CHEM` is the one exclusion
    # that stays silent, because a section header is not a compound and never
    # was.
    gaps: list[NameGap] = []
    for cid, name_text, src in asserted:
        if not _CHEM.search(name_text):
            continue                                  # a section header
        if _PHANE.search(name_text):
            _losses.record("asserted_name_unparsed", patent_id, cid=cid,
                           name=name_text, reason=OP.REASON_GRAMMAR,
                           opsin_error="", assertion=src,
                           detail="phane / macrocycle: OPSIN 2.9.0 does not "
                                  "implement IUPAC P-26")
            continue
        if _CONJUNCTION.search(name_text):
            _losses.record("asserted_name_unparsed", patent_id, cid=cid,
                           name=name_text, reason=_REASON_TWO_COMPOUNDS,
                           opsin_error="", assertion=src)
            continue
        gaps.append(NameGap(patent_id=patent_id, cid=cid, name_text=name_text,
                            clean_names=clean, source=src))

    if not gaps:
        return []

    doc = anchor_text(xml, patent_id)
    # Always run the diagnosis — `stereo_blocked` reads it, so it decides
    # which gaps leave the loop, not only what the prompt says.
    # `with_opsin_errors` controls whether it reaches the MODEL, which is the
    # thing the harness ablates; it must not also control whether the loop
    # knows what it is looking at.
    errs = _opsin_errors([g.name_text for g in gaps])
    blocked = OP.stereo_blocked([g.name_text for g in gaps], errs, patent_id)
    kept: list[NameGap] = []
    dropped = 0
    for g in gaps:
        err = errs.get(g.name_text, "")
        if g.name_text in blocked:
            # LEAVING THE LOOP IS NOT THE SAME AS LEAVING THE ARTIFACT. This
            # branch used to be a bare `continue` with a count in a log line,
            # so a compound the patent named, that we read correctly, and that
            # is blocked only by a parser limit, was indistinguishable from
            # one nothing ever looked at. It is unresolved either way — the
            # record is what makes it addressable later.
            dropped += 1
            _losses.record("asserted_name_unparsed", patent_id, cid=g.cid,
                           name=g.name_text, reason=OP.REASON_STEREO,
                           opsin_error=err, assertion=g.source)
            continue
        g.doc_text = doc
        g.opsin_error = err if with_opsin_errors else ""
        kept.append(g)
    if dropped:
        logger.info("name_gap: %s — %d gap(s) recorded as %s and excluded "
                    "from the loop; a character rule cannot add a grammar",
                    patent_id, dropped, OP.REASON_STEREO)
    gaps = kept

    n_tbl = sum(1 for g in gaps if g.source == "table")
    logger.info("name_gap: %s — %d asserted compounds unresolved "
                "(%d heading, %d table)", patent_id, len(gaps),
                len(gaps) - n_tbl, n_tbl)
    return gaps
