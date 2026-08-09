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
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

from ..core import config
from ..sources import iupac_names as IN
from ..sources.anchor import anchor_text

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
_CONJUNCTION = re.compile(r"\)\s+and\s+\(|\band its enantiomer|\band\s+\(\d", re.I)


@dataclass
class NameGap:
    """One compound the patent named that we could not resolve."""
    patent_id: str
    cid: str
    name_text: str                 # the heading's own name text, framing peeled
    opsin_error: str = ""          # OPSIN's own diagnosis, verbatim
    # Names from THIS patent that already parse. Two uses, both load-bearing:
    # `name_rules.ground()` measures collateral damage against them, and the
    # prompt shows a couple so the model can see what an intact name here looks
    # like.
    clean_names: list[str] = field(default_factory=list)
    doc_text: str = ""             # flattened patent text, for corroboration

    @property
    def key(self) -> str:
        return f"{self.patent_id}::{self.cid}"


def _opsin_errors(names: list[str]) -> dict[str, str]:
    """OPSIN's own error line per name, by running the jar directly.

    `py2opsin` returns only the SMILES column and discards stderr, so the one
    party that actually knows why a name failed is silent by the time the
    result reaches us. That diagnosis is the single most informative thing
    available about a failure — it distinguishes `unmatched opening bracket`
    from `uninterpretable` from `unable to assign all locants`, which are three
    different repairs — so it is worth one extra subprocess per patent.

    Best-effort: any failure here yields an empty mapping and the loop carries
    on without the annotation rather than losing the gap.
    """
    if not names:
        return {}
    try:
        import py2opsin
        jar = None
        base = os.path.dirname(py2opsin.__file__)
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".jar"):
                    jar = os.path.join(root, f)
                    break
            if jar:
                break
        if not jar:
            return {}
        with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False,
                dir=str(config.OUTPUT_DIR)) as fh:
            fh.write("\n".join(n.replace("\n", " ") for n in names))
            path = fh.name
        proc = subprocess.run(["java", "-jar", jar, "-o", "smi", path],
                              capture_output=True, text=True, timeout=300)
        os.unlink(path)
    except Exception as e:                       # java missing, timeout, ...
        logger.info("name_gap: OPSIN diagnosis unavailable: %r", e)
        return {}

    out: dict[str, str] = {}
    for line in proc.stderr.splitlines():
        low = line.lower()
        if not line.strip() or "info" in low[:20]:
            continue
        for n in names:
            if n in out:
                continue
            head = n[:60]
            if head and head in line:
                out[n] = line.strip()[:300]
                break
    return out


def find_name_gaps(xml: str, patent_id: str = "",
                   *, with_opsin_errors: bool = True) -> list[NameGap]:
    """Every compound this patent asserts and `extract_names` did not resolve.

    Runs the real extractor, not a re-implementation of it — "we produced
    nothing" has to mean the shipping code produced nothing, or the loop buys
    rules for failures that do not exist.
    """
    try:
        resolved = IN.extract_names(xml, patent_id)
    except Exception as e:
        logger.warning("name_gap: %s — extract_names failed: %r", patent_id, e)
        return []

    got_cids = {nc.cid for nc in resolved if nc.cid}
    clean = [nc.name for nc in resolved][:400]

    gaps: list[NameGap] = []
    for cid, name_text in IN._heading_texts(xml):
        if cid in got_cids:
            continue
        if not _CHEM.search(name_text):
            continue                                  # a section header
        if _PHANE.search(name_text) or _CONJUNCTION.search(name_text):
            continue                                  # not repairable here
        gaps.append(NameGap(patent_id=patent_id, cid=cid, name_text=name_text,
                            clean_names=clean))

    if not gaps:
        return []

    doc = anchor_text(xml, patent_id)
    errs = _opsin_errors([g.name_text for g in gaps]) if with_opsin_errors else {}
    for g in gaps:
        g.doc_text = doc
        g.opsin_error = errs.get(g.name_text, "")

    logger.info("name_gap: %s — %d asserted compounds unresolved", patent_id, len(gaps))
    return gaps
