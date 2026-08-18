"""The ONE OPSIN batch wrapper. Every caller goes through here.

WHY THIS EXISTS
---------------
Three modules had their own copy of this function — `iupac_names._opsin`,
`table_names._opsin_batch`, `name_repair._opsin_batch` — written by three agents
that could not see each other's files. All three ended in the identical line:

    return list(out) + [""] * (len(names) - len(out))

and all three were wrong in the same way. Two copies of one rule drift; three
copies of one BUG is worse, because fixing the copy you happen to be reading
looks like fixing the problem.

WHAT THE BUG WAS
----------------
OPSIN is a Java subprocess and its output is matched to its input BY POSITION —
every caller does `zip(names, results)`. On US10730863 it returned **4,973
outputs for a 4,972-name batch**, observed in the loss log, cause unknown.

When `len(out) > len(names)` the padding expression multiplies a list by a
NEGATIVE number, which yields `[]`, so the oversized list is returned unchanged
and `zip` silently truncates it. Whether that is harmless depends entirely on
WHERE the extra element came from, and the caller cannot tell:

  - an extra APPENDED at the end -> `zip` drops it, every pairing is correct
  - an extra INSERTED mid-list  -> every pairing after that point is shifted by
    one, and each name receives the NEXT name's structure

The second case ships a wrong structure for a real compound under a real
compound number, with nothing raising and nothing logged in two of the three
copies. It is the exact failure this pipeline exists to prevent, and it is
undetectable downstream: the SMILES is valid, the InChIKey is valid, the name is
valid, they are simply not each other's.

WHAT IT DOES NOW
----------------
**On any length mismatch it refuses the whole batch.** Not padding, not
truncation — every result comes back empty and the event is recorded.

That is deliberately drastic. The alternative is guessing which of the two
shapes occurred, and there is no evidence available at this layer to guess
from. Losing a batch costs coverage, which is recoverable and visible; shipping
a shifted batch costs correctness, which is neither. The same asymmetry already
governs `name_repair`: never accept a repair you cannot confirm.

If a mismatch turns out to be common rather than a one-off, the fix is to find
out why OPSIN does it — not to soften this.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile

from ..core import config
from . import losses as _losses

logger = logging.getLogger(__name__)


def radicals(names: list[str], patent_id: str = "") -> list[str]:
    """Resolve SUBSTITUENT names. One entry per input, in order, `""` on refusal.

    `batch` refuses a substituent by design — OPSIN's own message is "a name is
    just a substituent" — because a fragment is not a compound and returning
    one as if it were would put half a molecule in the artifact. A substituent
    table asks the opposite question: its cells are substituents and nothing
    else, and each one needs to come back with its attachment point marked.

    `--wildcardRadicals` is that answer: `5-chloro-2,4-difluoro-phenyl` returns
    `*C1=C(C=C(C(=C1)Cl)F)F`, a `*`-marked fragment in exactly the form
    `markush.build_text_group` joins.

    IT ALSO SEPARATES A NAME FROM A FORMULA, WHICH NOTHING ELSE DOES. Measured
    on US9718825's slot values: every substituent name resolved, and `CH3`,
    `NH2`, `3-CH3` and `2-F` were all refused. So a cell this resolves is a
    name and its leading digits are locants of its OWN ring, not a position on
    the scaffold — which is what stops `5-chloro-...-phenyl` being read as
    "locant 5, group chloro-...-phenyl" and built into a wrong molecule.

    This is the ONE OPSIN wrapper still. Do not write a second.
    """
    return batch(names, patent_id=patent_id, wildcard_radicals=True)


def batch(names: list[str], fmt: str = "SMILES", patent_id: str = "",
          *, allow_bad_stereo: bool = False,
          wildcard_radicals: bool = False) -> list[str]:
    """Resolve `names` through OPSIN. Returns one entry per input, in order.

    An entry is `""` when that name did not resolve — and, per the module
    docstring, EVERY entry is `""` when the batch could not be trusted at all.
    Callers may therefore keep pairing by position, which is what they all do.

    `tmp_fpath` is pid-scoped because py2opsin writes its input to a shared
    temp file whose default name is a constant: two processes running
    concurrently overwrite each other's input and silently receive each
    other's answers.

    `allow_bad_stereo` IS A DIAGNOSTIC, NOT A RESOLVER. It passes OPSIN's
    `--allowUninterpretableStereo`, which makes the parser drop a stereo
    descriptor it cannot map onto the skeleton and return the skeleton anyway.
    The structure that comes back therefore stands for a SET of stereoisomers,
    which is the same claim a Markush name makes and carries the same
    treatment — see `classify()` below, which is the only caller in this
    package. Nothing resolves through this flag; it exists to tell "OPSIN has
    no grammar for this" apart from "the text is broken", which are different
    populations owned by different parts of the pipeline.
    """
    if not names:
        return []
    from py2opsin import py2opsin

    tmp = str(config.OUTPUT_DIR / f".opsin_in_{os.getpid()}.txt")
    try:
        out = py2opsin(names, output_format=fmt, tmp_fpath=tmp,
                       allow_bad_stereo=allow_bad_stereo,
                       allow_radicals=wildcard_radicals,
                       wildcard_radicals=wildcard_radicals)
    except Exception as e:                       # OPSIN is a java subprocess
        logger.warning("opsin: batch of %d failed: %r", len(names), e)
        if _losses.ENABLED:
            _losses.record("opsin_batch_exception", patent_id, fmt=fmt,
                           batch_size=len(names), error=repr(e)[:200])
        return [""] * len(names)

    if isinstance(out, str):                     # single-name convenience form
        out = [out]

    # THE REFUSAL. See the module docstring: a length mismatch means the
    # position-to-position correspondence every caller relies on is no longer
    # established, and nothing here can tell a harmless trailing extra from a
    # mid-list insertion that shifts every subsequent pairing.
    if not isinstance(out, list) or len(out) != len(names):
        logger.warning(
            "opsin: batch of %d returned %s — REFUSING the whole batch rather "
            "than risk mis-pairing names with other names' structures",
            len(names), "a non-list" if not isinstance(out, list) else f"{len(out)}")
        if _losses.ENABLED:
            _losses.record(
                "opsin_batch_refused", patent_id, fmt=fmt,
                batch_size=len(names),
                returned=(len(out) if isinstance(out, list) else -1),
                reason=("malformed_output" if not isinstance(out, list)
                        else "length_mismatch"))
        return [""] * len(names)

    return out


def errors(names: list[str], patent_id: str = "") -> dict[str, str]:
    """OPSIN's own error line per name, by running the jar directly.

    `py2opsin` returns only the SMILES column and discards stderr, so the one
    party that actually knows why a name failed is silent by the time the
    result reaches us. That diagnosis is the single most informative thing
    available about a failure — it distinguishes `unmatched opening bracket`
    from `uninterpretable` from `unable to assign all locants`, which are three
    different repairs — so it is worth one extra subprocess per patent.

    Best-effort: any failure here yields an empty mapping and the caller
    carries on without the annotation rather than losing the record.

    This lived in `repair/name_gap.py` until `sources/image_ocr.py` needed the
    same diagnosis. A `sources` module cannot import from `repair` — the
    layering runs the other way — and a second copy is precisely what this
    module's docstring exists to prevent, so it moved here. `name_gap` calls
    it; nothing about its behaviour changed in the move.
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
        logger.info("opsin: diagnosis unavailable for %s: %r", patent_id, e)
        return {}

    # POSITIONAL, WITH A CONTENT FALLBACK — and getting this wrong hid the
    # single most useful piece of context from 85% of prompts.
    #
    # Two content-matching versions were tried first and both under-captured.
    # A literal `name[:60] in line` test matched 32 of 217 real gaps, because
    # OPSIN echoes the name back in ITS normalisation (`′` U+2032 becomes `'`).
    # Matching on alphanumerics only got 49. The reason neither works is that
    # OPSIN's error frequently does not contain the name AT ALL:
    #
    #     Could not find atom that: <stereoChemistry locant="1" type="RorS"
    #     value="R" stereoGroup="Abs">1r</stereoChemistry> appeared to be
    #     referring to
    #
    # Measured directly: OPSIN emits one stderr line per FAILURE (not per
    # name), in input order — 77 lines for 77 failing names, and 2 lines for a
    # 5-name batch containing 2 failures. Every name here is already known to
    # have failed, so the lines correspond one-to-one and IN ORDER.
    #
    # The count check is the guard: a name that emits two lines (an error plus
    # a locant warning) would shift every later pairing, and a silently shifted
    # diagnosis is worse than none — it would attribute one compound's failure
    # to another. When the counts disagree, fall back to content matching,
    # which under-captures but cannot mis-attribute.
    lines = [l.strip() for l in proc.stderr.splitlines()
             if l.strip() and "info" not in l.lower()[:20]]
    if len(lines) == len(names):
        return {n: l[:300] for n, l in zip(names, lines)}

    logger.info("opsin: %s emitted %d lines for %d names — falling back "
                "to content matching rather than risk mis-pairing",
                patent_id, len(lines), len(names))

    def _key(s: str) -> str:
        return "".join(c for c in s if c.isalnum()).lower()

    keyed = [(_key(n)[:40], n) for n in names]
    out: dict[str, str] = {}
    for line in lines:
        lk = _key(line)
        for head, n in keyed:
            if n not in out and head and head in lk:
                out[n] = line[:300]
                break
    return out


# THE ONE REASON CODE THIS MODULE ISSUES. A name carrying it is not corrupt
# text and no character rule will ever recover it, so it must leave the repair
# loop — but it also must not vanish, which is what happened before: the whole
# population was `continue`d with nothing but a count in a log line.
REASON_STEREO = "stereo_unmappable"
# A grammar OPSIN 2.9.0 does not implement at all — phane / macrocyclic
# nomenclature (IUPAC P-26), verified against textbook examples including
# `[2.2]paracyclophane`. Detected by the caller, from the NAME rather than
# from an error, because OPSIN's refusal of these is not distinctive.
REASON_GRAMMAR = "grammar_unsupported"
# The complement of both, so a caller that records EVERY outcome uses one
# vocabulary. Not issued by anything here — `stereo_blocked` answers one
# question and the caller names the other side.
REASON_TEXT = "text_defect"

# OPSIN QUOTING ITS OWN PARSE BACK AT US. The element echo means OPSIN read the
# descriptor, built a `<stereoChemistry>` node for it, and then could not place
# that node on the skeleton — which is a statement about the parser, not about
# the string.
_STEREO_ELEMENT = re.compile(r"<stereoChemistry", re.I)


def stereo_blocked(names: list[str], errs: dict[str, str],
                   patent_id: str = "") -> set[str]:
    """Of `names`, the ones OPSIN refuses ONLY because of stereochemistry.

    THE TEST IS A MEASUREMENT, NOT A PATTERN. Ask OPSIN the same question again
    with `--allowUninterpretableStereo`: a name that flips from refused to
    accepted was never broken text — the only thing standing between it and a
    structure is a descriptor the parser cannot map onto the skeleton.

    Measured over 630 asserted-but-unresolved names from 30 patents:

        parses with `-s`                     52
        `<stereoChemistry` echo, no flip      1     `(3a,9bR)`, alphaOrBeta
        union                                53   (8.4%)

    The regex this replaced (`could not find atom that|<stereoChemistry`)
    caught 46 — it missed the 7 OPSIN reports in prose instead of by element
    echo (`endoExoSynAnti stereochemistry is not currently interpretable`,
    `Failed to assign CIP stereochemistry`), which no amount of widening the
    pattern would have found reliably. The element echo is KEPT as a second
    arm because it catches the one case `-s` does not rescue, and because it
    cannot mis-fire: the string only appears when OPSIN is quoting a stereo
    node it built itself.

    The other 577 are NOT this. 405 are `uninterpretable substring` and 97 an
    unmatched bracket — text defects, which is exactly the repair loop's
    population, and they stay there.
    """
    if not names:
        return set()
    uniq = sorted(set(names))
    lax = batch(uniq, "SMILES", patent_id, allow_bad_stereo=True)
    flipped = {n for n, s in zip(uniq, lax) if s.strip()}
    return {n for n in uniq
            if n in flipped or _STEREO_ELEMENT.search(errs.get(n, ""))}
