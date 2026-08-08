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

from ..core import config
from . import losses as _losses

logger = logging.getLogger(__name__)


def batch(names: list[str], fmt: str = "SMILES", patent_id: str = "") -> list[str]:
    """Resolve `names` through OPSIN. Returns one entry per input, in order.

    An entry is `""` when that name did not resolve — and, per the module
    docstring, EVERY entry is `""` when the batch could not be trusted at all.
    Callers may therefore keep pairing by position, which is what they all do.

    `tmp_fpath` is pid-scoped because py2opsin writes its input to a shared
    temp file whose default name is a constant: two processes running
    concurrently overwrite each other's input and silently receive each
    other's answers.
    """
    if not names:
        return []
    from py2opsin import py2opsin

    tmp = str(config.OUTPUT_DIR / f".opsin_in_{os.getpid()}.txt")
    try:
        out = py2opsin(names, output_format=fmt, tmp_fpath=tmp)
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
