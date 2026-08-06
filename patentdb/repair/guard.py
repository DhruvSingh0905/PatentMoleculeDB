"""The one barrier in front of every write to TRACKED SOURCE.

WHAT WENT WRONG. `patentdb/sources/uspto_assays.py` was rewritten by ordinary
extraction runs. Reproduced on US20240010684A1 at default settings, with every
paid call blocked:

    REPAIR unset  -> uspto_assays.py sha e1982d48 -> 88d774ec, journal 37 -> 38
    REPAIR=0      -> unchanged, journal unchanged

so the `REPAIR=0` gate in `process_patent` was never the hole. The DEFAULT was.
The chain is four hops, and every one of them was ON by default and doing
exactly what it was written to do:

    process_patent.py:2186   if config.REPAIR_ENABLED          (REPAIR=1)
    process_patent.py:2237   maybe_escalate(...)
    autoheal.py:173          if not config.REPAIR_AUTOHEAL     (REPAIR_AUTOHEAL=1)
    capability.py:1100       do_apply = config.PARSER_REPAIR_APPLY  (=1)
    capability.py:1280       mod.write_text(text)              <- the write

Two things made it silent rather than merely surprising. The model call is
CACHED, and `propose_capability_patch` reads the cache (capability.py:759)
before it checks for an API key (:765) — so a patch bought once replays for
ever at zero cost and zero network. And `_annotate_prior_attempts`
(capability.py:954) skips any journal entry whose `gap_rows_recovered > 0`,
which is what feeds the `retry` counter that is supposed to be IN the cache key
(capability.py:744). A patch that WORKED therefore never increments `retry`,
the key never changes, and the same proposal is re-applied on every run. The
journal shows it: `0030`, `0034`, `0036`, `0038` share the id suffix
`09a09002`, and that suffix is `sha256(after_source + signature)` — provably one
proposal applied four times.

WHY A FLAG WAS NOT ENOUGH. `PARSER_REPAIR_APPLY` already existed for this and
defaulted ON, with the reasoning that "a fix that waits on a human is a queue"
and that safety is reversibility, not permission. That trade is defensible for
an operator who typed `capability_repair --repair`. It is not defensible for
`process_patent`, because it breaks something the whole project rests on: an
artifact can no longer be attributed to the code that produced it. Two agents
lost measurements to exactly that, one with `uspto_assays.py` rewritten
mid-benchmark.

So the gate here is not another boolean. It is PROVENANCE: did somebody ask?

    an operator ran a repair CLI   -> `operator_request()` is entered, writes OK
    SELF_PATCH=1 is set            -> writes OK, for a deliberate healing run
    anything else, incl. the       -> REFUSED, loudly, and journaled as a
    pipeline                          declined proposal that `--force` can apply

`PARSER_REPAIR_APPLY=0` keeps its old meaning and still vetoes everything; it is
now a kill switch over the top rather than the only thing standing there.

The tier is NOT removed. Every diagnosis still runs, every proposal is still
journaled with its full before/after source, and `parser_health --force <id>`
applies one afterwards. What used to happen without being asked now happens
when asked, which is the only change.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from ..core import config

logger = logging.getLogger(__name__)

# Set by the repair CLIs, which ARE the operator asking. Thread-local rather
# than global because `verify_patch` runs the corpus probe in a subprocess and
# the greedy/iterate loops measure candidates on worker threads; a global would
# leak the permission into work the operator did not name.
_asked = threading.local()

#: Opt in from the environment, for a deliberate self-healing run that is not
#: going through one of the CLIs — `SELF_PATCH=1 python3 -c "...process_patent..."`.
#: Read on every call, never captured at import, so a test can set it.
SELF_PATCH_ENV = "SELF_PATCH"


def self_patch_opted_in() -> bool:
    return os.environ.get(SELF_PATCH_ENV, "0") == "1"


def operator_requested() -> bool:
    return bool(getattr(_asked, "on", False))


@contextmanager
def operator_request():
    """Mark this thread as an explicit operator request to patch the tree.

    Entered by the repair CLIs. Nesting-safe and restored on the way out, so a
    CLI that calls another entry point does not leave the permission set.
    """
    prev = getattr(_asked, "on", False)
    _asked.on = True
    try:
        yield
    finally:
        _asked.on = prev


def may_write_tracked_source(*, explicit: bool | None = None) -> bool:
    """May the caller rewrite a tracked `.py` right now?

    `explicit` is for a caller that names its own intent — `revert(id)` and
    `apply_journaled(id)` are an operator typing a journal id, and the eval
    harnesses pass it straight through from their own `--patch` flag.
    """
    if not config.PARSER_REPAIR_APPLY:
        return False                       # the kill switch still wins outright
    if explicit is not None:
        return bool(explicit)
    return operator_requested() or self_patch_opted_in()


class Outcome:
    """Why `write_tracked_source` did or did not write. Three cases, not two."""

    WRITTEN = "written"
    REFUSED = "refused"
    UNCHANGED = "unchanged"


def write_tracked_source(module: Path, text: str, *, what: str = "",
                         explicit: bool | None = None) -> str:
    """The ONLY way `repair/` may rewrite a tracked source file.

    Returns one of `Outcome`. `UNCHANGED` is deliberately distinct from
    `WRITTEN`: a cached proposal replayed onto a tree that already holds it is
    not an application, and journalling it as one is how `0034`-`0038` came to
    claim four applications of a single patch.
    """
    module = Path(module)
    if not may_write_tracked_source(explicit=explicit):
        logger.warning(
            "REFUSED to modify tracked source %s%s — nothing asked for it. "
            "An extraction run must not rewrite the code it is being measured "
            "against. The proposal is journaled; apply it deliberately with "
            "`parser_health --force <id>`, `capability_repair --repair`, or by "
            "setting %s=1.",
            module.name, f" ({what})" if what else "", SELF_PATCH_ENV)
        return Outcome.REFUSED
    try:
        if module.read_text() == text:
            logger.info("tracked source %s already holds this patch%s — not "
                        "recording it as an application",
                        module.name, f" ({what})" if what else "")
            return Outcome.UNCHANGED
    except OSError:
        pass
    module.write_text(text)
    logger.warning("MODIFIED tracked source %s%s", module.name,
                   f" ({what})" if what else "")
    return Outcome.WRITTEN
