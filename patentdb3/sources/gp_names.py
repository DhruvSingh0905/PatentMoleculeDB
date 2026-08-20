"""Bulk import of Google Patents' own chemical-name annotations, kept only
where OPSIN resolves them and where we do not already hold that structure.

WHAT THIS IS, AND WHY IT IS SEPARATE FROM `gp_images.py`
----------------------------------------------------------
`gp_images.py` reads a Google Patents page for RENDERED IMAGE URLS —
`patentimages.storage.googleapis.com/.../<filename>.png` — and its own
docstring explains why that is the only thing on the page this codebase reads:
the page's chemical annotations carry no compound number, so they cannot join
an assay row.

This module reads the SAME PAGE for a different thing: the `name` text of
every chemical annotation, run through `sources/opsin.py` — the one OPSIN
wrapper — to see whether IT resolves a structure. That structure still carries
no compound number and still cannot join an assay row. The point is not to
recover a measurement; it is bulk structural coverage with no measurement
attached, which CLAUDE.md's owner has already accepted as the intended shape
of this output.

`_fetch` IS REUSED, NOT REWRITTEN. This module imports
`gp_images._fetch(patent_id) -> str` for the one HTTP GET a patent needs.
Writing a second `urllib.request` call here is exactly the copy CLAUDE.md's
`opsin.py` docstring warns about ("two copies of one rule drift; three copies
of one bug is worse"), so there is only one.

`_fetch`'S OWN CACHE CANNOT BE REUSED, ONLY ITS CALL. `gp_images.image_urls`
caches the URL MAP it parses out of the page, not the page itself — so a
second HTML-derived question (this module's compound-block microdata) cannot
be answered from that cache without a second fetch. This module therefore
keeps its OWN cache, `_CACHE_DIR` (`output_v3/gp_names/`, beside and never
inside `config.GP_IMAGE_DIR`), holding the PARSED BLOCKS rather than the raw
page — smaller, and it is all any caller here ever needs.

WHAT A COMPOUND BLOCK LOOKS LIKE
----------------------------------
Verified against five patents' live HTML (US10004738, US10030020,
US10214537, US10730877, US9718825; 2026-08-19):

    <li itemprop="match" itemscope repeat>
      <span itemprop="id">...</span>
      <span itemprop="name">...</span>
      <span itemprop="domain">...</span>
      <span itemprop="svg_large"></span>
      <span itemprop="svg_small"></span>
      <span itemprop="smiles">...</span>
      <span itemprop="inchi_key">...</span>
      <span itemprop="similarity">...</span>
      <span itemprop="sections" repeat>...</span>   (0 or more)
      <span itemprop="count">...</span>
    </li>

one `<li>` per annotated concept — the same eleven fields `gp_images.py`'s
docstring names, confirmed present verbatim.

THE MEASURED DECISION: "GP NAMES ARE NOT ALL COMPOUNDS"
-----------------------------------------------------------
`gp_images.py`'s docstring already warned that `domain` reads "Chemical
class" on some blocks whose `name` is a bare scaffold or the literal string
"compounds". Measured over the same five patents, 4,927 blocks total:

    domain                total   with GP's own `smiles`
    Chemical compound      2,300   2,300   (100%)
    Substances                515      12
    Diseases                  363       0
    Chemical group             329     146
    Effects                    284       0
    Methods                    224       0
    Drugs                      182       0
    Nutrition                  163       0
    Inorganic materials         92       4
    Chemical class              90      20
    Proteins                    90       0
    ... (7 more domains, all 0 with smiles)
    -----------------------------------------------------
    overall                  4,927   2,523   (51.2%)

`domain` is NOT a clean compound/non-compound split by itself: the "Chemical
class" bucket contains BOTH a fully elaborated target name with a GP-supplied
SMILES (`8-[3-(1-cyclopropylpyrazol-4-yl)-1H-pyrazolo[4,3-d]pyrimidin-5-yl]-
3-methyl-3,8-diazabicyclo[3.2.1]octan-2-one`) AND the bare literal `compounds`
with no SMILES at all — same domain string, opposite cases. A domain-string
allowlist would therefore have to special-case exactly the failure
`gp_images.py` already flagged.

**The decision: prefilter on the block's OWN `smiles` field being non-empty,
not on `domain`.** That field is GP's own claim "we grounded this text span to
a molecule", and it lines up with the chemistry-bearing domains far more
cleanly than the domain label itself does — every domain outside the
chemistry cluster above is at or near 0% with a SMILES, while "Chemical
compound" is 100%. This is stated plainly as an EFFICIENCY prefilter, not a
correctness gate: it exists only to avoid sending ~2,400 obviously non-
chemical annotations ("anti-inflammatory agent", "receptor antagonist") to a
Java subprocess that would refuse every one. It changes nothing about
correctness, because the real accept/reject decision is the one design
invariant this module leans on hardest —

**OPSIN is the hard gate, per the task's own step order.** A block that
clears the prefilter but is not a real compound name (a bare scaffold like
"1h-pyrrolo[3,2-b]pyridine", or a substituent fragment like "phenyl group")
either (a) resolves to a real, small molecule — which OPSIN is entitled to
do, since it IS one, and this module ships it exactly as it ships any other
resolved structure, labeled by `sources/reagents.py` (see below) — or (b) is
a bare substituent name, which `opsin.batch` (not `opsin.radicals`) refuses
BY DESIGN per that module's own docstring ("a name is just a substituent").
No heuristic here tries to out-guess that; `opsin.py` already draws the line
this module needs.

REAGENTS ARE LABELED, NOT DROPPED
-----------------------------------
A GP-annotated name can be a real, OPSIN-resolvable molecule that is still a
laboratory reagent rather than one of the patent's own compounds — a solvent,
a base, a coupling reagent named in an experimental paragraph and picked up
by GP's own entity tagger the same way a target compound is. `sources/
reagents.py` already exists to tell the two apart, and its OWN documented
design is "LABEL, never delete" (see that module's docstring, `THE LABEL
DESIGN`) — this module reuses `reagents.classify(name, smiles)` for exactly
that reason: it never drops a structure this module resolved, only tags it,
so a caller can filter afterward with full information rather than have that
judgment call made silently at import time.

DEDUP: TWO PASSES, ONE KEY (InChIKey), SCOPED PER PATENT
------------------------------------------------------------
CLAUDE.md's brief: "Drop any name whose structure we already hold. Our
structures are in `structures.tsv`, column `inchikey`, PER `patent_id`." That
scoping is deliberate and is followed literally — a structure already held
under one patent does not block a GP name in a DIFFERENT patent from being
added, because `structures.tsv` states everything per patent already
(`build_columns`'s reach and the corpus's own compound numbering are both
patent-scoped; nothing here changes that).

Two passes, because GP's OWN `inchi_key` field, when present, is a FREE dedup
key available before any OPSIN call is spent:

  1. **Pre-OPSIN.** Drop a block whose own `inchi_key` (GP's, not ours) is
     already in `structures.tsv` for this `patent_id`. A block with no GP
     `inchi_key` cannot be judged here and proceeds.
  2. **Post-OPSIN.** Of what survives and what OPSIN resolves, drop anything
     whose OPSIN-derived `StdInChIKey` is already held — catching the case
     GP's own key was blank, or (unmeasured, but structurally possible)
     disagreed with what OPSIN computes for the same name. This is the
     authoritative check; pass 1 exists only to save the OPSIN call.

A `StdInChIKey` is fetched with a SECOND `opsin.batch` call over the same
name list, in the same order the SMILES batch used — exactly the pattern
`sources/iupac_names.py` already uses (one batch for SMILES, one for
InChIKey, over the same list) and NOT `opsin.errors`/one-name-at-a-time,
which would multiply subprocess launches for no reason.

NO COMPOUND NUMBER, ON PURPOSE
---------------------------------
Per CLAUDE.md: this output carries no `cid` and joins no assay row. It is
finished structures with no measurement attached. `GPStructure` therefore has
no `cid` field at all — not blank, absent — so nothing downstream can
mistake it for a joinable record by looking at the schema.

THE OUTPUT ARTIFACT
----------------------
`OUT_PATH` = `patentdb3/out/gp_names.tsv` — a NEW path. `DUMP`, `STRUCTURES`
and `MANIFEST` (`core/config.py`) are `verify.dump()`'s own three artifacts
and this module writes none of them; it is not called from `verify.py` and
nothing there was changed to accommodate it.

GATING AND CACHING
----------------------
Gated on `config.GP_ENABLED`, the SAME switch `gp_images.py` uses — not a
second flag. Off (the default), `compound_blocks` returns `[]` and touches
neither the network nor `_CACHE_DIR`. On, a page is fetched once per patent
and its parsed blocks are cached at `_CACHE_DIR/<patent_id>.json`; a second
call reads the cache and fetches nothing.
"""
from __future__ import annotations

import csv
import html as _html
import json
import logging
import re
import urllib.error
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..core import config
from . import losses as _losses
from . import opsin as _opsin
from . import reagents as _reagents
from .gp_images import _fetch as _gp_fetch

logger = logging.getLogger(__name__)

# A NEW cache directory, deliberately not `config.GP_IMAGE_DIR` — see module
# docstring ("`_fetch`'S OWN CACHE CANNOT BE REUSED"). Module-level so a test
# can `monkeypatch.setattr(gp_names, "_CACHE_DIR", tmp_path / ...)` exactly as
# `test_gp_images.py` does for `config.GP_IMAGE_DIR`.
_CACHE_DIR = config.OUTPUT_DIR / "gp_names"

# This module's own artifact. Not `config.STRUCTURES` — that file is owned by
# `verify.dump()` and this module is never called from there. A new filename
# beside it, in the same `out/` directory `verify.py` already treats as its
# output area.
OUT_PATH = config.PACKAGE_ROOT / "out" / "gp_names.tsv"

# `main()`'s OWN loss sink — NEVER `losses.LOSS_LOG`. `sources/losses.py`
# truncates its shared file on the FIRST `record()` call of a PROCESS (see
# that module's docstring, "TRUNCATE ON FIRST WRITE, PER PROCESS"), which is
# correct for `verify.dump()`, the one caller it was built for, and wrong for
# anything else that shares the same file: a standalone run of this module IS
# a separate process, so its first `record()` call truncates whatever the
# last real `verify.dump()` run wrote — discovered the hard way while
# measuring this module (see the task report): a same-day 88,194-event
# corpus loss log was replaced by 3,077 `gp_name_*` records, with nothing
# raising. `main()` calls `losses.reset(_LOSS_LOG)` before doing any work, for
# exactly the reason `tests/conftest.py`'s session fixture does the same
# thing for the test suite — the production path must be unreachable from a
# process that does not own it, not just "remembered not to write there".
_LOSS_LOG = config.OUTPUT_DIR / "gp_names" / "loss_log.jsonl"


# ── parsing the compound-block microdata ────────────────────────────────

_BLOCK_RE = re.compile(r'<li itemprop="match" itemscope repeat>(.*?)</li>', re.S)


def _field_re(tag: str) -> re.Pattern[str]:
    return re.compile(rf'<span itemprop="{tag}">(.*?)</span>', re.S)


_F_ID = _field_re("id")
_F_NAME = _field_re("name")
_F_DOMAIN = _field_re("domain")
_F_SMILES = _field_re("smiles")
_F_INCHIKEY = _field_re("inchi_key")


@dataclass(frozen=True)
class GPBlock:
    """One `<li itemprop="match">` annotation, straight off the page.

    `smiles` and `inchikey` are GP's OWN values — never touched by OPSIN.
    Both are commonly `""`; see module docstring for what that means and how
    it is used (an efficiency prefilter and a free pre-OPSIN dedup key, never
    the structure this module ships).
    """
    id: str
    name: str
    domain: str
    smiles: str
    inchikey: str


def _text(rx: re.Pattern[str], block: str) -> str:
    m = rx.search(block)
    return _html.unescape(m.group(1)).strip() if m else ""


def _parse_blocks(html: str) -> list[GPBlock]:
    """Every compound-annotation block on a Google Patents page, in document
    order, duplicates and all — dedup is the caller's job (`new_structures`
    dedups on NAME, which is a decision the parser should not make for it).
    """
    out: list[GPBlock] = []
    for raw in _BLOCK_RE.findall(html):
        out.append(GPBlock(
            id=_text(_F_ID, raw), name=_text(_F_NAME, raw),
            domain=_text(_F_DOMAIN, raw), smiles=_text(_F_SMILES, raw),
            inchikey=_text(_F_INCHIKEY, raw)))
    return out


def _cache_path(patent_id: str) -> Path:
    return _CACHE_DIR / f"{patent_id}.json"


def compound_blocks(patent_id: str, *, allow_fetch: bool = True) -> list[GPBlock]:
    """Every GP compound-annotation block for `patent_id`. `[]` — never an
    exception — when the flag is off, the network is unavailable, GP has no
    page, or the page has no chemical annotations. Same contract as
    `gp_images.image_urls`, for the same reason: a caller cannot and should
    not try to tell those apart from the return value alone.
    """
    if not config.GP_ENABLED:
        return []

    path = _cache_path(patent_id)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            return [GPBlock(**b) for b in raw]
        except (OSError, ValueError, TypeError) as e:
            logger.warning("gp_names: %s — unreadable cache %s: %r",
                           patent_id, path, e)
            # fall through and re-fetch, exactly as `gp_images` does: a
            # corrupt cache is not a reason to lose the patent
    if not allow_fetch:
        return []

    try:
        html = _gp_fetch(patent_id)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning("gp_names: %s — fetch failed: %r", patent_id, e)
        return []

    blocks = _parse_blocks(html)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(b) for b in blocks], indent=1))
    except OSError as e:
        logger.warning("gp_names: %s — could not write cache: %r", patent_id, e)
    logger.info("gp_names: %s — %d compound block(s)", patent_id, len(blocks))
    return blocks


# ── what we already hold ─────────────────────────────────────────────────

def load_held_inchikeys(structures_path: "str | Path | None" = None
                        ) -> dict[str, set[str]]:
    """`{patent_id -> {inchikey, ...}}` read from `structures.tsv` (default
    `config.STRUCTURES`). Normalized upper/stripped; rows with no `inchikey`
    contribute nothing. `structures_path` exists so a test — or a caller
    measuring against a specific dump — never has to touch the real corpus
    artifact to exercise this.
    """
    path = Path(structures_path) if structures_path else config.STRUCTURES
    held: dict[str, set[str]] = {}
    if not path.exists():
        return held
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            pid = (row.get("patent_id") or "").strip()
            ik = (row.get("inchikey") or "").strip().upper()
            if pid and ik:
                held.setdefault(pid, set()).add(ik)
    return held


# ── the pipeline ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GPStructure:
    """One net-new structure. No `cid` field — see module docstring, "NO
    COMPOUND NUMBER, ON PURPOSE": this is deliberate, not an omission.
    """
    patent_id: str
    gp_name: str
    domain: str
    gp_smiles: str
    gp_inchikey: str
    smiles: str        # OPSIN's own SMILES for `gp_name`
    inchikey: str        # OPSIN's own StdInChIKey for `gp_name`
    label: str            # sources.reagents.classify(...).label
    reason: str            # sources.reagents.classify(...).reason


FIELDS: tuple[str, ...] = tuple(f.name for f in fields(GPStructure))


@dataclass
class Stats:
    """One row of the yield table CLAUDE.md's brief asks for. Every field
    names the function/stage that produced it — see `new_structures`.
    """
    patent_id: str
    gp_blocks: int = 0          # `compound_blocks` — every annotation on the page
    gp_grounded: int = 0        # blocks with GP's own `smiles` (the prefilter)
    distinct_names: int = 0     # after dedup on NAME (one OPSIN call each)
    dedup_dropped_pre: int = 0  # dropped: GP's own inchikey already held
    opsin_sent: int = 0         # names actually sent to OPSIN
    opsin_kept: int = 0         # OPSIN resolved (SMILES and StdInChIKey both)
    opsin_refused: int = 0      # OPSIN refused (either representation empty)
    dedup_dropped_post: int = 0  # dropped: OPSIN's own inchikey already held
    net_new: int = 0            # rows actually returned


def new_structures(patent_id: str, *, allow_fetch: bool = True,
                    held: "dict[str, set[str]] | None" = None,
                    blocks: "list[GPBlock] | None" = None,
                   ) -> tuple[list[GPStructure], Stats]:
    """The whole pipeline for one patent: GP blocks -> prefilter -> dedup ->
    OPSIN -> dedup -> `GPStructure` rows. See module docstring for what each
    stage does and why it is ordered this way.

    `held` lets a multi-patent caller load `structures.tsv` ONCE (see `main`
    below) instead of once per patent. `blocks` lets a test inject blocks
    directly, bypassing `compound_blocks` (and therefore `config.GP_ENABLED`)
    entirely — the dedup/OPSIN/label logic below is what this module actually
    needs to prove correct without a network or a live GP page.
    """
    stats = Stats(patent_id=patent_id)
    blocks = compound_blocks(patent_id, allow_fetch=allow_fetch) if blocks is None else blocks
    stats.gp_blocks = len(blocks)
    if not blocks:
        return [], stats

    held_all = held if held is not None else load_held_inchikeys()
    held_here: set[str] = set(held_all.get(patent_id, ()))

    # THE MEASURED PREFILTER. See module docstring: GP's own `smiles` field,
    # not `domain`, because `domain` alone does not separate "compounds"
    # (Chemical class, no smiles) from a fully elaborated target name (also
    # Chemical class, WITH smiles). Efficiency only — OPSIN is the real gate.
    grounded = [b for b in blocks if b.smiles]
    stats.gp_grounded = len(grounded)

    # One OPSIN call per distinct NAME. A reagent mentioned forty times in
    # one patent's annotations should cost one call, not forty; first
    # occurrence wins, which is immaterial here since every occurrence of the
    # same name carries the same GP fields by construction (GP deduplicates
    # its own annotations per concept before annotating the page).
    by_name: dict[str, GPBlock] = {}
    for b in grounded:
        by_name.setdefault(b.name, b)
    stats.distinct_names = len(by_name)

    # PASS 1 (pre-OPSIN, free): GP's own inchi_key against what we hold.
    survivors: list[GPBlock] = []
    for b in by_name.values():
        ik = b.inchikey.strip().upper()
        if ik and ik in held_here:
            stats.dedup_dropped_pre += 1
            if _losses.ENABLED:
                _losses.record("gp_name_already_held", patent_id,
                               name=b.name, inchikey=ik, stage="pre_opsin")
            continue
        survivors.append(b)

    if not survivors:
        return [], stats

    names = [b.name for b in survivors]
    stats.opsin_sent = len(names)
    # Two batches over the SAME list, positionally paired — the pattern
    # `sources/iupac_names.py` already uses. `opsin.batch` refuses a WHOLE
    # batch (all "") on any length mismatch rather than risk a shifted
    # pairing; see `sources/opsin.py`'s own docstring for why.
    smiles = _opsin.batch(names, "SMILES", patent_id)
    keys = _opsin.batch(names, "StdInChIKey", patent_id)

    out: list[GPStructure] = []
    for b, smi, ik in zip(survivors, smiles, keys):
        if not smi or not ik:
            stats.opsin_refused += 1
            if _losses.ENABLED:
                _losses.record("gp_name_opsin_refused", patent_id, name=b.name)
            continue
        stats.opsin_kept += 1

        # PASS 2 (post-OPSIN, authoritative): OPSIN's own key against what we
        # hold, including anything this same call already added — two GP
        # names that are synonyms of one molecule must not both survive.
        ik_u = ik.strip().upper()
        if ik_u in held_here:
            stats.dedup_dropped_post += 1
            if _losses.ENABLED:
                _losses.record("gp_name_already_held", patent_id,
                               name=b.name, inchikey=ik_u, stage="post_opsin")
            continue
        held_here.add(ik_u)

        verdict = _reagents.classify(b.name, smi)
        out.append(GPStructure(
            patent_id=patent_id, gp_name=b.name, domain=b.domain,
            gp_smiles=b.smiles, gp_inchikey=b.inchikey.strip().upper(),
            smiles=smi, inchikey=ik_u, label=verdict.label,
            reason=verdict.reason))

    stats.net_new = len(out)
    return out, stats


def write_tsv(rows: list[GPStructure], path: "str | Path | None" = None) -> Path:
    """Write `rows` to `path` (default `OUT_PATH`), overwriting it — the same
    rule `DUMP`/`STRUCTURES` follow: one canonical file, replaced whole every
    run, never appended to.
    """
    out = Path(path) if path else OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([getattr(r, f) for f in FIELDS])
    return out


# ── measurement / CLI ────────────────────────────────────────────────────
#
# Not a `verify.py`/`cli.py` subcommand and not a second file — CLAUDE.md
# caps this task at ONE new file, and no module under `sources/` carries its
# own `__main__` (that pattern lives only in the package's top-level scripts:
# `verify.py`, `decimer.py`, `to_excel.py`, `cli.py`). This is a measurement
# tool for exactly the question CLAUDE.md's brief asks — the yield table per
# patent — kept small and scoped to this file rather than growing a second
# entry point elsewhere.
def main(argv: "list[str] | None" = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="python3 -m patentdb3.sources.gp_names",
        description="Measure (and optionally write) net-new OPSIN structures "
                    "from Google Patents' own chemical-name annotations.")
    ap.add_argument("patent_ids", nargs="+", help="e.g. US10004738")
    ap.add_argument("--fetch", action="store_true",
                    help="allow a network fetch for a patent not already in "
                         "the cache (needs GP_ENABLED=1 either way)")
    ap.add_argument("--write", action="store_true",
                    help=f"write survivors to {OUT_PATH}")
    ap.add_argument("--sample", type=int, default=0,
                    help="print this many (gp_name, inchikey) survivors")
    a = ap.parse_args(argv)

    # REDIRECT BEFORE ANY `record()` CALL — see `_LOSS_LOG` above. This is
    # the one line standing between a standalone run of this module and
    # truncating `verify.dump()`'s own production loss log.
    _losses.reset(_LOSS_LOG)

    if not config.GP_ENABLED:
        print("GP_ENABLED=0 — this module spends nothing and returns no "
              "rows. Set GP_ENABLED=1 to read Google Patents' own name "
              "annotations.")

    held = load_held_inchikeys()
    all_rows: list[GPStructure] = []
    hdr = (f"{'patent':<14}{'gp_blocks':>10}{'grounded':>9}{'distinct':>9}"
           f"{'dedup_pre':>10}{'sent':>7}{'kept':>7}{'refused':>9}"
           f"{'dedup_post':>11}{'net_new':>9}")
    print(hdr)
    for pid in a.patent_ids:
        rows, stats = new_structures(pid, allow_fetch=a.fetch, held=held)
        all_rows.extend(rows)
        print(f"{pid:<14}{stats.gp_blocks:>10}{stats.gp_grounded:>9}"
              f"{stats.distinct_names:>9}{stats.dedup_dropped_pre:>10}"
              f"{stats.opsin_sent:>7}{stats.opsin_kept:>7}"
              f"{stats.opsin_refused:>9}{stats.dedup_dropped_post:>11}"
              f"{stats.net_new:>9}")

    if a.sample and all_rows:
        print(f"\nsample of {min(a.sample, len(all_rows))} survivor(s):")
        for r in all_rows[:a.sample]:
            print(f"  {r.gp_name[:70]:<72}{r.inchikey}")

    if a.write:
        out = write_tsv(all_rows)
        print(f"\nwrote {len(all_rows):,} structure(s) -> {out}")

    _losses.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
