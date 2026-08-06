"""Step 2.2 — IUPAC → SMILES via a fault-tolerant cascade.

Stage 0:  PubChem name lookup (free, network, stereo-aware)
Stage 1:  OPSIN direct (free, deterministic, correct if it succeeds)
Stage 2:  Rule-based cleaning → OPSIN retry (free)
Stage 2d: Relaxed OPSIN — permissive flags, stereo may be dropped (free)
Stage 3a: LLM normalize the name → OPSIN retry   (OFF: LLM_NAME_REPAIR)
Stage 3b: LLM direct SMILES, last resort         (OFF: LLM_NAME_REPAIR)

Two further stages used to sit between 2 and 2d and are gone, because
neither could ever execute:

  - **Levenshtein autocorrect** imported `.ocr_autocorrect`, a module that
    has never existed here, inside `except ImportError: pass`. It raised
    and was swallowed on every call.
  - **Vision OCR** required `compound.source_page`, which no constructor
    feeding this cascade sets (see
    `tests/test_cascade_dead_stages_removed.py` for the trace).

Both existed to repair MinerU OCR damage, and MinerU is no longer read.
Removing them changed 0 records across all 22 corpus patents.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from py2opsin import py2opsin
import threading
import os
import tempfile
import uuid
import warnings
from typing import Iterable

# py2opsin uses a shared temp file — not thread-safe AND not
# process-safe. The default temp filename is `py2opsin_temp_input.txt`
# in the cwd; multiple parallel patent runs all write to the same path
# and clobber each other's input, causing OPSIN to return SMILES for
# the WRONG patent's compound. We work around this by giving each
# process its own temp file (PID-stamped) AND serializing access from
# threads within the same process.
_opsin_lock = threading.Lock()
_OPSIN_TMP_FPATH = os.path.join(
    tempfile.gettempdir(),
    f"py2opsin_input_{os.getpid()}.txt",
)

from . import config
from .api_client import call_claude_text
from .cost_tracker import cost_tracker
from .models import Compound, MarkushContext
from .smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey,
    strip_salt, compute_drug_likeness, molecular_weight,
)

logger = logging.getLogger(__name__)

MAX_PARALLEL = 10
MIN_SMILES_MW = 150    # Catches iodine (127.9) and other single-atom SMILES
MAX_SMILES_MW = 1500   # Catches polymer/garbage SMILES from DECIMER
MIN_SMILES_LENGTH = 10 # Drug molecules have SMILES ≥10 chars; "I", "Cl", "CC" are not drugs

# OPSIN failure log
OPSIN_FAILURES_LOG = config.LOGS_DIR / "opsin_failures.jsonl"


# ============================================================
# Stage 2: Rule-based IUPAC cleaning
# ============================================================

def rule_based_clean(name: str) -> str:
    """Deterministic fixes for known OPSIN failure patterns.

    Built from empirical analysis of US10214537 (169 compounds):
    - pyrolo→pyrrolo: Claude drops an 'r' during extraction (30+ compounds)
    - pyranyl→pyran: extraction artifact
    - thionorpholine→thiomorpholine: OPSIN needs standard name
    - Salt suffix stripping
    - Stereo prefix normalization
    - Misplaced parentheses from Claude extraction
    """
    n = name.strip()

    # --- GENERALIZABLE FIXES (safe across all patents) ---

    # Fix common LLM/OCR typos in ring names
    n = n.replace('isodolin', 'isoindolin')  # Missing 'in' in isoindoline
    n = n.replace('Isodolin', 'Isoindolin')
    n = n.replace('isquinolin', 'isoquinolin')  # Missing 'o' in isoquinoline
    n = n.replace('Isquinolin', 'Isoquinolin')
    n = n.replace('benzontir', 'benzonitr')  # Transposed letters in benzonitrile
    n = n.replace('Benzontir', 'Benzonitr')
    n = n.replace('benzotir', 'benzonitr')   # Another variant
    n = n.replace('isoquinoin', 'isoquinolin')  # Missing 'l'
    n = n.replace('pyrolo[', 'pyrrolo[')    # Dropped double-r
    n = n.replace('Pyrolo[', 'Pyrrolo[')
    n = n.replace('pyro1o[', 'pyrrolo[')    # OCR reads l as 1
    n = n.replace('Pyro1o[', 'Pyrrolo[')
    n = n.replace('pyrran', 'pyran')         # Extra r
    n = re.sub(r'pyranyl-(\d)-yl', r'pyran-\1-yl', n)  # pyranyl-4-yl → pyran-4-yl
    n = n.replace('thionorpholine', 'thiomorpholine')  # Wrong ring name

    # Fix table OCR artifacts: spaces inside compound names
    n = re.sub(r'\)\s+pyrrolo\[', ')pyrrolo[', n)   # ") pyrrolo[" → ")pyrrolo["
    n = re.sub(r'\)\s+pyrolo\[', ')pyrrolo[', n)     # ") pyrolo[" → ")pyrrolo["
    n = re.sub(r'\)\s+Pyrrolo\[', ')Pyrrolo[', n)

    # Fix OCR periods in fusion descriptors: [1.2,4] → [1,2,4]
    n = re.sub(r'\[(\d+)\.(\d+),(\d+)\]', r'[\1,\2,\3]', n)  # [1.2,4] → [1,2,4]
    n = re.sub(r'\[(\d+)\.,(\d+),(\d+)\]', r'[\1,\2,\3]', n)  # [1.,2,4] → [1,2,4]

    # Fix missing locant in fusion: [2,4] → [1,2,4] (only for triazin context)
    n = re.sub(r'pyrrolo\[2,1-f\]\[2,4\]', 'pyrrolo[2,1-f][1,2,4]', n)
    n = re.sub(r'pyrrolo\[2,1-f\]\[2\.4\]', 'pyrrolo[2,1-f][1,2,4]', n)

    # Fix OCR: [1,4] in triazin context → [1,2,4]
    n = re.sub(r'\[1,4\]triazin', '[1,2,4]triazin', n)

    # Fix more table OCR artifacts
    n = n.replace('pyrrolo[2,1-1]', 'pyrrolo[2,1-f]')   # f OCR'd as 1
    n = n.replace('pyrrolo[2.1-f]', 'pyrrolo[2,1-f]')   # period in fusion
    n = n.replace('pyrzol', 'pyrazol')                    # missing 'a'
    n = re.sub(r'pyrrolo\s+\[', 'pyrrolo[', n)           # space before [
    # Fix missing opening bracket: pyrrolo[2,1-f]1,2,4] → pyrrolo[2,1-f][1,2,4]
    n = re.sub(r'pyrrolo\[2,1-f\](\d)', r'pyrrolo[2,1-f][\1', n)
    # Fix [1]2,4] → [1,2,4]
    n = re.sub(r'\[1\]2,4\]', '[1,2,4]', n)

    # Fix misplaced parens: "pyrazol-5-(yl)" → "pyrazol-5-yl)" (Claude artifact)
    n = re.sub(r'-(\d+)-\(yl\)', r'-\1-yl)', n)

    # Strip salt suffixes (OPSIN can't parse these)
    n = re.sub(r',\s*\d*\s*HCl$', '', n)
    n = re.sub(r',\s*\d*\s*TFA$', '', n)
    n = re.sub(r'\s+as the \w+ salt$', '', n, flags=re.IGNORECASE)

    # Strip ONLY cis/trans prefixes (OPSIN can't resolve these)
    # KEEP R/S/E/Z prefixes — OPSIN handles them correctly (tested on 21 compounds)
    n = re.sub(r'^\(\((?:C|c)is\)\)-?', '', n)   # ((Cis))- artifact
    n = re.sub(r'^\((cis|trans)\)-', '', n, flags=re.IGNORECASE)

    # Strip a leading racemate marker. OPSIN parses "cis-2-{...}" but not
    # "racemic cis-2-{...}", so this one word was failing otherwise-clean
    # names: on US8952177's compound table it alone blocked 16 of 35 names
    # (54% -> 100% parse rate). It carries no connectivity — it says the
    # sample is a racemate, which the relative cis/trans descriptor that
    # follows already implies — so removing it loses nothing OPSIN would
    # have used. Bare "rac-"/"(±)-"/"(+/-)-" are the same statement.
    n = re.sub(r'^\s*(racemic|rac|\(\s*[±±]\s*\)|\(\s*\+/-\s*\))[-\s]+',
               '', n, flags=re.IGNORECASE)

    # NOTE: Fusion descriptors ([1,2-f] vs [2,1-f]) and locants (triazin-9 vs -7)
    # are NOT corrected here — they specify different molecules. Changing them
    # would silently produce wrong structures. These go to the LLM fallback.

    return n


# ============================================================
# Stage 3: LLM cleaning with error context
# ============================================================

def _try_pubchem(name: str) -> str | None:
    """Try PubChem name lookup for stereo-aware SMILES.

    PubChem has 111M+ compounds with resolved stereochemistry.
    This bypasses OPSIN's stereo limitations.

    OFF by default — see `config.PUBCHEM_NAME_LOOKUP_ENABLED` for the
    measurement. One uncached HTTPS round-trip per name at 24.3% of traced
    wall time, for 11 of 18,039 shipped records, none of which the free
    OPSIN cascade fails to reproduce.
    """
    if not config.PUBCHEM_NAME_LOOKUP_ENABLED:
        return None

    try:
        import pubchempy as pcp
        results = pcp.get_compounds(name, 'name')
        if results and len(results) > 0:
            smiles = results[0].isomeric_smiles or results[0].canonical_smiles
            if smiles and validate_smiles(smiles) and len(smiles) >= MIN_SMILES_LENGTH:
                mw = molecular_weight(smiles)
                if mw and mw >= MIN_SMILES_MW:
                    return smiles
    except Exception:
        pass
    return None


CLEANING_PROMPT ="""OPSIN (a deterministic IUPAC-to-structure parser) failed to parse this chemical name. Your job: normalize the name into the form OPSIN can parse, WITHOUT changing the molecule's identity.

Raw name (from patent): {raw_name}
OPSIN error: {error}

Normalization rules (apply only the ones needed):
1. Fix typos in standard ring names (pyrolo→pyrrolo, isodolin→isoindolin, thionorpholine→thiomorpholine, pyrzol→pyrazol).
2. Balance parentheses and square brackets — every ( needs a ), every [ needs a ]. Missing brackets are extraction artifacts; reconstruct from context, do NOT delete substituents.
3. Replace non-standard fusion notation with IUPAC equivalents:
   - "benzenacyclo..." → equivalent IUPAC bridged-ring name
   - "X-aza-Y-oxa-bridged" → use the systematic von Baeyer name
4. Strip salt suffixes (", HCl", ", 2 TFA", "as the hydrochloride salt").
5. Normalize stereo prefixes only if they BLOCK parsing — NEVER strip R/S/E/Z descriptors that OPSIN handles (only ((cis))/((trans)) artifacts).
6. Preserve every locant and substituent EXACTLY. Do not "guess" a different regiochemistry to make OPSIN happy — that produces a different molecule.

Output ONLY the corrected IUPAC name on a single line. No prose, no explanation, no code fences."""

SMILES_FALLBACK_PROMPT = """Convert this IUPAC chemical name to a canonical SMILES string.
The OPSIN parser cannot handle this name, so generate the SMILES directly.
Preserve all stereochemistry (R/S, E/Z, cis/trans).

Name: {raw_name}

Output ONLY the SMILES string. Nothing else."""


def _llm_clean(raw_name: str, error_msg: str, patent_id: str, compound_id: str) -> str | None:
    """Stage 3a: Use the default model (Sonnet) to normalize an IUPAC name
    OPSIN can't parse.

    Name normalization / SMILES recovery does NOT need Opus — Sonnet
    handles non-standard nomenclature and macrocycles at 1/5th the cost
    ($3/$15 vs $15/$75 per Mtok). Cost is further bounded by the
    per-patent LM cap (config.PER_PATENT_LM_CAP).
    """
    if not config.LLM_NAME_REPAIR_ENABLED:
        # OFF by default — see config.LLM_NAME_REPAIR_ENABLED for the
        # measurement. This paid a model to rewrite a name OPSIN rejected; 86.1% of what it
        # was credited with resolves for free through OPSIN.
        return None

    prompt = CLEANING_PROMPT.format(error=error_msg, raw_name=raw_name)

    response = call_claude_text(
        prompt=prompt,
        model=config.DEFAULT_MODEL,
        patent_id=patent_id,
        compound_id=f"{compound_id}_clean",
        max_tokens=300,
    )

    if response:
        return response.strip().split("\n")[0].strip()
    return None


def _llm_direct_smiles(raw_name: str, patent_id: str, compound_id: str) -> str | None:
    """Stage 3b: Last resort — ask the default model (Sonnet) to generate
    SMILES directly.

    Only used when OPSIN fundamentally can't parse the ring system
    (e.g., pyrrolo[1,2-f] fusion descriptors). Sonnet is sufficient and
    5× cheaper than Opus; verified on US10273259 where the batched
    Sonnet/Opus fallback recovered 9/15 OPSIN-unparseable macrocyclic
    names.
    """
    if not config.LLM_NAME_REPAIR_ENABLED:
        # OFF by default — see config.LLM_NAME_REPAIR_ENABLED for the
        # measurement. This paid a model to guess a SMILES from a name; 86.1% of what it
        # was credited with resolves for free through OPSIN.
        return None

    prompt = SMILES_FALLBACK_PROMPT.format(raw_name=raw_name)

    response = call_claude_text(
        prompt=prompt,
        model=config.DEFAULT_MODEL,
        patent_id=patent_id,
        compound_id=f"{compound_id}_smiles_fallback",
        max_tokens=200,
    )

    if response:
        smiles = response.strip().split("\n")[0].strip()
        if validate_smiles(smiles) and len(smiles) >= MIN_SMILES_LENGTH:
            mw = molecular_weight(smiles)
            if mw and mw >= MIN_SMILES_MW:
                return smiles
    return None


# ============================================================
# Core conversion pipeline
# ============================================================

# OCR-markup signatures we DON'T want fed to OPSIN. Even if OPSIN
# happens to ignore the tags and parse the underlying name correctly,
# the SMILES it returns is for whatever substring it managed to parse —
# which may be wrong without us realizing. Strict mode rejects these
# outright so the cascade falls through to OCR-cleanup stages.
_OPSIN_INPUT_GARBAGE_PAT = re.compile(
    r"<\|[a-z/_]+\|>"           # MinerU detection-tag pollution
    r"|\[\[\s*\d+\s*,\s*\d+",   # MinerU bbox-coord block: [[114,99,...
    re.IGNORECASE,
)


# ============================================================
# OPSIN, batched — one JVM for a patent instead of one per name
# ============================================================
#
# `py2opsin` shells out to `subprocess.run(["java", "-jar", opsin-cli.jar…])`
# on EVERY call (py2opsin.py:137). A call trace over three patents measured
# 3,726 `_try_opsin` calls at ~0.19 s each = 721.3 s, 58.9% of all wall time,
# and essentially all of that is JVM startup, not parsing.
#
# py2opsin already takes a list (py2opsin.py:41) and writes it one name per
# line (py2opsin.py:117-121), so a whole patent costs one JVM. What a list
# call does NOT return is per-name stderr: py2opsin raises a single warning
# carrying the concatenated stderr of the entire run, and OPSIN's messages are
# not 1:1 with failures — 600 corpus names produced 54 empty results but 56
# stderr lines, and lines like "APPEARS_AMBIGUOUS: …" and "hydrogen addition
# at locant: 13 …" name no compound at all, so neither positional nor
# prefix attribution works. `_try_opsin`'s strict mode READS that message
# (an "Unmatched bracket" warning rejects an otherwise-valid parse, see the
# warning gate below), so losing attribution would silently change which
# structures ship.
#
# Attribution is therefore done with SENTINELS: an unparsable per-process
# nonce is written between every pair of real names. Each nonce yields one
# empty stdout line and at least one stderr line containing its own token, so
# the stderr stream partitions exactly, and the per-name message is rebuilt by
# inverting py2opsin's own formatting. Results land in `_OPSIN_MEMO`, which
# `_try_opsin` reads before shelling out — so a call site opts in by calling
# `prefetch_opsin` before its loop and changes nothing else.
#
# Every assumption is CHECKED at run time (stdout length, sentinel stdout
# empty, sentinels seen in ascending order, no stderr outside a sentinel
# span). Any violation discards the whole batch and leaves the memo untouched,
# which degrades to exactly the old per-name behaviour rather than to a
# mis-assigned structure. `tests/test_opsin_batch_identity.py` asserts the
# batched result is byte-identical to the per-call result over corpus names.

# (name, relaxed) -> (raw py2opsin result, py2opsin warning message)
_OPSIN_MEMO: dict[tuple[str, bool], tuple[str, str]] = {}
_opsin_memo_lock = threading.Lock()

# Names per JVM launch. Bounds the temp file and the blast radius of a
# verification failure — a rejected batch costs its own chunk, not the patent.
_OPSIN_BATCH_CHUNK = 500
# Entries kept before the memo is dropped wholesale. OPSIN is a pure function
# of (name, flags), so eviction is only a memory bound, never a correctness
# one. The 22-patent corpus holds 16,127 distinct names.
_OPSIN_MEMO_MAX = 200_000

# py2opsin.py:146 — the literal header it puts on every stderr warning.
_OPSIN_WARN_HEADER = "OPSIN raised the following error(s) while parsing:"
_OPSIN_WARN_SEP = "\n > "

# Underscores make the token unparsable to OPSIN under every flag combination
# (verified with allow_bad_stereo/allow_acid/allow_radicals all set); the
# nonce makes it impossible for patent text to collide with it.
_OPSIN_SENTINEL_NONCE = uuid.uuid4().hex[:12]


def _opsin_sentinel(i: int) -> str:
    return f"zq_{_OPSIN_SENTINEL_NONCE}_sep{i}_qz"


def _opsin_memo_store(items: dict[tuple[str, bool], tuple[str, str]]) -> None:
    with _opsin_memo_lock:
        if len(_OPSIN_MEMO) + len(items) > _OPSIN_MEMO_MAX:
            _OPSIN_MEMO.clear()
        _OPSIN_MEMO.update(items)


def clear_opsin_memo() -> None:
    """Drop every memoised OPSIN result. For tests that need a cold path."""
    with _opsin_memo_lock:
        _OPSIN_MEMO.clear()


def _opsin_call(names, *, relaxed: bool):
    """One py2opsin invocation. `names` is a str or a list of str.

    Returns (result, warning_messages). Raises whatever py2opsin raises —
    `_try_opsin` turns that into its (None, str(e)) return, as before.
    """
    with _opsin_lock:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = py2opsin(
                names,
                tmp_fpath=_OPSIN_TMP_FPATH,
                allow_bad_stereo=relaxed,
                allow_acid=relaxed,
                allow_radicals=relaxed,
            )
            messages = [str(w.message) for w in caught]
    return result, messages


def _opsin_raw(name: str, *, relaxed: bool) -> tuple[object, str]:
    """The raw OPSIN answer for ONE name: (py2opsin result, warning message).

    Served from `_OPSIN_MEMO` when `prefetch_opsin` already resolved the name,
    otherwise one JVM launch — identical to what `_try_opsin` did inline
    before batching existed.
    """
    key = (name, relaxed)
    hit = _OPSIN_MEMO.get(key)
    if hit is not None:
        return hit

    result, messages = _opsin_call(name, relaxed=relaxed)
    # py2opsin issues at most one warning per call; `caught[-1]` is what the
    # inline version read, so keep taking the last.
    entry = (result, messages[-1] if messages else "")
    if isinstance(result, str):
        _opsin_memo_store({key: entry})
    return entry


def _opsin_atoms(name: str) -> Iterable[str]:
    """The strings `_try_opsin(name)` will actually hand to OPSIN.

    A `;` multi-component name never reaches OPSIN whole — `_try_opsin`
    splits it and recurses on each part (see the multi-component block), so
    the parts are what a prefetch must resolve. Blank names and names holding
    a newline are excluded: a newline would break the one-name-per-line
    contract the batch depends on, and a truly empty name draws an "Input
    chemical name was blank!" line in a batch that a lone `py2opsin("")` does
    not, which would change the returned error text.
    """
    if not isinstance(name, str):
        return
    if ";" in name:
        parts = [p.strip() for p in name.split(";") if p.strip()]
        if len(parts) > 1:
            for p in parts:
                yield from _opsin_atoms(p)
            return
    if not name.strip():
        return
    if "\n" in name or "\r" in name:
        return
    yield name


def _opsin_stderr_lines(messages: list[str]) -> list[str] | None:
    """Undo py2opsin's warning formatting back to OPSIN's stderr lines.

    py2opsin.py:146 builds the message as
    HEADER + "\\n > " + stderr.replace("\\n", "\\n > ", stderr.count("\\n") - 1),
    i.e. every newline but the last becomes the separator. Splitting on that
    separator therefore recovers the original lines exactly. Returns None if
    anything other than a py2opsin warning landed in the record, so the caller
    can abandon the batch instead of guessing.
    """
    prefix = _OPSIN_WARN_HEADER + _OPSIN_WARN_SEP
    lines: list[str] = []
    for m in messages:
        if not m.startswith(prefix):
            return None
        body = m[len(prefix):]
        if body.endswith("\n"):
            body = body[:-1]
        lines.extend(body.split(_OPSIN_WARN_SEP))
    return lines


def _opsin_partition(err_lines: list[str], n_names: int) -> list[list[str]] | None:
    """Split OPSIN's stderr into one bucket per name, using the sentinels.

    Returns None — meaning "do not trust this batch" — if the sentinels are
    not all present in ascending order, or if a line falls outside any
    sentinel span.
    """
    buckets: list[list[str]] = [[] for _ in range(n_names)]
    tokens = [_opsin_sentinel(i) for i in range(n_names + 1)]
    current = -1        # index of the sentinel most recently opened
    next_expected = 0
    for ln in err_lines:
        if next_expected <= n_names and tokens[next_expected] in ln:
            current = next_expected
            next_expected += 1
            continue
        if current >= 0 and tokens[current] in ln:
            continue    # a sentinel whose own message ran to several lines
        if current < 0 or current >= n_names:
            return None  # stderr before the first / after the last sentinel
        buckets[current].append(ln)
    if next_expected != n_names + 1:
        return None      # a sentinel never reported — spans are unreliable
    return buckets


def _opsin_format_warning(lines: list[str]) -> str:
    """Rebuild the exact warning text py2opsin would have raised for a lone
    call whose stderr was `lines`."""
    if not lines:
        return ""
    return _OPSIN_WARN_HEADER + _OPSIN_WARN_SEP + _OPSIN_WARN_SEP.join(lines) + "\n"


def _opsin_batch(
    names: list[str], *, relaxed: bool,
) -> dict[str, tuple[str, str]] | None:
    """Resolve `names` in ONE JVM launch. Returns name -> (result, warning),
    or None when any structural check fails and the batch must be discarded.
    """
    lines: list[str] = []
    for i, n in enumerate(names):
        lines.append(_opsin_sentinel(i))
        lines.append(n)
    lines.append(_opsin_sentinel(len(names)))

    try:
        out, messages = _opsin_call(lines, relaxed=relaxed)
    except Exception as e:
        logger.warning(
            "OPSIN batch of %d names raised (%s) — resolving one at a time",
            len(names), e,
        )
        return None

    if not isinstance(out, list) or len(out) != len(lines):
        logger.warning(
            "OPSIN batch returned %s lines for %d inputs — discarding batch",
            len(out) if isinstance(out, list) else type(out).__name__, len(lines),
        )
        return None
    for i in range(0, len(lines), 2):
        if out[i]:
            logger.warning(
                "OPSIN batch: sentinel at line %d parsed to %r — discarding batch",
                i, out[i][:40],
            )
            return None

    err_lines = _opsin_stderr_lines(messages)
    if err_lines is None:
        logger.warning(
            "OPSIN batch: unrecognised warning in the record — discarding batch",
        )
        return None
    buckets = _opsin_partition(err_lines, len(names))
    if buckets is None:
        logger.warning(
            "OPSIN batch: stderr did not partition on the sentinels — "
            "discarding batch",
        )
        return None

    return {
        n: (out[2 * i + 1], _opsin_format_warning(buckets[i]))
        for i, n in enumerate(names)
    }


def prefetch_opsin(names: Iterable[str], *, relaxed: bool = False) -> int:
    """Resolve `names` through OPSIN in batched JVM launches.

    Fills the memo `_try_opsin` reads, so a caller that knows its names up
    front pays one JVM per `_OPSIN_BATCH_CHUNK` names instead of one per name.
    Call it before the loop; the loop body needs no change, and a name the
    batch could not cover simply resolves on its own as it always did.

    Returns the number of names newly resolved.
    """
    wanted: list[str] = []
    seen: set[str] = set()
    for name in names:
        for atom in _opsin_atoms(name):
            if atom in seen or (atom, relaxed) in _OPSIN_MEMO:
                continue
            seen.add(atom)
            wanted.append(atom)
    if not wanted:
        return 0

    n_new = 0
    for i in range(0, len(wanted), _OPSIN_BATCH_CHUNK):
        chunk = wanted[i:i + _OPSIN_BATCH_CHUNK]
        resolved = _opsin_batch(chunk, relaxed=relaxed)
        if resolved is None:
            continue        # discarded — `_try_opsin` falls back per name
        _opsin_memo_store({(n, relaxed): v for n, v in resolved.items()})
        n_new += len(resolved)
    logger.debug(
        "prefetch_opsin: %d names resolved in %d batches (relaxed=%s)",
        n_new, (len(wanted) + _OPSIN_BATCH_CHUNK - 1) // _OPSIN_BATCH_CHUNK,
        relaxed,
    )
    return n_new


def _opsin_unresolved(names: Iterable[str]) -> list[str]:
    """Names whose strict-flag OPSIN result is known and EMPTY — i.e. the ones
    that will go on to try the relaxed stage. Names the memo doesn't hold are
    excluded: their outcome isn't known yet, so prefetching relaxed for them
    would be guessing."""
    out = []
    for n in names:
        hit = _OPSIN_MEMO.get((n, False))
        if hit is not None and not hit[0]:
            out.append(n)
    return out


def prefetch_cascade(
    names: Iterable[str], *, rule_cleaned: bool = True, relaxed: bool = True,
) -> None:
    """Warm the memo for every OPSIN call the cascade will make over `names`.

    Mirrors `_convert_single`'s free stages — raw name, `rule_based_clean`
    retry, relaxed retry — so a caller with its names in hand pays three
    batched JVM launches per 500 names instead of up to four per name.

    The relaxed pass is issued ONLY for names both strict passes left empty,
    which is exactly the set `_convert_single` would carry into Stage 2d.
    Callers that never reach a stage turn it off: `rule_cleaned=False` for a
    loop that only tries the raw name, `relaxed=False` for a clean-text
    conversion, which skips Stage 2d entirely.
    """
    wanted = [n for n in names if n]
    if not wanted:
        return
    prefetch_opsin(wanted)
    cleaned: list[str] = []
    if rule_cleaned:
        cleaned = [rule_based_clean(n) for n in wanted]
        prefetch_opsin(cleaned)
    if relaxed:
        stuck = _opsin_unresolved(wanted) + _opsin_unresolved(cleaned)
        if stuck:
            prefetch_opsin(stuck, relaxed=True)


def _try_opsin(
    name: str, *, strict: bool = False, relaxed: bool = False,
) -> tuple[str | None, str]:
    """Try OPSIN conversion. Returns (smiles, error_msg).
    Thread-safe: py2opsin uses a shared temp file internally.

    `relaxed=True` enables OPSIN's own permissive flags:
      - allow_bad_stereo  — parse the connectivity and drop stereo
        descriptors OPSIN can't interpret, instead of failing the whole
        name. OCR mangles stereo terms far more often than it mangles
        ring systems, so this rescues names whose skeleton is intact.
      - allow_acid        — accept "acetic" for "acetic acid", a common
        truncation at a line break.
      - allow_radicals    — accept substituent-only fragments.

    This is deliberately a SECOND attempt, never the first: a relaxed
    parse can silently discard real stereochemistry, so we only reach for
    it once the strict parse has already failed. The alternative for
    those names is the LLM stages, which measured 0-35% exact match
    against BindingDB versus 37-71% for OPSIN paths — a connectivity-
    correct structure beats a paid guess.

    With strict=True, additional gates beyond "RDKit accepts the SMILES":
      - Input must not contain MinerU/OCR markup tags (<|ref|>, [[bbox]])
      - Input must not start with a non-letter (truncation hint)
      - SMILES heavy-atom count must be ≥ 0.4× the input's IUPAC token
        count for long names (≥30 tokens) — rejects partial-substring parses
      - OPSIN warnings of "Unmatched bracket" / "Could not interpret" /
        "couldn't" are HARD failures even if a SMILES is returned

    Strict mode is gated by the caller — wired from `_convert_single`
    when `is_clean_text=False` (i.e., the IUPAC came from MinerU markdown
    or another OCR'd source). Clean-HTML extractions stay lenient because
    they lack the OCR-garbage signals to filter on.
    """
    # A MULTI-COMPONENT name — PubChem's `;` notation for a substance whose
    # parts are not bonded to one another: `dicesium;carbonate` is Cs2CO3,
    # `palladium;tetrakis(triphenylphosphane)` is Pd(PPh3)4. OPSIN expects ONE
    # connected name and fails on the list, which is 652 of the corpus's name
    # failures.
    #
    # Each component is resolved separately and the results are joined with
    # `.`, NOT with `;`. `.` is SMILES' own separator for disconnected
    # components, so the join produces one valid multi-component structure with
    # one InChIKey — which is the point, since the substance IS one substance.
    # A `;`-joined string would not parse as SMILES at all. Component ORDER is
    # preserved so the structure reads in the same sequence as the name.
    #
    # All-or-nothing: if any component fails, the whole name fails. A partial
    # join would silently record Cs2CO3 as "carbonate".
    if ";" in name:
        parts = [p.strip() for p in name.split(";") if p.strip()]
        if len(parts) > 1:
            out = []
            for p in parts:
                smi, err = _try_opsin(p, strict=strict, relaxed=relaxed)
                if not smi:
                    return None, f"multi-component: {p[:40]!r} failed ({err})"
                out.append(smi)
            return ".".join(out), ""

    if strict:
        # Pre-filter: reject obvious OCR/markup pollution in the input
        if _OPSIN_INPUT_GARBAGE_PAT.search(name):
            return None, "input contains OCR markup tags (strict mode)"
        # Reject names starting with punctuation/digit (truncation hint)
        stripped = name.strip()
        if stripped and not (stripped[0].isalpha() or stripped[0] in "([{"):
            return None, "input starts with non-letter (strict mode; truncated?)"

    # One JVM launch, unless `prefetch_opsin` already resolved this name in a
    # batch — see the batching block above. Either way the answer, and the
    # warning text OPSIN produced with it, are what a lone py2opsin call
    # returns; a per-process tmp_fpath keeps parallel patent runs from
    # clobbering each other's OPSIN input file.
    try:
        result, error_msg = _opsin_raw(name, relaxed=relaxed)
    except Exception as e:
        return None, str(e)

    if not (result and isinstance(result, str) and len(result) > 3):
        return None, error_msg or "OPSIN returned empty/invalid result"
    if not validate_smiles(result):
        return None, error_msg or "OPSIN returned empty/invalid result"

    if strict:
        # Coverage check: heavy-atom count vs. IUPAC token count.
        # Long names (≥30 alpha tokens) that resolve to a tiny SMILES
        # (≤ 0.4× tokens worth of heavy atoms) are almost always
        # partial-substring parses — OPSIN matched a known fragment in
        # the middle of a corrupted name and returned the SMILES for
        # that fragment, not the full molecule.
        try:
            from rdkit import Chem as _RDChem
            mol = _RDChem.MolFromSmiles(result)
            n_heavy = mol.GetNumHeavyAtoms() if mol else 0
        except Exception:
            n_heavy = 0
        n_tokens = len(re.findall(r"[A-Za-z]+", name))
        if n_tokens >= 30 and n_heavy < 0.4 * n_tokens:
            return None, (
                f"strict: coverage too low ({n_heavy} heavy atoms vs "
                f"{n_tokens} tokens) — likely partial-substring parse"
            )
        # Warning gate — these phrases mean OPSIN had to skip part of
        # the name to produce a SMILES. Reject; the cascade has cleanup
        # stages that may recover the full name.
        if error_msg:
            low = error_msg.lower()
            if any(w in low for w in ("unmatched bracket", "could not interpret", "couldn't")):
                return None, f"strict: OPSIN warning ({error_msg})"

    return result, error_msg


def _convert_single(
    compound: Compound,
    is_clean_text: bool = False,
    route_hint: str | None = None,
) -> Compound:
    """Convert one compound through the staged pipeline.

    Args:
        compound: Compound with `iupac_name` set.
        is_clean_text: True when name source is high-quality clean text
            (e.g., Google Patents XML). Skips the OCR-targeted stages
            (rule_clean, relaxed OPSIN) since they add 0 wins on clean
            text per gauntlet evidence and waste compute.
        route_hint: Tags the compound's structured provenance with the route
            this conversion is happening under. See `_finalize` for valid
            values. Defaults to inferring from the existing extraction_method.
    """
    if not compound.iupac_name:
        compound.failure_reason = "no_iupac_name"
        return compound

    raw_name = compound.iupac_name
    cleaned = raw_name  # Used as fallback by later stages

    # Strict OPSIN gates fire when the source isn't known-clean. Markdown
    # carries OCR garbage (`<|ref|>`, `[[bbox]]`, mid-name punctuation
    # corruption) that OPSIN will sometimes parse a substring of and
    # return the wrong SMILES — strict mode rejects those so the
    # cascade's cleanup stages get a chance.
    strict_opsin = not is_clean_text

    # Stage 0: PubChem lookup (stereo-aware, authoritative)
    pubchem_smiles = _try_pubchem(raw_name)
    if pubchem_smiles:
        return _finalize(compound, pubchem_smiles, stage="pubchem_direct", route_hint=route_hint)

    # Stage 1: OPSIN on raw name (free, instant)
    smiles, error = _try_opsin(raw_name, strict=strict_opsin)
    if smiles:
        return _finalize(compound, smiles, stage="opsin_direct", route_hint=route_hint)

    # Stages 2 / 2c / 2b — OCR-targeted cleanup. Skip for known-clean text.
    if not is_clean_text:
        # Stage 2: Rule-based clean → OPSIN (free, instant).
        # rule_based_clean strips OCR markup tags too, so the cleaned name
        # passes the strict input-cleanliness gate.
        cleaned = rule_based_clean(raw_name)
        # Track if stereo was stripped
        stereo_stripped = any(p in raw_name.lower() for p in ['(cis)', '(trans)']) \
                          and not any(p in cleaned.lower() for p in ['(cis)', '(trans)'])
        if cleaned != raw_name:
            smiles, error = _try_opsin(cleaned, strict=strict_opsin)
            if smiles:
                compound.inferred_stereochemistry = stereo_stripped
                return _finalize(compound, smiles, stage="rule_cleaned", route_hint=route_hint)

        # Stage 2d: relaxed OPSIN — last free attempt before anything paid.
        # Retries the raw and cleaned names with OPSIN's permissive flags
        # (see `_try_opsin`). Only reached once every strict parse above has
        # failed, so it cannot change an existing successful result; it can
        # only rescue a name that was otherwise headed for the LLM stages.
        # Tagged distinctly in provenance because a relaxed parse may have
        # dropped stereochemistry — these records are connectivity-trustworthy,
        # not stereo-trustworthy, and downstream consumers must be able to see
        # the difference.
        for candidate in (cleaned, raw_name):
            if not candidate:
                continue
            smiles, error = _try_opsin(candidate, strict=False, relaxed=True)
            if smiles:
                return _finalize(
                    compound, smiles,
                    stage="opsin_relaxed_stereo_dropped",
                    route_hint=route_hint,
                )

    # Per-patent LM cost cap guard — Stages 3a/3b skipped if exceeded
    if cost_tracker.patent_lm_exceeded(compound.patent_id):
        compound.failure_reason = "patent_lm_cap_exceeded"
        compound.processing_status = "failed"
        _log_opsin_failure(
            patent_id=compound.patent_id,
            compound_id=compound.example_number or "unknown",
            raw_name=raw_name,
            cleaned_name=cleaned if cleaned != raw_name else None,
            llm_cleaned_name=None,
            error=f"LM cap (${config.PER_PATENT_LM_CAP}) exceeded; skipped LM stages",
            is_truncated=raw_name.count('(') != raw_name.count(')'),
        )
        return compound

    # Stage 3a: LLM normalize IUPAC name → OPSIN (Opus, accuracy-tuned)
    llm_cleaned = _llm_clean(
        raw_name, error,
        compound.patent_id,
        compound.example_number or "unknown",
    )
    if llm_cleaned:
        smiles, error2 = _try_opsin(llm_cleaned)
        if smiles:
            return _finalize(compound, smiles, stage="llm_cleaned", route_hint=route_hint)

    # Re-check cap before Stage 3b (since 3a already spent some budget)
    if cost_tracker.patent_lm_exceeded(compound.patent_id):
        compound.failure_reason = "patent_lm_cap_exceeded_after_stage_3a"
        compound.processing_status = "failed"
        return compound

    # Stage 3b: Direct SMILES from Opus (last resort for OPSIN-unfixable names)
    direct_smiles = _llm_direct_smiles(
        raw_name,
        compound.patent_id,
        compound.example_number or "unknown",
    )
    if direct_smiles:
        return _finalize(compound, direct_smiles, stage="llm_direct_smiles", route_hint=route_hint)

    # All stages failed — log and flag
    # Check if this is a complex macrocyclic name that no tool can handle
    is_macrocycle = any(kw in raw_name.lower() for kw in [
        'methano', 'cyclotridecine', 'cyclododecine', 'cyclopentadecine',
        'methanobenzo', 'methanodipyrido', 'triazacyclo',
    ])
    if is_macrocycle:
        compound.failure_reason = "unparsable_macrocycle_frontier_limitation"
    else:
        compound.failure_reason = "all_stages_failed"
    compound.processing_status = "failed"

    _log_opsin_failure(
        patent_id=compound.patent_id,
        compound_id=compound.example_number or "unknown",
        raw_name=raw_name,
        cleaned_name=cleaned if cleaned != raw_name else None,
        llm_cleaned_name=llm_cleaned,
        error=error,
        is_truncated=raw_name.count('(') != raw_name.count(')'),
    )

    return compound


def _finalize(
    compound: Compound,
    smiles: str,
    stage: str,
    route_hint: str | None = None,
) -> Compound:
    """Finalize a compound with a validated SMILES. Sets provenance metadata.

    Args:
        compound: Compound being finalized.
        smiles: Raw SMILES from the converter.
        stage: One of {pubchem_direct, opsin_direct, rule_cleaned,
            opsin_relaxed_stereo_dropped, llm_cleaned, llm_direct_smiles, ...}.
        route_hint: Optional route this compound came from
            ("google_patents_example", "google_patents_table", "image_pipeline",
             "synthesis_block", "page_extraction", etc.). If None, falls back to
             the existing extraction_method when set, otherwise "unknown".
    """
    from .models import CompoundProvenance

    compound.extraction_method = stage
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        compound.failure_reason = f"rdkit_rejected_opsin_output"
        compound.processing_status = "failed"
        return compound

    if len(canonical) < MIN_SMILES_LENGTH:
        compound.failure_reason = f"smiles_too_short_{len(canonical)}"
        compound.processing_status = "failed"
        return compound

    mw = molecular_weight(canonical)
    if mw and mw < MIN_SMILES_MW:
        compound.failure_reason = f"mw_too_low_{mw:.0f}"
        compound.processing_status = "failed"
        return compound
    if mw and mw > MAX_SMILES_MW:
        compound.failure_reason = f"mw_too_high_{mw:.0f}"
        compound.processing_status = "failed"
        return compound

    compound.smiles_from_text = smiles
    compound.canonical_smiles = canonical
    compound.inchikey = get_inchikey(canonical)

    salt_result = strip_salt(canonical)
    compound.parent_smiles = salt_result.get("parent_smiles")
    compound.parent_inchikey = salt_result.get("parent_inchikey")

    compound.drug_likeness = compute_drug_likeness(canonical)

    # MW cross-check against MS data (intermediate validation)
    if compound.ms_mh_plus:
        from rdkit import Chem as RDChem
        from rdkit.Chem import Descriptors as RDDescriptors
        mol = RDChem.MolFromSmiles(canonical)
        if mol:
            exact_mw = RDDescriptors.ExactMolWt(mol)
            expected_mw = compound.ms_mh_plus - 1.008  # (M+H)+ = MW + proton
            delta = abs(exact_mw - expected_mw)
            compound.mw_validated = delta < 1.5
            if not compound.mw_validated:
                # MW mismatch is a WARNING, not a failure.
                # Common causes: Claude assigned wrong MS to this compound,
                # or IUPAC name was partially truncated (missing substituent).
                # OPSIN SMILES is still likely correct — the MS assignment is unreliable.
                logger.warning(
                    f"{compound.example_number}: MW MISMATCH (warning) — "
                    f"OPSIN={exact_mw:.1f} vs MS={expected_mw:.1f} (Δ={delta:.1f}Da)"
                )

    # Set stereo trust based on extraction method
    if stage == "llm_direct_smiles":
        compound.stereo_trusted = False
    if compound.inferred_stereochemistry:
        compound.stereo_trusted = False

    # Mark LM-derived compounds for benchmark filtering
    if stage in ("llm_cleaned", "llm_direct_smiles", "synthesis_block_lm"):
        compound.lm_normalized = True

    # Populate structured provenance
    # Resolve route: caller hint > existing extraction_method prefix > "unknown"
    route = route_hint
    if not route:
        # Infer route from existing extraction_method prefix (set by extractor before _finalize)
        prev_method = compound.extraction_method or ""
        if prev_method.startswith("google_patents_table"):
            route = "google_patents_table"
        elif prev_method.startswith("google_patents"):
            route = "google_patents_example"
        elif prev_method == "synthesis_block_lm":
            route = "synthesis_block"
        elif prev_method.startswith(("decimer", "sonnet_vision", "opus_vision")):
            route = "image_pipeline"
        elif prev_method.startswith("adaptive"):
            route = "adaptive"
        else:
            route = "page_extraction"

    chain = [stage]
    if compound.provenance:
        # Shouldn't normally happen — _finalize is called once. But if it does,
        # preserve the previous chain for debugging.
        chain = (compound.provenance.chain or []) + chain
    compound.provenance = CompoundProvenance(
        route=route,
        stage=stage,
        page=compound.source_page,
        image_bbox=compound.image_bbox,
        image_path=compound.image_path,
        chain=chain,
    )

    compound.processing_status = "validated"
    return compound


def _log_opsin_failure(
    patent_id: str, compound_id: str, raw_name: str,
    cleaned_name: str | None, llm_cleaned_name: str | None,
    error: str, is_truncated: bool,
):
    """Log OPSIN failure for analysis and iteration."""
    # Categorize
    if is_truncated:
        category = "truncated_name"
    elif 'pyrolo' in raw_name.lower():
        category = "fused_ring_typo"
    elif re.search(r',\s*\d*\s*HCl', raw_name):
        category = "salt_suffix"
    elif re.search(r'^\(?(cis|trans)', raw_name, re.IGNORECASE):
        category = "stereochem"
    else:
        category = "other"

    entry = {
        "patent_id": patent_id,
        "compound_id": compound_id,
        "raw_iupac": raw_name[:200],
        "cleaned_iupac": cleaned_name[:200] if cleaned_name else None,
        "llm_cleaned": llm_cleaned_name[:200] if llm_cleaned_name else None,
        "opsin_error": error[:200],
        "failure_category": category,
        "is_truncated": is_truncated,
    }

    with open(OPSIN_FAILURES_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================
# Public API
# ============================================================

# Backward-compatible single conversion (used by tests)
def convert_iupac_to_smiles(
    compound: Compound,
    markush_context: MarkushContext | None = None,
) -> Compound:
    return _convert_single(compound)

# Keep the name for pipeline.py
finalize_compound = _finalize


def convert_batch(
    compounds: list[Compound],
    markush_context: MarkushContext | None = None,
) -> list[Compound]:
    """Convert all compounds via 3-stage pipeline in parallel."""
    to_convert = [c for c in compounds if c.iupac_name]
    if not to_convert:
        return compounds

    patent_id = compounds[0].patent_id if compounds else "unknown"
    logger.info(f"Patent {patent_id}: Converting {len(to_convert)} IUPAC names (OPSIN + rules + LLM fallback)")

    # Resolve the whole set through OPSIN first, in batched JVM launches, so
    # the per-compound cascade below reads answers instead of shelling out.
    # `_convert_single` here runs with is_clean_text=False, so all three free
    # stages are in play. The threads remain — the LLM stages are still one
    # network call each — but the OPSIN stages no longer are.
    prefetch_cascade([c.iupac_name for c in to_convert])

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(_convert_single, c): c for c in to_convert}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                c = futures[future]
                logger.error(f"Conversion error for {c.example_number}: {e}")
                c.failure_reason = str(e)
                c.processing_status = "failed"

    # Stats
    validated = sum(1 for c in to_convert if c.processing_status == "validated")
    failed = sum(1 for c in to_convert if c.processing_status == "failed")
    logger.info(f"Patent {patent_id}: IUPAC→SMILES complete. Validated: {validated}, Failed: {failed}")

    return compounds
