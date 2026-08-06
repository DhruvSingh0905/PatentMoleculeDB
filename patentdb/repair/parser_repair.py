"""The second repair tier: fix the READER, not the layout.

Why this exists as a separate thing
-----------------------------------
`repair/loop.py` buys one rule per layout fingerprint. That is the right unit
for a table we cannot interpret — the shape recurs across patents, so one paid
question amortises. It is the WRONG unit for a bug in our own parser, and the
difference is not academic:

  `_parse_row` matched `<entry/>` with the paired-tag branch, so every empty
  cell swallowed its neighbour. That is ONE regex. It corrupted US11613531
  (2,359 cells), US11254686, US10172859 and every other patent whose tables use
  positional placeholders — which is most of them. The loop's response was to
  synthesize renames and bin keys, per layout, per patent, forever, each one
  encoding the corruption rather than removing it.

A layout rule costs ~$0.002 and helps the patents sharing that shape. A parser
fix costs one question and helps EVERY patent, past and future, permanently.
Paying per-layout to route around a global defect is the single most expensive
mistake this loop can make, and it produces wrong data while doing it.

So the unit of work here is a defect, found once across the whole corpus, with
one minimal reproduction — not a patent, and not a layout.

Why a model can safely be trusted to patch code here
----------------------------------------------------
Automated program repair usually has a weak oracle: a test suite that a patch
can satisfy by deleting the failing behaviour. Kali and GenProg-style results
found the overwhelming majority of "successful" patches were of exactly that
kind, and this repo already guards the data-rule path against the analogue.

`parse_fidelity` is a much better oracle, for a structural reason: it counts
elements in the SOURCE against cells we produced. A patch that deletes
behaviour produces FEWER cells and scores strictly worse. The classic cheat is
not available. Combined with "every other patent must stay clean, the suite must
stay green, and extraction totals must not fall", the acceptance test is cheap,
precise and hard to game.

Nothing here writes to the working tree unless somebody ASKED — see
repair/guard.py. The default is to verify a patch fully, journal it, and hand
back the diff.

That sentence used to read "unless `PARSER_REPAIR_APPLY` is set", which was
true and misleading in the same breath: the flag defaults to 1, so the
documented default behaviour was the opposite of what the module did. An
extraction run rewrote `sources/uspto_assays.py` under that default.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config
from ..sources.uspto_xml import parse_fidelity
from . import guard

logger = logging.getLogger(__name__)

# Ground-truth entry scanner: self-closing FIRST, so `<entry/>` cannot be taken
# for a paired tag. Deliberately a local copy — this module must be able to
# describe what the source says even when the parser it is auditing is wrong.
_ENTRY = re.compile(r"<entry\b([^>]*?)/>|<entry\b([^>]*)>(.*?)</entry>", re.S)
_ROW = re.compile(r"<row\b[^>]*>.*?</row>", re.S)


@dataclass
class Defect:
    """One way the reader disagrees with the source, wherever it occurs."""
    signature: str
    repro: str                                  # smallest XML exhibiting it
    cells_lost: int = 0
    patents: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    @property
    def blast_radius(self) -> str:
        return (f"{len(self.patents)} patent(s), {len(self.tables)} table(s), "
                f"{self.cells_lost} cells")


def _row_shape(row_xml: str) -> str:
    """The row's entry pattern as ground truth: `e`=self-closing, `E`=paired.

    Run-length compressed, so `<entry/>x6` then three paired entries is `e6E3`.
    Two blocks that break the same way compress to the same string, which is
    what lets one question answer for the whole corpus.
    """
    forms = ["e" if m.group(1) is not None else "E" for m in _ENTRY.finditer(row_xml)]
    out, i = [], 0
    while i < len(forms):
        j = i
        while j < len(forms) and forms[j] == forms[i]:
            j += 1
        out.append(f"{forms[i]}{j - i}" if j - i > 1 else forms[i])
        i = j
    return "".join(out)


def minimal_repro(block_xml: str) -> tuple[str, str, int]:
    """The smallest `<row>` in this block that the reader gets wrong.

    A 6,514-entry table is not a bug report. One row is. Returns
    `(row_xml, shape, cells_lost)`; the row is what a model is shown and what a
    regression test is written against.
    """
    from ..sources.uspto_xml import _parse_row

    worst = ("", "", 0)
    for m in _ROW.finditer(block_xml):
        row = m.group(0)
        want = len(_ENTRY.findall(row))
        got = len(_parse_row(row))
        if want == got:
            continue
        lost = want - got
        # Prefer the SMALLEST row that still exhibits the defect; among equals,
        # the one losing most cells. A short repro is a better question.
        if not worst[0] or len(row) < len(worst[0]):
            worst = (_reduce(row), "", lost)
    if worst[0]:
        worst = (worst[0], _row_shape(worst[0]), worst[2])
    return worst


def _reduce(row_xml: str) -> str:
    """Delta-debug the row down to the smallest fragment that still breaks.

    Without this the signature is the row's full shape, and ONE bug fragments
    into 60 "distinct" defects — `eE`, `eE2`, `e2E`, `e2E3` — which is 60 paid
    questions for a single regex. Greedily dropping entries while the
    discrepancy survives collapses all of them onto the same minimal motif, so
    the corpus asks once.
    """
    from ..sources.uspto_xml import _parse_row

    def broken(entries: list[str]) -> bool:
        frag = "<row>" + "".join(entries) + "</row>"
        return len(_ENTRY.findall(frag)) != len(_parse_row(frag))

    entries = [m.group(0) for m in _ENTRY.finditer(row_xml)]
    if not broken(entries):
        return row_xml
    changed = True
    while changed and len(entries) > 1:
        changed = False
        for i in range(len(entries)):
            trial = entries[:i] + entries[i + 1:]
            if trial and broken(trial):
                entries = trial
                changed = True
                break
    return "<row>" + "".join(entries) + "</row>"


def corpus_defects(xml_dir: Path | None = None, limit: int | None = None) -> list[Defect]:
    """Every reader defect across every cached patent, grouped and deduplicated.

    This is the "it exists elsewhere" step, and it is the whole economic
    argument for this tier. Diagnosing per patent would have asked the same
    question 63 times; grouping by the structural shape of the failing row asks
    it once and reports the blast radius.
    """
    xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")
    found: dict[str, Defect] = {}
    paths = sorted(xml_dir.glob("*.xml"))[:limit]
    for path in paths:
        try:
            xml = path.read_text()
        except OSError:
            continue
        for d in parse_fidelity(xml):
            block = re.search(
                r"<tables\b[^>]*id=\"" + re.escape(d["table_id"]) + r"\".*?</tables>",
                xml, re.S)
            if not block:
                continue
            repro, shape, lost = minimal_repro(block.group(0))
            if not repro:
                continue
            defect = found.get(shape)
            if defect is None:
                defect = found[shape] = Defect(signature=shape, repro=repro)
            defect.cells_lost += d["source_entries"] - d["parsed_cells"]
            if path.stem not in defect.patents:
                defect.patents.append(path.stem)
            defect.tables.append(f"{path.stem}:{d['table_id']}")
    return sorted(found.values(), key=lambda d: -d.cells_lost)


# ── the acceptance test ───────────────────────────────────────────

def verify_patch(module: Path, new_source: str, *, xml_dir: Path | None = None,
                 baseline: dict | None = None,
                 also: dict[Path, str] | None = None,
                 repair_pid: str | None = None) -> dict:
    """Run a candidate patch against the whole corpus in a scratch copy.

    Never touches the working tree. Four conditions, all mandatory:

      1. the module still imports and the suite is green;
      2. `parse_fidelity` is clean across EVERY cached patent — not just the
         one that motivated the patch, because a reader change is global;
      3. total usable records do not fall (the anti-deletion condition — a
         patch that "fixes" fidelity by parsing less is the classic APR cheat);
      4. per-patent counts do not fall for any patent that was FIDELITY-CLEAN
         at baseline, so a gain on one patent cannot hide a loss on another.

    Condition 4 is scoped deliberately, and the scoping was earned. The first
    real run rejected a patch that was byte-for-byte equivalent to the correct
    fix, because US9233167 went 35 -> 0. Those 35 records were an artifact of
    the corruption: with cells shifted left, junk happened to parse. The
    baseline had been measured with the broken parser, so the check was
    demanding the patch preserve accidents.

    A corrupted baseline is not a specification. A patent whose own source
    disagreed with our reader cannot supply a floor for the patch that fixes
    it; its change is reported for review instead.
    """
    xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")
    root = Path(__file__).resolve().parent.parent.parent
    with tempfile.TemporaryDirectory(prefix="parser-repair-") as tmp:
        sandbox = Path(tmp)
        try:
            tracked = subprocess.run(["git", "ls-files"], cwd=root, check=True,
                                     capture_output=True, text=True).stdout.split()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return {"ok": False, "why": f"cannot enumerate tracked files: {e!r}"}
        for rel in tracked:
            src = root / rel
            if not src.is_file():
                continue
            (sandbox / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, sandbox / rel)

        # `also` carries the rest of a MULTI-FUNCTION patch. Some shapes cannot
        # be fixed one function at a time: US9302989 needs `classify_column` to
        # see a two-assay header AND the emitter to split one cell into two
        # records, and either alone is inert. They must be verified together or
        # each half looks like a patch that changes nothing.
        for mod, src_text in {module: new_source, **(also or {})}.items():
            target = sandbox / Path(mod).relative_to(root)
            if not target.exists():
                return {"ok": False, "why": f"{mod} is not a tracked file"}
            target.write_text(src_text)

        # `repair_pid` asks the probe for the count one patent reaches through
        # the FULL path — deterministic parse plus the cached rules. A
        # capability patch is often only half a fix on its own: promoting a
        # column of `+`/`++` to ASSAY produces records with a grade and no
        # number, which `extract_from_patent` scores as zero usable, and the
        # bin_key rule that turns the grade into a range lives in the repair
        # loop. Measured on `extract_from_patent` alone such a patch looks
        # inert; measured through `repair_patent` it recovers 1,238 rows.
        probe = (_PROBE.replace("__XML_DIR__", str(xml_dir.resolve()))
                       .replace("__REPAIR_PID__", repair_pid or ""))
        (sandbox / "_probe.py").write_text(probe)
        run = subprocess.run([sys.executable, "_probe.py"], cwd=sandbox,
                             capture_output=True, text=True, timeout=900)
        if run.returncode != 0:
            return {"ok": False, "why": "probe failed",
                    "stderr": run.stderr[-2000:]}
        import json
        try:
            got = json.loads(run.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "why": "probe produced no result",
                    "stdout": run.stdout[-2000:]}

        tests = subprocess.run([sys.executable, "-m", "pytest", "patentdb/tests/", "-q"],
                               cwd=sandbox, capture_output=True, text=True, timeout=900)
        got["tests_pass"] = tests.returncode == 0
        got["tests_tail"] = tests.stdout.strip().splitlines()[-1] if tests.stdout else ""

        # ONE blocking condition: does the patched code read FEWER compounds
        # than before? Everything else is recorded, not enforced.
        #
        # This is the same call already made for model-proposed RULES, applied
        # to model-proposed CODE. A fixed gate cannot anticipate the layouts
        # patents actually use, and every judgement-shaped gate here has been
        # wrong at least as often as right — a correct column_map scored 0/23,
        # a 49% floor on a rule whose real fault was a regex in the reader, an
        # anti-deletion baseline measured with a broken parser. Twice more in
        # this tier: an inert-patch check that declined a patch recovering
        # 1,238 rows, because it measured the parse and the fix needed the
        # loop. Blocking is not what makes this safe; the journal is, because
        # anything applied is revertible and its coverage delta is recorded.
        #
        # Coverage cannot be argued with, which is why it is the one that
        # stays. It also catches a patch that does not import or crashes: the
        # probe scores an exception as -1 and the total falls.
        #
        # What it CANNOT see is a patch that raises the count with wrong
        # values — the 10x bin-scale class. No count check ever will. That is
        # what the journal and a human reading the diff are for.
        evidence = []
        if got["discrepant_blocks"]:
            evidence.append(f"{got['discrepant_blocks']} block(s) disagree with "
                            f"their source")
        if not got["tests_pass"]:
            evidence.append(f"test suite fails: {got['tests_tail']}")
        # A patch that THROWS is not a patch that found less. The probe scores an
        # exception as -1 and `total_usable` drops, which the coverage condition
        # then reports as "picks up FEWER compounds" — true in arithmetic and
        # useless as feedback, because the model is told to find more data when
        # what it needs to hear is that its code raises. Named separately so the
        # next attempt has something to act on.
        crashed = sorted(p for p, v in got["per_patent"].items() if v < 0)
        got["crashed_patents"] = crashed
        if crashed:
            evidence.append(f"extraction RAISES on {len(crashed)} patent(s): "
                            + ", ".join(crashed[:5])
                            + ("..." if len(crashed) > 5 else ""))

        if baseline:
            trusted = baseline.get("_clean", set(baseline) - {"_clean"})
            total_before = sum(v for k, v in baseline.items() if k != "_clean")
            # The repaired count when we have it: a capability patch is often
            # half a fix on its own, finished by a rule in the loop.
            after = got["total_usable"]
            if got.get("repaired_usable") is not None and got["repaired_usable"] >= 0:
                got["full_path_usable"] = got["repaired_usable"]
            if after < total_before:
                got.update(ok=False, why=(
                    (f"extraction RAISES on {len(crashed)} patent(s) "
                     f"({', '.join(crashed[:3])}) — fix the exception first; "
                     f"compounds {total_before} -> {after}."
                     if crashed else
                     f"picks up FEWER compounds: {total_before} -> {after}. "
                     f"The only condition.")))
                got["objections"] = evidence
                return got
            moved = {p: (baseline[p], got["per_patent"].get(p, 0))
                     for p in baseline if p != "_clean"
                     and got["per_patent"].get(p, 0) < baseline[p]}
            lost = {p: v for p, v in moved.items() if p in trusted}
            if lost:
                evidence.append(f"per-patent falls on fidelity-clean patents: {lost}")
            got["changed_on_corrupt_baseline"] = {
                p: v for p, v in moved.items() if p not in trusted}

        # ...and the question `baseline` structurally cannot answer: is any
        # patent now below the best it has EVER scored?
        #
        # `baseline` is the state immediately before this patch, so the check
        # above only ever sees one step. A sequence that takes a patent 860 ->
        # 800 -> 700 clears the corpus condition three times if other patents
        # rise, and nothing remembers the 860. That is the shape of the loss
        # this tier's history is built around — US10660877 went 860 -> 0 on an
        # `_is_namelike` patch that touched none of its rows.
        #
        # Recorded, never enforced. A patch that genuinely supersedes an old
        # reading — the corrupted-baseline case, where junk parsed because
        # cells were shifted left — must be allowed to lower it, and a gate
        # here would be the fourth judgement-shaped gate this file has had to
        # remove. The journal plus a named regression is what makes it
        # reviewable.
        try:
            from . import ledger
            got["below_best"] = ledger.regressions_vs_best(got["per_patent"])
        except Exception as e:                   # bookkeeping never blocks
            logger.warning("ledger: best-known check skipped (%r)", e)
            got["below_best"] = {}
        if got["below_best"]:
            worst = sorted(got["below_best"].items(),
                           key=lambda kv: kv[1][0] - kv[1][1], reverse=True)
            evidence.append(
                f"{len(worst)} patent(s) now below their best-ever count: "
                + ", ".join(f"{p} {b}->{n}" for p, (b, n) in worst[:5])
                + ("..." if len(worst) > 5 else ""))

        got["objections"] = evidence
        if evidence:
            logger.warning("patch applied over %d objection(s): %s",
                           len(evidence), "; ".join(evidence)[:220])
        got["ok"] = True
        return got


def adopt_baseline(verdict: dict, *, journal_id: str | None = None) -> int:
    """The patch is in the tree; the probe already measured the tree. Adopt it.

    MUST be called after the modules are written — `ledger.record` stamps the
    entries with the CURRENT code key, and calling this first would file the
    patched corpus under the unpatched tree's key and permanently serve a
    baseline nobody ever measured.

    This is what stops a landed patch from costing the next gap a full rescan.
    `verify_patch` ran the patched modules over every cached XML in a sandbox
    copy of the tracked tree, which is byte-for-byte what the tree becomes when
    `write_text` returns, so re-deriving those counts would be recomputing a
    number we are holding in `verdict["per_patent"]`.
    """
    from . import ledger

    try:
        return ledger.record(verdict.get("per_patent") or {},
                             per_clean=verdict.get("per_clean"),
                             journal_id=journal_id)
    except Exception as e:                       # bookkeeping never breaks a run
        logger.warning("ledger: could not adopt the probe's counts (%r)", e)
        return 0


_PROBE = '''
import json, pathlib, sys
sys.path.insert(0, ".")
from patentdb.sources.uspto_xml import parse_fidelity
from patentdb.sources.uspto_assays import extract_from_patent

discrepant, per_patent, per_clean = 0, {}, {}
for p in sorted(pathlib.Path("__XML_DIR__").glob("*.xml")):
    xml = p.read_text()
    # PER PATENT, not just the total. `parse_fidelity` is already being called
    # here and its per-file verdict was being summed away — that verdict is
    # exactly `baseline_counts`'s `_clean` set, which decides whose count is a
    # trustworthy floor. Recording it lets `repair/ledger.py` adopt this run as
    # the next baseline complete, instead of carrying a stale `clean` flag
    # forward across a patch that changed fidelity.
    fid = parse_fidelity(xml)
    discrepant += len(fid)
    per_clean[p.stem] = not fid
    try:
        # COMPOUNDS, not records. A patch that splits one cell into two rows
        # doubles the record count without finding anything new; the question
        # is whether the patent gave up more of its molecules.
        per_patent[p.stem] = len({r.cid for r in extract_from_patent(xml)
                                  if r.is_usable and r.cid})
    except Exception:
        per_patent[p.stem] = -1
repaired = None
bad_values = None
pid = "__REPAIR_PID__"
if pid:
    try:
        from patentdb.repair.loop import repair_patent
        xml = (pathlib.Path("__XML_DIR__") / (pid + ".xml")).read_text()
        # `repair_patent` returns the deterministic baseline UNIONED with what
        # the rules recovered; adding `extract_from_patent` again here would
        # count every baseline record twice and inflate the gate it feeds.
        extra, _ = repair_patent(pid, xml, max_calls=0)
        repaired = sum(1 for r in extra if r.is_usable)
        # Are the NUMBERS right? A coverage check cannot tell recovered data
        # from invented data; BindingDB publishes the values, so this is a
        # lookup. Run inside the sandbox so it measures the PATCHED code.
        from patentdb.repair.value_check import check_patent
        bad_values = check_patent(pid, list(extra))["bad"]
    except Exception:
        repaired = -1
print(json.dumps({"discrepant_blocks": discrepant, "per_patent": per_patent,
                  "per_clean": per_clean,
                  "repaired_usable": repaired, "bad_values": bad_values,
                  "total_usable": sum(v for v in per_patent.values() if v > 0)}))
'''


def baseline_counts(xml_dir: Path | None = None) -> dict:
    """Distinct usable COMPOUNDS per patent — the floor a patch must hold.

    Compounds, and the unit is the whole point. This counted usable RECORDS
    while `_PROBE` counted `len({r.cid ...})`, and `verify_patch` compared the
    two directly:

        total_before = 75004     records, from here
        after        = 31114     compounds, from the probe
        if after < total_before: -> "picks up FEWER compounds"

    There are always more records than compounds — a patent with three assay
    columns emits ~3 records per molecule — so the condition was arithmetically
    unsatisfiable and the capability tier could never apply anything. Every
    decline this session read "75004 -> 31114" whatever was patched, because
    31114 is simply the corpus's compound count and no patch moved it much; the
    one assembler patch that DID move it scored 31093 and was declined for
    twenty-one compounds against a baseline in the wrong unit.

    The docstring above `verify_patch` already says compounds, `_PROBE` already
    explains why records are the wrong unit, and this function silently
    disagreed with both.

    REMEMBERED, NOT RE-DERIVED. This re-extracted all 137 cached patents on every
    call. Measured on US20240010684A1 — 15 compounds, a gap worth 3 rows — that
    was 14.04 s of the tier's 39.53 s untraced, and 47.69 s of 74.68 s under the
    tracer. The number is a pure function of (extraction code, XML), so it is now
    read from `repair/ledger.py` and only the patents whose code state moved are
    measured again. See that module for why a landed patch does not cost a rescan
    either: the probe that verified it already measured the corpus under the new
    code, and `verify_patch` hands that straight to the ledger.

    The in-process `_BASELINE_CACHE` this used to keep is gone with it. It was
    keyed on the patchable modules' `st_mtime_ns`, which is a weaker key than the
    ledger's content hash — a `--revert` restores the exact bytes that were
    measured before and moves the mtime — and it could not outlive the process,
    which is the case that mattered: a corpus run starts a fresh one per patent.

    The POPULATION is still the whole corpus, deliberately. `verify_patch`
    compares this sum against a corpus-wide probe total, so a baseline covering
    fewer patents would compare two numbers drawn from different populations —
    see `ledger.counts` for why narrowing it was rejected rather than overlooked.
    """
    from ..sources.uspto_assays import extract_from_patent
    from ..sources.uspto_xml import parse_fidelity

    from . import ledger

    xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")

    def measure(path: Path) -> tuple[int, bool]:
        xml = path.read_text()
        n = len({r.cid for r in extract_from_patent(xml)
                 if r.is_usable and r.cid})
        # Only a fidelity-clean patent's count is a trustworthy floor — see
        # verify_patch. A corrupted baseline is not a specification.
        return n, not parse_fidelity(xml)

    return ledger.counts(xml_dir, measure)


# ── the question ──────────────────────────────────────────────────

PATCH_SYSTEM = """You are repairing a table READER, not describing a table.

You are shown one row of OASIS/CALS XML from a US patent, what the source \
declares, what our reader produced, and the current source of the function that \
reads it. Return a corrected version of that function.

The reader must account for every element the source declares. A cell that is \
empty in the source is a POSITION, not an absence: CALS writes a label sitting \
over columns 6-8 of a nine-column table as six empty entries and three full \
ones, and dropping them shifts every following cell left.

Constraints:
  - Return the COMPLETE function, not a diff and not a fragment.
  - Change as little as possible. You are fixing one defect, not refactoring.
  - Keep the signature, the name, and the docstring's meaning.
  - Never make the reader produce FEWER cells. Your patch is checked against \
every patent in the corpus and rejected if any of them loses records; a patch \
that satisfies the count by parsing less is the failure mode we are guarding \
against, and it will not pass.

Your patch is applied to a scratch copy, run over the whole corpus and the full \
test suite, and discarded unless every check passes. Describe the fix; the \
harness decides."""

PATCH_TOOL = {
    "name": "propose_parser_patch",
    "description": "Return the corrected function body for a reader defect.",
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": ("One or two sentences: what the current code does "
                                "that loses information."),
            },
            "function_source": {
                "type": "string",
                "description": ("The complete corrected function, from its `def` "
                                "line through its final `return`, correctly "
                                "indented at module level."),
            },
        },
        "required": ["diagnosis", "function_source"],
    },
}


def propose_patch(defect: Defect, func_name: str, module: Path,
                  *, model: str | None = None) -> dict | None:
    """One paid question per DEFECT — not per patent, and not per layout."""
    import anthropic

    from ..core.api_cache import get_cached, store_cached
    from ..core.cost_tracker import cost_tracker
    from ..sources.uspto_xml import _parse_row

    model = model or config.MODEL_HAIKU
    source = module.read_text()
    m = re.search(rf"^def {re.escape(func_name)}\(.*?(?=\n(?:def |@|# ──|\Z))",
                  source, re.S | re.M)
    if not m:
        logger.warning("parser_repair: cannot locate %s in %s", func_name, module)
        return None
    current = m.group(0).rstrip()

    want = len(_ENTRY.findall(defect.repro))
    got = [c.text for c in _parse_row(defect.repro)]
    prompt = (
        f"DEFECT (shape {defect.signature}) affecting {defect.blast_radius}.\n\n"
        f"ONE ROW OF SOURCE XML:\n{defect.repro[:2000]}\n\n"
        f"The source declares {want} <entry> elements in that row.\n"
        f"Our reader produced {len(got)} cells: {got!r}\n\n"
        f"CURRENT SOURCE of `{func_name}` in {module.name}:\n"
        f"```python\n{current}\n```\n")

    key = f"patch::{defect.signature}::{func_name}::{model}"
    cached = get_cached(model, key)
    if cached is not None:
        import json
        try:
            return json.loads(cached)
        except ValueError:
            pass
    if not config.ANTHROPIC_API_KEY:
        logger.info("parser_repair: no API key; cannot propose a patch")
        return None

    from ..core.api_client import resilient_client
    client = resilient_client()
    resp = client.messages.create(
        model=model, max_tokens=2000, temperature=0,
        system=[{"type": "text", "text": PATCH_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[PATCH_TOOL], tool_choice={"type": "tool", "name": PATCH_TOOL["name"]},
        messages=[{"role": "user", "content": prompt}])
    cost_tracker.record(resp.usage.input_tokens, resp.usage.output_tokens, model,
                        patent_id="", cost_category="lm")
    call = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
    if call is None:
        return None
    out = dict(call.input)
    out["_current"] = current
    import json
    store_cached(model, key, json.dumps(out),
                 input_tokens=resp.usage.input_tokens,
                 output_tokens=resp.usage.output_tokens)
    return out


def repair_reader(func_name: str = "_parse_row", *, apply: bool | None = None,
                  limit: int | None = None) -> dict:
    """Find reader defects corpus-wide, buy one patch each, verify, APPLY.

    The loop acts on its own conclusion. It does not hand back a diff and wait
    for someone to agree — a fix that needs a human to flip a switch is a queue,
    not self-healing, and the defect it found is corrupting every patent while
    the switch sits unflipped.

    What replaces the permission gate is the record. Every proposal, applied or
    declined, is journaled with its complete before/after source and the
    per-patent coverage it moved, so `revert(id)` restores any earlier state
    exactly and `apply_journaled(id)` can overrule a declined one. Nothing is
    refused permanently and nothing changes without a trace.

    `verify_patch` still runs, and a patch that fails it is not written. That is
    the agent checking its own work, not a gate on the agent: it would be
    reckless to install a reader change that fails the corpus it was measured
    on. The verdict is journaled either way, because that check has already been
    wrong once.
    """
    module = Path(__file__).resolve().parent.parent / "sources" / "uspto_xml.py"
    # TRI-STATE — see repair/guard.py. `None` no longer resolves to the
    # PARSER_REPAIR_APPLY default (1); it defers to whether an operator ASKED.
    defects = corpus_defects(limit=limit)
    report = {"defects": len(defects), "applied": 0, "declined": 0, "results": []}
    if not defects:
        return report

    base = baseline_counts()
    for d in defects:
        proposal = propose_patch(d, func_name, module)
        if not proposal:
            report["results"].append({"signature": d.signature, "ok": False,
                                      "why": "model returned no proposal"})
            continue
        before = proposal["_current"]
        after = proposal["function_source"].rstrip()
        patched = module.read_text().replace(before, after)
        if patched == module.read_text():
            report["results"].append({"signature": d.signature, "ok": False,
                                      "why": "patch did not apply cleanly"})
            report["declined"] += 1
            continue

        verdict = verify_patch(module, patched, baseline=base)
        verdict.update(signature=d.signature, blast_radius=d.blast_radius,
                       diagnosis=proposal.get("diagnosis", ""))

        # Coverage delta per patent — the thing worth keeping. A total can hide
        # a patent going to zero, which is exactly what happened the first time
        # this ran.
        after_counts = verdict.get("per_patent", {})
        moved = {k: [base[k], after_counts.get(k, 0)]
                 for k in base if k != "_clean"
                 and after_counts.get(k, 0) != base[k]}

        # The WRITE HAPPENS FIRST, so `applied` in the journal is what the tree
        # actually did rather than what this loop intended. `applied: True` on
        # an entry whose text never reached the file is the specific lie that
        # made entries 0034-0038 unreadable, and it is what `revert` trusts.
        outcome = guard.Outcome.REFUSED
        if verdict.get("ok"):
            outcome = guard.write_tracked_source(module, patched, what=func_name,
                                                 explicit=apply)
        entry_id = journal_append({
            "action": "patch", "module": str(module), "function": func_name,
            "signature": d.signature, "blast_radius": d.blast_radius,
            "diagnosis": proposal.get("diagnosis", ""),
            "before_source": before, "after_source": after,
            "verified": bool(verdict.get("ok")), "why": verdict.get("why"),
            "tests_pass": verdict.get("tests_pass"),
            "discrepant_blocks_after": verdict.get("discrepant_blocks"),
            "total_usable_before": sum(v for k, v in base.items() if k != "_clean"),
            "total_usable_after": verdict.get("total_usable"),
            "coverage_moved": moved,
            "applied": outcome == guard.Outcome.WRITTEN,
            "write_outcome": outcome,
        })
        verdict["journal_id"] = entry_id
        verdict["coverage_moved"] = moved
        report["results"].append(verdict)

        if not verdict.get("ok"):
            report["declined"] += 1
            logger.warning("parser_repair: %s declined — %s (journaled; "
                           "`parser_health --force %s` applies it anyway)",
                           entry_id, verdict.get("why"), entry_id.split("-")[0])
            continue
        if outcome == guard.Outcome.WRITTEN:
            adopt_baseline(verdict, journal_id=entry_id)
            report["applied"] += 1
            logger.info("parser_repair: %s APPLIED for %s (%s) — revert with "
                        "`parser_health --revert %s`",
                        entry_id, d.signature, d.blast_radius,
                        entry_id.split("-")[0])
        else:
            verdict["patch"] = after
            logger.warning(
                "parser_repair: %s verified a patch to %s and did NOT write it "
                "(%s). Apply it with `parser_health --force %s`.",
                entry_id, module.name, outcome, entry_id.split("-")[0])
    return report


# ── the journal ───────────────────────────────────────────────────
#
# Authority without an audit trail is recklessness; an audit trail without
# authority is a queue. This is the second half of letting the loop patch its
# own reader: every proposal, applied or declined, is appended here with the
# COMPLETE before and after source and the per-patent coverage it moved.
#
# Revert is therefore self-contained — it does not need git, a clean tree, or
# the patch to still be the newest thing in the file. It restores the exact
# text that was there before, from the record written when it changed.
#
# Declined patches are journaled too, and that is not bookkeeping. The
# acceptance test has already been wrong once: it rejected a patch equivalent to
# the correct fix because the baseline it compared against had been measured
# with the broken reader. A rejection that leaves no trace is a decision nobody
# can review, so `--force` can apply any journaled proposal after the fact.

def _journal_path() -> Path:
    return config.PARSER_REPAIR_JOURNAL


def journal_append(entry: dict) -> str:
    """Record one proposal. Returns its id."""
    import hashlib
    import json

    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(journal_read()) + 1
    entry = dict(entry)
    entry["id"] = f"{n:04d}-" + hashlib.sha256(
        (entry.get("after_source", "") + entry.get("signature", "")).encode()
    ).hexdigest()[:8]
    with path.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry["id"]


def journal_read() -> list[dict]:
    import json

    path = _journal_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def journal_find(entry_id: str) -> dict | None:
    """Match on the full id or its numeric prefix — `0003` is enough to type."""
    for e in journal_read():
        if e.get("id") == entry_id or e.get("id", "").startswith(f"{entry_id}-"):
            return e
    return None


def _entry_parts(entry: dict) -> list[dict]:
    """The (module, before, after) triples an entry moved. One or several."""
    return entry.get("patches") or [{"module": entry.get("module", ""),
                                     "before_source": entry.get("before_source", ""),
                                     "after_source": entry.get("after_source", "")}]


def journal_state() -> list[dict]:
    """For every entry claiming `applied`, is its text still IN the tree?

    `applied: True` is a claim about the past that readers use as a claim about
    the present, and the two came apart. Six entries — `0030`, `0031`, `0034`,
    `0035`, `0036`, `0037` — say applied, while `sources/uspto_assays.py` is
    byte-identical to HEAD and holds none of them. Three of those are the SAME
    proposal replayed from the model cache: an id is
    `sha256(after_source + signature)[:8]`, so `0030`/`0034`/`0036` sharing
    `09a09002` is proof, not resemblance.

    `live` means `revert(id)` would find its text and succeed. `stale` means it
    would refuse — which is the safe answer and always was, but it is only safe
    by accident of the text having moved, and nothing showed an operator which
    of the twenty applied entries were which before they typed the command.
    """
    out = []
    cache: dict[str, str] = {}
    seen: dict[str, str] = {}
    for e in journal_read():
        if not e.get("applied") or e.get("action") not in (
                "capability_patch", "patch", "force-apply"):
            continue
        parts = _entry_parts(e)
        present = []
        for p in parts:
            mod = p.get("module") or ""
            if mod not in cache:
                try:
                    cache[mod] = Path(mod).read_text()
                except OSError:
                    cache[mod] = ""
            present.append(bool(p.get("after_source")) and
                           p["after_source"] in cache[mod])
        live = bool(present) and all(present)
        sig = e.get("id", "")[5:]
        out.append({
            "id": e.get("id"), "action": e.get("action"),
            "target": e.get("target") or e.get("function"),
            "modules": sorted({Path(p.get("module") or "").name for p in parts}),
            "state": "live" if live else "stale",
            # A duplicate is the same after_source AND signature as an earlier
            # entry — the id suffix already encodes exactly that pair.
            "duplicate_of": seen.get(sig),
        })
        seen.setdefault(sig, e.get("id"))
    return out


def reconcile_journal() -> dict:
    """Mark, without deleting, which applied entries the tree no longer holds.

    The journal is the revert mechanism and the only record of what this loop
    has done, so nothing is ever removed from it. Reconciliation appends ONE
    `reconcile` entry naming the stale ids and the duplicates, which is the
    record a later reader needs in order to trust `--history`.

    Idempotent: reconciling twice with an unchanged tree appends nothing.
    """
    state = journal_state()
    stale = [s["id"] for s in state if s["state"] == "stale"]
    dupes = {s["id"]: s["duplicate_of"] for s in state if s["duplicate_of"]}
    payload = {"stale": stale, "duplicates": dupes}
    for e in reversed(journal_read()):
        if e.get("action") == "reconcile":
            if e.get("stale") == stale and e.get("duplicates") == dupes:
                return {"appended": False, "stale": len(stale),
                        "duplicates": len(dupes), "id": e.get("id")}
            break
    jid = journal_append({
        "action": "reconcile", "applied": False,
        "why": (f"{len(stale)} applied entr(ies) name text the tree no longer "
                f"holds; {len(dupes)} are re-applications of an earlier "
                f"proposal (same after_source + signature)."),
        **payload,
    })
    return {"appended": True, "stale": len(stale), "duplicates": len(dupes),
            "id": jid}


def revert(entry_id: str) -> dict:
    """Put back exactly what was there before that entry was applied."""
    entry = journal_find(entry_id)
    if entry is None:
        return {"ok": False, "why": f"no journal entry {entry_id!r}"}
    if not entry.get("applied"):
        return {"ok": False, "why": f"{entry['id']} was never applied"}
    # A capability patch may span several functions and modules; undoing one
    # half of a paired change leaves the tree in a state neither version ever
    # ran in, so the whole group goes back or none of it does.
    parts = _entry_parts(entry)
    for part in parts:
        m = Path(part["module"])
        # CONTRACT, not judgement. `str.replace("", x, 1)` PREPENDS — it does
        # not no-op — so an entry with an empty `after_source` would inject its
        # whole before-image at the top of a tracked file and report success.
        # `0002-58499a23` carries exactly that (both fields empty, so it is
        # inert today) and every `not in` test passes vacuously for "".
        if not part.get("after_source") or not part.get("before_source"):
            return {"ok": False, "why": (f"{entry['id']} records an empty "
                                         f"before/after source for {m.name}; "
                                         f"reverting it would corrupt the file")}
        if part["after_source"] not in m.read_text():
            return {"ok": False, "why": (f"{m.name} no longer contains the text "
                                         f"{entry['id']} wrote — it has been "
                                         f"edited since (see `--reconcile`); "
                                         f"revert by hand")}
    for part in parts:
        m = Path(part["module"])
        # `explicit=True`: an operator typed a journal id. That IS the request.
        guard.write_tracked_source(
            m, m.read_text().replace(part["after_source"],
                                     part["before_source"], 1),
            what=f"revert {entry['id']}", explicit=True)
    module = Path(parts[0]["module"])
    journal_append({
        "action": "revert", "reverted": entry["id"], "module": str(module),
        "signature": entry.get("signature", ""),
        "before_source": entry["after_source"],
        "after_source": entry["before_source"],
        "applied": True,
    })
    return {"ok": True, "reverted": entry["id"], "module": str(module)}


def apply_journaled(entry_id: str) -> dict:
    """Apply a proposal the acceptance test declined. Nothing is refused forever."""
    entry = journal_find(entry_id)
    if entry is None:
        return {"ok": False, "why": f"no journal entry {entry_id!r}"}
    parts = _entry_parts(entry)
    for part in parts:
        m = Path(part["module"])
        if not part.get("after_source") or not part.get("before_source"):
            return {"ok": False, "why": (f"{entry['id']} records an empty "
                                         f"before/after source for {m.name}; "
                                         f"applying it would corrupt the file")}
        if part["before_source"] not in m.read_text():
            return {"ok": False, "why": f"{m.name} no longer matches the "
                                        f"pre-image {entry['id']} was written against"}
    for part in parts:
        m = Path(part["module"])
        guard.write_tracked_source(
            m, m.read_text().replace(part["before_source"],
                                     part["after_source"], 1),
            what=f"force-apply {entry['id']}", explicit=True)
    module = Path(parts[0]["module"])
    journal_append({
        "action": "force-apply", "forced": entry["id"], "module": str(module),
        "signature": entry.get("signature", ""),
        "before_source": entry["before_source"],
        "after_source": entry["after_source"],
        "verdict_when_proposed": entry.get("why"),
        "applied": True,
    })
    return {"ok": True, "applied": entry["id"]}
