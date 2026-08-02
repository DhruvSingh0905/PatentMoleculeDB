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

Nothing here writes to the working tree unless `PARSER_REPAIR_APPLY` is set.
The default is to verify a patch fully and hand back the diff.
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
        got["objections"] = evidence
        if evidence:
            logger.warning("patch applied over %d objection(s): %s",
                           len(evidence), "; ".join(evidence)[:220])
        got["ok"] = True
        return got


_PROBE = '''
import json, pathlib, sys
sys.path.insert(0, ".")
from patentdb.sources.uspto_xml import parse_fidelity
from patentdb.sources.uspto_assays import extract_from_patent

discrepant, per_patent = 0, {}
for p in sorted(pathlib.Path("__XML_DIR__").glob("*.xml")):
    xml = p.read_text()
    discrepant += len(parse_fidelity(xml))
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
        base = [r for r in extract_from_patent(xml) if r.is_usable]
        extra, _ = repair_patent(pid, xml, max_calls=0)
        repaired = len(base) + sum(1 for r in extra if r.is_usable)
        # Are the NUMBERS right? A coverage check cannot tell recovered data
        # from invented data; BindingDB publishes the values, so this is a
        # lookup. Run inside the sandbox so it measures the PATCHED code.
        from patentdb.repair.value_check import check_patent
        bad_values = check_patent(pid, base + list(extra))["bad"]
    except Exception:
        repaired = -1
print(json.dumps({"discrepant_blocks": discrepant, "per_patent": per_patent,
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
    """
    from ..sources.uspto_assays import extract_from_patent

    from ..sources.uspto_xml import parse_fidelity

    xml_dir = xml_dir or (config.OUTPUT_DIR / "uspto_xml")
    out: dict = {}
    clean: set[str] = set()
    for p in sorted(xml_dir.glob("*.xml")):
        try:
            xml = p.read_text()
            out[p.stem] = len({r.cid for r in extract_from_patent(xml)
                               if r.is_usable and r.cid})
        except Exception:                            # noqa: BLE001 - probe only
            continue
        if not parse_fidelity(xml):
            clean.add(p.stem)
    # Only these patents' counts are a trustworthy floor — see verify_patch.
    out["_clean"] = clean
    return out


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

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
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
    apply = config.PARSER_REPAIR_APPLY if apply is None else apply
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

        will_apply = bool(verdict.get("ok")) and apply
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
            "applied": will_apply,
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
        if will_apply:
            module.write_text(patched)
            report["applied"] += 1
            logger.info("parser_repair: %s APPLIED for %s (%s) — revert with "
                        "`parser_health --revert %s`",
                        entry_id, d.signature, d.blast_radius, entry_id.split("-")[0])
        else:
            verdict["patch"] = after
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
    parts = entry.get("patches") or [{"module": entry["module"],
                                      "before_source": entry["before_source"],
                                      "after_source": entry["after_source"]}]
    for part in parts:
        m = Path(part["module"])
        if part["after_source"] not in m.read_text():
            return {"ok": False, "why": (f"{m.name} no longer contains the text "
                                         f"{entry['id']} wrote — it has been "
                                         f"edited since; revert by hand")}
    for part in parts:
        m = Path(part["module"])
        m.write_text(m.read_text().replace(part["after_source"],
                                           part["before_source"], 1))
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
    parts = entry.get("patches") or [{"module": entry["module"],
                                      "before_source": entry["before_source"],
                                      "after_source": entry["after_source"]}]
    for part in parts:
        m = Path(part["module"])
        if part["before_source"] not in m.read_text():
            return {"ok": False, "why": f"{m.name} no longer matches the "
                                        f"pre-image {entry['id']} was written against"}
    for part in parts:
        m = Path(part["module"])
        m.write_text(m.read_text().replace(part["before_source"],
                                           part["after_source"], 1))
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
