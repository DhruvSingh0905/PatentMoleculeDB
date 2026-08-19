"""The rule gate: what it keeps, what it retries, what it refuses to remember.

These run against REAL cached patent XML and the REAL reader — no fixtures
describing a table that does not exist. The only thing stubbed is the model
call, because the point of every test here is what the loop does with an
answer, not what the answer is.

Written after the gate was rewritten from prediction (score the rule before it
runs) to measurement (run it and look). The behaviours below are the ones that
rewrite is FOR, so a regression in any of them is a regression in the whole
argument for it.
"""
from __future__ import annotations

import json
import re

import pytest

from patentdb3.core import config
from patentdb3.repair import loop as loop_mod
from patentdb3.repair.gap import find_gaps, layout_signature
from patentdb3.repair.loop import apply_rule, repair_patent
from patentdb3.repair.outcome import measure
from patentdb3.repair.rules import (
    COLUMN_MAP, ESCALATE, Invalid, Rule, RuleLibrary, ground,
)
from patentdb3.sources.uspto_assays import (
    _header_rows_of, extract_from_patent, merge_header,
)
from patentdb3.sources.uspto_xml import assemble_blocks, parse_tables

PID = "US8952177"
# A patent with exactly ONE gap no rule covers, so attempt accounting in the
# retry tests is unambiguous. US8952177 cannot be used for those: it extracts
# completely (359/359 usable) and raises no gap at all, so `propose` is never
# reached and every assertion about retries passes vacuously.
#
# Was US10870641 until `bin_legend` learned to read `"+" is <0.1 μM` — a quoted
# symbol, the verb `is`, and a bare comparison, none of which the key grammar
# accepted. That patent now extracts 855 of 855 with a range and raises no gap
# at all, so both escalation tests below passed vacuously against it. A fixture
# that stops being a gap is the good outcome; it just cannot go on standing in
# for one. US10953012 raises exactly one, and a column_map over it mints 11
# records, so the retry and escalation paths are both really exercised.
GAP_PID = "US10953012"
# A block whose column 3 is headed `MS` and whose cells PARSE, so mapping it as
# an assay really does mint 83 records. US9493446's ms columns are the wrong
# fixture: its cells read `MS (ESI) m/z 489.2 [M+H]+`, which `parse_value`
# refuses, so the rule yields nothing and the gate catches it on condition 1
# instead — testing a different branch than the one intended. Across 60 patents
# there are 130 (block, non-assay column) pairs whose cells do parse, so this is
# the common case, not a contrived one.
MS_PID, MS_TABLE, MS_COL = "US10030020", "TABLE-US-00001", 3


def _xml(pid: str = PID) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="ignore")


@pytest.fixture(scope="module")
def table():
    """A real assay block with a compound-id column and numeric assay columns."""
    for t in assemble_blocks(parse_tables(_xml())):
        if t.n_cols >= 3 and len(_header_rows_of(t)[1]) >= 10:
            return t
    pytest.skip("no suitable block")


def _column_map(tbl, name="FLAP Binding wild type HTRF Ki") -> Rule:
    return Rule(fingerprint="test", kind=COLUMN_MAP,
                payload={"cid": 0, "assays": [{"index": 1, "name": name}]})


# ── the gate itself ──────────────────────────────────────────────────

def test_a_rule_that_produces_records_is_positive(table):
    rule = _column_map(table)
    ground(rule, table)
    oc = measure(rule, apply_rule(rule, table, PID), table)
    assert oc.positive
    assert oc.records > 0 and oc.compounds > 0


def test_a_rule_that_produces_nothing_is_negative_and_says_so(table):
    """The commonest failure, and the one US9302989 sat behind for 1,561 rows."""
    rule = Rule(fingerprint="test", kind=COLUMN_MAP,
                payload={"cid": 0, "assays": [{"index": 1, "name": "x"}]})
    oc = measure(rule, [], table)
    assert not oc.positive
    assert "no records" in oc.detail


def test_usability_is_reported_but_never_gated(table):
    """A record missing only its UNIT is still a repair.

    Measured across the corpus: gating on `is_usable` dropped 301 rows from 9
    blocks and every one was missing exactly the unit — `% Aβ Reduction = 42.0`,
    `Hep CLint = 37.66`. The reader itself ships unitless rows, so gating a
    repair on usability held it to a standard the deterministic path does not
    meet.
    """
    rule = _column_map(table)
    ground(rule, table)
    produced = apply_rule(rule, table, PID)
    for r in produced:
        r.unit = None                       # make every record unusable
    oc = measure(rule, produced, table)
    assert oc.usable == 0
    assert oc.positive, "a unitless but otherwise complete record is still a repair"


def test_claiming_a_non_assay_column_is_a_conflict():
    """Condition 3 — the replacement for the adversarial probe battery.

    The battery ran a proposed regex against five synthetic NMR/MS/MW/RT lines.
    This asks whether the rule actually pulled a value out of a column of a real
    patent that the reader identified as mass-spec — a record that is perfectly
    usable and contradicts nothing, whose value happens to be an m/z. Conditions
    1 and 2 are both blind to it.
    """
    from patentdb3.sources.uspto_assays import build_columns

    tbl = next((t for t in assemble_blocks(parse_tables(_xml(MS_PID)))
                if t.table_id == MS_TABLE), None)
    if tbl is None:
        pytest.skip(f"{MS_PID} {MS_TABLE} not present")
    cols = build_columns(tbl)
    assert cols[MS_COL].kind == "ms", f"fixture drifted: column {MS_COL} is {cols[MS_COL].kind}"

    rule = Rule(fingerprint="test", kind=COLUMN_MAP,
                payload={"cid": 0, "assays": [{"index": MS_COL, "name": "IC50"}]})
    produced = apply_rule(rule, tbl, MS_PID)
    oc = measure(rule, produced, tbl)
    assert produced, "the rule does produce records — that is why the gate matters"
    assert not oc.positive
    assert "not a measurement" in oc.detail


# ── the contract half, which still blocks ────────────────────────────

def test_ground_rejects_a_rule_that_cannot_run(table):
    with pytest.raises(Invalid):
        ground(Rule(fingerprint="t", kind=COLUMN_MAP,
                    payload={"cid": 0, "assays": [{"index": 999, "name": "IC50"}]}), table)


def test_ground_drops_an_invented_unit_rather_than_the_rule(table):
    """US9221791: a proposed `nM` on a patent that never says nM, 440 records
    1000x low. No measurement of the output can catch this, so it is checked
    against the DOCUMENT — and the column survives, only the invention is cut."""
    rule = Rule(fingerprint="t", kind=COLUMN_MAP,
                payload={"cid": 0, "assays": [
                    {"index": 1, "name": "FLAP Binding wild type HTRF Ki",
                     "unit": "furlongs"}]})
    ground(rule, table)
    assert rule.payload["assays"][0]["unit"] is None
    assert len(rule.payload["assays"]) == 1, "the column itself must survive"


# ── unit grounding: is this text in the document, and nothing else ───

@pytest.mark.parametrize("source,unit,grounded,why", [
    # Units the old 7-entry allowlist could not express at all. 36 of 55 drops.
    ("Human microsomes T1/2 (min)", "min", True, "time"),
    ("Clearance CL int (μL/min/mg protein)", "µL/min/mg protein", True, "clearance"),
    ("AUCall (min · ng/mL)", "min·ng/mL", True, "whitespace differs from the header"),
    # Micro has three spellings and they are one unit. US11649247 heads sixteen
    # columns `IC50 (μM)` and the model answered `uM`.
    ("IC50 (μM) IC50 (μM)", "uM", True, "ASCII u vs greek mu"),
    ("IC50 (µM)", "uM", True, "micro sign vs ASCII u"),
    ("IC50 (uM)", "μM", True, "greek mu vs ASCII u"),
    ("Cmax (ng/mL)", "ng/ml", True, "litre case carries no meaning"),
    ("values are 10 nM", "nM", True, "in prose"),
    ("cells at 10nM", "nM", True, "no space before the unit"),
    # NANOMETRE IS NOT NANOMOLAR. Both of these are real text from the corpus,
    # and under the old `re.I` each one certified a proposed nanomolar.
    ("purified, monitor 240 and 254 nm). The", "nM", False, "US9221791 wavelength"),
    ("CombiFlash Rf 200, 80 g silica column, λ=220 nm)", "nM", False,
     "US10172859 wavelength"),
    ("column length 250 mm", "mM", False, "millimetre is not millimolar"),
    # Folding micro to `u` stays safe because the M stays capital: US10172859's
    # block holds 19 lowercase `um`, every one inside an ordinary word.
    ("sodium sulfate, tetrahydrofuran, pyrimidin-4-yl", "uM", False, "'um' inside words"),
    ("for the determination of purity", "min", False, "substring inside a word"),
    ("", "nM", False, "empty source"),
])
def test_states_unit(source, unit, grounded, why):
    from patentdb3.repair.rules import _states_unit
    assert _states_unit(source, unit) is grounded, why


def test_body_cells_count_as_the_document(table):
    """The unit row our own header detection misfiled as data still grounds.

    US9682141's merged header is `mTO RC | PI3K α | …` while body rows 0-1 read
    `IC50 | IC50 | …` and `(nM) | (nM) | …` — a three-row header of which
    `_header_rows_of` claimed one. Excluding the body meant the model got blamed
    for reassembling it correctly.
    """
    from patentdb3.sources.uspto_assays import _header_rows_of, table_legend

    tbl = next((t for t in assemble_blocks(parse_tables(_xml("US9682141")))
                if t.table_id == "TABLE-US-00002"), None)
    if tbl is None:
        pytest.skip("US9682141 not cached")
    hr, data = _header_rows_of(tbl)
    header_only = " ".join(c.text for r in hr for c in r) + (tbl.caption or "")
    with_body = header_only + " " + " ".join(c.text for r in data for c in r)
    from patentdb3.repair.rules import _states_unit
    assert not _states_unit(header_only, "nM"), "the header alone does not state it"
    assert _states_unit(with_body, "nM"), "the body does"


def test_raw_source_is_unescaped_but_keeps_its_structure():
    """The model must read `μM`, not `&#x3bc;M` — while colspec and the
    thead/tbody boundary survive, since that is why the source is shown."""
    from patentdb3.repair.gap import raw_block

    xml = _xml("US10030020")
    before = re.search(r"<tables\b[^>]*id=\"TABLE-US-00002\".*?</tables>", xml, re.S)
    if not before or "&#x3bc;" not in before.group(0):
        pytest.skip("no micro entity in this block")
    after = raw_block(xml, "TABLE-US-00002")
    assert "&#x3bc;" not in after and "μ" in after
    assert after.count("<entry") == before.group(0).count("<entry")
    assert "colspec" in after
    # markup entities are LEFT ALONE: `&lt;0.1` unescaped inside an XML view
    # reads as a malformed tag, which is a worse ambiguity than the one it fixes
    assert after.count("&lt;") == before.group(0).count("&lt;")


# ── the retry loop ───────────────────────────────────────────────────

class _Scripted:
    """Stands in for `synthesize.propose`, returning a fixed sequence."""

    def __init__(self, *rules):
        self.rules, self.calls, self.seen_feedback = list(rules), 0, []
        self.seen_models: list[str] = []
        self.seen_full: list[bool] = []

    def __call__(self, gap, *, model="", patent_id="", attempts=None, digest="",
                 full_context=False):
        # `full_context` mirrors `synthesize.propose`. A stub whose signature
        # drifts from the function it stands in for does not fail where the
        # drift is — it fails in four unrelated tests with a TypeError, which
        # is how a new keyword on `propose` read as a broken retry loop.
        self.seen_feedback.append(list(attempts or []))
        self.seen_models.append(model)
        self.seen_full.append(full_context)
        self.calls += 1
        return self.rules.pop(0) if self.rules else None


def _gap_table():
    """The block behind GAP_PID's single gap, plus a column_map that works."""
    from patentdb3.sources.uspto_assays import ASSAY, CID, build_columns

    xml = _xml(GAP_PID)
    tables = assemble_blocks(parse_tables(xml))
    gaps = find_gaps(GAP_PID, tables, extract_from_patent(xml), _source_xml=xml)
    if not gaps:
        pytest.skip(f"{GAP_PID} raises no gap")
    tbl = {t.table_id: t for t in tables}[gaps[0].table_id]
    cols = build_columns(tbl)
    cid = next(i for i, c in enumerate(cols) if c.kind == CID)
    assay = next(i for i, c in enumerate(cols) if c.kind == ASSAY)
    good = Rule(fingerprint="g", kind=COLUMN_MAP,
                payload={"cid": cid, "assays": [
                    {"index": assay, "name": cols[assay].header or "assay"}]})
    return tbl, good


def _inert():
    """Grounds cleanly, applies cleanly, matches nothing — so the retry under
    test is driven by the MEASURED outcome, not by a contract violation."""
    return Rule(fingerprint="x", kind="value_pattern",
                payload={"pattern": r"^(?P<num>ZZZNEVERMATCH\d+)$"})


def _run_with(monkeypatch, tmp_path, scripted, pid=None):
    """One patent through the loop with `propose` scripted and a fresh library."""
    import patentdb3.repair.synthesize as synth
    monkeypatch.setattr(synth, "propose", scripted)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    lib = RuleLibrary(path=tmp_path / "rules.json")
    # journal into tmp too — see repair_patent's `journal` parameter
    return repair_patent(pid or GAP_PID, _xml(pid or GAP_PID), library=lib,
                         max_calls=9, journal=tmp_path / "journal.jsonl"), lib


def test_a_bad_rule_is_retried_with_its_own_measurement(monkeypatch, tmp_path):
    """Attempt 1 yields nothing; attempt 2 works. The loop must ask twice, and
    the second ask must CARRY what the first one did — otherwise a retry at
    temperature 0 is byte-for-byte the same call and cannot produce a different
    answer. That carrying is the entire reason a retry is worth paying for."""
    _, good = _gap_table()
    scripted = _Scripted(_inert(), good)
    (_, rep), lib = _run_with(monkeypatch, tmp_path, scripted)

    assert scripted.calls >= 2, "a failing rule must be retried"
    assert scripted.seen_feedback[0] == [], "the first ask carries no history"
    assert scripted.seen_feedback[1], "the second ask must carry the first outcome"
    assert "no records" in scripted.seen_feedback[1][0]["why"]
    assert rep.adopted >= 1
    assert any(r.kind == COLUMN_MAP for r in lib._rules.values())


def test_three_failures_revert_and_skip(monkeypatch, tmp_path):
    """Nothing the model proposed may survive — not in the library, not in the
    output. What IS remembered is that the layout resisted, as an escalation,
    so the next run does not re-buy the same three failures. An escalation
    expires on SYNTH_EPOCH, so improving the prompt re-opens it automatically."""
    scripted = _Scripted(*[_inert() for _ in range(6)])
    (_, rep), lib = _run_with(monkeypatch, tmp_path, scripted)

    assert rep.adopted == 0
    kinds = {r.kind for r in lib._rules.values()}
    assert kinds <= {ESCALATE}, f"a failed proposal must not be remembered as a rule: {kinds}"
    assert any("attempts" in (r.payload or {}) for r in lib._rules.values())
    assert any("no rule survived" in str(e.get("capability")) for e in rep.escalations)


def test_the_loop_never_exceeds_max_attempts_per_layout(monkeypatch, tmp_path):
    scripted = _Scripted(*[_inert() for _ in range(40)])
    (_, rep), _ = _run_with(monkeypatch, tmp_path, scripted)
    unknown = max(1, rep.gaps_found - rep.already_known)
    assert rep.proposed <= loop_mod.MAX_ATTEMPTS * unknown
    assert scripted.calls <= loop_mod.MAX_ATTEMPTS * unknown


def test_a_failed_layout_is_not_re_bought_on_the_next_run(monkeypatch, tmp_path):
    """The amortisation argument. Three attempts is a per-layout budget, not a
    per-run one — otherwise every corpus pass pays for the same three failures
    again, which is how the cost advantage over HARVEST is lost."""
    first = _Scripted(*[_inert() for _ in range(6)])
    (_, rep1), lib = _run_with(monkeypatch, tmp_path, first)
    assert first.calls > 0

    import patentdb3.repair.synthesize as synth
    second = _Scripted(*[_inert() for _ in range(6)])
    monkeypatch.setattr(synth, "propose", second)
    repair_patent(GAP_PID, _xml(GAP_PID), library=lib, max_calls=9,
                  journal=tmp_path / "journal.jsonl")
    assert second.calls == 0, "an exhausted layout must not be paid for twice"


def test_no_api_key_means_no_rule_and_no_crash(tmp_path):
    """The state this session actually hit: `propose` returns None because the
    account has no credit. The loop must escalate rather than remember an empty
    answer or raise, and the reader's own records must still come back."""
    lib = RuleLibrary(path=tmp_path / "nokey.json")
    lib._rules.clear()
    import patentdb3.repair.synthesize as synth
    old = config.ANTHROPIC_API_KEY
    try:
        config.ANTHROPIC_API_KEY = synth.config.ANTHROPIC_API_KEY = ""
        recs, rep = repair_patent(GAP_PID, _xml(GAP_PID), library=lib, max_calls=3,
                                  journal=tmp_path / "journal.jsonl")
    finally:
        config.ANTHROPIC_API_KEY = synth.config.ANTHROPIC_API_KEY = old
    assert len(recs) > 0, "the baseline reader's records must still come back"
    assert not any(r.kind == COLUMN_MAP for r in lib._rules.values())
    assert rep.adopted == 0


# ── the digest ───────────────────────────────────────────────────────

def test_digest_excludes_rules_that_cannot_inform_this_layout(tmp_path):
    lib = RuleLibrary(path=tmp_path / "d.json")
    lib._rules.clear()
    target = "5|cid,num,num,text,num|example+no,herg+ic,flap+ki,notes,cl"
    cases = {
        "same_shape_worked": ("5|cid,bin,bin,text,num|example+no,herg+ic,pk,notes,cl", 880),
        "exact_same_layout": (target, 500),
        "wrong_column_count": ("3|cid,num,text|compound,ic,notes", 5000),
        "never_produced_a_row": (target[:-1] + "x", 0),
    }
    for fp, (sig, rows) in cases.items():
        lib._rules[fp] = Rule(fingerprint=fp, kind=COLUMN_MAP, signature=sig,
                              rows_yielded=rows, times_applied=1,
                              payload={"cid": 0, "assays": [{"index": 1, "name": "IC50"}]})
    got = {r.fingerprint for _, r in lib.similar(target)}
    assert got == {"same_shape_worked"}, got
    assert lib.digest(target), "a neighbour exists, so the digest must render"
    assert len(lib.digest(target)) < 1200, "the digest must stay small by construction"


def test_signature_is_the_fingerprints_preimage():
    """If these two ever describe different things, similarity is measured on
    something other than what the library is keyed by."""
    import hashlib

    from patentdb3.repair.gap import layout_fingerprint

    for t in assemble_blocks(parse_tables(_xml()))[:20]:
        h = merge_header(t, _header_rows_of(t)[0])
        sig = layout_signature(t, h)
        assert layout_fingerprint(t, h) == hashlib.sha256(sig.encode()).hexdigest()[:16]


# ── the reader is not in scope and must not move ─────────────────────

def test_the_reader_is_untouched_by_any_of_this():
    recs = extract_from_patent(_xml())
    assert len(recs) == 359
    assert len({r.cid for r in recs}) == 190
    assert all(r.table_id and r.column_header and r.unit for r in recs)


# ── the self-heal switch ─────────────────────────────────────────────

def test_self_heal_flag_selects_the_pipeline(monkeypatch, tmp_path):
    """`--dump` must be able to run reader-only, and must SAY which it did.

    `dump()` called `extract_from_patent` directly until the loop was wired in,
    so every dump was a reader measurement that read like a pipeline one. The
    flag is only half the fix — the manifest recording the mode is the other
    half, because two dumps produced by different pipelines are otherwise
    indistinguishable files.
    """
    from patentdb3 import verify as V

    monkeypatch.setattr(V, "DUMP_PATH", tmp_path / "d.tsv")
    monkeypatch.setattr(V, "MANIFEST_PATH", tmp_path / "m.json")
    # STRUCT_PATH was missing here, and `dump()` opens all three. This test
    # predates the structures artifact — when `STRUCT_PATH` was added to
    # `dump()` the redirect was not — so every run of the suite overwrote the
    # real `out/structures.tsv` with one patent's worth of rows while leaving
    # `reader_dump.tsv` and `latest.json` correct. The manifest then described
    # 64,465 rows across 137 patents next to a file holding 328 from one, with
    # nothing raising. Found by an audit reading the artifact mid-suite, not by
    # the suite. `conftest.py` now redirects all three session-wide; this line
    # is belt-and-braces so the test is correct read on its own.
    monkeypatch.setattr(V, "STRUCT_PATH", tmp_path / "s.tsv")

    V.dump([PID], heal=False)
    man = json.loads((tmp_path / "m.json").read_text())
    assert man["self_heal"] is False
    assert man["usd_spent"] == 0.0
    assert man["rows"] > 0
    reader_rows = man["rows"]

    # healing on: US8952177 raises no gap, so this costs nothing and must
    # produce exactly what the reader produced — the loop returns
    # `baseline + recovered`, and recovered is empty.
    V.dump([PID], heal=True)
    man = json.loads((tmp_path / "m.json").read_text())
    assert man["self_heal"] is True
    assert man["rows"] == reader_rows, "a patent with no gaps must be unchanged"
    assert man["usd_spent"] == 0.0, "no gaps means no paid call"


def test_config_default_is_heal_on(monkeypatch):
    """A self-heal that must be remembered is one that does not run.

    Reads the DEFAULT, not the ambient value. The first version asserted
    `config.SELF_HEAL is True` directly and so failed for anyone who had
    exported `SELF_HEAL=0` — which is exactly what an agent was told to do
    while working on the identity route, and it then reported a green suite as
    "1 pre-existing failure". A test that asserts a config default has to
    control the environment it reads, or it is asserting the shell.
    """
    import importlib

    monkeypatch.delenv("SELF_HEAL", raising=False)
    fresh = importlib.reload(config)
    try:
        assert fresh.SELF_HEAL is True
    finally:
        importlib.reload(config)      # restore whatever the caller's env says


# ── normalize_cid's length bound ─────────────────────────────────────────

def test_normalize_cid_refuses_a_compound_NAME_as_an_id():
    """THE REGRESSION. `build_columns` can mistake a `Name` column for the id
    column, and when it did, the compound's own IUPAC name became its cid —
    410 assay rows across US9611261 and US9018217, each keyed by a 70-344
    character "id" that can never join to anything.

    Nothing downstream could tell that from a real id: it is a non-empty
    string in the cid field, so every consumer treated it as one. The upstream
    defect is real and lives in `build_columns`; this is the contract check
    that stops it becoming silent data.
    """
    from patentdb3.sources.uspto_assays import normalize_cid
    name = ("5,7-dimethyl-2((1-methyl-4-phenyl-1H-imidazol-2- ylthio)methyl)"
            "imidazo[1,2-a]pyrimidine")
    assert normalize_cid(name) == ""


def test_normalize_cid_keeps_every_real_id_shape_in_this_corpus():
    """The bound is a LENGTH, not a shape, because `_CID_SHAPE` permits only
    one trailing letter and 505 perfectly good corpus ids already fail it.
    `α-6-mPEG1-O-Codeine` (19 chars, US9233167) is the longest real one seen
    and it must survive — the corpus is bimodal with an empty gap between 29
    and 70 characters, which is where the bound sits.
    """
    from patentdb3.sources.uspto_assays import normalize_cid
    for raw, want in [("7", "7"), ("Example 007", "7"), ("Cpd. No. 7", "7"),
                      ("100AA", "100AA"), ("10-1", "10-1"), ("I-0020", "I-20"),
                      ("α-6-mPEG1-O-Codeine", "α-6-mPEG1-O-Codeine")]:
        assert normalize_cid(raw) == want, raw


def test_the_last_attempt_escalates_to_the_strongest_model(monkeypatch, tmp_path):
    """Two attempts have already failed on this layout with the cheap model and
    the small view; a third of the same buys the same answer. The last one goes
    to Opus with the wider views served up front.

    Bounded, because the first version was not: firing Opus on every gap spent
    $1.04 on US11566007 against a $0.20 soft cap, and still adopted nothing."""
    scripted = _Scripted(*[_inert() for _ in range(6)])
    _run_with(monkeypatch, tmp_path, scripted)

    assert scripted.calls >= 2, "the loop must retry before it escalates"
    assert not scripted.seen_full[0], "the first attempt is the cheap one"
    assert scripted.seen_models[0] != config.MODEL_OPUS
    # The escalation fires at most once, on the last attempt for one gap.
    assert scripted.seen_full.count(True) <= 1
    if True in scripted.seen_full:
        i = scripted.seen_full.index(True)
        assert scripted.seen_models[i] == config.MODEL_OPUS, \
            "the escalation must go to the strongest model, not just get more context"
