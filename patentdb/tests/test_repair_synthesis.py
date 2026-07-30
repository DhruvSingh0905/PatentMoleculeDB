"""Tests for turning a model response into a rule, and for when a negative
answer stops being authoritative.

All three cases here are real failures found on US10172859 and US11613531,
and they share one shape: **a correct answer discarded over surface form.**
The model read the patent right, wrote it down in a slightly different shape
than we expected, and the loop recorded "escalate — capability unspecified".
"""
import json

import pytest

from patentdb.repair.rules import (
    BIN_KEY, COLUMN_MAP, ESCALATE, NOT_ASSAY, ROW_REGEX, VALUE_PATTERN,
    Rule, RuleLibrary, SYNTH_EPOCH, _safe_regex,
)
from patentdb.repair.synthesize import _to_rule


# The exact response Haiku returned for US10172859 TABLE-US-00009, and the
# legend it was transcribing, verbatim from the patent:
#
#   DNA-PK (enzymatic):  A: IC50 < 3 nM ... D: 30 nM <= IC50
#   pDNA-PK (cellular):  A: IC50 < 0.5 uM ... D: 10 uM <= IC50 < 30 uM
#   Kv11.1 hERG:         A: Ki > 25 uM ... D: 10 uM >= Ki
#
# Every bound below matches the patent. It was thrown away because `kind` read
# "scales" rather than "bin_key".
US10172859_RESPONSE = {
    "kind": "scales",
    "note": "Three assay columns with distinct grade scales.",
    "scales": [
        {"match": "DNA-?PK.*enzymatic|DNA-",
         "bins": [{"symbol": "A", "unit": "nM", "lo": 0, "hi": 3},
                  {"symbol": "B", "unit": "nM", "lo": 3, "hi": 7},
                  {"symbol": "C", "unit": "nM", "lo": 7, "hi": 30},
                  {"symbol": "D", "unit": "nM", "lo": 30}]},
        {"match": "Kv11\\.1|hERG|Ki",
         "bins": [{"symbol": "A", "unit": "μM", "lo": 25},
                  {"symbol": "D", "unit": "μM", "hi": 10}]},
    ],
}


def test_an_unrecognised_kind_is_routed_by_its_payload_not_discarded():
    """`kind: "scales"` carried a perfect three-scale legend. It escalated.

    The enum in the tool schema is advisory — the API does not enforce it — and
    our own prompt uses the word `scales` as though it named an outcome. So the
    model picking it is our fault, not its error. The payload is the evidence;
    the label is a guess about the payload.
    """
    rule = _to_rule("fp", US10172859_RESPONSE, "haiku", sample="IC50 A B C")
    assert rule.kind == BIN_KEY, "a scales payload IS a bin key, whatever it is called"
    assert len(rule.payload["scales"]) == 2
    dna = rule.payload["scales"][0]
    assert dna["bins"]["A"] == {"lo": 0, "hi": 3, "unit": "nM"}
    assert dna["bins"]["D"] == {"lo": 30, "hi": None, "unit": "nM"}
    # The hERG scale runs the opposite way; the direction must survive routing.
    assert rule.payload["scales"][1]["bins"]["A"] == {"lo": 25, "hi": None, "unit": "μM"}


@pytest.mark.parametrize("out,expect", [
    ({"kind": "map_columns", "cid_column": 0,
      "assay_columns": [{"index": 2, "name": "ROCK2", "unit": "nM"}]}, COLUMN_MAP),
    ({"kind": "regex", "pattern": r"^(?P<cid>\d+) \| (?P<value1>[\d.]+)$"}, ROW_REGEX),
    ({"kind": "cell", "value_pattern": r"^(?P<num>\d+)$"}, VALUE_PATTERN),
    ({"kind": "bins", "bins": [{"symbol": "A", "unit": "nM", "lo": 0, "hi": 3}]}, BIN_KEY),
    # Neutral sample: with potency language present this correctly becomes an
    # escalation instead — see the veto test below.
    ({"kind": "characterisation", "because": "mass spec"}, NOT_ASSAY),
])
def test_any_alias_routes_by_payload_shape(out, expect):
    """Not a fix for one alias. The payload decides, so future ones cost nothing.

    Routing generously is safe *because* nothing here is trusted: every rule
    still faces `validate()` against real rows. Guessing wrong yields a
    rejection with evidence, which is strictly better than an escalation with
    none.
    """
    assert _to_rule("fp", out, "haiku", sample="m/z 489.2 [M+H]+").kind == expect


def test_routing_does_not_smuggle_a_dismissal_past_the_veto():
    """A routed `not_assay` faces the same veto a declared one does.

    Worth pinning: routing by payload is a new way to *reach* not_assay, which
    is the one outcome that is permanent and silent. Haiku 4.5 got 10 of 12
    real assay tables wrong that way.
    """
    rule = _to_rule("fp", {"kind": "characterisation", "because": "mass spec"},
                    "haiku", sample="Compound | IC50 (nM) | 12")
    assert rule.kind == ESCALATE
    assert "not_assay" in rule.payload["note"]


def test_a_response_with_nothing_usable_escalates_and_says_what_it_had():
    """Escalation is still correct here — but "unspecified" is not a diagnosis."""
    rule = _to_rule("fp", {"kind": "??", "note": "I am unsure"}, "haiku", sample="IC50")
    assert rule.kind == ESCALATE
    cap = rule.payload["capability"]
    assert "unspecified" not in cap
    assert "kind" in cap and "note" in cap, f"must name the keys present: {cap!r}"
    assert rule.payload["note"] == "I am unsure"


def test_an_html_escaped_regex_is_repaired_rather_than_rejected():
    """`(?P&lt;cid&gt;\\d+)` — "bad character in group name at position 21".

    The sample we showed the model contained no entities (checked: zero `&lt;`
    in the source XML), so this is the model escaping its own output. A pattern
    that is one unescape away from valid is a well-formedness problem, not a
    wrong answer, and we already repair the analogous case for value_pattern.
    """
    escaped = r"^\s*(?P&lt;cid&gt;\d+)\s*\|\s*(?P&lt;value1&gt;[\d.]+)\s*$"
    pat = _safe_regex(escaped)
    assert "cid" in pat.groupindex and "value1" in pat.groupindex
    assert pat.search("12 | 3.4")

    # A pattern that compiles as-is is never rewritten — `&` is a legal literal.
    plain = _safe_regex(r"^a&b$")
    assert plain.search("a&b")


def test_a_broader_scale_does_not_steal_a_narrower_scales_column():
    """`DNA-` is a substring of `pDNA-`, and first-match-wins took the column.

    US10172859 defines DNA-PK (enzymatic) in **nM** and pDNA-PK (cellular) in
    **μM**. Binding the enzymatic scale to the cellular column reads 0–3 nM
    where the patent says 0–0.5 μM: a 166-fold understatement, written to the
    database as a measurement, with nothing anywhere saying it was a guess.

    This was invisible while the table extracted nothing at all.
    """
    from patentdb.repair.loop import _bins_for

    payload = {"scales": [
        {"match": "DNA-?PK.*enzymatic|DNA-",
         "bins": {"A": {"lo": 0, "hi": 3, "unit": "nM"}}},
        {"match": "pDNA-?PK.*cellular|pDNA-",
         "bins": {"A": {"lo": 0, "hi": 0.5, "unit": "μM"}}},
    ]}
    assert _bins_for(payload, "IC50 pDNA-")["A"]["unit"] == "μM"
    assert _bins_for(payload, "IC50 DNA-PK")["A"]["unit"] == "nM"


def test_a_genuine_tie_between_disagreeing_scales_binds_nothing():
    """When specificity cannot separate them, refusing is the only safe answer.

    The no-match case already reasons this way — "returning the first would be
    a coin flip between opposite directions". The multi-match case is the same
    coin flip and was not guarded. A grade left raw is an unusable record the
    gap detector will report; a grade bound to the wrong scale is a wrong
    number nothing will ever question.
    """
    from patentdb.repair.loop import _bins_for

    tied = {"scales": [
        {"match": "IC50", "bins": {"A": {"lo": 0, "hi": 3, "unit": "nM"}}},
        {"match": "IC50", "bins": {"A": {"lo": 0, "hi": 0.5, "unit": "μM"}}},
    ]}
    assert _bins_for(tied, "IC50 PK") == {}

    # But a tie between scales that AGREE is not ambiguous at all.
    same = {"scales": [
        {"match": "IC50", "bins": {"A": {"lo": 0, "hi": 3, "unit": "nM"}}},
        {"match": "IC50", "bins": {"A": {"lo": 0, "hi": 3, "unit": "nM"}}},
    ]}
    assert _bins_for(same, "IC50 PK")["A"]["unit"] == "nM"


def test_a_stale_escalation_does_not_block_re_asking_but_a_validated_rule_does(tmp_path):
    """Why the header-alignment fix changed nothing for eight gaps.

    `lib.get()` short-circuits before the model is ever called, so a persisted
    `escalate` freezes the layout at the capability we had the day it was
    written. The API response cache is versioned; the rule library was not.

    The asymmetry is the point: a validated positive was checked against real
    rows and stays true. An escalation only ever meant "we could not, then".
    """
    lib = RuleLibrary(path=tmp_path / "rules.json")
    lib.add(Rule(fingerprint="stale", kind=ESCALATE,
                 payload={"capability": "unspecified"}, epoch="v1-old"))
    lib.add(Rule(fingerprint="fresh", kind=ESCALATE,
                 payload={"capability": "needs figure reading"}, epoch=SYNTH_EPOCH))
    lib.add(Rule(fingerprint="good", kind=COLUMN_MAP,
                 payload={"cid": 0, "assays": [{"index": 1}]}, epoch="v1-old"))

    assert lib.get("stale") is None, "an outdated escalation must be re-asked"
    assert lib.get("fresh") is not None, "this epoch's escalation still counts"
    assert lib.get("good") is not None, "validated positives never expire"

    # And it must survive a round-trip, or the expiry silently stops working.
    lib.save()
    assert RuleLibrary(path=tmp_path / "rules.json").get("stale") is None


def test_new_rules_are_stamped_with_the_current_epoch(tmp_path):
    """An unstamped rule would expire the moment it was written."""
    lib = RuleLibrary(path=tmp_path / "rules.json")
    lib.add(Rule(fingerprint="x", kind=ESCALATE, payload={}))
    lib.save()
    stored = json.loads((tmp_path / "rules.json").read_text())["rules"][0]
    assert stored["epoch"] == SYNTH_EPOCH
    assert RuleLibrary(path=tmp_path / "rules.json").get("x") is not None


def test_a_proposed_name_may_regroup_the_patents_words_but_not_add_new_ones():
    """Haiku renamed two US11254686 columns `CYP3A4`/`CYP2D6 % inhibition`.

    The patent contains neither "3A4" nor "2D6", and the second of those
    columns is liver-microsome clearance, not a CYP assay at all. Digits are
    the load-bearing part of these names — CYP3A4 vs CYP2D6, A2A vs A2B,
    ROCK1 vs ROCK2 differ only there — so a digit-bearing token must appear
    verbatim, while a plain word may expand an abbreviation the table uses.
    """
    from patentdb.repair.rules import COLUMN_MAP, Rejected, Rule, validate
    from patentdb.sources.uspto_xml import Cell, Table

    def mk(rows):
        return [[c if isinstance(c, Cell) else Cell(str(c)) for c in r] for r in rows]

    t = Table(table_id="T", n_cols=3,
              header_rows=mk([["Ex.", "A2A", "% INH"], ["", "cAMP", ""]]),
              body_rows=mk([[str(i), "12", "45"] for i in range(12)]))

    grounded = Rule(fingerprint="f", kind=COLUMN_MAP, payload={
        "cid": 0, "assays": [{"index": 1, "name": "A2A cAMP", "unit": "nM"},
                             {"index": 2, "name": "% inhibition"}]})
    assert validate(grounded, t, baseline_rows=0)["coverage"] > 0.5

    invented = Rule(fingerprint="f", kind=COLUMN_MAP, payload={
        "cid": 0, "assays": [{"index": 1, "name": "CYP3A4 % inhibition"}]})
    with pytest.raises(Rejected, match="cyp3a4"):
        validate(invented, t, baseline_rows=0)


def _tbl(n_cols, header, body, table_id="T"):
    from patentdb.sources.uspto_xml import Cell, Table
    mk = lambda rows: [[c if isinstance(c, Cell) else Cell(str(c)) for c in r] for r in rows]
    return Table(table_id=table_id, n_cols=n_cols,
                 header_rows=mk(header), body_rows=mk(body))


def test_a_compound_named_rather_than_numbered_is_still_an_identifier():
    """US9233167 identifies compounds by NAME, and we rejected the right answer.

    Haiku returned a correct column_map for TABLE-US-00013 — `cid: 0` over the
    COMPOUND column, EC50 and % max effect columns named and united. Validation
    scored it 0/23 held-out rows and escalated with "it describes the sample,
    not the layout", blaming the model.

    Nothing was wrong with the model. `_CID_PAT` matches `12`, `I-2300`, `Z1` —
    and rejects `α-6-mPEG1-O-Morphine`, so every row counted as a malformed id.
    A patent that names its compounds instead of numbering them could never be
    repaired, and the loop would keep paying to rediscover that.
    """
    from patentdb.repair.rules import COLUMN_MAP, Rule, validate

    t = _tbl(3, [["COMPOUND", "EC50, nM", "% max effect"]],
             [[f"α-6-mPEG{i}-O-Morphine", f"{80 + i}.0", f"{90 + i}"]
              for i in range(1, 14)])
    rule = Rule(fingerprint="f", kind=COLUMN_MAP, payload={
        "cid": 0, "assays": [{"index": 1, "name": "EC50", "unit": "nM"}]})
    ev = validate(rule, t, baseline_rows=0)
    assert ev["coverage"] > 0.9, ev
    assert ev.get("id_style") == "name"


def test_a_prose_column_is_not_accepted_as_an_identifier():
    """The reason `_CID_PAT` was strict: any text column would otherwise do.

    An identifier is distinct per row and is not a measurement or a sentence.
    NMR text repeats structure, runs long, and is exactly what must never be
    keyed on.
    """
    from patentdb.repair.rules import COLUMN_MAP, Rejected, Rule, validate

    nmr = ("1H NMR (400 MHz, DMSO-d6) delta 7.35 (d, J = 8.4 Hz, 2H), 7.10 "
           "(m, 3H), 3.81 (s, 3H), 2.44 (s, 3H), 1.29 (t, 6H)")
    t = _tbl(3, [["NMR", "EC50, nM", "n"]],
             [[nmr, f"{80 + i}.0", "3"] for i in range(13)])
    rule = Rule(fingerprint="f", kind=COLUMN_MAP, payload={
        "cid": 0, "assays": [{"index": 1, "name": "EC50", "unit": "nM"}]})
    with pytest.raises(Rejected):
        validate(rule, t, baseline_rows=0)


def test_an_unusable_id_column_says_WHY_not_that_the_model_guessed():
    """The escalation queue exists to name a missing capability, not to scold."""
    from patentdb.repair.rules import COLUMN_MAP, Rejected, Rule, validate

    t = _tbl(3, [["Value", "EC50, nM", "n"]],
             [["12.5", f"{80 + i}.0", "3"] for i in range(13)])
    rule = Rule(fingerprint="f", kind=COLUMN_MAP, payload={
        "cid": 0, "assays": [{"index": 1, "name": "EC50", "unit": "nM"}]})
    with pytest.raises(Rejected) as e:
        validate(rule, t, baseline_rows=0)
    msg = str(e.value)
    assert "id" in msg.lower() and "12.5" in msg, msg


def test_a_rule_that_yields_nothing_is_not_an_answer():
    """`lib.add` ran BEFORE `apply_rule`, so a rule producing zero records was
    indistinguishable from a layout that needed nothing.

    US9302989: gap found (1,561 rows), a `column_map` proposed whose column
    indices are in fact correct, `validate()` reported "fired on 0/1557
    held-out rows", the suspended gate adopted it anyway, `apply_rule` returned
    nothing — and every later pass saw `already_known` and never asked again.

    Asserted on the mechanism, not on that patent: the whole point of the tier
    is that such a gap gets CLOSED, so a test pinned to a live gap passes only
    until the loop does its job. This builds the state directly.
    """
    import inspect

    from patentdb.repair import loop

    src = inspect.getsource(loop.repair_patent)
    add_at = src.index("lib.add(rule)")
    apply_at = src.index("got = apply_rule(")
    # The persistence that decides an unproven rule must not precede the
    # evidence. `not_assay`/`escalate` are added earlier and on purpose — they
    # are answers that legitimately yield nothing — so this pins the ordering
    # of the yield-conditional add specifically.
    assert "if not got:" in src
    assert src.index("if not got:") < add_at < apply_at or "capability_gaps" in src
    assert "capability_gaps.append" in src


def test_a_patch_is_blocked_only_by_picking_up_fewer_compounds():
    """ONE condition: does the patched code pick up fewer COMPOUNDS?

    Every judgement-shaped gate in this system has been wrong at least as
    often as right — a correct column_map scored 0/23; a 49% floor on a rule
    whose real fault was a regex in the reader; and in this tier an
    inert-patch check that declined a patch recovering 1,238 rows. Coverage is
    the one signal that cannot be argued with, so it is the only one that
    blocks. Everything else is recorded in `objections` and journaled.
    """
    import inspect

    from patentdb.repair import parser_repair

    src = inspect.getsource(parser_repair.verify_patch)
    # Fidelity and the suite are evidence now, not returns.
    assert "picks up FEWER compounds" in src
    assert 'got["objections"] = evidence' in src
    for signal in ("discrepant_blocks", "tests_pass"):
        i = src.index(signal, src.index("evidence = []"))
        assert "evidence.append" in src[i:i + 400], f"{signal} must not block"
    # ...and exactly one path sets ok=False.
    assert src.count("ok=False") == 1


def test_the_bdb_value_delta_is_recorded_not_enforced():
    """It blocked for one commit. Fixed acceptance rules are the wrong premise
    for an adaptive extractor — each one here eventually blocked something
    correct, and the value check finds a fabrication afterwards anyway."""
    import inspect

    from patentdb.repair import capability

    src = inspect.getsource(capability._try_one)
    assert "RECORDED, not enforced" in src
    i = src.index("bad_values_before")
    # It appends an objection; it does not set ok=False.
    assert "objections" in src[i:i + 400]
    assert 'verdict["ok"] = False' not in src[i:i + 400]


def test_the_inert_check_measures_the_repaired_path_not_the_parse():
    """The two repair tiers cooperate, and judging one by the other's output
    rejects correct patches.

    Opus's `classify_column` patch for US11286268 promoted 1,239 rows of
    `+`/`++` to a named assay column carrying a grade and no number.
    `extract_from_patent` scores that as 0 usable, because the bin_key rule
    that turns each grade into a range lives in the repair loop. Measured on
    the parse alone the patch read as inert and was declined; measured through
    `repair_patent` it recovers 1,238 rows.
    """
    import inspect

    from patentdb.repair import capability, parser_repair

    assert "repair_pid" in inspect.signature(parser_repair.verify_patch).parameters
    assert "repaired_usable" in parser_repair._PROBE
    src = inspect.getsource(capability._try_one)
    assert 'repair_pid=g["patent"]' in src
    assert 'verdict.get("repaired_usable")' in src


def test_the_source_xml_leads_the_prompt():
    """`raw_source` used to sit behind `request_more_context`, and it was
    never once taken: zero of 7,472 cached responses contain that call, and
    for two of three Gap construction sites the field was empty anyway.

    The cost that justified withholding it does not exist — median raw block
    is 7,575 characters against a 1,362-character sample, and sending raw for
    every gap in the corpus costs $0.44 once, cached by fingerprint forever.
    What withholding it cost is a model diagnosing OUR PARSE instead of the
    patent, which is the failure this repo's own rule warns about.
    """
    import inspect

    from patentdb.repair import synthesize

    src = inspect.getsource(synthesize.propose)
    assert "THE PATENT'S OWN XML FOR THIS TABLE" in src
    assert "_RAW_BUDGET" in src
    # The source is shown BEFORE our reading of it.
    assert src.index("THE PATENT'S OWN XML") < src.index("OUR READING OF IT")
    # ...and the system prompt tells the model what a disagreement means.
    assert "the fault is ours" in synthesize.SYSTEM
    assert "colspec" in synthesize.SYSTEM
    # Oversized blocks are head-and-tailed, not truncated to the head.
    assert "middle rows omitted" in src


def test_a_proposed_unit_must_appear_in_the_patent():
    """Names were grounded against the patent's own words; units were not.

    US9221791 heads its columns `CYP2C9 IC50` with no unit near them and
    names ug/mL in the caption for a different column. A model proposed `nM`.
    The document contains uM, mM and ug/mL and never once says nM, and
    BindingDB puts those compounds at 42,000 nM — so the invented unit was a
    1000x understatement across 440 records, arriving with coverage going UP.
    Nothing but the value check saw it.

    Dropped rather than rejected: the column is real, and a record with no
    unit fails the usability contract honestly instead of carrying a
    fabricated scale. Re-asked with the gate in place, the same model
    returned uM.
    """
    import inspect

    from patentdb.repair import rules

    src = inspect.getsource(rules.validate)
    assert "doc_units" in src
    assert 'a["unit"] = None' in src
    # Grounded against the SAME text the names are grounded against.
    assert src.index("doc_units") > src.index("source_toks")


def test_a_contract_violation_blocks_even_when_the_gates_are_suspended():
    """`RULE_GATES_ENFORCE` suspends JUDGEMENTS, and should never have
    suspended CONTRACTS.

    "Is this rule good?" — coverage floors, the adversarial battery,
    grounding — are opinions, and this codebase's have been wrong at least as
    often as right. "Can this rule run?" is not an opinion. Three rules
    entered the library over the objection "value_pattern must capture a
    named group `num`" and then crashed two whole patents at apply time.
    """
    from patentdb.core import config
    from patentdb.repair.rules import (VALUE_PATTERN, Invalid, Rejected, Rule,
                                       validate)
    from patentdb.sources.uspto_xml import Cell, Table

    assert config.RULE_GATES_ENFORCE is False, "judgement gates stay suspended"
    t = Table(table_id="T1", n_cols=2,
              header_rows=[[Cell("Example"), Cell("IC50 (nM)")]],
              body_rows=[[Cell(str(i)), Cell(f"{i}†")] for i in range(1, 8)])

    # Cannot ever yield a number -> Invalid, which loop catches FIRST.
    with pytest.raises(Invalid):
        validate(Rule(fingerprint="f", kind=VALUE_PATTERN,
                      payload={"pattern": r"^[A-Z]\d{2}[a-z]\d$"}), t)
    # Fixed rules about what a good patch looks like are the wrong premise for
    # an extractor built to adapt: every one here eventually blocked something
    # correct. The BindingDB value delta was a blocking condition for exactly
    # one commit and is now evidence, like fidelity and the suite.
    try:
        validate(Rule(fingerprint="g", kind=VALUE_PATTERN,
                      payload={"pattern": r"^\s*(?P<num>\d+)\s*$"}), t)
    except Invalid:
        raise AssertionError("a coverage verdict must not be a contract failure")
    except Rejected:
        pass


def test_a_gap_that_raises_is_recorded_not_swallowed():
    """Three patents were skipped entirely by a corpus run and the totals
    looked healthy, because `repair_patent` raised and the runner logged a
    line. A failure that preserves the appearance of the counts is the shape
    of every defect found this week, so a crash is now a report field.
    """
    import inspect

    from patentdb.repair import loop

    assert "crashed" in {f.name for f in
                         __import__("dataclasses").fields(loop.RepairReport)}
    src = inspect.getsource(loop.repair_patent)
    # Both sites that execute a MODEL-SUPPLIED regex are covered.
    assert src.count('"stage": "validate"') == 1
    assert src.count('"stage": "apply"') == 1
    assert "CRASHED" in inspect.getsource(loop.RepairReport.summary)
