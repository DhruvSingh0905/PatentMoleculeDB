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
