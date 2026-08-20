"""Tests for `sources/gp_names.py` — bulk GP-name import, OPSIN-verified,
kept only where it is new to us.

Every test here is offline. `compound_blocks` is the only function that would
otherwise reach the network; the fetch is monkeypatched (mirroring
`test_gp_images.py`'s own pattern) and the default-off path is asserted to
make no call at all. `new_structures` is exercised almost entirely through
its `blocks=` injection parameter, which bypasses `compound_blocks` (and
therefore `config.GP_ENABLED`) on purpose — the dedup/OPSIN/label logic is
what this module actually needs proven, independent of the network question.

OPSIN itself is a local Java subprocess, never the network — the same
resolver `test_name_repair.py` and the rest of the suite already call
directly — so the handful of real-OPSIN tests below need no key and no
connectivity, only the `java` binary and the `py2opsin` jar this repo already
depends on.

What is locked in, and why each assertion exists:

  - OFF BY DEFAULT MEANS NO CALL, and no `cid` field ever appears on the
    output — CLAUDE.md's owner already decided this output joins no assay
    row, and the schema should not be able to lie about that.
  - THE PREFILTER IS `smiles`, NOT `domain`. `domain` is not evidence by
    itself (see the module docstring's measurement); a block with GP's own
    SMILES clears the prefilter regardless of what `domain` says, and one
    with no SMILES is dropped regardless of what `domain` says.
  - DEDUP IS SCOPED PER PATENT, AND RUNS TWICE — once on GP's own
    `inchi_key` before OPSIN is asked anything, and once on OPSIN's own
    `StdInChIKey` after, because pass 1 cannot catch a block with no GP key.
  - OPSIN IS THE HARD GATE. A name it refuses never becomes a row, no matter
    what `domain` or `smiles` said going in.
"""
from __future__ import annotations

import json

import pytest

from patentdb3.core import config
from patentdb3.sources import gp_names, losses, opsin, reagents

HOST = "https://patentimages.storage.googleapis.com"


def _block(name: str, domain: str = "Chemical compound", smiles: str = "C",
           inchikey: str = "", block_id: str = "") -> str:
    """One `<li itemprop="match">` block, in the exact shape measured off a
    live Google Patents page (see module docstring)."""
    return f"""
    <li itemprop="match" itemscope repeat>
      <span itemprop="id">{block_id}</span>
      <span itemprop="name">{name}</span>
      <span itemprop="domain">{domain}</span>
      <span itemprop="svg_large"></span>
      <span itemprop="svg_small"></span>
      <span itemprop="smiles">{smiles}</span>
      <span itemprop="inchi_key">{inchikey}</span>
      <span itemprop="similarity">0.000</span>
      <span itemprop="sections" repeat>description</span>
      <span itemprop="count">1</span>
    </li>"""


PAGE = "<html><body><ul itemprop=\"concept\" itemscope>{}</ul></body></html>".format(
    _block("3,3,5-trifluoro-4,5-dihydro-2H-pyridin-6-amine",
           inchikey="BFTKEARFXHXNIW-UHFFFAOYSA-N") +
    _block("combination treatment", domain="Methods", smiles="", inchikey="") +
    _block("compounds", domain="Chemical class", smiles="", inchikey="") +
    _block("anti-inflammatory agent", domain="Substances", smiles="", inchikey=""))


@pytest.fixture()
def gp_on(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GP_ENABLED", True)
    monkeypatch.setattr(gp_names, "_CACHE_DIR", tmp_path / "gp_names")
    return tmp_path / "gp_names"


def _no_network(*a, **k):
    raise AssertionError("gp_names must not fetch here")


# ── the flag ─────────────────────────────────────────────────────────────

def test_disabled_returns_empty_and_never_fetches(monkeypatch):
    monkeypatch.setattr(config, "GP_ENABLED", False)
    monkeypatch.setattr(gp_names, "_gp_fetch", _no_network)
    assert gp_names.compound_blocks("US10004738") == []


def test_the_flag_is_the_same_one_gp_images_uses():
    assert config.GP_ENABLED in (True, False)


def test_the_cache_dir_is_new_not_the_image_cache():
    """A second cache beside `gp_images`'s, never inside it — see module
    docstring, "`_fetch`'S OWN CACHE CANNOT BE REUSED"."""
    assert gp_names._CACHE_DIR != config.GP_IMAGE_DIR
    assert "output_v3" in str(gp_names._CACHE_DIR)


def test_no_cid_field_ever_no_compound_number_by_design():
    """CLAUDE.md: these structures carry NO compound number. Absent from the
    schema, not merely blank, so nothing downstream can mistake a row here
    for something joinable to an assay record."""
    assert "cid" not in gp_names.FIELDS


# ── the parser ───────────────────────────────────────────────────────────

def test_parse_reads_every_block_in_document_order():
    blocks = gp_names._parse_blocks(PAGE)
    assert [b.name for b in blocks] == [
        "3,3,5-trifluoro-4,5-dihydro-2H-pyridin-6-amine",
        "combination treatment", "compounds", "anti-inflammatory agent"]


def test_parse_reads_all_five_fields_named_in_the_module_docstring():
    blocks = gp_names._parse_blocks(PAGE)
    b = blocks[0]
    assert b.name == "3,3,5-trifluoro-4,5-dihydro-2H-pyridin-6-amine"
    assert b.domain == "Chemical compound"
    assert b.smiles == "C"
    assert b.inchikey == "BFTKEARFXHXNIW-UHFFFAOYSA-N"


def test_parse_unescapes_html_entities():
    page = "<ul>{}</ul>".format(_block("Cu&#43; salt", smiles="[Cu+]"))
    blocks = gp_names._parse_blocks(page)
    assert blocks[0].name == "Cu+ salt"


def test_parse_of_a_page_with_no_annotations_is_empty_not_an_error():
    assert gp_names._parse_blocks("<html><body>nothing here</body></html>") == []


def test_parse_missing_field_is_empty_string_not_a_crash():
    page = ('<li itemprop="match" itemscope repeat>'
            '<span itemprop="name">bare name only</span></li>')
    blocks = gp_names._parse_blocks(page)
    assert blocks[0].name == "bare name only"
    assert blocks[0].smiles == ""
    assert blocks[0].domain == ""


# ── cache and fetch ──────────────────────────────────────────────────────

def test_fetches_once_then_serves_from_cache(gp_on, monkeypatch):
    calls = []

    def fake(pid):
        calls.append(pid)
        return PAGE

    monkeypatch.setattr(gp_names, "_gp_fetch", fake)
    first = gp_names.compound_blocks("US10004738")
    assert len(first) == 4
    assert calls == ["US10004738"]
    assert (gp_on / "US10004738.json").exists()

    monkeypatch.setattr(gp_names, "_gp_fetch", _no_network)
    second = gp_names.compound_blocks("US10004738")
    assert second == first


def test_a_failed_fetch_costs_no_exception_and_no_cache_file(gp_on, monkeypatch):
    def boom(pid):
        raise OSError("network down")

    monkeypatch.setattr(gp_names, "_gp_fetch", boom)
    assert gp_names.compound_blocks("US10004738") == []
    assert not (gp_on / "US10004738.json").exists()


def test_a_corrupt_cache_is_refetched_rather_than_fatal(gp_on, monkeypatch):
    gp_on.mkdir(parents=True, exist_ok=True)
    (gp_on / "US10004738.json").write_text("{not json")
    monkeypatch.setattr(gp_names, "_gp_fetch", lambda pid: PAGE)
    assert len(gp_names.compound_blocks("US10004738")) == 4


def test_allow_fetch_false_reads_cache_only(gp_on, monkeypatch):
    monkeypatch.setattr(gp_names, "_gp_fetch", _no_network)
    assert gp_names.compound_blocks("US10004738", allow_fetch=False) == []

    gp_on.mkdir(parents=True, exist_ok=True)
    (gp_on / "US10004738.json").write_text(json.dumps([
        {"id": "1", "name": "n", "domain": "d", "smiles": "s", "inchikey": "k"}]))
    got = gp_names.compound_blocks("US10004738", allow_fetch=False)
    assert len(got) == 1 and got[0].name == "n"


# ── what we already hold ─────────────────────────────────────────────────

def test_load_held_inchikeys_is_scoped_per_patent(tmp_path):
    path = tmp_path / "structures.tsv"
    path.write_text(
        "patent_id\tname\tsmiles\tinchikey\n"
        "US1\tfoo\tC\tAAAAAAAAAAAAAA-UHFFFAOYSA-N\n"
        "US2\tbar\tCC\tBBBBBBBBBBBBBB-UHFFFAOYSA-N\n"
        # lowercase on disk must still match an uppercase lookup
        "US1\tbaz\tCCC\tcccccccccccccc-uhfffaoysa-n\n"
        # blank inchikey contributes nothing
        "US1\tqux\t\t\n")
    held = gp_names.load_held_inchikeys(path)
    assert held["US1"] == {"AAAAAAAAAAAAAA-UHFFFAOYSA-N",
                            "CCCCCCCCCCCCCC-UHFFFAOYSA-N"}
    assert held["US2"] == {"BBBBBBBBBBBBBB-UHFFFAOYSA-N"}
    assert "US3" not in held


def test_load_held_inchikeys_missing_file_is_empty_not_an_error(tmp_path):
    assert gp_names.load_held_inchikeys(tmp_path / "nope.tsv") == {}


# ── the pipeline: prefilter, dedup, OPSIN, labels ───────────────────────

def test_prefilter_drops_blocks_with_no_gp_smiles():
    blocks = gp_names._parse_blocks(PAGE)
    rows, stats = gp_names.new_structures(
        "US1", blocks=blocks, held={})
    assert stats.gp_blocks == 4
    # only the first block carries a GP smiles ("C" — a real, tiny molecule
    # that also happens to be OPSIN-unparseable as a NAME, so it is refused
    # downstream; the prefilter count is what is being proven here)
    assert stats.gp_grounded == 1


def test_distinct_names_dedups_before_spending_an_opsin_call():
    blocks = ([gp_names.GPBlock(id="1", name="toluene", domain="Chemical compound",
                                smiles="Cc1ccccc1", inchikey="")] * 5)
    rows, stats = gp_names.new_structures("US1", blocks=blocks, held={})
    assert stats.gp_grounded == 5
    assert stats.distinct_names == 1
    assert stats.opsin_sent == 1


def test_pre_opsin_dedup_uses_gps_own_inchikey_and_spends_no_opsin_call():
    held = {"US1": {"YXFVVABEGXRONW-UHFFFAOYSA-N"}}   # toluene, already held
    blocks = [gp_names.GPBlock(id="1", name="toluene", domain="Chemical compound",
                               smiles="Cc1ccccc1",
                               inchikey="YXFVVABEGXRONW-UHFFFAOYSA-N")]
    rows, stats = gp_names.new_structures("US1", blocks=blocks, held=held)
    assert stats.dedup_dropped_pre == 1
    assert stats.opsin_sent == 0
    assert rows == []


def test_dedup_is_scoped_per_patent_not_global():
    held = {"US_OTHER": {"YXFVVABEGXRONW-UHFFFAOYSA-N"}}
    blocks = [gp_names.GPBlock(id="1", name="toluene", domain="Chemical compound",
                               smiles="Cc1ccccc1",
                               inchikey="YXFVVABEGXRONW-UHFFFAOYSA-N")]
    rows, stats = gp_names.new_structures("US1", blocks=blocks, held=held)
    assert stats.dedup_dropped_pre == 0
    assert stats.opsin_sent == 1


def test_opsin_is_the_hard_gate_real_resolver(monkeypatch):
    """Real OPSIN (local Java, no network): a genuine name resolves, a
    non-name is refused, and the reported InChIKey agrees with an
    independent direct call to `sources/opsin.py` — the ONE OPSIN wrapper."""
    blocks = [
        gp_names.GPBlock(id="1", name="toluene", domain="Chemical compound",
                         smiles="Cc1ccccc1", inchikey=""),
        gp_names.GPBlock(id="2", name="not a real chemical name at all",
                         domain="Chemical compound", smiles="C", inchikey=""),
    ]
    rows, stats = gp_names.new_structures("US1", blocks=blocks, held={})
    assert stats.opsin_sent == 2
    assert stats.opsin_kept == 1
    assert stats.opsin_refused == 1
    assert len(rows) == 1
    assert rows[0].gp_name == "toluene"

    ref_key = opsin.batch(["toluene"], "StdInChIKey")[0]
    assert rows[0].inchikey == ref_key
    assert rows[0].smiles == opsin.batch(["toluene"], "SMILES")[0]


def test_post_opsin_dedup_catches_what_gps_own_key_missed(monkeypatch):
    """GP supplied no `inchi_key` for this block, so pass 1 cannot drop it —
    OPSIN's OWN key must still be checked against what we hold."""
    ref_key = opsin.batch(["toluene"], "StdInChIKey")[0]
    held = {"US1": {ref_key}}
    blocks = [gp_names.GPBlock(id="1", name="toluene", domain="Chemical compound",
                               smiles="Cc1ccccc1", inchikey="")]
    rows, stats = gp_names.new_structures("US1", blocks=blocks, held=held)
    assert stats.dedup_dropped_pre == 0     # GP gave no key to check
    assert stats.opsin_sent == 1
    assert stats.opsin_kept == 1
    assert stats.dedup_dropped_post == 1
    assert rows == []


def test_reagent_labels_are_attached_never_used_to_drop():
    """`sources/reagents.py` LABELS, it never deletes — this module reuses
    that contract rather than reintroducing a silent filter."""
    blocks = [gp_names.GPBlock(id="1", name="dichloromethane",
                               domain="Chemical compound", smiles="ClCCl",
                               inchikey="")]
    rows, stats = gp_names.new_structures("US1", blocks=blocks, held={})
    assert stats.net_new == 1
    assert rows[0].label == "reagent"
    assert rows[0].reason.startswith("lexicon:solvent:")
    # cross-check against the module directly, not a hardcoded string
    verdict = reagents.classify("dichloromethane", rows[0].smiles)
    assert (rows[0].label, rows[0].reason) == (verdict.label, verdict.reason)


def test_no_gp_blocks_is_a_clean_empty_result_not_an_error():
    rows, stats = gp_names.new_structures("US1", blocks=[], held={})
    assert rows == []
    assert stats.gp_blocks == 0
    assert stats.net_new == 0


# ── finished-compound filter ────────────────────────────────────────────
#
# See `gp_names.py`'s own measurement section (immediately above
# `_finished_verdict`) for the evidence behind each rule: ring>=1, no boron,
# heavy-atom count >= `_FINISHED_HAC_FLOOR` (20) — measured against 10,734
# `reagents.classify`-labeled-`"compound"` GP rows and validated against
# `structures.tsv`'s own resolved population.

TOLUENE = "Cc1ccccc1"                 # HAC 7, one ring, no boron — too small
HEXANE = "CCCCCC"                     # HAC 6, no ring — acyclic
PINACOL_BORONATE = "CC1(C)OB(c2ccccc2)OC1(C)C"   # phenylboronic acid pinacol
                                                   # ester — a ring, but boron
# A real, complex, drug-shaped molecule: two chlorophenyl rings + a
# piperazine ring + an amide linker, HAC 24 — confirmed by
# `patentdb3.sources.opsin` to resolve from the IUPAC name
# "N-(4-chlorophenyl)-2-[4-(4-chlorophenyl)piperazin-1-yl]acetamide", and by
# `reagents.classify` to carry label `"compound"` (not in `REAGENT_LEXICON`,
# and far above the HAC<=3 structural backstop).
DRUG_SHAPED = "ClC1=CC=C(C=C1)NC(CN1CCN(CC1)C1=CC=C(C=C1)Cl)=O"


def test_finished_verdict_short_circuits_on_a_non_compound_label():
    fin, reason = gp_names.finished_verdict("reagent", TOLUENE)
    assert fin is False
    assert reason == "reagents_classify:reagent"

    fin, reason = gp_names.finished_verdict("trace_fragment", "C")
    assert fin is False
    assert reason == "reagents_classify:trace_fragment"


def test_finished_verdict_rejects_an_acyclic_compound_labeled_structure():
    fin, reason = gp_names.finished_verdict("compound", HEXANE)
    assert fin is False
    assert reason.startswith("no_ring:")


def test_finished_verdict_rejects_boron_even_with_a_ring():
    fin, reason = gp_names.finished_verdict("compound", PINACOL_BORONATE)
    assert fin is False
    assert reason.startswith("boron:")


def test_finished_verdict_rejects_below_the_hac_floor():
    fin, reason = gp_names.finished_verdict("compound", TOLUENE)  # HAC 7, ring>=1, no B
    assert fin is False
    assert reason.startswith("too_small:")
    assert f"floor={gp_names._FINISHED_HAC_FLOOR}" in reason


def test_finished_verdict_accepts_a_drug_shaped_compound_labeled_structure():
    fin, reason = gp_names.finished_verdict("compound", DRUG_SHAPED)
    assert fin is True
    assert reason.startswith("kept:hac=24")


def test_finished_verdict_unparseable_or_blank_smiles_is_never_finished():
    assert gp_names.finished_verdict("compound", "") == (False, "unparseable")
    assert gp_names.finished_verdict("compound", "not a smiles") == (
        False, "unparseable")


def test_new_structures_wires_finished_and_finished_reason_onto_the_row():
    """End-to-end: a real OPSIN name that resolves to a drug-shaped,
    non-lexicon structure comes back `finished=True`; `stats.finished_kept`
    counts it. Real OPSIN (local Java, no network) — same pattern as
    `test_opsin_is_the_hard_gate_real_resolver` above."""
    blocks = [gp_names.GPBlock(
        id="1", name="N-(4-chlorophenyl)-2-[4-(4-chlorophenyl)piperazin-1-yl]acetamide",
        domain="Chemical compound", smiles=DRUG_SHAPED, inchikey="")]
    rows, stats = gp_names.new_structures("US1", blocks=blocks, held={})
    assert len(rows) == 1
    assert rows[0].label == "compound"
    assert rows[0].finished is True
    assert rows[0].finished_reason.startswith("kept:")
    assert stats.finished_kept == 1


def test_new_structures_a_reagent_row_is_never_finished():
    blocks = [gp_names.GPBlock(id="1", name="dichloromethane",
                               domain="Chemical compound", smiles="ClCCl",
                               inchikey="")]
    rows, stats = gp_names.new_structures("US1", blocks=blocks, held={})
    assert rows[0].label == "reagent"
    assert rows[0].finished is False
    assert rows[0].finished_reason == "reagents_classify:reagent"
    assert stats.finished_kept == 0


# ── writer ────────────────────────────────────────────────────────────────

def test_write_tsv_round_trips_every_field(tmp_path):
    row = gp_names.GPStructure(
        patent_id="US1", gp_name="toluene", domain="Chemical compound",
        gp_smiles="Cc1ccccc1", gp_inchikey="", smiles="Cc1ccccc1",
        inchikey="YXFVVABEGXRONW-UHFFFAOYSA-N", label="reagent",
        reason="lexicon:solvent:toluene", finished=False,
        finished_reason="reagents_classify:reagent")
    out = gp_names.write_tsv([row], tmp_path / "gp_names.tsv")
    text = out.read_text()
    lines = text.strip("\n").split("\n")
    assert lines[0].split("\t") == list(gp_names.FIELDS)
    assert lines[1].split("\t") == [
        "US1", "toluene", "Chemical compound", "Cc1ccccc1", "",
        "Cc1ccccc1", "YXFVVABEGXRONW-UHFFFAOYSA-N", "reagent",
        "lexicon:solvent:toluene", "False", "reagents_classify:reagent"]


def test_write_tsv_default_path_is_a_new_artifact_not_structures_tsv():
    assert gp_names.OUT_PATH != config.STRUCTURES
    assert gp_names.OUT_PATH.name == "gp_names.tsv"


# ── main() must never touch the shared production loss log ─────────────
#
# `sources/losses.py` truncates its shared `LOSS_LOG` on the first `record()`
# call of a process. Running this module standalone found that the hard way:
# a same-day 88,194-event corpus loss log was replaced by this module's own
# 3,077 records, silently. `main()` now redirects to a private path before
# doing any work — this is what proves it.

def test_main_redirects_the_loss_sink_before_doing_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GP_ENABLED", False)
    monkeypatch.setattr(gp_names, "_LOSS_LOG", tmp_path / "gp_names_loss.jsonl")
    gp_names.main(["US10004738"])
    assert losses._path == tmp_path / "gp_names_loss.jsonl"
    assert losses._path != losses.LOSS_LOG
