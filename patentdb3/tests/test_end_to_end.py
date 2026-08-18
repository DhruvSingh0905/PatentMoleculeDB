"""End-to-end tests: a real cached patent through the whole pipeline, and the
SHAPE + INVARIANTS of what comes out — not one regex, one helper, one field.

The other 33+ files in this directory are almost entirely microtests. This
file is the complement: run `verify.dump()` (or the extractors it calls)
against real cached XML and assert properties that only exist once every
stage has run — the manifest agreeing with the artifact it describes, a
markush row never carrying an InChIKey no matter which of the three identity
routes produced it, a crash in one route costing nothing in another, an
assay record actually joining a structure on `cid`.

FOUR PATENTS, chosen because each is required to exercise something the
others cannot (see CLAUDE.md's own "pick 2-4 patents, say why" instruction):

  CLEAN_PID = US8952177        zero assay gaps (repair is a true no-op, so a
                                healed dump is safe to build without a real
                                rule library — see `redirected` below for why
                                that still isn't free); cid_first AND the
                                description/heading route both mint markush
                                rows on it, directly measured this session:
                                146 cid_first names (29 markush, all
                                `relative_stereo`), 328 description/heading
                                names (30 markush), 0 table names.
  DUAL_ROUTE_PID = US10214537  the one patent in `test_structures_wiring.py`
                                already established as "both identity routes
                                non-empty" — reused here for a DIFFERENT
                                purpose, crash-isolating the identity route
                                that file doesn't crash (it crashes the table
                                route; this crashes cid_first) — and to supply
                                real, non-vacuous `table`-route rows for the
                                markush/InChIKey check (710 of them; measured
                                to contain 0 markush rows in this sample,
                                which the test states rather than explains).
  MARKUSH_DRAWN_PID = US11548900  the only cached patent (of 137 scanned) where
                                cid_first emits BOTH `drawn_ref` rows (20) and
                                `substituent_table:` markush rows (88) in one
                                document — needed to make the "these two never
                                carry the other's marker" check non-vacuous on
                                both sides at once. US9718825 held this role
                                until its Table 4 was recognised as a
                                substituent table, which correctly emptied its
                                drawn_ref side.
  HEAL_PID = US10030020        gaps_found == already_known == 2: every gap
                                the repair loop finds already has a rule in
                                the SHIPPED library, so healing it recovers
                                real rows (baseline 1,468 -> healed 4,404 rows,
                                directly measured this session) by APPLYING
                                already-bought rules, at $0 — no synthesis
                                call, no API key needed, nothing to stub.

All four numbers above were produced by running the current tree during this
session (`extract_by_cid`, `extract_names`, `extract_table_names`,
`extract_from_patent`, `repair_patent` called directly, then reverted/
discarded — see `redirected` below), never carried forward from a comment.

A REAL HAZARD FOUND WHILE WRITING THIS FILE, FIXED HERE RATHER THAN LEFT FOR
THE NEXT PERSON TO REDISCOVER:

`conftest.py`'s session fixture redirects `verify.DUMP_PATH` / `STRUCT_PATH` /
`MANIFEST_PATH` and the loss log. It does NOT redirect the rule library or its
journal. `verify.dump()` builds the library as `RuleLibrary()` with no path
override (`verify.py`: `lib = RuleLibrary() if heal else None`), which
defaults to the TRACKED `patentdb3/data/layout_rules.json`; `repair_patent`
calls `lib.save()` UNCONDITIONALLY at the end of every run, healed or not
(`repair/loop.py`). A zero-gap patent happens to round-trip byte-identical
(nothing in `_rules` changed), which is the only reason any EXISTING test
calling `verify.dump(heal=True)` has been safe. A patent with even one
already-known gap increments `times_applied` / `rows_yielded` counters on the
matched rule and WRITES that back to the tracked file. Confirmed directly,
not inferred: an unguarded exploratory `repair_patent()` sweep run while
surveying candidate patents for this file left `git diff
patentdb3/data/layout_rules.json` showing 26 changed lines (13 counter
pairs) before it was reverted with `git checkout --`. The same run's default
`journal=None` landed 17 real entries in `output_v3/rule_adoption_journal.jsonl`
(confirmed by exact per-patent count match, then deleted — the file did not
exist before that sweep). Every test below that calls `verify.dump(heal=True)`
therefore goes through the `redirected` fixture, which ALSO monkeypatches
`verify.RuleLibrary` to a copy of the tracked file living in `tmp_path` and
redirects `config.RULE_JOURNAL` — extending the established
artifact-redirection pattern to a path it never covered. `ANTHROPIC_API_KEY`
is forced empty on top, defense in depth: HEAL_PID needs no live call (its
gaps are already-known), and CLEAN_PID raises no gap at all, but a future
change to either patent's extraction must not turn this file into a spender.

Everything else in this file (the identity-route calls: `extract_by_cid`,
`extract_names`, `extract_table_names`) has NO path to the network at all —
confirmed by reading `sources/name_repair.py`: it is pure regex/dewrap
candidate generation, zero references to `ANTHROPIC_API_KEY` or any client.
The LLM-backed name-repair tier is a SEPARATE module (`repair/name_loop.py` +
`repair/name_synthesize.py`) that nothing in this file imports or calls.
"""
from __future__ import annotations

import csv
import json
import shutil

import pytest

from patentdb3 import verify
from patentdb3.core import config
from patentdb3.repair.rules import RuleLibrary
from patentdb3.sources import losses
from patentdb3.sources.cid_first import extract_by_cid
from patentdb3.sources.iupac_names import extract_names
from patentdb3.sources.table_names import extract_table_names
from patentdb3.sources.uspto_assays import extract_from_patent

CLEAN_PID = "US8952177"
DUAL_ROUTE_PID = "US10214537"
# US9718825 held this role until the marker learned to read a scaffold
# introduced in prose. Its 8 `drawn_ref` rows were Table 4's, and Table 4 is a
# substituent table — so they became markush rows, correctly, and this patent
# stopped having both kinds. US11548900 now does: 20 and 88, measured.
MARKUSH_DRAWN_PID = "US11548900"
HEAL_PID = "US10030020"


def _xml(pid: str) -> str:
    path = config.XML_INPUT_DIR / f"{pid}.xml"
    if not path.exists():
        pytest.skip(f"{pid}.xml not cached")
    return path.read_text(errors="ignore")


@pytest.fixture()
def redirected(tmp_path, monkeypatch):
    """Every artifact path AND the rule library/journal, pointed at scratch.

    See the module docstring's "A REAL HAZARD" section for why the second
    half of this fixture exists: `conftest.py` covers the dump artifacts and
    the loss log; it does not cover `RuleLibrary`'s default path or
    `config.RULE_JOURNAL`, and `verify.dump()` gives no parameter to redirect
    either one itself. `verify.RuleLibrary` is monkeypatched to a callable
    returning a library backed by a COPY of the tracked file in `tmp_path` —
    a copy, not the original, so `lib.save()` at the end of every
    `repair_patent` call (healed or not, gaps or not) writes to scratch. The
    copy is seeded with the real 172 rules so a test on HEAL_PID's
    already-known gaps still finds them and actually recovers rows; an empty
    library would just escalate instead, and the point of that test is a
    real, non-mocked recovery.
    """
    monkeypatch.setattr(verify, "DUMP_PATH", tmp_path / "reader_dump.tsv")
    monkeypatch.setattr(verify, "STRUCT_PATH", tmp_path / "structures.tsv")
    monkeypatch.setattr(verify, "MANIFEST_PATH", tmp_path / "latest.json")
    losses.reset(tmp_path / "loss_log.jsonl")

    lib_copy = tmp_path / "layout_rules.json"
    shutil.copy(config.PACKAGE_ROOT / "data" / "layout_rules.json", lib_copy)
    monkeypatch.setattr(verify, "RuleLibrary", lambda: RuleLibrary(path=lib_copy))
    monkeypatch.setattr(config, "RULE_JOURNAL", tmp_path / "journal.jsonl")

    # Defense in depth — see module docstring. Neither patent used with
    # heal=True in this file needs a live call, but a future change to either
    # one's extraction must not silently turn this file into a spender.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    import patentdb3.repair.synthesize as synth
    monkeypatch.setattr(synth.config, "ANTHROPIC_API_KEY", "")

    yield tmp_path
    # NOT losses.reset(losses.LOSS_LOG) — see test_structures_wiring.py's
    # `redirected` fixture for why that specific call is the one that
    # truncates the production log.
    losses.reset(tmp_path / "teardown_loss_log.jsonl")


def _read_tsv(path):
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return header, [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


# ── the manifest must describe the artifacts it names, exactly ───────────

def test_dump_manifest_and_both_artifacts_agree_with_each_other(redirected, monkeypatch):
    """The manifest is the ONLY thing a consumer is supposed to trust for "how
    many rows, what fields, from what sources" — CLAUDE.md's own "Audit wiring
    before reporting any number" names `latest.json` as the artifact to check
    before trusting a row count. If it ever disagrees with the files it
    describes, every number quoted from it is a claim about a file that no
    longer exists in that shape — the exact failure `test_structures_wiring.py`
    found once already (a 64,465-row manifest sitting next to a 328-row file).
    This test does the general version of that check, not a specific replay:
    every declared count against an actual line count, every declared field
    list against an actual header, for BOTH artifacts from ONE dump call.

    CLEAN_PID because it raises zero gaps (verified in `test_repair_gate.py`:
    "extracts completely (359/359 usable) and raises no gap at all"), so
    `heal=True` here is a genuine no-op — the baseline reader's own count,
    recomputed fresh below rather than hardcoded, must equal the dump's row
    count exactly.
    """
    pytest.importorskip("py2opsin")
    xml = _xml(CLEAN_PID)
    monkeypatch.setattr(config, "IUPAC_NAMES", True)

    expected_baseline = len(extract_from_patent(xml))

    verify.dump([CLEAN_PID], heal=True)

    dump_header, dump_rows = _read_tsv(redirected / "reader_dump.tsv")
    struct_header, struct_rows = _read_tsv(redirected / "structures.tsv")
    manifest = json.loads((redirected / "latest.json").read_text())

    # the assay dump
    assert manifest["patents"] == [CLEAN_PID]
    assert manifest["rows"] == len(dump_rows)
    assert manifest["fields"] == dump_header
    assert manifest["self_heal"] is True
    assert manifest["gaps_found"] == 0, "CLEAN_PID is chosen specifically because it has none"
    assert manifest["rows_recovered"] == 0
    assert manifest["usd_spent"] == 0.0
    assert manifest["rows"] == expected_baseline, (
        "a zero-gap patent's healed row count must equal the bare reader's own "
        "count, freshly computed — not a number carried over from a comment")

    # the structures dump — same discipline, second artifact
    assert manifest["structures_rows"] == len(struct_rows)
    assert manifest["structures_fields"] == struct_header
    if struct_rows:
        actual_sources = {r["source"] for r in struct_rows}
        assert set(manifest["structures_sources"]) == actual_sources, (
            "the manifest's source tally must name exactly the sources present "
            "in the file, no more and no fewer")
        assert sum(manifest["structures_sources"].values()) == len(struct_rows), (
            "the tally must SUM to the row count — a manifest that undercounts "
            "or overcounts its own sources is describing a different file")
        assert manifest["structures_repaired"] == sum(1 for r in struct_rows if r["repair"])


# ── a crash in one identity route must not cost the others their rows ────

def test_identity_route_crash_costs_neither_the_table_route_nor_the_assay_dump(
        redirected, monkeypatch):
    """`test_structures_wiring.py::test_a_crashing_table_route_costs_neither_
    other_artifact` crashes the TABLE route and checks cid_first + the assay
    dump survive. This is the complementary case it doesn't cover: crash the
    IDENTITY route (`cid_first`, the default under `config.IDENTITY_ROUTE`)
    and check the table route + the assay dump survive it.

    DUAL_ROUTE_PID because both routes are independently non-empty here
    (established in `test_structures_wiring.py`) — if the table route's
    survival were checked against a patent where it normally yields nothing,
    "0 rows after the crash" would be indistinguishable from "0 rows anyway."
    """
    xml = _xml(DUAL_ROUTE_PID)
    monkeypatch.setattr(config, "IUPAC_NAMES", True)
    monkeypatch.setattr(config, "IDENTITY_ROUTE", "cid_first")

    def boom(xml, pid=""):
        raise RuntimeError("cid_first route exploded")

    monkeypatch.setattr(verify, "extract_by_cid", boom)
    verify.dump([DUAL_ROUTE_PID], heal=False)

    assay_rows = (redirected / "reader_dump.tsv").read_text().count("\n")
    assert assay_rows > 1, "the assay dump must be untouched by an identity-route crash"

    _, struct_rows = _read_tsv(redirected / "structures.tsv")
    assert struct_rows, "the table route must still have produced real rows"
    assert {r["source"] for r in struct_rows} == {"table"}, (
        "cid_first contributed nothing (it crashed); every surviving row must "
        "be attributable to the table route and nothing else")

    manifest = json.loads((redirected / "latest.json").read_text())
    # Both routes log under the SAME loss-log key regardless of which one
    # crashed (`verify.py` calls `losses.record("extract_names_exception", ...)`
    # whether `_route` is `extract_by_cid` or `extract_names`) — worth stating
    # directly rather than assuming, since it means this key alone can never
    # say WHICH identity route failed, only that one did.
    assert manifest["loss_counts"].get("extract_names_exception") == 1


# ── markush rows never carry an InChIKey, across all three producers ─────

def test_markush_rows_never_carry_an_inchikey_across_all_three_identity_routes():
    """The invariant is stated independently in three modules — `cid_first.py`
    ("THE INVARIANT, and it is the same one, not a second copy"),
    `iupac_names.py` ("SAME RULE AS THE PROSE ROUTE"), `table_names.py`
    ("A MARKUSH NAME GETS NO InChIKey, same rule and same reason") — because a
    relative-stereo name (`(1R*,2S*)-...`) or a stereo-stripped one denotes a
    SET of stereoisomers, and OPSIN still returns SOME concrete InChIKey for
    it since it has to pick one; keeping that key would let a downstream join
    assert a single-structure identity the name never claimed. Three separate
    call sites means three separate chances to forget the `if is_markush: ik
    = ""` line — this holds all three to it AT ONCE, on their REAL output,
    rather than trusting that a comment next to each one stays true.

    CLEAN_PID supplies non-vacuous markush rows from cid_first (29,
    `relative_stereo`) and the description/heading route (30) — both directly
    measured this session. Its own table route is empty (0 names), so
    DUAL_ROUTE_PID's table output (710 names) is folded in too, so the table
    producer is exercised on real rows rather than skipped by construction.
    Directly measured: DUAL_ROUTE_PID's table route contributes 0 markush rows
    in this sample — stated because it is what was measured, not because a
    reason for it was verified.
    """
    pytest.importorskip("py2opsin")
    xml_clean = _xml(CLEAN_PID)
    xml_dual = _xml(DUAL_ROUTE_PID)

    combined = []
    combined += extract_by_cid(xml_clean, CLEAN_PID)
    combined += extract_names(xml_clean, CLEAN_PID)
    combined += extract_table_names(xml_clean, CLEAN_PID)
    combined += extract_table_names(xml_dual, DUAL_ROUTE_PID)

    markush_rows = [r for r in combined if r.markush]
    assert markush_rows, "the check is vacuous if no producer here ever marks markush"

    violations = [r for r in markush_rows if r.inchikey]
    assert not violations, (
        f"{len(violations)} markush row(s) carry a non-empty InChIKey — a "
        f"single-structure identity claim about what the name itself states "
        f"is a SET of stereoisomers. Example: "
        f"{(violations[0].name or violations[0].raw_cell, violations[0].inchikey) if violations else None}"
    )


# ── drawn_ref and substituent-table markush never overlap ────────────────

def test_drawn_ref_and_substituent_table_markush_are_mutually_exclusive(monkeypatch):
    """Two related but distinct "we cannot name this compound" markers, both
    set only by `cid_first.py`:

      - `drawn_ref`: the patent DREW this compound and named nothing; the row
        must assert NO structure at all (name/smiles/inchikey empty) — see
        `_resolve`'s own comment: "the row asserts no structure, it records
        where the structure is and that it is not text."
      - `markush_reason` starting `substituent_table:`: the compound is a
        scaffold-plus-substituents entry whose row picture is one
        SUBSTITUENT, not the molecule — emitted with `markush=True` and, per
        `_resolve`: "`drawn_ref` stays EMPTY rather than pointing at a
        fragment and calling it the structure."

    A row asserting both would be self-contradictory: "here is an openable
    picture of the whole molecule" and "the only picture on this row is a
    substituent" cannot both be true of the same row. MARKUSH_DRAWN_PID is the
    one cached patent (of 137 scanned) where cid_first emits BOTH kinds in the
    same document — 20 `drawn_ref` rows and 88 `substituent_table:` rows,
    directly measured — so the check has real teeth on both sides rather than
    passing vacuously because one side is empty. It asserts non-vacuity for
    that reason: when US9718825 held this role, a marker fix emptied its
    drawn_ref side and this test said so rather than quietly passing.
    """
    pytest.importorskip("py2opsin")
    assert config.GP_ENABLED is False, (
        "this test's drawn_url assertion below assumes GP is off by default; "
        "turning it on here would be the wrong way to find that out")
    xml = _xml(MARKUSH_DRAWN_PID)
    names = extract_by_cid(xml, MARKUSH_DRAWN_PID)

    drawn = [n for n in names if n.drawn_ref]
    substituent = [n for n in names if n.markush_reason.startswith("substituent_table:")]
    assert drawn, "no drawn_ref rows — the check is vacuous on this side"
    assert substituent, "no substituent_table rows — the check is vacuous on this side"

    for n in drawn:
        assert not (n.name or n.smiles or n.inchikey), (
            f"cid {n.cid!r} carries drawn_ref {n.drawn_ref!r} AND asserts a "
            f"structure — a drawn row must assert none")
        # GP is off by default (config.GP_ENABLED), so the picture is
        # findable inside the XML (`drawn_ref`) but not yet openable as a URL.
        assert n.drawn_url == "", "GP_ENABLED is off; no row should carry a URL"

    for n in substituent:
        assert not n.drawn_ref, (
            f"cid {n.cid!r} is markush_reason={n.markush_reason!r} AND carries "
            f"drawn_ref {n.drawn_ref!r} — see `_resolve`: a substituent-table "
            f"row's picture is one fragment, never the whole structure, so "
            f"drawn_ref must stay empty on it")

    # the two sets must be disjoint by cid as well as by construction
    assert not ({n.cid for n in drawn} & {n.cid for n in substituent})


# ── assay records actually join structures on cid ─────────────────────────

def test_assay_records_join_to_structures_on_cid_is_non_trivial():
    """Track A (assay values) and Track B (compound identity) "do not join" —
    ARCH.md states this as the single biggest gap in v3 relative to the
    product goal. That is a statement about the ASSEMBLY (nothing merges the
    two dumps into one row), not about the DATA: both tracks key off the same
    `cid` the patent itself assigns, so a real corpus join is already possible
    for whoever builds the assembly. This asserts that possibility is real —
    not that a fraction of `cid`s happens to collide by accident, but that a
    meaningful majority of the assay compounds this patent measures also have
    a resolved structure.

    CLEAN_PID: 190 distinct assay cids (measured), cid_first resolves 146 of
    them to a name+structure or a `drawn_ref` marker — a join, not a
    coincidence.
    """
    pytest.importorskip("py2opsin")
    xml = _xml(CLEAN_PID)
    recs = extract_from_patent(xml)
    names = extract_by_cid(xml, CLEAN_PID)

    assay_cids = {r.cid for r in recs if r.cid}
    struct_cids = {n.cid for n in names if n.cid}
    join = assay_cids & struct_cids

    assert assay_cids, "no assay cids at all — nothing to join against"
    assert join, "zero overlap between assay cids and structure cids — the join is dead"
    coverage = len(join) / len(assay_cids)
    assert coverage > 0.5, (
        f"only {coverage:.0%} of {len(assay_cids)} assay cids join a structure "
        f"({len(join)} joined) — too small to call the join non-trivial")
    # every joined row must carry SOMETHING — a name+structure or, failing
    # that, a drawn_ref marker — never a bare cid with nothing behind it
    by_cid = {n.cid: n for n in names if n.cid}
    for cid in join:
        n = by_cid[cid]
        assert n.smiles or n.inchikey or n.drawn_ref, (
            f"cid {cid!r} is counted in the join but carries neither a "
            f"structure nor a drawn_ref marker")


# ── self-heal recovers real rows from the SHIPPED library, at $0 ─────────

def test_self_heal_recovers_real_rows_from_the_shipped_library_at_zero_cost(
        redirected, monkeypatch):
    """`test_repair_gate.py::test_self_heal_flag_selects_the_pipeline` proves
    the flag is wired using a ZERO-gap patent, so it can only show
    `usd_spent == 0.0` trivially (nothing ran) and `rows == baseline` (nothing
    changed). This proves the flag matters: HEAL_PID has 2 real gaps, and
    BOTH already have a rule in the 172 shipped in `data/layout_rules.json`
    (`already_known == gaps_found == 2`, directly measured), so healing it
    recovers real, substantial rows (baseline 1,468 -> healed 4,404, measured
    this session) by APPLYING already-bought rules — no synthesis call, no
    API key, `usd_spent` must still read exactly 0.0.

    `config.IUPAC_NAMES` is forced off here so this test measures only the
    repair tier, not OPSIN time on top of it — the two are independent stages
    and this one is about the assay dump.
    """
    xml = _xml(HEAL_PID)
    monkeypatch.setattr(config, "IUPAC_NAMES", False)
    baseline_count = len(extract_from_patent(xml))

    verify.dump([HEAL_PID], heal=True)
    manifest = json.loads((redirected / "latest.json").read_text())
    dump_rows = (redirected / "reader_dump.tsv").read_text().count("\n") - 1

    assert manifest["self_heal"] is True
    assert manifest["usd_spent"] == 0.0, (
        "every gap here is already-known — a nonzero spend means a live "
        "synthesis call happened when the library should have made one "
        "unnecessary")
    assert manifest["gaps_found"] and manifest["gaps_found"] >= 1
    assert manifest["rules_adopted"] and manifest["rules_adopted"] >= 1
    assert manifest["rows"] == dump_rows, "the manifest must match the file it describes"
    assert manifest["rows"] > baseline_count, (
        "healing must have added rows the bare reader did not produce")

    # `rows_recovered == rows - baseline` HELD ONLY WHILE THE UNION DUPLICATED.
    # `repair_patent` used to return `baseline + recovered` with no dedup, so
    # the total was exactly the sum and this assertion passed. It was encoding
    # a defect: where the loop IMPROVED a record, the reader's unusable copy
    # shipped beside the repaired one. Measured on US10172859, 1,123 of 1,290
    # keys were duplicated (87.1%), 1,121 of them one usable copy beside one
    # unusable copy of the same fact — and every affected table then scored a
    # yield of exactly 0.500 against `find_gaps(min_yield=0.5)`, so the
    # repair's own output pinned the detector at its threshold.
    #
    # A repaired record now supersedes its unusable twin, so the total is no
    # longer a sum and must not be asserted as one. What must hold instead:
    # no key may carry a usable and an unusable record at once.
    assert manifest["rows_recovered"] >= 1
    assert manifest.get("superseded", 0) >= 0
    rows = list(csv.DictReader(
        (redirected / "reader_dump.tsv").open(), delimiter="\t"))
    seen: dict[tuple, set] = {}
    for r in rows:
        key = (r["patent_id"], r.get("cid"), r.get("assay_name"),
               r.get("table_id"))
        seen.setdefault(key, set()).add(bool(r.get("value_numeric")
                                             or r.get("range_lo")
                                             or r.get("range_hi")))
    both = [k for k, v in seen.items() if v == {True, False}]
    assert not both, (
        f"{len(both)} key(s) carry a valued AND an unvalued record — the "
        f"repaired row must supersede its unusable twin, e.g. {both[:3]}")

    # the $0 baseline claim this whole test is stated against
    verify.dump([HEAL_PID], heal=False)
    manifest_cold = json.loads((redirected / "latest.json").read_text())
    assert manifest_cold["self_heal"] is False
    assert manifest_cold["usd_spent"] == 0.0
    assert manifest_cold["rows"] == baseline_count, (
        "reader-only must reproduce the freshly-computed baseline exactly")


# ── two independent "off" switches must stay independently readable ──────

def test_self_heal_off_and_identity_routes_off_stay_distinguishable_in_one_manifest(
        redirected, monkeypatch):
    """`SELF_HEAL` and `IUPAC_NAMES` are two orthogonal switches
    (`core/config.py`: "Two orthogonal questions need two switches:
    collapsing them means a route cannot be turned off without turning off
    the feature"). `test_repair_gate.py` and `test_structures_wiring.py` each
    check ONE of them off in isolation. This checks both off AT ONCE, in the
    SAME manifest from the SAME dump call, because the failure this guards
    against is specifically a manifest that conflates the two — e.g. a bug
    that reads `self_heal` to decide whether to report `iupac_names`, which
    neither existing test could catch since each leaves the other flag at its
    default.
    """
    xml = _xml(CLEAN_PID)
    monkeypatch.setattr(config, "IUPAC_NAMES", False)
    baseline_count = len(extract_from_patent(xml))

    verify.dump([CLEAN_PID], heal=False)
    manifest = json.loads((redirected / "latest.json").read_text())

    assert manifest["self_heal"] is False
    assert manifest["usd_spent"] == 0.0
    assert manifest["gaps_found"] is None, "off means None, not 0 — 0 would claim it ran and found nothing"
    assert manifest["rules_adopted"] is None
    assert manifest["rows_recovered"] is None

    assert manifest["iupac_names"] is False
    assert manifest["structures_rows"] == 0
    assert manifest["structures_sources"] == {}
    assert manifest["structures_repaired"] == 0

    # neither OFF flag may cost the reader itself its rows
    assert manifest["rows"] == baseline_count and manifest["rows"] > 0
