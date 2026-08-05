"""Cross-patent LABEL leak in the assay pattern library.

A library entry stores a purely STRUCTURAL row-regex (`(?P<cid>\\d+)\\s+
(?P<value0>\\d+\\.\\d+)` matches "an integer, then a decimal" in any document
ever written) beside `column_assays` — the literal column names read off the
header of the ONE patent that discovered it. The regex is safe to reuse; the
labels are not, and only the header anchor decides whether they travel.

Measured on this checkout (2026-08-04), the anchor was not strong enough to
decide it. US20240010684A1 is a MASP-1/MASP-2 complement patent — 226 mentions
of "MASP", exactly ONE of "RORgamma", in a list of unrelated therapeutics
("modulators of the RORc/RORgamma transcription factor"). US10273259's pattern
`97a7860874c14541` carries `column_assays = ["RORγ Binding IC50 μM"]`, whose
anchor degrades to the single token `rorgamma`; that one prose word opened the
gate and the structural regex then ran over the whole document, emitting
**593 rows** of `RORγ Binding IC50 μM`. None of them is an assay value:

    Column: Agilent, POROSHELL 120, 3x150 mm, SB-C18 2.7 μm
        -> compound_id 18, "RORγ Binding IC50" = 2.7 μM

an HPLC particle size filed as a nanomolar-class potency against a target the
patent does not study. 0 of 116 library entries carry a real `header_text`, so
every anchor in the library is this degraded `column_assays` fallback and 30 of
them are a single token.

What must NOT break: genuine cross-patent reuse is the library's whole economic
argument. US9745328 correctly inherits 3,048 rows from US8952177's pattern
`3654a884e244b617` — an 8-token anchor that matches because US9745328 prints
that exact header itself, over TABLE 5 and TABLE 6, immediately above the rows.
Every one of those 3,048 rows sits within 17,292 chars of its anchor; every one
of the 593 leaked rows sits at least 30,782 chars from its anchor, in a
different section of the document.
"""
from __future__ import annotations

import json

import pytest

from patentdb.core.assay_fsm import assay_pattern_library as lib


# ── fixtures ──────────────────────────────────────────────────────
#
# Hermetic, not corpus-backed: the real reproduction needs 271 MB of cached
# patent text. Each fixture below is the measured shape of one real case —
# the same regex, the same `column_assays`, the same anchor token count, and
# host prose copied from the patent that exhibited it.

# US10273259's pattern, verbatim from
# patentdb/data/assay_patterns.discoveries.json.
_ROR_PATTERN = {
    "key": "97a7860874c14541",
    "regex": r"(?P<cid>\d+)\s+(?P<value0>\d+\.\d+)",
    "column_assays": ["RORγ Binding IC50 μM"],
    "header_text": "",
    "example_match": "1 0.070",
    "status": "pending",
    "fingerprints_observed": ["US10273259"],
    "first_seen_patent": "US10273259",
    "n_observations": 2,
}

# US20240010684A1: one passing mention of RORgamma, then HPLC method prose.
# Both spans are copied from the patent's Google Patents text.
_MASP_HOST_TEXT = (
    "The present invention relates to inhibitors of MASP-1 and MASP-2, "
    "serine proteases of the lectin pathway of complement activation. "
    + "MASP-2 inhibition is described herein. " * 60
    + "Combination partners include agents inhibiting the differentiation "
    "of Th17 T cells, for example modulators of the RORc/RORgamma "
    "transcription factor; compounds antagonizing the Th17 T cell response "
    "for example anti IL-17 and anti IL-23 antibodies.\n"
    + "The MASP-2 assay buffer is described below. " * 400
    + "\nMethod 7\n"
    "Instrument type MS: Thermo Scientific LTQ-Orbitrap-XL; Equipment type "
    "HPLC: Agilent 1200SL; Column: Agilent, POROSHELL 120, 3x150 mm, "
    "SB-C18 2.7 μm; eluent A: 1 L water+0.1% trifluoroacetic acid.\n"
    "Method 8\n"
    "Instrument type MS: Waters TOF; Equipment type UPLC: Waters Acquity "
    "I-CLASS; Column: Waters, HSST3, 2.1x50 mm, C18 1.8 μm.\n"
)

# US8952177's pattern -> US9745328. Legitimate reuse: 8-token anchor, and the
# host prints that header itself directly above its own rows.
_FLAP_PATTERN = {
    "key": "3654a884e244b617",
    "regex": r"(?P<cid>\d+)\s+(?P<value0>\d+\.\d+)\s+(?P<value1>\d+\.\d+)",
    "column_assays": [
        "FLAP Binding wild type HTRF Ki (μM)",
        "Human Whole Blood LTB4 IC50 (μM)",
    ],
    "header_text": "",
    "example_match": "1 0.217 0.860",
    "status": "pending",
    "fingerprints_observed": ["US8952177"],
    "first_seen_patent": "US8952177",
    "n_observations": 2,
}

# US9745328's TABLE 5, verbatim header + first rows.
_FLAP_HOST_TEXT = (
    "Compounds of the invention were tested as FLAP inhibitors. " * 40
    + "\nTABLE 5\n"
    "FLAP binding and Human Whole Blood assay data\n"
    "Cmp  FLAP Binding wild type HTRF  Human Whole Blood LTB4 IC 50\n"
    "No.  K i (μM)  1:1 (μM)\n"
    "1  0.217  0.860\n"
    "2  0.286  0.358\n"
    "4  0.001  0.049\n"
    "5  0.038  0.077\n"
)


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Point the module at a throwaway library file."""
    def _install(*entries: dict) -> None:
        path = tmp_path / "assay_patterns.discoveries.json"
        path.write_text(json.dumps(
            {"schema_version": "1.0", "tokens": list(entries)}, indent=1))
        monkeypatch.setattr(lib, "_PATTERNS_PATH", path)
    return _install


def _labels(rows: list[dict]) -> set[str]:
    return {r["assay_name"] for r in rows}


# ── the leak ──────────────────────────────────────────────────────

def test_foreign_label_does_not_fire_on_unrelated_patent(library):
    """A label learned on patent A must not be stamped onto patent B.

    US10273259 (RORγ) -> US20240010684A1 (MASP-1/MASP-2). The only thing the
    two documents share is one prose occurrence of the word "RORgamma"; the
    rows the pattern then matches are HPLC column particle sizes.

    Failed before the cross-patent gate: `_header_anchor` degrades to the
    tokens of `column_assays` for all 116 library entries (0 carry a real
    `header_text`) and 30 of those anchors are a single token, so one word of
    prose opened the gate — while `first_seen_patent`, recorded since the
    entry was written, was never consulted at fire time.
    """
    library(_ROR_PATTERN)

    rows = lib.apply_patterns_to_text(_MASP_HOST_TEXT, "US20240010684A1")

    assert rows == [], (
        f"{len(rows)} foreign-label row(s) leaked onto US20240010684A1: "
        f"{_labels(rows)}; first={rows[0] if rows else None}"
    )


def test_leak_is_the_hplc_particle_size(library):
    """Pins the exact fabricated record, so a partial fix can't hide it.

    `SB-C18 2.7 μm` -> `compound_id 18, RORγ Binding IC50 = 2.7 μM`.

    Nothing downstream catches this one: `output_validator` corroborates
    `(cid, value)` and never inspects `assay_name`, so a fabricated label
    pinned to the host's own adjacent numbers corroborates trivially.
    """
    library(_ROR_PATTERN)

    rows = lib.apply_patterns_to_text(_MASP_HOST_TEXT, "US20240010684A1")
    fabricated = [
        r for r in rows if r["compound_id"] == "18" and r["value"] == 2.7
    ]

    assert not fabricated, (
        "an HPLC particle size is being reported as a micromolar potency: "
        f"{fabricated[0]}"
    )


# ── what the fix must not break ───────────────────────────────────

def test_legitimate_cross_patent_reuse_survives(library):
    """US8952177 -> US9745328 must keep firing.

    The host prints `FLAP Binding wild type HTRF ... Human Whole Blood LTB4
    IC 50` itself, directly above its own data rows. This is the case the
    library exists for: one paid discovery, reused free wherever that layout
    reappears. A fix that blocks all cross-patent reuse fails here.
    """
    library(_FLAP_PATTERN)

    rows = lib.apply_patterns_to_text(_FLAP_HOST_TEXT, "US9745328")

    assert rows, "legitimate cross-patent reuse was blocked"
    assert _labels(rows) == {
        "FLAP Binding wild type HTRF Ki (μM)",
        "Human Whole Blood LTB4 IC50 (μM)",
    }
    by_cid = {(r["compound_id"], r["assay_name"]): r["value"] for r in rows}
    assert by_cid[("1", "FLAP Binding wild type HTRF Ki (μM)")] == 0.217
    assert by_cid[("1", "Human Whole Blood LTB4 IC50 (μM)")] == 0.860


def test_pattern_still_fires_on_the_patent_that_discovered_it(library):
    """The originating patent's own extraction must be untouched.

    The gate keys on provenance, so the cheapest wrong fix — block every
    entry whose anchor is thin — would also silence the pattern on the patent
    whose header the labels were actually read from. 113,046 of the corpus's
    117,739 pattern-library rows are this native case.
    """
    library(_ROR_PATTERN)

    native_text = (
        "This invention relates to modulators of RORgamma.\n"
        "TABLE 3\n"
        "Example No.  RORγ Binding IC50 μM\n"
        "1  0.070\n"
        "2  0.115\n"
    )
    rows = lib.apply_patterns_to_text(native_text, "US10273259")

    assert rows, "the discovering patent lost its own rows"
    assert _labels(rows) == {"RORγ Binding IC50 μM"}


def test_fresh_patterns_are_native_to_the_patent_being_processed(library):
    """`fresh_patterns` are discovered on THIS patent during THIS run.

    They arrive from `harvest/orchestrator.py` without a `first_seen_patent`
    key. Treating an absent provenance as "foreign" would apply the strict
    gate to the run that just paid for the pattern — the pattern would be
    bought and then not used.
    """
    library()  # empty library; everything comes from fresh_patterns

    fresh = [{k: v for k, v in _ROR_PATTERN.items()
              if k not in ("first_seen_patent", "fingerprints_observed")}]
    native_text = (
        "Modulators of RORgamma are disclosed.\n"
        "Example No.  RORγ Binding IC50 μM\n"
        "1  0.070\n2  0.115\n"
    )

    rows = lib.apply_patterns_to_text(
        native_text, "US10273259", fresh_patterns=fresh)

    assert rows, "a pattern discovered on this patent did not fire on it"
