"""Some patents print the example number AFTER the name, not before it.

Both existing free passes — `_merge_example_iupacs_from_gp_description` and
`_merge_explicit_example_iupacs` — anchor on `Example N` PRECEDING the name.
Two corpus patents never write it that way:

    US20240335431A1   <name> [Example 24];          (a claim's compound list)
    US11312727        Synthesis of <name> (Compound 1)

so every compound in them is invisible to both passes. `example_index.json`
for those two patents is literally `{}` and `{3 records}`.

The danger in reading a name BACKWARDS from a marker is that there is no
header to anchor on, so a loose pattern picks up prose, or the wrong compound,
on the other twenty patents. Three shapes are pinned below because each was
measured producing a WRONG (cid → structure) pair on real text:

  * `(intermediate 5a, 5.9 g)` — US11312727 numbers intermediates in the same
    space as its examples. Its `intermediate 5a` is
    1-(allyloxy)-2-(chloro(phenyl)methyl)benzene; its `Compound 5A` is
    (18R,Z)-12-hydroxy-18-phenyl-…-11,13-dione. Two unrelated molecules whose
    ids differ only by case. Emitting intermediates scored 3 agreeing and 79
    disagreeing InChIKeys against BindingDB on that patent alone.
  * `(example 3, step (ii))` — US9718825. A cross-reference to how something
    was made, not a name for it; the name in front belongs to the intermediate
    being described.
  * `to give <name>` — the name must not absorb the verb in front of it.

The rule that falls out: read backwards, take name-shaped tokens only, stop at
the first thing that is English.
"""
from __future__ import annotations

import pytest

from patentdb.routes import process_patent as PP
from patentdb.core import patent_text as PT


NAME = "2-[4-(3-chlorophenyl)piperazin-1-yl]-N-(pyridin-3-ylmethyl)acetamide"
NAME2 = "1-(4-methoxyphenyl)-4-(pyridin-4-ylmethyl)piperazine-2,6-dione"


@pytest.fixture
def sources(monkeypatch):
    """Serve one text as the patent's grant XML; GP description is empty."""
    holder = {"xml": ""}
    monkeypatch.setattr(PP, "load_uspto_description", lambda pid: holder["xml"])
    monkeypatch.setattr(PT, "load_gp_description", lambda pid: "")
    return holder


def _run(holder, text, index=None):
    holder["xml"] = text
    idx = {} if index is None else index
    n = PP._merge_trailing_id_example_iupacs("USTEST", idx)
    return n, idx


# ── the two shapes this exists for ─────────────────────────────────


def test_bracketed_example_after_the_name_is_lifted(sources):
    """US20240335431A1's claim list: `<name> [Example 1];`"""
    text = (
        "wherein the compound is selected from the group consisting of:\n\n"
        f"{NAME} [Example 1];\n\n"
        f"{NAME2} [Example 2];\n\n"
    )
    n, idx = _run(sources, text)
    assert n == 2
    assert idx["1"]["iupac_name"] == NAME
    assert idx["2"]["iupac_name"] == NAME2
    assert idx["1"]["canonical_smiles"] and idx["1"]["inchikey"]


def test_parenthesised_compound_after_the_name_is_lifted(sources):
    """US11312727's header: `Example 1: Synthesis of <name> (Compound 1)`"""
    text = f"Example 1: Synthesis of {NAME} (Compound 1)\n\nSynthesis of ..."
    n, idx = _run(sources, text)
    assert n == 1
    assert idx["1"]["iupac_name"] == NAME


def test_a_counter_ion_tail_does_not_cost_the_parent_structure(sources):
    """`<name> 1·formic acid [Example 11]` — seven of US20240335431A1's 55.

    OPSIN cannot parse the middle-dot stoichiometry, and the salt is not the
    compound the patent is claiming. The existing passes already strip
    `as the TFA salt` for the same reason.
    """
    text = f"{NAME} 1·formic acid [Example 11];\n\n"
    n, idx = _run(sources, text)
    assert n == 1
    assert idx["11"]["canonical_smiles"]


# ── what it must NOT pick up ───────────────────────────────────────


def test_an_intermediate_is_not_an_example(sources):
    """Verbatim from US11312727's grant XML."""
    text = (
        "The mixture was concentrated to dryness and co-evaporated with "
        "toluene to give 1-(allyloxy)-2-(chloro(phenyl)methyl)benzene "
        "(intermediate 5a, 5.9 g), which was used as such in the next step."
    )
    n, idx = _run(sources, text)
    assert n == 0
    assert idx == {}


def test_a_lowercase_suffix_id_is_not_an_example_id(sources):
    """`Example 5a` and `Example 5A` are the same id under every downstream
    normaliser, and on US11312727 they are different molecules. The leading
    passes' `_EXAMPLE_HEADER_PAT` accepts an UPPERCASE suffix only; this one
    matches that grammar rather than widening it."""
    text = f"Synthesis of {NAME} (Compound 5a)\n\n"
    n, idx = _run(sources, text)
    assert n == 0


def test_a_step_cross_reference_is_not_a_naming(sources):
    """Verbatim from US9718825. The name in front is an intermediate of
    example 3, not example 3's own compound."""
    text = (
        "4-[6-Chloro-1-(tetrahydro-pyran-2-yl)-1H-pyrazolo[3,4-d]pyrimidin-"
        "4-yloxy]-cyclohexanol (2.0 g) (example 3, step (ii)) was added"
    )
    n, idx = _run(sources, text)
    assert n == 0


def test_the_verb_in_front_of_the_name_is_not_swallowed(sources):
    """`to give <name> (Compound 7)` stores the name, not `give <name>`."""
    text = (
        "The pure fractions were collected and concentrated under reduced "
        f"pressure to give {NAME} (Compound 7).\n\n"
    )
    n, idx = _run(sources, text)
    assert n == 1
    assert idx["7"]["iupac_name"] == NAME


def test_a_marker_with_no_name_in_front_yields_nothing(sources):
    text = "The title compound was prepared as described in (Example 12).\n\n"
    n, idx = _run(sources, text)
    assert n == 0


# ── interaction with the passes that already ran ───────────────────


def test_a_cid_the_leading_passes_already_resolved_is_left_alone(sources):
    """A trailing marker is weaker evidence than a header — it can be a
    reagent reference. This pass ADDS; it never overrides."""
    text = f"{NAME} [Example 1];\n\n"
    existing = {"1": {
        "compound_id": "Example 1", "iupac_name": NAME2,
        "canonical_smiles": "CCO", "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "source": "examples", "extraction_method": "explicit_example_header",
    }}
    n, idx = _run(sources, text, existing)
    assert n == 0
    assert idx["1"]["extraction_method"] == "explicit_example_header"


def test_an_ms_stub_without_a_structure_is_filled_in(sources):
    """A cid present but structureless is not "already found"."""
    text = f"{NAME} [Example 1];\n\n"
    existing = {"1": {"compound_id": "Example 1", "iupac_name": "",
                      "canonical_smiles": "", "inchikey": "",
                      "source": "ms_stub", "ms_mh_plus_found": 331.1}}
    n, idx = _run(sources, text, existing)
    assert n == 1
    assert idx["1"]["canonical_smiles"]
    assert idx["1"]["ms_mh_plus_found"] == 331.1


# ── cost shape ─────────────────────────────────────────────────────


def test_opsin_is_warmed_once_for_the_whole_patent(sources, monkeypatch):
    """One batched JVM launch for the patent, not one per name. OPSIN is a
    ~270 ms subprocess; a per-name loop over US20250163061A1's 64 names is
    17 s of JVM startup for a pass that is meant to be free."""
    from patentdb.core import iupac_to_smiles as ITS
    seen = []
    real = ITS.prefetch_cascade
    monkeypatch.setattr(
        ITS, "prefetch_cascade",
        lambda names, **kw: (seen.append(list(names)), real(names, **kw))[1],
    )
    text = f"{NAME} [Example 1];\n\n{NAME2} [Example 2];\n\n"
    _run(sources, text)
    assert len(seen) == 1, "prefetch_cascade must be called once, with a LIST"
    assert sorted(seen[0]) == sorted([NAME, NAME2])
