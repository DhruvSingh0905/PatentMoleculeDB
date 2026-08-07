"""A shipped record's `iupac_name` and `canonical_smiles` must describe the
SAME molecule.

Measured defect (corpus run 2026-08-06, the 22 shipped
`output_v2/text_extraction/*/example_index.json` written by `_write_outputs`;
read, not recomputed):

    records carrying `inchikey_aliases`              1,151
      their `iupac_name` OPSINs to a DIFFERENT key    1,087
      their `iupac_name` OPSINs to their primary          0
      their `iupac_name` OPSIN cannot parse              64

In all 1,087 the name's key is *in that record's own* `inchikey_aliases`, the
name's molecule has a different molecular FORMULA from the structure shipped
beside it (so this is not a stereo layer OPSIN dropped — they are different
compounds), and the name's molecule is already present on another cid of the
same patent, 1,068 of them self-consistently.

The producer is `_bridge_gp_to_harvest_cids`, the `mode == "overwrite"` branch:
it replaces the target's `canonical_smiles` + `inchikey` with a GP-embedded
structure and pushes the displaced key into `inchikey_aliases`, and it never
touches `iupac_name`. US10899738 cid 71 is the vivid instance — a clean
`1-benzyl-N-phenethylpiperidine-4-carboxamide` matching its own structure was
overwritten with `S=S=S=S=S=S=S=S=S=S=S=S.[C-10]....`

The three cases below are real records, verbatim: their names, the structure
the name resolves to, and the structure the bridge put there instead.

What this file must NOT be satisfied by: disabling the bridge. Stage A's
InChIKey merge, Stage B's positional rename and an overwrite the target's own
name does not contradict are each pinned below.
"""
import pytest

from patentdb.routes.process_patent import _bridge_gp_to_harvest_cids

PID = "US_TEST_NAME_AGREEMENT"

# ── US10214537 cid 169 (`opsin_direct`) ───────────────────────────────
NAME_169 = (
    "7-(3-((methyl(tetrahydro-2H-pyran-4-yl)amino)methyl)phenyl)-5-"
    "(1-(tetrahydro-2H-pyran-4-yl)-1H-pyrazol-5-yl)pyrrolo[2,1-f]"
    "[1,2,4]triazin-4-amine"
)
SMI_169_NAMED = "CN(Cc1cccc(-c2cc(-c3ccnn3C3CCOCC3)c3c(N)ncnn23)c1)C1CCOCC1"
SMI_169_IMPOSED = (
    "COc1ccc(-c2cc(-c3ccnn3C(C)C(F)(F)F)c3c(N)ncnn23)cc1N1CCN(C(C)=O)"
    "C(C)(C)C1=O"
)

# ── US9694016 cid 15 (`opsin_direct`) ─────────────────────────────────
NAME_15 = (
    "2-(2-cyanopropan-2-yl)-N-(3-(2,6-dimorpholinopyrimidin-4-yl)-4-"
    "methylphenyl)isonicotinamide"
)
SMI_15_NAMED = (
    "Cc1ccc(NC(=O)c2ccnc(C(C)(C)C#N)c2)cc1-c1cc(N2CCOCC2)nc(N2CCOCC2)n1"
)
SMI_15_IMPOSED = "Cc1ncc(N)cc1-c1cc(N2CCOCC2)nc(Cl)c1F"

# ── US9302989 cid 646 (`explicit_example_header`) ─────────────────────
NAME_646 = (
    "N-[4-(1-isobutyrylpiperidin-4-yl)phenyl]-1,3-dihydro-2H-pyrrolo"
    "[3,4-c]pyridine-2-carboxamide"
)
SMI_646_NAMED = "CC(C)C(=O)N1CCC(c2ccc(NC(=O)N3Cc4ccncc4C3)cc2)CC1"
SMI_646_IMPOSED = "O=C(Nc1ccc(C(=O)N2CCCC3(CCNCC3)C2)cc1)N1Cc2ccccc2C1"

CASES = [
    ("169", NAME_169, SMI_169_NAMED, SMI_169_IMPOSED),
    ("15", NAME_15, SMI_15_NAMED, SMI_15_IMPOSED),
    ("646", NAME_646, SMI_646_NAMED, SMI_646_IMPOSED),
]


def _ik(smiles):
    from patentdb.core.smiles_utils import get_inchikey
    return get_inchikey(smiles)


def _opsin_ik(name):
    """The key the record's own name resolves to — the same free local parse
    `_name_backs_current_structure` uses, so the test grades the invariant and
    not a second opinion about how to read a name."""
    from patentdb.core.iupac_to_smiles import _try_opsin
    smi, _ = _try_opsin(name)
    return _ik(smi) if smi else None


def _rows(n):
    return [{"assay_name": "IC50", "value_numeric": 10.0 + i, "unit": "nM",
             "qualifier": "", "n_runs": None, "source": "t"} for i in range(n)]


def _overwrite_case():
    """The shipped shape: for each cid N, a self-consistent patent record and
    an orphan `GP{N}` holding a different molecule.

    Three bare-digit assay cids each matching a GP index is what sets
    `digit_alignment_trusted`, hence `overwrite_trusted` — without it there is
    no overwrite to refuse and the fixture would prove nothing.

    The GP records carry no `iupac_name` on purpose: Strategy 0 reads a
    structure out of the GP HTML, and the veto is about the TARGET's name, so
    giving the GP one would test a case the bridge does not have.
    """
    example_index = {}
    assay_tables = {}
    for cid, name, named, imposed in CASES:
        example_index[cid] = {
            "compound_id": f"Cpd. No. {cid}", "iupac_name": name,
            "canonical_smiles": named, "inchikey": _ik(named),
            "extraction_method": "opsin_direct",
        }
        example_index[f"GP{cid}"] = {
            "compound_id": f"GP{cid}", "canonical_smiles": imposed,
            "inchikey": _ik(imposed), "extraction_method": "gp_embedded_meta",
        }
        assay_tables[cid] = _rows(4)
    return example_index, assay_tables


def _disagreements(index):
    """Every output record whose own name resolves to a structure other than
    the one it holds. A name OPSIN cannot parse is not a disagreement — it is
    silence, and silence is not evidence."""
    out = []
    for cid, rec in index.items():
        cur = (rec.get("inchikey") or "").strip()
        nm = (rec.get("iupac_name") or "").strip()
        if not cur or not nm:
            continue
        k = _opsin_ik(nm)
        if k and k != cur:
            out.append((cid, nm[:48], cur, k))
    return out


def _primary_iks(index):
    return {(r.get("inchikey") or "").strip()
            for r in index.values() if (r.get("inchikey") or "").strip()}


# ── the invariant ────────────────────────────────────────────────────

def test_no_output_record_disagrees_with_its_own_name():
    """FAILS before the fix with all three cids disagreeing — the shipped
    corpus's 1,087, reproduced on three of its own records."""
    example_index, assay_tables = _overwrite_case()

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    bad = _disagreements(out)
    assert not bad, (
        f"{len(bad)} record(s) ship a structure their own name contradicts: "
        + "; ".join(f"{c} name→{k} but holds {h}" for c, _n, h, k in bad)
    )


@pytest.mark.parametrize("cid,name,named,imposed", CASES)
def test_overwrite_refused_when_the_targets_name_backs_its_structure(
    cid, name, named, imposed,
):
    """Record-by-record: the cid keeps the molecule its name describes."""
    example_index, assay_tables = _overwrite_case()

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    assert out[cid]["inchikey"] == _ik(named)
    assert out[cid]["canonical_smiles"] == named
    assert _ik(imposed) not in _primary_iks({cid: out[cid]}), (
        "the GP structure was written onto a cid whose own name names a "
        "different molecule"
    )


def test_refusing_the_overwrite_costs_no_molecule():
    """A refusal must not be a deletion. The GP molecule is not renamed, so
    it stays under its own cid — Stage C's invariant (every InChIKey that is
    a primary on the way in is a primary on the way out) holds through the
    refusal rather than because of a restore."""
    example_index, assay_tables = _overwrite_case()
    before = _primary_iks(example_index)

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    missing = before - _primary_iks(out)
    assert not missing, f"bridge lost {len(missing)} molecule(s): {sorted(missing)}"
    for cid, _n, _named, imposed in CASES:
        assert out[f"GP{cid}"]["canonical_smiles"] == imposed, (
            "the refused GP record must remain a molecule under its own cid"
        )


def test_refused_records_carry_no_alias_because_nothing_was_displaced():
    """`inchikey_aliases` is the fingerprint the corpus measurement counted.
    A refused overwrite displaces nothing, so it must leave none behind."""
    example_index, assay_tables = _overwrite_case()

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    for cid, *_ in CASES:
        assert not out[cid].get("inchikey_aliases"), (
            f"{cid} still records a displaced key — the overwrite happened"
        )


# ── the bridge is not disabled ───────────────────────────────────────

def test_overwrite_still_fires_when_the_name_is_not_evidence():
    """Three ways of not knowing, all of which must leave the old behaviour
    exactly as it was: no name at all, a name OPSIN cannot parse, and a name
    that resolves to some THIRD structure. The veto refuses a contradiction;
    it does not require corroboration."""
    example_index = {
        # (a) no `iupac_name`
        "10": {"compound_id": "Cpd. No. 10", "canonical_smiles": SMI_15_NAMED,
               "inchikey": _ik(SMI_15_NAMED)},
        # (b) a name OPSIN cannot parse — 64 shipped records are this shape
        "11": {"compound_id": "Cpd. No. 11",
               "iupac_name": "compound of formula (I) wherein R1 is C1-C4 alkyl",
               "canonical_smiles": SMI_646_NAMED, "inchikey": _ik(SMI_646_NAMED)},
        # (c) a name that names NEITHER structure in play
        "12": {"compound_id": "Cpd. No. 12", "iupac_name": NAME_169,
               "canonical_smiles": SMI_646_NAMED, "inchikey": _ik(SMI_646_NAMED)},
        "GP10": {"compound_id": "GP10", "canonical_smiles": SMI_169_IMPOSED,
                 "inchikey": _ik(SMI_169_IMPOSED)},
        "GP11": {"compound_id": "GP11", "canonical_smiles": SMI_15_IMPOSED,
                 "inchikey": _ik(SMI_15_IMPOSED)},
        "GP12": {"compound_id": "GP12", "canonical_smiles": SMI_646_IMPOSED,
                 "inchikey": _ik(SMI_646_IMPOSED)},
    }
    assay_tables = {"10": _rows(2), "11": _rows(2), "12": _rows(2)}

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    assert out["10"]["inchikey"] == _ik(SMI_169_IMPOSED)
    assert out["11"]["inchikey"] == _ik(SMI_15_IMPOSED)
    assert out["12"]["inchikey"] == _ik(SMI_646_IMPOSED)
    for cid in ("10", "11", "12"):
        assert out[cid].get("inchikey_aliases"), \
            "the displaced key must still be preserved for the BDB cross-ref"


def test_stage_a_inchikey_merge_still_works():
    """CLAUDE.md's documented Stage A — "merges GP107 → 107 on exact InChIKey
    match" — runs before any of this and must be untouched. The patent cid
    here even carries a name that OPSINs to the molecule, which is precisely
    the shape the veto looks at; Stage A must still collapse the duplicate,
    because the veto is scoped to Stage B's overwrite and nothing else."""
    example_index = {
        "GP646": {"compound_id": "GP646", "canonical_smiles": SMI_646_NAMED,
                  "inchikey": _ik(SMI_646_NAMED),
                  "extraction_method": "gp_embedded_meta"},
        "646": {"compound_id": "Cpd. No. 646", "iupac_name": NAME_646,
                "canonical_smiles": "", "inchikey": _ik(SMI_646_NAMED)},
    }
    assay_tables = {"646": _rows(4)}

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    assert "GP646" not in out, "Stage A must still collapse the GP duplicate"
    assert out["646"]["canonical_smiles"] == SMI_646_NAMED
    assert len([c for c, r in out.items()
                if (r.get("inchikey") or "").strip() == _ik(SMI_646_NAMED)]) == 1


def test_stage_b_positional_rename_still_works():
    """The other half of the bridge: an orphan GP whose positional cid holds
    no molecule is renamed onto it so the assay rows join. There is no
    structure to contradict, so the veto must never see this path."""
    example_index = {
        "GP7": {"compound_id": "GP7", "canonical_smiles": SMI_169_NAMED,
                "inchikey": _ik(SMI_169_NAMED),
                "extraction_method": "gp_embedded_meta"},
        "GP8": {"compound_id": "GP8", "canonical_smiles": SMI_15_NAMED,
                "inchikey": _ik(SMI_15_NAMED),
                "extraction_method": "gp_embedded_meta"},
        # a `fill`: the cid exists with a name but no InChIKey yet
        "9": {"compound_id": "Cpd. No. 9", "iupac_name": NAME_646,
              "canonical_smiles": "", "inchikey": ""},
        "GP9": {"compound_id": "GP9", "canonical_smiles": SMI_646_NAMED,
                "inchikey": _ik(SMI_646_NAMED),
                "extraction_method": "gp_embedded_meta"},
    }
    assay_tables = {"7": _rows(2), "8": _rows(2), "9": _rows(2)}

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    assert out["7"]["inchikey"] == _ik(SMI_169_NAMED) and "GP7" not in out
    assert out["8"]["inchikey"] == _ik(SMI_15_NAMED) and "GP8" not in out
    assert out["9"]["inchikey"] == _ik(SMI_646_NAMED), "the `fill` must still fill"
