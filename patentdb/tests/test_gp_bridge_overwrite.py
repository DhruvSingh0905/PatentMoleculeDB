"""Stage A + Stage B must not annihilate a molecule between them.

Measured defect (corpus run 2026-08-05, `output_v2/text_extraction/`):
735 GP-embedded molecules across 10 patents ended the run with no
`example_index` entry whose primary `inchikey` was theirs — they survived
only as a string in some other entry's `inchikey_aliases`, which no
downstream join treats as a molecule. 90 of the 91 cids in the original
audit are in that set; US9694016 alone lost 245.

The mechanism is an INTERACTION, not one branch:

  1. Upstream mis-assignment puts molecule B's InChIKey on patent cid 87
     (US9694016: `gp_description_example_header` read Example 87 and
     stored the structure that belongs to Example 99).
  2. Stage A sees InChIKey B under both `GP99` and `87`, calls it a
     duplicate, and pops `GP99` — unconditionally, at
     `process_patent.py:1079`.
  3. Stage B then finds `GP87` orphaned, classifies cid 87 as
     `"overwrite"` (its IK differs from GP87's), and re-points cid 87 at
     molecule A. That is CORRECT for cid 87 — but it retracts the only
     thing that justified step 2, and molecule B is now nowhere.

Note what is *not* wrong: the overwrite itself improves cid 87, and cid
99 was free and carrying 14 real assay rows. Had Stage A left `GP99`
alone, Stage B would have renamed it onto cid 99 as a plain "fresh"
rename and both molecules would have landed correctly. So the fix
restores the loser rather than blocking the overwrite — a missing assay
is recoverable, a misattributed one is not, and a deleted molecule is
neither.
"""
from patentdb.routes.process_patent import _bridge_gp_to_harvest_cids

PID = "US_TEST_BRIDGE"

# Three distinct, RDKit-parseable drug-sized molecules. MW matters: the
# bridge calls `molecular_weight` on the GP record before deciding.
SMI_A = "CC(=O)Nc1ccc(OCCN2CCCCC2)cc1C(=O)Nc1ccccc1Cl"
SMI_B = "COc1ccc(CNC(=O)c2ccc(N3CCOCC3)nc2)cc1OCC1CC1"
SMI_C = "Cc1nc(-c2ccccc2F)cc(N2CCN(C(=O)c3ccncc3)CC2)n1"


def _iks():
    from patentdb.core.smiles_utils import get_inchikey
    return get_inchikey(SMI_A), get_inchikey(SMI_B), get_inchikey(SMI_C)


def _assay_rows(n):
    return [{"assay_name": "IC50", "value_numeric": 10.0 + i, "unit": "nM",
             "qualifier": "", "n_runs": None, "source": "t"} for i in range(n)]


def _primary_iks(index):
    return {(r.get("inchikey") or "").strip()
            for r in index.values() if (r.get("inchikey") or "").strip()}


def _displacement_case():
    """Reproduces US9694016's cid 87/99 collision exactly.

    Returns (example_index, assay_tables, ik_a, ik_b, ik_c).
    """
    ik_a, ik_b, ik_c = _iks()
    example_index = {
        # Strategy 0, positional. GP87 really is Example 87's molecule.
        "GP87": {"compound_id": "GP87", "canonical_smiles": SMI_A,
                 "inchikey": ik_a, "extraction_method": "gp_embedded_meta"},
        "GP88": {"compound_id": "GP88", "canonical_smiles": SMI_C,
                 "inchikey": ik_c, "extraction_method": "gp_embedded_meta"},
        # GP99 really is Example 99's molecule.
        "GP99": {"compound_id": "GP99", "canonical_smiles": SMI_B,
                 "inchikey": ik_b, "extraction_method": "gp_embedded_meta"},
        # Upstream mis-assignment: cid 87 carries molecule B's structure.
        "87": {"compound_id": "Example 87", "canonical_smiles": SMI_B,
               "inchikey": ik_b,
               "extraction_method": "gp_description_example_header"},
        # A correctly-aligned sibling, so Stage A has an honest merge to do.
        "88": {"compound_id": "Example 88", "canonical_smiles": SMI_C,
               "inchikey": ik_c,
               "extraction_method": "gp_description_example_header"},
    }
    # Three bare-digit assay cids all matching a GP index → this is what
    # sets `digit_alignment_trusted`, hence `overwrite_trusted`.
    assay_tables = {"87": _assay_rows(14), "88": _assay_rows(3),
                    "99": _assay_rows(14)}
    return example_index, assay_tables, ik_a, ik_b, ik_c


def test_bridge_never_drops_a_molecule_it_was_given():
    """The invariant. Every InChIKey that entered the bridge as some
    entry's primary must still be some entry's primary on the way out.

    Before the fix this failed with molecule B missing: Stage A popped
    `GP99`, Stage B overwrote cid 87, and B existed only as the string
    in `example_index["87"]["inchikey_aliases"]`.
    """
    example_index, assay_tables, ik_a, ik_b, ik_c = _displacement_case()
    before = _primary_iks(example_index)

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    missing = before - _primary_iks(out)
    assert not missing, (
        f"bridge destroyed {len(missing)} molecule(s): {sorted(missing)}. "
        "An InChIKey demoted to `inchikey_aliases` is not a molecule — no "
        "downstream join reads it as one."
    )


def test_displaced_molecule_lands_on_its_own_assay_cid():
    """Stronger than survival: molecule B belongs at cid 99, which was
    free and holding 14 assay rows the run would otherwise orphan.

    This pins the RECOVERY, not just the non-deletion — restoring B under
    a synthetic key would satisfy the invariant above while still leaving
    cid 99's rows unattributable.
    """
    example_index, assay_tables, ik_a, ik_b, ik_c = _displacement_case()

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    holders = [c for c, r in out.items()
               if (r.get("inchikey") or "").strip() == ik_b]
    assert holders == ["99"], (
        f"molecule B should be recovered at cid 99, found at {holders}"
    )
    assert out["99"]["canonical_smiles"] == SMI_B


def test_overwrite_still_corrects_the_mislabelled_cid():
    """The fix must not be a rollback of the overwrite. Cid 87 held the
    WRONG structure on the way in; the bridge is supposed to re-point it
    at GP87's molecule, and that behaviour is load-bearing (it is how the
    +41 v2_has_ay on US10246453 was earned, per the comment at :1203).
    """
    example_index, assay_tables, ik_a, ik_b, ik_c = _displacement_case()

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    assert out["87"]["inchikey"] == ik_a
    assert out["87"]["canonical_smiles"] == SMI_A
    # the displaced key stays reachable as an alias for the BDB cross-ref
    assert ik_b in (out["87"].get("inchikey_aliases") or [])


def test_stage_a_inchikey_merge_still_works():
    """Guard against fixing this by disabling the bridge.

    The documented Stage A contract (CLAUDE.md: "Stage A merges GP107 →
    107 on exact InChIKey match") must be untouched: a GP record whose
    InChIKey genuinely matches a patent cid that KEEPS that InChIKey is
    still collapsed into the patent cid, and no duplicate is resurrected.
    """
    ik_a, ik_b, ik_c = _iks()
    example_index = {
        "GP5": {"compound_id": "GP5", "canonical_smiles": SMI_A,
                "inchikey": ik_a, "extraction_method": "gp_embedded_meta"},
        # Stage A keys on InChIKey, so both sides must already carry one
        # (:1052 skips empty). The patent cid agrees on the molecule but
        # never got a structure — that is the case Stage A exists to fix.
        "5": {"compound_id": "Example 5", "iupac_name": "whatever",
              "canonical_smiles": "", "inchikey": ik_a},
    }
    assay_tables = {"5": _assay_rows(4)}

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    assert "GP5" not in out, "Stage A must still collapse the GP duplicate"
    assert out["5"]["inchikey"] == ik_a
    assert out["5"]["canonical_smiles"] == SMI_A
    assert len([c for c, r in out.items()
                if (r.get("inchikey") or "").strip() == ik_a]) == 1, \
        "exactly one entry should hold the molecule — no resurrected twin"


def test_stage_b_fresh_rename_still_works():
    """The other half of the bridge's job: an orphan GP whose positional
    cid is absent from example_index is renamed onto it so the assay rows
    join. Pins that the restore pass didn't change the ordinary path.
    """
    ik_a, ik_b, ik_c = _iks()
    example_index = {
        "GP7": {"compound_id": "GP7", "canonical_smiles": SMI_A,
                "inchikey": ik_a, "extraction_method": "gp_embedded_meta"},
        "GP8": {"compound_id": "GP8", "canonical_smiles": SMI_B,
                "inchikey": ik_b, "extraction_method": "gp_embedded_meta"},
        "GP9": {"compound_id": "GP9", "canonical_smiles": SMI_C,
                "inchikey": ik_c, "extraction_method": "gp_embedded_meta"},
    }
    assay_tables = {"7": _assay_rows(2), "8": _assay_rows(2),
                    "9": _assay_rows(2)}

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    assert out["7"]["inchikey"] == ik_a
    assert "GP7" not in out


def test_restore_pass_is_insert_only():
    """Stage C's cost budget, pinned.

    The restore pass runs unconditionally rather than behind a trust flag,
    which is only defensible because it cannot take anything: it writes to
    cids nothing else claimed and never moves an assay row. So every cid
    that held a molecule before the restore must hold the SAME molecule
    after, and the set of cids may only grow.
    """
    example_index, assay_tables, ik_a, ik_b, ik_c = _displacement_case()
    ay_before = {c: len(rows) for c, rows in assay_tables.items()}

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    # nothing Stage C added displaced an earlier decision
    assert out["87"]["inchikey"] == ik_a          # Stage B's overwrite stands
    assert out["88"]["inchikey"] == ik_c          # Stage A's merge stands
    assert {c: len(r) for c, r in assay_tables.items()} == ay_before, \
        "the bridge must not move assay rows"
    # every restored record is traceable to where it came from
    for cid, rec in out.items():
        if "restored_from" in rec:
            assert rec["restored_from"] not in out, \
                f"{cid} restored from a cid that still exists — that is a copy, not a rescue"


def test_untrusted_patent_restores_without_claiming_a_cid():
    """With no alignment signal there is no overwrite to undo, but if a
    molecule is ever lost the restore must not invent a positional
    binding to save it. Molecule survival is unconditional; claiming a
    patent cid is not — that asymmetry is the whole point of the module's
    founding rule (a missing assay is recoverable, a misattributed one is
    not).
    """
    ik_a, ik_b, ik_c = _iks()
    example_index = {
        "GP2": {"compound_id": "GP2", "canonical_smiles": SMI_A,
                "inchikey": ik_a, "extraction_method": "gp_embedded_meta"},
        "2": {"compound_id": "Example 2", "canonical_smiles": SMI_B,
              "inchikey": ik_b,
              "extraction_method": "gp_description_example_header"},
    }
    # one assay cid → n_ay_digits=1 < 3, so digit alignment is untrusted,
    # and one Stage A merge is far short of the >=5 positional threshold
    assay_tables = {"2": _assay_rows(1)}

    out = _bridge_gp_to_harvest_cids(PID, dict(example_index), assay_tables)

    primaries = _primary_iks(out)
    assert ik_a in primaries and ik_b in primaries, "both molecules survive"
    assert out["2"]["inchikey"] == ik_b, \
        "no trust signal → cid 2 keeps the molecule it came with"
