"""`core/assay_fsm/output_validator.py` is gone. It deleted real assays.

CLAUDE.md described it as keeping a `(cid, value)` only when corroborated
by a MinerU `<table>` block or the GP flat description within 200 chars
of a standalone cid token. It had not done that for some time:

  DEAD — `_load_table_blocks`, `_value_near_cid`, and
  `_value_near_cid_in_flat_text` (the entire corroboration mechanism)
  had no callers anywhere in the tree.

  INERT — the two surviving keep/drop branches read `validation_reason`
  and `source_quote`. The caller (`process_patent.py:1873-1889`) builds
  each row with six keys — assay_name, value_numeric, unit, qualifier,
  n_runs, source — and `harvest/orchestrator.py:68` writes
  `source=t.validation_reason`, i.e. it RENAMES the very field branch 1
  looks for. Shipped rows show it plainly: `source:
  "pattern_library:2d44394d6b85a293"`. So `n_pattern_kept` and
  `n_prose_dropped` were structurally pinned at 0.

  LIVE AND WRONG — one branch added later (`3d38050`) survived: a
  property-table heuristic that drops a value whose only co-occurrence
  with its cid falls inside an `LCMS m/z` / `HPLC tR` span.

Measured by replaying `extract_for_patent` from cache (zero paid calls)
and diffing the validator in vs. out:

    patent        rows in   dropped
    US8952177         492         0
    US9718790        4739         4
    US10214537       3553        12
    US10273259       1542       707   <- 46% of rows, 110 whole compounds

Every drop came from the property branch; the prose branch dropped 0,
confirming its inertness on real data. And the drops were not property
values: 681 of the 707 on US10273259 were `RORγ Binding IC50 μM`, the
patent's PRIMARY assay. Flattening the raw grant XML (per CLAUDE.md's
"never diagnose from the parsed view") put 46 of a 60-row sample —
77% — adjacent to their cid in the patent's own text. Only 111 of the
681 survived via the USPTO XML arm, so the rest were lost outright.

It was also no longer catching the defect it was written for: on
US9718790, the patent whose HPLC retention times motivated `3d38050`,
it dropped 4 rows and all four were μM IC50 values.

Removing it ADDS rows back; it cannot lose any. That asymmetry is why
this landed rather than being reported and left.
"""
from __future__ import annotations

import importlib

import pytest


def test_output_validator_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("patentdb.core.assay_fsm.output_validator")


def test_pipeline_does_not_filter_assay_rows():
    """No survivor of the module may be wired back into the orchestrator.

    Asserts on WIRING (an import or a call), not on prose: the comment
    left at the old call site names the module deliberately, so that the
    707-row measurement stays next to the code it explains.
    """
    import inspect
    import re

    from patentdb.routes import process_patent as pp

    src = inspect.getsource(pp)
    assert "filter_assays_to_table_blocks(" not in src
    assert not re.search(r"^\s*(from|import)\s+.*output_validator", src, re.M)
