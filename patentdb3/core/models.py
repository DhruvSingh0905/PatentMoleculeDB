"""The serialised shape of an assay row.

Carried into v3 from `patentdb/core/models.py`, which is 375 lines of which
only `AssayRow` is reachable from the reader — `uspto_assays.AssayRecord.as_dict`
at :416 constructs one, and `AssayRow._ALIASES` is what lets the old key
spellings still read. Nothing else in that file is used by the move set, so
copying it whole would import the assembly's vocabulary along with it.
"""
from __future__ import annotations


class AssayRow(dict):
    """The ONE serialised shape of an assay row, shared by both binned paths.

    `routes/letter_bin_assays.py` and `sources/uspto_assays.AssayRecord` both
    write ranges into the same `assay_tables.json`, and used to spell them
    differently — `value_low/value_high/value_raw/bin` (8,020 shipped rows,
    read by both JSON consumers) versus `range_lo/range_hi/letter_grade/
    value_text` (0 shipped rows, never exercised by a reader). A consumer had
    to know both.

    Written keys are the first spelling only: iteration, `keys()` and
    `json.dumps` see one schema. The second spelling stays READABLE via `[]`
    and `.get()` so the dataclass-side vocabulary — which `range_lo`/`range_hi`
    remain, six eval scripts read them off `AssayRecord` with getattr — does
    not have to be relearned at the dict boundary.
    """
    _ALIASES = {
        "range_lo": "value_low",
        "range_hi": "value_high",
        "letter_grade": "bin",
        "value_text": "value_raw",
    }

    def __missing__(self, key):
        canonical = self._ALIASES.get(key)
        if canonical is not None and dict.__contains__(self, canonical):
            return dict.__getitem__(self, canonical)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
