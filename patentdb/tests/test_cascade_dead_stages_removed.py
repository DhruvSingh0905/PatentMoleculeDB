"""The IUPAC cascade has FOUR stages, not six. Two were unreachable.

Both removed stages existed to repair OCR damage, and OCR (MinerU) is no
longer a source the pipeline reads — so they had nothing left to repair
even in principle. But they were already dead before that:

  - **Levenshtein autocorrect** imported `core.ocr_autocorrect`, a module
    that has never existed in this tree, inside `try: ... except
    ImportError: pass`. Every call raised ImportError and was swallowed,
    so the stage was a no-op that *looked* live in the stage list and in
    ARCH.md. A silently-swallowed import of a missing module is
    indistinguishable from a stage that ran and found nothing.

  - **Vision OCR** was gated on `compound.source_page is not None`. The
    only live constructors that SET `source_page` are
    `routes/google_tables.py:347,409`, and those compounds go to
    `_compounds_to_example_index` — they never reach `_convert_single`.
    Every compound that DOES reach the cascade
    (`process_patent.py:230,375`, `google_patents.py:1115`) is built
    without `source_page`, so the gate was never open.

Measured: removing both changed 0 records across all 22 corpus patents —
no shipped record in `output_v2/text_extraction/*/example_index.json`
carries `extraction_method` of `autocorrected`, `vision_ocr`, or
`vision_ocr_cleaned`.

These tests declare the removal so the stages cannot return unnoticed.
"""
from __future__ import annotations

import inspect

from patentdb.core import iupac_to_smiles as I
from patentdb.core.models import Compound


def _code_without_module_docstring() -> str:
    """Module source minus its docstring.

    The docstring deliberately NAMES both removed stages so the reason
    they went survives in the file — CLAUDE.md's rule that a whole-file
    rewrite must not drop the reasoning. These assertions are about
    executable code, so the prose that explains the removal must not
    trip them.
    """
    src = inspect.getsource(I)
    doc = I.__doc__ or ""
    return src.replace(doc, "", 1)


def test_no_import_of_nonexistent_ocr_autocorrect():
    """The cascade must not import a module that does not exist.

    `except ImportError: pass` around a real dependency hides a genuine
    packaging break; around a module that was never written it is a
    permanently-dead branch pretending to be a stage.
    """
    src = _code_without_module_docstring()
    assert "ocr_autocorrect" not in src
    assert "autocorrect_iupac" not in src


def test_no_swallowed_importerror_in_cascade():
    """No bare `except ImportError` remains to hide the next dead stage."""
    assert "except ImportError" not in _code_without_module_docstring()


def test_vision_ocr_helpers_are_gone():
    """`_vision_ocr_name` and its only gate `_is_truncated` are removed."""
    assert not hasattr(I, "_vision_ocr_name")
    assert not hasattr(I, "_is_truncated")


def test_cascade_never_calls_vision_even_with_source_page_set():
    """A truncated name WITH `source_page` set must not reach Vision.

    This is the state the removed gate was waiting for — unbalanced
    parens plus a page number. Constructing it by hand proves the path is
    gone rather than merely unreached by production callers.
    """
    called = []

    import patentdb.core.api_client as api

    orig = getattr(api, "call_claude_vision", None)
    api.call_claude_vision = lambda *a, **k: called.append(1) or "x"
    try:
        c = Compound(
            patent_id="USTEST",
            example_number="1",
            # Deliberately truncated: unbalanced parens -> old `_is_truncated`
            iupac_name="2-(4-chlorophenyl)-N-[3-(morpholin-4-yl)propyl",
            source_page=3,
        )
        I._convert_single(c, is_clean_text=False)
    finally:
        if orig is not None:
            api.call_claude_vision = orig

    assert called == [], "Vision OCR stage fired; it should not exist"


def test_finalize_does_not_know_removed_stage_names():
    """`_finalize`'s stereo-trust table must not list removed stages."""
    src = inspect.getsource(I._finalize)
    assert "vision_ocr" not in src
    assert "vision_ocr_cleaned" not in src
