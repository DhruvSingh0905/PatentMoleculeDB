"""DSPy modules for patent extraction — typed, optimizable LM calls.

Wraps key LM-heavy stages as DSPy modules with typed signatures.
Can be optimized against BindingDB metrics using DSPy's optimizers.
"""

from __future__ import annotations

import dspy
from . import config


def configure_dspy():
    """Configure DSPy with Anthropic Claude."""
    lm = dspy.LM("anthropic/claude-sonnet-4-6", api_key=config.ANTHROPIC_API_KEY)
    dspy.configure(lm=lm)


class CleanIUPACName(dspy.Signature):
    """Fix a noisy IUPAC chemical name so the OPSIN parser can parse it.
    Common issues: OCR typos (isquinolin→isoquinolin, pyrolo→pyrrolo),
    missing letters, transposed characters, broken brackets."""
    noisy_name: str = dspy.InputField(desc="IUPAC name with OCR errors")
    opsin_error: str = dspy.InputField(desc="Error from OPSIN parser")
    corrected_name: str = dspy.OutputField(desc="Corrected IUPAC name")


class ExtractCompounds(dspy.Signature):
    """Extract compound identifiers and IUPAC names from patent page text."""
    page_text: str = dspy.InputField(desc="Markdown text of a patent page")
    compounds: list[dict] = dspy.OutputField(desc="List of {compound_id, iupac_name, ms_mh_plus}")


class InferTableSchema(dspy.Signature):
    """Identify column roles in a patent compound table."""
    header_html: str = dspy.InputField(desc="HTML of the table header row(s)")
    schema: dict = dspy.OutputField(desc="Mapping of column index to role: compound_id, structure, name, mh_plus, assay")


# Instantiate predictors
_cleaner = None
_extractor = None
_schema_inferrer = None


def get_cleaner():
    global _cleaner
    if _cleaner is None:
        configure_dspy()
        _cleaner = dspy.Predict(CleanIUPACName)
    return _cleaner


def clean_iupac_dspy(noisy_name: str, opsin_error: str) -> str | None:
    """Use DSPy to clean an IUPAC name."""
    try:
        result = get_cleaner()(noisy_name=noisy_name, opsin_error=opsin_error)
        return result.corrected_name
    except Exception:
        return None
