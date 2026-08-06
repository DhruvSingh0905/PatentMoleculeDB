"""The CALS reader was made faster. These say it was not made different.

Every optimisation in `sources/uspto_assays.py`, `sources/uspto_xml.py` and
`sources/bin_legend.py` is one of three shapes — a memo on a pure function, an
invariant hoisted out of a loop, or a pattern compiled once — and each replaced
an expression that is still written out below. The tests hold the replacement
against the original over the real vocabulary and a fuzz corpus, because
"obviously equivalent" is how a caching bug gets into a reader that scores
99.9% exact-match and cannot be checked by eye afterwards.

The one that is NOT a matter of taste is `test_classify_column_never_shares`:
`Column` is mutable and `build_columns` mutates what `classify_column` returns,
so a memo that handed back its own instance would let one table silently
rewrite every later table's classification.
"""
from __future__ import annotations

import random
import re

from patentdb.sources import bin_legend, uspto_xml
from patentdb.sources.uspto_assays import (ASSAY, CID, Column, _assay_lemma_re,
                                           _count_matching, _header_rows_of,
                                           _is_spacer, _shapes_of, _split_rows,
                                           _column_shapes, _vocab,
                                           classify_column, split_top_level)
from patentdb.sources.uspto_xml import Cell, Table


# ── the memo may not share a mutable Column ───────────────────────

def test_classify_column_never_shares():
    """Two calls with the same arguments must not return the same object.

    `build_columns` writes `c.index`, `c.kind`, `c.unit` and `c.assay_name` on
    what it gets back. If the cache returned its stored instance, the second
    table with this header would inherit the first table's index and unit.
    """
    first = classify_column("hERG IC50 (nM)", [])
    assert first.kind == ASSAY and first.unit == "nM"

    first.index = 7
    first.kind = CID
    first.unit = "mM"
    first.assay_name = "clobbered"

    second = classify_column("hERG IC50 (nM)", [])
    assert second is not first
    assert (second.index, second.kind, second.unit, second.assay_name) == \
           (-1, ASSAY, "nM", "hERG IC50 (nM)")


def test_classify_column_distinguishes_its_samples():
    """The samples are part of the key, not decoration on it.

    An unrecognised header over plus-bins is promoted to ASSAY by the data
    shape alone; the same header over prose is not. Keying on the header only
    would return whichever was asked for first.
    """
    binned = classify_column("FP", ["+", "++", "+++", "++"])
    prose = classify_column("FP", ["a phrase", "another phrase", "a third"])
    assert binned.kind == ASSAY
    assert prose.kind != ASSAY


def test_classify_column_survives_unhashable_samples():
    """An unhashable sample list falls through to the function, not an error."""
    col = classify_column("hERG IC50 (nM)", [["not", "a", "string"]])
    assert col.kind == ASSAY


# ── the vocabulary scan ───────────────────────────────────────────

def test_lemma_regex_is_the_substring_scan():
    """`_assay_lemma_re()` must accept exactly what the comprehension did."""
    lemmas = _vocab()[0]

    def old(low: str) -> bool:
        return any(a in low for a in lemmas if len(a) > 2)

    new = _assay_lemma_re()
    headers = [
        "", "Compound No.", "hERG IC50 (nM)", "Ave A2B cAMP IC50", "FP",
        "MAGL % Inh 1 uM (mouse)", "1H NMR (CD3OD, 400 MHz) δ", "pIC50",
        "LM CLint (uL/min/mg/protein)", "Ratio", "m/z [M+H]+", "MW",
        "Structure", "R1", "(n)", "Ki app.", "-log10 IC50", "potency",
        "kd(app)", "k d", "Example", "TABLE 569", "0.0125", "nd",
    ]
    headers += [a for a in lemmas]
    headers += [f"x{a}y" for a in lemmas]
    for h in headers:
        low = h.lower()
        assert old(low) == bool(new.search(low)), h


def test_empty_vocabulary_matches_nothing():
    """`re.compile("")` matches everything; a missing vocabulary must not."""
    assert re.compile(r"(?!)").search("hERG IC50") is None


# ── the counting helper ───────────────────────────────────────────

def test_count_matching_is_the_sum_of_bools():
    pat = re.compile(r"^\d+$")
    for vals in ([], ["1"], ["1", "a"], ["1", "2", "3"], ["", "0"]):
        assert _count_matching(pat, vals) == sum(bool(pat.match(v)) for v in vals)


# ── the row/text predicates ───────────────────────────────────────

def test_is_spacer_matches_the_strip_form():
    rnd = random.Random(11)
    alphabet = ["", " ", "\t", "\n", " ", "0", "a", " a ", "  ", " "]
    for _ in range(3000):
        row = [Cell(rnd.choice(alphabet)) for _ in range(rnd.randint(0, 5))]
        assert _is_spacer(row) == (not any(c.text.strip() for c in row))


def test_text_matches_the_regex_form():
    """`" ".join(s.split())` against `re.sub(r"\\s+", " ", s).strip()`."""
    ws = re.compile(r"\s+")

    def old(fragment: str) -> str:
        import html
        s = ws.sub("", "") or fragment
        s = re.sub(r"<[^>]+>", "", s)
        s = html.unescape(s)
        return ws.sub(" ", s).strip()

    rnd = random.Random(5)
    chars = " \t\n\r\f\v   <sub>&amp;0.5aA-"
    cases = ["", " ", "IC<sub>50</sub>", "[M+H]<sup>+</sup>", "&lt;10&nbsp;nM",
             "  0.5  ", "  ", "a b"]
    for _ in range(4000):
        cases.append("".join(rnd.choice(chars) for _ in range(rnd.randint(0, 14))))
    for c in cases:
        assert uspto_xml._text(c) == old(c), repr(c)


def test_split_top_level_fast_paths():
    """The no-comma and no-bracket shortcuts against the depth loop."""
    def old(text: str) -> list[str]:
        parts, depth, cur = [], 0, []
        for ch in text or "":
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        parts.append("".join(cur).strip())
        return [p for p in parts if p]

    rnd = random.Random(3)
    cases = ["", "probe 1, probe 2", "1,234.5", "HT1080 (R132C, aKG) IC50 (uM)",
             ",,,", " a , b ", "[a,b],c", "((a,b)", "a)b,c", "no commas here"]
    for _ in range(4000):
        cases.append("".join(rnd.choice("ab,()[]{} .0") for _ in range(rnd.randint(0, 16))))
    for c in cases:
        assert split_top_level(c) == old(c), repr(c)


def test_looks_like_key_matches_the_two_pattern_form():
    a = re.compile(r"is\s+marked|\bkey\s*:|\*\s*key", re.I)
    b = re.compile(
        rf"{bin_legend._SYMBOL}{bin_legend._DEFINES}"
        rf"(?:IC\s*50|EC\s*50|{bin_legend._NUM}|[<>≤≥≦≧⩽⩾])", re.I)

    def old(text: str) -> bool:
        if not text:
            return False
        return bool(a.search(text)) or bool(b.search(text))

    rnd = random.Random(17)
    cases = ["", "+", "A", "0.5", "I-117", "++++: IC50 ≥ 1 uM",
             "+ refers to ≤10 nM", "A = <10 nM", "Key: + is 1 uM",
             "the * key", "activity is marked", "B: 10-50 nM", "nd", "n.d.",
             "IC 50 : A <= 10 nM; 10 nM < B <= 100 nM"]
    for _ in range(4000):
        cases.append("".join(rnd.choice("+ABCDE:=<>0.9 refersto key*marked")
                             for _ in range(rnd.randint(0, 18))))
    for c in cases:
        assert bin_legend.looks_like_key(c) == old(c), repr(c)


# ── the per-table memos ───────────────────────────────────────────

def _table() -> Table:
    return Table(
        table_id="TABLE-US-00001", n_cols=3,
        header_rows=[[Cell("Cmpd No."), Cell("hERG IC50 (nM)"), Cell("(n)")]],
        body_rows=[[Cell("1"), Cell("12.5"), Cell("(3)")],
                   [Cell("2"), Cell("0.44"), Cell("(4)")],
                   [Cell(""), Cell(""), Cell("")]],
        caption="Table 1. hERG inhibition.")


def test_split_rows_agrees_with_the_uncached_primitive():
    t = _table()
    assert _split_rows(t) == _header_rows_of(t)
    assert _split_rows(t) is _split_rows(t)          # served from the memo


def test_shapes_of_agrees_and_is_keyed_on_the_rows_it_was_given():
    t = _table()
    _, data = _split_rows(t)
    assert _shapes_of(t, data) == _column_shapes(t, data)

    # Handed a DIFFERENT list, the memo must recompute rather than replay.
    other = [[Cell("prose one"), Cell("prose two"), Cell("prose three")]]
    assert _shapes_of(t, other) == _column_shapes(t, other)
    assert _shapes_of(t, data) == _column_shapes(t, data)


def test_two_tables_do_not_share_a_memo():
    a, b = _table(), _table()
    b.body_rows = [[Cell("prose here"), Cell("more prose"), Cell("and more")]]
    _, da = _split_rows(a)
    _, db = _split_rows(b)
    assert _shapes_of(a, da) == _column_shapes(a, da)
    assert _shapes_of(b, db) == _column_shapes(b, db)


# ── the eval module is out of the live import set ─────────────────

def test_value_check_does_not_import_from_scripts_eval():
    """`repair/value_check.py` is production and must not reach into evals.

    It used to do `from ..scripts.eval.reference_bench import _EXAMPLE_REF,
    _norm_cid` inside `load_reference`, which put an eval module in the live
    set — `capability._bad_values_now` reached `scripts/eval/` in the middle of
    an extraction run. The definitions live with the reference reader now.
    """
    import inspect

    from patentdb.repair import value_check

    assert "scripts.eval" not in inspect.getsource(value_check.load_reference)
    assert "scripts.eval" not in "".join(
        l for l in inspect.getsource(value_check).splitlines()
        if l.strip().startswith(("import ", "from ")))


def test_reference_bench_re_exports_the_same_objects():
    """The benchmark's names are the package's, not copies of them."""
    from patentdb.repair import value_check
    from patentdb.scripts.eval import reference_bench

    assert reference_bench._EXAMPLE_REF is value_check._EXAMPLE_REF
    assert reference_bench._norm_cid is value_check._norm_cid


def test_norm_cid_still_canonicalises_bindingdbs_padding():
    """The memo must not change the answer that cost 1,119 compounds."""
    from patentdb.repair.value_check import _norm_cid

    assert _norm_cid("I-0117") == "I-117"
    assert _norm_cid("Example 007") == "7"
    assert _norm_cid("Cpd. No. 7") == "7"
    assert _norm_cid("007") == "7"
