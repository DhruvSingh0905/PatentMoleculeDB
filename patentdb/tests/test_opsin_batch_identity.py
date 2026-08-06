"""Batched OPSIN must return byte-identically what one-name-at-a-time returned.

`py2opsin` shells out to `java -jar opsin-cli.jar` on every call. A call trace
over three patents measured 3,726 `_try_opsin` calls at ~0.19 s each — 721.3 s,
58.9% of all wall time — almost all of it JVM startup. py2opsin already accepts
a list and writes it one name per line, so a whole patent's names cost one JVM.

The reason that is not a one-line change is stderr. A list call raises ONE
warning holding the concatenated stderr of the run, and OPSIN's messages are
not 1:1 with failures: 600 corpus names produced 54 empty results and 56 stderr
lines, several of which ("APPEARS_AMBIGUOUS: …", "hydrogen addition at locant:
13 …") name no compound, so neither positional nor prefix attribution works.
`_try_opsin`'s strict mode READS that message — an "Unmatched bracket" warning
rejects a parse that otherwise returns a SMILES — so a mis-attributed message
silently changes which structures ship. Mis-assigning a structure to a compound
is the worst failure this codebase has.

So the contract this file pins is IDENTITY, at two levels:

  * `_opsin_raw` — the SMILES string AND the warning text, byte for byte,
  * `_try_opsin` — the gated (smiles, error) tuple, under strict and lenient
    mode, relaxed and not.

and one safety property: when the batch's structural checks fail, the batch is
DISCARDED and every name resolves alone, rather than a guess being memoised.

ZERO paid calls: nothing here touches an LLM, and `anthropic.Anthropic` is
replaced by a raiser so a leak is an error. OPSIN is a local jar; PubChem is
never contacted because `config.PUBCHEM_NAME_LOOKUP_ENABLED` defaults off.

Set OPSIN_IDENTITY_N to widen the corpus sample (each extra name costs one JVM
launch in the per-call baseline, ~0.19 s). Default is small so the suite stays
fast; the reported measurement was taken with it raised.
"""
from __future__ import annotations

import glob
import json
import os
import warnings

import pytest

import anthropic

from patentdb.core import config
from patentdb.core import iupac_to_smiles as its
from patentdb.core.iupac_to_smiles import (
    _opsin_atoms,
    _opsin_batch,
    _opsin_format_warning,
    _opsin_raw,
    _opsin_sentinel,
    _try_opsin,
    clear_opsin_memo,
    prefetch_opsin,
)


@pytest.fixture(autouse=True)
def _no_paid_calls(monkeypatch):
    def _raise(*a, **k):
        raise AssertionError("paid API call attempted from an OPSIN test")
    monkeypatch.setattr(anthropic, "Anthropic", _raise)


@pytest.fixture(autouse=True)
def _cold_memo():
    clear_opsin_memo()
    yield
    clear_opsin_memo()


# ── Names, verbatim from the corpus ───────────────────────────────
#
# Chosen to cover every branch the batch has to reproduce: clean parses,
# hard failures, a parse that SUCCEEDS while emitting a warning (the strict
# warning gate reads exactly those), `;` multi-component names, OCR typos the
# rule cleaner targets, and a name long enough to trip the strict coverage
# check. Sourced from output_v2/text_extraction/*/example_index.json.
CORPUS_NAMES = [
    # plain, parses
    "benzene",
    "toluene",
    "2,4-dichlorophenol",
    "(3R)-3-methylhexane",
    "1-(1-benzylpiperidin-4-yl)-1-cyclopentyl-1,2,3,4-tetrahydroisoquinoline",
    "N-[4-(3-Amino-1H-pyrazolo[3,4-d]pyrimidin-6-yl)-phenyl]-5-chloro-2-fluoro-"
    "benzenesulfonamide",
    "(1S,2R)-2-(1-(4-Bromobenzyl)-6-((5-methylpyridin-2-yl)methoxy)-1H-"
    "benzo[d]imidazol-2-yl)cyclohexanecarboxylic acid",
    "COc1nc(-c2ccc(N)cc2)nc2[nH]nc(N)c12",          # a SMILES, not a name
    # hard failures — these are what put lines on stderr
    "notachemicalnameatall",
    "pyrolo[2,1-f]triazine",
    "654.1 2.39 (M + H)+",
    "1H NMR (400 MHz, DMSO-d6)",
    # multi-component: `_try_opsin` splits on `;` and joins with `.`
    "dicesium;carbonate",
    "benzene;toluene",
    "benzene;notachemicalnameatall",
    "3-methyl-6-pyrazol-1-yl-1H-1,3,5-triazine-2,4-dione;"
    "pyrazole-1-carboximidamide;hydrochloride",
    # OCR-shaped inputs the strict pre-filters and rule cleaner target
    "racemic cis-2-(4-fluorophenyl)cyclopropane-1-carboxylic acid",
    "(cis)-methyl 4-(3-(4-amino-5-(1-(tetrahydro-2H-pyran-4-yl)-1H-pyrazol-5-yl)"
    "pyro1ol[2,1-f][1,2,4]triazin-7-yl)phenyl)-3,5-dimethylpiperazine-1-carboxylate",
    "7-(3-(2-(dimethylamino) propyl)phenyl)-5-(1-(tetrahydro-2H-pyran-4-yl)-1H-"
    "pyra- zol-5-yl)pyrrolo[2,1-f][1,2,4]triazin-4-amine",
    "2-methylpropanoic acid, HCl",
    # a parse that returns a SMILES *and* warns — the strict warning gate's case
    "1,2,3,4,5,6,7,8,9,10,11,12,13-tridecahydroanthracene",
]


def _corpus_sample(n: int) -> list[str]:
    """Distinct `iupac_name` values from the shipped artifacts, if present."""
    names: list[str] = []
    seen: set[str] = set()
    pattern = str(
        config.OUTPUT_DIR / "text_extraction" / "*" / "example_index.json"
    )
    for path in sorted(glob.glob(pattern)):
        try:
            index = json.loads(open(path).read())
        except (OSError, ValueError):
            continue
        if not isinstance(index, dict):
            continue
        for rec in index.values():
            if not isinstance(rec, dict):
                continue
            name = (rec.get("iupac_name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
            if len(names) >= n:
                return names
    return names


def _sample() -> list[str]:
    extra = int(os.environ.get("OPSIN_IDENTITY_N", "12"))
    names = list(CORPUS_NAMES)
    seen = set(names)
    for n in _corpus_sample(extra + len(names)):
        if n not in seen and len(names) < len(CORPUS_NAMES) + extra:
            seen.add(n)
            names.append(n)
    return names


# ── The identity proof ────────────────────────────────────────────

@pytest.fixture(scope="module")
def per_call():
    """The pre-batching behaviour: every name in its own JVM launch.

    Computed once for the file — it is the expensive half (one `java -jar` per
    name per flag set) and it is the ground truth every test here compares
    against. The memo is cleared before each name so no call can be served
    from a sibling's result.
    """
    names = _sample()
    baseline: dict[bool, list[tuple]] = {}
    for relaxed in (False, True):
        out = []
        for n in names:
            clear_opsin_memo()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out.append(_opsin_raw(n, relaxed=relaxed))
        baseline[relaxed] = out
    clear_opsin_memo()
    return names, baseline


@pytest.mark.parametrize("relaxed", [False, True])
def test_batched_raw_result_is_byte_identical(per_call, relaxed):
    """The SMILES *and* OPSIN's warning text must survive batching intact.

    Compared per name so a failure names the offender rather than dumping two
    lists — a mis-ordered batch shows up here as a pair of names swapped.
    """
    names, baseline = per_call
    assert len(names) >= 20, "sample too small to mean anything"
    expected = baseline[relaxed]

    clear_opsin_memo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prefetch_opsin(names, relaxed=relaxed)
        actual = [_opsin_raw(n, relaxed=relaxed) for n in names]

    assert len(actual) == len(expected)
    for name, exp, act in zip(names, expected, actual):
        assert act == exp, (
            f"batched OPSIN diverged on {name[:70]!r}\n"
            f"  per-call: {exp!r}\n"
            f"  batched : {act!r}"
        )


@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_gated_try_opsin_is_identical(per_call, strict, relaxed):
    """Same, one level up: the tuple `_try_opsin` hands its callers.

    This is the level that covers the `;` multi-component split (which never
    reaches `_opsin_raw` whole), the strict input pre-filters, the strict
    coverage check and the strict warning gate.

    The "expected" side runs `_try_opsin` against a memo holding the per-call
    answers — one JVM launch per name, recorded by the fixture — so it is the
    pre-batching result without paying for it four more times.
    """
    names, baseline = per_call

    clear_opsin_memo()
    its._OPSIN_MEMO.update({
        (n, relaxed): v for n, v in zip(names, baseline[relaxed])
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = [_try_opsin(n, strict=strict, relaxed=relaxed) for n in names]

    clear_opsin_memo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prefetch_opsin(names, relaxed=relaxed)
        actual = [_try_opsin(n, strict=strict, relaxed=relaxed) for n in names]

    for name, exp, act in zip(names, expected, actual):
        assert act == exp, (
            f"strict={strict} relaxed={relaxed} diverged on {name[:70]!r}\n"
            f"  per-call: {exp!r}\n"
            f"  batched : {act!r}"
        )


def test_multicomponent_parts_are_prefetched_not_re_run(monkeypatch):
    """A `;` name never reaches OPSIN whole, so the prefetch must expand it.

    If it did not, `dicesium;carbonate` would still cost two JVM launches at
    resolve time and the batching would be silently ineffective for the 788
    `;` names in the corpus.
    """
    assert list(_opsin_atoms("dicesium;carbonate")) == ["dicesium", "carbonate"]
    # a trailing `;` leaves ONE part — `_try_opsin` then sends the whole string
    assert list(_opsin_atoms("benzene;")) == ["benzene;"]

    names = ["dicesium;carbonate", "benzene;toluene"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prefetch_opsin(names)

        calls = []
        real = its._opsin_call
        monkeypatch.setattr(
            its, "_opsin_call",
            lambda n, **k: (calls.append(n), real(n, **k))[1],
        )
        joined, _e1 = _try_opsin("benzene;toluene")
        # all-or-nothing: `dicesium` alone is not a name OPSIN accepts, so the
        # whole multi-component name fails rather than recording "carbonate"
        partial, e2 = _try_opsin("dicesium;carbonate")
    assert calls == [], f"prefetched parts still shelled out: {calls}"
    assert joined == "C1=CC=CC=C1.CC1=CC=CC=C1", joined
    assert partial is None and "multi-component" in e2, (partial, e2)


def test_discarded_batch_leaves_the_memo_empty(monkeypatch):
    """A batch that fails its structural checks must poison nothing.

    Truncating stdout by one line is exactly the shape of the failure that
    would otherwise shift every subsequent name onto its neighbour's answer.
    """
    real = its._opsin_call

    def _truncating(names, **kw):
        out, msgs = real(names, **kw)
        if isinstance(out, list):
            out = out[:-1]
        return out, msgs

    monkeypatch.setattr(its, "_opsin_call", _truncating)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _opsin_batch(["benzene", "toluene"], relaxed=False) is None
        assert prefetch_opsin(["benzene", "toluene"]) == 0
    assert its._OPSIN_MEMO == {}

    # and the caller still gets the right answer, one JVM at a time
    monkeypatch.setattr(its, "_opsin_call", real)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _try_opsin("benzene")[0] == "C1=CC=CC=C1"


def test_batch_discarded_when_a_sentinel_parses(monkeypatch):
    """The sentinels are the whole attribution mechanism. If one ever came
    back with a structure, the spans would be wrong and the batch is refused.
    """
    real = its._opsin_call

    def _sentinel_parses(names, **kw):
        out, msgs = real(names, **kw)
        if isinstance(out, list) and out:
            out = ["C1=CC=CC=C1"] + out[1:]
        return out, msgs

    monkeypatch.setattr(its, "_opsin_call", _sentinel_parses)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _opsin_batch(["benzene"], relaxed=False) is None
    assert its._OPSIN_MEMO == {}


def test_batch_discarded_when_stderr_escapes_its_span(monkeypatch):
    """An OPSIN error emitted after the last sentinel belongs to no name.
    Attributing it to one would be a guess, so the batch is refused instead.
    """
    real = its._opsin_call

    def _stray_stderr(names, **kw):
        out, msgs = real(names, **kw)
        return out, list(msgs) + [
            _opsin_format_warning(["stray line attributable to nothing"]),
        ]

    monkeypatch.setattr(its, "_opsin_call", _stray_stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _opsin_batch(["benzene", "notachemicalnameatall"],
                            relaxed=False) is None


def test_warning_formatting_round_trips_py2opsin_exactly():
    """`_opsin_format_warning` must rebuild py2opsin's message character for
    character, including the multi-line case.

    Not reachable from the corpus: prefetching all 16,127 distinct names
    produced 901 warnings and every one of them was a SINGLE stderr line,
    where the separator and a plain newline are indistinguishable. A mutation
    that replaced the separator with "\\n" therefore passed the identity tests
    above. OPSIN emitting two lines for one name is not something this
    codebase controls, so the format is pinned directly against py2opsin.py:146
    instead of against whatever the corpus happens to contain.
    """
    for lines in (["one line"],
                  ["first line", "second line"],
                  ["a", "b", "c"]):
        stderr = "\n".join(lines) + "\n"
        # verbatim from py2opsin.py:146-150
        py2opsin_message = (
            "OPSIN raised the following error(s) while parsing:"
            "\n > " + stderr.replace("\n", "\n > ", stderr.count("\n") - 1)
        )
        assert _opsin_format_warning(lines) == py2opsin_message, lines
        # and the split back out is its inverse
        assert its._opsin_stderr_lines([py2opsin_message]) == lines

    assert _opsin_format_warning([]) == ""


def test_sentinels_are_unparsable_under_every_flag():
    """The separator must fail OPSIN even with its permissive flags on —
    a sentinel that parsed would break the partition."""
    tokens = [_opsin_sentinel(i) for i in range(3)]
    for relaxed in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clear_opsin_memo()
            for tok in tokens:
                result, msg = _opsin_raw(tok, relaxed=relaxed)
                assert result == "", (tok, relaxed, result)
                assert tok in msg, (tok, relaxed, msg)


def test_prefetch_costs_one_jvm_launch_per_chunk(monkeypatch):
    """The point of the exercise: N names, one `java -jar`, not N."""
    names = [n for n in _sample() if ";" not in n][:40]
    assert len(names) >= 20

    launches = []
    real = its._opsin_call
    monkeypatch.setattr(
        its, "_opsin_call",
        lambda n, **k: (launches.append(n), real(n, **k))[1],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prefetch_opsin(names)
        for n in names:
            _try_opsin(n)

    assert len(launches) == 1, (
        f"{len(names)} names took {len(launches)} JVM launches"
    )
    assert isinstance(launches[0], list)


def test_batch_chunking_covers_every_name(monkeypatch):
    """More names than a chunk holds: still one launch per chunk, and no name
    is dropped at a boundary."""
    monkeypatch.setattr(its, "_OPSIN_BATCH_CHUNK", 5)
    names = [n for n in _sample() if ";" not in n][:17]

    launches = []
    real = its._opsin_call
    monkeypatch.setattr(
        its, "_opsin_call",
        lambda n, **k: (launches.append(n), real(n, **k))[1],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_new = prefetch_opsin(names)

    assert len(launches) == 4                       # ceil(17 / 5)
    assert n_new == len(set(names))
    for n in names:
        assert (n, False) in its._OPSIN_MEMO
