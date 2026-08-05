"""Regression tests for name-column selection in `routes/google_tables.py`.

US11292791's example tables were extracted with an NMR readout as the
`iupac_name` for 147 of the compounds stored in its `example_index.json`.
The mechanism was in `parse_compound_table` and was source-independent:

    elif 'name' in h and 'structure' not in h:   # header disqualified
        name_col = i
    ...
    if name_col is None and ex_col is not None:
        name_col = len(header_cells) - 1          # falls to the LAST column

MinerU reads that patent's header row as

    ['example number', 'structure and compound name', '1h nmr, lcms']

so the literal header "Structure and Compound Name" — the most common name
header in the corpus — was vetoed by the word "structure", `name_col` fell
through to the last column, and the last column is the NMR column.

Measured over the 11 corpus patents that have MinerU markdown, before → after:

    patent        compounds     named   spectral-names   OPSIN-parsable
    US11292791    168 → 112   164 → 94       153 → 1           3 → 5
    US10214537    831 → 831  830 → 829         0 → 0       344 → 348
    US10246453     40 → 112   40 → 112         7 → 0          4 → 12
    TOTAL        1130 → 1111 1122 → 1088     160 → 1       377 → 391

No name that OPSIN could parse was lost in any patent.

These tests pin the ORDERING — an explicit name header beats the positional
fallback — and the rejector, because both failures are silent: a wrong name
still produces a row, a plausible-looking `example_index.json`, and paid
downstream cascade calls that can never resolve.
"""
from patentdb.routes.google_tables import _clean_name, parse_compound_table

# The real header row, verbatim from data/patents/US11292791/all_pages/.
US11292791_HEADER = [
    'Example Number Structure and Compound Name',
    'LRMS m/z [M + H]*',
    '1HNMR',
]
# The other layout in the same patent — same words, split across three cells.
US11292791_SPLIT_HEADER = [
    'Example Number',
    'Structure and Compound Name',
    '1H NMR, LCMS',
]

NMR = ('1H-NMR (CDCl3, 300 MHz) δ (ppm): 7.43-7.41 (m,1H),7.22-7.19 (m,3H),'
       '4.81-4.79 (m,1H),3.82 (s,3H)')
NAME = '2-(4-chlorophenyl)-N-methylpyridine-3-carboxamide'


def _table(header, *rows):
    cells = lambda r: ''.join(f'<td>{c}</td>' for c in r)
    body = ''.join(f'<tr>{cells(r)}</tr>' for r in rows)
    return f'<table><tr>{cells(header)}</tr>{body}</table>'


def test_split_name_header_wins_over_the_last_column():
    """"Structure and Compound Name" names the name column, "structure" or not.

    This is the exact shape that produced 153 spectral names in US11292791:
    the name is in column 1 and column 2 is NMR.
    """
    out = parse_compound_table(_table(US11292791_SPLIT_HEADER, ['101', NAME, NMR]))
    assert len(out) == 1
    assert out[0]['example_number'] == 'Example 101'
    assert out[0]['iupac_name'] == NAME


def test_merged_name_header_does_not_fall_through_to_the_nmr_column():
    """The merged header is BOTH the ex column and the name column.

    The ex-branch of the header loop consumes it (it contains 'ex' and 'no'),
    which is why name detection cannot live in the same elif chain: it has to
    be tried separately or the positional fallback gets the row.
    """
    out = parse_compound_table(
        _table(US11292791_HEADER, [f'101 {NAME}', '247.1', NMR]))
    assert len(out) == 1
    assert out[0]['iupac_name'] == NAME       # not the NMR cell
    assert out[0]['mh_plus'] == 247.1


def test_an_nmr_cell_is_never_stored_as_a_name():
    """The rejector, independent of which column it arrived in.

    Column choice alone is not enough — MinerU spills NMR text into the name
    column of the very tables it fails to align — so a cell that reads as
    characterisation data must be dropped wherever it came from.
    """
    out = parse_compound_table(
        _table(US11292791_SPLIT_HEADER, ['5', NMR, NMR]))
    assert out == [] or out[0]['iupac_name'] is None


def test_last_column_fallback_skips_a_characterisation_column():
    """No header names a name column, so the fallback runs — but not onto NMR.

    The old rule ("Name is always the LAST text column") took the last column
    unconditionally. Here the name sits in column 1 under a blank header and
    the NMR column is last.
    """
    out = parse_compound_table(
        _table(['Example No.', '', '1H NMR'], ['7', NAME, NMR]))
    assert len(out) == 1
    assert out[0]['iupac_name'] == NAME


def test_fallback_skips_a_drawn_structure_column():
    """A column headed "Structure" with no "name" holds a picture.

    US10273259's tables are `Ex.# | Structure | LCMS m/z | HPLC tr | HPLC
    method` and carry no name at all; reading its structure column produced
    67 rows of `Homochiral from peak 2` and LaTeX-wrapped `(M + H)+`.
    """
    out = parse_compound_table(_table(
        ['Ex.#', 'Structure', 'LCMS m/z observed', 'HPLC tr (min)', 'HPLC method'],
        ['41', 'Homochiral from peak 2', '576.2 (M + H)+', '1.08', 'B']))
    assert all(r['iupac_name'] is None for r in out)


def test_a_body_row_segmented_finer_than_its_header_still_finds_the_name():
    """US10246453: 2 header cells over 4 body cells.

    `Example No.Name | MS (ES+) m/z 1H NMR` merges the ex and name headers, so
    the name column resolves to the ex column — but the body row is split, and
    the name is one cell to the right. Reading only the merged column dropped
    Examples 325-327 with their names intact.
    """
    out = parse_compound_table(_table(
        ['Example No.Name', 'MS (ES+) m/z 1H NMR'],
        ['326', '(S)-3-chloro-4-((1-(2,6-difluorobenzyl)pyrrolidin-3-yl)(methyl)amino)',
         '500.0 (M+ 1)', NMR]))
    assert len(out) == 1
    assert out[0]['iupac_name'].startswith('(S)-3-chloro-4-')


def test_a_leading_example_number_is_stripped_but_a_locant_is_not():
    """`79 (R)-4-...` is a number then a name; `3-chloro-...` opens on a 3.

    Whitespace after the number is what tells them apart. Eating the locant
    would corrupt every name in a merged cell whose example number OCR lost.
    """
    assert _clean_name('79 (R)-4-(1-benzylpyrrolidin-3-ylamino)-3-chloro',
                       strip_example_number=True).startswith('(R)-4-')
    assert _clean_name('3-chloro-N-(thiazol-2-yl)benzenesulfonamide',
                       strip_example_number=True).startswith('3-chloro')


def test_prose_and_latex_cells_are_not_names():
    """Cells that pass the spectra check and are still not names.

    `\\boldsymbol` supplies the lowercase run that `looks_like_spectra` has no
    opinion on. Over the 11 MinerU patents, 74 extracted names carried neither
    a bracket nor a locant and exactly one of those parsed in OPSIN —
    `trifluoroacetate`, a counter-ion.
    """
    assert _clean_name('Homochiral from peak 2') is None
    assert _clean_name('57 -structure and name are in the Example-') is None
    assert _clean_name(r'576.2 $( \boldsymbol{\mathrm{M}} + \mathrm{H} )^{+}$') is None


def test_a_real_name_survives_every_guard():
    """The guards must not be so eager that they take the thing we came for."""
    for name in (
        '1-(3-(3-(4-amino-5-(1-(tetrahydro-2H-pyran-4-yl)-1H-pyrazol-5-yl)'
        'pyrrolo[2,1-f][1,2,4]triazin-7-yl)phenyl)oxetan-3-yl)piperidin-4-one',
        '(S)-4-(1-benzylpyrrolidin-3-yloxy)-3-chloro-N-(1,2,4-thiadiazol-5-yl)'
        'benzenesulfonamide',
        '3-(4-methyloxazol-2-yl)propanal',
    ):
        assert _clean_name(name) == name


def test_a_name_followed_by_its_own_ms_keeps_the_name():
    """terminate_name cuts at the data, it does not discard the row."""
    got = _clean_name('2-(4-chlorophenyl)-N-methylpyridine-3-carboxamide '
                      'MS (ESI) m/z 435.2')
    assert got == '2-(4-chlorophenyl)-N-methylpyridine-3-carboxamide'
