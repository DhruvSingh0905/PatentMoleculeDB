"""Potency-bin legends: `+`, `++`, `A`-`E` → explicit numeric ranges.

Many patents do not publish a number per compound. They publish a symbol and,
somewhere nearby, a key that defines what the symbol means:

    *Key: ++++: IC50 ≥ 1 uM   +++: 1 uM > IC50 ≥ 0.1 uM   ...

    an IC50 value of greater than or equal to 0.001 μM and less than or equal
    to 0.01 μM is marked "++++"; a value greater than 0.01 μM and less than or
    equal to 0.1 μM is marked "+++"; ...

Seven forms appear in this corpus and each one is somebody's house style. The
separator is the only thing that varies, and it varies completely:

    A: IC50 < 3 nM                          a colon              US10172859
    + (greater than 10 microMolar)          a parenthesis        US9682141
    A ≦ 10 nM;                              nothing at all       US9656988
    *** is less than 100 nM                 the verb `is`        US10030020
    ... are labelled as "+++"               range stated first   US11286268
    "A" represents an IC50 of less than …   a quoted symbol      US20240166635
    <1.00 nM=A                              symbol stated last   US9688680

They are all read here, because a legend that is not read costs every record
under it, and 9 of these 15 patents publish nothing but grades.

**A key is applied to thousands of rows at once, so a wrong key is silent.**
That is what the rest of this module is defending against, and there are four
distinct ways to get one:

  * THE SAME SYMBOL, DIFFERENT SCALES. US11566007 uses `++++` for ≥1 μM in one
    table and 1-10 μM in the next; US11292791 uses it for 0.001-0.01 μM. Keys
    are resolved per table, never shared, and a symbol with no key in scope
    yields no range at all.
  * THE SAME SYMBOL, DIFFERENT COLUMNS. US10172859 defines `A`-`D` three times
    on one page, once per assay, so `B` is 3-7 nM, 0.5-5 μM or 15-25 μM
    depending only on which column it sits in. `parse_sectioned_key` keeps the
    scales apart and `section_for_column` matches one to a column by name.
  * THE SAME SYMBOL, DIFFERENT QUANTITIES. US10030020 grades potency in nM with
    `*` and inhibition in percent with `#`, in one sentence. `compatible`
    refuses a bin whose dimension the column contradicts.
  * THE LEGEND IS A TABLE, NOT A SENTENCE. US9221791 prints `value | rating`
    rows, and flattening them hands every grade the NEXT grade's number.
    `_VALUE_BEFORE` refuses a symbol that merely follows a value.

Every one of those was live in the corpus, and none of them showed up in the
output as anything other than a plausible number.

This turns a symbol into an honest interval, which is real, usable data — a
compound known to be 0.1-1 μM is a meaningful record for a screening cascade.
It will never match a point value in BindingDB, and should not be expected to:
that is a limitation of exact-match scoring, not of the extraction.
"""
from __future__ import annotations

import re

# Units accepted inside a key definition, normalised to the project's spelling.
#
# A bin scale is NOT always a concentration. The same patent grades potency in
# nM and inhibition in percent, and a legend that omits the unit on some lines
# used to inherit whichever unit the rest of the document showed. US10030020
# defines `### is ≥75%` for six columns headed `% Inh 1 μM (mouse)`, and `%`
# was not a unit here — so those bins parsed unitless, the backfill below
# stamped `nM` on them, and 1,243 records claimed a nanomolar potency the
# patent never measured. A dimension we cannot name must stay unnamed.
_UNIT = (r"(?:nM|pM|mM|[μuµ]M|micromolar|nanomolar|millimolar|picomolar"
         r"|%|mg/kg|[μuµ]g/m[lL]|[-\s]?fold)")
_UNIT_CANON = {
    "um": "uM", "µm": "uM", "μm": "uM", "nm": "nM", "mm": "mM", "pm": "pM",
    "micromolar": "uM", "nanomolar": "nM", "millimolar": "mM", "picomolar": "pM",
    "%": "%", "mg/kg": "mg/kg", "fold": "fold", "-fold": "fold",
    "μg/ml": "ug/mL", "µg/ml": "ug/mL", "ug/ml": "ug/mL",
}

# The symbol a bin is written with: a run of '+', '*' or '#', or a letter grade.
#
# `*` and `#` were absent, and their absence is invisible rather than noisy:
# the grade is still READ off the data table, so records appear, carry a
# symbol, and carry no range. US10030020 and US9133148 grade with `*`/`**`/
# `***` for potency and `#`/`##`/`###` for percent inhibition; US12351648 uses
# `***`/`****`. 2,598 records between them.
#
# The letter grades are UPPERCASE-only — `(?-i:...)` holds even though every
# pattern here compiles with re.I for the English verbs around it. Case-folding
# the grade turns the last letter of an ordinary word into one: the phrase
# `the following designations are used: <1.00 nM=A` defines `A`, and also
# matched `d` — the tail of "used", followed by a colon, followed by a range.
# A grade that exists only inside another word is always a false positive.
# U+2212 MINUS is a grade too. A two-level scale writes `+` and `−`, and
# leaving `−` out cost twice over: its own rows were dropped, and — worse —
# `_NEXT_DEF` could not see where the next grade's clause began, so the
# prose body for `+` ran straight through `− indicates ≥10 μm`, read BOTH
# bounds, and overwrote `≤10 μM` with the point interval `10..10`.
# US10953012 shipped 270 records saying a compound is exactly 10 μM.
# Only U+2212, never the ASCII hyphen: that is a range separator here.
_SYMBOL = r"(?:\++|\*+|#+|\u2212|(?-i:[A-E]))"

# Patents quote the symbol as often as not: `"A" represents ...`, `is marked
# "+++"`. Straight and curly, single and double.
_QUOTE = r"[\"“”'‘’`]"
# Capturing and non-capturing forms of "a symbol, optionally quoted".
#
# The lookbehind is the second half of the `used:` guard above: a symbol may
# not be glued to the end of a WORD. It is written before the optional quote so
# that `“A”` still passes — the character before `A` is the quote, not a letter.
#
# Digits are deliberately NOT excluded. US11485738 runs its sentence into its
# key with no space — `the results are shown in Table 15A=<250 nM` — and
# rejecting a grade after a digit drops that key and the 117 records under it.
_SYM_Q = rf"(?<![A-Za-z]){_QUOTE}?(?P<sym>{_SYMBOL}){_QUOTE}?"
_SYM_QN = rf"(?<![A-Za-z]){_QUOTE}?(?:{_SYMBOL}){_QUOTE}?"

# A NUMBER MAY BE GROUPED. `10,000` was read as `10` and `1,500` as `1`, so a
# bin defined as `IC50 ≥ 10,000 nM` came out as `≥10 nM` — a 1000x error that
# ships a plausible number rather than a blank, which is the worse failure. The
# unit went with it: the match ended at the comma, so `nM` never attached and
# the bound recorded no unit at all.
#
# The grouped form is tried FIRST and requires exactly three digits after each
# comma, so a list — `A, B` or `10, 20` — cannot be read as one number.
#
# Same family as `mass_gate._NUMBER` taking `69` out of `0.69`: a numeric
# pattern that can stop mid-token will, and the result stays plausible.
#
# WRAPPED. This string is interpolated into larger patterns, several of them
# bare rather than inside a group, so a top-level `|` here would split the
# whole surrounding expression instead of just the number.
_NUM = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"


def _f(s: str) -> float:
    """A matched `_NUM` as a float. THE ONE PLACE that knows the separator.

    Nine call sites converted these groups and exactly one of them stripped the
    comma, which is why the defect survived: the fix existed and was not where
    the failing bound was read.
    """
    return float(str(s).replace(",", ""))

# The metric a bin definition may name. ONE constant, because the cheap
# pre-filter and the full parser must accept the same set: `_KEY_HINT` listed
# only IC50/EC50 while `_KEY_COMPACT` also accepted Ki/Kd, so `A: Ki > 25 uM`
# was rejected before the parser that could read it ever ran. US10172859 states
# its whole hERG scale that way — 431 records, and worse than a plain miss,
# because `parse_sectioned_key` then reads the unparsed definition row as the
# next section's HEADING and shifts every section boundary after it.
_METRIC = r"(?:IC\s*50|EC\s*50|Ki|Kd|value)"

# What separates a symbol from its definition. `:` and `=` are the common
# forms; US10376513 writes "+ refers to ≤10 nM" and states the key for all 348
# of its compounds that way, so the verb forms are equally a separator.
#
# The last three alternatives are the ones that were missing, and between them
# they account for more unresolved records than every other cause in this file
# combined. A legend is not required to punctuate itself:
#
#   `+ (greater than 10 microMolar)`   US9682141, US11547697   4,619 records
#   `A ≦ 10 nM;`                       US9656988               1,351 records
#   `*** is less than 100 nM;`         US10030020, US9133148   1,568 records
#
# None of these carry a colon, an equals sign or one of the verbs, so the cheap
# pre-filter rejected them and the parser that could read every one was never
# reached. The bare-comparison form is a LOOKAHEAD: it consumes nothing, so the
# operator it detects is still there for the bound parser to read.
_DEFINES = (
    r"(?:"
    r"\s*[:=]\s*"
    r"|\s+(?:refers?\s+to|means|indicates?|represents?|denotes?|is|are|provided?|gave|gives?|showed?|shows|had|has|exhibited?|exhibits)\s+"
    r"|\s*\(\s*"
    r"|\s*(?=[<>≤≥≦≧⩽⩾])"
    # A SPACE IS A SEPARATOR WHEN THE METRIC FOLLOWS IT. US12351648 writes
    # `+ IC 50 value in the range of 5 μM to 50 μM` — no colon, no verb, no
    # bracket, 865 records. The metric must actually be there, so a bare
    # `A 10 nM` is still refused: that shape is a value sitting beside a symbol,
    # which `_VALUE_BEFORE` exists to reject.
    r"|\s+(?=" + _METRIC + r")"
    r")"
)

# How a patent says "this range is written with this symbol" when it states the
# range FIRST. `is marked` was the only form read; each of the others appears in
# this corpus and reads identically to a chemist.
_MARKED = (
    r"(?:is|are)\s+(?:marked|labell?ed|designated|indicated|denoted|shown|"
    r"reported|represented)(?:\s+(?:as|by|with))?"
)

# Form 1 — compact key:  "++++: IC50 ≥ 1 uM"  /  "+++: 1 uM > IC50 ≥ 0.1 uM"
#                        "++ refers to >10 nM to 50 nM"
# The trailing upper bound matters: without it "++ refers to >10 nM to 50 nM"
# parses as lo=10 with NO upper bound, turning a 10-50 nM bin into "anything
# above 10 nM". That is the same class of silent widening that a shared bin key
# causes, so the tail is captured rather than tolerated. US10626094 writes the
# same range as "A: IC 50 >200 nM−<800 nM" with a U+2212 MINUS between the
# bounds, and read every one of its bins as unbounded above until this landed.
#
# The trailing bound is OPTIONAL, and that is not cosmetic. `D: 10 uM >= Ki`
# states its only bound BEFORE the metric and puts nothing after it. While a
# number was required there, the leading group could not be kept — the engine
# backtracked, gave up `hi=10`, and re-matched the same `10` as `lo`, turning
# `Ki <= 10 uM` into `Ki >= 10 uM`. An inverted bin, silently, on every row
# written in that order. Both bounds optional; the caller drops a match that
# yields neither.
#
# The LEADING group is a third place a bound can sit: before the symbol itself.
# US9656988 brackets each grade from both sides — `10 nM < B ≦ 100 nM` — and
# reading left-to-right from `B` sees only the upper bound, so every interior
# grade came out unbounded below. `A ≦ 10 nM` and `E > 10 μM`, the two ends of
# the same scale, have nothing on the left and were already right, which is why
# the scale looked plausible while three of its five bins were wrong.
# WORDS BETWEEN THE SEPARATOR AND THE BOUND. A patent does not stop writing
# English where the pattern stops reading it:
#
#     +  IC 50 value in the range of 5 μM to 50 μM     US12351648    865 records
#     B  provided an IC 50 10-100 nM                   US9987276     461
#     ++ indicates an IC 50 of 100 to 500 nM           US11053246      3
#
# Every one of those already had a form the parser reads — `+: IC 50 5 μM to
# 50 μM` parses, `B provided IC 50 10-100 nM` parses, `++ indicates 100 to
# 500 nM` parses. What defeated them was an article, a preposition or a second
# noun: exactly ONE optional metric token was allowed there and nothing else.
# So this was never a missing range form. It was a gap between the separator
# and the bound, and only filler could sit in it.
#
# Letters and spaces only, so it can never cross a number and claim a bound
# that belongs to another clause. `_NEXT_DEF`'s guard is reused for the other
# direction: a verb separator is made of letters too, so without it the filler
# for `A is potent` would run through `B is 10 nM` and hand A the 10.
# It must also not eat the METRIC'S OWN LETTERS. Without the second lookahead
# the filler swallows `IC ` and the bound group then reads the `50` of `IC 50`
# as a number: every grade in US9987276 came back as `50..inf nM`, which is
# worse than the blank it replaced because it is a plausible bin.
# AND IT MUST NOT EAT AN ENGLISH COMPARISON. `less than` and `greater than`
# are letters and spaces like any other filler, so an unguarded run swallowed
# the operator and left a bare number: `*** is less than 100 nM` came back as
# lo=100 — the bin inverted, silently. Forms 2 and 3 read those phrasings; the
# filler's job is only to skip words that mean nothing to a bound.
_CMP_WORD = (r"(?:greater|less|more|least|most|equal|between|from|above|below"
             r"|higher|lower|under|over|exceed)")
_FILL = (rf"(?:(?!{_SYM_QN}{_DEFINES})(?!{_METRIC})(?!{_CMP_WORD})"
         rf"[A-Za-z\s]){{0,24}}")

_KEY_COMPACT = re.compile(
    rf"(?:(?P<pre>{_NUM})\s*(?P<preu>{_UNIT})?\s*(?P<preop>>=|>|≥|≧|⩾|≤|≦|⩽|<=|<)\s*)?"
    rf"{_SYM_Q}(?P<sep>{_DEFINES})"
    rf"(?:(?P<hi>{_NUM})\s*(?P<hiu>{_UNIT})?\s*(?P<hiop>>=|>|≥|≧|⩾|≤|≦|⩽|<=|<)\s*)?"
    # Filler, then up to two metric words each followed by more filler.
    # US12351648 writes TWO in a row — `IC 50 value in the range of` — and one
    # optional `_METRIC` leaves the second stranded in front of the bound.
    rf"{_FILL}(?:{_METRIC}{_FILL}){{0,2}}"
    rf"(?:(?P<loop>>=|>|≥|≧|⩾|≤|≦|⩽|<=|<)?\s*(?P<lo>{_NUM})\s*(?P<lou>{_UNIT})?"
    # The upper end may restate its direction IN WORDS. US10189840 writes
    # `an IC 50 value in the range of 1 μM to less than 10 μM`; reading only
    # the symbols here stopped the match at `to` and left grade B holding one
    # bound instead of two — a half-open bin where the document states a
    # closed one.
    rf"(?:\s*(?:to|[-–—−])\s*"
    rf"(?:[<>≤≥≦≧⩽⩾]|less\s+than|greater\s+than|under|over)?\s*"
    rf"(?P<hi2>{_NUM})\s*(?P<hi2u>{_UNIT})?)?)?",
    re.I)

# Form 2 — prose:  "a value greater than 0.01 μM and less than or equal to
#                   0.1 μM is marked "+++""
# NOTE the character class is `[^;]`, not `[^.;]`. Excluding the period looks
# safer but silently truncates every clause containing a decimal — the body of
# "greater than or equal to 0.001 μM and less than or equal to 0.01 μM" cannot
# be spanned without crossing a '.', so the lower bound was being dropped and
# every bin came out unbounded-below. Clauses in these legends are separated by
# semicolons, which is the boundary that actually holds.
#
# `is marked` was one phrasing of many. US11286268 writes "IC50 values of less
# than 0.05 μM are labelled as "+++"" — the same sentence, the same meaning,
# and it read as prose because the verb did not match. See `_MARKED`.
_KEY_PROSE = re.compile(
    rf"(?P<body>(?:greater|less|more|from)[^;]{{0,200}}?{_NUM}\s*{_UNIT}[^;]{{0,200}}?)"
    rf"{_MARKED}\s*{_SYM_Q}",
    re.I)

# Form 3 — the symbol is defined by a prose clause: "A = IC50 of less than
# 10 nM". Bounded by `;` for the same reason `_KEY_PROSE` is: a period cannot
# be a clause boundary when every value contains one.
# The body may NOT cross into the next symbol's definition. Without the
# lookahead, `+ refers to ≤10 nM ++ refers to >10 nM to 50 nM` gives `+` a body
# running through `++`'s clause, and `+` comes out as 10..10 — the exact bleed
# that made `following` unusable for keys, reappearing inside a single string.
_NEXT_DEF = rf"(?:(?!{_SYM_QN}{_DEFINES})[^;])"
_KEY_WORDY = re.compile(
    rf"{_SYM_Q}(?P<sep>{_DEFINES})(?P<body>{_NEXT_DEF}{{0,60}}?"
    rf"(?:greater\s+than|less\s+than|at\s+least|at\s+most|[<>≤≥≦≧⩽⩾]|or\s+(?:greater|less|more|higher|lower|below|above))"
    rf"{_NEXT_DEF}{{0,240}})",
    re.I)

# Form 4 — the range is stated FIRST and the symbol assigned to it:
#   "<1.00 nM=A; 1.01-10.0 nM=B; 10.01-100.0 nM=C; >100 nM=D"
# Forms 1-3 all read left-to-right from the symbol, so every one of them reads
# this backwards: `=A` has nothing after it to bound, and the number before is
# never reached. US9688680 states both of its scales this way and resolved none
# of its 950 records.
_RANGE_EXPR = re.compile(
    rf"(?:^|[;,:])\s*"
    rf"(?P<op>[<>≤≥≦≧⩽⩾]|greater\s+than|less\s+than|at\s+least|at\s+most)?\s*"
    rf"(?P<a>{_NUM})\s*(?P<au>{_UNIT})?"
    rf"(?:\s*(?:to|[-–—−])\s*(?P<b>{_NUM})\s*(?P<bu>{_UNIT})?)?"
    rf"\s*=\s*{_SYM_Q}", re.I)

# A value sitting immediately BEFORE a symbol, with no comparison between them.
# That is the `value | symbol` table shape — `80-100 A`, `60-79 B`, `40-59 C`,
# `<40 D` — and once flattened to prose it reads left-to-right as `C` followed
# by `<40`, which is the value from D's ROW. US9221791 rates fungicides that
# way and every grade came out holding the next grade's number.
#
# The trailing `[<>...]?` in the middle matters: `10 nM < B ≦ 100 nM` also puts
# a value before the symbol, but with an operator pointing AT the symbol, which
# is a real lower bound. This pattern only fires when nothing points anywhere.
# Form 5 — the whole INTERVAL is stated first and the symbol assigned after it:
#     `1000 nM < IC50 <= 10000 nM: +++`
# Form 4 reads `<value> = <symbol>` and stops at the first number, so a bound on
# each side leaves it nothing to bind. US11229631 states its whole scale this
# way in a footer under the table — 308 records.
_KEY_SPAN_FIRST = re.compile(
    rf"({_NUM})\s*({_UNIT})?\s*[<>≤≥≦≧⩽⩾]\s*[^;:]{{0,40}}?"
    rf"[<>≤≥≦≧⩽⩾]\s*({_NUM})\s*({_UNIT})?\s*[:=]\s*{_SYM_Q}", re.I)

# Form 6 — the symbol, then an interval whose middle names an ARBITRARY metric:
#     `A 0 < PI3K Delta Activity < 50 nM`
# `_METRIC` lists the four metric names a bin definition normally uses, and this
# patent puts the assay's own name there instead. Bounded to 40 characters so it
# cannot span into the next grade's clause.
_KEY_SPAN_AFTER = re.compile(
    rf"{_SYM_Q}\s+({_NUM})\s*({_UNIT})?\s*[<>≤≥≦≧⩽⩾]\s*[A-Za-z][^;:<>]{{0,40}}?"
    rf"[<>≤≥≦≧⩽⩾]\s*({_NUM})\s*({_UNIT})?", re.I)

_VALUE_BEFORE = re.compile(
    rf"(?:^|[\s;,])[<>≤≥≦≧⩽⩾]?\s*{_NUM}(?:\s*(?:to|[-–—−])\s*{_NUM})?"
    rf"\s*(?:{_UNIT})?\s*$")

# A bound whose direction is stated after it: `1 μM or greater`, `10 nM or
# less`. US20240166635 defines its top grade that way and nothing else in
# this file reads right-to-left.
_GT_AFTER = re.compile(rf"({_NUM})\s*({_UNIT})?\s+or\s+(?:greater|more|higher|above)", re.I)
_LT_AFTER = re.compile(rf"({_NUM})\s*({_UNIT})?\s+or\s+(?:less|lower|below|fewer)", re.I)
# `from 1 to 0.05 μM` — a range written high-to-low. Read in order it inverts.
_SPAN = re.compile(rf"from\s+({_NUM})\s*({_UNIT})?\s+to\s+({_NUM})\s*({_UNIT})?", re.I)
_GT = re.compile(rf"(?:greater than or equal to|at least|≥|≧|⩾|>=)\s*({_NUM})\s*({_UNIT})?", re.I)
_GT_STRICT = re.compile(rf"(?:greater than|>)\s*({_NUM})\s*({_UNIT})?", re.I)
_LT = re.compile(rf"(?:less than or equal to|at most|≤|≦|⩽|<=)\s*({_NUM})\s*({_UNIT})?", re.I)
_LT_STRICT = re.compile(rf"(?:less than|<)\s*({_NUM})\s*({_UNIT})?", re.I)


def _canon_unit(u: str | None) -> str | None:
    if not u:
        return None
    return _UNIT_CANON.get(u.strip().lower(), u.strip())


class BinRange:
    """A half-open-ish numeric interval for one bin symbol.

    `lo`/`hi` are in `unit`. Either bound may be None (unbounded on that side).
    """

    __slots__ = ("symbol", "lo", "hi", "unit")

    def __init__(self, symbol: str, lo: float | None, hi: float | None, unit: str | None):
        self.symbol, self.lo, self.hi, self.unit = symbol, lo, hi, unit

    def __repr__(self) -> str:
        lo = "-inf" if self.lo is None else f"{self.lo:g}"
        hi = "inf" if self.hi is None else f"{self.hi:g}"
        return f"BinRange({self.symbol!r}, {lo}..{hi} {self.unit or '?'})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, BinRange) and self.symbol == other.symbol
                and self.lo == other.lo and self.hi == other.hi
                and self.unit == other.unit)


# Relative size of each unit, for reconciling a clause that states its two
# bounds in DIFFERENT units.
_SCALE = {"pM": 1e-3, "nM": 1.0, "uM": 1e3, "mM": 1e6}


def _reconcile(lo, lo_u, hi, hi_u):
    """Put a bound pair on ONE scale: (lo, hi, unit).

    A clause is free to state its two bounds in different units, and the two
    readings differ by whatever the ratio between them is — 1,000x for the
    nM/uM pair that actually occurs. Converting to the finer of the two keeps
    integers integral and never rounds a bound toward zero.
    """
    if (lo is not None and hi is not None and lo_u and hi_u and lo_u != hi_u
            and lo_u in _SCALE and hi_u in _SCALE):
        if _SCALE[lo_u] <= _SCALE[hi_u]:
            hi, hi_u = hi * _SCALE[hi_u] / _SCALE[lo_u], lo_u
        else:
            lo, lo_u = lo * _SCALE[lo_u] / _SCALE[hi_u], hi_u
    return lo, hi, lo_u or hi_u


def _prose_bounds(body: str) -> tuple[float | None, float | None, str | None]:
    """Pull (lo, hi, unit) out of one prose clause.

    Each bound carries its OWN unit, because patents mix them inside a single
    clause: US11752149 defines `C = IC50 less than 1 μM (1,000 nM) but greater
    than or equal to 100 nM`. Taking the first unit seen and applying it to
    both returned lo=100, hi=1 — an interval inverted and 1,000x wrong, which
    then silently labels every `C` compound in the patent.
    """
    lo = hi = None
    lo_u = hi_u = None
    # `from 1 to 0.05 μM` — a span written high-to-low. Read left to right it
    # inverts, so the ends are sorted rather than assigned by position.
    m = _SPAN.search(body)
    if m:
        a, b = _f(m.group(1)), _f(m.group(3))
        u = _canon_unit(m.group(4) or m.group(2))
        lo, hi = min(a, b), max(a, b)
        lo_u = hi_u = u
        return _reconcile(lo, lo_u, hi, hi_u)
    # A bound whose direction is stated after the value.
    for pat, is_lo in ((_GT_AFTER, True), (_LT_AFTER, False)):
        m = pat.search(body)
        if not m:
            continue
        v, u = _f(m.group(1)), _canon_unit(m.group(2))
        if is_lo:
            lo, lo_u = v, u
        else:
            hi, hi_u = v, u
    for pat, is_lo in ((_GT, True), (_GT_STRICT, True), (_LT, False), (_LT_STRICT, False)):
        m = pat.search(body)
        if not m:
            continue
        val = _f(m.group(1))
        u = _canon_unit(m.group(2))
        if is_lo and lo is None:
            lo, lo_u = val, u
        elif not is_lo and hi is None:
            hi, hi_u = val, u
    return _reconcile(lo, lo_u, hi, hi_u)


def _adopt(here: dict, conflicts: set, br: BinRange) -> None:
    """Take a symbol's definition, or mark it as contested.

    A legend defines each grade ONCE. When the same pass reads a second,
    different definition of the same symbol, the text in scope is not one
    legend — it is two, and nothing here says which governs a given column.

    That case is common and it is the expensive one. US9688680 states two
    scales in one paragraph, `for D816V activity` and `For wild-type Kit
    activity`, and `A` is `<1.00 nM` in the first and `<10 nM` in the second.
    US9133148 grades potency in nM and inhibition in percent with the same
    `***`. Taking the first and applying it to every column is a silent 10x on
    half the records; taking neither costs the records and states the truth.
    """
    # WHY AN IMPOSSIBLE INTERVAL IS NOT REFUSED HERE. US9221791 prints its own
    # scale with a typo — `>1.5-1.5 B` where `B` should read `>0.5-1.5` — so
    # 17 records carry lo == hi == 1.5. Refusing that reading was tried and
    # REVERTED: a later pass then reads the same text as the open-ended `>1.5`,
    # which is wrong in the other direction (a B compound is at most 1.5, not
    # above it) and silently overlaps C and D. `check_impossible_interval`
    # already reports lo == hi, so the defect is visible; the refusal traded a
    # flagged defect for a hidden one, which is the worse of the two.
    prev = here.get(br.symbol)
    if prev is None:
        here[br.symbol] = br
    elif (prev.lo, prev.hi, prev.unit) != (br.lo, br.hi, br.unit):
        conflicts.add(br.symbol)


def _merge(out: dict, here: dict) -> None:
    """Fold one pass's readings in. An earlier pass's reading always wins."""
    for sym, br in here.items():
        out.setdefault(sym, br)


def _alphabet(symbol: str) -> str:
    """Which family a symbol belongs to. `*` and `#` are different scales."""
    return symbol[0] if symbol[0] in "+*#" else "A"


def parse_bin_key(text: str) -> dict[str, BinRange]:
    """Extract a symbol → range mapping from legend/footnote text.

    Returns {} when no key is present. Never falls back to a default mapping:
    two patents in this corpus assign incompatible ranges to `++++`, so a
    guessed key is worse than none.
    """
    if not text:
        return {}
    out: dict[str, BinRange] = {}
    # Symbols this text defines MORE THAN ONCE, incompatibly. See `_adopt`.
    conflicts: set[str] = set()

    # Form 4 first — `<1.00 nM=A` pins both its bound and its symbol with no
    # room to read either wrongly, and running it ahead of the left-to-right
    # forms means none of them can claim the symbol on a partial reading.
    here: dict[str, BinRange] = {}
    for m in _RANGE_EXPR.finditer(text):
        sym = m.group("sym")
        lo = hi = None
        unit = _canon_unit(m.group("bu") or m.group("au"))
        a = _f(m.group("a"))
        if m.group("b") is not None:
            lo, hi = a, _f(m.group("b"))
        else:
            op = (m.group("op") or "").lower()
            if op in ("<", "≤", "≦", "⩽", "less than", "at most"):
                hi = a
            elif op in (">", "≥", "≧", "⩾", "greater than", "at least"):
                lo = a
            else:
                # A bare `100 nM=B` states no direction. It is a bin edge with
                # no side, and choosing one would be inventing the interval.
                continue
        _adopt(here, conflicts, BinRange(sym, lo, hi, unit))
    _merge(out, here)

    # Forms 5 and 6 — an interval with a bound on EACH side of a metric name.
    # Run before the left-to-right forms for the same reason Form 4 is: both
    # ends are pinned, so there is nothing for a partial reading to claim.
    for pat, sym_first in ((_KEY_SPAN_FIRST, False), (_KEY_SPAN_AFTER, True)):
        here = {}
        for m in pat.finditer(text):
            a, au, b, bu = (m.group(i) for i in (2, 3, 4, 5)) if sym_first \
                else (m.group(1), m.group(2), m.group(3), m.group(4))
            lo, hi = _f(a), _f(b)
            unit = _canon_unit(bu or au)
            if lo > hi:
                lo, hi = hi, lo
            _adopt(here, conflicts, BinRange(m.group("sym"), lo, hi, unit))
        _merge(out, here)

    # Prose form — unambiguous when present.
    here = {}
    for m in _KEY_PROSE.finditer(text):
        lo, hi, unit = _prose_bounds(m.group("body"))
        sym = m.group("sym")
        if lo is not None or hi is not None:
            _adopt(here, conflicts, BinRange(sym, lo, hi, unit))
    _merge(out, here)

    # Compact form: "++++: IC50 ≥ 1 uM", "+++: 1 uM > IC50 ≥ 0.1 uM"
    here = {}
    for m in _KEY_COMPACT.finditer(text):
        sym = m.group("sym")
        # The bare-comparison separator consumes nothing, so a symbol can be
        # bound to a value that is not its own. It is only a separator when
        # nothing else already sits between the symbol and the number — and
        # when the symbol is not itself the tail of a `value symbol` pair.
        if not (m.group("sep") or "").strip() \
                and _VALUE_BEFORE.search(text[:m.start("sym")]):
            continue
        # A grade, a colon and a number is not a key. Every part of Form 1
        # except the symbol is optional, so `A: 4` and `E=20` match it — and a
        # patent's chemistry prose is full of both. 97 such matches in
        # US20230365584A1 alone. Harmless while the harvest text was a few
        # hundred characters of local legend; the preceding window is now
        # 6,000, so junk is one widening away from being adopted as a scale.
        #
        # A real key always carries at least one of: a unit, a comparison, the
        # name of the metric, or a RANGE. The range is the one that had to be
        # learned: US11254686 writes `A=<10 nM  B=10-50 nM  ...  D=100-500`,
        # stating the unit early and dropping it later, so `D=100-500` has no
        # unit, no comparison and no metric — and requiring only the first
        # three cost that patent 372 records. Two numbers joined as a range is
        # itself the evidence, and `A: 4` / `E=20` have neither.
        if not (m.group("lou") or m.group("hiu") or m.group("hi2u")
                or m.group("loop") or m.group("hiop") or m.group("hi2")
                or m.group("preop")
                or re.search(_METRIC, m.group(0), re.I)):
            continue
        # Each bound carries its OWN unit, for the reason `_prose_bounds`
        # documents: a single clause may state its two bounds on different
        # scales. US9656988 writes `100 nM < C ≦ 1 μM`, and taking one unit for
        # the pair reported C as `100..1 uM` — an interval whose lower bound is
        # above its upper, 1,000x out, and still shaped like a valid answer.
        lo = hi = None
        lo_u = hi_u = None
        # A bound stated BEFORE the symbol. `10 nM < B` reads "10 nM is less
        # than B", so the number is B's LOWER bound — the operator points at
        # the symbol, and the sense is therefore the reverse of the same
        # operator written after it.
        if m.group("pre") is not None:
            v, u = _f(m.group("pre")), _canon_unit(m.group("preu"))
            if m.group("preop") in ("<", "<=", "≤", "≦", "⩽"):
                lo, lo_u = v, u
            else:
                hi, hi_u = v, u
        lo_op, lo_val = m.group("loop"), m.group("lo")
        if lo_val is not None:
            v, u = _f(lo_val), _canon_unit(m.group("lou"))
            if lo_op in (">", ">=", "≥", "≧", "⩾", None):
                lo, lo_u = v, u
            else:
                hi, hi_u = v, u
        if m.group("hi") is not None:
            v, u = _f(m.group("hi")), _canon_unit(m.group("hiu"))
            # "1 uM > IC50" means 1 uM is the UPPER bound.
            if m.group("hiop") in (">", ">=", "≥", "≧", "⩾"):
                hi, hi_u = v, u
            else:
                lo, lo_u = v, u
        if m.group("hi2") is not None and hi is None:
            hi, hi_u = _f(m.group("hi2")), _canon_unit(m.group("hi2u"))
        lo, hi, unit = _reconcile(lo, lo_u, hi, hi_u)
        if lo is not None or hi is not None:
            _adopt(here, conflicts, BinRange(sym, lo, hi, unit))
    _merge(out, here)

    # Form 3 — a symbol defined by a prose CLAUSE rather than an operator:
    #   "A = IC50 of less than 10 nM; B = IC50 less than 100 nM but greater
    #    than or equal to 10 nM"
    # Form 1 needs a comparison symbol or a bare number straight after the
    # separator and finds neither, so US11752149's whole key parsed as {} and
    # all 47 of its graded records came back with no value. Runs last, so a
    # clause Form 1 can already read keeps Form 1's reading.
    seen_wordy: dict[str, BinRange] = {}
    for m in _KEY_WORDY.finditer(text):
        sym = m.group("sym")
        # Same `value symbol` guard as Form 1 — this pass reads the identical
        # span through a different pattern and would re-adopt what that one
        # refused.
        if not (m.group("sep") or "").strip() \
                and _VALUE_BEFORE.search(text[:m.start("sym")]):
            continue
        lo, hi, unit = _prose_bounds(m.group("body"))
        if lo is None and hi is None:
            continue
        # Same contest test as the other passes, applied to this pass's own
        # readings so a REFINEMENT of an earlier pass is not mistaken for one.
        _adopt(seen_wordy, conflicts, BinRange(sym, lo, hi, unit))
        prev = out.get(sym)
        if prev is None:
            out[sym] = BinRange(sym, lo, hi, unit)
            continue
        # A symbol Form 1 already read. Replace it ONLY when the prose reading
        # is a strict REFINEMENT — same unit, and the new interval sits inside
        # the old one. US9670210 writes `++ refers to IC50 >100 nM and ≦500 nM`
        # and Form 1 stops at the lower bound because `and` is not a range
        # separator, so `++` came out as "anything above 100 nM" when the
        # patent bounds it at 500. Requiring containment means this can only
        # ever tighten a bin, never move or widen one — the failure mode this
        # file exists to prevent.
        if prev.unit and unit and prev.unit != unit:
            continue
        tighter = ((prev.lo is None or (lo is not None and lo >= prev.lo))
                   and (prev.hi is None or (hi is not None and hi <= prev.hi))
                   and (lo, hi) != (prev.lo, prev.hi))
        if tighter:
            out[sym] = BinRange(sym, lo, hi, unit or prev.unit)

    # A symbol the text defines two incompatible ways yields nothing. Dropped
    # BEFORE the unit backfill below, so a contested reading cannot be the one
    # unit that the backfill then spreads over the rest of the key.
    for sym in conflicts:
        out.pop(sym, None)

    # A key defined without units on every line usually states it once; adopt
    # the single unit seen, rather than leaving most entries unitless.
    #
    # Scoped to ONE SYMBOL FAMILY, because a document that grades with two
    # alphabets is grading two different things. US10030020 uses `*` for IC50
    # in nM and `#` for percent inhibition, in one legend; pooling their units
    # left `%` looking like the odd one out and stamped `nM` across both.
    # A family is the smallest group the document itself keeps together.
    for family in {_alphabet(s) for s in out}:
        bins = [b for s, b in out.items() if _alphabet(s) == family]
        units = {b.unit for b in bins if b.unit}
        if len(units) == 1:
            only = units.pop()
            for b in bins:
                if not b.unit:
                    b.unit = only
    return out


# A legend heading is a label, not a sentence. Anything longer is prose that
# happens to sit between two definition rows.
_HEADING_MAX = 120

# A heading's parenthetical qualifier describes the ASSAY FORMAT, not the
# target: `DNA-PK (enzymatic)` and `pDNA-PK (cellular)` differ by the letter p,
# not by the words in brackets. Matching on the bracketed text would pair
# `pDNA-PK (cellular)` with a column reading `cellular IC50` for a different
# target entirely, so it is dropped before matching.
_QUALIFIER = re.compile(r"\([^)]*\)|\[[^\]]*\]")


def _norm_match(text: str) -> str:
    """Letters and digits only, lowercased — the form headings are matched in.

    Punctuation is noise here and dropping it is what makes the match survive
    the patent's own typo: US10172859 heads the legend section `Kv11.1 hERG`
    and the data column `Ki [Kv1.11 hERG]`, transposing the digits around the
    decimal point. Both normalise to `kv111herg`.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


# A line that OPENS with a grade symbol and a separator is a definition row,
# whether or not the parser managed to read it.
_LOOKS_DEFINED = re.compile(rf"^\W*{_SYMBOL}{_DEFINES}", re.I)


def _clean_heading(line: str) -> str:
    """A section heading, or "" when the line is not one.

    A definition row the parser could not read must NOT become a heading. It
    would open a section that does not exist and push the rows below it out of
    the section they belong to — so one unreadable grade mislabels a whole
    scale rather than costing its own row. Failing closed here means a parser
    gap stays a gap.
    """
    h = line.strip().rstrip(":").strip()
    if not h or not re.search(r"[A-Za-z]", h) or _LOOKS_DEFINED.match(line.strip()):
        return ""
    return h


# A cell holding nothing but a grade symbol, with optional trailing colon.
_BARE_SYMBOL = re.compile(rf"^\W*{_QUOTE}?({_SYMBOL}){_QUOTE}?\s*:?\s*$")


# The unit a legend table states in its own heading, not on its rows.
# US9221791 heads its columns `MIC (μg/mL` — the closing bracket is lost in
# the CALS split — and `% Disease Control @ 50 ppm`, then prints bare numbers
# beneath. Without this the bins are correct intervals of nothing.
_TABLE_UNIT = re.compile(rf"\(?\s*({_UNIT})", re.I)


def parse_bin_table(rows, unit_hint: str = "") -> dict[str, BinRange]:
    """Read a legend laid out as a TABLE, in either column order.

    `rows` is a list of cell lists. A legend table puts the symbol in one cell
    and its range in another, and patents disagree about which comes first:

        ['', 'A:', 'IC50 < 3 nM']        symbol then range   US10172859
        ['', '≦0.5', 'A']                range then symbol   US9221791

    Reading a row's cells is what makes the second order safe. Flattened into
    text it is unreadable — `40-59 C <40 D` binds `C` to the value printed on
    D's row, which is why `_VALUE_BEFORE` refuses that shape in prose. Inside
    one row there is no such ambiguity: whichever cell is nothing but a symbol
    is the symbol, and the other is its range.
    """
    out: dict[str, BinRange] = {}
    conflicts: set[str] = set()
    for row in rows:
        cells = [c.strip() for c in row if c and c.strip()]
        if len(cells) != 2:
            continue
        head, tail = cells
        m_head, m_tail = _BARE_SYMBOL.match(head), _BARE_SYMBOL.match(tail)
        if m_tail and not m_head:
            sym, expr = m_tail.group(1), head
        elif m_head:
            sym, expr = m_head.group(1), tail
        else:
            continue
        # `sym: expr` is the form every pattern in this module already reads,
        # so the two orders converge here rather than growing a second parser.
        bins = parse_bin_key(f"{sym}: {expr}")
        br = bins.get(sym)
        if br is not None:
            _adopt(out, conflicts, br)
    for sym in conflicts:
        out.pop(sym, None)
    m = _TABLE_UNIT.search(unit_hint or "")
    if m:
        unit = _canon_unit(m.group(1))
        for br in out.values():
            if not br.unit:
                br.unit = unit
    return out


# How a patent introduces one of several scales in running prose:
#   "In the Table below, for D816V activity, the following designations are
#    used: ...  For wild-type Kit activity, the following designations are
#    used: ..."
# The subject of the `for` clause names the column the scale governs.
_PROSE_SECTION = re.compile(
    r"\b(?:for|in)\s+(?P<subject>[^,.;:]{2,60}?)\s*,?\s+"
    r"the\s+following\s+\w+\s+(?:are|is)\s+used\s*:?", re.I)


def split_prose_sections(text: str):
    """[(heading, body)] for prose that states more than one scale.

    Returns [] when the text introduces fewer than two, so the caller keeps
    its ordinary single-key path. US9688680 states both of its scales in one
    paragraph and they use THE SAME LETTERS with different numbers, so read as
    one key they contest each other and neither survives — correct, and worth
    950 records to improve on.
    """
    marks = list(_PROSE_SECTION.finditer(text or ""))
    if len(marks) < 2:
        return []
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group("subject").strip(), text[m.end():end]))
    return out


def parse_sectioned_key(rows) -> dict[str, dict[str, BinRange]]:
    """Split a legend into its per-column scales: {heading: {symbol: range}}.

    `rows` is the legend's lines in document order, each already joined across
    its cells. A line that defines bins extends the current section; a short
    line that does not opens a new one.

    A legend with several sections is not a stylistic variation — it is a
    different scale per column, using THE SAME LETTERS. US10172859 defines
    `A`-`D` three times over: `B` is 3-7 nM for enzymatic DNA-PK, 0.5-5 uM for
    the cellular assay, and 15-25 uM for hERG. Flattening the three and calling
    `parse_bin_key` returns the first, because that function takes the first
    definition of each symbol by design — so every hERG bin would come out on a
    nM scale, a 1,000x error applied uniformly and invisibly to 431 records.

    Sections are therefore kept apart and matched to a column by name, and a
    column that matches none keeps no range at all.
    """
    sections: dict[str, dict[str, BinRange]] = {}
    current = ""
    for line in rows:
        line = (line or "").strip()
        if not line:
            continue
        bins = parse_bin_key(line) if looks_like_key(line) else {}
        if bins:
            slot = sections.setdefault(current, {})
            for sym, br in bins.items():
                # First definition of a symbol within a section wins, which is
                # `parse_bin_key`'s own rule applied one level up.
                slot.setdefault(sym, br)
        elif len(line) <= _HEADING_MAX:
            heading = _clean_heading(line)
            if heading:
                current = heading
    return {k: v for k, v in sections.items() if v}


# Words that appear in a heading without telling you which column it means.
_STOP_WORD = {
    "activity", "assay", "assays", "data", "value", "values", "inhibition",
    "rating", "scale", "the", "and", "of", "for", "in", "a", "an", "results",
}


def _heading_tokens(heading: str) -> list[str]:
    """The parts of a heading that could name a column, longest first."""
    words = re.findall(r"[A-Za-z0-9]+", _QUALIFIER.sub(" ", heading or ""))
    keep = [w.lower() for w in words if w.lower() not in _STOP_WORD]
    return sorted(set(keep), key=len, reverse=True)


def section_for_column(column: str, sections) -> str | None:
    """Which legend section governs `column`, or None when it is not decidable.

    A heading matches a column when one of its distinctive WORDS appears in
    that column, and the LONGEST such word wins. Length settles the case this
    exists for: `DNA-PK` is a substring of `pDNA-PK`, so `IC50 pDNA-PK` matches
    both headings, and the shorter one is wrong by exactly the margin that
    makes it look right.

    Words, rather than the whole heading, because a heading is a phrase and a
    column is an abbreviation of it. US9688680 heads a scale `for D816V
    activity` and the column `D816V IC50 (nM)`; nothing contains anything, and
    the one word that matters is in both.

    A tie returns None. Two headings that fit a column equally well means the
    document does not say which scale applies, and guessing is the failure this
    module is written to prevent.
    """
    col = _norm_match(column or "")
    if not col:
        return None
    best: str | None = None
    best_len = 0
    tied = False
    for heading in sections:
        hit = max((len(t) for t in _heading_tokens(heading)
                   if _norm_match(t) and _norm_match(t) in col), default=0)
        if hit > best_len:
            best, best_len, tied = heading, hit, False
        elif hit == best_len and hit and heading != best:
            tied = True
    return None if tied else best


def assign_sections(columns, sections) -> dict[str, str]:
    """{column: heading}, pairing each graded column with the scale it takes.

    Name matching first. What it cannot reach, COUNTING can: when the legend
    states exactly as many scales as the table has graded columns, and all but
    one column has been claimed by name, the last pairing is forced rather than
    guessed. US9688680 needs that step and nothing else will do it — its two
    scales are headed `D816V activity` and `wild-type Kit activity`, its two
    columns are `D816V IC50 (nM)` and `WT IC50 (nM)`, and `WT` shares not one
    character with `wild-type`. `D816V` pins the first column and excludes the
    second, so the remainder is determined.

    Anything short of determined returns nothing for that column, which leaves
    its grades without a range — the outcome this module prefers to a guess.
    """
    out: dict[str, str] = {}
    for col in columns:
        heading = section_for_column(col, sections)
        if heading is not None:
            out[col] = heading
    if len(sections) != len(columns):
        return out
    free_cols = [c for c in columns if c not in out]
    free_secs = [s for s in sections if s not in out.values()]
    if len(free_cols) == 1 and len(free_secs) == 1:
        out[free_cols[0]] = free_secs[0]
    return out


# What kind of quantity a unit measures. A bin key and the column it is applied
# to must agree on this much, or they are not describing the same measurement.
_DIMENSION = {
    "pM": "conc", "nM": "conc", "uM": "conc", "mM": "conc",
    "%": "percent", "mg/kg": "dose", "fold": "ratio",
}


# The markers that name a dimension in a column heading outright.
_HEADER_MARK = {"%": "percent", "mg/kg": "dose", "fold": "ratio"}


def compatible(column_unit: str | None, key_unit: str | None,
               column_header: str = "") -> bool:
    """May a bin stated in `key_unit` be attached to a column in `column_unit`?

    Unknown on either side is permitted — most columns state no unit and most
    keys are read from text that does. Only a stated DISAGREEMENT refuses.

    This is the last guard before a range reaches a record, and it catches what
    the parser cannot: a key that is read perfectly and then applied to the
    wrong column. US12351648 defines `*`-`****` twice, once for Ki in μM and
    once for a MASP-2-versus-thrombin selectivity ratio in fold, and the
    selectivity scale reached three columns headed `Ki (μM)` — 774 records
    carrying a fold-change where the patent prints a concentration. The column
    header said `μM` the whole time.

    `column_header` can only ever OVERRULE A REFUSAL, never cause one. A
    heading names a condition as often as a unit, and the column unit reader
    cannot always tell which: `MAGL % Inh 1 μM (mouse)` reports a percentage AT
    a concentration of 1 μM, and reads as a μM column, so a percent key was
    refused for 807 records that state their dimension in the heading itself.
    When the heading spells out the key's own dimension, that is the patent
    agreeing, and it settles the question.
    """
    a, b = _DIMENSION.get(column_unit or ""), _DIMENSION.get(key_unit or "")
    if not (a and b) or a == b:
        return True
    return any(mark in (column_header or "").lower() and dim == b
               for mark, dim in _HEADER_MARK.items())


def parse_bin_key_lines(lines) -> dict[str, BinRange]:
    """Resolve a key from table rows, each row read ON ITS OWN.

    Rows are a list, not a paragraph, and joining them lets a definition take
    its number from the row below. US9221791 prints its legend as a two-column
    table — `40-59 | C`, `<40 | D` — and concatenated that reads as `40-59 C
    <40 D`, so `C` is followed by `<40` and comes out as "under 40", which is
    verbatim the D bin. The value belongs to the row it is printed on.

    Rows still contest each other: a symbol two rows define differently is two
    scales in one block, and `_adopt` drops it.
    """
    out: dict[str, BinRange] = {}
    conflicts: set[str] = set()
    for line in lines:
        for sym, br in parse_bin_key(line).items():
            _adopt(out, conflicts, br)
    for sym in conflicts:
        out.pop(sym, None)
    return out


def parse_bin_key_layered(parts) -> dict[str, BinRange]:
    """Resolve a key from ordered sources, nearest first.

    `parts` are a block's legend sources in priority order — its caption, its
    own legend fragments, its rows, the key printed above it. Each is parsed
    ALONE and the first reading of a symbol wins, which is the precedence the
    concatenated form had by accident and this states on purpose.

    Parsing them apart is what makes the contest test in `_adopt` usable. Two
    definitions in ONE source are two legends with equal claim, and neither can
    be trusted. Two definitions in DIFFERENT sources are a near one and a far
    one, and the near one is simply right: US11566007 prints a four-grade key
    directly above TABLE-US-00006 and the FIVE-grade key of the table that
    follows in rows at its foot, so `++++` is `≥1 uM` in one and `1-10 uM` in
    the other. Flattened, that reads as a contradiction and costs 192 records;
    layered, the key above the table wins and the trailing rows are what they
    are — the next table's.
    """
    out: dict[str, BinRange] = {}
    for part in parts:
        if not part:
            continue
        for sym, br in parse_bin_key(part).items():
            out.setdefault(sym, br)
    return out


def nearest_key_before(text: str) -> str:
    """The LAST key-shaped span in a table's preceding text, or "".

    A fixed slice of the look-back cannot find these. US11566007's bin tables
    print their compound lists inline, so 1,200 characters before TABLE 7 are
    entirely the previous table's `A###,` tokens and its own key — sitting
    just beyond them — was never read: ten blocks, 110 rows each, silent.

    Nearest-PRECEDING is the safe direction and the reason is asymmetry. A key
    printed above a table governs it until the document redefines it, so
    reaching further back can only find an older definition of the same
    grades. Reaching FORWARD finds the next table's key, which is how a
    four-grade scale gets read with five-grade ranges — the 10x error this
    file's header warns about. `following` was tried and reverted for exactly
    that; this searches backwards only, and takes the closest match.
    """
    if not text:
        return ""
    # A symbol that merely FOLLOWS a value does not begin a key, and rejecting
    # it here matters more than rejecting it later. The span this returns is
    # cut at the first hit, so a false start does not just add noise — it
    # amputates the value that would have exposed it. US9221791's rating table
    # flattens to `... 40-59 C <40 D ...`; `C` looked like a key start, the
    # span began at `C`, and `40-59` was left outside it, so the guard in
    # `parse_bin_key` saw a symbol at the head of a string with nothing before.
    hits = []
    for m in re.finditer(r"(?:[*\s]|^)(?:Key\s*:\s*)?(" + _SYMBOL + r")"
                         + rf"(?P<sep>{_DEFINES})", text, re.I):
        if not (m.group("sep") or "").strip() \
                and _VALUE_BEFORE.search(text[:m.start(1)]):
            continue
        hits.append((m.start(), (m.group(1) or "").strip()))
    if not hits:
        return ""
    # A key is a RUN of grade definitions, not one. Taking the last match alone
    # returned `+: IC50 < 0.01 uM` — the final line of a four-line key — and a
    # one-grade scale reads three quarters of the table as unbinned.
    #
    # A run ends where a symbol REPEATS, not at some character distance. A key
    # defines each grade once, so seeing `++++` a second time means a second
    # key began — and distance cannot tell you that: in US11566007 two keys are
    # separated by hundreds of inline compound ids, and nothing says the next
    # patent will not print them back to back. Splitting on the repeat is what
    # makes "nearest wins" hold, because `parse_bin_key` takes the FIRST
    # definition of a symbol and merging two keys would hand back the older.
    # WALK FROM THE END, because the end is what the caller wants.
    #
    # Splitting forward puts the boundary in the wrong place whenever a later
    # key opens with a symbol the earlier key never used. Reading forward,
    # `+++++` is simply a symbol not yet seen, so it joins the run it follows
    # instead of opening its own; the repeat one symbol later then starts the
    # new run WITHOUT it. US11566007 is exactly that: its five-grade key runs
    # `+++++ ++++ +++ ++ +`, the nearest run came back as the last four, and
    # `parse_bin_key` returned a key that defines every grade except the one
    # the run began with. 413 records on TABLE-US-00007 carried `+++++` and no
    # range, while `++`, `+++` and `++++` all resolved — the signature of a
    # boundary drawn one symbol too late.
    #
    # Walking backwards, the first run IS the last key, complete, and a repeat
    # can only mean we have stepped into the key before it. Same rule, applied
    # from the side that defines the answer.
    runs: list[list[int]] = []
    cur: list[int] = []
    seen: set[str] = set()
    for pos, sym in reversed(hits):
        if sym in seen:
            runs.append(cur[::-1])
            cur, seen = [], set()
        cur.append(pos)
        seen.add(sym)
    runs.append(cur[::-1])
    for run in runs:                     # built nearest-first already
        if not run:
            continue
        span = text[run[0]:run[-1] + 260]
        if parse_bin_key(span):
            return span
    return ""


# The two shapes `looks_like_key` tests for, as one pattern compiled at import.
#
# They were built inline and searched separately, which meant an f-string
# interpolation, a lookup of a ~90-character pattern in `re`'s module cache and
# TWO scans on every call — and `extract_from_patent` calls this on EVERY CELL
# of every block (604,468 times over a 3-patent trace) before it calls it once
# on the joined text.
#
# `bool(A.search(t)) or bool(B.search(t))` is `bool((?:A)|(?:B)).search(t)`:
# each branch keeps its own group so alternation cannot straddle them, the
# flags are the same on both, and only the boolean is used — which alternative
# matched, and where, was never read. Built where `_SYMBOL`/`_DEFINES`/`_NUM`
# are already defined, so a capability patch to any of those still lands.
#
# The follow-set must list the PROSE comparisons too, not only the operator
# glyphs. `+ (greater than 10 microMolar)` and `*** is less than 100 nM` reach
# the parser's own patterns cleanly and were rejected one step earlier, because
# what follows the separator is the word "greater", which was not in the set.
# A filter in front of a parser has to accept everything the parser accepts;
# every record lost to this file was lost at a filter, not at a parser.
_CMP = r"(?:greater|less|more|at\s+least|at\s+most|between|about|under|over)"
_KEY_HINT = re.compile(
    rf"(?:{_MARKED}\s*{_QUOTE}?{_SYMBOL}|\bkey\s*:|\*\s*key)"
    # `(?:[a-z]+\s+){0,4}` — a few filler words may sit between the separator
    # and the evidence. US20240166635 writes `"A" represents a calculated IC50
    # value of less than 10 nM`: two words, "a calculated", and the hint said no
    # while the parser read it perfectly.
    rf"|(?:{_SYM_QN}{_DEFINES}(?:[a-z]+\s+){{0,4}}(?:{_METRIC}|{_CMP}|{_NUM}|[<>≤≥≦≧⩽⩾]))"
    rf"|(?:{_NUM}\s*{_UNIT}?\s*=\s*{_QUOTE}?{_SYMBOL})"
    # Forms 5 and 6, added to the PARSER and — the first time — not to this.
    # `1000 nM < IC50 <= 10000 nM: +++` and `A 0 < PI3K Delta Activity < 50 nM`
    # both parse cleanly and were refused here, so 499 records were lost at the
    # filter and never at the parser. That is the failure this file's own
    # gotcha names, repeated: a filter in front of a parser must accept
    # everything the parser accepts.
    rf"|(?:[<>≤≥≦≧⩽⩾]\s*{_NUM}[^;:]{{0,40}}?[:=]\s*{_QUOTE}?{_SYMBOL})"
    rf"|(?:{_SYM_QN}\s+{_NUM}\s*{_UNIT}?\s*[<>≤≥≦≧⩽⩾])", re.I)


def looks_like_key(text: str) -> bool:
    """Cheap test for whether text is worth running the full parser over."""
    if not text:
        return False
    return bool(_KEY_HINT.search(text))
