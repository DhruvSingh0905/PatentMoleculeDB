"""Potency-bin legends: `+`, `++`, `A`-`E` → explicit numeric ranges.

Many patents do not publish a number per compound. They publish a symbol and,
somewhere nearby, a key that defines what the symbol means:

    *Key: ++++: IC50 ≥ 1 uM   +++: 1 uM > IC50 ≥ 0.1 uM   ...

    an IC50 value of greater than or equal to 0.001 μM and less than or equal
    to 0.01 μM is marked "++++"; a value greater than 0.01 μM and less than or
    equal to 0.1 μM is marked "+++"; ...

Both forms appear, and the two examples above are from different patents with
*different scales for the same symbol* — US11566007 uses `++++` for ≥1 μM while
US11292791 uses `++++` for 0.001-0.01 μM. **A global symbol→range mapping would
silently corrupt one of them.** Keys are therefore always resolved per table,
never shared, and a symbol with no key in scope yields no range at all.

This turns a symbol into an honest interval, which is real, usable data — a
compound known to be 0.1-1 μM is a meaningful record for a screening cascade.
It will never match a point value in BindingDB, and should not be expected to:
that is a limitation of exact-match scoring, not of the extraction.
"""
from __future__ import annotations

import re

# Units accepted inside a key definition, normalised to the project's spelling.
_UNIT = r"(?:nM|pM|mM|[μuµ]M|micromolar|nanomolar|millimolar|picomolar)"
_UNIT_CANON = {
    "um": "uM", "µm": "uM", "μm": "uM", "nm": "nM", "mm": "mM", "pm": "pM",
    "micromolar": "uM", "nanomolar": "nM", "millimolar": "mM", "picomolar": "pM",
}

# The symbol a bin is written with: a run of '+', or a single letter grade.
_SYMBOL = r"(?:\++|[A-E])"

_NUM = r"\d+(?:\.\d+)?"

# What separates a symbol from its definition. `:` and `=` are the common
# forms; US10376513 writes "+ refers to ≤10 nM" and states the key for all 348
# of its compounds that way, so the verb forms are equally a separator.
_DEFINES = r"(?:\s*[:=]\s*|\s+(?:refers?\s+to|means|indicates?|represents?|denotes?)\s+)"

# Form 1 — compact key:  "++++: IC50 ≥ 1 uM"  /  "+++: 1 uM > IC50 ≥ 0.1 uM"
#                        "++ refers to >10 nM to 50 nM"
# The trailing upper bound matters: without it "++ refers to >10 nM to 50 nM"
# parses as lo=10 with NO upper bound, turning a 10-50 nM bin into "anything
# above 10 nM". That is the same class of silent widening that a shared bin key
# causes, so the tail is captured rather than tolerated. US10626094 writes the
# same range as "A: IC 50 >200 nM−<800 nM" with a U+2212 MINUS between the
# bounds, and read every one of its bins as unbounded above until this landed.
_KEY_COMPACT = re.compile(
    rf"({_SYMBOL}){_DEFINES}"
    rf"(?:(?P<hi>{_NUM})\s*(?P<hiu>{_UNIT})?\s*(?P<hiop>>=|>|≥|≤|<=|<)\s*)?"
    rf"(?:IC\s*50|EC\s*50|Ki|Kd|value)?\s*"
    rf"(?P<loop>>=|>|≥|≤|<=|<)?\s*(?P<lo>{_NUM})\s*(?P<lou>{_UNIT})?"
    rf"(?:\s*(?:to|[-–—−])\s*[<>≤≥]?\s*(?P<hi2>{_NUM})\s*(?P<hi2u>{_UNIT})?)?",
    re.I)

# Form 2 — prose:  "a value greater than 0.01 μM and less than or equal to
#                   0.1 μM is marked "+++""
# NOTE the character class is `[^;]`, not `[^.;]`. Excluding the period looks
# safer but silently truncates every clause containing a decimal — the body of
# "greater than or equal to 0.001 μM and less than or equal to 0.01 μM" cannot
# be spanned without crossing a '.', so the lower bound was being dropped and
# every bin came out unbounded-below. Clauses in these legends are separated by
# semicolons, which is the boundary that actually holds.
_KEY_PROSE = re.compile(
    rf"(?P<body>(?:greater|less|more)[^;]{{0,200}}?{_NUM}\s*{_UNIT}[^;]{{0,200}}?)"
    rf"is\s+marked\s*[\"“'`]?({_SYMBOL})[\"”'`]?",
    re.I)

_GT = re.compile(rf"(?:greater than or equal to|at least|≥|>=)\s*({_NUM})\s*({_UNIT})?", re.I)
_GT_STRICT = re.compile(rf"(?:greater than|>)\s*({_NUM})\s*({_UNIT})?", re.I)
_LT = re.compile(rf"(?:less than or equal to|at most|≤|<=)\s*({_NUM})\s*({_UNIT})?", re.I)
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

    @property
    def midpoint(self) -> float | None:
        """Geometric midpoint, for consumers that need a single number.

        Geometric rather than arithmetic because potency spans orders of
        magnitude — the middle of 0.01-0.1 μM is 0.032, not 0.055. Provided for
        ranking only; never store it as if it were a measured value.
        """
        if self.lo is not None and self.hi is not None and self.lo > 0:
            return (self.lo * self.hi) ** 0.5
        return self.hi if self.lo is None else self.lo

    def contains(self, value: float) -> bool:
        if self.lo is not None and value < self.lo:
            return False
        if self.hi is not None and value > self.hi:
            return False
        return True

    def __repr__(self) -> str:
        lo = "-inf" if self.lo is None else f"{self.lo:g}"
        hi = "inf" if self.hi is None else f"{self.hi:g}"
        return f"BinRange({self.symbol!r}, {lo}..{hi} {self.unit or '?'})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, BinRange) and self.symbol == other.symbol
                and self.lo == other.lo and self.hi == other.hi
                and self.unit == other.unit)


def _prose_bounds(body: str) -> tuple[float | None, float | None, str | None]:
    """Pull (lo, hi, unit) out of one prose clause."""
    lo = hi = None
    unit = None
    for pat, is_lo in ((_GT, True), (_GT_STRICT, True), (_LT, False), (_LT_STRICT, False)):
        m = pat.search(body)
        if not m:
            continue
        val = float(m.group(1))
        unit = unit or _canon_unit(m.group(2))
        if is_lo and lo is None:
            lo = val
        elif not is_lo and hi is None:
            hi = val
    return lo, hi, unit


def parse_bin_key(text: str) -> dict[str, BinRange]:
    """Extract a symbol → range mapping from legend/footnote text.

    Returns {} when no key is present. Never falls back to a default mapping:
    two patents in this corpus assign incompatible ranges to `++++`, so a
    guessed key is worse than none.
    """
    if not text:
        return {}
    out: dict[str, BinRange] = {}

    # Prose form first — it is unambiguous when present.
    for m in _KEY_PROSE.finditer(text):
        lo, hi, unit = _prose_bounds(m.group("body"))
        sym = m.group(2)
        if (lo is not None or hi is not None) and sym not in out:
            out[sym] = BinRange(sym, lo, hi, unit)

    # Compact form: "++++: IC50 ≥ 1 uM", "+++: 1 uM > IC50 ≥ 0.1 uM"
    for m in _KEY_COMPACT.finditer(text):
        sym = m.group(1)
        if sym in out:
            continue
        lo = hi = None
        unit = _canon_unit(m.group("lou") or m.group("hiu") or m.group("hi2u"))
        lo_op, lo_val = m.group("loop"), m.group("lo")
        if lo_val is not None:
            v = float(lo_val)
            if lo_op in (">", ">=", "≥", None):
                lo = v
            else:
                hi = v
        if m.group("hi") is not None:
            v = float(m.group("hi"))
            # "1 uM > IC50" means 1 uM is the UPPER bound.
            if m.group("hiop") in (">", ">=", "≥"):
                hi = v
            else:
                lo = v
        if m.group("hi2") is not None and hi is None:
            hi = float(m.group("hi2"))
        if lo is not None or hi is not None:
            out[sym] = BinRange(sym, lo, hi, unit)

    # A key defined without units on every line usually states it once; adopt
    # the single unit seen, rather than leaving most entries unitless.
    units = {b.unit for b in out.values() if b.unit}
    if len(units) == 1:
        only = units.pop()
        for b in out.values():
            if not b.unit:
                b.unit = only
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
    hits = [(m.start(), (m.group(1) or "").strip())
            for m in re.finditer(r"(?:[*\s]|^)(?:Key\s*:\s*)?(" + _SYMBOL + r")"
                                 + _DEFINES, text, re.I)]
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
    runs: list[list[int]] = []
    cur: list[int] = []
    seen: set[str] = set()
    for pos, sym in hits:
        if sym in seen:
            runs.append(cur)
            cur, seen = [], set()
        cur.append(pos)
        seen.add(sym)
    runs.append(cur)
    for run in reversed(runs):
        if not run:
            continue
        span = text[run[0]:run[-1] + 260]
        if parse_bin_key(span):
            return span
    return ""


def looks_like_key(text: str) -> bool:
    """Cheap test for whether text is worth running the full parser over."""
    if not text:
        return False
    return bool(re.search(r"is\s+marked|\bkey\s*:|\*\s*key", text, re.I)) or \
        bool(re.search(rf"{_SYMBOL}{_DEFINES}(?:IC\s*50|EC\s*50|{_NUM}|[<>≤≥])",
                       text, re.I))
