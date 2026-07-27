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

# Form 1 — compact key:  "++++: IC50 ≥ 1 uM"  /  "+++: 1 uM > IC50 ≥ 0.1 uM"
_KEY_COMPACT = re.compile(
    rf"({_SYMBOL})\s*[:=]\s*"
    rf"(?:(?P<hi>{_NUM})\s*(?P<hiu>{_UNIT})?\s*(?P<hiop>>=|>|≥|≤|<=|<)\s*)?"
    rf"(?:IC\s*50|EC\s*50|Ki|Kd|value)?\s*"
    rf"(?P<loop>>=|>|≥|≤|<=|<)?\s*(?P<lo>{_NUM})\s*(?P<lou>{_UNIT})?",
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
        unit = _canon_unit(m.group("lou") or m.group("hiu"))
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


def looks_like_key(text: str) -> bool:
    """Cheap test for whether text is worth running the full parser over."""
    if not text:
        return False
    return bool(re.search(r"is\s+marked|\bkey\s*:|\*\s*key", text, re.I)) or \
        bool(re.search(rf"{_SYMBOL}\s*[:=]\s*(?:IC\s*50|EC\s*50|{_NUM})", text))
