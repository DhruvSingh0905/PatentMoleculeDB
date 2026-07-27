"""The learned rule library: propose → validate → persist → reuse.

A rule is keyed by LAYOUT FINGERPRINT, never by patent. The whole economic
argument for this loop is amortisation: one paid question per distinct table
shape, reused free on every patent that shares it. Keying by patent would
reduce it to HARVEST with extra steps.

Four rule kinds, because "emit a regex or give up" is too narrow for what
actually goes wrong:

  column_map  — say which column is the id, which are assays, their units.
                Covers most failures; no regex, nothing to over-match.
  row_regex   — a pattern applied per row when the cell structure is lost.
  value_pattern — how to read an individual CELL. This is what makes the loop
                self-healing for parser bugs rather than only layout bugs. When
                `_VALUE_PAT` rejected any number >= 1000, the columns were
                mapped perfectly and the cells were plainly numeric; nothing in
                the other rule kinds could say "your number regex is too
                narrow". Adopting one is gated hard: it must parse the cells we
                currently fail on, agree with the existing parser on every cell
                we already read, and still reject the adversarial battery.
  not_assay   — a NEGATIVE rule. US11566007's `A1 | 944.5` table is mass-spec.
                The correct outcome is "never treat this shape as assay data",
                and recording it stops us paying to rediscover that forever.
  escalate    — the model could not produce a safe rule. Carries what
                capability is missing so a human reads a queue, not a log.

Nothing is trusted because a model proposed it. `validate()` re-runs the
proposed rule against the table it came from and against held-out rows, and a
rule that cannot reproduce what a human can see in the sample is rejected.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core import config

COLUMN_MAP = "column_map"
ROW_REGEX = "row_regex"
VALUE_PATTERN = "value_pattern"
BIN_KEY = "bin_key"
NOT_ASSAY = "not_assay"
ESCALATE = "escalate"

_LIBRARY = config.PACKAGE_ROOT / "data" / "layout_rules.json"

# Regex constructs that make catastrophic backtracking possible. A synthesized
# pattern runs over thousands of rows, so a quadratic blowup is a hang, and the
# model has no idea it wrote one.
_REDOS = re.compile(r"\((?=[^)]*[+*])[^)]*\)\s*[+*]|\(\?[:=!][^)]*[+*][^)]*\)[+*]")


@dataclass
class Rule:
    fingerprint: str
    kind: str
    # column_map: {"cid": 0, "assays": [{"index": 1, "name": ..., "unit": ...}]}
    # row_regex:  {"pattern": ..., "fields": {...}}
    # not_assay:  {"because": "mass-spec characterisation"}
    # escalate:   {"capability": ..., "note": ...}
    payload: dict = field(default_factory=dict)
    source: str = "llm"                  # llm | human | builtin
    model: str = ""
    created: str = ""
    # Evidence, kept so a bad rule can be traced and revoked.
    validated_on: dict = field(default_factory=dict)
    times_applied: int = 0
    rows_yielded: int = 0

    def to_json(self) -> dict:
        return asdict(self)


class RuleLibrary:
    """Persistent, append-only-ish store of validated layout rules."""

    def __init__(self, path: Path | None = None):
        self.path = path or _LIBRARY
        self._rules: dict[str, Rule] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for r in raw.get("rules", []):
            try:
                self._rules[r["fingerprint"]] = Rule(**r)
            except TypeError:
                continue

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"schema": 1, "rules": [r.to_json() for r in self._rules.values()]},
            indent=2))
        tmp.replace(self.path)

    def get(self, fingerprint: str) -> Rule | None:
        return self._rules.get(fingerprint)

    def add(self, rule: Rule) -> None:
        rule.created = rule.created or time.strftime("%Y-%m-%dT%H:%M:%S")
        self._rules[rule.fingerprint] = rule

    def knows(self, fingerprint: str) -> bool:
        """True when this layout has already been answered — including
        `not_assay` and `escalate`. Re-asking a question we have paid for is
        the single easiest way to lose the cost advantage."""
        return fingerprint in self._rules

    def record_use(self, fingerprint: str, rows: int) -> None:
        r = self._rules.get(fingerprint)
        if r:
            r.times_applied += 1
            r.rows_yielded += rows

    def stats(self) -> dict:
        from collections import Counter
        return {
            "rules": len(self._rules),
            "by_kind": dict(Counter(r.kind for r in self._rules.values())),
            "rows_yielded": sum(r.rows_yielded for r in self._rules.values()),
        }


# ── validation ────────────────────────────────────────────────────

class Rejected(Exception):
    """A proposed rule failed its gate. Carries why, for the escalation queue."""


try:                                    # `regex` supports a match timeout; `re` does not
    import regex as _re_engine
    _HAS_TIMEOUT = True
except ImportError:                     # pragma: no cover - fallback path
    _re_engine = re
    _HAS_TIMEOUT = False

MATCH_TIMEOUT_S = 0.25


def _safe_regex(pattern: str):
    """Compile a synthesized pattern, refusing the obviously dangerous shapes.

    The static check is a filter, NOT a guarantee. Automata-based ReDoS
    detectors are documented to miss polynomial blowup and anything involving
    lookaround or backreferences, so the real backstop is the per-match timeout
    applied in `_search` — a synthesized pattern runs over thousands of rows,
    and a quadratic one is an outage, not a slow query.
    """
    if len(pattern) > 400:
        raise Rejected("regex implausibly long; likely memorised the sample")
    if _REDOS.search(pattern):
        raise Rejected("regex has nested quantifiers (catastrophic backtracking risk)")
    try:
        return _re_engine.compile(pattern)
    except Exception as e:
        raise Rejected(f"regex does not compile: {e}") from e


def _search(pat, text: str):
    """Run a synthesized pattern under a hard wall-clock cap."""
    try:
        if _HAS_TIMEOUT:
            return pat.search(text, timeout=MATCH_TIMEOUT_S)
        return pat.search(text)
    except TimeoutError as e:
        raise Rejected(
            f"regex exceeded {MATCH_TIMEOUT_S}s on a single row — catastrophic "
            f"backtracking the static check did not catch") from e


# Strings that a careless pattern will happily swallow but that are never assay
# values. Drawn from the failure modes this pipeline already knows about: NMR
# shifts and coupling constants, MS m/z, molecular weights, retention times.
# No OSS tool measures over-generalisation, so this battery is the substitute:
# a rule that matches any of these is describing "a number", not a measurement.
ADVERSARIAL_ROWS = [
    "1 | 1H NMR (CDCl3, 400 MHz) δ 7.35 (d, J = 8.4 Hz, 2H)",
    "2 | MS (ESI) m/z 489.2 [M+H]+",
    "3 | MW 403.53 | calc'd 404.3",
    "4 | RT 1.24 min | purity 98.7%",
    "5 | LCMS: 523.2 (M + H), method QC-ACN-AA-XB",
]


def _validate_bin_key(rule: Rule, table) -> dict:
    """Gate a symbol -> numeric range mapping read out of the patent's own text.

    This is what lets the loop learn a legend GRAMMAR it has never seen. Rather
    than adding a parser per format — `A <= 10 nM; 10 nM < B <= 100 nM`,
    `++++: IC50 >= 1 uM`, `is marked "+++"` — the model reads whichever form the
    patent used and returns the mapping. The gate checks the mapping, not the
    prose, so any future format is covered for free.

    Rejected unless the bins are internally coherent and actually cover the
    grades in the table: a key that explains none of the symbols present is
    either hallucinated or belongs to a different table.
    """
    from ..sources.uspto_assays import _header_rows_of

    # A key may define one scale (`bins`) or several column-scoped ones
    # (`scales`). Validate the union: every bin of every scale must be coherent,
    # and between them they must explain the grades this table actually uses.
    scales = rule.payload.get("scales") or []
    if scales:
        for s in scales:
            if not (s.get("bins") or {}):
                raise Rejected("a scale carries no bins")
            pat = (s.get("match") or "").strip()
            if pat:
                try:
                    re.compile(pat)
                except re.error as e:
                    raise Rejected(f"scale match {pat!r} is not a regex: {e}")
        if sum(1 for s in scales if not (s.get("match") or "").strip()) > 1:
            raise Rejected("more than one unscoped scale — which governs which "
                           "column is then undefined")
        bins = {k: v for s in scales for k, v in (s.get("bins") or {}).items()}
    else:
        bins = rule.payload.get("bins") or {}
    if not bins:
        raise Rejected("bin_key carries no bins")
    present = set()
    _, data = _header_rows_of(table)
    for r in data:
        for c in r:
            v = c.text.strip()
            if v and (len(v) == 1 and v.isalpha() or set(v) == {"+"}):
                present.add(v.upper())
    if not present:
        raise Rejected("no grade symbols found in this table to apply a key to")

    covered = {k.upper() for k in bins} & present
    if not covered:
        raise Rejected(
            f"key defines {sorted(bins)} but the table uses {sorted(present)} — "
            f"it belongs to a different table")

    for sym, b in bins.items():
        lo, hi = b.get("lo"), b.get("hi")
        if lo is None and hi is None:
            raise Rejected(f"bin {sym!r} has neither bound")
        if lo is not None and hi is not None and lo >= hi:
            raise Rejected(f"bin {sym!r} has lo >= hi ({lo} >= {hi})")
        if not b.get("unit"):
            raise Rejected(f"bin {sym!r} has no unit; a range without one is unusable")

    return {"symbols_in_table": sorted(present), "symbols_defined": sorted(bins),
            "coverage": round(len(covered) / len(present), 3), "kind": BIN_KEY}


def _validate_value_pattern(rule: Rule, table) -> dict:
    """Gate a proposed CELL parser. Three conditions, all mandatory.

    A value pattern is the most dangerous rule kind, because it changes how
    every cell in scope is read rather than which cells are read. So it is only
    adopted when it is a strict improvement:

      1. RESCUES — it parses cells the current parser rejects. A pattern that
         adds nothing is not a repair.
      2. NO REGRESSION — on every cell the current parser already reads, it
         must agree on the number. A pattern that rescues 100 cells while
         changing the value of 1,000 others is catastrophic, and it would sail
         through a naive "did coverage go up" check.
      3. NO OVER-REACH — it must still refuse the adversarial battery. This is
         what stops "just match any number" from being adopted, which is the
         obvious and wrong way to make condition 1 pass.
    """
    from ..sources.uspto_assays import _header_rows_of, parse_value

    raw = rule.payload.get("pattern", "")
    pat = _safe_regex(raw)
    if "num" not in (pat.groupindex or {}):
        # A single unnamed capture group is unambiguous — it can only be the
        # number — so promote it rather than failing the call over syntax.
        if pat.groups == 1 and not pat.groupindex:
            raw = raw.replace("(", "(?P<num>", 1)
            pat = _safe_regex(raw)
            rule.payload["pattern"] = raw
        else:
            raise Rejected(
                "value_pattern must capture a named group `num` "
                f"(got groups={pat.groups}, named={list(pat.groupindex)})")

    _, data = _header_rows_of(table)
    cols = rule.payload.get("columns")
    rescued = agreed = 0
    regressions: list[str] = []
    for r in data:
        for i, cell in enumerate(r):
            if cols and i not in cols:
                continue
            s = cell.text.strip()
            if not s:
                continue
            current = parse_value(s)
            m = _search(pat, s)
            proposed = None
            if m:
                try:
                    proposed = float(m.group("num").replace(",", ""))
                except (TypeError, ValueError):
                    proposed = None
            if current and current.get("value_numeric") is not None:
                if proposed is None or abs(proposed - current["value_numeric"]) > 1e-9:
                    regressions.append(f"{s!r}: {current['value_numeric']} -> {proposed}")
                else:
                    agreed += 1
            elif proposed is not None:
                rescued += 1

    if regressions:
        raise Rejected(
            f"pattern changes {len(regressions)} cells the parser already reads "
            f"correctly (e.g. {regressions[0]}) — a value rule must be a strict "
            f"superset, never a reinterpretation")
    if rescued < 3:
        raise Rejected(
            f"pattern rescues only {rescued} unparsed cells; it is not repairing "
            f"anything the current parser cannot already do")
    for probe in ("1H NMR (CDCl3, 400 MHz)", "m/z 489.2 [M+H]+", "QC-ACN-AA-XB",
                  "RT 1.24 min", "Method F"):
        if _search(pat, probe):
            raise Rejected(
                f"pattern also matches {probe!r} — it is matching text, not a "
                f"measurement")
    return {"rescued": rescued, "agreed_with_parser": agreed,
            "regressions": 0, "kind": VALUE_PATTERN}


def validate(rule: Rule, table, *, min_yield: float = 0.5,
             baseline_rows: int = 0, shown_rows: int = 4) -> dict:
    """Run a proposed rule against the table it came from. Raise on failure.

    Three things are checked, in increasing order of what they catch:

      1. It executes at all (compiles, indexes in range, no ReDoS shape).
      2. It yields records on a MAJORITY of the table's data rows. A rule that
         fires on the two rows the model was shown and nothing else has
         memorised the sample rather than described the layout.
      3. What it yields is shaped like a measurement — ids that look like ids,
         values that parse as numbers. Without this a rule can happily return
         the molecular-weight column and call it an IC50.

    Returns evidence to store alongside the rule.
    """
    from ..sources.uspto_assays import _CID_PAT, _VALUE_PAT, _header_rows_of

    if rule.kind in (NOT_ASSAY, ESCALATE):
        return {"checked": "none required", "kind": rule.kind}

    if rule.kind == VALUE_PATTERN:
        return _validate_value_pattern(rule, table)

    if rule.kind == BIN_KEY:
        return _validate_bin_key(rule, table)

    _, data = _header_rows_of(table)
    all_rows = [r for r in data if any(c.text.strip() for c in r)]
    if not all_rows:
        raise Rejected("table has no data rows to validate against")

    # Score on rows the model was NOT shown. Automated-repair studies find
    # patches that pass 100% of the tests that motivated them generalise to
    # only ~69-75% of held-out tests, so grading a rule on its own prompt
    # sample measures memorisation, not generalisation.
    rows = all_rows[shown_rows:] or all_rows
    held_out = len(all_rows) - len(rows) if rows is not all_rows else 0

    hits = 0
    bad_cid = bad_val = 0

    if rule.kind == COLUMN_MAP:
        cid_i = rule.payload.get("cid")
        assays = rule.payload.get("assays") or []
        if not isinstance(cid_i, int) or not assays:
            raise Rejected("column_map needs an integer `cid` and ≥1 assay column")
        idxs = [cid_i] + [a.get("index") for a in assays]
        if any(not isinstance(i, int) or i < 0 or i >= table.n_cols for i in idxs):
            raise Rejected(f"column index out of range for a {table.n_cols}-column table")
        if cid_i in [a.get("index") for a in assays]:
            raise Rejected("the id column is also listed as an assay column")

        # Semantic gate. Everything above is structural, and a molecular-weight
        # column passes every structural check: the values are numbers, the ids
        # are ids, coverage is 100%. Recording MW as an IC50 is precisely the
        # corruption this pipeline exists to prevent, and no amount of shape
        # checking will catch it — the header has to be consulted.
        from ..sources.uspto_assays import (
            MS, MW, NMR, RT, STRUCTURE, classify_column, merge_header,
        )
        hdr_rows, _ = _header_rows_of(table)
        headers = merge_header(table, hdr_rows)
        banned = {MW, MS, NMR, RT, STRUCTURE}
        for a in assays:
            i = a["index"]
            head = headers[i] if i < len(headers) else ""
            kind = classify_column(head, []).kind
            if kind in banned:
                raise Rejected(
                    f"column {i} is headed {head.strip()!r}, which reads as "
                    f"{kind} — refusing to record it as an assay")
        # An unlabelled column is the ambiguous case: we cannot read its header
        # because it has none. Accept it only when the surrounding prose says
        # this table is assay data, mirroring the caption gate used elsewhere.
        from ..sources.uspto_assays import caption_assay_hint, table_legend
        unlabelled = [a["index"] for a in assays
                      if not (headers[a["index"]] if a["index"] < len(headers) else "").strip()]
        if unlabelled:
            ctx = f"{table.caption} {table_legend(table)}"
            if not caption_assay_hint(ctx)[0]:
                raise Rejected(
                    f"columns {unlabelled} have no header and no assay language in "
                    f"the caption or legend — cannot confirm they are measurements")
        for r in rows:
            if len(r) <= cid_i:
                continue
            cid = r[cid_i].text.strip()
            if not cid:
                continue
            if not _CID_PAT.match(cid):
                bad_cid += 1
                continue
            got = False
            for a in assays:
                i = a["index"]
                if len(r) > i and _VALUE_PAT.match(r[i].text.strip()):
                    got = True
            if got:
                hits += 1
            else:
                bad_val += 1

    elif rule.kind == ROW_REGEX:
        pat = _safe_regex(rule.payload.get("pattern", ""))
        if "cid" not in (pat.groupindex or {}):
            raise Rejected("row_regex must capture a named group `cid`")
        if not any(g.startswith("value") for g in pat.groupindex):
            raise Rejected("row_regex must capture at least one `value…` group")
        # Over-generalisation gate: a pattern that fires on an NMR shift or an
        # m/z value has learned "a number appears here", which is how molecular
        # weights end up recorded as IC50s.
        for probe in ADVERSARIAL_ROWS:
            hit = _search(pat, probe)
            if hit and hit.group("cid"):
                raise Rejected(
                    f"pattern also matches a known non-assay row ({probe[:46]}…) — "
                    f"it is matching numbers, not measurements")
        for r in rows:
            line = " | ".join(c.text.strip() for c in r)
            m = _search(pat, line)
            if not m:
                continue
            if not _CID_PAT.match((m.group("cid") or "").strip()):
                bad_cid += 1
                continue
            vals = [v for g, v in m.groupdict().items()
                    if g.startswith("value") and v is not None]
            if vals and all(_VALUE_PAT.match(v.strip()) for v in vals):
                hits += 1
            else:
                bad_val += 1
    else:
        raise Rejected(f"unknown rule kind {rule.kind!r}")

    coverage = hits / len(rows)
    if coverage < min_yield:
        raise Rejected(
            f"rule fired on {hits}/{len(rows)} held-out rows ({coverage:.0%}); below "
            f"the {min_yield:.0%} floor — it describes the sample, not the layout")
    if bad_cid > hits:
        raise Rejected(f"more malformed ids ({bad_cid}) than good rows ({hits})")

    # Anti-deletion guard. In automated program repair, the overwhelming
    # majority of patches that passed their gates "worked" by deleting the
    # failing behaviour — only 2 of 105 were genuinely correct. The extraction
    # analogue is a rule that resolves a gap by quietly parsing FEWER rows than
    # the parser already manages. A repair must add coverage, not merely stop
    # complaining, so it has to beat the baseline it claims to fix.
    if baseline_rows and hits <= baseline_rows:
        raise Rejected(
            f"rule yields {hits} rows but the existing parser already gets "
            f"{baseline_rows} — a repair must increase coverage, not reduce it")

    return {
        "rows_total": len(rows), "rows_matched": hits,
        "coverage": round(coverage, 3),
        "rejected_ids": bad_cid, "rejected_values": bad_val,
        "held_out_excluded": held_out, "baseline_rows": baseline_rows,
    }
