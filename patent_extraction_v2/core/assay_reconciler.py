"""Canonical assay/target reconciliation.

Maps a patent assay fragment ("P2X3 IC50 (μM)", "CBP IC50 (μM gmean)",
"RORγ Binding IC50 μM") to a canonical target name, so the rest of the pipeline
can stop being confused by surface-form variation:

  - `assay_name_guard` gains a POSITIVE signal — a fragment that resolves to a
    known target is definitely a real assay (today the guard only rejects
    known-bad names).
  - the hand-check workbook's BDB cross-reference can compare canonical ↔
    canonical instead of "P2X3 IC50 (μM)" vs "P2X purinoceptor 3", lifting the
    match rate past name fragmentation.
  - the pattern header-anchor can treat a known-target token as high-confidence.

How it works (your design: "search the dataset with our assay fragment, find
the canon name; build as we ingest"):

  Build ONCE → a compact token→canonical index, cached on disk. Thereafter
  `reconcile(fragment)` is a hashmap lookup — no scanning of any big file.

Sources, joined on UniProt ID:
  - BindingDB_All.tsv (7.9 GB)  → every MEASURED target: Target Name, UniProt
    Recommended Name, Entry-Name abbreviation (CBP_HUMAN → CBP), UniProt ID.
    Pre-extracted once to output/reference/bdb_targets.tsv.
  - HGNC complete set           → gene symbol ↔ alias_symbol / prev_symbol;
    joined to BDB targets via UniProt ID so a patent abbreviation that is a
    gene alias (P2X3 → P2RX3, Nav1.5 → SCN5A) resolves to the BDB target.
  - curated cell-line seed      → MV4;11, MOLM13, H358, … (cell-based readouts).
  - observed aliases            → accumulated as patents are ingested.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_REF_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "reference"
_BDB_TARGETS = _REF_DIR / "bdb_targets.tsv"           # cut from BindingDB_All.tsv
_HGNC = _REF_DIR / "hgnc_complete_set.txt"
_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "assay_target_index.json"

_GREEK = {"α":"alpha","β":"beta","γ":"gamma","δ":"delta","ε":"epsilon","ζ":"zeta",
          "η":"eta","θ":"theta","ι":"iota","κ":"kappa","λ":"lambda","μ":"mu",
          "ν":"nu","ξ":"xi","ο":"omicron","π":"pi","ρ":"rho","σ":"sigma","τ":"tau",
          "υ":"upsilon","φ":"phi","χ":"chi","ψ":"psi","ω":"omega"}


def _norm(s: str) -> str:
    """Lowercase + Greek→Latin + strip non-alphanumerics (matches the
    normalization used by the pattern library's header anchor)."""
    return re.sub(r"[^a-z0-9]", "", "".join(_GREEK.get(c, c) for c in (s or "").lower()))


# Assay-type / method / unit / organism words to peel off a fragment so the
# distinguishing TARGET token is what's left.
_STRIP_WORDS = {
    "ic50", "ec50", "ki", "kd", "cc50", "ki50", "ed50", "gi50", "ld50",
    "binding", "affinity", "activity", "potency", "inhibition", "agonist",
    "antagonist", "gmean", "geomean", "mean", "value", "values", "assay",
    "flux", "ep", "fret", "htrf", "elisa", "tr", "fl",
    "um", "μm", "nm", "mm", "pm", "molar",
    "human", "rat", "mouse", "rabbit", "dog", "monkey", "hu", "h",
    "cell", "cellular", "viability", "proliferation",
}
# Compared against `_norm(token)`, so store the normalized forms (μM → "mum").
_STRIP_NORM = {_norm(w) for w in _STRIP_WORDS}
# Ambiguous bare tokens that mis-resolve when matched ALONE: the index keys
# them to the WRONG target because the surface form collides with an unrelated
# gene/abbreviation. Skipping them lets a more-specific sibling token win
# (e.g. "A2A adenosine receptor" → match on "A2A" → Adenosine receptor A2a,
# instead of "adenosine" → Adenosine KINASE) or, when there's no better token,
# fall through to None so the caller slugifies the surface form verbatim.
#   adenosine → Adenosine kinase  (vs the receptor the fragment usually means)
#   flap      → Flap endonuclease 1  (vs ALOX5AP = 5-LO-activating protein)
#   a1        → ATPase H+ V0 subunit a1  (vs Adenosine receptor A1)
_AMBIGUOUS_TOKENS = {_norm(w) for w in ("adenosine", "flap", "a1")}
# Curated cell-line seed (build-as-ingest extends this).
_CELL_LINE_SEED = {
    "mv411": "MV4;11 (AML cell line)", "mv4;11": "MV4;11 (AML cell line)",
    "molm13": "MOLM-13 (AML cell line)", "molm-13": "MOLM-13 (AML cell line)",
    "h358": "NCI-H358 (lung cell line)", "ncih358": "NCI-H358 (lung cell line)",
    "a375": "A375 (melanoma cell line)", "hct116": "HCT116 (colon cell line)",
}

# Greek-letter WORDS. A bare Greek letter is never a standalone target — it
# only qualifies a preceding family token ("PI3K" + "delta" → the delta
# isoform). Matching "delta" alone wrongly hit a gene aliased 'delta'
# (proteasome 20S subunit beta 6), so these are never tried in isolation;
# `_target_tokens` folds them into the preceding token instead.
_GREEK_WORDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
}
# Curated combined-token → gene-symbol aliases for isoforms whose "{family}
# {greek}" surface form (or "{family}{symbol}", e.g. PI3Kδ) doesn't carry the
# gene symbol the index is keyed on. Resolved THROUGH the index so the
# canonical name stays in sync with BindingDB. Both "PI3K delta" (combined to
# pi3kdelta) and "PI3Kδ" (_norm → pi3kdelta) land here.
_COMBINED_ALIASES = {
    "pi3kalpha": "pik3ca", "pi3kbeta": "pik3cb",
    "pi3kgamma": "pik3cg", "pi3kdelta": "pik3cd",
}


class _Reconciler:
    def __init__(self) -> None:
        self._index: dict[str, str] = {}
        self._observed: dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if _INDEX_PATH.exists():
            try:
                d = json.loads(_INDEX_PATH.read_text())
                self._index = d.get("tokens", {})
                self._observed = d.get("observed", {})
            except (ValueError, OSError) as e:
                logger.warning("assay_reconciler: index load failed: %r", e)
        self._loaded = True

    # ── reconciliation (the hot path — O(1) hashmap) ────────────────
    def reconcile(self, fragment: str) -> tuple[str, str] | None:
        """Return (canonical_target, matched_token) for `fragment`, or None.
        Pure lookup — never scans a source file."""
        self._load()
        for tok in self._target_tokens(fragment):
            n = _norm(tok)
            if not n:
                continue
            hit = (self._index.get(n)
                   or self._index.get(_COMBINED_ALIASES.get(n, ""))
                   or self._observed.get(n)
                   or _CELL_LINE_SEED.get(n))
            if hit:
                return hit, tok
        return None

    @staticmethod
    def _target_tokens(fragment: str) -> list[str]:
        """Distinguishing target tokens from a fragment, longest first (so
        `P2X3` is tried before `P2`). Token class includes `;` so cell-line
        names like `MV4;11` stay intact. A bare Greek-letter word
        (`alpha`/`delta`/…) is never emitted on its own — it would wrongly
        match a gene aliased by that letter (e.g. `delta` → proteasome) — but
        is folded into the preceding family token (`PI3K` + `delta` →
        `PI3Kdelta`) so the isoform still resolves via `_COMBINED_ALIASES`."""
        toks: list[str] = []
        prev_target: str | None = None
        for raw in re.findall(r"[A-Za-zα-ωΑ-Ω][A-Za-zα-ωΑ-Ω0-9.;\-]+", fragment or ""):
            if _norm(raw) in _STRIP_NORM or len(raw) < 2:
                continue
            if _norm(raw) in _AMBIGUOUS_TOKENS:
                continue  # never matched alone — see _AMBIGUOUS_TOKENS
            if _norm(raw) in _GREEK_WORDS:
                if prev_target is not None:
                    toks.append(prev_target + raw)  # PI3K + delta → PI3Kdelta
                # never emit the bare Greek word as a standalone token
                continue
            # Identifier filter: a real target token carries an uppercase
            # letter or a digit (CBP, A2A, CYP3A4, hERG, P2X3, Nav1.5). Plain
            # all-lowercase words ("wild", "type", "adenosine", "receptor")
            # are English and only match the index by coincidence (→ wrong
            # target, e.g. "type" → sarcoglycan gamma), so are never matched
            # alone. Greek words are handled above (folded into the prefix).
            if not any(c.isupper() or c.isdigit() for c in raw):
                continue
            toks.append(raw)
            prev_target = raw
        return sorted(set(toks), key=lambda t: -len(t))

    # ── build-as-ingest ─────────────────────────────────────────────
    def observe(self, name: str, canonical: str | None = None) -> None:
        """Accumulate a validated assay name. If it already resolves, no-op;
        otherwise record its target token → canonical (defaults to the name
        itself when no canonical is supplied)."""
        self._load()
        if self.reconcile(name):
            return
        toks = self._target_tokens(name)
        if not toks:
            return
        key = _norm(toks[0])
        if key and key not in self._observed:
            self._observed[key] = canonical or name
            self._persist_observed()

    def _persist_observed(self) -> None:
        try:
            d = {"tokens": self._index, "observed": self._observed}
            _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            _INDEX_PATH.write_text(json.dumps(d))
        except OSError as e:
            logger.warning("assay_reconciler: observed-persist failed: %r", e)


_R = _Reconciler()
reconcile = _R.reconcile
observe = _R.observe


# ── one-time index build (no LLM, no network beyond the cached files) ──

def build_index() -> dict:
    """Build the token→canonical index from BDB targets + HGNC. Run once;
    cached to `assay_target_index.json`. Returns build stats."""
    if not _BDB_TARGETS.exists():
        raise FileNotFoundError(
            f"{_BDB_TARGETS} missing — extract it with:\n"
            f"  awk -F'\\t' 'NR>1 && !s[$7 FS $43 FS $44 FS $45]++ "
            f"{{print $7\"\\t\"$43\"\\t\"$44\"\\t\"$45}}' BindingDB_All.tsv "
            f"> {_BDB_TARGETS}")
    tokens: dict[str, str] = {}
    uniprot_to_canon: dict[str, str] = {}

    def add(tok: str, canon: str) -> None:
        n = _norm(tok)
        if len(n) >= 2 and n not in tokens:
            tokens[n] = canon

    # 1) BindingDB measured targets
    n_targets = 0
    for line in _BDB_TARGETS.read_text(errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        target_name, recname, entryname, uniprot_id = parts[0], parts[1], parts[2], parts[3]
        canon = (recname or target_name).strip()
        if not canon:
            continue
        n_targets += 1
        add(target_name, canon)
        add(recname, canon)
        # distinctive words of the recommended name (≥3 chars, skip generic)
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", recname or ""):
            if _norm(w) not in _STRIP_WORDS and _norm(w) not in (
                "protein", "receptor", "kinase", "subunit", "type", "alpha",
                "beta", "gamma", "delta", "channel", "factor", "domain",
                "containing", "related", "nuclear", "serine", "threonine"):
                add(w, canon)
        # entry-name abbreviation prefix:  CBP_HUMAN → CBP
        if "_" in (entryname or ""):
            add(entryname.split("_")[0], canon)
        # UniProt IDs (primary + secondary) → canonical, for the HGNC join
        for up in re.split(r"[\s,;|]+", uniprot_id or ""):
            up = up.strip()
            if up:
                uniprot_to_canon.setdefault(up, canon)

    # 2) HGNC gene symbol ↔ aliases, joined to BDB targets via UniProt ID
    n_hgnc_join = 0
    if _HGNC.exists():
        hl = _HGNC.read_text(errors="replace").splitlines()
        if hl:
            hdr = hl[0].split("\t")
            col = {name: i for i, name in enumerate(hdr)}
            ci_sym, ci_name = col.get("symbol"), col.get("name")
            ci_alias, ci_prev = col.get("alias_symbol"), col.get("prev_symbol")
            ci_up = col.get("uniprot_ids")
            for line in hl[1:]:
                p = line.split("\t")
                if len(p) <= max(filter(None, [ci_sym, ci_name, ci_alias, ci_prev, ci_up])):
                    continue
                def g(i): return (p[i].strip().strip('"') if i is not None and i < len(p) else "")
                ups = [u for u in re.split(r"[\s,;|]+", g(ci_up)) if u]
                canon = next((uniprot_to_canon[u] for u in ups if u in uniprot_to_canon), None)
                # If this gene's UniProt isn't a BDB target, fall back to the
                # HGNC approved name so the symbol still resolves to *something*.
                target = canon or g(ci_name)
                if not target:
                    continue
                if canon:
                    n_hgnc_join += 1
                for field in (g(ci_sym), g(ci_alias), g(ci_prev)):
                    for sym in re.split(r"[|,;]\s*", field):
                        sym = sym.strip().strip('"')
                        if sym:
                            add(sym, target)

    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_observed = {}
    if _INDEX_PATH.exists():
        try:
            existing_observed = json.loads(_INDEX_PATH.read_text()).get("observed", {})
        except (ValueError, OSError):
            pass
    _INDEX_PATH.write_text(json.dumps({"tokens": tokens, "observed": existing_observed}))
    # reset the in-memory singleton so it reloads
    _R._loaded = False
    stats = {"bdb_targets": n_targets, "uniprot_ids": len(uniprot_to_canon),
             "hgnc_joined": n_hgnc_join, "index_tokens": len(tokens),
             "index_path": str(_INDEX_PATH)}
    logger.info("assay_reconciler: built index %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(build_index(), indent=2))
