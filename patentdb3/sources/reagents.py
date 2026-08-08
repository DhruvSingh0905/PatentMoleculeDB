"""Tell a laboratory reagent apart from a patent's own compound.

WHY THIS EXISTS
----------------
`iupac_names.extract_names` resolves every OPSIN-parseable name in a patent's
description, with a 12-character minimum seed (`config.IUPAC_MIN_SEED`). That
minimum exists to reject bare substituent fragments (`ethyl`, `2-chloro`), but
it does nothing to stop a full, valid IUPAC name for a solvent, base, acid or
coupling reagent from being admitted as if it were one of the patent's own
compounds — a "Preparation of Intermediate" paragraph names `triethylamine`
and `dichloromethane` exactly the way it names the intermediate itself, and
OPSIN parses all of them equally well.

A character-count proxy (`>=45 chars`) was tried first and rejected as a
proxy, not a rule:

    patent        all structures    >=45 chars
    US8952177          238             180
    US10214537       1,018             887
    US10544143         222             165

Length correlates with "is a real target/intermediate" only because complex
drug-like scaffolds tend to have long names — but simple, short building
blocks the patent legitimately depends on (`4-bromobenzylamine`,
`(3-bromophenyl)boronic acid`, `4-fluoropiperidine`) are just as short as a
solvent name, and a handful of reagents used in this corpus are LONGER than
45 characters when spelled out in full rather than abbreviated
(`benzotriazole-1-yl-oxy-tris-(dimethylamino)-phosphonium` = PyBOP,
`2,2'-bis(diphenylphosphino)-1,1'-binaphthalene` = BINAP). Measured on the
same 1,478 structures across the three patents above: the length proxy
discards 246 structures at <45 chars. Manual review of that discarded set
(printed in full during development) shows the overwhelming majority — 203 of
246 — are genuine synthetic building blocks specific to the patent's own
chemistry (halogenated anilines/pyridines/phenols, boronic acids, Boc/Cbz-
protected amines, azetidines, piperidines), not reagents. Meanwhile 11
spelled-out catalysts/ligands/coupling reagents clear 45 characters and would
have been silently kept.

THE RULE
--------
Two tiers, in order:

1. **A curated name lexicon** (`REAGENT_LEXICON`, ~230 entries) of solvents,
   bases, acids, coupling/activating reagents, reducing and oxidizing agents,
   catalysts and ligands, drying/workup chemicals, inert gases, common
   inorganic salts, and silylating/protecting reagents — the closed,
   well-known vocabulary of organic synthesis that appears in essentially any
   med-chem patent's experimental section, independent of what that patent
   claims. Matching is on the FULL normalized name (case-folded, unicode
   quote/dash variants collapsed, whitespace collapsed), never a substring —
   substring matching would misfire on real compounds that merely contain a
   reagent-like fragment (e.g. a claimed compound bearing a `(trifluoro-
   methyl)silyl` protecting group is not "silane"). A small set of salt
   suffixes (` hydrochloride`, ` hydrate`, ` mesylate`, ...) is stripped
   before a second lookup, so `1H-benzo[d][1,2,3]triazol-1-ol hydrate`
   (HOBt hydrate) still matches without every salt form having to be listed.

2. **A narrow structural backstop** — heavy-atom count <= 3, computed by
   RDKit from the resolved SMILES — for whatever isn't in the lexicon at all:
   OPSIN boundary debris (`hydrochloride`, `carbon-carbon` parsed as
   fragments, not molecules) and bare metal/ion names (`palladium(II)`,
   `titanium(IV)`). Nothing at 3 heavy atoms or fewer has ever been a claimed
   or exemplified compound in this corpus; every HAC<=3 item found during
   development was either a catalyst, an extraction artifact, or water/
   acetone-class solvent debris. The backstop is deliberately NOT a general
   "small = reagent" rule: `2,6-lutidine` and `4-methylmorpholine` are real
   bases sitting at HAC 7-8, right next to genuine short intermediates like
   `4-fluoropiperidine` (HAC 7) and `oxetan-3-amine` (HAC 5) that must NOT be
   flagged. Above HAC 3, only the lexicon decides.

MEASURED, on the 1,478 OPSIN-accepted structures from the three patents above
(the only ones read for this module; the corpus-wide false-negative rate is
NOT measured — see below):

    patent        structures   flagged reagent   of which >=45 chars
    US8952177          238            5                  0
    US10214537       1,018           37                  8
    US10544143         222           12                  3
    TOTAL            1,478           54                 11

54/1,478 (3.7%) are classified as reagent-or-fragment. Net effect versus the
length proxy: 1,424 kept as compounds here vs. 1,232 kept by `>=45 chars` —
this rule recovers 203 real building blocks the length cutoff discarded,
while ALSO catching 11 spelled-out reagents the length cutoff would have kept.

WHAT IT GETS WRONG, both directions:

- **False negatives (reagent not caught) — the main known gap.** This is an
  exact-match lexicon; anything not on the list passes through labeled
  `"compound"` with no warning. Development found and fixed several real
  misses this way (spelled-out Buchwald ligands — XPhos, SPhos, RuPhos;
  BINAP; the HATU/PyBOP cationic cores written out instead of abbreviated;
  TMSCF3 as `(trifluoromethyl)trimethylsilane`) by grep-auditing the
  unflagged output for chemist-recognizable keywords
  (`silane|phosphin|boran|chromate|...`) — a manual process with no
  guarantee of completeness. The lexicon was built and audited against only
  3 of the 137 cached patents; an unmeasured number of reagents specific to
  other chemistry (e.g. a patent doing heavy fluorination, glycosylation, or
  peptide coupling chemistry not represented in this sample) will not be on
  it and will be silently kept as "compound."
- **False positives (real compound flagged as reagent) — not observed here,
  but structurally possible.** Exact-name matching means a patent whose own
  subject IS one of the lexicon substances (e.g. a patent about pyridine
  analogs that names bare `pyridine` itself as a reference compound) would
  have that occurrence labeled `"reagent"`. None of the 54 flags in the
  measured set were disputable on manual review — every one is a generic
  synthesis chemical (solvent, base, catalyst, ligand, coupling reagent) with
  no synthetic-target role in its patent — but the risk is real and unmeasured
  outside this sample. Because this module LABELS rather than deletes (see
  below), a false positive here costs a wrong tag, not a lost molecule.
- **Neither reagent nor compound.** A few OPSIN-accepted spans are not real
  substances at all — `water-acetonitrile` (an HPLC mobile-phase mention,
  not a molecule), `cyclohexanedicarboxylic` (a truncated fragment) — and
  this module does not catch them; that is a boundary-extraction defect in
  the candidate-span search, not a reagent/compound judgment, and is out of
  scope here.

THE LABEL DESIGN — why two flag labels, not more
--------------------------------------------------
This module LABELS every structure; it never deletes one. A caller that used
to do `if is_reagent: drop(nc)` is exactly the shape this refuses to
reintroduce: a resolved structure adjacent to the target compound —
intermediate, building block, starting material, or purchased reagent — is
useful even with no assay value, and which of those it is is a judgment this
module has no evidence to make. A boronic acid could be bought off a shelf or
built in scheme 2 of the very same patent; nothing here can tell those apart,
so nothing here tries to.

What the two tiers above CAN support, and no more, is exactly two flag
labels, plus the default pass-through:

- `"compound"` — matched neither tier. The default; not evidence the
  structure IS one of the patent's own compounds, only that nothing here
  says otherwise.
- `"reagent"` — matched `REAGENT_LEXICON` by exact normalized name. This is
  the strong claim: a human curated this list, entry by entry, and manual
  review of all 54 hits in the measured set found zero disputable false
  positives. The lexicon's ~12 internal categories (solvent, base, acid, ...)
  are curator bookkeeping — which comment block a name was typed under while
  building the list — not independently-evidenced classes; they ride along
  in `reason` for audit, but are deliberately NOT promoted to separate
  top-level labels, because nothing here re-derives or checks them from the
  chemistry itself (they were never adversarially tested the way the
  reagent/compound split was).
- `"trace_fragment"` — matched the structural HAC<=3 backstop and NOT the
  lexicon. This is a weaker and DIFFERENT claim than `"reagent"`, which is
  why it is not folded into that label: this tier catches "too small to be a
  claimed or exemplified compound in this corpus," and what actually sits
  behind that, per the docstring above, is a mix of real catalysts/ions
  (`palladium(II)`) and pure extraction debris that is not a molecule at all
  (`carbon-carbon`, a bracket-balancing artifact). Calling debris a
  "reagent" would overclaim what was identified; `trace_fragment` says only
  what was measured — small enough, by a fixed threshold, to not be a
  target — without asserting it is a named, known chemical.

Padding this out to the lexicon's ~12 categories, or splitting further into
"intermediate" vs "starting material" vs "byproduct," is exactly what the
evidence here cannot support: those distinctions were never audited, and for
several of them (lexicon category boundaries) the curator's own bucketing is
not mutually exclusive in real chemistry. Two flag labels is what this
lexicon and this backstop actually separate; shipping more would be
decoration, not measurement.

USAGE
-----
Not wired into `iupac_names.py` or `verify.py` — those are owned elsewhere,
and the whole point of labeling instead of deleting is that the consumer
decides what to do with each label. Call `classify(name, smiles)` per
`NamedCompound` and inspect `.label`:

    from patentdb3.sources.reagents import classify
    verdict = classify(nc.name, nc.smiles)
    # verdict.label in {"compound", "reagent", "trace_fragment"}
    # verdict.reason explains WHY (see ReagentVerdict docstring)
    ...  # nc keeps flowing through either way; the caller decides what to
         # do with a non-"compound" label, including nothing at all

`label_of(name, smiles="")` is a convenience wrapper returning just the label
string. `smiles` is optional; without it, only the lexicon tier runs (no
structural backstop, no crash — RDKit is imported lazily and only when a
SMILES string is actually given).

DETERMINISM
-----------
Same `(name, smiles)` in must produce the same `(label, reason)` out, every
run, on any machine. Concretely:

- No randomness, no sampling, no wall-clock or locale dependence anywhere in
  this module.
- `REAGENT_LEXICON` is looked up by exact key (`dict.__getitem__` /
  `in`), never iterated, so its build order (dict-literal insertion via
  `_add`) cannot leak into the answer.
- `_SALT_SUFFIXES` IS iterated, but it is a `list`, not a `set` — Python
  guarantees list iteration order, and the order was already chosen
  deliberately (longer, more specific suffixes like `" acetate salt"` are
  tried before the bare `" salt"` they'd otherwise be shadowed by).
- The RDKit tier (`_heavy_atom_count`) is a plain parse (`Chem.MolFromSmiles`)
  followed by a count over the parsed atom list (`GetNumHeavyAtoms`). Neither
  step samples, seeds, hashes, or canonicalizes anything — the count is
  invariant to the parser's internal atom ordering, which is itself fixed by
  the input string, not by process/thread/machine state. Nothing here uses
  RDKit's canonical-SMILES generation (the one RDKit facility with
  version-sensitive output); only atom counting. If RDKit is not importable,
  the tier is skipped, not approximated — never a different, less-safe
  computation standing in.
- See `patentdb3/tests/test_reagents.py::test_classification_is_deterministic`
  for the proof: the full lexicon+corpus is classified twice in-process and
  once more in a fresh subprocess interpreter, and all three outputs are
  compared byte-for-byte.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Unicode variants of quote/prime and dash characters that show up in patent
# text depending on the source's typesetting (`1,1'-...` vs `1,1′-...`).
# Both sides of every lexicon match go through this, so either spelling
# resolves to the same key.
_QUOTE_DASH_MAP = str.maketrans({
    "′": "'", "’": "'", "ʹ": "'", "`": "'",   # primes/quotes
    "–": "-", "—": "-",                             # en/em dash
})


def _normalize(name: str) -> str:
    s = name.strip().translate(_QUOTE_DASH_MAP)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,;:-").lower()


# Common salt/hydrate suffixes a building block carries that a bare reagent
# name in the lexicon would not. Stripped once, left-to-right, so
# `X hydrochloride` is tested against `X` without every salt form of every
# reagent having to be listed by hand. Order matters and is fixed (list, not
# set): `" acetate salt"` must be tried before the bare `" salt"` it would
# otherwise be shadowed by — see DETERMINISM above.
_SALT_SUFFIXES = [
    " hydrochloride", " hydrobromide", " hydroiodide", " sulfate", " sulphate",
    " trihydrate", " dihydrate", " hydrate", " mesylate", " tosylate",
    " oxalate", " citrate", " maleate", " fumarate", " acetate salt", " salt",
]

REAGENT_LEXICON: dict[str, str] = {}


def _add(category: str, *names: str) -> None:
    for n in names:
        REAGENT_LEXICON[_normalize(n)] = category


# ── Solvents ─────────────────────────────────────────────────────────────
_add("solvent",
     "water", "methanol", "ethanol", "propan-2-ol", "isopropanol",
     "isopropyl alcohol", "butan-1-ol", "butanol", "tert-butanol",
     "tert-butyl alcohol", "2-methylpropan-2-ol",
     "dichloromethane", "chloroform", "chloroform-d", "carbon tetrachloride",
     "tetrahydrofuran", "2-methyltetrahydrofuran", "1,4-dioxane", "dioxane",
     "diethyl ether", "tert-butyl methyl ether", "methyl tert-butyl ether",
     "t-butyl-methylether", "ethyl acetate", "acetone", "propan-2-one",
     "acetonitrile", "n,n-dimethylformamide", "n,n-dimethylacetamide",
     "dimethyl sulfoxide", "n-methyl-2-pyrrolidone", "n-methylpyrrolidinone",
     "n-methylpyrrolidone", "hexane", "hexanes", "heptane", "heptanes",
     "pentane", "cyclohexane", "toluene", "benzene", "xylene", "xylenes",
     "1,2-dichloroethane", "chlorobenzene", "nitromethane",
     "2,2,2-trifluoroethanol", "hexafluoropropan-2-ol",
     "hexafluoroisopropanol", "petroleum ether", "butan-2-one",
     "methyl ethyl ketone", "1,2-dimethoxyethane", "glyme", "diglyme",
     "sulfolane", "hexamethylphosphoramide", "ethylene glycol",
     "ethane-1,2-diol", "propylene glycol", "n,n-dimethylglycine")

# ── Bases ────────────────────────────────────────────────────────────────
_add("base",
     "triethylamine", "n,n-diisopropylethylamine",
     "n-ethyl-n-isopropylpropan-2-amine", "hunig's base", "pyridine",
     "2,6-lutidine", "2,6-dimethylpyridine", "4-dimethylaminopyridine",
     "4-(dimethylamino)pyridine", "n-methylmorpholine", "4-methylmorpholine",
     "imidazole", "1,8-diazabicyclo[5.4.0]undec-7-ene",
     "1,5-diazabicyclo[4.3.0]non-5-ene", "sodium hydroxide",
     "potassium hydroxide", "lithium hydroxide", "sodium carbonate",
     "potassium carbonate", "cesium carbonate", "caesium carbonate",
     "sodium bicarbonate", "potassium bicarbonate",
     "sodium hydrogen carbonate", "sodium hydride", "potassium hydride",
     "potassium tert-butoxide", "sodium tert-butoxide",
     "lithium diisopropylamide", "n-butyllithium", "tert-butyllithium",
     "sec-butyllithium", "lithium bis(trimethylsilyl)amide",
     "sodium bis(trimethylsilyl)amide", "bis(trimethylsilyl)amide",
     "potassium bis(trimethylsilyl)amide",
     "2-tert-butyl-1,1,3,3-tetramethylguanidine", "ammonium hydroxide",
     "ammonia", "n,n,n',n'-tetramethylethylenediamine")

# ── Acids ────────────────────────────────────────────────────────────────
_add("acid",
     "hydrochloric acid", "sulfuric acid", "trifluoroacetic acid",
     "acetic acid", "formic acid", "p-toluenesulfonic acid",
     "4-methylbenzenesulfonic acid", "methanesulfonic acid",
     "phosphoric acid", "boric acid", "nitric acid", "hydrobromic acid",
     "camphorsulfonic acid", "2-bromoacetic acid", "acetic anhydride")

# ── Coupling / activating reagents ──────────────────────────────────────
# Includes both the common abbreviation-adjacent names AND the spelled-out
# forms actually seen in patent text (HATU/PyBOP's cationic core written out
# rather than abbreviated) — the abbreviation itself is usually too short to
# ever reach OPSIN as a seed, so the lexicon has to carry the long form.
_add("coupling_activating",
     "hatu", "hbtu", "tbtu", "hobt", "1-hydroxybenzotriazole",
     "1h-benzo[d][1,2,3]triazol-1-ol", "1-hydroxy-7-azabenzotriazole",
     "hoat", "edc", "edci", "1-ethyl-3-(3-dimethylaminopropyl)carbodiimide",
     "dcc", "n,n'-dicyclohexylcarbodiimide", "dic",
     "n,n'-diisopropylcarbodiimide", "pybop", "t3p",
     "propylphosphonic anhydride", "cdi", "1,1'-carbonyldiimidazole",
     "thionyl chloride", "oxalyl chloride", "phosphorus oxychloride",
     "phosphorus pentachloride", "di-tert-butyl dicarbonate",
     "n,n'-disuccinimidyl carbonate", "benzyl chloroformate",
     "methanesulfonyl chloride", "p-toluenesulfonyl chloride",
     "tosyl chloride", "4-methylbenzenesulfonyl chloride",
     "benzotriazole-1-yl-oxy-tris-(dimethylamino)-phosphonium",
     "tetramethyl-o-(7-azabenzotriazol-1-yl)uronium",
     "acetyl chloride", "benzoyl chloride", "phosgene", "triphosgene")

# ── Reducing agents ──────────────────────────────────────────────────────
_add("reducing",
     "sodium borohydride", "lithium aluminum hydride",
     "lithium aluminium hydride", "sodium cyanoborohydride",
     "sodium triacetoxyborohydride", "lithium borohydride",
     "diisobutylaluminum hydride", "diisobutylaluminium hydride", "borane",
     "borane-tetrahydrofuran complex", "borane dimethyl sulfide complex",
     "hydrazine", "hydrazine hydrate", "raney nickel", "palladium on carbon",
     "platinum oxide", "platinum(iv) oxide", "tin(ii) chloride", "zinc",
     "zinc dust", "iron", "iron powder")

# ── Oxidizing agents ─────────────────────────────────────────────────────
_add("oxidizing",
     "manganese dioxide", "pyridinium chlorochromate",
     "dess-martin periodinane", "sodium periodate",
     "potassium permanganate", "m-chloroperoxybenzoic acid",
     "3-chlorobenzenecarboperoxoic acid", "hydrogen peroxide", "oxone",
     "n-bromosuccinimide", "n-chlorosuccinimide", "n-iodosuccinimide",
     "bromosuccinimide", "selectfluor", "tert-butyl hydroperoxide",
     "sodium hypochlorite")

# ── Catalysts and ligands ────────────────────────────────────────────────
# Buchwald-type biaryl phosphines (XPhos, SPhos, RuPhos, ...) and BINAP are
# listed by their full systematic-ish names because that is how OPSIN's own
# candidate span reads them off the page — the trade names never appear as
# parseable IUPAC strings.
_add("catalyst_ligand",
     "palladium", "palladium(ii) acetate",
     "tetrakis(triphenylphosphine)palladium(0)", "palladium(ii) chloride",
     "bis(triphenylphosphine)palladium(ii) dichloride",
     "1,1'-bis(diphenylphosphino)ferrocene", "bis(diphenylphosphino)ferrocene",
     "tris(dibenzylideneacetone)dipalladium(0)", "triphenylphosphine",
     "copper(i) iodide", "copper(ii) sulfate", "copper bromide",
     "nickel chloride", "ruthenium", "ruthenium(ii)",
     "chloro(pentamethylcyclopentadienyl)ruthenium", "rhodium", "platinum",
     "palladium(ii)", "titanium(iv)", "4,4'-di-tert-butyl-2,2'-bipyridine",
     "1,1'-bis(di-tert-butylphosphino)ferrocene",
     "bis(di-tert-butylphosphino)ferrocene",
     "dicyclohexyl(2',4',6'-triisopropylbiphenyl-2-yl)phosphine",        # XPhos
     "2,2'-bis(diphenylphosphino)-1,1'-binaphthalene",                   # BINAP
     "2-dicyclohexylphosphino-2',6'-dimethoxy-1,1'-biphenyl",            # SPhos
     "1,1'-bis(diphenylphosphino)ferrocene-palladium(ii)",               # Pd(dppf)
     "1,1'-bis(diphenylphosphino)ferrocene-palladium(ii)dichloride",     # Pd(dppf)Cl2
     "dicyclohexylphosphino-2',6'-diisopropoxy-1,1'-biphenyl",           # RuPhos
     "dicyclohexylphosphino-2',4',6'-triisopropyl-1,1'-biphenyl",        # XPhos (alt name)
     "(di-t-butylphosphino)-3-methoxy-6-methyl-2',4',6'-tri-i-propyl-1,1'-biphenyl")  # tBuXPhos-class

# ── Drying / workup ──────────────────────────────────────────────────────
_add("drying_workup",
     "magnesium sulfate", "sodium sulfate", "calcium chloride",
     "silica gel", "celite", "molecular sieves", "brine")

# ── Inert atmosphere ─────────────────────────────────────────────────────
_add("inert_gas", "nitrogen", "argon", "hydrogen", "helium")

# ── Common inorganic salts / misc reagents ──────────────────────────────
_add("salt_misc",
     "ammonium chloride", "sodium chloride", "potassium chloride",
     "lithium chloride", "ammonium acetate", "sodium acetate",
     "sodium azide", "trimethylsilyl azide", "sodium nitrite",
     "isoamyl nitrite", "tert-butyl nitrite", "iodomethane", "methyl iodide",
     "dimethyl sulfate", "bromine", "iodine")

# ── Silylating / protecting-group reagents ──────────────────────────────
_add("protecting_silyl",
     "benzyl bromide", "tert-butyldimethylsilyl chloride",
     "tert-butylchlorodiphenylsilane", "trimethylsilyl chloride",
     "tris(trimethylsilyl)silane", "n,o-dimethylhydroxylamine",
     "n,o-dimethylhydroxylamine hydrochloride", "tms-acetylene")

# ── Fluorinating / silylating reagents ──────────────────────────────────
_add("fluorinating_silylating",
     "(trifluoromethyl)trimethylsilane", "trimethyl(trifluoromethyl)silane",
     "(trifluoromethyl)silane", "diethylaminosulfur trifluoride",
     "bis(2-methoxyethyl)aminosulfur trifluoride", "phenylsilane")


def _lexicon_lookup(name: str) -> tuple[str, str, str] | None:
    """`(category, matched_key, salt_suffix)` for the FIRST lexicon hit on
    `name`, or `None`. `matched_key` is the exact normalized string that lives
    in `REAGENT_LEXICON` — together with `category` this is enough to look the
    entry back up by hand, which is the audit trail `reason` carries. Order is
    fixed (direct match, then `_SALT_SUFFIXES` in its declared list order) so
    the result cannot vary between runs — see DETERMINISM in the module
    docstring.
    """
    n = _normalize(name)
    if n in REAGENT_LEXICON:
        return REAGENT_LEXICON[n], n, ""
    for suf in _SALT_SUFFIXES:
        if n.endswith(suf):
            base = n[: -len(suf)].strip()
            if base in REAGENT_LEXICON:
                return REAGENT_LEXICON[base], base, suf.strip()
    return None


# Heavy-atom count at or below which a structure is treated as too small to
# be a claimed or exemplified compound in this corpus, regardless of whether
# its name is in the lexicon. See module docstring for what was checked at
# this boundary before picking 3.
_HAC_FLOOR = 3

# The complete, closed label set this module ever emits. `"compound"` is the
# default (neither tier fired); `"reagent"` and `"trace_fragment"` are the
# two flag labels the two tiers respectively support — see THE LABEL DESIGN
# in the module docstring for why there are exactly two, not the lexicon's
# ~12 internal categories and not a finer synthesis-role taxonomy.
LABELS: tuple[str, str, str] = ("compound", "reagent", "trace_fragment")


def _heavy_atom_count(smiles: str) -> int | None:
    """RDKit heavy-atom count for `smiles`, or `None` if RDKit is not
    installed or the string does not parse.

    Deterministic (see module docstring): `MolFromSmiles` is a pure parse of
    the given string with no seed/sampling/global state, and
    `GetNumHeavyAtoms` is a count over the resulting atom list — invariant to
    parse-internal atom ordering, which is itself fixed by the input string.
    Nothing here touches RDKit's canonical-SMILES writer, the one facility
    whose output has drifted across RDKit versions; only atom counting.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


@dataclass(frozen=True)
class ReagentVerdict:
    """What a structure looks like, and WHY — never whether to keep it.

    `label` is always one of `LABELS`:
      - `"compound"`        — matched neither tier (the default: no evidence
                               this is anything other than one of the
                               patent's own compounds).
      - `"reagent"`         — exact normalized-name match in `REAGENT_LEXICON`.
      - `"trace_fragment"`  — no lexicon match, but RDKit heavy-atom count
                               <= `_HAC_FLOOR`.

    `reason` is machine-readable and lets a human audit any single verdict
    without re-running anything:
      - `""` for `"compound"`.
      - `"lexicon:<category>:<matched_key>"` for `"reagent"`, e.g.
        `"lexicon:solvent:dichloromethane"`. If a salt/hydrate suffix was
        stripped before matching, a third field is appended:
        `"lexicon:base:triethylamine:salt=hydrochloride"`.
      - `"structural:hac=<n>:floor=<_HAC_FLOOR>"` for `"trace_fragment"`,
        e.g. `"structural:hac=1:floor=3"`.
    """
    label: str
    reason: str = ""


def classify(name: str, smiles: str = "") -> ReagentVerdict:
    """Is this resolved structure a laboratory reagent, structural trace, or
    (by default) one of the patent's own compounds?

    `name` is matched against `REAGENT_LEXICON` first (cheap, no RDKit
    needed). If that misses and `smiles` is given, RDKit is imported lazily
    to check heavy-atom count as a narrow backstop. Everything else —
    including every structure RDKit can't parse — is labeled `"compound"`:
    the default here, and at every call site, has to be "no flag" rather than
    "flag," because a false-negative reagent just ships one extra row
    labeled `"compound"` while a false-positive `"reagent"`/`"trace_fragment"`
    only ever adds a label, never removes the structure — this module does
    not delete anything, so the cost of a wrong flag is a wrong tag, not a
    lost molecule.
    """
    hit = _lexicon_lookup(name)
    if hit is not None:
        category, matched_key, salt = hit
        reason = f"lexicon:{category}:{matched_key}"
        if salt:
            reason += f":salt={salt}"
        return ReagentVerdict("reagent", reason)
    if smiles:
        hac = _heavy_atom_count(smiles)
        if hac is not None and hac <= _HAC_FLOOR:
            return ReagentVerdict("trace_fragment",
                                   f"structural:hac={hac}:floor={_HAC_FLOOR}")
    return ReagentVerdict("compound")


def label_of(name: str, smiles: str = "") -> str:
    """Convenience wrapper over `classify` for just the label string."""
    return classify(name, smiles).label
