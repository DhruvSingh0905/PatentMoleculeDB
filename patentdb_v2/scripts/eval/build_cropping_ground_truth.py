"""Build ground_truth_structures.json from BindingDB (Step 1 of the
label-association plan).

Joins our hand-curated assay-table fixture (which gives (patent_id,
assay_page, compound_id)) with BindingDB's per-patent TSV (Ligand SMILES),
re-derives all canonical-SMILES + InChIKey forms via RDKit ourselves, and
locates each compound's STRUCTURE page by grepping PaddleX markdown.

Output schema: list of records
    {
      patent_id: str,
      compound_id: str,
      assay_page: int,                # page where compound appears in an assay table
      structure_page: int,            # page where the structure DRAWING lives
      heading_pattern: str,           # which regex matched

      bdb_smiles: str,                # raw SMILES as BDB stored it
      full_canonical_smiles: str,     # RDKit canonical with stereo
      flat_canonical_smiles: str,     # RDKit canonical with stereo stripped
      full_inchikey: str,             # 27-char InChIKey via RDKit (NOT trusted from BDB)
      connectivity_inchikey: str,     # first 14 chars of full_inchikey
      flat_inchikey: str,             # 27-char InChIKey of the flat (stereo-stripped) molecule
      flat_connectivity_inchikey: str,# first 14 chars of flat_inchikey

      bdb_ligand_name: str,
    }

Why we re-derive everything ourselves
-------------------------------------
- BDB's `Ligand InChI Key` field is missing for ~20% of US9718825 rows and
  100% of US20250163061 rows — relying on it locks us out of those compounds.
- Stereo handling matters. DECIMER often flattens stereo even when the
  patent drawing has wedge bonds; comparing against BDB's full-stereo
  InChIKey rejects DECIMER's correct connectivity recovery as a "miss."
  The eval scores against `flat_inchikey` so connectivity-correct + stereo-
  loss is `OK`, not `DECIMER_MISS`.
- Canonical SMILES (full + flat) are also exported for the bench's
  diverse-sample output, so a human can read them directly without
  re-running RDKit on every comparison.

Compounds without a structure page (Markush substituent-table compounds
where the "structure" is a scaffold + R-group row) are still DROPPED —
cropping isn't the right tool for them.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]  # repo root (Patent (1)/)
BDB_TSV = ROOT / "output" / "bindingdb" / "our_patents.tsv"
GT_ASSAY = Path(__file__).parent / "ground_truth_assay_tables.json"
GT_OUT = Path(__file__).parent / "ground_truth_structures.json"

# Match "Cpd. No 350" / "Cpd. No. 350" / "Example 67" / "Compound 5A" / "Cpd 5A"
# Compound IDs in patents can be alphanumeric (5A, 8B, 100AA).
_NAME_RE = re.compile(
    r"(?:Example|Cpd\.?\s*No\.?|Compound|Cpd)\s*"
    r"([0-9]+[A-Za-z]{0,3})",
    re.IGNORECASE,
)


# Heading patterns that locate a compound on a page.
# Unanchored — PaddleX prefixes title lines with `<|ref|>title<|/ref|><|det|>...`,
# so anchored "^# Example N" misses real hits.
# Negative-lookbehind/word-boundary prevent matching "Example 67" in "Example 670".
_HEADING_PATTERNS = [
    ("Example",  r"(?<![A-Za-z])Example\s+{cid}(?![0-9A-Za-z])"),
    ("Cpd. No",  r"(?<![A-Za-z])Cpd\.?\s*No\.?\s*{cid}(?![0-9A-Za-z])"),
    ("Compound", r"(?<![A-Za-z])Compound\s+{cid}(?![0-9A-Za-z])"),
]

_PADDLEX_BLOCK_RE = re.compile(
    r"<\|ref\|>(\w+)<\|/ref\|><\|det\|>"
    r"\[\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\]"
    r"<\|/det\|>(.*?)(?=<\|ref\||\Z)",
    re.DOTALL,
)


def _page_has_compound_table_id(text: str, compound_id: str) -> bool:
    """True if PaddleX has a 'Cpd. No.' / 'No.' / '#' header somewhere on the
    page AND a tiny text block whose content is exactly `compound_id`.

    This catches structure tables like US10899738 p27 (Table 1) where each
    compound ID appears as a separate one-cell text block ('37', '38', ...,
    '41') alongside its image — bare numbers that the heading-pattern regex
    won't match because they're not preceded by 'Cpd. No' or 'Example'.

    Requires both signals (table-header + bare-id block) to avoid false
    positives from random integers (page numbers, NMR coupling constants).
    """
    target = compound_id.strip()
    # Collect all bare-id text blocks: any block whose entire content is
    # exactly an alphanumeric compound id (e.g. "41", "5A").
    bare_id_blocks: list[str] = []
    target_present = False
    for _btype, content in ((m.group(1), m.group(2)) for m in _PADDLEX_BLOCK_RE.finditer(text)):
        cleaned = re.sub(r"\s+", "", content)
        if not cleaned:
            continue
        if re.fullmatch(r"[0-9]{1,4}[A-Za-z]{0,3}", cleaned):
            bare_id_blocks.append(cleaned)
            if cleaned == target:
                target_present = True
    if not target_present:
        return False

    # Strong signal #1: an explicit table header on the same page
    has_table_header = bool(re.search(
        r"(?<![A-Za-z])(?:Cpd\.?\s*No\.?|Compound\s*No\.?|Example\s*No\.?)",
        text,
    )) or bool(re.search(
        r"(?<![A-Za-z])TABLE\s*\d+(?:-continued|\s+continued)?",
        text, re.IGNORECASE,
    ))
    if has_table_header:
        return True
    # Strong signal #2: at least 3 bare-id blocks AND they form a tight
    # numerical sequence (e.g. ['53','54','55','56','57']). This is the
    # "TABLE 1-continued" page where PaddleX missed the continuation header.
    if len(bare_id_blocks) >= 3:
        nums = sorted(int(b) for b in bare_id_blocks if b.isdigit())
        if len(nums) >= 3 and nums[-1] - nums[0] <= len(nums) + 2:
            return True
    return False


def find_structure_page(patent_id: str, compound_id: str,
                        all_pages_dir: Path) -> tuple[int | None, str | None]:
    """Find the page where this compound's structure is DRAWN.

    A page qualifies only if BOTH:
      (a) PaddleX has at least one `<|ref|>image<|/ref|>` block on it
          (otherwise it's a pure-text page like an NMR data section)
      (b) the compound is identifiable on that page, by EITHER:
          (b1) a heading-pattern match: "Example N" / "Cpd. No N" /
               "Compound N" anywhere in the markdown, OR
          (b2) a bare-id text block ("41") with a "Cpd. No." header
               on the same page — the structure-table layout
               (US10899738 p27 etc.)

    Returns (None, None) for compounds with no qualifying page — these
    are Markush substituent-table compounds where the "structure" is
    a scaffold + R-group row, not a drawn molecule.
    """
    if not all_pages_dir.exists():
        return (None, None)
    image_block_re = re.compile(r"<\|ref\|>image<\|/ref\|>")
    for md in sorted(all_pages_dir.glob("page_*.md")):
        try:
            text = md.read_text(errors="replace")
        except Exception:
            continue
        if not image_block_re.search(text):
            continue
        # (b1) heading match
        for label, tmpl in _HEADING_PATTERNS:
            pattern = tmpl.format(cid=re.escape(compound_id))
            if re.search(pattern, text):
                m = re.search(r"page_(\d+)\.md", md.name)
                if m:
                    return (int(m.group(1)), label)
        # (b2) compound-table bare-id match
        if _page_has_compound_table_id(text, compound_id):
            m = re.search(r"page_(\d+)\.md", md.name)
            if m:
                return (int(m.group(1)), "table_bare_id")
    return (None, None)


def parse_compound_id(ligand_name: str) -> str | None:
    """Pull the compound/example/Cpd.No identifier from a BDB ligand name.

    BDB names look like:
        "<IUPAC name or CHEMBLid>::US10899738, Cpd. No 350"
        "...::US9718825, Example 67"
        "US11312727, Compound 5A"
    """
    if not ligand_name:
        return None
    m = _NAME_RE.search(ligand_name)
    if not m:
        return None
    return m.group(1).strip()


def derive_smiles_keys(bdb_smiles: str) -> dict | None:
    """Run BDB SMILES through RDKit to compute canonical SMILES + InChIKeys
    in BOTH full-stereo and flat (stereo-stripped) form.

    Returns None if RDKit can't parse the SMILES (rare for BDB rows).

    The flat form is what the eval scores against — DECIMER often loses
    stereo on small wedges, so connectivity-correct should pass.
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey

    if not bdb_smiles:
        return None
    mol = Chem.MolFromSmiles(bdb_smiles)
    if mol is None:
        return None

    full_canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    flat_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)

    # Re-parse for clean InChIKey derivation (so the molecule object
    # reflects exactly what we canonicalized to)
    mol_full = Chem.MolFromSmiles(full_canonical)
    mol_flat = Chem.MolFromSmiles(flat_canonical)
    if mol_full is None or mol_flat is None:
        return None

    full_ikey = MolToInchiKey(mol_full) or ""
    flat_ikey = MolToInchiKey(mol_flat) or ""
    if not full_ikey or not flat_ikey:
        return None

    return {
        "full_canonical_smiles": full_canonical,
        "flat_canonical_smiles": flat_canonical,
        "full_inchikey": full_ikey,
        "connectivity_inchikey": full_ikey[:14],
        "flat_inchikey": flat_ikey,
        "flat_connectivity_inchikey": flat_ikey[:14],
    }


def load_bdb_index() -> dict[tuple[str, str], dict]:
    """Build index: (patent_id, compound_id_normalized) -> bdb row enriched
    with RDKit-derived canonical SMILES + InChIKeys.

    We do NOT require BDB's `Ligand InChI Key` field — it's missing for
    ~20% of US9718825 and 100% of US20250163061. As long as `Ligand SMILES`
    parses, we keep the row.
    """
    if not BDB_TSV.exists():
        sys.exit(f"BDB TSV not found at {BDB_TSV}")
    idx: dict[tuple[str, str], dict] = {}
    n_rows = 0
    n_indexed = 0
    n_skipped_no_cpd = 0
    n_skipped_smiles_parse = 0
    with BDB_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_rows += 1
            pat = row.get("Patent Number", "").strip()
            if not pat:
                continue
            cpd = parse_compound_id(row.get("BindingDB Ligand Name", ""))
            if not cpd:
                n_skipped_no_cpd += 1
                continue
            bdb_smiles = row.get("Ligand SMILES", "").strip()
            derived = derive_smiles_keys(bdb_smiles)
            if derived is None:
                n_skipped_smiles_parse += 1
                continue
            key = (pat, cpd.upper())
            # First occurrence wins (BDB has duplicates from multi-target assays)
            if key not in idx:
                idx[key] = {
                    "bdb_smiles": bdb_smiles,
                    "bdb_ligand_name": row.get("BindingDB Ligand Name", "").strip(),
                    **derived,
                }
                n_indexed += 1
    print(
        f"BDB: scanned {n_rows} rows, indexed {n_indexed} unique "
        f"(patent, compound) pairs (skipped {n_skipped_no_cpd} no-cpd-id, "
        f"{n_skipped_smiles_parse} unparseable-SMILES)"
    )
    return idx


def main() -> int:
    bdb = load_bdb_index()

    if not GT_ASSAY.exists():
        sys.exit(f"Assay-table ground truth missing at {GT_ASSAY}")
    with GT_ASSAY.open() as f:
        assay_gt = json.load(f)

    out: list[dict] = []
    misses_bdb: list[tuple[str, int, str]] = []
    misses_struct: list[tuple[str, int, str]] = []
    seen_compounds: set[tuple[str, str]] = set()
    seen_skip: set[tuple[str, str]] = set()

    for table in assay_gt.get("tables", []):
        pat = table["patent_id"]
        assay_page = int(table["page"])
        for row in table.get("expected", []):
            cpd = str(row["compound_id"]).strip()
            key_compound = (pat, cpd.upper())
            if key_compound in seen_compounds or key_compound in seen_skip:
                continue

            hit = bdb.get(key_compound)
            if not hit:
                misses_bdb.append((pat, assay_page, cpd))
                seen_skip.add(key_compound)
                continue

            # Find the structure page by grepping PaddleX markdown
            md_dir = ROOT / pat / "all_pages"
            struct_page, heading = find_structure_page(pat, cpd, md_dir)
            if struct_page is None:
                misses_struct.append((pat, assay_page, cpd))
                seen_skip.add(key_compound)
                continue

            seen_compounds.add(key_compound)
            out.append({
                "patent_id": pat,
                "compound_id": cpd,
                "assay_page": assay_page,
                "structure_page": struct_page,
                "heading_pattern": heading,
                **hit,
            })

    # Group for reporting
    by_patent: dict[str, int] = defaultdict(int)
    for r in out:
        by_patent[r["patent_id"]] += 1

    print(f"\nMatched {len(out)} (patent, compound) -> structure_page + InChIKey")
    for pat, n in sorted(by_patent.items()):
        struct_pages = sorted({r["structure_page"] for r in out if r["patent_id"] == pat})
        print(f"  {pat}: {n} compounds across {len(struct_pages)} structure pages: {struct_pages[:10]}{'...' if len(struct_pages)>10 else ''}")

    print(f"\nDropped (compound NOT in BDB): {len(misses_bdb)}")
    for pat, page, cpd in misses_bdb[:10]:
        print(f"  {pat} p{page} cpd={cpd}")

    print(f"\nDropped (no Example/Cpd/Compound heading found in PaddleX md — "
          f"likely a Markush substituent-table compound): {len(misses_struct)}")
    for pat, page, cpd in misses_struct[:20]:
        print(f"  {pat} (assay p{page}) cpd={cpd}")
    if len(misses_struct) > 20:
        print(f"  ... +{len(misses_struct) - 20} more")

    GT_OUT.write_text(json.dumps({
        "_meta": {
            "purpose": "Ground-truth (patent, compound, structure_page, BDB-derived SMILES + InChIKey) "
                       "for label-association eval.",
            "source": "ground_truth_assay_tables.json (compound IDs + assay pages) joined to "
                      "output/bindingdb/our_patents.tsv (Ligand SMILES) by parsing "
                      "'Cpd. No N' / 'Example N' / 'Compound N' from BDB ligand names. "
                      "All InChIKeys + canonical SMILES are RDKit-derived locally — we DO NOT "
                      "trust BDB's `Ligand InChI Key` field (missing for ~20% of US9718825 + "
                      "100% of US20250163061).",
            "scoring": "Score against `flat_inchikey` (stereo-stripped). DECIMER often loses stereo "
                       "even on correct connectivity recovery; full-stereo comparison would "
                       "incorrectly bucket those as DECIMER_MISS. Use full_inchikey only for "
                       "stereo-aware downstream verification.",
            "n_records": len(out),
            "by_patent": dict(by_patent),
        },
        "structures": out,
    }, indent=2))
    print(f"\nWrote {GT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
