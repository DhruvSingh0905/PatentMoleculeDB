"""Single source for loading BindingDB rows from `output/bindingdb/our_patents.tsv`.

Before this module: 3 separate loaders (fidelity_check, build_handcheck_xlsx,
build_cropping_ground_truth). Each parsed columns by index, each with
slightly different field selection. This module reads the TSV header
ONCE per process, exposes column indices by name, and provides:

  - `iter_rows_for_patent(patent_id)` — yields one BdbRow per BDB
    entry that mentions `patent_id` in its name field. BdbRow has
    every column we care about including source attribution
    (Patent Number / Article DOI / PMID / Curation Source).

  - `BdbRow` dataclass — typed access to all the columns.

Compound-id parsing reuses `core.compound_id.parse_compound_id` so
the prefix surface forms are consistent everywhere.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import config
from .compound_id import parse_compound_id

logger = logging.getLogger(__name__)


def bdb_tsv_path() -> Path:
    """Canonical filesystem path of the patent-filtered BDB TSV."""
    return config.REPO_ROOT / "output" / "bindingdb" / "our_patents.tsv"


@dataclass
class BdbRow:
    """One BindingDB measurement row, with source attribution."""
    compound_id: str             # parsed via core.compound_id.parse_compound_id
    raw_ligand_name: str
    smiles: str = ""
    inchi: str = ""
    inchi_key: str = ""
    ki_nM: str = ""              # raw string — caller decides float / null
    ic50_nM: str = ""
    kd_nM: str = ""
    ec50_nM: str = ""
    ph: str = ""
    temp_c: str = ""
    target_name: str = ""
    organism: str = ""
    # Source attribution — these are the columns that prove or refute
    # the "BDB enriched from external publications" hypothesis
    patent_number: str = ""
    article_doi: str = ""
    pmid: str = ""
    curation_source: str = ""
    # Pubchem etc. references for cross-DB linking
    pubchem_cid: str = ""
    chembl_id: str = ""

    def numeric_assays(self) -> dict[str, float]:
        """Return a {assay_name → value_nM} dict for non-empty columns."""
        out = {}
        for label, raw in [("Ki", self.ki_nM), ("IC50", self.ic50_nM),
                           ("Kd", self.kd_nM), ("EC50", self.ec50_nM)]:
            if not raw:
                continue
            try:
                out[label] = float(raw)
            except ValueError:
                continue
        return out


# Column-index cache (header parsed once)
_COLUMNS: dict[str, int] = {}


def _ensure_columns() -> dict[str, int]:
    """Lazy header parse. Returns a column-name → index map."""
    if _COLUMNS:
        return _COLUMNS
    p = bdb_tsv_path()
    if not p.exists():
        raise FileNotFoundError(f"BDB TSV not found at {p}")
    with p.open() as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr)
    for i, name in enumerate(header):
        _COLUMNS[name] = i
    return _COLUMNS


def iter_rows_for_patent(
    patent_id: str,
    *,
    skip_compound_id_parse_errors: bool = True,
) -> Iterator[BdbRow]:
    """Yield one `BdbRow` per BDB row whose ligand-name field mentions
    `patent_id`. Patent attribution (which patent the value comes from
    according to BDB) is in `BdbRow.patent_number` — caller compares
    that to `patent_id` if needed.
    """
    cols = _ensure_columns()
    p = bdb_tsv_path()

    def col(row: list[str], name: str, default: str = "") -> str:
        idx = cols.get(name)
        if idx is None or idx >= len(row):
            return default
        return (row[idx] or "").strip()

    name_idx = cols["BindingDB Ligand Name"]
    with p.open() as f:
        rdr = csv.reader(f, delimiter="\t")
        next(rdr)   # skip header
        for row in rdr:
            if len(row) <= name_idx:
                continue
            name = row[name_idx] or ""
            if patent_id not in name:
                continue
            cid = parse_compound_id(name) or ""
            if not cid and not skip_compound_id_parse_errors:
                logger.debug("BDB row had no parseable compound id: %r", name[:80])
            yield BdbRow(
                compound_id=cid.lower(),
                raw_ligand_name=name,
                smiles=col(row, "Ligand SMILES"),
                inchi=col(row, "Ligand InChI"),
                inchi_key=col(row, "Ligand InChI Key"),
                ki_nM=col(row, "Ki (nM)"),
                ic50_nM=col(row, "IC50 (nM)"),
                kd_nM=col(row, "Kd (nM)"),
                ec50_nM=col(row, "EC50 (nM)"),
                ph=col(row, "pH"),
                temp_c=col(row, "Temp (C)"),
                target_name=col(row, "Target Name"),
                organism=col(row, "Target Source Organism According to Curator or DataSource"),
                patent_number=col(row, "Patent Number"),
                article_doi=col(row, "Article DOI"),
                pmid=col(row, "PMID"),
                curation_source=col(row, "Curation/DataSource"),
                pubchem_cid=col(row, "PubChem CID"),
                chembl_id=col(row, "ChEMBL ID of Ligand"),
            )


def attribute_source(row: BdbRow, our_patent_id: str) -> str:
    """Human-readable description of where BDB attributes this measurement.

    Empirical finding from our 3 test patents: every row attributes
    to the same patent it's filtered into. So most rows return
    "this patent (USXXX)". External-publication rows (DOI/PMID set)
    or different-patent rows are rare in our test set but possible
    in BDB at large.
    """
    pn = row.patent_number.replace(",", "").replace(" ", "").strip()
    if pn and pn.replace("US", "") == our_patent_id.replace("US", ""):
        return f"this patent ({our_patent_id})"
    if pn:
        return f"different patent: {row.patent_number}"
    if row.article_doi:
        return f"external publication (DOI: {row.article_doi[:40]})"
    if row.pmid:
        return f"external publication (PMID: {row.pmid})"
    if row.curation_source:
        return f"unattributed; curation source = {row.curation_source}"
    return "no source listed in BDB"
