"""Compound images: fetch them, list the work, score what comes back.

WHAT THE XML GIVES US, AND WHAT IT DOES NOT
---------------------------------------------
The XML is ground truth for WHICH image belongs to which compound. A row names
a `<chemistry>` id; that element names a file:

    row for compound 100  ->  CHEM-US-00315
    CHEM-US-00315         ->  US10730877-20200804-C00315.TIF

It does NOT hold the image. The file name is the key, and it is the only part
of the join that survives outside the XML — so it, not the `<chemistry>` id,
is what every downstream step carries.

The `C00315` number is a document-order counter over every `<chemistry>` in
the file, numbered 1..N. It is NOT the compound number: US10730877 holds 257
structures in its body (scaffolds, schemes, intermediates) which push the
counter ahead of the example numbering. Measured corpus-wide, `cid == C-number`
in 545 of 2,834 cases (19.2%), and those are coincidence. Never compute an
offset; read the link the XML states.

TWO SOURCES FOR THE PIXELS
----------------------------
`config.IMAGE_SOURCE` picks one. Neither is trusted: the file name the XML
states is checked against what arrives.

  gpatents  (default) one GET per image, ~3 KB. Measured coverage of the files
            the XML names: 99.7%, 5,834 of 5,849 across 15 patents.
  uspto     ground truth, and impractical on a laptop. Red Book product
            PTGRDT ships the TIFs beside the XML, but only as weekly TARs of
            ~3.7 GB; there are 1,642 of them and no per-patent split
            (`PTGRDT-SPLT` returns an error). ~130 grant weeks for this corpus.

WHAT DECIMER ACTUALLY WANTS
------------------------------
Read off the two repositories rather than assumed, because both facts shape
this module:

  DECIMER-Image-Segmentation
      segment_chemical_structures(page: np.ndarray, expand=True) -> list[np.ndarray]
      segment_chemical_structures_from_file(path) -> list[np.ndarray]
    Returns a LIST. More than one segment from one of our images is a signal,
    not an error — a patent image can carry two depictions, and many carry the
    compound's IUPAC name rendered beside the structure, which segmentation is
    what removes.

  DECIMER-Image_Transformer
      predict_SMILES(image_path: str) -> str
    Takes a FILE PATH, not an array. So images must be on disk before
    recognition, which is why `fetch` writes them rather than returning bytes.
    It returns a bare SMILES with NO confidence score.

Both want Python 3.10 and TensorFlow 2.10.1, so recognition runs in its own
environment. This module never imports DECIMER. It prepares the work and
scores the result, and the two halves meet at two TSV files.

CORRECTNESS CANNOT COME FROM THE MODEL
-----------------------------------------
`predict_SMILES` returns no confidence, so a returned SMILES is a claim with
nothing behind it. The check is external and it already exists in the corpus:
**3,074 compounds have a structure we derived from the patent's own words AND
a picture in their own row.** Recognise those and compare InChIKeys. That
measures accuracy. The other 2,871 have no text answer and are the recovery
work — they cannot be checked this way, which is exactly why the first group
must be run first.
"""
from __future__ import annotations

import csv
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..core import config

logger = logging.getLogger(__name__)

WORKLIST = config.OUTPUT_DIR / "image_worklist.tsv"
RESULTS = config.OUTPUT_DIR / "image_results.tsv"

WORK_FIELDS = ("patent_id", "cid", "job", "chemistry_id", "image_file",
               "local_path", "url", "expected_inchikey", "n_assays", "assays")
# `recogniser` and `confidence` exist so more than one model can write into
# ONE results file and be compared on identical images.
#
#   recogniser  which model produced the row (today only "decimer").
#               The row key is (patent_id, cid, recogniser), so two models
#               scoring the same compound are two rows, never a silent
#               overwrite — and `score()` reports each separately, because an
#               undifferentiated accuracy over two models measures neither.
#   confidence  the model's own estimate, 0-1, or "" when it does not offer
#               one. NOT comparable ACROSS models: DECIMER's is the mean of
#               per-token decoder confidences and another model's will mean
#               something else. Blank is "not offered", never "low".
RESULT_FIELDS = ("patent_id", "cid", "image_file", "n_segments", "smiles",
                 "inchikey", "confidence", "recogniser", "error")

_UA = "Mozilla/5.0 (compatible; patentdb3/1.0)"
_TIMEOUT_S = 45


@dataclass
class WorkItem:
    patent_id: str
    cid: str
    job: str                      # "VALIDATE" | "RECOVER"
    chemistry_id: str
    image_file: str               # as the XML names it, e.g. ...C00315.TIF
    expected_inchikey: str = ""
    n_assays: int = 0
    assays: str = ""
    url: str = ""                 # optional; filled by `emit(with_urls=True)`

    @property
    def stem(self) -> str:
        return self.image_file.rsplit(".", 1)[0]

    @property
    def local_path(self) -> Path:
        """Where the pixels land. `predict_SMILES` needs a path, not bytes."""
        return config.IMAGE_DIR / self.patent_id / f"{self.stem}.png"


def emit(*, with_urls: bool = False, path: Path | None = None) -> dict:
    """Build the queue from the shipped artifacts. Deterministic, $0.

    One row per compound that has a picture in its OWN table row and at least
    one assay value. Split by whether the answer is already known:

      VALIDATE  a structure was derived from the patent's own text. Recognise
                the image and compare InChIKeys. This measures ACCURACY.
      RECOVER   no text answer exists. This is the coverage work, and it
                cannot be checked — which is why VALIDATE must run first.

    Markush rows are excluded: their picture is a substituent, not the
    compound. They keep their scaffold and fragment references for later
    enumeration (see `cid_first._header_scaffolds`).

    `with_urls` adds a Google Patents URL per row. Off by default because the
    machine that runs the models may hold the Red Book TARs and need no URL,
    and filling it costs one page fetch per patent.
    """
    import json
    from collections import defaultdict
    from .cid_first import _drawing_refs, _header_scaffolds, _markush_cids

    man = json.loads(config.MANIFEST.read_text())
    st = list(csv.DictReader(open(man["structures"]), delimiter="\t"))
    dump = list(csv.DictReader(open(man["dump"]), delimiter="\t"))

    # A VALIDATE ROW IS A CLAIM THAT WE ALREADY KNOW THE ANSWER, and it is the
    # only thing an image reader can ever be SCORED against. A wrong answer
    # here does not merely fail to help — it inverts the measurement: the
    # reader recognises the drawing correctly, disagrees with the reagent
    # someone anchored to that compound number, and is recorded as an error.
    # That is a score that cannot improve.
    #
    # It is not hypothetical. On US10730863 ten compounds carried a synthesis
    # intermediate as their text answer, every one of them 300+ Da from the
    # mass the patent prints in that compound's own row. Scoring against those
    # ten produced "3 agree, 14 disagree" for a caption reader that was right
    # every time the patent could referee it.
    #
    # So an answer the mass gate flags is NOT truth. It is demoted to RECOVER
    # — the work is still queued, it simply carries no expected value. The row
    # is not deleted, because "we have a disputed answer" and "we have no
    # answer" are different states and the tally keeps them apart.
    known: dict[str, dict[str, str]] = defaultdict(dict)
    disputed = 0
    for r in st:
        if r.get("cid") and r.get("inchikey"):
            if r.get("mass_check") == "contradicts":
                disputed += 1
                continue
            known[r["patent_id"]].setdefault(r["cid"], r["inchikey"])
    assays: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in dump:
        if r.get("cid"):
            assays[r["patent_id"]][r["cid"]].append(
                f"{r.get('assay_name','')}={r.get('value_numeric','')}"
                f"{r.get('unit','')}")

    items: list[WorkItem] = []
    tally: dict[str, int] = defaultdict(int)
    for pid in sorted({r["patent_id"] for r in st}):
        xml_path = config.XML_INPUT_DIR / f"{pid}.xml"
        if not xml_path.exists():
            continue
        xml = xml_path.read_text(errors="replace")
        refs = _drawing_refs(xml)
        scaf = _header_scaffolds(xml)
        mk = _markush_cids(xml)
        files = {m.group(1): m.group(2) for m in re.finditer(
            r'<chemistry[^>]*id="([^"]+)"(?:(?!</chemistry>).)*?'
            r'<img[^>]*file="([^"]+)"', xml, re.S)}
        urls = {}
        if with_urls:
            from .gp_images import image_urls
            urls = image_urls(pid)
        for cid, ref in sorted(refs.items()):
            if not assays[pid].get(cid):
                tally["skipped_no_assay_value"] += 1
                continue
            if cid in mk or ref in scaf:
                tally["skipped_markush_part"] += 1
                continue
            f = files.get(ref)
            if not f:
                tally["skipped_no_image_named"] += 1
                continue
            job = "VALIDATE" if cid in known.get(pid, {}) else "RECOVER"
            tally[job] += 1
            items.append(WorkItem(
                patent_id=pid, cid=cid, job=job, chemistry_id=ref,
                image_file=f, expected_inchikey=known.get(pid, {}).get(cid, ""),
                n_assays=len(assays[pid][cid]),
                assays="; ".join(assays[pid][cid][:4]),
                url=urls.get(f.rsplit(".", 1)[0], "")))

    out = write_worklist(items, path)
    # Counted, not silent. A demotion moves a row from VALIDATE to RECOVER, and
    # without this line the two totals simply shift and nothing says why.
    tally["demoted_mass_contradicts"] = disputed
    tally["rows"] = len(items)
    tally["path"] = str(out)
    logger.info("images: emitted %d rows (%d VALIDATE, %d RECOVER) -> %s",
                len(items), tally["VALIDATE"], tally["RECOVER"], out)
    return dict(tally)


def fetch(item: WorkItem, *, source: str = "") -> Path | None:
    """Put this item's image on disk. Returns the path, or None.

    Only needed when recognition runs HERE. When the models run elsewhere the
    queue carries the file name (and optionally a URL) and that machine
    fetches its own pixels.

    The file name is verified, not assumed: a source that serves a different
    file is refused rather than silently recognised as this compound.
    """
    src = source or config.IMAGE_SOURCE
    dest = item.local_path
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if src == "gpatents":
        from .gp_images import image_urls
        urls = image_urls(item.patent_id)
        url = urls.get(item.stem)
        if not url:
            logger.warning("images: %s %s — %s not served by Google Patents",
                           item.patent_id, item.cid, item.stem)
            return None
        # THE NAME IS THE KEY. A URL whose file name is not the one the XML
        # states is a different image, whatever else is right about it.
        if not url.rsplit("/", 1)[-1].startswith(item.stem):
            logger.warning("images: %s — url does not name %s", item.cid, item.stem)
            return None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                raw = r.read()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            logger.warning("images: %s %s — fetch failed %r",
                           item.patent_id, item.cid, e)
            return None
    elif src == "uspto":
        raise NotImplementedError(
            "IMAGE_SOURCE=uspto needs the Red Book weekly TAR "
            "(product PTGRDT, ~3.7 GB per grant week). Not implemented — see "
            "this module's docstring for why it is impractical on a laptop.")
    else:
        raise ValueError(f"unknown IMAGE_SOURCE {src!r}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".png.tmp")
    tmp.write_bytes(raw)
    tmp.replace(dest)
    return dest


def write_worklist(items: list[WorkItem], path: Path | None = None) -> Path:
    """The handoff. One row per compound with a picture worth recognising."""
    p = path or WORKLIST
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(WORK_FIELDS), delimiter="\t")
        w.writeheader()
        for it in items:
            w.writerow({"patent_id": it.patent_id, "cid": it.cid, "job": it.job,
                        "chemistry_id": it.chemistry_id,
                        "image_file": it.image_file,
                        "local_path": str(it.local_path), "url": it.url,
                        "expected_inchikey": it.expected_inchikey,
                        "n_assays": it.n_assays, "assays": it.assays})
    return p


def read_worklist(path: Path | None = None) -> list[WorkItem]:
    p = path or WORKLIST
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(open(p), delimiter="\t"):
        out.append(WorkItem(
            patent_id=r["patent_id"], cid=r["cid"], job=r["job"],
            chemistry_id=r["chemistry_id"], image_file=r["image_file"],
            expected_inchikey=r.get("expected_inchikey", ""),
            n_assays=int(r.get("n_assays") or 0), assays=r.get("assays", ""),
            url=r.get("url", "")))
    return out


@dataclass
class Score:
    """What the VALIDATE rows say about the recogniser. Every count is over
    compounds whose answer we already knew from the patent's own text."""
    validated: int = 0
    agreed: int = 0
    disagreed: int = 0
    no_smiles: int = 0
    multi_segment: int = 0
    recovered: int = 0            # RECOVER rows that produced a structure
    disagreements: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        n = self.agreed + self.disagreed
        return self.agreed / n if n else 0.0


def score(results_path: Path | None = None,
          worklist_path: Path | None = None) -> Score:
    """Compare what the recogniser returned against what the text already said.

    Only VALIDATE rows can be scored. A RECOVER row has no known answer, so it
    is counted but never treated as evidence of accuracy — reporting a
    recovery rate as if it were an accuracy rate is the specific mistake this
    split exists to prevent.
    """
    rp = results_path or RESULTS
    if not rp.exists():
        return Score()
    expected = {(w.patent_id, w.cid): w for w in read_worklist(worklist_path)}
    s = Score()
    for r in csv.DictReader(open(rp), delimiter="\t"):
        w = expected.get((r["patent_id"], r["cid"]))
        if w is None:
            continue
        if int(r.get("n_segments") or 0) > 1:
            s.multi_segment += 1
        got = (r.get("inchikey") or "").strip()
        if w.job == "RECOVER":
            if r.get("smiles"):
                s.recovered += 1
            continue
        s.validated += 1
        if not got:
            s.no_smiles += 1
        elif got == w.expected_inchikey:
            s.agreed += 1
        else:
            s.disagreed += 1
            if len(s.disagreements) < 25:
                s.disagreements.append(
                    {"patent_id": r["patent_id"], "cid": r["cid"],
                     "expected": w.expected_inchikey, "got": got,
                     "image_file": r.get("image_file", "")})
    return s


def recovered(results_path: Path | None = None,
              worklist_path: Path | None = None) -> list:
    """Structures the recogniser produced, as `NamedCompound` rows.

    RECOVER rows only. A VALIDATE row already has a structure from the
    patent's own text, and that text answer WINS — it is derived from what the
    document says, while a recognised structure is a model's reading of a
    picture. Emitting both would put two structures on one compound and let a
    downstream join pick either.

    `source="image"` so a consumer can always tell how a structure was
    obtained, and `drawn_ref`/`drawn_file` are carried so any row can be traced
    back to the exact picture it came from.
    """
    from .iupac_names import NamedCompound

    rp = results_path or RESULTS
    if not rp.exists():
        return []
    work = {(w.patent_id, w.cid): w for w in read_worklist(worklist_path)}
    out = []
    for r in csv.DictReader(open(rp), delimiter="\t"):
        w = work.get((r["patent_id"], r["cid"]))
        if w is None or w.job != "RECOVER":
            continue
        smi = (r.get("smiles") or "").strip()
        if not smi or r.get("error"):
            continue
        out.append(NamedCompound(
            patent_id=r["patent_id"], name="", smiles=smi,
            inchikey=(r.get("inchikey") or "").strip(),
            start=-1, source="image", cid=r["cid"],
            drawn_ref=w.chemistry_id, drawn_file=w.image_file))
    return out


def report(results_path: Path | None = None,
           worklist_path: Path | None = None) -> str:
    """One readable block. Accuracy first, because it bounds everything else."""
    s = score(results_path, worklist_path)
    rec = recovered(results_path, worklist_path)
    L = [f"VALIDATE  {s.validated:,} scored",
         f"   agreed with the text answer   {s.agreed:,}",
         f"   disagreed                     {s.disagreed:,}",
         f"   no structure returned         {s.no_smiles:,}",
         f"   ACCURACY                      {100*s.accuracy:.1f}%",
         f"RECOVER   {s.recovered:,} produced a structure "
         f"({len(rec):,} usable rows)",
         f"images yielding >1 segment       {s.multi_segment:,}"]
    if s.disagreements:
        L.append("\ndisagreements (text answer vs recognised):")
        for d in s.disagreements[:8]:
            L.append(f"   {d['patent_id']} cid={d['cid']}  "
                     f"{d['expected'][:22]} != {d['got'][:22]}")
    if not s.validated:
        L.append("\nNo VALIDATE rows scored. Accuracy is UNKNOWN — a recovery "
                 "count alone is not evidence the structures are right.")
    return "\n".join(L)
