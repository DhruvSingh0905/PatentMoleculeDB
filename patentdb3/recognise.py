"""Turn drawings into structures, wherever the GPU happens to be.

THE CONTRACT IS A FILE, NOT A SESSION
--------------------------------------
`decimer.py` drives a Colab notebook: it installs a venv, uploads a tarball,
polls for a worker, pulls results back. That works, and it is the wrong thing
for a compute node to depend on. A SLURM job has no browser, no `colab`
binary, and no interest in our staging directory layout.

So nothing here talks to a runner. The pipeline writes a MANIFEST — a plain
JSON list of `{id, file}` — and reads back a RESULTS file. Any machine that
can turn the first into the second is a valid backend: Colab, a login node, a
laptop with a GPU, someone running the model by hand. Neither file imports
`patentdb3`, so the worker never needs this package installed.

    manifest.json  ->  [ any GPU, any scheduler ]  ->  results.tsv
                       recognise_worker.py is one

That is the whole decoupling. `decimer.py` is now ONE backend behind this
interface rather than the interface itself.

WHY THIS EXISTS SEPARATELY FROM images.py
------------------------------------------
`sources/images.py` answers "which drawings are worth reading, and was the
answer right" — the work list and the scoring. This answers "get me the
structures for one patent's drawings, from wherever they already are". The
markush assembly tier needs the second and must not care about the first.

`structures()` is the whole public surface the pipeline uses. It returns
`{chemistry_id: smiles}` because that is what `repair/markush_loop` joins on —
a `<chemistry>` id, not a compound number. A drawing has an id even when no
compound claims it, which is exactly the case for a shared scaffold.

OFF IS THE DEFAULT AND COSTS NOTHING
-------------------------------------
With `RECOGNISER_BACKEND=off`, `structures()` returns `{}` and every markush
gap reports `scaffold_drawing_not_recognised`. No molecule is invented and no
network call is made. Turning it on is one environment variable.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from .core import config
from .sources import images as IM

logger = logging.getLogger(__name__)

# What a worker is handed and what it hands back. Both live beside the staged
# images so a whole job is one directory to copy.
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.tsv"

# The columns a worker must write. A subset of `images.RESULT_FIELDS` — the
# rest (`cid`, `n_segments`) are ours to fill in on ingest, and asking a remote
# worker for them would couple it to our work list.
WORKER_FIELDS = ("chemistry_id", "image_file", "smiles", "confidence",
                 "recogniser", "error")

BACKENDS = ("off", "file", "colab")


def backend() -> str:
    b = config.RECOGNISER_BACKEND
    if b not in BACKENDS:
        logger.warning("unknown RECOGNISER_BACKEND %r — treating as off", b)
        return "off"
    return b


def job_dir(patent_id: str) -> Path:
    return config.OUTPUT_DIR / "recognise" / patent_id


def write_manifest(patent_id: str, drawings: dict[str, Path]) -> Path:
    """`{chemistry_id -> image path}` -> a manifest a worker can read alone.

    Paths are stored RELATIVE to the job directory. An absolute path from this
    machine is meaningless on the node that runs the model, and a worker that
    has to rewrite paths is a worker that can get it wrong.
    """
    d = job_dir(patent_id)
    (d / "images").mkdir(parents=True, exist_ok=True)
    entries = []
    for chem_id, src in sorted(drawings.items()):
        if not src or not Path(src).exists():
            continue
        dst = d / "images" / Path(src).name
        if not dst.exists():
            dst.write_bytes(Path(src).read_bytes())
        entries.append({"id": chem_id, "file": f"images/{dst.name}"})
    path = d / MANIFEST_NAME
    path.write_text(json.dumps(
        {"patent_id": patent_id, "recogniser": config.RECOGNISER,
         "images": entries}, indent=2), encoding="utf-8")
    logger.info("manifest: %s — %d drawing(s)", path, len(entries))
    return path


def read_results(patent_id: str, path: Path | None = None) -> dict[str, str]:
    """`{chemistry_id -> smiles}` from a worker's results file. `{}` if absent.

    A row with an `error` and no `smiles` is skipped rather than stored as an
    empty structure: "the model failed" and "the model says this drawing is
    nothing" must not become the same blank downstream.
    """
    p = path or (job_dir(patent_id) / RESULTS_NAME)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            smi = (row.get("smiles") or "").strip()
            chem = (row.get("chemistry_id") or "").strip()
            if chem and smi:
                out[chem] = smi
    return out


def structures(patent_id: str) -> dict[str, str]:
    """THE ONE CALL THE PIPELINE MAKES. `{chemistry_id -> smiles}`.

    Never raises and never blocks: a missing file, a missing backend and a
    backend that is switched off all return `{}`, and the caller's own gate
    then reports the drawings as unrecognised. A repair tier that could be
    stalled by a GPU queue would be untestable without one.
    """
    b = backend()
    if b == "off":
        return {}
    if b == "colab":
        # The Colab driver's own staging, so a session already run through
        # `decimer.py` is usable here without being re-run.
        from . import decimer
        legacy = decimer.WORK / patent_id / "results.tsv"
        got = read_results(patent_id)
        return got or _from_legacy(legacy)
    return read_results(patent_id)


def _from_legacy(path: Path) -> dict[str, str]:
    """`decimer.py`'s results, which are keyed by cid rather than drawing.

    Joined back to a `<chemistry>` id through the work list, because that is
    where the two were paired in the first place. Kept so an existing Colab
    run is not wasted, not because cid is the right key.
    """
    if not path.exists():
        return {}
    by_file = {w.image_file: w.chemistry_id for w in IM.read_worklist()}
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            chem = by_file.get((row.get("image_file") or "").strip(), "")
            smi = (row.get("smiles") or "").strip()
            if chem and smi:
                out[chem] = smi
    return out


def markush_drawings(patent_id: str, xml: str) -> dict[str, Path]:
    """Every drawing the assembly tier needs for one patent, already fetched.

    Scaffolds AND fragments, from the gaps themselves rather than from the
    image work list — the work list is built from assay cids and a shared
    scaffold belongs to no cid, so it is not in there.
    """
    from .repair.markush_gap import find_gaps

    want: set[str] = set()
    for g in find_gaps(patent_id, xml):
        if g.scaffold_ref:
            want.add(g.scaffold_ref)
        want.update(r.fragment_ref for r in g.rows if r.fragment_ref)
    if not want:
        return {}
    have = {}
    for w in IM.read_worklist():
        if w.chemistry_id in want and w.local_path:
            p = Path(w.local_path)
            if p.exists():
                have[w.chemistry_id] = p
    missing = want - set(have)
    if missing:
        logger.info("%s: %d drawing(s) not fetched yet — %s",
                    patent_id, len(missing), sorted(missing)[:4])
    return have


def ingest(patent_id: str, src: Path) -> str:
    """Copy a worker's results into this patent's job directory."""
    if not src.exists():
        return f"no results at {src}"
    d = job_dir(patent_id)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / RESULTS_NAME
    dst.write_bytes(src.read_bytes())
    got = read_results(patent_id)
    return f"{patent_id}: {len(got)} structure(s) ingested -> {dst}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="patentdb3.recognise",
        description="Stage drawings for a GPU, and read the answers back.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stage", help="write a manifest for one patent")
    p.add_argument("patent_id")

    p = sub.add_parser("ingest", help="take a worker's results.tsv")
    p.add_argument("patent_id")
    p.add_argument("results", type=Path)

    p = sub.add_parser("status", help="what is recognised for one patent")
    p.add_argument("patent_id")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.cmd == "stage":
        xml_path = config.XML_INPUT_DIR / f"{a.patent_id}.xml"
        if not xml_path.exists():
            print(f"no cached XML for {a.patent_id}")
            return 1
        drawings = markush_drawings(a.patent_id,
                                    xml_path.read_text(errors="replace"))
        if not drawings:
            print(f"{a.patent_id}: no markush drawings to recognise")
            return 0
        m = write_manifest(a.patent_id, drawings)
        print(f"staged {len(drawings)} drawing(s) -> {m.parent}")
        print("run any recogniser over it, then:")
        print(f"  python3 -m patentdb3.recognise ingest {a.patent_id} <results.tsv>")
        return 0

    if a.cmd == "ingest":
        print(ingest(a.patent_id, a.results))
        return 0

    got = structures(a.patent_id)
    print(f"backend={backend()}  recognised={len(got)} drawing(s)")
    for k, v in list(got.items())[:8]:
        print(f"  {k:<22} {v[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
