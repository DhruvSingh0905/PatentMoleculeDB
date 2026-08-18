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
import re
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


def markush_drawings(patent_id: str, xml: str, *,
                     fetch: bool = False) -> dict[str, Path]:
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
    # THE FILE NAME COMES FROM THE XML, NOT FROM THE WORK LIST.
    #
    # Two reasons, and the first cost a staging run. A `<chemistry>` id is
    # unique WITHIN a document and nowhere else — `CHEM-US-00022` exists in
    # almost every patent in the corpus — so a work list spanning 137 patents
    # matched on id alone staged 405 drawings for US9718825 of which the first
    # three belonged to US8722692 and US9051265. The loop would then have
    # joined one patent's scaffold to another's fragment and called it clean.
    #
    # And the work list is built from ASSAY CIDS. A shared scaffold belongs to
    # no compound number, so it is not in there at all — which is precisely
    # the drawing this tier cannot proceed without.
    chem_files = {m.group(1): m.group(2) for m in re.finditer(
        r'<chemistry[^>]*id="([^"]+)"(?:(?!</chemistry>).)*?'
        r'<img[^>]*file="([^"]+)"', xml, re.S)}
    have: dict[str, Path] = {}
    unnamed = []
    for chem_id in sorted(want):
        f = chem_files.get(chem_id)
        if not f:
            unnamed.append(chem_id)
            continue
        item = IM.WorkItem(patent_id=patent_id, cid="", job="RECOVER",
                           chemistry_id=chem_id, image_file=f)
        p = item.local_path
        if p.exists() and p.stat().st_size:
            have[chem_id] = p
        elif fetch:
            got = IM.fetch(item)
            if got:
                have[chem_id] = got
    if unnamed:
        logger.warning("%s: %d drawing(s) name no image file", patent_id,
                       len(unnamed))
    missing = want - set(have) - set(unnamed)
    if missing:
        logger.info("%s: %d drawing(s) still unfetched — %s",
                    patent_id, len(missing), sorted(missing)[:4])
    return have


# Installed once per VM and reused. MolScribe's own `requirements.txt` pins
# `numpy<2` and PyPI metadata does not — with numpy 2.x the import dies inside
# a compiled extension with a message naming neither numpy nor the cause. Read
# the requirements file, not the package metadata; that lesson cost two runs.
COLAB_SETUP = """
import subprocess, os
MARK = "/content/.molscribe_ready"
def sh(c):
    print("$", c, flush=True); return subprocess.run(c, shell=True).returncode
# IDEMPOTENT, BECAUSE THE FIRST RUN WILL LOOK LIKE A FAILURE. `colab exec`
# holds a websocket and abandons it after 600 seconds; this install takes
# longer, so the kernel finishes and prints SETUP_RC 0 while THIS side raises
# TimeoutError. The marker makes the retry cost seconds instead of repeating
# a ten-minute install on a billable runtime.
if os.path.exists(MARK):
    print("SETUP_RC 0  (already installed)")
else:
    sh("pip -q install 'numpy<2' torch torchvision "
       "'transformers<4.40' huggingface_hub rdkit OpenNMT-py==2.2.0 albumentations==1.1.0")
    sh("pip -q install git+https://github.com/thomas0809/MolScribe.git")
    rc = sh("python -c \\"import molscribe, torch;"
            "print('molscribe ok, cuda', torch.cuda.is_available())\\"")
    if rc == 0:
        open(MARK, "w").close()
    print("SETUP_RC", rc)
"""


def colab_run(patent_id: str, session: str, *, tries: int = 80,
              every: int = 45) -> int:
    """Drive one Colab session over this patent's manifest. A BACKEND, not the interface.

    Everything here is convenience. The job directory is already a complete,
    self-contained unit of work — `recognise_worker.py` plus a manifest plus
    the pixels — so this only saves someone the copying. A compute node that
    cannot run `colab` loses nothing.

    NEVER ALLOCATES A SESSION AND NEVER STOPS ONE. Same rule `decimer.py`
    holds to: a runtime is billable, and a tool that quietly starts one is a
    tool that quietly bills. The caller passes a session it already owns.
    """
    import shutil
    import subprocess

    d = job_dir(patent_id)
    if not (d / MANIFEST_NAME).exists():
        print(f"no manifest at {d} — run `stage {patent_id} --fetch` first")
        return 1
    n = len(json.loads((d / MANIFEST_NAME).read_text())["images"])
    if not n:
        print(f"{patent_id}: manifest is empty")
        return 1

    def _colab(*args, code=""):
        print(f"$ colab {' '.join(args)}" + (" <<code" if code else ""), flush=True)
        return subprocess.run(["colab", *args], input=code or None,
                              check=True, text=True)

    remote = f"/content/recognise/{patent_id}"
    # THE TARBALL LANDS FLAT IN /content, NOT BESIDE ITS DESTINATION.
    # `colab upload` writes through the Jupyter contents API, which does not
    # create parent directories: uploading to `/content/recognise/<pid>.tar`
    # returns a bare `500 Internal Server Error` naming no missing path.
    # Upload flat, then extract into the tree the next cell makes.
    upload_to = f"/content/{patent_id}.tar"
    # ONE UPLOAD, NOT ONE PER IMAGE. Every `colab` invocation authenticates and
    # exits, so 714 uploads is 714 round trips for a few MB of PNGs.
    tar = d.with_suffix(".tar")
    shutil.make_archive(str(d), "tar", root_dir=d)
    print(f"{patent_id}: {n} drawing(s), {tar.stat().st_size/1024:.0f} KB")
    _colab("upload", "-s", session, str(tar), upload_to)
    _colab("exec", "-s", session, code=(
        f"import shutil, tarfile, os\n"
        f"shutil.rmtree({remote!r}, ignore_errors=True)\n"
        f"os.makedirs({remote!r}, exist_ok=True)\n"
        f"tarfile.open({upload_to!r}).extractall({remote!r})\n"
        f"print(len(os.listdir({remote + '/images'!r})), 'images')\n"))
    _colab("exec", "-s", session, code=COLAB_SETUP)
    _colab("upload", "-s", session,
           str(Path(__file__).with_name("recognise_worker.py")),
           "/content/recognise_worker.py")

    # DETACHED, THEN POLLED. `colab exec` holds a websocket and gives up after
    # 600 seconds while the kernel keeps working, so a long job "fails" here
    # and succeeds there. Loading the model alone can exceed that.
    _colab("exec", "-s", session, code=(
        "import subprocess\n"
        "subprocess.Popen(['python', '/content/recognise_worker.py',"
        f" {remote!r}],\n"
        "                 stdout=open('/content/worker.log','w'),\n"
        "                 stderr=subprocess.STDOUT)\n"
        "print('worker launched detached')\n"))

    for i in range(tries):
        subprocess.run(["sleep", str(every)], check=False)
        r = subprocess.run(
            ["colab", "exec", "-s", session], text=True, capture_output=True,
            input=(f"import os\np={remote + '/results.tsv'!r}\n"
                   "print('ROWS', sum(1 for _ in open(p))-1 if os.path.exists(p) else -1)\n"
                   "print(open('/content/worker.log').read()[-300:])\n"))
        out = (r.stdout or "") + (r.stderr or "")
        rows = next((int(w) for ln in out.splitlines() if ln.startswith("ROWS")
                     for w in ln.split()[1:2]), -1)
        print(f"  poll {i+1}: {rows}/{n}", flush=True)
        if rows >= n:
            break
    else:
        print("worker did not finish; /content/worker.log on the VM has why")
        return 1

    _colab("download", "-s", session, f"{remote}/results.tsv",
           str(d / RESULTS_NAME))
    got = read_results(patent_id)
    print(f"\n{patent_id}: {len(got)}/{n} recognised -> {d / RESULTS_NAME}")
    print(f"session {session} is STILL RUNNING and billable. Stop it with:\n"
          f"    colab stop -s {session}")
    return 0


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
    p.add_argument("--fetch", action="store_true",
                   help="download drawings this patent needs but has not got. "
                        "A shared scaffold is never in the image work list, "
                        "so without this the tier's most important drawing is "
                        "always missing.")

    p = sub.add_parser("run", help="drive a Colab session over the manifest")
    p.add_argument("patent_id")
    p.add_argument("-s", "--session", required=True,
                   help="a session you already own. This never allocates one "
                        "and never stops one — a runtime is billable.")

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
                                    xml_path.read_text(errors="replace"),
                                    fetch=a.fetch)
        if not drawings:
            print(f"{a.patent_id}: no markush drawings to recognise")
            return 0
        m = write_manifest(a.patent_id, drawings)
        print(f"staged {len(drawings)} drawing(s) -> {m.parent}")
        print("run any recogniser over it, then:")
        print(f"  python3 -m patentdb3.recognise ingest {a.patent_id} <results.tsv>")
        return 0

    if a.cmd == "run":
        return colab_run(a.patent_id, a.session)

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
