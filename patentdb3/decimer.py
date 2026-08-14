"""Drive DECIMER on a Colab GPU, ONE PATENT AT A TIME.

    python3 -m patentdb3.decimer plan    US10730863
    python3 -m patentdb3.decimer run     US10730863 -s decimer
    python3 -m patentdb3.decimer ingest  US10730863

WHY PATENT BY PATENT. A recogniser with no confidence score can only be
trusted as far as it has been measured, and the measurement lives in the
worklist's VALIDATE half — compounds whose structure the patent's own text
already gave. One patent is one measurement you can read, argue with, and act
on. A corpus-wide run before that produces a single number nobody can
attribute.

THE ORDER MATTERS, AND IT IS CHEAPEST-FIRST
--------------------------------------------
    1. fetch      the patent's images                    local, ~3 KB each
    2. OCR        read the name printed in the picture   local, $0, EXACT
    3. DECIMER    recognise what is left                 remote, GPU, billable

Step 2 is not an optimisation. Where a patent prints the IUPAC name inside the
drawing, OCR plus OPSIN gives an answer that passes the same gate every
text-derived structure passes, and DECIMER's output cannot be checked at all
on a RECOVER row. Sending an image DECIMER does not need spends GPU time to
get a WEAKER answer than the one already available. On the two captioned
patents measured, that crossed off 39 compounds before any GPU ran.

WHAT THIS COSTS
----------------
`colab new --gpu` allocates a billable VM and NOTHING reclaims it but a 24h
cap. This module never calls `colab new` and never calls `colab stop`: it
attaches to a session you named, so the thing that spends money stays an
explicit act you performed. `run` prints the stop command every time.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from .core import config
from .sources import images as IM

VM_SCRIPT = Path(__file__).with_name("decimer_vm.py")
WORK = config.OUTPUT_DIR / "decimer"          # per-patent staging, local side
REMOTE = "/content/work"

# THE VERSIONS BOTH DECIMER REPOS PIN, and Colab does not provide.
# `DECIMER-Image_Transformer` and `DECIMER-Image-Segmentation` both state
# Python 3.10.0 / TensorFlow 2.10.1. Colab's kernel is 3.12 with TF 2.20, and
# the kernel's interpreter cannot be swapped underneath a running session.
#
# `decimer` 2.8.0 alone DOES install on 3.12 (its PyPI metadata allows
# tensorflow<=2.20) and produced the 93.4% measurement. What does NOT install
# is `decimer-segmentation`, which is why `n_segments` came back 0 for every
# image — a value that reads as "no image held two structures" when it means
# "nobody looked".
#
# So the pin is honoured in a SEPARATE interpreter beside the kernel rather
# than by fighting it: `uv` builds a 3.10 venv, both packages install there,
# and the worker runs as a subprocess under it. The kernel stays 3.12 and is
# only used to launch things.
VENV = "/content/decimer-venv"
# 3.11, NOT the 3.10 both DECIMER repos state. Pinning 3.10 forces
# `tensorflow==2.10.1` (what `decimer-segmentation` needs), and TF 2.10.1
# caps `decimer` at 2.2.2 — a MEASURED, SILENT DOWNGRADE from the 2.8.0 that
# produced our 93.4%, and 2.2.2 has no confidence output at all (verified:
# zero occurrences of "confid" in its source). The pin cost the newer model
# and the only per-structure quality signal DECIMER offers, to buy a
# structure COUNT on images that hold one structure 98.9% of the time.
#
# So the pin is refused, and `decimer_segmentation` is a separate opt-in
# environment (`config.DECIMER_SEGMENTATION`) rather than a constraint on
# this one. State the trade where it is made, do not bury it in a version.
PY_PIN = "3.11"
# `--python <interpreter>` IS LOAD-BEARING. `VIRTUAL_ENV=... uv pip install`
# does NOT redirect uv — it reported "Using Python 3.12.13 environment at:
# /usr" and then failed with "tensorflow==2.10.1 has no wheels with a matching
# Python ABI tag (cp312)", which reads like a version conflict and is actually
# uv installing into the wrong interpreter.
#
# `numpy<2` IS ALSO LOAD-BEARING. TF 2.10.1 predates the numpy 2 ABI break, and
# with numpy 2.x installed `import tensorflow` dies inside
# `_pywrap_bfloat16.TF_bfloat16_type()` with "Unable to convert function
# return value to a Python type" — a message that names neither numpy nor the
# real cause.
SETUP = f"""
import subprocess, os
def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode
if not os.path.exists("{VENV}/bin/python"):
    sh("pip -q install uv")
    sh("uv venv --python {PY_PIN} {VENV}")
# `--python <interp>` IS LOAD-BEARING. `VIRTUAL_ENV=... uv pip install` does
# NOT redirect uv: it reported "Using Python 3.12.13 environment at: /usr" and
# then failed with "no wheels with a matching Python ABI tag (cp312)", which
# reads like a version conflict and was uv installing into the wrong place.
# `keras_preprocessing` is NOT a stated dependency and is required anyway.
# DECIMER ships its tokenizer as a PICKLE whose class lives in that module,
# and TF 2.20 no longer bundles it, so `import DECIMER` dies in
# `load_tokenizer` with `ModuleNotFoundError: No module named
# 'keras_preprocessing'` — after the 664 MB model download has already run.
sh("uv pip install -q --python {VENV}/bin/python 'decimer>=2.8' rdkit "
   "keras_preprocessing")
rc = sh("{VENV}/bin/python -c \\"import sys, DECIMER, inspect;"
        "import DECIMER.decimer as D;"
        "print('venv python', sys.version.split()[0]);"
        "print('DECIMER', DECIMER.__version__);"
        "import tensorflow as tf; print('tf', tf.__version__);"
        "sig=str(inspect.signature(D.predict_SMILES));print('sig', sig);"
        "assert 'confidence' in sig, 'this DECIMER cannot report confidence'\\"")
print("SETUP_RC", rc)
"""


def plan(pid: str, *, fetch: bool = True) -> tuple[list, list, list]:
    """`(needs_decimer, read_by_ocr, no_image)` for one patent.

    Reads the shipped worklist rather than rebuilding it, so what is sent to
    the GPU is exactly what the pipeline says is outstanding.
    """
    items = [w for w in IM.read_worklist() if w.patent_id == pid]
    if not items:
        return [], [], []

    if fetch:
        want = [it for it in items if not it.local_path.exists()]
        # SAY WHY NOTHING ARRIVED. `images.fetch` goes through `gp_images`,
        # which returns `{}` when `config.GP_ENABLED` is off and cannot raise.
        # Without this line a switched-off fetch and a patent Google Patents
        # does not serve produce the identical output — "N with no image" —
        # and the first is a flag you forgot while the second is a fact.
        if want and not config.GP_ENABLED:
            print(f"{len(want)} image(s) are not on disk and GP_ENABLED=0, so "
                  f"none will be fetched.\n"
                  f"    GP_ENABLED=1 python3 -m patentdb3.decimer plan {pid}")
        else:
            got = sum(1 for it in want if IM.fetch(it) is not None)
            print(f"fetched {got} of {len(want)} missing image(s)")
    have = [it for it in items if it.local_path.exists()]
    missing = [it for it in items if not it.local_path.exists()]

    # STEP 2. Free, exact, and it must run first — see the module docstring.
    read = []
    if have:
        from .sources import image_ocr
        prev = config.IMAGE_OCR
        config.IMAGE_OCR = True
        try:
            read = image_ocr.resolve(have, pid)
        finally:
            config.IMAGE_OCR = prev
    by_ocr = {r.cid for r in read}
    # THE VM CAN FETCH ITS OWN PIXELS NOW, so an image that is not on this
    # machine is still work — it goes in the manifest with a URL and the VM
    # pulls it into the Drive cache. It does NOT get an OCR pass, because OCR
    # runs here and here is where the image is not. That is a real cost: the
    # OCR route is free, exact, and self-checking, and DECIMER is none of the
    # three. Fetching locally first (`GP_ENABLED=1`) buys those back.
    needs = [it for it in have if it.cid not in by_ocr] + missing
    return needs, read, missing


def stage(pid: str, needs: list) -> Path:
    """Lay out `<WORK>/<pid>/` as the VM expects: `manifest.tsv` and `img/`."""
    d = WORK / pid
    if d.exists():
        shutil.rmtree(d)
    (d / "img").mkdir(parents=True)
    # THE URL COLUMN IS WHAT LETS THE VM FETCH ITS OWN PIXELS. Resolving the
    # Google Patents URL happens HERE, once per patent, because that code is
    # already written and already checks the served file name against the one
    # the XML states. The VM only downloads bytes for a name we resolved.
    urls = {}
    if config.GP_ENABLED:
        try:
            from .sources.gp_images import image_urls
            urls = image_urls(needs[0].patent_id) if needs else {}
        except Exception as e:                       # network, no GP page, ...
            print(f"gp_images: {e!r} — the VM will only see staged images")

    staged = 0
    with (d / "manifest.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(("patent_id", "cid", "image_file", "url"))
        for it in needs:
            # The file the XML NAMES is the key everywhere in this package, so
            # the VM sees that exact name and nothing derived from it.
            if it.local_path.exists():
                shutil.copy2(it.local_path, d / "img" / it.image_file)
                staged += 1
            w.writerow((it.patent_id, it.cid, it.image_file,
                        urls.get(it.stem, "")))
    n_url = sum(1 for it in needs if urls.get(it.stem))
    print(f"staged {staged} image(s) locally; {n_url} URL(s) for the VM to "
          f"fetch into its Drive cache")
    return d


def _colab(*args: str, code: str = "", check: bool = True):
    """Run one `colab` command. `code` is piped to stdin.

    `colab exec` takes a script with `-f` or python on STDIN. It has no `-c`,
    and passing one exits 2 with a usage message rather than doing nothing —
    which is the good failure, but only if the caller does not swallow it.
    """
    print(f"$ colab {' '.join(args)}" + (" <<code" if code else ""), flush=True)
    return subprocess.run(["colab", *args], input=code or None,
                          check=check, text=True)


def check_contract() -> None:
    """The VM script cannot import `images.RESULT_FIELDS`, so it restates the
    header as a literal. That is two declarations of one contract, and the
    failure mode — a column added here, results silently written with the old
    shape, `score()` reading a missing key as "" — is exactly the kind this
    package refuses to leave to discipline. Checked before every upload.
    """
    src = VM_SCRIPT.read_text()
    for f in IM.RESULT_FIELDS:
        if f'"{f}"' not in src:
            raise SystemExit(
                f"decimer_vm.py does not write the {f!r} column that "
                f"images.RESULT_FIELDS requires — fix it before running.")


def run(pid: str, session: str) -> int:
    """Stage, upload, recognise, download. Attaches to `session`; never
    allocates one and never stops one."""
    check_contract()
    needs, read, missing = plan(pid)
    print(f"{pid}: {len(needs)} for DECIMER, {len(read)} read by OCR, "
          f"{len(missing)} with no image")
    if not needs:
        print("nothing to recognise")
        return 0
    d = stage(pid, needs)

    # ONE UPLOAD, NOT ONE PER IMAGE. Every `colab` invocation authenticates
    # and exits, so 67 uploads is 67 round trips for ~200 KB of PNGs. Tar the
    # staging directory and unpack it on the VM.
    tar = d.with_suffix(".tar")
    shutil.make_archive(str(d), "tar", root_dir=d)
    print(f"staged {len(needs)} image(s) -> {tar.name} "
          f"({tar.stat().st_size/1024:.0f} KB)")
    _colab("upload", "-s", session, str(tar), f"{REMOTE}.tar")
    _colab("exec", "-s", session, code=(
        f"import shutil, tarfile, os\n"
        f"shutil.rmtree({REMOTE!r}, ignore_errors=True)\n"
        f"os.makedirs({REMOTE!r}, exist_ok=True)\n"
        f"tarfile.open({REMOTE + '.tar'!r}).extractall({REMOTE!r})\n"
        f"print(sorted(os.listdir({REMOTE!r})),"
        f" len(os.listdir({REMOTE + '/img'!r})), 'images')\n"))
    # THE PINNED INTERPRETER. Built once per VM and reused; see `SETUP`.
    _colab("exec", "-s", session, code=SETUP)
    # The worker runs UNDER that interpreter, not the kernel's. `colab exec -f`
    # would run it at Python 3.12, which is what silently disabled
    # segmentation on the first run.
    _colab("upload", "-s", session, str(VM_SCRIPT), "/content/decimer_vm.py")
    _colab("exec", "-s", session, code=(
        "import subprocess\n"
        f"rc = subprocess.run(['{VENV}/bin/python', '/content/decimer_vm.py'])\n"
        "print('WORKER_RC', rc.returncode)\n"))
    _colab("download", "-s", session, f"{REMOTE}/results.tsv",
           str(d / "results.tsv"))
    print(f"\nresults -> {d / 'results.tsv'}")
    print(f"session {session} is STILL RUNNING and billable. Stop it with:\n"
          f"    colab stop -s {session}")
    return 0


def ingest(pid: str) -> str:
    """Merge this patent's results into the canonical results file and score.

    Merged rather than overwritten, because the whole point of going patent by
    patent is to accumulate a measurement across several. Keyed on
    `(patent_id, cid)` so re-running one patent replaces its own rows and
    touches nothing else.
    """
    src = WORK / pid / "results.tsv"
    if not src.exists():
        return f"no results at {src} — run `decimer run {pid} -s <session>` first"
    fresh = list(csv.DictReader(open(src), delimiter="\t"))

    keep = []
    if IM.RESULTS.exists():
        keep = [r for r in csv.DictReader(open(IM.RESULTS), delimiter="\t")
                if r["patent_id"] != pid]
    IM.RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with IM.RESULTS.open("w", newline="") as fh:
        w = csv.DictWriter(fh, IM.RESULT_FIELDS, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(keep + fresh)
    return (f"{pid}: {len(fresh)} row(s) merged into {IM.RESULTS}\n\n"
            + IM.report())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="patentdb3.decimer")
    ap.add_argument("action", choices=("plan", "run", "ingest"))
    ap.add_argument("patent_id")
    ap.add_argument("-s", "--session", default="")
    a = ap.parse_args(argv[1:])

    if a.action == "plan":
        needs, read, missing = plan(a.patent_id)
        print(f"{a.patent_id}")
        print(f"  read by OCR, no GPU needed  {len(read):5d}")
        print(f"  for DECIMER                 {len(needs):5d}")
        print(f"  no image on disk            {len(missing):5d}")
        job = {}
        for it in needs:
            job[it.job] = job.get(it.job, 0) + 1
        print(f"  of those, by job            {job}")
        return 0
    if a.action == "run":
        if not a.session:
            print("run needs -s <session>. Create one yourself first:\n"
                  "    colab new -s decimer --gpu T4\n"
                  "It is billable and nothing reclaims it but a 24h cap.")
            return 2
        return run(a.patent_id, a.session)
    print(ingest(a.patent_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
