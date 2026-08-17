"""RUNS ON THE COLAB VM, NOT HERE. Recognise one patent's drawings.

This file is uploaded and executed by `patentdb3/decimer.py`. It is the only
code in this tree that runs somewhere else, and it therefore imports NOTHING
from `patentdb3` — the VM has the package's images and a manifest, and nothing
else. Keep it self-contained; an import of a sibling module fails there and
succeeds in every local test, which is the worst way to find out.

WHAT IT WRITES
---------------
`/content/work/results.tsv`, whose columns are `images.RESULT_FIELDS` exactly:

    patent_id  cid  image_file  n_segments  smiles  inchikey
    confidence  recogniser  error

`sources/images.py::score` and `::recovered` already read that shape. This
script's whole job is to fill it, so the header is stated here as a literal
rather than imported, and `decimer.py` asserts the two agree before uploading.

WHY IT PREDICTS ON THE WHOLE IMAGE
-----------------------------------
Segmentation runs, but only to COUNT. A full census of the corpus found 2,866
of 2,897 drawn compounds (98.9%) hold a whole-compound image in their own row,
so the row picture is the molecule and cropping it is more likely to lose an
substituent than to help. `n_segments` is recorded so the exception — a row
whose picture holds two structures — is visible in the artifact instead of
being silently mis-recognised. Decide what to do about those AFTER seeing how
many there are, not before.

THE CONFIDENCE COLUMN, AND WHAT IT IS NOT
------------------------------------------
`predict_SMILES(image, confidence=True)` returns `(smiles, [(token, conf)])`
in DECIMER >= 2.8; earlier builds have no such flag at all and the row says so
in `error` rather than leaving a blank that reads as certainty.

It is the model's own decoder confidence, collapsed to a MEAN over tokens.

IT HAS NOW BEEN MEASURED AGAINST THE VALIDATE HALF, AND IT DOES NOT WORK.
US10730863, 61 compounds whose structure the patent's own text already gave:

    correct   n=57   mean 0.9923   min 0.9772   max 0.9976
    WRONG     n=4    mean 0.9889   min 0.9832   max 0.9965

The distributions overlap completely. The single most confident answer among
the four wrong ones — compound 96, where a fluorine was read as a methyl —
scored 0.997, HIGHER than 51 of the 57 correct answers. Catching all four
errors means flagging 51 correct ones with them: an 89% false-alarm rate.
There is no threshold.

So the column is recorded and must NOT be used as a filter, a sort key, or a
quality signal. It sits at ~0.99 whether the answer is right or wrong. It is
kept because a negative result that is not written down gets re-discovered,
and because a future model's confidence may behave differently — the column
is the place to check that, not a claim that this one means anything.

Accuracy comes from the known answers. Never from a confidence, never from a
recovery count.
"""
import csv
import os
import sys
import traceback
import urllib.request

WORK = "/content/work"
RESULT_FIELDS = ("patent_id", "cid", "image_file", "n_segments", "smiles",
                 "inchikey", "confidence", "recogniser", "error")
RECOGNISER = "decimer"

# WHERE PIXELS ARE CACHED, AND WHY IT IS DRIVE. `/content` dies with the VM,
# so every restart re-fetches every image from Google Patents — thousands of
# requests to re-obtain bytes that never change. Drive survives the VM, so an
# image is fetched ONCE, ever. Set `DECIMER_CACHE` to override; the default is
# used only when Drive is actually mounted, and `/content/imgcache` otherwise
# so an unmounted Drive degrades to "slow" rather than "crash".
_DRIVE_ROOT = "/content/drive/MyDrive"
_DRIVE = f"{_DRIVE_ROOT}/patentdb3/images"
CACHE = os.environ.get("DECIMER_CACHE") or (
    _DRIVE if os.path.isdir(_DRIVE_ROOT) else "/content/imgcache")
_UA = "Mozilla/5.0 (compatible; patentdb3/1.0)"

# EVERY IMAGE IN ONE FLAT DIRECTORY, KEYED BY THE FILE NAME THE XML STATES.
# Not one directory per patent: the file name already carries the patent
# number (`US10730863-20200804-C00996.TIF`), so a per-patent tree adds a
# second place the same bytes can live and a second way to miss a cache hit.
# One name, one file, one lookup — the same rule the rest of this package
# applies to the XML's own file name.


def survey_cache() -> dict:
    """What is ALREADY in the cache, before anything is fetched.

    Run first and printed, because the failure this guards against is silent:
    a second copy of 5,000 images under a slightly different path costs Drive
    quota and makes two directories that disagree about what is cached. If
    this reports a stray directory, deal with it before fetching.
    """
    out = {"cache": CACHE, "exists": os.path.isdir(CACHE), "files": 0,
           "other_dirs": []}
    if out["exists"]:
        out["files"] = sum(1 for f in os.listdir(CACHE)
                           if not f.startswith(".") and not f.endswith(".tmp"))
    base = os.path.dirname(CACHE)
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            if os.path.isdir(p) and p != CACHE:
                n = sum(1 for _ in os.listdir(p))
                if n:
                    out["other_dirs"].append((p, n))
    return out


def fetch_images(rows) -> tuple[int, int, int]:
    """Fill the cache from the `url` column. `(cached, fetched, failed)`.

    THE FILE NAME IS THE KEY, exactly as `sources/images.fetch` treats it
    locally: a URL whose last path segment does not name the file the XML
    states is a different image, whatever else is right about it. Getting this
    wrong attaches one compound's picture to another compound's number, which
    nothing downstream could ever detect.
    """
    os.makedirs(CACHE, exist_ok=True)
    cached = fetched = failed = 0
    for r in rows:
        dest = os.path.join(CACHE, r["image_file"])
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            cached += 1
            continue
        url = (r.get("url") or "").strip()
        stem = r["image_file"].rsplit(".", 1)[0]
        if not url or not url.rsplit("/", 1)[-1].startswith(stem):
            failed += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read()
            tmp = dest + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, dest)
            fetched += 1
        except Exception:
            failed += 1
    return cached, fetched, failed


def environment() -> None:
    """Say what is actually installed. A version mismatch here looks exactly
    like a recognition failure, and it has already cost one measurement."""
    print(f"python {sys.version.split()[0]}", flush=True)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print(f"tensorflow {tf.__version__}   GPU {[g.name for g in gpus]}",
              flush=True)
        if not gpus:
            print("WARNING: no GPU visible — this will run, slowly, on CPU",
                  flush=True)
    except Exception as e:
        print(f"tensorflow UNAVAILABLE: {e!r}", flush=True)


def _inchikey(smiles: str) -> str:
    """InChIKey for a SMILES, or "". The join key every other route uses."""
    if not smiles:
        return ""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol) if mol is not None else ""
    except Exception:
        return ""


def main() -> int:
    environment()
    manifest = os.path.join(WORK, "manifest.tsv")
    if not os.path.exists(manifest):
        print(f"no manifest at {manifest}", flush=True)
        return 2
    rows = list(csv.DictReader(open(manifest), delimiter="\t"))
    print(f"{len(rows)} image(s) to recognise", flush=True)

    s = survey_cache()
    print(f"cache: {s['cache']}  exists={s['exists']}  holds {s['files']} file(s)",
          flush=True)
    for p, n in s["other_dirs"]:
        print(f"  ALSO PRESENT: {p} ({n} entries) — a second image directory "
              f"means two places disagree about what is cached", flush=True)
    if not os.path.isdir(_DRIVE_ROOT):
        print("DRIVE IS NOT MOUNTED. This cache dies with the VM and every "
              "image is re-fetched next session. `colab drivemount` needs a "
              "human at the terminal; run it yourself:\n"
              "    colab drivemount -s <session>", flush=True)

    cached, fetched, failed = fetch_images(rows)
    print(f"images: {cached} already cached, {fetched} fetched, "
          f"{failed} unavailable", flush=True)

    from DECIMER import predict_SMILES
    try:
        import numpy as np
        from decimer_segmentation import segment_chemical_structures
        from PIL import Image
        can_segment = True
    except Exception as e:
        print(f"segmentation unavailable ({e!r}); n_segments will be 0",
              flush=True)
        can_segment = False

    out_path = os.path.join(WORK, "results.tsv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(RESULT_FIELDS)
        for i, r in enumerate(rows, 1):
            # Staged copy first, then the Drive cache. `decimer.py` only
            # stages images it already holds locally; everything else arrives
            # through `fetch_images` above.
            img = os.path.join(WORK, "img", r["image_file"])
            if not os.path.exists(img):
                img = os.path.join(CACHE, r["image_file"])
            n_seg, smiles, conf, err = 0, "", "", ""
            if not os.path.exists(img):
                err = "image not staged and not fetchable"
            else:
                if can_segment:
                    try:
                        arr = np.array(Image.open(img).convert("RGB"))
                        n_seg = len(segment_chemical_structures(arr))
                    except Exception as e:
                        err = f"segment: {e!r}"[:120]
                try:
                    # `confidence=True` returns (smiles, [(token, conf), ...]).
                    # The per-token list is collapsed to its MEAN and the
                    # MINIMUM is not used: a single low token is normal at a
                    # ring closure and would flag almost everything, while the
                    # mean moved usefully in the one place we can check it.
                    got = predict_SMILES(img, confidence=True)
                    if isinstance(got, tuple):
                        smiles = (got[0] or "").strip()
                        pairs = got[1] or []
                        # `float(...)`, NOT `isinstance(c, float)`. DECIMER
                        # returns NUMPY scalars, and `isinstance(np.float32(x),
                        # float)` is False — an isinstance filter drops every
                        # value, leaves an EMPTY list rather than raising, and
                        # writes a blank confidence with no error beside it.
                        # That silently produced 67 blank rows on the first
                        # run and looked exactly like "the model offers none".
                        toks = []
                        for item in pairs:
                            try:
                                toks.append(float(item[1]))
                            except (TypeError, ValueError, IndexError):
                                pass
                        if toks:
                            conf = round(sum(toks) / len(toks), 4)
                        elif pairs:
                            # Values came back and none survived. Say so.
                            err = (f"confidence unreadable: {len(pairs)} pair(s),"
                                   f" first={pairs[0]!r}"[:150])
                    else:                       # older DECIMER: no confidence
                        smiles, conf = (got or "").strip(), ""
                except TypeError:
                    # This DECIMER predates the flag. Say so in the row rather
                    # than leaving a blank that reads as "the model was unsure
                    # of nothing" — see `images.RESULT_FIELDS`.
                    smiles = (predict_SMILES(img) or "").strip()
                    err = "confidence unavailable in this DECIMER build"
                except Exception as e:
                    err = (err + " | " if err else "") + f"predict: {e!r}"[:160]
            w.writerow([r["patent_id"], r["cid"], r["image_file"], n_seg,
                        smiles, _inchikey(smiles), conf, RECOGNISER, err])
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)
                fh.flush()
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
