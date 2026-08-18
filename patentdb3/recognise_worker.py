#!/usr/bin/env python3
"""Read a manifest of drawings, write a results.tsv. Runs anywhere.

THIS FILE IMPORTS NOTHING FROM patentdb3, AND MUST NOT START.
That is the entire point. Copy it and the job directory to a GPU node, a Colab
runtime, a workstation — anywhere the model is installed — and run it. It has
no idea what a patent is, what a compound number is, or where the results are
going afterwards.

    python3 recognise_worker.py <job_dir> [--recogniser molscribe|decimer]

The job directory holds `manifest.json` and an `images/` folder. The worker
writes `results.tsv` beside them and stops. Nothing is uploaded, nothing is
polled, and no session has to stay open.

WHY A MANIFEST AND NOT A GLOB OF THE FOLDER
--------------------------------------------
The manifest carries the `<chemistry>` id for each file. A filename is not a
stable key — the same drawing appears under different names across a document
and its Google Patents mirror — and a worker that joined on filename would
hand back answers the pipeline has to re-attribute by guessing.

ERRORS ARE ROWS, NOT EXCEPTIONS. One drawing the model chokes on must not
cost the other 600. Each failure is written as a row with an `error` and no
`smiles`, so the count of attempts always equals the count of images, and a
missing row means the worker died rather than that a drawing was hard.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

FIELDS = ("chemistry_id", "image_file", "smiles", "confidence",
          "recogniser", "error")


def load_molscribe():
    """MolScribe. Returns `predict(path) -> (smiles, confidence)`."""
    import torch
    from molscribe import MolScribe
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
    model = MolScribe(ckpt, torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))

    def predict(path: str):
        r = model.predict_image_file(path)
        return r.get("smiles", ""), r.get("confidence", "")
    return predict


def load_decimer():
    """DECIMER >= 2.8. Returns `predict(path) -> (smiles, confidence)`.

    `confidence=True` returns `(smiles, [(token, score), ...])`. The mean of
    those scores is the number recorded — and it is recorded because it is
    offered, not because it predicts anything. Measured on 67 images it does
    not: correct and incorrect reads overlap completely, and the most
    confident wrong answer scored above 51 of the 57 correct ones.
    """
    from DECIMER import predict_SMILES

    def predict(path: str):
        out = predict_SMILES(path, confidence=True)
        if isinstance(out, tuple) and len(out) == 2:
            smi, toks = out
            try:
                # float(), never isinstance(x, float): these arrive as
                # numpy.float32, for which isinstance is False, and a filter
                # written that way silently discards every score.
                vals = [float(t[1]) for t in toks]
                return smi, (sum(vals) / len(vals) if vals else "")
            except Exception:
                return smi, ""
        return out, ""
    return predict


LOADERS = {"molscribe": load_molscribe, "decimer": load_decimer}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recognise_worker")
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--recogniser", default="", help="molscribe (default) or decimer")
    a = ap.parse_args(argv)

    manifest = json.loads((a.job_dir / "manifest.json").read_text(encoding="utf-8"))
    name = a.recogniser or manifest.get("recogniser") or "molscribe"
    if name not in LOADERS:
        print(f"unknown recogniser {name!r}; choose from {sorted(LOADERS)}",
              file=sys.stderr)
        return 2

    images = manifest.get("images", [])
    print(f"{name}: {len(images)} drawing(s) from {a.job_dir}", flush=True)
    predict = LOADERS[name]()

    # WRITTEN AS IT GOES, NOT AT THE END. A caller polling this file is the
    # only way to see progress from outside — the process is detached, because
    # any interactive channel to a GPU runtime times out long before a
    # 649-image job finishes. Writing once at the end made the file absent for
    # twenty minutes, during which a healthy run and a dead one look exactly
    # the same. It also means a crash at image 600 keeps the first 599.
    out = a.job_dir / "results.tsv"
    rows = []
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for i, item in enumerate(images, 1):
            path = a.job_dir / item["file"]
            row = {"chemistry_id": item["id"],
                   "image_file": Path(item["file"]).name,
                   "smiles": "", "confidence": "", "recogniser": name,
                   "error": ""}
            try:
                smi, conf = predict(str(path))
                row["smiles"] = smi or ""
                row["confidence"] = "" if conf == "" else f"{float(conf):.4f}"
                if not smi:
                    row["error"] = "empty prediction"
            except Exception as e:                # one bad drawing, not the run
                row["error"] = f"{type(e).__name__}: {e}"[:200]
                traceback.print_exc(limit=1)
            w.writerow(row)
            rows.append(row)
            if i % 25 == 0 or i == len(images):
                fh.flush()
                print(f"  {i}/{len(images)}", flush=True)

    ok = sum(1 for r in rows if r["smiles"])
    print(f"wrote {out}  —  {ok}/{len(rows)} recognised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
