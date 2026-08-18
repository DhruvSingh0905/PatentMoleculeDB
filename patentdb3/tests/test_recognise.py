"""The GPU boundary: a manifest goes out, a results file comes back.

What matters here is that neither side needs the other to be running. The
pipeline must work with no GPU anywhere (returning nothing, blocking nothing),
and a worker must be able to satisfy the contract without importing this
package. Both are tested below by construction: the worker's output is written
by hand, exactly as a machine that has never seen patentdb3 would write it.
"""
from __future__ import annotations

import csv
import json

import pytest

from patentdb3 import recognise as R
from patentdb3.core import config


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    return tmp_path


def _worker_writes(path, rows):
    """Exactly what `recognise_worker.py` emits. No patentdb3 involved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, R.WORKER_FIELDS, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def test_off_is_the_default_and_returns_nothing(job, monkeypatch):
    """A plain dump must never wait on a GPU, or reach for one."""
    monkeypatch.setattr(config, "RECOGNISER_BACKEND", "off")
    assert R.structures("USTEST") == {}


def test_an_unknown_backend_degrades_to_off_rather_than_raising(job, monkeypatch):
    monkeypatch.setattr(config, "RECOGNISER_BACKEND", "slurm-someday")
    assert R.backend() == "off"
    assert R.structures("USTEST") == {}


def test_a_missing_results_file_is_empty_not_an_error(job, monkeypatch):
    monkeypatch.setattr(config, "RECOGNISER_BACKEND", "file")
    assert R.structures("USTEST") == {}


def test_the_round_trip(job, monkeypatch, tmp_path):
    """Stage -> a worker writes -> ingest -> the pipeline sees structures."""
    monkeypatch.setattr(config, "RECOGNISER_BACKEND", "file")
    img = tmp_path / "C00001.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    m = R.write_manifest("USTEST", {"CHEM-US-00001": img})
    data = json.loads(m.read_text(encoding="utf-8"))
    assert data["images"] == [{"id": "CHEM-US-00001", "file": "images/C00001.png"}]
    # RELATIVE, always: an absolute path from this machine means nothing on the
    # node that runs the model.
    assert not data["images"][0]["file"].startswith("/")

    out = tmp_path / "results.tsv"
    _worker_writes(out, [{"chemistry_id": "CHEM-US-00001",
                          "image_file": "C00001.png", "smiles": "c1ccccc1",
                          "confidence": "0.98", "recogniser": "molscribe",
                          "error": ""}])
    R.ingest("USTEST", out)
    assert R.structures("USTEST") == {"CHEM-US-00001": "c1ccccc1"}


def test_a_failed_drawing_is_not_stored_as_an_empty_structure(job, monkeypatch, tmp_path):
    """"The model failed" and "this drawing is nothing" must not become the
    same blank downstream."""
    monkeypatch.setattr(config, "RECOGNISER_BACKEND", "file")
    out = tmp_path / "results.tsv"
    _worker_writes(out, [
        {"chemistry_id": "CHEM-1", "image_file": "a.png", "smiles": "",
         "confidence": "", "recogniser": "molscribe", "error": "CUDA OOM"},
        {"chemistry_id": "CHEM-2", "image_file": "b.png", "smiles": "CCO",
         "confidence": "0.9", "recogniser": "molscribe", "error": ""},
    ])
    R.ingest("USTEST", out)
    got = R.structures("USTEST")
    assert got == {"CHEM-2": "CCO"}
    assert "CHEM-1" not in got


def test_the_worker_imports_nothing_from_this_package():
    """THE DECOUPLING, asserted rather than intended.

    `recognise_worker.py` is copied to a GPU node on its own. The moment it
    imports patentdb3, that stops working and nobody finds out until a job
    fails on a machine with no repo checked out.
    """
    import ast
    from pathlib import Path

    src = Path(R.__file__).with_name("recognise_worker.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                assert not n.name.startswith("patentdb3"), n.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import — implies a package"
            assert not (node.module or "").startswith("patentdb3"), node.module


def test_the_worker_writes_the_columns_recognise_reads():
    """One contract, asserted from both ends so they cannot drift apart."""
    import ast
    from pathlib import Path

    src = Path(R.__file__).with_name("recognise_worker.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fields = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == "FIELDS"):
            fields = tuple(e.value for e in node.value.elts)
    assert fields == R.WORKER_FIELDS


def test_the_loop_blocks_rather_than_stalls_when_nothing_is_recognised(monkeypatch):
    """The tier must be testable without a GPU. With the backend off every
    table reports a named blocker and no molecule is invented."""
    from patentdb3.repair import markush_loop as ML

    monkeypatch.setattr(config, "MARKUSH_ASSEMBLY", True)
    monkeypatch.setattr(config, "RECOGNISER_BACKEND", "off")
    p = config.XML_INPUT_DIR / "US9718825.xml"
    if not p.exists():
        pytest.skip("US9718825.xml not cached")
    reports = ML.repair_patent("US9718825", p.read_text(errors="replace"))
    assert reports
    assert all(r.blocked == ML.BLOCK_NO_SCAFFOLD for r in reports)
    assert not any(r.structures for r in reports)
