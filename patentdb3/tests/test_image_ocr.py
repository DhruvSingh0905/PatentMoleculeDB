"""The caption reader, and what it does with a caption it cannot read.

Every string in this file is a REAL OCR result, copied from the run over the
33 captioned images in US10730863 and US11292791. None is invented: a made-up
corruption tests the rule that was written for it and nothing else, which is
how a repair tier ends up with rules for defects the corpus does not have.
"""
import json

import pytest

from patentdb3.sources import image_ocr as IO
from patentdb3.sources import losses, opsin


# ---------------------------------------------------------------- candidates

def _parses(text: str) -> str:
    """The plainest candidate for `text` that OPSIN accepts, or ""."""
    cands = IO.name_candidates(text)
    smiles = opsin.batch(cands, "SMILES", "test")
    for c, s in zip(cands, smiles):
        if s.strip():
            return c
    return ""


def test_leading_atom_labels_are_dropped():
    """`F (E)-3-...` — OCR read a label off the structure and prepended it."""
    assert IO._strip_labels("F (E)-3-(2-(4-yl)methoxy)").startswith("(E)-")
    assert IO._strip_labels("HO- (E)-7-(2-").startswith("(E)-")
    assert IO._strip_labels("OH C Cl 4-(5-(4-").startswith("4-(5-")


def test_a_stereo_descriptor_is_never_mistaken_for_a_label():
    """`(E)-3-(2-...` is ONE token, so a token-wise strip cannot eat it.

    A character class holding `(`, `)` and `-` did exactly that, and the
    result still parsed often enough to be adopted — which is why the strip
    is token-wise and why this test exists.
    """
    assert IO._strip_labels("(E)-3-(2-phenyl)").startswith("(E)-")


def test_a_strip_that_leaves_nothing_namelike_is_refused():
    assert IO._strip_labels("OH CH3 NH2") == "OH CH3 NH2"


def test_one_caption_needs_two_different_readings_of_J():
    """`2Joctan` wants `]` and `vinylJimidazo` wants `)`, in one name.

    A global replace produces one reading or the other and neither is the
    name. The sites resolve independently.
    """
    out = list(IO._resolve_sites("bicyclo[2.2.2Joctan-1-yl)vinylJimidazo"))
    assert "bicyclo[2.2.2]octan-1-yl)vinyl)imidazo" in out


def test_a_lost_opening_bracket_is_restored():
    """`E)2-(4(2-...` — the patent wrote `(E)-2-`, OCR dropped the `(`."""
    assert "(E)2-(4(2-" in list(IO._repair_opening("E)2-(4(2-"))


def test_a_name_that_already_opens_correctly_is_left_alone():
    """The test is structural: a `)` before any `(` closes nothing."""
    for good in ("(1S,4R)-4-((S)-6-", "4-((7S)-6-(methoxycarbonyl)",
                 "methyl (S)-2-benzyl-7-methyl"):
        assert list(IO._repair_opening(good)) == [good]


def test_a_bracket_printed_twice_across_a_wrap_offers_both_drops():
    out = list(IO._doubled_close("yloxy } Jethyl"))
    assert "yloxy Jethyl" in out          # the first member was the echo
    assert "yloxy }ethyl" in out          # the second was
    assert out[0] == "yloxy } Jethyl"     # untouched first — `) )` is legal


def test_a_letter_to_letter_space_is_a_site_not_a_deletion():
    """`carboxylic acid` and `y loxy` are the same shape. Both are offered."""
    out = list(IO._resolve_spaces("carboxylic acid"))
    assert out[0] == "carboxylic acid"    # plainest reading first, always
    assert "carboxylicacid" in out        # the render split one word
    assert "carboxylic)acid" in out       # a `)` came back as whitespace


def test_space_enumeration_stops_when_the_text_is_too_damaged():
    many = " ".join("abcd" for _ in range(IO._MAX_SPACE_SITES + 3))
    assert list(IO._resolve_spaces(many)) == [many]


def test_annotation_after_the_name_is_cut_at_the_suffix():
    text = ("(1S,4R)-4-(phenyl)cyclohexane-1-carboxylic acid 1st eluting "
            "isomer")
    assert any(c.endswith("carboxylic acid") for c in IO._trim_annotation(text))


def test_a_cut_can_never_land_inside_the_name():
    """The cut is placed AFTER a name-ending word, so `...acid` survives."""
    text = "4-(phenyl)cyclohexane-1-carboxylic acid"
    assert list(IO._trim_annotation(text)) == [text]


def test_text_shorter_than_a_name_yields_no_candidates():
    """Median OCR length over 114 images was 15 characters — atom labels."""
    assert IO.name_candidates("CH3 NH2 OH") == []
    assert IO.name_candidates("") == []


# ------------------------------------------------------- real captions, OPSIN

@pytest.mark.parametrize("caption,ends", [
    # a leading label, a `J` for `]`, and a wrap space
    ("F (E)-3-(2-(4-((5-cyclopropyl-3-(2 - (trifluoromethoxy )phenyl)isoxazol-4 "
     "yl)methoxy Jbicyclo[2.2.2]octan-1-yl)vinyl)benzoic acid", "acid"),
    # a lost `(` on the descriptor AND a hyphen missing before `(`
    ("E)2-(4(2-(4-((5- cyclopropyl-3-(3,5-dichloropyridin-4-yl) isoxazol-4-yl)"
     "methoxy )bicyclo[2.2.2]octan- 1-yl)vinyl)phenyl)-N,N-dimethylacetamide",
     "acetamide"),
    # a split word, a doubled bracket, `l` for `1`, and a trailing annotation
    ("OH 4-((7S)-6-(methoxycarbonyl)- 7-methyl-2-(1-(pyridin-2- y loxy } "
     "Jethyl)-6,7,8,9-tetrahydro-3H-imidazo[4,5-fJquinolin - 3-yl)cyclohexane-"
     "l-carboxylic acid eluting isomer", "acid"),
])
def test_a_real_damaged_caption_reaches_a_name_opsin_accepts(caption, ends):
    got = _parses(caption)
    assert got, f"no candidate parsed for {caption[:60]!r}"
    assert got.endswith(ends)


def test_nothing_is_accepted_on_a_transformation_alone():
    """OPSIN is the gate. A caption of pure noise yields no structure."""
    noise = "Jbicyclo _ octan l yl methoxy " * 3
    assert _parses(noise) == ""


# ------------------------------------------------------------------ the mark

class _Item:
    """Stands in for `images.WorkItem`; only the fields `_mark_unread` reads."""
    def __init__(self, cid, pid="US11292791"):
        self.cid, self.patent_id = cid, pid
        self.chemistry_id, self.image_file = f"CHEM-{cid}", f"{pid}-C{cid}.TIF"


def _record_for(caption, tmp_path):
    losses.reset(tmp_path / "loss.jsonl")
    have = [_Item("646")]
    plan = [(0, c) for c in IO.name_candidates(caption)]
    smiles = opsin.batch([c for _, c in plan], "SMILES", "test")
    best = {}
    for (i, c), s in zip(plan, smiles):
        if s.strip() and i not in best:
            best[i] = (c, s)
    IO._mark_unread(have, plan, best, "US11292791")
    losses.flush()
    p = tmp_path / "loss.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_a_caption_blocked_only_by_stereo_is_recorded_not_dropped(tmp_path):
    """`(1S,4R)` on a cyclohexane. OPSIN cannot map those locants.

    Every one of the five captions this pipeline still cannot resolve is this
    compound family. The repair produced the exactly-correct string — checked
    against a hand-corrected name, similarity 1.000 — and OPSIN refused it, so
    the defect is not ours to fix and a rule bought for it would be garbage.
    """
    rows = _record_for(
        "(1S,4R)-4-((S)-6-(methoxycarbonyl)-7-methyl-2-(2- phenylpropan-2-yl)"
        "-6,7,8,9-tetrahydro-3H-imidazo[4,5- f]quinolin-3-yl)cyclohexane-l-"
        "carboxylic acid Nu_", tmp_path)
    assert len(rows) == 1
    assert rows[0]["reason"] == opsin.REASON_STEREO
    assert rows[0]["cid"] == "646"
    assert rows[0]["drawn_file"].endswith(".TIF")   # traceable to the picture
    assert rows[0]["opsin_error"]


def test_a_caption_that_parses_is_not_recorded_as_a_loss(tmp_path):
    rows = _record_for(
        "F (E)-3-(2-(4-((5-cyclopropyl-3-(2 - (trifluoromethoxy )phenyl)"
        "isoxazol-4 yl)methoxy Jbicyclo[2.2.2]octan-1-yl)vinyl)benzoic acid",
        tmp_path)
    assert rows == []


def test_an_image_with_no_caption_is_not_a_loss(tmp_path):
    """It is DECIMER's input. The queue already knows about it."""
    losses.reset(tmp_path / "loss.jsonl")
    IO._mark_unread([_Item("1")], [], {}, "US11292791")
    losses.flush()
    assert (tmp_path / "loss.jsonl").read_text().strip() == ""


# ------------------------------------------------------------- precedence

class _Row:
    """A structure row, in the two shapes `supersede` has to tell apart."""
    def __init__(self, cid, inchikey="", drawn_file="", source="cid_first"):
        self.cid, self.inchikey = cid, inchikey
        self.drawn_file, self.drawn_ref = drawn_file, "CHEM-1"
        self.source = source


def _fake_resolve(monkeypatch, answers):
    """Stand in for the OCR, so precedence is tested without pixels.

    `local_path` is pointed at this test file, which exists — the gate being
    exercised here is the precedence rule, not the file check, and that one
    has its own test below.
    """
    from pathlib import Path

    import patentdb3.sources.images as _images
    from patentdb3.core import config
    monkeypatch.setattr(config, "IMAGE_OCR", True)
    monkeypatch.setattr(IO, "resolve", lambda items, pid: list(answers))
    monkeypatch.setattr(_images.WorkItem, "local_path",
                        property(lambda self: Path(__file__)))


def test_a_caption_replaces_the_drawn_marker_it_answers(monkeypatch):
    """One compound, one row. Adding instead of replacing is the bug.

    A drawn marker and a resolved caption are the same cid. Concatenating them
    puts a blank row and an answer under one compound number, and every
    downstream count reads that compound twice — the shape that pinned the
    repair loop's yield at exactly 0.500.
    """
    answer = _Row("7", inchikey="KEY", drawn_file="a.TIF", source="image_ocr")
    _fake_resolve(monkeypatch, [answer])
    rows = [_Row("7", drawn_file="a.TIF")]
    out, n = IO.supersede(rows, "US1")
    assert n == 1
    assert len(out) == len(rows)                 # never a union
    assert [r.source for r in out] == ["image_ocr"]


def test_a_structure_the_text_track_resolved_is_never_overwritten(monkeypatch):
    """A caption has not earned the right to overrule an asserted name."""
    answer = _Row("7", inchikey="OCRKEY", drawn_file="a.TIF", source="image_ocr")
    _fake_resolve(monkeypatch, [answer])
    rows = [_Row("7", inchikey="TEXTKEY", drawn_file="a.TIF", source="table")]
    out, n = IO.supersede(rows, "US1")
    assert n == 0
    assert out[0].inchikey == "TEXTKEY"


def test_a_row_with_no_picture_is_left_alone(monkeypatch):
    answer = _Row("7", inchikey="KEY", drawn_file="a.TIF", source="image_ocr")
    _fake_resolve(monkeypatch, [answer])
    rows = [_Row("7")]                            # no drawn_file
    assert IO.supersede(rows, "US1") == (rows, 0)


def test_the_route_is_off_by_default():
    """CPU, not risk — see `config.IMAGE_OCR`. It must cost nothing when off."""
    from patentdb3.core import config
    assert config.IMAGE_OCR is False
    rows = [_Row("7", drawn_file="a.TIF")]
    assert IO.supersede(rows, "US1") == (rows, 0)


def test_a_missing_ocr_engine_is_recorded_not_swallowed(tmp_path, monkeypatch):
    """Without easyocr every image returns "" and the run reads as
    "no picture carried a caption" — a finding, not a missing dependency.

    That exact silence already cost one measurement, which reported
    "0 of 24 images have a caption" while the real cause was a loader error.
    """
    import builtins
    from patentdb3.sources import losses
    losses.reset(tmp_path / "loss.jsonl")
    real = builtins.__import__

    def no_easyocr(name, *a, **k):
        if name == "easyocr":
            raise ImportError("no easyocr")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_easyocr)
    out = IO.read_text([tmp_path / "a.png", tmp_path / "b.png"])
    monkeypatch.undo()
    losses.flush()
    assert out == {tmp_path / "a.png": "", tmp_path / "b.png": ""}
    rows = [json.loads(l) for l in
            (tmp_path / "loss.jsonl").read_text().splitlines() if l.strip()]
    assert [r["loss_type"] for r in rows] == ["image_ocr_unavailable"]
    assert rows[0]["images"] == 2


def test_no_local_image_means_no_work_and_no_easyocr(monkeypatch):
    """Self-gating. A patent whose pictures are not on disk must not import
    easyocr, let alone load a model."""
    from patentdb3.core import config
    monkeypatch.setattr(config, "IMAGE_OCR", True)
    monkeypatch.setattr(IO, "resolve", lambda items, pid: pytest.fail(
        "resolve() ran with no local image"))
    rows = [_Row("7", drawn_file="nothing-on-disk.TIF")]
    assert IO.supersede(rows, "US1") == (rows, 0)


# ------------------------------------------- the shared classifier, both tiers

def test_stereo_blocked_uses_the_measurement_not_the_wording():
    """A name OPSIN accepts with stereo ignored was never broken text.

    The second name below is refused by OPSIN for a reason that has nothing to
    do with stereochemistry, and must not be swept into the same bucket.
    """
    stereo = ("(1S,4R)-4-((S)-6-(methoxycarbonyl)-7-methyl-2-(2-phenylpropan-"
              "2-yl)-6,7,8,9-tetrahydro-3H-imidazo[4,5-f]quinolin-3-yl)"
              "cyclohexane-1-carboxylic acid")
    broken = "4-(4-qqqphenyl)-1-methylpiperidine"
    errs = opsin.errors([stereo, broken], "test")
    got = opsin.stereo_blocked([stereo, broken], errs, "test")
    assert stereo in got
    assert broken not in got


def test_the_stereo_flag_never_resolves_anything():
    """It is a diagnostic. `batch` with the flag is called only to classify."""
    name = ("(1S,4R)-4-((S)-6-(methoxycarbonyl)-7-methyl-2-(2-phenylpropan-2-"
            "yl)-6,7,8,9-tetrahydro-3H-imidazo[4,5-f]quinolin-3-yl)cyclohexane"
            "-1-carboxylic acid")
    assert opsin.batch([name], "SMILES", "test")[0].strip() == ""
    assert opsin.batch([name], "SMILES", "test",
                       allow_bad_stereo=True)[0].strip() != ""
