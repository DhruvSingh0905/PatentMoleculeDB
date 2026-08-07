"""Where UTF-8 first got decoded as Latin-1, and everything that followed.

Google Patents answers `GET /patent/{id}/en` with `Content-Type: text/html`
and **no charset parameter**. `requests` then applies RFC-2616's default for
`text/*` (`requests.utils.get_encoding_from_headers` -> `ISO-8859-1`), so
`Response.text` hands back a UTF-8 page decoded as Latin-1. Measured live
2026-08-07 against `https://patents.google.com/patent/US10214537/en`:

    Content-Type: 'text/html'          r.encoding: ISO-8859-1
    content  b' is: (i) Cl, Br, I, \\xe2\\x80\\x94CN, \\xe2\\x80\\x94'
    r.text    ' is: (i) Cl, Br, I, \\xe2\\x80\\x94CN, \\xe2\\x80\\x94'   (as Latin-1)

`routes/google_patents.fetch_patent_text` reads `r.text` and writes it to
`output_v2/gpatents_cache/{pid}.json`, so the misdecode is persisted once and
read forever. Every one of the 209 cached files carries it — 661,943 lead/
continuation pairs — while all 137 `output_v2/uspto_xml/*.xml` carry none
(the XML spells the same characters `&#x2014;`).

Three consequences, each of which was patched downstream of here before:

  * `normalizer.repair_mojibake`'s lookup table only knew a handful of the
    sequences, so `═` (`\\xe2\\x95\\x90`), `≡`, `“”` and the em/en spaces
    survived Stage 0 into `route_audit.json` snippets.
  * The HARVEST model, shown `â` followed by two invisible C1 bytes,
    transcribed the visible lead byte alone — 18 of 116 stored regexes in
    `assay_patterns.discoveries.json` contain `â`/`Â`, 12 inside a character
    class. The cached response for US10246453 says so in the model's own
    words: "The compound id prefix is `â` (an em-dash)".
  * `f74170a7e3509c03` = `(?P<cid>\\d+)\\s+(?P<value0>[\\d.â]+)\\s+(?P<value1>[\\d.â]+)`
    matched US10214537's empty `CD69 IC 50 value (nM)` column — 375 cells that
    read `—` in the patent's own CALS — and shipped them as `0.0`, because the
    null guard in `apply_patterns_to_text` lists `"—"` and never saw one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from patentdb.core.assay_fsm.normalizer import normalize_page, repair_mojibake


# ── The misdecode, reproduced from the bytes GP actually serves ──────────────

# `—CN` — the em-dash Google Patents serves as UTF-8 and requests decodes as
# Latin-1 when the response carries no charset.
_UTF8_BYTES = " is: (i) Cl, Br, I, —CN".encode("utf-8")
_MISDECODED = _UTF8_BYTES.decode("latin-1")


class _FakeResponse:
    """Just enough of `requests.Response` to exercise the decode decision.

    `text` is a property on the real class that consults `self.encoding`,
    which `requests` sets from the Content-Type header. Reproducing that
    relationship is the whole point of the test.
    """

    def __init__(self, body: bytes, content_type: str) -> None:
        self.content = body
        self.status_code = 200
        self.headers = {"content-type": content_type}
        from requests.utils import get_encoding_from_headers
        self.encoding = get_encoding_from_headers(self.headers)

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", "replace")


def test_requests_defaults_a_charsetless_html_response_to_latin1():
    """The premise. If this ever stops holding, the fix below is dead code."""
    r = _FakeResponse(_UTF8_BYTES, "text/html")
    assert r.encoding == "ISO-8859-1"
    assert r.text == _MISDECODED
    assert "—" not in r.text


def test_fetch_patent_text_decodes_utf8_when_the_server_omits_charset(
    tmp_path, monkeypatch
):
    """The fix, at the line where the bytes are first turned into characters.

    A charsetless `text/html` response must be read as UTF-8 — what HTML5
    mandates and what Google Patents actually sends — not as Latin-1.
    """
    from patentdb.routes import google_patents as gp

    body = (
        '<section itemprop="description" x="1">'
        "<p>Compound 1, IC 50 — , 5 μM, 1.6 ± 0.1</p>"
        "</section>"
        '<section itemprop="claims"><p>A compound of —CH 3 .</p></section>'
    ).encode("utf-8")

    monkeypatch.setattr(gp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        gp.requests, "get",
        lambda *a, **k: _FakeResponse(body, "text/html"),
    )
    monkeypatch.setattr(gp.time, "sleep", lambda *_a, **_k: None)

    out = gp.fetch_patent_text("USTEST1")
    assert out is not None
    for field in ("description", "claims"):
        assert "â" not in out[field], (
            f"{field} still carries the Latin-1 lead byte")
    assert "—" in out["description"]
    assert "μM" in out["description"]
    assert "1.6 ± 0.1" in out["description"]

    # And what was written to disk is what a later reader will see.
    on_disk = json.loads((tmp_path / "USTEST1.json").read_text(encoding="utf-8"))
    assert "—" in on_disk["description"]
    assert "â" not in on_disk["description"]


def test_fetch_patent_text_honours_an_explicit_charset():
    """A server that DOES declare its charset must still be believed."""
    r = _FakeResponse("café — 5 μM".encode("utf-8"),
                      "text/html; charset=UTF-8")
    assert r.encoding == "UTF-8"
    assert r.text == "café — 5 μM"


def test_stale_cache_is_repaired_on_read(tmp_path, monkeypatch):
    """209 cache files already hold the misdecode and a re-fetch is not free.

    `fetch_patent_text` already hydrates old caches defensively (it
    HTML-unescapes SMILES written before that ran at fetch time). Same idiom.
    """
    from patentdb.routes import google_patents as gp

    monkeypatch.setattr(gp, "CACHE_DIR", tmp_path)
    (tmp_path / "USTEST2.json").write_text(json.dumps({
        "description": _MISDECODED,
        "claims": "5 Î¼M",
        "embedded_compounds": [],
        "figure_image_urls": [],
    }), encoding="utf-8")

    out = gp.fetch_patent_text("USTEST2")
    assert out is not None
    assert "—CN" in out["description"]
    assert out["claims"] == "5 μM"


# ── Stage 0 must repair the whole family, not a hand-listed few ─────────────

@pytest.mark.parametrize("moji, real, what", [
    ("â\x80\x94", "—", "em-dash — the null marker in assay tables"),
    ("â\x80\x93", "–", "en-dash"),
    ("â\x88\x92", "−", "minus sign"),
    ("Î¼", "μ", "micro"),
    ("Â±", "±", "plus-minus"),
    # These four survived Stage 0 into route_audit.json snippets on 13 of 22
    # patents, because the lookup table spelled its keys in cp1252
    # (`â€™`, `â‰¤`, `â‚‚`) while the corruption is Latin-1 (`â\x80\x99`).
    ("â\x95\x90", "═", "double horizontal — a drawn double bond"),
    ("â\x89¡", "≡", "identical to — a drawn triple bond"),
    ("â\x80\x9c", "“", "left double quote"),
    ("â\x80\x83", "\u2003", "em space — US10246453's cid separator"),
    ("â\x80\xa0", "†", "dagger"),
    ("â\x89¤", "≤", "less-than-or-equal — a qualifier"),
    ("â\x82\x82", "₂", "subscript two"),
])
def test_repair_mojibake_covers_the_whole_family(moji, real, what):
    assert repair_mojibake(f"x {moji} y") == f"x {real} y", what


def test_repair_mojibake_is_idempotent():
    once = repair_mojibake("IC 50 â\x80\x94 5 Î¼M")
    assert repair_mojibake(once) == once


def test_repair_mojibake_leaves_clean_text_alone():
    """The XML tier is already correct; Stage 0 must not touch it."""
    clean = "IC 50 — 5 μM, 1.6 ± 0.1, ═O, δ (ppm), °C"
    assert repair_mojibake(clean) == clean


def test_repair_mojibake_repairs_a_mixed_document():
    """`load_full_patent_text` concatenates the clean XML with the corrupt GP
    scrape and normalises the JOIN. A whole-string `encode('latin-1')` fails on
    the XML's real `—`, so a whole-string repair silently does nothing to
    the GP half. The repair has to be per-run."""
    mixed = "XML: 5 μM — done\nGP: 5 Î¼M â\x80\x94 done"
    assert repair_mojibake(mixed) == (
        "XML: 5 μM — done\nGP: 5 μM — done")


def test_normalize_page_repairs_before_nfkc():
    out = normalize_page("Ex. 1 13 â\x80\x94 2 16 â\x80\x94")
    assert "â" not in out
    assert out.count("—") == 2


# ── The em-dash reaches the null guard, so no `0.0` is fabricated ───────────

def test_empty_column_no_longer_ships_as_a_measurement():
    """US10214537 TABLE: `Ex. No. | PI3K delta IC 50 (nM) | CD69 IC 50 (nM)`
    with the CD69 column entirely em-dashes. 375 shipped rows read
    `CD69 IC50 (nM) = 0.0`. The real value in the FIRST column must survive."""
    from patentdb.core.assay_fsm import assay_pattern_library as lib

    row = "1 13 — \n2 16 — \n3 3 — "
    entry = {
        "key": "test",
        "regex": r"(?P<cid>\d+)\s+(?P<value0>[\d.—]+)\s+(?P<value1>[\d.—]+)",
        "column_assays": ["PI3K delta IC50 (nM)", "CD69 IC50 (nM)"],
        "header_text": "Ex. No. PI3K delta IC 50 value (nM) CD69 IC 50 value (nM)",
        "first_seen_patent": "US10214537",
        "status": "pending",
    }
    header = ("Ex. No. PI3K delta IC 50 value (nM) CD69 IC 50 value (nM)\n")
    rows = lib.apply_patterns_to_text(
        header + row, "US10214537", fresh_patterns=[entry])
    by_assay = {}
    for r in rows:
        by_assay.setdefault(r["assay_name"], []).append(r)
    assert "CD69 IC50 (nM)" not in by_assay, (
        "an em-dash cell shipped as a measurement: "
        f"{by_assay.get('CD69 IC50 (nM)')}")
    # Row count is not asserted: the repaired library entry `f74170a7e3509c03`
    # now carries this exact shape, so it fires alongside the fresh one. What
    # must hold is that every compound keeps its real first-column value.
    got = {(r["compound_id"], r["value"])
           for r in by_assay.get("PI3K delta IC50 (nM)", [])}
    assert got == {("1", 13.0), ("2", 16.0), ("3", 3.0)}, (
        "the real first-column values must survive the null second column")


# ── The learned artifact must stop carrying the corruption ──────────────────

_DISCOVERIES = (Path(__file__).resolve().parents[1]
                / "data" / "assay_patterns.discoveries.json")
_LEAD = "ÂÃÎÏâ"


@pytest.mark.skipif(not _DISCOVERIES.exists(),
                    reason="runtime accumulation; absent in a fresh clone")
def test_no_stored_pattern_captures_a_mojibake_fragment():
    """A value group whose character class admits `â` matches the LEAD BYTE of
    a character that no longer exists once the source is decoded correctly.
    `f74170a7e3509c03` is the one that shipped 375 fabricated rows."""
    data = json.loads(_DISCOVERIES.read_text(encoding="utf-8"))
    offenders = []
    for t in data.get("tokens", []):
        rx = t.get("regex") or ""
        for cls in __import__("re").finditer(r"\[[^\]]*\]", rx):
            if any(c in _LEAD for c in cls.group(0)):
                offenders.append((t.get("key"), cls.group(0)))
                break
    assert not offenders, (
        f"{len(offenders)} stored regex(es) still admit a mojibake lead byte "
        f"inside a character class: {offenders[:5]}")
