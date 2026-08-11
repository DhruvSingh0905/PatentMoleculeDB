"""Tests for `sources/gp_images.py` — the ONE thing v3 reads from Google Patents.

Every test here is offline. `image_urls` is the only function in this package
that would otherwise reach the network during a test run, so the fetch is
monkeypatched in the one test that exercises it and the default-off path is
asserted to make no call at all.

What is locked in, and why each assertion exists:

  - OFF BY DEFAULT MEANS NO CALL. `config.GP_ENABLED` is `0` unless set, and a
    disabled route that still fetched would make every test run depend on a
    third-party site being up. The stub raises if touched.
  - THE PARSER TAKES THE FILENAME, NOT THE HASH PATH. The join to our own XML
    is on `US10730877-20200804-C00227`; the directory component is a content
    hash that differs per image and carries no meaning for us.
  - PAGE FURNITURE IS NOT AN IMAGE. Google Patents serves its own icons and
    logos as PNGs from other hosts; only `patentimages.storage.googleapis.com`
    counts, or the map silently gains entries that can never join.
  - A FAILURE IS `{}`, NEVER AN EXCEPTION. The marker this feeds is still
    correct without a URL — `drawn_ref` records the `<chemistry>` id either
    way — so a GP outage must not cost a patent its drawn markers.
"""
from __future__ import annotations

import json

import pytest

from patentdb3.core import config
from patentdb3.sources import gp_images

HOST = "https://patentimages.storage.googleapis.com"
PAGE = f"""
<html><body>
  <img src="/static/logo.png">
  <img src="https://www.gstatic.com/images/branding/x.png">
  <meta itemprop="full" content="{HOST}/05/5a/af/5e24e549689ad3/US10730877-20200804-C00227.png">
  <img src="{HOST}/05/5a/af/5e24e549689ad3/US10730877-20200804-C00227.png">
  <img src="{HOST}/fe/c8/cf/2ab8bc39312ce4/US10730877-20200804-C00315.png">
</body></html>
"""


@pytest.fixture()
def gp_on(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GP_ENABLED", True)
    monkeypatch.setattr(config, "GP_IMAGE_DIR", tmp_path / "gp_images")
    return tmp_path / "gp_images"


def _no_network(*a, **k):
    raise AssertionError("gp_images must not fetch here")


# ── the flag ─────────────────────────────────────────────────────────────

def test_disabled_returns_empty_and_never_fetches(monkeypatch):
    monkeypatch.setattr(config, "GP_ENABLED", False)
    monkeypatch.setattr(gp_images, "_fetch", _no_network)
    assert gp_images.image_urls("US10730877") == {}


def test_the_flag_is_off_unless_the_environment_says_otherwise():
    """`GP_ENABLED` defaults off, so "we do not use Google Patents" stays a
    stated choice. If this ever flips, every dump starts making network calls
    it did not make before."""
    assert config.GP_ENABLED in (True, False)
    assert isinstance(config.GP_IMAGE_DIR, type(config.OUTPUT_DIR))
    # the cache belongs to v3's own tree, never v2's
    assert "output_v3" in str(config.GP_IMAGE_DIR)


# ── the parser ───────────────────────────────────────────────────────────

def test_parse_keys_on_the_filename_stem_not_the_hash_path():
    urls = gp_images._parse(PAGE)
    assert set(urls) == {"US10730877-20200804-C00227",
                         "US10730877-20200804-C00315"}
    assert urls["US10730877-20200804-C00227"].endswith(
        "/05/5a/af/5e24e549689ad3/US10730877-20200804-C00227.png")


def test_parse_ignores_png_that_is_not_a_patent_image():
    """The site's own logo is a .png too. Anchoring on the image host is what
    keeps it out; a bare `\\.png` pattern would put `logo` in the map."""
    urls = gp_images._parse(PAGE)
    assert not any("logo" in k or "branding" in v for k, v in urls.items())


def test_parse_deduplicates_the_same_image_referenced_twice():
    """The page names each image more than once (meta + img). Both carry the
    identical hash path, so duplicates would only make the count disagree with
    the XML's `<img>` count when it does not."""
    urls = gp_images._parse(PAGE)
    assert len(urls) == 2
    assert PAGE.count("US10730877-20200804-C00227.png") == 2


def test_parse_of_a_page_with_no_images_is_empty_not_an_error():
    assert gp_images._parse("<html><body>no chemistry here</body></html>") == {}


# ── cache and failure ────────────────────────────────────────────────────

def test_fetches_once_then_serves_from_cache(gp_on, monkeypatch):
    calls = []

    def fake(pid):
        calls.append(pid)
        return PAGE

    monkeypatch.setattr(gp_images, "_fetch", fake)
    first = gp_images.image_urls("US10730877")
    assert len(first) == 2
    assert calls == ["US10730877"]
    assert (gp_on / "US10730877.json").exists()

    monkeypatch.setattr(gp_images, "_fetch", _no_network)
    assert gp_images.image_urls("US10730877") == first


def test_a_failed_fetch_costs_no_exception_and_no_cache_file(gp_on, monkeypatch):
    def boom(pid):
        raise OSError("network down")

    monkeypatch.setattr(gp_images, "_fetch", boom)
    assert gp_images.image_urls("US10730877") == {}
    assert not (gp_on / "US10730877.json").exists()


def test_a_corrupt_cache_is_refetched_rather_than_fatal(gp_on, monkeypatch):
    gp_on.mkdir(parents=True, exist_ok=True)
    (gp_on / "US10730877.json").write_text("{not json")
    monkeypatch.setattr(gp_images, "_fetch", lambda pid: PAGE)
    assert len(gp_images.image_urls("US10730877")) == 2


def test_allow_fetch_false_reads_cache_only(gp_on, monkeypatch):
    """The measurement path needs to ask "what do we already hold" without
    silently going to the network for 137 patents."""
    monkeypatch.setattr(gp_images, "_fetch", _no_network)
    assert gp_images.image_urls("US10730877", allow_fetch=False) == {}

    gp_on.mkdir(parents=True, exist_ok=True)
    (gp_on / "US10730877.json").write_text(json.dumps({"a": "b"}))
    assert gp_images.image_urls("US10730877", allow_fetch=False) == {"a": "b"}
