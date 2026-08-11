"""Rendered images for compounds the patent DREW instead of naming.

WHY THIS IS AN IMAGE MODULE AND NOT A STRUCTURE MODULE
--------------------------------------------------------
Google Patents publishes chemical annotations per patent, and the obvious hope
is that they fill the drawn-structure gap directly. Measured against the live
HTML, they do not, and the reason is structural rather than fixable:

Every GP compound block carries eleven microdata fields — `name`, `id`,
`count`, `smiles`, `inchi_key`, `domain`, `sections`, `match`, `similarity`,
`svg_small`, `svg_large`. `domain` reads **"Chemical class"**, and the names
include bare scaffolds (`1h-pyrrolo[3,2-b]pyridine`) and the literal string
`compounds`. `id` is the InChIKey itself, or a numeric entity id
(`150000001875`). `sections` says which SECTION of the document the entity was
seen in — `description`, `claims`, `abstract`, `title`.

That is entity annotation over text, the same model SureChEMBL uses. It is not
per-drawing recognition, and the counts say so plainly: US10214537 carries
2,078 annotated compounds against 1,342 images. None of those eleven fields
names the patent's own compound number, which is the only key our assay rows
carry — so GP's structures cannot be joined to our measurements at all. This
module therefore does not read `smiles`, `inchi_key`, or any other annotation
field, and adding them later would not make them joinable.

WHAT IT DOES READ, AND WHY THAT PART IS EXACT
-----------------------------------------------
`figure_image_urls` matches our own XML's `<img file="...">` list one for one,
`.TIF` -> `.png`: 1,323 = 1,323 on US10730877, 1,342 = 1,342 on US10214537.
The filename (`US10730877-20200804-C00227`) is a stable identifier present on
BOTH sides, so the join is a dictionary lookup rather than a heuristic. Run end
to end — cid -> `<chemistry>` -> image filename -> GP URL — it resolved
1,065/1,065, 480/480 and 843/843 on the three patents tested.

The URL cannot be constructed. Its directory component is a content hash
(`.../05/5a/af/5e24e549689ad3/US10730877-20200804-C00227.png`), which is the
entire reason this module fetches and caches anything instead of deriving a
path from the filename.

WHAT THIS BUYS, STATED HONESTLY
---------------------------------
A URL is not a structure. This does not resolve one drawn compound; it turns
`drawn_ref`, an opaque `<chemistry>` id meaningful only inside the XML, into
something a human can open and an image-to-structure model could be pointed
at. That is the whole claim. Any actual recovery would come from a recognition
step that does not exist in this tree.

Gated on `config.GP_ENABLED` (default off). Disabled, `image_urls` returns
`{}` — never a partial or stale answer.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

from ..core import config

logger = logging.getLogger(__name__)

# The rendered-image host. Anchored to `patentimages.storage.googleapis.com`
# rather than any `.png` on the page so that page furniture — icons, logos,
# the site's own chrome — cannot enter the map.
_IMAGE_URL = re.compile(
    r"https://patentimages\.storage\.googleapis\.com/[A-Za-z0-9/_.-]+?"
    r"/([A-Za-z0-9_-]+)\.png")

_GP_URL = "https://patents.google.com/patent/{pid}/en"
_UA = "Mozilla/5.0 (compatible; patentdb3/1.0)"
_TIMEOUT_S = 60


def _cache_path(patent_id: str):
    return config.GP_IMAGE_DIR / f"{patent_id}.json"


def _parse(html: str) -> dict[str, str]:
    """`{image filename stem -> url}` from a Google Patents page.

    Deduplicated by stem, first occurrence winning: the same image is
    referenced more than once on the page (thumbnail and full view) and both
    references carry the identical hash path, so the choice is immaterial —
    but leaving duplicates in would make the count look like it disagreed with
    the XML's `<img>` count when it does not.
    """
    out: dict[str, str] = {}
    for m in _IMAGE_URL.finditer(html):
        out.setdefault(m.group(1), m.group(0))
    return out


def _fetch(patent_id: str) -> str:
    req = urllib.request.Request(_GP_URL.format(pid=patent_id),
                                 headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
        return r.read().decode("utf-8", errors="replace")


def image_urls(patent_id: str, *, allow_fetch: bool = True) -> dict[str, str]:
    """`{image filename stem -> rendered png URL}` for one patent.

    Returns `{}` — never raises — when the flag is off, when the network is
    unavailable, or when GP has no page for this patent. A caller cannot tell
    those apart from the return value alone and should not try to: the point
    of the marker this feeds is that a missing URL costs nothing, because the
    `<chemistry>` id is still recorded either way.
    """
    if not config.GP_ENABLED:
        return {}

    path = _cache_path(patent_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as e:
            logger.warning("gp_images: %s — unreadable cache %s: %r",
                           patent_id, path, e)
            # fall through and re-fetch rather than fail; a corrupt cache is
            # not a reason to lose the patent
    if not allow_fetch:
        return {}

    try:
        html = _fetch(patent_id)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning("gp_images: %s — fetch failed: %r", patent_id, e)
        return {}

    urls = _parse(html)
    try:
        config.GP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(urls, indent=1, sort_keys=True))
    except OSError as e:
        logger.warning("gp_images: %s — could not write cache: %r", patent_id, e)
    logger.info("gp_images: %s — %d rendered image URL(s)", patent_id, len(urls))
    return urls
