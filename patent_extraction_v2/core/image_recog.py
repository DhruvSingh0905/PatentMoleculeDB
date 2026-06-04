"""Image-recognition arm (v2) — direct DECIMER on Google-Patents C##### figures.

v2's structure figures arrive as Google's *already-cropped, single-structure*
PNG URLs (`…-C#####.png`). Validated by eyeballing real samples: each C-number
is one structure (the first one or two are the generic Markush scaffold,
"Formula I"; the rest are specific complete molecules). So we feed each PNG
straight to DECIMER — no PDF cropping, no DECIMER-Segmentation.

Routing is NON-vision by design (project decision: vision models do poorly on
Markush structures). DECIMER itself separates the two classes for free:

  - Complete molecule  → DECIMER emits a parseable SMILES whose main fragment
    validates with MW in range  → emit a Compound (extraction_method=image_decimer).
  - Markush scaffold   → DECIMER emits pseudo-atom tokens ([R1], [X10], [G], …)
    that fail RDKit parsing  → NOT emitted; recorded as a markush tag for the
    later scaffold+substituent enumeration phase.

The emit gate is purely `validate_smiles(main) AND MIN_MW<=MW<=MAX_MW`; the
R-group token regex only *labels* a non-emitted figure markush-vs-fail for the
audit (so a legitimate bracketed element like boron is never wrongly excluded —
it would pass validate_smiles and be emitted).

SEQUENTIAL by design: DECIMER is a heavy local TF model; parallel inference
crashes the user's machine (same constraint as MinerU). Do NOT add threads here.
"""
from __future__ import annotations

import logging
import re
import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .models import (
    Compound, CompoundProvenance, CompoundSource, ConfidenceTier, IupacSource,
)
from .smiles_utils import (
    validate_smiles, canonicalize_smiles, get_inchikey, molecular_weight,
    strip_salt,
)

logger = logging.getLogger(__name__)

MIN_MW = 150.0
MAX_MW = 1500.0

# Pseudo-atom tokens DECIMER emits at Markush variable positions, e.g.
# [R1] [R9] [R8d] [R11] [X] [X1] [X10] [G] [L] [W] [Z] [Q] [A]. Boron ([B])
# and yttrium ([Y]) are deliberately excluded — they're real elements; if a
# compound legitimately contains them it will parse + be emitted on its own
# merits. This regex is for LABELLING non-emitted figures only.
_RGROUP_TOKEN = re.compile(r"\[(?:R\d*[a-z]?|X\d*|G|L|W|Z|Q|A)\]")

# Parse the C-number out of a Google figure URL/filename:
#   US10544143-20200128-C00444.png  →  C00444
_CNUM_RE = re.compile(r"-(C\d+)\.png$", re.IGNORECASE)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


@dataclass
class ImageRecogResult:
    complete_compounds: list[Compound] = field(default_factory=list)
    markush_tags: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _cnum_from_url(url: str) -> str:
    m = _CNUM_RE.search(url)
    return m.group(1) if m else Path(url).stem


def _download(url: str, dest: Path) -> bool:
    """Download a figure PNG to `dest` (cached; skip if already present)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as r:
            data = r.read()
        if not data:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:  # network / 404 / timeout — log and move on
        logger.warning(f"image_recog: download failed {url}: {e}")
        return False


def classify_decimer_output(raw: str) -> tuple[str, str]:
    """Map a raw DECIMER prediction → (verdict, main_fragment_smiles).

    verdict ∈ {"complete", "markush", "fail"}:
      - complete: main fragment is a valid molecule with MW in range
      - markush:  not a valid molecule AND carries R-group placeholder tokens
      - fail:     not valid and no obvious R-group tokens (OCR miss)
    """
    raw = (raw or "").strip()
    if not raw:
        return "fail", ""
    main = raw.split(".")[0].strip()
    if validate_smiles(main):
        mw = molecular_weight(main)
        if mw is not None and MIN_MW <= mw <= MAX_MW:
            return "complete", main
        return "fail", main  # parses but implausible size → don't trust
    # Not a valid molecule. Is it a Markush scaffold or just an OCR miss?
    if _RGROUP_TOKEN.search(raw) or "*" in raw:
        return "markush", main
    return "fail", main


def _decimer_predict(image_path: str) -> str:
    """Lazy DECIMER call (importing DECIMER loads TensorFlow — keep it out of
    module import so unrelated callers don't pay the cost)."""
    from DECIMER import predict_SMILES  # type: ignore[import-not-found]
    return predict_SMILES(image_path) or ""


def run_image_recognition(
    patent_id: str,
    figure_urls: list[str],
    existing_inchikeys: set[str] | None = None,
    limit: int | None = None,
    images_dir: Path | None = None,
) -> ImageRecogResult:
    """Run the direct-DECIMER arm over a patent's C##### figure URLs.

    Args:
        patent_id: e.g. "US10544143".
        figure_urls: ordered list of Google C##### PNG URLs.
        existing_inchikeys: InChIKeys already extracted (text/GP) — complete
            molecules whose InChIKey is in this set are skipped (dedup, so we
            only surface genuinely novel image-recovered compounds).
        limit: process only the first N figures (sampling / smoke runs).
        images_dir: download cache dir (default output_v2/images/{patent_id}).

    Returns ImageRecogResult(complete_compounds, markush_tags, stats). SEQUENTIAL.
    """
    existing = set(existing_inchikeys or set())
    if images_dir is None:
        images_dir = config.IMAGES_DIR / patent_id
    images_dir.mkdir(parents=True, exist_ok=True)

    urls = figure_urls[:limit] if limit else list(figure_urls)
    result = ImageRecogResult()
    seen_emitted: set[str] = set()
    n_download_fail = n_complete = n_markush = n_fail = n_dup = 0

    for url in urls:
        cnum = _cnum_from_url(url)
        dest = images_dir / f"{patent_id}-{cnum}.png" if not url.endswith(".png") \
            else images_dir / Path(url).name
        if not _download(url, dest):
            n_download_fail += 1
            continue

        try:
            raw = _decimer_predict(str(dest))
        except Exception as e:
            logger.warning(f"image_recog: DECIMER failed on {dest.name}: {e}")
            n_fail += 1
            continue

        verdict, main = classify_decimer_output(raw)
        if verdict == "markush":
            n_markush += 1
            result.markush_tags.append({
                "cnum": cnum, "url": url, "raw_smiles": raw[:200],
                "reason": "r_group_tokens",
            })
            continue
        if verdict == "fail":
            n_fail += 1
            continue

        # verdict == "complete"
        canonical = canonicalize_smiles(main)
        if not canonical:
            n_fail += 1
            continue
        ik = get_inchikey(canonical)
        if not ik:
            n_fail += 1
            continue
        if ik in existing or ik in seen_emitted:
            n_dup += 1
            continue
        seen_emitted.add(ik)

        salt = strip_salt(canonical)
        prov = CompoundProvenance(
            route="image_pipeline",          # closed Literal — must be this value
            stage="decimer_local",
            image_path=str(dest),
            chain=["decimer_local"],
        )
        result.complete_compounds.append(Compound(
            patent_id=patent_id,
            example_number=f"IMG_{cnum}",
            iupac_name=None,
            iupac_source=IupacSource.GENERATED,
            smiles_from_image=main,
            canonical_smiles=canonical,
            inchikey=ik,
            parent_smiles=salt.get("parent_smiles"),
            parent_inchikey=salt.get("parent_inchikey"),
            source=CompoundSource.EXEMPLIFIED,
            confidence_tier=ConfidenceTier.SINGLE_VALIDATED,
            extraction_method="image_decimer",
            provenance=prov,
            image_path=str(dest),
            processing_status="validated",
        ))
        n_complete += 1

    result.stats = {
        "patent_id": patent_id,
        "figures_seen": len(urls),
        "download_failed": n_download_fail,
        "complete_emitted": n_complete,
        "markush_tagged": n_markush,
        "decimer_fail": n_fail,
        "dedup_skipped": n_dup,
    }
    logger.info(
        f"image_recog {patent_id}: {len(urls)} figs → {n_complete} complete "
        f"(novel), {n_markush} markush, {n_fail} fail, {n_dup} dup, "
        f"{n_download_fail} dl-fail"
    )
    return result
