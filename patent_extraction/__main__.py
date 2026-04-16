"""CLI entry point: python3 -m patent_extraction US10899738

Usage:
    python3 -m patent_extraction US10899738          # single patent
    python3 -m patent_extraction                     # all 8 patents
    python3 -m patent_extraction --force US10899738  # ignore step cache
    python3 -m patent_extraction --skip-images       # skip image pipeline
"""

import argparse
import logging
import sys

from . import config
from .pipeline import run_patent


def main():
    parser = argparse.ArgumentParser(
        description="Patent compound extraction pipeline"
    )
    parser.add_argument(
        "patent_ids", nargs="*", default=None,
        help="Patent IDs to process (default: all configured patents)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore step cache, re-run everything"
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Skip image extraction (saves cost)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    ids = args.patent_ids or config.PATENT_IDS

    for pid in ids:
        try:
            result = run_patent(
                pid,
                skip_images=args.skip_images,
                force=args.force,
            )
            print(f"\n{pid}: {len(result.exemplified_compounds)} compounds → "
                  f"{config.RESULTS_DIR / pid / 'combined.json'}")
        except Exception as e:
            logging.error(f"{pid}: FAILED — {e}")
            if len(ids) == 1:
                raise


if __name__ == "__main__":
    main()
