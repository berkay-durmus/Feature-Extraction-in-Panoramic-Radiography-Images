from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from panoramic_features import __version__
from panoramic_features.export import write_table
from panoramic_features.pipeline import find_pairs, process_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panoramic-features",
        description="Extract third-molar and mandibular canal features from panoramic radiographs.",
    )
    parser.add_argument("input_dir", type=Path, help="folder with images and same-named .xml files")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("FeatureList.xlsx"), help=".xlsx or .csv file"
    )
    parser.add_argument(
        "--missing",
        type=float,
        default=None,
        help="value written for features that cannot be computed (default: empty cell)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.input_dir.is_dir():
        logging.error("Not a directory: %s", args.input_dir)
        return 2
    if not find_pairs(args.input_dir):
        logging.error("No image/annotation pairs found in %s", args.input_dir)
        return 1

    start = time.perf_counter()
    count = write_table(process_folder(args.input_dir), args.output, args.missing)
    logging.info("Wrote %d rows to %s in %.1fs", count, args.output, time.perf_counter() - start)
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
