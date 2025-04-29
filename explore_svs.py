#!/usr/bin/env python3
"""
Dumps TIFF metadata from an SVS file.

Usage examples:

# Print metadata in text format (default)
python explore_svs.py some_image.svs

# Print metadata in JSON format
python explore_svs.py some_image.svs --format json
"""

import argparse
import json
from pathlib import Path

import tiffparser


def page_dict(page):
    return {
        "description": page.description.strip(),
        "dtype": str(page.dtype),
        "shape": page.shape,
        "samples_per_pixel": page.samplesperpixel,
        "compression": page.compression.name
        if hasattr(page.compression, "name")
        else page.compression,
        "tags": {k: v.value for k, v in page.tags.items()},
    }


def show(path: Path, out_format: str):
    with path.open("rb") as fp:
        t = tiffparser.TiffFile(fp)
        pages = [page_dict(p) for p in t.pages]

    if out_format == "json":
        print(json.dumps({"pages": pages}, indent=2))
        return

    for i, p in enumerate(pages):
        print(f"Page {i}")
        print(f"  description: {p['description']}")
        print(
            f"  shape: {p['shape']}  dtype: {p['dtype']}  spp: {p['samples_per_pixel']}"
        )
        print(f"  compression: {p['compression']}")
        for tag, val in p["tags"].items():
            print(f"    {tag}: {val}")
        print()


def main():
    ap = argparse.ArgumentParser(description="Dump all TIFF metadata from an SVS file")
    ap.add_argument("svs", help=".svs file to inspect")
    ap.add_argument(
        "-f",
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format (default text)",
    )
    args = ap.parse_args()
    show(Path(args.svs), args.format)


if __name__ == "__main__":
    main()
