#!/usr/bin/env python3
"""
De-identifies Whole Slide Image (WSI) files by removing PHI from metadata and macro images.

This script processes WSI files (.svs, .tif, .tiff) to remove potentially identifying information.
It can work with pre-identified bounding boxes from identify_boxes.py or use default/specified
rectangles for masking.

Usage Examples
-------------
# Process slides using pre-identified bounding boxes
uv run python deidentify.py "sample/identified/*.svs" --salt "your-secret-salt" --boxes-json identified_boxes.json

# Basic de-identification without modifying macro images
uv run python deidentify.py "sample/identified/*.{svs,tif,tiff}" \\
    --salt "your-secret-salt-here" \\
    -o sample/deidentified \\
    -m sample/hash_mapping.csv

# Default centered rectangle for macro masking
uv run python deidentify.py "sample/identified/*.{svs,tif,tiff}" \\
    --salt "your-secret-salt-here" \\
    -o sample/deidentified \\
    --macro-description "macro" \\
    --rect 0 0 0 0

# Specify custom rectangle coordinates (x0 y0 x1 y1)
uv run python deidentify.py "path/to/slides/*.svs" \\
    --salt "your-secret-salt-here" \\
    -o output_dir \\
    --rect 100 150 500 600

The script will:
1. Copy input slides to a specified output directory
2. Rename them using a salted hash derived from the original filename
3. Remove the "label" image (which often contains PHI)
4. Strip non-technical metadata fields 
5. Optionally redact the "macro" image by masking areas containing PHI (only if --rect or --boxes-json is specified)
6. Generate a CSV mapping of original to hashed filenames
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import tiffparser
from replace_macro import replace_macro


###############################################################################
# helpers
###############################################################################
def delete_associated_image(slide_path: Path, image_type: str) -> None:
    allowed = {"label", "macro"}
    if image_type not in allowed:
        raise ValueError("image_type must be 'label' or 'macro'")

    with slide_path.open("r+b") as fp:
        t = tiffparser.TiffFile(fp)
        p0 = t.pages[0]
        if "Aperio Image Library" in p0.description:
            pages = [p for p in t.pages if image_type in p.description]
        elif "Aperio Leica Biosystems GT450" in p0.description:
            pages = [t.pages[-2]] if image_type == "label" else [t.pages[-1]]
        else:
            pages = [p for p in t.pages if image_type in p.description]
        if len(pages) != 1:
            return
        page = pages[0]

        fmt, sz = t.tiff.ifdoffsetformat, t.tiff.ifdoffsetsize
        tagnoformat, tagnosize = t.tiff.tagnoformat, t.tiff.tagnosize
        tagsize, unpack = t.tiff.tagsize, struct.unpack

        ifds = [{"this": p.offset} for p in t.pages]
        for i in ifds:
            fp.seek(i["this"])
            (ntags,) = unpack(tagnoformat, fp.read(tagnosize))
            fp.seek(ntags * tagsize, 1)
            i["next_ifd_offset"] = fp.tell()
            (i["next_ifd_value"],) = unpack(fmt, fp.read(sz))

        p_ifd = next(i for i in ifds if i["this"] == page.offset)
        prev_ifd = next((i for i in ifds if i["next_ifd_value"] == page.offset), None)
        if prev_ifd is None:
            return

        for off, bc in zip(
            page.tags["StripOffsets"].value, page.tags["StripByteCounts"].value
        ):
            fp.seek(off)
            fp.write(b"\0" * bc)

        for tag in page.tags.values():
            fp.seek(tag.valueoffset)
            fp.write(b"\0" * tag.count)

        pagebytes = (p_ifd["next_ifd_offset"] - p_ifd["this"]) + sz
        fp.seek(p_ifd["this"])
        fp.write(b"\0" * pagebytes)

        fp.seek(prev_ifd["next_ifd_offset"])
        fp.write(struct.pack(fmt, p_ifd["next_ifd_value"]))


TECH_KEYS = {
    "Aperio Image Library",
    "AppMag",
    "MPP",
    "MPP X",
    "MPP Y",
    "Left",
    "Top",
    "Right",
    "Bottom",
    "Compression",
    "JPEG Quality",
    "ICCProfile",
}


def _sanitize(desc: str) -> str:
    parts = re.split(r"[|;]\s*", desc)
    kept, header = [], False
    for p in parts:
        if p.startswith("Aperio") and not header:
            kept.append(p.strip())
            header = True
            continue
        m = re.match(r"\s*([A-Za-z0-9 _+-]+?)\s*=\s*(.*)", p)
        if not m:
            continue
        key = m.group(1).strip()
        if key in TECH_KEYS:
            kept.append(f"{key}={m.group(2)}")
    return " | ".join(kept)


def strip_metadata(path: Path) -> None:
    with path.open("r+b") as fp:
        t = tiffparser.TiffFile(fp)
        for page in t.pages:
            tag = page.tags.get("ImageDescription")
            if not tag:
                continue
            raw = tag.value
            orig = raw if isinstance(raw, str) else raw.decode(errors="ignore")
            redacted = _sanitize(orig)
            if orig == redacted:
                continue
            data = redacted.encode()  # ASCII
            pad = tag.count - len(data)
            if pad < 0:  # very rare – truncate
                data = data[: tag.count]
                pad = 0
            fp.seek(tag.valueoffset)
            fp.write(data)
            if pad:
                fp.write(b"\0" * pad)


###############################################################################
# Helper Functions (including new one)
###############################################################################
def check_macro_exists(src_file_path: Path, macro_description: str) -> bool:
    """Checks if a macro image with the given description exists in the original TIFF file."""
    try:
        with src_file_path.open("rb") as fp:
            t = tiffparser.TiffFile(fp)
            for page in t.pages:
                desc_tag = page.tags.get("ImageDescription")
                if desc_tag:
                    desc = desc_tag.value
                    if isinstance(desc, bytes):
                        desc = desc.decode(errors="ignore")
                    if macro_description.lower() in desc.lower():
                        return True
    except Exception as e:
        print(
            f"  Warning: Could not reliably check for macro in source {src_file_path}: {e}",
            file=sys.stderr,
        )
        return False
    return False


###############################################################################
# CLI
###############################################################################
def hash_id(slide_id: str, salt: str) -> str:
    return hashlib.sha256((salt + slide_id).encode()).hexdigest()


def process_slide(
    src: Path,
    out_dir: Path,
    salt: str,
    writer,
    macro_description: str,
    rect_coords: tuple[int, int, int, int] | None,
) -> None:
    if src.suffix.lower() not in [".svs", ".tif", ".tiff"]:
        return
    slide_id = src.stem
    hashed_id = hash_id(slide_id, salt)
    dst = out_dir / f"{hashed_id}{src.suffix.lower()}"

    dst.parent.mkdir(parents=True, exist_ok=True)

    macro_exists_in_source = False
    if rect_coords is not None:
        macro_exists_in_source = check_macro_exists(src, macro_description)
        if macro_exists_in_source:
            print(
                f"  Found macro ('{macro_description}') in source file {src}. Will attempt replacement after copy."
            )
        else:
            print(
                f"  Macro ('{macro_description}') not found or check failed in source file {src}. Skipping replacement."
            )

    shutil.copyfile(src, dst)

    delete_associated_image(dst, "label")
    # strip_metadata(dst)

    if rect_coords is not None and macro_exists_in_source:
        print(f"  Attempting macro replacement ({macro_description}) in {dst}")
        try:
            replace_macro(
                str(dst),
                str(dst),
                macro_description=macro_description,
                rect_coords=rect_coords,
            )
            print(f"  Successfully replaced macro in {dst}.")
        except Exception as e:
            print(
                f"  Error during macro replacement in {dst}: {e}",
                file=sys.stderr,
            )
    elif rect_coords is not None and not macro_exists_in_source:
        pass
    else:
        print(
            "  No rectangle coordinates provided or macro not found in source, keeping macro image unchanged."
        )

    writer.writerow(
        {
            "slide_id": slide_id,
            "hashed_id": hashed_id,
            "src_path": src.resolve(),
            "dst_path": dst.resolve(),
        }
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Copy .svs or .tif slides, strip label/macro images, scrub metadata, and rename using salted hash"
    )
    p.add_argument(
        "slides", nargs="+", help="paths or glob patterns to .svs or .tif files"
    )
    p.add_argument("-o", "--out", default="deidentified", help="output directory")
    p.add_argument("--salt", required=True, help="secret salt")
    p.add_argument("-m", "--map", default="hash_mapping.csv", help="mapping CSV")
    p.add_argument(
        "--macro-description",
        default="macro",
        help="String identifier for the macro image (default: 'macro'). Case-insensitive.",
    )
    p.add_argument(
        "--rect",
        metavar="N",
        type=int,
        nargs=4,
        default=None,
        help="Coordinates [x0 y0 x1 y1] for the redaction rectangle (optional). Defaults to a centered rectangle of 1/4 image size.",
    )
    p.add_argument(
        "--boxes-json",
        help="Path to a JSON file containing pre-identified bounding boxes for each file. Generated by identify_boxes.py.",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    boxes_by_path = {}
    if args.boxes_json:
        try:
            with open(args.boxes_json, "r") as f:
                boxes_data = json.load(f)
                for item in boxes_data:
                    file_path = item.get("file_path")
                    rect_coords = item.get("rect_coords")
                    if file_path and rect_coords:
                        boxes_by_path[file_path] = tuple(rect_coords)
                        print(f"Loaded bounding box for {file_path}: {rect_coords}")
                print(
                    f"Loaded bounding boxes for {len(boxes_by_path)} files from {args.boxes_json}"
                )
        except Exception as e:
            print(f"Error loading boxes JSON file: {e}", file=sys.stderr)
            return

    mapping_exists = Path(args.map).is_file()
    with open(args.map, "a", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["slide_id", "hashed_id", "src_path", "dst_path"]
        )
        if not mapping_exists:
            writer.writeheader()

        all_paths = set()
        for pattern in args.slides:
            print(f"Processing pattern: {pattern}")
            expanded_patterns = []
            match = re.match(r"(.*)\{(.*)\}(.*)", pattern)
            if match:
                base, exts_str, suffix = match.groups()
                extensions = exts_str.split(",")
                expanded_patterns = [f"{base}{ext}{suffix}" for ext in extensions]
                print(f"  Expanded to: {expanded_patterns}")
            else:
                expanded_patterns = [pattern]

            for exp_pattern in expanded_patterns:
                pattern_paths = list(Path().glob(exp_pattern))
                if pattern_paths:
                    print(
                        f"  Found {len(pattern_paths)} files for pattern '{exp_pattern}'"
                    )
                else:
                    print(f"  No files found for pattern '{exp_pattern}'")
                for path in pattern_paths:
                    all_paths.add(path)

        print(f"Found {len(all_paths)} unique files to process.")

        if not all_paths:
            print("No files found matching the provided patterns.")
            return

        for path in sorted(list(all_paths)):
            print(f"Processing path: {path}")
            path_str = str(path.resolve())

            rect_coords = None
            if path_str in boxes_by_path:
                rect_coords = boxes_by_path[path_str]
                print(f"Using pre-identified bounding box: {rect_coords}")
            elif args.rect:
                rect_coords = tuple(args.rect)
                print(f"Using command-line rectangle: {rect_coords}")

            try:
                process_slide(
                    path,
                    out_dir,
                    args.salt,
                    writer,
                    args.macro_description,
                    rect_coords,
                )
                print(f"✓ {path} → {out_dir}")
            except Exception as e:
                print(f"✗ {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
