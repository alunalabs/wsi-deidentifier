#!/usr/bin/env python3
"""
De-identifies Whole Slide Image (WSI) files by removing PHI from metadata and macro images.

This script processes WSI files (.svs, .tif, .tiff) to remove potentially identifying information.
It can work with pre-identified bounding boxes from identify_boxes.py or use default/specified
rectangles for masking. Optionally, filenames can be hashed using a salt for additional anonymization.

Usage Examples
-------------
# Process slides using pre-identified bounding boxes with salt (hashed filenames)
uv run python deidentify.py "sample/identified/*.svs" --salt "your-secret-salt" --boxes-json identified_boxes.json

# Basic de-identification with salt (hashed filenames)
uv run python deidentify.py "sample/identified/*.{svs,tif,tiff}" \\
    --salt "your-secret-salt-here" \\
    -o sample/deidentified \\
    -m sample/hash_mapping.csv

# Process slides without salt (preserves original filenames)
uv run python deidentify.py "sample/identified/*.{svs,tif,tiff}" \\
    -o sample/deidentified \\
    -m sample/hash_mapping.csv

# Default centered rectangle for macro masking with salt
uv run python deidentify.py "sample/identified/*.{svs,tif,tiff}" \\
    --salt "your-secret-salt-here" \\
    -o sample/deidentified \\
    --macro-description "macro" \\
    --rect 0 0 0 0

# Specify custom rectangle coordinates without salt (preserves filenames)
uv run python deidentify.py "path/to/slides/*.svs" \\
    -o output_dir \\
    --rect 100 150 500 600

# Strip all associated images (label and macro) completely without salt
uv run python deidentify.py "sample/identified/*.{svs,tif,tiff}" \
    -o sample/deidentified \
    --strip-all-images

The script will:
1. Copy input slides to a specified output directory
2. Optionally rename them using a salted hash derived from the original filename (if --salt is provided)
3. Remove the "label" image (which often contains PHI)
4. Strip non-technical metadata fields 
5. Optionally redact the "macro" image by masking areas containing PHI (only if --rect or --boxes-json is specified)
6. Generate a CSV mapping of original to processed filenames (hashed if salt provided, original if not)
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

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import ImageColor

import tiffparser
from replace_macro import load_macro_openslide, replace_macro


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
def parse_color(color_string: str) -> tuple[int, int, int]:
    """Parses a color string (name or hex) into an RGB tuple."""
    if color_string.startswith("#"):
        return ImageColor.getrgb(color_string)
    # For common color names, ImageColor.getrgb can also handle them
    try:
        return ImageColor.getrgb(color_string)
    except ValueError:
        # Fallback for a few basic names if ImageColor doesn't get them directly without #
        # This is more of a safeguard; ImageColor is usually robust.
        color_map = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
        }
        if color_string.lower() in color_map:
            return color_map[color_string.lower()]
        raise ValueError(
            f"Invalid color string: {color_string}. Use common names or hex codes (e.g., '#FF0000')."
        )


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
def hash_id(slide_id: str, salt: str | None) -> str:
    if salt is None:
        return slide_id
    return hashlib.sha256((salt + slide_id).encode()).hexdigest()


def process_slide(
    src: Path,
    out_dir: Path,
    salt: str | None,
    writer,
    macro_description: str,
    rect_coords: tuple[int, int, int, int] | None,
    strip_all_images: bool,
    macro_text: str | None,
    macro_font_size: int | None,
    fill_color_rgb: tuple[int, int, int],
    font_color_rgb: tuple[int, int, int],
) -> None:
    if src.suffix.lower() not in [".svs", ".tif", ".tiff"]:
        return
    slide_id = src.stem
    # Use original filename if salt is not provided
    output_filename_base = hash_id(slide_id, salt)
    dst = out_dir / f"{output_filename_base}{src.suffix.lower()}"

    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(src, dst)

    if strip_all_images:
        print(f"  Stripping all associated images from {dst}")
        try:
            delete_associated_image(dst, "label")
            print("    Removed label image.")
        except Exception as e:
            print(f"    Warning: Could not remove label image: {e}", file=sys.stderr)
        try:
            delete_associated_image(dst, "macro")
            print("    Removed macro image.")
        except Exception as e:
            print(f"    Warning: Could not remove macro image: {e}", file=sys.stderr)
        # Optionally strip metadata as well if desired in this mode
        # strip_metadata(dst)
    else:
        # Original logic for label deletion and macro handling
        try:
            delete_associated_image(dst, "label")
            print(f"  Removed label image from {dst}")
        except Exception as e:
            print(
                f"  Warning: Could not remove label image from {dst}: {e}",
                file=sys.stderr,
            )

        macro_exists_in_source = False
        if rect_coords is not None:
            # Check in the *copied* file now, as the source might not be accessible
            # or it's safer to check the file we are modifying.
            # Note: This check might be less reliable after copying or label removal.
            # Consider simplifying if macro existence check isn't strictly needed before replacement.
            try:
                with dst.open("rb") as fp_check:
                    t_check = tiffparser.TiffFile(fp_check)
                    for page in t_check.pages:
                        desc_tag = page.tags.get("ImageDescription")
                        if desc_tag:
                            desc = desc_tag.value
                            if isinstance(desc, bytes):
                                desc = desc.decode(errors="ignore")
                            if macro_description.lower() in desc.lower():
                                macro_exists_in_source = True
                                break
            except Exception as e:
                print(
                    f"  Warning: Could not check for macro in destination {dst}: {e}",
                    file=sys.stderr,
                )

            if macro_exists_in_source:
                print(
                    f"  Found macro ('{macro_description}') in destination file {dst}. Will attempt replacement."
                )
            else:
                print(
                    f"  Macro ('{macro_description}') not found or check failed in destination file {dst}. Skipping replacement."
                )

        if rect_coords is not None and not _rect_is_valid(rect_coords):
            print(
                f"  Warning: Invalid rectangle coordinates provided {rect_coords}, skipping macro replacement.",
                file=sys.stderr,
            )
            rect_coords = None  # bad data, treat as if no rect was given

        if rect_coords is not None and macro_exists_in_source:
            print(f"  Attempting macro replacement ({macro_description}) in {dst}")
            tmp_out = dst.with_suffix(dst.suffix + ".tmp")
            try:
                replace_macro(
                    str(dst),
                    str(tmp_out),
                    macro_description=macro_description,
                    rect_coords=rect_coords,
                    fill_color=fill_color_rgb,
                    text=macro_text,
                    font_size=macro_font_size if macro_font_size is not None else 14,
                    font_color=font_color_rgb,
                )
                # Atomic replace original file
                tmp_out.replace(dst)
                print(f"  Successfully replaced macro in {dst}.")
            except Exception as e:
                print(
                    f"  Error during macro replacement in {dst}: {e}",
                    file=sys.stderr,
                )
            finally:
                # Ensure the temporary file is removed if it still exists
                if tmp_out.exists():
                    try:
                        tmp_out.unlink()
                        print(f"  Cleaned up temporary file {tmp_out}")
                    except OSError as unlink_err:
                        print(
                            f"  Warning: Could not remove temporary file {tmp_out}: {unlink_err}",
                            file=sys.stderr,
                        )
        elif rect_coords is not None and not macro_exists_in_source:
            # Rect provided but macro wasn't found or check failed
            pass  # Already printed a message
        else:
            # No rectangle coords provided or rect was invalid
            print(
                "  No rectangle coordinates provided or macro not found, keeping macro image unchanged."
            )

        # Strip metadata after potential macro replacement
        # strip_metadata(dst)

    # Write mapping regardless of image stripping mode
    writer.writerow(
        {
            "slide_id": slide_id,
            "hashed_id": output_filename_base,
            "src_path": src.resolve(),
            "dst_path": dst.resolve(),
        }
    )


###############################################################################
# Interactive labeling helper
###############################################################################


def interactive_label_slides(
    slide_paths: list[Path],
    boxes_by_path: dict,
    macro_description: str,
    boxes_json_path: Path,
):
    """Launch an interactive UI (matplotlib RectangleSelector) to label bounding boxes.

    Already-labeled slides contained in *boxes_by_path* are skipped.  After each
    successful annotation the JSON file at *boxes_json_path* is updated so that
    progress is saved incrementally (important if the user aborts early).
    """

    slide_paths_sorted = sorted(slide_paths)

    for slide in slide_paths_sorted:
        slide_abs = str(slide.resolve())
        if slide_abs in boxes_by_path:
            # already labelled – skip
            continue

        print(f"\nInteractive annotation for: {slide}")
        print(
            "  Instructions:  Draw a rectangle covering PHI on the macro image, then\n"
            "                press <Enter> to accept or close the window to skip."
        )

        # ------------------------------------------------------------------
        # Load the macro thumbnail
        # ------------------------------------------------------------------
        try:
            macro_img = load_macro_openslide(str(slide))
        except Exception as exc:
            print(f"  [warning] Could not load macro for {slide}: {exc}")
            continue

        if macro_img is None:
            print("  [warning] No macro image embedded – skipping.")
            continue

        # Convert to RGB in case it is not
        macro_img = macro_img.convert("RGB")

        # ------------------------------------------------------------------
        # Start an interactive Matplotlib session
        # ------------------------------------------------------------------
        coords = []  # will hold [x0, y0, x1, y1]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(macro_img)
        ax.set_title(slide.name, fontsize=10)
        ax.axis("off")

        def onselect(eclick, erelease):
            nonlocal coords
            x0, y0 = int(eclick.xdata), int(eclick.ydata)
            x1, y1 = int(erelease.xdata), int(erelease.ydata)
            # normalise so x0<x1, y0<y1
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            coords = [x0, y0, x1, y1]

        toggle_selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],  # left mouse button only
            interactive=True,
        )

        def accept(event):
            if event.key in ("enter", "return"):
                plt.close(event.canvas.figure)

        fig.canvas.mpl_connect("key_press_event", accept)

        plt.show()

        # ------------------------------------------------------------------
        # Store annotation
        # ------------------------------------------------------------------
        if coords and coords[0] < coords[2] and coords[1] < coords[3]:
            print(f"  Saved rectangle: {coords}")
            boxes_by_path[slide_abs] = tuple(coords)
        elif coords:
            print(
                "  Ignored zero-size rectangle (click rather than drag). Please drag to create a box."
            )
        # Write out to JSON after every annotation to be safe
        try:
            # store as list of dicts (legacy format)
            records = [
                {"file_path": p, "rect_coords": list(rc)}
                for p, rc in boxes_by_path.items()
            ]
            with open(boxes_json_path, "w") as jf:
                json.dump(records, jf, indent=2)
            print(f"  JSON updated → {boxes_json_path}")
        except Exception as exc:
            print(f"  [warning] Could not update JSON file: {exc}")

    return boxes_by_path


def _rect_is_valid(rect: tuple[int, int, int, int] | list[int]) -> bool:
    """Return True if rect coords are well-formed (x0<x1 and y0<y1)."""
    if len(rect) != 4:
        return False
    x0, y0, x1, y1 = rect
    return x1 > x0 and y1 > y0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Copy .svs or .tif slides, strip label/macro images, scrub metadata, and optionally rename using salted hash"
    )
    p.add_argument(
        "slides", nargs="+", help="paths or glob patterns to .svs or .tif files"
    )
    p.add_argument("-o", "--out", default="deidentified", help="output directory")
    p.add_argument(
        "--salt",
        help="secret salt for filename hashing (optional). If not provided, original filenames are preserved.",
    )
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
        default="identified_boxes.json",
        help=(
            "Path to a JSON file containing bounding boxes (default: identified_boxes.json). If the file exists, its contents are loaded. "
            "If interactive labeling is enabled (see --interactive-label) the same path will be updated with new annotations "
            "after every slide is labeled."
        ),
    )
    p.add_argument(
        "--interactive-label",
        action="store_true",
        help=(
            "Launch an interactive bounding-box labeling UI for slides that do not yet have annotations. "
            "Progress is saved incrementally to the JSON file specified by --boxes-json (or 'identified_boxes.json' by default).",
        ),
    )
    p.add_argument(
        "--strip-all-images",
        action="store_true",
        help="Completely remove all associated images (label, macro) instead of masking. Overrides --rect, --boxes-json, and --interactive-label.",
    )
    p.add_argument(
        "--macro-text",
        help="Text to add at center of macro image (optional).",
        default=None,
    )
    p.add_argument(
        "--macro-font-size",
        type=int,
        help="Font size for the text on the macro image (optional, default: 14).",
        default=14,
    )
    p.add_argument(
        "--fill-color",
        type=str,
        default="black",
        help="Fill color for the redaction rectangle on the macro image (e.g., 'black', '#000000'). Default: black.",
    )
    p.add_argument(
        "--font-color",
        type=str,
        default="white",
        help="Font color for the text on the macro image (e.g., 'white', '#FFFFFF'). Default: white.",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    boxes_by_path: dict[str, tuple[int, int, int, int]] = {}
    if args.boxes_json:
        try:
            if Path(args.boxes_json).is_file():
                with open(args.boxes_json, "r") as f:
                    boxes_data = json.load(f)
                    for item in boxes_data:
                        file_path = item.get("file_path")
                        rect_coords = item.get("rect_coords")
                        if file_path and rect_coords and _rect_is_valid(rect_coords):
                            boxes_by_path[file_path] = tuple(rect_coords)
                            print(f"Loaded bounding box for {file_path}: {rect_coords}")
                        elif file_path and rect_coords:
                            print(
                                f"Ignored invalid rectangle in JSON for {file_path}: {rect_coords}",
                                file=sys.stderr,
                            )
                    print(
                        f"Loaded bounding boxes for {len(boxes_by_path)} files from {args.boxes_json}"
                    )
            else:
                print(
                    f"No existing bounding-box JSON found at {args.boxes_json}. A new file will be created upon saving.",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"Error loading boxes JSON file: {e}", file=sys.stderr)

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
                # Handle absolute vs relative patterns
                if Path(exp_pattern).is_absolute():
                    # For absolute patterns, use the parent directory and relative pattern
                    pattern_path = Path(exp_pattern)
                    if pattern_path.is_dir():
                        # If it's a directory, glob all supported files in it
                        pattern_paths = []
                        for ext in [".svs", ".tif", ".tiff"]:
                            pattern_paths.extend(pattern_path.glob(f"*{ext}"))
                    else:
                        # If it's a file pattern, use the parent and name
                        parent_dir = pattern_path.parent
                        pattern_name = pattern_path.name
                        pattern_paths = list(parent_dir.glob(pattern_name))
                else:
                    # For relative patterns, use current working directory
                    pattern_paths = list(Path().glob(exp_pattern))

                # Filter out macOS resource fork files (._filename)
                pattern_paths = [
                    p for p in pattern_paths if not p.name.startswith("._")
                ]

                if pattern_paths:
                    print(
                        f"  Found {len(pattern_paths)} files for pattern '{exp_pattern}'"
                    )
                else:
                    print(f"  No files found for pattern '{exp_pattern}'")
                for path in pattern_paths:
                    all_paths.add(path)

        print(f"Found {len(all_paths)} unique files to process.")

        # ------------------------------------------------------------------
        # Interactive labeling (optional) ----------------------------------
        # ------------------------------------------------------------------
        if args.interactive_label:
            if not all_paths:
                print("No slides found for interactive labeling.")
            else:
                print("\n=== Interactive labeling mode enabled ===")
                boxes_by_path = interactive_label_slides(
                    slide_paths=list(all_paths),
                    boxes_by_path=boxes_by_path,
                    macro_description=args.macro_description,
                    boxes_json_path=Path(args.boxes_json),
                )
                print("=== Labeling complete ===\n")

        if not all_paths:
            print("No files found matching the provided patterns.")
            return

        try:
            parsed_fill_color = parse_color(args.fill_color)
        except ValueError as e:
            print(f"Error parsing --fill-color: {e}", file=sys.stderr)
            return
        try:
            parsed_font_color = parse_color(args.font_color)
        except ValueError as e:
            print(f"Error parsing --font-color: {e}", file=sys.stderr)
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
                    args.strip_all_images,
                    args.macro_text,
                    args.macro_font_size,
                    parsed_fill_color,
                    parsed_font_color,
                )
                print(f"✓ {path} → {out_dir}")
            except Exception as e:
                print(f"✗ {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
