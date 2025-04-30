#!/usr/bin/env python3
"""
De-identifies Whole Slide Image (WSI) files by removing PHI from metadata and macro images.

This script processes WSI files (.svs, .tif, .tiff) to remove potentially identifying information.
It can work with pre-identified bounding boxes from identify_boxes.py, use default/specified
rectangles for masking, or launch an interactive UI for drawing bounding boxes.

Usage Examples
-------------
# Process slides using pre-identified bounding boxes (from JSON)
# (Draws black boxes on macro based on JSON coords)
uv run python deidentify.py "sample/identified/*.svs" --salt "your-secret-salt" --boxes-json identified_boxes.json

# Process slides using interactive annotation
# (Pops up UI for each slide with a macro, saves annotations)
uv run python deidentify.py "sample/identified/*.svs" --salt "your-secret-salt" --interactive-annotate --annotations-out-dir ./macro_annotations

# Process slides using a single fixed rectangle for all macros
# (Draws the same black box on all found macros)
uv run python deidentify.py "path/to/slides/*.svs" --salt "your-secret-salt" -o output_dir --rect 100 150 500 600

# Basic de-identification without modifying macro images at all
# (Skips macro finding and replacement)
uv run python deidentify.py "sample/identified/*.{svs,tif,tiff}" \
    --salt "your-secret-salt-here" \
    -o sample/deidentified \
    -m sample/hash_mapping.csv

# Specify macro description preference order
uv run python deidentify.py "*.svs" --salt "salt" --rect 0 0 1 1 --macro-description-prefs thumbnail macro overview

The script will:
1. Copy input slides to a specified output directory
2. Rename them using a salted hash derived from the original filename
3. Remove the "label" image (which often contains PHI)
4. Strip non-technical metadata fields (Optional, currently commented out)
5. Optionally redact the "macro" image by masking areas containing PHI using one of the specified methods (--interactive-annotate, --boxes-json, --rect).
6. If using interactive mode, save original macros and annotation coordinates.
7. Generate a CSV mapping of original to hashed filenames
"""

import argparse
import csv
import hashlib
import io
import json
import re
import struct
import sys
from pathlib import Path

from PIL import ImageDraw

import tiffparser

# Import the interactive UI runner function
from interactive_annotator import run_annotator

# Import necessary functions from refactored replace_macro.py
from replace_macro import (
    COMMON_MACRO_DESCRIPTIONS,
    find_and_load_macro_openslide,
    replace_macro_with_image,
)


###############################################################################
# helpers
###############################################################################
def delete_associated_image(data: bytearray, image_type: str) -> bool:
    """Modifies the data bytearray in-place to remove the specified associated image.

    Returns True on success, False if the image wasn't found or an error occurred.
    """
    allowed = {"label", "macro"}
    if image_type not in allowed:
        raise ValueError("image_type must be 'label' or 'macro'")

    try:
        # Use BytesIO to treat the bytearray like a file
        with io.BytesIO(data) as fp:
            t = tiffparser.TiffFile(fp)
            p0 = t.pages[0]
            if "Aperio Image Library" in p0.description:
                pages = [p for p in t.pages if image_type in p.description]
            elif "Aperio Leica Biosystems GT450" in p0.description:
                pages = [t.pages[-2]] if image_type == "label" else [t.pages[-1]]
            else:
                pages = [p for p in t.pages if image_type in p.description]
            if len(pages) != 1:
                print(
                    f"  Info: Did not find exactly one '{image_type}' page. Skipping deletion."
                )
                return False  # Indicate image not found/deleted
            page = pages[0]

            fmt, sz = t.tiff.ifdoffsetformat, t.tiff.ifdoffsetsize
            tagnoformat, tagnosize = t.tiff.tagnoformat, t.tiff.tagnosize
            tagsize, unpack = t.tiff.tagsize, struct.unpack

            ifds = []
            # Need to iterate through IFDs by offset, reading from the bytearray
            next_ifd_offset_in_mem = t.tiff.firstifd
            while next_ifd_offset_in_mem:
                fp.seek(next_ifd_offset_in_mem)
                (ntags,) = unpack(tagnoformat, fp.read(tagnosize))
                ifd_start_pos = next_ifd_offset_in_mem
                fp.seek(ntags * tagsize, 1)
                next_ifd_ptr_pos = fp.tell()
                (next_ifd_val,) = unpack(fmt, fp.read(sz))
                ifds.append(
                    {
                        "this": ifd_start_pos,
                        "next_ifd_offset": next_ifd_ptr_pos,
                        "next_ifd_value": next_ifd_val,
                    }
                )
                next_ifd_offset_in_mem = next_ifd_val

            # Find the IFD entry for the page to delete and the one pointing to it
            p_ifd = next((i for i in ifds if i["this"] == page.offset), None)
            prev_ifd = next(
                (i for i in ifds if i["next_ifd_value"] == page.offset), None
            )

            if p_ifd is None:
                print(
                    f"  Warning: Could not find IFD structure for page offset {page.offset}. Skipping deletion."
                )
                return False
            if prev_ifd is None:
                # This case should be rare for label/macro but possible if it's the first associated image.
                # We might need to update the main image IFD pointer if this happens.
                # For now, consider it an error or unsupported case for simplicity.
                print(
                    f"  Warning: Could not find previous IFD pointing to page offset {page.offset}. Deletion logic might be incomplete."
                )
                # To proceed safely, we should zero out the data but not attempt to link.
                # However, returning False for now to indicate incomplete deletion.
                return False

            # Zero out strip/tile data referenced by the page
            strip_offsets_tag = page.tags.get("StripOffsets")
            strip_counts_tag = page.tags.get("StripByteCounts")
            tile_offsets_tag = page.tags.get("TileOffsets")
            tile_counts_tag = page.tags.get("TileByteCounts")

            offsets = []
            counts = []
            if strip_offsets_tag and strip_counts_tag:
                offsets = strip_offsets_tag.value
                counts = strip_counts_tag.value
            elif tile_offsets_tag and tile_counts_tag:
                offsets = tile_offsets_tag.value
                counts = tile_counts_tag.value
            else:
                print(
                    f"  Warning: Could not find Strip/Tile Offset/Count tags for page {page.offset}. Cannot zero image data."
                )

            for off, bc in zip(offsets, counts):
                if off + bc <= len(data):  # Boundary check
                    # Modify the underlying bytearray directly
                    data[off : off + bc] = b"\0" * bc
                else:
                    print(
                        f"  Warning: Invalid offset/count ({off}/{bc}) for image data. Skipping zeroing."
                    )

            # Zero out data referenced by tags *if* stored externally
            # Using the simple original loop, accepting potential inaccuracies for non-byte types,
            # as this wasn't the header corruption cause and full correction is complex.
            for tag in page.tags.values():
                try:
                    # Check if data is inline (valueoffset might be the data itself)
                    type_size_map = {
                        1: 1,
                        2: 1,
                        3: 2,
                        4: 4,
                        5: 8,
                        6: 1,
                        7: 1,
                        8: 2,
                        9: 4,
                        10: 8,
                        11: 4,
                        12: 8,
                        13: 4,
                        16: 8,
                        17: 8,
                        18: 8,
                    }
                    tag_type = getattr(
                        tag, "type", None
                    )  # Use .type based on prior attempt
                    type_size = type_size_map.get(tag_type)
                    if type_size:
                        byte_count_needed = tag.count * type_size
                        is_inline = byte_count_needed <= sz
                        if not is_inline and tag.valueoffset + byte_count_needed <= len(
                            data
                        ):  # boundary check for external data
                            data[
                                tag.valueoffset : tag.valueoffset + byte_count_needed
                            ] = b"\0" * byte_count_needed
                    # If type info is missing or inline, we don't zero here (IFD zeroing handles inline)
                except Exception as e:
                    print(
                        f"  Warning: Error zeroing tag {tag.code} data at {getattr(tag, 'valueoffset', 'N/A')}: {e}",
                        file=sys.stderr,
                    )

            # Zero out the IFD structure itself in the bytearray
            ifd_byte_count = (p_ifd["next_ifd_offset"] - p_ifd["this"]) + sz
            if p_ifd["this"] + ifd_byte_count <= len(data):
                data[p_ifd["this"] : p_ifd["this"] + ifd_byte_count] = (
                    b"\0" * ifd_byte_count
                )
            else:
                print(
                    f"  Warning: Invalid offset/count for IFD zeroing ({p_ifd['this']}/{ifd_byte_count}). Skipping."
                )
                return False  # Cannot safely proceed

            # Update the previous IFD's 'next IFD' pointer in the bytearray
            pointer_bytes = struct.pack(fmt, p_ifd["next_ifd_value"])
            ptr_offset = prev_ifd["next_ifd_offset"]
            if ptr_offset + sz <= len(data):
                data[ptr_offset : ptr_offset + sz] = pointer_bytes
            else:
                print(
                    f"  Warning: Invalid offset for next IFD pointer update ({ptr_offset}). Skipping."
                )
                return False  # Cannot safely proceed

        # If we reached here, modification was attempted successfully
        return True

    except Exception as e:
        print(
            f"  Error during in-memory associated image deletion: {e}", file=sys.stderr
        )
        import traceback

        traceback.print_exc()
        return False  # Indicate failure


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
# Removed check_macro_exists - logic is now in find_and_load_macro_openslide


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
    macro_annotation_mode: str | None,  # 'interactive', 'json', 'rect', or None
    macro_description_prefs: list[str],
    rect_coords_arg: tuple[int, int, int, int] | None,
    boxes_json_data: dict[
        str, list[tuple[int, int, int, int]]
    ],  # Map path_str to list of rects
    annotations_out_dir: Path | None,
    verbose: bool,
) -> None:
    if src.suffix.lower() not in [".svs", ".tif", ".tiff"]:
        return
    slide_id = src.stem
    hashed_id = hash_id(slide_id, salt)
    dst = out_dir / f"{hashed_id}{src.suffix.lower()}"
    src_str = str(src.resolve())  # Use resolved path for lookups

    dst.parent.mkdir(parents=True, exist_ok=True)

    # --- Macro Handling Setup ---
    original_macro_pil = None
    found_macro_desc = None
    modified_macro_pil = None
    rects_to_apply = []  # List of (x0, y0, x1, y1) tuples

    # --- Read Source File into Memory ---
    print(f"  Reading {src} into memory...")
    try:
        file_data = bytearray(src.read_bytes())
        print(f"  Read {len(file_data)} bytes.")
    except Exception as e:
        print(f"  Error reading source file {src}: {e}", file=sys.stderr)
        # Skip processing this file if read fails
        writer.writerow(
            {
                "slide_id": slide_id,
                "hashed_id": hashed_id,
                "src_path": src.resolve(),
                "dst_path": dst.resolve(),
                "status": f"Error reading source: {e}",  # Add status field
            }
        )
        return

    # Only try loading macro if an annotation mode is active
    # Load from the original source path, as in-memory data might change
    if macro_annotation_mode:
        print(f"  Attempting to load macro for {src} (from original file)...", end="")
        original_macro_pil, found_macro_desc = find_and_load_macro_openslide(
            str(src), macro_description_prefs
        )
        if original_macro_pil and found_macro_desc:
            print(f" Found ('{found_macro_desc}')")
            # Keep a copy for modification
            modified_macro_pil = original_macro_pil.copy()
            # Ensure it's RGB for drawing
            if modified_macro_pil.mode != "RGB":
                modified_macro_pil = modified_macro_pil.convert("RGB")
        else:
            print(" Not found or load failed.")
            # If macro not found, we can skip annotation steps for this file
            macro_annotation_mode = None  # Disable further macro processing

    # --- In-Memory De-identification ---
    # Remove the copy step - we operate in memory now
    # print(f"  Copying {src} to {dst}...")
    # shutil.copyfile(src, dst)

    print("  Deleting label image from in-memory data...")
    label_deleted = False
    try:
        # Call the refactored function which modifies file_data in-place
        label_deleted = delete_associated_image(file_data, "label")
        if label_deleted:
            print("  Label deletion successful.")
        else:
            print("  Label deletion skipped (image not found or error occurred).")

    except Exception as e:
        # Catch errors from delete_associated_image itself
        print(
            f"  Warning: Failed during in-memory label deletion: {e}",
            file=sys.stderr,
        )

    # print(f"  Stripping metadata from {dst}...") # Metadata stripping needs similar refactor
    # try:
    #     strip_metadata(dst)
    # except Exception as e:
    #     print(f"  Warning: Failed to strip metadata from {dst}: {e}", file=sys.stderr)

    # --- Macro Annotation/Redaction (operates on PIL image, data modified later) ---
    if macro_annotation_mode == "interactive" and modified_macro_pil:
        if not annotations_out_dir:
            print(
                "  Error: --annotations-out-dir is required for --interactive-annotate",
                file=sys.stderr,
            )
            # Decide how to handle this - skip annotation? Exit?
            # For now, skip annotation for this file.
            macro_annotation_mode = None
        else:
            annotations_out_dir.mkdir(parents=True, exist_ok=True)
            # Save original macro
            original_macro_path = (
                annotations_out_dir / f"{hashed_id}_macro_original.png"
            )
            print(f"  Saving original macro to {original_macro_path}...")
            try:
                original_macro_pil.save(original_macro_path, "PNG")
            except Exception as e:
                print(f"  Warning: Failed to save original macro: {e}", file=sys.stderr)

            # Launch interactive annotator
            print(f"  Launching interactive annotator for {src}...")
            # Call the runner function which handles the QApplication loop
            saved_rects = run_annotator(modified_macro_pil)
            # The run_annotator function blocks until the window is closed,
            # and returns the results (or None if cancelled).

            if saved_rects is not None:  # User saved, not cancelled
                rects_to_apply = saved_rects
                # Save annotations to JSON
                annotation_json_path = (
                    annotations_out_dir / f"{hashed_id}_annotations.json"
                )
                print(
                    f"  Saving {len(rects_to_apply)} annotations to {annotation_json_path}..."
                )
                annotation_data = {
                    "slide_id": slide_id,
                    "hashed_id": hashed_id,
                    "original_path": src_str,
                    "macro_description_found": found_macro_desc,
                    "annotations": rects_to_apply,
                }
                try:
                    with open(annotation_json_path, "w") as f_json:
                        json.dump(annotation_data, f_json, indent=2)
                except Exception as e:
                    print(
                        f"  Warning: Failed to save annotation JSON: {e}",
                        file=sys.stderr,
                    )
            else:
                print("  Annotation cancelled by user.")
                macro_annotation_mode = None  # Don't proceed with replacement

    elif macro_annotation_mode == "json" and modified_macro_pil:
        if src_str in boxes_json_data:
            rects_to_apply = boxes_json_data[src_str]
            print(f"  Using {len(rects_to_apply)} bounding box(es) from JSON for {src}")
        else:
            print(
                f"  No bounding box found in JSON for {src}. Skipping macro replacement."
            )
            macro_annotation_mode = None

    elif macro_annotation_mode == "rect" and modified_macro_pil:
        if rect_coords_arg:
            # Use the single rectangle provided via CLI arg
            rects_to_apply = [rect_coords_arg]
            print(f"  Using command-line rectangle {rect_coords_arg} for {src}")
        else:
            # Should not happen if mode is 'rect', but handle defensively
            print(
                "  Error: Macro mode is 'rect' but no --rect argument provided? Skipping.",
                file=sys.stderr,
            )
            macro_annotation_mode = None

    # --- Apply Rectangles and Replace Macro (if applicable) ---
    if (
        macro_annotation_mode
        and rects_to_apply
        and modified_macro_pil
        and found_macro_desc
    ):
        print(f"  Drawing {len(rects_to_apply)} black rectangle(s) on macro...")
        draw = ImageDraw.Draw(modified_macro_pil)
        for x0, y0, x1, y1 in rects_to_apply:
            # Ensure coords are int and ordered correctly
            ix0, ix1 = sorted((int(x0), int(x1)))
            iy0, iy1 = sorted((int(y0), int(y1)))
            # Draw rectangle (fill is black)
            draw.rectangle([ix0, iy0, ix1, iy1], fill=(0, 0, 0))

        # Call the refactored replace_macro_with_image which works on bytearray
        print("  Replacing macro image in-memory data...")
        try:
            # Pass the current file_data, get the modified version back
            file_data = replace_macro_with_image(
                data=file_data,  # Pass bytearray
                new_macro_pil_image=modified_macro_pil,
                macro_description_to_find=found_macro_desc,
                verbose=verbose,
            )
            print("  Successfully replaced macro in-memory.")
        except Exception as e:
            print(
                f"  Error during in-memory macro replacement: {e}",
                file=sys.stderr,
            )
    elif macro_annotation_mode:
        # Handles cases where mode was set but no rects were generated
        # (e.g., interactive cancelled, JSON key missing, rect arg missing)
        print(
            f"  Skipping macro replacement for {dst} (no valid annotations found/provided)."
        )
    else:
        print("  Macro image will not be modified.")

    # --- Write Final Data ---
    print(f"  Writing final modified data to {dst}...")
    try:
        with open(dst, "wb") as f_out:
            f_out.write(file_data)
        print(f"  Successfully wrote {len(file_data)} bytes.")
    except Exception as e:
        print(f"  Error writing final file {dst}: {e}", file=sys.stderr)
        # Even if write fails, record the attempt in map
        writer.writerow(
            {
                "slide_id": slide_id,
                "hashed_id": hashed_id,
                "src_path": src.resolve(),
                "dst_path": dst.resolve(),
                "status": f"Error writing output: {e}",
            }
        )
        return

    # --- Finalize Mapping ---
    writer.writerow(
        {
            "slide_id": slide_id,
            "hashed_id": hashed_id,
            "src_path": src.resolve(),
            "dst_path": dst.resolve(),
            "status": "Success"
            if label_deleted
            else "Success (label not found/deleted)",  # Add status field
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
        "--macro-description-prefs",
        nargs="+",
        default=COMMON_MACRO_DESCRIPTIONS,
        help=f"List of preferred macro description strings (default: {' '.join(COMMON_MACRO_DESCRIPTIONS)}). Case-insensitive.",
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
    p.add_argument(
        "--interactive-annotate",
        action="store_true",
        help="Launch an interactive UI for drawing bounding boxes on the macro image.",
    )
    p.add_argument(
        "--annotations-out-dir",
        default="./macro_annotations",
        help="Directory to save original macros and annotation JSON files when using --interactive-annotate (default: ./macro_annotations)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    # --- Determine Macro Annotation Mode ---
    macro_mode = None
    if args.interactive_annotate:
        if args.boxes_json or args.rect:
            print(
                "Warning: --interactive-annotate specified; ignoring --boxes-json and --rect.",
                file=sys.stderr,
            )
        macro_mode = "interactive"
        print("Macro annotation mode: Interactive")
    elif args.boxes_json:
        if args.rect:
            print("Warning: --boxes-json specified; ignoring --rect.", file=sys.stderr)
        macro_mode = "json"
        print("Macro annotation mode: JSON Boxes")
    elif args.rect:
        macro_mode = "rect"
        print(f"Macro annotation mode: Fixed Rectangle {args.rect}")
    else:
        print("Macro annotation mode: None (macros will not be modified)")

    # --- Load Boxes JSON if needed ---
    boxes_by_path = {}
    if macro_mode == "json":
        try:
            with open(args.boxes_json, "r") as f:
                boxes_data = json.load(f)
                # Expecting format: [{ "file_path": "...", "rect_coords": [x0,y0,x1,y1] }, ...]
                # Convert to: { "resolved_file_path_str": [(x0,y0,x1,y1), ...], ... }
                # Note: Original identify_boxes.py puts a single rect, adapting here to expect/store a list
                count = 0
                for item in boxes_data:
                    file_path_str = item.get("file_path")
                    rect_coords = item.get("rect_coords")  # This is a single tuple
                    if (
                        file_path_str
                        and isinstance(rect_coords, (list, tuple))
                        and len(rect_coords) == 4
                    ):
                        resolved_path_str = str(Path(file_path_str).resolve())
                        # Store as a list containing the single rectangle
                        boxes_by_path[resolved_path_str] = [tuple(rect_coords)]
                        count += 1
                    else:
                        print(
                            f"Warning: Skipping invalid entry in {args.boxes_json}: {item}",
                            file=sys.stderr,
                        )
                print(f"Loaded bounding boxes for {count} files from {args.boxes_json}")
        except FileNotFoundError:
            print(
                f"Error: Boxes JSON file not found: {args.boxes_json}", file=sys.stderr
            )
            return 1  # Exit if JSON mode selected but file missing
        except Exception as e:
            print(
                f"Error loading or parsing boxes JSON file ({args.boxes_json}): {e}",
                file=sys.stderr,
            )
            return 1  # Exit on other JSON errors

    # --- Prepare Annotation Output Dir if needed ---
    annotations_dir = None
    if macro_mode == "interactive":
        annotations_dir = Path(args.annotations_out_dir)
        # The directory itself is created within process_slide if needed
        print(f"Annotations and original macros will be saved to: {annotations_dir}")

    # --- Process Slides ---
    mapping_exists = Path(args.map).is_file()
    with open(args.map, "a", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "slide_id",
                "hashed_id",
                "src_path",
                "dst_path",
                "status",
            ],  # Add status field
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

            # Determine rectangle source for this specific file
            # This logic is now handled inside process_slide based on macro_mode

            try:
                process_slide(
                    path,
                    out_dir,
                    args.salt,
                    writer,
                    macro_mode,  # Pass the determined mode
                    args.macro_description_prefs,
                    tuple(args.rect) if args.rect else None,
                    boxes_by_path,  # Pass the loaded JSON data
                    annotations_dir,  # Pass the output dir for interactive mode
                    args.verbose,  # Pass verbose flag
                )
                print(f"✓ {path} → {out_dir}")
            except Exception as e:
                print(f"✗ {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
