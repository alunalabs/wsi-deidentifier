#!/usr/bin/env python3
"""
Overwrite the "Macro" associated image in an SVS file
with one that has a solid black rectangle hiding PHI, or with a provided image.

Usage
-----
# Using coordinates (draws a black rectangle)
python replace_macro.py INPUT.SVS [OUTPUT.SVS] [x0 y0 x1 y1] [-v|--verbose]

• OUTPUT.SVS defaults to "INPUT_macro_covered.svs".
• If no coordinates are given, a rectangle the size of ¼ the macro's
  width × height is centred in the image.
• -v / --verbose enables debug-level logging.
"""

import argparse
import io
import logging
import os
import shutil
import struct
import sys
from io import BytesIO
from typing import List, Optional, Tuple

import openslide
from PIL import Image, ImageDraw, ImageFile

# allow Pillow to open incomplete JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Common descriptions for macro images
COMMON_MACRO_DESCRIPTIONS = ["macro", "thumbnail", "overview"]

TAG_IMAGE_WIDTH = 0x0100
TAG_IMAGE_LENGTH = 0x0101
TAG_BITS_PER_SAMPLE = 0x0102
TAG_COMPRESSION = 0x0103
TAG_IMAGE_DESCRIPTION = 0x010E
TAG_PHOTOMETRIC = 0x0106
TAG_SAMPLES_PER_PIXEL = 0x0115
TAG_ROWS_PER_STRIP = 0x0116
TAG_STRIP_OFFSETS = 0x0111
TAG_STRIP_BYTE_COUNTS = 0x0117


class IFDEntry:
    __slots__ = ("tag", "typ", "count", "value", "pos")

    def __init__(self, tag, typ, cnt, val, pos):
        self.tag, self.typ, self.count, self.value, self.pos = (tag, typ, cnt, val, pos)


# Helper function to find an entry by tag
def find_entry(entries: List[IFDEntry], tag: int) -> Optional[IFDEntry]:
    return next((e for e in entries if e.tag == tag), None)


def read_ifd(buf, off, endian):
    n = struct.unpack(endian + "H", buf[off : off + 2])[0]
    entries, cur = [], off + 2
    for _ in range(n):
        tag, typ, cnt, val = struct.unpack(endian + "HHII", buf[cur : cur + 12])
        entries.append(IFDEntry(tag, typ, cnt, val, cur + 8))
        cur += 12
    nxt = struct.unpack(endian + "I", buf[cur : cur + 4])[0]
    return entries, nxt


def ascii_val(entries, tag, data, endian):
    for e in entries:
        if e.tag == tag:
            raw = (
                struct.pack(endian + "I", e.value)[: e.count]
                if e.count <= 4
                else data[e.value : e.value + e.count]
            )
            return raw.rstrip(b"\0").decode("ascii", "ignore")
    return ""


def short_val(entry, data, endian):
    if entry.typ != 3 or entry.count < 1:
        return None
    raw = (
        struct.pack(endian + "I", entry.value)[:2]
        if entry.count * 2 <= 4
        else data[entry.value : entry.value + 2]
    )
    return struct.unpack(endian + "H", raw)[0]


def int_val(entries, tag, data, endian):
    e = find_entry(entries, tag)
    if e:
        return short_val(e, data, endian) if e.typ == 3 else e.value
    return None


def array_val(entry, data, endian):
    size = 2 if entry.typ == 3 else 4
    raw = (
        struct.pack(endian + "I", entry.value)[: entry.count * size]
        if entry.count * size <= 4
        else data[entry.value : entry.value + entry.count * size]
    )
    fmt = endian + ("H" if entry.typ == 3 else "I")
    return [struct.unpack(fmt, raw[i : i + size])[0] for i in range(0, len(raw), size)]


def set_int(buf, entry, v, endian):
    struct.pack_into(endian + "I", buf, entry.pos, v)
    entry.value = v


def set_count(buf, entry, c, endian):
    struct.pack_into(endian + "I", buf, entry.pos - 4, c)
    entry.count = c


def align4(n):
    return (n + 3) & ~3


def find_and_load_macro_openslide(
    path: str, preferred_descriptions: List[str] = COMMON_MACRO_DESCRIPTIONS
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Tries to load the macro image using OpenSlide, checking common descriptions."""
    try:
        with openslide.OpenSlide(path) as slide:
            for desc in preferred_descriptions:
                if desc in slide.associated_images:
                    logging.debug(
                        f"Found macro image with description '{desc}' via OpenSlide."
                    )
                    img = slide.associated_images[desc]
                    return img, desc
            # Fallback: Check if *any* associated image exists if specific ones aren't found
            if slide.associated_images:
                first_key = next(iter(slide.associated_images))
                logging.warning(
                    f"Could not find preferred macro descriptions {preferred_descriptions}. "
                    f"Falling back to the first associated image found: '{first_key}'."
                )
                return slide.associated_images[first_key], first_key
            logging.warning("No associated images found via OpenSlide.")
            return None, None
    except Exception as e:
        logging.error(f"OpenSlide error loading {path}: {e}")
        return None, None


def load_macro_fallback(data, offsets, counts, w, h, cmp, endian):
    if cmp == 1:
        raw = b"".join(data[o : o + c] for o, c in zip(offsets, counts))
        return Image.frombytes("RGB", (w, h), raw)

    if cmp in (6, 7):  # JPEG compression (6 = strip, 7 = tiled)
        strips = []
        for o, c in zip(offsets, counts):
            strips.append(Image.open(BytesIO(data[o : o + c])).convert("RGB"))
        if len(strips) == 1:
            return strips[0]
        img = Image.new("RGB", (w, h))
        y = 0
        for s in strips:
            img.paste(s, (0, y))
            y += s.height
        return img

    return None


def fail(msg, **ctx):
    if ctx:
        logging.error(
            "%s -- context: %s", msg, ", ".join(f"{k}={v}" for k, v in ctx.items())
        )
    sys.exit(msg)


def replace_macro_with_image(
    data: bytearray,
    new_macro_pil_image: Image.Image,
    macro_description_to_find: str,
    verbose: bool = False,
) -> bytearray:
    """
    Replaces the macro image in the provided bytearray with the given PIL Image.

    Args:
        data (bytearray): The SVS/TIFF file content as a mutable byte array.
        new_macro_pil_image (Image.Image): The new macro image (must be RGB).
        macro_description_to_find (str): The description string used to identify the correct IFD.
        verbose (bool): Enable verbose logging.

    Returns:
        bytearray: The modified byte array with the macro replaced.

    Raises:
        ValueError: If the macro IFD cannot be found or tags cannot be updated.
        Exception: For other file processing errors.
    """
    # logging setup is handled by the calling script (deidentify.py)
    logging.info(
        "Replacing macro in-memory (targeting description '%s')",
        macro_description_to_find,
    )

    # Operate on the bytearray directly using BytesIO for parsing convenience
    try:
        with io.BytesIO(data) as fp:  # Use BytesIO for reading IFDs etc.
            # Read header info directly from bytearray
            if len(data) < 8:
                raise ValueError("Data too short for TIFF header")

            endian_char = data[:2].decode("ascii", "ignore")
            endian = {"II": "<", "MM": ">"}.get(endian_char)
            magic_number = None
            if endian:
                try:
                    magic_number = struct.unpack(endian + "H", data[2:4])[0]
                except struct.error:
                    pass  # magic_number remains None

            # Allow both Classic TIFF (42) and BigTIFF (43) magic numbers
            if not endian or magic_number not in (42, 43):
                raise ValueError(
                    f"Not a TIFF/SVS format – bad byte order or invalid magic number (read {magic_number}, expected 42 or 43)"
                )

            # TODO: Handle BigTIFF specific structures (e.g., 64-bit offsets) if necessary later.
            # For now, assume the IFDs we need to parse are compatible enough or
            # that openslide handled the initial load transparently.
            if magic_number == 43:
                logging.warning(
                    "Detected BigTIFF format (magic number 43). Proceeding with Classic TIFF assumptions for IFD parsing."
                )

            # Assuming Classic TIFF for initial IFD offset reading (first 4 bytes after header)
            # BigTIFF might have a different header structure if version != 42
            first_ifd_offset = struct.unpack(endian + "I", data[4:8])[0]

            # Find the macro IFD by iterating through offsets
            macro_ifd_entries = None
            found_macro_ifd = False
            current_ifd_offset = first_ifd_offset
            while current_ifd_offset:
                # Need to read IFD structure directly from bytearray
                fp.seek(current_ifd_offset)
                num_entries = struct.unpack(endian + "H", fp.read(2))[0]
                entries_bytes = fp.read(num_entries * 12)
                next_ifd_offset_bytes = fp.read(4)
                if len(next_ifd_offset_bytes) < 4:
                    logging.warning(
                        f"Could not read next IFD offset at {fp.tell() - 4}. Stopping IFD scan."
                    )
                    current_ifd_offset = 0  # Stop loop
                else:
                    current_ifd_offset = struct.unpack(
                        endian + "I", next_ifd_offset_bytes
                    )[0]

                # Parse entries from the bytes we read
                entries = []
                for i in range(num_entries):
                    entry_bytes = entries_bytes[i * 12 : (i + 1) * 12]
                    tag, typ, cnt, val = struct.unpack(endian + "HHII", entry_bytes)
                    # Position here is relative to IFD start + 2 bytes for num_entries
                    value_pos_in_ifd = current_ifd_offset + 2 + i * 12 + 8
                    entries.append(IFDEntry(tag, typ, cnt, val, value_pos_in_ifd))

                # Check description for this IFD
                desc = ascii_val(entries, TAG_IMAGE_DESCRIPTION, data, endian).lower()
                if macro_description_to_find.lower() in desc:
                    macro_ifd_entries = entries
                    found_macro_ifd = True
                    logging.debug(
                        f"Found IFD for macro description '{macro_description_to_find}' at offset {current_ifd_offset}"
                    )
                    break  # Found it!

            if not found_macro_ifd:
                raise ValueError(
                    f"Could not find the specific macro IFD with description '{macro_description_to_find}' during replacement phase."
                )

    except Exception as e:
        logging.error(f"Error parsing in-memory TIFF structure: {e}")
        raise  # Re-raise after logging

    # --- Modify the bytearray based on found IFD entries ---

    # Prepare the new image data (uncompressed RGB)
    if new_macro_pil_image.mode != "RGB":
        new_macro_pil_image = new_macro_pil_image.convert("RGB")

    img_bytes = new_macro_pil_image.tobytes()
    original_length = len(data)
    new_data_offset = align4(original_length)  # Append after original data
    padding_size = new_data_offset - original_length
    new_byte_count = len(img_bytes)
    w, h = new_macro_pil_image.size

    # Modify existing tags in the macro IFD within the main 'data' bytearray
    updated_tags = False
    for e in macro_ifd_entries:
        # Important: Use struct.pack_into to modify the bytearray IN PLACE
        if e.tag == TAG_COMPRESSION:
            struct.pack_into(endian + "I", data, e.pos, 1)  # 1 = Uncompressed
            e.value = 1  # Update IFDEntry object state for consistency if needed
            updated_tags = True
        elif e.tag == TAG_PHOTOMETRIC:
            struct.pack_into(endian + "I", data, e.pos, 2)  # 2 = RGB
            e.value = 2
            updated_tags = True
        elif e.tag == TAG_STRIP_OFFSETS:
            if e.count != 1:
                struct.pack_into(endian + "I", data, e.pos - 4, 1)  # Update count field
                e.count = 1
            struct.pack_into(
                endian + "I", data, e.pos, new_data_offset
            )  # Update value (offset)
            e.value = new_data_offset
            updated_tags = True
        elif e.tag == TAG_STRIP_BYTE_COUNTS:
            if e.count != 1:
                struct.pack_into(endian + "I", data, e.pos - 4, 1)  # Update count field
                e.count = 1
            struct.pack_into(
                endian + "I", data, e.pos, new_byte_count
            )  # Update value (byte count)
            e.value = new_byte_count
            updated_tags = True
        elif e.tag == TAG_ROWS_PER_STRIP:
            if e.count != 1:
                struct.pack_into(endian + "I", data, e.pos - 4, 1)
                e.count = 1
            struct.pack_into(endian + "I", data, e.pos, h)  # Update value (height)
            e.value = h
            updated_tags = True
        elif e.tag == TAG_IMAGE_WIDTH:
            # Read current value before potentially updating
            current_w = int_val([e], TAG_IMAGE_WIDTH, data, endian)
            if current_w != w:
                logging.warning(
                    f"Original width {current_w} != new width {w}. Updating."
                )
                if e.typ == 3:  # SHORT
                    struct.pack_into(endian + "H", data, e.pos, w)
                else:  # Assume LONG
                    struct.pack_into(endian + "I", data, e.pos, w)
                e.value = w
        elif e.tag == TAG_IMAGE_LENGTH:
            current_h = int_val([e], TAG_IMAGE_LENGTH, data, endian)
            if current_h != h:
                logging.warning(
                    f"Original height {current_h} != new height {h}. Updating."
                )
                if e.typ == 3:  # SHORT
                    struct.pack_into(endian + "H", data, e.pos, h)
                else:  # Assume LONG
                    struct.pack_into(endian + "I", data, e.pos, h)
                e.value = h

    if not updated_tags:
        raise ValueError(
            "Failed to find and update necessary tags (Compression, Photometric, Strips) in the macro IFD."
        )

    # Append padding and new image data to the bytearray
    padding = b"\0" * padding_size
    final_data = data + padding + img_bytes

    logging.info(
        f"In-memory macro replacement complete. New data length: {len(final_data)}"
    )
    return final_data


def replace_macro(
    input_path: str,
    output_path: str,
    rect_coords: Optional[Tuple[int, int, int, int]] = None,
    verbose: bool = False,
    macro_description_prefs: List[str] = COMMON_MACRO_DESCRIPTIONS,
):
    """
    Replaces the macro image in an SVS file by drawing a black rectangle.

    Args:
        input_path (str): Path to the input SVS file.
        output_path (str): Path to save the modified SVS file.
        rect_coords (tuple, optional): Coordinates (x0, y0, x1, y1) for the redaction rectangle.
                                     Defaults to a centered rectangle if None.
        verbose (bool): Enable verbose logging.
        macro_description_prefs (List[str]): Preferred description strings for the macro.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.info("Attempting to load macro from %s", input_path)

    # Try loading with OpenSlide first using preferred descriptions
    img, found_desc = find_and_load_macro_openslide(input_path, macro_description_prefs)

    if img is None:
        # TODO: Add fallback using TIFF parsing if OpenSlide fails?
        # For now, we rely on OpenSlide as it handles JPEG decompression etc.
        logging.warning(
            f"Could not load macro image using OpenSlide for descriptions {macro_description_prefs}. Skipping replacement."
        )
        # Optional: copy input to output if names differ, to signify no change?
        if input_path != output_path:
            try:
                shutil.copyfile(input_path, output_path)
                logging.info(
                    f"Copied input {input_path} to {output_path} as macro was not found/loaded."
                )
            except Exception as e:
                fail(f"Failed to copy input file {input_path} to {output_path}: {e}")
        return  # Exit gracefully if no macro found

    logging.info(f"Successfully loaded macro image ('{found_desc}')")
    img = img.convert("RGB")
    w, h = img.size

    if rect_coords:
        x0, y0, x1, y1 = rect_coords
    else:
        # Default: Center a rectangle 1/4 width and height
        dw, dh = w // 4, h // 4
        x0, y0 = (w - dw) // 2, (h - dh) // 2
        x1, y1 = x0 + dw, y0 + dh
        logging.debug("Defaulting to centered rectangle: %s", (x0, y0, x1, y1))

    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        fail(
            "Rectangle outside image bounds",
            rect=f"{x0},{y0},{x1},{y1}",
            image=f"{w}x{h}",
        )

    # Draw black rectangle
    draw = ImageDraw.Draw(img)
    # Use black (0, 0, 0) instead of red
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(0, 0, 0))
    logging.debug("Black rectangle drawn at %s", (x0, y0, x1, y1))

    # Call the function that handles TIFF writing with the modified PIL image
    replace_macro_with_image(
        output_path=output_path,
        new_macro_pil_image=img,
        macro_description_to_find=found_desc,
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overwrite the macro image in an SVS file with a redacted version.",
        usage="%(prog)s INPUT.SVS [OUTPUT.SVS] [x0 y0 x1 y1] [-v|--verbose]",
    )
    parser.add_argument(
        "input_svs", metavar="INPUT.SVS", help="Path to the input SVS file."
    )
    parser.add_argument(
        "output_svs",
        metavar="OUTPUT.SVS",
        nargs="?",
        default=None,
        help="Path for the output SVS file (optional). If omitted, overwrites input file.",
    )
    parser.add_argument(
        "rect",
        metavar="N",
        type=int,
        nargs="*",
        help="Coordinates [x0 y0 x1 y1] for the redaction rectangle (optional). Defaults to a centered rectangle of 1/4 image size.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug-level logging."
    )
    parser.add_argument(
        "--macro-description-prefs",
        nargs="+",
        default=COMMON_MACRO_DESCRIPTIONS,
        help=f"List of preferred macro description strings (default: {' '.join(COMMON_MACRO_DESCRIPTIONS)}). Case-insensitive.",
    )

    parsed_args = parser.parse_args()

    inp_path = parsed_args.input_svs
    # Default output to input if not specified
    out_path = (
        parsed_args.output_svs if parsed_args.output_svs is not None else inp_path
    )

    # Default rect_coords to None if not provided, triggering default centered box logic
    rect_coords_arg = tuple(parsed_args.rect) if parsed_args.rect else None
    if rect_coords_arg and len(rect_coords_arg) != 4:
        parser.error("Rectangle requires 4 integer coordinates: x0 y0 x1 y1.")

    verbosity = parsed_args.verbose
    macro_prefs = parsed_args.macro_description_prefs

    # Check if input exists before proceeding
    if not os.path.exists(inp_path):
        fail(f"Input file not found: {inp_path}")

    # --- This part needs modification for standalone use ---
    # 1. Read input_path into bytearray
    # 2. Create PIL image (either black rect or from file)
    # 3. Call replace_macro_with_image(bytearray, pil_image, ...)
    # 4. Write returned bytearray to output_path
    print(
        "Standalone execution of replace_macro.py needs to be adapted for the new in-memory function."
    )
    # --- End modification needed ---

    # Replace the call to the old function
    # replace_macro(inp_path, out_path, rect_coords_arg, verbosity, macro_prefs)
