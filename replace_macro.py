#!/usr/bin/env python3
"""
Overwrite the "Macro" associated image in an SVS file
with one that has a solid red rectangle hiding PHI.

Usage
-----
python replace_macro.py INPUT.SVS [OUTPUT.SVS] [x0 y0 x1 y1] [-v|--verbose]

• OUTPUT.SVS defaults to "INPUT_macro_covered.svs".
• If no coordinates are given, a rectangle the size of ¼ the macro's
  width × height is centred in the image.
• -v / --verbose enables debug-level logging.
"""

import argparse
import logging
import os
import struct
import sys
from io import BytesIO

import openslide
from PIL import Image, ImageDraw, ImageFile

# allow Pillow to open incomplete JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    for e in entries:
        if e.tag == tag:
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


def load_macro_openslide(path):
    try:
        with openslide.OpenSlide(path) as slide:
            return slide.associated_images.get("macro", None)
    except Exception:  # structural corruption or vendor-specific corner cases
        return None


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


def replace_macro(
    input_path, output_path, rect_coords=None, verbose=False, macro_description="macro"
):
    """
    Replaces the macro image in an SVS file.

    Args:
        input_path (str): Path to the input SVS file.
        output_path (str): Path to save the modified SVS file.
        rect_coords (tuple, optional): Coordinates (x0, y0, x1, y1) for the redaction rectangle.
                                     Defaults to a centered rectangle if None.
        verbose (bool): Enable verbose logging.
        macro_description (str): The string expected in the ImageDescription tag for the macro.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.info("Opening %s", input_path)
    try:
        with open(input_path, "rb") as f:
            data = bytearray(f.read())
    except FileNotFoundError:
        fail(f"Input file not found: {input_path}")
    except Exception as e:
        fail(f"Error reading input file {input_path}: {e}")

    endian = {"II": "<", "MM": ">"}.get(data[:2].decode("ascii", "ignore"))
    if not endian or struct.unpack(endian + "H", data[2:4])[0] != 42:
        fail("Not a TIFF/SVS – bad byte order or missing magic number")

    ifd_off, macro = struct.unpack(endian + "I", data[4:8])[0], None
    found_macro_ifd = False
    while ifd_off:
        ent, nxt = read_ifd(data, ifd_off, endian)
        desc = ascii_val(ent, TAG_IMAGE_DESCRIPTION, data, endian).lower()
        if macro_description.lower() in desc:
            macro = ent
            found_macro_ifd = True
            break
        ifd_off = nxt
    if not found_macro_ifd:
        logging.info(
            "Macro image not found (description '%s') - skipping replacement.",
            macro_description,
        )
        return

    w = int_val(macro, TAG_IMAGE_WIDTH, data, endian)
    h = int_val(macro, TAG_IMAGE_LENGTH, data, endian)
    spp = int_val(macro, TAG_SAMPLES_PER_PIXEL, data, endian) or 3
    bps = int_val(macro, TAG_BITS_PER_SAMPLE, data, endian) or 8
    cmp = int_val(macro, TAG_COMPRESSION, data, endian) or 1
    logging.debug("Macro: %dx%d, spp=%d, bps=%d, compression=%d", w, h, spp, bps, cmp)

    if not (spp == 3 and bps == 8):
        fail("Unsupported macro format – need 24-bit RGB", spp=spp, bits_per_sample=bps)

    img = load_macro_openslide(input_path)
    if img is None:
        fail("Unable to decode macro", compression=cmp)

    img = img.convert("RGB")

    if rect_coords:
        x0, y0, x1, y1 = rect_coords
    else:
        dw, dh = w // 4, h // 4
        x0, y0 = (w - dw) // 2, (h - dh) // 2
        x1, y1 = x0 + dw, y0 + dh
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        fail(
            "Rectangle outside image bounds",
            rect=f"{x0},{y0},{x1},{y1}",
            image=f"{w}x{h}",
        )

    ImageDraw.Draw(img).rectangle([x0, y0, x1 - 1, y1 - 1], fill=(255, 0, 0))
    logging.debug("Red rectangle drawn at %s", (x0, y0, x1, y1))

    buf = bytearray(img.tobytes())
    new_off = align4(len(data))
    data.extend(b"\0" * (new_off - len(data)))
    data.extend(buf)
    bc = len(buf)

    def entry(tag):  # closure for quick lookup
        return next((e for e in macro if e.tag == tag), None)

    for tag, val in ((TAG_COMPRESSION, 1), (TAG_PHOTOMETRIC, 2)):
        if e := entry(tag):
            set_int(data, e, val, endian)

    for tag, val in (
        (TAG_STRIP_OFFSETS, new_off),
        (TAG_STRIP_BYTE_COUNTS, bc),
        (TAG_ROWS_PER_STRIP, h),
    ):
        e = entry(tag)
        if not e:
            fail("Required strip tag missing", tag=hex(tag))
        if e.count != 1:
            set_count(data, e, 1, endian)
        set_int(data, e, val, endian)

    try:
        with open(output_path, "wb") as fh:
            fh.write(data)
        logging.info("Saved %s", output_path)
    except Exception as e:
        fail(f"Error writing output file {output_path}: {e}")


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
        help="Path for the output SVS file (optional). Defaults to 'INPUT_macro_covered.svs'.",
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
        "--macro-description",
        default="macro",
        help="String identifier for the macro image in ImageDescription tag (default: 'macro'). Case-insensitive.",
    )

    parsed_args = parser.parse_args()

    inp_path = parsed_args.input_svs
    out_path = parsed_args.output_svs
    rect_coords = tuple(parsed_args.rect) if parsed_args.rect else None
    verbosity = parsed_args.verbose
    macro_desc = parsed_args.macro_description

    if out_path is None:
        input_dir = os.path.dirname(inp_path)
        input_basename = os.path.basename(inp_path)
        output_dir = os.path.join(input_dir, "..", "macro_covered")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        out_path = os.path.join(output_dir, input_basename)

    if rect_coords and len(rect_coords) != 4:
        parser.error("Rectangle requires 4 integer coordinates: x0 y0 x1 y1.")

    replace_macro(inp_path, out_path, rect_coords, verbosity, macro_desc)
