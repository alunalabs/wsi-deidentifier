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

# Constants for TIFF variants
TIFF_MAGIC_STANDARD = 42  # Classic TIFF
TIFF_MAGIC_BIG = 43  # BigTIFF (64-bit offsets)


class IFDEntry:
    """Represents one IFD (Image File Directory) entry.

    For classic TIFF (`big == False`) offsets are 32-bit and the *value* and
    *count* fields are 4 bytes.  In BigTIFF (`big == True`) both are 64-bit
    which changes the entry size from 12 bytes to 20 bytes.
    """

    __slots__ = (
        "tag",
        "typ",
        "count",
        "value",
        "value_pos",  # absolute position (byte offset) of the value/offset field
        "count_pos",  # absolute position of the count field
        "big",  # bool – True if BigTIFF (uses 8-byte count/value)
    )

    def __init__(self, tag, typ, cnt, val, value_pos, count_pos, big):
        self.tag = tag
        self.typ = typ
        self.count = cnt
        self.value = val
        self.value_pos = value_pos
        self.count_pos = count_pos
        self.big = big


# --- IFD readers -----------------------------------------------------------


def _read_ifd_standard(buf: bytes | bytearray, off: int, endian: str):
    """Read a classic TIFF IFD (32-bit offsets)."""

    num_entries = struct.unpack(endian + "H", buf[off : off + 2])[0]
    entries: list[IFDEntry] = []
    cur = off + 2

    for _ in range(num_entries):
        tag, typ, cnt, val = struct.unpack(endian + "HHII", buf[cur : cur + 12])
        # value field starts at cur+8, count at cur+4
        entries.append(IFDEntry(tag, typ, cnt, val, cur + 8, cur + 4, False))
        cur += 12

    nxt = struct.unpack(endian + "I", buf[cur : cur + 4])[0]
    return entries, nxt


def _read_ifd_big(buf: bytes | bytearray, off: int, endian: str):
    """Read a BigTIFF IFD (64-bit offsets)."""

    num_entries = struct.unpack(endian + "Q", buf[off : off + 8])[0]
    entries: list[IFDEntry] = []
    cur = off + 8

    for _ in range(num_entries):
        tag, typ = struct.unpack(endian + "HH", buf[cur : cur + 4])
        cnt = struct.unpack(endian + "Q", buf[cur + 4 : cur + 12])[0]
        val = struct.unpack(endian + "Q", buf[cur + 12 : cur + 20])[0]
        # In BigTIFF, value starts at cur+12, count at cur+4
        entries.append(IFDEntry(tag, typ, cnt, val, cur + 12, cur + 4, True))
        cur += 20

    nxt = struct.unpack(endian + "Q", buf[cur : cur + 8])[0]
    return entries, nxt


def read_ifd(buf: bytes | bytearray, off: int, endian: str, *, big: bool = False):
    """Dispatch to the correct IFD reader depending on *big*."""

    return (
        _read_ifd_big(buf, off, endian) if big else _read_ifd_standard(buf, off, endian)
    )


# --- Convenience helpers ---------------------------------------------------


def ascii_val(entries, tag, data: bytes | bytearray, endian: str):
    """Return ASCII value of *tag* from *entries*, or empty string."""

    for e in entries:
        if e.tag == tag:
            inline_bytes = 8 if e.big else 4
            if e.count <= inline_bytes:
                raw = struct.pack(endian + ("Q" if e.big else "I"), e.value)[: e.count]
            else:
                raw = data[e.value : e.value + e.count]
            return raw.rstrip(b"\0").decode("ascii", "ignore")
    return ""


def _short_from_raw(raw: bytes, endian: str) -> int:
    return struct.unpack(endian + "H", raw)[0]


def short_val(entry: IFDEntry, data: bytes | bytearray, endian: str):
    """Return first SHORT value from *entry*."""

    if entry.typ != 3 or entry.count < 1:
        return None

    inline_bytes = 8 if entry.big else 4
    if entry.count * 2 <= inline_bytes:
        raw = struct.pack(endian + ("Q" if entry.big else "I"), entry.value)[:2]
    else:
        raw = data[entry.value : entry.value + 2]

    return _short_from_raw(raw, endian)


def int_val(entries, tag, data, endian):
    for e in entries:
        if e.tag == tag:
            return short_val(e, data, endian) if e.typ == 3 else e.value
    return None


def array_val(entry: IFDEntry, data: bytes | bytearray, endian: str):
    size = 2 if entry.typ == 3 else (8 if entry.big else 4)
    inline_bytes = 8 if entry.big else 4

    if entry.count * size <= inline_bytes:
        raw = struct.pack(endian + ("Q" if entry.big else "I"), entry.value)[
            : entry.count * size
        ]
    else:
        raw = data[entry.value : entry.value + entry.count * size]

    fmt = endian + ("H" if entry.typ == 3 else ("Q" if entry.big else "I"))
    return [struct.unpack(fmt, raw[i : i + size])[0] for i in range(0, len(raw), size)]


def set_int(buf: bytearray, entry: IFDEntry, v: int, endian: str):
    fmt = "Q" if entry.big else "I"
    struct.pack_into(endian + fmt, buf, entry.value_pos, v)
    entry.value = v


def set_count(buf: bytearray, entry: IFDEntry, c: int, endian: str):
    fmt = "Q" if entry.big else "I"
    struct.pack_into(endian + fmt, buf, entry.count_pos, c)
    entry.count = c


def align(n: int, big: bool) -> int:
    """Round *n* up to a 4-byte (classic TIFF) or 8-byte (BigTIFF) boundary."""
    return (n + (7 if big else 3)) & (~7 if big else ~3)


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


# ---------------------------------------------------------------------------
# Public helpers ------------------------------------------------------------


def read_svs_macro_info(input_path, *, macro_description="macro", verbose=False):
    """Parse *input_path* and locate the macro IFD.

    Returns a dictionary with the following keys::

        data            – the full file contents (bytearray – mutable)
        endian          – byte-order mark ("<" little, ">" big)
        bigtiff         – bool – *True* if BigTIFF (64-bit offsets)
        macro_entries   – list[IFDEntry] for the macro image
        width, height   – macro dimensions (pixels)
        spp, bps, cmp   – samples/pixel, bits/sample, compression

    The function terminates the program via *fail()* if the slide is not a
    TIFF or no macro image could be found.  It does **not** attempt to decode
    the actual macro bitmap – that is left to the caller.
    """

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logging.info("Opening %s", input_path)

    try:
        with open(input_path, "rb") as fh:
            data = bytearray(fh.read())
    except FileNotFoundError:
        fail(f"Input file not found: {input_path}")
    except Exception as exc:
        fail(f"Error reading input file {input_path}: {exc}")

    # --- Detect byte order + classic vs BigTIFF -------------------------
    endian = {"II": "<", "MM": ">"}.get(data[:2].decode("ascii", "ignore"))
    if not endian:
        fail("Not a TIFF/SVS – could not determine byte order")

    magic = struct.unpack(endian + "H", data[2:4])[0]
    bigtiff = magic == TIFF_MAGIC_BIG  # 43 ⇒ BigTIFF, 42 ⇒ classic

    if bigtiff:
        # BigTIFF header: bytes 4-6 contain offset size (should be 8) and
        # reserved; first IFD offset is 8-byte @8
        ifd_off = struct.unpack(endian + "Q", data[8:16])[0]
    else:
        ifd_off = struct.unpack(endian + "I", data[4:8])[0]

    macro_entries: list[IFDEntry] | None = None

    while ifd_off:
        entries, nxt = read_ifd(data, ifd_off, endian, big=bigtiff)
        desc = ascii_val(entries, TAG_IMAGE_DESCRIPTION, data, endian).lower()
        if macro_description.lower() in desc:
            macro_entries = entries
            break
        ifd_off = nxt

    if macro_entries is None:
        fail("Macro image not found", description=macro_description)

    width = int_val(macro_entries, TAG_IMAGE_WIDTH, data, endian)
    height = int_val(macro_entries, TAG_IMAGE_LENGTH, data, endian)
    spp = int_val(macro_entries, TAG_SAMPLES_PER_PIXEL, data, endian) or 3
    bps = int_val(macro_entries, TAG_BITS_PER_SAMPLE, data, endian) or 8
    cmp = int_val(macro_entries, TAG_COMPRESSION, data, endian) or 1

    logging.debug(
        "Macro: %dx%d, spp=%d, bps=%d, compression=%d",
        width,
        height,
        spp,
        bps,
        cmp,
    )

    if spp != 3:
        fail("Unsupported macro format – need RGB samples_per_pixel=3", spp=spp)

    return {
        "data": data,
        "endian": endian,
        "bigtiff": bigtiff,
        "macro_entries": macro_entries,
        "width": width,
        "height": height,
        "spp": spp,
        "bps": bps,
        "cmp": cmp,
    }


def _write_slide_with_new_macro(
    *,
    info: dict,
    img: Image.Image,
    input_path: str,
    output_path: str,
    verbose: bool = False,
):
    """Patch *input_path* so that the macro image is replaced with *img*.

    The heavy lifting for *replace_macro* and *replace_macro_with_image* lives
    here so both variants share the exact same byte-level logic.
    """

    data: bytearray = info["data"]
    endian: str = info["endian"]
    bigtiff: bool = info["bigtiff"]
    macro: list[IFDEntry] = info["macro_entries"]
    w: int = info["width"]
    h: int = info["height"]

    # Ensure correct colour space & size ---------------------------------
    img = img.convert("RGB")
    if (img.width, img.height) != (w, h):
        logging.debug(
            "Resizing replacement image from %dx%d → %dx%d", img.width, img.height, w, h
        )
        img = img.resize((w, h), Image.LANCZOS)

    buf = bytearray(img.tobytes())

    original_size = len(data)
    new_off = align(original_size, bigtiff)
    bc = len(buf)

    # Helper for quick tag lookup in the macro IFD -----------------------
    def entry(tag):
        return next((e for e in macro if e.tag == tag), None)

    patches: list[tuple[int, bytes]] = []  # (file_position, packed_bytes)

    def record_int_patch(e: IFDEntry, v: int):
        fmt = "Q" if e.big else "I"
        patches.append((e.value_pos, struct.pack(endian + fmt, v)))

    def record_count_patch(e: IFDEntry, c: int):
        fmt = "Q" if e.big else "I"
        patches.append((e.count_pos, struct.pack(endian + fmt, c)))

    # Set Compression=1 (uncompressed) & PhotometricInterpretation=2 (RGB)
    for tag, val in ((TAG_COMPRESSION, 1), (TAG_PHOTOMETRIC, 2)):
        if e := entry(tag):
            set_int(data, e, val, endian)
            record_int_patch(e, val)

    # Update strip tags ---------------------------------------------------
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
            record_count_patch(e, 1)
        set_int(data, e, val, endian)
        record_int_patch(e, val)

    # ------------------------------------------------------------------
    # Create the output file efficiently (hard-link fall-back copy) -----
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
        os.link(input_path, output_path)
    except (OSError, AttributeError):
        import shutil

        shutil.copy2(input_path, output_path)

    # ------------------------------------------------------------------
    # Patch header + append new pixel buffer -----------------------------
    try:
        with open(output_path, "r+b") as fh:
            for pos, packed in patches:
                fh.seek(pos)
                fh.write(packed)

            fh.seek(0, os.SEEK_END)
            cur_size = fh.tell()
            if cur_size < new_off:
                fh.write(b"\0" * (new_off - cur_size))
            fh.write(buf)

        logging.info("Saved %s", output_path)
    except Exception as exc:
        fail(f"Error writing output file {output_path}: {exc}")


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------


def replace_macro(
    input_path,
    output_path,
    rect_coords=None,
    verbose=False,
    macro_description="macro",
    fill_color=(255, 0, 0),
):
    """High-level helper – redact the macro by drawing a rectangle.

    This convenience wrapper keeps the original CLI behaviour intact while
    delegating all heavy lifting to :func:`read_svs_macro_info` and
    :func:`_write_slide_with_new_macro`.
    """

    # Gather metadata + verify slide ------------------------------------
    info = read_svs_macro_info(
        input_path, macro_description=macro_description, verbose=verbose
    )

    w = info["width"]
    h = info["height"]

    # Decode existing macro so we can paint the rectangle ---------------
    img = load_macro_openslide(input_path)
    if img is None:
        fail("Unable to decode macro", compression=info["cmp"])
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

    ImageDraw.Draw(img).rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill_color)
    logging.debug("Rectangle drawn at %s", (x0, y0, x1, y1))

    _write_slide_with_new_macro(
        info=info,
        img=img,
        input_path=input_path,
        output_path=output_path,
        verbose=verbose,
    )


def replace_macro_with_image(
    *,
    input_path: str,
    output_path: str,
    replacement_img: "Image.Image",
    verbose: bool = False,
    macro_description: str = "macro",
):
    """Replace the macro thumbnail with *replacement_img*.

    The *replacement_img* is converted to RGB and resized if necessary to the
    dimensions of the original macro.
    """

    info = read_svs_macro_info(
        input_path, macro_description=macro_description, verbose=verbose
    )

    _write_slide_with_new_macro(
        info=info,
        img=replacement_img,
        input_path=input_path,
        output_path=output_path,
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
        output_dir = os.path.join(os.path.dirname(input_dir), "macro_covered")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        out_path = os.path.join(output_dir, input_basename)

    if rect_coords and len(rect_coords) != 4:
        parser.error("Rectangle requires 4 integer coordinates: x0 y0 x1 y1.")

    replace_macro(inp_path, out_path, rect_coords, verbosity, macro_desc)
