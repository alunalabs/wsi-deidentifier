#!/usr/bin/env python3
import argparse
import csv
import hashlib
import re
import shutil
import struct
import sys
from pathlib import Path

import tiffparser


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
# CLI
###############################################################################
def hash_id(slide_id: str, salt: str) -> str:
    return hashlib.sha256((salt + slide_id).encode()).hexdigest()


def process_slide(src: Path, out_dir: Path, salt: str, writer) -> None:
    if src.suffix.lower() not in [".svs", ".tif", ".tiff"]:
        return
    slide_id = src.stem
    hashed_id = hash_id(slide_id, salt)
    dst = out_dir / f"{hashed_id}{src.suffix.lower()}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    delete_associated_image(dst, "label")
    # delete_associated_image(dst, "macro")
    strip_metadata(dst)

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
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    mapping_exists = Path(args.map).is_file()
    with open(args.map, "a", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["slide_id", "hashed_id", "src_path", "dst_path"]
        )
        if not mapping_exists:
            writer.writeheader()
        for pattern in args.slides:
            for path in Path().glob(pattern):
                try:
                    process_slide(path, out_dir, args.salt, writer)
                    print(f"✓ {path} → {out_dir}")
                except Exception as e:
                    print(f"✗ {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
