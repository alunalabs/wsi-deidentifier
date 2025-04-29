#!/usr/bin/env python3
# process_slides.py

import argparse
import csv
import hashlib
import re
import shutil
import struct
import sys
from pathlib import Path
import asyncio
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import fitz 

import tiffparser
from replace_macro import replace_macro
from find_identifying_boxes import find_barcodes, find_text_boxes, process_image
from pdf_processor import process_pdf


###############################################################################
# helpers (TIFF)
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
            data = redacted.encode()
            pad = tag.count - len(data)
            if pad < 0:
                data = data[: tag.count]
                pad = 0
            fp.seek(tag.valueoffset)
            fp.write(data)
            if pad:
                fp.write(b"\0" * pad)


###############################################################################
# PDF helpers
###############################################################################
def extract_images_from_pdf(
    pdf_path: Path,
) -> list[tuple[int, Image.Image, tuple[int, int, int, int]]]:
    reader = PdfReader(str(pdf_path))
    out: list[tuple[int, Image.Image, tuple[int, int, int, int]]] = []

    for page_num, page in enumerate(reader.pages):
        res = page.get("/Resources")
        if "/XObject" not in res:
            continue
        x_objects = res["/XObject"].get_object()
        for obj in x_objects.values():
            if obj.get("/Subtype") != "/Image":
                continue
            if obj.get("/Filter") == "/FlateDecode":
                img = Image.open(BytesIO(obj.get_data()))
                bbox = obj["/BBox"]  # [llx, lly, urx, ury]
                out.append((page_num, img, bbox))

    return out


def process_pdf_images(pdf_path: Path) -> list[tuple[int, int, int, int, int]]:
    images = extract_images_from_pdf(pdf_path)
    regions: list[tuple[int, int, int, int, int]] = []

    for page_num, img, bbox in images:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        for b in find_barcodes(cv_img):
            x, y, w, h = b["rect"]
            regions.append((page_num, bbox[0] + x, bbox[1] + y, w, h))

        temp = Path("temp_img.jpg")
        img.save(temp)
        for x, y, w, h in find_text_boxes(str(temp)):
            regions.append((page_num, bbox[0] + x, bbox[1] + y, w, h))
        temp.unlink(missing_ok=True)

    return regions


def redact_pdf_regions(pdf_path: Path, regions: list[tuple[int, int, int, int, int]]) -> None:
    if not regions:
        # Still clear metadata
        with fitz.open(pdf_path) as doc:
            doc.set_metadata({})
            doc.save(pdf_path, garbage=4, deflate=True)
        return

    # Group regions per page
    page_map: dict[int, list[tuple[int, int, int, int]]] = {}
    for page_idx, x, y, w, h in regions:
        page_map.setdefault(page_idx, []).append((x, y, w, h))

    tmp_path = pdf_path.with_suffix(".redacted.pdf")
    with fitz.open(pdf_path) as doc:
        for page_idx, rects in page_map.items():
            page = doc[page_idx]
            page_h = page.rect.height
            for x, y, w, h in rects:
                # convert PDF bottom-left coords → MuPDF top-left coords
                x0 = x
                y0 = page_h - (y + h)
                x1 = x + w
                y1 = page_h - y
                page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(0, 0, 0))
            page.apply_redactions()  # images default to pixelise

        doc.set_metadata({})
        doc.save(tmp_path, garbage=4, deflate=True)

    tmp_path.replace(pdf_path)


###############################################################################
# CLI helpers
###############################################################################
def check_macro_exists(file_path: Path, macro_description: str) -> bool:
    try:
        with file_path.open("rb") as fp:
            t = tiffparser.TiffFile(fp)
            for page in t.pages:
                desc_tag = page.tags.get("ImageDescription")
                if not desc_tag:
                    continue
                desc = desc_tag.value
                if isinstance(desc, bytes):
                    desc = desc.decode(errors="ignore")
                if macro_description.lower() in desc.lower():
                    return True
    except Exception as e:
        print(f"  Warning: Could not check for macro in {file_path}: {e}", file=sys.stderr)
        return False
    return False


###############################################################################
# main processing
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
    if src.suffix.lower() not in {".svs", ".tif", ".tiff", ".pdf"}:
        return
    slide_id = src.stem
    hashed_id = hash_id(slide_id, salt)
    dst = out_dir / f"{hashed_id}{src.suffix.lower()}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    if dst.suffix.lower() == ".pdf":
        process_pdf(dst)
    else:
        delete_associated_image(dst, "label")
        strip_metadata(dst)

        if check_macro_exists(dst, macro_description):
            try:
                replace_macro(
                    str(dst),
                    str(dst),
                    macro_description=macro_description,
                    rect_coords=rect_coords,
                )
            except Exception as e:
                print(f"  Error replacing macro in {dst}: {e}", file=sys.stderr)
        else:
            print(f"  Macro ({macro_description}) not found in {dst}, skipping.")

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
        description=(
            "Copy .svs, .tif, or .pdf files, scrub metadata, remove label/macro images "
            "and fully redact PHI regions."
        )
    )
    p.add_argument("slides", nargs="+", help="paths or glob patterns to slide/PDF files")
    p.add_argument("-o", "--out", default="deidentified", help="output directory")
    p.add_argument("--salt", required=True, help="secret salt")
    p.add_argument("-m", "--map", default="hash_mapping.csv", help="mapping CSV")
    p.add_argument(
        "--macro-description",
        default="macro",
        help="String identifier for the macro image (case-insensitive).",
    )
    p.add_argument(
        "--rect",
        metavar="N",
        type=int,
        nargs=4,
        default=None,
        help="[x0 y0 x1 y1] replacement rectangle for macro images.",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    mapping_exists = Path(args.map).is_file()
    with open(args.map, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["slide_id", "hashed_id", "src_path", "dst_path"])
        if not mapping_exists:
            writer.writeheader()

        all_paths: set[Path] = set()
        for pattern in args.slides:
            brace = re.match(r"(.*)\{(.*)\}(.*)", pattern)
            if brace:
                base, exts, suff = brace.groups()
                for ext in exts.split(","):
                    all_paths.update(Path().glob(f"{base}{ext}{suff}"))
            else:
                all_paths.update(Path().glob(pattern))

        if not all_paths:
            print("No files found.")
            return

        for path in sorted(all_paths):
            try:
                process_slide(
                    path,
                    out_dir,
                    args.salt,
                    writer,
                    args.macro_description,
                    tuple(args.rect) if args.rect else None,
                )
                print(f"✓ {path} → {out_dir}")
            except Exception as e:
                print(f"✗ {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
