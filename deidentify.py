#!/usr/bin/env python3
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

import tiffparser
from replace_macro import replace_macro
from PyPDF2 import PdfReader, PdfWriter
from find_identifying_boxes import find_barcodes, find_text_boxes, process_image


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


def extract_images_from_pdf(pdf_path: Path) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    """Extract images from a PDF along with their positions."""
    reader = PdfReader(str(pdf_path))
    images = []
    
    for page_num, page in enumerate(reader.pages):
        if '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject'].get_object()
            
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    # Get image data
                    if xObject[obj]['/Filter'] == '/FlateDecode':
                        data = xObject[obj].get_data()
                        img = Image.open(BytesIO(data))
                        # Get image position
                        bbox = xObject[obj]['/BBox']
                        images.append((img, bbox))
    
    return images

def process_pdf_images(pdf_path: Path) -> list[tuple[int, int, int, int]]:
    """Process images in a PDF to find PII regions."""
    images = extract_images_from_pdf(pdf_path)
    pii_regions = []
    
    for img, bbox in images:
        # Convert PIL Image to OpenCV format
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Find barcodes
        barcode_boxes = find_barcodes(cv_img)
        
        # Find text regions
        # Save image temporarily for GCP Vision
        temp_path = Path("temp_image.jpg")
        img.save(temp_path)
        text_boxes = find_text_boxes(str(temp_path))
        temp_path.unlink()
        
        # Process with Gemini for PII detection
        # Note: This would need to be adapted to work with the image directly
        # For now, we'll use the barcode and text boxes as PII regions
        for box in barcode_boxes:
            x, y, w, h = box['rect']
            # Convert to PDF coordinates
            pdf_x = bbox[0] + x
            pdf_y = bbox[1] + y
            pii_regions.append((pdf_x, pdf_y, w, h))
            
        for box in text_boxes:
            x, y, w, h = box
            # Convert to PDF coordinates
            pdf_x = bbox[0] + x
            pdf_y = bbox[1] + y
            pii_regions.append((pdf_x, pdf_y, w, h))
    
    return pii_regions

def redact_pdf_regions(pdf_path: Path, regions: list[tuple[int, int, int, int]]) -> None:
    """Redact identified PII regions in a PDF."""
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    
    for page in reader.pages:
        # Create a new page with redactions
        page = page.get_object()
        
        # Add redaction rectangles for each region
        for x, y, w, h in regions:
            # Create a redaction rectangle
            page.merge_page({
                '/Type': '/Annot',
                '/Subtype': '/Redact',
                '/Rect': [x, y, x + w, y + h],
                '/OC': {
                    '/Type': '/OCG',
                    '/Name': 'Redaction',
                    '/Usage': {
                        '/View': {'/ViewState': 'OFF'},
                        '/Print': {'/PrintState': 'OFF'},
                    }
                }
            })
        
        writer.add_page(page)
    
    # Save the redacted PDF
    with pdf_path.open('wb') as output_file:
        writer.write(output_file)

def strip_pdf_metadata(path: Path) -> None:
    """Strip metadata and redact PII from a PDF file."""
    # First process images to find PII
    print(f"Processing images in {path} for PII detection...")
    pii_regions = process_pdf_images(path)
    
    if pii_regions:
        print(f"Found {len(pii_regions)} PII regions to redact")
        # Redact the identified regions
        redact_pdf_regions(path, pii_regions)
    else:
        print("No PII regions found")
    
    # Remove metadata
    reader = PdfReader(str(path))
    writer = PdfWriter()

    # Copy all pages
    for page in reader.pages:
        writer.add_page(page)

    # Remove all metadata
    writer.metadata = {}

    # Save the new PDF
    with path.open("wb") as output_file:
        writer.write(output_file)


###############################################################################
# Helper Functions (including new one)
###############################################################################
def check_macro_exists(file_path: Path, macro_description: str) -> bool:
    """Checks if a macro image with the given description exists in the TIFF file."""
    try:
        with file_path.open("rb") as fp:
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
            f"  Warning: Could not check for macro in {file_path}: {e}", file=sys.stderr
        )
        # If we can't check, assume it doesn't exist or is inaccessible
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
    if src.suffix.lower() not in [".svs", ".tif", ".tiff", ".pdf"]:
        return
    slide_id = src.stem
    hashed_id = hash_id(slide_id, salt)
    dst = out_dir / f"{hashed_id}{src.suffix.lower()}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    if src.suffix.lower() == ".pdf":
        strip_pdf_metadata(dst)
    else:
        delete_associated_image(dst, "label")
        strip_metadata(dst)

        # Check if macro exists before trying to replace it
        if check_macro_exists(dst, macro_description):
            print(f"  Replacing macro ({macro_description}) in {dst}")
            try:
                replace_macro(
                    str(dst),
                    str(dst),
                    macro_description=macro_description,
                    rect_coords=rect_coords,
                )
            except Exception as e:
                print(
                    f"  Error replacing macro in {dst}: {e}",
                    file=sys.stderr,
                )
        else:
            print(
                f"  Macro ({macro_description}) not found or inaccessible in {dst}, skipping replacement."
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
        description="Copy .svs, .tif, or .pdf files, strip label/macro images (for slides), scrub metadata, and rename using salted hash"
    )
    p.add_argument(
        "slides", nargs="+", help="paths or glob patterns to .svs, .tif, or .pdf files"
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

        all_paths = set()
        for pattern in args.slides:
            print(f"Processing pattern: {pattern}")
            expanded_patterns = []
            # Basic brace expansion
            match = re.match(r"(.*)\{(.*)\}(.*)", pattern)
            if match:
                base, exts_str, suffix = match.groups()
                extensions = exts_str.split(",")
                expanded_patterns = [f"{base}{ext}{suffix}" for ext in extensions]
                print(f"  Expanded to: {expanded_patterns}")
            else:  # No braces or malformed, use as is
                expanded_patterns = [pattern]

            for exp_pattern in expanded_patterns:
                for path in Path().glob(exp_pattern):
                    all_paths.add(path)  # Collect unique paths

        print(f"Found {len(all_paths)} unique files to process.")

        if not all_paths:
            print("No files found matching the provided patterns.")
            return  # Exit early if no files found

        for path in sorted(list(all_paths)):  # Process in sorted order
            print(f"Processing path: {path}")
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
