#!/usr/bin/env python3
"""
Identifies bounding boxes that contain identifying information in WSI files.
Saves the results to a JSON file for later use with deidentify.py.

Usage
-----
python identify_boxes.py --input "sample/identified/*.svs" --output boxes.json

The script will:
1. Identify all slide files matching the given pattern(s)
2. Extract and analyze macro images for each file
3. Generate bounding boxes for areas containing PHI
4. Save results to a JSON file for later inspection and use with deidentify.py
"""

import argparse
import asyncio
import glob
import json
import os
import sys
from pathlib import Path

import cv2

from gcp_textract import detect_text
from gemini_extract import gemini_extract

# https://github.com/NaturalHistoryMuseum/pyzbar/issues/131
# export DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix zbar)/lib/:$(brew --prefix libdmtx)/lib/


async def process_slide(
    file_path,
    project=None,
    location=None,
):
    """
    Process a single slide file to identify bounding boxes containing PHI.
    Returns a dictionary with the file path and identified boxes.
    """
    print(f"--- Processing {file_path} ---")

    # First, extract macro image
    # This would normally happen inside find_identifying_boxes.py
    # But we need to gather the coordinates here

    # Assuming that the macro image has been extracted to a predictable location
    # We're going to use the file_path to determine a likely macro image path
    slide_basename = os.path.basename(file_path)
    slide_name = os.path.splitext(slide_basename)[0]

    # Try to find the corresponding macro image
    macro_dir = os.path.join(
        os.path.dirname(os.path.dirname(file_path)), "macro_images"
    )
    potential_macro_paths = glob.glob(os.path.join(macro_dir, f"{slide_name}*_macro.*"))

    if not potential_macro_paths:
        print(f"Warning: Could not find macro image for {file_path}")
        return {
            "file_path": str(file_path),
            "boxes": [],
            "error": "No macro image found",
        }

    macro_path = potential_macro_paths[0]
    print(f"Found macro image: {macro_path}")

    # Load the image to get dimensions
    try:
        image = cv2.imread(macro_path)
        if image is None:
            print(
                f"Error: Could not load image from path: {macro_path}", file=sys.stderr
            )
            return {
                "file_path": str(file_path),
                "boxes": [],
                "error": "Failed to load macro image",
            }
        img_height, img_width = image.shape[:2]
    except Exception as e:
        print(f"Error loading image {macro_path}: {e}", file=sys.stderr)
        return {
            "file_path": str(file_path),
            "boxes": [],
            "error": f"Failed to process macro image: {e}",
        }

    # Process with GCP Vision and Gemini concurrently
    gemini_response, text_boxes = await asyncio.gather(
        gemini_extract(file_path=macro_path, project=project, location=location),
        asyncio.to_thread(detect_text, macro_path),
    )

    # Process text boxes from GCP OCR
    all_text_boxes = []
    if text_boxes and len(text_boxes) > 1:
        for text_annotation in text_boxes[1:]:
            vertices = text_annotation.bounding_poly.vertices
            # Convert vertices to a list of points
            points = [(v.x, v.y) for v in vertices]

            # Calculate the bounding box (x, y, w, h)
            np_points = [(int(x), int(y)) for x, y in points]
            if np_points:
                x = min(p[0] for p in np_points)
                y = min(p[1] for p in np_points)
                max_x = max(p[0] for p in np_points)
                max_y = max(p[1] for p in np_points)
                w = max_x - x
                h = max_y - y

                # Basic filtering (ensure width and height are positive)
                if w > 0 and h > 0:
                    all_text_boxes.append((x, y, w, h))

    # Process Gemini text boxes
    gemini_text_boxes_abs = []
    gemini_tissue_boxes_abs = []

    for box in gemini_response:
        x_abs = int(box["x"] * img_width)
        y_abs = int(box["y"] * img_height)
        w_abs = int(box["width"] * img_width)
        h_abs = int(box["height"] * img_height)

        # Basic validation
        if w_abs <= 0 or h_abs <= 0:
            continue

        if box["label"] == "tissue":
            gemini_tissue_boxes_abs.append((x_abs, y_abs, w_abs, h_abs))
        elif box["label"] == "text":
            gemini_text_boxes_abs.append((x_abs, y_abs, w_abs, h_abs))

    # Combine text boxes (both GCP and Gemini)
    all_boxes = all_text_boxes + gemini_text_boxes_abs

    # Filter boxes that intersect with tissue
    filtered_boxes = []

    def check_intersection(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        overlap_x = (x1 < x2 + w2) and (x1 + w1 > x2)
        overlap_y = (y1 < y2 + h2) and (y1 + h1 > y2)
        return overlap_x and overlap_y

    if gemini_tissue_boxes_abs:
        for box in all_boxes:
            intersect = False
            for tissue_box in gemini_tissue_boxes_abs:
                if check_intersection(box, tissue_box):
                    intersect = True
                    break
            if not intersect:
                filtered_boxes.append(box)
    else:
        filtered_boxes = all_boxes

    # Calculate encompassing box for all filtered boxes
    encompassing_box = None
    if filtered_boxes:
        min_x = min(box[0] for box in filtered_boxes)
        min_y = min(box[1] for box in filtered_boxes)
        max_x_w = max(box[0] + box[2] for box in filtered_boxes)
        max_y_h = max(box[1] + box[3] for box in filtered_boxes)
        enc_w = max_x_w - min_x
        enc_h = max_y_h - min_y
        encompassing_box = (min_x, min_y, enc_w, enc_h)

    # Convert box format to match what replace_macro.py expects (x0, y0, x1, y1)
    formatted_encompassing_box = None
    if encompassing_box:
        x, y, w, h = encompassing_box
        formatted_encompassing_box = (x, y, x + w, y + h)

    result = {
        "file_path": str(file_path),
        "macro_path": str(macro_path),
        "rect_coords": formatted_encompassing_box,
        "individual_boxes": [
            {"x": x, "y": y, "width": w, "height": h} for x, y, w, h in filtered_boxes
        ],
    }

    return result


async def main():
    parser = argparse.ArgumentParser(
        description="Identify bounding boxes containing PHI in WSI files and save to JSON."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Paths or glob patterns to .svs or .tif files to process.",
    )
    parser.add_argument(
        "--output",
        default="identified_boxes.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--project",
        help="Google Cloud project ID to use for Vertex AI (Gemini).",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
    )
    parser.add_argument(
        "--location",
        help="Google Cloud location for Vertex AI (Gemini).",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )

    args = parser.parse_args()

    # Expand input patterns
    all_files = set()
    for pattern in args.input:
        expanded_patterns = []
        # Handle brace expansion similar to deidentify.py
        import re

        match = re.match(r"(.*)\{(.*)\}(.*)", pattern)
        if match:
            base, exts_str, suffix = match.groups()
            extensions = exts_str.split(",")
            expanded_patterns = [f"{base}{ext}{suffix}" for ext in extensions]
        else:
            expanded_patterns = [pattern]

        for exp_pattern in expanded_patterns:
            for path in Path().glob(exp_pattern):
                if path.suffix.lower() in [".svs", ".tif", ".tiff"]:
                    all_files.add(path)

    if not all_files:
        print("No files found matching the provided patterns.")
        return

    print(f"Found {len(all_files)} files to process.")

    # Process files with limited concurrency
    semaphore = asyncio.Semaphore(5)  # Process 5 files at a time

    async def process_with_semaphore(file_path):
        async with semaphore:
            return await process_slide(
                file_path, project=args.project, location=args.location
            )

    tasks = [process_with_semaphore(file_path) for file_path in sorted(all_files)]
    all_results = await asyncio.gather(*tasks)

    # Save results to JSON
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {args.output}")
    print(f"Processed {len(all_results)} files. Use this JSON file with deidentify.py.")


if __name__ == "__main__":
    asyncio.run(main())
