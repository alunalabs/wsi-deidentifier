#!/usr/bin/env python3
"""
Simple test script for verifying Docker container functionality.
"""
import argparse
import os
import sys

try:
    import cv2
    import numpy as np
    from pyzbar import pyzbar
except Exception:
    import pytest
    pytest.skip(
        "Required libraries for docker_test not available",
        allow_module_level=True,
    )

def find_barcodes(image):
    """Finds all supported barcodes (QR codes, etc.) using pyzbar."""
    barcode_boxes = []
    try:
        pyzbar_barcodes = pyzbar.decode(image)
        for barcode in pyzbar_barcodes:
            (x, y, w, h) = barcode.rect
            barcode_type = barcode.type
            barcode_data = barcode.data.decode("utf-8")
            barcode_boxes.append({
                "rect": (x, y, w, h),
                "type": f"pyzbar_{barcode_type}",
                "data": barcode_data,
            })
    except Exception as e:
        print(f"Error during barcode detection: {e}", file=sys.stderr)
    return barcode_boxes

def draw_boxes(image, barcode_boxes):
    """Draws bounding boxes on the image."""
    output_image = image.copy()
    # Draw Barcode boxes (Green)
    for barcode_info in barcode_boxes:
        x, y, w, h = barcode_info["rect"]
        barcode_type = barcode_info["type"]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output_image,
            barcode_type,
            (x, y - 10 if y > 10 else y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return output_image

def process_image(image_path, output_path=None):
    """Processes a single image: loads, finds boxes, draws/saves."""
    print(f"--- Processing {image_path} ---")

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from path: {image_path}", file=sys.stderr)
        return False  # Indicate failure
    print(f"  Image loaded successfully.")

    # --- Barcode Detection ---
    print("  Finding barcodes (QR codes, etc.)...")
    barcode_boxes = find_barcodes(image)
    print(f"  Found {len(barcode_boxes)} barcode(s).")
    
    # Draw boxes on the image
    output_image = draw_boxes(image, barcode_boxes)
    
    # Save the annotated image
    if output_path:
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(output_path, output_image)
            print(f"  Saved output image with boxes to: {output_path}")
        except Exception as e:
            print(f"  Error saving image to {output_path}: {e}", file=sys.stderr)
            return False
    
    print(f"--- Finished processing {image_path} ---")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Simple Docker test script to detect QR codes in images."
    )
    parser.add_argument(
        "input_path",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the annotated image.",
        default=None,
    )

    args = parser.parse_args()
    success = process_image(args.input_path, args.output)
    print("--- Processing Complete ---")
    if success:
        print("✅ Docker test successful!")
    else:
        print("❌ Docker test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()