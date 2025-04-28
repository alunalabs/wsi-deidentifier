import argparse
import os  # For GCP auth check and path handling
import sys  # Import sys for exit
import time  # Add time import

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode as dmtx_decode
from pyzbar import pyzbar

from gcp_textract import detect_text  # Import the function

# Optional: Default Tesseract path (can be overridden by command line argument)
# On Windows, it might be: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On macOS (via Homebrew): '/opt/homebrew/bin/tesseract' or '/usr/local/bin/tesseract'
# On Linux (default install): '/usr/bin/tesseract'
# Set this if Tesseract is not in your system's PATH
# pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'

# https://github.com/NaturalHistoryMuseum/pyzbar/issues/131
# export DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix zbar)/lib/
# export DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix libdmtx)/lib/

# export DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix zbar)/lib/:$(brew --prefix libdmtx)/lib/


def find_barcodes(image):
    """Finds all supported barcodes (QR, DataMatrix, etc.) using pyzbar and pylibdmtx,
    and returns their bounding boxes."""
    start_time = time.time()
    barcode_boxes = []

    print("  pyzbar detection started")
    # --- Use pyzbar ---
    pyzbar_start = time.time()
    try:
        pyzbar_barcodes = pyzbar.decode(image)
        for barcode in pyzbar_barcodes:
            (x, y, w, h) = barcode.rect
            barcode_type = barcode.type
            barcode_data = barcode.data.decode("utf-8")
            barcode_boxes.append(
                {
                    "rect": (x, y, w, h),
                    "type": f"pyzbar_{barcode_type}",
                    "data": barcode_data,
                }  # Prefix type
            )
            # print(f"  - Found {barcode_type} (pyzbar): {barcode_data} at {(x, y, w, h)}")
    except Exception as e:
        print(f"Error during pyzbar detection: {e}", file=sys.stderr)
    pyzbar_duration = time.time() - pyzbar_start
    print(f"  pyzbar detection took: {pyzbar_duration:.4f} seconds")

    print("  pylibdmtx detection started")
    # --- Use pylibdmtx ---
    dmtx_start = time.time()
    try:
        raise "temp disable"
        # pylibdmtx works best with grayscale images
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dmtx_barcodes = dmtx_decode(gray_image)
        for barcode in dmtx_barcodes:
            # pylibdmtx gives corner points (polygon), calculate bounding box
            x = barcode.rect.left
            y = barcode.rect.top
            w = barcode.rect.width
            h = barcode.rect.height
            barcode_type = "DATAMATRIX"  # pylibdmtx only detects DataMatrix
            barcode_data = barcode.data.decode("utf-8")
            # TODO: Add mechanism to avoid adding duplicate barcodes if detected by both libs
            barcode_boxes.append(
                {
                    "rect": (x, y, w, h),
                    "type": f"dmtx_{barcode_type}",
                    "data": barcode_data,
                }  # Prefix type
            )
            # print(f"  - Found {barcode_type} (dmtx): {barcode_data} at {(x, y, w, h)}")
    except ImportError:
        # This might happen if pylibdmtx or its dependencies are not correctly installed
        print(
            "Warning: pylibdmtx library not found or not properly installed. Skipping DataMatrix detection.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Error during pylibdmtx detection: {e}", file=sys.stderr)
    dmtx_duration = time.time() - dmtx_start
    print(f"  pylibdmtx detection took: {dmtx_duration:.4f} seconds")

    total_duration = time.time() - start_time
    print(f"Total barcode detection took: {total_duration:.4f} seconds")
    return barcode_boxes


def find_text_boxes(image_path):
    """Finds text regions using Google Cloud Vision API."""
    start_time = time.time()
    all_text_boxes = []
    # debug_boxes_raw = [] # No longer needed for GCP polygons, but keep for now if we draw them

    print(f"  Running Google Cloud Vision text detection on {image_path}...")
    try:
        # Call the imported detect_text function
        gcp_texts = detect_text(image_path)
        if not gcp_texts:
            print("  Google Cloud Vision returned no text results.")
            return []
    except Exception as e:
        print(
            f"An error occurred during Google Cloud Vision processing: {e}",
            file=sys.stderr,
        )
        return []  # Return empty list on error

    gcp_duration = time.time() - start_time
    print(f"  Google Cloud Vision detection took: {gcp_duration:.4f} seconds")

    # Process GCP results
    # The first text annotation is the full text block, skip it.
    if len(gcp_texts) > 1:
        for text_annotation in gcp_texts[1:]:
            vertices = text_annotation.bounding_poly.vertices
            # Convert vertices to a NumPy array for easier calculation
            np_points = np.array([(v.x, v.y) for v in vertices], dtype=np.int32)

            # Calculate the upright bounding box (x, y, w, h) from the polygon vertices
            if len(np_points) > 0:
                x, y, w, h = cv2.boundingRect(np_points)

                # Basic filtering (ensure width and height are positive)
                if w > 0 and h > 0:
                    all_text_boxes.append((x, y, w, h))
                    # debug_boxes_raw.append(np_points) # Keep if needed for polygon drawing later
                    # Optional: Print detected text info
                    # print(f"    - Text: '{text_annotation.description}' at {(x, y, w, h)}")
    else:
        print("  Google Cloud Vision returned only the full text block or nothing.")

    # Debug image saving logic removed as it was specific to PaddleOCR output format/needs.
    # Could be re-added later if needed, using np_points.

    print(f"  Found {len(all_text_boxes)} text boxes from GCP Vision.")

    total_duration = time.time() - start_time
    print(f"Total text box finding took: {total_duration:.4f} seconds")
    # Return both the list of (x,y,w,h) boxes and the raw polygon points for potential drawing
    return all_text_boxes  # Only return the list of upright bounding boxes


def draw_boxes(image, barcode_boxes, text_boxes, encompassing_box=None):
    """Draws bounding boxes on the image."""
    output_image = image.copy()
    # Draw Barcode boxes (Green) - Now uses type
    for barcode_info in barcode_boxes:
        x, y, w, h = barcode_info["rect"]
        barcode_type = barcode_info["type"]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output_image,
            barcode_type,  # Use the actual type
            (x, y - 10 if y > 10 else y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Draw Text boxes (Red)
    for x, y, w, h in text_boxes:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Adding text label for text boxes can be very cluttered, uncomment if needed
        # cv2.putText(output_image, 'Text', (x, y - 10 if y > 10 else y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw Encompassing box (Green)
    if encompassing_box:
        x, y, w, h = encompassing_box
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return output_image


def display_image(window_name, image):
    """Displays the image in a window, resizing if necessary."""
    start_time = time.time()
    try:
        max_display_dim = 1000  # Max width/height for the display window
        h_img, w_img = image.shape[:2]

        # Calculate scaling factor to fit within max dimensions while preserving aspect ratio
        scale = min(max_display_dim / w_img, max_display_dim / h_img, 1.0)

        if scale < 1.0:
            new_width = int(w_img * scale)
            new_height = int(h_img * scale)
            display_img = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        else:
            display_img = image

        cv2.imshow(window_name, display_img)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        # Handle errors that might occur if no display environment is available (e.g., SSH session)
        print("\nWarning: Could not display the image window.", file=sys.stderr)
        print(f"OpenCV Error: {e}", file=sys.stderr)
        print(
            "This might happen if you are running in an environment without a graphical display.",
            file=sys.stderr,
        )
        print(
            "Consider using the --output argument to save the image instead.",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"\nAn unexpected error occurred during image display: {e}", file=sys.stderr
        )
        cv2.destroyAllWindows()  # Attempt to close any windows that might have opened partially
    duration = time.time() - start_time
    print(f"Image display function took: {duration:.4f} seconds")


def main():
    total_start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Detect QR codes and text regions in an image and output bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Corrected syntax/ensure no corruption here
    )
    parser.add_argument(
        "image_path", help="Path to the input image file."
    )  # Corrected syntax
    parser.add_argument(
        "--output",
        help="Optional path to save the image with bounding boxes drawn. If not provided, tries to display the image.",
        default=None,
    )
    parser.add_argument(
        "--hide-window",
        action="store_true",
        help="Do not display the image window, even if no output path is specified.",
    )

    args = parser.parse_args()

    # --- Add GCP Authentication Check ---
    creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    default_creds_path = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )
    if not creds_env and not os.path.exists(default_creds_path):
        print(
            "Warning: Google Cloud authentication credentials not found.",
            file=sys.stderr,
        )
        print(
            "Ensure you have run 'gcloud auth application-default login' or set the GOOGLE_APPLICATION_CREDENTIALS environment variable.",
            file=sys.stderr,
        )
        # sys.exit(1) # Exit if credentials are required
    elif creds_env:
        print(f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {creds_env}")
    else:
        print(f"Using application default credentials found at: {default_creds_path}")
    # --- End GCP Auth Check ---

    # Load the image using OpenCV
    load_start = time.time()
    image = cv2.imread(args.image_path)
    load_duration = time.time() - load_start
    if image is None:
        print(
            f"Error: Could not load image from path: {args.image_path}", file=sys.stderr
        )
        sys.exit(1)
    print(f"Processing image: {args.image_path} (Load time: {load_duration:.4f}s)")

    # --- Barcode Detection ---
    print("Finding barcodes (QR, DataMatrix, etc.)...")
    barcode_start = time.time()
    barcode_boxes = find_barcodes(image)  # Use renamed function
    barcode_duration = time.time() - barcode_start
    print(
        f"Found {len(barcode_boxes)} barcode(s). (Detection time: {barcode_duration:.4f}s)"
    )
    if barcode_boxes:
        print("Barcode Details:")
        for box_info in barcode_boxes:
            print(
                f"  - Type: {box_info['type']}, Rect: {box_info['rect']}, Data: {box_info['data'][:30]}..."
            )  # Print type and truncated data

    # --- Text Detection ---
    print("Finding text boxes (using Google Cloud Vision)...")  # Updated print
    text_start = time.time()
    # Now returns only a list of boxes
    text_boxes = find_text_boxes(args.image_path)
    text_duration = time.time() - text_start

    # Simplified error check (find_text_boxes returns [] on init or processing error)
    if not isinstance(text_boxes, list):
        # This case shouldn't happen with current logic, but as a safeguard
        print(
            "An unexpected error occurred during GCP text detection.", file=sys.stderr
        )
        sys.exit(1)

    # Proceed with reporting text box findings
    if text_boxes:
        print(
            f"Found {len(text_boxes)} potential text box(es) via GCP. (Detection time: {text_duration:.4f}s)"
        )
        print("Text Bounding Boxes (x, y, width, height):")
        for box in text_boxes:
            print(f"  - {box}")
    else:
        # Adjust message slightly if it could be an empty list vs None
        # (find_text_boxes now always returns a list)
        print(f"Found 0 text boxes via GCP. (Detection time: {text_duration:.4f}s)")

    # Combine boxes for drawing/counting (only if text_boxes is a list)
    all_boxes = []
    if barcode_boxes:  # Check if barcode_boxes is not None/empty
        all_boxes.extend([b["rect"] for b in barcode_boxes])
    if text_boxes:  # Check if text_boxes is not None/empty
        all_boxes.extend(text_boxes)

    # Calculate encompassing box
    encompassing_box = None
    if all_boxes:
        min_x = min(box[0] for box in all_boxes)
        min_y = min(box[1] for box in all_boxes)
        max_x_w = max(box[0] + box[2] for box in all_boxes)
        max_y_h = max(box[1] + box[3] for box in all_boxes)
        enc_w = max_x_w - min_x
        enc_h = max_y_h - min_y
        encompassing_box = (min_x, min_y, enc_w, enc_h)
        print(f"\nCalculated encompassing box: {encompassing_box}")

    print(f"\nTotal identifying boxes found: {len(all_boxes)}")

    # --- Output / Display ---
    draw_start = time.time()
    output_image_generated = False
    if args.output or (all_boxes and not args.hide_window):
        print("Drawing bounding boxes on image...")
        # Pass the barcode_boxes list (contains dicts) and text_boxes list to draw_boxes
        output_image = draw_boxes(image, barcode_boxes, text_boxes, encompassing_box)
        output_image_generated = True
    else:
        output_image = None  # No drawing needed
    draw_duration = time.time() - draw_start
    if output_image_generated:
        print(f"Drawing boxes took: {draw_duration:.4f} seconds")

    save_display_start = time.time()
    if args.output:
        # Save the image
        try:
            cv2.imwrite(args.output, output_image)
            print(f"Saved output image with boxes to: {args.output}")
        except cv2.error as e:
            print(
                f"Error saving image to {args.output}. OpenCV error: {e}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"An unexpected error occurred saving image to {args.output}: {e}",
                file=sys.stderr,
            )
    elif output_image_generated and not args.hide_window:
        # Display the image if boxes were found, no output path given, and not hidden
        print("Displaying image with detected boxes...")
        display_image(
            "Detected Boxes (Barcode: Green, Text: Red)", output_image
        )  # Update window title
    elif not all_boxes:
        print("No barcodes or text boxes were found to display or save.")
    elif args.hide_window:
        print("Image display skipped due to --hide-window flag.")

    save_display_duration = time.time() - save_display_start
    if args.output or (output_image_generated and not args.hide_window):
        print(f"Saving/Displaying image took: {save_display_duration:.4f} seconds")

    total_duration = time.time() - total_start_time
    print(f"\nTotal script execution time: {total_duration:.4f} seconds")


if __name__ == "__main__":
    main()
