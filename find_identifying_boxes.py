import argparse
import sys  # Import sys for exit
import time  # Add time import

import cv2
import easyocr
import numpy as np
from pylibdmtx.pylibdmtx import decode as dmtx_decode
from pyzbar import pyzbar

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


# Initialize EasyOCR Reader (specify languages, e.g., English)
# Doing this globally to load the model only once.
# Use gpu=False if you don't have a compatible GPU or CUDA setup.
# Consider making GPU usage configurable via argparse if needed.
try:
    print("Initializing EasyOCR Reader (this may download models on first run)...")
    reader = easyocr.Reader(["en"], gpu=True)  # Use gpu=False if no CUDA/GPU
    print("EasyOCR Reader initialized.")
except Exception as e:
    print(f"Error initializing EasyOCR Reader: {e}", file=sys.stderr)
    print("Please ensure PyTorch and EasyOCR are installed correctly.", file=sys.stderr)
    # Indicate critical failure - maybe set reader to None and check in find_text_boxes
    reader = None
    # sys.exit(1) # Or exit immediately


def find_text_boxes(image_cv):
    """Finds text regions using EasyOCR."""
    start_time = time.time()
    all_text_boxes = []

    if reader is None:
        print(
            "EasyOCR Reader failed to initialize. Skipping text detection.",
            file=sys.stderr,
        )
        return []  # Return empty list if reader couldn't be created

    # EasyOCR works directly with NumPy arrays (BGR format is fine)
    # No need for PIL conversion or rotations
    print("  Running EasyOCR text detection...")
    try:
        # detail=0 gives only bounding boxes and text
        # detail=1 gives boxes, text, and confidence
        # paragraph=True might help group text lines, but False gives individual word boxes
        # Set batch_size based on your GPU memory if using GPU
        ocr_results = reader.readtext(image_cv, detail=1, paragraph=False)
        # Example result item: ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], 'text', confidence_score)

    except Exception as e:
        print(f"An error occurred during EasyOCR processing: {e}", file=sys.stderr)
        return []  # Return empty on error

    ocr_duration = time.time() - start_time
    print(f"  EasyOCR detection took: {ocr_duration:.4f} seconds")

    min_confidence = 0.2  # EasyOCR confidence threshold (adjust as needed)

    debug_boxes_raw = []  # For debug image
    for bbox, text, confidence in ocr_results:
        if confidence >= min_confidence:
            # bbox is a list of 4 points: [top_left, top_right, bottom_right, bottom_left]
            # Extract coordinates for drawing and processing
            top_left = tuple(map(int, bbox[0]))
            top_right = tuple(map(int, bbox[1]))  # Needed for drawing polygon
            bottom_right = tuple(map(int, bbox[2]))
            bottom_left = tuple(map(int, bbox[3]))  # Needed for drawing polygon

            # Calculate (x, y, w, h) format for return value
            x = top_left[0]
            y = top_left[1]
            w = bottom_right[0] - top_left[0]
            h = bottom_right[1] - top_left[1]

            # Basic filtering (ensure width and height are positive)
            if w > 0 and h > 0:
                all_text_boxes.append((x, y, w, h))
                # Add points for drawing polygon on debug image
                debug_boxes_raw.append(
                    np.array(
                        [top_left, top_right, bottom_right, bottom_left], dtype=np.int32
                    )
                )
                # Optional: Print detected text info
                # print(f"    - Text: '{text}' at {(x, y, w, h)}, Conf: {confidence:.4f}")

    # --- Debug: Save image with raw EasyOCR boxes ---
    if debug_boxes_raw:
        try:
            debug_img = image_cv.copy()
            # Draw raw boxes (Polygons, Blue for distinction)
            cv2.polylines(
                debug_img,
                debug_boxes_raw,
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
            # Alternative: Draw bounding rectangles using the calculated (x,y,w,h)
            # for x_d, y_d, w_d, h_d in all_text_boxes:
            #     cv2.rectangle(debug_img, (x_d, y_d), (x_d + w_d, y_d + h_d), (255, 0, 0), 2)

            debug_filename = "debug_easyocr_boxes.png"
            cv2.imwrite(debug_filename, debug_img)
            print(f"    [Debug] Saved image with EasyOCR boxes to: {debug_filename}")
        except Exception as e:
            print(
                f"    [Debug] Error saving debug image: {e}",
                file=sys.stderr,
            )
    # --- End Debug ---

    print(f"  Found {len(all_text_boxes)} text boxes meeting confidence threshold.")

    total_duration = time.time() - start_time
    print(f"Total text box finding took: {total_duration:.4f} seconds")
    # No need for unique list conversion as rotation isn't used
    return all_text_boxes


def draw_boxes(image, barcode_boxes, text_boxes):
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
    print("Finding text boxes (using EasyOCR)...")
    text_start = time.time()
    # Now returns only a list of boxes
    text_boxes = find_text_boxes(image)
    text_duration = time.time() - text_start

    # Simplified error check (find_text_boxes returns [] on init or processing error)
    if not isinstance(text_boxes, list):
        # This case shouldn't happen with current logic, but as a safeguard
        print("An unexpected error occurred during text detection.", file=sys.stderr)
        sys.exit(1)

    # Proceed with reporting text box findings
    if text_boxes:
        print(
            f"Found {len(text_boxes)} potential text box(es). (Detection time: {text_duration:.4f}s)"
        )
        print("Text Bounding Boxes (x, y, width, height):")
        for box in text_boxes:
            print(f"  - {box}")
    else:
        # Adjust message slightly if it could be an empty list vs None
        if text_boxes is not None:
            print(f"Found 0 text boxes. (Detection time: {text_duration:.4f}s)")
        # The None case (Tesseract error) is handled above

    # Combine boxes for drawing/counting (only if text_boxes is a list)
    all_boxes = []
    if barcode_boxes:  # Check if barcode_boxes is not None/empty
        all_boxes.extend([b["rect"] for b in barcode_boxes])
    if text_boxes:  # Check if text_boxes is not None/empty
        all_boxes.extend(text_boxes)

    # all_boxes = [
    #     b["rect"] for b in barcode_boxes
    # ] + text_boxes # Extract rects for counting/drawing
    print(f"\nTotal identifying boxes found: {len(all_boxes)}")

    # --- Output / Display ---
    draw_start = time.time()
    output_image_generated = False
    if args.output or (all_boxes and not args.hide_window):
        print("Drawing bounding boxes on image...")
        # Pass the barcode_boxes list (contains dicts) to draw_boxes
        output_image = draw_boxes(image, barcode_boxes, text_boxes)
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
