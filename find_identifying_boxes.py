import argparse
import sys  # Import sys for exit

import cv2
import pytesseract
from PIL import Image
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
    barcode_boxes = []

    # --- Use pyzbar ---
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

    # --- Use pylibdmtx ---
    try:
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

    return barcode_boxes


def find_text_boxes(image_cv):
    """Finds text regions in an image using Tesseract and returns their bounding boxes."""
    # Convert OpenCV image (BGR) to PIL image (RGB)
    try:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
    except cv2.error as e:
        print(f"Error converting image for Tesseract: {e}", file=sys.stderr)
        return []

    # Use image_to_data to get bounding box information for each word
    # Config options:
    # --oem 3: Use default OCR Engine mode (based on what's available/installed)
    # --psm 3: Fully automatic page segmentation (default)
    # You might experiment with other psm values (e.g., 6 for assuming a single uniform block of text)
    # if default results are poor. See `tesseract --help-psm` for details.
    custom_config = r"--oem 3 --psm 3"
    try:
        # Request dictionary output
        data = pytesseract.image_to_data(
            pil_img, output_type=pytesseract.Output.DICT, config=custom_config
        )
    except pytesseract.TesseractNotFoundError:
        print(
            "\nError: Tesseract OCR is not installed or not found in your system's PATH.",
            file=sys.stderr,
        )
        print(
            "Please install Tesseract for your OS: https://github.com/tesseract-ocr/tesseract#installing-tesseract",
            file=sys.stderr,
        )
        print(
            "You may also need to set the TESSERACT_CMD environment variable or use the --tesseract-path argument.",
            file=sys.stderr,
        )
        # Return None to indicate a critical error preventing text detection
        return None
    except Exception as e:
        print(f"An error occurred during Tesseract processing: {e}", file=sys.stderr)
        return []  # Return empty list for non-critical errors

    text_boxes = []
    n_boxes = len(data["level"])
    min_confidence = 60  # Minimum confidence score to consider a detected word valid

    for i in range(n_boxes):
        # Level 5 corresponds to word-level boxes in Tesseract's output
        # Check confidence level. Note: Tesseract returns confidence as string, convert to float/int.
        try:
            confidence = int(float(data["conf"][i]))
        except ValueError:
            confidence = -1  # Handle cases where confidence is not a number

        if data["level"][i] == 5 and confidence > min_confidence:
            text = data["text"][i].strip()
            # Only include boxes with actual text content (not just whitespace)
            if text:
                (x, y, w, h) = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                # Basic filtering: ignore unreasonably small boxes
                if w > 5 and h > 5:
                    text_boxes.append((x, y, w, h))
                    # Optional: print detected text and confidence
                    # print(f"  - Text: '{text}' at {(x, y, w, h)}, Conf: {confidence}")

    return text_boxes


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


def main():
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
        "--tesseract-path",
        help="Optional path to the Tesseract executable if it's not in your system PATH.",
        default=None,
    )
    parser.add_argument(
        "--hide-window",
        action="store_true",
        help="Do not display the image window, even if no output path is specified.",
    )

    args = parser.parse_args()

    # Set Tesseract command path if provided
    if args.tesseract_path:
        try:
            # Basic check if the path seems plausible (optional)
            # import os
            # if not os.path.exists(args.tesseract_path):
            #     print(f"Warning: Provided Tesseract path does not exist: {args.tesseract_path}", file=sys.stderr)
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
        except Exception as e:
            print(f"Error setting Tesseract path: {e}", file=sys.stderr)
            sys.exit(1)  # Exit if setting path fails critically

    # Load the image using OpenCV
    image = cv2.imread(args.image_path)
    if image is None:
        print(
            f"Error: Could not load image from path: {args.image_path}", file=sys.stderr
        )
        sys.exit(1)

    print(f"Processing image: {args.image_path}")

    # --- Barcode Detection ---
    print("Finding barcodes (QR, DataMatrix, etc.)...")
    barcode_boxes = find_barcodes(image)  # Use renamed function
    print(f"Found {len(barcode_boxes)} barcode(s).")
    if barcode_boxes:
        print("Barcode Details:")
        for box_info in barcode_boxes:
            print(
                f"  - Type: {box_info['type']}, Rect: {box_info['rect']}, Data: {box_info['data'][:30]}..."
            )  # Print type and truncated data

    # --- Text Detection ---
    print("Finding text boxes (using Tesseract OCR)...")
    text_boxes = find_text_boxes(
        image
    )  # This handles TesseractNotFoundError internally

    if text_boxes is None:
        # Tesseract not found, main function should exit as text detection failed critically.
        print("Exiting due to Tesseract setup issue.", file=sys.stderr)
        sys.exit(1)
    elif text_boxes:
        print(f"Found {len(text_boxes)} potential text box(es).")
        print("Text Bounding Boxes (x, y, width, height):")
        for box in text_boxes:
            print(f"  - {box}")
    else:
        print("Found 0 text boxes.")

    all_boxes = [
        b["rect"] for b in barcode_boxes
    ] + text_boxes  # Extract rects for counting/drawing
    print(f"\nTotal identifying boxes found: {len(all_boxes)}")

    # --- Output / Display ---
    output_image_generated = False
    if args.output or (all_boxes and not args.hide_window):
        print("Drawing bounding boxes on image...")
        # Pass the barcode_boxes list (contains dicts) to draw_boxes
        output_image = draw_boxes(image, barcode_boxes, text_boxes)
        output_image_generated = True
    else:
        output_image = None  # No drawing needed

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


if __name__ == "__main__":
    main()
