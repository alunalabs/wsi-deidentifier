"""
Opens a Whole Slide Image (WSI) file using openslide, extracts the associated
macro image, and displays it in a GUI window using OpenCV.

Usage:
    python view_macro_image.py <path_to_wsi_file>
"""

import argparse
import sys

import cv2
import numpy as np
import openslide


def view_macro(wsi_path):
    """
    Opens a WSI file, extracts and displays the macro image.

    Args:
        wsi_path (str): Path to the WSI file.
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
    except openslide.OpenSlideError as e:
        print(f"Error opening WSI file '{wsi_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found at '{wsi_path}'", file=sys.stderr)
        sys.exit(1)

    if "macro" not in slide.associated_images:
        print(f"No 'macro' associated image found in '{wsi_path}'.", file=sys.stderr)
        # Optionally list available associated images
        available_images = list(slide.associated_images.keys())
        if available_images:
            print(
                f"Available associated images: {', '.join(available_images)}",
                file=sys.stderr,
            )
        else:
            print("No associated images found.", file=sys.stderr)
        slide.close()
        sys.exit(1)

    try:
        macro_image_pil = slide.associated_images["macro"]
        # Convert PIL image to NumPy array (OpenCV format - BGR)
        macro_image_np = np.array(macro_image_pil)
        # Check if the image has an alpha channel and remove it if necessary
        if macro_image_np.shape[2] == 4:
            macro_image_np = cv2.cvtColor(macro_image_np, cv2.COLOR_RGBA2BGR)
        else:
            macro_image_np = cv2.cvtColor(macro_image_np, cv2.COLOR_RGB2BGR)

        # Display the image
        window_name = f"Macro Image - {wsi_path}"
        cv2.imshow(window_name, macro_image_np)
        print(f"Displaying macro image from {wsi_path}. Press any key to close.")
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing or displaying macro image: {e}", file=sys.stderr)
    finally:
        slide.close()  # Ensure the slide is closed


def main():
    parser = argparse.ArgumentParser(
        description="""Opens a WSI file, extracts the associated macro image,
                       and displays it using OpenCV."""
    )
    parser.add_argument(
        "wsi_file", help="Path to the Whole Slide Image file (e.g., .svs, .tif)."
    )
    args = parser.parse_args()

    view_macro(args.wsi_file)


if __name__ == "__main__":
    main()
