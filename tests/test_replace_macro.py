import glob
import os
import sys

import pytest

try:
    import openslide  # noqa: F401
except Exception:
    pytest.skip("openslide not available", allow_module_level=True)

# Add the project root to the Python path to allow importing replace_macro
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import replace_macro  # noqa E402

# --- Test Configuration ---
TEST_INPUT_DIR = os.path.join(project_root, "sample", "identified")
TEST_FILES = glob.glob(os.path.join(TEST_INPUT_DIR, "*.svs")) + glob.glob(
    os.path.join(TEST_INPUT_DIR, "*.tif")
)
TEST_RECT_COORDS = (0, 0, 100, 100)  # Top-left 100x100 square
TEST_FILL_COLOR = (255, 165, 0)  # Orange
COLOR_TOLERANCE = 20  # Allow difference due to compression
MACRO_DESCRIPTION = "macro"  # Default description used by the script


# --- Helper Function ---
def check_color_similarity(color1, color2, tolerance):
    """Checks if two RGB colors are similar within a given tolerance."""
    return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))


# --- Test Function ---
@pytest.mark.parametrize("input_svs_path", TEST_FILES)
def test_replace_macro_with_orange_square(input_svs_path, tmp_path):
    """
    Tests if replace_macro correctly adds an orange square to the macro image.
    """
    test_output_filename = f"test_output_{os.path.basename(input_svs_path)}"
    output_svs_path = tmp_path / test_output_filename

    print(f"Testing file: {input_svs_path}")
    print(f"Output path: {output_svs_path}")

    # Run the replace_macro function
    replace_macro.replace_macro(
        input_path=input_svs_path,
        output_path=str(output_svs_path),
        rect_coords=TEST_RECT_COORDS,
        fill_color=TEST_FILL_COLOR,
        macro_description=MACRO_DESCRIPTION,  # Ensure we target the correct image
        verbose=True,  # Enable verbose logging for debugging if needed
    )

    # Verify the output file was created
    assert output_svs_path.exists(), f"Output file not created: {output_svs_path}"

    # Verify the macro image was modified using openslide
    try:
        with openslide.OpenSlide(str(output_svs_path)) as slide:
            # Check if the specific macro image exists (based on description matching)
            macro_img = None
            for key, img in slide.associated_images.items():
                # Attempt to read description from TIFF tags if possible
                # This is complex with openslide alone, rely on name matching for now
                # print(f"Checking associated image: {key}")
                if MACRO_DESCRIPTION.lower() in key.lower():
                    macro_img = img
                    break

            assert macro_img is not None, (
                f"Macro image with description '{MACRO_DESCRIPTION}' not found in {output_svs_path}"
            )

            macro_img_rgb = macro_img.convert("RGB")

            # Check the color of a pixel inside the drawn rectangle
            # Using (50, 50) as it's the center of the 100x100 square
            center_x, center_y = 50, 50
            pixel_color = macro_img_rgb.getpixel((center_x, center_y))

            print(f"Pixel color at ({center_x},{center_y}): {pixel_color}")
            print(f"Expected color: {TEST_FILL_COLOR}")

            assert check_color_similarity(
                pixel_color, TEST_FILL_COLOR, COLOR_TOLERANCE
            ), (
                f"Pixel color {pixel_color} at ({center_x},{center_y}) is not close enough to expected {TEST_FILL_COLOR} (Tolerance: {COLOR_TOLERANCE})"
            )

            print(f"Test passed for {input_svs_path}")

    except openslide.OpenSlideError as e:
        pytest.fail(
            f"Failed to open output SVS file {output_svs_path} with openslide: {e}"
        )
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during verification: {e}")


# Optional: Add a check if no test files were found
if not TEST_FILES:
    print(f"Warning: No test files found in {TEST_INPUT_DIR}. Skipping tests.")
    # You might want to raise an error or handle this case appropriately
    # pytest.skip("No test files found", allow_module_level=True)
