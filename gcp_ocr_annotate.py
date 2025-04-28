import argparse
import io
import os

from google.cloud import vision
from PIL import Image, ImageDraw


def detect_text(path):
    """Detects text in the file using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    print(f"Sending request to Google Cloud Vision API for {path}...")
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Received response from API.")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return texts


def draw_boxes(image_path, texts, output_path):
    """Draws bounding boxes on the image."""
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        print(f"Drawing {len(texts) - 1} bounding boxes...")
        # Skip the first annotation which is the full text block
        for text in texts[1:]:
            vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
            # Draw polygon to handle potentially skewed boxes
            draw.polygon(vertices, outline="red", width=3)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(f"Output directory exists: {output_dir}")

        # Save the annotated image
        image.save(output_path)
        print(f"Saved annotated image to {output_path}")

    except Exception as e:
        print(f"Error drawing boxes or saving image: {e}")


def main(image_path, output_dir):
    """Main function to detect text and draw boxes."""
    if not os.path.isfile(image_path):
        print(f"Error: Input file not found at {image_path}")
        return

    print(f"Processing {image_path}...")
    try:
        texts = detect_text(image_path)
        if not texts:
            print("No text detected.")
            return

        # Construct output filename
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        # Ensure extension is .jpg or similar image format if needed, but keep original for now
        output_filename = f"{name}_gcp_annotated{ext}"
        output_path = os.path.join(output_dir, output_filename)

        draw_boxes(image_path, texts, output_path)

    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    # Check if authentication is set up - provide a warning if not obvious
    creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    default_creds_path = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )
    if not creds_env and not os.path.exists(default_creds_path):
        print("Warning: Google Cloud authentication credentials not found.")
        print(
            "Ensure you have run 'gcloud auth application-default login' or set the GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )
        # Optionally exit if credentials are strictly required, but let it try for now
        # sys.exit(1)
    elif creds_env:
        print(f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {creds_env}")
    else:
        print(f"Using application default credentials found at: {default_creds_path}")

    parser = argparse.ArgumentParser(
        description="Detect text in an image using Google Cloud Vision API and draw bounding boxes."
    )
    parser.add_argument("image_path", help="Path to the input JPG image file.")
    parser.add_argument(
        "--output_dir",
        default="./sample/gcp_output",
        help="Directory to save the annotated image (default: ./sample/gcp_output).",
    )

    args = parser.parse_args()

    main(args.image_path, args.output_dir)
