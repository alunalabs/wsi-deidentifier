# WSI De-identifier Docker Usage

This document describes how to use the WSI De-identifier tools in a Docker container.

## Prerequisites

- Docker installed and running on your system
- Google Cloud Platform credentials for text detection (if using GCP Vision features)

## Docker Image

The included Dockerfile creates an environment with all necessary dependencies for the WSI De-identifier tools, including:

- Python 3.10
- OpenCV, Pillow, and other image processing libraries
- Google Cloud Vision libraries
- QR code and barcode detection libraries (ZBar, libdmtx)
- All Python dependencies specified in pyproject.toml

## Building the Docker Image

```bash
docker build -t wsi-deidentifier:latest .
```

## Using the Wrapper Script

For convenience, a wrapper script (`run_wsi_identifier.sh`) is provided to simplify using the Docker container.

### Basic Usage

```bash
# Build and run in one command
./run_wsi_identifier.sh --build --input-path "./sample/macro_images/GP14-5551_A_HE_macro.jpg" --output-path "./sample/output.jpg"

# Process multiple images with a glob pattern
./run_wsi_identifier.sh --input-path "./sample/macro_images/*.jpg" --output-path "./sample/macro_images_annotated/"

# Run with hide-window option (for headless environments)
./run_wsi_identifier.sh --input-path "./sample/macro_images/*.jpg" --output-path "./sample/macro_images_annotated/" --hide-window
```

### Options

- `--build`: Build the Docker image before running
- `--input-path`: Path to input image(s), can be a glob pattern
- `--output-path`: Path to output directory or file
- `--hide-window`: Hide the image window (useful in headless environments)
- `--help`: Display help message

## Authentication & API Keys

## Google Cloud Authentication

The script automatically mounts Google Cloud credentials if they are available. For text detection functionality, set up authentication before using the container:

1. **Application Default Credentials** (recommended):
   ```bash
   gcloud auth application-default login
   ```
   This will create credentials at `~/.config/gcloud/application_default_credentials.json` which will be automatically mounted by the script.

2. **Service Account JSON**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```
   The script will mount this file into the container.

If no credentials are provided, the GCP Vision API features will not work, but basic barcode detection will still function.

## Google Gemini API Key

For tissue detection functionality, you need to provide a Google Gemini API key:

```bash
./run_wsi_identifier.sh --gemini-api-key "YOUR_API_KEY" --input-path "./sample/macro_images/*.jpg" --output-path "./output/"
```

You can obtain a Gemini API key from the [Google AI Studio](https://ai.google.dev/) or Google Cloud Console.

If no Gemini API key is provided, tissue detection will not work, but text and barcode detection will still function.

## Running the Docker Container Directly

If you prefer not to use the wrapper script, you can run the Docker container directly:

```bash
# Mount local directories and run the container
docker run --rm \
  -v /path/to/credentials.json:/tmp/gcp-credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json \
  -e GEMINI_API_KEY="your-gemini-api-key" \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  wsi-deidentifier:latest \
  "/data/input/image.jpg" --output "/data/output/annotated.jpg"
```

## Working with Different Tools

The Docker image contains all scripts from the WSI De-identifier project. To use a different script than the default `find_identifying_boxes.py`, you can override the entrypoint:

```bash
docker run --rm \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  --entrypoint python \
  wsi-deidentifier:latest \
  deidentify.py "/data/input/*.svs" --salt "your-secret-salt" -o /data/output
```