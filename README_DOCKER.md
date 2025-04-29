# WSI De-identifier Docker Usage

This document describes how to use the WSI De-identifier tools in a Docker container.

## Prerequisites

- Docker installed and running on your system
- Google Cloud Platform credentials for text detection and tissue detection

## Docker Image

The included Dockerfile creates an environment with all necessary dependencies for the WSI De-identifier tools, including:

- Python 3.10
- OpenCV, Pillow, and other image processing libraries
- Google Cloud Vision and Vertex AI libraries
- QR code and barcode detection libraries (ZBar, libdmtx)
- All Python dependencies installed via uv from pyproject.toml

## Building the Docker Image

```bash
docker build -t wsi-deidentifier:latest .
```

## Using the Wrapper Script

For convenience, a wrapper script (`run_wsi_identifier.sh`) is provided to simplify using the Docker container.

### Basic Usage

```bash
# Build and run in one command
./run_wsi_identifier.sh --build --input-path "./sample/macro_images/test_macro_image.jpg" --output-path "./sample/output.jpg"

# Process multiple images with a glob pattern
./run_wsi_identifier.sh --input-path "./sample/macro_images/*.jpg" --output-path "./sample/macro_images_annotated/"

# Run with hide-window option (for headless environments)
./run_wsi_identifier.sh --input-path "./sample/macro_images/*.jpg" --output-path "./sample/macro_images_annotated/" --hide-window

# Specify Google Cloud project for tissue detection
./run_wsi_identifier.sh --project "your-gcp-project-id" --input-path "./sample/macro_images/*.jpg" --output-path "./sample/output/"
```

### Options

- `--build`: Build the Docker image before running
- `--input-path`: Path to input image(s), can be a glob pattern
- `--output-path`: Path to output directory or file
- `--hide-window`: Hide the image window (useful in headless environments)
- `--project`: Google Cloud project ID for Vertex AI (tissue detection)
- `--location`: Google Cloud location for Vertex AI (default: us-central1)
- `--help`: Display help message

## Authentication & Environment Variables

### Google Cloud Authentication

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

### Google Cloud Project and Location

For tissue detection using Vertex AI (Gemini), you must specify a Google Cloud project:

1. **Using command line arguments**:

   ```bash
   ./run_wsi_identifier.sh --project "your-gcp-project-id" --location "us-central1" --input-path "./sample/macro_images/*.jpg"
   ```

2. **Using environment variables**:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
   export GOOGLE_CLOUD_LOCATION="us-central1"  # Optional, defaults to us-central1
   ./run_wsi_identifier.sh --input-path "./sample/macro_images/*.jpg"
   ```

If no project is specified, tissue detection will not work, but text and barcode detection will still function.

## Running the Docker Container Directly

If you prefer not to use the wrapper script, you can run the Docker container directly:

```bash
# Mount local directories and run the container
docker run --rm \
  -v /path/to/credentials.json:/tmp/gcp-credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json \
  -e GOOGLE_CLOUD_PROJECT="your-gcp-project-id" \
  -e GOOGLE_CLOUD_LOCATION="us-central1" \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  wsi-deidentifier:latest \
  "/data/input/image.jpg" --output "/data/output/annotated.jpg" --project "your-gcp-project-id" --location "us-central1"
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
