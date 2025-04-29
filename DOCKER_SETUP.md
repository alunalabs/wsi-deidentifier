# Docker Setup for WSI De-identifier

This document summarizes the Docker setup created for the WSI De-identifier project.

## Files Created

1. **Dockerfile**
   - Based on Python 3.10-slim
   - Installs required system dependencies (libzbar0, libdmtx0b, libgl1)
   - Uses uv for dependency management (installed via astral.sh install script)
   - Installs Python dependencies from pyproject.toml via `uv sync`
   - Sets up file structure for input/output

2. **run_wsi_identifier.sh**
   - Wrapper script for running the Docker container
   - Handles various command-line options including:
     - Input/output paths
     - Google Cloud credentials
     - Google Cloud Project and Location for Vertex AI (Gemini)
     - Building the Docker image

3. **README_DOCKER.md**
   - Documentation for using the Docker container
   - Explains authentication options
   - Provides usage examples

## Key Features

### Authentication Support

- **Google Cloud Vision API**
  - Automatically mounts application default credentials or service account JSON
  - Required for text detection in images

- **Google Vertex AI (Gemini)**
  - Sets GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables
  - Passes project and location parameters to find_identifying_boxes.py
  - Required for tissue detection in images

### Usage Flexibility

- Process single images or multiple images using glob patterns
- Output to individual files or directories
- Option to hide display window (useful in headless environments)

## Quick Start

1. Build the Docker image:
   ```bash
   docker build -t wsi-deidentifier:latest .
   ```

2. Setup authentication:
   ```bash
   # For Google Cloud Vision and Vertex AI
   gcloud auth application-default login
   
   # Set required environment variables
   export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
   ```

3. Run the tool:
   ```bash
   ./run_wsi_identifier.sh \
     --input-path "./sample/macro_images/GP14-5551_A_HE_macro.jpg" \
     --output-path "./sample/output/annotated.jpg" \
     --project "your-gcp-project-id" \
     --hide-window
   ```

## Notes

- Even without Vertex AI project configuration, the barcode and text detection using Google Cloud Vision will still work
- If no Google Cloud authentication is available, basic barcode detection will still function
- For production usage, consider using a service account JSON file instead of application default credentials