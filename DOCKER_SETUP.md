# Docker Setup for WSI De-identifier

This document summarizes the Docker setup created for the WSI De-identifier project.

## Files Created

1. **Dockerfile**
   - Based on Python 3.10-slim
   - Installs required system dependencies (libzbar0, libdmtx0b, libgl1)
   - Installs Python dependencies including Google Cloud Vision and Gemini
   - Sets up file structure for input/output

2. **run_wsi_identifier.sh**
   - Wrapper script for running the Docker container
   - Handles various command-line options including:
     - Input/output paths
     - Google Cloud credentials
     - Gemini API key for tissue detection
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

- **Google Gemini API**
  - Accepts API key via command-line parameter `--gemini-api-key`
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
   # For Google Cloud Vision
   gcloud auth application-default login
   
   # For Gemini (obtain API key from https://ai.google.dev/)
   ```

3. Run the tool:
   ```bash
   ./run_wsi_identifier.sh \
     --input-path "./sample/macro_images/GP14-5551_A_HE_macro.jpg" \
     --output-path "./sample/output/annotated.jpg" \
     --gemini-api-key "YOUR_GEMINI_API_KEY" \
     --hide-window
   ```

## Notes

- Even without a Gemini API key, the barcode and text detection using Google Cloud Vision will still work
- If neither authentication method is available, basic barcode detection will still function
- For production usage, consider using a service account JSON file instead of application default credentials