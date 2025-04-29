#!/bin/bash

# Exit on error
set -e

echo "Building Docker image..."
docker build -t wsi-deidentifier:latest .

echo "Creating output directory..."
mkdir -p ./sample/docker_test_output

# Check for Google Cloud credentials
CREDS_PATH="$HOME/.config/gcloud/application_default_credentials.json"
if [ ! -f "$CREDS_PATH" ]; then
  echo "Error: Google Cloud credentials not found at $CREDS_PATH"
  echo "Run 'gcloud auth application-default login' first"
  exit 1
fi

echo "Running Docker container with Google Cloud credentials..."
docker run --rm \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  -v "$(pwd)/sample/macro_images:/data/input" \
  -v "$(pwd)/sample/docker_test_output:/data/output" \
  wsi-deidentifier:latest \
  "/data/input/GP14-5551_A_HE_macro.jpg" \
  --output "/data/output/gcp_test_result.jpg" \
  --hide-window

echo "Checking if output file was created..."
if [ -f "./sample/docker_test_output/gcp_test_result.jpg" ]; then
  echo "✅ Test successful! Output file was created."
  echo "Output file: ./sample/docker_test_output/gcp_test_result.jpg"
else
  echo "❌ Test failed! No output file was created."
  exit 1
fi