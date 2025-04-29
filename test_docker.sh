#!/bin/bash

# Exit on error
set -e

echo "Building Docker image..."
docker build -t wsi-deidentifier:latest .

echo "Creating output directory..."
mkdir -p ./sample/docker_test_output

echo "Running Docker container with test script..."
docker run --rm \
  -v "$(pwd)/sample/macro_images:/data/input" \
  -v "$(pwd)/sample/docker_test_output:/data/output" \
  wsi-deidentifier:latest \
  "/data/input/GP14-5551_A_HE_macro.jpg" --output "/data/output/test_result.jpg"

echo "Checking if output file was created..."
if [ -f "./sample/docker_test_output/test_result.jpg" ]; then
  echo "✅ Test successful! Output file was created."
  echo "Output file: ./sample/docker_test_output/test_result.jpg"
else
  echo "❌ Test failed! No output file was created."
  exit 1
fi