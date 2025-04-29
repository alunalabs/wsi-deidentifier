#!/bin/bash

# Exit on error
set -e

# Function to display help message
show_help() {
  echo "WSI De-identifier Find Identifying Boxes Script"
  echo "----------------------------------------------"
  echo "This script helps identify text and barcodes in WSI macro images."
  echo
  echo "Usage:"
  echo "  ./run_wsi_identifier.sh [OPTIONS]"
  echo
  echo "Options:"
  echo "  --input-path PATH      Path to input image files (can be glob pattern)"
  echo "  --output-path PATH     Path to output directory or file"
  echo "  --help                 Display this help message and exit"
  echo "  --build                Build the Docker image before running"
  echo "  --hide-window          Hide the image window (useful in headless environments)"
  echo "  --project PROJECT_ID   Google Cloud project ID for Vertex AI (Gemini)"
  echo "  --location LOCATION    Google Cloud location for Vertex AI (default: us-central1)"
  echo
  echo "Examples:"
  echo "  ./run_wsi_identifier.sh --input-path './sample/macro_images/*.jpg' --output-path './sample/macro_images_annotated/'"
  echo "  ./run_wsi_identifier.sh --build --input-path './sample/macro_images/GP14-5551_A_HE_macro.jpg' --output-path './sample/output.jpg'"
  echo "  ./run_wsi_identifier.sh --project 'my-gcp-project' --input-path './sample/macro_images/*.jpg' --output-path './sample/output/'"
  echo
  echo "Note: GCP authentication is required. Make sure to set GOOGLE_APPLICATION_CREDENTIALS"
  echo "      or run 'gcloud auth application-default login' before using this script."
  echo "      For tissue detection, set GOOGLE_CLOUD_PROJECT in your environment or use --project."
}

# Default values
BUILD_IMAGE=false
HIDE_WINDOW=false
INPUT_PATH=""
OUTPUT_PATH=""
PROJECT=""
LOCATION=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      show_help
      exit 0
      ;;
    --build)
      BUILD_IMAGE=true
      shift
      ;;
    --input-path)
      INPUT_PATH="$2"
      shift 2
      ;;
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --hide-window)
      HIDE_WINDOW=true
      shift
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --location)
      LOCATION="$2"
      shift 2
      ;;
    *)
      echo "Error: Unknown option $1"
      show_help
      exit 1
      ;;
  esac
done

# Check for required arguments
if [ -z "$INPUT_PATH" ]; then
  echo "Error: Input path is required"
  show_help
  exit 1
fi

# Build Docker image if requested
if [ "$BUILD_IMAGE" = true ]; then
  echo "Building Docker image..."
  docker build -t wsi-deidentifier:latest .
fi

# Prepare Docker command
DOCKER_CMD="docker run --rm"

# Add Google Cloud Project and Location if provided
if [ ! -z "$PROJECT" ]; then
  DOCKER_CMD="$DOCKER_CMD -e GOOGLE_CLOUD_PROJECT=\"$PROJECT\""
  echo "Using provided Google Cloud project: $PROJECT"
elif [ ! -z "$GOOGLE_CLOUD_PROJECT" ]; then
  DOCKER_CMD="$DOCKER_CMD -e GOOGLE_CLOUD_PROJECT=\"$GOOGLE_CLOUD_PROJECT\""
  echo "Using Google Cloud project from environment: $GOOGLE_CLOUD_PROJECT"
else
  echo "Warning: No Google Cloud project specified. Tissue detection may not work."
  echo "To enable tissue detection, set GOOGLE_CLOUD_PROJECT or use --project"
fi

if [ ! -z "$LOCATION" ]; then
  DOCKER_CMD="$DOCKER_CMD -e GOOGLE_CLOUD_LOCATION=\"$LOCATION\""
  echo "Using provided Google Cloud location: $LOCATION"
elif [ ! -z "$GOOGLE_CLOUD_LOCATION" ]; then
  DOCKER_CMD="$DOCKER_CMD -e GOOGLE_CLOUD_LOCATION=\"$GOOGLE_CLOUD_LOCATION\""
  echo "Using Google Cloud location from environment: $GOOGLE_CLOUD_LOCATION"
else
  echo "Using default Google Cloud location: us-central1"
fi

# Mount the GCP credentials if they exist
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  # Service account credentials
  CREDS_PATH="$GOOGLE_APPLICATION_CREDENTIALS"
  DOCKER_CMD="$DOCKER_CMD -v $CREDS_PATH:/tmp/gcp-credentials.json -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json"
  echo "Using service account credentials from GOOGLE_APPLICATION_CREDENTIALS"
elif [ -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
  # Application default credentials
  mkdir -p /tmp/gcloud_config
  cp -r $HOME/.config/gcloud /tmp/gcloud_config/
  DOCKER_CMD="$DOCKER_CMD -v /tmp/gcloud_config/gcloud:/root/.config/gcloud:ro"
  echo "Using application default credentials from gcloud"
else
  echo "Warning: No Google Cloud credentials found. Text detection using GCP Vision API will not work."
  echo "Run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS."
fi

# Determine the input directory to mount
INPUT_DIR=$(dirname "$INPUT_PATH")
INPUT_BASE=$(basename "$INPUT_PATH")
DOCKER_CMD="$DOCKER_CMD -v $INPUT_DIR:/data/input"

# Handle output path
if [ ! -z "$OUTPUT_PATH" ]; then
  # Determine if output is a directory or file
  if [[ "$OUTPUT_PATH" == */ ]] || [ -d "$OUTPUT_PATH" ]; then
    # It's a directory
    OUTPUT_DIR="$OUTPUT_PATH"
    OUTPUT_ARG="--output /data/output"
  else
    # It's a file
    OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
    OUTPUT_FILE=$(basename "$OUTPUT_PATH")
    OUTPUT_ARG="--output /data/output/$OUTPUT_FILE"
  fi
  
  mkdir -p "$OUTPUT_DIR"
  DOCKER_CMD="$DOCKER_CMD -v $OUTPUT_DIR:/data/output"
else
  # No output path provided - create a temporary output directory
  TEMP_OUTPUT_DIR="$(pwd)/tmp_docker_output"
  mkdir -p "$TEMP_OUTPUT_DIR"
  DOCKER_CMD="$DOCKER_CMD -v $TEMP_OUTPUT_DIR:/data/output"
  OUTPUT_ARG=""
  echo "No output path specified. Results will not be saved."
fi

# Add hide-window flag if needed
if [ "$HIDE_WINDOW" = true ]; then
  OUTPUT_ARG="$OUTPUT_ARG --hide-window"
fi

# Add project and location to command if specified
if [ ! -z "$PROJECT" ]; then
  OUTPUT_ARG="$OUTPUT_ARG --project \"$PROJECT\""
fi

if [ ! -z "$LOCATION" ]; then
  OUTPUT_ARG="$OUTPUT_ARG --location \"$LOCATION\""
fi

# Run the Docker container
echo "Running WSI identifier..."
$DOCKER_CMD wsi-deidentifier:latest "/data/input/$INPUT_BASE" $OUTPUT_ARG

# Clean up any temporary directories
if [ -d "/tmp/gcloud_config" ]; then
  rm -rf /tmp/gcloud_config
fi

echo "Done!"