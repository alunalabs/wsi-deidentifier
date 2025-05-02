#!/bin/bash

set -e

IMAGE_NAME="wsi-deidentifier"
IMAGE_TAG="latest"

function print_help {
  echo "WSI-Deidentifier Docker Helper"
  echo ""
  echo "Usage: ./start.sh [command] [options]"
  echo ""
  echo "Commands:"
  echo "  build                     Build the Docker image"
  echo "  run                       Run the Docker container"
  echo "  help                      Show this help message"
  echo ""
  echo "Options for 'run':"
  echo "  --fastapi-port=PORT       Port for FastAPI server (default: 8000)"
  echo "  --nextjs-port=PORT        Port for Next.js server (default: 3000)" 
  echo "  --slide-pattern=PATTERN   Glob pattern for slides (default: sample/identified/*.{svs,tif,tiff})"
  echo "  --json-path=PATH          Path to store JSON (default: boxes.json)"
  echo "  --deidentified-dir=DIR    Directory for deidentified images (default: deidentified)"
  echo "  --credentials=PATH        Path to Google Cloud credentials file"
  echo "  -v, --volume=SRC:DEST     Add a volume mapping (can be used multiple times)"
  echo "  -d, --detach              Run container in detached mode"
  echo ""
  echo "Examples:"
  echo "  ./start.sh build          Build the Docker image"
  echo "  ./start.sh run            Run with default settings"
  echo "  ./start.sh run --fastapi-port=9000 --nextjs-port=4000"
  echo "  ./start.sh run --credentials=/path/to/credentials.json"
  echo "  ./start.sh run -v /path/to/slides:/app/sample/identified"
  echo "  ./start.sh run -d         Run in detached mode"
  echo ""
  echo "Quick Start Example:"
  echo "  ./start.sh build && ./start.sh run -v $(pwd)/sample/identified:/app/sample/identified -v $(pwd)/deidentified:/app/deidentified"
}

function build_image {
  echo "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
  docker build -t "$IMAGE_NAME:$IMAGE_TAG" .
  echo "Build complete"
}

function run_container {
  # Default values
  local fastapi_port=8000
  local nextjs_port=3000
  local slide_pattern="sample/identified/*.{svs,tif,tiff}"
  local json_path="boxes.json"
  local deidentified_dir="deidentified"
  local credentials=""
  local volumes=()
  local add_sample_volume=true
  local add_deidentified_volume=true
  local detach_mode=false

  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --fastapi-port=*)
        fastapi_port="${1#*=}"
        shift
        ;;
      --nextjs-port=*)
        nextjs_port="${1#*=}"
        shift
        ;;
      --slide-pattern=*)
        slide_pattern="${1#*=}"
        shift
        ;;
      --json-path=*)
        json_path="${1#*=}"
        shift
        ;;
      --deidentified-dir=*)
        deidentified_dir="${1#*=}"
        shift
        ;;
      --credentials=*)
        credentials="${1#*=}"
        shift
        ;;
      -v=*|--volume=*)
        volumes+=("${1#*=}")
        # Check if the user already mounts the sample/identified directory
        if [[ "${1#*=}" == *":/app/sample/identified"* ]]; then
          add_sample_volume=false
        fi
        # Check if the user already mounts the deidentified directory
        if [[ "${1#*=}" == *":/app/deidentified"* ]]; then
          add_deidentified_volume=false
        fi
        shift
        ;;
      -v|--volume)
        if [[ -n "$2" && "$2" != -* ]]; then
          volumes+=("$2")
          # Check if the user already mounts the sample/identified directory
          if [[ "$2" == *":/app/sample/identified"* ]]; then
            add_sample_volume=false
          fi
          # Check if the user already mounts the deidentified directory
          if [[ "$2" == *":/app/deidentified"* ]]; then
            add_deidentified_volume=false
          fi
          shift 2
        else
          echo "Error: -v/--volume requires an argument"
          exit 1
        fi
        ;;
      -d|--detach)
        detach_mode=true
        shift
        ;;
      *)
        echo "Unknown option: $1"
        print_help
        exit 1
        ;;
    esac
  done

  # Build docker run command
  cmd=("docker" "run")
  
  # Add detach flag if requested
  if [[ "$detach_mode" = true ]]; then
    cmd+=("-d")
  else
    cmd+=("-it")
  fi
  
  cmd+=("--rm")
  
  # Add port mappings
  cmd+=("-p" "${fastapi_port}:${fastapi_port}" "-p" "${nextjs_port}:${nextjs_port}")
  
  # Add environment variables
  cmd+=("-e" "FASTAPI_PORT=${fastapi_port}" "-e" "NEXTJS_PORT=${nextjs_port}")
  cmd+=("-e" "SLIDE_PATTERN=${slide_pattern}" "-e" "PERSIST_JSON_PATH=${json_path}")
  cmd+=("-e" "DEIDENTIFIED_DIR=${deidentified_dir}")
  
  # Add credentials if provided
  if [[ -n "$credentials" ]]; then
    if [[ -f "$credentials" ]]; then
      cmd+=("-e" "GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json")
      cmd+=("-v" "${credentials}:/app/credentials.json:ro")
    else
      echo "Warning: Credentials file not found at ${credentials}"
    fi
  fi
  
  # Add volumes
  for vol in "${volumes[@]}"; do
    cmd+=("-v" "$vol")
  done
  
  # Add default sample/identified volume if not already mounted
  if [[ "$add_sample_volume" = true ]]; then
    local sample_dir="$(pwd)/sample/identified"
    if [[ -d "$sample_dir" ]]; then
      echo "Adding default volume mapping for slides: $sample_dir:/app/sample/identified"
      cmd+=("-v" "$sample_dir:/app/sample/identified")
    else
      echo "Warning: Default sample slides directory not found at $sample_dir."
      echo "No slides will be available unless you've specified a custom volume."
    fi
  fi

  # Add default deidentified volume if not already mounted
  if [[ "$add_deidentified_volume" = true ]]; then
    local deidentified_path="$(pwd)/deidentified"
    # Create the directory if it doesn't exist
    mkdir -p "$deidentified_path"
    echo "Adding default volume mapping for deidentified slides: $deidentified_path:/app/deidentified"
    cmd+=("-v" "$deidentified_path:/app/deidentified")
  fi
  
  # Add image name
  cmd+=("$IMAGE_NAME:$IMAGE_TAG")
  
  # Execute the command
  echo "Starting container with the following configuration:"
  echo "  FastAPI Port: ${fastapi_port}"
  echo "  Next.js Port: ${nextjs_port}"
  echo "  Slide Pattern: ${slide_pattern}"
  echo "  JSON Path: ${json_path}"
  echo "  Deidentified Directory: ${deidentified_dir}"
  if [[ -n "$credentials" ]]; then
    echo "  Credentials: ${credentials}"
  fi
  if [[ "$detach_mode" = true ]]; then
    echo "  Mode: Detached"
  fi
  echo ""
  
  echo "Running command: ${cmd[*]}"
  "${cmd[@]}"
  
  # Print helpful information if detached
  if [[ "$detach_mode" = true ]]; then
    echo ""
    echo "Container started in detached mode."
    echo "Access FastAPI at: http://localhost:${fastapi_port}"
    echo "Access Next.js at: http://localhost:${nextjs_port}"
    echo ""
    echo "To view logs: docker logs $(docker ps -q -f ancestor=${IMAGE_NAME}:${IMAGE_TAG} -n 1)"
    echo "To stop container: docker stop $(docker ps -q -f ancestor=${IMAGE_NAME}:${IMAGE_TAG} -n 1)"
  fi
}

# Main script logic
case "$1" in
  build)
    build_image
    ;;
  run)
    shift
    run_container "$@"
    ;;
  help|--help|-h)
    print_help
    ;;
  *)
    print_help
    exit 1
    ;;
esac

exit 0