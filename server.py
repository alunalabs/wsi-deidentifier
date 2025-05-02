#!/usr/bin/env python3
import base64
import io
import json
import os
import re
import sys
import tempfile
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import openslide

# Add Azure dependencies
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from replace_macro import replace_macro

# --- Globals / State ---

# Will store Path objects for found slides
slide_paths: List[Path] = []
# Maps filename stem to Path object or Azure blob URL
filename_to_path: Dict[str, Path | str] = {}
# In-memory storage for bounding boxes (filename_stem -> [x0, y0, x1, y1] or [-1,-1,-1,-1] for no-box-needed)
boxes_store: Dict[str, List[int]] = {}
# Path for persisting boxes
PERSIST_JSON_PATH: Path | None = None
# Directory for deidentified slides (can be local path or Azure SAS URL)
DEIDENTIFIED_DIR: Path | str | None = None


# --- Azure Storage Helpers ---


def is_azure_sas_url(url: str) -> bool:
    """Check if a string is an Azure Blob Storage SAS URL."""
    return (
        url.startswith("https://") and "blob.core.windows.net" in url and "sv=" in url
    )


def parse_azure_sas_url(sas_url: str) -> Tuple[str, str, str, str]:
    """
    Parse an Azure SAS URL into its components.
    Returns: (account_name, container_name, blob_path, sas_token)
    """
    parsed_url = urllib.parse.urlparse(sas_url)

    # Extract account name from the hostname
    account_name = parsed_url.netloc.split(".")[0]

    # Extract container name and blob path
    path_parts = parsed_url.path.strip("/").split("/")
    container_name = path_parts[0] if path_parts else ""
    blob_path = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""

    # Extract SAS token (everything after the '?')
    sas_token = parsed_url.query

    return account_name, container_name, blob_path, sas_token


def get_container_client_from_sas(sas_url: str) -> ContainerClient:
    """Get a container client from a SAS URL."""
    account_name, container_name, _, sas_token = parse_azure_sas_url(sas_url)

    # Construct the account URL
    account_url = f"https://{account_name}.blob.core.windows.net"

    # Create a BlobServiceClient using the account URL and SAS token
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=sas_token
    )

    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)

    return container_client


def list_azure_blobs(sas_url: str, pattern: str = "*.{svs,tif,tiff}") -> List[str]:
    """
    List blobs in Azure container matching the given pattern.
    Returns full SAS URLs for each matching blob.
    """
    # Extract parts from the SAS URL
    account_name, container_name, prefix_path, sas_token = parse_azure_sas_url(sas_url)

    # Get container client
    container_client = get_container_client_from_sas(sas_url)

    # Convert glob pattern to regex for filtering
    # This is a simplified version - may need enhancement for complex patterns
    pattern_regex = (
        pattern.replace(".", "\\.")
        .replace("*", ".*")
        .replace("{", "(")
        .replace("}", ")")
        .replace(",", "|")
    )
    regex = re.compile(pattern_regex)

    # List blobs with the prefix
    matching_blobs = []
    try:
        # List all blobs if prefix is empty, otherwise use the prefix
        blobs = container_client.list_blobs(name_starts_with=prefix_path)

        # Filter blobs by pattern and construct full SAS URLs
        base_url = f"https://{account_name}.blob.core.windows.net/{container_name}"

        for blob in blobs:
            # Extract just the filename for pattern matching
            blob_name = blob.name
            if prefix_path:
                # Remove prefix for pattern matching if it exists
                relative_path = (
                    blob_name[len(prefix_path) :]
                    if blob_name.startswith(prefix_path)
                    else blob_name
                )
            else:
                relative_path = blob_name

            if regex.match(relative_path):
                # Construct full SAS URL for the blob
                blob_url = f"{base_url}/{blob_name}?{sas_token}"
                matching_blobs.append(blob_url)

        print(f"Found {len(matching_blobs)} matching blobs in Azure container")
    except Exception as e:
        print(f"Error listing Azure blobs: {e}", file=sys.stderr)

    return matching_blobs


def download_azure_blob(blob_url: str, local_path: str) -> None:
    """Download a blob from Azure to a local path."""
    account_name, container_name, blob_path, sas_token = parse_azure_sas_url(blob_url)

    # Construct the account URL
    account_url = f"https://{account_name}.blob.core.windows.net"

    # Create the blob client
    blob_client = BlobClient(
        account_url=account_url,
        container_name=container_name,
        blob_name=blob_path,
        credential=sas_token,
    )

    # Download the blob
    with open(local_path, "wb") as file:
        blob_data = blob_client.download_blob()
        file.write(blob_data.readall())


def upload_to_azure(local_path: str, destination_sas_url: str, filename: str) -> str:
    """
    Upload a file to Azure Blob Storage using SAS URL.
    Returns the full URL of the uploaded blob.
    """
    account_name, container_name, prefix_path, sas_token = parse_azure_sas_url(
        destination_sas_url
    )

    # Construct the blob name with prefix if any
    blob_name = f"{prefix_path}/{filename}" if prefix_path else filename
    blob_name = blob_name.lstrip("/")  # Ensure no leading slash

    # Construct the account URL
    account_url = f"https://{account_name}.blob.core.windows.net"

    # Create the blob client
    blob_client = BlobClient(
        account_url=account_url,
        container_name=container_name,
        blob_name=blob_name,
        credential=sas_token,
    )

    # Upload the file
    with open(local_path, "rb") as file:
        blob_client.upload_blob(file, overwrite=True)

    # Return the full URL of the uploaded blob
    return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"


def get_filename_from_url(url: str) -> str:
    """Extract the filename from a URL."""
    path = urllib.parse.urlparse(url).path
    return os.path.basename(path)


def get_filenames_from_azure(pattern: str) -> Dict[str, str]:
    """
    Get filenames from Azure blobs that match the pattern.
    Returns a dict of {filename_stem: full_blob_url}
    """
    # Check if the pattern is an Azure SAS URL with a pattern
    if is_azure_sas_url(pattern):
        # For Azure, we need to extract any glob pattern that might be in the URL
        # Simplify by just listing all blobs and filtering client-side
        base_url = pattern.split("*")[0] if "*" in pattern else pattern

        # If URL ends with a path separator, remove it
        base_url = base_url.rstrip("/")

        # Extract the pattern from the original string
        pattern_part = pattern.split("/")[-1] if "/" in pattern else ""
        if not ("*" in pattern_part or "{" in pattern_part):
            # Default to common slide formats if no specific pattern is given
            pattern_part = "*.{svs,tif,tiff}"

        azure_blobs = list_azure_blobs(base_url, pattern_part)
        file_dict = {}

        for blob_url in azure_blobs:
            filename = get_filename_from_url(blob_url)
            filename_stem = os.path.splitext(filename)[0]
            file_dict[filename_stem] = blob_url

        return file_dict

    return {}  # Return empty dict if not an Azure URL


# --- Pydantic Models ---


class SlideListResponse(BaseModel):
    """Response model for listing slide filenames."""

    slides: List[str] = Field(..., description="List of slide filenames (stems) found.")


class BoundingBoxInput(BaseModel):
    """Input model for storing bounding box coordinates."""

    coords: List[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box coordinates [x0, y0, x1, y1].",
    )

    # Removing the validation constraint x0<x1, y0<y1 here, will validate in the endpoint after checking special values
    # @field_validator("coords")
    # @classmethod
    # def validate_coords(cls, v: List[int]) -> List[int]:
    #     if len(v) != 4:
    #         # This check is technically redundant due to min/max_length, but good practice
    #         raise ValueError("Coordinates list must contain exactly 4 integers.")
    #     x0, y0, x1, y1 = v
    #     if not (
    #         isinstance(x0, int)
    #         and isinstance(y0, int)
    #         and isinstance(x1, int)
    #         and isinstance(y1, int)
    #     ):
    #         raise ValueError("All coordinates must be integers.")

    #     # Special case: Allow [0,0,0,0] for box deletion
    #     # Special case: Allow [-1,-1,-1,-1] for "no box needed"
    #     if v == [0, 0, 0, 0] or v == [-1, -1, -1, -1]:
    #         return v

    #     # Validation moved to endpoint logic
    #     # if x0 >= x1 or y0 >= y1:
    #     #     raise ValueError(
    #     #         "Invalid coordinates: x0 must be less than x1, and y0 must be less than y1."
    #     #     )
    #     return v


class BoundingBoxResponse(BaseModel):
    """Response model for retrieving bounding box coordinates."""

    slide_filename: str = Field(..., description="The filename stem of the slide.")
    coords: List[int] = Field(
        default_factory=list,
        min_length=0,  # Allow empty list for deleted boxes or [-1,-1,-1,-1]
        description="Bounding box coordinates [x0, y0, x1, y1]. Empty list if no box is set. [-1,-1,-1,-1] if marked as 'no box needed'.",
    )


class SlideImageResponse(BaseModel):
    """Response model for returning slide image data."""

    slide_filename: str = Field(..., description="The filename stem of the slide.")
    image_data: str = Field(..., description="Base64 encoded PNG image data.")


class DeidentifyResponse(BaseModel):
    """Response model for deidentification operation."""

    slide_filename: str = Field(..., description="The filename stem of the slide.")
    output_path: str = Field(..., description="Path to the deidentified slide.")


class BulkDeidentifyResponse(BaseModel):
    """Response model for bulk deidentification operation."""

    results: List[DeidentifyResponse] = Field(
        ..., description="Results for each processed slide."
    )
    skipped: List[str] = Field(
        ..., description="Slides that were skipped due to errors or missing boxes."
    )


class LabelStatsResponse(BaseModel):
    """Response model for label statistics."""

    total: int = Field(..., description="Total number of slides")
    labeled: int = Field(..., description="Number of slides that have been labeled")
    unlabeled: int = Field(..., description="Number of slides that are not labeled yet")
    no_box_needed: int = Field(
        ..., description="Number of slides marked as not needing a box"
    )


# --- Helper Functions ---


def _find_slides(pattern: str) -> None:
    """Finds slides based on glob pattern or Azure SAS URL and populates global state."""
    global slide_paths, filename_to_path
    print(f"Searching for slides using pattern: {pattern}")

    # Check if pattern is an Azure SAS URL
    if is_azure_sas_url(pattern):
        print("Detected Azure SAS URL. Searching for slides in Azure.")
        azure_files = get_filenames_from_azure(pattern)
        filename_to_path = azure_files
        slide_paths = [Path(get_filename_from_url(url)) for url in azure_files.values()]
        print(f"Found {len(slide_paths)} unique slide files in Azure.")
        return

    # Original local filesystem search
    all_paths_set = set()
    expanded_patterns = []
    # Basic brace expansion handling (like {svs,tif,tiff})
    match = re.match(r"(.*)\{(.*)\}(.*)", pattern)
    if match:
        base, exts_str, suffix = match.groups()
        extensions = exts_str.split(",")
        expanded_patterns = [f"{base}{ext}{suffix}" for ext in extensions]
        print(f"  Expanded pattern to: {expanded_patterns}")
    else:
        expanded_patterns = [pattern]

    for exp_pattern in expanded_patterns:
        try:
            found_paths = list(Path().glob(exp_pattern))
            if found_paths:
                print(
                    f"  Found {len(found_paths)} files for sub-pattern '{exp_pattern}'"
                )
                all_paths_set.update(found_paths)
            else:
                print(f"  No files found for sub-pattern '{exp_pattern}'")
        except Exception as e:
            print(f"Error processing pattern '{exp_pattern}': {e}", file=sys.stderr)

    slide_paths = sorted([p.resolve() for p in all_paths_set if p.is_file()])
    filename_to_path = {p.stem: p for p in slide_paths}
    print(f"Found {len(slide_paths)} unique slide files on local filesystem.")


def _load_boxes():
    """Loads boxes from the JSON file defined by PERSIST_JSON_PATH."""
    global boxes_store
    if PERSIST_JSON_PATH and PERSIST_JSON_PATH.exists():
        print(f"Loading existing boxes from {PERSIST_JSON_PATH}")
        try:
            with open(PERSIST_JSON_PATH, "r") as f:
                loaded_data = json.load(f)
                # Basic validation: ensure it's a dictionary and values are lists of 4 ints
                if isinstance(loaded_data, dict):
                    valid_data = {}
                    invalid_count = 0
                    for k, v in loaded_data.items():
                        if (
                            isinstance(v, list)
                            and len(v) == 4
                            and all(isinstance(i, int) for i in v)
                        ):
                            valid_data[k] = v
                        else:
                            invalid_count += 1
                            print(
                                f"Warning: Invalid data format for key '{k}' in {PERSIST_JSON_PATH}. Skipping.",
                                file=sys.stderr,
                            )
                    boxes_store = valid_data
                    if invalid_count > 0:
                        print(
                            f"Warning: Skipped {invalid_count} invalid entries from {PERSIST_JSON_PATH}.",
                            file=sys.stderr,
                        )
                    print(f"Loaded {len(boxes_store)} box entries.")
                else:
                    print(
                        f"Warning: Invalid format in {PERSIST_JSON_PATH}. Expected a JSON object. Starting with empty store.",
                        file=sys.stderr,
                    )
                    boxes_store = {}
        except json.JSONDecodeError:
            print(
                f"Error: Could not decode JSON from {PERSIST_JSON_PATH}. Starting with empty store.",
                file=sys.stderr,
            )
            boxes_store = {}
        except Exception as e:
            print(
                f"Error loading boxes from {PERSIST_JSON_PATH}: {e}. Starting with empty store.",
                file=sys.stderr,
            )
            boxes_store = {}
    else:
        print(
            "Persistence file not found or not specified. Starting with empty box store."
        )
        boxes_store = {}


def _save_boxes():
    """Saves the current boxes_store to the JSON file defined by PERSIST_JSON_PATH."""
    if PERSIST_JSON_PATH:
        print(f"Saving {len(boxes_store)} box entries to {PERSIST_JSON_PATH}")
        try:
            # Ensure parent directory exists
            PERSIST_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PERSIST_JSON_PATH, "w") as f:
                json.dump(boxes_store, f, indent=2)  # Use indent for readability
        except Exception as e:
            print(f"Error saving boxes to {PERSIST_JSON_PATH}: {e}", file=sys.stderr)
    else:
        print("Skipping box persistence: PERSIST_JSON_PATH not set.")


# --- FastAPI Lifecycle ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    slide_pattern = os.environ.get("SLIDE_PATTERN")
    if not slide_pattern:
        print("Error: SLIDE_PATTERN environment variable not set.", file=sys.stderr)
        # Or raise an exception if you prefer the app not to start
        yield  # Allow startup to finish, but endpoints might fail gracefully
        # Shutdown
        print("Shutting down server.")
        return

    global PERSIST_JSON_PATH, DEIDENTIFIED_DIR
    persist_path_str = os.environ.get("PERSIST_JSON_PATH")
    if persist_path_str:
        # Check if this is an Azure SAS URL
        if is_azure_sas_url(persist_path_str):
            # For the proof of concept, we're not implementing Azure storage for the JSON
            # In a real implementation, you would download/upload the JSON as needed
            print(
                "Warning: Azure SAS URL for PERSIST_JSON_PATH not fully supported yet."
            )
            print("Will use local persistence.")
            PERSIST_JSON_PATH = Path("boxes.json").resolve()
        else:
            PERSIST_JSON_PATH = Path(persist_path_str).resolve()
        print(
            f"Persistence enabled. Boxes will be loaded/saved at: {PERSIST_JSON_PATH}"
        )
    else:
        print(
            "Warning: PERSIST_JSON_PATH environment variable not set. Box data will not be persisted.",
            file=sys.stderr,
        )

    deidentified_dir_str = os.environ.get("DEIDENTIFIED_DIR")
    if deidentified_dir_str:
        if is_azure_sas_url(deidentified_dir_str):
            # For Azure, we'll keep the SAS URL as a string
            DEIDENTIFIED_DIR = deidentified_dir_str
            print(f"Deidentified slides will be uploaded to Azure: {DEIDENTIFIED_DIR}")

            # Verify connection by attempting to list blobs
            try:
                container_client = get_container_client_from_sas(DEIDENTIFIED_DIR)
                _ = list(
                    container_client.list_blobs(max_results=1)
                )  # Just check if we can list
                print("Successfully connected to Azure output container.")
            except Exception as e:
                print(
                    f"Warning: Could not connect to Azure output container: {e}",
                    file=sys.stderr,
                )
                print("Deidentification operations may fail.", file=sys.stderr)
        else:
            # Local path
            DEIDENTIFIED_DIR = Path(deidentified_dir_str).resolve()
            os.makedirs(DEIDENTIFIED_DIR, exist_ok=True)
            print(
                f"Deidentified slides will be saved to local directory: {DEIDENTIFIED_DIR}"
            )
    else:
        print(
            "Warning: DEIDENTIFIED_DIR environment variable not set. Deidentification endpoint will not be available.",
            file=sys.stderr,
        )

    _find_slides(slide_pattern)
    _load_boxes()  # Load existing boxes from file
    print("Server startup complete.")
    yield
    # Shutdown
    _save_boxes()  # Save boxes on shutdown
    print("Shutting down server.")


# --- FastAPI App ---

app = FastAPI(
    title="WSI De-identifier API",
    description="API for managing WSI slide de-identification tasks.",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- API Endpoints ---


@app.get(
    "/slides",
    response_model=SlideListResponse,
    summary="List Available Slides",
    description="Retrieves a list of filenames (stems) for all slides found based on the startup pattern.",
)
async def get_slides():
    """Returns the list of discovered slide filenames (stems)."""
    return SlideListResponse(slides=list(filename_to_path.keys()))


@app.get(
    "/boxes/{slide_filename}",
    response_model=BoundingBoxResponse,
    summary="Get Bounding Box",
    description="Retrieves the stored bounding box coordinates for a given slide filename (stem). Returns an empty list if no box is set.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Slide filename not found"},
    },
)
async def get_bounding_box(slide_filename: str):
    """Returns the bounding box for the specified slide filename. Returns empty coords if box not set."""
    if slide_filename not in filename_to_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide '{slide_filename}' not found.",
        )
    coords = boxes_store.get(slide_filename, [])
    return BoundingBoxResponse(slide_filename=slide_filename, coords=coords)


@app.put(
    "/boxes/{slide_filename}",
    response_model=BoundingBoxResponse,
    status_code=status.HTTP_200_OK,  # Use 200 for update, 201 for creation (optional distinction)
    summary="Set or Update Bounding Box",
    description="Sets or updates the bounding box coordinates for a given slide filename (stem). Use coords [0,0,0,0] to delete (mark as unlabeled), and [-1,-1,-1,-1] to mark as 'no box needed'.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Slide not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Invalid bounding box data"
        },
    },
)
async def set_bounding_box(slide_filename: str, box_input: BoundingBoxInput):
    """Sets or updates the bounding box for the specified slide filename."""
    if slide_filename not in filename_to_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide '{slide_filename}' not found and cannot set box.",
        )

    # Basic type/length validation is handled by Pydantic model BoundingBoxInput
    coords = box_input.coords

    # Special case for deletion ([0,0,0,0] coordinates) -> Mark as unlabeled
    if coords == [0, 0, 0, 0]:
        if slide_filename in boxes_store:
            del boxes_store[slide_filename]
            print(
                f"Deleted bounding box entry for {slide_filename} (marked as unlabeled)."
            )
            _save_boxes()  # Persist change
        return BoundingBoxResponse(slide_filename=slide_filename, coords=[])

    # Special case for "No box needed" ([-1,-1,-1,-1] coordinates)
    elif coords == [-1, -1, -1, -1]:
        boxes_store[slide_filename] = coords
        print(f"Marked {slide_filename} as 'no box needed'.")
        _save_boxes()  # Persist change
        return BoundingBoxResponse(slide_filename=slide_filename, coords=coords)

    # Normal box - Perform additional validation
    else:
        x0, y0, x1, y1 = coords
        if x0 >= x1 or y0 >= y1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid coordinates: x0 must be less than x1, and y0 must be less than y1.",
            )

        # Store the validated coordinates
        boxes_store[slide_filename] = coords
        print(f"Stored bounding box for {slide_filename}: {coords}")
        _save_boxes()  # Persist change
        return BoundingBoxResponse(slide_filename=slide_filename, coords=coords)


@app.get(
    "/boxes/status",
    response_model=Dict[str, str],
    summary="Get Status of All Boxes",
    description="Returns the annotation status for all known slides ('labeled', 'unlabeled', 'no_box_needed').",
)
async def get_boxes_status():
    """Calculates and returns the status of each slide based on boxes_store."""
    status_map = {}
    for filename in filename_to_path.keys():
        if filename in boxes_store:
            coords = boxes_store[filename]
            if coords == [-1, -1, -1, -1]:
                status_map[filename] = "no_box_needed"
            elif (
                isinstance(coords, list) and len(coords) == 4
            ):  # Assume valid box otherwise
                status_map[filename] = "labeled"
            else:
                # Should not happen with current logic, but handle defensively
                status_map[filename] = "unlabeled"  # Treat unexpected data as unlabeled
        else:
            status_map[filename] = "unlabeled"
    return status_map


@app.get(
    "/label-stats",
    response_model=LabelStatsResponse,
    summary="Get Label Statistics",
    description="Returns statistics about labeled, unlabeled, and no-box-needed slides.",
)
async def get_label_stats():
    """Calculates and returns statistics about slide annotation status."""
    total = len(filename_to_path)
    labeled = 0
    no_box_needed = 0

    for filename in filename_to_path.keys():
        if filename in boxes_store:
            coords = boxes_store[filename]
            if coords == [-1, -1, -1, -1]:
                no_box_needed += 1
            elif isinstance(coords, list) and len(coords) == 4:
                labeled += 1

    # Unlabeled is the remainder
    unlabeled = total - labeled - no_box_needed

    return LabelStatsResponse(
        total=total, labeled=labeled, unlabeled=unlabeled, no_box_needed=no_box_needed
    )


@app.get(
    "/slides/{slide_filename}/image",
    response_model=SlideImageResponse,
    summary="Get Slide Image",
    description="Retrieves a representative image (macro, thumbnail, or generated) for a slide as base64 PNG.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Slide not found"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Error processing slide image"
        },
    },
)
async def get_slide_image(slide_filename: str):
    """Returns a base64 encoded PNG image for the specified slide."""
    if slide_filename not in filename_to_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide '{slide_filename}' not found.",
        )

    slide_path = filename_to_path[slide_filename]
    local_path = None
    temp_file = None

    try:
        # Handle Azure SAS URL
        if isinstance(slide_path, str) and is_azure_sas_url(slide_path):
            # Create a temporary file to download the slide
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svs")
            local_path = temp_file.name
            temp_file.close()  # Close so we can write to it

            # Download the slide from Azure
            print(f"Downloading slide {slide_filename} from Azure")
            download_azure_blob(slide_path, local_path)
            slide = openslide.OpenSlide(local_path)
        else:
            # Local file
            slide = openslide.OpenSlide(str(slide_path))

        img: Image.Image | None = None
        # Prioritize associated images often used for labels/thumbnails
        if "macro" in slide.associated_images:
            img = slide.associated_images["macro"].convert("RGB")
            print(f"Using 'macro' image for {slide_filename}")
        elif "thumbnail" in slide.associated_images:
            img = slide.associated_images["thumbnail"].convert("RGB")
            print(f"Using 'thumbnail' image for {slide_filename}")
        else:
            # Generate a thumbnail if no suitable associated image is found
            # Adjust size as needed for frontend display
            thumbnail_size = (1024, 1024)  # Example size, adjust as necessary
            img = slide.get_thumbnail(thumbnail_size).convert("RGB")
            print(f"Generated thumbnail ({thumbnail_size}) for {slide_filename}")

        if img is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not extract or generate an image for slide '{slide_filename}'.",
            )

        # Convert PIL image to base64 PNG
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return SlideImageResponse(slide_filename=slide_filename, image_data=img_str)

    except openslide.OpenSlideError as e:
        print(f"OpenSlideError for {slide_filename}: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error opening or processing slide '{slide_filename}': {e}",
        )
    except Exception as e:
        print(
            f"Unexpected error processing image for {slide_filename}: {e}",
            file=sys.stderr,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing image for slide '{slide_filename}'.",
        )
    finally:
        if "slide" in locals() and slide:
            slide.close()
        # Clean up temporary file if we created one
        if temp_file and local_path and os.path.exists(local_path):
            try:
                os.unlink(local_path)
            except Exception as e:
                print(
                    f"Warning: Could not delete temporary file {local_path}: {e}",
                    file=sys.stderr,
                )


@app.post(
    "/slides/{slide_filename}/deidentify",
    response_model=DeidentifyResponse,
    summary="Deidentify Slide",
    description="Deidentifies a slide by applying a bounding box to redact PHI in the macro image.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Slide not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Slide has no bounding box defined"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Error processing slide"
        },
    },
)
async def deidentify_slide(slide_filename: str):
    """Deidentifies a slide using its stored bounding box coordinates."""
    if slide_filename not in filename_to_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide '{slide_filename}' not found.",
        )

    if not DEIDENTIFIED_DIR:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DEIDENTIFIED_DIR environment variable not set. Deidentification is not available.",
        )

    coords = boxes_store.get(slide_filename, [])

    # Check if we have coordinates
    if not coords:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No bounding box defined for slide '{slide_filename}'. Please define a box first.",
        )

    slide_path = filename_to_path[slide_filename]
    local_input_path = None
    local_output_path = None
    temp_input_file = None
    temp_output_file = None
    output_path = ""

    try:
        # Download from Azure if needed
        if isinstance(slide_path, str) and is_azure_sas_url(slide_path):
            # Create temporary files for input and output
            temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svs")
            local_input_path = temp_input_file.name
            temp_input_file.close()

            print(f"Downloading slide {slide_filename} from Azure")
            download_azure_blob(slide_path, local_input_path)
        else:
            # Local file
            local_input_path = str(slide_path)

        # Create temporary output file if outputting to Azure
        if isinstance(DEIDENTIFIED_DIR, str) and is_azure_sas_url(DEIDENTIFIED_DIR):
            temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svs")
            local_output_path = temp_output_file.name
            temp_output_file.close()
        else:
            # Local directory
            local_output_path = str(Path(DEIDENTIFIED_DIR) / f"{slide_filename}.svs")

        # Check if this is marked as no-box-needed ([-1,-1,-1,-1])
        if coords == [-1, -1, -1, -1]:
            # For slides marked as not needing a box, we'll just copy the slide
            try:
                if os.path.exists(local_output_path):
                    os.remove(local_output_path)

                import shutil

                shutil.copy2(local_input_path, local_output_path)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error copying slide '{slide_filename}': {str(e)}",
                )
        else:
            # Regular case - apply redaction box
            replace_macro(
                local_input_path, local_output_path, coords, fill_color=(0, 0, 0)
            )

        # Upload to Azure if needed
        if isinstance(DEIDENTIFIED_DIR, str) and is_azure_sas_url(DEIDENTIFIED_DIR):
            print(f"Uploading deidentified slide {slide_filename} to Azure")
            output_path = upload_to_azure(
                local_output_path, DEIDENTIFIED_DIR, f"{slide_filename}.svs"
            )
        else:
            output_path = local_output_path

        return DeidentifyResponse(
            slide_filename=slide_filename, output_path=output_path
        )

    except Exception as e:
        print(f"Error deidentifying slide '{slide_filename}': {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deidentifying slide '{slide_filename}': {str(e)}",
        )
    finally:
        # Clean up temporary files
        if temp_input_file and local_input_path and os.path.exists(local_input_path):
            try:
                os.unlink(local_input_path)
            except Exception as e:
                print(
                    f"Warning: Could not delete temporary input file: {e}",
                    file=sys.stderr,
                )

        if temp_output_file and local_output_path and os.path.exists(local_output_path):
            try:
                os.unlink(local_output_path)
            except Exception as e:
                print(
                    f"Warning: Could not delete temporary output file: {e}",
                    file=sys.stderr,
                )


@app.post(
    "/slides/deidentify-all",
    response_model=BulkDeidentifyResponse,
    summary="Bulk Deidentify Slides",
    description="Deidentifies all slides that have bounding boxes defined.",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Error processing slides"
        },
    },
)
async def deidentify_all_slides():
    """Deidentifies all slides that have defined bounding boxes."""
    if not DEIDENTIFIED_DIR:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DEIDENTIFIED_DIR environment variable not set. Deidentification is not available.",
        )

    results = []
    skipped = []

    # Process all slides with defined bounding boxes
    for slide_filename in filename_to_path.keys():
        coords = boxes_store.get(slide_filename, [])

        # Skip slides without coordinates
        if not coords:
            skipped.append(slide_filename)
            continue

        slide_path = filename_to_path[slide_filename]
        local_input_path = None
        local_output_path = None
        temp_input_file = None
        temp_output_file = None
        output_path = ""

        try:
            # Download from Azure if needed
            if isinstance(slide_path, str) and is_azure_sas_url(slide_path):
                # Create temporary files for input and output
                temp_input_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".svs"
                )
                local_input_path = temp_input_file.name
                temp_input_file.close()

                print(f"Downloading slide {slide_filename} from Azure")
                download_azure_blob(slide_path, local_input_path)
            else:
                # Local file
                local_input_path = str(slide_path)

            # Create temporary output file if outputting to Azure
            if isinstance(DEIDENTIFIED_DIR, str) and is_azure_sas_url(DEIDENTIFIED_DIR):
                temp_output_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".svs"
                )
                local_output_path = temp_output_file.name
                temp_output_file.close()
            else:
                # Local directory
                local_output_path = str(
                    Path(DEIDENTIFIED_DIR) / f"{slide_filename}.svs"
                )

            # Check if this is marked as no-box-needed ([-1,-1,-1,-1])
            if coords == [-1, -1, -1, -1]:
                # For slides marked as not needing a box, just copy the slide
                if os.path.exists(local_output_path):
                    os.remove(local_output_path)

                import shutil

                shutil.copy2(local_input_path, local_output_path)
            else:
                # Regular case - apply redaction
                replace_macro(
                    local_input_path, local_output_path, coords, fill_color=(0, 0, 0)
                )

            # Upload to Azure if needed
            if isinstance(DEIDENTIFIED_DIR, str) and is_azure_sas_url(DEIDENTIFIED_DIR):
                print(f"Uploading deidentified slide {slide_filename} to Azure")
                output_path = upload_to_azure(
                    local_output_path, DEIDENTIFIED_DIR, f"{slide_filename}.svs"
                )
            else:
                output_path = local_output_path

            results.append(
                DeidentifyResponse(
                    slide_filename=slide_filename, output_path=output_path
                )
            )

        except Exception as e:
            print(f"Error deidentifying slide '{slide_filename}': {e}", file=sys.stderr)
            skipped.append(slide_filename)
        finally:
            # Clean up temporary files
            if (
                temp_input_file
                and local_input_path
                and os.path.exists(local_input_path)
            ):
                try:
                    os.unlink(local_input_path)
                except Exception as e:
                    print(
                        f"Warning: Could not delete temporary input file: {e}",
                        file=sys.stderr,
                    )

            if (
                temp_output_file
                and local_output_path
                and os.path.exists(local_output_path)
            ):
                try:
                    os.unlink(local_output_path)
                except Exception as e:
                    print(
                        f"Warning: Could not delete temporary output file: {e}",
                        file=sys.stderr,
                    )

    return BulkDeidentifyResponse(results=results, skipped=skipped)


# --- Main execution (for running with `python server.py`) ---
# Note: It's generally better to run with `uvicorn server:app --reload`

if __name__ == "__main__":
    import uvicorn

    print("Starting server with uvicorn. Use Ctrl+C to stop.")
    print("Ensure required environment variables are set, e.g.:")
    print(
        'SLIDE_PATTERN="sample/identified/*.svs" PERSIST_JSON_PATH="boxes.json" DEIDENTIFIED_DIR="deidentified" python server.py'
    )
    # Read pattern here ONLY if running directly, otherwise rely on lifespan
    if not os.environ.get("SLIDE_PATTERN"):
        print(
            "Warning: SLIDE_PATTERN not set. No slides will be loaded if run this way.",
            file=sys.stderr,
        )
        # Set a default or raise error if needed when run directly
        # os.environ["SLIDE_PATTERN"] = "sample/identified/*.svs" # Example default
    if not os.environ.get("PERSIST_JSON_PATH"):
        print(
            "Warning: PERSIST_JSON_PATH not set. Box data will not be persisted if run this way.",
            file=sys.stderr,
        )
    if not os.environ.get("DEIDENTIFIED_DIR"):
        print(
            "Warning: DEIDENTIFIED_DIR not set. Deidentification endpoints will not be available if run this way.",
            file=sys.stderr,
        )

    # Note: Lifespan events work better with uvicorn CLI runner
    uvicorn.run(
        app, host="0.0.0.0", port=8000
    )  # Add reload=True for development if desired
