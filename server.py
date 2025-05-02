#!/usr/bin/env python3
import base64
import io
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import openslide
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from replace_macro import replace_macro

# --- Globals / State ---

# Will store Path objects for found slides
slide_paths: List[Path] = []
# Maps filename stem to Path object
filename_to_path: Dict[str, Path] = {}
# In-memory storage for bounding boxes (filename_stem -> [x0, y0, x1, y1] or [-1,-1,-1,-1] for no-box-needed)
boxes_store: Dict[str, List[int]] = {}
# Path for persisting boxes
PERSIST_JSON_PATH: Path | None = None
# Directory for deidentified slides
DEIDENTIFIED_DIR: Path | None = None


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

    results: List[DeidentifyResponse] = Field(..., description="Results for each processed slide.")
    skipped: List[str] = Field(..., description="Slides that were skipped due to errors or missing boxes.")
    
    
class LabelStatsResponse(BaseModel):
    """Response model for label statistics."""
    
    total: int = Field(..., description="Total number of slides")
    labeled: int = Field(..., description="Number of slides that have been labeled")
    unlabeled: int = Field(..., description="Number of slides that are not labeled yet")
    no_box_needed: int = Field(..., description="Number of slides marked as not needing a box")


# --- Helper Functions ---


def _find_slides(pattern: str) -> None:
    """Finds slides based on glob pattern and populates global state."""
    global slide_paths, filename_to_path
    print(f"Searching for slides using pattern: {pattern}")

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
    print(f"Found {len(slide_paths)} unique slide files.")
    # print("Full paths found:")
    # for p in slide_paths:
    #     print(f"  - {p}")
    # print("Filename stem mapping:")
    # for stem, path in filename_to_path.items():
    #     print(f"  - {stem}: {path}")


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
        DEIDENTIFIED_DIR = Path(deidentified_dir_str).resolve()
        os.makedirs(DEIDENTIFIED_DIR, exist_ok=True)
        print(f"Deidentified slides will be saved to: {DEIDENTIFIED_DIR}")
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
        total=total,
        labeled=labeled,
        unlabeled=unlabeled,
        no_box_needed=no_box_needed
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
    try:
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


@app.post(
    "/slides/{slide_filename}/deidentify",
    response_model=DeidentifyResponse,
    summary="Deidentify Slide",
    description="Deidentifies a slide by applying a bounding box to redact PHI in the macro image.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Slide not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Slide has no bounding box defined"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Error processing slide"},
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
    
    # Check if this is marked as no-box-needed ([-1,-1,-1,-1])
    if coords == [-1, -1, -1, -1]:
        # For slides marked as not needing a box, we'll just copy the slide to the output directory
        input_path = str(filename_to_path[slide_filename])
        output_path = str(DEIDENTIFIED_DIR / f"{slide_filename}.svs")
        
        try:
            # Just create a hard link or copy the file
            if os.path.exists(output_path):
                os.remove(output_path)
            try:
                os.link(input_path, output_path)
            except (OSError, AttributeError):
                import shutil
                shutil.copy2(input_path, output_path)
                
            return DeidentifyResponse(slide_filename=slide_filename, output_path=output_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error copying slide '{slide_filename}': {str(e)}",
            )
    
    # Regular case - apply redaction box
    input_path = str(filename_to_path[slide_filename])
    output_path = str(DEIDENTIFIED_DIR / f"{slide_filename}.svs")
    
    try:
        replace_macro(input_path, output_path, coords, fill_color=(0, 0, 0))
        return DeidentifyResponse(slide_filename=slide_filename, output_path=output_path)
    except Exception as e:
        print(f"Error deidentifying slide '{slide_filename}': {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deidentifying slide '{slide_filename}': {str(e)}",
        )


@app.post(
    "/slides/deidentify-all",
    response_model=BulkDeidentifyResponse,
    summary="Bulk Deidentify Slides",
    description="Deidentifies all slides that have bounding boxes defined.",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Error processing slides"},
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
        
        input_path = str(filename_to_path[slide_filename])
        output_path = str(DEIDENTIFIED_DIR / f"{slide_filename}.svs")
        
        try:
            # Check if this is marked as no-box-needed ([-1,-1,-1,-1])
            if coords == [-1, -1, -1, -1]:
                # For slides marked as not needing a box, just copy the slide
                if os.path.exists(output_path):
                    os.remove(output_path)
                try:
                    os.link(input_path, output_path)
                except (OSError, AttributeError):
                    import shutil
                    shutil.copy2(input_path, output_path)
            else:
                # Regular case - apply redaction
                replace_macro(input_path, output_path, coords, fill_color=(0, 0, 0))
                
            results.append(DeidentifyResponse(slide_filename=slide_filename, output_path=output_path))
        except Exception as e:
            print(f"Error deidentifying slide '{slide_filename}': {e}", file=sys.stderr)
            skipped.append(slide_filename)
    
    return BulkDeidentifyResponse(results=results, skipped=skipped)


# --- Main execution (for running with `python server.py`) ---
# Note: It's generally better to run with `uvicorn server:app --reload`

if __name__ == "__main__":
    import uvicorn

    print("Starting server with uvicorn. Use Ctrl+C to stop.")
    print("Ensure required environment variables are set, e.g.:")
    print('SLIDE_PATTERN="sample/identified/*.svs" PERSIST_JSON_PATH="boxes.json" DEIDENTIFIED_DIR="deidentified" python server.py')
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
