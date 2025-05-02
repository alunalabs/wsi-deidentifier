#!/usr/bin/env python3
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --- Globals / State ---

# Will store Path objects for found slides
slide_paths: List[Path] = []
# Maps filename stem to Path object
filename_to_path: Dict[str, Path] = {}
# In-memory storage for bounding boxes (filename_stem -> [x0, y0, x1, y1])
boxes_store: Dict[str, List[int]] = {}


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

    @field_validator("coords")
    @classmethod
    def validate_coords(cls, v: List[int]) -> List[int]:
        if len(v) != 4:
            # This check is technically redundant due to min/max_length, but good practice
            raise ValueError("Coordinates list must contain exactly 4 integers.")
        x0, y0, x1, y1 = v
        if not (
            isinstance(x0, int)
            and isinstance(y0, int)
            and isinstance(x1, int)
            and isinstance(y1, int)
        ):
            raise ValueError("All coordinates must be integers.")
        if x0 >= x1 or y0 >= y1:
            raise ValueError(
                "Invalid coordinates: x0 must be less than x1, and y0 must be less than y1."
            )
        return v


class BoundingBoxResponse(BaseModel):
    """Response model for retrieving bounding box coordinates."""

    slide_filename: str = Field(..., description="The filename stem of the slide.")
    coords: List[int] = Field(
        ..., description="Bounding box coordinates [x0, y0, x1, y1]."
    )


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

    _find_slides(slide_pattern)
    # TODO: Optionally load existing boxes from a file here
    print("Server startup complete.")
    yield
    # Shutdown
    # TODO: Optionally save boxes to a file here
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
    description="Retrieves the stored bounding box coordinates for a given slide filename (stem).",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Bounding box not found for this slide"
        },
    },
)
async def get_bounding_box(slide_filename: str):
    """Returns the bounding box for the specified slide filename."""
    if slide_filename not in filename_to_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide '{slide_filename}' not found.",
        )
    if slide_filename not in boxes_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bounding box not set for slide '{slide_filename}'.",
        )
    return BoundingBoxResponse(
        slide_filename=slide_filename, coords=boxes_store[slide_filename]
    )


@app.put(
    "/boxes/{slide_filename}",
    response_model=BoundingBoxResponse,
    status_code=status.HTTP_200_OK,  # Use 200 for update, 201 for creation (optional distinction)
    summary="Set or Update Bounding Box",
    description="Sets or updates the bounding box coordinates for a given slide filename (stem).",
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

    # Basic validation is handled by Pydantic model BoundingBoxInput
    # Store the validated coordinates
    boxes_store[slide_filename] = box_input.coords
    print(f"Stored bounding box for {slide_filename}: {box_input.coords}")

    return BoundingBoxResponse(slide_filename=slide_filename, coords=box_input.coords)


# --- Main execution (for running with `python server.py`) ---
# Note: It's generally better to run with `uvicorn server:app --reload`

if __name__ == "__main__":
    import uvicorn

    print("Starting server with uvicorn. Use Ctrl+C to stop.")
    print("Ensure SLIDE_PATTERN environment variable is set, e.g.:")
    print('SLIDE_PATTERN="sample/identified/*.svs" python server.py')
    # Read pattern here ONLY if running directly, otherwise rely on lifespan
    if not os.environ.get("SLIDE_PATTERN"):
        print(
            "Warning: SLIDE_PATTERN not set. No slides will be loaded if run this way.",
            file=sys.stderr,
        )
        # Set a default or raise error if needed when run directly
        # os.environ["SLIDE_PATTERN"] = "sample/identified/*.svs" # Example default

    # Note: Lifespan events work better with uvicorn CLI runner
    uvicorn.run(
        app, host="0.0.0.0", port=8000
    )  # Add reload=True for development if desired
