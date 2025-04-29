# PHI De-identifier

This tool processes Whole Slide Image (WSI) files (`.svs`, `.tif`, `.tiff`) to remove potentially identifying information. It performs the following steps:

1.  **Copies** input slides to a specified output directory.
2.  **Renames** the copied slides using a unique, salted SHA-256 hash derived from the original filename.
3.  **Removes** the associated "label" image from the copied slide.
4.  **Redacts** the associated "macro" image by drawing a solid red rectangle over a central portion (or user-specified coordinates).
5.  **Strips** non-technical metadata fields from the TIFF image description tag.
6.  **Generates** a CSV mapping file (`hash_mapping.csv` by default) to link original filenames to their hashed counterparts and track file paths.

# Usage

```bash
# Default centered rectangle
uv run deidentify.py "sample/identified/*.{svs,tif,tiff}" \
    --salt "your-secret-salt-here" \
    -o sample/deidentified \
    -m sample/hash_mapping.csv \
    --macro-description "macro"

# Specify custom rectangle coordinates (x0 y0 x1 y1)
uv run deidentify.py "path/to/slides/*.svs" \
    --salt "your-secret-salt-here" \
    -o output_dir \
    --rect 100 150 500 600
```

# Options

- `slides`: One or more input file paths or glob patterns (e.g., `"*.svs"`, `"path/to/slide.tif"`). Supports basic brace expansion like `{svs,tif}`.
- `-o, --out`: Specifies the output directory for de-identified slides (default: `deidentified`).
- `--salt`: A required secret string used for hashing filenames. **Keep this secure!**
- `-m, --map`: Specifies the path for the output CSV mapping file (default: `hash_mapping.csv`).
- `--macro-description`: String used to identify the macro image's description tag (case-insensitive, default: `macro`).
- `--rect`: Four integer coordinates `x0 y0 x1 y1` defining the redaction rectangle in the macro image. If omitted, a default rectangle (1/4 of the image dimensions, centered) is used.
