#!/bin/bash
# Script to reorganize the project structure

# Ensure the package directory exists
mkdir -p wsi_deidentifier

# Copy Python files to the package
for file in deidentify.py identify_boxes.py replace_macro.py tiffparser.py server.py gcp_ocr_annotate.py gcp_textract.py gemini_extract.py; do
  if [ -f "$file" ]; then
    echo "Copying $file to wsi_deidentifier/"
    cp "$file" wsi_deidentifier/
  else
    echo "Warning: $file not found, skipping"
  fi
done

# Install in development mode
echo "Running uv sync to install the package"
uv sync

echo "Done! Your package structure has been reorganized."
echo "You can now import modules from the wsi_deidentifier package."
echo "Example: from wsi_deidentifier import deidentify" 