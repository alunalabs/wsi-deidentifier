#!/usr/bin/env python3

import os
import glob
from pathlib import Path
import openslide
from PIL import Image

def extract_label_from_svs(svs_path, output_path):
    """Extract label image from SVS file and save as PNG"""
    try:
        slide = openslide.OpenSlide(svs_path)
        
        # Try to get the label image (usually at index 2 for most SVS files)
        # Some files may have it at different indices
        label_img = None
        
        # Check if there's an associated image called 'label'
        if 'label' in slide.associated_images:
            label_img = slide.associated_images['label']
        elif 'macro' in slide.associated_images:
            # Sometimes the label is stored as 'macro'
            label_img = slide.associated_images['macro']
        else:
            print(f"No label/macro image found in {svs_path}")
            slide.close()
            return False
        
        # Save as PNG
        label_img.save(output_path, 'PNG')
        slide.close()
        print(f"Extracted label from {svs_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {svs_path}: {e}")
        return False

def main():
    # Get all SVS files in current directory
    svs_files = glob.glob("*.svs")
    
    if not svs_files:
        print("No SVS files found in current directory")
        return
    
    print(f"Found {len(svs_files)} SVS files")
    
    success_count = 0
    
    for svs_file in svs_files:
        # Create output filename by replacing .svs with .png
        png_file = svs_file.replace('.svs', '.png')
        
        if extract_label_from_svs(svs_file, png_file):
            success_count += 1
    
    print(f"Successfully extracted {success_count}/{len(svs_files)} label images")

if __name__ == "__main__":
    main()