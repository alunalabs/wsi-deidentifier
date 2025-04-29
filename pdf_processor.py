#!/usr/bin/env python3

import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
import fitz
from PyPDF2 import PdfReader
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import vision
import base64
import json


def extract_images_from_pdf(
    pdf_path: Path,
) -> list[tuple[int, Image.Image, tuple[int, int, int, int]]]:
    """Extract images from a PDF along with their positions."""
    print(f"Extracting images from {pdf_path}")
    reader = PdfReader(str(pdf_path))
    out: list[tuple[int, Image.Image, tuple[int, int, int, int]]] = []

    for page_num, page in enumerate(reader.pages):
        print(f"Processing page {page_num + 1}")
        try:
            resources = page["/Resources"].get_object()
            if "/XObject" not in resources:
                continue
            x_objects = resources["/XObject"].get_object()
            
            for obj in x_objects.values():
                if not isinstance(obj, dict):
                    obj = obj.get_object()
                if obj.get("/Subtype", "") != "/Image":
                    continue

                try:
                    filter_type = obj.get("/Filter", "")
                    if isinstance(filter_type, list):
                        filter_type = filter_type[0]

                    # Get image data
                    data = obj.get_data()
                    if not data:
                        print(f"No image data found for object")
                        continue

                    # Handle different image formats
                    if filter_type in ["/DCTDecode", "/JPXDecode"]:
                        # JPEG or JPEG2000
                        img = Image.open(BytesIO(data))
                    elif filter_type == "/FlateDecode":
                        width = obj.get("/Width", 0)
                        height = obj.get("/Height", 0)
                        color_space = obj.get("/ColorSpace", "/DeviceRGB")
                        bits_per_component = obj.get("/BitsPerComponent", 8)

                        if not (width and height):
                            print(f"Missing dimensions for image")
                            continue

                        # Handle different color spaces
                        if color_space == "/DeviceRGB" or (isinstance(color_space, list) and color_space[0] == "/CalRGB"):
                            mode = "RGB"
                            size = (width, height)
                        elif color_space == "/DeviceGray":
                            mode = "L"
                            size = (width, height)
                        else:
                            print(f"Unsupported color space: {color_space}")
                            continue

                        try:
                            # Create image from raw data
                            img = Image.frombytes(mode, size, data)
                        except Exception as e:
                            print(f"Error creating image from bytes: {e}")
                            continue
                    else:
                        print(f"Unsupported filter type: {filter_type}")
                        continue

                    # Ensure image is in RGB mode
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Use page dimensions if BBox not available
                    bbox = obj.get("/BBox", [0, 0, page.mediabox.width, page.mediabox.height])
                    print(f"Found image with bbox: {bbox}")
                    out.append((page_num, img, bbox))
                except Exception as e:
                    print(f"Error processing image object: {e}")

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")

    return out


def detect_text_with_vision(image: Image.Image) -> list[tuple[float, float, float, float, str]]:
    """Detect text using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    
    # Convert PIL Image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    vision_image = vision.Image(content=img_byte_arr)
    response = client.text_detection(image=vision_image)
    
    if response.error.message:
        raise Exception(f"Error from Vision API: {response.error.message}")
    
    regions = []
    # Skip the first annotation which is the full text
    for text in response.text_annotations[1:]:
        vertices = text.bounding_poly.vertices
        x = min(v.x for v in vertices)
        y = min(v.y for v in vertices)
        width = max(v.x for v in vertices) - x
        height = max(v.y for v in vertices) - y
        regions.append((x, y, width, height, text.description))
    
    return regions


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image to improve text detection."""
    # Convert to numpy array for OpenCV processing
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply multiple preprocessing steps
    # 1. Adaptive thresholding for text with varying contrast
    binary1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 2. Otsu's thresholding for global threshold
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Combine both thresholding results
    combined = cv2.bitwise_or(binary1, binary2)
    
    # 4. Denoise
    denoised = cv2.fastNlMeansDenoising(combined)
    
    # 5. Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 6. Additional preprocessing for faint text
    kernel = np.ones((1,1), np.uint8)
    dilated = cv2.dilate(enhanced, kernel, iterations=1)
    
    # 7. Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(dilated, -1, kernel)
    
    # Convert back to PIL Image
    return Image.fromarray(sharpened)


def save_debug_image(image: Image.Image, suffix: str, page_num: int):
    """Save image for debugging purposes."""
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)
    output_path = debug_dir / f"page_{page_num}_{suffix}.jpg"
    image.save(output_path)
    print(f"Saved debug image to {output_path}")


def analyze_pdf_with_gemini(pdf_path: Path) -> list[tuple[int, int, int, int, int]]:
    """Analyze PDF with Gemini to identify regions containing PII."""
    print(f"Analyzing PDF with Gemini: {pdf_path}")
    
    # Initialize Vertex AI
    vertexai.init(project="fresh-span-458302-k9", location="us-central1")
    model = GenerativeModel("gemini-2.5-flash-preview-04-17")
    
    # Extract images from PDF
    images = extract_images_from_pdf(pdf_path)
    regions = []
    
    for page_idx, img, bbox in images:
        try:
            # Preprocess image
            processed_img = preprocess_image(img)
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Convert PIL Image to base64
            buffered = BytesIO()
            processed_img.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Create prompt for Gemini to focus only on text regions
            prompt = """
            You are a PII detection expert. Your task is to identify ONLY text regions containing personally identifiable information (PII) 
            or sensitive information in this medical document. DO NOT mark any image regions or graphics. Focus only on text areas.

            This includes but is not limited to:

            1. Personal Identifiers:
               - Names (first, last, full names)
               - Addresses (street, city, state, zip)
               - Phone numbers
               - Email addresses
               - Social security numbers
               - Medical record numbers
               - Patient IDs
               - Dates of birth
               - Driver's license numbers

            2. Medical Information:
               - Patient names
               - Medical record numbers
               - Diagnosis codes
               - Procedure codes
               - Treatment information
               - Test results
               - Physician names
               - Hospital/clinic names
               - Department names
               - Lab results
               - Pathology reports
               - Specimen IDs
               - Slide numbers
               - Block numbers
               - Case numbers

            3. Business Information:
               - Company names
               - Department names
               - Employee IDs
               - Internal reference numbers
               - Case numbers
               - Document IDs
               - Any identifying numbers or codes
               - Location information
               - Branch names
               - Division names
               - Lab names
               - Facility names

            4. Location Information:
               - Hospital/clinic addresses
               - Department locations
               - Room numbers
               - Building names
               - Floor numbers
               - Wing names
               - Unit names
               - Lab locations
               - Facility locations

            5. Document Metadata:
               - Document numbers
               - Reference numbers
               - Case numbers
               - Accession numbers
               - Barcode numbers
               - QR codes
               - Watermarks
               - Stamps
               - Signatures
               - Dates
               - Times
               - Document IDs

            For each text region found, provide the coordinates in JSON format:
            {
                "regions": [
                    {
                        "x": x_coordinate,
                        "y": y_coordinate,
                        "width": width,
                        "height": height,
                        "type": "type_of_pii"
                    }
                ]
            }

            Important guidelines:
            1. ONLY mark text regions - ignore all images, graphics, or visual elements
            2. Be thorough - if you see ANY text that could identify a person, place, or organization, mark it
            3. Include ALL headers, footers, and watermarks that might contain identifying information
            4. Mark any reference numbers, codes, or identifiers even if their meaning isn't clear
            5. When in doubt, mark the region - it's better to over-redact than under-redact
            6. Pay special attention to:
               - Document headers and footers
               - Stamps and watermarks
               - Reference numbers
               - Department names
               - Location information
               - Any text that could be used to identify a person or organization
               - Barcodes and QR codes
               - Document numbers and IDs
               - Case numbers and references
               - Dates and times
               - Any numbers that could be identifiers
            7. Remember:
               - Redact ALL identifying information
               - Redact ALL dates and times
               - Redact ALL reference numbers
               - Redact ALL location information
               - Redact ALL department names
               - Redact ALL facility names
               - When in doubt, redact it
            """
            
            # Call Gemini
            response = model.generate_content(
                [
                    Part.from_text(prompt),
                    Part.from_data(img_str, mime_type="image/jpeg")
                ]
            )
            
            try:
                # Extract JSON from the response text
                response_text = response.text
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end != -1:
                        response_text = response_text[json_start:json_end].strip()
                
                # Parse Gemini's response
                result = json.loads(response_text)
                
                # If Gemini didn't find any regions, try Vision API as fallback
                if not result.get("regions"):
                    vision_regions = detect_text_with_vision(processed_img)
                    for x, y, w, h, text in vision_regions:
                        # Convert coordinates to PDF space
                        pdf_x = bbox[0] + (x * bbox[2] / img_width)
                        pdf_y = bbox[1] + (y * bbox[3] / img_height)
                        pdf_w = w * bbox[2] / img_width
                        pdf_h = h * bbox[3] / img_height
                        regions.append((page_idx, pdf_x, pdf_y, pdf_w, pdf_h))
                else:
                    for region in result.get("regions", []):
                        # Convert coordinates to PDF space
                        pdf_x = bbox[0] + (region["x"] * bbox[2] / img_width)
                        pdf_y = bbox[1] + (region["y"] * bbox[3] / img_height)
                        pdf_w = region["width"] * bbox[2] / img_width
                        pdf_h = region["height"] * bbox[3] / img_height
                        regions.append((page_idx, pdf_x, pdf_y, pdf_w, pdf_h))
                        
            except Exception as e:
                print(f"Error parsing Gemini response: {e}")
                # Try Vision API as fallback
                try:
                    vision_regions = detect_text_with_vision(processed_img)
                    for x, y, w, h, text in vision_regions:
                        # Convert coordinates to PDF space
                        pdf_x = bbox[0] + (x * bbox[2] / img_width)
                        pdf_y = bbox[1] + (y * bbox[3] / img_height)
                        pdf_w = w * bbox[2] / img_width
                        pdf_h = h * bbox[3] / img_height
                        regions.append((page_idx, pdf_x, pdf_y, pdf_w, pdf_h))
                except Exception as ve:
                    print(f"Vision API fallback also failed: {ve}")
        
        except Exception as e:
            print(f"Error processing page {page_idx}: {e}")
            continue
    
    return regions


def redact_pdf_regions(pdf_path: Path, regions: list[tuple[int, int, int, int, int]], output_path: Path) -> None:
    """Redact regions in a PDF and clear metadata."""
    print(f"Applying redaction from {pdf_path} to {output_path}")

    # Group regions per page
    page_map: dict[int, list[tuple[int, int, int, int]]] = {}
    for page_idx, x, y, w, h in regions:
        # Convert all coordinates to float
        x, y, w, h = float(x), float(y), float(w), float(h)
        page_map.setdefault(page_idx, []).append((x, y, w, h))

    with fitz.open(pdf_path) as doc:
        for page_idx, rects in page_map.items():
            page = doc[page_idx]
            page_h = float(page.rect.height)  # Convert page height to float
            print(f"Processing page {page_idx + 1} with height {page_h}")
            for x, y, w, h in rects:
                # convert PDF bottom-left coords → MuPDF top-left coords
                x0 = x
                y0 = page_h - (y + h)  # Flip y-coordinate and adjust for height
                x1 = x + w
                y1 = page_h - y  # Flip y-coordinate
                print(f"Adding redaction box on page {page_idx + 1}: {(x0, y0, x1, y1)}")
                page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(0, 0, 0))
            page.apply_redactions()

        # Clear metadata
        doc.set_metadata({})
        print("Clearing metadata")
        doc.save(output_path, garbage=4, deflate=True)
        print(f"Saved redacted PDF to {output_path}")


def process_pdf(pdf_path: Path, output_path: Path | None = None) -> None:
    """Main entry point for PDF processing. Handles all PDF-related operations."""
    if output_path is None:
        output_path = pdf_path.with_stem(pdf_path.stem + "_redacted")
        
    print(f"Processing PDF: {pdf_path}")
    try:
        # Analyze PDF with Gemini to find PII regions
        regions = analyze_pdf_with_gemini(pdf_path)
        print(f"Found {len(regions)} regions to redact")
        
        # Apply redaction to new file
        redact_pdf_regions(pdf_path, regions, output_path)
        print("✓ PDF processing completed successfully")
    except Exception as e:
        print(f"✗ Error processing PDF: {e}")