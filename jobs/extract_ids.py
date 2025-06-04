#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "flask>=3.1.1",
#   "openai>=1.84.0",
#   "python-dotenv>=1.1.0",
# ]
# ///


import asyncio
import base64
import csv
import glob
import os
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


def encode_image(image_path):
    """Encode image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def extract_id_from_image(
    client: AsyncOpenAI,
    image_path: str,
    example_images: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Extract ID from label image using OpenAI vision model"""
    try:
        base64_image = encode_image(image_path)

        # Build the content array with text and example images
        content = [
            {
                "type": "text",
                "text": """Look at this medical slide label image and extract the specimen ID. The ID typically follows a pattern like GCXX-XXXX YYYYY where XX is a 2-digit number and XXXX is a 4-digit number. YYYYY is typically found near the bottom of the image in smaller text.""",
            }
        ]

        # Add example images if provided
        if example_images:
            for example_file, expected_id in example_images.items():
                if os.path.exists(example_file):
                    example_base64 = encode_image(example_file)
                    content.extend(
                        [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{example_base64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": f"{expected_id}",
                            },
                        ]
                    )

        content.extend(
            [
                {
                    "type": "text",
                    "text": "Now extract the ID from this image. Return ONLY the ID in the format GCXX-XXXX YYYYY, nothing else. Do not include any other text or comments.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )

        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=100,
        )

        extracted_id = response.choices[0].message.content.strip()
        print(f"Extracted ID from {image_path}: {extracted_id}")
        return extracted_id

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


async def main():
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get all PNG files in current directory
    png_files = glob.glob("*.png")

    if not png_files:
        print("No PNG files found in current directory")
        return

    print(f"Found {len(png_files)} PNG files")

    # Define example images with known IDs
    example_images = {
        "@D.png": "GC21-1399 35392",
        "=D.png": "GC21-1398 35389",
        "FD.png": "GC21-1399 35398",
    }

    # Process files concurrently
    tasks = [
        extract_id_from_image(client, png_file, example_images)
        for png_file in png_files
    ]
    results = []

    for png_file, extracted_id in zip(png_files, await asyncio.gather(*tasks)):
        results.append({"filename": png_file, "extracted_id": extracted_id})

    # Write results to CSV
    with open("filename_to_id_mapping.csv", "w", newline="") as csvfile:
        fieldnames = ["filename", "extracted_id"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print("Results saved to filename_to_id_mapping.csv")

    # Print summary
    successful_extractions = sum(1 for r in results if r["extracted_id"] is not None)
    print(
        f"Successfully extracted IDs from {successful_extractions}/{len(png_files)} files"
    )


if __name__ == "__main__":
    asyncio.run(main())
