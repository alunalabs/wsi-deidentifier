#!/usr/bin/env python3

from pathlib import Path
from pdf_processor import process_pdf

def main():
    # Define input and output paths
    input_pdf = Path("test.pdf")
    output_pdf = Path("test_redacted.pdf")
    
    # Process the PDF
    print(f"Processing {input_pdf}...")
    process_pdf(input_pdf, output_pdf)
    print(f"Completed processing. Output saved to {output_pdf}")

if __name__ == "__main__":
    main() 