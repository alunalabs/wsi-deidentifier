FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libdmtx0b \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY *.py ./

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    google-cloud-vision>=3.10.1 \
    google-genai>=1.12.1 \
    opencv-python>=4.11.0.86 \
    pillow>=11.2.1 \
    pylibdmtx>=0.1.10 \
    pytesseract>=0.3.13 \
    pyzbar>=0.1.9 \
    tifffile>=2025.3.30 \
    tinynumpy>=1.2.1

# Create directories for processed files
RUN mkdir -p /data/input /data/output

# Default command
ENTRYPOINT ["python", "find_identifying_boxes.py"]
CMD ["--help"]