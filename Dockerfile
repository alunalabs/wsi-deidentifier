FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libdmtx0b \
    libgl1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY *.py ./

# Install Python dependencies using uv
RUN uv sync

# Create directories for processed files
RUN mkdir -p /data/input /data/output

# Create entry script to ensure virtual environment is activated
RUN echo '#!/bin/bash\n\
source .venv/bin/activate\n\
python find_identifying_boxes.py "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]