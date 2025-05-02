# Multi-stage Dockerfile for WSI-Deidentifier app with Python and Node.js

###############################################################################
# Stage 1: Python Build with UV
###############################################################################
FROM python:3.10-slim-bookworm AS python-builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libzbar0 \
    libdmtx0b \
    libgl1 \
    libglib2.0-0 \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

# Install pip packages from requirements
WORKDIR /app

# Copy Python dependencies
COPY pyproject.toml ./

# Install dependencies using pip
RUN pip install --no-cache-dir -e .

# Copy Python application code
COPY *.py ./

# Create required directories
RUN mkdir -p /app/deidentified /app/sample

###############################################################################
# Stage 2: Node.js Build with Bun
###############################################################################
FROM oven/bun:latest AS node-builder

WORKDIR /app

# Copy Next.js app
COPY nextjs/package.json nextjs/bun.lock ./

# Install dependencies with Bun
RUN bun install --frozen-lockfile

# Copy source code
COPY nextjs/. ./

# Build Next.js app
RUN bun run build

###############################################################################
# Stage 3: Final image
###############################################################################
FROM python:3.10-slim-bookworm

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    openslide-tools \
    libzbar0 \
    libdmtx0b \
    libgl1 \
    libglib2.0-0 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python app
COPY --from=python-builder /app /app
COPY --from=python-builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy Next.js app
COPY --from=node-builder /app/.next /app/nextjs/.next
COPY --from=node-builder /app/node_modules /app/nextjs/node_modules
COPY --from=node-builder /app/public /app/nextjs/public
COPY --from=node-builder /app/package.json /app/nextjs/package.json
COPY --from=node-builder /app/next.config.ts /app/nextjs/next.config.ts

# Create required directories
RUN mkdir -p /app/deidentified /app/sample/identified

# Environment variables (with defaults)
ENV FASTAPI_PORT=8000
ENV NEXTJS_PORT=3000
ENV SLIDE_PATTERN="sample/identified/*.{svs,tif,tiff}"
ENV PERSIST_JSON_PATH="boxes.json"
ENV DEIDENTIFIED_DIR="deidentified"

# Expose ports
EXPOSE $FASTAPI_PORT $NEXTJS_PORT

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Check if credentials exist and are accessible\n\
if [ -f "/root/.config/gcloud/application_default_credentials.json" ]; then\n\
  echo "Using application default credentials from mounted volume"\n\
elif [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then\n\
  echo "Using service account credentials from GOOGLE_APPLICATION_CREDENTIALS"\n\
else\n\
  echo "Warning: No Google Cloud credentials found. Text detection using GCP Vision API may not work fully."\n\
fi\n\
\n\
# Start FastAPI in background\n\
cd /app\n\
python -m uvicorn server:app --reload --host 0.0.0.0 --port $FASTAPI_PORT &\n\
FASTAPI_PID=$!\n\
\n\
# Start Next.js\n\
cd /app/nextjs\n\
npx next start -p $NEXTJS_PORT &\n\
NEXTJS_PID=$!\n\
\n\
# Handle shutdown gracefully\n\
function cleanup() {\n\
  echo "Shutting down services..."\n\
  kill $FASTAPI_PID $NEXTJS_PID\n\
  wait\n\
}\n\
trap cleanup SIGTERM SIGINT\n\
\n\
echo "FastAPI running on http://localhost:$FASTAPI_PORT"\n\
echo "Next.js running on http://localhost:$NEXTJS_PORT"\n\
echo "SLIDE_PATTERN set to: $SLIDE_PATTERN"\n\
echo "PERSIST_JSON_PATH set to: $PERSIST_JSON_PATH"\n\
echo "DEIDENTIFIED_DIR set to: $DEIDENTIFIED_DIR"\n\
\n\
# Wait for processes to finish\n\
wait $FASTAPI_PID $NEXTJS_PID\n\
' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]