# syntax=docker/dockerfile:1.6
# TSN v2 Dockerfile

#Removed the No Cache layer to reduce image size and build time.

# Base image with Python 3.11
FROM python:3.11-slim as base

# Install system dependencies (cached)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
        libsndfile1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Avoid large wheel download timeouts during pip install
ENV PIP_DEFAULT_TIMEOUT=180

# Set working directory
WORKDIR /app

# Copy requirements
COPY pyproject.toml README.md ./

# Install Python dependencies with cached wheels
RUN --mount=type=cache,target=/root/.cache/pip pip install -e .

# Copy source code
COPY tsn_common/ ./tsn_common/
COPY tsn_cli/ ./tsn_cli/

# Torch base to avoid re-downloading CUDA wheels
FROM base as torch-base

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch>=2.1 --index-url https://download.pytorch.org/whl/cu118

# Server target (includes transcription and analysis)
FROM torch-base as server

# Copy server code
COPY tsn_server/ ./tsn_server/
COPY tsn_orchestrator.py ./

# Create directories
RUN mkdir -p /incoming /storage

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run orchestrator
CMD ["python", "tsn_orchestrator.py"]

# Node target (lightweight, no GPU)
FROM base as node

# Copy node code
COPY tsn_node/ ./tsn_node/
COPY tsn_orchestrator.py ./

# Create directories
RUN mkdir -p /node/incoming /node/archive

# Run orchestrator
CMD ["python", "tsn_orchestrator.py"]

# Development target (includes dev tools)
FROM server as dev

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest>=7.4 \
    pytest-asyncio>=0.23 \
    pytest-cov>=4.1 \
    black>=24.1 \
    ruff>=0.1 \
    mypy>=1.8

# Copy tests
COPY tests/ ./tests/

# Run tests by default
CMD ["pytest", "-v", "--cov=tsn_common", "--cov=tsn_server", "--cov=tsn_node"]
