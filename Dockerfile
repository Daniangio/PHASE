#
# Dockerfile for the AllosKin Backend Service
# This uses a multi-stage build for efficiency.
# Build context: Root of the AllosKin directory
#

# --- 1. Base Stage ---
# Use a slim Python image.
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye AS base
WORKDIR /app

# Set non-interactive frontend for package managers
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install build tools
RUN pip install --upgrade pip setuptools wheel

# Install core + backend requirements
# We copy requirements first to leverage build cache
COPY requirements.txt .
COPY backend/requirements.txt ./backend_requirements.txt
# Ensure dependencies are installed in the Base stage
RUN pip install -r requirements.txt
RUN pip install -r backend_requirements.txt

# --- 2. Builder Stage (Packages the application code) ---
FROM base AS builder
# Copy the source code for the alloskin library
COPY pyproject.toml .
COPY README.md .
COPY alloskin/ alloskin/
# Install the package
RUN pip install --no-build-isolation --no-deps .

# --- 3. Final App Stage (Lean Production Image) ---
FROM python:3.11-slim AS final
WORKDIR /app
# Set environment variables again
ENV PYTHONUNBUFFERED=1

# 1. Copy the Python site-packages (libraries)
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# 2. Copy the executables (uvicorn, fastapi, etc.) to the $PATH location
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the backend app code
COPY backend/ backend/

# When WORKDIR is /app, the 'backend' folder is correctly seen as the 'backend' module.
WORKDIR /app

EXPOSE 8000

# Run the FastAPI server
# This command is overridden in docker-compose.override.yml for development.
# The `backend.main:app` module is now correctly resolved from the /app directory.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
