FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for heavy ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
# We use --system to install into the system python, avoiding venv complexity in Docker
RUN uv pip install --system -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/ src/

# Set python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["prefect", "server", "start"]
