# Multi-stage build for smaller image size
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch to reduce image size
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Second stage: runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Download NLTK data during build to avoid runtime downloads
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create non-root user for security
RUN useradd -m appuser
USER appuser

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Create a volume mount point for model assets
VOLUME ["/app/model_assets"]

# Use entrypoint script to start either FastAPI or Streamlit based on args
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["api"]