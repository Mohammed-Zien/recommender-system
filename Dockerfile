# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY start.sh .

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

ENV PATH=/root/.local/bin:$PATH

# Set permissions and expose port
EXPOSE 8000 8501
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
