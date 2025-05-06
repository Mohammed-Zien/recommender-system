FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install essential OS packages (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir huggingface_hub

# Copy the source code (excluding model assets via .dockerignore)
COPY . .

# Download model assets from Hugging Face
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='MohammedZien/hybrid-news-recommender-assets', revision='main', local_dir='app/model_assets')"

# Make start script executable
COPY start.sh . 
RUN chmod +x start.sh

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["./start.sh"]
