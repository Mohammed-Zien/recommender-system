    FROM python:3.10-slim

    WORKDIR /app

    # Avoid interactive prompts
    ENV DEBIAN_FRONTEND=noninteractive

    # Install OS packages
    RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && rm -rf /var/lib/apt/lists/*

    # Copy files
    COPY . .

    # Install Python deps
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt

    EXPOSE 8501

    COPY start.sh .
    RUN chmod +x start.sh

    CMD ["./start.sh"]

