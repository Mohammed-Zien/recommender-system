FROM python:3.11.2-slim

# Use Google's DNS

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /code

# Copy everything into the image
COPY . .

# Install required system packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt


# Download NLTK data
RUN python -m nltk.downloader punkt wordnet stopwords

# Expose backend and frontend ports
EXPOSE 7860
EXPOSE 8000

# Run both FastAPI and Streamlit via start.sh
CMD ["./start.sh"]
