#!/bin/bash
set -e

# Script to launch either FastAPI or Streamlit component
if [ "$1" = "api" ]; then
    echo "Starting FastAPI server..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
elif [ "$1" = "streamlit" ]; then
    echo "Starting Streamlit app..."
    exec streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
else
    echo "Command not recognized. Use 'api' for FastAPI or 'streamlit' for Streamlit app."
    exit 1
fi
