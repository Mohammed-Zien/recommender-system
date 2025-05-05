#!/bin/bash

# Start FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend
streamlit run app/streamlit_app.py --server.port=8501 --server.enableCORS=false
