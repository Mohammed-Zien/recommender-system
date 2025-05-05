#!/bin/bash
# Start the FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait a bit to make sure the backend is up before Streamlit starts
sleep 5

# Start the Streamlit frontend
streamlit run app/app.py --server.port=7860 --server.address=0.0.0.0
