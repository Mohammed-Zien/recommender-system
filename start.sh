#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

sleep 5

exec streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
