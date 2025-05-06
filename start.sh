#!/bin/bash
# In start.sh, before uvicorn...
cd "$(dirname "$0")"

echo "Contents of model_assets:"
ls -R app/model_assets

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
streamlit run app/streamlit_app.py
