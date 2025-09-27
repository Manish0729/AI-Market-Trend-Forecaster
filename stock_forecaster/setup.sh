#!/bin/bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
" > ~/.streamlit/config.toml

# Use python -m streamlit instead of streamlit command
python -m streamlit run app.py
