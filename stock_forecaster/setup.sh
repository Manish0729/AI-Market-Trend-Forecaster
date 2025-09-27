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

# Use python3 -m streamlit instead of streamlit command
python3 -m streamlit run app.py
