#!/bin/bash

# Kill any existing Streamlit processes on port 8503
lsof -ti :8503 | xargs -r kill -9

# Activate virtual environment and start the app
source "/Users/manish./AI for Market Trend Analysis/.venv/bin/activate"
cd "/Users/manish./AI for Market Trend Analysis/stock_forecaster"

echo "🚀 Starting Sentiment-Augmented Stock Trend Forecaster with SHAP integration..."
echo "📊 App will be available at: http://localhost:8503"
echo ""

streamlit run app.py --server.headless true --server.port 8503

