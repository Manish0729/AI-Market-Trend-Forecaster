## Sentiment-Augmented Stock Trend Forecaster

An interactive Streamlit app that forecasts future stock price trends by combining:

- Historical OHLCV data from `yfinance`
- Daily news sentiment from simulated headlines using Hugging Face Transformers

The model uses Prophet with the 7-day rolling mean of sentiment as an extra regressor to improve forecasts.

### Why

Markets often react to narratives and sentiment. By incorporating a sentiment signal derived from news headlines (simulated for this demo), we can study how narrative trends may affect price movements.

### What it does

1. Downloads the last 2 years of daily stock data for a given ticker
2. Generates realistic, ticker-aware sample news headlines across the same timeframe
3. Analyzes per-headline sentiment using `distilbert-base-uncased-finetuned-sst-2-english`
4. Aggregates daily mean sentiment and computes a 7-day rolling mean
5. Trains a Prophet model with `sentiment_7d` as an extra regressor
6. Forecasts the next 90 days and visualizes results with Plotly

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Then open the local URL in your browser and enter a ticker (e.g., `AAPL`) and click "Run Forecast".

### Notes

- Uses simulated headlines for demonstration only; no external news API is required.
- GPU is optional. If CUDA is available for PyTorch, the sentiment pipeline will use it; otherwise it runs on CPU.
- Sentiment influences Prophet via an extra regressor (`sentiment_7d`), which is held constant at the last known value for future dates.
- If the environment is offline or the model cannot be loaded, the app falls back to neutral sentiment so it remains usable.

### Project Structure

```
stock_forecaster/
├─ app.py
├─ requirements.txt
├─ README.md
└─ src/
   ├─ __init__.py
   ├─ data_handler.py
   ├─ sentiment_analyzer.py
   └─ model_trainer.py
```



