# Sentiment-Augmented Stock Trend Forecaster

A sophisticated AI-powered stock market forecasting application that combines historical price data, sentiment analysis of news headlines, and advanced technical indicators to predict future market trends. The application uses machine learning models to provide accurate forecasts with explainable AI insights.

## 🚀 Features

* **Automatic Model Showdown**: Tests multiple sentiment window configurations and selects the best-performing model
* **Multi-Indicator Analysis**: Integrates RSI, MACD, and SMA for comprehensive technical analysis
* **Sentiment Intelligence**: Analyzes news headlines using Hugging Face NLP models
* **Explainable AI (SHAP)**: Provides transparent insights into model predictions
* **Interactive Dashboard**: Professional Streamlit interface with real-time charts
* **Comprehensive Coverage**: Supports Indian stocks, US markets, forex, commodities, and cryptocurrencies

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **ML**: Prophet (Facebook's time series forecasting)
* **NLP**: Hugging Face Transformers
* **Explainability**: SHAP
* **Technical Analysis**: pandas-ta
* **Data**: yfinance, pandas, NumPy
* **Visualization**: Plotly

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Manish0729/AI-Market-Trend-Forecaster.git
cd AI-Market-Trend-Forecaster

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
cd stock_forecaster
pip install -r requirements.txt
```

## 🎯 Usage

### Run Locally

```bash
cd stock_forecaster
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Using the Launch Script

```bash
# From project root
./run_all.sh
```

This launches both the homepage (port 5500) and Streamlit app (port 8502)

## 🌐 Live Demo

Visit the deployed app on Streamlit Cloud: [AI Market Trend Forecaster](https://ai-market-trend-forecaster.streamlit.app)

## 📊 How It Works

1. **Select a ticker** from the dropdown or enter a custom symbol
2. **Choose forecast horizon** (7-180 days)
3. **Select sentiment windows** for model comparison
4. **Click "Find Best Model & Run Forecast"**
5. **Review results** including:
   - Performance metrics comparison
   - Price forecasts with confidence intervals
   - Technical indicator overlays (RSI, MACD, SMA)
   - SHAP explainability charts

## 📁 Project Structure

```
AI-Market-Trend-Forecaster/
├── stock_forecaster/
│   ├── app.py                    # Main Streamlit application
│   ├── requirements.txt          # Python dependencies
│   └── src/
│       ├── data_handler.py       # Data fetching and processing
│       ├── sentiment_analyzer.py # News sentiment analysis
│       └── model_trainer.py      # Prophet model training and SHAP
├── index.html                    # Landing page
├── favicon.svg                   # App icon
├── Procfile                      # Heroku/Streamlit deployment config
├── runtime.txt                   # Python version specification
└── README.md                     # This file
```

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market investments carry risk. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Built with ❤️ using Prophet, Hugging Face, and Streamlit

