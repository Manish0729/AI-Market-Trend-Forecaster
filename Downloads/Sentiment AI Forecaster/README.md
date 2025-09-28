# Sentiment-Augmented Market Trend Forecaster

A sophisticated AI-powered stock market forecasting application that combines historical price data, sentiment analysis of news headlines, and advanced technical indicators to predict future market trends. The application uses machine learning models to provide accurate forecasts with explainable AI insights, making it a powerful tool for traders and investors.

## Key Features

- **Automatic Model Showdown**: Tests multiple sentiment window configurations and automatically selects the best-performing model based on cross-validation metrics
- **Multi-Indicator Analysis**: Integrates RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), and SMA (Simple Moving Average) for comprehensive technical analysis
- **Sentiment Intelligence**: Analyzes news headlines using Hugging Face's state-of-the-art NLP models to gauge market sentiment
- **Explainable AI (SHAP)**: Provides transparent insights into model predictions, showing which factors most influence the forecasts
- **Interactive Dashboard**: Clean, professional Streamlit interface with real-time charts and performance metrics
- **Comprehensive Asset Coverage**: Supports Indian stocks, US markets, forex pairs, commodities, and cryptocurrencies
- **Advanced Forecasting**: Uses Facebook's Prophet model with custom regressors for robust time series predictions

## Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Prophet (Facebook's time series forecasting)
- **NLP & Sentiment**: Hugging Face Transformers
- **Explainable AI**: SHAP (SHapley Additive exPlanations)
- **Technical Analysis**: pandas-ta
- **Data Processing**: pandas, NumPy
- **Visualization**: Plotly
- **Data Source**: yfinance (Yahoo Finance API)
- **Language**: Python 3.11+

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd AI-for-Market-Trend-Analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. **Install dependencies**:
   ```bash
   cd stock_forecaster
   pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```bash
   streamlit --version
   ```

## How to Run

To launch the application, navigate to the `stock_forecaster` directory and run:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

### Alternative Launch Methods

- **Using the provided script** (from project root):
  ```bash
  ./run_all.sh
  ```
  This will launch both the homepage and the Streamlit app.

- **Quick restart** (from stock_forecaster directory):
  ```bash
  ./start_app.sh
  ```

## Usage

1. **Select a ticker** from the dropdown or enter a custom one
2. **Choose forecast horizon** (7-180 days)
3. **Select sentiment windows** for model comparison
4. **Click "Find Best Model & Run Forecast"**
5. **Review results** including performance metrics, forecasts, and SHAP explanations

## Project Structure

```
AI-for-Market-Trend-Analysis/
├── stock_forecaster/
│   ├── app.py                 # Main Streamlit application
│   ├── requirements.txt       # Python dependencies
│   ├── README.md             # Project documentation
│   └── src/
│       ├── __init__.py
│       ├── data_handler.py    # Data fetching and processing
│       ├── sentiment_analyzer.py  # News sentiment analysis
│       └── model_trainer.py   # Prophet model training and SHAP
├── index.html                # Landing page
├── favicon.svg              # Website icon
├── run_all.sh              # Launch script
└── README.md               # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Disclaimer

This application is for educational and research purposes only. Stock market investments carry risk, and past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.
