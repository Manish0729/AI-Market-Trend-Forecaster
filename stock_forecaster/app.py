from __future__ import annotations

"""Streamlit app: Sentiment-Augmented Stock Trend Forecaster.

This app downloads 2 years of OHLCV data from yfinance, generates simulated
news headlines, analyzes sentiment with a Hugging Face pipeline, and trains a
Prophet model that uses a 7-day rolling sentiment signal as a regressor to
forecast the next 90 days of closing prices.
"""

import warnings
from datetime import timezone
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_handler import load_data
from src.sentiment_analyzer import analyze_sentiment
from src.model_trainer import find_best_model_and_forecast


# Global determinism for synthetic operations
np.random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)


st.set_page_config(page_title="Sentiment-Augmented Stock Trend Forecaster", layout="wide")

# Lightweight, theme-friendly CSS for a simple hero + card feel
st.markdown(
    """
    <style>
      .sf-hero{ text-align:center; padding: 6px 0 10px; }
      .sf-hero h1{ margin:0; font-size: 1.8rem; letter-spacing: -0.01em; }
      .sf-hero p{ margin:.35rem 0 0; color: var(--text-color-secondary, #a7b0c2); }
      .sf-section-title{ font-weight:700; margin: 4px 0 10px; font-size: 1.1rem; }
      /* Sticky footer styles */
      .sf-footer{
        position: fixed;
        left: 0; bottom: 0; width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #262730;
        z-index: 9999;
      }
      /* Ensure main content doesn't get hidden behind footer */
      main .block-container{ padding-bottom: 72px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sf-hero">
      <h1>📈 Sentiment-Augmented Stock Trend Forecaster</h1>
      <p><em>AI + Technical Indicators for Smarter Market Insights</em></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar toggle for RSI overlay (secondary Y axis)
show_rsi_overlay = st.sidebar.checkbox("Overlay RSI (0-100) on Forecast Chart", value=False)
show_macd_overlay = st.sidebar.checkbox("Overlay MACD on Forecast Chart", value=False)
overlay_sma = st.sidebar.checkbox("Overlay SMA (50-Day) on Chart", value=False)


@st.cache_data(show_spinner=False)
def cached_load_data(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stock_df, news_df = load_data(ticker)
    # Compute RSI_14 using pandas-ta and attach to stock_df
    try:
        import pandas_ta as ta  # lazy import to avoid hard failure on environments with numpy 2
        stock_df = stock_df.copy()
        stock_df.ta.rsi(length=14, append=True)
        if "RSI_14" not in stock_df.columns:
            # Normalize any RSI column name to RSI_14
            for col in stock_df.columns:
                if str(col).lower().startswith("rsi"):
                    stock_df.rename(columns={col: "RSI_14"}, inplace=True)
                    break
        # Compute MACD set (MACD, Signal, Histogram)
        stock_df.ta.macd(append=True)
        # Normalize column names if library changes minor naming
        if "MACD_12_26_9" not in stock_df.columns:
            macd_cols = [c for c in stock_df.columns if str(c).upper().startswith("MACD_") and "_" in str(c)]
            if macd_cols:
                stock_df.rename(columns={macd_cols[0]: "MACD_12_26_9"}, inplace=True)
        if "MACDs_12_26_9" not in stock_df.columns:
            sig_cols = [c for c in stock_df.columns if str(c).upper().startswith("MACDS_")]
            if sig_cols:
                stock_df.rename(columns={sig_cols[0]: "MACDs_12_26_9"}, inplace=True)
        if "MACDh_12_26_9" not in stock_df.columns:
            hist_cols = [c for c in stock_df.columns if str(c).upper().startswith("MACDH_")]
            if hist_cols:
                stock_df.rename(columns={hist_cols[0]: "MACDh_12_26_9"}, inplace=True)
        # 50-day Simple Moving Average
        stock_df.ta.sma(length=50, append=True)
        if "SMA_50" not in stock_df.columns:
            sma_cols = [c for c in stock_df.columns if str(c).upper().startswith("SMA_50") or str(c).upper()=="SMA_50"]
            if sma_cols:
                stock_df.rename(columns={sma_cols[0]: "SMA_50"}, inplace=True)
        # Guarantee SMA_50 exists
        if "SMA_50" not in stock_df.columns:
            stock_df["SMA_50"] = pd.to_numeric(stock_df["close"], errors="coerce").rolling(50).mean()
    except Exception:
        # Fallback minimal RSI if ta is unavailable
        close = pd.to_numeric(stock_df["close"], errors="coerce")
        delta = close.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        stock_df["RSI_14"] = rsi.fillna(method="bfill").fillna(method="ffill")
        # Fallback MACD (12, 26, 9) using EMA
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        stock_df["MACD_12_26_9"] = macd
        stock_df["MACDs_12_26_9"] = signal
        stock_df["MACDh_12_26_9"] = hist
        # Fallback SMA(50)
        stock_df["SMA_50"] = close.rolling(window=50, min_periods=1).mean()

    return stock_df, news_df


@st.cache_data(show_spinner=False)
def cached_analyze_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    return analyze_sentiment(news_df)





def _compute_sentiment_7d(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge prices with daily sentiment and compute 7-day rolling sentiment."""

    left = stock_df[["date", "close"]].copy()
    left["date"] = pd.to_datetime(left["date"], utc=True).dt.floor("D")
    right = sentiment_df.copy()
    right["date"] = pd.to_datetime(right["date"], utc=True).dt.floor("D")

    merged = pd.merge(left, right, on="date", how="left").sort_values("date")
    merged["sentiment_score"] = merged["sentiment_score"].ffill().bfill().fillna(0.0)
    merged["sentiment_7d"] = merged["sentiment_score"].rolling(7, min_periods=1).mean()
    return merged


default_tickers = [
    # Indian Indices & Stocks
    "^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "^BSESN",
    # US Stocks & Indices
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "^GSPC", "^IXIC",
    # Forex, Commodities & Crypto
    "EURUSD=X", "USDINR=X", "GC=F", "CL=F", "BTC-USD"
]

# Main page controls (like earlier)
ctl1, ctl2, ctl3 = st.columns([2, 2, 3])
with ctl1:
    base_choice = st.selectbox("Choose Ticker", default_tickers, index=0)
with ctl2:
    custom_ticker = st.text_input("Or enter custom ticker (overrides)", value="")
with ctl3:
    horizon = st.slider(
        "Select Forecast Horizon (Days)", min_value=7, max_value=180, value=30, step=7
    )

ctl4, ctl5, ctl6 = st.columns([2, 2, 2])
with ctl4:
    windows = st.multiselect(
        "Select Sentiment Windows (Days)", options=[7, 15, 30], default=[7, 15]  # Default to 7 and 15 days for better accuracy
    )
with ctl5:
    overlay_sent = st.checkbox("Overlay 7D Sentiment on Forecast", value=False)
with ctl6:
    trigger_showdown = st.button("Find Best Model & Run Forecast")

# Final ticker resolution
ticker = (custom_ticker.strip().upper() or base_choice)

# Initialize variables for SHAP display
shap_figure = None

if trigger_showdown:
    if not ticker:
        st.error("Please enter a valid ticker symbol.")
        st.stop()

    try:
        with st.spinner("Running analysis... This may take a few minutes"):
            stock_df, news_df = cached_load_data(ticker)
            sentiment_df = cached_analyze_sentiment(news_df)

        candidate_windows = windows or [7, 15, 30]

        # Progress bar for selection process
        st.write("Model selection in progress...")
        progress = st.progress(0)
        progress.progress(5)

        with st.spinner("Finding best model — this may take several minutes..."):
            perf_df, forecast, best_model, shap_figure = find_best_model_and_forecast(
                stock_df,
                sentiment_df,
                windows=candidate_windows,
                horizon=int(horizon),
            )
        progress.progress(100)

    except Exception as exc:  # Robustness for network/model errors
        import traceback
        st.error(
            f"Operation failed: {exc}\n\n"
            "Tips: Check your internet connection, retry with another ticker, or try again later."
        )
        with st.expander("Show error details"):
            st.code("\n".join(traceback.format_exc().splitlines()[-200:]))
        st.stop()

    st.markdown("<div class='sf-section-title'>🔍 Data Snapshots</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Stock OHLCV (head)", expanded=False):
            st.dataframe(stock_df.head(10), width="stretch")
    with c2:
        with st.expander("Sample News (head)", expanded=False):
            st.dataframe(news_df.head(10), width="stretch")

    # Build helper frame with 7d sentiment for metrics and plot 1
    merged = _compute_sentiment_7d(stock_df, sentiment_df)

    # Plot 1: History + daily sentiment (secondary y)
    st.markdown("### 📊 Stock Price vs Daily Sentiment")
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(
        go.Scatter(
            x=merged["date"], y=merged["close"], name="Close", mode="lines", line=dict(color="#1f77b4")
        ),
        secondary_y=False,
    )
    fig1.add_trace(
        go.Bar(
            x=merged["date"], y=merged["sentiment_score"], name="Daily Sentiment", marker_color="#ff7f0e", opacity=0.5
        ),
        secondary_y=True,
    )
    fig1.update_layout(
        title=f"{ticker} Price vs. Daily Sentiment",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=450,
    )
    fig1.update_yaxes(title_text="Close Price", secondary_y=False)
    fig1.update_yaxes(title_text="Sentiment", secondary_y=True, range=[-1, 1])
    st.plotly_chart(fig1, width="stretch")

    # Model Performance Comparison
    st.markdown("<div class='sf-section-title'>⚡ Model Performance Comparison</div>", unsafe_allow_html=True)
    # Aggregate by window to show a concise summary
    try:
        perf_summary = (
            perf_df.groupby("window", as_index=False)[["rmse", "mape", "mae"]].mean().sort_values("rmse")
        )
        # Highlight the best (lowest rmse)
        styler = (
            perf_summary.style.highlight_min(subset=["rmse"], color="#163b2e")
            .format({"rmse": "{:.3f}", "mape": "{:.4f}", "mae": "{:.3f}"})
        )
        st.dataframe(styler, width="stretch")
    except Exception:
        st.dataframe(perf_df.sort_values("rmse"), width="stretch")

    # Forecast from Best Model
    st.markdown("<div class='sf-section-title'>📊 Forecast Results</div>", unsafe_allow_html=True)
    # Plot 2: Forecast with uncertainty + history
    hist = stock_df.copy()
    hist["date"] = pd.to_datetime(hist["date"], utc=True)
    # Build figure: if MACD overlay requested, add a second subplot
    if show_macd_overlay:
        forecast_plot = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            row_heights=[0.7, 0.3],
            vertical_spacing=0.08,
        )
    else:
        forecast_plot = make_subplots(specs=[[{"secondary_y": True}]])

    # Uncertainty band
    forecast_plot.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="Upper",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    forecast_plot.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            fillcolor="rgba(31,119,180,0.15)",
            showlegend=False,
            hoverinfo="skip",
            name="Lower",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    # Forecast line
    forecast_plot.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Forecast",
            mode="lines",
            line=dict(color="#2ca02c"),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    # Actual closes
    forecast_plot.add_trace(
        go.Scatter(
            x=hist["date"], y=hist["close"], name="Close", mode="lines", line=dict(color="#1f77b4")
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    # Optional SMA(50) overlay on primary axis
    if overlay_sma and "SMA_50" in stock_df.columns:
        forecast_plot.add_trace(
            go.Scatter(
                x=hist["date"],
                y=stock_df["SMA_50"],
                name="SMA 50",
                mode="lines",
                line=dict(color="#FFA500"),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    # Optional overlay of 7D sentiment (scaled to price range)
    if overlay_sent:
        merged_for_overlay = _compute_sentiment_7d(stock_df, sentiment_df)
        y_min = float(hist["close"].min())
        y_max = float(hist["close"].max())
        scale = (y_max - y_min) / 4.0
        baseline = y_min + (y_max - y_min) * 0.1
        overlay_series = baseline + merged_for_overlay["sentiment_7d"] * scale
        forecast_plot.add_trace(
            go.Scatter(
                x=merged_for_overlay["date"],
                y=overlay_series,
                name="Sentiment 7D (scaled)",
                mode="lines",
                line=dict(color="#9467bd", dash="dot"),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # Optional RSI overlay on secondary y-axis
    if show_rsi_overlay and "RSI_14" in stock_df.columns:
        forecast_plot.add_trace(
            go.Scatter(
                x=hist["date"],
                y=stock_df["RSI_14"],
                name="RSI",
                mode="lines",
                line=dict(color="#FFD700", dash="dash"),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        # Add guide bands at RSI 30/70 on the secondary Y-axis
        x0 = pd.to_datetime(forecast["ds"]).min()
        x1 = pd.to_datetime(forecast["ds"]).max()
        for level in (30, 70):
            forecast_plot.add_shape(
                type="line",
                x0=x0,
                x1=x1,
                y0=level,
                y1=level,
                xref="x",
                yref="y2",
                line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
            )

    # Optional MACD subplot
    if show_macd_overlay and {"MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"}.issubset(set(stock_df.columns)):
        # Histogram bars
        forecast_plot.add_trace(
            go.Bar(
                x=hist["date"],
                y=stock_df["MACDh_12_26_9"],
                name="MACD Hist",
                marker_color="rgba(100, 149, 237, 0.5)",
            ),
            row=2,
            col=1,
        )
        # MACD line
        forecast_plot.add_trace(
            go.Scatter(
                x=hist["date"],
                y=stock_df["MACD_12_26_9"],
                name="MACD",
                mode="lines",
                line=dict(color="#00CED1"),
            ),
            row=2,
            col=1,
        )
        # Signal line
        forecast_plot.add_trace(
            go.Scatter(
                x=hist["date"],
                y=stock_df["MACDs_12_26_9"],
                name="Signal",
                mode="lines",
                line=dict(color="#FF6347"),
            ),
            row=2,
            col=1,
        )

    forecast_plot.update_layout(
        title=f"📈 Forecast with Technical Indicators (Prophet + MACD + RSI + SMA)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=650 if show_macd_overlay else 500,
    )
    forecast_plot.update_yaxes(title_text="Close Price", secondary_y=False)
    if show_rsi_overlay:
        forecast_plot.update_yaxes(title_text="RSI (0-100)", secondary_y=True, range=[0, 100])
    if show_macd_overlay:
        forecast_plot.update_yaxes(title_text="MACD", row=2, col=1)
    st.plotly_chart(forecast_plot, width="stretch")

    # Day-by-day forecast table just below the chart
    st.subheader("Day-by-Day Forecast Breakdown")
    try:
        last_hist = pd.to_datetime(stock_df["date"]).dt.tz_localize(None).max()
        ds_series = pd.to_datetime(forecast["ds"])  # tz-naive already
        future_only = forecast.loc[ds_series > last_hist, ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_table = future_only.rename(
            columns={
                "ds": "Date",
                "yhat": "Predicted Price",
                "yhat_lower": "Predicted Low",
                "yhat_upper": "Predicted High",
            }
        )
        # Optional: round for readability
        for col in ["Predicted Price", "Predicted Low", "Predicted High"]:
            if col in forecast_table.columns:
                forecast_table[col] = pd.to_numeric(forecast_table[col], errors="coerce").round(2)
        st.dataframe(forecast_table, width="stretch")
    except Exception:
        # Non-fatal; continue rendering the rest of the UI
        pass

    # KPIs
    k1, k2, k3 = st.columns(3)
    last_close = float(merged["close"].iloc[-1])
    last_sent7 = float(merged["sentiment_7d"].iloc[-1])
    last_rsi = float(stock_df["RSI_14"].dropna().iloc[-1]) if "RSI_14" in stock_df.columns else float("nan")
    last_macd = float(stock_df["MACD_12_26_9"].dropna().iloc[-1]) if "MACD_12_26_9" in stock_df.columns else float("nan")

    # Normalize datetime types before comparison
    # - forecast['ds'] is already tz-naive; ensure dtype datetime64[ns]
    ds_naive = pd.to_datetime(forecast["ds"])  # tz-naive
    # - merged['date'] is UTC-aware; convert column to tz-naive and take last
    hist_dates_naive = pd.to_datetime(merged["date"]).dt.tz_localize(None)
    last_hist_date = hist_dates_naive.iloc[-1]
    next_30_mask = ds_naive > last_hist_date
    _n = int(min(30, int(horizon)))
    next_30_avg = float(forecast.loc[next_30_mask, "yhat"].head(_n).mean())

    k1.metric("Last Close", f"{last_close:,.2f}")
    k2.metric("Last RSI (14)", f"{last_rsi:,.1f}")
    k3.metric(f"Next {_n}D Avg Forecast", f"{next_30_avg:,.2f}")
    st.caption(f"Latest 7D sentiment avg: {last_sent7:+.3f} | Last MACD: {last_macd:+.3f}")

    # Show performance table if available (from showdown)
    if 'perf_df' in locals():
        st.markdown("Best model selection (lower RMSE is better):")
        st.dataframe(perf_df[["window", "rmse"]].sort_values("rmse"), width="stretch")

    # Download CSV of forecast
    csv_bytes = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Forecast CSV",
        data=csv_bytes,
        file_name=f"{ticker}_forecast_{horizon}d.csv",
        mime="text/csv",
    )

    # Explainability section (SHAP)
    with st.expander("Why This Forecast? (Explainable AI Insights)", expanded=False):
        st.write(
            "This chart shows the impact of each feature on the final forecast. "
            "Positive values indicate upward influence; negative values indicate downward influence."
        )
        if shap_figure is not None:
            st.pyplot(shap_figure, use_container_width=True)
        else:
            st.info("SHAP explanation is unavailable for this run (no regressors or SHAP not installed).")


else:
    st.info("Use the controls above to configure the ticker, horizon and windows, then click 'Find Best Model & Run Forecast'.")

# Small footer for branding
st.markdown(
    """
    <div class="sf-footer">
      🚀 Built with Prophet, Hugging Face, and pandas‑ta | Demo Project (2025)
    </div>
    """,
    unsafe_allow_html=True,
)









