"""Data loading and sample news generation utilities.

This module provides a function to fetch historical OHLCV stock data using
`yfinance` and generates a simulated news headlines DataFrame covering the same
date range. The news is synthetic but ticker-aware to enable sentiment
experiments without relying on external news APIs.

All returned dataframes include a normalized `date` column in UTC timezone.

Python version: 3.10+
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import time
import requests

# Set deterministic seeds for reproducibility of synthetic news sampling
np.random.seed(42)


@dataclass
class LoadResult:
    """Container for loaded stock and news data.

    Attributes:
        stock_df: Historical OHLCV data with normalized columns and UTC dates.
        news_df: Simulated news headlines with UTC dates.
    """

    stock_df: pd.DataFrame
    news_df: pd.DataFrame


def _normalize_stock_columns(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Normalize yfinance columns and ensure UTC date column.

    Args:
        df: Raw DataFrame from yfinance with DatetimeIndex and OHLCV columns.

    Returns:
        DataFrame with columns: ["date","open","high","low","close","adj_close","volume"].
    """

    if df.empty:
        return df

    # Ensure index is timezone-aware UTC, then reset to a column named `date`
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df.index = df.index.tz_convert(timezone.utc)

    def _norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "_")

    known_fields = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "adj_close": "adj_close",
        "volume": "volume",
        "adj_close": "adj_close",
        "adj_close": "adj_close",
    }

    tmp = pd.DataFrame(index=df.index)

    if isinstance(df.columns, pd.MultiIndex):
        # Determine which level contains field names
        level0 = [_norm(x) for x in df.columns.get_level_values(0)]
        level1 = [_norm(x) for x in df.columns.get_level_values(1)]
        fields_in_lvl0 = any(x in known_fields for x in level0)
        fields_in_lvl1 = any(x in known_fields for x in level1)

        field_level = 0 if fields_in_lvl0 else 1
        ticker_level = 1 - field_level

        # Choose ticker column
        chosen_ticker = None
        if symbol is not None:
            symbols = list(df.columns.get_level_values(ticker_level).unique())
            for s in symbols:
                if _norm(s) == _norm(symbol):
                    chosen_ticker = s
                    break
        if chosen_ticker is None:
            chosen_ticker = df.columns.get_level_values(ticker_level).unique()[0]

        # For each desired field, find matching label and extract the series
        for raw_label in set(df.columns.get_level_values(field_level)):
            key = _norm(raw_label).replace("adj_close", "adjclose") if _norm(raw_label) == "adj_close" else _norm(raw_label)
        
        def _find_field(label_set) -> dict[str, object]:
            mapping: dict[str, object] = {}
            for lbl in label_set:
                n = _norm(lbl)
                if n in ("adj_close", "adjclose", "adj_close"):
                    mapping["adj_close"] = lbl
                elif n in known_fields:
                    mapping[known_fields[n]] = lbl
            return mapping

        label_set = list(df.columns.get_level_values(field_level).unique())
        field_label_map = _find_field(label_set)

        for canonical, lbl in field_label_map.items():
            if field_level == 0:
                tmp[canonical] = df[(lbl, chosen_ticker)]
            else:
                tmp[canonical] = df[(chosen_ticker, lbl)]

        # Ensure we have adj_close; if missing, fallback to close
        if "adj_close" not in tmp.columns and "close" in tmp.columns:
            tmp["adj_close"] = tmp["close"]

        # Reset index to `date`
        tmp.index = (
            tmp.index.tz_localize(timezone.utc) if tmp.index.tz is None else tmp.index.tz_convert(timezone.utc)
        )
        df = tmp.reset_index().rename(columns={"index": "date", "Date": "date"})
    else:
        # Single-level columns
        rename = {}
        for c in df.columns:
            n = _norm(c)
            if n == "adj_close" or n == "adjclose" or n == "adj_close":
                rename[c] = "adj_close"
            elif n in known_fields:
                rename[c] = known_fields[n]
        tmp = df.rename(columns=rename)
        tmp.index = (
            tmp.index.tz_localize(timezone.utc) if tmp.index.tz is None else tmp.index.tz_convert(timezone.utc)
        )
        df = tmp.reset_index().rename(columns={"index": "date", "Date": "date"})

    # Enforce column order and dtypes
    wanted_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in wanted_cols:
        if col not in df.columns:
            # If Adj Close may be missing, fallback to Close
            if col == "adj_close" and "adj_close" not in df.columns and "close" in df.columns:
                df["adj_close"] = df["close"]
            else:
                df[col] = np.nan

    df = df[wanted_cols].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Cast numeric columns
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _generate_sample_headlines(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Generate synthetic, ticker-aware headlines between start and end dates.

    We aim for ~2-3 headlines per week over ~2 years (~120-300 headlines).

    Args:
        ticker: Stock ticker symbol, used in headline templates.
        start: Inclusive start date (UTC Timestamp).
        end: Inclusive end date (UTC Timestamp).

    Returns:
        DataFrame with columns ["date", "headline"]. Dates are UTC normalized
        to calendar days.
    """

    # Build weekly buckets; choose 2 or 3 random weekdays per week
    start_day = pd.Timestamp(start.date(), tz=timezone.utc)
    end_day = pd.Timestamp(end.date(), tz=timezone.utc)
    all_days = pd.date_range(start_day, end_day, freq="D", tz=timezone.utc)
    if len(all_days) == 0:
        return pd.DataFrame({"date": [], "headline": []})

    # Week start (Mon) grouping
    weeks = all_days.to_period("W-MON").unique()

    base_templates: List[str] = [
        f"{ticker} shares rally as analysts raise price targets",
        f"{ticker} dips after mixed earnings; guidance draws scrutiny",
        f"{ticker} unveils new product line amid strong demand",
        f"Regulatory chatter weighs on {ticker} ahead of key decision",
        f"Supply chain improvements boost {ticker}'s outlook",
        f"{ticker} expands into emerging markets, investors take note",
        f"Macro headwinds pressure tech; {ticker} remains resilient",
        f"{ticker} announces partnership to accelerate AI initiatives",
        f"Short sellers retreat as {ticker} posts margin gains",
        f"Dividend hike by {ticker} signals confidence in cash flows",
        f"{ticker} faces lawsuit; management reassures stakeholders",
        f"Buyback plan from {ticker} draws positive market reaction",
        f"{ticker} CFO commentary highlights prudent cost controls",
        f"Market rotation benefits {ticker} after sector slump",
        f"{ticker} volatility rises ahead of product launch",
    ]

    headlines: List[str] = []
    dates: List[pd.Timestamp] = []

    for wk in weeks:
        # Pick 2 or 3 days within the week, skew toward Tue-Thu
        per_week = 2 + int(np.random.rand() > 0.4)  # 2 or 3
        week_start: pd.Timestamp = wk.start_time
        week_days = pd.date_range(week_start, periods=7, freq="D", tz=timezone.utc)
        # Prefer business days Tue-Thu
        candidates = [d for d in week_days if d.weekday() in (1, 2, 3)] or list(week_days)
        chosen = list(np.random.choice(candidates, size=min(per_week, len(candidates)), replace=False))
        chosen.sort()

        for d in chosen:
            template = np.random.choice(base_templates)
            # Add small variations
            suffix = np.random.choice([
                "per analysts",
                "amid macro uncertainty",
                "on Wall Street chatter",
                "per management update",
                "as valuation debates intensify",
                "following investor day",
                "on broader market rally",
                "after product reviews",
                "amid supply constraints",
                "as growth accelerates",
            ])
            headline = f"{template} {suffix}."
            headlines.append(headline)
            dates.append(pd.Timestamp(d.date(), tz=timezone.utc))

    news_df = pd.DataFrame({"date": dates, "headline": headlines})
    news_df = news_df.sort_values("date").reset_index(drop=True)

    # Ensure at least ~120 rows. If not, duplicate with minor variation.
    while len(news_df) < 120:
        extra = news_df.sample(n=min(len(news_df), 60), replace=True, random_state=42).copy()
        extra["headline"] = extra["headline"] + " (update)"
        news_df = pd.concat([news_df, extra], ignore_index=True)
        news_df = news_df.sort_values("date").reset_index(drop=True)

    return news_df


def load_data(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load last 2 years of daily OHLCV data and generate sample news.

    Args:
        ticker: Ticker symbol to download from yfinance.

    Returns:
        Tuple of (stock_df, news_df).

        - stock_df columns: ["date","open","high","low","close","adj_close","volume"]
        - news_df columns: ["date","headline"] with ~120+ rows spread across the range
    """

    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")

    symbol = ticker.upper().strip()

    # Download last 2 years of daily data with multiple attempts
    raw = None
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # Try different parameters for better compatibility
            if attempt == 0:
                raw = yf.download(symbol, period="2y", interval="1d", auto_adjust=False, progress=False)
            elif attempt == 1:
                # Try with auto_adjust=True
                raw = yf.download(symbol, period="2y", interval="1d", auto_adjust=True, progress=False)
            else:
                # Try with a shorter period
                raw = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
            
            if raw is not None and not raw.empty:
                break
                
        except Exception as exc:
            if attempt == max_attempts - 1:
                # Last attempt failed
                raise RuntimeError(f"Failed to download data for {symbol} after {max_attempts} attempts: {exc}") from exc
            # Wait before retry
            time.sleep(1)
            continue
    
    if raw is None or raw.empty:
        # Suggest alternative tickers
        suggestions = {
            'BTC-USD': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            '^NSEI': ['^GSPC', '^DJI', 'AAPL', 'MSFT'],
            'TCS.NS': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'HDFCBANK.NS': ['JPM', 'BAC', 'WFC', 'AAPL']
        }
        
        alt_tickers = suggestions.get(symbol, ['AAPL', 'MSFT', 'GOOGL', 'TSLA', '^GSPC'])
        alt_list = ', '.join(alt_tickers)
        
        raise RuntimeError(
            f"No data returned for {symbol}. Try these alternatives: {alt_list}. "
            f"This may be due to market hours, data provider issues, or invalid ticker symbol."
        )

    stock_df = _normalize_stock_columns(raw, symbol=symbol)
    if stock_df.empty:
        raise RuntimeError(
            f"Downloaded frame for {symbol} is empty after normalization. Please retry later."
        )

    # Generate synthetic headlines over the available stock date window
    start_date = pd.to_datetime(stock_df["date"].min()).tz_convert(timezone.utc)
    end_date = pd.to_datetime(stock_df["date"].max()).tz_convert(timezone.utc)
    news_df = _generate_sample_headlines(symbol, start=start_date, end=end_date)

    # Final dtype normalization
    news_df["date"] = pd.to_datetime(news_df["date"], utc=True)
    news_df = news_df[["date", "headline"]].copy()

    return stock_df, news_df


__all__ = [
    "load_data",
]


