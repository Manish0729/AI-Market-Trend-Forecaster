"""News sentiment analysis utilities using Hugging Face transformers.

This module exposes a function to analyze per-headline sentiment using the
`distilbert-base-uncased-finetuned-sst-2-english` model and aggregates the
results to daily mean sentiment scores in the range [-1, 1].

If the model download or inference fails (e.g., offline), the function falls
back to neutral sentiment (0.0) to keep the pipeline operational.

Python version: 3.10+
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

try:  # Lazy/deferred imports to speed module import
    import torch
    from transformers import pipeline
except Exception:  # pragma: no cover - handled gracefully at runtime
    torch = None  # type: ignore
    pipeline = None  # type: ignore


def _select_device() -> int:
    """Return device index for transformers pipeline (0 for CUDA, -1 for CPU)."""

    if torch is not None:
        try:
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
    return -1


def _to_numeric_scores(labels: List[str], probs: List[float]) -> np.ndarray:
    """Convert class labels to signed scores in [-1, 1].

    POSITIVE -> +prob, NEGATIVE -> -prob.
    """

    signs = np.array([1.0 if (lbl.upper().startswith("POS")) else -1.0 for lbl in labels], dtype=float)
    prob_arr = np.clip(np.array(probs, dtype=float), 0.0, 1.0)
    return signs * prob_arr


def analyze_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze sentiment of headlines and aggregate to daily mean.

    Args:
        news_df: DataFrame with columns ["date", "headline"].

    Returns:
        DataFrame with columns ["date", "sentiment_score"] containing one row per
        calendar date present in `news_df` and the mean sentiment score for that date.
    """

    if news_df is None or news_df.empty:
        return pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns, UTC]"), "sentiment_score": pd.Series([], dtype=float)})

    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")
    df["headline"] = df["headline"].astype(str)

    device = _select_device()

    # Attempt model inference; fallback to neutral sentiment on failures
    scores: np.ndarray
    try:
        if pipeline is None:
            raise RuntimeError("transformers not available in this environment")

        clf = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        )

        headlines: List[str] = df["headline"].tolist()
        batch_size: int = 32
        labels: List[str] = []
        probs: List[float] = []

        for start in range(0, len(headlines), batch_size):
            batch = headlines[start : start + batch_size]
            outputs = clf(batch, truncation=True)
            for out in outputs:
                labels.append(str(out.get("label", "POSITIVE")))
                probs.append(float(out.get("score", 0.0)))

        scores = _to_numeric_scores(labels, probs)
    except Exception:
        # Neutral fallback ensures the app remains usable offline
        scores = np.zeros(len(df), dtype=float)

    df["sentiment_score"] = scores

    daily = (
        df[["date", "sentiment_score"]]
        .groupby("date", as_index=False)
        .mean(numeric_only=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily[["date", "sentiment_score"]]


__all__ = ["analyze_sentiment"]



