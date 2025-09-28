from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Try to import Prophet and SHAP, fallback if not available
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

def find_best_model_and_forecast(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    windows: List[int],
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, any, plt.Figure]:
    """
    Tests multiple feature configurations, evaluates them using Prophet's backtesting,
    selects the best model, retrains it, generates a forecast, and provides a SHAP explanation.
    """
    if not PROPHET_AVAILABLE:
        # Fallback: simple linear trend forecast
        return _fallback_forecast(stock_df, sentiment_df, windows, horizon)
    
    performance_records = []

    # --- Step 1: Model Showdown (Cross-Validation) ---
    for w in windows:
        try:
            sentiment_feat_name = f"sentiment_{w}d"
            sentiment_feat = (
                sentiment_df.set_index("date")["sentiment_score"]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index()
                .rename(columns={"sentiment_score": sentiment_feat_name})
            )
            
            merged_df = pd.merge(stock_df, sentiment_feat, on="date", how="left")
            merged_df = merged_df.ffill().bfill()

            df_prophet = pd.DataFrame({
                "ds": pd.to_datetime(merged_df["date"]).dt.tz_localize(None),  # Remove timezone
                "y": merged_df["close"],
                sentiment_feat_name: merged_df[sentiment_feat_name],
                "RSI_14": merged_df["RSI_14"],
                "MACD_12_26_9": merged_df["MACD_12_26_9"],
            })

            model = Prophet()
            model.add_regressor(sentiment_feat_name)
            model.add_regressor("RSI_14")
            model.add_regressor("MACD_12_26_9")
            model.fit(df_prophet)
            
            df_cv = cross_validation(model, initial="365 days", period="90 days", horizon="30 days", parallel="processes")
            df_p = performance_metrics(df_cv)
            performance_records.append({"window": w, "rmse": df_p["rmse"].mean()})

        except Exception as e:
            print(f"Failed to test window {w}: {e}")
            performance_records.append({"window": w, "rmse": float("inf")})

    if not performance_records:
        raise ValueError("Model selection failed for all windows.")

    # --- Step 2: Select and Retrain the Best Model ---
    performance_df = pd.DataFrame(performance_records).sort_values(by="rmse").reset_index(drop=True)
    best_window = int(performance_df.iloc[0]["window"])

    best_sentiment_feat_name = f"sentiment_{best_window}d"
    best_sentiment = (
        sentiment_df.set_index("date")["sentiment_score"]
        .rolling(window=best_window, min_periods=1)
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": best_sentiment_feat_name})
    )
    
    final_merged_df = pd.merge(stock_df, best_sentiment, on="date", how="left")
    final_merged_df = final_merged_df.ffill().bfill()

    final_train_df = pd.DataFrame({
        "ds": pd.to_datetime(final_merged_df["date"]).dt.tz_localize(None),  # Remove timezone
        "y": final_merged_df["close"],
        best_sentiment_feat_name: final_merged_df[best_sentiment_feat_name],
        "RSI_14": final_merged_df["RSI_14"],
        "MACD_12_26_9": final_merged_df["MACD_12_26_9"],
    })

    best_model = Prophet()
    best_model.add_regressor(best_sentiment_feat_name)
    best_model.add_regressor("RSI_14")
    best_model.add_regressor("MACD_12_26_9")
    best_model.fit(final_train_df)

    # --- Step 3: Generate the Final Forecast ---
    future = best_model.make_future_dataframe(periods=horizon, freq="D")
    for col in best_model.extra_regressors:
        future[col] = final_train_df[col].iloc[-1]
    final_forecast = best_model.predict(future)

    # --- Step 4: Generate SHAP Explanation ---
    shap_figure = None
    if SHAP_AVAILABLE:
        try:
            regressor_columns = list(best_model.extra_regressors)
            train_regressors = final_train_df[regressor_columns]

            def predict_model_for_shap(X):
                X_df = pd.DataFrame(X, columns=regressor_columns)
                # Create a DataFrame with the same number of rows as X
                predict_df = pd.DataFrame()
                # Use a subset of dates that matches the input size
                n_rows = len(X_df)
                predict_df['ds'] = final_train_df['ds'].iloc[:n_rows].reset_index(drop=True)
                for col in regressor_columns:
                    predict_df[col] = X_df[col].values
                # Ensure ds column is timezone-naive
                predict_df['ds'] = pd.to_datetime(predict_df['ds']).dt.tz_localize(None)
                return best_model.predict(predict_df)['yhat'].values

            explainer = shap.KernelExplainer(predict_model_for_shap, shap.kmeans(train_regressors, 25))
            
            data_to_explain = train_regressors.sample(n=50, random_state=42)
            shap_values = explainer.shap_values(data_to_explain)

            plt.style.use('dark_background')
            plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
            shap.summary_plot(shap_values, data_to_explain, plot_type="bar", show=False)
            shap_figure = plt.gcf()
            plt.close()
        except Exception as e:
            print(f"SHAP explanation failed: {e}")

    return performance_df, final_forecast, best_model, shap_figure


def _fallback_forecast(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    windows: List[int],
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, any, plt.Figure]:
    """
    Fallback forecast when Prophet is not available.
    Uses simple linear trend with sentiment overlay.
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # Simple performance comparison
    performance_records = []
    for w in windows:
        try:
            # Calculate sentiment feature
            sentiment_feat = (
                sentiment_df.set_index("date")["sentiment_score"]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index()
                .rename(columns={"sentiment_score": f"sentiment_{w}d"})
            )
            
            merged_df = pd.merge(stock_df, sentiment_feat, on="date", how="left")
            merged_df = merged_df.ffill().bfill()
            
            # Simple linear regression
            X = merged_df[["RSI_14", "MACD_12_26_9", f"sentiment_{w}d"]].fillna(0)
            y = merged_df["close"]
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Simple RMSE calculation
            y_pred = model.predict(X)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            performance_records.append({"window": w, "rmse": rmse})
            
        except Exception as e:
            print(f"Failed to test window {w}: {e}")
            performance_records.append({"window": w, "rmse": float("inf")})
    
    # Select best window
    performance_df = pd.DataFrame(performance_records).sort_values(by="rmse").reset_index(drop=True)
    best_window = int(performance_df.iloc[0]["window"])
    
    # Create final model
    sentiment_feat = (
        sentiment_df.set_index("date")["sentiment_score"]
        .rolling(window=best_window, min_periods=1)
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": f"sentiment_{best_window}d"})
    )
    
    merged_df = pd.merge(stock_df, sentiment_feat, on="date", how="left")
    merged_df = merged_df.ffill().bfill()
    
    X = merged_df[["RSI_14", "MACD_12_26_9", f"sentiment_{best_window}d"]].fillna(0)
    y = merged_df["close"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate forecast
    last_date = pd.to_datetime(stock_df["date"]).max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    
    # Use last known values for features
    last_features = X.iloc[-1].values
    future_features = np.tile(last_features, (horizon, 1))
    
    future_predictions = model.predict(future_features)
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions,
        'yhat_lower': future_predictions * 0.95,  # Simple uncertainty
        'yhat_upper': future_predictions * 1.05
    })
    
    return performance_df, forecast_df, model, None