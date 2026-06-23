"""
forecasting.py
==============
Sales forecasting module using Scikit-Learn Linear Regression.

Pipeline:
  1. Aggregate net_revenue by month (from feature-engineered data)
  2. Encode time as numeric features (month index, year, month-of-year)
  3. Train / test split (80 / 20)
  4. Fit LinearRegression
  5. Return model, metrics, and forecast for next N months

Author : Sales Analytics Dashboard
Purpose: Portfolio / Internship project
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics       import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_monthly_ts(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("year_month")
        .agg(total_revenue=("net_revenue", "sum"))
        .reset_index()
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    monthly["t"]         = np.arange(len(monthly))           # numeric time index
    monthly["month_num"] = pd.to_datetime(
        monthly["year_month"], format="%Y-%m"
    ).dt.month
    monthly["year"]      = pd.to_datetime(
        monthly["year_month"], format="%Y-%m"
    ).dt.year

    # Sine / cosine seasonality features
    monthly["sin_month"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
    monthly["cos_month"] = np.cos(2 * np.pi * monthly["month_num"] / 12)

    return monthly


FEATURE_COLS = ["t", "year", "sin_month", "cos_month"]
TARGET_COL   = "total_revenue"


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_forecast_model(df: pd.DataFrame, test_size: float = 0.2):
    monthly = _prepare_monthly_ts(df)

    X = monthly[FEATURE_COLS].values
    y = monthly[TARGET_COL].values

    # Chronological split (not random) to respect time order
    split_at  = max(1, int(len(X) * (1 - test_size)))
    X_train, X_test = X[:split_at], X[split_at:]
    y_train, y_test = y[:split_at], y[split_at:]
    train_idx = list(range(split_at))
    test_idx  = list(range(split_at, len(X)))

    # Standardise
    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test  = model.predict(X_test_s) if len(X_test_s) > 0 else np.array([])

    # Guard: if test set is empty, compute metrics on train
    y_eval  = y_test  if len(y_test)  > 0 else y_train
    yp_eval = y_pred_test if len(y_pred_test) > 0 else y_pred_train

    mae  = mean_absolute_error(y_eval, yp_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, yp_eval))
    r2   = r2_score(y_eval, yp_eval)

    metrics = {
        "MAE":  round(mae,  2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2,   4),
    }

    print(f"[forecasting] Model metrics -> MAE: {mae:,.0f} | "
          f"RMSE: {rmse:,.0f} | R2: {r2:.4f}")

    return {
        "model":         model,
        "scaler":        scaler,
        "monthly":       monthly,
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "y_pred_train":  y_pred_train,
        "y_pred_test":   y_pred_test,
        "metrics":       metrics,
        "train_idx":     train_idx,
        "test_idx":      test_idx,
    }


# ---------------------------------------------------------------------------
# Future forecast
# ---------------------------------------------------------------------------

def forecast_future(result: dict, n_months: int = 6) -> pd.DataFrame:
    monthly = result["monthly"]
    model   = result["model"]
    scaler  = result["scaler"]

    last_t          = int(monthly["t"].max())
    last_month_year = pd.to_datetime(monthly["year_month"].iloc[-1], format="%Y-%m")

    future_rows = []
    for i in range(1, n_months + 1):
        future_date = last_month_year + pd.DateOffset(months=i)
        t_val       = last_t + i
        month_num   = future_date.month
        year_val    = future_date.year
        sin_m       = np.sin(2 * np.pi * month_num / 12)
        cos_m       = np.cos(2 * np.pi * month_num / 12)
        future_rows.append({
            "year_month": future_date.strftime("%Y-%m"),
            "t":          t_val,
            "year":       year_val,
            "sin_month":  sin_m,
            "cos_month":  cos_m,
        })

    future_df = pd.DataFrame(future_rows)
    X_future  = future_df[FEATURE_COLS].values
    X_future_s = scaler.transform(X_future)
    future_df["predicted_revenue"] = model.predict(X_future_s)

    return future_df[["year_month", "predicted_revenue"]]


# ---------------------------------------------------------------------------
# Product-level forecasting (Hot Take prediction)
# ---------------------------------------------------------------------------

def _prepare_product_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data to monthly level per product."""
    monthly = (
        df.groupby(["year_month", "product"])
        .agg(total_revenue=("net_revenue", "sum"))
        .reset_index()
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    # Assign time index 't' for each month
    unique_months = sorted(monthly["year_month"].unique())
    month_to_t = {m: i for i, m in enumerate(unique_months)}
    monthly["t"] = monthly["year_month"].map(month_to_t)

    monthly["month_num"] = pd.to_datetime(
        monthly["year_month"], format="%Y-%m"
    ).dt.month
    monthly["year"] = pd.to_datetime(
        monthly["year_month"], format="%Y-%m"
    ).dt.year

    monthly["sin_month"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
    monthly["cos_month"] = np.cos(2 * np.pi * monthly["month_num"] / 12)

    return monthly

def train_product_forecast_models(df: pd.DataFrame):
    """Train a separate Linear Regression model for each product."""
    monthly = _prepare_product_ts(df)
    products = monthly["product"].unique()
    
    models = {}
    
    for prod in products:
        prod_df = monthly[monthly["product"] == prod]
        if len(prod_df) < 3: # Need at least some data to fit
            continue
            
        X = prod_df[FEATURE_COLS].values
        y = prod_df[TARGET_COL].values
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_s, y)
        
        models[prod] = {
            "model": model,
            "scaler": scaler,
            "last_t": int(prod_df["t"].max()),
            "last_month_year": pd.to_datetime(prod_df["year_month"].iloc[-1], format="%Y-%m")
        }
        
    return models, monthly

def analyze_product_growth(df: pd.DataFrame, n_months: int = 6) -> pd.DataFrame:
    """
    Train models, predict next n_months, and compare with the last n_months 
    to calculate projected growth % and identify 'hot takes'.
    """
    models, historical_monthly = train_product_forecast_models(df)
    
    results = []
    
    # 1. Calculate historical baseline (last 6 months of actuals)
    for prod, info in models.items():
        prod_hist = historical_monthly[historical_monthly["product"] == prod]
        # Get the last n_months of actual data
        recent_actuals = prod_hist.tail(n_months)
        past_revenue = recent_actuals["total_revenue"].sum()
        
        # 2. Predict future n_months
        model = info["model"]
        scaler = info["scaler"]
        last_t = info["last_t"]
        last_month = info["last_month_year"]
        
        future_rows = []
        for i in range(1, n_months + 1):
            future_date = last_month + pd.DateOffset(months=i)
            t_val = last_t + i
            month_num = future_date.month
            year_val = future_date.year
            sin_m = np.sin(2 * np.pi * month_num / 12)
            cos_m = np.cos(2 * np.pi * month_num / 12)
            future_rows.append([t_val, year_val, sin_m, cos_m])
            
        X_future = np.array(future_rows)
        X_future_s = scaler.transform(X_future)
        future_preds = model.predict(X_future_s)
        
        # Avoid negative predictions for revenue
        future_preds = np.maximum(future_preds, 0)
        future_revenue = future_preds.sum()
        
        # 3. Calculate growth
        if past_revenue > 0:
            growth_pct = ((future_revenue - past_revenue) / past_revenue) * 100
        else:
            growth_pct = 0.0
            
        results.append({
            "product": prod,
            "past_revenue": past_revenue,
            "predicted_revenue": future_revenue,
            "growth_pct": growth_pct
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("growth_pct", ascending=False).reset_index(drop=True)
    return res_df


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader          import load_raw_data
    from data_cleaning        import clean_data
    from feature_engineering  import engineer_features

    raw    = load_raw_data()
    clean  = clean_data(raw)
    feat   = engineer_features(clean)

    result = train_forecast_model(feat)
    print("Metrics:", result["metrics"])

    future = forecast_future(result, n_months=6)
    print("\nFuture forecast:")
    print(future.to_string())
