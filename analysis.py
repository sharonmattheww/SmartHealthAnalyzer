from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def moving_average_forecast(series: pd.Series, steps: int = 5, window: int = 8) -> np.ndarray:
    clean_series = series.dropna().astype(float)
    if clean_series.empty:
        return np.array([0.0] * steps)

    values = clean_series.to_list()
    forecasts = []
    for _ in range(steps):
        lookback = values[-window:] if len(values) >= window else values
        next_value = float(np.mean(lookback))
        forecasts.append(next_value)
        values.append(next_value)
    return np.array(forecasts)


def rolling_regression_forecast(series: pd.Series, steps: int = 5, window: int = 12) -> np.ndarray:
    clean_series = series.dropna().astype(float).reset_index(drop=True)
    if len(clean_series) < 3:
        return moving_average_forecast(clean_series, steps=steps, window=window)

    recent = clean_series.tail(window).reset_index(drop=True)
    x = np.arange(len(recent))
    y = recent.values

    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(recent), len(recent) + steps)
    forecast = slope * future_x + intercept
    return np.clip(forecast, 0, 100)


def add_zscore_anomaly_flags(df: pd.DataFrame, cols: list[str], threshold: float = 2.0) -> pd.DataFrame:
    output = df.copy()
    if output.empty:
        output["anomaly_count"] = []
        output["is_anomaly"] = []
        return output

    anomaly_flags = []
    for col in cols:
        std = output[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            z_scores = np.zeros(len(output))
        else:
            z_scores = (output[col] - output[col].mean()) / std
        flag = np.abs(z_scores) >= threshold
        output[f"{col}_anomaly"] = flag
        anomaly_flags.append(flag.astype(int))

    output["anomaly_count"] = np.sum(np.vstack(anomaly_flags), axis=0)
    output["is_anomaly"] = output["anomaly_count"] > 0
    return output


def risk_projection_10min(df: pd.DataFrame, refresh_seconds: int = 2) -> Tuple[str, float]:
    if df.empty:
        return "Stable", 75.0

    recent = df["health_score"].tail(25).astype(float).reset_index(drop=True)
    if len(recent) < 3:
        projected = float(recent.iloc[-1])
    else:
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent.values, 1)
        future_steps = int(600 / max(refresh_seconds, 1))
        projected = float(np.clip(slope * (len(recent) + future_steps) + intercept, 0, 100))

    if projected < 45:
        return "High Risk", projected
    if projected < 70:
        return "Moderate Risk", projected
    return "Stable", projected


def build_heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    matrix = df.pivot_table(
        index="department",
        columns="campus_zone",
        values="health_score",
        aggfunc="mean",
    )
    return matrix.sort_index()
