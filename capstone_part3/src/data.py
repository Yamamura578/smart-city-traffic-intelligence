"""Shared data loading, proxy-label construction and train/test splitting.

Every Part 3 module imports from here so that all models see an identical
feature matrix and an identical split. This is what makes the metric
comparisons in the report meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.config import FEATURES_CSV, LOW_VISIBILITY_WEATHER, RANDOM_STATE, SEVERE_WEATHER
from common.logging_config import get_logger

logger = get_logger(__name__)

# Feature set shared by the classifier, the regressor and the neural network.
# Time features, weather encodings, a holiday flag, and cyclical encodings of
# both hour and day of week, as the brief requires.
FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "is_precipitation", "is_low_visibility", "has_rain", "has_snow",
    "weather_Clear", "weather_Clouds", "weather_Drizzle", "weather_Fog",
    "weather_Haze", "weather_Mist", "weather_Other", "weather_Rain",
    "weather_Snow", "weather_Thunderstorm",
]


def load_features(path=FEATURES_CSV) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["date_time"])
    except FileNotFoundError:
        logger.error("Feature table not found at %s — run Part 2 first", path, exc_info=True)
        raise
    logger.info("Loaded feature table | rows=%d | columns=%d", len(df), df.shape[1])
    return df


def add_proxy_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the documented proxy accident-risk label.

    NO ACCIDENT DATA WAS SUPPLIED WITH THIS CAPSTONE. This label is a
    deterministic function of congestion and weather:

        high_risk = (congestion in {High, Severe}) AND
                    (weather is severe OR low visibility)

    It demonstrates the classification workflow. It is NOT a prediction of
    real accidents, and its limitations are discussed in the bias and
    fairness report.
    """
    df = df.copy()
    high_congestion = df["congestion_level"].isin(["High", "Severe"])
    risky_weather = (
        df["weather_main"].isin(SEVERE_WEATHER) | (df["is_low_visibility"] == 1)
    )
    df["high_risk"] = (high_congestion & risky_weather).astype(int)

    rate = df["high_risk"].mean()
    logger.info(
        "Created proxy accident-risk label | positive=%d (%.2f%%) | negative=%d",
        int(df["high_risk"].sum()), 100 * rate, int((1 - df["high_risk"]).sum()),
    )
    if rate < 0.05:
        logger.warning(
            "Proxy label is heavily imbalanced | positive rate=%.3f | "
            "accuracy will be a misleading metric; report recall and ROC AUC",
            rate,
        )
    return df


def get_xy(df: pd.DataFrame, target: str):
    """Return the shared feature matrix X and the requested target y."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Feature columns missing from dataset: %s", missing)
        raise KeyError(f"Missing feature columns: {missing}")
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[target]
    logger.info("Prepared X/y | features=%d | target='%s'", X.shape[1], target)
    return X, y


def split(X, y, stratify=None, test_size=0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )
    logger.info(
        "Split data | train=%d | test=%d | test_size=%.2f", len(X_tr), len(X_te), test_size
    )
    return X_tr, X_te, y_tr, y_te


def prepare(target: str, stratify_target: bool = False):
    """One call used by every model script: load, label, split."""
    df = add_proxy_risk_label(load_features())
    X, y = get_xy(df, target)
    strat = y if stratify_target else None
    return (df, X, y, *split(X, y, stratify=strat))
