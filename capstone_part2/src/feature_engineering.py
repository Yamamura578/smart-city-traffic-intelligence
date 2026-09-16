"""Part 2, Task 2 — build ML-ready features from the cleaned dataset.

Run from the repository root:

    python -m capstone_part2.src.feature_engineering
    python -m capstone_part2.src.feature_engineering --debug

Reads  data/processed/traffic_cleaned.csv   (output of pipeline.py)
Writes data/processed/traffic_features.csv  (input to Part 3)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common.config import (
    CLEANED_CSV,
    FEATURES_CSV,
    LOG_DIR,
    LOW_VISIBILITY_WEATHER,
    SEVERE_WEATHER,
)
from common.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Weather categories below this count are folded into "Other" before
# one-hot encoding. Squall has 4 observations in the raw data; a dummy
# variable built on 4 rows is noise, and will not generalise.
RARE_CATEGORY_THRESHOLD = 100


def load_cleaned(path=CLEANED_CSV) -> pd.DataFrame:
    """Read the cleaned dataset produced by pipeline.py."""
    try:
        df = pd.read_csv(path, parse_dates=["date_time"])
    except FileNotFoundError:
        logger.error(
            "Cleaned dataset not found at %s — run pipeline.py first", path,
            exc_info=True,
        )
        raise
    logger.info("Loaded cleaned data | rows=%d | columns=%d", len(df), df.shape[1])
    return df


# --------------------------------------------------------------------------
# Time features
# --------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, day of week, weekend flag, month, and cyclical encodings.

    Cyclical encoding matters here: as an integer, hour 23 and hour 0 are 23
    units apart, when in reality they are adjacent. Projecting onto sine and
    cosine puts them next to each other on a circle, which is what any
    distance-based or linear model needs.
    """
    df = df.copy()
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek          # Monday = 0
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info(
        "Added time features | hour, day_of_week, month, year, is_weekend, "
        "hour_sin/cos, dow_sin/cos"
    )
    logger.debug(
        "Weekend share: %.3f | weekday rows=%d | weekend rows=%d",
        df["is_weekend"].mean(),
        int((df["is_weekend"] == 0).sum()),
        int((df["is_weekend"] == 1).sum()),
    )
    return df


# --------------------------------------------------------------------------
# Weather features
# --------------------------------------------------------------------------
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rare weather categories, one-hot encode, add derived flags."""
    df = df.copy()

    counts = df["weather_main"].value_counts()
    rare = counts[counts < RARE_CATEGORY_THRESHOLD].index.tolist()
    logger.debug("weather_main value counts: %s", counts.to_dict())

    if rare:
        n_rare = int(df["weather_main"].isin(rare).sum())
        df["weather_grouped"] = df["weather_main"].where(
            ~df["weather_main"].isin(rare), "Other"
        )
        logger.warning(
            "Collapsed %d rows across %d rare weather categories (%s) into "
            "'Other' | reason: fewer than %d observations each",
            n_rare, len(rare), ", ".join(rare), RARE_CATEGORY_THRESHOLD,
        )
    else:
        df["weather_grouped"] = df["weather_main"]

    dummies = pd.get_dummies(df["weather_grouped"], prefix="weather", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    logger.info(
        "One-hot encoded weather into %d indicator columns", dummies.shape[1]
    )

    # Derived indicators: these carry information the one-hot columns
    # fragment across several categories.
    df["is_precipitation"] = df["weather_main"].isin(SEVERE_WEATHER).astype(int)
    df["is_low_visibility"] = df["weather_main"].isin(LOW_VISIBILITY_WEATHER).astype(int)
    df["has_rain"] = (df["rain_1h"] > 0).astype(int)
    df["has_snow"] = (df["snow_1h"] > 0).astype(int)
    df["temp_celsius"] = df["temp"] - 273.15

    logger.info(
        "Added derived weather indicators | is_precipitation, is_low_visibility, "
        "has_rain, has_snow, temp_celsius"
    )
    logger.debug(
        "Indicator rates | precipitation=%.3f | low visibility=%.3f | rain=%.3f | snow=%.3f",
        df["is_precipitation"].mean(), df["is_low_visibility"].mean(),
        df["has_rain"].mean(), df["has_snow"].mean(),
    )
    return df


# --------------------------------------------------------------------------
# Scaled numerical features
# --------------------------------------------------------------------------
def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardised (z-score) and min-max scaled versions of continuous vars.

    Both are produced because they serve different models: z-scores for
    linear and distance-based methods, min-max for the neural network in
    Part 3. Scaling parameters are logged at DEBUG so the transformation can
    be reproduced or inverted.
    """
    df = df.copy()
    for column in ("temp", "clouds_all"):
        mean, std = df[column].mean(), df[column].std(ddof=0)
        lo, hi = df[column].min(), df[column].max()

        df[f"{column}_z"] = (df[column] - mean) / std
        df[f"{column}_minmax"] = (df[column] - lo) / (hi - lo)

        logger.debug(
            "Scaling '%s' | mean=%.4f std=%.4f min=%.4f max=%.4f",
            column, mean, std, lo, hi,
        )

    logger.info("Added scaled features | temp_z, temp_minmax, clouds_all_z, clouds_all_minmax")
    return df


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
def add_congestion_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a data-driven congestion category from volume quartiles.

    Logic, as required by the brief:

    The Part 1 analysis used a fixed threshold of 5,500 vehicles/hour. That
    is defensible for reporting but arbitrary for modelling, and it produces
    badly imbalanced classes (about 15% positive). Here the boundaries are
    taken from the data itself — the 25th, 50th and 75th percentiles of
    observed hourly volume — giving four classes of roughly equal size:

        Low       volume <= Q1
        Moderate  Q1 < volume <= Q2 (median)
        High      Q2 < volume <= Q3
        Severe    volume > Q3

    A binary `is_congested` flag is also retained at the Part 1 threshold so
    that results stay comparable across the two parts.
    """
    df = df.copy()
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.50, 0.75])
    logger.debug(
        "Congestion quartile thresholds | Q1=%.1f | Q2=%.1f | Q3=%.1f", q1, q2, q3
    )

    df["congestion_level"] = pd.cut(
        df["traffic_volume"],
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=["Low", "Moderate", "High", "Severe"],
    )
    df["congestion_code"] = df["congestion_level"].cat.codes
    df["is_congested"] = (df["traffic_volume"] > 5500).astype(int)

    distribution = df["congestion_level"].value_counts().sort_index().to_dict()
    logger.info("Created congestion_level target | class counts: %s", distribution)
    logger.info(
        "Created binary is_congested flag at the Part 1 threshold of 5500 | "
        "positive rate=%.4f", df["is_congested"].mean(),
    )
    return df


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_feature_engineering() -> pd.DataFrame:
    df = load_cleaned()
    logger.info(
        "Dataset shape BEFORE feature engineering | rows=%d | columns=%d",
        len(df), df.shape[1],
    )

    for step in (
        add_time_features,
        add_weather_features,
        add_scaled_features,
        add_congestion_target,
    ):
        df = step(df)

    logger.info(
        "Dataset shape AFTER feature engineering | rows=%d | columns=%d",
        len(df), df.shape[1],
    )

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    logger.info("Feature table saved | path=%s", FEATURES_CSV)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Engineer ML-ready features from the cleaned traffic dataset."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Emit DEBUG-level intermediate values (quartile thresholds, scaling parameters).",
    )
    args = parser.parse_args()

    setup_logging(LOG_DIR / "pipeline.log", debug=args.debug)

    try:
        run_feature_engineering()
    except Exception:
        logger.error("Feature engineering aborted before completion", exc_info=True)
        return 1

    logger.info("Feature engineering completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
