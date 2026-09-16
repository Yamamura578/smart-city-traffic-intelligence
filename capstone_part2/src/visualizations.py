"""Part 2, Task 3 — Matplotlib visualisations of the engineered dataset.

Run from the repository root:

    python -m capstone_part2.src.visualizations

Reads  data/processed/traffic_features.csv
Writes capstone_part2/figures/*.png

Four figures are produced. Interpretations are in the Part 2 report; each
function's docstring states what the figure is for.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed to save files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import FEATURES_CSV, LOG_DIR, PART2_FIGURES
from common.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

FIGSIZE = (10, 6)
DPI = 150


def load_features(path=FEATURES_CSV) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["date_time"])
    except FileNotFoundError:
        logger.error(
            "Feature table not found at %s — run feature_engineering.py first",
            path, exc_info=True,
        )
        raise
    logger.info("Loaded feature table | rows=%d | columns=%d", len(df), df.shape[1])
    return df


def _save(fig, name: str):
    """Save a figure and log its path — the confirmation trail for Task 3."""
    PART2_FIGURES.mkdir(parents=True, exist_ok=True)
    path = PART2_FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Figure saved | path=%s", path)
    return path


def figure_hourly_profile(df: pd.DataFrame):
    """Mean volume by hour, weekday vs weekend overlaid.

    The central finding of the whole project: the daily cycle dwarfs every
    other effect, and weekday and weekend follow different shapes.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for flag, label, style in ((0, "Weekday", "-o"), (1, "Weekend", "--s")):
        subset = df[df["is_weekend"] == flag]
        hourly = subset.groupby("hour")["traffic_volume"].mean()
        ax.plot(hourly.index, hourly.values, style, label=label, markersize=4)
        logger.debug(
            "%s profile | peak hour=%d (%.0f) | trough hour=%d (%.0f)",
            label, hourly.idxmax(), hourly.max(), hourly.idxmin(), hourly.min(),
        )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean traffic volume (vehicles/hour)")
    ax.set_title("Traffic demand by hour: weekday vs weekend")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "01_hourly_profile.png")


def figure_weather_impact(df: pd.DataFrame):
    """Mean volume by weather category, annotated with sample size.

    The annotation is the point: Squall's low mean rests on 4 observations,
    and a bar chart without n invites a conclusion the data cannot support.
    """
    stats = (
        df.groupby("weather_main")["traffic_volume"]
          .agg(["mean", "count"])
          .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = ["#c44e52" if n < 100 else "#4c72b0" for n in stats["count"]]
    ax.barh(stats.index, stats["mean"], color=colors)

    for i, (mean, count) in enumerate(zip(stats["mean"], stats["count"])):
        ax.text(mean + 40, i, f"n={count:,}", va="center", fontsize=8)

    ax.set_xlabel("Mean traffic volume (vehicles/hour)")
    ax.set_title("Traffic by weather condition (red = fewer than 100 observations)")
    ax.set_xlim(0, stats["mean"].max() * 1.18)
    ax.grid(axis="x", alpha=0.3)

    reliable = stats[stats["count"] >= 100]
    logger.debug(
        "Weather range | overall %.1f-%.1f (diff %.1f) | n>=100 only %.1f-%.1f (diff %.1f)",
        stats["mean"].min(), stats["mean"].max(),
        stats["mean"].max() - stats["mean"].min(),
        reliable["mean"].min(), reliable["mean"].max(),
        reliable["mean"].max() - reliable["mean"].min(),
    )
    return _save(fig, "02_weather_impact.png")


def figure_temperature_scatter(df: pd.DataFrame):
    """Temperature vs volume, with the correlation stated on the figure.

    Plotted after cleaning, so the -273 degree sensor failures are gone; the
    weakness of the relationship is the finding.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        df["temp_celsius"], df["traffic_volume"],
        s=3, alpha=0.08, color="#4c72b0", edgecolors="none",
    )

    # Binned means make the weak trend visible through the point cloud.
    bins = np.arange(-30, 40, 2.5)
    binned = df.groupby(pd.cut(df["temp_celsius"], bins), observed=True)["traffic_volume"].mean()
    centres = [interval.mid for interval in binned.index]
    ax.plot(centres, binned.values, "-o", color="#c44e52", markersize=4,
            label="Mean volume per 2.5 °C bin")

    r = df["temp_celsius"].corr(df["traffic_volume"])
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Traffic volume (vehicles/hour)")
    ax.set_title(f"Temperature vs traffic volume (Pearson r = {r:.3f}, r² = {r**2:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    logger.debug("Temperature-traffic correlation on cleaned data: r=%.4f", r)
    return _save(fig, "03_temperature_scatter.png")


def figure_coverage_and_distribution(df: pd.DataFrame):
    """Two panels: monthly record counts, and the volume distribution.

    The left panel makes the 2014-2015 sensor outage visible; the right
    shows why the mean is a poor summary of hourly volume.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    monthly = df.groupby(df["date_time"].dt.to_period("M")).size()
    ax1.plot(monthly.index.to_timestamp(), monthly.values, color="#4c72b0")
    ax1.axhline(monthly.median(), color="#c44e52", linestyle="--",
                label=f"Median = {monthly.median():.0f} hours/month")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Hours recorded")
    ax1.set_title("Data coverage over time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.hist(df["traffic_volume"], bins=60, color="#4c72b0", edgecolor="white")
    ax2.axvline(df["traffic_volume"].mean(), color="#c44e52", linestyle="--",
                label=f"Mean = {df['traffic_volume'].mean():.0f}")
    ax2.axvline(df["traffic_volume"].median(), color="#55a868", linestyle="--",
                label=f"Median = {df['traffic_volume'].median():.0f}")
    ax2.set_xlabel("Traffic volume (vehicles/hour)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of hourly traffic volume")
    ax2.legend()
    ax2.grid(alpha=0.3)

    gap = monthly[monthly < 200]
    logger.debug("Months with fewer than 200 recorded hours: %s",
                 [str(p) for p in gap.index])
    return _save(fig, "04_coverage_and_distribution.png")


def run_visualisations() -> list:
    df = load_features()
    paths = [
        figure_hourly_profile(df),
        figure_weather_impact(df),
        figure_temperature_scatter(df),
        figure_coverage_and_distribution(df),
    ]
    logger.info("All %d figures generated successfully", len(paths))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Part 2 Matplotlib figures.")
    parser.add_argument("--debug", action="store_true",
                        help="Emit DEBUG-level detail about each figure's underlying values.")
    args = parser.parse_args()

    setup_logging(LOG_DIR / "pipeline.log", debug=args.debug)

    try:
        run_visualisations()
    except Exception:
        logger.error("Visualisation step aborted before completion", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
