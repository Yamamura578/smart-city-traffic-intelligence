"""Shared paths and constants.

Keeping these in one place means Part 2 and Part 3 read the same files and
use the same thresholds, which is what the "integration" criteria reward.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
# config.py lives in <repo>/common/, so parents[1] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
LOG_DIR = REPO_ROOT / "logs"

RAW_CSV = DATA_RAW / "Metro_Interstate_Traffic_Volume.csv"
CLEANED_CSV = DATA_PROCESSED / "traffic_cleaned.csv"      # output of Part 2 pipeline
FEATURES_CSV = DATA_PROCESSED / "traffic_features.csv"    # output of Part 2 feature engineering
SQLITE_DB = DATA_PROCESSED / "traffic.db"                 # Part 1

PART2_FIGURES = REPO_ROOT / "capstone_part2" / "figures"
PART3_MODELS = REPO_ROOT / "capstone_part3" / "models"

# --- Schema --------------------------------------------------------------
EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

# --- Physical plausibility bounds (used for outlier detection) -----------
# temp is in Kelvin; 0 K rows are sensor failures, not cold days.
TEMP_MIN_K = 200.0        # ~ -73 C, far below any Minneapolis record
TEMP_MAX_K = 330.0        # ~ 57 C
RAIN_MAX_MM = 9000.0      # the brief's stated implausible-rainfall cut-off
CLOUDS_RANGE = (0, 100)

# --- Part 1 analysis constants ------------------------------------------
CONGESTION_THRESHOLD = 5500   # traffic_volume > 5500 == "congestion"
HIGH_TEMP_K = 292.0           # "high temperature" for conditional probability

# --- Part 3 proxy-label constants ---------------------------------------
# Weather categories treated as severe for the proxy accident-risk label.
SEVERE_WEATHER = ["Rain", "Snow", "Thunderstorm", "Squall"]
# Categories treated as low visibility.
LOW_VISIBILITY_WEATHER = ["Fog", "Mist", "Haze", "Smoke"]

RANDOM_STATE = 42
