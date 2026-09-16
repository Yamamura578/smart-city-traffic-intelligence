"""Part 2, Task 1 — data loading, schema validation and cleaning.

Run from the repository root:

    python -m capstone_part2.src.pipeline
    python -m capstone_part2.src.pipeline --debug

Each cleaning step is its own function so that each can emit its own log
message (the brief explicitly forbids a single "Cleaning completed." line).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from common.config import (
    CLEANED_CSV,
    EXPECTED_COLUMNS,
    LOG_DIR,
    RAIN_MAX_MM,
    RAW_CSV,
    TEMP_MAX_K,
    TEMP_MIN_K,
)
from common.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV, converting file/parse failures into a clean error.

    Note the specific exception types — a bare `except:` is explicitly
    disallowed by the brief.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Raw CSV not found at %s", path, exc_info=True)
        raise
    except (pd.errors.ParserError, UnicodeDecodeError):
        logger.error("Could not parse raw CSV at %s", path, exc_info=True)
        raise

    logger.info("Loaded raw data | rows=%d | columns=%d | source=%s", len(df), df.shape[1], path)
    return df


# --------------------------------------------------------------------------
# Validate
# --------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if expected columns are missing. Runs before any cleaning."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing:
        logger.error("Schema validation failed | missing columns: %s", missing)
        raise ValueError(f"Missing expected columns: {missing}")
    if extra:
        logger.warning("Unexpected columns present (kept, not dropped): %s", extra)

    logger.info("Schema validation passed | %d expected columns present", len(EXPECTED_COLUMNS))


# --------------------------------------------------------------------------
# Clean — one function per step, one WARNING per change
# --------------------------------------------------------------------------
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time, drop or flag unparseable rows.

    TODO: use pd.to_datetime(errors="coerce"), count NaT rows, log a WARNING
    with the count and reason before dropping them.
    """
    raise NotImplementedError


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records.

    TODO: there are two distinct cases in this dataset and you should treat
    them separately and log them separately:
      1. 17 fully identical rows  -> safe to drop outright.
      2. ~7,600 repeated date_time values with different weather_description
         -> the same hour logged under several weather categories. Decide a
         rule (keep the first, keep the most severe, or aggregate) and
         document it in the report. Do not silently drop them.
    """
    raise NotImplementedError


def standardise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise categorical text.

    TODO: strip whitespace and normalise case on weather_main /
    weather_description; forward-fill `holiday` across each holiday date
    (it is only tagged on the 00:00 row, so 48,143 of 48,204 rows are null)
    and fill the remainder with "None". Log the number of rows affected.
    """
    raise NotImplementedError


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect physically impossible values and median-impute them.

    Targets in this dataset:
      * temp == 0 K (10 rows) and anything outside [TEMP_MIN_K, TEMP_MAX_K]
      * rain_1h > RAIN_MAX_MM (1 row, 9831.3 mm)

    TODO: impute by MONTH rather than one global median — the brief asks for
    an explicit loop or groupby over months. Log a WARNING per affected
    column with the row count and the reason, and log the monthly median
    values used at DEBUG level.
    """
    raise NotImplementedError


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle remaining missing values. Log count and strategy per column."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_pipeline(debug: bool = False) -> pd.DataFrame:
    df = load_raw()
    validate_schema(df)

    for step in (
        parse_datetime,
        drop_duplicates,
        standardise_categories,
        handle_outliers,
        handle_missing,
    ):
        df = step(df)

    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)
    logger.info("Cleaned dataset saved | rows=%d | columns=%d | path=%s",
                len(df), df.shape[1], CLEANED_CSV)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean the Metro Interstate traffic dataset.")
    parser.add_argument("--debug", action="store_true",
                        help="Emit DEBUG-level intermediate values (thresholds, imputation values).")
    args = parser.parse_args()

    setup_logging(LOG_DIR / "pipeline.log", debug=args.debug)

    try:
        run_pipeline(debug=args.debug)
    except Exception:
        # Log once, at the top level, with the traceback captured in the log
        # file rather than dumped raw to the user's terminal.
        logger.error("Pipeline aborted before completion", exc_info=True)
        return 1

    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
