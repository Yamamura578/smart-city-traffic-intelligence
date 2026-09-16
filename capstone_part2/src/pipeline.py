"""Part 2, Task 1 — data loading, schema validation and cleaning.

Run from the repository root:

    python -m capstone_part2.src.pipeline
    python -m capstone_part2.src.pipeline --debug

Each cleaning step is a separate function so that each emits its own log
record; the brief explicitly forbids a single "Cleaning completed." message.

Cleaning decisions and their rationale are documented at each step. The
issues addressed here were identified during the Part 1 analysis.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
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

# Severity ranking used to resolve duplicate timestamps. Where the same hour
# was logged under several weather descriptions, the most severe condition is
# retained: for traffic and safety analysis, under-reporting severe weather is
# the more costly error.
WEATHER_SEVERITY = {
    "Squall": 10, "Thunderstorm": 9, "Snow": 8, "Rain": 7, "Drizzle": 6,
    "Fog": 5, "Smoke": 4, "Haze": 3, "Mist": 2, "Clouds": 1, "Clear": 0,
}


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV, converting file and parse failures into clear errors.

    Specific exception types are caught; a bare `except:` is disallowed by
    the brief and would mask genuine bugs.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Raw CSV not found at %s", path, exc_info=True)
        raise
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        logger.error("Could not parse raw CSV at %s", path, exc_info=True)
        raise

    logger.info(
        "Loaded raw data | rows=%d | columns=%d | source=%s",
        len(df), df.shape[1], path,
    )
    logger.debug("Raw dtypes: %s", df.dtypes.to_dict())
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
        logger.warning("Unexpected columns present (retained, not dropped): %s", extra)

    logger.info(
        "Schema validation passed | %d expected columns present", len(EXPECTED_COLUMNS)
    )


# --------------------------------------------------------------------------
# Clean — one function per step, one WARNING per change
# --------------------------------------------------------------------------
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time and drop any row whose timestamp cannot be read."""
    df = df.copy()
    before = len(df)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    unparseable = int(df["date_time"].isna().sum())
    if unparseable:
        df = df.dropna(subset=["date_time"])
        logger.warning(
            "Dropped %d rows | reason: date_time could not be parsed", unparseable
        )
    else:
        logger.info("Parsed date_time | all %d timestamps valid", before)

    logger.info(
        "Date range | first=%s | last=%s",
        df["date_time"].min(), df["date_time"].max(),
    )
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates in two distinct passes, logged separately.

    1. Fully identical rows are unambiguous errors and are dropped outright.
    2. Repeated timestamps arise where one hour was logged under several
       weather descriptions. These are resolved by keeping the most severe
       weather condition for that hour (see WEATHER_SEVERITY).
    """
    df = df.copy()

    exact = int(df.duplicated().sum())
    if exact:
        df = df.drop_duplicates()
        logger.warning(
            "Dropped %d rows | reason: fully duplicated records", exact
        )
    else:
        logger.info("No fully duplicated rows found")

    repeated = int(df["date_time"].duplicated().sum())
    if repeated:
        df["_severity"] = df["weather_main"].map(WEATHER_SEVERITY).fillna(-1)
        df = (
            df.sort_values(["date_time", "_severity"], ascending=[True, False])
              .drop_duplicates(subset="date_time", keep="first")
              .drop(columns="_severity")
        )
        logger.warning(
            "Dropped %d rows | reason: repeated timestamps; retained the most "
            "severe weather observation per hour", repeated,
        )
    else:
        logger.info("No repeated timestamps found")

    logger.info("Duplicate handling complete | rows remaining=%d", len(df))
    return df.reset_index(drop=True)


def standardise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise categorical text and make the holiday column usable.

    Two issues are handled here. Categorical text is stripped of stray
    whitespace so that group-by operations do not split a single category.

    More importantly, `holiday` is tagged only on the 00:00 row of each
    holiday (61 of 48,204 rows; the remainder hold the placeholder "None",
    which pandas reads as a null). The flag is forward-filled across the
    whole of each holiday date so that it is usable as a feature. This runs
    BEFORE duplicate removal, so that a tagged midnight row cannot be
    discarded by the de-duplication rule before its date has been recorded.
    """
    df = df.copy()

    for col in ("weather_main", "weather_description", "holiday"):
        present = df[col].notna()
        stripped = df.loc[present, col].astype(str).str.strip()
        changed = int((df.loc[present, col] != stripped).sum())
        df.loc[present, col] = stripped
        if changed:
            logger.warning(
                "Modified %d values in '%s' | reason: leading/trailing whitespace",
                changed, col,
            )
    logger.info("Standardised categorical text in weather and holiday columns")

    categories = sorted(df["weather_main"].dropna().unique())
    logger.debug("weather_main categories after standardisation: %s", categories)

    tagged = int(df["holiday"].notna().sum())
    holiday_dates = (
        df.loc[df["holiday"].notna(), ["date_time", "holiday"]]
          .assign(_date=lambda d: d["date_time"].dt.date)
          .set_index("_date")["holiday"]
          .to_dict()
    )
    df["holiday"] = df["date_time"].dt.date.map(holiday_dates)
    df["is_holiday"] = df["holiday"].notna().astype(int)
    newly_filled = int(df["holiday"].notna().sum()) - tagged
    logger.warning(
        "Modified %d rows in 'holiday' | reason: forward-filled from %d tagged "
        "midnight rows across %d holiday dates",
        newly_filled, tagged, len(holiday_dates),
    )
    logger.debug("Holiday dates identified: %s", sorted(map(str, holiday_dates)))
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physically impossible readings with the median for that month.

    Imputation is done month by month in an explicit loop, as required by the
    brief: a single global median would assign a July temperature to a
    January sensor failure.

    Targets: temp outside [TEMP_MIN_K, TEMP_MAX_K] (includes the 0 K sensor
    failures) and rain_1h above RAIN_MAX_MM.
    """
    df = df.copy()
    df["_month"] = df["date_time"].dt.month

    checks = {
        "temp": (df["temp"] < TEMP_MIN_K) | (df["temp"] > TEMP_MAX_K),
        "rain_1h": df["rain_1h"] > RAIN_MAX_MM,
    }

    for column, is_bad in checks.items():
        n_bad = int(is_bad.sum())
        if not n_bad:
            logger.info("No implausible values detected in '%s'", column)
            continue

        # Medians computed from valid rows only, so bad values cannot
        # contaminate the replacement value.
        valid = df.loc[~is_bad, [column, "_month"]]
        for month in sorted(df["_month"].unique()):
            month_median = valid.loc[valid["_month"] == month, column].median()
            target = is_bad & (df["_month"] == month)
            n_month = int(target.sum())
            if n_month:
                df.loc[target, column] = month_median
                logger.debug(
                    "Imputed %d rows in '%s' for month %02d using median %.2f",
                    n_month, column, month, month_median,
                )

        logger.warning(
            "Imputed %d rows in '%s' | reason: value outside physically "
            "plausible range; replaced with monthly median",
            n_bad, column,
        )

    zero_volume = int((df["traffic_volume"] == 0).sum())
    if zero_volume:
        logger.warning(
            "Retained %d rows with traffic_volume = 0 | reason: implausible for "
            "an interstate but not corrected, as the true value is unknown",
            zero_volume,
        )

    return df.drop(columns="_month")


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Report and fill any remaining missing values, column by column."""
    df = df.copy()
    numeric = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]

    for column in numeric:
        n_missing = int(df[column].isna().sum())
        if n_missing:
            fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)
            logger.warning(
                "Imputed %d missing values in '%s' | reason: median fill (%.2f)",
                n_missing, column, fill_value,
            )

    for column in ("weather_main", "weather_description"):
        n_missing = int(df[column].isna().sum())
        if n_missing:
            df[column] = df[column].fillna("Unknown")
            logger.warning(
                "Filled %d missing values in '%s' | reason: replaced with 'Unknown'",
                n_missing, column,
            )

    remaining = int(df[numeric].isna().sum().sum())
    logger.info("Missing-value handling complete | remaining nulls in numeric columns=%d", remaining)
    return df


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_pipeline() -> pd.DataFrame:
    """Execute every stage in order and write the cleaned dataset."""
    df = load_raw()
    validate_schema(df)
    rows_in = len(df)

    # Order matters: holiday tags are forward-filled before de-duplication so
    # that a tagged midnight row cannot be discarded before it has been used.
    for step in (
        parse_datetime,
        standardise_categories,
        drop_duplicates,
        handle_outliers,
        handle_missing,
    ):
        df = step(df)

    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)
    logger.info(
        "Cleaned dataset saved | rows=%d (from %d) | columns=%d | path=%s",
        len(df), rows_in, df.shape[1], CLEANED_CSV,
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean the Metro Interstate traffic dataset."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Emit DEBUG-level intermediate values (monthly medians, holiday dates).",
    )
    args = parser.parse_args()

    setup_logging(LOG_DIR / "pipeline.log", debug=args.debug)

    try:
        run_pipeline()
    except Exception:
        # Logged once, at the top level, with the traceback captured in the
        # log file rather than dumped raw to the user's terminal.
        logger.error("Pipeline aborted before completion", exc_info=True)
        return 1

    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
