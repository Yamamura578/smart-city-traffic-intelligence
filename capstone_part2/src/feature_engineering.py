"""Part 2, Task 2 — build ML-ready features from the cleaned dataset.

Run:  python -m capstone_part2.src.feature_engineering [--debug]

Required feature groups:
  Time     : hour, day_of_week, is_weekend, >=1 cyclical encoding
             (hour_sin/hour_cos = sin/cos(2*pi*hour/24), same idea for dow).
  Weather  : encode weather_main (collapse the long tail -- Squall=4 rows,
             Smoke=20 -- before one-hot), plus derived indicators such as
             is_precipitation / is_low_visibility.
  Numeric  : scaled versions of at least two continuous variables
             (temp and clouds_all are the obvious pair).
  Target   : data-driven congestion category from traffic_volume quartiles.
             Log the quartile thresholds at DEBUG level, and document the
             bucket logic in the report.

Logging: INFO with dataset shape BEFORE and AFTER; DEBUG for intermediate
values; INFO when the feature table is written.
"""

from __future__ import annotations

from common.logging_config import get_logger

logger = get_logger(__name__)

# TODO: implement add_time_features, add_weather_features,
# add_scaled_features, add_congestion_target, and a main() that calls
# setup_logging(LOG_DIR / "pipeline.log", debug=args.debug).
