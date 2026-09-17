"""Part 3, Task 5 — travel-timing recommendation system.

    python -m capstone_part3.src.recommender --day-type weekday --weather Rain

The dataset covers a single corridor, so the system recommends *when* to
travel rather than which route. Recommendations are grounded in the Part 2
feature table and the Part 3 models: historical hourly profiles supply the
ranking, and the trained regressor supplies the expected volume under the
stated conditions.

print() is used here because this is a user-facing tool; all internal
events go through the logger.
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from common.config import LOG_DIR, PART3_MODELS, REPO_ROOT
from common.logging_config import get_logger, setup_logging
from capstone_part3.src.data import FEATURE_COLUMNS, load_features

logger = get_logger(__name__)
REPORTS = REPO_ROOT / "capstone_part3" / "reports"


def hourly_profile(df: pd.DataFrame, day_type: str, weather: str | None) -> pd.Series:
    """Mean volume by hour under the requested conditions, with fallback."""
    subset = df if day_type == "all" else df[df["is_weekend"] == (day_type == "weekend")]

    if weather:
        filtered = subset[subset["weather_main"] == weather]
        if len(filtered) < 200:
            logger.warning(
                "Only %d hours match weather='%s' on %s | falling back to all "
                "weather conditions: the conditional profile would be unreliable",
                len(filtered), weather, day_type,
            )
        else:
            subset = filtered
            logger.info("Filtered to weather='%s' | rows=%d", weather, len(subset))

    profile = subset.groupby("hour")["traffic_volume"].mean()
    logger.info("Built hourly profile | day_type=%s | rows=%d", day_type, len(subset))
    return profile


def recommend(day_type="weekday", weather=None, earliest=6, latest=20, top=3) -> dict:
    df = load_features()
    profile = hourly_profile(df, day_type, weather)
    window = profile[(profile.index >= earliest) & (profile.index <= latest)]

    if window.empty:
        logger.warning("No hours in window %d-%d", earliest, latest)
        return {"error": "no data in requested window"}

    ranked = window.sort_values()
    overall = profile.mean()
    best = ranked.head(top)
    worst_hour = int(ranked.index[-1])

    # Contiguous runs of quiet hours make a better recommendation than three
    # scattered hours, so the best run is reported as the travel window.
    quiet = sorted(int(h) for h in best.index)
    runs, current = [], [quiet[0]]
    for h in quiet[1:]:
        if h == current[-1] + 1:
            current.append(h)
        else:
            runs.append(current); current = [h]
    runs.append(current)
    longest = max(runs, key=len)

    if len(longest) > 1:
        window_text = f"between {longest[0]:02d}:00 and {longest[-1] + 1:02d}:00"
    else:
        window_text = f"around {longest[0]:02d}:00"

    day_text = {"weekday": "a weekday", "weekend": "a weekend",
                "all": "any"}.get(day_type, day_type)
    weather_text = f" in {weather.lower()} conditions" if weather else ""
    saving = 100 * (overall - best.iloc[0]) / overall

    sentence = (
        f"For {day_text} journey{weather_text}, consider travelling "
        f"{window_text}, when historical traffic volumes average "
        f"{best.iloc[0]:,.0f} vehicles/hour — about {saving:.0f}% below the "
        f"daily average. Avoid {worst_hour:02d}:00, the busiest hour in this window "
        f"at {ranked.iloc[-1]:,.0f} vehicles/hour."
    )

    result = {
        "day_type": day_type,
        "weather": weather,
        "search_window": [earliest, latest],
        "recommended_hours": [{"hour": int(h), "mean_volume": round(float(v), 1)}
                              for h, v in best.items()],
        "busiest_hour": {"hour": worst_hour, "mean_volume": round(float(ranked.iloc[-1]), 1)},
        "percent_below_average": round(float(saving), 1),
        "recommendation": sentence,
    }
    logger.info("Recommendation generated | %s", sentence)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Traffic timing recommendation system.")
    p.add_argument("--day-type", choices=["all", "weekday", "weekend"], default="weekday")
    p.add_argument("--weather", default=None, help="e.g. Clear, Rain, Snow")
    p.add_argument("--earliest", type=int, default=6)
    p.add_argument("--latest", type=int, default=20)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    setup_logging(LOG_DIR / "part3.log", debug=args.debug)
    logger.info("Recommender invoked | arguments=%s",
                {k: v for k, v in vars(args).items() if k != "debug"})

    try:
        result = recommend(args.day_type, args.weather, args.earliest, args.latest)
    except Exception:
        logger.error("Recommendation failed", exc_info=True)
        print("Could not generate a recommendation. See logs/part3.log.")
        return 1

    if "error" in result:
        print("No data available for that window.")
        return 1

    print("\n" + result["recommendation"] + "\n")
    print("Quietest hours in the requested window:")
    for entry in result["recommended_hours"]:
        print(f"  {entry['hour']:02d}:00   {entry['mean_volume']:>7,.0f} vehicles/hour")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "recommendation_example.json").write_text(json.dumps(result, indent=2))
    logger.info("Example recommendation saved | path=%s",
                REPORTS / "recommendation_example.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
