"""Part 2, Task 4 — command-line traffic analytics application.

Run from the repository root:

    python -m capstone_part2.src.traffic_app lookup  --datetime "2017-05-03 08:00"
    python -m capstone_part2.src.traffic_app peak    --day-type weekday
    python -m capstone_part2.src.traffic_app compare --weather Clear --weather Snow
    python -m capstone_part2.src.traffic_app advise  --day-type weekday --earliest 6 --latest 20

Note on print() vs logging: print() appears here and only here, because this
is a user-facing tool and the answer to a query is output, not status. All
internal events — which command ran, bad input, load failures — go through
the logger, per the brief.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from common.config import FEATURES_CSV, LOG_DIR
from common.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def load_data(path=FEATURES_CSV) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["date_time"])
    except FileNotFoundError:
        logger.error(
            "Processed dataset not found at %s — run pipeline.py and "
            "feature_engineering.py first", path, exc_info=True,
        )
        raise
    logger.info("Loaded dataset for query | rows=%d", len(df))
    return df


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_lookup(df: pd.DataFrame, args) -> int:
    """Traffic and weather for one specific hour."""
    try:
        target = pd.to_datetime(args.datetime)
    except (ValueError, TypeError):
        logger.error(
            "Invalid --datetime value %r | expected a format such as "
            "'2017-05-03 08:00'", args.datetime,
        )
        print(f"Could not read '{args.datetime}' as a date and time.")
        print("Try a format like: 2017-05-03 08:00")
        return 2

    target = target.floor("h")
    match = df[df["date_time"] == target]

    if match.empty:
        nearest = df.iloc[(df["date_time"] - target).abs().argsort()[:1]]
        logger.warning("No record for %s | reporting nearest available hour", target)
        print(f"No record for {target:%Y-%m-%d %H:%M}.")
        row = nearest.iloc[0]
        print(f"Nearest available: {row['date_time']:%Y-%m-%d %H:%M}")
    else:
        row = match.iloc[0]
        print(f"{row['date_time']:%Y-%m-%d %H:%M} ({row['date_time']:%A})")

    print(f"  Traffic volume : {int(row['traffic_volume']):,} vehicles/hour")
    print(f"  Congestion     : {row['congestion_level']}")
    print(f"  Weather        : {row['weather_main']} ({row['weather_description']})")
    print(f"  Temperature    : {row['temp_celsius']:.1f} °C")
    print(f"  Cloud cover    : {int(row['clouds_all'])}%")
    if pd.notna(row.get("holiday")):
        print(f"  Holiday        : {row['holiday']}")
    return 0


def cmd_peak(df: pd.DataFrame, args) -> int:
    """Busiest and quietest hours, optionally split by day type."""
    subset = df
    label = "all days"
    if args.day_type == "weekday":
        subset, label = df[df["is_weekend"] == 0], "weekdays"
    elif args.day_type == "weekend":
        subset, label = df[df["is_weekend"] == 1], "weekends"

    hourly = subset.groupby("hour")["traffic_volume"].agg(["mean", "count"])
    busiest = hourly["mean"].nlargest(args.top)
    quietest = hourly["mean"].nsmallest(args.top)

    print(f"Traffic by hour — {label} (n = {len(subset):,} hours)\n")
    print(f"  Busiest {args.top} hours:")
    for hour, mean in busiest.items():
        print(f"    {hour:02d}:00   {mean:>7,.0f} vehicles/hour")
    print(f"\n  Quietest {args.top} hours:")
    for hour, mean in quietest.items():
        print(f"    {hour:02d}:00   {mean:>7,.0f} vehicles/hour")
    print(f"\n  Peak-to-trough ratio: {busiest.iloc[0] / quietest.iloc[0]:.1f}x")
    return 0


def cmd_compare(df: pd.DataFrame, args) -> int:
    """Compare mean traffic across weather conditions, or weekday vs weekend."""
    if args.weather:
        available = set(df["weather_main"].unique())
        unknown = [w for w in args.weather if w not in available]
        if unknown:
            logger.error(
                "Unknown weather condition(s): %s | available: %s",
                ", ".join(unknown), ", ".join(sorted(available)),
            )
            print(f"Unknown weather condition(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(sorted(available))}")
            return 2

        print("Mean traffic volume by weather condition\n")
        for condition in args.weather:
            subset = df[df["weather_main"] == condition]
            flag = "  (small sample)" if len(subset) < 100 else ""
            print(f"  {condition:<14} {subset['traffic_volume'].mean():>7,.0f} "
                  f"vehicles/hour   n={len(subset):,}{flag}")
    else:
        print("Weekday vs weekend traffic\n")
        for flag, name in ((0, "Weekday"), (1, "Weekend")):
            subset = df[df["is_weekend"] == flag]
            print(f"  {name:<9} mean={subset['traffic_volume'].mean():>7,.0f}   "
                  f"median={subset['traffic_volume'].median():>7,.0f}   n={len(subset):,}")
        weekday = df[df["is_weekend"] == 0]["traffic_volume"].mean()
        weekend = df[df["is_weekend"] == 1]["traffic_volume"].mean()
        print(f"\n  Difference: {weekday - weekend:,.0f} vehicles/hour "
              f"({100 * (weekday - weekend) / weekend:+.1f}% vs weekend)")
    return 0


def cmd_advise(df: pd.DataFrame, args) -> int:
    """Recommend travel windows — the quietest hours within a time range.

    This command is the seed of the Part 3 recommendation component.
    """
    if not 0 <= args.earliest <= 23 or not 0 <= args.latest <= 23:
        logger.error(
            "Invalid hour range | earliest=%d latest=%d | both must be 0-23",
            args.earliest, args.latest,
        )
        print("Hours must be between 0 and 23.")
        return 2
    if args.earliest > args.latest:
        logger.error(
            "Invalid hour range | earliest=%d is after latest=%d",
            args.earliest, args.latest,
        )
        print(f"Earliest hour ({args.earliest}) cannot be after latest ({args.latest}).")
        return 2

    subset = df if args.day_type == "all" else (
        df[df["is_weekend"] == (0 if args.day_type == "weekday" else 1)]
    )
    window = subset[(subset["hour"] >= args.earliest) & (subset["hour"] <= args.latest)]

    if window.empty:
        logger.warning("No data in requested window | %d:00-%d:00 on %s",
                       args.earliest, args.latest, args.day_type)
        print("No data available for that window.")
        return 1

    hourly = window.groupby("hour")["traffic_volume"].mean().sort_values()
    overall = subset["traffic_volume"].mean()

    print(f"Recommended travel times — {args.day_type}, "
          f"between {args.earliest:02d}:00 and {args.latest:02d}:00\n")
    for rank, (hour, mean) in enumerate(hourly.head(3).items(), start=1):
        # Negative means below the day-type average, i.e. a quieter hour.
        relative = 100 * (mean - overall) / overall
        print(f"  {rank}. {hour:02d}:00   {mean:>7,.0f} vehicles/hour   "
              f"({relative:+.0f}% vs average)")

    worst_hour, worst_mean = hourly.index[-1], hourly.iloc[-1]
    print(f"\n  Avoid {worst_hour:02d}:00 — {worst_mean:,.0f} vehicles/hour, "
          f"the busiest hour in this window.")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic_app",
        description="Query the processed Metro Interstate traffic dataset.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lookup", help="Traffic and weather for a specific date and time.")
    p.add_argument("--datetime", required=True, help='e.g. "2017-05-03 08:00"')

    p = sub.add_parser("peak", help="Busiest and quietest hours.")
    p.add_argument("--day-type", choices=["all", "weekday", "weekend"], default="all")
    p.add_argument("--top", type=int, default=3, help="How many hours to list (default 3).")

    p = sub.add_parser("compare", help="Compare traffic across conditions.")
    p.add_argument("--weather", action="append",
                   help="Weather condition; repeat for several. Omit to compare weekday vs weekend.")

    p = sub.add_parser("advise", help="Recommend the quietest travel times in a window.")
    p.add_argument("--day-type", choices=["all", "weekday", "weekend"], default="weekday")
    p.add_argument("--earliest", type=int, default=6, help="Earliest acceptable hour (default 6).")
    p.add_argument("--latest", type=int, default=20, help="Latest acceptable hour (default 20).")

    return parser


COMMANDS = {
    "lookup": cmd_lookup,
    "peak": cmd_peak,
    "compare": cmd_compare,
    "advise": cmd_advise,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(LOG_DIR / "traffic_app.log", debug=args.debug)

    arguments = {k: v for k, v in vars(args).items() if k not in ("command", "debug")}
    logger.info("Command invoked | command=%s | arguments=%s", args.command, arguments)

    try:
        df = load_data()
        return COMMANDS[args.command](df, args)
    except FileNotFoundError:
        print("Processed data not found. Run the pipeline first:")
        print("  python -m capstone_part2.src.pipeline")
        print("  python -m capstone_part2.src.feature_engineering")
        return 1
    except Exception:
        logger.error("Command failed | command=%s", args.command, exc_info=True)
        print("Something went wrong. See logs/traffic_app.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
