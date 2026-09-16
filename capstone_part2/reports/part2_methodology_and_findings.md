# Part 2 — Methodology and Findings

**Smart City Traffic Intelligence** · Python data pipeline, feature engineering, visualisation and CLI
**Author:** Stefan Mueller

---

## 1. Pipeline design

The pipeline is four modules sharing one configuration and one logging setup:

```
pipeline.py            raw CSV  →  data/processed/traffic_cleaned.csv
feature_engineering.py cleaned  →  data/processed/traffic_features.csv
visualizations.py      features →  capstone_part2/figures/*.png
traffic_app.py         features →  answers to user queries
```

Every module calls `get_logger(__name__)`; handlers are attached only in entry
points, via `setup_logging()` in `common/logging_config.py`. The feature table is
the handoff point to Part 3.

Each cleaning step is its own function, so each emits its own log record and the
processing history can be read off the log file without reading the code.

## 2. Cleaning decisions

Input: **48,204 rows × 9 columns**. Output: **40,575 rows × 10 columns**.

| Step | Action | Rows affected |
|------|--------|--------------:|
| Datetime parsing | `errors="coerce"`, drop unparseable | 0 (all valid) |
| Holiday standardisation | Forward-fill tags across each holiday date | 1,348 filled from 61 tags |
| Exact duplicates | Dropped | 17 |
| Repeated timestamps | Kept most severe weather per hour | 7,612 |
| Temperature outliers | Monthly-median imputation | 10 |
| Rainfall outliers | Monthly-median imputation | 1 |
| Zero traffic volume | Flagged, retained | 2 |

Three decisions deserve justification.

**Duplicate timestamps (7,612 rows, 19% of the data).** The raw file logs some
hours several times, once per weather description. The analysis grain is the
hour, so one row per hour is required. Rather than keeping the first row
arbitrarily, the row with the **most severe weather** is retained, using an
explicit severity ranking (Squall > Thunderstorm > Snow > Rain > … > Clear). For
traffic and safety work, under-reporting severe weather is the more costly
error. The result is exactly 40,575 rows — one per distinct timestamp.

**Monthly rather than global median imputation.** A single global median would
assign an annual average temperature to a January sensor failure. Medians are
computed per calendar month, in an explicit loop, from valid rows only so that
the impossible values cannot contaminate their own replacement. Ten rows
recorded 0 K (absolute zero) and one recorded 9,831.3 mm of rain in an hour.

**Ordering matters.** Holiday forward-fill runs *before* de-duplication. In an
earlier version it ran after, and the de-duplication rule silently discarded 8
of the 61 tagged midnight rows before their dates had been recorded — losing
eight holidays. The fixed order captures all 61.

## 3. Feature engineering

**40,575 × 10 → 40,575 × 42.**

**Time (9 features).** `hour`, `day_of_week`, `month`, `year`, `is_weekend`, plus
cyclical encodings `hour_sin`/`hour_cos` and `dow_sin`/`dow_cos`. Cyclical
encoding matters because as integers, hour 23 and hour 0 are 23 units apart when
in reality they are adjacent; projecting onto a circle fixes this for any linear
or distance-based model.

**Weather (14 features).** Categories with fewer than 100 observations (Smoke
n=16, Squall n=4) are collapsed into `Other` before one-hot encoding — a dummy
variable fitted on 4 rows is noise. Derived flags: `is_precipitation`,
`is_low_visibility`, `has_rain`, `has_snow`, and `temp_celsius`.

**Scaled (4 features).** Z-score and min-max versions of `temp` and
`clouds_all`. Both are produced because they serve different models: z-scores
for linear and distance-based methods, min-max for the Part 3 neural network.
Scaling parameters are logged at DEBUG so the transform can be reproduced.

**Target.** Part 1 used a fixed threshold of 5,500 vehicles/hour, which is
defensible for reporting but arbitrary for modelling and badly imbalanced (15%
positive). The modelling target is therefore **quartile-based**:

| Class | Rule | Rows |
|-------|------|-----:|
| Low | ≤ 1,248.5 | 10,144 |
| Moderate | ≤ 3,427.0 | 10,144 |
| High | ≤ 4,952.0 | 10,145 |
| Severe | > 4,952.0 | 10,142 |

Four classes of near-equal size. The binary `is_congested` flag at the Part 1
threshold is retained so results stay comparable across parts.

## 4. Visualisations and findings

**`01_hourly_profile.png` — demand by hour, weekday vs weekend.** The headline
result. Weekday traffic is twin-peaked: 6,061 vehicles/hour at 07:00 and 6,241
at 16:00, against a trough of 303 at 02:00 — a **20.6× peak-to-trough ratio**.
Weekend traffic has no morning peak at all, rising gradually to a flat plateau
of ~4,400 between 12:00 and 16:00. These are two different demand regimes, and
any model that ignores day type will fit neither well.

**`02_weather_impact.png` — traffic by weather, annotated with n.** Bars for
categories with fewer than 100 observations are coloured red. Across all
categories the range is 2,061.8 (Squall) to 3,616.7 (Clouds), a difference of
1,555. Excluding small samples, the range narrows to 2,653.8 (Fog) to 3,616.7,
a difference of **962.9** — about a sixth of the swing produced by hour of day
alone. Squall's apparent effect rests on four observations and should not
inform any decision.

**`03_temperature_scatter.png` — temperature vs volume.** Pearson r = **0.140**
on the cleaned data (0.132 on raw), r² = 0.020. The binned-mean overlay shows a
gentle rise from about −20 °C to +25 °C, then a slight fall. Temperature
explains roughly 2% of variance in hourly volume. The relationship is almost
certainly confounded: warm hours are daytime hours.

**`04_coverage_and_distribution.png` — coverage and distribution.** The left
panel makes the sensor outage unmissable: monthly record counts collapse to 136
hours in August 2014 and do not recover until mid-2015. The right panel shows
why the mean is a poor summary — the distribution is broad and near-bimodal,
with mean and median close together but neither describing a typical hour.

## 5. Command-line application

Four commands, each logging its invocation and arguments at INFO:

| Command | Purpose |
|---------|---------|
| `lookup --datetime` | Traffic, weather and congestion for one hour |
| `peak --day-type --top` | Busiest and quietest hours, with peak-to-trough ratio |
| `compare --weather` | Mean traffic by weather condition, or weekday vs weekend |
| `advise --day-type --earliest --latest` | Quietest travel times in a window |

Invalid input produces a logged ERROR and a plain-language message, never a
traceback: unparseable dates, unknown weather conditions, and inverted or
out-of-range hour arguments are all handled explicitly. `advise` is the seed of
the Part 3 recommendation component.

`print()` appears only in this module, and only for answers the user asked for.

## 6. Logging

Both console and `logs/pipeline.log`, formatted as
`timestamp | level | module | message`. A representative run produces 53 lines,
of which 7 are WARNING (every row-level change) and 20 are DEBUG (only with
`--debug`).

| Level | Used for | Example |
|-------|----------|---------|
| DEBUG | Internal values | Quartile thresholds Q1=1248.5, Q2=3427.0, Q3=4952.0 |
| INFO | Milestones | `Loaded raw data \| rows=48204 \| columns=9` |
| WARNING | Recoverable changes | `Imputed 10 rows in 'temp' \| reason: outside plausible range` |
| ERROR | Pipeline-stopping faults | Logged once with `exc_info=True`, exit code 1 |

Third-party loggers (matplotlib, PIL) are pinned to WARNING so that `--debug`
shows this project's diagnostics rather than hundreds of font-cache lines.

## 7. Limitations

- **19% of rows were discarded** to reach one row per hour. An alternative would
  be aggregating weather per hour rather than selecting one observation.
- **The coverage gap is not imputed.** Nine months are absent; filling them
  would fabricate data. Models in Part 3 inherit this gap.
- **Two zero-volume rows are retained.** They are implausible, but the true
  value is unknown and imputing would be a guess.
- **Scaling is fitted on the full dataset**, which leaks distributional
  information into the Part 3 test split. For a production model, scalers should
  be fitted on training data only.
