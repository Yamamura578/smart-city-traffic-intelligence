# Smart City Traffic Intelligence

Capstone project for the NUS SOC Advanced Certificate in AI, Machine Learning and Data Science.

End-to-end analysis of ~48,000 hourly traffic records for westbound I-94 near
Minneapolis–St Paul (October 2012 – September 2018), progressing from SQL and
dashboard analytics through a reproducible Python pipeline to machine learning,
explainability and a simulated deployment.

| Part | Focus | Folder |
| --- | --- | --- |
| 1 | SQL, descriptive statistics, probability, Power BI | `capstone_part1/` |
| 2 | Python pipeline, feature engineering, visualisation, CLI app | `capstone_part2/` |
| 3 | Supervised/unsupervised ML, deep learning, MLOps, recommendations | `capstone_part3/` |

## Repository structure

```
.
├── common/                     # shared across parts 2 and 3
│   ├── config.py               # paths, schema, thresholds, constants
│   └── logging_config.py       # setup_logging() and get_logger()
├── data/
│   ├── raw/                    # Metro_Interstate_Traffic_Volume.csv (as supplied)
│   └── processed/              # pipeline outputs + traffic.db (git-ignored)
├── logs/                       # pipeline.log — sample output committed for review
├── capstone_part1/
│   ├── sql/                    # SQLite load + analysis queries
│   ├── notebooks/              # statistics and probability analysis
│   ├── powerbi/                # .pbix dashboard + screenshots
│   └── reports/                # Data Analytics Insights Report (1–2 pages)
├── capstone_part2/
│   ├── src/                    # pipeline.py, feature_engineering.py,
│   │                           # visualizations.py, traffic_app.py
│   ├── figures/                # saved Matplotlib output
│   └── reports/                # methodology and findings (1–2 pages)
└── capstone_part3/
    ├── src/                    # modelling, clustering, recommender, monitoring
    ├── notebooks/              # model development and explainability
    ├── models/                 # serialised models
    ├── api/                    # FastAPI deployment mock-up
    ├── mlruns/                 # MLflow tracking output
    └── reports/                # final capstone report + bias and fairness report
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

All commands are run **from the repository root** so that `common/` resolves as
a package:

```bash
# Part 2 — clean the raw data
python -m capstone_part2.src.pipeline
python -m capstone_part2.src.pipeline --debug     # adds intermediate values

# Part 2 — features and figures
python -m capstone_part2.src.feature_engineering
python -m capstone_part2.src.visualizations

# Part 2 — CLI application
python -m capstone_part2.src.traffic_app peak --day-type weekday
```

Part 1 SQL:

```bash
sqlite3 data/processed/traffic.db < capstone_part1/sql/01_load_and_verify.sql
```

## Logging

Logging uses the standard library `logging` module throughout. There are no
`print()` statements for internal status or progress; `print()` appears only in
`traffic_app.py`, for output the end user asked for.

**Where logs are written.** Every run writes to both the console and
`logs/pipeline.log`, in append mode, so the file is a cumulative audit trail. A
sample `pipeline.log` is committed so it can be reviewed without re-running the
project.

**How it is configured.** `common/logging_config.py` exposes two functions:

- `setup_logging(log_file, debug=False)` attaches the console and file handlers.
  It is called **only** from entry-point scripts, inside `if __name__ == "__main__"`.
- `get_logger(__name__)` is called at the top of every module. No module
  configures handlers itself and the root logger is never used directly, so the
  modules stay importable from notebooks without side effects.

**Format.** `timestamp | level | module name | message`, for example:

```
2026-09-14 09:12:04 | WARNING  | capstone_part2.src.pipeline | Imputed 10 rows in 'temp' (0 K sensor failures) using monthly median
```

**Levels.**

| Level | Used for |
| --- | --- |
| `DEBUG` | Intermediate values that exist only for troubleshooting — quartile thresholds, per-month imputation values. Off unless `--debug` is passed. |
| `INFO` | Expected milestones — data loaded (with row/column counts), a cleaning step completed, dataset shape before and after feature engineering, a figure saved (with its path), a model saved, a CLI command invoked. |
| `WARNING` | Recoverable but noteworthy events — rows dropped, values imputed, outliers capped, a monitoring alert triggered. Always states the affected row count and the reason. |
| `ERROR` | Something that stops the pipeline completing. Logged once at the top level with `exc_info=True`, then the process exits gracefully. |

## Data notes

Characteristics of the supplied dataset that shape the analysis and are
discussed in the reports:

- **Coverage gap.** Recording is sparse from August 2014 to mid-2015 (August 2014
  has 136 rows; June 2015 has 186). Raw yearly totals therefore reflect sensor
  uptime rather than demand, and yearly comparisons are reported as mean hourly
  volume alongside totals.
- **Sensor failures.** 10 rows record `temp = 0 K`; one row records
  `rain_1h = 9831.3 mm`. Both are imputed from the monthly median.
- **Duplicates.** 17 fully identical rows; ~7,600 repeated timestamps where the
  same hour was logged under multiple weather descriptions.
- **Holiday sparsity.** `holiday` is populated only on the 00:00 row of each
  holiday (61 of 48,204 rows) and is forward-filled across the day before use.

## Accident data

No accident dataset was supplied with this capstone. The classification task in
Part 3 uses a **documented proxy label** derived from congestion quartiles
combined with severe or low-visibility weather. It demonstrates the
classification workflow and is **not** a prediction of real accidents. Its
limitations are discussed in `capstone_part3/reports/`.
