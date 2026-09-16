# Part 2 — Reproducible Traffic Analytics Pipeline

Python data pipeline, feature engineering, visualisation and a command-line
application for the Metro Interstate Traffic Volume dataset.

## Contents

```
capstone_part2/
├── src/
│   ├── pipeline.py             Task 1 — load, validate, clean
│   ├── feature_engineering.py  Task 2 — ML-ready features
│   ├── visualizations.py       Task 3 — four Matplotlib figures
│   └── traffic_app.py          Task 4 — CLI application
├── figures/                    Generated PNGs
├── reports/                    Methodology and findings (1–2 pages)
├── pipeline.log                Sample log output for review
└── README.md
```

## Running

All commands are run **from the repository root** so that `common/` resolves
as a package. Add `--debug` to any of them to see intermediate values.

```bash
python -m capstone_part2.src.pipeline                # → data/processed/traffic_cleaned.csv
python -m capstone_part2.src.feature_engineering     # → data/processed/traffic_features.csv
python -m capstone_part2.src.visualizations          # → capstone_part2/figures/*.png
```

## CLI application

```bash
python -m capstone_part2.src.traffic_app lookup  --datetime "2017-05-03 08:00"
python -m capstone_part2.src.traffic_app peak    --day-type weekday --top 3
python -m capstone_part2.src.traffic_app compare --weather Clear --weather Snow
python -m capstone_part2.src.traffic_app advise  --day-type weekday --earliest 6 --latest 20
```

`compare` with no `--weather` argument compares weekday against weekend.

## Data flow

| Stage | Input | Output | Rows |
|-------|-------|--------|-----:|
| pipeline | `data/raw/Metro_Interstate_Traffic_Volume.csv` | `traffic_cleaned.csv` | 48,204 → 40,575 |
| features | `traffic_cleaned.csv` | `traffic_features.csv` | 40,575 (10 → 42 cols) |
| figures | `traffic_features.csv` | 4 PNGs | — |

Part 3 reads `traffic_features.csv`.

## Logging

See the root `README.md` for the full logging documentation. In brief: every
module uses `get_logger(__name__)`; handlers are attached only in entry points;
output goes to both console and `logs/pipeline.log` with the format
`timestamp | level | module | message`. DEBUG output requires `--debug`.
`pipeline.log` in this folder is a committed sample so the logging can be
reviewed without re-running the project.
