# Part 3 — Machine Learning, Deep Learning and AI

## Accident data: a documented proxy was used

**No real accident dataset was sourced.** None was supplied with the capstone.
The classification task uses a documented proxy label:

```
high_risk = congestion in {High, Severe} AND (severe weather OR low visibility)
```

This demonstrates the classification workflow. It is **not** a prediction of
real accidents, and its metrics must not be read as accident-prediction
accuracy. The label is partly circular — weather appears in both the inputs
and the label definition — which is why the classifier scores so highly. See
`reports/bias_and_fairness_report.md` section 1.

## Contents

```
capstone_part3/
├── src/
│   ├── data.py            Shared loading, proxy label, train/test split
│   ├── supervised.py      Task 1 — classification + regression, MLflow
│   ├── unsupervised.py    Task 2 — K-means + association rules
│   ├── deep_learning.py   Task 3 — LSTM (PyTorch) + SHAP
│   ├── recommender.py     Task 5 — travel-timing recommendations
│   └── monitoring.py      Tasks 6.4/6.5 — drift detection + alerting
├── api/main.py            Task 6.3 — FastAPI deployment mock-up
├── models/                Serialised models (.joblib, .pt)
├── figures/               Cluster, SHAP and LSTM plots
├── mlruns/ + mlflow.db    Task 4 / 6.2 — MLflow tracking store
└── reports/
    ├── final_capstone_report.md
    ├── bias_and_fairness_report.md
    ├── model_results.json, unsupervised_results.json,
    ├── deep_learning_results.json, monitoring_report.json
    └── association_rules.csv
```

## Running

From the repository root, after Part 2 has produced `traffic_features.csv`:

```bash
python -m capstone_part3.src.supervised          # ~1 min
python -m capstone_part3.src.unsupervised        # ~30 s
python -m capstone_part3.src.deep_learning       # ~2 min (15 epochs)
python -m capstone_part3.src.monitoring          # ~10 s
python -m capstone_part3.src.recommender --day-type weekday --weather Rain
```

Add `--debug` to any of them for intermediate values.

API:

```bash
uvicorn capstone_part3.api.main:app --reload
# interactive docs at http://127.0.0.1:8000/docs
```

MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///capstone_part3/mlflow.db
```

## Headline results

| Task | Model | Result |
|------|-------|--------|
| Regression | Gradient boosting | MAE 230.3, R² 0.9649 |
| Regression | **LSTM** | **MAE 196.4, R² 0.9786** |
| Classification | Random forest | F1 0.9543, ROC AUC 0.9988 *(proxy label — see caveat)* |
| Clustering | K-means, k=8 | Silhouette 0.3268 |
| Association rules | apriori | 142 rules, top lift 3.79 |
| Monitoring | PSI + error ratio | **ALERT** (3/27 features drifted; error improved) |

SHAP on the surrogate: `hour_cos` (1451), `hour` (427), `hour_sin` (193)
dominate. No weather feature reaches the top six.

## Framework note

TensorFlow segfaults on Python 3.13 / macOS arm64, so the deep learning task
uses **PyTorch 2.14**. MLflow 3.x rejects the filesystem tracking backend, so
the tracking store is SQLite.
