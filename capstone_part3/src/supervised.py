"""Part 3, Task 1 — supervised models, with MLflow experiment tracking.

    python -m capstone_part3.src.supervised

Classification: proxy accident-risk label, logistic regression vs random forest.
Regression:     traffic volume, linear regression vs gradient boosting.

Every run is logged to MLflow (Task 4 and Task 6.2): parameters, metrics and
the fitted model, under two named experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.config import LOG_DIR, PART3_MODELS, RANDOM_STATE, REPO_ROOT
from common.logging_config import get_logger, setup_logging
from capstone_part3.src.data import prepare

logger = get_logger(__name__)

# MLflow 3.x rejects the legacy filesystem tracking backend, so the tracking
# store is SQLite (its recommended replacement) and artifacts are written
# alongside it. Both live inside capstone_part3/ and are committed, so the
# experiment history can be reviewed without re-running anything.
MLFLOW_DB = REPO_ROOT / "capstone_part3" / "mlflow.db"
MLFLOW_ARTIFACTS = REPO_ROOT / "capstone_part3" / "mlruns"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB}"


def init_mlflow(experiment: str):
    """Point MLflow at the SQLite store and select an experiment."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment)
    logger.info("MLflow experiment selected | %s", experiment)
RESULTS_PATH = REPO_ROOT / "capstone_part3" / "reports" / "model_results.json"


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def evaluate_classifier(name, model, X_te, y_te) -> dict:
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, y_prob),
    }
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    logger.info(
        "%s | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        name, metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["roc_auc"],
    )
    logger.info("%s | confusion matrix: TN=%d FP=%d FN=%d TP=%d", name, tn, fp, fn, tp)
    if metrics["recall"] < 0.5:
        logger.warning(
            "%s recalls only %.1f%% of high-risk hours | reason: class imbalance",
            name, 100 * metrics["recall"],
        )
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def run_classification() -> dict:
    logger.info("=== Task 1a: classification on the proxy accident-risk label ===")
    df, X, y, X_tr, X_te, y_tr, y_te = prepare("high_risk", stratify_target=True)

    baseline = float(1 - y_te.mean())
    logger.warning(
        "Majority-class baseline accuracy is %.4f | accuracy alone is not an "
        "adequate metric for this target", baseline,
    )

    candidates = {
        # Logistic regression needs scaling; the forest does not, but a
        # pipeline keeps the two interfaces identical.
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=18, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }

    init_mlflow("traffic-accident-risk-classification")
    results = {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            start = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_seconds = time.perf_counter() - start

            metrics = evaluate_classifier(name, model, X_te, y_te)
            mlflow.log_params({
                "model": name,
                "n_features": X.shape[1],
                "n_train": len(X_tr),
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            })
            mlflow.log_metrics({**{k: float(v) for k, v in metrics.items()},
                                "fit_seconds": fit_seconds,
                                "baseline_accuracy": baseline})
            # MLflow 3.x defaults to skops serialisation, which refuses to
            # round-trip sklearn tree objects; cloudpickle handles them.
            mlflow.sklearn.log_model(
                model, name=name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            logger.info("%s | fitted in %.2fs | logged to MLflow", name, fit_seconds)

            PART3_MODELS.mkdir(parents=True, exist_ok=True)
            path = PART3_MODELS / f"classifier_{name}.joblib"
            joblib.dump(model, path)
            logger.info("Model saved | path=%s", path)
            results[name] = {**metrics, "fit_seconds": fit_seconds}

    best = max(results, key=lambda k: results[k]["roc_auc"])
    logger.info("Best classifier by ROC AUC: %s (%.4f)", best, results[best]["roc_auc"])
    results["_baseline_accuracy"] = baseline
    results["_best"] = best
    return results


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------
def evaluate_regressor(name, model, X_te, y_te) -> dict:
    y_pred = model.predict(X_te)
    metrics = {
        "mae": float(mean_absolute_error(y_te, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_te - y_pred) ** 2))),
        "r2": float(r2_score(y_te, y_pred)),
    }
    logger.info("%s | MAE=%.1f RMSE=%.1f R2=%.4f",
                name, metrics["mae"], metrics["rmse"], metrics["r2"])
    return metrics


def run_regression() -> dict:
    logger.info("=== Task 1b: regression on hourly traffic volume ===")
    df, X, y, X_tr, X_te, y_tr, y_te = prepare("traffic_volume")

    naive_mae = float(np.mean(np.abs(y_te - y_tr.mean())))
    logger.warning(
        "Mean-prediction baseline MAE is %.1f vehicles/hour | any model must "
        "beat this", naive_mae,
    )

    candidates = {
        "linear_regression": Pipeline([
            ("scale", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.9, random_state=RANDOM_STATE,
        ),
    }

    init_mlflow("traffic-volume-regression")
    results = {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            start = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_seconds = time.perf_counter() - start

            metrics = evaluate_regressor(name, model, X_te, y_te)
            mlflow.log_params({
                "model": name, "n_features": X.shape[1], "n_train": len(X_tr),
                "random_state": RANDOM_STATE,
            })
            mlflow.log_metrics({**metrics, "fit_seconds": fit_seconds,
                                "baseline_mae": naive_mae})
            # MLflow 3.x defaults to skops serialisation, which refuses to
            # round-trip sklearn tree objects; cloudpickle handles them.
            mlflow.sklearn.log_model(
                model, name=name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            logger.info("%s | fitted in %.2fs | logged to MLflow", name, fit_seconds)

            path = PART3_MODELS / f"regressor_{name}.joblib"
            joblib.dump(model, path)
            logger.info("Model saved | path=%s", path)
            results[name] = {**metrics, "fit_seconds": fit_seconds}

    best = max(results, key=lambda k: results[k]["r2"])
    improvement = 100 * (naive_mae - results[best]["mae"]) / naive_mae
    logger.info(
        "Best regressor by R2: %s (R2=%.4f, MAE=%.1f, %.1f%% better than baseline)",
        best, results[best]["r2"], results[best]["mae"], improvement,
    )
    results["_baseline_mae"] = naive_mae
    results["_best"] = best
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and compare Part 3 supervised models."
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(LOG_DIR / "part3.log", debug=args.debug)
    logger.info("MLflow tracking URI | %s", MLFLOW_URI)

    try:
        results = {
            "classification": run_classification(),
            "regression": run_regression(),
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        logger.info("Results written | path=%s", RESULTS_PATH)
    except Exception:
        logger.error("Supervised modelling aborted", exc_info=True)
        return 1

    logger.info("Supervised modelling completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
