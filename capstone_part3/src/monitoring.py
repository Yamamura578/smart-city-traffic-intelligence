"""Part 3, Tasks 6.4 and 6.5 — drift monitoring and alerting.

    python -m capstone_part3.src.monitoring

Simulates production monitoring by splitting the data chronologically into a
"reference" period (what the model was trained on) and a "current" period
(what it is now seeing), then checking two things:

  * Feature distribution drift  — Population Stability Index per feature
  * Prediction error drift      — MAE on current data vs the reference MAE

Each check emits PASS or ALERT. Thresholds are the conventional PSI bands.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from common.config import LOG_DIR, PART3_MODELS, REPO_ROOT
from common.logging_config import get_logger, setup_logging
from capstone_part3.src.data import FEATURE_COLUMNS, load_features

logger = get_logger(__name__)
REPORTS = REPO_ROOT / "capstone_part3" / "reports"

PSI_WARN, PSI_ALERT = 0.10, 0.25      # conventional bands
ERROR_ALERT_RATIO = 1.25              # 25% worse than reference MAE


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    logger.info("=== Task 6.4a: feature distribution drift (PSI) ===")
    results, drifted = {}, []
    for column in FEATURE_COLUMNS:
        value = psi(reference[column].to_numpy(), current[column].to_numpy())
        status = "ALERT" if value >= PSI_ALERT else "WARN" if value >= PSI_WARN else "PASS"
        results[column] = {"psi": round(value, 4), "status": status}
        if status != "PASS":
            drifted.append((column, value, status))

    for column, value, status in sorted(drifted, key=lambda x: -x[1]):
        logger.warning("Feature drift %s | %s | PSI=%.4f (warn>=%.2f, alert>=%.2f)",
                       status, column, value, PSI_WARN, PSI_ALERT)
    if not drifted:
        logger.info("No feature drift detected across %d features", len(FEATURE_COLUMNS))

    overall = ("ALERT" if any(s == "ALERT" for _, _, s in drifted)
               else "WARN" if drifted else "PASS")
    logger.info("Feature drift overall status | %s", overall)
    return {"status": overall, "n_drifted": len(drifted), "per_feature": results}


def check_error_drift(model, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    logger.info("=== Task 6.4b: prediction error drift ===")
    ref_mae = float(mean_absolute_error(
        reference["traffic_volume"], model.predict(reference[FEATURE_COLUMNS])))
    cur_mae = float(mean_absolute_error(
        current["traffic_volume"], model.predict(current[FEATURE_COLUMNS])))
    ratio = cur_mae / ref_mae if ref_mae else float("inf")
    status = "ALERT" if ratio >= ERROR_ALERT_RATIO else "PASS"

    logger.info("Reference MAE=%.1f | current MAE=%.1f | ratio=%.3f",
                ref_mae, cur_mae, ratio)
    if status == "ALERT":
        logger.warning(
            "Prediction error drift ALERT | current MAE is %.1f%% worse than "
            "reference | threshold=%.0f%%",
            100 * (ratio - 1), 100 * (ERROR_ALERT_RATIO - 1),
        )
    return {"status": status, "reference_mae": round(ref_mae, 1),
            "current_mae": round(cur_mae, 1), "ratio": round(ratio, 3)}


def main() -> int:
    p = argparse.ArgumentParser(description="Part 3 drift monitoring and alerting.")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--split-date", default="2017-06-01",
                   help="Reference period ends / current period begins here.")
    args = p.parse_args()
    setup_logging(LOG_DIR / "part3.log", debug=args.debug)

    try:
        df = load_features()
        cutoff = pd.Timestamp(args.split_date)
        reference = df[df["date_time"] < cutoff]
        current = df[df["date_time"] >= cutoff]
        logger.info(
            "Monitoring windows | reference=%d rows (<%s) | current=%d rows (>=%s)",
            len(reference), cutoff.date(), len(current), cutoff.date(),
        )
        if len(current) < 500:
            logger.warning("Current window has only %d rows | drift estimates "
                           "will be noisy", len(current))

        model_path = PART3_MODELS / "regressor_gradient_boosting.joblib"
        model = joblib.load(model_path)
        logger.info("Loaded monitored model | path=%s", model_path)

        feature_drift = check_feature_drift(reference, current)
        error_drift = check_error_drift(model, reference, current)

        overall = "ALERT" if "ALERT" in (feature_drift["status"], error_drift["status"]) \
            else "WARN" if feature_drift["status"] == "WARN" else "PASS"

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "split_date": args.split_date,
            "reference_rows": len(reference),
            "current_rows": len(current),
            "feature_drift": feature_drift,
            "error_drift": error_drift,
            "overall_status": overall,
        }
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "monitoring_report.json").write_text(json.dumps(report, indent=2))
        logger.info("Monitoring report written | path=%s",
                    REPORTS / "monitoring_report.json")

        # Task 6.5 — the alerting surface, for a human to read.
        banner = {"PASS": "PASS / Normal",
                  "WARN": "WARN / Monitor",
                  "ALERT": "ALERT / Requires investigation"}[overall]
        print("\n" + "=" * 58)
        print(f"  MODEL MONITORING STATUS:  {banner}")
        print("=" * 58)
        print(f"  Feature drift    : {feature_drift['status']:<6} "
              f"({feature_drift['n_drifted']} of {len(FEATURE_COLUMNS)} features drifted)")
        print(f"  Prediction error : {error_drift['status']:<6} "
              f"(MAE {error_drift['reference_mae']:.0f} -> {error_drift['current_mae']:.0f}, "
              f"ratio {error_drift['ratio']:.2f})")
        print("=" * 58 + "\n")
        if overall != "PASS":
            worst = sorted(
                ((k, v["psi"]) for k, v in feature_drift["per_feature"].items()
                 if v["status"] != "PASS"), key=lambda x: -x[1])[:5]
            print("  Features to investigate:")
            for name, value in worst:
                print(f"    {name:<22} PSI = {value:.3f}")
            print()
    except Exception:
        logger.error("Monitoring run aborted", exc_info=True)
        return 1

    logger.info("Monitoring completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
