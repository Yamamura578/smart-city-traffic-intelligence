"""Part 3, Task 3 — deep learning with explainability.

    python -m capstone_part3.src.deep_learning

Model: an LSTM for next-hour traffic demand. The data is an hourly time
series, so a recurrent model is the natural choice and is what the brief
recommends.

Explainability: SHAP is applied to a gradient-boosting surrogate trained on
the same inputs, not to the LSTM directly. The brief permits this explicitly
for sequence models. The justification is in the report: SHAP's Deep/Gradient
explainers assume feature independence, which is violated by construction in
a 24-step lag window where consecutive steps are strongly autocorrelated, so
attributions over the raw sequence would be misleading.

Framework note: TensorFlow segfaults on this project's target platform
(Python 3.13 / macOS arm64), so PyTorch is used instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from common.config import LOG_DIR, PART3_MODELS, RANDOM_STATE, REPO_ROOT
from common.logging_config import get_logger, setup_logging
from capstone_part3.src.data import FEATURE_COLUMNS, load_features

logger = get_logger(__name__)

FIGURES = REPO_ROOT / "capstone_part3" / "figures"
REPORTS = REPO_ROOT / "capstone_part3" / "reports"
MLFLOW_URI = f"sqlite:///{REPO_ROOT / 'capstone_part3' / 'mlflow.db'}"

SEQ_LEN = 24          # one full day of history
EPOCHS = 15
BATCH_SIZE = 256
HIDDEN = 64


class TrafficLSTM(nn.Module):
    """Two-layer LSTM over a 24-hour window, predicting the next hour."""

    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_sequences(df: pd.DataFrame, columns, target="traffic_volume"):
    """Turn the hourly table into (n, SEQ_LEN, f) windows.

    Windows are only emitted where the 24 preceding hours are genuinely
    consecutive; the dataset has a nine-month coverage gap and thousands of
    single-hour holes, and spanning them would fabricate sequences.
    """
    df = df.sort_values("date_time").reset_index(drop=True)
    values = df[columns].to_numpy(dtype=np.float32)
    target_values = df[target].to_numpy(dtype=np.float32)
    times = df["date_time"].to_numpy()

    X, y, idx = [], [], []
    expected = np.timedelta64(SEQ_LEN, "h")
    for i in range(SEQ_LEN, len(df)):
        if times[i] - times[i - SEQ_LEN] != expected:
            continue          # discontinuous window: skip
        X.append(values[i - SEQ_LEN:i])
        y.append(target_values[i])
        idx.append(i)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    skipped = (len(df) - SEQ_LEN) - len(X)
    logger.warning(
        "Dropped %d candidate windows | reason: the %d preceding hours were "
        "not consecutive (coverage gaps and de-duplicated hours)", skipped, SEQ_LEN,
    )
    logger.info("Built sequences | windows=%d | shape=%s", len(X), X.shape)
    return X, y, np.asarray(idx)


def train_lstm(X_tr, y_tr, X_te, y_te, y_mean, y_std, epochs=EPOCHS) -> tuple:
    torch.manual_seed(RANDOM_STATE)
    device = "cpu"
    model = TrafficLSTM(X_tr.shape[2]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    tr = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = torch.utils.data.DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
    X_te_t = torch.from_numpy(X_te)

    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(tr)

        model.eval()
        with torch.no_grad():
            pred = model(X_te_t).numpy()
        mae = mean_absolute_error(y_te * y_std + y_mean, pred * y_std + y_mean)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "test_mae": float(mae)})
        logger.info("Epoch %2d/%d | train_loss=%.4f | test MAE=%.1f vehicles/hour",
                    epoch, epochs, epoch_loss, mae)

    seconds = time.perf_counter() - start
    with torch.no_grad():
        pred = model(X_te_t).numpy()
    pred = pred * y_std + y_mean
    truth = y_te * y_std + y_mean
    metrics = {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(np.mean((truth - pred) ** 2))),
        "r2": float(r2_score(truth, pred)),
        "train_seconds": seconds,
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    logger.info("LSTM | MAE=%.1f RMSE=%.1f R2=%.4f | %d parameters | trained in %.1fs",
                metrics["mae"], metrics["rmse"], metrics["r2"],
                metrics["parameters"], seconds)
    return model, metrics, history, pred, truth


def run_shap_surrogate(df: pd.DataFrame) -> dict:
    """SHAP on a gradient-boosting surrogate over the same feature set."""
    logger.info("=== SHAP explainability (gradient-boosting surrogate) ===")
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["traffic_volume"]
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    surrogate = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
    )
    surrogate.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, surrogate.predict(X_te))
    r2 = r2_score(y_te, surrogate.predict(X_te))
    logger.info("Surrogate | MAE=%.1f | R2=%.4f", mae, r2)

    sample = X_te.sample(min(2000, len(X_te)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(sample)
    logger.info("SHAP values computed | sample=%d rows", len(sample))

    importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
    )
    logger.info("Top 10 features by mean |SHAP|:\n%s",
                importance.head(10).round(2).to_string())

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.title("SHAP feature importance — traffic volume surrogate", pad=20)
    plt.tight_layout()
    path = FIGURES / "06_shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved | path=%s", path)

    return {
        "surrogate_mae": float(mae),
        "surrogate_r2": float(r2),
        "top_features": importance.head(10).round(3).to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 3 deep learning with SHAP.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    setup_logging(LOG_DIR / "part3.log", debug=args.debug)
    epochs = args.epochs

    try:
        df = load_features().sort_values("date_time").reset_index(drop=True)
        logger.info("=== Task 3: LSTM demand prediction (PyTorch) ===")
        logger.warning(
            "Using PyTorch rather than TensorFlow | reason: TensorFlow segfaults "
            "on the target platform (Python 3.13 / macOS arm64)"
        )

        columns = ["traffic_volume"] + [c for c in FEATURE_COLUMNS if c != "hour"]
        X, y, _ = build_sequences(df, columns)

        # Chronological split: a random split would let the model see the
        # future, which for a time series is leakage.
        split = int(len(X) * 0.8)
        X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
        logger.info("Chronological split | train=%d | test=%d", len(X_tr), len(X_te))

        # Scale using training statistics only.
        mu = X_tr.reshape(-1, X_tr.shape[2]).mean(axis=0)
        sd = X_tr.reshape(-1, X_tr.shape[2]).std(axis=0) + 1e-8
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
        y_tr_s = (y_tr - y_mean) / y_std
        y_te_s = (y_te - y_mean) / y_std

        mlflow.set_tracking_uri(MLFLOW_URI)
        if mlflow.get_experiment_by_name("traffic-deep-learning") is None:
            mlflow.create_experiment(
                "traffic-deep-learning",
                artifact_location=(REPO_ROOT / "capstone_part3" / "mlruns").as_uri(),
            )
        mlflow.set_experiment("traffic-deep-learning")

        with mlflow.start_run(run_name="lstm_demand"):
            model, metrics, history, pred, truth = train_lstm(
                X_tr, y_tr_s, X_te, y_te_s, y_mean, y_std, epochs=epochs
            )
            mlflow.log_params({
                "architecture": "LSTM(2 layers)", "hidden": HIDDEN,
                "seq_len": SEQ_LEN, "epochs": epochs, "batch_size": BATCH_SIZE,
                "framework": f"pytorch {torch.__version__}",
            })
            mlflow.log_metrics({k: v for k, v in metrics.items()})

            PART3_MODELS.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), PART3_MODELS / "lstm_demand.pt")
            logger.info("Model saved | path=%s", PART3_MODELS / "lstm_demand.pt")

        # Prediction-vs-actual figure
        fig, ax = plt.subplots(figsize=(12, 5))
        n = min(500, len(truth))
        ax.plot(truth[:n], label="Actual", linewidth=1)
        ax.plot(pred[:n], label="LSTM prediction", linewidth=1, alpha=0.8)
        ax.set_xlabel("Test-set hour")
        ax.set_ylabel("Traffic volume")
        ax.set_title(f"LSTM next-hour demand (MAE = {metrics['mae']:.0f} vehicles/hour)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "07_lstm_predictions.png", dpi=150)
        plt.close(fig)
        logger.info("Figure saved | path=%s", FIGURES / "07_lstm_predictions.png")

        shap_results = run_shap_surrogate(df)

        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "deep_learning_results.json").write_text(json.dumps(
            {"lstm": metrics, "history": history, "shap": shap_results}, indent=2
        ))
        logger.info("Results written | path=%s", REPORTS / "deep_learning_results.json")
    except Exception:
        logger.error("Deep learning step aborted", exc_info=True)
        return 1

    logger.info("Deep learning and explainability completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
