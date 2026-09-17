"""Part 3, Task 2 — unsupervised learning.

    python -m capstone_part3.src.unsupervised

K-means clusters traffic *conditions* (no accident data exists to cluster).
Association rule mining discretises time, day type and weather, then mines
rules that predict congestion level.
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common.config import LOG_DIR, RANDOM_STATE, REPO_ROOT
from common.logging_config import get_logger, setup_logging
from capstone_part3.src.data import add_proxy_risk_label, load_features

logger = get_logger(__name__)

FIGURES = REPO_ROOT / "capstone_part3" / "figures"
REPORTS = REPO_ROOT / "capstone_part3" / "reports"

# Weather severity, used as a single ordinal feature for clustering.
SEVERITY = {
    "Clear": 0, "Clouds": 1, "Mist": 2, "Haze": 2, "Smoke": 2, "Fog": 3,
    "Drizzle": 3, "Rain": 4, "Snow": 5, "Thunderstorm": 6, "Squall": 7,
}


# --------------------------------------------------------------------------
# K-means
# --------------------------------------------------------------------------
def run_clustering(df: pd.DataFrame, k_range=range(2, 9)) -> dict:
    """Cluster traffic conditions on hour, weather severity and volume."""
    logger.info("=== Task 2a: K-means clustering of traffic conditions ===")

    work = df.copy()
    work["weather_severity"] = work["weather_main"].map(SEVERITY).fillna(0)
    features = ["hour", "weather_severity", "traffic_volume", "temp_celsius", "is_weekend"]
    X = StandardScaler().fit_transform(work[features])

    # Choose k by silhouette score on a subsample (silhouette is O(n^2)).
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        scores[k] = float(silhouette_score(X[sample_idx], labels[sample_idx]))
        logger.debug("k=%d | silhouette=%.4f | inertia=%.1f", k, scores[k], km.inertia_)

    best_k = max(scores, key=scores.get)
    logger.info("Selected k=%d by silhouette score (%.4f)", best_k, scores[best_k])

    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    work["cluster"] = km.fit_predict(X)

    profile = work.groupby("cluster").agg(
        n=("traffic_volume", "size"),
        mean_hour=("hour", "mean"),
        mean_volume=("traffic_volume", "mean"),
        mean_severity=("weather_severity", "mean"),
        mean_temp=("temp_celsius", "mean"),
        weekend_share=("is_weekend", "mean"),
    ).round(2)
    logger.info("Cluster profiles:\n%s", profile.to_string())

    # Plain-language interpretation, derived from the profile rather than
    # asserted: each cluster is described by where it sits on volume and hour.
    interpretations = {}
    vol_median = work["traffic_volume"].median()
    for cid, row in profile.iterrows():
        when = ("overnight" if row["mean_hour"] < 6 else
                "morning" if row["mean_hour"] < 11 else
                "midday" if row["mean_hour"] < 15 else
                "afternoon/evening" if row["mean_hour"] < 20 else "late evening")
        load = "high-volume" if row["mean_volume"] > vol_median else "low-volume"
        weather = "adverse weather" if row["mean_severity"] > 2 else "mild weather"
        day = "mostly weekend" if row["weekend_share"] > 0.5 else "mostly weekday"
        interpretations[int(cid)] = (
            f"{load} {when} hours, {day}, {weather} "
            f"(n={int(row['n']):,}, mean volume {row['mean_volume']:.0f})"
        )
        logger.info("Cluster %d: %s", cid, interpretations[int(cid)])

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(list(scores), list(scores.values()), "-o")
    ax1.axvline(best_k, color="#c44e52", linestyle="--", label=f"selected k={best_k}")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Silhouette score")
    ax1.set_title("Cluster count selection")
    ax1.legend(); ax1.grid(alpha=0.3)

    for cid in sorted(work["cluster"].unique()):
        subset = work[work["cluster"] == cid]
        ax2.scatter(subset["hour"], subset["traffic_volume"], s=4, alpha=0.25,
                    label=f"Cluster {cid}")
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Traffic volume")
    ax2.set_title("Traffic condition clusters")
    ax2.legend(markerscale=3, fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "05_clusters.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info("Figure saved | path=%s", path)

    return {
        "best_k": best_k,
        "silhouette_scores": scores,
        "profiles": profile.to_dict(orient="index"),
        "interpretations": interpretations,
    }


# --------------------------------------------------------------------------
# Association rules
# --------------------------------------------------------------------------
def run_association_rules(df: pd.DataFrame, min_support=0.02, min_lift=1.2) -> dict:
    """Mine rules linking time, day type and weather to congestion level."""
    logger.info("=== Task 2b: association rule mining ===")

    work = df.copy()
    work["time_band"] = pd.cut(
        work["hour"],
        bins=[-1, 5, 9, 15, 19, 23],
        labels=["night", "morning_peak", "midday", "evening_peak", "late_evening"],
    )
    work["day_type"] = np.where(work["is_weekend"] == 1, "weekend", "weekday")
    work["weather_band"] = np.where(
        work["weather_main"].isin(["Clear", "Clouds"]), "fair_weather", "adverse_weather"
    )
    work["temp_band"] = pd.cut(
        work["temp_celsius"], bins=[-40, 0, 15, 50],
        labels=["freezing", "cool", "warm"],
    )

    basket = pd.get_dummies(
        work[["time_band", "day_type", "weather_band", "temp_band", "congestion_level"]]
        .astype(str),
        prefix_sep="=",
    ).astype(bool)
    logger.info("Built transaction matrix | rows=%d | items=%d",
                basket.shape[0], basket.shape[1])

    itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    logger.info("Frequent itemsets found | count=%d | min_support=%.3f",
                len(itemsets), min_support)

    rules = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    # Keep only rules that *predict* congestion, which is the question asked.
    rules = rules[rules["consequents"].apply(
        lambda c: all(str(i).startswith("congestion_level=") for i in c)
    )]
    rules = rules.sort_values("lift", ascending=False)
    logger.info("Rules predicting congestion | count=%d | min_lift=%.2f",
                len(rules), min_lift)

    if rules.empty:
        logger.warning("No rules met the thresholds | support=%.3f lift=%.2f",
                       min_support, min_lift)
        return {"n_rules": 0, "top_rules": []}

    def fmt(items):
        return ", ".join(sorted(str(i) for i in items))

    top = []
    for _, r in rules.head(8).iterrows():
        entry = {
            "antecedents": fmt(r["antecedents"]),
            "consequents": fmt(r["consequents"]),
            "support": round(float(r["support"]), 4),
            "confidence": round(float(r["confidence"]), 4),
            "lift": round(float(r["lift"]), 3),
        }
        entry["plain_language"] = (
            f"When {entry['antecedents'].replace('=', ' is ')}, "
            f"{entry['consequents'].replace('congestion_level=', 'congestion is ')} "
            f"in {100 * entry['confidence']:.0f}% of hours — "
            f"{entry['lift']:.2f}x more likely than the base rate."
        )
        top.append(entry)
        logger.info("Rule | %s", entry["plain_language"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    rules.head(25).to_csv(REPORTS / "association_rules.csv", index=False)
    logger.info("Rules saved | path=%s", REPORTS / "association_rules.csv")

    return {"n_rules": int(len(rules)), "top_rules": top}


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 3 unsupervised learning.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    setup_logging(LOG_DIR / "part3.log", debug=args.debug)

    try:
        df = add_proxy_risk_label(load_features())
        results = {
            "clustering": run_clustering(df),
            "association_rules": run_association_rules(df),
        }
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "unsupervised_results.json").write_text(json.dumps(results, indent=2))
        logger.info("Results written | path=%s", REPORTS / "unsupervised_results.json")
    except Exception:
        logger.error("Unsupervised analysis aborted", exc_info=True)
        return 1

    logger.info("Unsupervised analysis completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
