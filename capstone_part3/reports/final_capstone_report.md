# Part 3 — Final Capstone Report

**Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility**
Metro Interstate Traffic Volume · westbound I-94, Minneapolis–St Paul
**Author:** Stefan Mueller

---

## Summary

Three parts, one pipeline. Part 1 established that hour of day dominates every
other signal on this corridor and that demand has been flat for six years.
Part 2 turned that into a reproducible cleaning and feature pipeline producing
40,575 hourly rows and 42 columns. Part 3 builds models on that output.

The central result is consistent across every model: **time features carry
almost all the predictive signal, and weather adds very little.** The best
demand model reaches R² = 0.979; SHAP shows that three hour-derived features
outweigh all weather features combined. The classification task rests on a
proxy label and its apparent near-perfection is an artefact of that label's
construction, discussed below and at length in the bias report.

## Task 1 — Supervised models

Shared 27-feature matrix: time features, cyclical encodings of hour and day of
week, weather one-hot encodings and derived indicators, and a holiday flag.
80/20 split, `random_state=42`, stratified for classification.

**Classification — proxy accident risk** (positive rate 13.59%)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Fit time |
|-------|---------:|----------:|-------:|---:|--------:|---------:|
| Logistic regression | 0.9675 | 0.8142 | 0.9855 | 0.8917 | 0.9953 | 0.09 s |
| **Random forest** | **0.9872** | **0.9251** | **0.9855** | **0.9543** | **0.9988** | 2.17 s |

Majority-class baseline accuracy is 0.8641, so both models beat it — but this
is the wrong way to read these numbers. **The proxy label is a deterministic
function of congestion and weather, and weather is a model input.** The
classifier is substantially re-deriving its own target. These metrics measure
the learnability of a rule, not the predictability of accidents. Both models
were configured with `class_weight="balanced"`, which is why recall (0.9855)
exceeds precision: for risk flagging, a missed hazard costs more than a false
alarm. Logistic regression produced 248 false positives against 16 false
negatives; the forest cut false positives to 88 at the same recall.

**Regression — hourly traffic volume**

| Model | MAE | RMSE | R² | Fit time |
|-------|----:|-----:|---:|---------:|
| Linear regression | 806.2 | 1029.8 | 0.7293 | 0.03 s |
| **Gradient boosting** | **230.3** | **370.8** | **0.9649** | 14.6 s |

Mean-prediction baseline MAE is 1,738.0, so gradient boosting is **86.7%
better than baseline**. The gap between the two models is the finding: linear
regression cannot represent the twin-peaked weekday profile, while a tree
ensemble captures it easily. The relationship between hour and volume is
strongly non-linear, which is why the cyclical encodings matter.

## Task 2 — Unsupervised learning

**K-means.** Features: hour, weather severity (ordinal), volume, temperature,
weekend flag, all standardised. Silhouette score selected **k = 8**
(score 0.3268 — moderate separation, as expected for continuous traffic data
with no natural cluster boundaries). The clusters are interpretable:

| Cluster | n | Mean volume | Character |
|--------:|--:|------------:|-----------|
| 3 | 6,986 | 5,254 | High-volume weekday midday, mild weather |
| 2 | 6,794 | 760 | Low-volume weekday overnight |
| 0 | 5,685 | 3,632 | Weekend afternoon/evening |
| 1 | 5,444 | 5,172 | High-volume weekday midday |
| 6 | 5,092 | 4,263 | High-volume weekday midday, **adverse weather** |
| 7 | 4,665 | 2,438 | Weekday late evening |
| 5 | 3,584 | 1,085 | Weekend overnight |
| 4 | 2,325 | 2,530 | Weekend midday, adverse weather |

The structure the algorithm found is time-of-day and day-type first, weather
second — the same ordering every other analysis in this project produced.
Clusters 1 and 3 are near-duplicates, suggesting k = 8 slightly over-segments.

**Association rules.** Time bands, day type, weather band and temperature band
were discretised into a 16-item transaction matrix; apriori at min_support 0.02
found 383 frequent itemsets, yielding **142 rules** predicting congestion at
lift ≥ 1.2. The strongest:

- *Weekend + freezing + night → congestion Low*, confidence 95%, **lift 3.79**
- *Weekend + night + adverse weather → Low*, confidence 91%, lift 3.63
- *Weekday + cool + morning peak → **Severe***, confidence 90%, **lift 3.62**

In plain language: on a weekday morning peak in cool weather, congestion is
severe in nine hours out of ten — 3.6 times more often than the base rate. The
rules confirm the models rather than adding to them, which is itself
informative: the signal is simple enough that a rule miner and a gradient
boosting ensemble find the same thing.

## Task 3 — Deep learning with explainability

**Model: LSTM (PyTorch).** Two layers, 64 hidden units, 24-hour input window,
59,201 parameters. Windows were emitted only where the 24 preceding hours are
genuinely consecutive — 11,680 candidate windows were discarded because the
coverage gap and de-duplicated hours would otherwise have produced fabricated
sequences. 28,871 windows remained, split **chronologically** 80/20 (a random
split would leak the future into training).

| | MAE | RMSE | R² |
|---|----:|-----:|---:|
| LSTM | **196.4** | 290.0 | **0.9786** |
| Gradient boosting (Task 1) | 230.3 | 370.8 | 0.9649 |

The LSTM is the best model in the project, 14.8% better than gradient boosting
on MAE. Whether that justifies its cost is addressed in the bias report's
sustainability section — the short answer is no.

**Framework note.** TensorFlow segfaults on the target platform
(Python 3.13 / macOS arm64); PyTorch 2.14 was substituted.

**Explainability: SHAP on a gradient-boosting surrogate.** The brief permits
applying SHAP to a comparable model when the primary model is a sequence
model, and that route was taken deliberately. SHAP's Deep and Gradient
explainers assume approximate feature independence, which is violated by
construction across a 24-step lag window of a strongly autocorrelated series;
attributions over the raw sequence would be confidently wrong. The surrogate
(MAE 239.8, R² 0.9586) performs close enough to the LSTM to be a fair proxy.

Mean |SHAP| values:

| Feature | Mean \|SHAP\| |
|---------|-------------:|
| hour_cos | 1451.4 |
| hour | 426.6 |
| hour_sin | 193.1 |
| is_weekend | 182.5 |
| dow_sin | 131.8 |
| day_of_week | 99.4 |

**The three hour features total 2,071 against no weather feature reaching the
top six.** This is the quantitative confirmation of what Part 1 suspected from
a correlation of r = 0.13: weather is nearly irrelevant to demand on this
corridor once time is known. It also validates the cyclical encoding decision —
`hour_cos` outranks raw `hour` by 3.4×.

## Task 4 — Advanced technique: MLflow

**Why.** Of the four options, MLflow connects directly to the MLOps
requirements in Task 6, so one implementation serves two tasks. The
alternatives were considered and rejected on fit: quantisation optimises a
model that is already 59k parameters and runs in milliseconds; a GAN would
generate synthetic traffic data for a dataset that has 40,575 real rows; and
anomaly detection overlaps with the drift monitoring already required.

**How.** Three experiments — `traffic-accident-risk-classification`,
`traffic-volume-regression`, `traffic-deep-learning` — with parameters, metrics
and serialised models logged per run. The tracking store is SQLite
(`capstone_part3/mlflow.db`) rather than the legacy filesystem backend, which
MLflow 3.x rejects. Models are logged with cloudpickle serialisation because
the 3.x default (skops) refuses to round-trip scikit-learn tree objects.

**Value.** The five runs are directly comparable after the fact, including fit
times, without re-running anything. This is what made the sustainability
analysis possible — the compute-versus-accuracy trade-off is read straight off
the logged `fit_seconds` and MAE.

**Limitations.** Local SQLite has no multi-user access control or remote
artifact store. Runs record what was executed but not the data version behind
them, so a run is only reproducible while the feature table is unchanged —
proper practice would hash the input data and log the digest.

## Task 5 — Recommendation system

Single corridor, so the system recommends **when** to travel, not which route.
It builds an hourly profile conditioned on day type and optionally weather,
ranks hours within the user's acceptable window, identifies the longest
contiguous run of quiet hours, and generates plain language:

> "For a weekday journey in rain conditions, consider travelling between 19:00
> and 21:00, when historical traffic volumes average 2,816 vehicles/hour —
> about 21% below the daily average. Avoid 16:00, the busiest hour in this
> window at 6,275 vehicles/hour."

When a weather filter leaves fewer than 200 matching hours the system logs a
WARNING and falls back to the unconditioned profile rather than returning a
confident recommendation from a thin sample. Available as a CLI and as
`GET /recommend`.

## Task 6 — MLOps and deployment simulation

**6.1 Versioning.** Five models saved to `capstone_part3/models/` with
performance recorded in `reports/model_results.json` and in MLflow. Versions:
v1 linear/logistic baselines, v2 ensembles, v3 LSTM.

**6.2 Tracking.** As Task 4.

**6.3 Deployment.** FastAPI (`capstone_part3/api/main.py`) with four endpoints:
`POST /predict/volume`, `POST /predict/risk`, `GET /recommend`, `GET /health`.
Pydantic validates inputs; a 27-column feature vector is assembled from
human-meaningful inputs (timestamp, weather, temperature) so callers need no
knowledge of the encoding. Every risk response carries an explicit proxy
warning. Verified working: a June weekday 08:00 in rain returns 5,612
vehicles/hour; a January 17:00 in snow returns risk probability 0.941.

This is a simulation, not a service. It has no authentication, rate limiting,
input provenance checking, model-registry lookup or request logging to a
durable store.

**6.4 Monitoring.** Chronological split at 2017-06-01 (reference 28,925 rows,
current 11,650). Population Stability Index per feature, plus prediction error
ratio.

**6.5 Alerting.** PASS / WARN / ALERT banner. Current status: **ALERT**.

| Check | Status | Detail |
|-------|--------|--------|
| Feature drift | **ALERT** | 3 of 27 features drifted: `clouds_all` PSI 0.333, `month` 0.185, `temp` 0.121 |
| Prediction error | PASS | MAE 222.9 → 193.1, ratio 0.87 |

This is the most instructive result in Task 6. **Input distributions drifted
significantly while accuracy improved.** The drift is seasonal composition, not
degradation — the two windows contain different mixes of months. A naive
monitoring rule that retrained on any feature-drift alert would have burned
compute to fix a problem that was not occurring. Drift alerts are a prompt to
investigate, not a trigger to retrain.

## Task 7 — Responsible and sustainable AI

Covered in full in `bias_and_fairness_report.md`. The three points that matter
most: the proxy label is circular and its metrics must not be read as accident
prediction; a nine-month coverage gap and single-corridor scope limit
generalisation; and the LSTM costs ~1,000× the compute of linear regression for
a 34-vehicle/hour improvement that no traveller would notice, making gradient
boosting the defensible production choice.

## Integration across the three parts

The parts form one chain rather than three exercises. Part 1's SQL established
the coverage gap, which determined Part 2's decision not to impute it, which
determined Part 3's sequence-construction rule that discards windows spanning
it. Part 1's correlation of r = 0.13 between temperature and volume anticipated
Part 3's SHAP finding that weather features carry almost no weight. Part 2's
quartile target feeds Part 3's proxy label directly. Part 2's `advise` CLI
command is the prototype of Part 3's recommendation engine. Each part reads the
previous part's output file rather than re-deriving it.
