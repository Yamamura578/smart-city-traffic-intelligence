# Bias and Fairness Report

**Smart City Traffic Intelligence** · Part 3, Task 7
**Author:** Stefan Mueller

---

## 1. The proxy label is the single largest risk in this project

**No accident dataset was supplied with this capstone.** The classification
task uses a proxy label defined entirely from data already in the feature set:

```
high_risk = congestion in {High, Severe} AND (severe weather OR low visibility)
```

This produces 5,514 positive cases (13.59% of 40,575 hours). Three consequences
follow, and all of them matter more than the headline metrics.

**The label is circular.** `weather_main` and `is_low_visibility` are model
*inputs*, and they also appear in the label *definition*. The classifier is
therefore partly re-deriving its own target rather than discovering a pattern.
The random forest's ROC AUC of **0.9988** and recall of **0.9855** are not
evidence of a good accident model; they are evidence that a deterministic rule
is easy to learn. Reporting those numbers without this caveat would be
misleading, and a reader who took them at face value would badly overestimate
what the model can do.

**The label encodes an assumption, not an observation.** It asserts that
congestion plus bad weather equals elevated accident risk. That is plausible,
but it is untested here. Real accident rates are influenced by road geometry,
lighting, enforcement, driver behaviour, vehicle mix and time since last
maintenance — none of which are in this dataset. The proxy may well be
*anti*-correlated with reality in places: drivers frequently slow down in
severe weather, and heavy congestion reduces the speed differentials that
cause the most serious collisions.

**Deploying it would be inappropriate.** Any operational use — resource
allocation, variable speed limits, public risk warnings — would encode an
untested assumption as if it were a measurement. The API returns an explicit
warning to this effect in every risk response, and the README states the proxy
status in its first section.

## 2. Sampling and coverage limitations

**A nine-month hole.** Recording collapses to 136 hours in August 2014 and does
not resume until June 2015 (186 hours). September 2014 to May 2015 is absent
entirely. Every model in Part 3 is trained on a record that silently omits two
winters' worth of one year. Seasonal patterns are therefore learned from five
winters, not six, and any model-derived claim about winter behaviour rests on
less evidence than the row count implies.

**One corridor, one direction.** The data covers westbound I-94 between
Minneapolis and St Paul. Nothing here generalises to other corridors, to the
eastbound direction, or to surface streets. A recommendation engine trained on
this data and deployed city-wide would give confident advice about roads it has
never seen.

**19% of raw rows were discarded.** Part 2 removed 7,612 duplicated timestamps
by keeping the most severe weather observation per hour. This deliberately
biases the retained weather distribution *toward* severe conditions. Since
severe weather also appears in the proxy label, this choice inflates the
positive class. A different de-duplication rule would produce a different
label distribution — the label is partly an artefact of a cleaning decision.

**Category imbalance.** Squall has 4 observations and Smoke 16, out of 40,575.
These were collapsed into `Other` before encoding. Any conclusion about squall
conditions would rest on four hours of data.

## 3. How errors are distributed unevenly

Errors are not spread evenly across conditions, and the uneven parts are
predictable:

**By hour.** SHAP analysis shows the three hour-derived features dominate the
regression: `hour_cos` (mean |SHAP| 1,451), `hour` (427) and `hour_sin` (193)
together outweigh every weather feature combined. The models are, in effect,
hour-of-day lookups with weather as a minor correction. Absolute error is
therefore largest during transition hours — the steep 04:00–07:00 ramp — where
small timing shifts produce large volume differences, and smallest overnight
where volume is both low and stable.

**By rarity.** Conditions with few training examples receive the least reliable
predictions. Snow hours (2,786), fog (617) and the collapsed `Other` category
(20) are exactly the conditions where a safety-oriented system most needs to be
right, and exactly where it has the least evidence. The system is most
confident where the stakes are lowest.

**By season.** The drift monitor found feature drift on `clouds_all`
(PSI 0.333, ALERT), `month` (0.185) and `temp` (0.121) between the pre- and
post-June-2017 periods — while prediction error actually *improved*
(MAE 222.9 → 193.1). Input distributions shift seasonally even when aggregate
accuracy holds, so an aggregate accuracy figure conceals condition-specific
degradation.

**By class.** Even with `class_weight="balanced"`, logistic regression produced
248 false positives against 16 false negatives. The asymmetry is a deliberate
choice — for a risk-flagging system, a missed hazard is worse than a false
alarm — but it means roughly one in five positive flags is wrong, and a
deployment that routed resources on each flag would waste them at that rate.

## 4. Governance: what should exist before this is trusted

1. **Real accident data, or no risk model at all.** The proxy demonstrates a
   workflow. It does not justify a deployment. The first governance gate is
   obtaining linked incident records and re-validating against them.
2. **Named human accountability.** A specific owner should sign off on the
   label definition, the cleaning rules and each retraining, with those
   decisions recorded. Several choices in this project — the severity ranking
   for de-duplication, the quartile boundaries, the PSI thresholds — are
   defensible but arbitrary, and were made by one person without review.
3. **Scheduled revalidation, not just drift alerts.** The monitor compares
   distributions; it cannot detect that the label itself has stopped meaning
   what it meant. Periodic human review of sampled predictions is required.
4. **Published limitations alongside any public output.** If timing advice
   reaches the public, the coverage gap, the single-corridor scope and the
   error profile should be published with it, not buried in an appendix.
5. **An appeal and override path.** Any system influencing public-facing
   decisions needs a documented route to challenge and override its output.
6. **Equity review before deployment.** Travel-timing advice is not neutral in
   its effects: people with rigid shift patterns cannot act on a
   recommendation to travel at 10:00, so benefits accrue disproportionately to
   those with flexible schedules. That distributional question should be
   examined before rollout, not after.

## 5. Sustainability and resource trade-offs

Training costs for this project were small and are worth stating precisely,
because they support the argument rather than decorate it:

| Model | Training time | Test MAE | R² |
|-------|--------------:|---------:|----:|
| Linear regression | 0.03 s | 806.2 | 0.729 |
| Gradient boosting | 14.6 s | 230.3 | 0.965 |
| LSTM (59,201 parameters) | 32.5 s | 196.4 | 0.979 |

The LSTM is the most accurate model and cost roughly **1,000× the compute of
linear regression** for a **14.8% MAE improvement** over gradient boosting.
Gradient boosting reached 96.5% of the variance explained in under half the
time and produces a model that runs on CPU with no framework dependency.

For this application the honest conclusion is that **the LSTM is not worth its
cost**. Its 34-vehicle/hour advantage is well inside the noise of an operational
decision about when to travel; nobody changes their departure time over 34
vehicles. Gradient boosting is the right production choice, and the LSTM is
best understood as a demonstration that sequence structure adds little here
beyond what hour-of-day features already capture.

Two further points. **Inference cost, not training cost, dominates in
production** — a model serving predictions continuously for years will consume
far more energy in aggregate than a one-off training run, which argues for the
smaller model independently of training economics. And **retraining frequency
should be evidence-driven**: the drift monitor showed error *improving* over
time, so a fixed monthly retraining schedule would burn compute to fix a
problem that was not occurring.
