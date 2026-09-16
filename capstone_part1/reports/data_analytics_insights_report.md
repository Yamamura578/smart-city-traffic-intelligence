# Part 1 — Data Analytics Insights Report

**Smart City Traffic Intelligence** · Metro Interstate Traffic Volume, westbound I‑94, Minneapolis–St Paul
**Period:** 2 October 2012 – 30 September 2018 · **Records:** 48,204 hourly observations
**Author:** Stefan Mueller

---

## 1. Headline findings

**Traffic demand on this corridor did not grow over six years.** Mean hourly volume moved only
between 3,169.4 (2016) and 3,340.7 (2017) — a spread of about 5%. Apparent year‑on‑year swings
in *total* volume of −44.2% and +243.3% are artefacts of sensor uptime, not demand.

**Hour of day is the only variable that matters.** In 2017, average volume ranges from 382
vehicles/hour at 03:00 to 5,798 at 16:00 — a fifteen‑fold swing. No weather or temperature effect
in this dataset approaches that magnitude.

**Weather is a poor standalone predictor of congestion.** The largest defensible gap between
weather categories is 914.7 vehicles/hour, and the apparent "clear weather reduces congestion"
signal is almost certainly confounding by time of day.

---

## 2. SQL analysis — annual trends and holidays

### 2.1 Annual traffic volume, 2012–2017

| Year | Hours recorded | Total volume | Δ total | Mean hourly | Δ mean |
|------|---------------:|-------------:|--------:|------------:|-------:|
| 2012 | 2,559  |  8,208,767 |   —     | 3,207.8 |   —    |
| 2013 | 8,573  | 28,177,412 | +243.3% | 3,286.8 | +2.5%  |
| 2014 | 4,839  | 15,731,289 | −44.2%  | 3,250.9 | −1.1%  |
| 2015 | 4,373  | 14,181,206 | −9.9%   | 3,242.9 | −0.2%  |
| 2016 | 9,306  | 29,494,821 | +108.0% | 3,169.4 | −2.3%  |
| 2017 | 10,605 | 35,428,156 | +20.1%  | 3,340.7 | +5.4%  |

**Observation 1 — the 2014–2015 collapse is a sensor outage.** Monthly record counts show
recording falling to 136 hours in August 2014, then nothing until June 2015 (186 hours). Nine
months are absent entirely. Totals track hours recorded almost exactly, while mean hourly volume
changes by roughly 1%. Any yearly comparison built on totals is measuring detector availability.

**Observation 2 — underlying demand is flat.** Across all six years the mean sits within a narrow
band. 2017 is the only year where both measures rise together (+20.1% total, +5.4% mean), making
it the sole year with a defensible claim to genuine growth. The 2012 figure covers October–December
only, so the +243.3% jump into 2013 is a partial‑year artefact.

**Implication:** capacity planning for this corridor should be driven by hourly and weekly
patterns, not annual growth projections, because there is no annual growth to project.

### 2.2 Holiday temperature patterns, 2015–2017

| Holiday | Year | Date | Hours | Mean temp (°C) | Δ temp | Mean traffic |
|---------|-----:|------|------:|---------------:|-------:|-------------:|
| Labor Day | 2015 | 2015‑09‑07 | 24 | 22.17 | — | 2,430.3 |
| Labor Day | 2016 | 2016‑09‑05 | 40 | 21.60 | −0.57 | 1,868.3 |
| Labor Day | 2017 | 2017‑09‑04 | 28 | 17.48 | −4.12 | 2,525.7 |
| New Year's Day | 2016 | 2016‑01‑01 | 19 | −6.12 | — | 1,815.2 |
| New Year's Day | 2017 | 2017‑01‑02 | 42 | −0.89 | +5.23 | 2,376.4 |

Labor Day cooled by 4.69 °C across the period; New Year's Day warmed by 5.23 °C. Traffic rose in
2017 on both holidays — alongside cooling on one and warming on the other. **With two to three
observations per holiday, these temperature differences establish no relationship with traffic.**

Three limitations are worth recording. New Year's Day 2015 is absent entirely (coverage gap).
The 2017 New Year observation falls on 2 January, because the federal holiday was observed on the
Monday. And `hours_recorded` exceeds 24 on three rows — the same hour logged under multiple weather
descriptions — so these means are weighted toward hours with more weather rows.

---

## 3. Descriptive statistics and correlation

| Statistic | Value |
|-----------|------:|
| Mean | 3,259.82 |
| Median | 3,380.00 |
| Standard deviation | 1,986.86 |
| Variance | 3,947,615.32 |
| Min / Max / Range | 0 / 7,280 / 7,280 |
| IQR | 1,193 – 4,933 |
| Skewness | −0.089 |
| Coefficient of variation | 0.610 |

**The average describes almost no actual hour.** One standard deviation is 61% of the mean and the
IQR spans 3,740 vehicles. The distribution is near‑symmetric (skew −0.089) but effectively bimodal,
reflecting the split between overnight and daytime hours rather than scatter around a centre. Any
decision based on the mean will be badly wrong at both ends of the day.

**Temperature vs traffic volume:** Pearson r = **+0.132** (n = 48,194, excluding 10 sensor‑failure
rows), r² = **0.018**. Spearman ρ = 0.133 — near‑identical, so the weakness is not a non‑linear
relationship hiding from a linear measure. Temperature explains under 2% of variation in hourly
volume. The p‑value is effectively zero (4.7 × 10⁻¹⁸⁷), but at n ≈ 48,000 that is expected and
carries no practical weight; **effect size, not significance, is the relevant question here.**

**Correlation is not causation.** The positive association almost certainly reflects a common
cause: temperature peaks in the afternoon and so does traffic. Hour of day drives both. The
relationship is further confounded by season, and could in principle run the other way, since dense
traffic generates heat. Establishing causation would require controlling for hour and season.

---

## 4. Probability and congestion

Congestion is defined as `traffic_volume > 5500`; high temperature as `temp > 292 K` (≈ 18.9 °C).

| Quantity | Value |
|----------|------:|
| P(Congestion) | 0.1473 (7,100 hours) |
| P(Clear) | 0.2778 (13,391 hours) |
| P(Congestion ∩ Clear) | 0.0366 (1,763 hours) |
| P(Clear \| Congestion) | 0.2483 |
| P(High Temp \| Congestion) | 0.2630 |
| P(Congestion) × P(Clear) | 0.0409 |
| Odds ratio (clear vs cloudy) | 0.7354 |

**Conditioning on congestion barely changes the weather picture.** P(Clear | Congestion) = 0.2483
against an unconditional P(Clear) of 0.2778 — knowing an hour is congested makes clear weather
slightly *less* likely.

**The events are dependent, but only weakly.** Observed joint probability is 0.0366 against 0.0409
under independence — 89% of expected. A chi‑squared test rejects independence (χ² = 35.9,
p = 2.1 × 10⁻⁹), but with n = 48,204 rejection is near‑guaranteed; the departure itself is small.

**Odds ratio:** congestion occurs in 1,763 of 13,391 clear hours (odds 0.1516) and 2,592 of 15,164
cloudy hours (odds 0.2062), giving OR = **0.735** — congestion is about 1.36× more likely when
cloudy.

Read naively this says good weather reduces congestion, which is implausible as causation. Clear
skies are over‑represented overnight and in winter; cloud cover during daytime hours. Because
congestion is overwhelmingly a daytime phenomenon, **the weather variable is partly acting as a
proxy for hour of day.**

---

## 5. Power BI dashboard

**Preparation.** 48,204 rows × 9 columns loaded into Power Query. Automatic type detection had
silently cast `rain_1h` and `snow_1h` as whole numbers, truncating all decimal values to zero —
corrected to decimal. Derived `Hour`, `Year`, `Temp_Celsius` and a three‑band `Traffic_Category`
(Low < 4,500 = 31,237 rows; Medium ≤ 5,500 = 9,867; High > 5,500 = 7,100).

**Hourly pattern (2017):** peak 5,798 at 16:00, trough 382 at 03:00 — a 15× swing, and the single
strongest pattern in the dataset.

**Weather impact:** highest average is **Clouds at 3,618.4**; lowest is **Squall at 2,061.8**,
a difference of **1,556.7**. Squall, however, has only 4 observations. Excluding categories with
n < 100, the lowest is **Fog at 2,703.7**, giving a robust difference of **914.7** — which is the
figure that should inform any decision.

**Temperature scatter:** a diffuse cloud consistent with r = 0.132, with higher volumes broadly
associated with 15–30 °C. A distinct vertical column of points sits at −273.15 °C: the 10
sensor‑failure rows.

**KPI cards:** 48,204 hours analysed · 3,259.8 mean volume · 8.06 °C mean temperature (8.11 °C
excluding the 0 K rows). Interactive slicers on hour range, weather condition and traffic category.

---

## 6. Data quality issues identified

| Issue | Scale | Handling |
|-------|------:|----------|
| Coverage gap, Sep 2014 – May 2015 | ~9 months | Reported; annual totals treated as unreliable |
| `temp` = 0 K (absolute zero) | 10 rows | Excluded from correlation; median‑imputed in Part 2 |
| `rain_1h` = 9,831.3 mm | 1 row | Flagged as implausible |
| Duplicate timestamps | 7,629 (40,575 distinct of 48,204) | Multiple weather rows per hour; resolved in Part 2 |
| Exact duplicate rows | 17 | Dropped in Part 2 |
| `holiday` populated only at 00:00 | 61 of 48,204 rows | Forward‑filled across each holiday date |
| `holiday` stores `"None"` as a string | 48,143 rows | Converted to true NULL |
| `traffic_volume` = 0 | 2 rows | Flagged as implausible for an interstate |

---

## 7. Conclusions for the mobility team

1. **Plan around the daily cycle, not annual growth.** Demand has been flat for six years; the
   15× intraday swing is where the operational problem lies.
2. **Do not use weather alone to forecast congestion.** Its apparent effect is largely a proxy for
   time of day. Weather should enter models only after conditioning on hour and day type.
3. **Treat the historical record as incomplete.** Nine months are missing, and several fields
   contain physically impossible values. Any model trained on raw data inherits these faults.
4. **Instrument reliability is itself a finding.** A nine‑month outage went undetected in the
   totals, and automatic type inference corrupted two columns. Monitoring data collection deserves
   the same attention as analysing it.
