-- Part 1, Task 1.3: temperature patterns on New Year's Day and Labor Day, 2015-2017.
--   sqlite3 data/processed/traffic.db < capstone_part1/sql/03_holiday_temperatures.sql
--
-- NOTE: `holiday` is populated only on the 00:00 row of each holiday, so a direct
-- filter on the column returns one row per year. We first collect the tagged
-- holiday DATES, then join back to pull all hours falling on those dates.

.headers on
.mode column

-- Confirm the sparsity of the holiday column before relying on it.
SELECT COUNT(*) AS tagged_rows FROM traffic WHERE holiday IS NOT NULL;

WITH holiday_dates AS (
    SELECT DISTINCT
        holiday,
        DATE(date_time)           AS holiday_date,
        strftime('%Y', date_time) AS year
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND year BETWEEN '2015' AND '2017'
),
holiday_hours AS (
    SELECT
        h.holiday,
        h.year,
        h.holiday_date,
        t.temp,
        t.temp - 273.15 AS temp_c,
        t.traffic_volume
    FROM holiday_dates h
    JOIN traffic t ON DATE(t.date_time) = h.holiday_date
)
SELECT
    holiday,
    year,
    holiday_date,
    COUNT(*)                      AS hours_recorded,
    ROUND(AVG(temp_c), 2)         AS mean_temp_c,
    ROUND(MIN(temp_c), 2)         AS min_temp_c,
    ROUND(MAX(temp_c), 2)         AS max_temp_c,
    ROUND(AVG(traffic_volume), 1) AS mean_traffic
FROM holiday_hours
GROUP BY holiday, year
ORDER BY holiday, year;

-- Year-on-year temperature change on each holiday.
WITH holiday_dates AS (
    SELECT DISTINCT
        holiday,
        DATE(date_time)           AS holiday_date,
        strftime('%Y', date_time) AS year
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND year BETWEEN '2015' AND '2017'
),
yearly AS (
    SELECT
        h.holiday,
        h.year,
        AVG(t.temp - 273.15)   AS mean_temp_c,
        AVG(t.traffic_volume)  AS mean_traffic
    FROM holiday_dates h
    JOIN traffic t ON DATE(t.date_time) = h.holiday_date
    GROUP BY h.holiday, h.year
)
SELECT
    holiday,
    year,
    ROUND(mean_temp_c, 2) AS mean_temp_c,
    ROUND(mean_temp_c - LAG(mean_temp_c) OVER (PARTITION BY holiday ORDER BY year), 2)
        AS temp_change_c,
    ROUND(mean_traffic, 1) AS mean_traffic,
    ROUND(mean_traffic - LAG(mean_traffic) OVER (PARTITION BY holiday ORDER BY year), 1)
        AS traffic_change
FROM yearly
ORDER BY holiday, year;

-- Context: the same calendar dates in non-holiday years are not comparable,
-- so compare each holiday against the annual mean for its own year instead.
SELECT
    strftime('%Y', date_time)     AS year,
    ROUND(AVG(temp - 273.15), 2)  AS annual_mean_temp_c,
    ROUND(AVG(traffic_volume), 1) AS annual_mean_traffic
FROM traffic
WHERE year BETWEEN '2015' AND '2017'
  AND temp > 0
GROUP BY year
ORDER BY year;
