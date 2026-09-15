-- Part 1, Task 1.2: annual traffic trends, 2012-2017.
--   sqlite3 data/processed/traffic.db < capstone_part1/sql/02_yearly_trends.sql

.headers on
.mode column

-- Total and mean volume per year, with the record count that produced them.
SELECT
    strftime('%Y', date_time)      AS year,
    COUNT(*)                       AS hours_recorded,
    SUM(traffic_volume)            AS total_volume,
    ROUND(AVG(traffic_volume), 1)  AS mean_hourly_volume
FROM traffic
WHERE year BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- Year-on-year change, on both measures.
WITH yearly AS (
    SELECT
        strftime('%Y', date_time)     AS year,
        COUNT(*)                      AS hours_recorded,
        SUM(traffic_volume)           AS total_volume,
        AVG(traffic_volume)           AS mean_volume
    FROM traffic
    WHERE year BETWEEN '2012' AND '2017'
    GROUP BY year
)
SELECT
    year,
    hours_recorded,
    total_volume,
    total_volume - LAG(total_volume) OVER (ORDER BY year) AS total_change,
    ROUND(100.0 * (total_volume - LAG(total_volume) OVER (ORDER BY year))
          / LAG(total_volume) OVER (ORDER BY year), 1)    AS total_pct_change,
    ROUND(mean_volume, 1)                                 AS mean_hourly_volume,
    ROUND(100.0 * (mean_volume - LAG(mean_volume) OVER (ORDER BY year))
          / LAG(mean_volume) OVER (ORDER BY year), 1)     AS mean_pct_change
FROM yearly
ORDER BY year;

-- Monthly record counts, to expose the coverage gap behind the totals.
SELECT
    strftime('%Y-%m', date_time) AS month,
    COUNT(*)                     AS hours_recorded
FROM traffic
WHERE month BETWEEN '2014-01' AND '2015-12'
GROUP BY month
ORDER BY month;
