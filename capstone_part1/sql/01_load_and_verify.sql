-- Part 1, Task 1.1: load the raw CSV into SQLite and verify.
-- Run from the repository root:
--   sqlite3 data/processed/traffic.db < capstone_part1/sql/01_load_and_verify.sql

DROP TABLE IF EXISTS traffic_raw;
DROP TABLE IF EXISTS traffic;

.mode csv
.import data/raw/Metro_Interstate_Traffic_Volume.csv traffic_raw

-- .import loads every column as TEXT, so build a typed table from it.
CREATE TABLE traffic AS
SELECT
    NULLIF(holiday, 'None')             AS holiday,
    CAST(temp AS REAL)              AS temp,
    CAST(rain_1h AS REAL)           AS rain_1h,
    CAST(snow_1h AS REAL)           AS snow_1h,
    CAST(clouds_all AS INTEGER)     AS clouds_all,
    weather_main,
    weather_description,
    date_time,
    CAST(traffic_volume AS INTEGER) AS traffic_volume
FROM traffic_raw;

-- Verification
.headers on
.mode column
SELECT COUNT(*) AS row_count FROM traffic;
SELECT MIN(date_time) AS first_ts, MAX(date_time) AS last_ts FROM traffic;
SELECT typeof(traffic_volume) AS vol_type, typeof(temp) AS temp_type FROM traffic LIMIT 1;
