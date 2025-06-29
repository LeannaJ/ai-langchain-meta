"""
trend_bot.py
Pulls the latest Spread × Intensity Top-25 list from Google Trends
and writes it to a CSV file.
"""

import datetime as dt
from google.cloud import bigquery
from dotenv import load_dotenv
import pandas as pd

load_dotenv()          # reads .env for credentials
client = bigquery.Client()

SQL = """
-- get latest partition date
DECLARE latest DATE DEFAULT (
  SELECT PARSE_DATE('%Y%m%d', MAX(partition_id))
  FROM   `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
  WHERE  table_name='top_rising_terms' AND partition_id<>'__NULL__'
);

WITH raw AS (
  SELECT term, dma_id, percent_gain, score, week
  FROM   `bigquery-public-data.google_trends.top_rising_terms`
  WHERE  refresh_date = latest
),
dedup AS (
  SELECT *
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY term, dma_id ORDER BY week DESC) AS rn
    FROM raw
  )
  WHERE rn = 1
),
stats AS (
  SELECT
    term,
    COUNT(DISTINCT dma_id) AS dma_hits,
    APPROX_QUANTILES(percent_gain,2)[OFFSET(1)] AS median_gain
  FROM dedup
  GROUP BY term
)
SELECT
  term,
  dma_hits,
  dma_hits / 210.0 AS coverage_ratio,
  median_gain,
  (dma_hits / 210.0) * median_gain AS spread_intensity_score
FROM stats
ORDER BY spread_intensity_score DESC
LIMIT 200;
"""

results = client.query(SQL).result().to_dataframe()
today = dt.date.today()

# Write to CSV
csv_file = f"trend_output_{today}.csv"
results.to_csv(csv_file, index=False)
print(f"Trend CSV generated: {csv_file}")
