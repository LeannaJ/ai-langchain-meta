"""
trend_bot.py

Pulls the latest Spread × Intensity Top-25 list from Google Trends,
keeps the **top-3 DMAs per term**, ranks by spread-intensity,
and writes the 200 best terms to a dated CSV.
"""

import datetime as dt
from google.cloud import bigquery
from dotenv import load_dotenv
import pandas as pd

load_dotenv()                       # reads .env for credentials
client = bigquery.Client()

SQL = """
-- ░░ Get latest partition date ░░
DECLARE latest DATE DEFAULT (
  SELECT PARSE_DATE('%Y%m%d', MAX(partition_id))
  FROM  `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
  WHERE  table_name='top_rising_terms' AND partition_id!='NULL'
);

-- ░░ Pull *all* rows for that date ░░
WITH raw AS (
  SELECT term, dma_id, percent_gain, score, week
  FROM   `bigquery-public-data.google_trends.top_rising_terms`
  WHERE  refresh_date = latest
  AND    percent_gain >= 0          -- no threshold
  AND    score         >= 0
),

-- ░░ Keep the top-3 DMAs *per term* by percent_gain ░░
dedup AS (
  SELECT *
  FROM (
    SELECT
      term, dma_id, percent_gain, score, week,
      RANK() OVER (PARTITION BY term ORDER BY percent_gain DESC) AS rk
    FROM raw
  )
  WHERE rk <= 3                     -- ← tweak to 1,2,3… as you like
),

-- ░░ Aggregate per term ░░
stats AS (
  SELECT
    term,
    COUNT(DISTINCT dma_id) AS dma_hits,
    APPROX_QUANTILES(percent_gain,2)[OFFSET(1)] AS median_gain
  FROM dedup
  GROUP BY term
)

-- ░░ Final score & pick the best 200 ░░
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

# ── Run query ──────────────────────────────────────────────────────────────────
results = client.query(SQL).result().to_dataframe()

# ── Save dated CSV ─────────────────────────────────────────────────────────────
today = dt.date.today().isoformat()
csv_file = f"trend_output_{today}.csv"
results.to_csv(csv_file, index=False)
print(f"Trend CSV generated: {csv_file}")
