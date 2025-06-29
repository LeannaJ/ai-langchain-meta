"""
trend_bot.py (v2)  –  pulls the **Top-200 U.S. rising Google-Trends topics**

How it works
------------
1. Looks at the most-recent *WINDOW* BigQuery partitions (days).
2. Optionally filters by percent_gain / score thresholds.
3. Deduplicates per -- term × DMA (keeps latest week).
4. Ranks by “spread × intensity” and returns the 200 strongest terms.
5. Saves a CSV named `trend_output_<YYYY-MM-DD>.csv`.
"""

import datetime as dt
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
WINDOW_DAYS      = 7     # how many latest partitions (days) to scan
MIN_PERCENT_GAIN = 300   # set to 0 to disable threshold
MIN_SCORE        = 20    # set to 0 to disable threshold
CSV_PREFIX       = "trend_output_"  # file name will be <prefix><date>.csv
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()                       # pulls GOOGLE_APPLICATION_CREDENTIALS, etc.
client = bigquery.Client()          # uses those creds

SQL = f"""
/* 1️⃣  get the latest {WINDOW_DAYS} partitions (one per refresh_date) */
WITH latest_partitions AS (
  SELECT DISTINCT
         PARSE_DATE('%Y%m%d', partition_id) AS dt
  FROM   `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
  WHERE  table_name = 'top_rising_terms'
         AND partition_id <> '__NULL__'
  ORDER  BY dt DESC
  LIMIT  {WINDOW_DAYS}
),

/* 2️⃣  pull rows for those dates */
raw AS (
  SELECT term,
         dma_id,
         percent_gain,
         score,
         week
  FROM   `bigquery-public-data.google_trends.top_rising_terms`
  WHERE  refresh_date IN (SELECT dt FROM latest_partitions)
         AND percent_gain >= {MIN_PERCENT_GAIN}
         AND score        >= {MIN_SCORE}
),

/* 3️⃣  dedupe: keep newest week per term×DMA */
dedup AS (
  SELECT *
  FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY term, dma_id
                                  ORDER BY week DESC) AS rn
        FROM   raw)
  WHERE rn = 1
),

/* 4️⃣  aggregate stats per term */
stats AS (
  SELECT
        term,
        COUNT(DISTINCT dma_id)                          AS dma_hits,
        APPROX_QUANTILES(percent_gain, 2)[OFFSET(1)]    AS median_gain
  FROM   dedup
  GROUP  BY term
)

/* 5️⃣  final ranking */
SELECT
    term,
    dma_hits,
    dma_hits / 210.0                        AS coverage_ratio,
    median_gain,
    (dma_hits / 210.0) * median_gain        AS spread_intensity_score
FROM stats
ORDER BY spread_intensity_score DESC
LIMIT 200;
"""

# ──────────────────────────────────────────────────────────────────────────────
# Execute and save
results = client.query(SQL).result().to_dataframe()
today   = dt.date.today().isoformat()
csv_file = f"{CSV_PREFIX}{today}.csv"
results.to_csv(csv_file, index=False)

print(f"[✓] Trend CSV generated → {csv_file}  (rows: {len(results)})")
