#!/usr/bin/env python3
"""
trend_bot.py – output two CSVs:

  • trend_rising_<date>.csv  → term, dma_hits, coverage_ratio, median_gain, spread_intensity_score
  • trend_top_<date>.csv     → term, dma_hits, avg_rank,      total_score,  coverage_ratio

Window: N latest daily partitions (default 7) from Google-Trends BigQuery.
"""

import datetime as dt
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
from caption_agent import main as run_captions

# ─── CONFIG ────────────────────────────────────────────────────────────────
WINDOW_DAYS    = 7          # how many latest partitions to scan
MIN_SCORE_GAIN = 0          # rising-table filter; leave 0 to disable
CSV_PREFIX     = "trend"    # output file basename
# ───────────────────────────────────────────────────────────────────────────

load_dotenv()                       # uses GOOGLE_APPLICATION_CREDENTIALS
client = bigquery.Client()

# ─── helper: most-recent partitions ───────────────────────────────────────
def recent_partitions(n: int) -> list[str]:
    sql = """
    SELECT PARSE_DATE('%Y%m%d', partition_id) AS dt
    FROM  `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
    WHERE table_name='top_rising_terms' AND partition_id<>'__NULL__'
    ORDER BY dt DESC
    LIMIT @n
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("n", "INT64", n)]
        ),
    )
    return [row["dt"].strftime("%Y-%m-%d") for row in job]

PARTS      = recent_partitions(WINDOW_DAYS)
PART_LIST  = ",".join(f"'{d}'" for d in PARTS)
LATEST_DAY = PARTS[0]                 # used in filenames

# ─── SQL builders ─────────────────────────────────────────────────────────
def sql_rising() -> str:
    """Top-200 *rising* terms with full DMA coverage statistics."""
    return f"""
    WITH raw AS (
      SELECT term,
             dma_id,
             percent_gain  AS metric
      FROM   `bigquery-public-data.google_trends.top_rising_terms`
      WHERE  refresh_date IN ({PART_LIST})
        AND  percent_gain >= {MIN_SCORE_GAIN}
    ),
    stats AS (
      SELECT
        term,
        COUNT(DISTINCT dma_id)                 AS dma_hits,
        COUNT(DISTINCT dma_id) / 210.0         AS coverage_ratio,
        APPROX_QUANTILES(metric, 2)[OFFSET(1)] AS median_gain,
        (COUNT(DISTINCT dma_id) / 210.0)
          * APPROX_QUANTILES(metric, 2)[OFFSET(1)] AS spread_intensity_score
      FROM raw
      GROUP BY term
    )
    SELECT *
    FROM   stats
    ORDER  BY spread_intensity_score DESC
    LIMIT  200
    """

def sql_top() -> str:
    """Top-200 *top_terms* with windowed strongest score & coverage."""
    return f"""
    WITH raw AS (
      SELECT term,
             dma_id,
             score AS metric,
             rank
      FROM   `bigquery-public-data.google_trends.top_terms`
      WHERE  refresh_date IN ({PART_LIST})
    ),
    best_dmas AS (                   -- keep strongest score per term-DMA across window
      SELECT * EXCEPT(rk) FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY term, dma_id ORDER BY metric DESC) rk
        FROM raw)
      WHERE rk = 1
    ),
    stats AS (
      SELECT
        term,
        COUNT(DISTINCT dma_id)       AS dma_hits,
        AVG(rank)                    AS avg_rank,
        MAX(metric)                  AS total_score,
        COUNT(DISTINCT dma_id)/210.0 AS coverage_ratio
      FROM best_dmas
      GROUP BY term
    )
    SELECT *
    FROM   stats
    ORDER  BY total_score DESC
    LIMIT  200
    """

# ─── exporter ─────────────────────────────────────────────────────────────
def export(prefix: str, sql: str) -> None:
    df = client.query(sql).result().to_dataframe()
    fname = f"{prefix}_{LATEST_DAY}.csv"
    df.to_csv(fname, index=False)
    print(f"[✓] {fname} written  (rows: {len(df)})")

# ─── main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    export(f"{CSV_PREFIX}_rising", sql_rising())
    export(f"{CSV_PREFIX}_top",    sql_top())
run_captions()