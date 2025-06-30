#!/usr/bin/env python3
"""
trend_bot.py -- pulls  ❑ Top-200 US *rising* terms
                  and  ❑ Top-200 US *top* terms
over the latest N partitions (default 7 days) of Google-Trends BigQuery
and writes two CSVs with the EXACT columns below.

Outputs
-------
trend_rising_YYYY-MM-DD.csv   ( 200 rows, 5 columns )
trend_top_YYYY-MM-DD.csv      ( 200 rows, 4 columns )
"""
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import datetime as dt
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
# ━━━━━━━━━━━━━━━━━━━  CONFIG  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WINDOW_DAYS   = 7          # look back this many partitions (days)
MIN_SCORE_GAIN = 0         # set >0 to filter low-quality rows in rising table
CSV_PREFIX     = "trend"   # base file name
# ━━━━━━━━━━━━━━━━━━━  INIT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
load_dotenv()                       # picks up GOOGLE_APPLICATION_CREDENTIALS
client = bigquery.Client()
# ━━━━━━━━━━━━━━━━━━━  HELPERS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_recent_partitions(n: int) -> list[str]:
    """Return the most-recent N partition YYYY-MM-DD strings in *top_rising_terms*."""
    sql = """
    SELECT PARSE_DATE('%Y%m%d', partition_id) AS dt
    FROM  `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
    WHERE table_name='top_rising_terms'
      AND partition_id<>'__NULL__'
    ORDER BY dt DESC
    LIMIT @n
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("n", "INT64", n)]
        ),
    )
    return [row["dt"].strftime("%Y-%m-%d") for row in job.result()]

PARTITIONS = get_recent_partitions(WINDOW_DAYS)
LATEST     = PARTITIONS[0]            # newest date, used in filenames
PARTITION_LIST = ",".join(f"'{d}'" for d in PARTITIONS)
# ━━━━━━━━━━━━━━━━━━━  SQL BUILDERS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def sql_rising() -> str:
    """Top-200 rising terms across WINDOW with requested metrics."""
    return f"""
    WITH raw AS (
      SELECT term,
             dma_id,
             percent_gain AS metric,
             refresh_date
      FROM `bigquery-public-data.google_trends.top_rising_terms`
      WHERE refresh_date IN ({PARTITION_LIST})
        AND percent_gain >= {MIN_SCORE_GAIN}
    ),
    dedup AS (            -- keep strongest dma per term/date
      SELECT * EXCEPT(rk) FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY term, refresh_date
                                     ORDER BY metric DESC) rk
        FROM raw)
      WHERE rk = 1
    ),
    stats AS (            -- aggregate across WINDOW
      SELECT
        term,
        COUNT(DISTINCT dma_id)                  AS dma_hits,
        COUNT(DISTINCT dma_id)/210.0            AS coverage_ratio,
        APPROX_QUANTILES(metric, 2)[OFFSET(1)]  AS median_gain,
        (COUNT(DISTINCT dma_id)/210.0) *
        APPROX_QUANTILES(metric, 2)[OFFSET(1)]  AS spread_intensity_score
      FROM dedup
      GROUP BY term
    )
    SELECT *
    FROM stats
    ORDER BY spread_intensity_score DESC
    LIMIT 200
    """

def sql_top() -> str:
    """Top-200 *top_terms* windowed metrics -- strongest score, plus avg-rank."""
    return f"""
    WITH raw AS (
      SELECT term,
             dma_id,
             score            AS metric,
             rank,
             refresh_date
      FROM `bigquery-public-data.google_trends.top_terms`
      WHERE refresh_date IN ({PARTITION_LIST})
    ),
    best_per_dma AS (      -- keep strongest score per term/dma/date
      SELECT * EXCEPT(rk) FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY term, dma_id, refresh_date
                                     ORDER BY metric DESC) rk
        FROM raw)
      WHERE rk = 1
    ),
    best_across_win AS (   -- take strongest score across WINDOW for each term/dma
      SELECT * EXCEPT(rk) FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY term, dma_id
                                     ORDER BY metric DESC) rk
        FROM best_per_dma)
      WHERE rk = 1
    ),
    stats AS (
      SELECT
        term,
        COUNT(DISTINCT dma_id)        AS dma_hits,
        AVG(rank)                     AS avg_rank,
        MAX(metric)                   AS total_score
      FROM best_across_win
      GROUP BY term
    )
    SELECT *
    FROM stats
    ORDER BY total_score DESC
    LIMIT 200
    """
# ━━━━━━━━━━━━━━━━━━━  EXPORT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def export(csv_prefix: str, sql: str) -> None:
    df = client.query(sql).result().to_dataframe()
    fname = f"{csv_prefix}_{LATEST}.csv"
    df.to_csv(fname, index=False)
    print(f"[✓] {fname} written  (rows: {len(df)})")
# ━━━━━━━━━━━━━━━━━━━  MAIN  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    export(f"{CSV_PREFIX}_rising", sql_rising())
    export(f"{CSV_PREFIX}_top",    sql_top())
