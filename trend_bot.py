#!/usr/bin/env python3
"""
trend_bot.py – pulls two Google-Trends exports every day
  • trend_rising_<date>.csv  (top_rising_terms, 200 rows max)
  • trend_top_<date>.csv     (top_terms,         200 rows max)
"""

import datetime as dt
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()                      # pick up GOOGLE_APPLICATION_CREDENTIALS, etc.
client = bigquery.Client()

# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────
def get_latest_partition():
    sql = """
        SELECT PARSE_DATE('%Y%m%d', MAX(partition_id)) AS latest_dt
        FROM  `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
        WHERE table_name=@tbl AND partition_id != '__NULL__'
    """
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter('tbl', 'STRING', 'top_rising_terms')]
    ))
    return job.result()[0]['latest_dt']          # returns a DATE

LATEST = get_latest_partition()

def build_query(table_name:str) -> str:
    if table_name == "top_rising_terms":
        return f"""
        -- rising terms (may be <200)
        WITH raw AS (
          SELECT term,
                 dma_id,
                 percent_gain,
                 score,
                 week
          FROM   `bigquery-public-data.google_trends.{table_name}`
          WHERE  refresh_date = DATE '{LATEST}'
        ),
        dedup AS (
          SELECT * EXCEPT(rn)
          FROM  (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY term, dma_id ORDER BY week DESC) AS rn
            FROM raw
          )
          WHERE rn = 1
        ),
        stats AS (
          SELECT term,
                 COUNT(DISTINCT dma_id)          AS dma_hits,
                 APPROX_QUANTILES(percent_gain,2)[OFFSET(1)] AS median_gain
          FROM dedup
          GROUP BY term
        )
        SELECT term,
               dma_hits,
               dma_hits/210.0                          AS coverage_ratio,
               median_gain,
               (dma_hits/210.0)*median_gain            AS spread_intensity_score
        FROM stats
        ORDER BY spread_intensity_score DESC
        LIMIT 200
        """
    else:  # top_terms table
        return f"""
        SELECT term,
               score
        FROM   `bigquery-public-data.google_trends.{table_name}`
        WHERE  refresh_date = DATE '{LATEST}'
        ORDER  BY score DESC
        LIMIT  200
        """

def export(csv_prefix:str, sql:str):
    df = client.query(sql).result().to_dataframe()
    fname = f"{csv_prefix}_{LATEST}.csv"
    df.to_csv(fname, index=False)
    print(f"[+]  {fname} written  (rows: {len(df)})")

# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    export("trend_rising", build_query("top_rising_terms"))
    export("trend_top",    build_query("top_terms"))

