#!/usr/bin/env python3
"""
trend_bot.py – pulls two Google-Trends exports every day

    • trend_rising_YYYY-MM-DD.csv   (top_rising_terms, ≤200 rows)
    • trend_top_YYYY-MM-DD.csv      (top_terms,        200 rows max)

How it works
============

1.   Looks at the most-recent *known* BigQuery partitions (by day).
2.   Optionally filters by percent_gain / score thresholds.
3.   Deduplicates per term ➜ DMA (keeps latest week).
4.   Ranks by “spread × intensity” and returns the 200 strongest terms.
5.   Saves  CSV  named  `<csv_prefix>_YYYY-MM-DD.csv`
"""

import datetime as dt
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()                     # pulls GOOGLE_APPLICATION_CREDENTIALS, etc.
client = bigquery.Client()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_DAYS    = 7               # how many latest partitions (days) to scan
MIN_PERCENT_GAIN = 0             # set >0 to enable threshold
MIN_SCORE        = 0             # set >0 to enable threshold
CSV_PREFIX       = "trend_output"  # file name will be <prefix><date>.csv

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_latest_partition() -> str:
    """
    Return the most recent partition date (YYYY-MM-DD) that exists in the
    Google-Trends public dataset.
    """
    sql = """
        SELECT PARSE_DATE('%Y%m%d', MAX(partition_id)) AS latest_dt
        FROM  `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
        WHERE table_name=@tbl
          AND partition_id!='__NULL__'
    """
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('tbl', 'STRING', 'top_rising_terms')
        ]
    )
    rows = client.query(sql, job_config=job_cfg).result()
    first_row = next(rows, None)         # RowIterator → first row
    if first_row is None or first_row['latest_dt'] is None:
        raise RuntimeError("No valid partition found in INFORMATION_SCHEMA")
    return first_row['latest_dt']        # returns a DATE (yyyy-mm-dd)


LATEST = get_latest_partition()          # used in all queries


def build_query(table_name: str) -> str:
    """
    Return a fully-formed SQL string for either the `top_rising_terms`
    or `top_terms` table.
    """
    if table_name == "top_rising_terms":       # rising terms (may be <200)
        return f"""
        WITH raw AS (
            SELECT
                term,
                dma_id,
                percent_gain,
                score,
                week
            FROM  `bigquery-public-data.google_trends.{table_name}`
            WHERE refresh_date = DATE('{LATEST}')
              AND percent_gain >= {MIN_PERCENT_GAIN}
              AND score        >= {MIN_SCORE}
        ),
        dedup AS (                               -- keep newest week per term×DMA
            SELECT * EXCEPT(rn)
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY term, dma_id
                                          ORDER BY week DESC) AS rn
                FROM raw
            )
            WHERE rn = 1
        ),
        stats AS (                               -- aggregate stats per term
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
            dma_hits/210.0                       AS coverage_ratio,
            median_gain,
            (dma_hits/210.0)*median_gain         AS spread_intensity_score
        FROM stats
        ORDER BY spread_intensity_score DESC
        LIMIT 200
        """
    else:                                       # `top_terms` table
        return f"""
        SELECT
            term,
            score
        FROM  `bigquery-public-data.google_trends.{table_name}`
        WHERE refresh_date = DATE('{LATEST}')
        ORDER BY score DESC
        LIMIT 200
        """


def export(csv_prefix: str, sql: str) -> None:
    """Run query → DataFrame → CSV; print row-count."""
    df     = client.query(sql).result().to_dataframe()
    fname  = f"{csv_prefix}_{LATEST}.csv"
    df.to_csv(fname, index=False)
    print(f"[✓]  {fname} written  (rows: {len(df)})")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    export("trend_rising", build_query("top_rising_terms"))
    export("trend_top",    build_query("top_terms"))
