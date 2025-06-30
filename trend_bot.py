#!/usr/bin/env python3
"""
trend_bot.py  ─  pulls the Top-200 U.S. *rising* & *top* Google-Trends terms
                 looking back over the latest N partitions (default 7 ≈ 7 days)
                 and writes two CSVs with identical columns.

Outputs
  trend_rising_YYYY-MM-DD.csv   (≤200 rows)
  trend_top_   YYYY-MM-DD.csv   (≤200 rows)
"""

import datetime as dt
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()                      # picks up GOOGLE_APPLICATION_CREDENTIALS
client = bigquery.Client()

# ─────────────── CONFIG ──────────────────────────────────────────────────
WINDOW_DAYS   = 7       # how many latest partitions to include
MIN_SCORE_GAIN = 0      # set >0 to filter low-quality rows
CSV_PREFIX    = "trend" # base name of output CSVs
# ─────────────────────────────────────────────────────────────────────────

# ─────────────── helpers ─────────────────────────────────────────────────
def get_recent_partitions(n_days: int) -> list[str]:
    """
    Return a *list* of DATE strings (YYYY-MM-DD) for the n newest partitions
    seen in `top_rising_terms`   (newest first).
    """
    sql = f"""
    SELECT PARSE_DATE('%Y%m%d', partition_id) AS dt
    FROM `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
    WHERE table_name='top_rising_terms'
      AND partition_id!='NULL'
    ORDER BY partition_id DESC
    LIMIT {n_days}
    """
    return [row["dt"] for row in client.query(sql).result()]

PARTITIONS = get_recent_partitions(WINDOW_DAYS)       # list of DATEs
LATEST     = PARTITIONS[0]                            # newest – for filenames

def build_query(table_name: str) -> str:
    """Return SQL (no parameters) for either table, scanning all PARTITIONS."""
    metric = 'percent_gain' if 'rising' in table_name else 'score'
    with_clause = ""

    # Pull *all* rows for the requested partitions
    base_select = f"""
      SELECT term, dma_id, {metric}, week, refresh_date
      FROM  `bigquery-public-data.google_trends.{table_name}`
      WHERE refresh_date IN ({', '.join([f"DATE '{d}'" for d in PARTITIONS])})
        AND {metric} >= {MIN_SCORE_GAIN}
    """

    if 'rising' in table_name:
        with_clause = f"""
        WITH raw AS ( {base_select} ),

        /* keep strongest DMA for each term*date          */
        dedup AS (
          SELECT *
          FROM (
            SELECT *,
                   ROW_NUMBER() OVER
                     (PARTITION BY term, refresh_date ORDER BY {metric} DESC) rk
            FROM raw)
          WHERE rk = 1
        ),

        /* aggregate across the WINDOW                  */
        stats AS (
          SELECT
            term,
            COUNT(DISTINCT dma_id)                           AS dma_hits,
            APPROX_QUANTILES({metric},2)[OFFSET(1)]          AS median_score
          FROM dedup
          GROUP BY term
        )
        SELECT
          term,
          dma_hits,
          dma_hits/210.0                       AS coverage_ratio,
          median_score,
          (dma_hits/210.0)*median_score        AS spread_intensity_score
        FROM stats
        ORDER BY spread_intensity_score DESC
        LIMIT 200
        """

        return with_clause

    # else: top_terms  → much simpler; aggregate strongest score across window
    return f"""
    WITH raw AS ( {base_select} ),
    stats AS (
      SELECT term, MAX({metric}) AS score
      FROM   raw
      GROUP  BY term
    )
    SELECT
      term,
      score                                          AS median_score,      -- keep column name stable
      0                     AS dma_hits,            -- fillers (top_terms has no DMA granularity)
      0                     AS coverage_ratio,
      score                 AS spread_intensity_score
    FROM stats
    ORDER BY score DESC
    LIMIT 200
    """

def export(prefix: str, sql: str) -> None:
    df   = client.query(sql).result().to_dataframe()
    file = f"{prefix}_{LATEST}.csv"
    df.to_csv(file, index=False)
    print(f"[+] {file} (rows: {len(df)})")

# ─────────────── main ────────────────────────────────────────────────────
if __name__ == "__main__":
    export(f"{CSV_PREFIX}_rising", build_query("top_rising_terms"))
    export(f"{CSV_PREFIX}_top",    build_query("top_terms"))
