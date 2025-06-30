#!/usr/bin/env python3
"""
trend_bot.py  –  pulls two Google-Trends exports every day

• trend_rising_YYYY-MM-DD.csv   (top_rising_terms, ≤200 rows)
• trend_top_YYYY-MM-DD.csv      (top_terms,        ≤200 rows)

Both CSVs have the same four quantitative columns so downstream
ranking / embedding code can treat them identically.
"""

import datetime as dt
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()                       # picks up GOOGLE_APPLICATION_CREDENTIALS
client = bigquery.Client()

# ────────────────────────────── CONFIG ──────────────────────────────
WINDOW_DAYS     = 7                 # how many latest partitions (days) to scan
MIN_SCORE_GAIN  = 0                 # set >0 to filter very low-quality rows
CSV_PREFIX      = "trend"           # base name of output CSVs
DMA_DENOM       = 210.0             # number of US DMAs, for coverage_ratio
# ────────────────────────────────────────────────────────────────────


# ───────────────────────────── helpers ──────────────────────────────
def get_recent_partitions(n_days: int) -> list[str]:
    """
    Return a *list* of partition dates (YYYY-MM-DD) – newest first – that are
    visible in *top_rising_terms*.  We use that to probe both tables.
    """
    sql = """
    SELECT PARSE_DATE('%Y%m%d', partition_id) AS dt
    FROM  `bigquery-public-data.google_trends.INFORMATION_SCHEMA.PARTITIONS`
    WHERE table_name='top_rising_terms'
      AND partition_id!='__NULL__'
    ORDER BY dt  DESC
    LIMIT  @N
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("N", "INT64", n_days)]
        ),
    )
    return [row["dt"] for row in job.result()]


PARTITIONS  = get_recent_partitions(WINDOW_DAYS)  # list of DATEs
LATEST      = PARTITIONS[0]                       # newest – used in filenames


def build_query(table_name: str) -> str:
    """
    Return SQL (NO parameters) for *either* table.  We keep the select-list
    identical (term, dma_hits, coverage_ratio, total_score, avg_rank) so that
    both CSVs line up.
    """
    metric_col = "percent_gain" if "rising" in table_name else "score"

    ###############
    # 1) raw rows #
    ###############
    base_select = f"""
    SELECT
        term,
        dma_id,
        {metric_col}               AS metric,
        rank,
        refresh_date
    FROM `bigquery-public-data.google_trends.{table_name}`
    WHERE refresh_date IN ({','.join([f"'{d}'" for d in PARTITIONS])})
      AND {metric_col} >= {MIN_SCORE_GAIN}
    """

    ############################################################################
    # rising-specific branch: keep *strongest DMA* per term/date, then         #
    # aggregate across WINDOW and compute spread_intensity_score exactly as   #
    # before (this path is almost unchanged from your working version).        #
    ############################################################################
    if "rising" in table_name:
        return f"""
        WITH raw AS ({base_select}),

        /* keep strongest DMA on each refresh_date for every term */
        dedup AS (
          SELECT *
          FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY term, refresh_date
                                      ORDER BY metric DESC) rk
            FROM raw)
          WHERE rk = 1
        ),

        /* aggregate across the WINDOW and rank by spread_intensity_score */
        stats AS (
          SELECT
             term,
             COUNT(DISTINCT dma_id)                       AS dma_hits,
             APPROX_QUANTILES(metric,2)[OFFSET(1)]        AS median_score
          FROM dedup
          GROUP BY term
        )
        SELECT
           term,
           dma_hits,
           dma_hits/{DMA_DENOM}                          AS coverage_ratio,
           median_score                                  AS total_score,
           0                                             AS avg_rank,          -- filler
           (dma_hits/{DMA_DENOM})*median_score           AS spread_intensity_score
        FROM stats
        ORDER BY spread_intensity_score DESC
        LIMIT 200
        """

    ###########################################################################
    # top-terms branch: MUCH simpler – just aggregate strongest score + rank  #
    # for every term over the WINDOW                                          #
    ###########################################################################
    return f"""
    WITH raw AS ({base_select}),

    stats AS (
      SELECT
         term,
         COUNT(DISTINCT dma_id)     AS dma_hits,
         SUM(metric)                AS total_score,     -- 0–2000 approx
         AVG(rank)                  AS avg_rank
      FROM raw
      GROUP BY term
    )
    SELECT
       term,
       dma_hits,
       dma_hits/{DMA_DENOM}         AS coverage_ratio,
       total_score,
       avg_rank,
       total_score                  AS spread_intensity_score   -- keep column name stable
    FROM stats
    ORDER BY total_score DESC
    LIMIT 200
    """


def export(csv_prefix: str, sql: str) -> None:
    df     = client.query(sql).result().to_dataframe()
    fname  = f"{csv_prefix}{LATEST}.csv"
    df.to_csv(fname, index=False)
    print(f"[✓] {fname} written   (rows: {len(df)})")


# ─────────────────────────────── main ───────────────────────────────
if __name__ == "__main__":
    export(f"{CSV_PREFIX}_rising_", build_query("top_rising_terms"))
    export(f"{CSV_PREFIX}_top_"   , build_query("top_terms"))
