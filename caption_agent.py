#!/usr/bin/env python3
"""
caption_agent.py – Generate Instagram‑style captions for trending topics.
Reads in a specified `trend_top_<date>.csv`, calls Gemini,
and outputs `captions_<date>.csv` with multiple examples per trend.
"""
import os
import json
from datetime import datetime
import argparse
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ─── CONFIG ────────────────────────────────────────────────────────────────
load_dotenv()  # Expect GENAI_API_KEY in .env
genai.api_key = os.getenv("GENAI_API_KEY")
MODEL = genai.GenerativeModel("models/gemini-2.0-flash")

# Prompt template: adjust tone, length, hashtags, JSON output format
PROMPT = """
Generate 6 catchy and 6 humorous Instagram post captions for the trending topic "{trend}".  
Each caption should be ≤200 characters, include 2–3 relevant hashtags, and be written
in an upbeat, engaging style. Output JSON as:
{{"captions": ["...","...", …]}}
"""

# Total number of captions expected per trend
NUM_CAPTIONS = 12

def generate_captions_for_trend(trend: str) -> list[str]:
    """Call the LLM to generate captions for a given trend."""
    prompt = PROMPT.format(trend=trend)
    response = MODEL.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "max_output_tokens": 150},
        safety_settings=[
            # reuse your existing safety settings from TikTok script
        ]
    )
    # Extract JSON from markdown or code fences
    text = response.text.strip().strip("```json").strip("```")
    payload = json.loads(text)
    captions = payload.get("captions", [])
    # Ensure we only return up to NUM_CAPTIONS
    return captions[:NUM_CAPTIONS]


def main():
    # Allow overriding which date to use (for historical trend data)
    parser = argparse.ArgumentParser(
        description="Generate Instagram-style captions from a trend_top_<date>.csv file"
    )
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD for which to generate captions (defaults to today UTC)",
        default=None,
    )
    args = parser.parse_args()
    date_to_use = args.date or datetime.utcnow().date().isoformat()

    # 1) Locate the trend CSV for the chosen date
    filename = f"trend_top_{date_to_use}.csv"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Trend file not found: {filename}")
    df = pd.read_csv(filename)

    # 2) Generate captions
    all_rows = []
    for trend in df["term"].unique():
        caps = generate_captions_for_trend(trend)
        for idx, cap in enumerate(caps, start=1):
            all_rows.append({
                "term": trend,
                "caption_id": idx,
                "caption": cap,
            })

    # 3) Write output CSV
    out_df = pd.DataFrame(all_rows)
    out_file = f"captions_{date_to_use}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"[✓] Wrote {len(out_df)} captions to {out_file}")


if __name__ == "__main__":
    main()
