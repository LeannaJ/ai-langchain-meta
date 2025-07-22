#!/usr/bin/env python3
"""
caption_agent.py – Generate Instagram‑style captions for trending topics.
Reads in the latest `trend_top_<date>.csv` (or rising), calls Gemini,
and outputs `captions_<date>.csv` with multiple examples per trend.
"""
import os
import json
from datetime import datetime
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

NUM_CAPTIONS = 5

def generate_captions_for_trend(trend: str, n: int = NUM_CAPTIONS) -> list[str]:
    prompt = PROMPT.format(trend=trend, n=n)
    response = MODEL.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "maxOutputTokens": 150},
        safety_settings=[
            # reuse your existing safety settings from TikTok script
        ]
    )
    # Extract JSON from markdown or code fences
    text = response.text.strip().strip("```json").strip("```")
    payload = json.loads(text)
    return payload.get("captions", [])

def main():
    # 1) Locate today's trend CSV
    today = datetime.utcnow().date().isoformat()
    # e.g. CHOOSE trend_top_<date>.csv by checking existence
    filename = f"trend_top_{today}.csv"
    df = pd.read_csv(filename)

    # 2) Generate captions
    all_rows = []
    for trend in df["term"].unique():
        caps = generate_captions_for_trend(trend)
        for idx, cap in enumerate(caps, start=1):
            all_rows.append({"term": trend, "caption_id": idx, "caption": cap})

    out_df = pd.DataFrame(all_rows)
    out_file = f"captions_{today}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"[✓] Wrote {len(out_df)} captions to {out_file}")

if __name__ == "__main__":
    main()
