#!/usr/bin/env python3
"""
caption_agent.py – Generate Instagram‑style captions for trending topics.
Reads from `trend_top_<date>.csv`, calls Gemini, sanitizes the JSON, and outputs `captions_<date>.csv` with example captions.
"""
import os
import json
import re
import argparse
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ─── CONFIG ────────────────────────────────────────────────────────────────
load_dotenv()  # Expect GENAI_API_KEY in .env
genai.api_key = os.getenv("GENAI_API_KEY")
MODEL = genai.GenerativeModel("models/gemini-2.0-flash")

# Prompt template, note double braces for Python formatting
PROMPT = """
Generate 6 catchy and 6 humorous Instagram post captions for the trending topic "{trend}".
Each caption should be ≤200 characters, include 2–3 relevant hashtags, and be written
in an upbeat, engaging style. Output JSON exactly as:
{{"captions": ["caption1", "caption2", …, "caption12"]}}
"""

# Number of captions to retain
NUM_CAPTIONS = 12


def generate_captions_for_trend(trend: str, n: int = NUM_CAPTIONS) -> list[str]:
    """Call the LLM, extract the JSON block, escape inner quotes, and parse captions."""
    prompt = PROMPT.format(trend=trend)
    response = MODEL.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "max_output_tokens": 2048},
        safety_settings=[]
    )
    raw = response.text.strip()

    # 1) Extract the first JSON-like block
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"Could not extract JSON from LLM response:\n{raw}")
    jtext = match.group()

    # 2) Escape any un-escaped double quotes inside "text" fields
    def _escape_inner_quotes(m: re.Match) -> str:
        prefix = m.group(1)
        body = m.group(2)
        safe_body = body.replace('"', '\\"')
        return f"{prefix}{safe_body}" + '"'

    jtext = re.sub(
        r'("text"\s*:\s*")(.+?)(")',
        _escape_inner_quotes,
        jtext,
        flags=re.DOTALL
    )

    # 3) Parse the sanitized JSON
    try:
        payload = json.loads(jtext)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e.msg}\nSanitized content:\n{jtext}")

    caps_obj = payload.get("captions", []) or []
    # If elements are dicts with 'text', extract the values
    if caps_obj and isinstance(caps_obj[0], dict):
        captions = [item.get("text", "") for item in caps_obj]
    else:
        captions = caps_obj

    # Return up to n captions
    return captions[:n]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Instagram captions from a trend_top_<date>.csv file"
    )
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD for which to generate captions (defaults to today UTC)",
        default=None,
    )
    args = parser.parse_args()
    date_to_use = args.date or datetime.utcnow().date().isoformat()

    filename = f"trend_top_{date_to_use}.csv"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Trend file not found: {filename}")
    df = pd.read_csv(filename)

    all_rows = []
    for trend in df.get("term", []):
        caps = generate_captions_for_trend(trend)
        for idx, cap in enumerate(caps, start=1):
            all_rows.append({"term": trend, "caption_id": idx, "caption": cap})

    out_df = pd.DataFrame(all_rows)
    out_file = f"captions_{date_to_use}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"[✓] Wrote {len(all_rows)} captions to {out_file}")


if __name__ == "__main__":
    main()

