#!/usr/bin/env python3
"""
caption_agent.py – Generate Instagram‑style captions from a CSV of Google Trends rising terms.
"""

import os
import re
import json
import argparse
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ─── CONFIG ────────────────────────────────────────────────────────────────
load_dotenv()  # expects GENAI_API_KEY in your .env
genai.api_key = os.getenv("GENAI_API_KEY")
MODEL = genai.GenerativeModel("models/gemini-2.0-flash")

PROMPT = """
Generate 6 catchy and 6 humorous Instagram post captions for the topic "{trend}".
Each caption should be ≤200 characters, include 2–3 relevant hashtags, and be written
in an upbeat, engaging style. Output JSON exactly as:
{{"captions": ["caption1", "caption2", …, "caption12"]}}
"""

NUM_CAPTIONS = 12


def generate_captions_for_trend(trend: str, n: int = NUM_CAPTIONS) -> list[str]:
    """
    Call Gemini, extract the JSON block, escape stray quotes and backslashes,
    parse, and return up to n captions.
    """
    response = MODEL.generate_content(
        PROMPT.format(trend=trend),
        generation_config={"temperature": 0.7, "max_output_tokens": 2048},
        safety_settings=[],
    )
    raw = response.text.strip()

    # 1) Extract the first {...} JSON block
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"Could not extract JSON from model response:\n{raw}")
    jtext = match.group()

    # 1a) Escape any stray backslashes so the JSON parser won't choke on invalid \x escapes
    jtext = jtext.replace("\\", "\\\\")

    # 2) Escape any un‑escaped " inside the caption text fields
    def _escape_inner(m):
        prefix, body = m.group(1), m.group(2)
        safe_body = body.replace('"', '\\"')
        return f"{prefix}{safe_body}\""

    jtext = re.sub(
        r'("text"\s*:\s*")(.+?)(")',
        _escape_inner,
        jtext,
        flags=re.DOTALL
    )

    # 3) Parse the sanitized JSON
    try:
        payload = json.loads(jtext)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e.msg}\nSanitized JSON:\n{jtext}")

    caps = payload.get("captions", []) or []
    # If Gemini returned objects, extract their "text" field
    if caps and isinstance(caps[0], dict):
        caps = [c.get("text", "") for c in caps]

    return caps[:n]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Instagram captions from Google Trends rising terms CSV."
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to input CSV (must contain a 'term' column). Defaults to trend_rising_<today>.csv",
        default=None,
    )
    parser.add_argument(
        "--n", type=int, default=NUM_CAPTIONS,
        help="Number of captions per term"
    )
    args = parser.parse_args()

    # Determine input file (either user‑supplied or yesterday's rising trends)
    if args.input:
        infile = args.input
    else:
        today = datetime.now(timezone.utc).date().isoformat()
        infile = f"trend_rising_{today}.csv"

    if not os.path.isfile(infile):
        raise FileNotFoundError(f"Input file not found: {infile}")

    df = pd.read_csv(infile)
    basename = os.path.splitext(os.path.basename(infile))[0]
    out_file = f"captions_{basename}.csv"

    rows = []
    for term in df.get("term", []):
        for idx, cap in enumerate(generate_captions_for_trend(term, args.n), start=1):
            rows.append({"term": term, "caption_id": idx, "caption": cap})

    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"[✓] Wrote {len(rows)} captions to {out_file}")


if __name__ == "__main__":
    main()
