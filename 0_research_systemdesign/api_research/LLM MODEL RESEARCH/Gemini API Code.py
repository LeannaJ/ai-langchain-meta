import os
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# === Load API key ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env!")
genai.configure(api_key=api_key)

# === Prompt Template ===
PROMPT_TEMPLATE = """
You are a social media trend analyst. Produce 3–5 emerging trends in JSON format.

Platform: {platform}
Category: {category}
Region: {region}
Timeframe: {timeframe}

Format the output exactly like:
{{
  "platform": "{platform}",
  "category": "{category}",
  "region": "{region}",
  "date": "YYYY-MM-DD",
  "trends": [
    {{
      "name": "trend name",
      "popularity_score": 0–100,
      "brief": "short explanation"
    }}
  ]
}}
"""

# === Functions ===
def get_trends(platform, category, region="US", timeframe="last 7 days"):
    prompt = PROMPT_TEMPLATE.format(
        platform=platform,
        category=category,
        region=region,
        timeframe=timeframe
    )
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

def parse_response(text):
    try:
        return json.loads(text)
    except:
        print("⚠️ Couldn't parse JSON, saving raw response.")
        return {"raw_response": text, "date": datetime.today().strftime("%Y-%m-%d")}

def save_json(data, filename="social_trends.json"):
    if os.path.exists(filename):
        with open(filename, "r+", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except:
                existing = []
            existing.append(data)
            f.seek(0)
            json.dump(existing, f, indent=2)
    else:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([data], f, indent=2)

def run_trend_scrape(platform, category):
    print(f"📤 Querying Gemini for {platform}/{category} trends...")
    text = get_trends(platform, category)
    result = parse_response(text)
    save_json(result)
    print("✅ Trend data saved to social_trends.json")

# === Run ===
if __name__ == "__main__":
    run_trend_scrape("TikTok", "fashion")



