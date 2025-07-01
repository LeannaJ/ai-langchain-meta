import traceback
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
import asyncio
from playwright.async_api import async_playwright

# === Load API Key ===
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env!")
genai.configure(api_key=api_key)

# === Gemini Prompt ===
PROMPT_TEMPLATE = """
You are a social media trend analyst. Output 3–5 trending TikTok hashtags as JSON.

Format it like this:
{
  "date": "YYYY-MM-DD",
  "platform": "TikTok",
  "category": "LLM-judged category name",
  "hashtags": [
    {"tag": "hashtag1", "popularity_score": 0-100},
    {"tag": "hashtag2", "popularity_score": 0-100}
  ]
}
"""

safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

def get_trending_hashtags():
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(
        PROMPT_TEMPLATE,
        generation_config={"temperature": 0},
        safety_settings=safety_settings
    )
    try:
        text = response.text.strip().strip('```json').strip('```')
        print("\n📨 Gemini raw response:\n", text)
        parsed = json.loads(text)
        return [h['tag'].lstrip('#') for h in parsed['hashtags']]
    except Exception as e:
        print("⚠️ Error parsing Gemini response:", e)
        return ["fashion", "style", "outfitoftheday"]

async def scrape_hashtag(tag, max_videos=10):
    results = []
    url = f"https://www.tiktok.com/tag/{tag}"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            print(f"🔍 Visiting: {url}")
            await page.goto(url, timeout=60000)
            await page.wait_for_timeout(5000)
            videos = await page.locator("div[data-e2e='search-video-item']").all()
            for video in videos[:max_videos]:
                try:
                    href = await video.locator("a").get_attribute("href")
                    desc = await video.locator("a").text_content()
                    results.append({"url": href, "description": desc})
                except Exception as ve:
                    print(f"⚠️ Failed to parse one video: {ve}")
        except Exception as e:
            print(f"❌ Error during Playwright scraping for #{tag}: {e}")
        await browser.close()
    return results

async def main():
    try:
        print("🤖 Selecting trending TikTok hashtags using Gemini...")
        hashtags = get_trending_hashtags()
        print(f"🔥 Gemini recommends: {hashtags}")

        all_results = []
        for tag in hashtags:
            try:
                result = await scrape_hashtag(tag, max_videos=10)
                all_results.extend(result)
            except Exception as e:
                print(f"❌ Failed to scrape #{tag}: {e}")
                traceback.print_exc()

        output_path = "tiktok_combined_llm_enriched.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n📆 Scraping complete. Total videos: {len(all_results)}")
        print(f"📦 Data saved to: {output_path}")
    except Exception as e:
        print("❌ Unhandled error in main():", e)
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("❌ Global Exception:", e)
        traceback.print_exc()
        exit(1)
