import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# === Load API Key ===
load_dotenv(dotenv_path=r"C:\\Users\\arman\\OneDrive\\Documents\\PURDUE\\SUMMER\\META IP\\.env")
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

# === Get Hashtags from Gemini ===
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

# === Scrape Metadata for a Single Video URL ===
def scrape_video_metadata(driver, url, hashtag):
    driver.get(url)
    time.sleep(4)

    def safe_find(selector):
        try:
            return driver.find_element(By.CSS_SELECTOR, selector).text
        except NoSuchElementException:
            return None

    return {
        "hashtag": hashtag,
        "url": url,
        "description": safe_find("h1[data-e2e='browse-video-desc']"),
        "likes": safe_find("strong[data-e2e='like-count']"),
        "comments": safe_find("strong[data-e2e='comment-count']"),
        "shares": safe_find("strong[data-e2e='share-count']"),
        "views": safe_find("strong[data-e2e='play-count']"),
        "scraped_at": datetime.now().astimezone().isoformat()
    }

# === Scrape Hashtag Page for Videos ===
def scrape_hashtag(tag, max_videos=50):
    print(f"🔍 Scraping TikTok for #{tag}...")

    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    driver.get(f"https://www.tiktok.com/tag/{tag}")
    time.sleep(5)

    video_urls = set()
    last_height = driver.execute_script("return document.body.scrollHeight")

    while len(video_urls) < max_videos:
        anchors = driver.find_elements(By.TAG_NAME, "a")
        for a in anchors:
            href = a.get_attribute("href")
            if href and "/video/" in href and href not in video_urls:
                video_urls.add(href)
                print(f"  ➕ Found video: {href}")
                if len(video_urls) >= max_videos:
                    break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    print(f"  ✅ Found {len(video_urls)} videos for #{tag}")

    results = []
    for url in list(video_urls)[:max_videos]:
        try:
            metadata = scrape_video_metadata(driver, url, tag)
            results.append(metadata)
        except Exception as e:
            print(f"⚠️ Error scraping metadata for {url}: {e}")
        time.sleep(2)

    driver.quit()
    return results

# === Main ===
if __name__ == "__main__":
    print("🤖 Selecting trending TikTok hashtags using Gemini...")
    hashtags = get_trending_hashtags()
    print(f"🔥 Gemini recommends: {hashtags}")

    all_results = []
    for tag in hashtags:
        all_results.extend(scrape_hashtag(tag, max_videos=10))  # You can raise to 1000 later

    output_path = r"C:\Users\arman\OneDrive\Documents\PURDUE\SUMMER\META IP\tiktok_combined_llm_enriched.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📆 Scraping complete. Total videos: {len(all_results)}")
    print(f"📦 Data saved to: {output_path}")
