import os
import json
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# === Load API Key ===
# Load from project-root `.env` if present; fallback to default search
load_dotenv()
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

# === Gemini Safety Settings ===
safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

# === Fetch trending hashtags from Gemini ===
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
        return ["fashion", "style", "outfitoftheday"]  # fallback

# === Scrape TikTok metadata per hashtag ===
def scrape_hashtag(tag, max_videos=250):
    print(f"🔍 Scraping TikTok for #{tag}...")
    url = f"https://www.tiktok.com/tag/{tag}"
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    time.sleep(5)

    post_links = set()
    data = []
    last_height = driver.execute_script("return document.body.scrollHeight")

    while len(post_links) < max_videos:
        anchors = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/video/"]')
        for a in anchors:
            href = a.get_attribute("href")
            if href and href not in post_links:
                post_links.add(href)
            if len(post_links) >= max_videos:
                break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    print(f"  ✅ Found {len(post_links)} video URLs, scraping metadata...")

    for url in list(post_links)[:max_videos]:
        try:
            driver.get(url)
            time.sleep(3)
            
            description_elem = driver.find_element(By.CSS_SELECTOR, '[data-e2e="browse-video-desc"]')
            likes = driver.find_element(By.CSS_SELECTOR, 'strong[data-e2e="like-count"]').text
            comments = driver.find_element(By.CSS_SELECTOR, 'strong[data-e2e="comment-count"]').text
            shares = driver.find_element(By.CSS_SELECTOR, 'strong[data-e2e="share-count"]').text

            data.append({
                "hashtag": tag,
                "url": url,
                "description": description_elem.text,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "scraped_at": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            print(f"  ⚠️ Skipped video: {e}")
            continue

    driver.quit()
    return data

# === Main ===
if __name__ == "__main__":
    print("🤖 Selecting trending TikTok hashtags using Gemini...")
    hashtags = get_trending_hashtags()
    print(f"🔥 Gemini recommends: {hashtags}")

    all_data = []
    for tag in hashtags:
        all_data.extend(scrape_hashtag(tag))

    # Save to repo-relative path under `1_data_collection` for portability
    os.makedirs("1_data_collection", exist_ok=True)
    save_path = os.path.join("1_data_collection", "tiktok_combined_llm_rich.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n📆 Scraping complete. Total videos: {len(all_data)}")
    print(f"📦 Data saved to: {save_path}")
