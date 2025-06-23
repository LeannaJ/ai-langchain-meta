import asyncio
import nest_asyncio
import json
from datetime import datetime
from pprint import pprint
from playwright.async_api import async_playwright
from TikTokApi import TikTokApi

nest_asyncio.apply()

async def get_single_ms_token(playwright):
    browser = await playwright.firefox.launch(headless=True, slow_mo=100)
    context = await browser.new_context()
    page = await context.new_page()

    print("🌐 Opening TikTok...")
    await page.goto("https://www.tiktok.com", timeout=60000)
    await page.wait_for_timeout(15000)
    cookies = await context.cookies()
    await browser.close()

    ms_tokens = [c["value"] for c in cookies if c["name"] == "msToken"]
    return ms_tokens[-1] if ms_tokens else None

async def collect_ms_tokens(n=6):
    tokens = []
    async with async_playwright() as p:
        for i in range(n):
            print(f"\n🔁 Session {i+1}")
            token = await get_single_ms_token(p)
            if token:
                print(f"✅ Token #{i+1}: {token[:50]}...")
                tokens.append(token)
            else:
                print(f"❌ Failed to retrieve token #{i+1}")
    return tokens

async def main():
    ms_token_list = await collect_ms_tokens(6)
    all_data = []

    async with async_playwright() as pw:
        api = TikTokApi.get_instance(custom_verify_fp="", playwright=pw)
        for i, token in enumerate(ms_token_list):
            print(f"\n📥 Scraping with token #{i+1}")
            count = 0
            try:
                videos = await api.trending(count=30, ms_token=token)
                for video in videos:
                    all_data.append(video.as_dict)
                    count += 1
                print(f"✅ Retrieved {count} videos from token #{i+1}")
            except Exception as e:
                print(f"⚠️ Failed on token #{i+1}: {e}")

    print(f"\n📊 Total videos collected: {len(all_data)}")

    # Deduplicate and transform
    unique_videos = {}
    for v in all_data:
        video_id = v.get("id")
        author_id = v.get("author", {}).get("uniqueId")

        if not video_id or video_id in unique_videos:
            continue

        unique_videos[video_id] = {
            "video_id": video_id,
            "author_id": author_id,
            "video_url": f"https://www.tiktok.com/@{author_id}/video/{video_id}" if author_id else None,
            "description": v.get("desc"),
            "create_time": datetime.fromtimestamp(v.get("createTime")).isoformat() if v.get("createTime") else None,
            "author_name": v.get("author", {}).get("nickname"),
            "likes": v.get("stats", {}).get("diggCount"),
            "views": v.get("stats", {}).get("playCount"),
            "comments": v.get("stats", {}).get("commentCount"),
            "shares": v.get("stats", {}).get("shareCount"),
            "music_title": v.get("music", {}).get("title"),
            "music_author_name": v.get("music", {}).get("authorName"),
            "video_duration": v.get("video", {}).get("duration"),
            "cover_image": v.get("video", {}).get("cover"),
            "hashtags": [tag.get("hashtagName") for tag in v.get("textExtra", []) if "hashtagName" in tag],
            "challenges": [c.get("title") for c in v.get("challenges", []) if "title" in c]
        }

    deduped_cleaned = list(unique_videos.values())
    print("📦 Number of unique videos after deduplication:", len(deduped_cleaned))

    output_file = "tiktok_trending_cleaned.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(deduped_cleaned, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n🚨 Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
