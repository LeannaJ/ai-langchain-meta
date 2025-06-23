import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright
import nest_asyncio
from TikTokApi import TikTokApi

nest_asyncio.apply()

async def get_single_ms_token(playwright):
    browser = await playwright.firefox.launch(headless=True, slow_mo=100)
    context = await browser.new_context()
    page = await context.new_page()

    print("\U0001f310 Opening TikTok...")
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
            print(f"\n\U0001f501 Session #{i+1}")
            token = await get_single_ms_token(p)
            if token:
                print(f"✅ Token #{i+1}: {token[:50]}...")
                tokens.append(token)
            else:
                print(f"❌ Failed to retrieve token #{i+1}")
    return tokens

async def scrape_trending_videos(ms_token_list):
    api = TikTokApi()
    await api.create_sessions(
        ms_tokens=ms_token_list,
        num_sessions=len(ms_token_list),
        sleep_after=3,
        browser="chromium",
        headless=True
    )

    all_data = []
    for i, session in enumerate(api.sessions):
        print(f"\U0001f4e5 Scraping with token #{i+1}")
        count = 0
        async for video in api.trending.videos(session=session, count=30):
            all_data.append(video.as_dict)
            count += 1
        print(f"✅ Retrieved {count} videos from token #{i+1}")
    return all_data

def clean_and_deduplicate(videos):
    unique_videos = {}
    for v in videos:
        video_id = v.get("id")
        author_id = v.get("author", {}).get("uniqueId")
        if not video_id:
            continue
        if video_id not in unique_videos:
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
    return list(unique_videos.values())

async def main():
    ms_token_list = await collect_ms_tokens(6)
    print("\n\U0001f3af All collected ms_tokens:")
    for i, t in enumerate(ms_token_list, 1):
        print(f"#{i}: {t}")

    all_data = await scrape_trending_videos(ms_token_list)
    print(f"\n\U0001f4ca Total videos collected: {len(all_data)}")

    deduped_cleaned = clean_and_deduplicate(all_data)
    print(f"\U0001f4e6 Number of unique videos after deduplication: {len(deduped_cleaned)}")

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
