#!/usr/bin/env python
# coding: utf-8

# In[21]:


import asyncio
from playwright.async_api import async_playwright

async def get_single_ms_token(playwright):
    browser = await playwright.firefox.launch(headless=True, slow_mo=100)
    context = await browser.new_context()
    page = await context.new_page()

    print("🌐 Opening TikTok...")
    await page.goto("https://www.tiktok.com", timeout=60000)
    await page.wait_for_timeout(15000)
    cookies = await context.cookies()
    await browser.close()

    # Extract last ms_token if available
    ms_tokens = [c["value"] for c in cookies if c["name"] == "msToken"]
    return ms_tokens[-1] if ms_tokens else None

async def collect_ms_tokens(n=6):
    tokens = []
    async with async_playwright() as p:
        for i in range(n):
            print(f"\n🔁 Session {i+1}")
            token = await get_single_ms_token(p)
            if token:
                print(f"✅ Token #{i+1}: {token[:50]}...")  # Preview first 50 chars
                tokens.append(token)
            else:
                print(f"❌ Failed to retrieve token #{i+1}")
    return tokens

# Run it
ms_token_list = asyncio.run(collect_ms_tokens(6))

# Optional: print all tokens
print("\n🎯 All collected ms_tokens:")
for i, t in enumerate(ms_token_list, 1):
    print(f"#{i}: {t}")


# In[22]:


import nest_asyncio
import asyncio
import json
from TikTokApi import TikTokApi

nest_asyncio.apply()


output_file = "tiktok_trending1.json"

async def main():
    all_data = []
    api = TikTokApi()
    await api.create_sessions(
        ms_tokens = ms_token_list,
        num_sessions=len(ms_token_list),
        sleep_after=3,
        browser="chromium",
        headless=True
    )

    for i in range(len(ms_token_list)):
        print(f"📥 Scraping with token #{i+1}")
        session = api.sessions[i]
        count = 0
        async for video in api.trending.videos(session=session, count=30):
            all_data.append(video.as_dict)
            count += 1
        print(f"✅ Retrieved {count} videos from token #{i+1}")


    print(f"\n📊 Total videos collected: {len(all_data)}")

    if all_data:
        print("\n🔍 Sample video:")
        print(json.dumps(all_data[0], indent=2))

    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n💾 Data saved to {output_file}")

# Run everything
if __name__ == "__main__":
    import asyncio
    cleaned_videos = asyncio.run(main())


# In[23]:


import json
from pprint import pprint

with open("/Users/hazel/Documents/Purdue/Advanced IP/Practice File/tiktok_trending1.json", "r") as f:
    videos = json.load(f)  # now the root is directly a list of videos

# Show results
print("📦 Number of videos:", len(videos))

if videos:
    print("🔍 First video:")
    pprint(videos[0], depth=3, width=100)
else:
    print("⚠️ No videos found.")


# In[ ]:


from datetime import datetime

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
            "music_author_name": v.get("music", {}).get("authorName"),  # ✅ added this line
            "video_duration": v.get("video", {}).get("duration"),
            "cover_image": v.get("video", {}).get("cover"),
            "hashtags": [tag.get("hashtagName") for tag in v.get("textExtra", []) if "hashtagName" in tag],
            "challenges": [c.get("title") for c in v.get("challenges", []) if "title" in c]
        }

# Final list
deduped_cleaned = list(unique_videos.values())
print("📦 Number of unique videos after deduplication:", len(deduped_cleaned))


# Save the cleaned and deduplicated data
output_cleaned_file = "tiktok_trending_cleaned.json"

with open(output_cleaned_file, "w", encoding="utf-8") as f:
    json.dump(deduped_cleaned, f, ensure_ascii=False, indent=2)

print(f"✅ Cleaned data saved to: {output_cleaned_file}")

# In[ ]:




