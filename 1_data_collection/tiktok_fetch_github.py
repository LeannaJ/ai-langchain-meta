import os
import json
import asyncio
from datetime import datetime
from TikTokApi import TikTokApi

INPUT_HASHTAGS_JSON = os.path.join("data", "trends24_us_hashtags.json")
OUTPUT_JSON = os.path.join("data", "tiktok_tiktokapi_results.json")

# Load hashtags
with open(INPUT_HASHTAGS_JSON, "r", encoding="utf-8") as f:
    hashtag_list = [tag.strip("#") for tag in json.load(f).get("hashtags", [])]

async def fetch_hashtag_videos(api, hashtag):
    print(f"🔍 Fetching videos for #{hashtag}")
    results = []
    try:
        hashtag_obj = api.hashtag(name=hashtag)
        gen = hashtag_obj.videos(count=5)

        i = 0
        while True:
            try:
                video = await anext(gen)
                results.append({
                    "hashtag": hashtag,
                    "url": f"https://www.tiktok.com/@{video.as_dict.get('author', {}).get('uniqueId', 'unknown')}/video/{video.as_dict.get('id', '')}",
                    "description": video.as_dict.get("desc", ""),
                    "likes": video.as_dict.get("stats", {}).get("diggCount", 0),
                    "comments": video.as_dict.get("stats", {}).get("commentCount", 0),
                    "shares": video.as_dict.get("stats", {}).get("shareCount", 0),
                    "views": video.as_dict.get("stats", {}).get("playCount", 0),
                    "scraped_at": datetime.now().astimezone().isoformat()
                })
                i += 1
                if i >= 5:
                    break
            except StopAsyncIteration:
                break
            except Exception as e:
                print(f"⚠️ Error getting video: {e}")
                break

    except Exception as e:
        print(f"⚠️ Error with #{hashtag}: {e}")
    return results

async def main():
    api = TikTokApi()
    await api.create_sessions(
        headless=True,          # ✅ must be True on GitHub Actions
        browser="chromium"      # ✅ more CI-friendly
    )


    all_results = []
    for tag in hashtag_list:
        videos = await fetch_hashtag_videos(api, tag)
        all_results.extend(videos)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved {len(all_results)} videos to {OUTPUT_JSON}")

if __name__ == "__main__":
    asyncio.run(main())
