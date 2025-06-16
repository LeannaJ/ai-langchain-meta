import os
import requests
import datetime
import pandas as pd
from bs4 import BeautifulSoup

def fetch_trends(region="united-states"):
    url = f"https://trends24.in/{region}/"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from {url}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    region_header = soup.find("h1", class_="mt-1")
    region_name = region_header.text.strip() if region_header else "Unknown Region"

    trends_data = []
    trend_cards = soup.select(".trend-card__list")

    for card in trend_cards:
        for rank, item in enumerate(card.select("li"), 1):
            trend_tag = item.find("a")
            if not trend_tag:
                continue

            trend_text = trend_tag.text.strip()
            trend_link = trend_tag["href"].strip()
            full_link = f"https://trends24.in{trend_link}"

            # Find metadata span
            meta_div = item.find("span", class_="tweet-count")
            top_position = tweet_count = duration = ""

            if meta_div:
                meta_parts = [p.strip() for p in meta_div.text.split("•")]

                if len(meta_parts) == 1:
                    tweet_count = meta_parts[0]
                elif len(meta_parts) == 2:
                    tweet_count, duration = meta_parts
                elif len(meta_parts) == 3:
                    top_position, tweet_count, duration = meta_parts

            trends_data.append({
                "region": region_name,
                "rank": rank,
                "trend": trend_text,
                "top_position": top_position,
                "tweet_count": tweet_count,
                "duration": duration,
                "link": full_link,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })

        if len(trends_data) >= 50:
            break

    return trends_data[:50]

def save_to_csv(data, filename="trends_output.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    try:
        trends = fetch_trends()
        save_to_csv(trends)
        print(f"[DONE] Extracted and saved {len(trends)} trends to trends_output.csv")
    except Exception as e:
        print(f"[ERROR] {e}")
