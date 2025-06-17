import os
import pandas as pd
import datetime
from playwright.sync_api import sync_playwright, TimeoutError

TOP_COUNTRIES = [
    "united-states",
    "india",
    "brazil",
    "united-kingdom",
    "indonesia",
    "mexico",
    "philippines",
    "canada",
    "germany",
    "japan"
]

def fetch_trends(region):
    print(f"Beginning extraction for region: {region}")
    url = f"https://trends24.in/{region}/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"[INFO] Fetching: {url}")
        page.goto(url, timeout=60000)

        try:
            print("[INFO] Forcing 'Table' tab to load...")
            page.evaluate("document.querySelector('#tab-link-table').click()")
            page.wait_for_selector("section#table table", timeout=15000)
        except TimeoutError:
            print("[WARN] Table tab may already be active or not visible.")

        rows = page.query_selector_all("section#table table tbody tr")
        print(f"[DEBUG] Found {len(rows)} rows in table for {region}")

        trends = []
        for row in rows:
            cols = row.query_selector_all("td")
            if len(cols) < 2:
                continue

            def safe_text(col_idx):
                return cols[col_idx].inner_text().strip() if col_idx < len(cols) else ""

            trends.append({
                "region": region,
                "rank": safe_text(0),
                "trend": safe_text(1),
                "top_position": safe_text(2),
                "tweet_count": safe_text(3),
                "duration": safe_text(4),
                "timestamp": datetime.datetime.utcnow().isoformat()
            })

        browser.close()
        return trends

def save_to_csv(all_data, filename="trends_output.csv"):
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"[DONE] Saved {len(df)} total trends to {filename}" if len(df) else "No trends found\nCreated empty CSV file")

if __name__ == "__main__":
    all_trends = []
    for country in TOP_COUNTRIES:
        trends = fetch_trends(country)
        all_trends.extend(trends)
    save_to_csv(all_trends)
