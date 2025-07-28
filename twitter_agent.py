import os
import pandas as pd
import datetime
import time
from playwright.sync_api import sync_playwright, TimeoutError

TOP_COUNTRIES = [
    "united-states"#,
    # "india",
    # "brazil",
    # "united-kingdom",
    # "indonesia",
    # "mexico",
    # "philippines",
    # "canada",
    # "germany",
    # "japan"
]

def fetch_trends(region):
    print(f"\nBeginning extraction for region: {region}")
    url = f"https://trends24.in/{region}/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Retry logic for page.goto
        for attempt in range(2):
            try:
                print(f"[INFO] Fetching: {url}")
                page.goto(url, timeout=60000)
                break
            except TimeoutError:
                if attempt == 1:
                    raise  # Re-raise after second failure
                print(f"[WARN] Timeout loading {url}, retrying in 5 seconds...")
                time.sleep(5)

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
    print(f"\n[DONE] Saved {len(df)} total trends to {filename}" if len(df) else "No trends found\nCreated empty CSV file")

if __name__ == "__main__":
    all_trends = []
    for country in TOP_COUNTRIES:
        try:
            trends = fetch_trends(country)
            all_trends.extend(trends)
        except TimeoutError:
            print(f"[ERROR] Timeout while processing {country}. Skipping...")
        except Exception as e:
            print(f"[ERROR] Unexpected error for {country}: {str(e)}. Skipping...")

    save_to_csv(all_trends)
