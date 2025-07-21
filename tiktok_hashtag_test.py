import asyncio
import csv
from playwright.async_api import async_playwright
import os
import requests
from datetime import datetime

# At the top or before return results
def upload_to_supabase(data):
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing Supabase URL or service key in environment variables")

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    # add timestamp for each record
    for item in data:
        item["views"] = int(item["views"])  # convert to number
        item["scraped_at"] = datetime.now().isoformat()

    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/tiktok_hashtags",
        headers=headers,
        json=data
    )

    print("🚀 Supabase upload status:", response.status_code)
    print(response.text)

async def scrape_tiktok_hashtags():
    print("🚀 Activating Browser...")
    results = []
    url = "https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  
        page = await browser.new_page()

        print("🌐 Opening TikTok Creative Center page...")
        await page.goto(url, timeout=60000)
        print("🌐 Verifying if page opened properly...")
        try:
            await page.wait_for_selector("main", timeout=60000)  # or another reliable root element
            print("✅ Page loaded successfully.")
        except Exception as e:
            print(f"❌ Page failed to load: {e}")
            await page.screenshot(path="error_screenshot.png", full_page=True)
            return  # exit early if critical


        # 📸 Take a screenshot of the initial page
        await page.screenshot(path="tiktok_hashtag_landing.png", full_page=True)
        print("📸 Screenshot saved as tiktok_hashtag_landing.png")

        print("⏳ Waiting hashtag Table...")
        await page.wait_for_selector("a[class*='CardPc_container___']", timeout=60000)

        print("🔁 Click『View more』till no more button shows up...")
        while True:
            try:
                load_more_button = page.locator("div.CcButton_common__aFDas.CcButton_secondary__N1HnA")
                if await load_more_button.is_visible():
                    await load_more_button.scroll_into_view_if_needed()
                    await load_more_button.click()
                    print("✅ Clicked『View more』")
                    await asyncio.sleep(2)
                else:
                    print("✅ No more buttons, data loaded complete")
                    break
            except Exception as e:
                print(f"⚠️ Click failed or can't find buttons：{e}")
                break

        print("🔍 Start extracting hashtag cards...")
        row_locator = page.locator("a[class*='CardPc_container___']")
        count = await row_locator.count()
        print(f"📦 Found {count} rows, start extracting...")
        
        for i in range(count):
            row = row_locator.nth(i)
            try:
                title_el = row.locator('span[class^="CardPc_titleText__"]').first
                title_raw = await title_el.inner_text()

                values = row.locator('span[class^="CardPc_itemValue__"]')
                value_count = await values.count()

                view_raw = await values.nth(0).inner_text() if value_count >= 1 else "N/A"

                # processing hashtags data
                hashtag = title_raw.strip().lstrip("#")

                view_clean = view_raw.strip().upper().replace(",", "")
                if "K" in view_clean:
                    views = str(int(float(view_clean.replace("K", "")) * 1000))
                elif "M" in view_clean:
                    views = str(int(float(view_clean.replace("M", "")) * 1_000_000))
                else:
                    views = view_clean  # fallback

                results.append({
                    "rank": i + 1,
                    "hashtag": hashtag,
                    "views": views
                })

                print(f"{i+1}. ✅ {hashtag} | Views: {views}")

            except Exception as e:
                print(f"⚠️ Failed at row {i+1}：{e}")


        await browser.close()

    print(f"\n🎉 Scraping completed. Found {len(results)} hashtag entries.")

    # ✅ Export to CSV
    filename = "tiktok_hashtags.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "hashtag", "views"])
        writer.writeheader()
        writer.writerows(results)

    print(f"📁 Results written to：{filename}")
  
    # ✅ Upload to Supabase instead of CSV
    upload_to_supabase(results)
  
    return results

    

# 🔧 Execution
if __name__ == "__main__":
    asyncio.run(scrape_tiktok_hashtags())
