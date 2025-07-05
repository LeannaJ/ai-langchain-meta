import asyncio
import csv
from playwright.async_api import async_playwright

async def scrape_tiktok_hashtags():
    print("🚀 Activating Browser...")
    results = []
    url = "https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  
        page = await browser.new_page()

        print("🌐 Opening TikTok Creative Center page...")
        await page.goto(url, timeout=90000)

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
                title = await title_el.inner_text()

                values = row.locator('span[class^="CardPc_itemValue__"]')
                value_count = await values.count()

                view = await values.nth(0).inner_text() if value_count >= 1 else "N/A"

                results.append({
                    "rank": i + 1,
                    "hashtag": title.strip(),
                    "views": view.strip()
                })

                print(f"{i+1}. ✅ #{title.strip()} | Views: {view.strip()}")

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
    return results

# 🔧 Execution
if __name__ == "__main__":
    asyncio.run(scrape_tiktok_hashtags())
