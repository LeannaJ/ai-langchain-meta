import os
import json
from datetime import datetime, timezone
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === Setup Chrome options ===
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

# === Start Chrome and load Trends24 US page ===
driver = webdriver.Chrome(options=options)
driver.get("https://trends24.in/united-states/")

# === Wait for trend-card list to load ===
try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".trend-card__list li a"))
    )
except:
    print("❌ Failed to load trend elements.")
    driver.quit()
    exit()

# === Extract trending hashtags ===
hashtags = set()
elements = driver.find_elements(By.CSS_SELECTOR, ".trend-card__list li a")
for elem in elements:
    text = elem.text.strip()
    if text.startswith("#"):
        hashtags.add(text)

driver.quit()

# === Save results ===
hashtag_list = sorted(hashtags)
output_data = {
    "source": "trends24.in",
    "region": "united-states",
    "date_collected": datetime.now(timezone.utc).isoformat(),
    "total_hashtags": len(hashtag_list),
    "hashtags": hashtag_list
}

# Save to relative data path
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "trends24_us_hashtags.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ Saved {len(hashtag_list)} hashtags to:\n{output_path}")
