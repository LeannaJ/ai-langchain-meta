import os
import requests
import json
import pandas as pd
from requests.auth import HTTPBasicAuth

# ── 1. Credentials from env vars ───────────────────────────────────────────
CLIENT_ID     = os.environ["REDDIT_CLIENT_ID"]
CLIENT_SECRET = os.environ["REDDIT_CLIENT_SECRET"]
USER_AGENT    = os.environ.get("REDDIT_USER_AGENT", "MyApp/0.1")

# ── 2. Get an OAuth bearer token ──────────────────────────────────────────
auth = HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
data = {"grant_type": "client_credentials"}
headers = {"User-Agent": USER_AGENT}

token_res = requests.post(
    "https://www.reddit.com/api/v1/access_token",
    auth=auth, data=data, headers=headers
)
token_res.raise_for_status()
bearer = token_res.json()["access_token"]

# ── 3. Page through r/all/hot until you have 1 000 posts ─────────────────
hot_url = "https://oauth.reddit.com/r/all/hot"
headers["Authorization"] = f"bearer {bearer}"

all_posts = []
after = None

while len(all_posts) < 1000:
    to_fetch = min(100, 1000 - len(all_posts))
    params = {"limit": to_fetch}
    if after:
        params["after"] = after

    res = requests.get(hot_url, headers=headers, params=params)
    res.raise_for_status()
    children = res.json()["data"].get("children", [])
    if not children:
        break

    all_posts.extend(children)
    after = res.json()["data"].get("after")
    if not after:
        break

all_posts = all_posts[:1000]

# ── 4. Dump to file ────────────────────────────────────────────────────────

# ── 4. Normalize and save to CSV ─────────────────────────────────────────────
# Extract the inner "data" dict from each post
posts_data = [post["data"] for post in all_posts]

# Flatten into a DataFrame
df = pd.json_normalize(posts_data)

# Write out to CSV
output_path = "reddit_hot_posts.csv"
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} posts to {output_path}")
