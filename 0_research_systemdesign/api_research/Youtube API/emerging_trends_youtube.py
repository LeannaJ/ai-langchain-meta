import csv
import os
import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from isodate import parse_duration
from dateutil import parser as date_parser

# --- CONFIGURATION ---

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# If you want to perform channel-level analysis (adds 1 API call per unique channel)
# This is required for "Subscriber Count", "Channel Published At", and "Breakout Channel" metrics
FETCH_CHANNEL_DATA = True

# A list of region codes to fetch trending videos from.
# Full list: https://gist.github.com/stpe/2951130
REGION_CODES = ['US', 'IN', 'GB', 'CA', 'DE', 'JP', 'BR', 'AU']

# --- MAIN SCRIPT ---

# Build YouTube service client
try:
    youtube = build("youtube", "v3", developerKey=API_KEY)
except Exception as e:
    print(f"Error building YouTube service: {e}")
    exit()

# Cache for channel data to avoid redundant API calls within a single run
channel_data_cache = {}

def get_channel_details(channel_id):
    """
    Fetches channel-specific data like subscriber count and creation date.
    Uses a cache to avoid re-fetching data for the same channel.
    """
    if channel_id in channel_data_cache:
        return channel_data_cache[channel_id]

    try:
        request = youtube.channels().list(
            part="snippet,statistics",
            id=channel_id
        )
        response = request.execute()
        if not response.get('items'):
            return None

        snippet = response['items'][0]['snippet']
        statistics = response['items'][0]['statistics']

        channel_details = {
            'subscriberCount': int(statistics.get('subscriberCount', 0)),
            'channelPublishedAt': snippet.get('publishedAt')
        }
        channel_data_cache[channel_id] = channel_details
        return channel_details
    except HttpError as e:
        print(f"Error fetching channel details for {channel_id}: {e}")
        return None

def main():
    """
    Main function to fetch trending videos and their calculated metrics.
    """
    # Define CSV column headers with the new metrics
    csv_columns = [
        # Core Video Info
        'Video ID', 'Title', 'Channel Title', 'Channel ID', 'Region Code',
        'Published At', 'Video Age (Hours)', 'Duration (Seconds)', 'Is Short',
        'Category ID', 'Tags',
        # Core Stats
        'View Count', 'Like Count', 'Comment Count',
        # Velocity Metrics
        'View Velocity (Views/Hour)', 'Like Velocity (Likes/Hour)', 'Comment Velocity (Comments/Hour)',
        # Ratio Metrics
        'Like-to-View Ratio (%)',
        # Channel Metrics
        'Subscriber Count', 'Channel Published At',
        # Description
        'Description'
    ]

    all_trending_videos = []
    utc_now = datetime.datetime.now(datetime.timezone.utc)

    # Loop through each region
    for region_code in REGION_CODES:
        print(f"Fetching trending videos for region: {region_code}...")
        next_page_token = None

        while True:
            try:
                request = youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    chart='mostPopular',
                    regionCode=region_code,
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()
            except HttpError as e:
                print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
                break # Stop processing this region on error

            for item in response.get('items', []):
                snippet = item['snippet']
                statistics = item.get('statistics', {}) # Use .get for safety
                content_details = item['contentDetails']
                
                # --- Basic Info ---
                video_id = item.get('id')
                channel_id = snippet.get('channelId')
                published_at_str = snippet.get('publishedAt')
                published_at_dt = date_parser.parse(published_at_str)
                
                # --- Calculations ---
                # 1. Video Age
                age_delta = utc_now - published_at_dt
                age_in_hours = max(1, round(age_delta.total_seconds() / 3600)) # Avoid division by zero, min age is 1 hour

                # 2. Counts (handle missing data)
                view_count = int(statistics.get('viewCount', 0))
                like_count = int(statistics.get('likeCount', 0))
                comment_count = int(statistics.get('commentCount', 0))

                # 3. Velocity Metrics
                view_velocity = round(view_count / age_in_hours, 2)
                like_velocity = round(like_count / age_in_hours, 2)
                comment_velocity = round(comment_count / age_in_hours, 2)
                
                # 4. Ratio Metrics
                like_to_view_ratio = round((like_count / view_count) * 100, 2) if view_count > 0 else 0

                # 5. Duration and 'Is Short' Flag
                try:
                    duration_seconds = parse_duration(content_details.get('duration')).total_seconds()
                except Exception:
                    duration_seconds = 0
                is_short = duration_seconds <= 60

                # --- Fetch Channel Data (Optional) ---
                channel_details = {'subscriberCount': None, 'channelPublishedAt': None}
                if FETCH_CHANNEL_DATA and channel_id:
                    details = get_channel_details(channel_id)
                    if details:
                        channel_details = details

                video_details = {
                    'Video ID': video_id,
                    'Title': snippet.get('title'),
                    'Channel Title': snippet.get('channelTitle'),
                    'Channel ID': channel_id,
                    'Region Code': region_code,
                    'Published At': published_at_str,
                    'Video Age (Hours)': age_in_hours,
                    'Duration (Seconds)': int(duration_seconds),
                    'Is Short': is_short,
                    'Category ID': snippet.get('categoryId'),
                    'Tags': ','.join(snippet.get('tags', [])),
                    'View Count': view_count,
                    'Like Count': like_count,
                    'Comment Count': comment_count,
                    'View Velocity (Views/Hour)': view_velocity,
                    'Like Velocity (Likes/Hour)': like_velocity,
                    'Comment Velocity (Comments/Hour)': comment_velocity,
                    'Like-to-View Ratio (%)': like_to_view_ratio,
                    'Subscriber Count': channel_details['subscriberCount'],
                    'Channel Published At': channel_details['channelPublishedAt'],
                    'Description': snippet.get('description'),
                }
                all_trending_videos.append(video_details)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

    # Write combined results to CSV
    today = datetime.date.today().isoformat()
    output_file = f'youtube_trending_analysis_{today}.csv'
    if not all_trending_videos:
        print("No videos found. Exiting.")
        return
        
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_trending_videos)

    print("\n--- Summary ---")
    print(f"Total videos collected: {len(all_trending_videos)}")
    print(f"Output file: {output_file}")
    print("Sample rows with new metrics:")
    for row in all_trending_videos[:3]:
        print(
            f"Title: {row['Title'][:40]}... | "
            f"Region: {row['Region Code']} | "
            f"View Velocity: {row['View Velocity (Views/Hour)']} v/hr | "
            f"Like/View Ratio: {row['Like-to-View Ratio (%)']}%"
        )

if __name__ == '__main__':
    main()