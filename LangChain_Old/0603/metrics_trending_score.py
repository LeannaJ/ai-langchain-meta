import math
from dateutil import parser as date_parser
from isodate import parse_duration

# YouTube
def calculate_youtube_metrics(row, utc_now=None):
    """
    Calculate derived YouTube video metrics such as velocity, ratios, etc. Accepts a single row dict (e.g., from a DataFrame) and extracts needed fields internally.
    Args:
        row (dict): Row of YouTube video data (flat dict)
        utc_now (datetime, optional): Reference UTC datetime. If None, uses current UTC time.
    Returns:
        dict: Calculated derived metrics only (no direct API or simple parse fields)
    """
    # YouTube
    if utc_now is None:
        import datetime
        utc_now = datetime.datetime.now(datetime.timezone.utc)

    published_at_str = row.get('Published At')
    published_at_dt = date_parser.parse(published_at_str) if published_at_str else utc_now
    age_delta = utc_now - published_at_dt
    age_in_hours = max(1, round(age_delta.total_seconds() / 3600))

    view_count = int(row.get('View Count', 0))
    like_count = int(row.get('Like Count', 0))
    comment_count = int(row.get('Comment Count', 0))

    # Derived metrics only
    view_velocity = round(view_count / age_in_hours, 2)
    like_velocity = round(like_count / age_in_hours, 2)
    comment_velocity = round(comment_count / age_in_hours, 2)
    like_to_view_ratio = round((like_count / view_count) * 100, 2) if view_count > 0 else 0

    return {
        'View Velocity (Views/Hour)': view_velocity,
        'Like Velocity (Likes/Hour)': like_velocity,
        'Comment Velocity (Comments/Hour)': comment_velocity,
        'Like-to-View Ratio (%)': like_to_view_ratio,
    }

# Reddit proxy metrics for cross-platform comparison
def calculate_reddit_proxy_metrics(post):
    """
    Calculate Reddit proxy metrics for cross-platform comparison (Likes Proxy, Views Proxy, Like-to-View Ratio).
    Args:
        post (dict): Reddit post data (one row from API or DataFrame)
    Returns:
        dict: Proxy metrics
    """
    K = 1000  # scaling constant, can be tuned
    score = int(post.get('score', 0))
    num_comments = int(post.get('num_comments', 0))
    total_awards = int(post.get('total_awards_received', 0))
    subscribers = int(post.get('subreddit_subscribers', 1)) or 1  # avoid division by zero

    reddit_likes_proxy = score  # score = upvotes - downvotes
    reddit_views_proxy = (score + num_comments + total_awards) * (K / subscribers)
    reddit_like_view_ratio = reddit_likes_proxy / reddit_views_proxy if reddit_views_proxy else 0

    return {
        'Reddit Likes Proxy': reddit_likes_proxy,
        'Reddit Views Proxy': reddit_views_proxy,
        'Reddit Like-to-View Ratio': reddit_like_view_ratio,
    }

# Reddit
def calculate_reddit_metrics(post, utc_now=None):
    """
    Calculate Reddit post metrics such as velocity, ratios, etc.
    Args:
        post (dict): Reddit post data (one row from API or DataFrame)
        utc_now (datetime, optional): Reference UTC datetime. If None, uses current UTC time.
    Returns:
        dict: Calculated metrics
    """
    if utc_now is None:
        from datetime import datetime, timezone
        utc_now = datetime.now(timezone.utc)

    # Post age in hours
    created_utc = post.get('created_utc')
    if created_utc is not None:
        from datetime import datetime, timezone
        post_time = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
        age_delta = utc_now - post_time
        age_in_hours = max(1, round(age_delta.total_seconds() / 3600))
    else:
        age_in_hours = None

    upvotes = int(post.get('ups', 0))
    score = int(post.get('score', 0))
    num_comments = int(post.get('num_comments', 0))
    view_count = post.get('view_count')
    if view_count is not None:
        view_count = int(view_count)

    # Get proxy metrics
    proxy_metrics = calculate_reddit_proxy_metrics(post)
    reddit_likes_proxy = proxy_metrics['Reddit Likes Proxy']
    reddit_views_proxy = proxy_metrics['Reddit Views Proxy']

    view_velocity = round(reddit_views_proxy / age_in_hours, 2) if age_in_hours else None
    like_velocity = round(reddit_likes_proxy / age_in_hours, 2) if age_in_hours else None
    comment_velocity = round(num_comments / age_in_hours, 2) if age_in_hours else None
    like_to_view_ratio = round((reddit_likes_proxy / reddit_views_proxy) * 100, 2) if reddit_views_proxy else None
    upvote_ratio = float(post.get('upvote_ratio', 0)) * 100  # Reddit gives this as 0~1

    return {
        'Post Age (Hours)': age_in_hours,
        'Upvotes': upvotes,
        'Score': score,
        'Num Comments': num_comments,
        'View Velocity (Views/Hour)': view_velocity,
        'Upvote Velocity (Upvotes/Hour)': like_velocity,
        'Comment Velocity (Comments/Hour)': comment_velocity,
        'Upvote-to-View Ratio (%)': like_to_view_ratio,
        'Upvote Ratio (%)': upvote_ratio,
        # Optionally, include proxy metrics as well:
        **proxy_metrics
    } 