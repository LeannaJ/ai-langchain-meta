#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain Workflow for Multi-Platform Trend Analysis
A comprehensive pipeline for collecting, processing, and analyzing trending content
from YouTube and Reddit using LDA topic modeling and LangChain tools.
Updated version with topic grouping functionality.
"""

import pandas as pd
import numpy as np
import glob
import os
from metrics_trending_score import calculate_youtube_metrics, calculate_reddit_metrics, calculate_reddit_proxy_metrics
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.utils import simple_preprocess
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import defaultdict
from difflib import SequenceMatcher
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# STEP 1: Data Collection Tools
def get_latest_csv(pattern):
    """Get the most recent CSV file matching the pattern"""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def fetch_youtube_popular():
    """Fetch latest YouTube trending analysis data with metrics"""
    latest = get_latest_csv("Scraped_Data/youtube_trending_analysis_*.csv")
    return pd.read_csv(latest)

def fetch_reddit_hot():
    """Fetch latest Reddit trend analysis data with metrics"""
    latest = get_latest_csv("Scraped_Data/reddit_hot_posts*.csv")
    return pd.read_csv(latest)

def fetch_tiktok_hashtags():
    """Fetch latest TikTok hashtags data"""
    latest = get_latest_csv("Scraped_Data/tiktok_hashtags*.csv")
    return pd.read_csv(latest)

def fetch_twitter_trends():
    """Fetch latest Twitter trends data"""
    latest = get_latest_csv("Scraped_Data/trends_output*.csv")
    return pd.read_csv(latest)

def fetch_google_trends():
    """Fetch latest Google Trends rising data"""
    latest = get_latest_csv("Scraped_Data/trend_rising_*.csv")
    return pd.read_csv(latest)

youtube_tool = Tool.from_function(
    name="FetchYouTubeData",
    func=fetch_youtube_popular,
    description="Fetch latest YouTube trending videos data with calculated metrics."
)

reddit_tool = Tool.from_function(
    name="FetchRedditData",
    func=fetch_reddit_hot,
    description="Fetch latest Reddit hot posts data with calculated metrics."
)

tiktok_tool = Tool.from_function(
    name="FetchTikTokData",
    func=fetch_tiktok_hashtags,
    description="Fetch latest TikTok hashtags data with view counts."
)

twitter_tool = Tool.from_function(
    name="FetchTwitterData",
    func=fetch_twitter_trends,
    description="Fetch latest Twitter trends data from multiple regions."
)

google_trends_tool = Tool.from_function(
    name="FetchGoogleTrendsData",
    func=fetch_google_trends,
    description="Fetch latest Google Trends rising terms data with spread intensity scores."
)

# STEP 2: Data Preprocessing
def preprocess_platform_data(platform_data_dict):
    """
    Preprocess raw data from all platforms to ensure consistent data types and formats
    Args:
        platform_data_dict: dict with platform names as keys and DataFrames as values
    Returns:
        dict with preprocessed DataFrames
    """
    preprocessed_data = {}
    
    for platform, df in platform_data_dict.items():
        if df is None or df.empty:
            print(f"Warning: {platform} data is empty or None")
            preprocessed_data[platform] = df
            continue
            
        print(f"Preprocessing {platform} data...")
        
        if platform == 'twitter':
            # Twitter specific preprocessing
            df_processed = df.copy()
            
            # Clean tweet_count: remove commas and convert to numeric
            if 'tweet_count' in df_processed.columns:
                df_processed['tweet_count'] = (
                    df_processed['tweet_count']
                    .astype(str)
                    .str.replace(',', '')
                    .str.replace('"', '')  # Remove quotes
                    .str.replace("'", '')  # Remove single quotes
                    .replace('', np.nan)   # Empty strings to NaN
                    .astype(float)
                )
                print(f"  - tweet_count: {df_processed['tweet_count'].notna().sum()} valid values")
            
            # Clean duration: extract hours and convert to numeric
            if 'duration' in df_processed.columns:
                # Extract numeric part from duration strings like "10 hrs", "8 hrs"
                df_processed['duration'] = (
                    df_processed['duration']
                    .astype(str)
                    .str.extract('(\d+)')  # Extract digits
                    .astype(float)
                )
                print(f"  - duration: {df_processed['duration'].notna().sum()} valid values")
            
            # Fill NaN values with reasonable defaults
            df_processed['tweet_count'] = df_processed['tweet_count'].fillna(0)
            df_processed['duration'] = df_processed['duration'].fillna(1)  # Default 1 hour
            
            preprocessed_data[platform] = df_processed
            
        elif platform == 'youtube':
            # YouTube data is already well-structured from metrics calculation
            preprocessed_data[platform] = df
            
        elif platform == 'reddit':
            # Reddit data is already well-structured from metrics calculation
            preprocessed_data[platform] = df
            
        elif platform == 'tiktok':
            # TikTok data is already well-structured
            preprocessed_data[platform] = df
            
        elif platform == 'google_trends':
            # Google Trends data is already well-structured
            preprocessed_data[platform] = df
            
        else:
            print(f"Warning: Unknown platform {platform}, skipping preprocessing")
            preprocessed_data[platform] = df
    
    return preprocessed_data

preprocess_data_tool = Tool.from_function(
    name="PreprocessPlatformData",
    func=preprocess_platform_data,
    description="Preprocess raw data from all platforms to ensure consistent data types and formats."
)

# STEP 3: LDA Topic Modeling for YouTube and Reddit 
# 3.1: LDA Preprocessing
def initialize_nltk_components():
    """Initialize NLTK components for LDA preprocessing"""
    # Download required NLTK data
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    return stop_words, lemmatizer

def preprocess_text(df, text_cols):
    """
    LDA-style preprocessing: lowercase, remove special chars, normalize whitespace, 
    remove stopwords, tokenize, lemmatize, and filter out meaningless words
    """
    # Initialize NLTK components
    stop_words, lemmatizer = initialize_nltk_components()
    
    # Extended stopwords and meaningless words
    extended_stopwords = {
        'http', 'https', 'www', 'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'html', 'php', 'asp', 'jsp', 'xml', 'css', 'js', 'json', 'api', 'url',
        'bit', 'ly', 'goo', 'gl', 'tiny', 'url', 'short', 'link', 'redirect',
        'click', 'visit', 'watch', 'subscribe', 'follow', 'like', 'share',
        'video', 'channel', 'playlist', 'upload', 'download', 'stream',
        'live', 'online', 'web', 'site', 'page', 'home', 'main', 'index',
        'search', 'find', 'get', 'go', 'see', 'look', 'check', 'view',
        'new', 'latest', 'recent', 'update', 'news', 'info', 'data',
        'free', 'download', 'buy', 'sell', 'shop', 'store', 'market',
        'help', 'support', 'contact', 'about', 'terms', 'privacy', 'policy'
    }
    
    # Combine with existing stopwords
    all_stopwords = stop_words.union(extended_stopwords)
    
    def preprocess(text):
        if pd.isna(text):
            return []
        # simple_preprocess: lowercases, strips accents, tokenizes on word boundaries, removes tokens <2 or >15 chars
        tokens = simple_preprocess(text, deacc=True)
        # Filter out stopwords, meaningless words, and very short/long tokens
        filtered_tokens = []
        for t in tokens:
            t_lemma = lemmatizer.lemmatize(t)
            if (t_lemma not in all_stopwords and 
                len(t_lemma) >= 3 and len(t_lemma) <= 20 and
                not t_lemma.isdigit() and  # Remove pure numbers
                not all(c.isdigit() or c == '.' for c in t_lemma)):  # Remove version numbers like "1.2.3"
                filtered_tokens.append(t_lemma)
        return filtered_tokens
    
    processed_texts = []
    for _, row in df.iterrows():
        combined_text = " ".join([str(row[col]) for col in text_cols if pd.notna(row[col])])
        processed_texts.append(preprocess(combined_text))
    
    return processed_texts

preprocess_tool = Tool.from_function(
    name="PreprocessText",
    func=lambda df: preprocess_text(df, ["Title", "Description"]),
    description="Preprocess text columns: lowercase, remove special chars, normalize whitespace, remove stopwords, tokenize, lemmatize."
)

# 3.2: LDA Topic Modeling
def topic_modeling(texts, num_topics=200):
    """
    LDA Topic Modeling using gensim
    Args:
        texts: list of text strings
        num_topics: number of topics to extract
    Returns:
        topics: topic assignments for each document
        topic_model: trained LDA model
        topic_labels: list of topic labels
        topic_keywords: list of keyword lists
        topic_keywords_str: list of comma-joined keyword strings
    """
    # Filter out empty documents
    non_empty_texts = [doc for doc in texts if doc]
    if not non_empty_texts:
        raise ValueError("No non-empty documents found for topic modeling")
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(non_empty_texts)
    corpus = [dictionary.doc2bow(doc) for doc in non_empty_texts]
    
    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=30,
        iterations=100,
        alpha='auto',
        per_word_topics=True
    )
    
    # Get topic assignments
    topics = []
    for doc in corpus:
        doc_topics = lda_model.get_document_topics(doc)
        if doc_topics:  # Check if document has topics
            topics.append(max(doc_topics, key=lambda x: x[1])[0])
        else:
            topics.append(0)  # Default topic for empty documents
    
    # Extract topic keywords
    topic_keywords = []
    topic_keywords_str = []
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=10)
        keywords = [word for word, _ in topic_words]
        topic_keywords.append(keywords)
        topic_keywords_str.append(", ".join(keywords))
    
    # Generate topic labels (simple keyword-based)
    topic_labels = []
    for keywords in topic_keywords:
        if keywords:
            label = keywords[0].title()
        else:
            label = "General"
        topic_labels.append(label)
    
    return topics, lda_model, topic_labels, topic_keywords, topic_keywords_str

topic_modeling_tool = Tool.from_function(
    name="TopicModeling",
    func=topic_modeling,
    description="Perform LDA topic modeling on preprocessed texts and return topics, model, labels, and keywords"
)

# 3.3: LLM-based Topic Labeling
def create_topic_labeling_chain():
    """Create LLM chain for natural topic labeling"""
    llm = OpenAI(temperature=0.3)  # Lower temperature for more consistent labels
    
    prompt_template = PromptTemplate(
        input_variables=["keywords", "topic_count"],
        template="""
        Given the following keywords from a topic cluster and the number of documents in this topic, 
        create a short, descriptive, and natural topic label (2-4 words maximum).
        
        Keywords: {keywords}
        Number of documents: {topic_count}
        
        Create a natural, descriptive label that captures the main theme:
        """
    )
    
    return LLMChain(llm=llm, prompt=prompt_template)

def label_topics_with_llm(topic_keywords_str, topic_counts, api_key=None):
    """
    Label topics using LLM for more natural descriptions
    Args:
        topic_keywords_str: list of comma-joined keyword strings
        topic_counts: list of document counts for each topic
        api_key: OpenAI API key (optional, can be set via environment)
    Returns:
        list of natural topic labels
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        chain = create_topic_labeling_chain()
        labels = []
        
        for keywords, count in zip(topic_keywords_str, topic_counts):
            try:
                result = chain.run(keywords=keywords, topic_count=count)
                # Clean up the label
                label = result.strip().replace('"', '').replace("'", "")
                labels.append(label)
            except Exception as e:
                print(f"Error labeling topic with keywords '{keywords}': {e}")
                # Fallback to first keyword
                first_keyword = keywords.split(',')[0].strip() if keywords else "General"
                labels.append(first_keyword.title())
        
        return labels
    except Exception as e:
        print(f"LLM labeling failed: {e}")
        # Fallback to keyword-based labels
        return [keywords.split(',')[0].strip().title() if keywords else "General" 
                for keywords in topic_keywords_str]

topic_labeling_tool_func = label_topics_with_llm
topic_labeling_tool = Tool.from_function(
    name="TopicLabeling",
    func=topic_labeling_tool_func,
    description="Label topics using LLM for natural descriptions"
)

def create_topic_dataframe(topics, topic_labels, topic_keywords_str, original_df, platform):
    """
    Convert topic modeling results to DataFrame format for Step 4
    Args:
        topics: topic assignments for each document
        topic_labels: list of topic labels
        topic_keywords_str: list of comma-joined keyword strings
        original_df: original DataFrame with engagement metrics
        platform: 'youtube' or 'reddit'
    Returns:
        DataFrame with columns: topic, video_count/doc_count, total_engagement
    """
    # Count documents per topic
    topic_counts = {}
    for topic_id in topics:
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
    
    # Create topic DataFrame
    topic_data = []
    for topic_id in range(len(topic_labels)):
        if topic_id in topic_counts:
            topic_data.append({
                'topic': topic_labels[topic_id],
                'topic_keywords': topic_keywords_str[topic_id],
                'topic_count': topic_counts[topic_id]
            })
    
    topic_df = pd.DataFrame(topic_data)
    
    if platform == 'youtube':
        # Calculate total_engagement for each topic
        topic_engagement = {}
        for topic_id, topic_label in enumerate(topic_labels):
            # Get documents belonging to this topic
            topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
            
            if topic_docs:
                # Calculate total engagement for this topic
                total_views = original_df.iloc[topic_docs]['View Count'].sum()
                total_likes = original_df.iloc[topic_docs]['Like Count'].sum()
                total_comments = original_df.iloc[topic_docs]['Comment Count'].sum()
                total_engagement = total_views + total_likes + total_comments
                
                topic_engagement[topic_label] = total_engagement
        
        # Add engagement to topic DataFrame
        topic_df['total_engagement'] = topic_df['topic'].map(topic_engagement)
        topic_df['video_count'] = topic_df['topic_count']
        
        return topic_df[['topic', 'video_count', 'total_engagement']]
    
    elif platform == 'reddit':
        # Calculate total_engagement for each topic
        topic_engagement = {}
        for topic_id, topic_label in enumerate(topic_labels):
            # Get documents belonging to this topic
            topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
            
            if topic_docs:
                # Calculate total engagement for this topic
                total_ups = original_df.iloc[topic_docs]['ups'].sum()
                total_comments = original_df.iloc[topic_docs]['num_comments'].sum()
                total_views = original_df.iloc[topic_docs]['view_count'].sum()
                total_engagement = total_ups + total_comments + total_views
                
                topic_engagement[topic_label] = total_engagement
        
        # Add engagement to topic DataFrame
        topic_df['total_engagement'] = topic_df['total_engagement'] = topic_df['topic'].map(topic_engagement)
        topic_df['doc_count'] = topic_df['topic_count']
        
        return topic_df[['topic', 'doc_count', 'total_engagement']]
    
    else:
        raise ValueError(f"Unsupported platform: {platform}")



# STEP 4: Topic Level aggregation of each platform
def extract_platform_metrics(df, platform):
    """
    Extract and standardize metrics for each platform
    Args:
        df: DataFrame with platform-specific data
        platform: platform name ('youtube', 'reddit', 'tiktok', 'twitter', 'google_trends')
    Returns:
        DataFrame with standardized columns: keyword, frequency, engagement, platform
    """
    if platform == 'youtube':
        # YouTube: topic, video_count, total_engagement (from topic extraction)
        return df[['topic', 'video_count', 'total_engagement']].rename(columns={
            'topic': 'keyword',
            'video_count': 'frequency',
            'total_engagement': 'engagement'
        }).assign(platform='YouTube')
    
    elif platform == 'reddit':
        # Reddit: topic, doc_count, total_engagement (from topic extraction)
        return df[['topic', 'doc_count', 'total_engagement']].rename(columns={
            'topic': 'keyword',
            'doc_count': 'frequency',
            'total_engagement': 'engagement'
        }).assign(platform='Reddit')
    
    elif platform == 'tiktok':
        # TikTok: hashtag, views (direct trending data)
        return df[['hashtag', 'views']].rename(columns={
            'hashtag': 'keyword',
            'views': 'frequency'
        }).assign(
            engagement=lambda x: x['frequency'],
            platform='TikTok'
        )
    
    elif platform == 'twitter':
        # Twitter: trend, duration, tweet_count (direct trending data)
        # duration as frequency (trend duration), tweet_count as engagement (user participation)
        return df[['trend', 'duration', 'tweet_count']].rename(columns={
            'trend': 'keyword',
            'duration': 'frequency',
            'tweet_count': 'engagement'
        }).assign(platform='X/Twitter')
    
    elif platform == 'google_trends':
        # Google Trends: term, median_gain (direct trending data)
        return df[['term', 'median_gain']].rename(columns={
            'term': 'keyword',
            'median_gain': 'frequency'
        }).assign(
            engagement=lambda x: x['frequency'],
            platform='Google Trends'
        )
    
    else:
        raise ValueError(f"Unsupported platform: {platform}")

def consolidate_platform_data(platform_data_dict):
    """
    Consolidate data from all platforms into one DataFrame
    Args:
        platform_data_dict: dict with platform names as keys and DataFrames as values
    Returns:
        Consolidated DataFrame with standardized columns
    """
    consolidated_data = []
    
    for platform, df in platform_data_dict.items():
        if df is not None and not df.empty:
            try:
                platform_metrics = extract_platform_metrics(df, platform)
                consolidated_data.append(platform_metrics)
            except Exception as e:
                print(f"Error processing {platform}: {e}")
                continue
    
    if not consolidated_data:
        raise ValueError("No valid platform data found")
    
    return pd.concat(consolidated_data, ignore_index=True)

# STEP 5: MinMax Scaling
def minmax_normalize(series):
    """Min-max normalization helper"""
    return (series - series.min()) / (series.max() - series.min())

def apply_minmax_scaling(df):
    """
    Apply min-max normalization per platform
    Args:
        df: Consolidated DataFrame with frequency and engagement columns
    Returns:
        DataFrame with normalized columns
    """
    df = df.copy()
    
    # Ensure metrics are numeric
    df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce')
    df['engagement'] = pd.to_numeric(df['engagement'], errors='coerce')
    
    # Apply normalization per platform
    df['frequency_norm'] = df.groupby('platform')['frequency'].transform(minmax_normalize)
    df['engagement_norm'] = df.groupby('platform')['engagement'].transform(minmax_normalize)
    
    return df

# STEP 6: Calculate Trend Scores (Media Type Specific)
def get_platform_weights(media_type='video'):
    """
    Get platform weights based on content type
    Args:
        media_type: 'text', 'video', 'image', or 'equal' for equal weights
    Returns:
        dict of platform weights
    """
    weights = {
        'equal': {
            'Reddit': 0.2,
            'X/Twitter': 0.2,
            'YouTube': 0.2,
            'TikTok': 0.2,
            'Google Trends': 0.2
        },
        'text': {
            'Reddit': 0.4,
            'X/Twitter': 0.4,
            'YouTube': 0.1,
            'TikTok': 0.1,
            'Google Trends': 0.0
        },
        'video': {
            'YouTube': 0.5,
            'TikTok': 0.4,
            'Reddit': 0.05,
            'X/Twitter': 0.05,
            'Google Trends': 0.0
        },
        'image': {
            'TikTok': 0.4,
            'Reddit': 0.3,
            'X/Twitter': 0.3,
            'YouTube': 0.0,
            'Google Trends': 0.0
        }
    }
    
    return weights.get(media_type, weights['video'])

def calculate_trend_scores(df, media_type='video', alpha=0.4, beta=0.4):
    """
    Calculate weighted trend scores
    Args:
        df: DataFrame with normalized metrics
        media_type: content type for weight selection
        alpha: weight for frequency (default 0.4)
        beta: weight for engagement (default 0.4)
    Returns:
        DataFrame with trend scores
    """
    df = df.copy()
    
    # Get platform weights
    platform_weights = get_platform_weights(media_type)
    
    # Apply platform weights
    df['platform_weight'] = df['platform'].map(platform_weights).fillna(0)
    
    # Calculate weighted trend score per platform
    df['trend_score'] = (
        (df['frequency_norm'] + df['engagement_norm']) / 2
    ) * df['platform_weight']
    
    # Calculate per-platform score S_p = w_p * (α·F_p + β·E_p)
    df['S_p'] = df['platform_weight'] * (
        alpha * df['frequency_norm'] + beta * df['engagement_norm']
    )
    
    # Aggregate across platforms to get final TrendScore per keyword
    final_scores = df.groupby('keyword', as_index=False)['S_p'].sum().rename(columns={'S_p': 'TrendScore'})
    final_scores = final_scores.sort_values('TrendScore', ascending=False)
    
    return df, final_scores

def save_consolidated_scores(df_with_scores, media_type='equal', workflow_timestamp=None, output_filename=None):
    """
    Save consolidated scores (before cross-platform bonus) to CSV file
    Args:
        df_with_scores: DataFrame with trend scores calculated
        media_type: content type for filename ('equal', 'video', 'text', 'image')
        workflow_timestamp: timestamp for consistent file naming
        output_filename: optional filename, will auto-generate if None
    Returns:
        filepath of saved CSV
    """
    if output_filename is None:
        if workflow_timestamp:
            if media_type == 'equal':
                output_filename = f"consolidated_scores_{workflow_timestamp}.csv"
            else:
                output_filename = f"consolidated_scores_{media_type}_{workflow_timestamp}.csv"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            if media_type == 'equal':
                output_filename = f"consolidated_scores_{timestamp}.csv"
            else:
                output_filename = f"consolidated_scores_{media_type}_{timestamp}.csv"
    
    # Create output directory
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    
    # Save to CSV
    df_with_scores.to_csv(filepath, index=False)
    print(f"Consolidated scores exported to '{filepath}'")
    print(f"Total records: {len(df_with_scores)}")
    
    return filepath

# STEP 7: Topic Clustering & Similarity Analysis (Common Processing)
def calculate_keyword_similarity(keywords1, keywords2):
    """
    Calculate similarity between two keyword strings using Jaccard and String similarity
    Args:
        keywords1: first keyword string
        keywords2: second keyword string
    Returns:
        similarity score (0-1)
    """
    # Convert to sets of words
    set1 = set(keywords1.lower().split())
    set2 = set(keywords2.lower().split())
    
    # Jaccard similarity
    if len(set1.union(set2)) == 0:
        jaccard_sim = 0
    else:
        jaccard_sim = len(set1.intersection(set2)) / len(set1.union(set2))
    
    # String similarity using SequenceMatcher
    string_sim = SequenceMatcher(None, keywords1.lower(), keywords2.lower()).ratio()
    
    # Combined similarity (weighted average)
    combined_sim = 0.6 * jaccard_sim + 0.4 * string_sim
    
    return combined_sim

def cluster_similar_topics(file_path, lambda_val=0.1, similarity_threshold=0.6):
    """
    Cluster similar topics and calculate platform boost scores
    
    Parameters:
    - file_path: path to CSV file
    - lambda_val: lambda value for platform boost calculation
    - similarity_threshold: threshold for considering topics similar (0-1)

    Returns:
    - List of dictionaries with clustered results
    - DataFrame with original data + cross-platform bonus and final scores
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Create a copy for final output
    final_df = df.copy()
    
    # Use the 'keyword' column for clustering
    topics = df['keyword'].tolist()
    groups = []
    assigned = [False] * len(topics)

    # Group similar topics
    for i, topic in enumerate(topics):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True

        # Find similar topics using the more sophisticated similarity calculation
        for j in range(i+1, len(topics)):
            if not assigned[j]:
                # Use calculate_keyword_similarity for better accuracy
                similarity = calculate_keyword_similarity(topic, topics[j])
                if similarity >= similarity_threshold:
                    group.append(j)
                    assigned[j] = True
        groups.append(group)

    # Aggregate scores and platforms for each group
    results = []
    group_counter = 0
    
    for group in groups:
        group_keywords = [topics[idx] for idx in group]
        group_df = df.iloc[group]

        # Sum the S_p scores
        sum_sp = group_df['S_p'].sum()

        # Get unique platforms
        platforms = set(group_df['platform'])

        # Calculate platform boost: λ × (number of platforms - 1)
        platform_boost = lambda_val * (len(platforms) - 1)

        # Final score = Sum of S_p + Platform Boost
        final_score = sum_sp + platform_boost

        # Add cross-platform bonus and final score to the original dataframe
        for idx in group:
            final_df.loc[idx, 'cross_platform_bonus'] = platform_boost
            final_df.loc[idx, 'final_trend_score'] = final_score
            final_df.loc[idx, 'group_id'] = group_counter
            final_df.loc[idx, 'group_name'] = group_keywords[0]

        results.append({
            'group_name': group_keywords[0],  # Use first keyword as group name
            'group_keywords': group_keywords,
            'keyword_count': len(group_keywords),
            'sum_S_p': sum_sp,
            'platforms': list(platforms),
            'platform_count': len(platforms),
            'platform_boost': platform_boost,
            'final_score': final_score
        })
        
        group_counter += 1

    # Fill NaN values for ungrouped items (if any)
    final_df['cross_platform_bonus'] = final_df['cross_platform_bonus'].fillna(0)
    final_df['final_trend_score'] = final_df['final_trend_score'].fillna(final_df['S_p'])
    final_df['group_id'] = final_df['group_id'].fillna(-1)  # -1 for ungrouped
    final_df['group_name'] = final_df['group_name'].fillna(final_df['keyword'])

    return results, final_df

def save_final_trend_scores(final_df, media_type='equal', workflow_timestamp=None, output_filename=None):
    """
    Save final trend scores with cross-platform bonus to CSV file
    Args:
        final_df: DataFrame with original data + cross-platform bonus and final scores
        media_type: content type for filename ('equal', 'video', 'text', 'image')
        workflow_timestamp: timestamp for consistent file naming
        output_filename: optional filename, will auto-generate if None
    Returns:
        filepath of saved CSV
    """
    if output_filename is None:
        if workflow_timestamp:
            if media_type == 'equal':
                output_filename = f"consolidated_scores_w_crossbonus_{workflow_timestamp}.csv"
            else:
                output_filename = f"consolidated_scores_w_crossbonus_{media_type}_{workflow_timestamp}.csv"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            if media_type == 'equal':
                output_filename = f"consolidated_scores_w_crossbonus_{timestamp}.csv"
            else:
                output_filename = f"consolidated_scores_w_crossbonus_{media_type}_{timestamp}.csv"
    
    # Create output directory
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    
    # Save to CSV
    final_df.to_csv(filepath, index=False)
    print(f"Final trend scores exported to '{filepath}'")
    print(f"Total records: {len(final_df)}")
    
    return filepath

# STEP 8: Final Grouped Results Generation (Common Processing)
def save_grouped_results(results, media_type='equal', workflow_timestamp=None, output_filename=None):
    """
    Save grouped results to CSV file
    Args:
        results: list of grouped topic dictionaries
        media_type: content type for filename ('equal', 'video', 'text', 'image')
        workflow_timestamp: timestamp for consistent file naming
        output_filename: optional filename, will auto-generate if None
    Returns:
        filepath of saved CSV
    """
    if output_filename is None:
        if workflow_timestamp:
            if media_type == 'equal':
                output_filename = f"final_grouped_trend_scores_{workflow_timestamp}.csv"
            else:
                output_filename = f"final_grouped_trend_scores_{media_type}_{workflow_timestamp}.csv"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            if media_type == 'equal':
                output_filename = f"final_grouped_trend_scores_{timestamp}.csv"
            else:
                output_filename = f"final_grouped_trend_scores_{media_type}_{timestamp}.csv"
    
    # Create output directory
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Grouped results exported to '{filepath}'")
    print(f"Total groups created: {len(df)}")
    
    return filepath

def generate_final_output(clustered_results, media_type='equal', workflow_timestamp=None, output_filename=None):
    """
    Generate final grouped output with summary statistics
    Args:
        clustered_results: list of clustered topic dictionaries
        media_type: content type for filename ('equal', 'video', 'text', 'image')
        workflow_timestamp: timestamp for consistent file naming
        output_filename: optional filename, will auto-generate if None
    Returns:
        dict with filepath and summary statistics
    """
    # Save grouped results
    filepath = save_grouped_results(clustered_results, media_type, workflow_timestamp, output_filename)
    
    # Calculate summary statistics
    total_groups = len(clustered_results)
    total_keywords = sum(result['keyword_count'] for result in clustered_results)
    avg_platform_count = sum(result['platform_count'] for result in clustered_results) / total_groups if total_groups > 0 else 0
    
    # Get top 10 results
    top_results = sorted(clustered_results, key=lambda x: x['final_score'], reverse=True)[:10]
    
    # Print summary
    print(f"\n=== Top 10 Grouped Topics ===")
    for i, result in enumerate(top_results, 1):
        print(f"{i}. {result['group_name']}")
        print(f"   Keywords: {result['keyword_count']}")
        print(f"   Platforms: {', '.join(result['platforms'])}")
        print(f"   Final Score: {result['final_score']:.4f}\n")
    
    print(f"=== Summary ===")
    print(f"Total topic groups: {total_groups}")
    print(f"Total keywords processed: {total_keywords}")
    print(f"Average platforms per group: {avg_platform_count:.2f}")
    print(f"Results saved to: {filepath}")
    
    return {
        'filepath': filepath,
        'total_groups': total_groups,
        'total_keywords': total_keywords,
        'avg_platform_count': avg_platform_count,
        'top_results': top_results
    }

# STEP 9: Category Classification (Common Processing)
def create_category_classification_chain():
    """Create LLM chain for topic category classification"""
    llm = OpenAI(temperature=0.1)  # Low temperature for consistent categorization
    
    prompt_template = PromptTemplate(
        input_variables=["topic", "categories"],
        template="""
        Classify the following topic into the most appropriate category from the given options.
        
        Topic: {topic}
        
        Available categories:
        {categories}
        
        Instructions:
        - Choose the SINGLE most appropriate category
        - Consider the main theme and context of the topic
        - If a topic could fit multiple categories, choose the most dominant one
        - Respond with ONLY the category name, nothing else
        
        Category: """
    )
    
    return LLMChain(llm=llm, prompt=prompt_template)

def classify_topic_category(topic, api_key=None):
    """
    Classify a single topic into a predefined category using LLM
    Args:
        topic: topic keyword/label to classify
        api_key: OpenAI API key (optional, can be set via environment)
    Returns:
        category: classified category name
    """
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    try:
        # Define categories
        categories = [
            "Beauty & Fashion",
            "Technology & Innovation", 
            "Lifestyle & Health",
            "News & Politics",
            "Sports & Fitness",
            "Education & Learning",
            "Business & Finance",
            "Entertainment & Media"
        ]
        
        # Create category list for prompt
        categories_text = "\n".join([f"- {cat}" for cat in categories])
        
        # Create and run classification chain
        chain = create_category_classification_chain()
        result = chain.run(topic=topic, categories=categories_text)
        
        # Clean and validate result
        category = result.strip()
        
        # Validate that result is one of the predefined categories
        if category not in categories:
            # If LLM returned something unexpected, try to match closest category
            category = match_closest_category(category, categories)
        
        return category
        
    except Exception as e:
        print(f"Error classifying topic '{topic}': {str(e)}")
        return "Entertainment & Media"  # Default fallback category

def match_closest_category(llm_result, categories):
    """
    Match LLM result to closest predefined category using similarity
    Args:
        llm_result: result from LLM
        categories: list of predefined categories
    Returns:
        closest_category: best matching category
    """
    from difflib import SequenceMatcher
    
    best_match = "Entertainment & Media"  # Default
    best_score = 0
    
    for category in categories:
        # Calculate similarity between LLM result and category
        similarity = SequenceMatcher(None, llm_result.lower(), category.lower()).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = category
    
    return best_match

def batch_classify_topics(topics_list, api_key=None, batch_size=10):
    """
    Classify multiple topics in batches to optimize API usage
    Args:
        topics_list: list of topic keywords/labels
        api_key: OpenAI API key
        batch_size: number of topics to process in each batch
    Returns:
        dict: mapping of topic to category
    """
    topic_to_category = {}
    
    # Process in batches
    for i in range(0, len(topics_list), batch_size):
        batch = topics_list[i:i + batch_size]
        print(f"  Classifying batch {i//batch_size + 1}/{(len(topics_list) + batch_size - 1)//batch_size} ({len(batch)} topics)")
        
        for topic in batch:
            category = classify_topic_category(topic, api_key)
            topic_to_category[topic] = category
    
    return topic_to_category

def add_category_column(final_results_filepath, api_key=None):
    """
    Add category column to final grouped results
    Args:
        final_results_filepath: path to final grouped results CSV
        api_key: OpenAI API key
    Returns:
        filepath: path to updated CSV with category column
    """
    print(f"Step 9: Adding category classification to {final_results_filepath}")
    
    # Load final results
    df = pd.read_csv(final_results_filepath)
    
    # Get unique topic names (group_name column)
    unique_topics = df['group_name'].unique().tolist()
    print(f"  Found {len(unique_topics)} unique topics to classify")
    
    # Classify all unique topics
    topic_to_category = batch_classify_topics(unique_topics, api_key)
    
    # Add category column to dataframe
    df['category'] = df['group_name'].map(topic_to_category)
    
    # Fill any missing categories with default
    df['category'] = df['category'].fillna("Entertainment & Media")
    
    # Save updated results
    output_filepath = final_results_filepath.replace('.csv', '_with_categories.csv')
    df.to_csv(output_filepath, index=False)
    
    print(f"  ✓ Category classification completed")
    print(f"  ✓ Updated results saved to: {output_filepath}")
    
    # Print category distribution
    category_counts = df['category'].value_counts()
    print(f"\n  Category Distribution:")
    for category, count in category_counts.items():
        print(f"    {category}: {count} topics")
    
    return output_filepath

def save_categorized_results(df, media_type='equal', workflow_timestamp=None, output_filename=None):
    """
    Save final results with category classification to CSV file
    Args:
        df: DataFrame with category column added
        media_type: content type for filename ('equal', 'video', 'text', 'image')
        workflow_timestamp: timestamp for consistent file naming
        output_filename: optional filename, will auto-generate if None
    Returns:
        filepath of saved CSV
    """
    if output_filename is None:
        if workflow_timestamp:
            if media_type == 'equal':
                output_filename = f"final_grouped_trend_scores_with_categories_{workflow_timestamp}.csv"
            else:
                output_filename = f"final_grouped_trend_scores_with_categories_{media_type}_{workflow_timestamp}.csv"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            if media_type == 'equal':
                output_filename = f"final_grouped_trend_scores_with_categories_{timestamp}.csv"
            else:
                output_filename = f"final_grouped_trend_scores_with_categories_{media_type}_{timestamp}.csv"
    
    # Create output directory
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Final categorized results exported to '{filepath}'")
    print(f"Total records: {len(df)}")
    
    return filepath

# STEP 10: Media Type Specific Final Score Calculation
def calculate_final_scores_by_media_type(categorized_results_filepath, media_type, workflow_timestamp):
    """
    Calculate final scores for specific media type using the categorized results
    Args:
        categorized_results_filepath: path to categorized results CSV
        media_type: media type for weight calculation
        workflow_timestamp: timestamp for file naming
    Returns:
        filepath: path to final media-type specific results
    """
    print(f"Step 10: Calculating final scores for {media_type} media type")
    
    # Load categorized results
    df = pd.read_csv(categorized_results_filepath)
    
    # Get platform weights for this media type
    platform_weights = get_platform_weights(media_type)
    
    # Recalculate S_p scores with media-specific weights
    df['platform_weight'] = df['platform'].map(platform_weights).fillna(0)
    df['S_p'] = df['platform_weight'] * (
        0.4 * df['frequency_norm'] + 0.4 * df['engagement_norm']
    )
    
    # Recalculate final trend score (S_p + cross_platform_bonus)
    df['final_trend_score'] = df['S_p'] + df['cross_platform_bonus']
    
    # Save media-type specific results
    if media_type == 'equal':
        output_filename = f"final_categorized_scores_{workflow_timestamp}.csv"
    else:
        output_filename = f"final_categorized_scores_{media_type}_{workflow_timestamp}.csv"
    
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    
    df.to_csv(filepath, index=False)
    print(f"  ✓ Final {media_type} scores saved to: {filepath}")
    
    return filepath

# Main workflow function that includes grouping
def run_complete_workflow_with_grouping(media_type='video', alpha=0.4, beta=0.4, 
                                       lambda_val=0.1, similarity_threshold=0.8,
                                       api_key=None, workflow_timestamp=None):
    """
    Run the complete workflow including topic grouping
    Args:
        media_type: content type for weight selection
        alpha: weight for frequency
        beta: weight for engagement
        lambda_val: lambda value for platform boost calculation
        similarity_threshold: threshold for considering topics similar
        api_key: OpenAI API key for LLM features
        workflow_timestamp: timestamp for output filenames
    Returns:
        dict with all results including grouped topics
    """
    print("Starting complete workflow with topic grouping...")
    
    # Step 1: Collect data from all platforms
    print("\n=== Step 1: Collecting data from platforms ===")
    platform_data = {}
    
    try:
        platform_data['youtube'] = fetch_youtube_popular()
        print("✓ YouTube data collected")
    except Exception as e:
        print(f"✗ YouTube data collection failed: {e}")
        platform_data['youtube'] = None
    
    try:
        platform_data['reddit'] = fetch_reddit_hot()
        print("✓ Reddit data collected")
    except Exception as e:
        print(f"✗ Reddit data collection failed: {e}")
        platform_data['reddit'] = None
    
    try:
        platform_data['tiktok'] = fetch_tiktok_hashtags()
        print("✓ TikTok data collected")
    except Exception as e:
        print(f"✗ TikTok data collection failed: {e}")
        platform_data['tiktok'] = None
    
    try:
        platform_data['twitter'] = fetch_twitter_trends()
        print("✓ Twitter data collected")
    except Exception as e:
        print(f"✗ Twitter data collection failed: {e}")
        platform_data['twitter'] = None
    
    try:
        platform_data['google_trends'] = fetch_google_trends()
        print("✓ Google Trends data collected")
    except Exception as e:
        print(f"✗ Google Trends data collection failed: {e}")
        platform_data['google_trends'] = None
    
    # Step 2: Preprocess data
    print("\n=== Step 2: Preprocessing data ===")
    preprocessed_data = preprocess_platform_data(platform_data)
    print("✓ Data preprocessing completed")
    
    # Step 3: LDA Topic Modeling for YouTube and Reddit
    print("\n=== Step 3: LDA Topic Modeling for YouTube and Reddit ===")
    
    # Process YouTube data if available
    if preprocessed_data.get('youtube') is not None:
        print("3.1: Preprocessing YouTube text data...")
        youtube_df = preprocessed_data['youtube']
        youtube_texts = preprocess_text(youtube_df, ['Title', 'Description'])
        print(f"✓ Preprocessed {len(youtube_texts)} YouTube documents")
        
        print("3.2: Performing LDA topic modeling on YouTube data...")
        youtube_topics, youtube_model, youtube_labels, youtube_keywords, youtube_keywords_str = topic_modeling(youtube_texts)
        print(f"✓ Extracted {len(youtube_labels)} topics from YouTube data")
        
        print("3.3: Labeling YouTube topics with LLM...")
        youtube_topic_counts = [youtube_topics.count(i) for i in range(len(youtube_keywords_str))]
        youtube_llm_labels = label_topics_with_llm(youtube_keywords_str, youtube_topic_counts, api_key)
        print(f"✓ LLM labeled {len(youtube_llm_labels)} YouTube topics")
        
        # Create YouTube topic DataFrame
        youtube_topic_df = create_topic_dataframe(youtube_topics, youtube_llm_labels, youtube_keywords_str, youtube_df, 'youtube')
        preprocessed_data['youtube'] = youtube_topic_df
        print(f"✓ Created YouTube topic DataFrame with {len(youtube_topic_df)} topics")
    else:
        print("✗ YouTube data not available for topic modeling")
    
    # Process Reddit data if available
    if preprocessed_data.get('reddit') is not None:
        print("3.1: Preprocessing Reddit text data...")
        reddit_df = preprocessed_data['reddit']
        reddit_texts = preprocess_text(reddit_df, ['title', 'selftext'])
        print(f"✓ Preprocessed {len(reddit_texts)} Reddit documents")
        
        print("3.2: Performing LDA topic modeling on Reddit data...")
        reddit_topics, reddit_model, reddit_labels, reddit_keywords, reddit_keywords_str = topic_modeling(reddit_texts)
        print(f"✓ Extracted {len(reddit_labels)} topics from Reddit data")
        
        print("3.3: Labeling Reddit topics with LLM...")
        reddit_topic_counts = [reddit_topics.count(i) for i in range(len(reddit_keywords_str))]
        reddit_llm_labels = label_topics_with_llm(reddit_keywords_str, reddit_topic_counts, api_key)
        print(f"✓ LLM labeled {len(reddit_llm_labels)} Reddit topics")
        
        # Create Reddit topic DataFrame
        reddit_topic_df = create_topic_dataframe(reddit_topics, reddit_llm_labels, reddit_keywords_str, reddit_df, 'reddit')
        preprocessed_data['reddit'] = reddit_topic_df
        print(f"✓ Created Reddit topic DataFrame with {len(reddit_topic_df)} topics")
    else:
        print("✗ Reddit data not available for topic modeling")
    
    print("✓ LDA Topic Modeling completed for YouTube and Reddit")
    
    # Step 4: Consolidate platform data
    print("\n=== Step 4: Consolidating platform data ===")
    consolidated_df = consolidate_platform_data(preprocessed_data)
    print(f"✓ Consolidated {len(consolidated_df)} records from all platforms")
    
    # Step 5: Apply min-max scaling
    print("\n=== Step 5: Applying min-max scaling ===")
    scaled_df = apply_minmax_scaling(consolidated_df)
    print("✓ Min-max scaling completed")
    
    # Step 6: Calculate trend scores and save consolidated results (before cross-platform bonus)
    print("\nStep 6: Calculating trend scores...")
    df_with_scores, final_scores = calculate_trend_scores(scaled_df, media_type, alpha, beta)
    consolidated_file = save_consolidated_scores(df_with_scores, media_type, workflow_timestamp)
    print(f"✓ Consolidated scores saved to {consolidated_file}")
    
    # Step 7: Cluster similar topics and save consolidated results with cross-platform bonus
    print("\n=== Step 7: Clustering similar topics ===")
    clustered_results, final_df = cluster_similar_topics(consolidated_file, lambda_val, similarity_threshold)
    print(f"✓ Clustered {len(clustered_results)} topic groups")
    consolidated_w_crossbonus_file = save_final_trend_scores(final_df, media_type, workflow_timestamp)
    print(f"✓ Consolidated scores with cross bonus saved to {consolidated_w_crossbonus_file}")
    
    # Step 8: Generate final grouped output
    print("\n=== Step 8: Generating final output ===")
    final_output_results = generate_final_output(clustered_results, media_type, workflow_timestamp)
    print("✓ Final output generated successfully")
    
    # Step 9: Add category classification
    print("\n=== Step 9: Adding category classification ===")
    categorized_results_filepath = add_category_column(final_output_results['filepath'], api_key)
    print(f"✓ Category classification added to {categorized_results_filepath}")
    
    # Step 10: Calculate media-type specific final scores
    print("\n=== Step 10: Calculating media-type specific final scores ===")
    final_scores_by_media_type_filepath = calculate_final_scores_by_media_type(categorized_results_filepath, media_type, workflow_timestamp)
    print(f"✓ Media-type specific final scores saved to {final_scores_by_media_type_filepath}")
    
    return {
        'consolidated_data': df_with_scores,
        'final_scores': final_scores,
        'clustered_results': clustered_results,
        'final_df': final_df,
        'consolidated_file': consolidated_file,
        'consolidated_w_crossbonus_file': consolidated_w_crossbonus_file,
        'final_output': final_output_results,
        'categorized_results_filepath': categorized_results_filepath,
        'final_scores_by_media_type_filepath': final_scores_by_media_type_filepath
    }

# Example usage (Use all kinds of media types)
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get API key for LLM operations
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
    
    # Define media types to process
    media_types = ['equal', 'video', 'text', 'image']
    
    print("=" * 60)
    print("OPTIMIZED LANGCHAIN WORKFLOW EXECUTION")
    print("=" * 60)
    
    # Generate consistent timestamp for all files in this workflow run
    workflow_timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    print(f"Workflow timestamp: {workflow_timestamp}")
    
    # STEP 1-5: Run once for all media types (common processing)
    print("\n" + "=" * 40)
    print("STEPS 1-5: COMMON PROCESSING (RUNNING ONCE)")
    print("=" * 40)
    
    try:
        # Step 1: Fetch latest data from all platforms
        print("\nStep 1: Fetching latest data from all platforms...")
        youtube_df = fetch_youtube_popular()
        reddit_df = fetch_reddit_hot()
        tiktok_df = fetch_tiktok_hashtags()
        twitter_df = fetch_twitter_trends()
        google_trends_df = fetch_google_trends()
        
        platform_data = {
            'youtube': youtube_df,
            'reddit': reddit_df,
            'tiktok': tiktok_df,
            'twitter': twitter_df,
            'google_trends': google_trends_df
        }
        print("✓ Data collection completed")
        
        # Step 1.5: Preprocess all platform data
        print("\nStep 1.5: Preprocessing all platform data...")
        preprocessed_data = preprocess_platform_data(platform_data)
        print("✓ Data preprocessing completed")
        
        # Step 2: LDA Preprocessing (NLTK initialization)
        print("\nStep 2: Initializing NLTK components for LDA...")
        stop_words, lemmatizer = initialize_nltk_components()
        print("✓ NLTK components initialized")
        
        # Step 3: LDA Topic Modeling for YouTube and Reddit
        print("\nStep 3: LDA Topic Modeling and LLM Labeling...")
        
        # Process YouTube data
        if preprocessed_data.get('youtube') is not None:
            print("  - Processing YouTube data...")
            youtube_texts = preprocess_text(youtube_df, ['Title', 'Description'])
            youtube_topics, youtube_model, youtube_labels, youtube_keywords, youtube_keywords_str = topic_modeling(youtube_texts)
            youtube_topic_counts = [youtube_topics.count(i) for i in range(len(youtube_keywords_str))]
            youtube_llm_labels = label_topics_with_llm(youtube_keywords_str, youtube_topic_counts, api_key)
            preprocessed_data['youtube'] = create_topic_dataframe(youtube_topics, youtube_llm_labels, youtube_keywords_str, youtube_df, 'youtube')
            print(f"  ✓ YouTube: {len(youtube_topics)} topics processed")
        
        # Process Reddit data
        if preprocessed_data.get('reddit') is not None:
            print("  - Processing Reddit data...")
            reddit_texts = preprocess_text(reddit_df, ['title', 'selftext'])
            reddit_topics, reddit_model, reddit_labels, reddit_keywords, reddit_keywords_str = topic_modeling(reddit_texts)
            reddit_topic_counts = [reddit_topics.count(i) for i in range(len(reddit_keywords_str))]
            reddit_llm_labels = label_topics_with_llm(reddit_keywords_str, reddit_topic_counts, api_key)
            preprocessed_data['reddit'] = create_topic_dataframe(reddit_topics, reddit_llm_labels, reddit_keywords_str, reddit_df, 'reddit')
            print(f"  ✓ Reddit: {len(reddit_topics)} topics processed")
        
        print("✓ LDA Topic Modeling and LLM Labeling completed")
        
        # Step 4: Extract platform metrics
        print("\nStep 4: Extracting platform metrics...")
        all_metrics = []
        for platform, df in preprocessed_data.items():
            if df is not None and not df.empty:
                metrics = extract_platform_metrics(df, platform)
                all_metrics.extend(metrics)
        print("✓ Platform metrics extraction completed")
        
        # Step 5: Consolidate and normalize data
        print("\nStep 5: Consolidating and normalizing data...")
        consolidated_df = consolidate_platform_data(preprocessed_data)
        normalized_df = apply_minmax_scaling(consolidated_df)
        print("✓ Data consolidation and normalization completed")
        
        print("\n" + "=" * 40)
        print("COMMON PROCESSING COMPLETED")
        print("=" * 40)
        
        # STEPS 6-8: Run for each media type
        print("\n" + "=" * 40)
        print("STEPS 6-8: MEDIA TYPE SPECIFIC PROCESSING")
        print("=" * 40)
        
        for media_type in media_types:
            print(f"\n--- Processing {media_type.upper()} media type ---")
            
            try:
                # Step 6: Calculate trend scores with media-specific weights
                print(f"  Step 6: Calculating trend scores for {media_type}...")
                df_with_scores, final_scores = calculate_trend_scores(normalized_df, media_type)
                
                # Save Step 6 results
                step6_filepath = save_consolidated_scores(df_with_scores, media_type, workflow_timestamp)
                print(f"  ✓ Step 6 results saved: {step6_filepath}")
                
                # Step 7: Topic clustering and cross-platform bonus
                print(f"  Step 7: Clustering similar topics for {media_type}...")
                clustered_results, final_df = cluster_similar_topics(step6_filepath)
                
                # Save Step 7 results
                step7_filepath = save_final_trend_scores(final_df, media_type, workflow_timestamp)
                print(f"  ✓ Step 7 results saved: {step7_filepath}")
                
                # Step 8: Generate final grouped output
                print(f"  Step 8: Generating final grouped output for {media_type}...")
                step8_filepath = generate_final_output(clustered_results, media_type, workflow_timestamp)
                print(f"  ✓ Step 8 results saved: {step8_filepath}")
                
                # Step 9: Add category classification
                print(f"  Step 9: Adding category classification for {media_type}...")
                step9_filepath = add_category_column(step8_filepath, api_key)
                print(f"  ✓ Step 9 results saved: {step9_filepath}")
                
                # Step 10: Calculate media-type specific final scores
                print(f"  Step 10: Calculating media-type specific final scores for {media_type}...")
                step10_filepath = calculate_final_scores_by_media_type(step9_filepath, media_type, workflow_timestamp)
                print(f"  ✓ Step 10 results saved: {step10_filepath}")
                
                print(f"  ✓ {media_type.upper()} processing completed successfully")
                
            except Exception as e:
                print(f"  ✗ Error processing {media_type}: {str(e)}")
                continue
        
        print("\n" + "=" * 60)
        print("WORKFLOW EXECUTION COMPLETED")
        print("=" * 60)
        print(f"Generated files for {len(media_types)} media types")
        print(f"All files use consistent timestamp: {workflow_timestamp}")
        
    except Exception as e:
        print(f"\n✗ Workflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
