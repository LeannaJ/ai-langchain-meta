#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain Workflow for Multi-Platform Trend Analysis
A comprehensive pipeline for collecting, processing, and analyzing trending content
from YouTube and Reddit using LDA topic modeling and LangChain tools.
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
    latest = get_latest_csv("Scraped_Data/reddit_trend_analysis_*.csv")
    return pd.read_csv(latest)

def fetch_tiktok_hashtags():
    """Fetch latest TikTok hashtags data"""
    latest = get_latest_csv("Scraped_Data/tiktok_hashtags_*.csv")
    return pd.read_csv(latest)

def fetch_twitter_trends():
    """Fetch latest Twitter trends data"""
    latest = get_latest_csv("Scraped_Data/twitter_trends_*.csv")
    return pd.read_csv(latest)

def fetch_google_trends():
    """Fetch latest Google Trends data"""
    latest = get_latest_csv("Scraped_Data/google_trends_*.csv")
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

# STEP 1.5: Data Preprocessing
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

# STEP 2: LDA Preprocessing
def preprocess_text(df, text_cols):
    """
    LDA-style preprocessing: lowercase, remove special chars, normalize whitespace, 
    remove stopwords, tokenize, lemmatize, and filter out meaningless words
    """
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

# STEP 3: LDA Topic Modeling
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

# STEP 3.5: LLM-based Topic Labeling
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

# STEP 6: Give Weight and Calculating TrendScore
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

# STEP 7: Topic Matching: Clustering & Semantic Analysis, LLM
def calculate_keyword_similarity(keywords1, keywords2):
    """
    Calculate similarity between two keyword strings
    Args:
        keywords1, keywords2: comma-separated keyword strings
    Returns:
        similarity score (0-1)
    """
    from difflib import SequenceMatcher
    
    # Convert to sets for better comparison
    kw1_set = set([k.strip().lower() for k in keywords1.split(',') if k.strip()])
    kw2_set = set([k.strip().lower() for k in keywords2.split(',') if k.strip()])
    
    if not kw1_set or not kw2_set:
        return 0.0
    
    # Jaccard similarity
    intersection = len(kw1_set & kw2_set)
    union = len(kw1_set | kw2_set)
    jaccard_sim = intersection / union if union > 0 else 0.0
    
    # String similarity for partial matches
    str1 = ' '.join(sorted(kw1_set))
    str2 = ' '.join(sorted(kw2_set))
    string_sim = SequenceMatcher(None, str1, str2).ratio()
    
    # Combined similarity (weighted average)
    combined_sim = 0.7 * jaccard_sim + 0.3 * string_sim
    
    return combined_sim

def create_topic_matching_chain():
    """Create LLM chain for topic matching"""
    llm = OpenAI(temperature=0.1)  # Very low temperature for consistent matching
    
    prompt_template = PromptTemplate(
        input_variables=["topic1_label", "topic1_keywords", "topic2_label", "topic2_keywords"],
        template="""
        Compare these two topics and determine if they represent the same or very similar subject matter.
        
        Topic 1:
        - Label: {topic1_label}
        - Keywords: {topic1_keywords}
        
        Topic 2:
        - Label: {topic2_label}
        - Keywords: {topic2_keywords}
        
        Answer with ONLY "YES" if these topics are the same or very similar, or "NO" if they are different topics.
        Consider synonyms, related concepts, and different ways to express the same idea.
        """
    )
    
    return LLMChain(llm=llm, prompt=prompt_template)

def llm_topic_matching(topic1_label, topic1_keywords, topic2_label, topic2_keywords, api_key=None):
    """
    Use LLM to determine if two topics are the same
    Args:
        topic1_label, topic1_keywords: first topic info
        topic2_label, topic2_keywords: second topic info
        api_key: OpenAI API key (optional)
    Returns:
        True if topics match, False otherwise
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        chain = create_topic_matching_chain()
        result = chain.run(
            topic1_label=topic1_label,
            topic1_keywords=topic1_keywords,
            topic2_label=topic2_label,
            topic2_keywords=topic2_keywords
        )
        
        # Parse result
        result = result.strip().upper()
        return result == "YES"
    except Exception as e:
        print(f"LLM topic matching failed: {e}")
        # Fallback to keyword similarity
        similarity = calculate_keyword_similarity(topic1_keywords, topic2_keywords)
        return similarity > 0.5  # Threshold for similarity

def advanced_topic_matching(topic_df1, topic_df2, similarity_threshold=0.6, use_llm=True, api_key=None):
    """
    Advanced topic matching using similarity and LLM
    Args:
        topic_df1: First platform topic DataFrame
        topic_df2: Second platform topic DataFrame
        similarity_threshold: minimum similarity score for matching
        use_llm: whether to use LLM for final verification
        api_key: OpenAI API key for LLM matching
    Returns:
        matched_pairs: list of (idx1, idx2) pairs
        similarity_scores: dict of similarity scores
    """
    matched_pairs = []
    similarity_scores = {}
    
    for idx1, row1 in topic_df1.iterrows():
        for idx2, row2 in topic_df2.iterrows():
            # Calculate keyword similarity
            kw_similarity = calculate_keyword_similarity(
                row1['keywords'], row2['keywords']
            )
            
            # Store similarity score
            pair_key = (idx1, idx2)
            similarity_scores[pair_key] = kw_similarity
            
            # Check if similarity meets threshold
            if kw_similarity >= similarity_threshold:
                # Use LLM for final verification if enabled
                if use_llm:
                    llm_match = llm_topic_matching(
                        row1['topic_label'], row1['keywords'],
                        row2['topic_label'], row2['keywords'],
                        api_key
                    )
                    if llm_match:
                        matched_pairs.append((idx1, idx2))
                else:
                    matched_pairs.append((idx1, idx2))
    
    return matched_pairs, similarity_scores

# Create tools for advanced matching
topic_similarity_tool = Tool.from_function(
    name="TopicSimilarity",
    func=calculate_keyword_similarity,
    description="Calculate similarity between two keyword strings"
)

topic_matching_tool = Tool.from_function(
    name="TopicMatching",
    func=lambda df1, df2, threshold=0.6: advanced_topic_matching(df1, df2, threshold),
    description="Match topics between platforms using similarity and LLM"
)
