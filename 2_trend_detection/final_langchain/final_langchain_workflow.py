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
    """Fetch latest Google Trends rising data"""
    latest = get_latest_csv("Scraped_Data/google_trends_rising_*.csv")
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

# STEP 6: Similar Topic Finding
# 6.1: Similarity Calculation (1st pass)
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

def initial_topic_clustering(df, similarity_threshold=0.6):
    """
    Initial clustering based on keyword similarity
    Args:
        df: DataFrame with topic data
        similarity_threshold: threshold for grouping similar topics
    Returns:
        list of topic groups (each group is a list of indices)
    """
    topics = df['keyword'].tolist()
    n_topics = len(topics)
    groups = []
    used = set()
    
    for i in range(n_topics):
        if i in used:
            continue
            
        current_group = [i]
        used.add(i)
        
        for j in range(i + 1, n_topics):
            if j in used:
                continue
                
            similarity = calculate_keyword_similarity(topics[i], topics[j])
            if similarity >= similarity_threshold:
                current_group.append(j)
                used.add(j)
        
        groups.append(current_group)
    
    return groups

# 6.2: LLM Enhancement (2nd pass)
def create_llm_grouping_chain():
    """Create LLM chain for semantic topic grouping"""
    llm = OpenAI(temperature=0.1)  # Low temperature for consistent grouping
    
    prompt_template = PromptTemplate(
        input_variables=["topics_text", "existing_groups_text"],
        template="""
        You are an expert at grouping similar topics semantically. Analyze the following topics and existing groups, then suggest improvements to group similar topics together.

        Topics to analyze:
        {topics_text}

        Current groups:
        {existing_groups_text}

        Instructions:
        1. Review the current grouping
        2. Identify topics that should be grouped together but aren't
        3. Suggest merging of existing groups if they are semantically similar
        4. Consider semantic meaning, not just keyword overlap
        5. Return ONLY the improved grouping as a list of lists, where each inner list contains topic indices

        Improved grouping: """
    )
    
    return LLMChain(llm=llm, prompt=prompt_template)

def llm_enhanced_grouping(initial_groups, topics_list, api_key=None):
    """
    Enhance initial grouping using LLM semantic analysis
    Args:
        initial_groups: list of initial topic groups
        topics_list: list of all topic keywords
        api_key: OpenAI API key
    Returns:
        enhanced groups list
    """
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    try:
        # Prepare topics text
        topics_text = "\n".join([f"{i}: {topic}" for i, topic in enumerate(topics_list)])
        
        # Prepare existing groups text
        groups_text = []
        for i, group in enumerate(initial_groups):
            group_topics = [topics_list[idx] for idx in group]
            groups_text.append(f"Group {i}: {', '.join(group_topics)}")
        existing_groups_text = "\n".join(groups_text)
        
        # Create and run LLM chain
        chain = create_llm_grouping_chain()
        result = chain.run(topics_text=topics_text, existing_groups_text=existing_groups_text)
        
        # Parse LLM result
        enhanced_groups = parse_llm_grouping_result(result, topics_list)
        
        return enhanced_groups
        
    except Exception as e:
        print(f"LLM grouping failed: {str(e)}")
        print("Falling back to initial groups")
        return initial_groups

def parse_llm_grouping_result(llm_result, topics_list):
    """
    Parse LLM grouping result into list of groups
    Args:
        llm_result: raw LLM response
        topics_list: list of all topics
    Returns:
        list of topic groups
    """
    try:
        # Simple parsing - extract numbers from the result
        import re
        numbers = re.findall(r'\d+', llm_result)
        
        # Convert to groups (assuming LLM returned topic indices)
        groups = []
        current_group = []
        
        for num in numbers:
            idx = int(num)
            if idx < len(topics_list):
                current_group.append(idx)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        
        if current_group:
            groups.append(current_group)
        
        return groups if groups else [[i] for i in range(len(topics_list))]
        
    except Exception as e:
        print(f"Error parsing LLM result: {str(e)}")
        # Fallback: return individual topics as groups
        return [[i] for i in range(len(topics_list))]

def cluster_similar_topics(df, lambda_val=0.1, similarity_threshold=0.6, api_key=None):
    """
    Enhanced topic clustering with LLM assistance
    Args:
        df: DataFrame with topic data
        lambda_val: lambda value for platform boost calculation
        similarity_threshold: threshold for similarity clustering
        api_key: OpenAI API key for LLM enhancement
    Returns:
        list of clustered results and enhanced DataFrame
    """
    print("Step 6.1: Initial similarity-based clustering...")
    
    # 6.1: Initial clustering based on similarity
    initial_groups = initial_topic_clustering(df, similarity_threshold)
    print(f"  ✓ Initial clustering created {len(initial_groups)} groups")
    
    # 6.2: LLM enhancement
    print("Step 6.2: LLM-enhanced semantic grouping...")
    topics_list = df['keyword'].tolist()
    enhanced_groups = llm_enhanced_grouping(initial_groups, topics_list, api_key)
    print(f"  ✓ LLM enhancement completed")
    
    # Process groups and calculate scores
    results = []
    final_df = df.copy()
    group_counter = 0
    
    for group in enhanced_groups:
        group_keywords = [topics_list[idx] for idx in group]
        group_df = df.iloc[group]
        
        # Calculate group statistics
        sum_sp = group_df['frequency_norm'].sum() + group_df['engagement_norm'].sum()
        platforms = set(group_df['platform'])
        platform_boost = lambda_val * (len(platforms) - 1)
        final_score = sum_sp + platform_boost
        
        # Update DataFrame
        for idx in group:
            final_df.loc[idx, 'platform_count'] = len(platforms)
            final_df.loc[idx, 'group_id'] = group_counter
            final_df.loc[idx, 'group_name'] = group_keywords[0]
        
        results.append({
            'group_name': group_keywords[0],
            'group_keywords': group_keywords,
            'keyword_count': len(group_keywords),
            'sum_S_p': sum_sp,
            'platforms': list(platforms),
            'platform_count': len(platforms),
            'platform_boost': platform_boost,
            'final_score': final_score
        })
        
        group_counter += 1
    
    # Fill NaN values
    final_df['platform_count'] = final_df['platform_count'].fillna(1)
    final_df['group_id'] = final_df['group_id'].fillna(-1)
    final_df['group_name'] = final_df['group_name'].fillna(final_df['keyword'])
    
    return results, final_df

# 6.3: Grouping Results Save
def save_grouping_results(final_df, workflow_timestamp):
    """
    Save detailed data with grouping information
    Args:
        final_df: DataFrame with original data + grouping info
        workflow_timestamp: timestamp for file naming
    Returns:
        detailed_filepath: path to saved detailed CSV
    """
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed data with grouping info (for Step 7 and 8)
    detailed_filename = f"detailed_data_with_grouping_{workflow_timestamp}.csv"
    detailed_filepath = os.path.join(output_dir, detailed_filename)
    final_df.to_csv(detailed_filepath, index=False)
    
    print(f"Step 6.3: Grouping results saved:")
    print(f"  Detailed: '{detailed_filepath}' ({len(final_df)} records)")
    
    return detailed_filepath

# STEP 7: Category Assignment
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
        closest_category: matched category name
    """
    try:
        # Simple string matching - find category with most common words
        llm_words = set(llm_result.lower().split())
        best_match = categories[0]
        best_score = 0
        
        for category in categories:
            category_words = set(category.lower().split())
            common_words = llm_words.intersection(category_words)
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_match = category
        
        return best_match
        
    except Exception as e:
        print(f"Error matching category: {str(e)}")
        return "Entertainment & Media"  # Default fallback

def batch_classify_topics(topics_list, api_key=None, batch_size=10):
    """
    Classify multiple topics in batches to avoid rate limits
    Args:
        topics_list: list of topics to classify
        api_key: OpenAI API key
        batch_size: number of topics to process in each batch
    Returns:
        dict: mapping of topic to category
    """
    topic_to_category = {}
    total_batches = (len(topics_list) + batch_size - 1) // batch_size
    
    print(f"  Classifying {len(topics_list)} topics in {total_batches} batches...")
    
    for i in range(0, len(topics_list), batch_size):
        batch = topics_list[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  Classifying batch {batch_num}/{total_batches} ({len(batch)} topics)")
        
        for topic in batch:
            category = classify_topic_category(topic, api_key)
            topic_to_category[topic] = category
    
    return topic_to_category

def add_category_column(detailed_filepath, api_key=None):
    """
    Add category column to detailed data with grouping information
    Args:
        detailed_filepath: path to detailed data CSV with grouping info
        api_key: OpenAI API key
    Returns:
        filepath: path to updated CSV with category column
    """
    print("Step 7: Adding category classification...")
    
    # Load detailed data
    df = pd.read_csv(detailed_filepath)
    
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
    output_filepath = detailed_filepath.replace('.csv', '_with_categories.csv')
    df.to_csv(output_filepath, index=False)
    
    print(f"  ✓ Category classification completed")
    print(f"  ✓ Updated results saved to: {output_filepath}")
    
    # Print category distribution
    category_counts = df['category'].value_counts()
    print(f"\n  Category Distribution:")
    for category, count in category_counts.items():
        print(f"    {category}: {count} topics")
    
    return output_filepath

# STEP 8: Score Calculation (Media Type Specific)
# 8.1: Weight assignment and S_p score calculation
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

def calculate_trend_scores(df, media_type='video', alpha=0.5, beta=0.5):
    """
    8.1: Calculate weighted trend scores
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

# 8.2: Cross-platform bonus calculation
def add_cross_platform_bonus(df, lambda_val=0.1):
    """
    8.2: Add cross-platform bonus for topics appearing on multiple platforms
    Args:
        df: DataFrame with trend scores calculated
        lambda_val: lambda value for platform boost calculation
    Returns:
        DataFrame with cross-platform bonus added
    """
    df = df.copy()
    
    # Calculate cross-platform bonus based on platform_count
    # Bonus = lambda_val * (platform_count - 1)
    df['cross_platform_bonus'] = lambda_val * (df['platform_count'] - 1)
    
    # Calculate final trend score: Final Score = S_p + Cross-platform Bonus
    df['final_trend_score'] = df['S_p'] + df['cross_platform_bonus']
    
    return df

# 8.3: Save consolidated scores with cross-platform bonus to CSV file
def save_consolidated_scores(df_with_scores, media_type='equal', workflow_timestamp=None, output_filename=None):
    """
    8.3: Save consolidated scores with cross-platform bonus to CSV file
    Args:
        df_with_scores: DataFrame with trend scores and cross-platform bonus calculated
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
    df_with_scores.to_csv(filepath, index=False)
    print(f"Consolidated scores with cross-platform bonus exported to '{filepath}'")
    print(f"Total records: {len(df_with_scores)}")
    
    return filepath
    main()