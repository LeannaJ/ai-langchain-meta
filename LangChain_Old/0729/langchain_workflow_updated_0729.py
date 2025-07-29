#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain Workflow for Multi-Platform Trend Analysis - Optimized Version 2
A comprehensive pipeline for collecting, processing, and analyzing trending content
from YouTube and Reddit using LDA topic modeling and LangChain tools.
Updated version with optimized processing flow and LLM-enhanced topic grouping.

WORKFLOW STRUCTURE:
Step 1-7: Common Processing (Run Once)
1. File Loading
2. Preprocessing  
3. LDA + LLM for YouTube, Reddit
   - 3.1 LDA Preprocessing
   - 3.2 LDA Topic Modeling
   - 3.3 LLM Labeling
4. Aggregation
5. Scaling
6. Similar Topic Finding
   - 6.1 Similarity Calculation (1st pass)
   - 6.2 LLM (2nd pass)
   - 6.3 Grouping Results Save (File Save - 1 file regardless of media type)
7. Category Assignment

Step 8: Score Calculation - Media Type Specific (Repeat for each media type)
- 8.1 Weight assignment and S_p score calculation
- 8.2 Cross-platform bonus calculation (for topics grouped in step 6)
- 8.3 Final score file (File Save - 4 files: equal, video, text, image)

Total Output Files: 5 files
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
    tokenize, remove stopwords, lemmatize
    Args:
        df: DataFrame with text columns
        text_cols: list of column names containing text
    Returns:
        list of preprocessed text documents
    """
    stop_words, lemmatizer = initialize_nltk_components()
    
    def preprocess(text):
        if pd.isna(text) or text == '':
            return []
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        
        return tokens
    
    # Process all text columns
    all_texts = []
    for col in text_cols:
        if col in df.columns:
            texts = df[col].apply(preprocess)
            all_texts.extend(texts.tolist())
    
    return all_texts

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
        os.environ['OPENAI_API_KEY'] = api_key
    
    try:
        chain = create_topic_labeling_chain()
        labels = []
        
        for keywords, count in zip(topic_keywords_str, topic_counts):
            try:
                result = chain.run(keywords=keywords, topic_count=count)
                # Clean the result
                label = result.strip().replace('"', '').replace("'", "")
                labels.append(label)
            except Exception as e:
                print(f"Error labeling topic '{keywords}': {str(e)}")
                labels.append(keywords.split(',')[0].title())  # Fallback to first keyword
        
        return labels
        
    except Exception as e:
        print(f"LLM labeling failed: {str(e)}")
        # Fallback to simple keyword-based labels
        return [keywords.split(',')[0].title() for keywords in topic_keywords_str]

def create_topic_dataframe(topics, topic_labels, topic_keywords_str, original_df, platform):
    """
    Create DataFrame with topic information
    Args:
        topics: topic assignments for each document
        topic_labels: list of topic labels
        topic_keywords_str: list of keyword strings
        original_df: original platform DataFrame
        platform: platform name
    Returns:
        DataFrame with topic information added
    """
    df = original_df.copy()
    
    # Add topic information
    df['topic_id'] = topics[:len(df)]  # Ensure we don't exceed DataFrame length
    df['topic_label'] = df['topic_id'].apply(lambda x: topic_labels[x] if x < len(topic_labels) else 'Unknown')
    df['topic_keywords'] = df['topic_id'].apply(lambda x: topic_keywords_str[x] if x < len(topic_keywords_str) else '')
    
    # Create keyword column for consistency
    df['keyword'] = df['topic_label']
    
    # Add platform information
    df['platform'] = platform
    
    return df

# STEP 4: Platform Metrics Extraction
def extract_platform_metrics(df, platform):
    """
    Extract and calculate platform-specific metrics
    Args:
        df: DataFrame with platform data
        platform: platform name
    Returns:
        list of dictionaries with metrics
    """
    metrics = []
    
    if platform == 'youtube':
        for _, row in df.iterrows():
            metrics.append({
                'keyword': row['keyword'],
                'platform': 'YouTube',
                'frequency': row.get('view_count', 0),
                'engagement': row.get('like_count', 0) + row.get('comment_count', 0),
                'topic_label': row.get('topic_label', ''),
                'topic_keywords': row.get('topic_keywords', '')
            })
    
    elif platform == 'reddit':
        for _, row in df.iterrows():
            metrics.append({
                'keyword': row['keyword'],
                'platform': 'Reddit',
                'frequency': row.get('score', 0),
                'engagement': row.get('num_comments', 0),
                'topic_label': row.get('topic_label', ''),
                'topic_keywords': row.get('topic_keywords', '')
            })
    
    elif platform == 'tiktok':
        for _, row in df.iterrows():
            metrics.append({
                'keyword': row['hashtag'],
                'platform': 'TikTok',
                'frequency': row.get('view_count', 0),
                'engagement': row.get('like_count', 0) + row.get('comment_count', 0),
                'topic_label': row['hashtag'],
                'topic_keywords': row['hashtag']
            })
    
    elif platform == 'twitter':
        for _, row in df.iterrows():
            metrics.append({
                'keyword': row['trend'],
                'platform': 'X/Twitter',
                'frequency': row.get('tweet_count', 0),
                'engagement': row.get('duration', 1),  # Using duration as engagement proxy
                'topic_label': row['trend'],
                'topic_keywords': row['trend']
            })
    
    elif platform == 'google_trends':
        for _, row in df.iterrows():
            metrics.append({
                'keyword': row['term'],
                'platform': 'Google Trends',
                'frequency': row.get('median_gain', 0),
                'engagement': row.get('median_gain', 0),  # Using same metric for both
                'topic_label': row['term'],
                'topic_keywords': row['term']
            })
    
    return metrics

# STEP 5: Data Consolidation and Scaling
def consolidate_platform_data(platform_data_dict):
    """
    Consolidate data from all platforms into a single DataFrame
    Args:
        platform_data_dict: dict with platform names as keys and DataFrames as values
    Returns:
        consolidated DataFrame
    """
    all_metrics = []
    
    for platform, df in platform_data_dict.items():
        if df is not None and not df.empty:
            metrics = extract_platform_metrics(df, platform)
            all_metrics.extend(metrics)
    
    if not all_metrics:
        raise ValueError("No metrics found from any platform")
    
    consolidated_df = pd.DataFrame(all_metrics)
    return consolidated_df

def minmax_normalize(series):
    """Min-max normalization for a pandas series"""
    if series.max() == series.min():
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def apply_minmax_scaling(df):
    """
    Apply min-max scaling to frequency and engagement metrics
    Args:
        df: DataFrame with frequency and engagement columns
    Returns:
        DataFrame with normalized metrics
    """
    df = df.copy()
    
    # Normalize frequency and engagement per platform
    for platform in df['platform'].unique():
        platform_mask = df['platform'] == platform
        df.loc[platform_mask, 'frequency_norm'] = minmax_normalize(df.loc[platform_mask, 'frequency'])
        df.loc[platform_mask, 'engagement_norm'] = minmax_normalize(df.loc[platform_mask, 'engagement'])
    
    return df

# STEP 6: Similar Topic Finding (Common Processing)
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
    Initial clustering based on keyword similarity (1st pass)
    Args:
        df: DataFrame with keywords
        similarity_threshold: threshold for considering topics similar
    Returns:
        list of initial groups
    """
    topics = df['keyword'].tolist()
    groups = []
    assigned = [False] * len(topics)

    # Group similar topics
    for i, topic in enumerate(topics):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True

        # Find similar topics
        for j in range(i+1, len(topics)):
            if not assigned[j]:
                similarity = calculate_keyword_similarity(topic, topics[j])
                if similarity >= similarity_threshold:
                    group.append(j)
                    assigned[j] = True
        groups.append(group)

    return groups

# 6.2: LLM (2nd pass)
def create_llm_grouping_chain():
    """Create LLM chain for semantic topic grouping"""
    llm = OpenAI(temperature=0.2)  # Low temperature for consistent grouping
    
    prompt_template = PromptTemplate(
        input_variables=["topics", "existing_groups"],
        template="""
        Analyze the following topics and existing groups to improve semantic grouping.
        
        Topics: {topics}
        Existing Groups: {existing_groups}
        
        Instructions:
        - Review the existing groups and identify topics that should be grouped together semantically
        - Consider meaning, context, and related concepts
        - Suggest improved groupings that capture semantic relationships
        - Respond with a list of improved groups, each containing semantically related topics
        
        Improved Groups:
        """
    )
    
    return LLMChain(llm=llm, prompt=prompt_template)

def llm_enhanced_grouping(initial_groups, topics_list, api_key=None):
    """
    Use LLM to enhance initial grouping with semantic understanding
    Args:
        initial_groups: list of initial groups from similarity clustering
        topics_list: list of all topics
        api_key: OpenAI API key
    Returns:
        list of enhanced groups
    """
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    try:
        # Prepare topics and existing groups for LLM
        topics_text = "\n".join([f"- {topic}" for topic in topics_list])
        existing_groups_text = "\n".join([
            f"Group {i+1}: {', '.join([topics_list[idx] for idx in group])}"
            for i, group in enumerate(initial_groups)
        ])
        
        # Create and run LLM chain
        chain = create_llm_grouping_chain()
        result = chain.run(topics=topics_text, existing_groups=existing_groups_text)
        
        # Parse LLM result and create enhanced groups
        # This is a simplified implementation - in practice, you'd need more robust parsing
        enhanced_groups = parse_llm_grouping_result(result, topics_list)
        
        return enhanced_groups if enhanced_groups else initial_groups
        
    except Exception as e:
        print(f"LLM grouping failed: {str(e)}")
        return initial_groups  # Fallback to initial groups

def parse_llm_grouping_result(llm_result, topics_list):
    """
    Parse LLM result to extract enhanced groups
    Args:
        llm_result: result from LLM grouping
        topics_list: list of all topics
    Returns:
        list of enhanced groups
    """
    # This is a simplified parser - in practice, you'd need more sophisticated parsing
    # For now, return the original groups
    return []

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
            final_df.loc[idx, 'cross_platform_bonus'] = platform_boost
            final_df.loc[idx, 'final_trend_score'] = final_score
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
    final_df['cross_platform_bonus'] = final_df['cross_platform_bonus'].fillna(0)
    final_df['final_trend_score'] = final_df['final_trend_score'].fillna(final_df['frequency_norm'] + final_df['engagement_norm'])
    final_df['group_id'] = final_df['group_id'].fillna(-1)
    final_df['group_name'] = final_df['group_name'].fillna(final_df['keyword'])
    
    return results, final_df

# 6.3: Grouping Results Save
def save_grouping_results(results, workflow_timestamp):
    """
    Save grouping results (common file regardless of media type)
    Args:
        results: list of grouped topic dictionaries
        workflow_timestamp: timestamp for file naming
    Returns:
        filepath of saved CSV
    """
    output_filename = f"topic_grouping_results_{workflow_timestamp}.csv"
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    
    print(f"Step 6.3: Grouping results saved to '{filepath}'")
    print(f"  Total groups created: {len(df)}")
    
    return filepath

# STEP 7: Category Assignment (Common Processing)
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

def add_category_column(grouping_results_filepath, api_key=None):
    """
    Add category column to grouping results
    Args:
        grouping_results_filepath: path to grouping results CSV
        api_key: OpenAI API key
    Returns:
        filepath: path to updated CSV with category column
    """
    print("Step 7: Adding category classification...")
    
    # Load grouping results
    df = pd.read_csv(grouping_results_filepath)
    
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
    output_filepath = grouping_results_filepath.replace('.csv', '_with_categories.csv')
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
    print(f"Step 8: Calculating final scores for {media_type} media type")
    
    # Load categorized results
    df = pd.read_csv(categorized_results_filepath)
    
    # Get platform weights for this media type
    platform_weights = get_platform_weights(media_type)
    
    # 8.1: Weight assignment and S_p score calculation
    print(f"  8.1: Weight assignment and S_p score calculation...")
    df['platform_weight'] = df['platform'].map(platform_weights).fillna(0)
    df['S_p'] = df['platform_weight'] * (
        0.4 * df['frequency_norm'] + 0.4 * df['engagement_norm']
    )
    
    # 8.2: Cross-platform bonus calculation
    print(f"  8.2: Cross-platform bonus calculation...")
    # Cross-platform bonus is already calculated in Step 6
    # We just need to ensure it's properly applied
    
    # 8.3: Final score calculation
    print(f"  8.3: Final score calculation...")
    df['final_trend_score'] = df['S_p'] + df['cross_platform_bonus']
    
    # Save media-type specific results
    if media_type == 'equal':
        output_filename = f"final_scores_equal_{workflow_timestamp}.csv"
    else:
        output_filename = f"final_scores_{media_type}_{workflow_timestamp}.csv"
    
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    
    df.to_csv(filepath, index=False)
    print(f"  ✓ Final {media_type} scores saved to: {filepath}")
    
    return filepath

# Main execution function
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
    print("OPTIMIZED LANGCHAIN WORKFLOW V2 EXECUTION")
    print("=" * 60)
    
    # Generate consistent timestamp for all files in this workflow run
    workflow_timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    print(f"Workflow timestamp: {workflow_timestamp}")
    
    # STEPS 1-7: COMMON PROCESSING (RUNNING ONCE)
    print("\n" + "=" * 40)
    print("STEPS 1-7: COMMON PROCESSING (RUNNING ONCE)")
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
        
        # Step 2: Preprocess all platform data
        print("\nStep 2: Preprocessing all platform data...")
        preprocessed_data = preprocess_platform_data(platform_data)
        print("✓ Data preprocessing completed")
        
        # Step 3: LDA Topic Modeling for YouTube and Reddit
        print("\nStep 3: LDA Topic Modeling and LLM Labeling...")
        
        # Process YouTube data
        if preprocessed_data.get('youtube') is not None:
            print("  3.1: LDA preprocessing for YouTube...")
            youtube_texts = preprocess_text(youtube_df, ['Title', 'Description'])
            print("  3.2: LDA topic modeling for YouTube...")
            youtube_topics, _, youtube_labels, _, youtube_keywords_str = topic_modeling(youtube_texts)
            youtube_topic_counts = [youtube_topics.count(i) for i in range(len(youtube_labels))]
            print("  3.3: LLM labeling for YouTube...")
            youtube_llm_labels = label_topics_with_llm(youtube_keywords_str, youtube_topic_counts, api_key)
            preprocessed_data['youtube'] = create_topic_dataframe(youtube_topics, youtube_llm_labels, youtube_keywords_str, youtube_df, 'youtube')
            print(f"  ✓ YouTube: {len(youtube_topics)} topics processed")
        
        # Process Reddit data
        if preprocessed_data.get('reddit') is not None:
            print("  3.1: LDA preprocessing for Reddit...")
            reddit_texts = preprocess_text(reddit_df, ['title', 'selftext'])
            print("  3.2: LDA topic modeling for Reddit...")
            reddit_topics, _, reddit_labels, _, reddit_keywords_str = topic_modeling(reddit_texts)
            reddit_topic_counts = [reddit_topics.count(i) for i in range(len(reddit_labels))]
            print("  3.3: LLM labeling for Reddit...")
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
        
        # Step 6: Similar topic finding
        print("\nStep 6: Similar topic finding...")
        clustered_results, final_df = cluster_similar_topics(normalized_df, api_key=api_key)
        grouping_filepath = save_grouping_results(clustered_results, workflow_timestamp)
        print("✓ Topic clustering and grouping completed")
        
        # Step 7: Category assignment
        print("\nStep 7: Category assignment...")
        categorized_filepath = add_category_column(grouping_filepath, api_key)
        print("✓ Category classification completed")
        
        print("\n" + "=" * 40)
        print("COMMON PROCESSING COMPLETED")
        print("=" * 40)
        
        # STEP 8: MEDIA TYPE SPECIFIC PROCESSING
        print("\n" + "=" * 40)
        print("STEP 8: MEDIA TYPE SPECIFIC PROCESSING")
        print("=" * 40)
        
        for media_type in media_types:
            print(f"\n--- Processing {media_type.upper()} media type ---")
            
            try:
                # Step 8: Calculate final scores for this media type
                final_scores_filepath = calculate_final_scores_by_media_type(categorized_filepath, media_type, workflow_timestamp)
                print(f"  ✓ {media_type.upper()} processing completed successfully")
                
            except Exception as e:
                print(f"  ✗ Error processing {media_type}: {str(e)}")
                continue
        
        print("\n" + "=" * 60)
        print("WORKFLOW EXECUTION COMPLETED")
        print("=" * 60)
        print(f"Generated files for {len(media_types)} media types")
        print(f"All files use consistent timestamp: {workflow_timestamp}")
        print(f"Total output files: 5 (1 common + 4 media-type specific)")
        
    except Exception as e:
        print(f"\n✗ Workflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
