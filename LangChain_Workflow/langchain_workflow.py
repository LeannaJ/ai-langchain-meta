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
        passes=10,
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

# STEP 4: Advanced Topic Matching Functions
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

def advanced_topic_matching(yt_topic_df, rd_topic_df, similarity_threshold=0.6, use_llm=True, api_key=None):
    """
    Advanced topic matching using similarity and LLM
    Args:
        yt_topic_df: YouTube topic DataFrame
        rd_topic_df: Reddit topic DataFrame
        similarity_threshold: minimum similarity score for matching
        use_llm: whether to use LLM for final verification
        api_key: OpenAI API key for LLM matching
    Returns:
        matched_pairs: list of (yt_idx, rd_idx) pairs
        similarity_scores: dict of similarity scores
    """
    matched_pairs = []
    similarity_scores = {}
    
    for yt_idx, yt_row in yt_topic_df.iterrows():
        for rd_idx, rd_row in rd_topic_df.iterrows():
            # Calculate keyword similarity
            kw_similarity = calculate_keyword_similarity(
                yt_row['keywords'], rd_row['keywords']
            )
            
            # Store similarity score
            pair_key = (yt_idx, rd_idx)
            similarity_scores[pair_key] = kw_similarity
            
            # Check if similarity meets threshold
            if kw_similarity >= similarity_threshold:
                # Use LLM for final verification if enabled
                if use_llm:
                    llm_match = llm_topic_matching(
                        yt_row['topic_label'], yt_row['keywords'],
                        rd_row['topic_label'], rd_row['keywords'],
                        api_key
                    )
                    if llm_match:
                        matched_pairs.append((yt_idx, rd_idx))
                else:
                    matched_pairs.append((yt_idx, rd_idx))
    
    return matched_pairs, similarity_scores

# Create tools for advanced matching
topic_similarity_tool = Tool.from_function(
    name="TopicSimilarity",
    func=calculate_keyword_similarity,
    description="Calculate similarity between two keyword strings"
)

topic_matching_tool = Tool.from_function(
    name="TopicMatching",
    func=lambda yt_df, rd_df, threshold=0.6: advanced_topic_matching(yt_df, rd_df, threshold),
    description="Match topics between YouTube and Reddit using similarity and LLM"
)

# STEP 5: Topic-level aggregation of original metrics - platform-specific topic metric aggregation
def aggregate_metrics_by_topic(df, topics, metrics_cols, agg_func='sum'):
    """Aggregate original metrics by topic (sum or mean)"""
    df = df.copy()
    df['topic_id'] = topics
    if agg_func == 'sum':
        agg_df = df.groupby('topic_id')[metrics_cols].sum()
    elif agg_func == 'mean':
        agg_df = df.groupby('topic_id')[metrics_cols].mean()
    else:
        raise ValueError('agg_func must be "sum" or "mean"')
    return agg_df.reset_index()

# STEP 6: Topic-level metrics calculation (custom equations) - topic-specific metric calculation
# Example metric weights for each platform
YOUTUBE_METRIC_WEIGHTS = {
    'View Velocity (Views/Hour)': 0.4,
    'Like Velocity (Likes/Hour)': 0.2,
    'Comment Velocity (Comments/Hour)': 0.2,
    'Like-to-View Ratio (%)': 0.2
}
REDDIT_METRIC_WEIGHTS = {
    'Upvote Velocity (Upvotes/Hour)': 0.4,
    'Comment Velocity (Comments/Hour)': 0.3,
    'Upvote-to-View Ratio (%)': 0.2,
    'Upvote Ratio (%)': 0.1
}

def calc_topic_metrics(agg_df, platform, metric_weights):
    """Apply custom metrics equation to aggregated topic metrics"""
    results = []
    for _, row in agg_df.iterrows():
        score = calc_platform_trend_score(row, metric_weights)
        results.append({
            'topic_id': row['topic_id'],
            f'{platform}_topic_score': score
        })
    return pd.DataFrame(results)

# STEP 7: Build and merge platform-specific topic DataFrames
def build_platform_topic_df(df, topics, topic_labels, topic_keywords_str, platform, agg_metrics_df):
    """
    Build a comprehensive topic DataFrame for each platform
    Args:
        df: Original DataFrame with content data
        topics: topic assignments for each row
        topic_labels: list of topic labels
        topic_keywords_str: list of comma-joined keyword strings
        platform: 'youtube' or 'reddit'
        agg_metrics_df: aggregated metrics DataFrame from Step 4
    Returns:
        DataFrame with all topic information and metrics
    """
    topic_nums = np.unique(topics)
    summary = []
    
    for i, topic_id in enumerate(topic_nums):
        mask = (topics == topic_id)
        topic_df = df[mask]
        
        # Get aggregated metrics for this topic
        topic_metrics = agg_metrics_df[agg_metrics_df['topic_id'] == topic_id]
        
        row = {
            'topic_num': topic_id,
            'topic_label': topic_labels[i] if i < len(topic_labels) else f'Topic_{topic_id}',
            'keywords': topic_keywords_str[i] if i < len(topic_keywords_str) else '',
            'topic_count': topic_df.shape[0],
            'is_youtube': 1 if platform == 'youtube' else 0,
            'is_reddit': 1 if platform == 'reddit' else 0
        }
        
        # Add platform-specific metrics
        if not topic_metrics.empty:
            for col in topic_metrics.columns:
                if col != 'topic_id':
                    row[f'{platform}_{col}'] = topic_metrics[col].iloc[0]
        
        summary.append(row)
    
    return pd.DataFrame(summary)

def save_and_merge_topic_dfs(yt_df, rd_df, yt_topics, rd_topics, yt_labels, rd_labels, 
                            yt_keywords_str, rd_keywords_str, yt_agg_metrics, rd_agg_metrics):
    """Save platform-specific topic DataFrames and merge them with outer join"""
    # Create Output directory if it doesn't exist
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build platform-specific topic DataFrames
    yt_topic_df = build_platform_topic_df(yt_df, yt_topics, yt_labels, yt_keywords_str, 'youtube', yt_agg_metrics)
    rd_topic_df = build_platform_topic_df(rd_df, rd_topics, rd_labels, rd_keywords_str, 'reddit', rd_agg_metrics)
    
    # Save individual platform DataFrames to Output folder
    yt_output_path = os.path.join(output_dir, 'youtube_topics_aggregated.csv')
    rd_output_path = os.path.join(output_dir, 'reddit_topics_aggregated.csv')
    
    yt_topic_df.to_csv(yt_output_path, index=False)
    rd_topic_df.to_csv(rd_output_path, index=False)
    
    print(f"Saved YouTube topics: {len(yt_topic_df)} topics")
    print(f"Saved Reddit topics: {len(rd_topic_df)} topics")
    
    # Debug: Print column names before merge
    print(f"YouTube columns: {list(yt_topic_df.columns)}")
    print(f"Reddit columns: {list(rd_topic_df.columns)}")
    
    # METHOD 1: Simple concatenation (no topic_num matching)
    merged_df = pd.concat([yt_topic_df, rd_topic_df], ignore_index=True, sort=False)
    
    # Fill NaN values with 0
    merged_df = merged_df.fillna(0)
    
    # Get actual column names from merged DataFrame
    actual_columns = list(merged_df.columns)
    print(f"Merged columns: {actual_columns}")
    
    # Create column order based on actual existing columns
    column_order = []
    
    # 1. topic_num (always exists)
    if 'topic_num' in actual_columns:
        column_order.append('topic_num')
    
    # 2. Common topic info columns
    topic_info_cols = [
        'topic_label', 'keywords', 'topic_count',
        'is_youtube', 'is_reddit'
    ]
    
    for col in topic_info_cols:
        if col in actual_columns:
            column_order.append(col)
    
    # 3. YouTube metrics columns (from agg_metrics_df)
    yt_metric_cols = [col for col in actual_columns if col.startswith('youtube_')]
    column_order.extend(yt_metric_cols)
    
    # 4. Reddit metrics columns (from agg_metrics_df)
    rd_metric_cols = [col for col in actual_columns if col.startswith('reddit_')]
    column_order.extend(rd_metric_cols)
    
    # 5. Any remaining columns
    remaining_cols = [col for col in actual_columns if col not in column_order]
    column_order.extend(remaining_cols)
    
    print(f"Final column order: {column_order}")
    
    # Reorder DataFrame
    merged_df = merged_df[column_order]
    
    # Save merged DataFrame to Output folder
    merged_output_path = os.path.join(output_dir, 'merged_topics_aggregated.csv')
    merged_df.to_csv(merged_output_path, index=False)
    
    print(f"Merged DataFrame shape: {merged_df.shape}")
    print(f"Final columns: {list(merged_df.columns)}")
    
    return merged_df

# Advanced merge using similarity and LLM matching
def save_and_merge_topic_dfs_advanced(yt_df, rd_df, yt_topics, rd_topics, yt_labels, rd_labels, 
                                     yt_keywords_str, rd_keywords_str, yt_agg_metrics, rd_agg_metrics,
                                     similarity_threshold=0.6, use_llm=True, api_key=None):
    """
    Save platform-specific topic DataFrames and merge them using advanced matching
    Args:
        yt_df, rd_df: Original DataFrames
        yt_topics, rd_topics: Topic assignments
        yt_labels, rd_labels: Topic labels
        yt_keywords_str, rd_keywords_str: Keyword strings
        yt_agg_metrics, rd_agg_metrics: Aggregated metrics
        similarity_threshold: Minimum similarity for matching
        use_llm: Whether to use LLM for verification
        api_key: OpenAI API key for LLM matching
    Returns:
        DataFrame with matched topics
    """
    # Create Output directory if it doesn't exist
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build platform-specific topic DataFrames
    yt_topic_df = build_platform_topic_df(yt_df, yt_topics, yt_labels, yt_keywords_str, 'youtube', yt_agg_metrics)
    rd_topic_df = build_platform_topic_df(rd_df, rd_topics, rd_labels, rd_keywords_str, 'reddit', rd_agg_metrics)
    
    # Save individual platform DataFrames
    yt_output_path = os.path.join(output_dir, 'youtube_topics_aggregated.csv')
    rd_output_path = os.path.join(output_dir, 'reddit_topics_aggregated.csv')
    
    yt_topic_df.to_csv(yt_output_path, index=False)
    rd_topic_df.to_csv(rd_output_path, index=False)
    
    print(f"Saved YouTube topics: {len(yt_topic_df)} topics")
    print(f"Saved Reddit topics: {len(rd_topic_df)} topics")
    
    # Advanced topic matching
    print("Performing advanced topic matching...")
    matched_pairs, similarity_scores = advanced_topic_matching(
        yt_topic_df, rd_topic_df, similarity_threshold, use_llm, api_key
    )
    
    print(f"Found {len(matched_pairs)} matched topic pairs")
    
    # Create merged DataFrame based on matches
    merged_rows = []
    
    # Add matched topics
    for yt_idx, rd_idx in matched_pairs:
        yt_row = yt_topic_df.iloc[yt_idx].copy()
        rd_row = rd_topic_df.iloc[rd_idx].copy()
        
        # Create merged row
        merged_row = {
            'topic_num': f"YT{yt_row['topic_num']}_RD{rd_row['topic_num']}",
            'topic_label_yt': yt_row['topic_label'],
            'topic_label_rd': rd_row['topic_label'],
            'keywords_yt': yt_row['keywords'],
            'keywords_rd': rd_row['keywords'],
            'topic_count_yt': yt_row['topic_count'],
            'topic_count_rd': rd_row['topic_count'],
            'is_youtube': 1,
            'is_reddit': 1,
            'similarity_score': similarity_scores[(yt_idx, rd_idx)]
        }
        
        # Add YouTube metrics
        for col in yt_row.index:
            if col.startswith('youtube_'):
                merged_row[col] = yt_row[col]
        
        # Add Reddit metrics
        for col in rd_row.index:
            if col.startswith('reddit_'):
                merged_row[col] = rd_row[col]
        
        merged_rows.append(merged_row)
    
    # Add unmatched YouTube topics
    matched_yt_indices = {pair[0] for pair in matched_pairs}
    for idx, row in yt_topic_df.iterrows():
        if idx not in matched_yt_indices:
            merged_row = {
                'topic_num': f"YT{row['topic_num']}",
                'topic_label_yt': row['topic_label'],
                'topic_label_rd': '',
                'keywords_yt': row['keywords'],
                'keywords_rd': '',
                'topic_count_yt': row['topic_count'],
                'topic_count_rd': 0,
                'is_youtube': 1,
                'is_reddit': 0,
                'similarity_score': 0.0
            }
            
            # Add YouTube metrics
            for col in row.index:
                if col.startswith('youtube_'):
                    merged_row[col] = row[col]
            
            merged_rows.append(merged_row)
    
    # Add unmatched Reddit topics
    matched_rd_indices = {pair[1] for pair in matched_pairs}
    for idx, row in rd_topic_df.iterrows():
        if idx not in matched_rd_indices:
            merged_row = {
                'topic_num': f"RD{row['topic_num']}",
                'topic_label_yt': '',
                'topic_label_rd': row['topic_label'],
                'keywords_yt': '',
                'keywords_rd': row['keywords'],
                'topic_count_yt': 0,
                'topic_count_rd': row['topic_count'],
                'is_youtube': 0,
                'is_reddit': 1,
                'similarity_score': 0.0
            }
            
            # Add Reddit metrics
            for col in row.index:
                if col.startswith('reddit_'):
                    merged_row[col] = row[col]
            
            merged_rows.append(merged_row)
    
    # Create final DataFrame
    merged_df = pd.DataFrame(merged_rows)
    
    # Fill NaN values with 0
    merged_df = merged_df.fillna(0)
    
    # Save merged DataFrame
    merged_output_path = os.path.join(output_dir, 'merged_topics_advanced.csv')
    merged_df.to_csv(merged_output_path, index=False)
    
    print(f"Advanced merged DataFrame shape: {merged_df.shape}")
    print(f"Matched topics: {len(matched_pairs)}")
    print(f"YouTube-only topics: {sum(merged_df['is_youtube'] == 1) - len(matched_pairs)}")
    print(f"Reddit-only topics: {sum(merged_df['is_reddit'] == 1) - len(matched_pairs)}")
    
    return merged_df

# Create tool for advanced merging
advanced_merge_tool = Tool.from_function(
    name="AdvancedTopicMerge",
    func=lambda yt_df, rd_df, yt_topics, rd_topics, yt_labels, rd_labels, 
                yt_keywords_str, rd_keywords_str, yt_agg_metrics, rd_agg_metrics: 
           save_and_merge_topic_dfs_advanced(
               yt_df, rd_df, yt_topics, rd_topics, yt_labels, rd_labels,
               yt_keywords_str, rd_keywords_str, yt_agg_metrics, rd_agg_metrics
           ),
    description="Merge YouTube and Reddit topics using advanced similarity and LLM matching"
)

# STEP 8: Rank topics by trend score for dashboard/analysis
def rank_topics(summary_df, score_col='Weighted Trend Score', top_n=20):
    """Rank topics by the specified trend score column (descending)"""
    ranked = summary_df.copy()
    ranked = ranked.sort_values(by=score_col, ascending=False)
    ranked['rank'] = range(1, len(ranked)+1)
    if top_n:
        ranked = ranked.head(top_n)
    return ranked

# STEP 9: Trend Score Calculation
# Example platform weights by Content Type
PLATFORM_WEIGHTS_TEXT = {'youtube': 0.4, 'reddit': 0.6}  # For text/image+text creators
PLATFORM_WEIGHTS_VIDEO = {'youtube': 0.6, 'reddit': 0.4}  # For video creators

# Calculate weighted trend score for a single platform
def calc_platform_trend_score(row, metric_weights):
    return sum(row.get(metric, 0) * weight for metric, weight in metric_weights.items())

# Calculate final trend score with platform weights
def calc_final_trend_score(youtube_score, reddit_score, platform_weights):
    return youtube_score * platform_weights['youtube'] + reddit_score * platform_weights['reddit']

# Main step 7 function: expects topic-level aggregation (topic_id as key)
def calc_trend_scores_by_topic(youtube_topic_scores, reddit_topic_scores, platform_weights):
    """Calculate trend scores by topic using metric and platform weights"""
    # topic_scores: dict {topic_id: platform_score}
    all_topics = set(youtube_topic_scores.keys()) | set(reddit_topic_scores.keys())
    results = {}
    for topic_id in all_topics:
        yt_score = youtube_topic_scores.get(topic_id, 0)
        rd_score = reddit_topic_scores.get(topic_id, 0)
        # 1. Simple summation (no platform weights)
        simple_sum = yt_score + rd_score
        # 2. Weighted trend score (with platform weights)
        weighted_score = yt_score * platform_weights['youtube'] + rd_score * platform_weights['reddit']
        results[topic_id] = {
            'YouTube Score': yt_score,
            'Reddit Score': rd_score,
            'Simple Summation Score': simple_sum,
            'Weighted Trend Score': weighted_score
        }
    return results