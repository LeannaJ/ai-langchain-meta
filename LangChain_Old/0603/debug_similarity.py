#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Similarity Score Issues
Test script to debug why similarity scores are coming out as 0.
"""

import pandas as pd
import numpy as np
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Or set environment variables manually.")

from langchain_workflow import *

def debug_similarity_calculation():
    """Debug the similarity calculation function"""
    print("=== Debugging Similarity Calculation ===")
    
    # Test cases
    test_cases = [
        ("ai, machine learning, artificial intelligence", "machine learning, ai, artificial intelligence"),
        ("gaming, video games, esports", "video games, gaming, esports"),
        ("cooking, recipes, food", "recipes, cooking, food"),
        ("tech, technology, innovation", "technology, tech, innovation"),
        ("ai, ml, artificial intelligence", "machine learning, ai, artificial intelligence"),
        ("", "ai, machine learning"),  # Empty keywords
        ("ai", "machine learning"),    # Single word
    ]
    
    for i, (kw1, kw2) in enumerate(test_cases):
        similarity = calculate_keyword_similarity(kw1, kw2)
        print(f"Test {i+1}: '{kw1}' vs '{kw2}' = {similarity:.3f}")
    
    print()

def debug_real_data_similarity():
    """Debug similarity with real data"""
    print("=== Debugging Real Data Similarity ===")
    
    # Get real data
    print("Fetching real data...")
    yt_df = youtube_tool.func()
    rd_df = reddit_tool.func()
    
    # Preprocess
    yt_df_processed = preprocess_text(yt_df, ["Title", "Description"])
    rd_df_processed = preprocess_text(rd_df, ["title", "selftext"])
    
    # Filter empty documents
    yt_non_empty_mask = [len(doc) > 0 for doc in yt_df_processed]
    rd_non_empty_mask = [len(doc) > 0 for doc in rd_df_processed]
    
    yt_df_filtered = yt_df[yt_non_empty_mask].reset_index(drop=True)
    rd_df_filtered = rd_df[rd_non_empty_mask].reset_index(drop=True)
    
    yt_df_processed = [doc for doc in yt_df_processed if len(doc) > 0]
    rd_df_processed = [doc for doc in rd_df_processed if len(doc) > 0]
    
    # Topic modeling
    print("Topic modeling...")
    yt_topics, yt_model, yt_labels, yt_keywords, yt_keywords_str = topic_modeling_tool.func(yt_df_processed)
    rd_topics, rd_model, rd_labels, rd_keywords, rd_keywords_str = topic_modeling_tool.func(rd_df_processed)
    
    # Build topic DataFrames
    yt_metrics_cols = ['View Count', 'Like Count', 'Comment Count']
    yt_agg_metrics = aggregate_metrics_by_topic(yt_df_filtered, yt_topics, yt_metrics_cols, 'sum')
    
    rd_metrics_cols = ['ups', 'score', 'num_comments']
    rd_agg_metrics = aggregate_metrics_by_topic(rd_df_filtered, rd_topics, rd_metrics_cols, 'sum')
    
    yt_topic_df = build_platform_topic_df(yt_df_filtered, yt_topics, yt_labels, yt_keywords_str, 'youtube', yt_agg_metrics)
    rd_topic_df = build_platform_topic_df(rd_df_filtered, rd_topics, rd_labels, rd_keywords_str, 'reddit', rd_agg_metrics)
    
    print(f"YouTube topics: {len(yt_topic_df)}")
    print(f"Reddit topics: {len(rd_topic_df)}")
    
    # Check keyword formats
    print("\n=== Keyword Format Check ===")
    print("YouTube keywords examples:")
    for i in range(min(5, len(yt_topic_df))):
        print(f"  Topic {i}: '{yt_topic_df.iloc[i]['keywords']}'")
    
    print("\nReddit keywords examples:")
    for i in range(min(5, len(rd_topic_df))):
        print(f"  Topic {i}: '{rd_topic_df.iloc[i]['keywords']}'")
    
    # Test similarity between first few topics
    print("\n=== Similarity Test with Real Data ===")
    for yt_idx in range(min(3, len(yt_topic_df))):
        for rd_idx in range(min(3, len(rd_topic_df))):
            yt_kw = yt_topic_df.iloc[yt_idx]['keywords']
            rd_kw = rd_topic_df.iloc[rd_idx]['keywords']
            similarity = calculate_keyword_similarity(yt_kw, rd_kw)
            print(f"YouTube Topic {yt_idx} vs Reddit Topic {rd_idx}: {similarity:.3f}")
            print(f"  YT: '{yt_kw}'")
            print(f"  RD: '{rd_kw}'")
            print()
    
    # Test with different thresholds
    print("=== Testing Different Thresholds ===")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for threshold in thresholds:
        matched_pairs, similarity_scores = advanced_topic_matching(
            yt_topic_df, rd_topic_df, threshold, use_llm=False
        )
        print(f"Threshold {threshold}: {len(matched_pairs)} matches")
        if len(matched_pairs) > 0:
            print(f"  Sample similarity scores: {list(similarity_scores.values())[:3]}")
    
    return yt_topic_df, rd_topic_df

def main():
    print("=== Similarity Score Debug Started ===\n")
    
    # Test 1: Basic similarity calculation
    debug_similarity_calculation()
    
    # Test 2: Real data similarity
    yt_topic_df, rd_topic_df = debug_real_data_similarity()
    
    print("=== Similarity Score Debug Completed ===")

if __name__ == "__main__":
    main() 