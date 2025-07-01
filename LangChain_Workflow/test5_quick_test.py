#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test for Advanced Topic Matching
Fast version without LLM for demonstration.
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

def main():
    print("=== Quick Advanced Topic Matching Test Started ===\n")
    
    # Step 1: Data Collection (CSV with metrics)
    print("=== Step 1: Data Collection ===")
    yt_df = youtube_tool.func()
    rd_df = reddit_tool.func()
    
    print(f"YouTube data: {yt_df.shape}")
    print(f"Reddit data: {rd_df.shape}\n")
    
    # Step 2: LDA Preprocessing
    print("=== Step 2: LDA Preprocessing ===")
    
    # Create platform-specific preprocessing functions
    def preprocess_youtube(df):
        return preprocess_text(df, ["Title", "Description"])
    
    def preprocess_reddit(df):
        return preprocess_text(df, ["title", "selftext"])
    
    # YouTube preprocessing
    yt_df_processed = preprocess_youtube(yt_df)
    rd_df_processed = preprocess_reddit(rd_df)
    
    # Filter out empty documents
    yt_non_empty_mask = [len(doc) > 0 for doc in yt_df_processed]
    rd_non_empty_mask = [len(doc) > 0 for doc in rd_df_processed]
    
    yt_df_filtered = yt_df[yt_non_empty_mask].reset_index(drop=True)
    rd_df_filtered = rd_df[rd_non_empty_mask].reset_index(drop=True)
    
    yt_df_processed = [doc for doc in yt_df_processed if len(doc) > 0]
    rd_df_processed = [doc for doc in rd_df_processed if len(doc) > 0]
    
    print(f"YouTube non-empty documents: {len(yt_df_processed)}")
    print(f"Reddit non-empty documents: {len(rd_df_processed)}\n")
    
    # Step 3: LDA Topic Modeling
    print("=== Step 3: LDA Topic Modeling ===")
    
    # YouTube topic modeling
    yt_topics, yt_model, yt_labels, yt_keywords, yt_keywords_str = topic_modeling_tool.func(yt_df_processed)
    rd_topics, rd_model, rd_labels, rd_keywords, rd_keywords_str = topic_modeling_tool.func(rd_df_processed)
    
    print(f"YouTube topics count: {len(set(yt_topics))}")
    print(f"Reddit topics count: {len(set(rd_topics))}\n")
    
    # Step 4: Topic-level Metric Aggregation
    print("=== Step 4: Topic-level Metric Aggregation ===")
    
    # YouTube metric aggregation
    yt_metrics_cols = ['View Count', 'Like Count', 'Comment Count']
    yt_agg_metrics = aggregate_metrics_by_topic(yt_df_filtered, yt_topics, yt_metrics_cols, 'sum')
    
    # Reddit metric aggregation
    rd_metrics_cols = ['ups', 'score', 'num_comments']
    rd_agg_metrics = aggregate_metrics_by_topic(rd_df_filtered, rd_topics, rd_metrics_cols, 'sum')
    
    print(f"YouTube aggregation completed: {yt_agg_metrics.shape}")
    print(f"Reddit aggregation completed: {rd_agg_metrics.shape}\n")
    
    # Step 5: Apply Platform-specific Metric Weights
    print("=== Step 5: Apply Platform-specific Metric Weights ===")
    
    # Apply YouTube weights
    yt_topic_scores = calc_topic_metrics(yt_agg_metrics, 'youtube', YOUTUBE_METRIC_WEIGHTS)
    rd_topic_scores = calc_topic_metrics(rd_agg_metrics, 'reddit', REDDIT_METRIC_WEIGHTS)
    
    print(f"YouTube topic scores: {yt_topic_scores.shape}")
    print(f"Reddit topic scores: {rd_topic_scores.shape}\n")
    
    # Step 6: Quick Advanced Matching (No LLM)
    print("=== Step 6: Quick Advanced Matching (No LLM) ===")
    
    # Use higher threshold and no LLM for speed
    merged_df = save_and_merge_topic_dfs_advanced(
        yt_df_filtered, rd_df_filtered, yt_topics, rd_topics, yt_labels, rd_labels,
        yt_keywords_str, rd_keywords_str, yt_agg_metrics, rd_agg_metrics,
        similarity_threshold=0.25, use_llm=False  # Higher threshold, no LLM
    )
    
    print(f"Merged DataFrame: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
    
    # Show results
    print("\n=== Results Summary ===")
    print(f"Total topics: {len(merged_df)}")
    
    if 'is_youtube' in merged_df.columns and 'is_reddit' in merged_df.columns:
        yt_only_count = sum((merged_df['is_youtube'] == 1) & (merged_df['is_reddit'] == 0))
        rd_only_count = sum((merged_df['is_youtube'] == 0) & (merged_df['is_reddit'] == 1))
        common_count = sum((merged_df['is_youtube'] == 1) & (merged_df['is_reddit'] == 1))
        
        print(f"YouTube-only topics: {yt_only_count}")
        print(f"Reddit-only topics: {rd_only_count}")
        print(f"Common topics (matched): {common_count}")
        
        # Show some matched topics with similarity scores
        if 'similarity_score' in merged_df.columns:
            matched_topics = merged_df[merged_df['is_youtube'] == 1][merged_df['is_reddit'] == 1]
            if len(matched_topics) > 0:
                print(f"\nTop 5 matched topics with similarity scores:")
                print(matched_topics[['topic_num', 'topic_label_yt', 'topic_label_rd', 'similarity_score']].head())
    
    print("\n=== Quick Test Completed ===")

if __name__ == "__main__":
    main() 