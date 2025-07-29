#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain Pipeline Test with Advanced Topic Matching - Step by Step
Test file for the advanced topic matching version of langchain_workflow.py step by step.
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
    print("=== LangChain Pipeline Test with Advanced Topic Matching Started ===\n")
    
    # Setup OpenAI API Key (if not already set)
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API Key not found in environment variables.")
        print("Please set your OpenAI API key:")
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("API key set successfully!")
        else:
            print("No API key provided. LLM features will be disabled.")
    else:
        print("OpenAI API Key found in environment variables.")
    
    print()
    
    # Step 1: Data Collection (CSV with metrics)
    print("=== Step 1: Data Collection ===")
    yt_df = youtube_tool.func()
    rd_df = reddit_tool.func()
    
    print(f"YouTube data: {yt_df.shape}")
    print(f"Reddit data: {rd_df.shape}")
    print(f"YouTube columns: {list(yt_df.columns)}")
    print(f"Reddit columns: {list(rd_df.columns)}\n")
    
    # Step 2: LDA Preprocessing
    print("=== Step 2: LDA Preprocessing ===")
    
    # Create platform-specific preprocessing functions
    def preprocess_youtube(df):
        return preprocess_text(df, ["Title", "Description"])
    
    def preprocess_reddit(df):
        return preprocess_text(df, ["title", "selftext"])
    
    # YouTube preprocessing
    yt_df_processed = preprocess_youtube(yt_df)
    print(f"YouTube preprocessing completed: {len(yt_df_processed)} documents")
    
    # Reddit preprocessing
    rd_df_processed = preprocess_reddit(rd_df)
    print(f"Reddit preprocessing completed: {len(rd_df_processed)} documents")
    
    # Filter out empty documents from original dataframes
    yt_non_empty_mask = [len(doc) > 0 for doc in yt_df_processed]
    rd_non_empty_mask = [len(doc) > 0 for doc in rd_df_processed]
    
    yt_df_filtered = yt_df[yt_non_empty_mask].reset_index(drop=True)
    rd_df_filtered = rd_df[rd_non_empty_mask].reset_index(drop=True)
    
    # Filter processed texts to only include non-empty documents
    yt_df_processed = [doc for doc in yt_df_processed if len(doc) > 0]
    rd_df_processed = [doc for doc in rd_df_processed if len(doc) > 0]
    
    print(f"YouTube non-empty documents: {len(yt_df_processed)}")
    print(f"Reddit non-empty documents: {len(rd_df_processed)}")
    print()
    
    # Step 3: LDA Topic Modeling
    print("=== Step 3: LDA Topic Modeling ===")
    
    # YouTube topic modeling (yt_df_processed is already a list of preprocessed texts)
    yt_topics, yt_model, yt_labels, yt_keywords, yt_keywords_str = topic_modeling_tool.func(yt_df_processed)
    print(f"YouTube topics count: {len(set(yt_topics))}")
    print(f"YouTube labels example: {yt_labels[:5]}")
    
    # Reddit topic modeling (rd_df_processed is already a list of preprocessed texts)
    rd_topics, rd_model, rd_labels, rd_keywords, rd_keywords_str = topic_modeling_tool.func(rd_df_processed)
    print(f"Reddit topics count: {len(set(rd_topics))}")
    print(f"Reddit labels example: {rd_labels[:5]}\n")
    
    # Step 3.5: LLM Topic Labeling (Optional)
    print("=== Step 3.5: LLM Topic Labeling ===")
    
    # Check if API key is available
    if os.getenv("OPENAI_API_KEY"):
        # YouTube topic labeling
        yt_topic_counts = pd.Series(yt_topics).value_counts().sort_index().tolist()
        try:
            yt_natural_labels = topic_labeling_tool.func(yt_keywords_str, yt_topic_counts)
            print(f"YouTube natural labels: {yt_natural_labels[:5]}")
            yt_labels = yt_natural_labels  # Use natural labels
        except Exception as e:
            print(f"LLM labeling failed (using default labels): {e}")
        
        # Reddit topic labeling
        rd_topic_counts = pd.Series(rd_topics).value_counts().sort_index().tolist()
        try:
            rd_natural_labels = topic_labeling_tool.func(rd_keywords_str, rd_topic_counts)
            print(f"Reddit natural labels: {rd_natural_labels[:5]}")
            rd_labels = rd_natural_labels  # Use natural labels
        except Exception as e:
            print(f"LLM labeling failed (using default labels): {e}")
    else:
        print("OpenAI API key not available. Using default keyword-based labels.")
    print()
    
    # Step 4: Topic-level Metric Aggregation
    print("=== Step 4: Topic-level Metric Aggregation ===")
    
    # YouTube metric aggregation (using filtered dataframe)
    yt_metrics_cols = ['View Count', 'Like Count', 'Comment Count', 
                       'View Velocity (Views/Hour)', 'Like Velocity (Likes/Hour)', 
                       'Comment Velocity (Comments/Hour)', 'Like-to-View Ratio (%)']
    yt_agg_metrics = aggregate_metrics_by_topic(yt_df_filtered, yt_topics, yt_metrics_cols, 'sum')
    print(f"YouTube aggregation completed: {yt_agg_metrics.shape}")
    
    # Reddit metric aggregation (using filtered dataframe)
    rd_metrics_cols = ['ups', 'score', 'num_comments', 'total_awards_received',
                       'Upvote Velocity (Upvotes/Hour)', 'Comment Velocity (Comments/Hour)',
                       'Upvote-to-View Ratio (%)', 'Upvote Ratio (%)']
    rd_agg_metrics = aggregate_metrics_by_topic(rd_df_filtered, rd_topics, rd_metrics_cols, 'sum')
    print(f"Reddit aggregation completed: {rd_agg_metrics.shape}\n")
    
    # Step 5: Apply Platform-specific Metric Weights
    print("=== Step 5: Apply Platform-specific Metric Weights ===")
    
    # Apply YouTube weights
    yt_topic_scores = calc_topic_metrics(yt_agg_metrics, 'youtube', YOUTUBE_METRIC_WEIGHTS)
    print(f"YouTube topic scores: {yt_topic_scores.shape}")
    
    # Apply Reddit weights
    rd_topic_scores = calc_topic_metrics(rd_agg_metrics, 'reddit', REDDIT_METRIC_WEIGHTS)
    print(f"Reddit topic scores: {rd_topic_scores.shape}\n")
    
    # Step 6: Create and Merge Platform-specific Topic DataFrames with Advanced Matching
    print("=== Step 6: Create and Merge Platform-specific Topic DataFrames with Advanced Matching ===")
    
    # Use advanced topic matching instead of simple concatenation
    merged_df = save_and_merge_topic_dfs_advanced(
        yt_df_filtered, rd_df_filtered, yt_topics, rd_topics, yt_labels, rd_labels,
        yt_keywords_str, rd_keywords_str, yt_agg_metrics, rd_agg_metrics,
        similarity_threshold=0.3, use_llm=True
    )
    
    print(f"Merged DataFrame: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
    print("\nMerged DataFrame top 5 rows:")
    print(merged_df.head())
    print()
    
    # Step 7: Topic Ranking
    print("=== Step 7: Topic Ranking ===")
    print("Ranking preparation completed")
    print("\nTop 10 topics:")
    
    # Check actual existing column names for advanced matching output
    available_cols = ['topic_num']
    if 'topic_label_yt' in merged_df.columns:
        available_cols.append('topic_label_yt')
    if 'topic_label_rd' in merged_df.columns:
        available_cols.append('topic_label_rd')
    if 'topic_count_yt' in merged_df.columns:
        available_cols.append('topic_count_yt')
    if 'topic_count_rd' in merged_df.columns:
        available_cols.append('topic_count_rd')
    if 'is_youtube' in merged_df.columns:
        available_cols.append('is_youtube')
    if 'is_reddit' in merged_df.columns:
        available_cols.append('is_reddit')
    if 'similarity_score' in merged_df.columns:
        available_cols.append('similarity_score')
    
    top_topics = merged_df.head(10)[available_cols]
    print(top_topics)
    print()
    
    # Final Results Check
    print("=== Final Results ===")
    print("Generated files:")
    print("- Output/youtube_topics_aggregated.csv")
    print("- Output/reddit_topics_aggregated.csv")
    print("- Output/merged_topics_advanced.csv")
    
    # Merged DataFrame Summary
    print(f"\nMerged DataFrame Summary:")
    print(f"Total topics: {len(merged_df)}")
    
    # Calculate using actual existing column names for advanced matching
    yt_only_count = 0
    rd_only_count = 0
    common_count = 0
    
    if 'is_youtube' in merged_df.columns and 'is_reddit' in merged_df.columns:
        yt_only_count = sum((merged_df['is_youtube'] == 1) & (merged_df['is_reddit'] == 0))
        rd_only_count = sum((merged_df['is_youtube'] == 0) & (merged_df['is_reddit'] == 1))
        common_count = sum((merged_df['is_youtube'] == 1) & (merged_df['is_reddit'] == 1))
    elif 'is_youtube' in merged_df.columns:
        yt_only_count = sum(merged_df['is_youtube'] == 1)
    elif 'is_reddit' in merged_df.columns:
        rd_only_count = sum(merged_df['is_reddit'] == 1)
    
    print(f"YouTube-only topics: {yt_only_count}")
    print(f"Reddit-only topics: {rd_only_count}")
    print(f"Common topics (matched): {common_count}")
    
    # Show similarity scores for matched topics
    if 'similarity_score' in merged_df.columns:
        matched_topics = merged_df[merged_df['is_youtube'] == 1][merged_df['is_reddit'] == 1]
        if len(matched_topics) > 0:
            print(f"\nMatched topics similarity scores:")
            print(matched_topics[['topic_num', 'topic_label_yt', 'topic_label_rd', 'similarity_score']].head())
    
    print("\n=== LangChain Pipeline Test with Advanced Topic Matching Completed ===")

if __name__ == "__main__":
    main() 