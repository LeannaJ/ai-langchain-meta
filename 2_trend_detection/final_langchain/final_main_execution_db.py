#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Execution Script for New Multi-Platform Trend Analysis Workflow
This script handles the execution of the optimized workflow structure.
"""

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import all necessary functions from the workflow module
from final_langchain_workflow_db import (
    # Step 1: File loading
    get_latest_csv,
    fetch_youtube_popular,
    fetch_reddit_hot,
    fetch_tiktok_hashtags,
    fetch_twitter_trends,
    fetch_google_trends,
    
    # Step 2: Data preprocessing
    preprocess_platform_data,
    
    # Step 3: LDA Topic Modeling
    initialize_nltk_components,
    preprocess_text,
    topic_modeling,
    create_topic_labeling_chain,
    label_topics_with_llm,
    create_topic_dataframe,
    
    # Step 4: Aggregation
    consolidate_platform_data,
    
    # Step 5: Scaling
    apply_minmax_scaling,
    
    # Step 6: Similar Topic Finding
    calculate_keyword_similarity,
    cluster_similar_topics,
    llm_enhanced_grouping,
    save_grouping_results,
    
    # Step 7: Category Assignment
    create_category_classification_chain,
    batch_classify_topics,
    add_category_column,
    
    # Step 8: Score Calculation
    calculate_trend_scores,
    add_cross_platform_bonus,
    save_consolidated_scores,
    upload_to_supabase
)

def main():
    """
    Main execution function for the new workflow structure
    """
    print("=== Starting New Multi-Platform Trend Analysis Workflow ===")
    
    # Load environment variables
    load_dotenv()
    
    # Generate workflow timestamp
    workflow_timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    print(f"Workflow timestamp: {workflow_timestamp}")
    
    # Define media types for final score calculation
    media_types = ['equal', 'video', 'text', 'image']
    
    try:
        # ========================================
        # STEPS 1-7: COMMON PROCESSING (Run Once)
        # ========================================
        
        print("\n" + "="*50)
        print("STEPS 1-7: COMMON PROCESSING")
        print("="*50)
        
        # Step 1: Load latest files from each platform
        print("\nStep 1: Loading latest files...")
        platform_data = {}
        
        # Load YouTube data
        try:
            youtube_data = fetch_youtube_popular()
            if youtube_data is not None and not youtube_data.empty:
                platform_data['youtube'] = youtube_data
                print("  ✓ YouTube data loaded")
        except Exception as e:
            print(f"  ⚠️  YouTube data loading failed: {e}")
        
        # Load Reddit data
        try:
            reddit_data = fetch_reddit_hot()
            if reddit_data is not None and not reddit_data.empty:
                platform_data['reddit'] = reddit_data
                print("  ✓ Reddit data loaded")
        except Exception as e:
            print(f"  ⚠️  Reddit data loading failed: {e}")
        
        # Load TikTok data
        try:
            tiktok_data = fetch_tiktok_hashtags()
            if tiktok_data is not None and not tiktok_data.empty:
                platform_data['tiktok'] = tiktok_data
                print("  ✓ TikTok data loaded")
        except Exception as e:
            print(f"  ⚠️  TikTok data loading failed: {e}")
        
        # Load Twitter data
        try:
            twitter_data = fetch_twitter_trends()
            if twitter_data is not None and not twitter_data.empty:
                platform_data['twitter'] = twitter_data
                print("  ✓ Twitter data loaded")
        except Exception as e:
            print(f"  ⚠️  Twitter data loading failed: {e}")
        
        # Load Google Trends data
        try:
            google_data = fetch_google_trends()
            if google_data is not None and not google_data.empty:
                platform_data['google_trends'] = google_data
                print("  ✓ Google Trends data loaded")
        except Exception as e:
            print(f"  ⚠️  Google Trends data loading failed: {e}")
        
        # Step 2: Preprocess all platform data
        print("\nStep 2: Preprocessing platform data...")
        preprocessed_data = preprocess_platform_data(platform_data)
        
        # Step 3: LDA Topic Modeling for YouTube and Reddit
        print("\nStep 3: LDA Topic Modeling for YouTube and Reddit...")
        
        # 3.1: Initialize NLTK components
        print("  3.1: Initializing NLTK components...")
        initialize_nltk_components()
        
        # 3.2: Topic modeling for YouTube
        if 'youtube' in preprocessed_data and preprocessed_data['youtube'] is not None:
            print("  3.2: YouTube topic modeling...")
            youtube_df = preprocessed_data['youtube']
            youtube_text_cols = ['Title', 'Description', 'Tags']
            youtube_texts = preprocess_text(youtube_df, youtube_text_cols)
            youtube_topics, _, youtube_topic_labels, _, youtube_keywords_str = topic_modeling(youtube_texts)
            youtube_topic_counts = [youtube_topics.count(i) for i in range(len(youtube_topic_labels))]
            
            # 3.3: LLM labeling for YouTube
            print("  3.3: YouTube LLM labeling...")
            youtube_labeled_topics = label_topics_with_llm(youtube_keywords_str, youtube_topic_counts)
            
            # Create topic DataFrame for YouTube
            youtube_topic_df = create_topic_dataframe(
                youtube_topics, 
                youtube_labeled_topics, 
                youtube_keywords_str, 
                youtube_df, 
                'youtube'
            )
            preprocessed_data['youtube'] = youtube_topic_df
        
        # 3.2: Topic modeling for Reddit
        if 'reddit' in preprocessed_data and preprocessed_data['reddit'] is not None:
            print("  3.2: Reddit topic modeling...")
            reddit_df = preprocessed_data['reddit']
            reddit_text_cols = ['title', 'selftext']
            reddit_texts = preprocess_text(reddit_df, reddit_text_cols)
            reddit_topics, _, reddit_topic_labels, _, reddit_keywords_str = topic_modeling(reddit_texts)
            reddit_topic_counts = [reddit_topics.count(i) for i in range(len(reddit_topic_labels))]
            
            # 3.3: LLM labeling for Reddit
            print("  3.3: Reddit LLM labeling...")
            reddit_labeled_topics = label_topics_with_llm(reddit_keywords_str, reddit_topic_counts)
            
            # Create topic DataFrame for Reddit
            reddit_topic_df = create_topic_dataframe(
                reddit_topics, 
                reddit_labeled_topics, 
                reddit_keywords_str, 
                reddit_df, 
                'reddit'
            )
            preprocessed_data['reddit'] = reddit_topic_df
        
        # Step 4: Aggregate all platform data
        print("\nStep 4: Aggregating platform data...")
        consolidated_df = consolidate_platform_data(preprocessed_data)
        
        # Step 5: Normalize metrics
        print("\nStep 5: Normalizing metrics...")
        normalized_df = apply_minmax_scaling(consolidated_df)
        
        # Step 6: Similar Topic Finding
        print("\nStep 6: Similar Topic Finding...")
        
        # 6.1: First-pass similarity calculation and clustering
        print("  6.1: First-pass similarity calculation and clustering...")
        clustered_results, final_df = cluster_similar_topics(normalized_df)
        
        # 6.3: Save grouping results
        print("  6.3: Saving grouping results...")
        detailed_data_filepath = save_grouping_results(final_df, workflow_timestamp)
        
        # Step 7: Category Assignment
        print("\nStep 7: Category Assignment...")
        categorized_data_filepath = add_category_column(detailed_data_filepath)
        
        # ========================================
        # STEP 8: SCORE CALCULATION (Media Type Specific)
        # ========================================
        
        print("\n" + "="*50)
        print("STEP 8: SCORE CALCULATION (Media Type Specific)")
        print("="*50)
        
        final_results = {}
        
        for media_type in media_types:
            print(f"\nProcessing {media_type} media type...")
            
            # 8.1: Weight assignment and S_p score calculation
            print(f"  8.1: Weight assignment and S_p score calculation...")
            df_with_scores, final_scores = calculate_trend_scores(
                pd.read_csv(categorized_data_filepath), media_type
            )
            
            # 8.2: Cross-platform bonus calculation
            print(f"  8.2: Cross-platform bonus calculation...")
            df_with_bonus = add_cross_platform_bonus(df_with_scores)
            
            # 8.3: Save consolidated scores with cross-platform bonus
            print(f"  8.3: Saving consolidated scores...")
            final_filepath = save_consolidated_scores(df_with_bonus, media_type, workflow_timestamp)
            final_results[media_type] = final_filepath
            
            # 8.4:Upload to Supabase
            table_name = f"trends_{media_type}"  # trends_equal, trends_video, trends_text, trends_image
            print(f"Uploading {media_type} results to Supabase...")
            upload_to_supabase(df_with_bonus, table_name)
        
        # ========================================
        # WORKFLOW COMPLETION SUMMARY
        # ========================================
        
        print("\n" + "="*50)
        print("WORKFLOW COMPLETION SUMMARY")
        print("="*50)
        
        print(f"\n✅ Workflow completed successfully!")
        print(f"📅 Timestamp: {workflow_timestamp}")
        print(f"📊 Total files generated: {len(final_results) + 2}")  # +2 for detailed_data and categorized_data
        
        print(f"\n📁 Generated files:")
        print(f"  • Detailed data with grouping: {detailed_data_filepath}")
        print(f"  • Categorized data: {categorized_data_filepath}")
        
        for media_type, filepath in final_results.items():
            print(f"  • {media_type} weight results: {filepath}")
        
        print(f"\n🎯 Media types processed: {', '.join(media_types)}")
        print(f"📈 Cross-platform bonus applied: Yes")
        print(f"🏷️  Category classification: Yes")
        
        return final_results
        
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
