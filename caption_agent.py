#!/usr/bin/env python3
# caption_agent.py - Updated to call Abacus.AI Intelligent Caption Generator Workflow

import os
from abacusai import ApiClient
import argparse

def run_caption_workflow(topic, workflow_id='586927fa'):
    """
    Calls the Abacus.AI AI Workflow 'Intelligent Caption Generator' to generate viral Instagram captions.
    
    Args:
        topic (str): The trending topic (e.g., "Luigi Mangione").
        workflow_id (str): The ID of the workflow (default: 586927fa).
    
    Returns:
        dict: The workflow's output (e.g., {'summary': ..., 'sentiment': ..., 'captions': [...]})
    """
    # Load API key from environment (GitHub Secret or local .env)
    api_key = os.getenv('ABACUSAPIKEY')
    if not api_key:
        raise ValueError("ABACUSAPIKEY not set in environment.")
    
    client = ApiClient(api_key=api_key)
    
    # Prepare input for the workflow
    inputs = {'topic': topic}  # Matches our prompt; adjust if your workflow needs more params
    
    # Run the workflow (similar to run_agent, but for workflows)
    response = client.run_workflow(
        workflow_id=workflow_id,
        inputs=inputs,
        timeout=300  # For research steps
    )
    
    return response

# Example usage (integrate into trend_bot.py or tests)
if __name__ == "__main__":
    topic = "Luigi Mangione"  # Test topic
    try:
        result = run_caption_workflow(topic)
        print("Workflow Output:")
        print(result)  # Should print summary, sentiment, captions
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate viral captions for a topic.")
    parser.add_argument('--topic', type=str, default="Luigi Mangione", help="Trending topic")
    args = parser.parse_args()
    
    try:
        result = run_caption_workflow(args.topic)
        print("Workflow Output:")
        print(result)
        # Optional: Save to CSV
        # import json; with open('output.json', 'w') as f: json.dump(result, f)
    except Exception as e:
        print(f"Error: {e}")