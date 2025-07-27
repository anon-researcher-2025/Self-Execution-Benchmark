#!/usr/bin/env python3
"""
Script to identify and reprocess failed LLM calls from question_sets.csv files.
This script distinguishes between different types of failures and creates
a reprocessing queue for API failures specifically.
"""

import os
import pandas as pd
import json
import re
from pathlib import Path

def analyze_failures(csv_path):
    """
    Analyze the question_sets.csv file to identify different types of failures.
    
    Returns:
        dict: Contains lists of different failure types
    """
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found")
        return None
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # API call failures - explicit failure message
    api_failures = df[df["raw_llm_ranking_output"] == "Combined prompt LLM call failed or returned no answer"]
    
    # Zero token responses (legitimate but need to distinguish)
    zero_token_responses = df[
        (df["combined_duration"] == 0) & 
        (df["raw_llm_ranking_output"] != "Combined prompt LLM call failed or returned no answer") &
        (df["raw_llm_ranking_output"].notna())
    ]
    
    # Parsing failures - have raw output but no valid ranking
    parsing_failures = df[
        (df["llm_ranking"].isna() | (df["llm_ranking"] == "")) & 
        (df["raw_llm_ranking_output"].notna()) & 
        (df["raw_llm_ranking_output"] != "Combined prompt LLM call failed or returned no answer") &
        (df["combined_duration"] > 0)
    ]
    
    # Empty or null responses
    empty_responses = df[
        (df["raw_llm_ranking_output"].isna()) | 
        (df["raw_llm_ranking_output"] == "") |
        (df["raw_llm_ranking_output"] == "nan")
    ]
    
    results = {
        "api_failures": api_failures,
        "zero_token_responses": zero_token_responses, 
        "parsing_failures": parsing_failures,
        "empty_responses": empty_responses,
        "total_rows": len(df)
    }
    
    return results

def create_reprocess_csv(failure_groups, model_name, output_dir):
    """
    Create a CSV file for reprocessing the failed groups.
    This CSV will be compatible with the existing reprocess script format.
    """
    if failure_groups.empty:
        print("No failure groups to reprocess")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with model name and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.replace('/', '_').replace(':', '_')}_failures_reprocess_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Save the group IDs that need reprocessing
    reprocess_data = []
    for _, row in failure_groups.iterrows():
        reprocess_data.append({
            "group_id": row["group_id"],
            "model": row["model"],
            "failure_type": "api_failure",
            "original_output": row["raw_llm_ranking_output"]
        })
    
    df_reprocess = pd.DataFrame(reprocess_data)
    df_reprocess.to_csv(output_path, index=False)
    
    print(f"Created reprocess file: {output_path}")
    return output_path

def analyze_all_models():
    """
    Analyze all model results in the results directory
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found")
        return
    
    # Create output directory for reprocess files
    reprocess_output_dir = "reprocess_failures"
    os.makedirs(reprocess_output_dir, exist_ok=True)
    
    all_analysis = {}
    
    # Walk through all model directories
    for root, dirs, files in os.walk(results_dir):
        if "question_sets.csv" in files:
            csv_path = os.path.join(root, "question_sets.csv")
            
            # Extract model name from path
            relative_path = os.path.relpath(root, results_dir)
            model_name = relative_path.replace(os.sep, "/")
            
            print(f"\n=== Analyzing {model_name} ===")
            
            analysis = analyze_failures(csv_path)
            if analysis is None:
                continue
                
            all_analysis[model_name] = analysis
            
            # Print summary
            print(f"Total rows: {analysis['total_rows']}")
            print(f"API failures: {len(analysis['api_failures'])}")
            print(f"Zero token responses: {len(analysis['zero_token_responses'])}")
            print(f"Parsing failures: {len(analysis['parsing_failures'])}")
            print(f"Empty responses: {len(analysis['empty_responses'])}")
            
            # Create reprocess file for API failures
            if len(analysis['api_failures']) > 0:
                print(f"\nAPI failure group IDs for {model_name}:")
                for idx, row in analysis['api_failures'].iterrows():
                    print(f"  {row['group_id']}")
                
                reprocess_file = create_reprocess_csv(
                    analysis['api_failures'], 
                    model_name, 
                    reprocess_output_dir
                )
            
            # Show some examples of parsing failures (first 3)
            if len(analysis['parsing_failures']) > 0:
                print(f"\nSample parsing failures for {model_name}:")
                for idx, row in analysis['parsing_failures'].head(3).iterrows():
                    print(f"  Group {row['group_id']}: {row['raw_llm_ranking_output'][:100]}...")
    
    # Create summary report
    summary_path = os.path.join(reprocess_output_dir, "failure_analysis_summary.json")
    summary_data = {}
    for model_name, analysis in all_analysis.items():
        summary_data[model_name] = {
            "total_rows": analysis['total_rows'],
            "api_failures": len(analysis['api_failures']),
            "zero_token_responses": len(analysis['zero_token_responses']),
            "parsing_failures": len(analysis['parsing_failures']),
            "empty_responses": len(analysis['empty_responses'])
        }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n=== Summary saved to {summary_path} ===")
    return all_analysis

def create_improved_reprocess_script(template_file="TimeTest_parallel_reprocess.py"):
    """
    Create an improved reprocessing script with better error handling
    """
    if not os.path.exists(template_file):
        print(f"Template file {template_file} not found")
        return
    
    # Read the original file
    with open(template_file, 'r') as f:
        original_content = f.read()
    
    # Create improved ask_llm function
    improved_ask_llm = '''def ask_llm(prompt, model=model_to_use, temperature=1.0, max_retries=3):
    """
    Sends a prompt to the LLM and returns its answer along with the token count and error info.
    Dynamically determines the provider based on the model name.
    Returns: (text, tokens, error_info)
    """
    provider = 'openrouter'
    
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                # Handle OpenAI provider
                import openai
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                    text = response.choices[0].message.content
                    tokens = response.usage.completion_tokens
                    return text, tokens, None
                except openai.error.RateLimitError:
                    return None, None, "Rate limit exceeded"
                except openai.error.InvalidRequestError as e:
                    return None, None, f"Invalid request: {e}"
                except Exception as e:
                    return None, None, f"OpenAI API error: {e}"
                    
            elif provider == "openrouter":
                import requests
                headers = {
                    "Authorization": f"Bearer {openrouter_api_key_val}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                
                try:
                    api_response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=120  # 2 minute timeout
                    )
                    api_response.raise_for_status()
                    response_json = api_response.json()
                    
                    # Check for API errors in response
                    if "error" in response_json:
                        return None, None, f"API returned error: {response_json['error']}"
                    
                    # Check for empty response
                    if "choices" not in response_json or len(response_json["choices"]) == 0:
                        return None, None, "Empty choices in API response"
                        
                    text = response_json["choices"][0]["message"]["content"]
                    
                    # Handle missing or empty content
                    if text is None or text.strip() == "":
                        return "", 0, "Empty content returned by model"
                    
                    # Get token count
                    tokens = 0
                    if "usage" in response_json and "completion_tokens" in response_json["usage"]:
                        tokens = response_json["usage"]["completion_tokens"]
                    
                    if tokens == 0 and text.strip() != "":
                        return text, 0, "Zero tokens reported but content exists"
                        
                    return text, tokens, None  # Success case
                    
                except requests.exceptions.Timeout:
                    return None, None, "Request timeout"
                except requests.exceptions.HTTPError as http_err:
                    return None, None, f"HTTP error: {http_err}"
                except requests.exceptions.ConnectionError:
                    return None, None, "Connection error to API"
                except json.JSONDecodeError:
                    return None, None, "Invalid JSON response from API"
                except Exception as parse_err:
                    return None, None, f"API response parsing error: {parse_err}"
            else:
                return None, None, f"Unknown provider: {provider}"
                
        except Exception as e:
            error_msg = f"Attempt {attempt+1} failed: {e}"
            if attempt < max_retries - 1:
                print(f"Error: {error_msg}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None, None, f"Failed after {max_retries} attempts: {e}"
    
    return None, None, "Unknown error after max retries"'''
    
    # Create improved process_group_wrapper section for error handling
    improved_error_handling = '''    combined_answer, combined_tokens, error_info = ask_llm(combined_prompt, model=current_model_to_use)
    
    if combined_answer is None:
        error_type = "api_failure"
        if error_info:
            if "timeout" in error_info.lower():
                error_type = "timeout"
            elif "rate limit" in error_info.lower():
                error_type = "rate_limit"
            elif "connection" in error_info.lower():
                error_type = "connection_error"
        
        print(f"Skipping group {group_id_val} due to combined prompt failure: {error_info}")
        
        return {
            "group_id": group_id_val, 
            "status": "failed_combined_prompt", 
            "original_ids": original_ids, 
            "model": current_model_to_use,
            "individual_results_entries": [], 
            "question_sets_entry": None,
            "combined_ids_entry": combined_ids_entry, 
            "group_correct_count": 0,
            "num_questions_in_group": len(group),
            "score": None,
            "raw_combined_answer": f"API error: {error_info}",
            "error_type": error_type,
            "error_details": error_info
        }
    
    # Handle empty content with error info
    raw_ranking_output_for_csv = combined_answer
    if error_info:
        raw_ranking_output_for_csv = f"Warning: {error_info} | Content: {combined_answer}"'''
    
    # Create the new improved script
    new_script_path = "TimeTest_parallel_reprocess_improved.py"
    
    # Replace the ask_llm function and error handling
    new_content = original_content
    
    # Find and replace the ask_llm function
    ask_llm_pattern = r'def ask_llm\(.*?\n    return None, None'
    new_content = re.sub(ask_llm_pattern, improved_ask_llm.replace('\n', '\n'), new_content, flags=re.DOTALL)
    
    # Add error handling imports at the top
    import_section = '''import os
import sys
import time
import json
import re
import random
import multiprocessing
import requests
import pandas as pd
from datetime import datetime'''
    
    # Replace imports
    new_content = re.sub(r'from datasets import load_dataset.*?import pandas as pd', import_section, new_content, flags=re.DOTALL)
    
    with open(new_script_path, 'w') as f:
        f.write(new_content)
    
    print(f"Created improved reprocessing script: {new_script_path}")
    return new_script_path

if __name__ == "__main__":
    print("=== LLM Failure Analysis and Reprocessing Setup ===\n")
    
    # Analyze all models
    analysis_results = analyze_all_models()
    
    print("\n=== Creating Improved Reprocessing Script ===")
    improved_script = create_improved_reprocess_script()
    
    print("\n=== Next Steps ===")
    print("1. Review the failure analysis in 'reprocess_failures/' directory")
    print("2. Use the generated CSV files to identify which groups need reprocessing")
    print("3. Run the improved reprocessing script for better error handling")
    print("4. Check the 'failure_analysis_summary.json' for an overview of all failures")
