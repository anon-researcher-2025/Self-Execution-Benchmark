#!/usr/bin/env python3
"""
Script to identify and re-run failed API calls with detailed error tracking.
This will help distinguish between different types of failures:
1. API connection errors
2. Zero tokens returned
3. Parsing failures
"""

import pandas as pd
import json
import os
import time
import requests
from datetime import datetime
import re

# Read API keys
key_file = "openai_key.txt"
if os.path.exists(key_file):
    with open(key_file, "r") as f:
        lines = f.read().strip().split('\n')
        openai_api_key_val = lines[0] if lines else None
        openrouter_api_key_val = lines[1] if len(lines) > 1 else None
else:
    raise FileNotFoundError(f"API key file '{key_file}' not found.")

def ask_llm_with_detailed_error_tracking(prompt, model, temperature=1.0, max_retries=3):
    """
    Enhanced LLM call with detailed error tracking.
    Returns: (response_text, token_count, error_type, error_details)
    """
    provider = 'openrouter'
    
    headers = {
        "Authorization": f"Bearer {openrouter_api_key_val}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries} for model {model}")
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Check HTTP status
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                if attempt == max_retries - 1:
                    return None, 0, "HTTP_ERROR", error_msg
                print(f"    HTTP error, retrying: {error_msg}")
                time.sleep(2 ** attempt)
                continue
            
            # Parse JSON response
            try:
                response_json = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                if attempt == max_retries - 1:
                    return None, 0, "JSON_DECODE_ERROR", error_msg
                print(f"    JSON decode error, retrying: {error_msg}")
                time.sleep(2 ** attempt)
                continue
            
            # Check for API errors in response
            if "error" in response_json:
                error_msg = f"API error: {response_json['error']}"
                if attempt == max_retries - 1:
                    return None, 0, "API_ERROR", error_msg
                print(f"    API error, retrying: {error_msg}")
                time.sleep(2 ** attempt)
                continue
            
            # Check for choices
            if "choices" not in response_json or len(response_json["choices"]) == 0:
                error_msg = "No choices in response"
                if attempt == max_retries - 1:
                    return None, 0, "NO_CHOICES", error_msg
                print(f"    No choices, retrying: {error_msg}")
                time.sleep(2 ** attempt)
                continue
            
            # Extract content
            content = response_json["choices"][0]["message"]["content"]
            if content is None:
                error_msg = "Content is None"
                if attempt == max_retries - 1:
                    return "", 0, "NULL_CONTENT", error_msg
                print(f"    Null content, retrying: {error_msg}")
                time.sleep(2 ** attempt)
                continue
            
            # Extract token count
            token_count = 0
            if "usage" in response_json and "completion_tokens" in response_json["usage"]:
                token_count = response_json["usage"]["completion_tokens"]
            
            # Check for zero tokens
            if token_count == 0:
                return content, 0, "ZERO_TOKENS", "Model returned 0 completion tokens"
            
            # Success case
            return content, token_count, None, None
            
        except requests.exceptions.Timeout:
            error_msg = "Request timeout"
            if attempt == max_retries - 1:
                return None, 0, "TIMEOUT", error_msg
            print(f"    Timeout, retrying: {error_msg}")
            time.sleep(2 ** attempt)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            if attempt == max_retries - 1:
                return None, 0, "CONNECTION_ERROR", error_msg
            print(f"    Connection error, retrying: {error_msg}")
            time.sleep(2 ** attempt)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if attempt == max_retries - 1:
                return None, 0, "UNEXPECTED_ERROR", error_msg
            print(f"    Unexpected error, retrying: {error_msg}")
            time.sleep(2 ** attempt)
    
    return None, 0, "MAX_RETRIES_EXCEEDED", f"Failed after {max_retries} attempts"

def extract_ranking_from_response(text):
    """Extract ranking from LLM response"""
    if not text:
        return None
    
    # Try to extract JSON
    try:
        # Look for JSON patterns
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
            if "ranking" in parsed:
                return ",".join(parsed["ranking"])
        
        # Look for direct JSON
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start != -1 and json_end != -1:
            json_text = text[json_start:json_end+1]
            parsed = json.loads(json_text)
            if "ranking" in parsed:
                return ",".join(parsed["ranking"])
                
    except json.JSONDecodeError:
        pass
    
    return None

def analyze_existing_failures():
    """Analyze existing CSV files to find failures"""
    results_dir = "results"
    failed_groups = []
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "question_sets.csv":
                csv_path = os.path.join(root, file)
                print(f"Analyzing {csv_path}")
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Find failures
                    failure_mask = df["raw_llm_ranking_output"] == "Combined prompt LLM call failed or returned no answer"
                    failures = df[failure_mask]
                    
                    for _, row in failures.iterrows():
                        failed_groups.append({
                            "csv_path": csv_path,
                            "group_id": row["group_id"],
                            "model": row["model"],
                            "original_score": row.get("score", "N/A")
                        })
                        
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
    
    return failed_groups

def rerun_failed_group(failed_group_info):
    """Re-run a single failed group with detailed tracking"""
    print(f"\n=== Re-running {failed_group_info['group_id']} with {failed_group_info['model']} ===")
    
    # Create a simple ranking prompt (you might want to customize this based on your actual prompt)
    prompt = """Please rank the following 4 questions (Q1, Q2, Q3, Q4) from fastest to slowest in terms of how many tokens you would need to provide a complete answer. Consider both the complexity of the reasoning required and the length of explanation needed.

Respond with a JSON object in this exact format:
{"ranking": ["Q1", "Q2", "Q3", "Q4"]}

Where the list shows questions ordered from fastest (fewest tokens) to slowest (most tokens) to answer completely."""
    
    # Make the API call with detailed tracking
    response_text, token_count, error_type, error_details = ask_llm_with_detailed_error_tracking(
        prompt, failed_group_info['model']
    )
    
    # Try to extract ranking
    extracted_ranking = None
    parsing_error = None
    if response_text:
        extracted_ranking = extract_ranking_from_response(response_text)
        if not extracted_ranking:
            parsing_error = "Could not extract valid ranking from response"
    
    return {
        "group_id": failed_group_info['group_id'],
        "model": failed_group_info['model'],
        "original_csv_path": failed_group_info['csv_path'],
        "rerun_timestamp": datetime.now().isoformat(),
        "response_text": response_text or "",
        "token_count": token_count,
        "error_type": error_type,
        "error_details": error_details or "",
        "extracted_ranking": extracted_ranking or "",
        "parsing_error": parsing_error or "",
        "rerun_success": error_type is None and extracted_ranking is not None
    }

def main():
    print("=== Identifying and Re-running Failed API Calls ===")
    
    # Find all failed groups
    print("Step 1: Analyzing existing results for failures...")
    failed_groups = analyze_existing_failures()
    
    if not failed_groups:
        print("No failed groups found!")
        return
    
    print(f"Found {len(failed_groups)} failed groups to re-run")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"rerun_failures_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Re-run each failed group
    print("\nStep 2: Re-running failed groups...")
    results = []
    
    for i, failed_group in enumerate(failed_groups):
        print(f"\nProgress: {i+1}/{len(failed_groups)}")
        result = rerun_failed_group(failed_group)
        results.append(result)
        
        # Small delay to be nice to the API
        time.sleep(1)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "rerun_results.csv")
    results_df.to_csv(results_file, index=False)
    
    # Create summary
    summary = {
        "total_reruns": len(results),
        "successful_reruns": sum(1 for r in results if r["rerun_success"]),
        "error_breakdown": {},
        "timestamp": timestamp
    }
    
    # Count error types
    for result in results:
        if result["error_type"]:
            error_type = result["error_type"]
            summary["error_breakdown"][error_type] = summary["error_breakdown"].get(error_type, 0) + 1
    
    summary_file = os.path.join(output_dir, "rerun_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Total re-runs: {summary['total_reruns']}")
    print(f"Successful re-runs: {summary['successful_reruns']}")
    print(f"Results saved to: {output_dir}/")
    print("\nError breakdown:")
    for error_type, count in summary["error_breakdown"].items():
        print(f"  {error_type}: {count}")

if __name__ == "__main__":
    main()
