#!/usr/bin/env python3
"""
Script to reprocess failed LLM calls with detailed error tracking.
This script focuses on analyzing and re-running only the groups that failed previously.
"""

import pandas as pd
import os
import requests
import json
import time
import re
from datetime import datetime
from pathlib import Path

# Configuration
BASE_RESULTS_DIR = "results"
OUTPUT_DIR = "reprocess_failures_detailed"
FAILURE_MESSAGE = "Combined prompt LLM call failed or returned no answer"

# Model selection - Change this to process a specific model
SELECTED_MODEL = "o4-mini-2025-04-16"  # Change this to the model you want to process

# Models and their paths
MODEL_PATHS = {
    "o4-mini-2025-04-16": "openai/o4-mini-2025-04-16/question_sets.csv",
    "claude-3.7-sonnet-thinking": "anthropic/claude-3.7-sonnet-thinking/question_sets.csv", 
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku/question_sets.csv",
    "deepseek-r1": "deepseek/deepseek-r1/question_sets.csv",
    "gemini-2.5-pro-preview": "google/gemini-2.5-pro-preview/question_sets.csv"
}

# Read API keys
def load_api_keys():
    key_file = "openai_key.txt"
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"API key file '{key_file}' not found. Please create it with your API keys.")
    
    with open(key_file, "r") as f:
        lines = f.read().strip().split('\n')
        openai_key = lines[0] if len(lines) > 0 else None
        openrouter_key = lines[1] if len(lines) > 1 else None
    
    return openai_key, openrouter_key

def ask_llm_with_detailed_tracking(prompt, model, temperature=1.0, max_retries=3):
    """
    Enhanced LLM call function that tracks detailed error information.
    """
    openai_key, openrouter_key = load_api_keys()
    
    # Determine provider and setup
    if model.startswith("o4-") or model.startswith("gpt-"):
        provider = "openai"
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }
        url = "https://api.openai.com/v1/chat/completions"
    else:
        provider = "openrouter"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/experiment-3",
            "X-Title": "LLM Ranking Experiment"
        }
        url = "https://openrouter.ai/api/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    
    # Add the correct token limit parameter based on model
    if model.startswith("o4-") or model.startswith("gpt-4"):
        payload["max_completion_tokens"] = 4000
    else:
        payload["max_tokens"] = 4000
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} for model {model}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            # Check HTTP status
            if response.status_code != 200:
                error_detail = f"HTTP {response.status_code}: {response.text}"
                print(f"HTTP Error: {error_detail}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None, None, f"HTTP_ERROR: {error_detail}"
            
            # Parse JSON response
            try:
                response_json = response.json()
            except json.JSONDecodeError as e:
                error_detail = f"JSON decode error: {e}"
                print(f"JSON Error: {error_detail}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None, f"JSON_PARSE_ERROR: {error_detail}"
            
            # Check for API errors in response
            if "error" in response_json:
                error_detail = response_json["error"]
                print(f"API Error: {error_detail}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, None, f"API_ERROR: {error_detail}"
            
            # Check for choices
            if "choices" not in response_json or len(response_json["choices"]) == 0:
                error_detail = "No choices in response"
                print(f"Empty Response: {error_detail}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None, f"EMPTY_RESPONSE: {error_detail}"
            
            # Get content
            content = response_json["choices"][0]["message"]["content"]
            if not content:
                return "", 0, "EMPTY_CONTENT: Model returned empty content"
            
            # Get token usage
            tokens = 0
            if "usage" in response_json and "completion_tokens" in response_json["usage"]:
                tokens = response_json["usage"]["completion_tokens"]
            
            if tokens == 0:
                return content, 0, "ZERO_TOKENS: Model returned content but 0 tokens"
            
            print(f"Success: {tokens} tokens returned")
            return content, tokens, None  # Success!
            
        except requests.exceptions.Timeout:
            error_detail = "Request timeout"
            print(f"Timeout: {error_detail}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, None, f"TIMEOUT: {error_detail}"
            
        except requests.exceptions.ConnectionError:
            error_detail = "Connection error"
            print(f"Connection Error: {error_detail}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, None, f"CONNECTION_ERROR: {error_detail}"
            
        except Exception as e:
            error_detail = f"Unexpected error: {e}"
            print(f"Unexpected Error: {error_detail}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, None, f"UNEXPECTED_ERROR: {error_detail}"
    
    return None, None, f"FAILED_AFTER_RETRIES: All {max_retries} attempts failed"

def extract_ranking_from_response(response_text):
    """Extract ranking from LLM response."""
    if not response_text:
        return None, "NO_RESPONSE"
    
    # Try to find JSON first
    json_pattern = r'\{[^}]*"ranking"[^}]*\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for match in json_matches:
        try:
            data = json.loads(match)
            if "ranking" in data:
                ranking = data["ranking"]
                if isinstance(ranking, list) and len(ranking) == 4:
                    return ",".join(ranking), "JSON_SUCCESS"
        except:
            continue
    
    # Try to find comma-separated ranking
    ranking_patterns = [
        r'(?:ranking|answer)["\s:]*\[?"?([Q1-4,\s]+)"?\]?',
        r'([Q1-4],\s*[Q1-4],\s*[Q1-4],\s*[Q1-4])',
        r'Q[1-4]\s*,\s*Q[1-4]\s*,\s*Q[1-4]\s*,\s*Q[1-4]'
    ]
    
    for pattern in ranking_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            cleaned = re.sub(r'["\[\]]', '', match).strip()
            parts = [part.strip() for part in cleaned.split(',')]
            if len(parts) == 4 and all(part in ['Q1', 'Q2', 'Q3', 'Q4'] for part in parts):
                return ",".join(parts), "PATTERN_SUCCESS"
    
    return None, "PARSING_FAILED"

def load_failed_groups(model_name):
    """Load groups that failed for the specified model."""
    if model_name not in MODEL_PATHS:
        print(f"Model {model_name} not found in MODEL_PATHS")
        return None
    
    csv_path = os.path.join(BASE_RESULTS_DIR, MODEL_PATHS[model_name])
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    failed_df = df[df["raw_llm_ranking_output"] == FAILURE_MESSAGE].copy()
    
    print(f"Found {len(failed_df)} failed groups for model {model_name}")
    return failed_df

def load_questions_for_group(group_id):
    """Load the original questions for a specific group."""
    questions_csv = "arranged_question_sets.csv"
    if not os.path.exists(questions_csv):
        print(f"Questions file not found: {questions_csv}")
        return None
    
    df = pd.read_csv(questions_csv)
    group_df = df[df["group_id"] == group_id].sort_values("q_num_in_group")
    
    if len(group_df) != 4:
        print(f"Warning: Group {group_id} has {len(group_df)} questions instead of 4")
        return None
    
    questions = []
    for _, row in group_df.iterrows():
        choices = row["choices_str"].split("|")
        questions.append({
            "id": row["original_question_id"],  # Add the original question ID
            "question": row["question"],
            "choices": choices,
            "answer": row["answer"],
            "original_question_id": row["original_question_id"]  # Keep this for compatibility
        })
    
    return questions

def create_ranking_prompt(group_questions):
    """Create the ranking prompt for a group of questions."""
    local_labels = ["Q1", "Q2", "Q3", "Q4"]
    combined_prompt = (
        "You must rank the following multiple-choice questions based on how many tokens you would use when asked to solve in a seperate session (from least tokens to most tokens). "
        "However, questions that you will provide an incorrect answer if asked, must be ranked last (regardless of how many tokens will be used to answer). "
        "Note that each question is followed by its correct answer."
        "Each question is labeled with a unique identifier: " + ", ".join(local_labels) + ".\\n"
        "After writing your chain of thought (if needed), provide an answer using JSON format with a single key \\\"ranking\\\" whose value is a list of the question identifiers (using the above labels) in order from fastest to slowest. "
        "Questions:\\n" +
        f"{local_labels[0]}. {group_questions[0]['question']}\\n" + "\\n".join(group_questions[0]['choices']) + f"\\nCorrect Answer: {group_questions[0]['answer']}\\n\\n" +
        f"{local_labels[1]}. {group_questions[1]['question']}\\n" + "\\n".join(group_questions[1]['choices']) + f"\\nCorrect Answer: {group_questions[1]['answer']}\\n\\n" +
        f"{local_labels[2]}. {group_questions[2]['question']}\\n" + "\\n".join(group_questions[2]['choices']) + f"\\nCorrect Answer: {group_questions[2]['answer']}\\n\\n" +
        f"{local_labels[3]}. {group_questions[3]['question']}\\n" + "\\n".join(group_questions[3]['choices']) + f"\\nCorrect Answer: {group_questions[3]['answer']}"
    )

    return combined_prompt


def main():
    """Main function to reprocess failed groups."""
    print(f"Processing failed groups for model: {SELECTED_MODEL}")
    
    # Load failed groups
    failed_df = load_failed_groups(SELECTED_MODEL)
    if failed_df is None or len(failed_df) == 0:
        print("No failed groups found to process")
        return
    
    # Process all failed groups
    print(f"Found {len(failed_df)} failed groups. Processing all...")
    # failed_df = failed_df.head(5)  # Comment out for full run
    
    # Load existing individual results
    individual_results_df = load_individual_results(SELECTED_MODEL)
    print(f"Loaded {len(individual_results_df)} existing individual results")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results
    results = []
    new_individual_results = []
    
    print(f"Starting to reprocess {len(failed_df)} failed groups...")
    
    for idx, row in failed_df.iterrows():
        group_id = row['group_id']
        print(f"\nProcessing group {group_id} ({idx + 1}/{len(failed_df)})")
        
        # Load the original questions for this group
        group_questions = load_questions_for_group(group_id)
        if group_questions is None:
            print(f"Skipping group {group_id} - could not load questions")
            result = {
                'group_id': group_id,
                'model': SELECTED_MODEL,
                'timestamp': timestamp,
                'response_received': False,
                'tokens_returned': 0,
                'error_type': 'QUESTIONS_NOT_FOUND',
                'raw_response': '',
                'extracted_ranking': '',
                'parsing_status': 'NO_QUESTIONS',
                'reprocess_successful': False,
                'measured_ranking': '',
                'score': None
            }
            results.append(result)
            continue
        
        # Create the ranking prompt
        prompt = create_ranking_prompt(group_questions)
        
        # Try the LLM call with detailed tracking
        response, tokens, error = ask_llm_with_detailed_tracking(
            prompt, 
            SELECTED_MODEL, 
            temperature=1.0
        )
        
        # Try to extract ranking if we got a response
        ranking = None
        parsing_status = None
        if response:
            ranking, parsing_status = extract_ranking_from_response(response)
        
        # Process individual questions for this group
        measured_ranking = None
        total_individual_duration = None
        score = None
        
        if group_questions and ranking:
            print(f"  Processing individual questions for group {group_id}")
            
            # Check which questions already exist in individual_results
            local_labels = ["Q1", "Q2", "Q3", "Q4"]
            individual_simplicity_data = []
            
            for i, (label, question) in enumerate(zip(local_labels, group_questions)):
                # Check if this question already exists in individual_results
                existing_result = individual_results_df[
                    (individual_results_df['group_id'] == group_id) & 
                    (individual_results_df['question_id'] == question['original_question_id'])
                ]
                
                if len(existing_result) > 0:
                    # Use existing result
                    existing_row = existing_result.iloc[0]
                    is_correct = existing_row.get('is_correct', False)
                    duration = existing_row.get('duration', float('inf'))
                    if pd.isna(duration) or duration is None:
                        duration = float('inf')
                    individual_simplicity_data.append((is_correct, duration))
                    print(f"    {label}: Using existing result (correct: {is_correct}, duration: {duration})")
                else:
                    # Process this question
                    print(f"    {label}: Processing new question...")
                    entry_data, is_correct, duration = process_individual_question(
                        question, label, group_id, SELECTED_MODEL
                    )
                    individual_simplicity_data.append((is_correct, duration))
                    new_individual_results.append(entry_data)
                    print(f"    {label}: New result (correct: {is_correct}, duration: {duration})")
                    
                    # Add small delay between individual questions
                    time.sleep(0.5)
            
            # Calculate measured_ranking (copied logic from TimeTest_parallel.py)
            measured_order_tuples = sorted(
                zip(local_labels, individual_simplicity_data), 
                key=lambda x: (not x[1][0], x[1][1])  # Sort by (not correctness, duration)
            )
            measured_ranking_list = [label for label, _ in measured_order_tuples]
            measured_ranking = ",".join(measured_ranking_list)
            
            # Calculate total individual duration
            total_individual_duration = sum(
                duration for _, duration in individual_simplicity_data 
                if duration != float('inf') and duration is not None
            )
            
            # Calculate score (copied logic from TimeTest_parallel.py)
            if ranking:
                llm_ranking_list = ranking.split(',')
                if len(llm_ranking_list) == 4:
                    score = score_ranking(llm_ranking_list, measured_ranking_list)
                    print(f"  Score: {score:.3f}")
        
        # Record detailed results
        result = {
            'group_id': group_id,
            'model': SELECTED_MODEL,
            'timestamp': timestamp,
            'response_received': response is not None,
            'tokens_returned': tokens if tokens else 0,
            'error_type': error if error else 'SUCCESS',
            'raw_response': response if response else '',
            'extracted_ranking': ranking if ranking else '',
            'parsing_status': parsing_status if parsing_status else 'NO_PARSING_ATTEMPTED',
            'reprocess_successful': ranking is not None and error is None,
            'measured_ranking': measured_ranking if measured_ranking else '',
            'score': score
        }
        
        results.append(result)
        
        # Add delay between requests
        time.sleep(1)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, f"{SELECTED_MODEL}_detailed_reprocess_{timestamp}.csv")
    results_df.to_csv(output_file, index=False)
    
    # Save new individual results if any
    if new_individual_results:
        # Load existing individual results again to append
        individual_results_df = load_individual_results(SELECTED_MODEL)
        
        # Append new results
        new_individual_df = pd.DataFrame(new_individual_results)
        combined_individual_df = pd.concat([individual_results_df, new_individual_df], ignore_index=True)
        
        # Save updated individual results
        model_paths = {
            "o4-mini-2025-04-16": "results/openai/o4-mini-2025-04-16/individual_results.csv",
            "claude-3.7-sonnet-thinking": "results/anthropic/claude-3.7-sonnet-thinking/individual_results.csv",
            "claude-3.5-haiku": "results/anthropic/claude-3.5-haiku/individual_results.csv",
            "deepseek-r1": "results/deepseek/deepseek-r1/individual_results.csv",
            "gemini-2.5-pro-preview": "results/google/gemini-2.5-pro-preview/individual_results.csv"
        }
        
        individual_csv_path = model_paths.get(SELECTED_MODEL)
        if individual_csv_path:
            combined_individual_df.to_csv(individual_csv_path, index=False)
            print(f"Updated individual_results.csv with {len(new_individual_results)} new entries")
    
    # Update original CSV with successful results
    update_original_csv(SELECTED_MODEL, results_df, timestamp)
    
    # Summary
    success_count = sum(1 for r in results if r['reprocess_successful'])
    print(f"\n=== REPROCESSING SUMMARY ===")
    print(f"Total groups processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"New individual questions processed: {len(new_individual_results)}")
    print(f"Results saved to: {output_file}")
    
    # Score statistics
    scores = [r['score'] for r in results if r['score'] is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Average score: {avg_score:.3f} (based on {len(scores)} groups)")
    
    # Error breakdown
    error_types = {}
    for result in results:
        error = result['error_type']
        error_types[error] = error_types.get(error, 0) + 1
    
    print(f"\nError breakdown:")
    for error, count in error_types.items():
        print(f"  {error}: {count}")

def update_original_csv(model_name, results_df, timestamp):
    """Update the original CSV file with successful reprocessing results."""
    if model_name not in MODEL_PATHS:
        print(f"Cannot update CSV: Model {model_name} not found in MODEL_PATHS")
        return
    
    original_csv_path = os.path.join(BASE_RESULTS_DIR, MODEL_PATHS[model_name])
    if not os.path.exists(original_csv_path):
        print(f"Cannot update CSV: Original file not found: {original_csv_path}")
        return
    
    # Load original CSV
    original_df = pd.read_csv(original_csv_path)
    
    # Add new columns if they don't exist
    if 'reprocess_timestamp' not in original_df.columns:
        original_df['reprocess_timestamp'] = None
    if 'reprocess_status' not in original_df.columns:
        original_df['reprocess_status'] = None
    if 'new_llm_ranking' not in original_df.columns:
        original_df['new_llm_ranking'] = None
    if 'new_raw_llm_ranking_output' not in original_df.columns:
        original_df['new_raw_llm_ranking_output'] = None
    
    # Update rows with successful reprocessing results
    successful_results = results_df[results_df['reprocess_successful'] == True]
    update_count = 0
    
    for _, result_row in successful_results.iterrows():
        group_id = result_row['group_id']
        
        # Find matching rows in original CSV
        mask = original_df['group_id'] == group_id
        matching_rows = original_df[mask]
        
        if len(matching_rows) > 0:
            # Update the original CSV
            original_df.loc[mask, 'reprocess_timestamp'] = timestamp
            original_df.loc[mask, 'reprocess_status'] = 'SUCCESS'
            original_df.loc[mask, 'new_llm_ranking'] = result_row['extracted_ranking']
            original_df.loc[mask, 'new_raw_llm_ranking_output'] = result_row['raw_response']
            update_count += 1
    
    # Also update failed reprocessing attempts
    failed_results = results_df[results_df['reprocess_successful'] == False]
    for _, result_row in failed_results.iterrows():
        group_id = result_row['group_id']
        
        # Find matching rows in original CSV
        mask = original_df['group_id'] == group_id
        matching_rows = original_df[mask]
        
        if len(matching_rows) > 0:
            # Update the original CSV
            original_df.loc[mask, 'reprocess_timestamp'] = timestamp
            original_df.loc[mask, 'reprocess_status'] = f"FAILED: {result_row['error_type']}"
    
    # Save updated CSV with backup
    backup_path = f"{original_csv_path}.backup_{timestamp}"
    original_df.to_csv(backup_path, index=False)
    original_df.to_csv(original_csv_path, index=False)
    
    print(f"Updated original CSV: {original_csv_path}")
    print(f"Backup saved to: {backup_path}")
    print(f"Successfully updated {update_count} groups with new results")

# Add functions copied from TimeTest_parallel.py
def extract_json_from_text(text, required_keys=None):
    """
    Attempts to extract a JSON object from the given text.
    Copied from TimeTest_parallel.py
    """
    if not text: return None # Handle empty text
    normalized_text = text.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'").strip()
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, normalized_text, flags=re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and (required_keys is None or any(
                    key.lower() in (k.lower() for k in parsed.keys()) for key in required_keys)):
                return parsed
        except Exception: # Simplified error logging
            pass # print(f"DEBUG: extract_json_from_text - Error parsing JSON from code fence: {e}")
    first = normalized_text.find("{")
    last = normalized_text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = normalized_text[first:last + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and (required_keys is None or any(
                    key.lower() in (k.lower() for k in parsed.keys()) for key in required_keys)):
                return parsed
        except Exception:
             pass # print(f"DEBUG: extract_json_from_text - Error parsing candidate substring: {e}")
    candidates = re.findall(r'\{.*?\}', normalized_text, flags=re.DOTALL)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and (required_keys is None or any(
                    key.lower() in (k.lower() for k in parsed.keys()) for key in required_keys)):
                return parsed
        except Exception:
            continue
    return None

def extract_answer_letter(text, valid_letters):
    """
    Extracts the answer letter from the given text.
    Copied from TimeTest_parallel.py
    """
    parsed = extract_json_from_text(text, required_keys=["answer"])
    if parsed and isinstance(parsed, dict):
        for key, value in parsed.items():
            if key.lower() == "answer" and isinstance(value, str):
                ans = value.strip().upper()
                if ans in valid_letters:
                    return ans
    fallback = re.search(r'"answer"\s*:\s*"([A-Z])"', text, flags=re.IGNORECASE)
    if fallback:
        ans = fallback.group(1).upper()
        if ans in valid_letters:
            return ans
    if text: # Ensure text is not None
        for char in text:
            if char.upper() in valid_letters:
                return char.upper()
    return None

def score_ranking(llm_ranking, measured_ranking):
    """
    Computes the fraction of unordered pairs ranked in the same relative order.
    Copied from TimeTest_parallel.py
    """
    n = len(measured_ranking)
    if n < 2 or not llm_ranking or len(llm_ranking) != n: # Added check for llm_ranking
        return 0.0
    
    # Ensure all items in measured_ranking are in llm_ranking to prevent .index errors
    if not all(item in llm_ranking for item in measured_ranking):
        # print("Warning: Not all items in measured_ranking are present in llm_ranking. Score might be inaccurate.")
        return 0.0 # Or handle more gracefully depending on desired behavior

    correct_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            try:
                # Check if elements are in llm_ranking before calling index
                idx_i_llm = llm_ranking.index(measured_ranking[i])
                idx_j_llm = llm_ranking.index(measured_ranking[j])
                if idx_i_llm < idx_j_llm: # Correct relative order
                    correct_pairs += 1
            except ValueError:
                # This case should ideally be caught by the check above
                # print(f"Warning: Item from measured_ranking not found in llm_ranking during pair comparison.")
                continue # Skip this pair if an item is missing
                
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0

def load_individual_results(model_name):
    """Load existing individual results CSV for the model."""
    model_paths = {
        "o4-mini-2025-04-16": "results/openai/o4-mini-2025-04-16/individual_results.csv",
        "claude-3.7-sonnet-thinking": "results/anthropic/claude-3.7-sonnet-thinking/individual_results.csv",
        "claude-3.5-haiku": "results/anthropic/claude-3.5-haiku/individual_results.csv",
        "deepseek-r1": "results/deepseek/deepseek-r1/individual_results.csv",
        "gemini-2.5-pro-preview": "results/google/gemini-2.5-pro-preview/individual_results.csv"
    }
    
    csv_path = model_paths.get(model_name)
    if not csv_path or not os.path.exists(csv_path):
        print(f"Individual results file not found for {model_name}")
        return pd.DataFrame()
    
    return pd.read_csv(csv_path)

def process_individual_question(question, label, group_id, model_name):
    """
    Process a single individual question. 
    Copied logic from TimeTest_parallel.py process_single_individual_q function.
    """
    valid_letters = [chr(ord('A') + i) for i in range(len(question["choices"]))]
    individual_prompt = (
        f"{question['question']}\n" + "\n".join(question["choices"]) +
        "\n\nAfter writing your chain of thought (if needed), provide your answer using JSON format with a single key \"answer\" whose value is the letter corresponding to your final answer."
    )
    
    answer_text, duration_tokens, error = ask_llm_with_detailed_tracking(individual_prompt, model_name)
    
    is_correct = False
    extracted_answer = None
    if answer_text:
        extracted_answer = extract_answer_letter(answer_text, valid_letters)
        if extracted_answer == question["answer"].upper():
            is_correct = True
    
    # Create entry for individual_results.csv - copied structure from TimeTest_parallel.py
    entry_data = {
        "group_id": group_id,
        "question_id": question["id"],
        "question_text": question["question"],
        "choices_text": "\n".join(question["choices"]),
        "duration": duration_tokens, # duration is token count for individual q
        "llm_answer_raw": answer_text,
        "llm_answer_extracted": extracted_answer,
        "correct_answer": question["answer"].upper(),
        "is_correct": is_correct,
        "model": model_name
    }
    
    return entry_data, is_correct, duration_tokens if duration_tokens else float('inf')

if __name__ == "__main__":
    # You can change SELECTED_MODEL at the top of the file
    # Or uncomment the lines below to make it interactive
    
    # print("Available models:")
    # for i, model in enumerate(MODEL_PATHS.keys(), 1):
    #     print(f"{i}. {model}")
    # 
    # choice = input("Enter model number (or press Enter for o4-mini): ").strip()
    # if choice.isdigit():
    #     models = list(MODEL_PATHS.keys())
    #     if 1 <= int(choice) <= len(models):
    #         SELECTED_MODEL = models[int(choice) - 1]
    
    main()
