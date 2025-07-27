# ===============================================
# REPROCESSING SCRIPT - Created automatically
# This script is configured to reprocess only specific question groups
# ===============================================

import time
import random
import re
import json
import pandas as pd
from datasets import load_dataset
import openai
import os
import requests
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# Read your OpenAI API key from an external file.
key_file = "openai_key.txt"
openai_api_key_val = None
openrouter_api_key_val = None # New variable for OpenRouter key

if os.path.exists(key_file):
    with open(key_file, "r") as f:
        lines = f.readlines()
        if len(lines) > 0:
            openai_api_key_val = lines[0].strip()
            openai.api_key = openai_api_key_val # Set for OpenAI provider
        if len(lines) > 1:
            openrouter_api_key_val = lines[1].strip() # Store OpenRouter key
        
        if not openai_api_key_val:
            print(f"OpenAI API key not found on the first line of '{key_file}'.")
        if not openrouter_api_key_val:
            print(f"OpenRouter API key not found on the second line of '{key_file}'.")
else:
    raise FileNotFoundError(f"API key file '{key_file}' not found. Please create it and paste your OpenAI API key on the first line and OpenRouter API key on the second line.")


models_to_use = [
    "o4-mini-2025-04-16",                       # OpenAI o4‑mini (a.k.a. GPT‑o4‑mini)
    "gpt-4.1-2025-04-14",                       # OpenAI GPT‑4.1 (full)
    "gpt-4.1-mini-2025-04-14",                  # OpenAI GPT‑4.1‑mini
    "deepseek/deepseek-r1",                 # DeepSeek R1
    "deepseek/deepseek-chat-v3",            # DeepSeek V3 (0324 release)
    "google/gemini-2.5-pro-preview",                # Gemini 2.5 Pro
    "google/gemini-2.5-flash-preview",              # Gemini 2.5 Flash
    "meta-llama/llama-4-scout",             # Llama 4 Scout
    "meta-llama/llama-3.1-8b-instruct",     # Llama 3.1 8B Instruct
    "meta-llama/llama-3.2-3b-instruct",     # Llama 3.2 3B Instruct
    "mistralai/mistral-7b-instruct",        # Mistral 7B Instruct
    "mistralai/mistral-small-3.1-24b-instruct",  # Mistral‑Small 3.1 (24B) Instruct
    "anthropic/claude-3.7-sonnet:thinking", # Claude 3.7 Sonnet ("thinking")
    "anthropic/claude-3.5-haiku",           # Claude 3.5 Haiku!
    "qwen/qwen-2.5-7b-instruct"              # Qwen 2.5 7B Instruct
]

index = 5
# Global model selection – change this value to run with a different LLM.
model_to_use = models_to_use[index]  # Default to the first model in the list

# Maximum groups to process in a single run.
max_groups_to_process = 250 # Process all groups

ARRANGED_QUESTIONS_CSV = "reprocess/anthropic_claude_3_7_sonnet:thinking_reprocess.csv"
PROCESSED_IDS_CSV = "processed_ids.csv"

def check_group_already_processed(group_questions, current_model, processed_ids_set):
    """
    Checks if all 4 questions in a group have already been processed by the current model.
    
    Args:
        group_questions: List of question dictionaries for a group
        current_model: The model being used
        processed_ids_set: Set of question_id_model strings from processed_ids.csv
        
    Returns:
        True if all questions in the group were already processed by this model, False otherwise
    """
    if not processed_ids_set:
        return False
        
    # Check if all question IDs from this group are in processed_ids_set with current model
    all_processed = True
    for question in group_questions:
        question_model_key = f"{question['id']}_{current_model}"
        if question_model_key not in processed_ids_set:
            all_processed = False
            break
            
    return all_processed

def check_individual_question_processed(question_id, current_model, processed_ids_set):
    """
    Checks if a specific question has already been processed by the current model.
    
    Args:
        question_id: The question ID to check
        current_model: The model being used
        processed_ids_set: Set of question_id_model strings from processed_ids.csv
        
    Returns:
        True if the question was already processed by this model, False otherwise
    """
    if not processed_ids_set:
        return False
        
    question_model_key = f"{question_id}_{current_model}"
    return question_model_key in processed_ids_set

def get_benchmark_questions(dataset_choice="mmlu", num_questions=10000, topics=None):
    """
    Loads benchmark questions from one of two datasets: MMLU or CommonSenseQA.
    (Identical to the original TimeTest.py)
    """
    questions = []
    if dataset_choice.lower() == "mmlu":
        if topics is None:
            topics = ["abstract_algebra", "anatomy", "astronomy"]
        all_items = []
        for topic in topics:
            try:
                ds_topic = load_dataset("cais/mmlu", topic, split="test")
                print(f"Loaded {len(ds_topic)} examples for topic '{topic}'.")
            except Exception as e:
                print(f"Error loading topic '{topic}' from MMLU: {e}")
                continue
            for i, item in enumerate(ds_topic):
                if i >= 100: # Limiting items per topic for faster testing as in original
                    break
                words = item["question"].split()
                first_three = "".join(word.capitalize() for word in words[:3])
                item["id"] = f"MMLU_{topic}_{first_three}_{i+1:04d}"
                all_items.append(item)
                
        print(f"Total examples loaded from all topics: {len(all_items)}")
        if len(all_items) < num_questions:
            print(f"Warning: only {len(all_items)} examples available; using all.")
            num_questions = len(all_items)
        random.shuffle(all_items)
        selected_items = all_items[:num_questions]
        for item in selected_items:
            full_question = item["question"]
            choices = [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item["choices"])]
            correct = chr(65 + int(item["answer"]))
            questions.append({
                "id": item["id"],
                "question": full_question,
                "choices": choices,
                "answer": correct
            })
    # elif dataset_choice.lower() == "commonsense":
    #     try:
    #         ds = load_dataset("commonsense_qa", split="validation")
    #     except Exception as e:
    #         print(f"Error loading CommonSenseQA dataset: {e}")
    #         return []
    #     ds = ds.shuffle(seed=42).select(range(min(num_questions, len(ds)))) # Ensure not to select more than available
    #     for i, item in enumerate(ds):
    #         words = item["question"].split()
    #         first_three = "".join(word.capitalize() for word in words[:3])
    #         question_text = item["question"]
    #         choices_list = item["choices"]["text"] # Accessing text from choices
    #         labels = item["choices"]["label"] # Accessing labels
            
    #         labeled_choices = []
    #         for j, choice_text in enumerate(choices_list):
    #             letter = labels[j] # Use provided labels
    #             labeled_choices.append(f"{letter}. {choice_text}")

    #         correct_letter = item["answerKey"].strip().upper() # CommonSenseQA uses 'answerKey'
    #         questions.append({
    #             "id": f"CommonSense_{first_three}_{i+1:04d}",
    #             "question": question_text,
    #             "choices": labeled_choices,
    #             "answer": correct_letter
    #         })
    else:
        print(f"Unknown dataset choice: {dataset_choice}")
        return []
    print(f"Total questions loaded: {len(questions)}")
    return questions

def ask_llm(prompt, model=model_to_use, temperature=1.0, max_retries=3):
    """
    Sends a prompt to the LLM and returns its answer along with the token count and error status.
    Returns: (text, tokens, error_info) where error_info is a dict with error details or None if successful
    """
    provider = 'openrouter'#"openrouter" # Default provider
    # if "/" in model and not model.startswith("openai/"):
    #     provider = "openrouter"
    
    # print(f"DEBUG: Using model: {model} with provider: {provider}") # Optional debug line

    for attempt in range(max_retries):
        try:
            if provider == "openai":
                if not openai.api_key:
                    # This refers to openai_api_key_val loaded globally
                    print("OpenAI API key not set globally for ask_llm.")
                    # This reload might be needed if openai.api_key is not correctly passed
                    # or set in the multiprocessing context.
                    if os.path.exists(key_file):
                        with open(key_file, "r") as f:
                            lines = f.readlines()
                            if len(lines) > 0:
                                openai.api_key = lines[0].strip()
                    if not openai.api_key:
                         print("OpenAI API key still not set after attempting reload in ask_llm.")
                         return None, None, {"type": "api_key_error", "message": "OpenAI API key not available"}

                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                tokens = response["usage"]["completion_tokens"]
                text = response["choices"][0]["message"]["content"]
                
                # Check for zero tokens
                if tokens == 0:
                    error_info = {"type": "zero_tokens", "message": "Response generated 0 tokens", "attempt": attempt + 1}
                    print(f"Warning: Zero tokens returned from {provider} (model: {model}, attempt: {attempt+1})")
                    if attempt < max_retries - 1:
                        print("Retrying due to zero tokens...")
                        time.sleep(1)
                        continue
                    return text, tokens, error_info
                
                # Check for blocked/empty content
                if not text or text.strip() == "":
                    error_info = {"type": "blocked_response", "message": "Empty or blocked response content", "attempt": attempt + 1}
                    print(f"Warning: Empty response from {provider} (model: {model}, attempt: {attempt+1})")
                    if attempt < max_retries - 1:
                        print("Retrying due to empty response...")
                        time.sleep(1)
                        continue
                    return text, tokens, error_info
                
            elif provider == "openrouter":
                global openrouter_api_key_val 
                if not openrouter_api_key_val: 
                    # Check if the global OpenRouter key is loaded
                    print(f"OpenRouter API key not available from '{key_file}' for ask_llm.")
                    # Attempt to reload if in a new process context
                    # This part might be redundant if global loading is robust
                    if os.path.exists(key_file):
                        with open(key_file, "r") as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                openrouter_api_key_val = lines[1].strip()
                    if not openrouter_api_key_val:
                        print("OpenRouter API key still not set after attempting reload in ask_llm.")
                        return None, None, {"type": "api_key_error", "message": "OpenRouter API key not available"}

                headers = {
                    "Authorization": f"Bearer {openrouter_api_key_val}", 
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                api_response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                # Check for API errors before processing response
                if api_response.status_code == 429:
                    error_info = {"type": "api_problem", "message": f"Rate limit exceeded (HTTP 429)", "status_code": 429, "attempt": attempt + 1}
                    print(f"Rate limit error from {provider} (model: {model}, attempt: {attempt+1}): HTTP 429")
                    if attempt < max_retries - 1:
                        print("Retrying due to rate limit...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None, None, error_info
                
                if api_response.status_code >= 400:
                    error_info = {"type": "api_problem", "message": f"HTTP error {api_response.status_code}", "status_code": api_response.status_code, "attempt": attempt + 1}
                    print(f"HTTP error from {provider} (model: {model}, attempt: {attempt+1}): {api_response.status_code}")
                    if attempt < max_retries - 1:
                        print(f"Retrying due to HTTP {api_response.status_code}...")
                        time.sleep(1)
                        continue
                    return None, None, error_info
                
                api_response.raise_for_status() 
                response_json = api_response.json()
                
                # Check for API-level errors in response
                if "error" in response_json:
                    error_info = {"type": "api_problem", "message": f"API error: {response_json['error']}", "attempt": attempt + 1}
                    print(f"API error from {provider} (model: {model}, attempt: {attempt+1}): {response_json['error']}")
                    if attempt < max_retries - 1:
                        print("Retrying due to API error...")
                        time.sleep(1)
                        continue
                    return None, None, error_info
                
                # Check if response was blocked by content filter
                if "choices" not in response_json or not response_json["choices"]:
                    error_info = {"type": "blocked_response", "message": "No choices in response (possibly blocked)", "attempt": attempt + 1}
                    print(f"No choices in response from {provider} (model: {model}, attempt: {attempt+1})")
                    if attempt < max_retries - 1:
                        print("Retrying due to blocked response...")
                        time.sleep(1)
                        continue
                    return None, None, error_info
                
                choice = response_json["choices"][0]
                if "finish_reason" in choice and choice["finish_reason"] == "content_filter":
                    error_info = {"type": "blocked_response", "message": "Response blocked by content filter", "attempt": attempt + 1}
                    print(f"Content filter blocked response from {provider} (model: {model}, attempt: {attempt+1})")
                    if attempt < max_retries - 1:
                        print("Retrying due to content filter...")
                        time.sleep(1)
                        continue
                    return None, None, error_info
                
                text = choice["message"]["content"]
                tokens = response_json["usage"]["completion_tokens"] if "usage" in response_json else 0
                
                # Check for zero tokens
                if tokens == 0:
                    error_info = {"type": "zero_tokens", "message": "Response generated 0 tokens", "attempt": attempt + 1}
                    print(f"Warning: Zero tokens returned from {provider} (model: {model}, attempt: {attempt+1})")
                    if attempt < max_retries - 1:
                        print("Retrying due to zero tokens...")
                        time.sleep(1)
                        continue
                    return text, tokens, error_info
                
                # Check for blocked/empty content
                if not text or text.strip() == "":
                    error_info = {"type": "blocked_response", "message": "Empty or blocked response content", "attempt": attempt + 1}
                    print(f"Warning: Empty response from {provider} (model: {model}, attempt: {attempt+1})")
                    if attempt < max_retries - 1:
                        print("Retrying due to empty response...")
                        time.sleep(1)
                        continue
                    return text, tokens, error_info
                    
            else:
                # This case should not be reached if logic is correct
                print(f"Internal error: Unknown provider determined: {provider}")
                return None, None, {"type": "internal_error", "message": f"Unknown provider: {provider}"}
                
            # Success case - return with no error
            return text, tokens, None
            
        except requests.exceptions.RequestException as e:
            error_info = {"type": "api_problem", "message": f"Request error: {str(e)}", "attempt": attempt + 1}
            print(f"Request error during LLM call with {provider} (model: {model}, attempt: {attempt+1}): {e}")
            if attempt < max_retries - 1:
                print("Retrying due to request error...")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None, None, error_info
            
        except Exception as e:
            error_info = {"type": "api_problem", "message": f"Unexpected error: {str(e)}", "attempt": attempt + 1}
            print(f"Unexpected error during LLM call with {provider} (model: {model}, attempt: {attempt+1}): {e}")
            if attempt < max_retries - 1:
                print("Retrying due to unexpected error...")
                time.sleep(1)
                continue
            return None, None, error_info
    
    # If we get here, all retries failed
    final_error = {"type": "api_problem", "message": f"Failed after {max_retries} attempts", "attempts": max_retries}
    print(f"Failed to get a response from {provider} (model: {model}) after {max_retries} attempts.")
    return None, None, final_error

def get_credits():
    """Fetches the remaining credits from OpenRouter API."""
    try:
        response = requests.get(
            url="https://openrouter.ai/api/v1/credits",
            headers={
                "Authorization": "Bearer " + openrouter_api_key_val,
            }
        )
        response_data = response.json()
        return response_data.get("data", {})
    except Exception as e:
        print(f"Error fetching credits: {e}")
        return None
    
def extract_json_from_text(text, required_keys=None):
    """
    Attempts to extract a JSON object from the given text.
    (Identical to the original TimeTest.py)
    """
    if not text: return None # Handle empty text
    normalized_text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'").strip()
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
    (Identical to the original TimeTest.py, minor debug print changes)
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

def extract_ranking(text):
    """
    Extracts the ranking list from the given text.
    (Identical to the original TimeTest.py, minor debug print changes)
    """
    parsed = extract_json_from_text(text, required_keys=["ranking"])
    if parsed and isinstance(parsed, dict):
        for key, value in parsed.items():
            if key.lower() == "ranking":
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    return [item.strip() for item in value.split(",") if item.strip()]
    return None

def score_ranking(llm_ranking, measured_ranking):
    """
    Computes the fraction of unordered pairs ranked in the same relative order.
    (Identical to the original TimeTest.py)
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


def process_group_wrapper(args):
    """
    Wrapper function to process a single group of questions.
    Individual question LLM calls are made concurrently using a ThreadPoolExecutor.
    """
    group, group_id_val, dataset_choice_val, current_model_to_use, processed_ids_set, existing_individual_results = args
    
    # Check if all questions in this group have already been processed by this model
    # Only skip entire group if ALL individual questions are done AND we're not retrying combined ranking
    if check_group_already_processed(group, current_model_to_use, processed_ids_set):
        # Check if there's an existing combined ranking result that failed due to API problems
        # For now, we'll process the group anyway to retry combined ranking
        print(f"All individual questions in group {group_id_val} already processed by {current_model_to_use}, but will check combined ranking...")
        skip_all_individuals = True
    else:
        skip_all_individuals = False
    
    # Ensure API keys are accessible in this process
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                openai_api_key_local = lines[0].strip()
                if openai_api_key_local:
                    openai.api_key = openai_api_key_local
            if len(lines) > 1:
                openrouter_api_key_local = lines[1].strip()
    
    local_labels = ["Q1", "Q2", "Q3", "Q4"]
    original_ids = [q["id"] for q in group]

    print(f"----- Processing Group (Group ID: {group_id_val}, Model: {current_model_to_use}) -----")

    combined_ids_entry = {
        "group_id": group_id_val,
        "question_ids": ",".join(original_ids),
        "model": current_model_to_use
    }

    combined_prompt = (
        "You must rank the following multiple-choice questions based on how many tokens you would use when asked to solve in a seperate session (from least tokens to most tokens). "
        "However, questions that you will provide an incorrect answer if asked, must be ranked last (regardless of how many tokens will be used to answer). "
        "Note that each question is followed by its correct answer."
        "Each question is labeled with a unique identifier: " + ", ".join(local_labels) + ".\\n"
        "After writing your chain of thought (if needed), provide an answer using JSON format with a single key \\\"ranking\\\" whose value is a list of the question identifiers (using the above labels) in order from fastest to slowest. "
        "Questions:\\n" +
        f"{local_labels[0]}. {group[0]['question']}\\n" + "\\n".join(group[0]['choices']) + f"\\nCorrect Answer: {group[0]['answer']}\\n\\n" +
        f"{local_labels[1]}. {group[1]['question']}\\n" + "\\n".join(group[1]['choices']) + f"\\nCorrect Answer: {group[1]['answer']}\\n\\n" +
        f"{local_labels[2]}. {group[2]['question']}\\n" + "\\n".join(group[2]['choices']) + f"\\nCorrect Answer: {group[2]['answer']}\\n\\n" +
        f"{local_labels[3]}. {group[3]['question']}\\n" + "\\n".join(group[3]['choices']) + f"\\nCorrect Answer: {group[3]['answer']}"
    )
    
    combined_answer, combined_tokens, combined_error = ask_llm(combined_prompt, model=current_model_to_use)
    # raw_ranking_output_for_csv = combined_answer # Store raw output regardless of parsing success # Deferred assignment

    if combined_answer is None or combined_error is not None:
        if combined_error:
            if combined_error["type"] == "api_problem":
                reason_for_failure = f"API PROBLEM (UNACCEPTABLE): {combined_error['message']}"
                status = "failed_api_problem"
            elif combined_error["type"] == "zero_tokens":
                reason_for_failure = f"ZERO TOKENS: {combined_error['message']}"
                status = "failed_zero_tokens"
            elif combined_error["type"] == "blocked_response":
                reason_for_failure = f"BLOCKED RESPONSE: {combined_error['message']}"
                status = "failed_blocked_response"
            else:
                reason_for_failure = f"ERROR: {combined_error['message']}"
                status = "failed_other_error"
        else:
            reason_for_failure = "Combined prompt LLM call failed or returned no answer"
            status = "failed_combined_prompt"
            
        print(f"Skipping group {group_id_val} due to combined prompt failure: {reason_for_failure}")
        return {
            "group_id": group_id_val, "status": status, 
            "original_ids": original_ids, "model": current_model_to_use,
            "individual_results_entries": [], "question_sets_entry": None,
            "combined_ids_entry": combined_ids_entry, "group_correct_count": 0,
            "num_questions_in_group": len(group), "score": None,
            "raw_combined_answer": reason_for_failure, # Store the reason string here
            "error_info": combined_error
        }

    # If we reach here, combined_answer is NOT None and no critical error occurred.
    raw_ranking_output_for_csv = combined_answer # Assign now for the success path.

    # Extract ranking from the combined answer - THIS WAS MISSING!
    llm_ranking_list = extract_ranking(combined_answer)

    individual_simplicity_data_map = {}  # Stores (is_correct, duration_tokens) by label
    individual_results_entries = []
    group_correct_count = 0

    def process_single_individual_q(label_arg, q_item_arg, model_arg, group_id_for_log):
        # Check if this individual question is already processed
        if check_individual_question_processed(q_item_arg["id"], model_arg, processed_ids_set):
            print(f"Skipping individual question {label_arg} ({q_item_arg['id']}) - already processed by {model_arg}")
            
            # Load existing results if available
            existing_data = existing_individual_results.get(q_item_arg["id"])
            if existing_data:
                is_correct_existing, duration_existing = existing_data
                return {
                    "label": label_arg,
                    "is_correct": is_correct_existing,
                    "duration_tokens": duration_existing,
                    "error_info": None,
                    "entry_data": None,  # No new entry needed
                    "already_processed": True
                }
            else:
                # Fallback if no existing data found
                return {
                    "label": label_arg,
                    "is_correct": True,  # Assume correct if processed
                    "duration_tokens": 1000,  # Default duration
                    "error_info": None,
                    "entry_data": None,
                    "already_processed": True
                }
        
        valid_letters_arg = [chr(ord('A') + i) for i in range(len(q_item_arg["choices"]))]
        individual_prompt_arg = (
            f"{q_item_arg['question']}\\n" + "\\n".join(q_item_arg["choices"]) +
            "\\n\\nAfter writing your chain of thought (if needed), provide your answer using JSON format with a single key \\\"answer\\\" whose value is the letter corresponding to your final answer."
        )
        answer_text_res, duration_tokens_res, error_info_res = ask_llm(individual_prompt_arg, model=model_arg)
        
        is_correct_flag_res = False
        extracted_ans_res = None
        error_reason = None
        
        if answer_text_res is not None and error_info_res is None:
            extracted_ans_res = extract_answer_letter(answer_text_res, valid_letters_arg)
            if extracted_ans_res == q_item_arg["answer"].upper():
                is_correct_flag_res = True
        elif error_info_res is not None:
            # Determine error reason for tracking
            if error_info_res["type"] == "api_problem":
                error_reason = f"API_PROBLEM: {error_info_res['message']}"
            elif error_info_res["type"] == "zero_tokens":
                error_reason = f"ZERO_TOKENS: {error_info_res['message']}"
            elif error_info_res["type"] == "blocked_response":
                error_reason = f"BLOCKED: {error_info_res['message']}"
            else:
                error_reason = f"ERROR: {error_info_res['message']}"
        
        return {
            "label": label_arg,
            "is_correct": is_correct_flag_res,
            "duration_tokens": duration_tokens_res if duration_tokens_res is not None else float('inf'),
            "error_info": error_info_res,
            "already_processed": False,
            "entry_data": {
                "group_id": group_id_for_log,
                "question_id": q_item_arg["id"],
                "question_text": q_item_arg["question"],
                "choices_text": "\\n".join(q_item_arg["choices"]),
                "duration": duration_tokens_res, # duration is token count for individual q
                "llm_answer_raw": answer_text_res,
                "llm_answer_extracted": extracted_ans_res,
                "correct_answer": q_item_arg["answer"].upper(),
                "is_correct": is_correct_flag_res,
                "model": model_arg,
                "error_reason": error_reason
            }
        }

    future_to_q_meta = {}
    # Max 4 threads for 4 questions; these are I/O bound.
    with ThreadPoolExecutor(max_workers=len(group)) as executor: 
        for lbl, q_itm in zip(local_labels, group):
            future = executor.submit(process_single_individual_q, lbl, q_itm, current_model_to_use, group_id_val)
            future_to_q_meta[future] = lbl

    for future_item in as_completed(future_to_q_meta):
        original_label_for_future = future_to_q_meta[future_item]
        try:
            result_item = future_item.result()
            
            # Handle already processed questions
            if result_item.get("already_processed", False):
                # Use the loaded existing data
                individual_simplicity_data_map[result_item["label"]] = (result_item["is_correct"], result_item["duration_tokens"])
                if result_item["is_correct"]:
                    group_correct_count += 1
                print(f"Using existing data for {original_label_for_future} in group {group_id_val} - Correct: {result_item['is_correct']}, Duration: {result_item['duration_tokens']}")
                continue
            
            entry_data_from_thread = result_item["entry_data"]
            # group_id is now added directly in process_single_individual_q
            if entry_data_from_thread is not None:
                individual_results_entries.append(entry_data_from_thread)
            
            individual_simplicity_data_map[result_item["label"]] = (result_item["is_correct"], result_item["duration_tokens"])
            if result_item["is_correct"]:
                group_correct_count += 1
                
            # Log specific error types for individual questions
            if result_item["error_info"] is not None:
                error_type = result_item["error_info"]["type"]
                if error_type == "api_problem":
                    print(f"API PROBLEM (UNACCEPTABLE) for {original_label_for_future} in group {group_id_val}: {result_item['error_info']['message']}")
                elif error_type == "zero_tokens":
                    print(f"ZERO TOKENS for {original_label_for_future} in group {group_id_val}: {result_item['error_info']['message']}")
                elif error_type == "blocked_response":
                    print(f"BLOCKED RESPONSE for {original_label_for_future} in group {group_id_val}: {result_item['error_info']['message']}")
                    
        except Exception as exc:
            print(f"PROCESSING ERROR for label {original_label_for_future} in group {group_id_val}: {exc}")
            individual_simplicity_data_map[original_label_for_future] = (False, float('inf'))
            failed_q_item = None
            # Find the question item that failed to populate its details
            for lbl_s, q_itm_s in zip(local_labels, group):
                if lbl_s == original_label_for_future:
                    failed_q_item = q_itm_s
                    break
            
            error_entry = {
                "group_id": group_id_val,
                "question_id": failed_q_item["id"] if failed_q_item else "UNKNOWN",
                "question_text": failed_q_item["question"] if failed_q_item else "UNKNOWN",
                "choices_text": "\\n".join(failed_q_item["choices"]) if failed_q_item else "UNKNOWN",
                "duration": None,
                "llm_answer_raw": "ERROR_PROCESSING",
                "llm_answer_extracted": "ERROR",
                "correct_answer": failed_q_item["answer"].upper() if failed_q_item else "UNKNOWN",
                "is_correct": False,
                "model": current_model_to_use,
                "error_reason": f"PROCESSING_ERROR: {str(exc)}"
            }
            individual_results_entries.append(error_entry)

    # Ensure individual_simplicity_data is ordered correctly according to local_labels for measured_ranking
    individual_simplicity_data = [individual_simplicity_data_map.get(lbl, (False, float('inf'))) for lbl in local_labels]
    total_individual_tokens = sum(x[1] for x in individual_simplicity_data if x[1] != float('inf') and x[1] is not None)
    
    # Determine why we don't have a valid ranking
    ranking_failure_reason = None
    if llm_ranking_list is None or not isinstance(llm_ranking_list, list) or len(llm_ranking_list) != 4:
        if combined_error is not None:
            if combined_error["type"] == "api_problem":
                ranking_failure_reason = "API_PROBLEM"
            elif combined_error["type"] == "zero_tokens":
                ranking_failure_reason = "ZERO_TOKENS"
            elif combined_error["type"] == "blocked_response":
                ranking_failure_reason = "BLOCKED_RESPONSE"
            else:
                ranking_failure_reason = "OTHER_ERROR"
        elif combined_tokens == 0:
            ranking_failure_reason = "ZERO_TOKENS"
        elif combined_answer is not None:
            ranking_failure_reason = "UNPARSABLE_OUTPUT"
            print(f"Warning: Failed to extract valid ranking for group {group_id_val} - UNPARSABLE OUTPUT. LLM output: {combined_answer[:200]}...")
        else:
            ranking_failure_reason = "NO_RESPONSE"
            
        llm_ranking_list = [] # Ensure it's a list for join

    # Sort by (not correctness, token_duration)
    measured_order_tuples = sorted(zip(local_labels, individual_simplicity_data), key=lambda x: (not x[1][0], x[1][1]))
    measured_ranking_list = [label for label, _ in measured_order_tuples]

    current_score = None
    if llm_ranking_list and len(llm_ranking_list) == 4: # Ensure valid ranking for scoring
        current_score = score_ranking(llm_ranking_list, measured_ranking_list)

    question_sets_entry = {
        "group_id": group_id_val,
        "llm_ranking": ",".join(llm_ranking_list),
        "raw_llm_ranking_output": raw_ranking_output_for_csv, # Add raw output here
        "combined_duration": combined_tokens, # Token count
        "total_individual_duration": total_individual_tokens, # Token count
        "measured_ranking": ",".join(measured_ranking_list),
        "score": current_score,
        "model": current_model_to_use,
        "ranking_failure_reason": ranking_failure_reason
    }

    return {
        "group_id": group_id_val,
        "status": "success",
        "combined_ids_entry": combined_ids_entry,
        "question_sets_entry": question_sets_entry,
        "individual_results_entries": individual_results_entries,
        "group_correct_count": group_correct_count,
        "num_questions_in_group": len(group),
        "score": current_score,
        "original_ids": original_ids,
        "ranking_failure_reason": ranking_failure_reason
    }


def load_arranged_questions_from_csv(csv_filepath):
    """
    Loads pre-arranged question groups from a CSV file.
    The CSV should have columns: group_id, q_num_in_group, original_question_id, 
                                 question, choices_str (pipe-separated), answer.
    Returns a list of tuples: (group_id_str, list_of_question_dicts).
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: Arranged questions CSV file not found at {csv_filepath}")
        return []
    
    try:
        df = pd.read_csv(csv_filepath)
        # Ensure required columns are present
        required_cols = ['group_id', 'q_num_in_group', 'original_question_id', 'question', 'choices_str', 'answer']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"Error: CSV file {csv_filepath} is missing required columns: {missing_cols}")
            return []
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return []

    all_groups_with_ids = []
        
    for group_id_from_csv, group_df in df.groupby('group_id'):
        current_group_questions = []
        # Sort by q_num_in_group to maintain order within the group
        group_df = group_df.sort_values(by='q_num_in_group')
        for _, row in group_df.iterrows():
            choices_list = []
            if pd.notna(row['choices_str']) and row['choices_str']:
                choices_list = row['choices_str'].split('|')
            
            question_dict = {
                "id": row['original_question_id'], # This is the original_question_id from the CSV
                "question": row['question'],
                "choices": choices_list,
                "answer": row['answer']
            }
            current_group_questions.append(question_dict)
        
        if len(current_group_questions) == 4: # Assuming fixed group size of 4
            all_groups_with_ids.append((str(group_id_from_csv), current_group_questions))
        else:
            print(f"Warning: Group {group_id_from_csv} in CSV does not have exactly 4 questions ({len(current_group_questions)} found). Skipping group.")
            
    print(f"Loaded {len(all_groups_with_ids)} question groups (each with its group_id) from {csv_filepath}")
    return all_groups_with_ids

def load_existing_individual_results(model_name):
    """
    Load existing individual results for a model to get data for already processed questions.
    Returns a dictionary mapping question_id to (is_correct, duration_tokens)
    """
    model_dir = f"results/{model_name.replace(':', '-')}"
    individual_results_file = f"{model_dir}/individual_results.csv"
    
    existing_results = {}
    if os.path.exists(individual_results_file):
        try:
            df = pd.read_csv(individual_results_file)
            for _, row in df.iterrows():
                question_id = row.get('question_id')
                is_correct = row.get('is_correct', False)
                duration = row.get('duration', float('inf'))
                if pd.notna(question_id):
                    existing_results[question_id] = (is_correct, duration if pd.notna(duration) else float('inf'))
            print(f"Loaded {len(existing_results)} existing individual results for {model_name}")
        except Exception as e:
            print(f"Error loading existing individual results for {model_name}: {e}")
    
    return existing_results

def opening_message():
    print(f"Using model: {model_to_use}")
    credits = get_credits()
    print(type(credits))
    print(f"the amount of credits are: {credits} ({(credits['total_credits'] - credits['total_usage']):.2f}). \n do you want to continue? (y/n)")
    user_input = input().strip().lower()
    if user_input != 'y':
        print("Exiting without processing.")
        exit(0)
def main():
    # Initialize result lists at the very start of main
    all_results_combined_ids = []
    all_results_question_sets = []
    all_results_individual = []
    overall_scores_values = []
    total_individual_correct_overall = 0
    total_individual_questions_overall = 0
    newly_processed_qids_for_checkpoint = set()
    global model_to_use
    opening_message()

    results_dir = f"results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    print(f"Loading arranged benchmark questions from {ARRANGED_QUESTIONS_CSV}...")
    # Load questions from the CSV file. This now returns list of (group_id, questions_list)
    all_question_groups_with_ids = load_arranged_questions_from_csv(ARRANGED_QUESTIONS_CSV)

    if not all_question_groups_with_ids:
        print("Failed to load any question groups from CSV. Exiting.")
        return

    # Load existing individual results to avoid reprocessing
    existing_individual_results = load_existing_individual_results(model_to_use)

    # Checkpoint file logic remains to avoid re-processing individual questions for the *same model*
    # if the script is run multiple times. The primary goal of using CSV is that *different models*
    # get the exact same question sets.
    checkpoint_file = PROCESSED_IDS_CSV
    processed_ids_set = set()
    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            if "question_model" in df_checkpoint.columns:
                processed_ids_set = set(df_checkpoint["question_model"].tolist())
                print(f"Found {len(processed_ids_set)} processed question-model keys in checkpoint.")
            else:
                print(f"Checkpoint file '{checkpoint_file}' is missing 'question_model' column.")
        except pd.errors.EmptyDataError:
            print(f"Checkpoint file '{checkpoint_file}' is empty.")
        except Exception as e:
            print(f"Error reading checkpoint file '{checkpoint_file}': {e}")

    tasks_for_pool = []
    groups_added_to_pool = 0

    for original_group_id, group_questions_list in all_question_groups_with_ids:
        if groups_added_to_pool >= max_groups_to_process:
            print(f"Reached max_groups_to_process ({max_groups_to_process}). Not adding more groups.")
            break
        
        # Ensure the group actually has 4 questions, though load_arranged_questions_from_csv should handle this.
        if len(group_questions_list) != 4:
            print(f"Warning: Group {original_group_id} from CSV does not have 4 questions after loading. Skipping.")
            continue

        # Adding the processed_ids_set to each task
        tasks_for_pool.append(
            (group_questions_list, 
             original_group_id,  # Use the group_id from the CSV
             "MMLU_ARRANGED_CSV", # Dataset identifier for this run
             model_to_use,
             processed_ids_set,  # Pass the processed IDs set to check if group already processed
             existing_individual_results)  # Pass existing individual results
        )
        groups_added_to_pool += 1

    if not tasks_for_pool:
        print("No groups to process after considering max_groups_to_process or other filters. Exiting.")
        return

    print(f"\nPrepared {len(tasks_for_pool)} groups for parallel processing using model: {model_to_use}.")
    
    # Determine number of processes
    num_worker_processes = min(len(tasks_for_pool), (os.cpu_count() or 1))
    print(f"Starting multiprocessing pool with {num_worker_processes} worker processes...")

    with multiprocessing.Pool(processes=num_worker_processes) as pool:
        # pool_results is a list of dictionaries returned by process_group_wrapper
        parallel_run_results = pool.map(process_group_wrapper, tasks_for_pool)

    print("\nParallel processing complete. Aggregating results...")

    newly_processed_qids_for_checkpoint = set()
    
    # Error tracking counters
    api_problem_count = 0
    zero_tokens_count = 0
    blocked_response_count = 0
    unparsable_output_count = 0
    processing_error_count = 0
    success_count = 0
    skipped_count = 0

    for result in parallel_run_results:
        if result and result.get("status") == "success":
            success_count += 1
            all_results_combined_ids.append(result["combined_ids_entry"])
            all_results_question_sets.append(result["question_sets_entry"])
            all_results_individual.extend(result["individual_results_entries"])
            
            total_individual_correct_overall += result["group_correct_count"]
            total_individual_questions_overall += result["num_questions_in_group"]
            if result["score"] is not None:
                overall_scores_values.append(result["score"])
            
            # Track ranking failures even in successful groups
            if result.get("ranking_failure_reason"):
                failure_reason = result["ranking_failure_reason"]
                if failure_reason == "API_PROBLEM":
                    api_problem_count += 1
                elif failure_reason == "ZERO_TOKENS":
                    zero_tokens_count += 1
                elif failure_reason == "BLOCKED_RESPONSE":
                    blocked_response_count += 1
                elif failure_reason == "UNPARSABLE_OUTPUT":
                    unparsable_output_count += 1
            
            # Add successfully processed original qids (with model) to checkpoint set
            for qid in result["original_ids"]:
                 newly_processed_qids_for_checkpoint.add(f"{qid}_{model_to_use}")
                
        elif result and result.get("status") == "skipped_already_processed":
            skipped_count += 1
            # No need to add these results to result files as they already exist
            print(f"Skipped group with IDs: {result.get('original_ids')} as all questions were already processed by {result.get('model')}")
            # We don't add the IDs to checkpoint file as they're already there
            
        elif result and result.get("status") in ["failed_api_problem", "failed_zero_tokens", "failed_blocked_response", "failed_other_error", "failed_combined_prompt"]:
            status = result.get("status")
            if status == "failed_api_problem":
                api_problem_count += 1
                print(f"API PROBLEM (UNACCEPTABLE) for group {result.get('group_id')}: {result.get('raw_combined_answer')}")
            elif status == "failed_zero_tokens":
                zero_tokens_count += 1
                print(f"ZERO TOKENS for group {result.get('group_id')}: {result.get('raw_combined_answer')}")
            elif status == "failed_blocked_response":
                blocked_response_count += 1
                print(f"BLOCKED RESPONSE for group {result.get('group_id')}: {result.get('raw_combined_answer')}")
            else:
                processing_error_count += 1
                print(f"Group {result.get('group_id')} had issues: {result.get('status')}")
                
            if "combined_ids_entry" in result and result["combined_ids_entry"] is not None:
                all_results_combined_ids.append(result["combined_ids_entry"])

            # Create an entry for question_sets.csv to log the raw output from combined prompt failure
            failed_qset_entry = {
                "group_id": result.get("group_id"),
                "llm_ranking": "", # Empty as ranking failed
                "raw_llm_ranking_output": result.get("raw_combined_answer"), # Save the raw output (which is None)
                "combined_duration": None,
                "total_individual_duration": None,
                "measured_ranking": "",
                "score": None,
                "model": result.get("model"),
                "ranking_failure_reason": status.replace("failed_", "").upper()
            }
            all_results_question_sets.append(failed_qset_entry)
            # Optionally, log original_ids to checkpoint for failed groups if you want to avoid reprocessing them
            # for qid in result.get("original_ids", []):
            # newly_processed_qids_for_checkpoint.add(f"{qid}_{result.get('model')}")

        elif result: # Handle other potential partial failures if any are defined
            processing_error_count += 1
            print(f"Group {result.get('group_id')} had issues: {result.get('status')}")
            # Ensure combined_ids_entry is logged if available, even for other failures
            if "combined_ids_entry" in result and result["combined_ids_entry"] is not None:
                all_results_combined_ids.append(result["combined_ids_entry"])

    # Create the model-specific directory if it doesn't exist
    model_to_use = model_to_use.replace(":", "-")
    model_dir = os.path.join(results_dir, model_to_use)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Created directory: {model_dir}")
    else:
        print(f"Using existing directory: {model_dir}")
    # Save aggregated results to CSV files
    if all_results_combined_ids:
        pd.DataFrame(all_results_combined_ids).to_csv(f"{results_dir}/{model_to_use}/combined_ids.csv", index=False)
        print(f"Saved {len(all_results_combined_ids)} entries to combined_ids.csv")
    if all_results_question_sets:
        pd.DataFrame(all_results_question_sets).to_csv(f"{results_dir}/{model_to_use}/question_sets.csv", index=False)
        print(f"Saved {len(all_results_question_sets)} entries to question_sets.csv")
    if all_results_individual:
        pd.DataFrame(all_results_individual).to_csv(f"{results_dir}/{model_to_use}/individual_results.csv", index=False)
        print(f"Saved {len(all_results_individual)} entries to individual_results.csv")

    if overall_scores_values:
        avg_score = sum(overall_scores_values) / len(overall_scores_values)
        print("\n===== Overall Results =====")
        print(f"Processed {len(tasks_for_pool)} groups (up to {max_groups_to_process} requested).") # tasks_for_pool has the actual count submitted
        print(f"Average ranking score: {avg_score:.3f} / 1 (based on {len(overall_scores_values)} scored groups)")
    else:
        print("\nNo valid ranking scores computed for any group.")

    # Comprehensive error reporting
    print("\n===== Error Analysis =====")
    total_groups = len(parallel_run_results)
    print(f"Total groups processed: {total_groups}")
    print(f"Successful groups: {success_count}")
    print(f"Skipped groups (already processed): {skipped_count}")
    print(f"Failed groups breakdown:")
    
    if api_problem_count > 0:
        print(f"  🚨 API PROBLEMS (UNACCEPTABLE): {api_problem_count} groups")
        print(f"     This represents {api_problem_count/total_groups*100:.1f}% of all groups")
    
    if zero_tokens_count > 0:
        print(f"  ⚠️  ZERO TOKENS: {zero_tokens_count} groups")
        print(f"     This represents {zero_tokens_count/total_groups*100:.1f}% of all groups")
    
    if blocked_response_count > 0:
        print(f"  🛡️  BLOCKED RESPONSES: {blocked_response_count} groups")
        print(f"     This represents {blocked_response_count/total_groups*100:.1f}% of all groups")
    
    if unparsable_output_count > 0:
        print(f"  🔧 UNPARSABLE OUTPUT: {unparsable_output_count} groups")
        print(f"     This represents {unparsable_output_count/total_groups*100:.1f}% of all groups")
    
    if processing_error_count > 0:
        print(f"  ❌ OTHER PROCESSING ERRORS: {processing_error_count} groups")
        print(f"     This represents {processing_error_count/total_groups*100:.1f}% of all groups")
    
    # Summary severity assessment
    if api_problem_count > 0:
        print(f"\n🚨 CRITICAL: {api_problem_count} API problems detected - these are unacceptable and need investigation!")
    
    failure_rate = (total_groups - success_count - skipped_count) / total_groups * 100 if total_groups > 0 else 0
    if failure_rate > 10:
        print(f"⚠️  HIGH FAILURE RATE: {failure_rate:.1f}% of groups failed - consider investigating")
    elif failure_rate > 5:
        print(f"⚠️  MODERATE FAILURE RATE: {failure_rate:.1f}% of groups failed")
    else:
        print(f"✅ LOW FAILURE RATE: {failure_rate:.1f}% of groups failed")

    if total_individual_questions_overall > 0:
        accuracy = (total_individual_correct_overall / total_individual_questions_overall) * 100
        print(f"\nOverall, the LLM answered {total_individual_correct_overall} out of {total_individual_questions_overall} individual questions correctly ({accuracy:.2f}%).")
    else:
        print("\nNo individual questions were processed.")

    # Update the checkpoint file
    if newly_processed_qids_for_checkpoint:
        # Combine with existing processed IDs from the start of the run
        final_processed_ids_for_checkpoint = list(processed_ids_set.union(newly_processed_qids_for_checkpoint))
        df_new_checkpoint = pd.DataFrame({"question_model": final_processed_ids_for_checkpoint})
        df_new_checkpoint.to_csv(checkpoint_file, index=False)
        print(f"Checkpoint file '{checkpoint_file}' updated with {len(newly_processed_qids_for_checkpoint)} new entries. Total: {len(final_processed_ids_for_checkpoint)}.")
    else:
        print("No new question-model keys to add to checkpoint.")


if __name__ == "__main__":
    # Required for multiprocessing to work correctly on Windows
    multiprocessing.freeze_support() 
    main()
