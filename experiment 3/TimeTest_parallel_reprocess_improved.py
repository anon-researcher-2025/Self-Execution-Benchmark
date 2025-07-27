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

index = 13
# Global model selection – change this value to run with a different LLM.
model_to_use = models_to_use[index]  # Default to the first model in the list

# Maximum groups to process in a single run.
max_groups_to_process = 250 # Allows processing up to 1000 questions (250 groups * 4 questions)

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
    
    return None, None, "Unknown error after max retries"

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
    group, group_id_val, dataset_choice_val, current_model_to_use, processed_ids_set = args
    
    # Check if all questions in this group have already been processed by this model
    if check_group_already_processed(group, current_model_to_use, processed_ids_set):
        print(f"Skipping group {group_id_val} - all questions already processed by {current_model_to_use}")
        return {
            "group_id": group_id_val,
            "status": "skipped_already_processed",
            "original_ids": [q["id"] for q in group],
            "model": current_model_to_use
        }
    
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
    
    combined_answer, combined_tokens = ask_llm(combined_prompt, model=current_model_to_use)
    # raw_ranking_output_for_csv = combined_answer # Store raw output regardless of parsing success # Deferred assignment

    if combined_answer is None:
        print(f"Skipping group {group_id_val} due to combined prompt failure (ask_llm returned None).")
        reason_for_failure = "Combined prompt LLM call failed or returned no answer"
        return {
            "group_id": group_id_val, "status": "failed_combined_prompt", 
            "original_ids": original_ids, "model": current_model_to_use,
            "individual_results_entries": [], "question_sets_entry": None,
            "combined_ids_entry": combined_ids_entry, "group_correct_count": 0,
            "num_questions_in_group": len(group), "score": None,
            "raw_combined_answer": reason_for_failure # Store the reason string here
        }

    # If we reach here, combined_answer is NOT None.
    raw_ranking_output_for_csv = combined_answer # Assign now for the success path.

    individual_simplicity_data_map = {}  # Stores (is_correct, duration_tokens) by label
    individual_results_entries = []
    group_correct_count = 0

    def process_single_individual_q(label_arg, q_item_arg, model_arg, group_id_for_log):
        valid_letters_arg = [chr(ord('A') + i) for i in range(len(q_item_arg["choices"]))]
        individual_prompt_arg = (
            f"{q_item_arg['question']}\\n" + "\\n".join(q_item_arg["choices"]) +
            "\\n\\nAfter writing your chain of thought (if needed), provide your answer using JSON format with a single key \\\"answer\\\" whose value is the letter corresponding to your final answer."
        )
        answer_text_res, duration_tokens_res = ask_llm(individual_prompt_arg, model=model_arg)
        
        is_correct_flag_res = False
        extracted_ans_res = None
        if answer_text_res is not None:
            extracted_ans_res = extract_answer_letter(answer_text_res, valid_letters_arg)
            if extracted_ans_res == q_item_arg["answer"].upper():
                is_correct_flag_res = True
        
        return {
            "label": label_arg,
            "is_correct": is_correct_flag_res,
            "duration_tokens": duration_tokens_res if duration_tokens_res is not None else float('inf'),
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
                "model": model_arg
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
            entry_data_from_thread = result_item["entry_data"]
            # group_id is now added directly in process_single_individual_q
            individual_results_entries.append(entry_data_from_thread)
            individual_simplicity_data_map[result_item["label"]] = (result_item["is_correct"], result_item["duration_tokens"])
            if result_item["is_correct"]:
                group_correct_count += 1
        except Exception as exc:
            print(f"Error processing individual question for label {original_label_for_future} in group {group_id_val}: {exc}")
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
                "error_message": str(exc)
            }
            individual_results_entries.append(error_entry)

    # Ensure individual_simplicity_data is ordered correctly according to local_labels for measured_ranking
    individual_simplicity_data = [individual_simplicity_data_map.get(lbl, (False, float('inf'))) for lbl in local_labels]
    total_individual_tokens = sum(x[1] for x in individual_simplicity_data if x[1] != float('inf') and x[1] is not None)
    
    llm_ranking_list = extract_ranking(combined_answer) # combined_answer might be None if ask_llm failed before the check
    if llm_ranking_list is None or not isinstance(llm_ranking_list, list) or len(llm_ranking_list) != 4:
        if combined_answer is not None: # Only print warning if there was an answer to parse
            print(f"Warning: Failed to extract valid ranking for group {group_id_val}. LLM output: {combined_answer}")
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
        "model": current_model_to_use
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
        "original_ids": original_ids 
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
             processed_ids_set)  # Pass the processed IDs set to check if group already processed
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

    for result in parallel_run_results:
        if result and result.get("status") == "success":
            all_results_combined_ids.append(result["combined_ids_entry"])
            all_results_question_sets.append(result["question_sets_entry"])
            all_results_individual.extend(result["individual_results_entries"])
            
            total_individual_correct_overall += result["group_correct_count"]
            total_individual_questions_overall += result["num_questions_in_group"]
            if result["score"] is not None:
                overall_scores_values.append(result["score"])
            
            # Add successfully processed original qids (with model) to checkpoint set
            for qid in result["original_ids"]:
                 newly_processed_qids_for_checkpoint.add(f"{qid}_{model_to_use}")
                
        elif result and result.get("status") == "skipped_already_processed":
            # אין צורך להוסיף את התוצאות האלה לקבצי התוצאות כי הם כבר קיימים
            print(f"Skipped group with IDs: {result.get('original_ids')} as all questions were already processed by {result.get('model')}")
            # אנחנו לא מוסיפים את ה-IDs לקובץ ה-checkpoint כי הם כבר שם
            
        elif result and result.get("status") == "failed_combined_prompt":
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
                "model": result.get("model")
            }
            all_results_question_sets.append(failed_qset_entry)
            # Optionally, log original_ids to checkpoint for failed groups if you want to avoid reprocessing them
            # for qid in result.get("original_ids", []):
            # newly_processed_qids_for_checkpoint.add(f"{qid}_{result.get('model')}")

        elif result: # Handle other potential partial failures if any are defined
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
