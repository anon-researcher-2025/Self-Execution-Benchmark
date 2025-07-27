import os
import pandas as pd
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import requests
import time
import re
import json

# Import functions from the original script
from TimeTest_parallel import (
    ask_llm, extract_ranking, extract_answer_letter, score_ranking,
    extract_json_from_text, get_credits
)

# Configuration
ARRANGED_QUESTIONS_CSV = "arranged_question_sets.csv"
PROCESSED_IDS_CSV = "processed_ids.csv"

# Read API keys
key_file = "openai_key.txt"
openai_api_key_val = None
openrouter_api_key_val = None

if os.path.exists(key_file):
    with open(key_file, "r") as f:
        lines = f.readlines()
        if len(lines) > 0:
            openai_api_key_val = lines[0].strip()
            openai.api_key = openai_api_key_val
        if len(lines) > 1:
            openrouter_api_key_val = lines[1].strip()
else:
    raise FileNotFoundError(f"API key file '{key_file}' not found.")

def load_arranged_questions_from_csv(csv_filepath):
    """Load pre-arranged question groups from CSV file"""
    if not os.path.exists(csv_filepath):
        print(f"Error: Arranged questions CSV file not found at {csv_filepath}")
        return []
    
    try:
        df = pd.read_csv(csv_filepath)
        required_cols = ['group_id', 'q_num_in_group', 'original_question_id', 'question', 'choices_str', 'answer']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"Error: CSV file is missing required columns: {missing_cols}")
            return []
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return []

    all_groups_with_ids = []
    
    for group_id_from_csv, group_df in df.groupby('group_id'):
        current_group_questions = []
        group_df = group_df.sort_values(by='q_num_in_group')
        for _, row in group_df.iterrows():
            choices_list = []
            if pd.notna(row['choices_str']) and row['choices_str']:
                choices_list = row['choices_str'].split('|')
            
            question_dict = {
                "id": row['original_question_id'],
                "question": row['question'],
                "choices": choices_list,
                "answer": row['answer']
            }
            current_group_questions.append(question_dict)
        
        if len(current_group_questions) == 4:
            all_groups_with_ids.append((str(group_id_from_csv), current_group_questions))
        else:
            print(f"Warning: Group {group_id_from_csv} does not have exactly 4 questions. Skipping.")
    
    return all_groups_with_ids

def find_incomplete_groups(model_name, all_groups_dict):
    """Find groups that are incomplete (missing or have no score) for the given model"""
    # Normalize model name for file path
    model_dir_name = model_name.replace(":", "-")
    results_file = f"results/{model_dir_name}/question_sets.csv"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print(f"All groups need to be processed for model: {model_name}")
        return list(all_groups_dict.keys())
    
    try:
        results_df = pd.read_csv(results_file)
        
        # Get all group IDs that exist in arranged_questions_sets.csv
        all_group_ids = set(all_groups_dict.keys())
        
        # Handle duplicates by keeping only the LATEST entry for each group_id
        # (assuming later entries are more recent/better)
        results_df = results_df.drop_duplicates(subset=['group_id'], keep='last')
        
        # Get group IDs that have results with valid scores
        completed_groups = set()
        for _, row in results_df.iterrows():
            if (pd.notna(row.get('score')) and 
                row.get('score') is not None and 
                row.get('group_id') in all_group_ids):
                completed_groups.add(row['group_id'])
        
        # Find incomplete groups (missing or with no score)
        incomplete_groups = list(all_group_ids - completed_groups)
        
        print(f"Model: {model_name}")
        print(f"Total groups available: {len(all_group_ids)}")
        print(f"Total rows in results file: {len(pd.read_csv(results_file))}")
        print(f"Unique groups after deduplication: {len(results_df)}")
        print(f"Completed groups with valid scores: {len(completed_groups)}")
        print(f"Incomplete groups to reprocess: {len(incomplete_groups)}")
        
        return incomplete_groups
        
    except Exception as e:
        print(f"Error reading results file {results_file}: {e}")
        return list(all_groups_dict.keys())

def process_single_individual_q(label_arg, q_item_arg, model_arg, group_id_for_log):
    """Process a single individual question"""
    valid_letters_arg = [chr(ord('A') + i) for i in range(len(q_item_arg["choices"]))]
    individual_prompt_arg = (
        f"{q_item_arg['question']}\n" + "\n".join(q_item_arg["choices"]) +
        "\n\nAfter writing your chain of thought (if needed), provide your answer using JSON format with a single key \"answer\" whose value is the letter corresponding to your final answer."
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
            "choices_text": "\n".join(q_item_arg["choices"]),
            "duration": duration_tokens_res,
            "llm_answer_raw": answer_text_res,
            "llm_answer_extracted": extracted_ans_res,
            "correct_answer": q_item_arg["answer"].upper(),
            "is_correct": is_correct_flag_res,
            "model": model_arg
        }
    }

def process_group_for_reprocessing(group_id_val, group_questions, current_model_to_use):
    """Process a single group of questions (similar to process_group_wrapper but simplified)"""
    # Ensure API keys are accessible
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
    original_ids = [q["id"] for q in group_questions]

    print(f"----- Reprocessing Group {group_id_val} (Model: {current_model_to_use}) -----")

    # Create combined prompt
    combined_prompt = (
        "You must rank the following multiple-choice questions based on how many tokens you would use when asked to solve in a seperate session (from least tokens to most tokens). "
        "However, questions that you will provide an incorrect answer if asked, must be ranked last (regardless of how many tokens will be used to answer). "
        "Note that each question is followed by its correct answer."
        "Each question is labeled with a unique identifier: " + ", ".join(local_labels) + ".\n"
        "After writing your chain of thought (if needed), provide an answer using JSON format with a single key \"ranking\" whose value is a list of the question identifiers (using the above labels) in order from fastest to slowest. "
        "Questions:\n" +
        f"{local_labels[0]}. {group_questions[0]['question']}\n" + "\n".join(group_questions[0]['choices']) + f"\nCorrect Answer: {group_questions[0]['answer']}\n\n" +
        f"{local_labels[1]}. {group_questions[1]['question']}\n" + "\n".join(group_questions[1]['choices']) + f"\nCorrect Answer: {group_questions[1]['answer']}\n\n" +
        f"{local_labels[2]}. {group_questions[2]['question']}\n" + "\n".join(group_questions[2]['choices']) + f"\nCorrect Answer: {group_questions[2]['answer']}\n\n" +
        f"{local_labels[3]}. {group_questions[3]['question']}\n" + "\n".join(group_questions[3]['choices']) + f"\nCorrect Answer: {group_questions[3]['answer']}"
    )
    
    combined_answer, combined_tokens = ask_llm(combined_prompt, model=current_model_to_use)
    
    if combined_answer is None:
        print(f"Failed to get combined answer for group {group_id_val}")
        return None

    # Process individual questions concurrently
    individual_simplicity_data_map = {}
    individual_results_entries = []
    group_correct_count = 0

    future_to_q_meta = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for lbl, q_itm in zip(local_labels, group_questions):
            future = executor.submit(process_single_individual_q, lbl, q_itm, current_model_to_use, group_id_val)
            future_to_q_meta[future] = lbl

    for future_item in as_completed(future_to_q_meta):
        original_label_for_future = future_to_q_meta[future_item]
        try:
            result_item = future_item.result()
            individual_results_entries.append(result_item["entry_data"])
            individual_simplicity_data_map[result_item["label"]] = (result_item["is_correct"], result_item["duration_tokens"])
            if result_item["is_correct"]:
                group_correct_count += 1
        except Exception as exc:
            print(f"Error processing individual question {original_label_for_future}: {exc}")
            individual_simplicity_data_map[original_label_for_future] = (False, float('inf'))

    # Calculate results
    individual_simplicity_data = [individual_simplicity_data_map.get(lbl, (False, float('inf'))) for lbl in local_labels]
    total_individual_tokens = sum(x[1] for x in individual_simplicity_data if x[1] != float('inf') and x[1] is not None)
    
    llm_ranking_list = extract_ranking(combined_answer)
    if llm_ranking_list is None or not isinstance(llm_ranking_list, list) or len(llm_ranking_list) != 4:
        print(f"Warning: Failed to extract valid ranking for group {group_id_val}")
        llm_ranking_list = []

    # Sort by (not correctness, token_duration)
    measured_order_tuples = sorted(zip(local_labels, individual_simplicity_data), key=lambda x: (not x[1][0], x[1][1]))
    measured_ranking_list = [label for label, _ in measured_order_tuples]

    current_score = None
    if llm_ranking_list and len(llm_ranking_list) == 4:
        current_score = score_ranking(llm_ranking_list, measured_ranking_list)

    # Create result entries
    combined_ids_entry = {
        "group_id": group_id_val,
        "question_ids": ",".join(original_ids),
        "model": current_model_to_use
    }

    question_sets_entry = {
        "group_id": group_id_val,
        "llm_ranking": ",".join(llm_ranking_list),
        "raw_llm_ranking_output": combined_answer,
        "combined_duration": combined_tokens,
        "total_individual_duration": total_individual_tokens,
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
        "num_questions_in_group": len(group_questions),
        "score": current_score,
        "original_ids": original_ids
    }

def save_single_group_results(model_name, combined_ids_entry, question_sets_entry, individual_results_entries):
    """Save results for a single group immediately after processing"""
    model_dir_name = model_name.replace(":", "-")
    model_dir = f"results/{model_dir_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save combined_ids.csv
    combined_file = f"{model_dir}/combined_ids.csv"
    if os.path.exists(combined_file):
        existing_df = pd.read_csv(combined_file)
        new_df = pd.DataFrame([combined_ids_entry])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(combined_file, index=False)
    else:
        pd.DataFrame([combined_ids_entry]).to_csv(combined_file, index=False)
    
    # Save question_sets.csv
    question_sets_file = f"{model_dir}/question_sets.csv"
    if os.path.exists(question_sets_file):
        existing_df = pd.read_csv(question_sets_file)
        new_df = pd.DataFrame([question_sets_entry])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(question_sets_file, index=False)
    else:
        pd.DataFrame([question_sets_entry]).to_csv(question_sets_file, index=False)
    
    # Save individual_results.csv
    individual_file = f"{model_dir}/individual_results.csv"
    if os.path.exists(individual_file):
        existing_df = pd.read_csv(individual_file)
        new_df = pd.DataFrame(individual_results_entries)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(individual_file, index=False)
    else:
        pd.DataFrame(individual_results_entries).to_csv(individual_file, index=False)

def update_checkpoint_for_group(model_name, original_ids):
    """Update checkpoint file immediately after processing a group"""
    newly_processed_qids = set()
    for qid in original_ids:
        newly_processed_qids.add(f"{qid}_{model_name}")
    
    if os.path.exists(PROCESSED_IDS_CSV):
        existing_df = pd.read_csv(PROCESSED_IDS_CSV)
        existing_ids = set(existing_df["question_model"].tolist())
    else:
        existing_ids = set()
    
    final_processed_ids = list(existing_ids.union(newly_processed_qids))
    df_new_checkpoint = pd.DataFrame({"question_model": final_processed_ids})
    df_new_checkpoint.to_csv(PROCESSED_IDS_CSV, index=False)

def clean_duplicate_entries(model_name):
    """Clean duplicate entries from existing CSV files, keeping the latest/best entry for each group"""
    model_dir_name = model_name.replace(":", "-")
    model_dir = f"results/{model_dir_name}"
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return
    
    # Clean question_sets.csv
    question_sets_file = f"{model_dir}/question_sets.csv"
    if os.path.exists(question_sets_file):
        print(f"Cleaning duplicates from {question_sets_file}...")
        df = pd.read_csv(question_sets_file)
        original_count = len(df)
        
        # Keep the last entry for each group_id (assuming it's the most recent/correct)
        df_cleaned = df.drop_duplicates(subset=['group_id'], keep='last')
        new_count = len(df_cleaned)
        
        # Save the cleaned data
        df_cleaned.to_csv(question_sets_file, index=False)
        print(f"  Removed {original_count - new_count} duplicate entries")
        print(f"  {new_count} unique groups remaining")
    
    # Clean combined_ids.csv
    combined_ids_file = f"{model_dir}/combined_ids.csv"
    if os.path.exists(combined_ids_file):
        print(f"Cleaning duplicates from {combined_ids_file}...")
        df = pd.read_csv(combined_ids_file)
        original_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=['group_id'], keep='last')
        new_count = len(df_cleaned)
        
        df_cleaned.to_csv(combined_ids_file, index=False)
        print(f"  Removed {original_count - new_count} duplicate entries")
    
    # Clean individual_results.csv
    individual_results_file = f"{model_dir}/individual_results.csv"
    if os.path.exists(individual_results_file):
        print(f"Cleaning duplicates from {individual_results_file}...")
        df = pd.read_csv(individual_results_file)
        original_count = len(df)
        
        # For individual results, we need to be more careful
        # Keep the latest set of individual results for each group
        df_cleaned = df.drop_duplicates(subset=['group_id', 'question_id'], keep='last')
        new_count = len(df_cleaned)
        
        df_cleaned.to_csv(individual_results_file, index=False)
        print(f"  Removed {original_count - new_count} duplicate entries")

def reprocess_incomplete_groups(model_name, max_groups=None):
    """Main function to reprocess incomplete groups for a given model"""
    print(f"=== Reprocessing Incomplete Groups for Model: {model_name} ===\n")
    
    # Show current credits
    credits = get_credits()
    if credits:
        remaining = credits.get('total_credits', 0) - credits.get('total_usage', 0)
        print(f"Current credits: {remaining:.2f}")
        user_input = input("Do you want to continue? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting without processing.")
            return
    
    # Load all question groups
    print("Loading arranged question groups...")
    all_groups_list = load_arranged_questions_from_csv(ARRANGED_QUESTIONS_CSV)
    all_groups_dict = {group_id: questions for group_id, questions in all_groups_list}
    
    # Find incomplete groups
    print("Analyzing incomplete groups...")
    incomplete_group_ids = find_incomplete_groups(model_name, all_groups_dict)
    
    if not incomplete_group_ids:
        print("No incomplete groups found. All groups appear to be processed successfully.")
        return
    
    # Limit the number of groups if specified
    if max_groups and len(incomplete_group_ids) > max_groups:
        print(f"Limiting to {max_groups} groups (out of {len(incomplete_group_ids)} incomplete)")
        incomplete_group_ids = incomplete_group_ids[:max_groups]
    
    print(f"Processing {len(incomplete_group_ids)} incomplete groups...")
    
    # Process groups
    overall_scores_values = []
    total_individual_correct_overall = 0
    total_individual_questions_overall = 0
    groups_processed = 0
    
    for i, group_id in enumerate(incomplete_group_ids, 1):
        print(f"\nProcessing group {i}/{len(incomplete_group_ids)}: {group_id}")
        group_questions = all_groups_dict[group_id]
        
        result = process_group_for_reprocessing(group_id, group_questions, model_name)
        
        if result and result.get("status") == "success":
            # Save results immediately after each group
            save_single_group_results(
                model_name, 
                result["combined_ids_entry"], 
                result["question_sets_entry"], 
                result["individual_results_entries"]
            )
            
            # Update checkpoint immediately
            update_checkpoint_for_group(model_name, result["original_ids"])
            
            # Update counters
            total_individual_correct_overall += result["group_correct_count"]
            total_individual_questions_overall += result["num_questions_in_group"]
            if result["score"] is not None:
                overall_scores_values.append(result["score"])
            groups_processed += 1
                
            # Fix the formatting issue with None scores
            score_str = f"{result['score']:.3f}" if result['score'] is not None else 'N/A'
            print(f"✓ Group {group_id} completed successfully (Score: {score_str}) - Results saved!")
        else:
            print(f"✗ Group {group_id} failed to process")
    
    # Print summary (no need for bulk saving since we save after each group)
    print(f"\n=== Reprocessing Summary ===")
    print(f"Groups processed: {groups_processed}")
    if overall_scores_values:
        avg_score = sum(overall_scores_values) / len(overall_scores_values)
        print(f"Average ranking score: {avg_score:.3f}")
    if total_individual_questions_overall > 0:
        accuracy = (total_individual_correct_overall / total_individual_questions_overall) * 100
        print(f"Individual question accuracy: {accuracy:.2f}%")
    print("All results have been saved immediately after each group was processed.")

if __name__ == "__main__":
    # Configuration - change the model name here
    MODEL_TO_REPROCESS = "meta-llama/llama-3.2-3b-instruct"  # Change this to the model you want to reprocess
    MAX_GROUPS_TO_PROCESS = None  # Set to None to process all incomplete groups
    
    # Ask user if they want to clean existing duplicates first
    print("=== Duplicate Cleanup Tool ===")
    user_choice = input("Do you want to clean existing duplicates before reprocessing? (y/n): ").strip().lower()
    if user_choice == 'y':
        clean_duplicate_entries(MODEL_TO_REPROCESS)
        print("\nDuplicate cleanup completed!")
        
        # Ask if they want to continue with reprocessing
        continue_choice = input("\nDo you want to continue with reprocessing incomplete groups? (y/n): ").strip().lower()
        if continue_choice == 'y':
            reprocess_incomplete_groups(MODEL_TO_REPROCESS, MAX_GROUPS_TO_PROCESS)
        else:
            print("Exiting without reprocessing.")
    else:
        reprocess_incomplete_groups(MODEL_TO_REPROCESS, MAX_GROUPS_TO_PROCESS)