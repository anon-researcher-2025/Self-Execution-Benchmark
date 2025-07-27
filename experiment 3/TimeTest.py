import time
import random
import re
import json
import pandas as pd
from datasets import load_dataset
import openai
import requests
import os

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

# Global model selection – change this value to run with a different LLM.
model_to_use = "meta-llama/llama-3.1-8b-instruct"  # "o3-mini"  # e.g., "gpt-4o-mini", "o3-mini", etc.

# Maximum groups to process in a single run.
max_groups_to_process = 5

def get_benchmark_questions(dataset_choice="mmlu", num_questions=10000, topics=None):
    """
    Loads benchmark questions from one of two datasets:

    1. MMLU:
       - Loads the entire test split of the "cais/mmlu" dataset for each specified topic.
       - Each example is expected to have the following fields:
             "topic", "question", "choices", "answer"
         (where "choices" is a list of 4 answer options and "answer" is an integer index (0-based)).
       - Each example is reformatted into a dictionary with keys:
             "id": a unique id in the form "MMLU_{topic}_{FirstThreeWordsCamel}_{order}"
             "question": the question text,
             "choices": a list of "A. …", "B. …", etc.
             "answer": the correct answer letter.
    2. CommonSenseQA:
       - Loads the validation split of the "commonsense_qa" dataset.
       - Each example is reformatted into a dictionary with keys:
             "id": a unique id in the form "CommonSense_{FirstThreeWordsCamel}_{order}"
             "question": the question text,
             "choices": a list of labeled choices,
             "answer": the correct answer letter.
    Returns a list of question dictionaries.
    """
    questions = []
    if dataset_choice.lower() == "mmlu":
        if topics is None:
            topics = ["abstract_algebra", "anatomy", "astronomy"]  # default topics; update as desired
        all_items = []
        for topic in topics:
            try:
                ds_topic = load_dataset("cais/mmlu", topic, split="test")
                print(f"Loaded {len(ds_topic)} examples for topic '{topic}'.")
            except Exception as e:
                print(f"Error loading topic '{topic}' from MMLU: {e}")
                continue
            for i, item in enumerate(ds_topic):
                if i >= 100:
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
    elif dataset_choice.lower() == "commonsense":
        try:
            ds = load_dataset("commonsense_qa", split="validation")
        except Exception as e:
            print(f"Error loading CommonSenseQA dataset: {e}")
            return []
        ds = ds.shuffle(seed=42).select(range(num_questions))
        for i, item in enumerate(ds):
            words = item["question"].split()
            first_three = "".join(word.capitalize() for word in words[:3])
            question_text = item["question"]
            choices_list = item["choices"]
            labeled_choices = []
            for j, choice in enumerate(choices_list):
                letter = chr(ord('A') + j)
                labeled_choices.append(f"{letter}. {choice}")
            correct_letter = item["answer"].strip().upper()
            questions.append({
                "id": f"CommonSense_{first_three}_{i+1:04d}",
                "question": question_text,
                "choices": labeled_choices,
                "answer": correct_letter
            })
    else:
        print(f"Unknown dataset choice: {dataset_choice}")
        return []
    print(f"Total questions loaded: {len(questions)}")
    return questions

def ask_llm(prompt, model=model_to_use, temperature=0.0, max_retries=3, provider="openrouter"):
    """
    Sends a prompt to the LLM and returns its answer along with the token count.
    (Identical to the original TimeTest.py)
    """
    # Removed OpenRouter key file reading from here
    
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                # Ensure openai.api_key is set (should be by module-level load)
                if not openai.api_key: # This refers to openai_api_key_val loaded globally
                    print("OpenAI API key not set globally.")
                    # Attempt to reload if in a new process context (though module load should handle)
                    # This part might be redundant if global loading is robust
                    if os.path.exists(key_file):
                        with open(key_file, "r") as f:
                            lines = f.readlines()
                            if len(lines) > 0:
                                openai.api_key = lines[0].strip()
                    if not openai.api_key:
                         return None, None

                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                tokens = response["usage"]["completion_tokens"]
                text = response["choices"][0]["message"]["content"]
                
            elif provider == "openrouter":
                if not openrouter_api_key_val: # Check if the global OpenRouter key is loaded
                    print(f"OpenRouter API key not available from '{key_file}'.")
                    return None, None
                
                headers = {
                    "Authorization": f"Bearer {openrouter_api_key_val}", # Use the globally loaded key
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                api_response = requests.post( # Renamed to avoid conflict
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                api_response.raise_for_status() # Check for HTTP errors
                response_json = api_response.json()
                text = response_json["choices"][0]["message"]["content"]
                tokens = response_json["usage"]["completion_tokens"]
            else:
                print(f"Unknown provider: {provider}. Use 'openai' or 'openrouter'.")
                return None, None
                
            return text, tokens
            
        except Exception as e:
            print(f"Error during LLM call with {provider} (model: {model}, attempt: {attempt+1}): {e}. Retrying...")
            time.sleep(1) # Simplified sleep
    
    print(f"Failed to get a response from {provider} (model: {model}) after {max_retries} attempts.")
    return None, None


def extract_json_from_text(text, required_keys=None):
    """
    Attempts to extract a JSON object from the given text.

    Steps:
      1. Normalize non‑standard quotes.
      2. Look for a code‑fenced JSON block delimited by ```json and ``` using regex.
      3. If not found, try to parse the substring from the first "{" to the last "}".
      4. Finally, fall back to iterating over all candidate JSON-like substrings.

    If required_keys is provided (a list of keys), the returned JSON object must contain at least one of those keys (checked case‑insensitively).
    """
    # Normalize quotes.
    normalized_text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'").strip()

    # 1. Try to capture a JSON code block delimited by ```json ... ```
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, normalized_text, flags=re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            print("DEBUG: extract_json_from_text - Found JSON in code fence:", parsed)
            if isinstance(parsed, dict) and (required_keys is None or any(
                    key.lower() in (k.lower() for k in parsed.keys()) for key in required_keys)):
                return parsed
        except Exception as e:
            print("DEBUG: extract_json_from_text - Error parsing JSON from code fence:", e)

    # 2. Try to extract the substring between the first '{' and the last '}'
    first = normalized_text.find("{")
    last = normalized_text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = normalized_text[first:last + 1]
        try:
            parsed = json.loads(candidate)
            # print("DEBUG: extract_json_from_text - Parsed JSON from candidate substring:", parsed)
            if isinstance(parsed, dict) and (required_keys is None or any(
                    key.lower() in (k.lower() for k in parsed.keys()) for key in required_keys)):
                return parsed
        except Exception as e:
            print("DEBUG: extract_json_from_text - Error parsing candidate substring:", e)

    # 3. Fallback: search for all potential JSON-like substrings.
    candidates = re.findall(r'\{.*?\}', normalized_text, flags=re.DOTALL)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and (required_keys is None or any(
                    key.lower() in (k.lower() for k in parsed.keys()) for key in required_keys)):
                # print("DEBUG: extract_json_from_text - Found JSON in fallback candidate:", parsed)
                return parsed
        except Exception:
            continue
    # print("DEBUG: extract_json_from_text - No valid JSON found.")
    return None


def extract_answer_letter(text, valid_letters):
    """
    Extracts the answer letter from the given text.

    This function calls extract_json_from_text with required_keys=["answer"].
    It then looks for a key (case‑insensitively) matching "answer" in the parsed JSON.
    If found and the corresponding value (after stripping whitespace) is one of the valid letters,
    it returns that letter.
    Otherwise, it falls back to a regex search.
    """
    # print("DEBUG: extract_answer_letter - Starting extraction")
    parsed = extract_json_from_text(text, required_keys=["answer"])
    # print("DEBUG: extract_answer_letter - Parsed JSON:", parsed)
    if parsed and isinstance(parsed, dict):
        for key, value in parsed.items():
            if key.lower() == "answer" and isinstance(value, str):
                ans = value.strip().upper()
                # print("DEBUG: extract_answer_letter - Found answer in JSON:", ans)
                if ans in valid_letters:
                    return ans
                else:
                    print("DEBUG: extract_answer_letter - Answer not in valid_letters:", valid_letters)
    # Fallback: search using regex for a pattern like "answer": "X"
    fallback = re.search(r'"answer"\s*:\s*"([A-Z])"', text, flags=re.IGNORECASE)
    if fallback:
        ans = fallback.group(1).upper()
        # print("DEBUG: extract_answer_letter - Fallback found answer:", ans)
        if ans in valid_letters:
            return ans
    # Final fallback: scan through the text for the first valid letter.
    for char in text:
        if char.upper() in valid_letters:
            print("DEBUG: extract_answer_letter - Final fallback found letter:", char.upper())
            return char.upper()
    # print("DEBUG: extract_answer_letter - No answer letter found")
    return None


def extract_ranking(text):
    """
    Extracts the ranking list from the given text.

    This function calls extract_json_from_text with required_keys=["ranking"].
    It then looks for a key (case‑insensitively) matching "ranking" in the parsed JSON.
    If the corresponding value is a list, it is returned; if it's a string (e.g. comma‑separated), it splits it.
    """
    # print("DEBUG: extract_ranking - Starting extraction")
    parsed = extract_json_from_text(text, required_keys=["ranking"])
    # print("DEBUG: extract_ranking - Parsed JSON:", parsed)
    if parsed and isinstance(parsed, dict):
        for key, value in parsed.items():
            if key.lower() == "ranking":
                if isinstance(value, list):
                    # print("DEBUG: extract_ranking - Found ranking list:", value)
                    return value
                elif isinstance(value, str):
                    ranking_list = [item.strip() for item in value.split(",") if item.strip()]
                    print("DEBUG: extract_ranking - Parsed ranking list from string:", ranking_list)
                    return ranking_list
    # print("DEBUG: extract_ranking - No valid ranking found in JSON")
    return None


def score_ranking(llm_ranking, measured_ranking):
    """
    Computes the fraction (between 0 and 1) of unordered pairs that are ranked in the same relative order
    in both rankings. For n items, there are C(n, 2) unordered pairs (e.g. 6 pairs when n=4).
    """
    n = len(measured_ranking)
    if n < 2:
        return 0.0
    correct_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if llm_ranking.index(measured_ranking[i]) < llm_ranking.index(measured_ranking[j]):
                correct_pairs += 1
    return correct_pairs / total_pairs



def main():
    DATASET_CHOICE = "mmlu"  # or "commonsense"
    MMLU_TOPICS = ["abstract_algebra", "college_physics", "high_school_geography", "miscellaneous", "moral_scenarios", "global_facts", "formal_logic", "international_law", "business_ethics", "high_school_mathematics"]
    # MMLU_TOPICS = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    #                "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    #                "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics",
    #                "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts",
    #                "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    #                "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    #                "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    #                "high_school_physics", "high_school_psychology", "high_school_statistics",
    #                "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    #                "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    #                "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    #                "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    #                "professional_psychology", "public_relations", "security_studies", "sociology",
    #                "us_foreign_policy", "virology", "world_religions"]

    print(f"Downloading benchmark questions from {DATASET_CHOICE.upper()}...")
    benchmark_questions = get_benchmark_questions(dataset_choice=DATASET_CHOICE, num_questions=1000, topics=MMLU_TOPICS)
    if not benchmark_questions:
        print("Failed to download benchmark questions.")
        return

    # Build a checkpoint using composite keys: question id + "_" + model.
    checkpoint_file = "processed_ids.csv"
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        df_checkpoint = pd.read_csv(checkpoint_file)
        processed_ids = set(df_checkpoint["question_model"].tolist())
        print(f"Found {len(processed_ids)} processed question-model keys in checkpoint.")
    remaining_questions = [q for q in benchmark_questions if f"{q['id']}_{model_to_use}" not in processed_ids]
    print(f"{len(remaining_questions)} questions remain to be processed for model {model_to_use}.")


    chunk_size =100
    chunks = [remaining_questions[i:i + chunk_size] for i in range(0, len(remaining_questions), chunk_size)]
    
    print(f"Divided questions into {len(chunks)} chunks of size {chunk_size}.")
    print(f"Processing {len(remaining_questions)} questions in {len(chunks)} chunks.")    

    chunks = chunks # For testing, limit to the first 5 chunks.

    # Initialize CSV output lists.
    combined_ids = []       # Mapping of group id to original question ids.
    question_sets = []      # Group-level info.
    individual_results = [] # Individual question details.
    overall_scores = []
    total_individual_correct = 0
    total_individual_count = 0
    new_processed_ids = []  # Newly processed question-model keys.

    groups_processed = 0

    for chunk in chunks:
        print(f"\n=== Processing Chunk with {len(chunk)} questions ===")
        groups = [chunk[i:i + 4] for i in range(0, len(chunk), 4)]
        for group_idx, group in enumerate(groups):
            if len(group) < 4:
                continue  # Skip incomplete groups.
            if groups_processed >= max_groups_to_process:
                print(f"\nReached maximum groups ({max_groups_to_process}) for this run.")
                break

            local_labels = ["Q1", "Q2", "Q3", "Q4"]
            original_ids = [q["id"] for q in group]
            group_id = f"{DATASET_CHOICE.upper()}_GROUP_{groups_processed+1:03d}"
            print(f"\n----- Processing Group (Group ID: {group_id}) -----")
            for label, q in zip(local_labels, group):
                print(f"{label} (ID: {q['id']}): {q['question']}")
                for choice in q["choices"]:
                    print(f"   {choice}")
            combined_ids.append({
                "group_id": group_id,
                "question_ids": ",".join(original_ids),
                "model": model_to_use
            })

            combined_prompt = (
                "You must rank the following multiple-choice questions based on how many tokens you would use when asked to solve in a seperate session (from least tokens to most tokens). "
                "However, questions that you will provide an incorrect answer if asked, must be ranked last (regardless of how many tokens will be used to answer). "
                "Note that each question is followed by its correct answer."
                #"You must answer this combined question faster than it would take you to answer each individual question separately.\n"
                "Each question is labeled with a unique identifier: " + ", ".join(local_labels) + ".\n"
                "After writing your chain of thought (if needed), provide an answer using JSON format with a single key \"ranking\" whose value is a list of the question identifiers (using the above labels) in order from fastest to slowest. "
                "Questions:\n"
                f"{local_labels[0]}. {group[0]['question']}\n" +
                "\n".join(group[0]['choices']) + "\n" +
                f"Correct Answer: {group[0]['answer']}\n\n" +
                f"{local_labels[1]}. {group[1]['question']}\n" +
                "\n".join(group[1]['choices']) + "\n" +
                f"Correct Answer: {group[1]['answer']}\n\n" +
                f"{local_labels[2]}. {group[2]['question']}\n" +
                "\n".join(group[2]['choices']) + "\n" +
                f"Correct Answer: {group[2]['answer']}\n\n" +
                f"{local_labels[3]}. {group[3]['question']}\n" +
                "\n".join(group[3]['choices']) + "\n" +
                f"Correct Answer: {group[3]['answer']}"
            )
            print("\n--- Combined Prompt ---")
            print(combined_prompt)

            print("\nRunning combined prompt...")
            combined_answer, Tokens = ask_llm(combined_prompt, model=model_to_use)
            if combined_answer is None:
                print("Skipping group due to combined prompt failure.")
                continue
            print("\nCombined Answer (raw output):")
            print(combined_answer)
            print(f"Combined Duration: {Tokens:.2f} seconds")

            individual_simplicity = []
            group_correct_count = 0
            for label, q in zip(local_labels, group):
                valid_letters = [chr(ord('A') + i) for i in range(len(q["choices"]))]
                individual_prompt = (
                    f"{q['question']}\n" +
                    "\n".join(q["choices"]) +
                    "\n\nAfter writing your chain of thought (if needed), provide your answer using JSON format with a single key \"answer\" whose value is the letter corresponding to your final answer."
                )
                print(f"\nRunning individual question {label} (ID: {q['id']}):")
                print(individual_prompt)
                answer, duration = ask_llm(individual_prompt, model=model_to_use)
                print(f"Duration for {label}: {duration:.2f} tokens")
                if answer is None:
                    print(f"Skipping question {label} due to API error.")
                    individual_simplicity.append((False, duration))
                    individual_results.append({
                        "question_id": q["id"],
                        "duration": duration,
                        "llm_answer": None,
                        "correct_answer": q["answer"].upper(),
                        "is_correct": False,
                        "model": model_to_use
                    })
                else:
                    print("Answer (raw output):")
                    print(answer)
                    extracted = extract_answer_letter(answer, valid_letters)
                    if extracted is None:
                        print("Could not extract a valid answer letter.")
                    else:
                        print(f"Extracted Answer: {extracted}")
                    correct = (extracted == q["answer"].upper())
                    if correct:
                        print("Result: Correct!")
                        group_correct_count += 1
                    else:
                        print(f"Result: Incorrect. Expected: {q['answer'].upper()}")
                    individual_simplicity.append((correct, duration))
                    individual_results.append({
                        "question_id": q["id"],
                        "duration": duration,
                        "llm_answer": extracted,
                        "correct_answer": q["answer"].upper(),
                        "is_correct": correct,
                        "model": model_to_use
                    })
                total_individual_count += 1
            total_individual_correct += group_correct_count

            print(f"\nGroup {groups_processed+1}: {group_correct_count} out of {len(local_labels)} individual questions answered correctly.")
            total_duration_individual = sum(x[1] for x in individual_simplicity)
            print(f"\nTotal Duration for Individual Questions: {total_duration_individual:.2f} seconds")

            if Tokens < total_duration_individual:
                print("Timing Check: Success: The combined prompt was answered faster than the sum of individual questions.")
            else:
                print("Timing Check: Failure: The combined prompt was NOT answered faster than the sum of individual questions.")

            llm_ranking = extract_ranking(combined_answer)
            if llm_ranking is None or not isinstance(llm_ranking, list) or len(llm_ranking) != 4:
                print("No valid JSON ranking could be extracted from the combined answer.")
                llm_ranking = []
            else:
                print("\nExtracted LLM Ranking (order from fastest to slowest):")
                print(llm_ranking)

            measured_order = sorted(zip(local_labels, individual_simplicity), key=lambda x: (not x[1][0], x[1][1]))
            measured_ranking = [label for label, _ in measured_order]
            print("\nMeasured Ranking based on Individual Durations (fastest first):")
            print(measured_ranking)

            if llm_ranking:
                score = score_ranking(llm_ranking, measured_ranking)
                print(f"\nScore for this group (weighted ranking): {score:.2f} / 1")
                overall_scores.append(score)
                question_sets.append({
                    "group_id": group_id,
                    "llm_ranking": ",".join(llm_ranking),
                    "combined_duration": Tokens,
                    "total_individual_duration": total_duration_individual,
                    "measured_ranking": ",".join(measured_ranking),
                    "score": score,
                    "model": model_to_use
                })
            else:
                print("No score computed due to missing or invalid ranking output.")
                question_sets.append({
                    "group_id": group_id,
                    "llm_ranking": "",
                    "combined_duration": Tokens,
                    "total_individual_duration": total_duration_individual,
                    "measured_ranking": ",".join(measured_ranking),
                    "score": None,
                    "model": model_to_use
                })

            new_processed_ids.extend([f"{qid}_{model_to_use}" for qid in original_ids])
            groups_processed += 1

            # Save CSV outputs after each group.
            pd.DataFrame(combined_ids).to_csv("combined_ids.csv", mode='a', index=False)
            pd.DataFrame(question_sets).to_csv("question_sets.csv", mode='a', index=False)
            pd.DataFrame(individual_results).to_csv("individual_results.csv", mode='a', index=False)
            print("CSV files updated.")

        if groups_processed >= max_groups_to_process:
            break

    if overall_scores:
        avg_score = sum(overall_scores) / len(overall_scores)
        print("\n===== Overall Results =====")
        print(f"Processed {groups_processed} groups with an average ranking score of {avg_score:.3f} / 1")
    else:
        print("No valid ranking scores computed.")

    print(f"\nOverall, the LLM answered {total_individual_correct} out of {total_individual_count} individual questions correctly.")

    # Update the checkpoint file with processed question-model keys.
    df_checkpoint = pd.DataFrame({"question_model": list(new_processed_ids)})
    if os.path.exists("processed_ids.csv"):
        df_existing = pd.read_csv("processed_ids.csv")
        df_checkpoint = pd.concat([df_existing, df_checkpoint]).drop_duplicates()
    df_checkpoint.to_csv("processed_ids.csv", index=False)
    print("Checkpoint file updated: processed_ids.csv")


if __name__ == "__main__":
    main()