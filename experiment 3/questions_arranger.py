import pandas as pd
import random
from datasets import load_dataset
import os

# Configuration
MMLU_TOPICS = [
    "abstract_algebra", "college_physics", "high_school_geography", 
    "miscellaneous", "moral_scenarios", "global_facts", "formal_logic", 
    "international_law", "business_ethics", "high_school_mathematics"
]
QUESTIONS_PER_TOPIC = 100
GROUP_SIZE = 4
OUTPUT_CSV_FILENAME = "arranged_question_sets.csv"
DATASET_NAME_IN_ID = "MMLU" # Used for question IDs and group IDs

def load_mmlu_questions_for_arranger(topics, num_questions_per_topic):
    """
    Loads and formats MMLU questions.
    """
    all_questions = []
    print("Starting to load MMLU questions...")
    for topic in topics:
        try:
            ds_topic = load_dataset("cais/mmlu", topic, split="test", trust_remote_code=True)
            print(f"Loaded {len(ds_topic)} examples for topic '{topic}'.")
        except Exception as e:
            print(f"Error loading topic '{topic}' from MMLU: {e}")
            continue

        processed_topic_questions = 0
        for i, item in enumerate(ds_topic):
            if processed_topic_questions >= num_questions_per_topic:
                break
            
            question_text = item["question"]
            # Create a unique ID for the question
            words = question_text.split()
            first_three_words = "".join(word.capitalize() for word in words[:3]) # Ensure this is robust
            if not first_three_words and len(words) > 0: # Handle very short questions
                first_three_words = words[0][:10].capitalize() 
            elif not first_three_words:
                 first_three_words = "Q" # Fallback if question is empty or very unusual

            # Ensure unique ID by including original index 'i' from dataset
            q_id = f"{DATASET_NAME_IN_ID}_{topic}_{first_three_words}_{i+1:04d}"

            # Format choices and answer
            formatted_choices = [f"{chr(65 + j)}. {choice}" for j, choice in enumerate(item["choices"])]
            correct_answer_letter = chr(65 + int(item["answer"]))

            all_questions.append({
                "original_question_id": q_id,
                "question": question_text,
                "choices_list": formatted_choices, # Keep as list for now, convert to string for CSV later
                "answer": correct_answer_letter,
                "topic": topic # Keep topic for potential future use or verification
            })
            processed_topic_questions += 1
        print(f"Processed {processed_topic_questions} questions for topic '{topic}'.")
            
    print(f"Total MMLU questions loaded and formatted: {len(all_questions)}")
    return all_questions

def main():
    print("Starting question arrangement process...")
    # Load questions
    benchmark_questions = load_mmlu_questions_for_arranger(MMLU_TOPICS, QUESTIONS_PER_TOPIC)

    if not benchmark_questions:
        print("No benchmark questions loaded. Exiting.")
        return

    # Shuffle questions
    random.shuffle(benchmark_questions)
    print(f"Shuffled {len(benchmark_questions)} questions.")

    # Group questions
    grouped_data_for_csv = []
    num_total_questions = len(benchmark_questions)
    num_groups = num_total_questions // GROUP_SIZE
    
    print(f"Creating {num_groups} groups of {GROUP_SIZE} questions.")

    for i in range(num_groups):
        group_id = f"{DATASET_NAME_IN_ID}_ARRANGED_GROUP_{i+1:03d}"
        group_questions = benchmark_questions[i*GROUP_SIZE : (i+1)*GROUP_SIZE]
        
        for q_idx_in_group, question_data in enumerate(group_questions):
            grouped_data_for_csv.append({
                "group_id": group_id,
                "q_num_in_group": q_idx_in_group + 1, # 1-indexed
                "original_question_id": question_data["original_question_id"],
                "question": question_data["question"],
                "choices_str": "|".join(question_data["choices_list"]), # Pipe-separated choices
                "answer": question_data["answer"]
            })

    if not grouped_data_for_csv:
        print("No groups were created. Exiting.")
        return

    # Create DataFrame and save to CSV
    df_arranged_questions = pd.DataFrame(grouped_data_for_csv)
    
    # Save to CSV in the project root
    # Assuming this script is run from the project root or the path is adjusted accordingly.
    # For this tool, the filePath is absolute.
    output_path = os.path.join(OUTPUT_CSV_FILENAME)
    
    try:
        df_arranged_questions.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully saved arranged questions to: {output_path}")
    except Exception as e:
        print(f"Error saving CSV to {output_path}: {e}")

if __name__ == "__main__":
    # Ensure the script can find 'datasets' and other necessary libraries.
    # May require specific environment setup if run outside a pre-configured one.
    main()
