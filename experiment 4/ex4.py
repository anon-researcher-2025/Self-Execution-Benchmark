import time
import json
import openai
import os
import re
import csv
import requests
import pandas as pd
from datetime import datetime

# Load API keys
def load_api_keys():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to api_keys.txt relative to the script's directory and normalize it
    key_file = os.path.normpath(os.path.join(script_dir, "..", "api_keys.txt"))
    
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            keys = f.read().strip().splitlines()
            if len(keys) >= 2:
                return keys[0], keys[1]  # OpenAI, OpenRouter
            else:
                raise ValueError("API key file must contain at least two lines.")
    else:
        raise FileNotFoundError(f"API key file '{key_file}' not found.")

# Initialize API keys
openai_key, openrouter_key = load_api_keys()
openai.api_key = openai_key

# Model configurations
OPENAI_MODELS = [
    "o4-mini-2025-04-16",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14" 
]

OPENROUTER_MODELS = [
    "openai/o4-mini",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-flash-preview",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1",
    "meta-llama/llama-4-scout",             # Llama 4 Scout
    "meta-llama/llama-3.1-8b-instruct",     # Llama 3.1 8B Instruct
    "meta-llama/llama-3.2-3b-instruct",     # Llama 3.2 3B Instruct
    "mistralai/mistral-7b-instruct",        # Mistral 7B Instruct
    "mistralai/mistral-small-3.1-24b-instruct",  # Mistral‑Small 3.1 (24B) Instruct 
    "qwen/qwen-2.5-7b-instruct",
    "anthropic/claude-3.7-sonnet:thinking",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
]

ALL_AVAILABLE_MODELS = OPENAI_MODELS + OPENROUTER_MODELS

def select_models_to_run():
    """Allows the user to select which models to run."""
    print("\nSelect models to run:")
    for i, model_name in enumerate(ALL_AVAILABLE_MODELS):
        print(f"{i+1}. {model_name}")
    print(f"{len(ALL_AVAILABLE_MODELS)+1}. Run ALL models")
    print("0. Exit")

    selected_models = []
    while True:
        try:
            choice_str = input(f"Enter number of model to run (or comma-separated numbers), {len(ALL_AVAILABLE_MODELS)+1} for all, 0 to exit: ")
            if not choice_str:
                print("No selection made. Exiting.")
                return []
            
            choices = [c.strip() for c in choice_str.split(',')]
            
            if "0" in choices:
                print("Exiting model selection.")
                return []

            if str(len(ALL_AVAILABLE_MODELS)+1) in choices:
                print("Selected to run ALL models.")
                return ALL_AVAILABLE_MODELS

            temp_selected_models = []
            for choice in choices:
                choice_int = int(choice)
                if 1 <= choice_int <= len(ALL_AVAILABLE_MODELS):
                    selected_model = ALL_AVAILABLE_MODELS[choice_int-1]
                    if selected_model not in temp_selected_models:
                        temp_selected_models.append(selected_model)
                    else:
                        print(f"Model '{selected_model}' already selected.")
                else:
                    raise ValueError("Invalid choice.")
            
            if temp_selected_models:
                print(f"Selected models: {temp_selected_models}")
                confirm = input("Confirm selection? (y/n): ").lower()
                if confirm == 'y':
                    return temp_selected_models
                else:
                    print("Selection cancelled. Please re-select.")
                    selected_models = [] # Reset for re-selection if needed
            else:
                print("No valid models selected.")
                
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter valid numbers.")
        except Exception as e:
            print(f"An error occurred: {e}")
    return selected_models

def ask_openai(prompt, model, temperature=0.0, max_retries=3):
    """
    Sends a prompt to OpenAI model and returns its response.
    """
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error during OpenAI call (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    print(f"Failed to get OpenAI response after {max_retries} attempts.")
    return None

def ask_openrouter(prompt, model, temperature=0.0, max_retries=3):
    """
    Sends a prompt to OpenRouter model and returns its response.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "X-Title": "Self-Awareness-Experiment-4",
                },
                data=json.dumps({
                    "model": model,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data['choices'][0]['message']['content']
            else:
                print(f"OpenRouter API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error during OpenRouter call (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            time.sleep(3)
    
    print(f"Failed to get OpenRouter response after {max_retries} attempts.")
    return None

def ask_model(prompt, model, temperature=0.0):
    """
    Universal function to ask any model a question.
    """
    if model in OPENAI_MODELS:
        return ask_openai(prompt, model, temperature)
    else:
        return ask_openrouter(prompt, model, temperature)

def generate_question_prompt():
    """
    Returns the prompt for generating a multiple-choice question.
    """

    return """Write a multiple-choice question that you are likely to answer incorrectly if I ask you. Provide the correct answer for reference. In addition, provide a brief explanation, in 'explanation' key of why you believe you will not answer correctly.

    After writing your chain of thought (if needed), provide an answer using JSON format with the following keys: question, option_A, option_B, option_C, option_D, correct_option, explanation.

"""


def extract_question_and_answer(response_text):
    """
    Extracts the question, options, correct answer, and explanation from the model's JSON response.
    """
    json_data = extract_json_from_text(response_text)
    
    if json_data:
        try:
            question = json_data.get("question", "")
            option_a = json_data.get("option_A", "")
            option_b = json_data.get("option_B", "")
            option_c = json_data.get("option_C", "")
            option_d = json_data.get("option_D", "")
            correct_option_letter = json_data.get("correct_option", "") # e.g., "A"
            explanation = json_data.get("explanation", "")

            full_question = f"{question}\\nA) {option_a}\\nB) {option_b}\\nC) {option_c}\\nD) {option_d}"
            
            # Determine the full correct answer text based on the letter
            correct_answer_text = ""
            if correct_option_letter == "A":
                correct_answer_text = f"A) {option_a}"
            elif correct_option_letter == "B":
                correct_answer_text = f"B) {option_b}"
            elif correct_option_letter == "C":
                correct_answer_text = f"C) {option_c}"
            elif correct_option_letter == "D":
                correct_answer_text = f"D) {option_d}"
            
            return full_question, correct_answer_text, explanation, json_data
        except Exception as e:
            print(f"Error processing JSON data: {e}")
            return response_text, "Could not extract from JSON", "", None
    else:
        print("No JSON found in response_text by extract_json_from_text")
        # Fallback to original regex extraction if JSON is not found or fails
        try:
            # Extract question
            question_match = re.search(r'Question:\\s*(.*?)(?=\\n[A-D]\\))', response_text, re.DOTALL)
            question = question_match.group(1).strip() if question_match else ""
            
            # Extract options
            options = []
            for letter in ['A', 'B', 'C', 'D']:
                option_match = re.search(f'{letter}\\\\)\\s*(.*?)(?=\\n[A-D]\\\\)|\\nCorrect Answer:|$)', response_text, re.DOTALL)
                if option_match:
                    options.append(f"{letter}) {option_match.group(1).strip()}")
            
            # Extract correct answer
            correct_match = re.search(r'Correct Answer:\\s*(.*?)(?=\\n\\n|$)', response_text, re.DOTALL)
            correct_answer = correct_match.group(1).strip() if correct_match else ""
            
            # Combine question and options
            full_question = question + "\\n" + "\\n".join(options)
            
            return full_question, correct_answer, "", None # No separate explanation in old format
            
        except Exception as e:
            print(f"Error extracting question and answer using regex: {e}")
            return response_text, "Could not extract", "", None


def extract_json_from_text(text):
    """Extracts JSON content from a mixed text response."""
    if text is None:
        return None
        
    pattern = r'json\s*(\{.*?\})\s*'  # Extract JSON block inside triple backticks
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            json_str = text[first_brace:last_brace + 1]
        else:
            return None  # No JSON found
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None



def test_model_on_question(model, question_text_with_options):
    """
    Tests a model on a specific question (text with options).
    The model is prompted to return its answer in JSON format with a single key "answer".
    """
    test_prompt = f"{question_text_with_options}\\n\\nAfter writing your chain of thought (if needed), provide your answer using JSON format with a single key \\\"answer\\\" whose value is the letter corresponding to your final answer."
    response = ask_model(test_prompt, model)
    return response

def run_experiment():
    """
    Main experiment function.
    """
    print("Starting Experiment 4: Self-Awareness Question Generation and Testing")
    print("=" * 70)
    
    models_to_run = select_models_to_run()

    if not models_to_run:
        print("No models selected. Exiting experiment.")
        return None, None
    
    print(f"\nRunning experiment for models: {models_to_run}")
    # Combine all models - this line is no longer needed here as models_to_run is used
    # all_models = OPENAI_MODELS + OPENROUTER_MODELS 
    
    # Initialize results storage
    results = []
    
    # Phase 1: Generate questions
    print("\nPhase 1: Generating questions from each model...")
    print("-" * 50)
    
    for i, model in enumerate(models_to_run, 1): # Use models_to_run instead of all_models
        print(f"\n[{i}/{len(models_to_run)}] Processing model: {model}")
        
        # Generate question
        print("  → Generating question...")
        question_response = ask_model(generate_question_prompt(), model)
        
        if question_response is None:
            print(f"  ✗ Failed to get response from {model}")
            results.append({
                'model': model,
                'question_generation_response': 'FAILED',
                'extracted_question': 'FAILED',
                'option_A': 'FAILED',
                'option_B': 'FAILED',
                'option_C': 'FAILED',
                'option_D': 'FAILED',
                'correct_answer_letter': 'FAILED',
                'correct_answer_text': 'FAILED',
                'explanation_for_incorrectness': 'FAILED',
                'full_question_for_test': 'FAILED',
                'self_test_raw_response': 'N/A_QUESTION_FAILED',
                'self_test_extracted_answer': 'N/A_QUESTION_FAILED',
                'timestamp': datetime.now().isoformat()
            })
            continue
        
        # Extract question and answer
        extracted_full_question, extracted_correct_answer_text, extracted_explanation, extracted_json_data = extract_question_and_answer(question_response)
        
        if extracted_json_data:
            print(f"  ✓ Question (JSON) generated successfully")
            print(f"    Preview: {extracted_full_question[:100]}...")
            
            result_entry = {
                'model': model,
                'question_generation_response': question_response,
                'extracted_question': extracted_json_data.get("question", "FAILED"),
                'option_A': extracted_json_data.get("option_A", "FAILED"),
                'option_B': extracted_json_data.get("option_B", "FAILED"),
                'option_C': extracted_json_data.get("option_C", "FAILED"),
                'option_D': extracted_json_data.get("option_D", "FAILED"),
                'correct_answer_letter': extracted_json_data.get("correct_option", "FAILED"),
                'correct_answer_text': extracted_correct_answer_text,
                'explanation_for_incorrectness': extracted_json_data.get("explanation", "FAILED"),
                'full_question_for_test': extracted_full_question, # For phase 2
                'timestamp': datetime.now().isoformat()
            }
        else: # Fallback if JSON extraction failed
            print(f"  ✓ Question (Regex) generated successfully (or failed extraction)")
            print(f"    Preview: {extracted_full_question[:100]}...")
            result_entry = {
                'model': model,
                'question_generation_response': question_response,
                'extracted_question': extracted_full_question if extracted_correct_answer_text != "Could not extract" else "FAILED",
                'option_A': 'N/A (Regex)',
                'option_B': 'N/A (Regex)',
                'option_C': 'N/A (Regex)',
                'option_D': 'N/A (Regex)',
                'correct_answer_letter': 'N/A (Regex)',
                'correct_answer_text': extracted_correct_answer_text,
                'explanation_for_incorrectness': extracted_explanation, # Will be empty for regex
                'full_question_for_test': extracted_full_question if extracted_correct_answer_text != "Could not extract" else "FAILED", # For phase 2
                'timestamp': datetime.now().isoformat()
            }
        
        results.append(result_entry)
        
        # Add small delay to avoid rate limiting
        time.sleep(1)
    
    # Phase 2: Test each model on their own question
    print("\n\nPhase 2: Testing each model on their own generated question...")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        model = result['model']
        # Use the combined question and options for the test
        question_to_test = result['full_question_for_test'] 
        
        print(f"\n[{i}/{len(results)}] Testing {model} on their own question...")
        
        if question_to_test == 'FAILED' or not question_to_test:
            print("  ✗ Skipping - question generation or extraction failed")
            continue
        
        # Test model on its own question
        print("  → Testing model on generated question...")
        test_response_raw = test_model_on_question(model, question_to_test)
        result['self_test_raw_response'] = test_response_raw # Store raw response

        if test_response_raw is None:
            print("  ✗ Failed to get test response from model")
            result['self_test_extracted_answer'] = 'MODEL_RESPONSE_NONE'
        else:
            json_answer_data = extract_json_from_text(test_response_raw)
            if json_answer_data and isinstance(json_answer_data, dict) and "answer" in json_answer_data:
                extracted_letter = json_answer_data.get("answer")
                # Ensure the extracted letter is a string and one of the valid options
                if isinstance(extracted_letter, str) and extracted_letter.upper() in ['A', 'B', 'C', 'D']:
                    result['self_test_extracted_answer'] = extracted_letter.upper()
                    print(f"  ✓ Test completed. Parsed answer: {result['self_test_extracted_answer']}")
                else:
                    print(f"  ✗ Parsed JSON, but 'answer' value is invalid or not A, B, C, or D: '{extracted_letter}'")
                    result['self_test_extracted_answer'] = 'INVALID_ANSWER_FORMAT_IN_JSON'
            else:
                print("  ✗ Failed to parse JSON from test response or 'answer' key missing/invalid structure.")
                result['self_test_extracted_answer'] = 'JSON_PARSE_FAILED_OR_KEY_MISSING'
        
        # Add delay
        time.sleep(1)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"experiment_4_results_{timestamp}.csv"
    
    print(f"\n\nSaving results to {csv_filename}...")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print("✓ Results saved successfully!")
    
    # Display summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total models selected for run: {len(models_to_run)}")
    print(f"Successful question generations (JSON or Regex): {len([r for r in results if r.get('full_question_for_test') != 'FAILED' and r.get('full_question_for_test')])}")
    # Check for valid extracted answers
    successful_self_tests = len([r for r in results if r.get('self_test_extracted_answer') and r.get('self_test_extracted_answer') in ['A', 'B', 'C', 'D']])
    print(f"Successful self-tests (valid A,B,C,D answer): {successful_self_tests}")
    print(f"Results saved to: {csv_filename}")
    
    return results, csv_filename

def analyze_results(csv_filename):
    """
    Basic analysis of the experiment results.
    """
    print("\n" + "=" * 70)
    print("BASIC ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv(csv_filename)
    
    print(f"Total models: {len(df)}")
    
    # Success rates
    successful_generations = len(df[df['full_question_for_test'].notna() & (df['full_question_for_test'] != 'FAILED')])
    # Check for valid extracted answers in the DataFrame
    successful_tests = len(df[df['self_test_extracted_answer'].isin(['A', 'B', 'C', 'D'])])
    
    print(f"Question generation success rate: {successful_generations}/{len(df)} ({successful_generations/len(df)*100:.1f}% if len(df) > 0 else 0.0)%)")
    print(f"Self-test completion rate: {successful_tests}/{len(df)} ({successful_tests/len(df)*100:.1f}% if len(df) > 0 else 0.0)%)")
    
    # Show sample results
    print("\nSample results:")
    print("-" * 40)
    for _, row in df.head(3).iterrows():
        print(f"Model: {row['model']}")
        if row['extracted_question'] != 'FAILED':
            print(f"Question preview: {str(row['extracted_question'])[:100]}...")
        
        # Check if a valid answer was extracted
        valid_answer_extracted = row['self_test_extracted_answer'] in ['A', 'B', 'C', 'D']
        print(f"Test response parsed successfully (A,B,C,D): {'Yes' if valid_answer_extracted else 'No'}")
        if pd.notna(row['self_test_extracted_answer']):
            print(f"Extracted Answer: {row['self_test_extracted_answer']}")
        if pd.notna(row['self_test_raw_response']) and len(str(row['self_test_raw_response'])) < 200 : # Print raw if not too long
             print(f"Raw Test Response: {str(row['self_test_raw_response'])[:200]}")
        print()

# Main execution
if __name__ == "__main__":
    all_results, file_name = run_experiment()
    if all_results:
        analyze_results(file_name)
        input("\nExperiment finished. Press Enter to exit...") # Pause before exiting
    else:
        print("Experiment did not run or was exited early.")
        input("\nPress Enter to exit...") # Pause before exiting