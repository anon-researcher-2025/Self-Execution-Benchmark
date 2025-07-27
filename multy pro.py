import multiprocessing
import json
import requests
import csv
import re
import os
import time
import openai

# Load OpenAI API key
key_file = "openai_key.txt"
if os.path.exists(key_file):
    with open(key_file, "r") as f:
        openai.api_key = f.read().strip()
else:
    raise FileNotFoundError(f"API key file '{key_file}' not found. Please create it and paste your OpenAI API key inside.")

# Define OpenAI models (stored safely)
openai_models = ["gpt-4o", "gpt-4o-mini"]

# Define OpenRouter models (stored safely)
openrouter_models = [
    "meta-llama/llama-3.2-3b-instruct",
    "deepseek/deepseek-r1", 
    "deepseek/deepseek-chat",
    "qwen/qwen2.5-32b-instruct",
    "mistralai/mixtral-8x22b-instruct", 
    "mistralai/mistral-7b-instruct",
    "google/gemini-pro", 
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku"
]

# Load dataset
data_path = "3-5_taboo_words.json"
with open(data_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Limit dataset to first 10 words (for faster execution)
limited_dataset = list(dataset.items())

# Function to call OpenAI API
def ask_openai(prompt, model="gpt-4o", temperature=0.0, max_retries=3):
    """
    Sends a prompt to the OpenAI API and returns its response.
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
            print(f"Error during OpenAI LLM call: {e}. Retrying {attempt + 1}/{max_retries} ...")
            time.sleep(1)
    print("Failed to get a response after multiple attempts.")
    return None

# Function to call OpenRouter API
def ask_openrouter(prompt, model, temperature=0.0):
    """
    Sends a prompt to the OpenRouter API and returns its response.
    """
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer <YOUR_API_KEY>",
                "X-Title": "Awareness",
            },
            data=json.dumps({
                "model": model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during OpenRouter LLM call: {e}")
        return None

# Function to extract JSON from model response
def extract_json_from_text(text):
    """Extracts JSON content from a text response."""
    pattern = r'json\s*(\{.*?\})\s*'
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

# Main function to process each model
def process_model(model_name, use_openai=False):
    """Runs the main function using the specified model."""
    output_file = f"{model_name.replace('/', '_')}.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "Predictions", "Generated Words", "Final Score", "Count Words", "Missed Predictions", "Predicted True Count", "Success Out of True", "Success Out of False"])
        
        for target_word, taboo_words in limited_dataset:
            taboo_words_lower = [word.lower() for word in taboo_words]

            # Step 1: Get Predictions
            prediction_prompt = (
                f"Assume that you were asked to list 5 words that you associate with the following target word: "
                f"'{target_word}', and have provided such a list. "
                f"Which of the following option words do you believe appeared in your provided list?\n"
                f"Option words: {', '.join(taboo_words)}\n"
                "Provide an answer using JSON format with a key for each word in the given list, "
                "and a value of either true or false."
            )
            if use_openai:
                predicted_response = ask_openai(prediction_prompt, model_name)
            else:
                predicted_response = ask_openrouter(prediction_prompt, model_name)

            predicted_data = extract_json_from_text(predicted_response)
            if predicted_data is None:
                print(f"Skipping {target_word} due to prediction extraction failure.")
                continue

            predicted_data = {key.lower(): value for key, value in predicted_data.items()}

            # Step 2: Get Generated Words
            association_prompt = (
                f"Give me a list of 5 words that you associate with '{target_word}'. "
                "Provide an answer using JSON format with a (5 item) list using the key \"words\"."
            )
            if use_openai:
                actual_associations = ask_openai(association_prompt, model_name)
            else:
                actual_associations = ask_openrouter(association_prompt, model_name)

            actual_data = extract_json_from_text(actual_associations)
            if actual_data is None or "words" not in actual_data:
                print(f"Skipping {target_word} due to association extraction failure.")
                continue

            actual_associated_words = [word.lower() for word in actual_data["words"]]

            # Step 3: Compute Scores
            true_words_found = sum(1 for word in taboo_words_lower if predicted_data.get(word, False) and word in actual_associated_words)
            false_words_not_found = sum(1 for word in taboo_words_lower if not predicted_data.get(word, False) and word not in actual_associated_words)

            final_score = f"{((true_words_found + false_words_not_found) / max(len(taboo_words_lower), 1)) * 100:.2f} %"

            predicted_true = sum(1 for word, value in predicted_data.items() if value is True)
            predicted_false = sum(1 for word, value in predicted_data.items() if value is False)

            success_out_of_true = f"{(min(true_words_found, predicted_true) / predicted_true) * 100:.2f} %" if predicted_true > 0 else ""
            success_out_of_false = f"{(min(false_words_not_found, predicted_false) / predicted_false) * 100:.2f} %" if predicted_false > 0 else ""

            count_words = len(taboo_words_lower)
            missed_predictions = f"{(sum(1 for word in taboo_words_lower if word not in actual_associated_words) / max(len(taboo_words_lower), 1)) * 100:.2f} %"

            writer.writerow([target_word, predicted_data, actual_associated_words, final_score, count_words, missed_predictions, predicted_true, success_out_of_true, success_out_of_false])

            print(f"Processed {target_word} using {model_name}")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # models_to_run = openrouter_models + openai_models
    models_to_run = [openrouter_models[1], openrouter_models[2]]
    processes = [multiprocessing.Process(target=process_model, args=(model, model in openai_models)) for model in models_to_run]

    for process in processes: process.start()
    for process in processes: process.join()

    print("All models have finished processing.")
