import time
import json
import openai
import os
import re
import csv
import requests

# Load OpenAI API key
key_file = "api_keys.txt"
if os.path.exists(key_file):
    with open(key_file, "r") as f:
        keys = f.read().strip().splitlines()
        if len(keys) >= 2:
            openai.api_key = keys[0]  # First line for OpenAI
            openrouter_key = keys[1]  # Second line for OpenRouter
        else:
            raise ValueError("API key file must contain at least two lines: one for OpenAI and one for OpenRouter.")
else:
    raise FileNotFoundError(f"API key file '{key_file}' not found. Please create it and add your keys in the format 'openai=YOUR_OPENAI_KEY' and 'openrouter=YOUR_OPENROUTER_KEY'.")

# Model selection options
openai_models = ["gpt-4o", "gpt-4o-mini"]
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

index = 0

# Prepare output file
output_file = openrouter_models[index].replace("/", "-") + ".csv"

# Select model type
use_openai = True  # Set to True to use OpenAI, False to use OpenRouter
model_to_use = openai_models[index] if use_openai else openrouter_models[index]

def ask_openai(prompt, model=model_to_use, temperature=0.0, max_retries=3):
    """
    Sends a prompt to the LLM using OpenAI API and returns its response.
    """
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error during OpenAI LLM call: {e}. Retrying {attempt + 1}/{max_retries} ...")
            time.sleep(1)
    print("Failed to get a response after multiple attempts.")
    return None

def ask_openrouter(prompt, model=model_to_use, temperature=0.0):
    """
    Sends a prompt to the LLM using OpenRouter API and returns its response.
    """
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + openrouter_key,
                "X-Title": "Awareness",
            },
            data=json.dumps({
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during OpenRouter LLM call: {e}")
        return None
def get_credits():
    """Fetches the remaining credits from OpenRouter API."""
    try:
        response = requests.get(
            url="https://openrouter.ai/api/v1/credits",
            headers={
                "Authorization": "Bearer " + openrouter_key,
            }
        )
        response_data = response.json()
        return response_data.get("data", {})
    except Exception as e:
        print(f"Error fetching credits: {e}")
        return None

get_credits() 
exit()

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



# Load dataset (JSON file with words as keys and lists of associated words as values)
data_path = "3-5_taboo_words.json"
with open(data_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Limit to first 10 words
limited_dataset = list(dataset.items())[:10]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Word", "Predictions", "Generated Words", "Final Score (association detection accuracy)", "Count Words", "If Only False Predictions", "Predicted True Count", "Success Out of True", "Success Out of False"])
    
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
        predicted_response = ask_openai(prediction_prompt) if use_openai else ask_openrouter(prediction_prompt)
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
        actual_associations = ask_openai(association_prompt) if use_openai else ask_openrouter(association_prompt)
        actual_data = extract_json_from_text(actual_associations) # the list the machine generated

        if actual_data is None or "words" not in actual_data:
            print(f"Skipping {target_word} due to association extraction failure.")
            continue

        actual_associated_words = [word.lower() for word in actual_data["words"]]

        # Step 3: Compute Scores
        # כמה מהמילים שהמודל אמר עליהן 'אמת' (שהן אסורות), באמת הופיעו ברשימת המילים בפועל
        true_words_found = sum(1 for word in taboo_words_lower if predicted_data.get(word, False) and word in actual_associated_words)
        false_words_not_found = sum(1 for word in taboo_words_lower if not predicted_data.get(word, False) and word not in actual_associated_words)

        # final_score = f"{((true_words_found + false_words_not_found) / max(len(predicted_data), 1)) * 100:.2f} %"
        final_score = f"{((true_words_found + false_words_not_found) / max(len(taboo_words_lower), 1)) * 100:.2f} %"

        # Count predictions marked as True & False
        predicted_true = sum(1 for word, value in predicted_data.items() if value is True)
        predicted_false = sum(1 for word, value in predicted_data.items() if value is False)

        # Prevent division by zero
        success_out_of_true = f"{(true_words_found / predicted_true) * 100:.2f} %" if predicted_true > 0 else ""
        success_out_of_false = f"{(min(false_words_not_found, predicted_false) / predicted_false) * 100:.2f} %" if predicted_false > 0 else ""

        count_words = len(predicted_data)
        missed_predictions = f"{(sum(1 for word in predicted_data if word not in actual_associated_words) / max(len(predicted_data), 1)) * 100:.2f} %"

        # Step 4: Write to File
        writer.writerow([target_word, predicted_data, actual_associated_words, final_score, count_words, missed_predictions, predicted_true, success_out_of_true, success_out_of_false])

        print(f"Processed {target_word}")

print(f"Results saved to {output_file}")
