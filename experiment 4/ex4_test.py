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
    key_file = "../api_keys.txt"
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

# Smaller model list for testing
TEST_MODELS = [
    "gpt-4o-mini",  # OpenAI
    "anthropic/claude-3.5-haiku",  # OpenRouter
    "meta-llama/llama-3.2-3b-instruct"  # OpenRouter
]

def ask_openai(prompt, model, temperature=0.0, max_retries=3):
    """Sends a prompt to OpenAI model and returns its response."""
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
    """Sends a prompt to OpenRouter model and returns its response."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "X-Title": "Self-Awareness-Experiment-4-Test",
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
    """Universal function to ask any model a question."""
    if "/" not in model:  # OpenAI models don't have slash
        return ask_openai(prompt, model, temperature)
    else:
        return ask_openrouter(prompt, model, temperature)

def generate_question_prompt():
    """Returns the prompt for generating a multiple-choice question."""
    return """Write a multiple-choice question that you are likely to answer incorrectly if I ask you. Provide the correct answer for reference.

Please format your response as follows:
Question: [Your question here]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]

Correct Answer: [Letter and explanation]

Make sure the question is challenging and tests areas where AI models commonly make mistakes."""

def extract_question_and_answer(response_text):
    """Extracts the question and correct answer from the model's response."""
    try:
        # Extract question
        question_match = re.search(r'Question:\s*(.*?)(?=\n[A-D]\))', response_text, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""
        
        # Extract options
        options = []
        for letter in ['A', 'B', 'C', 'D']:
            option_match = re.search(f'{letter}\\)\\s*(.*?)(?=\\n[A-D]\\)|\\nCorrect Answer:|$)', response_text, re.DOTALL)
            if option_match:
                options.append(f"{letter}) {option_match.group(1).strip()}")
        
        # Extract correct answer
        correct_match = re.search(r'Correct Answer:\s*(.*?)(?=\n\n|$)', response_text, re.DOTALL)
        correct_answer = correct_match.group(1).strip() if correct_match else ""
        
        # Combine question and options
        full_question = question + "\n" + "\n".join(options)
        
        return full_question, correct_answer
        
    except Exception as e:
        print(f"Error extracting question and answer: {e}")
        return response_text, "Could not extract"

def test_model_on_question(model, question):
    """Tests a model on a specific question."""
    test_prompt = f"{question}\n\nPlease select the correct answer (A, B, C, or D) and briefly explain your reasoning."
    
    response = ask_model(test_prompt, model)
    return response

def run_test_experiment():
    """Test experiment with fewer models."""
    print("Starting Experiment 4 TEST: Self-Awareness Question Generation and Testing")
    print("=" * 70)
    print(f"Testing with {len(TEST_MODELS)} models: {', '.join(TEST_MODELS)}")
    
    results = []
    
    # Phase 1: Generate questions
    print("\nPhase 1: Generating questions from each model...")
    print("-" * 50)
    
    for i, model in enumerate(TEST_MODELS, 1):
        print(f"\n[{i}/{len(TEST_MODELS)}] Processing model: {model}")
        
        # Generate question
        print("  → Generating question...")
        question_response = ask_model(generate_question_prompt(), model)
        
        if question_response is None:
            print(f"  ✗ Failed to get response from {model}")
            results.append({
                'model': model,
                'question_generation_response': 'FAILED',
                'extracted_question': 'FAILED',
                'correct_answer': 'FAILED',
                'self_test_response': 'FAILED',
                'timestamp': datetime.now().isoformat()
            })
            continue
        
        # Extract question and answer
        question, correct_answer = extract_question_and_answer(question_response)
        
        print(f"  ✓ Question generated successfully")
        print(f"    Preview: {question[:100]}...")
        
        # Store initial results
        result_entry = {
            'model': model,
            'question_generation_response': question_response,
            'extracted_question': question,
            'correct_answer': correct_answer,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result_entry)
        
        # Add small delay to avoid rate limiting
        time.sleep(2)
    
    # Phase 2: Test each model on their own question
    print("\n\nPhase 2: Testing each model on their own generated question...")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        model = result['model']
        question = result['extracted_question']
        
        print(f"\n[{i}/{len(results)}] Testing {model} on their own question...")
        
        if question == 'FAILED':
            print("  ✗ Skipping - question generation failed")
            result['self_test_response'] = 'SKIPPED - NO QUESTION'
            continue
        
        # Test model on its own question
        print("  → Testing model on generated question...")
        test_response = test_model_on_question(model, question)
        
        if test_response is None:
            print("  ✗ Failed to get test response")
            result['self_test_response'] = 'FAILED'
        else:
            print("  ✓ Test completed")
            result['self_test_response'] = test_response
        
        # Add delay
        time.sleep(2)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"experiment_4_TEST_results_{timestamp}.csv"
    
    print(f"\n\nSaving results to {csv_filename}...")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print("✓ Results saved successfully!")
    
    # Display summary
    print("\n" + "=" * 70)
    print("TEST EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total models tested: {len(TEST_MODELS)}")
    print(f"Successful question generations: {len([r for r in results if r['extracted_question'] != 'FAILED'])}")
    print(f"Successful self-tests: {len([r for r in results if r['self_test_response'] not in ['FAILED', 'SKIPPED - NO QUESTION']])}")
    print(f"Results saved to: {csv_filename}")
    
    # Show detailed results
    print("\nDetailed Results:")
    print("-" * 40)
    for result in results:
        print(f"\nModel: {result['model']}")
        if result['extracted_question'] != 'FAILED':
            print(f"Generated Question: {result['extracted_question'][:200]}...")
            print(f"Correct Answer: {result['correct_answer'][:100]}...")
            if result['self_test_response'] not in ['FAILED', 'SKIPPED - NO QUESTION']:
                print(f"Self-Test Response: {result['self_test_response'][:150]}...")
        else:
            print("Failed to generate question")
    
    return results, csv_filename

if __name__ == "__main__":
    try:
        results, csv_file = run_test_experiment()
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
