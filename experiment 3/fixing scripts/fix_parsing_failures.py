import os
import pandas as pd
import re
import json
from TimeTest_parallel import extract_ranking, score_ranking

def extract_ranking_flexible(raw_output):
    """
    Try multiple methods to extract ranking from raw LLM output
    Returns the ranking list if found, None otherwise
    """
    if not raw_output or pd.isna(raw_output):
        return None
    
    # Method 1: Try the original extract_ranking function first
    try:
        ranking = extract_ranking(raw_output)
        if ranking and len(ranking) == 4:
            return ranking
    except:
        pass
    
    # Method 2: Look for JSON-like patterns
    json_patterns = [
        r'"ranking"\s*:\s*\[(.*?)\]',
        r'"ranking"\s*:\s*\[(.*?)\]',
        r'ranking.*?\[(.*?)\]',
        r'\[(.*?)\]'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, raw_output, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                # Clean the match and split by comma
                items = [item.strip().strip('"\'') for item in match.split(',')]
                # Filter for Q1, Q2, Q3, Q4
                valid_items = [item for item in items if item in ['Q1', 'Q2', 'Q3', 'Q4']]
                if len(valid_items) == 4 and len(set(valid_items)) == 4:
                    return valid_items
            except:
                continue
    
    # Method 3: Look for ordered list patterns
    list_patterns = [
        r'1\.\s*(Q[1-4])',
        r'[Ff]irst:?\s*(Q[1-4])',
        r'[Ss]econd:?\s*(Q[1-4])',
        r'[Tt]hird:?\s*(Q[1-4])',
        r'[Ff]ourth:?\s*(Q[1-4])',
        r'[Ll]ast:?\s*(Q[1-4])'
    ]
    
    found_items = []
    for i, pattern in enumerate(list_patterns):
        matches = re.findall(pattern, raw_output, re.IGNORECASE)
        if matches:
            found_items.extend(matches)
    
    # Remove duplicates while preserving order
    unique_items = []
    seen = set()
    for item in found_items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    
    if len(unique_items) == 4 and all(item in ['Q1', 'Q2', 'Q3', 'Q4'] for item in unique_items):
        return unique_items
    
    # Method 4: Look for simple comma-separated Q1,Q2,Q3,Q4 patterns
    simple_patterns = [
        r'(Q[1-4],\s*Q[1-4],\s*Q[1-4],\s*Q[1-4])',
        r'(Q[1-4]\s+Q[1-4]\s+Q[1-4]\s+Q[1-4])',
        r'(Q[1-4]-Q[1-4]-Q[1-4]-Q[1-4])',
        r'(Q[1-4]\s*,\s*Q[1-4]\s*,\s*Q[1-4]\s*,\s*Q[1-4])'
    ]
    
    for pattern in simple_patterns:
        matches = re.findall(pattern, raw_output)
        for match in matches:
            try:
                # Split by various delimiters
                items = re.split(r'[,\s\-]+', match)
                items = [item.strip() for item in items if item.strip()]
                if len(items) == 4 and all(item in ['Q1', 'Q2', 'Q3', 'Q4'] for item in items):
                    return items
            except:
                continue
    
    # Method 5: Look for explicit ranking statements
    ranking_patterns = [
        r'[Rr]anking:?\s*(.*?)(?:\n|$)',
        r'[Oo]rder:?\s*(.*?)(?:\n|$)',
        r'[Ff]rom fastest to slowest:?\s*(.*?)(?:\n|$)',
        r'[Aa]nswer:?\s*(.*?)(?:\n|$)'
    ]
    
    for pattern in ranking_patterns:
        matches = re.findall(pattern, raw_output, re.IGNORECASE)
        for match in matches:
            # Try to extract Q1-Q4 from this line
            q_items = re.findall(r'Q[1-4]', match)
            if len(q_items) == 4 and len(set(q_items)) == 4:
                return q_items
    
    return None

def display_sample_failures(df, num_samples=5):
    """Display sample parsing failures to help understand the patterns"""
    # Find rows with empty ranking but valid raw output
    empty_ranking = df['llm_ranking'].isna() | (df['llm_ranking'] == '')
    valid_raw_output = df['raw_llm_ranking_output'].notna() & (df['raw_llm_ranking_output'] != '')
    
    parsing_failures = df[empty_ranking & valid_raw_output]
    
    if len(parsing_failures) == 0:
        print("No parsing failures found!")
        return
    
    print(f"Found {len(parsing_failures)} parsing failures")
    print(f"\nShowing {min(num_samples, len(parsing_failures))} sample failures:\n")
    
    for i, (idx, row) in enumerate(parsing_failures.head(num_samples).iterrows()):
        print(f"=== Sample {i+1}: Group {row['group_id']} ===")
        raw_output = str(row['raw_llm_ranking_output'])
        print(f"Raw output length: {len(raw_output)}")
        print(f"Raw output preview (first 300 chars):")
        print(raw_output[:300])
        print(f"Raw output end (last 100 chars):")
        print(raw_output[-100:])
        print()

def fix_parsing_failures(model_name, dry_run=True, show_samples=True):
    """
    Fix parsing failures for a specific model by re-extracting rankings from raw output
    
    Args:
        model_name: Name of the model (e.g., 'meta-llama/llama-3.2-3b-instruct')
        dry_run: If True, only show what would be fixed without making changes
        show_samples: If True, show sample failures before processing
    """
    # Normalize model name for file path
    model_dir_name = model_name.replace(":", "-").replace("/", "/")
    results_file = f"results/{model_dir_name}/question_sets.csv"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    print(f"Processing model: {model_name}")
    print(f"Results file: {results_file}")
    
    # Load the data
    df = pd.read_csv(results_file)
    print(f"Total rows: {len(df)}")
    
    # Find parsing failures
    empty_ranking = df['llm_ranking'].isna() | (df['llm_ranking'] == '')
    valid_raw_output = df['raw_llm_ranking_output'].notna() & (df['raw_llm_ranking_output'] != '')
    parsing_failures = df[empty_ranking & valid_raw_output]
    
    print(f"Groups with NaN scores: {df['score'].isna().sum()}")
    print(f"Parsing failures (empty ranking but has raw output): {len(parsing_failures)}")
    
    if show_samples and len(parsing_failures) > 0:
        display_sample_failures(df)
        
        # Ask user if they want to continue
        user_input = input("\nDo you want to continue with fixing these failures? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting without making changes.")
            return
    
    if len(parsing_failures) == 0:
        print("No parsing failures found to fix!")
        return
    
    # Process each failure
    fixed_count = 0
    for idx, row in parsing_failures.iterrows():
        group_id = row['group_id']
        raw_output = row['raw_llm_ranking_output']
        
        # Try to extract ranking
        extracted_ranking = extract_ranking_flexible(raw_output)
        
        if extracted_ranking:
            if dry_run:
                print(f"Would fix group {group_id}: {extracted_ranking}")
            else:
                # Update the dataframe
                df.at[idx, 'llm_ranking'] = ','.join(extracted_ranking)
                
                # Recalculate score if we have measured_ranking
                measured_ranking_str = row.get('measured_ranking')
                if measured_ranking_str and pd.notna(measured_ranking_str):
                    measured_ranking = measured_ranking_str.split(',')
                    if len(measured_ranking) == 4:
                        try:
                            score = score_ranking(extracted_ranking, measured_ranking)
                            df.at[idx, 'score'] = score
                            print(f"Fixed group {group_id}: {extracted_ranking} (Score: {score:.3f})")
                        except:
                            print(f"Fixed group {group_id}: {extracted_ranking} (Score calculation failed)")
                    else:
                        print(f"Fixed group {group_id}: {extracted_ranking} (No measured ranking)")
                else:
                    print(f"Fixed group {group_id}: {extracted_ranking} (No measured ranking)")
            
            fixed_count += 1
        else:
            if dry_run:
                print(f"Could not extract ranking for group {group_id}")
    
    print(f"\n=== Summary ===")
    print(f"Total parsing failures: {len(parsing_failures)}")
    print(f"Successfully extracted rankings: {fixed_count}")
    print(f"Still failed: {len(parsing_failures) - fixed_count}")
    
    if not dry_run and fixed_count > 0:
        # Save the fixed data
        backup_file = results_file.replace('.csv', '_before_parsing_fix.csv')
        
        # Create backup
        original_df = pd.read_csv(results_file)
        original_df.to_csv(backup_file, index=False)
        print(f"Backup saved to: {backup_file}")
        
        # Save fixed data
        df.to_csv(results_file, index=False)
        print(f"Fixed data saved to: {results_file}")
        
        # Show final statistics
        final_nan_count = df['score'].isna().sum()
        print(f"NaN scores after fix: {final_nan_count}")
    elif dry_run:
        print("\nThis was a dry run. Use dry_run=False to actually make the changes.")

if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "openai/o4-mini-2025-04-16"  # Change this to the model you want to fix
    
    print("=== Parsing Failure Fix Tool ===")
    print(f"Target model: {MODEL_NAME}")
    print()
    
    # First, run in dry-run mode to see what would be fixed
    print("Running in DRY RUN mode first...")
    fix_parsing_failures(MODEL_NAME, dry_run=True, show_samples=True)
    
    print("\n" + "="*50)
    user_choice = input("Do you want to apply the fixes? (y/n): ").strip().lower()
    if user_choice == 'y':
        print("Applying fixes...")
        fix_parsing_failures(MODEL_NAME, dry_run=False, show_samples=False)
    else:
        print("No changes made.")