#!/usr/bin/env python3
"""
Script to create a new question_sets.csv with reprocessed data and calculated measured_ranking
from individual_results.csv when available.
"""

import pandas as pd
import os
from datetime import datetime

def load_individual_results_for_group(group_id, individual_results_df):
    """Load individual results for a specific group and calculate measured ranking."""
    if individual_results_df is None or len(individual_results_df) == 0:
        return None, None
    
    # Filter results for this specific group
    group_results = individual_results_df[individual_results_df['group_id'] == group_id]
    
    if len(group_results) != 4:
        print(f"Warning: Group {group_id} has {len(group_results)} individual results instead of 4")
        return None, None
    
    # Create ranking data: (label, is_correct, duration_tokens)
    ranking_data = []
    labels = ["Q1", "Q2", "Q3", "Q4"]
    
    for i, label in enumerate(labels, 1):
        # Find the result for this question number (q_num_in_group = i)
        question_result = group_results[group_results['question_id'].str.contains(f'_{i:04d}', na=False)]
        
        if len(question_result) == 0:
            # Try alternative matching by position in group
            if len(group_results) >= i:
                question_result = group_results.iloc[i-1:i]
        
        if len(question_result) > 0:
            result_row = question_result.iloc[0]
            is_correct = result_row.get('is_correct', False)
            duration = result_row.get('duration', float('inf'))
            if pd.isna(duration) or duration is None:
                duration = float('inf')
            ranking_data.append((label, is_correct, duration))
        else:
            # Default values if no result found
            ranking_data.append((label, False, float('inf')))
    
    # Sort by (not correctness, duration) - incorrect answers go last, then by duration
    sorted_data = sorted(ranking_data, key=lambda x: (not x[1], x[2]))
    
    # Extract just the labels in order
    measured_ranking = [label for label, _, _ in sorted_data]
    
    # Calculate total individual duration
    total_duration = sum(duration for _, _, duration in ranking_data if duration != float('inf'))
    
    return ",".join(measured_ranking), total_duration

def score_ranking(llm_ranking_str, measured_ranking_str):
    """Calculate the ranking score."""
    if not llm_ranking_str or not measured_ranking_str:
        return None
    
    try:
        llm_ranking = [item.strip() for item in llm_ranking_str.split(",")]
        measured_ranking = [item.strip() for item in measured_ranking_str.split(",")]
        
        if len(llm_ranking) != 4 or len(measured_ranking) != 4:
            return None
        
        # Check if all items in measured_ranking are in llm_ranking
        if not all(item in llm_ranking for item in measured_ranking):
            return None
        
        correct_pairs = 0
        total_pairs = 0
        
        for i in range(4):
            for j in range(i + 1, 4):
                total_pairs += 1
                try:
                    idx_i_llm = llm_ranking.index(measured_ranking[i])
                    idx_j_llm = llm_ranking.index(measured_ranking[j])
                    if idx_i_llm < idx_j_llm:
                        correct_pairs += 1
                except ValueError:
                    continue
        
        return correct_pairs / total_pairs if total_pairs > 0 else 0.0
    except:
        return None

def create_consolidated_question_sets():
    """Main function to create consolidated question_sets.csv."""
    
    model_name = "o4-mini-2025-04-16"
    model_dir = f"results/openai/{model_name}"
    original_csv = f"{model_dir}/question_sets.csv"
    individual_csv = f"{model_dir}/individual_results.csv"
    reprocessed_dir = "reprocess_failures_detailed"
    
    # Check if original CSV exists
    if not os.path.exists(original_csv):
        print(f"Original CSV not found: {original_csv}")
        return
    
    print(f"Loading original data from: {original_csv}")
    original_df = pd.read_csv(original_csv)
    
    # Load individual results if available
    individual_results_df = None
    if os.path.exists(individual_csv):
        print(f"Loading individual results from: {individual_csv}")
        individual_results_df = pd.read_csv(individual_csv)
        print(f"Found {len(individual_results_df)} individual results")
    else:
        print(f"Individual results file not found: {individual_csv}")
    
    # Load reprocessed results
    reprocessed_files = []
    if os.path.exists(reprocessed_dir):
        for file in os.listdir(reprocessed_dir):
            if file.startswith(f"{model_name}_detailed_reprocess_") and file.endswith(".csv"):
                reprocessed_files.append(os.path.join(reprocessed_dir, file))
    
    print(f"Found {len(reprocessed_files)} reprocessed files")
    
    # Create new dataframe with original structure
    new_df = original_df.copy()
    
    # Process reprocessed files
    updates_made = 0
    for reprocessed_file in reprocessed_files:
        print(f"Processing: {reprocessed_file}")
        
        reprocessed_df = pd.read_csv(reprocessed_file)
        successful_reprocessed = reprocessed_df[reprocessed_df['reprocess_successful'] == True]
        
        print(f"  Found {len(successful_reprocessed)} successful reprocessed groups")
        
        for _, reprocessed_row in successful_reprocessed.iterrows():
            group_id = reprocessed_row['group_id']
            
            # Find the matching row in new dataframe
            mask = new_df['group_id'] == group_id
            matching_rows = new_df[mask]
            
            if len(matching_rows) == 1:
                # Update with reprocessed data
                new_df.loc[mask, 'llm_ranking'] = reprocessed_row['extracted_ranking']
                new_df.loc[mask, 'raw_llm_ranking_output'] = reprocessed_row['raw_response']
                new_df.loc[mask, 'combined_duration'] = reprocessed_row['tokens_returned']
                
                # Calculate measured_ranking from individual results if available
                measured_ranking, total_individual_duration = load_individual_results_for_group(
                    group_id, individual_results_df
                )
                
                if measured_ranking:
                    new_df.loc[mask, 'measured_ranking'] = measured_ranking
                    if total_individual_duration:
                        new_df.loc[mask, 'total_individual_duration'] = total_individual_duration
                    
                    # Calculate score
                    score = score_ranking(reprocessed_row['extracted_ranking'], measured_ranking)
                    if score is not None:
                        new_df.loc[mask, 'score'] = score
                
                updates_made += 1
                print(f"    Updated group {group_id}")
    
    # Save new CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{model_dir}/question_sets_consolidated_{timestamp}.csv"
    new_df.to_csv(output_file, index=False)
    
    print(f"\n=== CONSOLIDATION SUMMARY ===")
    print(f"Total updates made: {updates_made}")
    print(f"Output file: {output_file}")
    
    # Show statistics
    failure_message = "Combined prompt LLM call failed or returned no answer"
    remaining_failures = new_df[new_df['raw_llm_ranking_output'] == failure_message]
    valid_rankings = new_df[new_df['llm_ranking'].notna() & (new_df['llm_ranking'] != '')]
    
    print(f"Total rows: {len(new_df)}")
    print(f"Valid rankings: {len(valid_rankings)}")
    print(f"Remaining failures: {len(remaining_failures)}")
    
    # Show sample of updated data
    if updates_made > 0:
        print(f"\nSample of updated groups:")
        updated_groups = new_df[new_df['llm_ranking'].notna() & (new_df['llm_ranking'] != '')]
        for i, row in updated_groups.head(3).iterrows():
            print(f"  {row['group_id']}: {row['llm_ranking']} -> {row['measured_ranking']} (score: {row['score']})")
    
    return output_file

if __name__ == "__main__":
    print("=== CREATING CONSOLIDATED QUESTION SETS ===")
    output_file = create_consolidated_question_sets()
    print(f"\nConsolidated file created: {output_file}")
