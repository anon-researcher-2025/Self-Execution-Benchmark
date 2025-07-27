#!/usr/bin/env python3
"""
Script to create a clean question_sets.csv that looks like everything was processed at once.
This script merges all reprocessed results into a clean final question_sets.csv.
"""

import pandas as pd
import os
from datetime import datetime

def create_final_question_sets():
    """Create a final clean question_sets.csv with all results."""
    
    model_name = "o4-mini-2025-04-16"
    model_dir = f"results/openai/{model_name}"
    original_csv = f"{model_dir}/question_sets.csv"
    reprocessed_dir = "reprocess_failures_detailed"
    
    print(f"=== CREATING FINAL QUESTION SETS CSV ===")
    
    # Load original data
    if not os.path.exists(original_csv):
        print(f"Original CSV not found: {original_csv}")
        return
    
    print(f"Loading original data from: {original_csv}")
    df = pd.read_csv(original_csv)
    
    # Find all reprocessed files for this model
    reprocessed_files = []
    if os.path.exists(reprocessed_dir):
        for file in os.listdir(reprocessed_dir):
            if file.startswith(f"{model_name}_detailed_reprocess_") and file.endswith(".csv"):
                reprocessed_files.append(os.path.join(reprocessed_dir, file))
    
    if not reprocessed_files:
        print(f"No reprocessed files found for {model_name}")
        return
    
    print(f"Found {len(reprocessed_files)} reprocessed files:")
    for file in reprocessed_files:
        print(f"  - {file}")
    
    # Collect all successful reprocessed results
    all_reprocessed_results = []
    for reprocessed_file in reprocessed_files:
        print(f"\nProcessing: {reprocessed_file}")
        
        reprocessed_df = pd.read_csv(reprocessed_file)
        successful_reprocessed = reprocessed_df[reprocessed_df['reprocess_successful'] == True]
        
        print(f"  Found {len(successful_reprocessed)} successful results")
        all_reprocessed_results.append(successful_reprocessed)
    
    # Combine all reprocessed results
    if all_reprocessed_results:
        combined_reprocessed = pd.concat(all_reprocessed_results, ignore_index=True)
        
        # Remove duplicates (keep the latest by timestamp)
        combined_reprocessed = combined_reprocessed.sort_values('timestamp').drop_duplicates(
            subset=['group_id'], keep='last'
        )
        
        print(f"\nTotal unique successfully reprocessed groups: {len(combined_reprocessed)}")
    else:
        print("No successful reprocessed results found")
        return
    
    # Create the final dataset
    final_df = df.copy()
    
    # Update rows with reprocessed data
    updates_made = 0
    for _, reprocessed_row in combined_reprocessed.iterrows():
        group_id = reprocessed_row['group_id']
        
        # Find the matching row in original data
        mask = final_df['group_id'] == group_id
        matching_rows = final_df[mask]
        
        if len(matching_rows) == 1:
            # Update with reprocessed data
            final_df.loc[mask, 'llm_ranking'] = reprocessed_row['extracted_ranking']
            final_df.loc[mask, 'raw_llm_ranking_output'] = reprocessed_row['raw_response']
            final_df.loc[mask, 'combined_duration'] = reprocessed_row['tokens_returned']
            
            # Update measured_ranking and score if available
            if pd.notna(reprocessed_row.get('measured_ranking')):
                final_df.loc[mask, 'measured_ranking'] = reprocessed_row['measured_ranking']
            
            if pd.notna(reprocessed_row.get('score')):
                final_df.loc[mask, 'score'] = reprocessed_row['score']
            
            # Calculate total_individual_duration from individual_results if available
            individual_csv = f"{model_dir}/individual_results.csv"
            if os.path.exists(individual_csv):
                individual_df = pd.read_csv(individual_csv)
                group_individual = individual_df[individual_df['group_id'] == group_id]
                if len(group_individual) > 0:
                    total_duration = group_individual['duration'].sum()
                    final_df.loc[mask, 'total_individual_duration'] = total_duration
            
            updates_made += 1
            print(f"  Updated group {group_id}")
    
    # Keep only the essential columns in the specified order
    essential_columns = [
        'group_id', 'llm_ranking', 'raw_llm_ranking_output', 'combined_duration',
        'total_individual_duration', 'measured_ranking', 'score', 'model'
    ]
    
    # Make sure all columns exist
    for col in essential_columns:
        if col not in final_df.columns:
            final_df[col] = None
    
    # Select only essential columns
    clean_df = final_df[essential_columns].copy()
    
    # Save the clean final version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_file = f"{model_dir}/question_sets_final_{timestamp}.csv"
    clean_df.to_csv(final_output_file, index=False)
    
    print(f"\n=== FINAL RESULTS SUMMARY ===")
    print(f"Total groups: {len(clean_df)}")
    print(f"Updated groups: {updates_made}")
    print(f"Final file saved to: {final_output_file}")
    
    # Statistics
    failure_message = "Combined prompt LLM call failed or returned no answer"
    remaining_failures = clean_df[clean_df['raw_llm_ranking_output'] == failure_message]
    valid_rankings = clean_df[clean_df['llm_ranking'].notna() & (clean_df['llm_ranking'] != '')]
    scored_groups = clean_df[clean_df['score'].notna()]
    
    print(f"\nStatistics:")
    print(f"  Valid rankings: {len(valid_rankings)}")
    print(f"  Remaining failures: {len(remaining_failures)}")
    print(f"  Groups with scores: {len(scored_groups)}")
    
    if len(scored_groups) > 0:
        avg_score = scored_groups['score'].mean()
        print(f"  Average score: {avg_score:.3f}")
    
    # Show sample of final data
    print(f"\nSample of final data:")
    for i, row in clean_df.head(3).iterrows():
        print(f"  {row['group_id']}: {row['llm_ranking']} -> {row['measured_ranking']} (score: {row['score']})")
    
    return final_output_file

if __name__ == "__main__":
    final_file = create_final_question_sets()
    print(f"\nFinal clean question_sets.csv created: {final_file}")
