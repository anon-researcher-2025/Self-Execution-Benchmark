#!/usr/bin/env python3
"""
Script to update the original CSV file with reprocessing results.
This script takes the detailed reprocessing results and adds them back to the original question_sets.csv
"""

import pandas as pd
import os
from datetime import datetime

def update_csv_from_reprocess_results(model_name, reprocess_csv_path):
    """Update original CSV with results from reprocessing."""
    
    # Model paths
    MODEL_PATHS = {
        "o4-mini-2025-04-16": "results/openai/o4-mini-2025-04-16/question_sets.csv",
        "claude-3.7-sonnet-thinking": "results/anthropic/claude-3.7-sonnet-thinking/question_sets.csv", 
        "claude-3.5-haiku": "results/anthropic/claude-3.5-haiku/question_sets.csv",
        "deepseek-r1": "results/deepseek/deepseek-r1/question_sets.csv",
        "gemini-2.5-pro-preview": "results/google/gemini-2.5-pro-preview/question_sets.csv"
    }
    
    if model_name not in MODEL_PATHS:
        print(f"Model {model_name} not found in MODEL_PATHS")
        return False
    
    original_csv_path = MODEL_PATHS[model_name]
    if not os.path.exists(original_csv_path):
        print(f"Original CSV not found: {original_csv_path}")
        return False
    
    if not os.path.exists(reprocess_csv_path):
        print(f"Reprocess results not found: {reprocess_csv_path}")
        return False
    
    # Load files
    print(f"Loading original CSV: {original_csv_path}")
    original_df = pd.read_csv(original_csv_path)
    
    print(f"Loading reprocess results: {reprocess_csv_path}")
    reprocess_df = pd.read_csv(reprocess_csv_path)
    
    # Add new columns if they don't exist
    if 'reprocess_timestamp' not in original_df.columns:
        original_df['reprocess_timestamp'] = None
    if 'reprocess_status' not in original_df.columns:
        original_df['reprocess_status'] = None
    if 'new_llm_ranking' not in original_df.columns:
        original_df['new_llm_ranking'] = None
    if 'new_raw_llm_ranking_output' not in original_df.columns:
        original_df['new_raw_llm_ranking_output'] = None
    
    # Get timestamp from reprocess results
    timestamp = reprocess_df['timestamp'].iloc[0] if 'timestamp' in reprocess_df.columns else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update rows with reprocessing results
    update_count = 0
    success_count = 0
    
    for _, result_row in reprocess_df.iterrows():
        group_id = result_row['group_id']
        
        # Find matching rows in original CSV
        mask = original_df['group_id'] == group_id
        matching_rows = original_df[mask]
        
        if len(matching_rows) > 0:
            # Update the original CSV
            original_df.loc[mask, 'reprocess_timestamp'] = timestamp
            
            if result_row['reprocess_successful']:
                original_df.loc[mask, 'reprocess_status'] = 'SUCCESS'
                original_df.loc[mask, 'new_llm_ranking'] = result_row['extracted_ranking']
                original_df.loc[mask, 'new_raw_llm_ranking_output'] = result_row['raw_response']
                success_count += 1
            else:
                error_type = result_row['error_type'] if 'error_type' in result_row else 'UNKNOWN_ERROR'
                original_df.loc[mask, 'reprocess_status'] = f"FAILED: {error_type}"
            
            update_count += 1
        else:
            print(f"Warning: Group {group_id} not found in original CSV")
    
    # Create backup
    backup_path = f"{original_csv_path}.backup_{timestamp}"
    print(f"Creating backup: {backup_path}")
    original_df.to_csv(backup_path, index=False)
    
    # Save updated CSV
    print(f"Updating original CSV: {original_csv_path}")
    original_df.to_csv(original_csv_path, index=False)
    
    print(f"\n=== UPDATE SUMMARY ===")
    print(f"Total groups updated: {update_count}")
    print(f"Successful reprocessing: {success_count}")
    print(f"Failed reprocessing: {update_count - success_count}")
    print(f"Backup saved to: {backup_path}")
    
    return True

def main():
    # Configuration
    model_name = "o4-mini-2025-04-16"
    
    # Find the most recent reprocess results file
    reprocess_dir = "reprocess_failures_detailed"
    if not os.path.exists(reprocess_dir):
        print(f"Reprocess directory not found: {reprocess_dir}")
        return
    
    # List all reprocess files for the model
    reprocess_files = [f for f in os.listdir(reprocess_dir) if f.startswith(f"{model_name}_detailed_reprocess_")]
    
    if not reprocess_files:
        print(f"No reprocess files found for model {model_name}")
        return
    
    # Sort by timestamp (newest first)
    reprocess_files.sort(reverse=True)
    
    print("Available reprocess files:")
    for i, file in enumerate(reprocess_files, 1):
        print(f"{i}. {file}")
    
    # Use the most recent file or let user choose
    choice = input(f"Enter file number (or press Enter for most recent): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(reprocess_files):
        selected_file = reprocess_files[int(choice) - 1]
    else:
        selected_file = reprocess_files[0]  # Most recent
    
    reprocess_csv_path = os.path.join(reprocess_dir, selected_file)
    print(f"Using file: {selected_file}")
    
    # Update the CSV
    success = update_csv_from_reprocess_results(model_name, reprocess_csv_path)
    
    if success:
        print("CSV update completed successfully!")
    else:
        print("CSV update failed!")

if __name__ == "__main__":
    main()
