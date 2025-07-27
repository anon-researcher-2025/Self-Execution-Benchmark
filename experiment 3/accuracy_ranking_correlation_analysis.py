#!/usr/bin/env python3
"""
Analysis script to calculate the correlation between model accuracy and question ranking ability.
This script combines accuracy data from model_accuracy_results.csv with ranking scores from all_model_statistics.csv
and calculates various correlation metrics.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_merge_data():
    """Load accuracy and ranking data and merge them by model name."""
    print("Loading accuracy data...")
    accuracy_df = pd.read_csv('model_accuracy_results.csv')
    
    print("Loading model statistics data...")
    stats_df = pd.read_csv('all_model_statistics.csv')
    
    # Merge the dataframes on the model name
    # Use full_model_name from accuracy_df to match model_name in stats_df
    merged_df = accuracy_df.merge(
        stats_df, 
        left_on='full_model_name', 
        right_on='model_name', 
        how='inner'
    )
    
    print(f"Successfully merged data for {len(merged_df)} models")
    print("Merged models:")
    for _, row in merged_df.iterrows():
        print(f"  - {row['full_model_name']}: {row['accuracy_percentage']:.2f}% accuracy, {row['avg_question_order_score']:.4f} ranking score")
    
    return merged_df

def calculate_correlations(df):
    """Calculate Pearson and Spearman correlations between accuracy and ranking score."""
    accuracy = df['accuracy_percentage']
    ranking_score = df['avg_question_order_score']
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(accuracy, ranking_score)
    spearman_corr, spearman_p = spearmanr(accuracy, ranking_score)
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS RESULTS")
    print("="*60)
    print(f"Number of models analyzed: {len(df)}")
    print(f"Accuracy range: {accuracy.min():.2f}% - {accuracy.max():.2f}%")
    print(f"Ranking score range: {ranking_score.min():.4f} - {ranking_score.max():.4f}")
    print()
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    # Interpretation
    print("\nInterpretation:")
    if abs(pearson_corr) > 0.7:
        strength = "strong"
    elif abs(pearson_corr) > 0.5:
        strength = "moderate"
    elif abs(pearson_corr) > 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    
    direction = "positive" if pearson_corr > 0 else "negative"
    print(f"There is a {strength} {direction} correlation between model accuracy and ranking ability.")
    
    if pearson_p < 0.05:
        print("The correlation is statistically significant (p < 0.05).")
    else:
        print("The correlation is not statistically significant (p >= 0.05).")
    
    return {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'n_models': len(df)
    }

def create_visualization(df, correlation_results):
    """Create scatter plot showing the relationship between accuracy and ranking score."""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(df['accuracy_percentage'], df['avg_question_order_score'], 
                alpha=0.7, s=100, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add model labels
    for _, row in df.iterrows():
        plt.annotate(row['model'], 
                    (row['accuracy_percentage'], row['avg_question_order_score']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    # Add trend line
    z = np.polyfit(df['accuracy_percentage'], df['avg_question_order_score'], 1)
    p = np.poly1d(z)
    plt.plot(df['accuracy_percentage'], p(df['accuracy_percentage']), 
             "r--", alpha=0.8, linewidth=2, label='Trend line')
    
    # Formatting
    plt.xlabel('Model Accuracy (%)', fontsize=12)
    plt.ylabel('Average Question Order Score', fontsize=12)
    plt.title('Correlation between Model Accuracy and Question Ranking Ability', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add correlation info as text
    textstr = f'Pearson r = {correlation_results["pearson_corr"]:.3f} (p = {correlation_results["pearson_p"]:.3f})\n'
    textstr += f'Spearman ρ = {correlation_results["spearman_corr"]:.3f} (p = {correlation_results["spearman_p"]:.3f})\n'
    textstr += f'N = {correlation_results["n_models"]} models'
    
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('accuracy_vs_ranking_correlation.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'accuracy_vs_ranking_correlation.png'")
    plt.show()

def save_detailed_results(df, correlation_results):
    """Save detailed results to CSV file."""
    # Create results dataframe
    results_df = df[['full_model_name', 'model', 'accuracy_percentage', 'avg_question_order_score']].copy()
    results_df = results_df.sort_values('accuracy_percentage', ascending=False)
    
    # Add ranking
    results_df['accuracy_rank'] = range(1, len(results_df) + 1)
    results_df['ranking_score_rank'] = results_df['avg_question_order_score'].rank(ascending=False, method='min')
    results_df['rank_difference'] = abs(results_df['accuracy_rank'] - results_df['ranking_score_rank'])
    
    # Save to CSV
    output_file = 'accuracy_ranking_correlation_detailed.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to '{output_file}'")
    
    # Save correlation summary
    summary_df = pd.DataFrame([correlation_results])
    summary_file = 'correlation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Correlation summary saved to '{summary_file}'")
    
    return results_df

def main():
    """Main analysis function."""
    print("Starting Accuracy vs Ranking Correlation Analysis")
    print("="*60)
    
    try:
        # Load and merge data
        merged_df = load_and_merge_data()
        
        # Calculate correlations
        correlation_results = calculate_correlations(merged_df)
        
        # Create visualization
        create_visualization(merged_df, correlation_results)
        
        # Save detailed results
        detailed_results = save_detailed_results(merged_df, correlation_results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Files generated:")
        print("  - accuracy_vs_ranking_correlation.png (visualization)")
        print("  - accuracy_ranking_correlation_detailed.csv (detailed results)")
        print("  - correlation_summary.csv (correlation statistics)")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
        print("Make sure 'model_accuracy_results.csv' and 'all_model_statistics.csv' exist in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
