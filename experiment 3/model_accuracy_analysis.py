import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_model_accuracy():
    """
    חישוב שיעור ההצלחה עבור כל מודל בתקיית results
    """
    results_dir = 'results'
    model_accuracy_data = []
    
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found!")
        return None
    
    # עבור כל ספק (provider)
    for provider in os.listdir(results_dir):
        provider_path = os.path.join(results_dir, provider)
        
        if not os.path.isdir(provider_path):
            continue
            
        # עבור כל מודל
        for model in os.listdir(provider_path):
            model_path = os.path.join(provider_path, model)
            
            if not os.path.isdir(model_path):
                continue
                
            # נתיב לקובץ individual_results.csv
            individual_results_path = os.path.join(model_path, 'individual_results.csv')
            
            if not os.path.exists(individual_results_path):
                print(f"Missing individual_results.csv for {provider}/{model}")
                continue
                
            try:
                # קריאת הנתונים
                df = pd.read_csv(individual_results_path)
                
                # בדיקה שיש עמודת is_correct
                if 'is_correct' not in df.columns:
                    print(f"Column 'is_correct' not found in {provider}/{model}")
                    continue
                
                # חישוב שיעור ההצלחה
                total_questions = len(df)
                correct_answers = df['is_correct'].sum()
                accuracy_rate = correct_answers / total_questions if total_questions > 0 else 0
                
                model_accuracy_data.append({
                    'provider': provider,
                    'model': model,
                    'full_model_name': f"{provider}/{model}",
                    'total_questions': total_questions,
                    'correct_answers': correct_answers,
                    'accuracy_rate': accuracy_rate,
                    'accuracy_percentage': accuracy_rate * 100
                })
                
                print(f"Processed {provider}/{model}: {correct_answers}/{total_questions} = {accuracy_rate:.3f}")
                
            except Exception as e:
                print(f"Error processing {provider}/{model}: {str(e)}")
                continue
    
    return pd.DataFrame(model_accuracy_data)

def display_results_table(accuracy_df):
    """
    הצגת התוצאות בטבלה
    """
    if accuracy_df is None or len(accuracy_df) == 0:
        print("No data to display")
        return
    
    # מיון לפי שיעור ההצלחה
    sorted_df = accuracy_df.sort_values('accuracy_percentage', ascending=False)
    
    print("\n" + "="*80)
    print("MODEL ACCURACY ANALYSIS")
    print("="*80)
    
    print(f"{'Rank':<4} {'Model':<40} {'Correct':<8} {'Total':<8} {'Accuracy':<10}")
    print("-" * 80)
    
    for i, row in sorted_df.iterrows():
        rank = sorted_df.index.get_loc(i) + 1
        model_name = row['full_model_name']
        correct = int(row['correct_answers'])
        total = int(row['total_questions'])
        accuracy = f"{row['accuracy_percentage']:.2f}%"
        
        print(f"{rank:<4} {model_name:<40} {correct:<8} {total:<8} {accuracy:<10}")
    
    # סטטיסטיקות כלליות
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Number of models analyzed: {len(sorted_df)}")
    print(f"Average accuracy: {sorted_df['accuracy_percentage'].mean():.2f}%")
    print(f"Best performing model: {sorted_df.iloc[0]['full_model_name']} ({sorted_df.iloc[0]['accuracy_percentage']:.2f}%)")
    print(f"Worst performing model: {sorted_df.iloc[-1]['full_model_name']} ({sorted_df.iloc[-1]['accuracy_percentage']:.2f}%)")
    print(f"Standard deviation: {sorted_df['accuracy_percentage'].std():.2f}%")

def create_accuracy_visualizations(accuracy_df):
    """
    יצירת גרפים להצגת התוצאות
    """
    if accuracy_df is None or len(accuracy_df) == 0:
        print("No data to visualize")
        return
    
    # מיון לפי שיעור ההצלחה
    sorted_df = accuracy_df.sort_values('accuracy_percentage', ascending=True)
    
    # יצירת גרף עמודות אופקי
    plt.figure(figsize=(12, max(8, len(sorted_df) * 0.4)))
    
    # צביעה לפי ספק
    providers = sorted_df['provider'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(providers)))
    provider_colors = dict(zip(providers, colors))
    
    bar_colors = [provider_colors[provider] for provider in sorted_df['provider']]
    
    bars = plt.barh(range(len(sorted_df)), sorted_df['accuracy_percentage'], color=bar_colors)
    
    # הוספת תוויות על העמודות
    for i, (bar, accuracy) in enumerate(zip(bars, sorted_df['accuracy_percentage'])):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{accuracy:.1f}%', va='center', ha='left', fontsize=9)
    
    # עיצוב הגרף
    plt.yticks(range(len(sorted_df)), sorted_df['full_model_name'])
    plt.xlabel('Accuracy Percentage (%)')
    plt.title('Model Accuracy Comparison\n(Individual Question Answering)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # הוספת מקרא לפי ספקים
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=provider_colors[provider], label=provider) 
                      for provider in providers]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # גרף התפלגות
    plt.figure(figsize=(10, 6))
    plt.hist(sorted_df['accuracy_percentage'], bins=min(10, len(sorted_df)), 
             alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Accuracy Percentage (%)')
    plt.ylabel('Number of Models')
    plt.title('Distribution of Model Accuracy Rates')
    plt.grid(alpha=0.3)
    
    # הוספת קו לממוצע
    mean_accuracy = sorted_df['accuracy_percentage'].mean()
    plt.axvline(mean_accuracy, color='red', linestyle='--', 
                label=f'Mean: {mean_accuracy:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_csv(accuracy_df, filename='model_accuracy_results.csv'):
    """
    שמירת התוצאות לקובץ CSV
    """
    if accuracy_df is None or len(accuracy_df) == 0:
        print("No data to save")
        return
    
    # מיון לפי שיעור ההצלחה
    sorted_df = accuracy_df.sort_values('accuracy_percentage', ascending=False)
    
    # הוספת דירוג
    sorted_df['rank'] = range(1, len(sorted_df) + 1)
    
    # שמירה
    sorted_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

def main():
    """
    הפונקציה הראשית
    """
    print("Starting Model Accuracy Analysis...")
    
    # חישוב שיעור ההצלחה
    accuracy_df = calculate_model_accuracy()
    
    if accuracy_df is None or len(accuracy_df) == 0:
        print("No model data found!")
        return
    
    # הצגת התוצאות בטבלה
    display_results_table(accuracy_df)
    
    # יצירת גרפים
    print("\nCreating visualizations...")
    create_accuracy_visualizations(accuracy_df)
    
    # שמירת התוצאות
    save_results_to_csv(accuracy_df)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()
