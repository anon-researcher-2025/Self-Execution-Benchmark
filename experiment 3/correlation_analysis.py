import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr

def load_accuracy_results():
    """
    טעינת תוצאות ה-accuracy שחישבנו
    """
    accuracy_file = 'model_accuracy_results.csv'
    
    if not os.path.exists(accuracy_file):
        print(f"File {accuracy_file} not found. Please run model_accuracy_analysis.py first.")
        return None
    
    df = pd.read_csv(accuracy_file)
    print(f"Loaded accuracy results for {len(df)} models")
    return df

def load_external_accuracy_results(external_file_path):
    """
    טעינת תוצאות ה-accuracy מהקובץ החיצוני
    """
    if not os.path.exists(external_file_path):
        print(f"External file {external_file_path} not found!")
        return None
    
    try:
        external_df = pd.read_csv(external_file_path)
        print(f"Loaded external accuracy results for {len(external_df)} models")
        print(f"Columns in external file: {external_df.columns.tolist()}")
        return external_df
    except Exception as e:
        print(f"Error loading external file: {e}")
        return None

def load_ranking_results():
    """
    טעינת תוצאות המיון מהקובץ all_model_statistics.csv
    """
    ranking_file = 'all_model_statistics.csv'
    
    if not os.path.exists(ranking_file):
        print(f"File {ranking_file} not found. Please run the statistics analysis first.")
        return None
    
    df = pd.read_csv(ranking_file)
    print(f"Loaded ranking results for {len(df)} models")
    return df

def merge_and_analyze(accuracy_df, external_df, ranking_df, 
                     model_col='full_model_name', 
                     external_model_col='model_name',
                     external_accuracy_col='accuracy_percentage'):
    """
    מיזוג הנתונים וחישוב קורלציות
    """
    
    # מיזוג נתוני ה-accuracy שלנו עם נתוני המיון
    if ranking_df is not None:
        merged_df = pd.merge(accuracy_df, ranking_df, 
                           left_on=model_col, right_on='model_name', 
                           how='inner')
        print(f"Merged accuracy and ranking data: {len(merged_df)} models")
    else:
        merged_df = accuracy_df.copy()
    
    # מיזוג עם הנתונים החיצוניים
    if external_df is not None:
        final_df = pd.merge(merged_df, external_df, 
                          left_on=model_col, right_on=external_model_col, 
                          how='inner')
        print(f"Final merged data: {len(final_df)} models")
    else:
        print("No external data provided, analyzing internal data only")
        final_df = merged_df.copy()
    
    return final_df

def calculate_correlations(df, internal_accuracy_col='accuracy_percentage', 
                         external_accuracy_col='external_accuracy',
                         ranking_score_col='avg_question_order_score'):
    """
    חישוב קורלציות בין המדדים השונים
    """
    correlations = {}
    
    # זמינות העמודות
    available_cols = df.columns.tolist()
    print(f"Available columns: {available_cols}")
    
    # קורלציה בין accuracy פנימי וחיצוני
    if external_accuracy_col in df.columns:
        internal_acc = df[internal_accuracy_col]
        external_acc = df[external_accuracy_col]
        
        # הסרת ערכים חסרים
        mask = ~(pd.isna(internal_acc) | pd.isna(external_acc))
        if mask.sum() > 1:
            pearson_corr, pearson_p = pearsonr(internal_acc[mask], external_acc[mask])
            spearman_corr, spearman_p = spearmanr(internal_acc[mask], external_acc[mask])
            
            correlations['internal_vs_external_accuracy'] = {
                'pearson': (pearson_corr, pearson_p),
                'spearman': (spearman_corr, spearman_p),
                'n_samples': mask.sum()
            }
            
            print(f"\n=== קורלציה בין Accuracy פנימי וחיצוני ===")
            print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
            print(f"Spearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
            print(f"Number of samples: {mask.sum()}")
    
    # קורלציה בין accuracy לציון מיון
    if ranking_score_col in df.columns:
        internal_acc = df[internal_accuracy_col]
        ranking_score = df[ranking_score_col]
        
        # הסרת ערכים חסרים
        mask = ~(pd.isna(internal_acc) | pd.isna(ranking_score))
        if mask.sum() > 1:
            pearson_corr, pearson_p = pearsonr(internal_acc[mask], ranking_score[mask])
            spearman_corr, spearman_p = spearmanr(internal_acc[mask], ranking_score[mask])
            
            correlations['accuracy_vs_ranking'] = {
                'pearson': (pearson_corr, pearson_p),
                'spearman': (spearman_corr, spearman_p),
                'n_samples': mask.sum()
            }
            
            print(f"\n=== קורלציה בין Accuracy וציון מיון ===")
            print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
            print(f"Spearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
            print(f"Number of samples: {mask.sum()}")
    
    # קורלציה בין accuracy חיצוני לציון מיון
    if external_accuracy_col in df.columns and ranking_score_col in df.columns:
        external_acc = df[external_accuracy_col]
        ranking_score = df[ranking_score_col]
        
        # הסרת ערכים חסרים
        mask = ~(pd.isna(external_acc) | pd.isna(ranking_score))
        if mask.sum() > 1:
            pearson_corr, pearson_p = pearsonr(external_acc[mask], ranking_score[mask])
            spearman_corr, spearman_p = spearmanr(external_acc[mask], ranking_score[mask])
            
            correlations['external_accuracy_vs_ranking'] = {
                'pearson': (pearson_corr, pearson_p),
                'spearman': (spearman_corr, spearman_p),
                'n_samples': mask.sum()
            }
            
            print(f"\n=== קורלציה בין Accuracy חיצוני וציון מיון ===")
            print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
            print(f"Spearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
            print(f"Number of samples: {mask.sum()}")
    
    return correlations

def create_correlation_plots(df, internal_accuracy_col='accuracy_percentage', 
                           external_accuracy_col='external_accuracy',
                           ranking_score_col='avg_question_order_score'):
    """
    יצירת גרפי פיזור להצגת הקורלציות
    """
    # קביעת מספר הגרפים
    available_plots = []
    if external_accuracy_col in df.columns:
        available_plots.append('internal_vs_external')
    if ranking_score_col in df.columns:
        available_plots.append('accuracy_vs_ranking')
    if external_accuracy_col in df.columns and ranking_score_col in df.columns:
        available_plots.append('external_vs_ranking')
    
    if not available_plots:
        print("No data available for correlation plots")
        return
    
    # יצירת גרפים
    n_plots = len(available_plots)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # גרף 1: Accuracy פנימי מול חיצוני
    if 'internal_vs_external' in available_plots:
        ax = axes[plot_idx]
        x = df[internal_accuracy_col]
        y = df[external_accuracy_col]
        
        # הסרת ערכים חסרים
        mask = ~(pd.isna(x) | pd.isna(y))
        if mask.sum() > 0:
            ax.scatter(x[mask], y[mask], alpha=0.7, s=50)
            
            # הוספת קו רגרסיה
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            ax.plot(x[mask], p(x[mask]), "r--", alpha=0.8)
            
            # חישוב קורלציה לכותרת
            corr, _ = pearsonr(x[mask], y[mask])
            
            ax.set_xlabel('Internal Accuracy (%)')
            ax.set_ylabel('External Accuracy (%)')
            ax.set_title(f'Internal vs External Accuracy\n(r = {corr:.3f})')
            ax.grid(True, alpha=0.3)
            
            # הוספת שמות מודלים
            for i, model in enumerate(df[mask]['full_model_name']):
                if i < 15:  # רק 15 הראשונים כדי לא לגדוס
                    ax.annotate(model.split('/')[-1], (x[mask].iloc[i], y[mask].iloc[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        plot_idx += 1
    
    # גרף 2: Accuracy מול ציון מיון
    if 'accuracy_vs_ranking' in available_plots:
        ax = axes[plot_idx]
        x = df[internal_accuracy_col]
        y = df[ranking_score_col]
        
        # הסרת ערכים חסרים
        mask = ~(pd.isna(x) | pd.isna(y))
        if mask.sum() > 0:
            ax.scatter(x[mask], y[mask], alpha=0.7, s=50, color='green')
            
            # הוספת קו רגרסיה
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            ax.plot(x[mask], p(x[mask]), "r--", alpha=0.8)
            
            # חישוב קורלציה לכותרת
            corr, _ = pearsonr(x[mask], y[mask])
            
            ax.set_xlabel('Question Accuracy (%)')
            ax.set_ylabel('Ranking Score')
            ax.set_title(f'Accuracy vs Ranking Ability\n(r = {corr:.3f})')
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # גרף 3: Accuracy חיצוני מול ציון מיון
    if 'external_vs_ranking' in available_plots:
        ax = axes[plot_idx]
        x = df[external_accuracy_col]
        y = df[ranking_score_col]
        
        # הסרת ערכים חסרים
        mask = ~(pd.isna(x) | pd.isna(y))
        if mask.sum() > 0:
            ax.scatter(x[mask], y[mask], alpha=0.7, s=50, color='orange')
            
            # הוספת קו רגרסיה
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            ax.plot(x[mask], p(x[mask]), "r--", alpha=0.8)
            
            # חישוב קורלציה לכותרת
            corr, _ = pearsonr(x[mask], y[mask])
            
            ax.set_xlabel('External Accuracy (%)')
            ax.set_ylabel('Ranking Score')
            ax.set_title(f'External Accuracy vs Ranking\n(r = {corr:.3f})')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_comprehensive_data():
    """
    טעינת הנתונים המקיפים והמיזוג עם מספר השאלות הנכון
    """
    # הנתונים החדשים שהבאת
    comprehensive_data = """model_name,failed_to_order,correct_individual_answers_count,avg_question_order_score,adjusted_avg_question_order_score,group_individual_duration_avg,group_individual_duration_std,group_individual_duration_std_percentage,group_individual_duration_std_percentage_median,combined_duration_avg,combined_duration_std
anthropic/claude-3.7-sonnet-thinking,20,813,0.7478260869565218,0.6880000000000001,3233.643665158371,3601.199899529569,0.9955223851572254,1.0238233847577534,4764.221739130435,2595.2801917637244
deepseek/deepseek-chat-v3,0,863,0.6653333333333332,0.6653333333333332,254.888,190.14705820675556,0.7081402477053116,0.6909459651577958,298.016,71.68225111209881
deepseek/deepseek-r1,0,903,0.7,0.7,860.63,642.3885237532919,0.6278133687117033,0.5828134116731337,2047.544,926.00045541984
google/gemini-2.5-flash-preview,0,696,0.6766666666666667,0.6766666666666667,1688.928,2606.1436925518146,0.7930596563680066,0.690026564160924,1284.78,501.74666362401587
google/gemini-2.5-pro-preview,14,291,0.7055555555555556,0.254,3778.2901234567903,3256.818546329808,0.8654252910713095,0.8570344800681841,7821.2555555555555,2545.740251545624
meta-llama/llama-3.1-8b-instruct,24,567,0.6172566371681416,0.558,679.261,774.9743099863707,0.5148931712393093,0.3458522232548711,641.26,1218.970040747911
meta-llama/llama-3.2-3b-instruct,134,479,0.5502873563218391,0.25533333333333336,447.908,417.0189520009422,0.5567050063874384,0.4547402198980831,274.308,111.74083044597002
meta-llama/llama-4-scout,0,851,0.6793333333333332,0.6793333333333332,395.415,145.2048444833118,0.3547398570670077,0.32287457338504844,690.288,112.25149732009106
mistralai/mistral-7b-instruct,109,504,0.5342789598108747,0.3013333333333333,322.258,288.077790118467,0.5475196204039512,0.515394061556179,333.912,101.62807347022316
mistralai/mistral-small-3.1-24b-instruct,6,803,0.5990437158469946,0.5846666666666667,292.021,134.71639696270205,0.4327274650988051,0.402310375461872,338.548,81.36792150843236
openai/gpt-4.1,1,883,0.7342704149933065,0.7313333333333333,536.547,714.0482729452445,0.539371979761881,0.4285045590390879,470.2,96.29335199562956
openai/gpt-4.1-mini,0,880,0.704,0.704,274.173,167.25231397377038,0.563537618767129,0.5321916839706088,534.992,107.7407267135835
openai/o4-mini,101,494,0.6588366890380314,0.39266666666666666,463.89233576642334,279.2046265175099,0.5350451156605831,0.49629725947312114,971.1140939597316,389.84006648259395
qwen/qwen-2.5-7b-instruct,3,691,0.6005398110661269,0.5933333333333334,292.506,135.98207489945975,0.4478293720885688,0.4360496247746619,308.604,90.84201821344774"""
    
    # שמירה לקובץ זמני
    with open('comprehensive_data.csv', 'w') as f:
        f.write(comprehensive_data)
    
    # טעינת הנתונים המקיפים
    df_comprehensive = pd.read_csv('comprehensive_data.csv')
    
    # טעינת מספר השאלות הנכון מהקובץ שכבר חישבנו
    accuracy_results = pd.read_csv('model_accuracy_results.csv')
    
    # מיזוג הנתונים לפי שם המודל
    df = pd.merge(df_comprehensive, 
                  accuracy_results[['full_model_name', 'total_questions']], 
                  left_on='model_name', 
                  right_on='full_model_name', 
                  how='left')
    
    # חישוב שיעור הצלחה באחוזים לפי מספר השאלות הנכון לכל מודל
    df['question_accuracy_percentage'] = (df['correct_individual_answers_count'] / df['total_questions']) * 100
    
    # דיוק המיון - זה ה-adjusted_avg_question_order_score שלוקח בחשבון כשלונות
    df['ranking_accuracy'] = df['adjusted_avg_question_order_score']
    
    # הצגת פירוט למודלים
    print(f"Loaded comprehensive data for {len(df)} models")
    print("\nמספר השאלות לכל מודל:")
    for _, row in df.iterrows():
        if pd.notna(row['total_questions']):
            print(f"  {row['model_name']}: {int(row['total_questions'])} שאלות, {row['correct_individual_answers_count']} נכונות = {row['question_accuracy_percentage']:.1f}%")
        else:
            print(f"  {row['model_name']}: לא נמצא מספר שאלות - משתמש ב-1000 כברירת מחדל")
            df.loc[df['model_name'] == row['model_name'], 'total_questions'] = 1000
            df.loc[df['model_name'] == row['model_name'], 'question_accuracy_percentage'] = (row['correct_individual_answers_count'] / 1000) * 100
    
    return df

def main():
    """
    הפונקציה הראשית - חישוב קורלציה בין הצלחה בשאלות להצלחה במיון
    """
    print("=== ניתוח קורלציה: הצלחה בשאלות VS הצלחה במיון ===\n")
    
    # טעינת הנתונים המקיפים
    df = load_comprehensive_data()    
    if len(df) == 0:
        print("No data found!")
        return
    
    print(f"=== סיכום נתונים ===")
    print(f"מספר מודלים: {len(df)}")
    print(f"טווח דיוק שאלות: {df['question_accuracy_percentage'].min():.1f}% - {df['question_accuracy_percentage'].max():.1f}%")
    print(f"טווח דיוק מיון: {df['ranking_accuracy'].min():.3f} - {df['ranking_accuracy'].max():.3f}")
    
    # הצגת טבלה מסודרת
    display_df = df[['model_name', 'correct_individual_answers_count', 'question_accuracy_percentage', 
                     'avg_question_order_score', 'adjusted_avg_question_order_score', 'ranking_accuracy']].copy()
    display_df = display_df.sort_values('question_accuracy_percentage', ascending=False)
    display_df['model_short'] = display_df['model_name'].str.split('/').str[-1]
    
    print(f"\n=== דירוג מודלים ===")
    print(f"{'Rank':<4} {'Model':<30} {'Quest_Acc':<10} {'Rank_Raw':<10} {'Rank_Adj':<10}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(display_df.iterrows(), 1):
        print(f"{i:<4} {row['model_short']:<30} {row['question_accuracy_percentage']:<9.1f}% {row['avg_question_order_score']:<9.3f} {row['ranking_accuracy']:<9.3f}")
    
    # חישוב קורלציות
    question_accuracy = df['question_accuracy_percentage']
    ranking_accuracy = df['ranking_accuracy']
      
    # הסרת ערכים חסרים אם יש
    mask = ~(pd.isna(question_accuracy) | pd.isna(ranking_accuracy))
    accuracy_clean = question_accuracy[mask]
    ranking_clean = ranking_accuracy[mask]
    
    if len(accuracy_clean) < 2:
        print("Not enough data for correlation analysis!")
        return
    
    # חישוב קורלציות
    pearson_corr, pearson_p = pearsonr(accuracy_clean, ranking_clean)
    spearman_corr, spearman_p = spearmanr(accuracy_clean, ranking_clean)
    
    print(f"\n=== תוצאות קורלציה ===")
    print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
    print(f"Spearman correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
    print(f"מספר דגימות: {len(accuracy_clean)}")
    
    # פרשנות התוצאות
    print(f"\n=== פרשנות ===")
    if abs(pearson_corr) > 0.7:
        strength = "חזקה מאוד"
    elif abs(pearson_corr) > 0.5:
        strength = "חזקה"
    elif abs(pearson_corr) > 0.3:
        strength = "בינונית"
    elif abs(pearson_corr) > 0.1:
        strength = "חלשה"
    else:
        strength = "חלשה מאוד"
    
    direction = "חיובית" if pearson_corr > 0 else "שלילית"
    significance = "מובהקת סטטיסטית" if pearson_p < 0.05 else "לא מובהקת סטטיסטית"
    
    print(f"קורלציה {strength} ו{direction} ({significance})")
    
    if pearson_corr > 0.5:
        print("מודלים שטובים יותר בשאלות נוטים להיות טובים יותר גם במיון שאלות")
    elif pearson_corr < -0.5:
        print("מודלים שטובים יותר בשאלות נוטים להיות גרועים יותר במיון שאלות (מפתיע!)")
    else:
        print("אין קשר חזק בין הצלחה בשאלות להצלחה במיון שאלות")
    
    # יצירת גרף פיזור
    plt.figure(figsize=(10, 8))
      # צביעה לפי ספק
    df['provider'] = df['model_name'].str.split('/').str[0]
    providers = df['provider'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(providers)))
    provider_colors = dict(zip(providers, colors))
    
    for provider in providers:
        provider_data = df[df['provider'] == provider]
        plt.scatter(provider_data['question_accuracy_percentage'], 
                   provider_data['ranking_accuracy'],
                   label=provider, alpha=0.8, s=100,
                   color=provider_colors[provider])
    
    # הוספת קו רגרסיה
    z = np.polyfit(accuracy_clean, ranking_clean, 1)
    p = np.poly1d(z)
    plt.plot(accuracy_clean, p(accuracy_clean), "r--", alpha=0.8, linewidth=2)
      # הוספת שמות מודלים
    for _, row in df.iterrows():
        model_short = row['model_name'].split('/')[-1]
        plt.annotate(model_short, 
                    (row['question_accuracy_percentage'], row['ranking_accuracy']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    plt.xlabel('Question Accuracy (%)')
    plt.ylabel('Ranking Accuracy (Adjusted Score)')
    plt.title(f'Correlation: Question Accuracy vs Ranking Accuracy\n(r = {pearson_corr:.3f}, p = {pearson_p:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('correlation_accuracy_vs_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()    # שמירת תוצאות
    df_output = df[['model_name', 'correct_individual_answers_count', 'question_accuracy_percentage', 
                    'avg_question_order_score', 'adjusted_avg_question_order_score', 'ranking_accuracy']].copy()
    df_output['question_accuracy_rank'] = df_output['question_accuracy_percentage'].rank(ascending=False)
    df_output['ranking_accuracy_rank'] = df_output['ranking_accuracy'].rank(ascending=False)
    df_output['rank_difference'] = df_output['question_accuracy_rank'] - df_output['ranking_accuracy_rank']
    
    df_output.to_csv('accuracy_vs_ranking_correlation.csv', index=False)
    print(f"\nתוצאות נשמרו בקובץ: accuracy_vs_ranking_correlation.csv")
    
    # ניקוי קובץ זמני
    import os
    os.remove('comprehensive_data.csv')
    
    print("\nניתוח הקורלציה הושלם!")

if __name__ == "__main__":
    main()
