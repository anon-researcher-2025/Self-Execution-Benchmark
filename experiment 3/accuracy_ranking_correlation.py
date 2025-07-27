import pandas as pd
import numpy as np

def calculate_accuracy_ranking_correlation():
    """
    חישוב קורלציה בין שיעור הצלחה בשאלות לבין יכולת מיון שאלות
    """
    
    # יצירת הנתונים מהטבלה שהבאת
    data = """model_name,failed_to_order,correct_individual_answers_count,avg_question_order_score,adjusted_avg_question_order_score,group_individual_duration_avg,group_individual_duration_std,group_individual_duration_std_percentage,group_individual_duration_std_percentage_median,combined_duration_avg,combined_duration_std
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
    
    # שמירת הנתונים לקובץ זמני
    with open('correlation_data.csv', 'w') as f:
        f.write(data)
    
    # טעינת הנתונים
    df = pd.read_csv('correlation_data.csv')
    
    # חישוב שיעור הצלחה באחוזים
    # נניח שכל מודל נבדק על 1000 שאלות (כמו שרוב המודלים)
    df['total_questions'] = 1000
    
    # תיקון עבור מודלים שיש להם מספר שאלות שונה
    # (על בסיס הנתונים מהקובץ המקורי)
    specific_totals = {
        'anthropic/claude-3.7-sonnet-thinking': 920,
        'google/gemini-2.5-pro-preview': 360,
        'openai/o4-mini': 596
    }
    
    for model, total in specific_totals.items():
        df.loc[df['model_name'] == model, 'total_questions'] = total
    
    # חישוב אחוז הצלחה
    df['accuracy_percentage'] = (df['correct_individual_answers_count'] / df['total_questions']) * 100
    
    print("=== ניתוח קורלציה בין הצלחה בשאלות ויכולת מיון ===\n")
    
    # הצגת הנתונים
    print(f"{'Model':<35} {'Accuracy %':<12} {'Ranking Score':<15} {'Questions':<10}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        model_short = row['model_name'].split('/')[-1]
        accuracy = row['accuracy_percentage']
        ranking = row['avg_question_order_score']
        questions = row['total_questions']
        
        print(f"{model_short:<35} {accuracy:<12.1f} {ranking:<15.3f} {questions:<10}")
    
    # חישוב קורלציות
    accuracy_values = df['accuracy_percentage'].values
    ranking_values = df['avg_question_order_score'].values
    
    # קורלציה פשוטה (Pearson)
    correlation_matrix = np.corrcoef(accuracy_values, ranking_values)
    pearson_correlation = correlation_matrix[0, 1]
    
    # קורלציה של Spearman (rank correlation) - פשוט יותר ללא scipy
    def spearman_correlation(x, y):
        """חישוב קורלציית Spearman פשוטה"""
        n = len(x)
        
        # יצירת דירוגים
        x_ranks = np.argsort(np.argsort(x)) + 1
        y_ranks = np.argsort(np.argsort(y)) + 1
        
        # חישוב הקורלציה
        x_mean = np.mean(x_ranks)
        y_mean = np.mean(y_ranks)
        
        numerator = np.sum((x_ranks - x_mean) * (y_ranks - y_mean))
        denominator = np.sqrt(np.sum((x_ranks - x_mean)**2) * np.sum((y_ranks - y_mean)**2))
        
        return numerator / denominator if denominator != 0 else 0
    
    spearman_corr = spearman_correlation(accuracy_values, ranking_values)
    
    # הצגת תוצאות הקורלציה
    print(f"\n=== תוצאות קורלציה ===")
    print(f"Pearson correlation:  {pearson_correlation:.3f}")
    print(f"Spearman correlation: {spearman_corr:.3f}")
    
    # פרשנות התוצאות
    print(f"\n=== פרשנות ===")
    
    def interpret_correlation(corr):
        abs_corr = abs(corr)
        if abs_corr < 0.1:
            return "קורלציה חלשה מאוד"
        elif abs_corr < 0.3:
            return "קורלציה חלשה"
        elif abs_corr < 0.5:
            return "קורלציה בינונית"
        elif abs_corr < 0.7:
            return "קורלציה חזקה"
        else:
            return "קורלציה חזקה מאוד"
    
    print(f"Pearson:  {interpret_correlation(pearson_correlation)}")
    print(f"Spearman: {interpret_correlation(spearman_corr)}")
    
    if pearson_correlation > 0:
        print("\nכיוון הקורלציה: חיובי")
        print("פירוש: ככל שמודל טוב יותר בשאלות בודדות, כך הוא גם טוב יותר במיון שאלות")
    else:
        print("\nכיוון הקורלציה: שלילי")
        print("פירוש: ככל שמודל טוב יותר בשאלות בודדות, כך הוא פחות טוב במיון שאלות")
    
    # זיהוי מודלים מעניינים
    print(f"\n=== מודלים מעניינים ===")
    
    # מודלים עם accuracy גבוה אבל ranking נמוך
    df['accuracy_rank'] = df['accuracy_percentage'].rank(ascending=False)
    df['ranking_rank'] = df['avg_question_order_score'].rank(ascending=False)
    df['rank_diff'] = df['accuracy_rank'] - df['ranking_rank']
    
    # מודלים שטובים בaccuracy אבל פחות טובים במיון
    accuracy_better = df[df['rank_diff'] < -3].sort_values('rank_diff')
    if len(accuracy_better) > 0:
        print("\nמודלים שטובים בשאלות אבל פחות טובים במיון:")
        for _, row in accuracy_better.iterrows():
            model = row['model_name'].split('/')[-1]
            acc = row['accuracy_percentage']
            rank_score = row['avg_question_order_score']
            print(f"  {model}: {acc:.1f}% accuracy, {rank_score:.3f} ranking")
    
    # מודלים שטובים במיון אבל פחות טובים בaccuracy
    ranking_better = df[df['rank_diff'] > 3].sort_values('rank_diff', ascending=False)
    if len(ranking_better) > 0:
        print("\nמודלים שטובים במיון אבל פחות טובים בשאלות:")
        for _, row in ranking_better.iterrows():
            model = row['model_name'].split('/')[-1]
            acc = row['accuracy_percentage']
            rank_score = row['avg_question_order_score']
            print(f"  {model}: {acc:.1f}% accuracy, {rank_score:.3f} ranking")
    
    # שמירת התוצאות
    df_output = df[['model_name', 'accuracy_percentage', 'avg_question_order_score', 'total_questions']].copy()
    df_output['pearson_correlation'] = pearson_correlation
    df_output['spearman_correlation'] = spearman_corr
    
    df_output.to_csv('accuracy_ranking_correlation_results.csv', index=False)
    print(f"\nתוצאות נשמרו בקובץ: accuracy_ranking_correlation_results.csv")
    
    # ניקוי
    import os
    os.remove('correlation_data.csv')
    
    return pearson_correlation, spearman_corr, df

if __name__ == "__main__":
    pearson, spearman, data = calculate_accuracy_ranking_correlation()
