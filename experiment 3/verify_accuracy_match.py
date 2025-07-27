import pandas as pd

def compare_accuracy_results():
    """
    השוואה בין תוצאות ה-accuracy מקבצים שונים
    """
    
    # טעינת הקובץ שחישבתי
    my_results = pd.read_csv('model_accuracy_results.csv')
    
    # יצירת הקובץ החדש מהנתונים שהבאת
    new_data = """model_name,failed_to_order,correct_individual_answers_count,avg_question_order_score,adjusted_avg_question_order_score,group_individual_duration_avg,group_individual_duration_std,group_individual_duration_std_percentage,group_individual_duration_std_percentage_median,combined_duration_avg,combined_duration_std
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
    
    # שמירת הנתונים החדשים לקובץ זמני
    with open('temp_new_results.csv', 'w') as f:
        f.write(new_data)
    
    # טעינת הקובץ החדש
    new_results = pd.read_csv('temp_new_results.csv')
    
    print("=== השוואת תוצאות Accuracy ===\n")
    
    # הכנת טבלת השוואה
    comparison_data = []
    
    for _, new_row in new_results.iterrows():
        model_name = new_row['model_name']
        new_correct = new_row['correct_individual_answers_count']
        
        # חיפוש המודל בתוצאות שלי
        my_row = my_results[my_results['full_model_name'] == model_name]
        
        if len(my_row) > 0:
            my_correct = my_row['correct_answers'].iloc[0]
            my_total = my_row['total_questions'].iloc[0]
            
            # חישוב accuracy מהנתונים החדשים (נניח שהסך הכולל זהה)
            new_accuracy = (new_correct / my_total) * 100 if my_total > 0 else 0
            my_accuracy = my_row['accuracy_percentage'].iloc[0]
            
            difference = my_correct - new_correct
            match = "✓" if difference == 0 else "✗"
            
            comparison_data.append({
                'model': model_name.split('/')[-1],  # רק שם המודל
                'my_correct': my_correct,
                'new_correct': new_correct,
                'difference': difference,
                'my_accuracy': f"{my_accuracy:.1f}%",
                'new_accuracy': f"{new_accuracy:.1f}%",
                'match': match
            })
        else:
            comparison_data.append({
                'model': model_name.split('/')[-1],
                'my_correct': 'N/A',
                'new_correct': new_correct,
                'difference': 'N/A',
                'my_accuracy': 'N/A',
                'new_accuracy': 'N/A',
                'match': '?'
            })
    
    # הצגת טבלת השוואה
    comparison_df = pd.DataFrame(comparison_data)
    
    print(f"{'Model':<30} {'My Count':<10} {'New Count':<10} {'Diff':<6} {'My %':<8} {'New %':<8} {'Match':<5}")
    print("-" * 85)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['model']:<30} {row['my_correct']:<10} {row['new_correct']:<10} {row['difference']:<6} {row['my_accuracy']:<8} {row['new_accuracy']:<8} {row['match']:<5}")
    
    # סיכום
    matches = comparison_df[comparison_df['match'] == '✓']
    mismatches = comparison_df[comparison_df['match'] == '✗']
    
    print(f"\n=== סיכום ===")
    print(f"התאמות מושלמות: {len(matches)}/{len(comparison_df)}")
    print(f"אי-התאמות: {len(mismatches)}")
    
    if len(mismatches) > 0:
        print(f"\nמודלים עם אי-התאמות:")
        for _, row in mismatches.iterrows():
            print(f"  {row['model']}: הפרש של {row['difference']} תשובות")
    
    # בדיקת מודלים חסרים
    my_models = set(my_results['full_model_name'])
    new_models = set(new_results['model_name'])
    
    missing_in_new = my_models - new_models
    missing_in_my = new_models - my_models
    
    if missing_in_new:
        print(f"\nמודלים שיש לי אבל לא בנתונים החדשים:")
        for model in missing_in_new:
            print(f"  {model}")
    
    if missing_in_my:
        print(f"\nמודלים שיש בנתונים החדשים אבל לא אצלי:")
        for model in missing_in_my:
            print(f"  {model}")
    
    # ניקוי קובץ זמני
    import os
    os.remove('temp_new_results.csv')
    
    return comparison_df

if __name__ == "__main__":
    comparison_df = compare_accuracy_results()
