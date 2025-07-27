import ast
import os
import re
import pandas as pd
import numpy as np

def count_list(data: list) -> int:
    """
    Count the number of items in a list.
    """
    return len(data)



def string_to_dict(data: str) -> dict:
    """
    Convert a string in dictionary format to a Python dictionary.
    """
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid dictionary string format")
    

def string_to_list(data: str) -> list:
    """
    Convert a string in list format to a Python list.
    """
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
            raise ValueError("Invalid list string format")
    

def get_files_in_directory(directory_path):
    """
    Returns a list of all files in the specified directory.
    """
    try:
        # Get a list of all files in the directory
        files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
        return files
    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def get_model_name(file_base_name):
    """
    Map file base names to benchmark model names.
    """
    mapping = {
        'anthropic-claude-3.7-sonnet-thinking': 'Claude 3.7 Sonnet (Thinking)',
        'anthropic_claude-3.5-haiku': 'Claude 3.5 Haiku',
        'deepseek-deepseek-chat-v3-0324': 'DeepSeek V3',
        'deepseek-deepseek-r1': 'DeepSeek R1',
        'google-gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
        'google-gemini-2.5-pro-preview-03-25': 'Gemini 2.5 Pro',
        'meta-llama-llama-3.1-8b-instruct': 'LLaMA 3.1 8B',
        'meta-llama-llama-4-scout': 'LLaMA 4 Scout',
        'meta-llama_llama-3.2-3b-instruct': 'LLaMA 3.2 3B',
        'mistralai-mistral-small-3.1-24b-instruct': 'Mistral?Small?3.1',
        'mistralai_mistral-7b-instruct': 'Mistral?7B',
        'openai-gpt-4.1-mini': 'GPT-4.1 Mini',
        'openai-gpt-4.1': 'GPT-4.1',
        'openai-o4-mini': 'GPT-4.1 Mini',  # Assuming this is the same as GPT-4.1 Mini
        'qwen-qwen-2.5-7b-instruct': 'Qwen 2.5 7B'
    }
    
    # Return the mapped name or the original if no mapping exists
    return mapping.get(file_base_name, file_base_name)


def costum_parse_exp_2(file_path):
    """Parse the irregularly formatted summarized_results CSV file."""
    # Read the raw file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by model sections (each starting with "The current model is:")
    model_sections = re.split(r'0\s*\n"The current model is:', content)[1:]
    
    # Process each model section
    results = []
    for section in model_sections:
        # Extract model name
        model_name = section.split('\n')[0].strip().strip('"')
        
        # Extract metrics using regex
        consistent_answers = int(re.search(r'Number of consistent answers: (\d+)', section).group(1))
        consistency_rate = float(re.search(r'Consistency rate: ([\d\.]+)', section).group(1))
        
        # Some models might not have overcuteness metrics
        overcuteness_match = re.search(r'Number of over cuteness: (\d+)', section)
        overcuteness = int(overcuteness_match.group(1)) if overcuteness_match else 0
        
        overcuteness_rate_match = re.search(r'Over cuteness rate: ([\d\.]+)', section)
        overcuteness_rate = float(overcuteness_rate_match.group(1)) if overcuteness_rate_match else 0
        
        provision_failure_match = re.search(r'Number of provision failure: (\d+)', section)
        provision_failure = int(provision_failure_match.group(1)) if provision_failure_match else 0
        
        verification_failure_match = re.search(r'Number of verification failure: (\d+)', section)
        verification_failure = int(verification_failure_match.group(1)) if verification_failure_match else 0
        
        # Add to results
        results.append({
            'model': model_name,
            'consistent_answers': consistent_answers,
            'consistency_rate': consistency_rate,
            'overcuteness': overcuteness,
            'overcuteness_rate': overcuteness_rate,
            'provision_failure': provision_failure,
            'verification_failure': verification_failure
        })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def clean_model_name(model_name):
    """
    Clean model names to show only the essential model identifier.
    Examples:
    - "qwen-qwen-2.5-7b-instruct" -> "qwen 2.5 7b"
    - "anthropic-claude-3.5-haiku" -> "claude 3.5 haiku"
    - "google-gemini-2.5-flash-preview" -> "gemini 2.5 flash"
    """
    if pd.isna(model_name) or model_name == "":
        return model_name
   
    # Convert to lowercase for processing
    name = str(model_name).lower()
    name = name.replace('/', '-')
    # Remove common prefixes
    prefixes_to_remove = ['anthropic-', 'google-', 'openai-', 'meta-', 'deepseek-', 'mistralai-', 'qwen-',
                          'anthropic/', 'google/', 'openai/', 'meta/', 'deepseek/', 'mistralai/', 'qwen/']
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    # Remove duplicate company names (e.g., "qwen-qwen-2.5" -> "qwen-2.5")
    parts = name.split('-')
    if len(parts) > 1 and parts[0] == parts[1]:
        parts = parts[1:]
        name = '-'.join(parts)
    
    # Remove common suffixes
    suffixes_to_remove = ['-instruct', '-chat', '-preview', '-thinking', '-001', '-0324', '-03-25']
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    # Replace hyphens with spaces and clean up
    name = name.replace('-', ' ')
    
    # Remove extra words that don't add value
    words_to_remove = ['instruct', 'chat', 'preview', 'thinking']
    words = name.split()
    words = [word for word in words if word not in words_to_remove]
    
    # Join and capitalize appropriately
    cleaned_name = ' '.join(words)
    
    # Special handling for specific models
    if 'claude' in cleaned_name:
        cleaned_name = cleaned_name.replace('claude', 'Claude')
    elif 'gemini' in cleaned_name:
        cleaned_name = cleaned_name.replace('gemini', 'Gemini')
    elif 'gpt' in cleaned_name:
        cleaned_name = cleaned_name.replace('gpt', 'GPT')
    elif 'llama' in cleaned_name:
        cleaned_name = cleaned_name.replace('llama', 'Llama')
    elif 'qwen' in cleaned_name:
        cleaned_name = cleaned_name.replace('qwen', 'Qwen')
    elif 'mistral' in cleaned_name or 'mistralai' in cleaned_name:
        cleaned_name = cleaned_name.title()
    elif 'deepseek' in cleaned_name:
        cleaned_name = cleaned_name.replace('deepseek', 'DeepSeek')
    
    return cleaned_name

def extract_company_name(model_name):
    """Extract company name from model name more reliably"""
    # Convert to lowercase and replace hyphens with slashes for consistency
    name = str(model_name).lower()
    
    # Map of model prefixes to company names
    company_mappings = {
        'openai': 'openai',
        'gpt': 'openai',
        'anthropic': 'anthropic',
        'claude': 'anthropic',
        'google': 'google',
        'gemini': 'google',
        'meta': 'meta',
        'llama': 'meta',
        'mistral': 'mistralai',
        'mistralai': 'mistralai',
        'deepseek': 'deepseek',
        'qwen': 'qwen'
    }
    
    # Check for company name in the model name
    for prefix, company in company_mappings.items():
        if prefix in name:
            return company
            
    # If we can't determine the company, use the first part before any separator
    if '/' in name:
        return name.split('/')[0]
    elif '-' in name:
        return name.split('-')[0]
    
    return name  # Return the whole name if no separator found


def sort_and_group_models(df, model_col='model', benchmark_col='consistency_rate'):
    """
    Sort models by benchmark average and group by company.
    
    Parameters:
    - df: DataFrame containing model data
    - model_col: Name of column containing model names (default: 'model')
    - benchmark_col: Name of column to use as benchmark (default: 'consistency_rate')
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract company name using our improved function
    df_copy['company'] =  df_copy[model_col].apply(extract_company_name)
    
    # Use the specified column as benchmark
    df_copy['benchmark_avg'] = df_copy[benchmark_col]
    
    # Group by company and calculate average benchmark score per company
    company_avg = df_copy.groupby('company')['benchmark_avg'].mean().reset_index()

    # Sort companies by their average benchmark score (ascending)
    company_avg = company_avg.sort_values('benchmark_avg', ascending=True)
    
    # Create ordered list of companies
    company_order = company_avg['company'].tolist()
    
    # Create a new column for company rank (for sorting)
    df_copy['company_rank'] = df_copy['company'].map({comp: i for i, comp in enumerate(company_order)})
    
    # Sort by company rank first, then by benchmark_avg within each company
    sorted_df = df_copy.sort_values(['company_rank', 'benchmark_avg'], ascending=[True, True])
    
    return sorted_df, company_order

def sort_dataframe_by_company_model_order(df, sorted_df, model_col='model'):
    """
    Sort a DataFrame to match the order of models in another sorted DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to be sorted
    sorted_df (pandas.DataFrame): The DataFrame with the desired sort order
    
    Returns:
    pandas.DataFrame: The sorted DataFrame
    """
    # Create a mapping of model names to their position in sorted_df
    model_order = {model: i for i, model in enumerate(sorted_df['model'])}
    print(f"Model order mapping: {model_order}")
    # Create a temporary column with the sort order
    df['sort_index'] = df[model_col].map(model_order)
    print(df.head(3))
    # Sort by this index, then drop the temporary column
    sorted_result = df.sort_values('sort_index').reset_index(drop=True)
    sorted_result = sorted_result.drop('sort_index', axis=1)
    
    return sorted_result