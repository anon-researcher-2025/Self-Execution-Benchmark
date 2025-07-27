import os
import json
from collections import Counter

# Specify the input directory containing JSON files and the output file
input_dir = r""
output_file = "3-5_taboo_words.json"

# Dictionary to store the filtered words
filtered_words = {}
word_count = Counter()

# Iterate over each category file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):  # Process only JSON files
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            for word, taboo_list in data.items():
                # Check if the word is a single word (no spaces) and taboo_list is a list
                if " " not in word and isinstance(taboo_list, list) and len(taboo_list) >= 3:
                    # Ensure between 3-5 words in the taboo list (keep at least 3 and max 5)
                    filtered_words[word] = taboo_list[:max(3, min(5, len(taboo_list)))]
                    word_count[len(filtered_words[word])] += 1

# Save the filtered words to a new JSON file
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(filtered_words, outfile, ensure_ascii=False, indent=2)

# Print the count of words with 3, 4, and 5 taboo words
for num in [3, 4, 5]:
    print(f"Words with {num} taboo words: {word_count[num]}")

print(f"Filtered data saved to {output_file}")



