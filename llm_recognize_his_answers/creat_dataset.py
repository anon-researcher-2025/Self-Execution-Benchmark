from datasets import load_dataset

# Load the dataset
dataset = load_dataset("launch/open_question_type")

# Extract the first 500 questions
questions = dataset['train']['question'][:500]

# Save to a text file
with open("first_500_questions.txt", "w", encoding="utf-8") as f:
    for question in questions:
        f.write(question + "\n")

print("Saved the first 500 questions to first_500_questions.txt")
