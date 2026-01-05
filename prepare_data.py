import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw_dataset.csv")

# Keep only required columns
df = df[['question', 'answer']]

# Clean dataset
df = df.dropna()
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)

# Convert to GPT training format
with open("data/education_dataset.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f"Q: {row['question']} A: {row['answer']}\n")

print("Dataset prepared successfully.")
