import json
import re
from datasets import load_dataset

def clean_html(text):
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    return clean.strip()

print("Downloading ServerFault subset (~20 MB)...")
# Load the serverfault.com configuration specifically
dataset = load_dataset("HuggingFaceTB/stackexchange_2025_md", "serverfault.com", split="train")

formatted_pairs = []

print("Extracting QA pairs...")
for item in dataset:
    question = clean_html(item.get("Title", "") or item.get("Body", ""))
    
    # Extract the accepted answer or first available answer
    answers = item.get("Answers", [])
    answer = ""
    if isinstance(answers, list) and len(answers) > 0:
        answer = clean_html(answers[0].get("Body", ""))
    elif isinstance(answers, str):
        answer = clean_html(answers)
        
    if len(answer) >= 150 and len(question) > 0:
        formatted_pairs.append({
            "question": question,
            "resolution_document": answer
        })
        
    if len(formatted_pairs) >= 1500:
        break

OUTPUT_PATH = "serverfault_dataset.json"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(formatted_pairs, f, indent=2)

print(f"Done! Successfully saved {len(formatted_pairs)} ServerFault pairs to {OUTPUT_PATH}.")