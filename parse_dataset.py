import json
from datasets import load_dataset

print("Loading TechQA dataset from Hugging Face...")
dataset = load_dataset("rojagtap/tech-qa", split="train")

formatted_pairs = []

print("Processing records...")
for item in dataset:
    question_text = item.get("question")
    resolution_doc = item.get("document") or item.get("answer")
    
    if question_text and resolution_doc:
        formatted_pairs.append({
            "question": question_text.strip(),
            "resolution_document": resolution_doc.strip()
        })

OUTPUT_PATH = "formatted_dataset.json"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(formatted_pairs, f, indent=2)

print(f"Successfully processed {len(formatted_pairs)} pairs into {OUTPUT_PATH}!")