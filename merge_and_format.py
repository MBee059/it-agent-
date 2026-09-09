import json

SYSTEM_PROMPT = "You are an expert IT support agent. Help the user resolve their technical issue using clear, accurate instructions."

def load_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found.")
        return []

print("Loading datasets...")
techqa_data = load_json("formatted_dataset.json")
serverfault_data = load_json("serverfault_dataset.json")

combined_data = techqa_data + serverfault_data
print(f"Loaded {len(techqa_data)} TechQA pairs and {len(serverfault_data)} ServerFault pairs.")
print(f"Total raw samples: {len(combined_data)}")

chatml_dataset = []

for item in combined_data:
    q = item.get("question", "").strip()
    a = item.get("resolution_document", "").strip()
    
    if q and a:
        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        }
        chatml_dataset.append(entry)

OUTPUT_FILE = "train_chatml.json"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chatml_dataset, f, indent=2)

print(f"Successfully formatted {len(chatml_dataset)} ChatML records into {OUTPUT_FILE}!")
