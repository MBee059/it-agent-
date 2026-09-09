import json

INPUT_PATH = "formatted_dataset.json"
OUTPUT_PATH = "train_chatml.json"

print("Loading dataset...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_chatml = []

SYSTEM_PROMPT = "You are an expert IT support agent. Help the user resolve their technical issue using clear, accurate instructions."

for item in data:
    chat_entry = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["resolution_document"]}
        ]
    }
    formatted_chatml.append(chat_entry)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(formatted_chatml, f, indent=2)

print(f"Successfully converted {len(formatted_chatml)} samples to ChatML format in {OUTPUT_PATH}!")