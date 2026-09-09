import json
import random
import re

SCHEMA_TEMPLATE = """## Symptom
{symptom}

## Likely Cause
{cause}

## Diagnostic Steps
1. Review the issue logs or error codes specified in the question.
2. Confirm the system environment and configuration details.

## Resolution Steps
{resolution}

## Verification
Verify that the resolution steps eliminated the reported symptom and normal operations are restored."""

def reformat_to_schema(question, resolution_doc):
    symptom = question.strip() if question else "Unspecified technical issue reported."
    
    # Simple heuristic: look for cause hints, otherwise state based on context
    cause_match = re.search(r'(due to|caused by|reason:?)(.*?)(?=\.|\n|$)', resolution_doc, re.IGNORECASE)
    if cause_match:
        cause = cause_match.group(0).strip()
    else:
        cause = "Misconfiguration, software bug, or missing environment settings."
        
    # Format resolution steps cleanly
    paragraphs = [p.strip() for p in resolution_doc.split("\n") if p.strip()]
    formatted_steps = []
    for idx, p in enumerate(paragraphs, start=1):
        formatted_steps.append(f"{idx}. {p}")
        
    resolution_str = "\n".join(formatted_steps) if formatted_steps else "1. " + resolution_doc
    
    return SCHEMA_TEMPLATE.format(
        symptom=symptom,
        cause=cause,
        resolution=resolution_str
    )

print("Loading raw merged dataset...")
with open("train_chatml.json", "r", encoding="utf-8") as f:
    chatml_data = json.load(f)

dataset_samples = []

for item in chatml_data:
    messages = item.get("messages", [])
    if len(messages) >= 3:
        user_input = messages[1]["content"]
        raw_output = messages[2]["content"]
        
        formatted_output = reformat_to_schema(user_input, raw_output)
        
        dataset_samples.append({
            "input": user_input,
            "output": formatted_output
        })

print(f"Total reformatted samples: {len(dataset_samples)}")

# Step 4: Split into Train (80%), Val (10%), Test (10%)
random.seed(42)  # For reproducible splitting
random.shuffle(dataset_samples)

total = len(dataset_samples)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_data = dataset_samples[:train_end]
val_data = dataset_samples[train_end:val_end]
test_data = dataset_samples[val_end:]

def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

save_jsonl(train_data, "train.jsonl")
save_jsonl(val_data, "val.jsonl")
save_jsonl(test_data, "test.jsonl")

print(f"Dataset successfully created and split:")
print(f" - train.jsonl: {len(train_data)} samples")
print(f" - val.jsonl:   {len(val_data)} samples")
print(f" - test.jsonl:  {len(test_data)} samples")