import json
import random

with open("training_data_leakage.jsonl") as f:
    examples = [json.loads(line) for line in f]

random.shuffle(examples)
split = int(len(examples) * 0.9)
train = examples[:split]
valid = examples[split:]

# Write to a separate directory so we don't overwrite the original data
import os
os.makedirs("data_leakage", exist_ok=True)

with open("data_leakage/train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

with open("data_leakage/valid.jsonl", "w") as f:
    for ex in valid:
        f.write(json.dumps(ex) + "\n")

print(f"Train: {len(train)}, Valid: {len(valid)}")

# Sanity check: no system prompts
for ex in train[:5]:
    roles = [m["role"] for m in ex["messages"]]
    assert "system" not in roles, f"Found system prompt in training data! Roles: {roles}"
print("Confirmed: no system prompts in training data")
