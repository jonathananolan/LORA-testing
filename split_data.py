# save as split_data.py
import json
import random

with open("training_data.jsonl") as f:
    examples = [json.loads(line) for line in f]

random.shuffle(examples)
split = int(len(examples) * 0.9)
train = examples[:split]
valid = examples[split:]

with open("data/train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

with open("data/valid.jsonl", "w") as f:
    for ex in valid:
        f.write(json.dumps(ex) + "\n")

print(f"Train: {len(train)}, Valid: {len(valid)}")