import json, os
from datasets import load_dataset

os.makedirs("hellaswag", exist_ok=True)
ds = load_dataset("Rowan/hellaswag", split="validation")

with open("hellaswag/hellaswag_val.jsonl", "w") as f:
    for ex in ds:
        f.write(json.dumps({
            "ctx": ex["ctx"],
            "endings": ex["endings"],
            "label": int(ex["label"]),   # HF stores labels as strings like "3"
        }) + "\n")

print(f"wrote {len(ds)} examples")