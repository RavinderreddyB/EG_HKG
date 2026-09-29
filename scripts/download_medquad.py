from datasets import load_dataset
import json
from pathlib import Path

Path("data/raw/medquad").mkdir(parents=True, exist_ok=True)

dataset = load_dataset("lavita/MedQuAD")

with open("data/raw/medquad/medquad.jsonl", "w") as f:
    for item in dataset["train"]:
        record = {
            "question": item["question"],
            "answer": item["answer"]
        }
        f.write(json.dumps(record) + "\n")

print("[OK] MedQuAD saved")