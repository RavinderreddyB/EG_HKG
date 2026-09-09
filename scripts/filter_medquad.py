import json

INPUT = "data/processed/medquad.json"
OUTPUT = "data/processed/medquad_filtered.json"

KEYWORDS = [
    "treatment",
    "therapy",
    "drug",
    "medication",
    "management",
    "antibiotic",
    "how to treat"
]

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = []

for item in data:
    q = item["question"].lower()

    if any(k in q for k in KEYWORDS):
        filtered.append(item)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(f"\n[OK] Filtered dataset size: {len(filtered)}")