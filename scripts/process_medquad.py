import json
import os

INPUT_DIR = "data/raw/medquad"
OUTPUT_FILE = "data/processed/medquad.json"

os.makedirs("data/processed", exist_ok=True)

dataset = []

# -----------------------------
# Process files
# -----------------------------
for filename in os.listdir(INPUT_DIR):

    filepath = os.path.join(INPUT_DIR, filename)

    # -------------------------
    # JSONL FILE (your case)
    # -------------------------
    if filename.endswith(".jsonl"):

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)

                    q = item.get("question")
                    a = item.get("answer")

                    if q and a:
                        dataset.append({
                            "question": q.strip(),
                            "answer": a.strip()
                        })

                except:
                    continue

    # -------------------------
    # JSON FILE
    # -------------------------
    elif filename.endswith(".json"):

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                q = item.get("question")
                a = item.get("answer")

                if q and a:
                    dataset.append({
                        "question": q.strip(),
                        "answer": a.strip()
                    })

# -----------------------------
# Remove duplicates
# -----------------------------
unique = []
seen = set()

for item in dataset:
    key = item["question"].lower()

    if key not in seen:
        seen.add(key)
        unique.append(item)

# -----------------------------
# Save
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2)

print(f"\n[OK] Processed {len(unique)} QA pairs")
print(f"Saved to {OUTPUT_FILE}")