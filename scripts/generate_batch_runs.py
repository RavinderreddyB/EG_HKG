import sys
import json
import random
from pathlib import Path
from neo4j import GraphDatabase

from pipeline.full_pipeline import MedicalPipeline
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

PROGRESS_PATH = Path("logs/batch_runs_progress.json")


def fetch_symptoms(driver):
    with driver.session() as session:
        return [r["name"] for r in session.run("MATCH (s:Symptom) RETURN s.name AS name")]


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed": 0, "errors": []}


def save_progress(progress):
    Path("logs").mkdir(exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    symptoms_pool = fetch_symptoms(driver)
    driver.close()
    print(f"Symptom pool size: {len(symptoms_pool)}")

    random.seed(seed)
    all_combos = [random.sample(symptoms_pool, 3) for _ in range(n)]

    progress = load_progress()
    start = progress["completed"]
    print(f"Target runs: {n} | Resuming from iteration {start}")

    pipeline = MedicalPipeline()

    for i in range(start, n):
        symptoms = all_combos[i]
        print(f"\n===== BATCH [{i + 1}/{n}] symptoms={symptoms} =====")
        try:
            pipeline.run(symptoms)
        except Exception as e:
            print(f"[ERROR] iteration {i} failed: {e}")
            progress["errors"].append({"iteration": i, "symptoms": symptoms, "error": str(e)[:300]})

        progress["completed"] = i + 1
        save_progress(progress)

    pipeline.close()
    print(f"\n[OK] Completed {progress['completed']}/{n} runs. Errors: {len(progress['errors'])}")


if __name__ == "__main__":
    main()
