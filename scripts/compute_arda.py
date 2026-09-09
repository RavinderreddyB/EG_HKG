"""Adaptive Ranking-Confidence Decision Accuracy (ARDA) -- scores the
check_ranking_confidence() flag as a decision under uncertainty: flag or
don't flag. Ground truth comes from a held-out symptom-masking trial: hide
half a disease's known symptoms, query find_diseases() with the rest, and
check whether the true disease actually landed rank 1.

A "correct" adaptive decision is:
  - flagging a case where the true disease was NOT top-ranked (correct warning)
  - not flagging a case where the true disease WAS top-ranked (correct confidence)

Run from the project root: python -m scripts.compute_arda
"""
import json
import random
from pipeline.full_pipeline import MedicalPipeline

SEED = 42
OUTPUT_PATH = "logs/adaptive_retrieval_decision_accuracy.json"


def main():
    random.seed(SEED)
    pipeline = MedicalPipeline()

    with pipeline.driver.session() as session:
        rows = session.run("""
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN d.name AS disease, collect(s.name) AS symptoms, d.source AS source
        """)
        disease_rows = [r.data() for r in rows]

    print(f"Diseases in graph: {len(disease_rows)}")

    trials = []
    for row in disease_rows:
        disease, symptoms, source = row["disease"], row["symptoms"], row["source"]
        if len(symptoms) < 2:
            continue

        shuffled = symptoms[:]
        random.shuffle(shuffled)
        keep_n = max(1, len(shuffled) // 2)
        query_symptoms = shuffled[:keep_n]

        matched, unmatched = pipeline.query_engine.validate_symptoms(query_symptoms)
        if not matched:
            continue
        ranked = pipeline.query_engine.find_diseases(matched)
        if not ranked:
            continue

        rank = next((i + 1 for i, r in enumerate(ranked) if r["disease"] == disease), None)
        hit1 = (rank == 1)
        hit3 = (rank is not None and rank <= 3)

        confidence = pipeline.query_engine.check_ranking_confidence(ranked)
        flagged = confidence["is_low_confidence"]

        trials.append({
            "disease": disease,
            "source": source,
            "query_symptoms": query_symptoms,
            "rank": rank,
            "hit1": hit1,
            "hit3": hit3,
            "flagged_low_confidence": flagged,
            "top_match_count": confidence["top_match_count"],
            "tied_diseases": confidence["tied_diseases"] if flagged else [],
        })

    pipeline.close()

    n = len(trials)
    hit1_rate = sum(t["hit1"] for t in trials) / n
    hit3_rate = sum(t["hit3"] for t in trials) / n
    flag_rate = sum(t["flagged_low_confidence"] for t in trials) / n

    tp = sum(1 for t in trials if t["flagged_low_confidence"] and not t["hit1"])
    fp = sum(1 for t in trials if t["flagged_low_confidence"] and t["hit1"])
    tn = sum(1 for t in trials if not t["flagged_low_confidence"] and t["hit1"])
    fn = sum(1 for t in trials if not t["flagged_low_confidence"] and not t["hit1"])

    arda_accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\nn trials: {n}")
    print(f"Hit@1: {hit1_rate:.4f}  Hit@3: {hit3_rate:.4f}  flag rate: {flag_rate:.4f}")
    print("Confusion matrix (flag decision vs. actual top-1 correctness):")
    print(f"  TP (correctly warned)    = {tp}")
    print(f"  FP (false alarm)         = {fp}")
    print(f"  TN (correctly confident) = {tn}")
    print(f"  FN (missed warning)      = {fn}")
    print(f"\nARDA accuracy : {arda_accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1             : {f1:.4f}")

    report = {
        "n_trials": n,
        "hit1_rate": hit1_rate,
        "hit3_rate": hit3_rate,
        "flag_rate": flag_rate,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "arda_accuracy": arda_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trials": trials,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
