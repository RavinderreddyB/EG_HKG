"""Unified, multi-seed held-out symptom-masking evaluation. Replaces two
previously disconnected single-run evaluations (an older, non-persisted
Table 6 script from an earlier session, and compute_arda.py's single
seed=42 run) that were incorrectly described as "consistent" despite
being two different single trials of a randomized split -- see Section
6.4/6.7 of OBJECTIVE_I_RESEARCH_PAPER.md for the discrepancy this fixes
(94.40% vs 96.80% Hit@1) and why a single seed cannot be trusted alone.

For each of N seeds: for every disease, half its known symptoms are kept
as query input (seeded shuffle+split), find_diseases() is queried, and
both the disease-ranking metrics (Hit@K, MRR, rank) AND the
check_ranking_confidence() flag decision (ARDA) are scored against the
same trial. Reports mean +/- population stdev across all N seeds for
every metric, not a single cherry-pickable number.

Run from the project root: python -m scripts.compute_holdout_eval
"""
import json
import math
import random
import statistics
from pipeline.full_pipeline import MedicalPipeline

N_SEEDS = 20
SEEDS = list(range(1, N_SEEDS + 1))  # 1..20, not reusing the old seed=42 specifically
OUTPUT_PATH = "logs/holdout_eval_multiseed_report.json"


def run_one_seed(pipeline, disease_rows, seed):
    rng = random.Random(seed)

    trials = []
    for row in disease_rows:
        disease, symptoms = row["disease"], row["symptoms"]
        if len(symptoms) < 2:
            continue

        shuffled = symptoms[:]
        rng.shuffle(shuffled)
        keep_n = max(1, len(shuffled) // 2)
        query_symptoms = shuffled[:keep_n]

        matched, _ = pipeline.query_engine.validate_symptoms(query_symptoms)
        if not matched:
            continue
        ranked = pipeline.query_engine.find_diseases(matched)
        if not ranked:
            continue

        rank = next((i + 1 for i, r in enumerate(ranked) if r["disease"] == disease), None)
        confidence = pipeline.query_engine.check_ranking_confidence(ranked)

        trials.append({
            "disease": disease,
            "rank": rank,
            "flagged": confidence["is_low_confidence"],
        })

    n = len(trials)
    if n == 0:
        return None

    def hit_at(k):
        return sum(1 for t in trials if t["rank"] is not None and t["rank"] <= k) / n

    ranks = [t["rank"] for t in trials if t["rank"] is not None]
    mrr = sum(1.0 / r for r in ranks) / n if ranks else 0.0
    ndcg10 = sum((1.0 / math.log2(r + 1)) for r in ranks if r <= 10) / n
    mean_rank = statistics.mean(ranks) if ranks else None
    median_rank = statistics.median(ranks) if ranks else None
    not_recovered_top10 = sum(1 for t in trials if t["rank"] is None or t["rank"] > 10)

    tp = sum(1 for t in trials if t["flagged"] and t["rank"] != 1)
    fp = sum(1 for t in trials if t["flagged"] and t["rank"] == 1)
    tn = sum(1 for t in trials if not t["flagged"] and t["rank"] == 1)
    fn = sum(1 for t in trials if not t["flagged"] and t["rank"] != 1)

    arda_accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "seed": seed, "n": n,
        "hit1": hit_at(1), "hit3": hit_at(3), "hit5": hit_at(5), "hit10": hit_at(10),
        "mrr": mrr, "ndcg10": ndcg10,
        "mean_rank": mean_rank, "median_rank": median_rank,
        "not_recovered_top10": not_recovered_top10,
        "flag_rate": sum(t["flagged"] for t in trials) / n,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "arda_accuracy": arda_accuracy, "precision": precision, "recall": recall, "f1": f1,
    }


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def main():
    pipeline = MedicalPipeline()

    with pipeline.driver.session() as session:
        rows = session.run("""
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN d.name AS disease, collect(s.name) AS symptoms, d.source AS source
        """)
        disease_rows = [r.data() for r in rows]
        source_by_disease = {r["disease"]: r["source"] for r in disease_rows}

    print(f"Diseases in graph: {len(disease_rows)}. Running {N_SEEDS} seeds...", flush=True)

    per_seed = []
    by_source_hit1_per_seed = {"DiseaseAndSymptoms": [], "MedlinePlus": []}
    for seed in SEEDS:
        result = run_one_seed(pipeline, disease_rows, seed)
        if result:
            per_seed.append(result)
            print(f"  seed {seed:3d} | n={result['n']} | Hit@1={result['hit1']:.4f} | "
                  f"Hit@3={result['hit3']:.4f} | ARDA_acc={result['arda_accuracy']:.4f} | "
                  f"flags={result['confusion_matrix']}", flush=True)

            rng = random.Random(seed)
            seed_by_source = {"DiseaseAndSymptoms": [], "MedlinePlus": []}
            for row in disease_rows:
                disease, symptoms = row["disease"], row["symptoms"]
                if len(symptoms) < 2:
                    continue
                shuffled = symptoms[:]
                rng.shuffle(shuffled)
                keep_n = max(1, len(shuffled) // 2)
                matched, _ = pipeline.query_engine.validate_symptoms(shuffled[:keep_n])
                if not matched:
                    continue
                ranked = pipeline.query_engine.find_diseases(matched)
                if not ranked:
                    continue
                rank = next((i + 1 for i, r in enumerate(ranked) if r["disease"] == disease), None)
                src = source_by_disease.get(disease, "unknown")
                if src in seed_by_source:
                    seed_by_source[src].append(1.0 if rank == 1 else 0.0)
            for src in by_source_hit1_per_seed:
                if seed_by_source[src]:
                    by_source_hit1_per_seed[src].append(sum(seed_by_source[src]) / len(seed_by_source[src]))

    pipeline.close()

    metrics = ["hit1", "hit3", "hit5", "hit10", "mrr", "ndcg10", "mean_rank", "median_rank",
               "flag_rate", "arda_accuracy", "precision", "recall", "f1"]
    summary = {}
    for m in metrics:
        mean, sd = mean_sd([s[m] for s in per_seed])
        summary[m] = {"mean": mean, "sd": sd}

    total_tp = sum(s["confusion_matrix"]["tp"] for s in per_seed)
    total_fp = sum(s["confusion_matrix"]["fp"] for s in per_seed)
    total_tn = sum(s["confusion_matrix"]["tn"] for s in per_seed)
    total_fn = sum(s["confusion_matrix"]["fn"] for s in per_seed)

    print(f"\n=== Aggregate across {len(per_seed)} seeds (mean +/- SD) ===")
    for m in metrics:
        mean, sd = summary[m]["mean"], summary[m]["sd"]
        if mean is None:
            continue
        print(f"  {m:15s}: {mean:.4f} +/- {sd:.4f}")
    print(f"\n  Pooled confusion matrix across all seeds: TP={total_tp} FP={total_fp} TN={total_tn} FN={total_fn}")

    by_source_summary = {}
    for src, vals in by_source_hit1_per_seed.items():
        if vals:
            m, sd = mean_sd(vals)
            by_source_summary[src] = {"mean_hit1": m, "sd_hit1": sd, "n_seeds": len(vals)}
            print(f"  {src:20s} Hit@1 mean+/-SD across {len(vals)} seeds: {m:.4f} +/- {sd:.4f}")

    report = {
        "n_seeds": len(per_seed),
        "seeds": SEEDS,
        "per_seed": per_seed,
        "summary_mean_sd": summary,
        "pooled_confusion_matrix": {"tp": total_tp, "fp": total_fp, "tn": total_tn, "fn": total_fn},
        "by_source_hit1": by_source_summary,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
