"""Faithfulness / hallucination evaluation -- current, correct methodology.
Includes drugs + treatment_modalities as evidence (fixes the gap that
under-scored Tier-2 MedlinePlus diseases), and uses real MedlinePlus
per-section evidence (via full_pipeline.py's get_medlineplus_evidence)
rather than the unreliable MedQuAD RAG fallback for those diseases.
Run from the project root: python -m scripts.eval_faithfulness_final
Points at whatever batch of run_*.json files you generate with
scripts/generate_batch_runs.py -- edit RUN_GLOB below to match your run.
"""
import json
import glob
from pathlib import Path
from evaluation.faithfulness_nli import FaithfulnessScorer

RUNS_DIR = Path("logs/pipeline_runs")
RUN_GLOB = "run_20260826_*.json"  # narrow this to a specific batch's timestamp range if needed
SEED_RANGE = ("123116", "123707")  # the 34-batch (seed 777) regenerated with reordered/fixed MODEL_FALLBACKS
OUTPUT_PATH = Path("logs/faithfulness_nli_final_report.json")
CHECKPOINT_PATH = Path("logs/faithfulness_nli_final_checkpoint.json")

MAX_SENTENCE_CHARS = 800  # defensive cap mirroring MAX_EVIDENCE_PIECE_CHARS


def build_evidence_pieces(disease_record):
    parts = []
    for c in disease_record.get("citations", []):
        snippet = c.get("abstract_snippet")
        if snippet:
            parts.append(snippet)
        elif c.get("title"):
            parts.append(c["title"])
    for e in disease_record.get("trusted_evidence", []):
        if e.get("text"):
            parts.append(e["text"])
    for drug in disease_record.get("drugs", []):
        parts.append(f"Treatment: {drug}")
    for modality in disease_record.get("treatment_modalities", []):
        parts.append(f"Treatment approach: {modality}")
    return parts


def main():
    run_files = sorted([
        p for p in RUNS_DIR.glob(RUN_GLOB)
        if SEED_RANGE[0] <= p.stem.split("_")[-1] <= SEED_RANGE[1]
    ])
    print(f"Isolated {len(run_files)} run files", flush=True)

    scorer = FaithfulnessScorer()

    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            results = json.load(f)
        print(f"Resuming from checkpoint: {len(results)} already scored", flush=True)
    else:
        results = []

    done_keys = {(r["run"], r["disease"]) for r in results}

    for path in run_files:
        with open(path) as f:
            run = json.load(f)

        for d in run.get("diseases", []):
            key = (path.name, d["disease"])
            if key in done_keys:
                continue

            print(f"  -- starting {path.name} | {d['disease']}", flush=True)
            answer = d.get("answer")
            if not answer:
                print("     [skip] no answer", flush=True)
                continue
            evidence_pieces = build_evidence_pieces(d)
            if not evidence_pieces:
                print("     [skip] no evidence", flush=True)
                continue

            capped_answer = ". ".join(s[:MAX_SENTENCE_CHARS] for s in answer.split(". "))

            try:
                scores = scorer.score_answer(evidence_pieces, capped_answer)
            except Exception as e:
                print(f"     [ERROR] {str(e)[:150]}", flush=True)
                continue

            scores["run"] = path.name
            scores["disease"] = d["disease"]
            scores["evidence_source"] = d.get("evidence_source")
            scores["generation_model"] = d.get("generation_model")
            results.append(scores)
            done_keys.add(key)
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"     done | avg_consistency {scores['avg_consistency']:.2f} | "
                  f"low_rate {scores['low_consistency_rate']:.2f} | claims {scores['num_claims']} | "
                  f"src {scores['evidence_source']}", flush=True)

    n = len(results)
    if n == 0:
        print("No scoreable disease records found.")
        return

    avg_consistency = sum(r["avg_consistency"] for r in results) / n
    avg_min_consistency = sum(r["min_consistency"] for r in results) / n
    avg_low_rate = sum(r["low_consistency_rate"] for r in results) / n

    by_source = {}
    for r in results:
        src = r.get("evidence_source") or "unknown"
        by_source.setdefault(src, []).append(r)

    by_model = {}
    for r in results:
        m = r.get("generation_model") or "unknown"
        by_model.setdefault(m, []).append(r)

    print(f"\n[OK] Scored {n} disease answers across {len(run_files)} runs")
    print(f"  Avg consistency (faithfulness)          : {avg_consistency:.4f}")
    print(f"  Avg min consistency (weakest claim)     : {avg_min_consistency:.4f}")
    print(f"  Avg low-consistency rate (hallucination) : {avg_low_rate:.4f}")

    report = {
        "per_disease": results,
        "avg_consistency": avg_consistency,
        "avg_min_consistency": avg_min_consistency,
        "avg_low_consistency_rate": avg_low_rate,
        "num_scored": n,
        "by_source": {},
        "by_generation_model": {},
    }

    print("\nBy evidence source:")
    for src, items in by_source.items():
        m = len(items)
        a = sum(r["avg_consistency"] for r in items) / m
        lr = sum(r["low_consistency_rate"] for r in items) / m
        print(f"  {src:20s} n={m:3d}  avg_consistency={a:.4f}  low_rate={lr:.4f}")
        report["by_source"][src] = {"n": m, "avg_consistency": a, "avg_low_consistency_rate": lr}

    print("\nBy generation model:")
    for model_name, items in by_model.items():
        m = len(items)
        a = sum(r["avg_consistency"] for r in items) / m
        lr = sum(r["low_consistency_rate"] for r in items) / m
        print(f"  {model_name:25s} n={m:3d}  avg_consistency={a:.4f}  low_rate={lr:.4f}")
        report["by_generation_model"][model_name] = {"n": m, "avg_consistency": a, "avg_low_consistency_rate": lr}

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
