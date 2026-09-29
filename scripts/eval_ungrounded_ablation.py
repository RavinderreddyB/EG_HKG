"""KG-grounded vs ungrounded (no-KG) generation ablation.
For every disease already scored by scripts/eval_faithfulness_final.py
(the KG-grounded condition), generate a second answer with
MedicalGenerator.generate_ungrounded() -- pure parametric Gemini, no
evidence/citations/drugs -- and score it with the SAME NLI scorer against
the SAME real KG evidence pieces used to score the grounded answer. This
is the standard RAG-ablation design: same gold evidence, different
generation-time grounding, isolating what the KG actually contributes.

Run from the project root: python -m scripts.eval_ungrounded_ablation
"""
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from evaluation.faithfulness_nli import FaithfulnessScorer
from pipeline.generator import MedicalGenerator

load_dotenv()

RUNS_DIR = Path("logs/pipeline_runs")
RUN_GLOB = "run_20260826_*.json"
SEED_RANGE = ("123116", "123707")  # same 34-batch (reordered/fixed MODEL_FALLBACKS) used for the grounded eval

OUTPUT_PATH = Path("logs/ungrounded_ablation_report.json")
CHECKPOINT_PATH = Path("logs/ungrounded_ablation_checkpoint.json")

MAX_SENTENCE_CHARS = 800


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

    generator = MedicalGenerator(api_key=os.getenv("GEMINI_API_KEY"))
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

            evidence_pieces = build_evidence_pieces(d)
            if not evidence_pieces or not d.get("answer"):
                print(f"  [skip] {path.name} | {d['disease']} -- no grounded answer/evidence", flush=True)
                continue

            print(f"  -- generating ungrounded | {d['disease']}", flush=True)
            ungrounded_answer = None
            ungrounded_model = None
            last_error = None
            for attempt in range(3):
                try:
                    ungrounded_answer, ungrounded_model = generator.generate_ungrounded(d["disease"])
                    break
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        print(f"     [WARN] Retrying ({attempt + 1}/2)... {str(e)[:100]}", flush=True)
                        time.sleep(3)
            if ungrounded_answer is None:
                print(f"     [ERROR generation] {str(last_error)[:150]}", flush=True)
                continue

            capped = ". ".join(s[:MAX_SENTENCE_CHARS] for s in ungrounded_answer.split(". "))

            try:
                scores = scorer.score_answer(evidence_pieces, capped)
            except Exception as e:
                print(f"     [ERROR scoring] {str(e)[:150]}", flush=True)
                continue

            scores["run"] = path.name
            scores["disease"] = d["disease"]
            scores["evidence_source"] = d.get("evidence_source")
            scores["ungrounded_answer"] = ungrounded_answer
            scores["ungrounded_model"] = ungrounded_model
            scores["grounded_model"] = d.get("generation_model")
            results.append(scores)
            done_keys.add(key)
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"     done | avg_consistency {scores['avg_consistency']:.2f} | "
                  f"low_rate {scores['low_consistency_rate']:.2f} | "
                  f"grounded_model={scores['grounded_model']} ungrounded_model={scores['ungrounded_model']}", flush=True)

    n = len(results)
    if n == 0:
        print("No scoreable disease records found.")
        return

    avg_consistency = sum(r["avg_consistency"] for r in results) / n
    avg_low_rate = sum(r["low_consistency_rate"] for r in results) / n

    by_source = {}
    for r in results:
        src = r.get("evidence_source") or "unknown"
        by_source.setdefault(src, []).append(r)

    grounded_models = {r.get("grounded_model") for r in results}
    ungrounded_models = {r.get("ungrounded_model") for r in results}

    print(f"\n[OK] Scored {n} ungrounded disease answers")
    print(f"  Avg consistency (faithfulness, ungrounded)   : {avg_consistency:.4f}")
    print(f"  Avg low-consistency rate (hallucination, ungrounded): {avg_low_rate:.4f}")
    print(f"\n  Distinct grounded-condition models seen: {grounded_models}")
    print(f"  Distinct ungrounded-condition models seen: {ungrounded_models}")
    if len(grounded_models) > 1 or len(ungrounded_models) > 1:
        print("  [WARN] More than one model appeared within a condition -- "
              "aggregate numbers are a blend across models, see per-model breakdown in the saved report.")
    else:
        print("  [OK] Every answer in each condition came from exactly one model -- no fallback-model confound.")

    report = {
        "per_disease": results,
        "avg_consistency": avg_consistency,
        "avg_low_consistency_rate": avg_low_rate,
        "num_scored": n,
        "by_source": {},
        "grounded_models_seen": sorted(str(m) for m in grounded_models),
        "ungrounded_models_seen": sorted(str(m) for m in ungrounded_models),
    }
    print("\nBy evidence source:")
    for src, items in by_source.items():
        m = len(items)
        a = sum(r["avg_consistency"] for r in items) / m
        lr = sum(r["low_consistency_rate"] for r in items) / m
        print(f"  {src:20s} n={m:3d}  avg_consistency={a:.4f}  low_rate={lr:.4f}")
        report["by_source"][src] = {"n": m, "avg_consistency": a, "avg_low_consistency_rate": lr}

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
