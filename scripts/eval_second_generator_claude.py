"""Second-generator-family faithfulness comparison (supervisor comment 6):
reuses the exact evidence-assembly logic already validated for the Gemini
grounded/ungrounded ablation (Section 5.8/6.2), but generates both arms
with Claude Sonnet 5 instead -- same prompts (build_grounded_prompt /
build_ungrounded_prompt from pipeline/generator.py), same evidence, same
HHEM scorer, same paired-McNemar's methodology already run for Gemini.

Source data: logs/evidence_only_runs/*.json (scripts/generate_batch_evidence_only.py)
n=34 queries, 102 disease-answer slots -- same protocol as the Gemini run.

Run from the project root: python -m scripts.eval_second_generator_claude
"""
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic
from statsmodels.stats.contingency_tables import mcnemar

load_dotenv()

from pipeline.generator import build_grounded_prompt, build_ungrounded_prompt
from evaluation.faithfulness_nli import FaithfulnessScorer, LOW_CONSISTENCY_THRESHOLD

EVIDENCE_DIR = Path("logs/evidence_only_runs")
OUTPUT_PATH = Path("logs/second_generator_claude_report.json")
CHECKPOINT_PATH = Path("logs/second_generator_claude_checkpoint.json")

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
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


def cap(text):
    return ". ".join(s[:MAX_SENTENCE_CHARS] for s in text.split(". "))


def generate(client, prompt, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if b.type == "text")
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(3)
    raise last_error


def main():
    client = Anthropic()
    scorer = FaithfulnessScorer()

    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} already done", flush=True)
    else:
        results = []
    done_keys = {(r["file"], r["disease"]) for r in results}

    files = sorted(EVIDENCE_DIR.glob("evidence_*.json"))
    print(f"{len(files)} evidence files", flush=True)

    for path in files:
        with open(path) as f:
            record = json.load(f)

        for d in record["diseases"]:
            key = (path.name, d["disease"])
            if key in done_keys:
                continue

            evidence_pieces = build_evidence_pieces(d)
            if not evidence_pieces:
                print(f"  [skip] {d['disease']} -- no evidence", flush=True)
                continue

            print(f"  -- {path.name} | {d['disease']}", flush=True)

            grounded_prompt = build_grounded_prompt(
                d["disease"], d["drugs"], d["ranked_evidence"], d["citations"],
                treatment_modalities=d["treatment_modalities"],
            )
            ungrounded_prompt = build_ungrounded_prompt(d["disease"])

            try:
                grounded_answer = generate(client, grounded_prompt)
                ungrounded_answer = generate(client, ungrounded_prompt)
            except Exception as e:
                print(f"     [ERROR generation] {str(e)[:150]}", flush=True)
                continue

            try:
                g_scores = scorer.score_answer(evidence_pieces, cap(grounded_answer))
                u_scores = scorer.score_answer(evidence_pieces, cap(ungrounded_answer))
            except Exception as e:
                print(f"     [ERROR scoring] {str(e)[:150]}", flush=True)
                continue

            entry = {
                "file": path.name,
                "disease": d["disease"],
                "evidence_source": d.get("evidence_source"),
                "grounded_answer": grounded_answer,
                "ungrounded_answer": ungrounded_answer,
                "grounded_avg_consistency": g_scores["avg_consistency"],
                "grounded_low_consistency_rate": g_scores["low_consistency_rate"],
                "ungrounded_avg_consistency": u_scores["avg_consistency"],
                "ungrounded_low_consistency_rate": u_scores["low_consistency_rate"],
                "model": MODEL,
            }
            results.append(entry)
            done_keys.add(key)
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"     grounded={g_scores['avg_consistency']:.3f}  ungrounded={u_scores['avg_consistency']:.3f}", flush=True)

    n = len(results)
    if n == 0:
        print("No results.")
        return

    g_mean = sum(r["grounded_avg_consistency"] for r in results) / n
    u_mean = sum(r["ungrounded_avg_consistency"] for r in results) / n
    g_hall = sum(r["grounded_low_consistency_rate"] for r in results) / n
    u_hall = sum(r["ungrounded_low_consistency_rate"] for r in results) / n

    both_fail = both_pass = g_fail_only = u_fail_only = 0
    for r in results:
        g_fail = r["grounded_avg_consistency"] < LOW_CONSISTENCY_THRESHOLD
        u_fail = r["ungrounded_avg_consistency"] < LOW_CONSISTENCY_THRESHOLD
        if g_fail and u_fail:
            both_fail += 1
        elif not g_fail and not u_fail:
            both_pass += 1
        elif g_fail and not u_fail:
            g_fail_only += 1
        else:
            u_fail_only += 1

    table = [[both_fail, g_fail_only], [u_fail_only, both_pass]]
    mc = mcnemar(table, exact=True)

    print(f"\n=== Claude Sonnet 5 (n={n}) ===")
    print(f"Grounded   mean_consistency={g_mean:.4f}  hallucination_rate={g_hall*100:.2f}%")
    print(f"Ungrounded mean_consistency={u_mean:.4f}  hallucination_rate={u_hall*100:.2f}%")
    print(f"McNemar exact test: statistic={mc.statistic}, p={mc.pvalue:.2e}")
    print(f"Discordant pairs: ungrounded-fails-only={u_fail_only}, grounded-fails-only={g_fail_only}")

    report = {
        "model": MODEL,
        "n": n,
        "grounded_mean_consistency": g_mean,
        "ungrounded_mean_consistency": u_mean,
        "grounded_hallucination_rate": g_hall,
        "ungrounded_hallucination_rate": u_hall,
        "mcnemar_statistic": mc.statistic,
        "mcnemar_pvalue": mc.pvalue,
        "confusion": {"both_fail": both_fail, "both_pass": both_pass,
                      "grounded_fail_only": g_fail_only, "ungrounded_fail_only": u_fail_only},
        "per_disease": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
